"""Evaluation: language-model perplexity and Rhen decision quality.

``python -m eos_research.evaluate --checkpoint runs/x/checkpoints/step_00001000 \\
    --decision-dir data/decisions --lm-data data/corpus --out report.json``

The decision report fits temperature scaling and a conformal threshold on the ``val`` split,
then reports raw and calibrated metrics on ``test`` and ``test_ood``, OOD detection AUROC
(``test`` vs ``test_ood``) and option-order consistency (how much a decision changes when the
same options are presented in another order).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from eos_research import device as devmod
from eos_research import distributed as distmod
from eos_research.calibration import DecisionCalibration, calibrate_records, fit_decision_calibration
from eos_research.data import DecisionExample, TokenWindows, lm_inputs, load_decision_jsonl
from eos_research.manifest import write_json
from eos_research.metrics import ood_detection, summarize
from rhen_loss import ORDER_FREE_PRIMITIVES, sample_option_permutation


def _probs_of(result: Dict[str, Any]) -> List[float]:
    tensor = result.get("probs_tensor")
    if tensor is not None:
        return [float(p) for p in tensor[0].detach().float().cpu().tolist()]
    return [float(p) for p in result.get("probabilities", [])]


@torch.no_grad()
def decision_records(
    model,
    examples: Sequence[DecisionExample],
    tokenizer=None,
    num_loops: Optional[int] = None,
    order_check: bool = False,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """One record per question (see ``eos_research.metrics``).

    With ``order_check``, each example is evaluated a second time with the options of every
    order-free question reordered, and the record gets ``consistency_tv`` (total variation
    between the two decisions, per option identity) and ``consistency_flip`` (whether the
    chosen option changed).
    """
    from rhen_packer import get_rhen_packer

    packer = get_rhen_packer(tokenizer)
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(int(seed))
    records: List[Dict[str, Any]] = []
    try:
        for ex in examples:
            items = [q.packer_item() for q in ex.questions]
            results = model.forward_rhen(packer.pack(state=ex.state, questions_and_options=items), num_loops=num_loops)
            perm_results, perms = None, None
            if order_check:
                perms = [
                    sample_option_permutation(len(q.options), gen) if q.primitive in ORDER_FREE_PRIMITIVES
                    else list(range(len(q.options)))
                    for q in ex.questions
                ]
                permuted = [(it[0], [it[1][i] for i in p], it[2], it[3]) for it, p in zip(items, perms)]
                perm_results = model.forward_rhen(packer.pack(state=ex.state, questions_and_options=permuted), num_loops=num_loops)
            for i, q in enumerate(ex.questions):
                res = results[i]
                probs = _probs_of(res)
                rec: Dict[str, Any] = {
                    "task": ex.task,
                    "split": ex.split,
                    "primitive": q.primitive,
                    "probs": probs,
                    "label_index": q.class_index(),
                    "free_energy": res.get("free_energy"),
                }
                if q.primitive == "score":
                    rec["label_value"] = float(q.label)
                if q.primitive == "multichoice":
                    rec["label_set"] = list(q.label)
                if perm_results is not None and q.primitive in ORDER_FREE_PRIMITIVES and len(probs) > 1:
                    permuted_probs = _probs_of(perm_results[i])
                    restored = [0.0] * len(probs)
                    for j, src in enumerate(perms[i]):
                        restored[src] = permuted_probs[j]
                    if q.primitive == "multichoice":
                        rec["consistency_tv"] = sum(abs(a - b) for a, b in zip(probs, restored)) / len(probs)
                        rec["consistency_flip"] = [a >= 0.5 for a in probs] != [b >= 0.5 for b in restored]
                    else:
                        rec["consistency_tv"] = 0.5 * sum(abs(a - b) for a, b in zip(probs, restored))
                        rec["consistency_flip"] = probs.index(max(probs)) != restored.index(max(restored))
                records.append(rec)
    finally:
        model.train(was_training)
    return records


@torch.no_grad()
def evaluate_lm(
    model,
    windows: TokenWindows,
    batch_size: int,
    max_batches: Optional[int],
    device: torch.device,
    policy: Optional[devmod.PrecisionPolicy] = None,
    ctx: Optional[distmod.DistributedContext] = None,
    num_loops: Optional[int] = None,
) -> Dict[str, float]:
    """Token-weighted mean cross-entropy and perplexity over sequential windows.

    Batches are split round-robin across ranks and the sums are all-reduced, so every rank
    returns the same numbers.
    """
    ctx = ctx or distmod.DistributedContext(device=device)
    policy = policy or devmod.resolve_precision("fp32", device)
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.float64, device=device)
    start = time.perf_counter()
    try:
        for i, window in enumerate(windows.sequential(batch_size, max_batches)):
            if i % ctx.world_size != ctx.rank:
                continue
            inputs, labels = lm_inputs(window)
            inputs, labels = inputs.to(device), labels.to(device)
            with devmod.autocast(device, policy):
                # No labels: the model's loss includes auxiliary terms (MoE, ponder, z-loss), so
                # perplexity is computed from the logits as pure next-token cross-entropy.
                logits = model(input_ids=inputs, num_loops=num_loops).logits
            n = labels[:, 1:].numel()
            ce = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
            total_loss += ce.double() * n
            total_tokens += n
    finally:
        model.train(was_training)
    if ctx.enabled:
        import torch.distributed as dist

        dist.all_reduce(total_loss)
        dist.all_reduce(total_tokens)
    tokens = float(total_tokens.item())
    loss = float(total_loss.item()) / max(tokens, 1.0)
    return {
        "lm_loss": loss,
        "lm_perplexity": math.exp(min(loss, 50.0)),
        "lm_eval_tokens": tokens,
        "lm_eval_seconds": time.perf_counter() - start,
    }


def decision_report(
    model,
    splits: Dict[str, Sequence[DecisionExample]],
    tokenizer=None,
    num_loops: Optional[int] = None,
    alpha: float = 0.1,
    order_check: bool = True,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Optional[DecisionCalibration]]:
    """Raw and calibrated metrics per split; calibration is fitted on ``val`` when present."""
    records = {
        name: decision_records(model, exs, tokenizer, num_loops, order_check=order_check and name != "val", seed=seed)
        for name, exs in splits.items()
    }
    report: Dict[str, Any] = {"splits": {}}
    calibration = None
    if records.get("val"):
        calibration = fit_decision_calibration(records["val"], alpha=alpha, fitted_on="val")
        report["calibration"] = {
            "temperature": calibration.temperature,
            "conformal_alpha": calibration.conformal_alpha,
            "conformal_qhat": calibration.conformal_qhat,
            "n_calibration": calibration.n_calibration,
        }
    for name, recs in records.items():
        entry = {"raw": summarize(recs)}
        if calibration is not None and name != "val":
            entry["calibrated"] = summarize(calibrate_records(recs, calibration), qhat=calibration.conformal_qhat)
        report["splits"][name] = entry
    if records.get("test") and records.get("test_ood"):
        report["ood_detection"] = ood_detection(records["test"], records["test_ood"])
    return report, calibration


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EOS checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory (contains model.pt)")
    parser.add_argument("--decision-dir", help="directory with val/test/test_ood .jsonl files")
    parser.add_argument("--decision-limit", type=int, default=None, help="max examples per split")
    parser.add_argument("--lm-data", help="prepared corpus directory (val.bin)")
    parser.add_argument("--lm-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--no-order-check", action="store_true")
    parser.add_argument("--out", default=None, help="report path (default: <checkpoint>/eval_report.json)")
    args = parser.parse_args(argv)

    from eos_research.checkpointing import load_model_from_checkpoint

    device = devmod.resolve_device(args.device)
    policy = devmod.resolve_precision(args.precision, device)
    model = load_model_from_checkpoint(args.checkpoint, device)
    report: Dict[str, Any] = {"checkpoint": os.path.abspath(args.checkpoint), "device": str(device)}

    if args.lm_data:
        windows = TokenWindows(args.lm_data, "val", args.seq_len)
        report["lm"] = evaluate_lm(model, windows, args.batch_size, args.lm_batches, device, policy, num_loops=args.num_loops)

    if args.decision_dir:
        tokenizer = None
        if args.tokenizer:
            from tokenizer import get_tokenizer

            tokenizer = get_tokenizer(args.tokenizer)
        splits = {}
        for name in ("val", "test", "test_ood"):
            path = os.path.join(args.decision_dir, f"{name}.jsonl")
            if os.path.exists(path):
                splits[name] = load_decision_jsonl(path)[: args.decision_limit]
        report["decisions"], calibration = decision_report(
            model, splits, tokenizer, args.num_loops, args.alpha, order_check=not args.no_order_check
        )
        if calibration is not None:
            calibration.save(os.path.join(args.checkpoint, "decision_calibration.json"))

    out = args.out or os.path.join(args.checkpoint, "eval_report.json")
    write_json(out, report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
