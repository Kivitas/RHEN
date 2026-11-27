#!/usr/bin/env python3
"""Flat baseline vs RHEN-2 workspace (plan 10.4 item 7).

Trains ``Rhen2Model`` and ``FlatBaselineModel`` with the same data, step budget, optimiser,
batch order and seeds on a mixture of synthetic choice / noul / score / multi families
(``rhen.v2.data.synth``), then evaluates both on the in-distribution ``test`` split, the
``test_ood`` split and the option-shuffle flip rate. The flat model is parameter-matched with
``matched_flat_config`` (default) or FLOP-matched (``--match flops``). Writes JSON with mean and
standard deviation over seeds, parameter counts and FLOPs estimates.

Tiny CPU models, synthetic data and few seeds: the output says nothing about 50M-scale behaviour.

    python3 benchmarks/suite/flat_vs_workspace.py --out /tmp/flat_vs_workspace.json
    python3 benchmarks/suite/flat_vs_workspace.py --preset r2-50m --seeds 0 --steps 20 --n-train 32 --n-eval 32
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FAMILIES = (
    "negation_choice,majority_vote,count_most_frequent,threshold_band,tolerance,unit_compare,policy_exceptions,"
    "noul_overdue,pair_threshold,pair_membership,score_pass_rate,score_delay_days,"
    "multi_mentions,multi_policies,multi_members"
)
METRICS = (
    "test_accuracy", "test_ece", "test_multi_exact", "test_multi_f1",
    "ood_accuracy", "ood_ece", "ood_multi_exact", "ood_multi_f1",
    "flip_rate", "final_train_loss", "train_seconds", "ms_per_example",
)


def build_data(families: List[str], n_train: int, n_eval: int, seed: int, packer) -> Dict[str, Any]:
    """Generate the splits and drop examples the model's token limits cannot hold (counted, not hidden)."""
    from benchmarks.suite import run_suite
    from rhen.v2.data import synth

    out: Dict[str, Any] = {"dropped": {}, "kept": {}}
    for split, n in (("train", n_train), ("test", n_eval), ("test_ood", n_eval)):
        tagged = synth.generate_tagged(split, n, seed, families)
        ex, fam, dropped = run_suite.fit_examples(packer, [e for _, e in tagged], [f for f, _ in tagged])
        out[split] = (ex, fam)
        out["dropped"][split] = dropped
        out["kept"][split] = len(ex)
    return out


def train(model, cfg, packer, train_ex, steps: int, seed: int, lr: float, batch_examples: int, is_flat: bool) -> Dict[str, float]:
    import torch
    from rhen.v2.workspace import LoopPlan

    rng = random.Random(seed)
    order = list(train_ex)
    rng.shuffle(order)
    batches = [packer.pack(order[i:i + batch_examples]) for i in range(0, len(order), batch_examples)]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    recent: List[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        batch = batches[step % len(batches)]
        plan = None if is_flat else LoopPlan.sample(cfg, gen)
        out = model(batch, plan)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent.append(float(out.loss.detach()))
    return {"final_train_loss": float(np.mean(recent[-10:])), "train_seconds": time.perf_counter() - t0}


def evaluate(model, packer, data, seed: int) -> Dict[str, Any]:
    from rhen.v2 import eval as ev

    res: Dict[str, Any] = {}
    timings: List[Dict[str, float]] = []
    base = None
    for split, key in (("test", "test"), ("test_ood", "ood")):
        ex, fam = data[split]
        recs = ev.evaluate_model(model, packer, ex, "medium", 64, fam, timings if split == "test" else None)
        d, mm = ev.discrete_metrics(recs), ev.multi_metrics(recs)
        res[f"{key}_accuracy"], res[f"{key}_ece"] = d.get("accuracy"), d.get("ece_15bin")
        res[f"{key}_multi_exact"], res[f"{key}_multi_f1"] = mm.get("exact_match"), mm.get("mean_option_f1")
        res[f"{key}_n_discrete"], res[f"{key}_n_multi"] = d.get("n", 0), mm.get("n", 0)
        if split == "test":
            base = recs
    flip = ev.option_shuffle_flip_rate(model, packer, data["test"][0], "medium", 64, seed, base)
    res["flip_rate"], res["flip_n"], res["flip_max_abs_prob_diff"] = flip["flip_rate"], flip["n"], flip["max_abs_prob_diff"]
    n_ex = max(len(data["test"][0]), 1)
    res["ms_per_example"] = 1000.0 * sum(t["seconds"] for t in timings) / n_ex
    return res


def mean_std(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, float]]:
    v = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
    if not v.size:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from rhen.v2.config import count_parameters, preset
    from rhen.v2.flat_baseline import FlatBaselineModel, flops_estimate, matched_flat_config
    from rhen.v2.model import Rhen2Model
    from rhen.v2.packing import Rhen2Packer

    families = args.families.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = preset(args.preset)
    packer = Rhen2Packer(cfg)
    ws_flops_ref = None
    runs: Dict[str, List[Dict[str, Any]]] = {"workspace": [], "flat": []}
    info: Dict[str, Any] = {}
    for seed in seeds:
        data = build_data(families, args.n_train, args.n_eval, 1000 + seed, packer)
        probe = packer.pack(data["test"][0][:16])
        flat_cfg_params = matched_flat_config(cfg)
        if args.match == "flops":
            # match the workspace model's FLOPs per example on a probe batch of the test split
            ref_ws = Rhen2Model(cfg)
            ws_flops_ref = flops_estimate(ref_ws, probe)
            n_q = float(probe.q_valid.sum()) / probe.batch_size
            per_q = ws_flops_ref / max(n_q, 1.0)
            ref = {"n_state": int(probe.state_mask.sum(-1).float().mean()), "n_q": int(probe.q_mask.sum(-1).float().mean()),
                   "k": int(probe.opt_valid.sum(-1).float().mean()), "opt_tokens": 4}
            flat_cfg = matched_flat_config(cfg, target_flops=per_q, ref=ref)
        else:
            flat_cfg = flat_cfg_params
        for name in ("workspace", "flat"):
            torch.manual_seed(seed)
            is_flat = name == "flat"
            model = FlatBaselineModel(flat_cfg) if is_flat else Rhen2Model(cfg)
            m_cfg = flat_cfg if is_flat else cfg
            tr = train(model, m_cfg, packer, data["train"][0], args.steps, seed, args.lr, args.batch_examples, is_flat)
            row = {"seed": seed, **tr, **evaluate(model, packer, data, seed)}
            row["flops_per_example_probe"] = flops_estimate(model, probe)
            row["n_params"] = model.num_parameters()
            row["dropped"], row["kept"] = data["dropped"], data["kept"]
            runs[name].append(row)
            print(f"seed {seed} {name}: test_acc={row['test_accuracy']} ood_acc={row['ood_accuracy']} "
                  f"flip={row['flip_rate']} loss={row['final_train_loss']:.3f} ({row['train_seconds']:.0f}s)", file=sys.stderr)
        info = {"flat_layers": flat_cfg.n_state_layers, "flat_ffn_mult": flat_cfg.ffn_mult}
    summary = {name: {k: mean_std(rows, k) for k in METRICS + ("flops_per_example_probe", "n_params")} for name, rows in runs.items()}
    paired = {}
    for k in METRICS:
        diffs = [f[k] - w[k] for w, f in zip(runs["workspace"], runs["flat"]) if w.get(k) is not None and f.get(k) is not None]
        if diffs:
            paired[k] = {"mean": float(np.mean(diffs)), "std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0}
    return {
        "benchmark": "flat-vs-workspace", "preset": args.preset, "match": args.match, "seeds": seeds,
        "steps": args.steps, "lr": args.lr, "batch_examples": args.batch_examples,
        "n_train": args.n_train, "n_eval": args.n_eval, "families": families,
        "param_counts": {"workspace": count_parameters(cfg)["total"], "flat": runs["flat"][0]["n_params"]},
        "flat_config": info, "summary": summary, "flat_minus_workspace_paired": paired, "runs": runs,
        "command": "python3 " + shlex.join(sys.argv),
        "caveat": "tiny CPU models, synthetic data, few seeds; says nothing about 50M-scale behaviour. "
                  "Std is the sample std over seeds (ddof=1).",
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="r2-tiny")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-train", type=int, default=384)
    p.add_argument("--n-eval", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-examples", type=int, default=16)
    p.add_argument("--families", default=DEFAULT_FAMILIES)
    p.add_argument("--match", choices=("params", "flops"), default="params")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "flat_vs_workspace.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    results = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{'metric':22s} {'workspace':>18s} {'flat':>18s}")
    for k in METRICS + ("flops_per_example_probe", "n_params"):
        cells = []
        for name in ("workspace", "flat"):
            e = results["summary"][name][k]
            cells.append("n/a" if e is None else f"{e['mean']:.4g} +- {e['std']:.2g}")
        print(f"{k:22s} {cells[0]:>18s} {cells[1]:>18s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
