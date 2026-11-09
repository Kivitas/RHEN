"""rhen-eval: CLI for evaluating RHEN decision accuracy, ECE calibration, Brier score, and OOD AUROC.

Evaluates decision quality metrics:
- Decision Accuracy: Top-1 categorical accuracy and multi-choice exact match
- ECE (Expected Calibration Error): Binned top-label confidence calibration
- Brier Score: Mean squared error of predicted probabilities
- OOD AUROC: Free energy and confidence separation between test and test_ood
- Option Order Consistency: Total variation and flip rate under option permutations
- Conformal Prediction: Empirical coverage and prediction set cardinality
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Sequence

from rhen.research import device as devmod
from rhen.research.checkpointing import load_model_from_checkpoint
from rhen.research.data import DecisionExample, load_decision_jsonl
from rhen.research.evaluate import decision_report
from rhen.research.manifest import write_json
from rhen.research.presets import preset_config
try:
    from rhen.config import EOSConfig
    from rhen.model import EOSModel
except (ImportError, ModuleNotFoundError):
    from config import EOSConfig
    from model import EOSModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhen-eval",
        description="Evaluate RHEN decision accuracy, calibration (ECE), Brier score, and OOD AUROC.",
    )
    # Model specification
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Path to trained checkpoint directory containing model.pt and model_config.json.",
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "small", "base"],
        default="tiny",
        help="Model preset if evaluating an initialized model without checkpoint (default: tiny).",
    )

    # Data specification
    parser.add_argument(
        "-d",
        "--decision-dir",
        "--data-dir",
        dest="decision_dir",
        default=None,
        help="Directory containing val.jsonl, test.jsonl, and test_ood.jsonl.",
    )
    parser.add_argument(
        "--val",
        default=None,
        help="Explicit path to validation split JSONL (used for calibration fitting).",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Explicit path to in-distribution test split JSONL.",
    )
    parser.add_argument(
        "--test-ood",
        "--ood",
        dest="test_ood",
        default=None,
        help="Explicit path to out-of-distribution test split JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate per split (useful for smoke tests).",
    )

    # Evaluation options
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal prediction error rate alpha (target coverage: 1 - alpha, default: 0.1).",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Number of recurrent computation loops (default: model config default).",
    )
    parser.add_argument(
        "--no-order-check",
        action="store_true",
        help="Disable option permutation invariance consistency checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for permutation evaluation (default: 0).",
    )

    # Hardware & System
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto, cpu, cuda, cuda:0, etc. (default: auto).",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="fp32",
        help="Compute precision policy: auto, fp32, bf16, fp16 (default: fp32).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or asset directory (default: bundled EOS tokenizer).",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="File path to write complete evaluation JSON report (default: <checkpoint>/eval_report.json or eval_report.json).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress terminal summary output.",
    )
    return parser


def format_summary_card(report: Dict[str, Any]) -> str:
    """Format evaluation results into a clean terminal report card."""
    lines = []
    lines.append("=" * 72)
    lines.append("                  RHEN DECISION EVALUATION REPORT                 ")
    lines.append("=" * 72)

    # Calibration section
    cal = report.get("decisions", {}).get("calibration")
    if cal:
        lines.append(f"Calibration Parameters (fitted on val, n={cal.get('n_calibration', 0)}):")
        lines.append(f"  * Temperature Scaling    : {cal.get('temperature', 1.0):.4f}")
        lines.append(f"  * Conformal Guarantee    : {1.0 - cal.get('conformal_alpha', 0.1):.1%}")
        lines.append(f"  * Conformal Threshold    : {cal.get('conformal_qhat', 0.0):.4f}")
        lines.append("-" * 72)

    # Table per split
    splits = report.get("decisions", {}).get("splits", {})
    col_w = [12, 10, 10, 10, 10, 14]
    header = f"{'SPLIT':<{col_w[0]}} | {'ACCURACY':<{col_w[1]}} | {'BRIER':<{col_w[2]}} | {'ECE':<{col_w[3]}} | {'NLL':<{col_w[4]}} | {'CONF COVERAGE':<{col_w[5]}}"
    lines.append(header)
    lines.append("-" * 72)

    for split_name in ("val", "test", "test_ood"):
        split_data = splits.get(split_name)
        if not split_data:
            continue
        # Prefer calibrated metrics if available, else raw
        raw_m = split_data.get("raw", {}).get("single_label", {})
        cal_m = split_data.get("calibrated", {}).get("single_label", raw_m)

        acc = f"{cal_m.get('accuracy', raw_m.get('accuracy', 0.0)):.2%}"
        brier = f"{cal_m.get('brier', raw_m.get('brier', 0.0)):.4f}"
        ece = f"{cal_m.get('ece', raw_m.get('ece', 0.0)):.4f}"
        nll = f"{cal_m.get('nll', raw_m.get('nll', 0.0)):.4f}"
        coverage = f"{cal_m.get('conformal_coverage', 0.0):.1%}" if "conformal_coverage" in cal_m else "-"

        lines.append(f"{split_name:<{col_w[0]}} | {acc:<{col_w[1]}} | {brier:<{col_w[2]}} | {ece:<{col_w[3]}} | {nll:<{col_w[4]}} | {coverage:<{col_w[5]}}")

    lines.append("-" * 72)

    # OOD Detection AUROC
    ood = report.get("decisions", {}).get("ood_detection")
    if ood:
        lines.append("Out-of-Distribution (OOD) Separation (test vs. test_ood):")
        lines.append(f"  * Free Energy AUROC      : {ood.get('auroc_free_energy', 0.0):.4f} (higher = better separation)")
        lines.append(f"  * Uncertainty AUROC      : {ood.get('auroc_max_prob', 0.0):.4f}")
        lines.append("-" * 72)

    # Permutation consistency
    test_split = splits.get("test", {})
    tv_info = test_split.get("raw", {}).get("permutation_consistency")
    if tv_info:
        lines.append("Option Order Invariance (Permutation Consistency on test):")
        lines.append(f"  * Mean Total Variation   : {tv_info.get('mean_total_variation', 0.0):.4f}")
        lines.append(f"  * Argmax Flip Rate       : {tv_info.get('argmax_flip_rate', 0.0):.2%}")
        lines.append("-" * 72)

    lines.append("=" * 72)
    return "\n".join(lines)


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = devmod.resolve_device(args.device)

    # Load or initialize model
    if args.checkpoint:
        if not os.path.isdir(args.checkpoint):
            sys.stderr.write(f"Error: Checkpoint directory not found: {args.checkpoint}\n")
            return 1
        model = load_model_from_checkpoint(args.checkpoint, device=device)
    else:
        cfg = EOSConfig(**preset_config(args.preset))
        model = EOSModel(cfg).to(device)
        model.eval()

    tokenizer = None
    if args.tokenizer:
        from tokenizer import get_tokenizer

        tokenizer = get_tokenizer(args.tokenizer)

    # Locate split datasets
    splits: Dict[str, Sequence[DecisionExample]] = {}
    split_paths = {
        "val": args.val or (os.path.join(args.decision_dir, "val.jsonl") if args.decision_dir else None),
        "test": args.test or (os.path.join(args.decision_dir, "test.jsonl") if args.decision_dir else None),
        "test_ood": args.test_ood or (os.path.join(args.decision_dir, "test_ood.jsonl") if args.decision_dir else None),
    }

    for name, path in split_paths.items():
        if path and os.path.exists(path):
            splits[name] = load_decision_jsonl(path)[: args.limit]

    if not splits:
        sys.stderr.write(
            "Error: No decision splits found. Specify --decision-dir or --val, --test, --test-ood.\n"
        )
        return 1

    report: Dict[str, Any] = {
        "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else f"preset:{args.preset}",
        "device": str(device),
    }

    report["decisions"], calibration = decision_report(
        model=model,
        splits=splits,
        tokenizer=tokenizer,
        num_loops=args.num_loops,
        alpha=args.alpha,
        order_check=not args.no_order_check,
        seed=args.seed,
    )

    if calibration is not None and args.checkpoint:
        cal_path = os.path.join(args.checkpoint, "decision_calibration.json")
        calibration.save(cal_path)

    # Determine report output path
    out_path = args.out
    if not out_path:
        out_path = (
            os.path.join(args.checkpoint, "eval_report.json")
            if args.checkpoint
            else "eval_report.json"
        )

    write_json(out_path, report)

    if not args.quiet:
        print(format_summary_card(report))
        print(f"Complete evaluation report written to: {out_path}\n")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
