"""rhen-train: CLI wrapper for RHEN decision and language-model training.

Executes single-process or distributed training of RHEN non-autoregressive decision models,
language-model backbones, or joint multi-task models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

from rhen.research import distributed as distmod
from rhen.research.train import (
    OPTIMIZERS,
    SCHEDULES,
    TrainConfig,
    Trainer,
    apply_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhen-train",
        description="Train RHEN decision models and/or language-model objectives.",
    )
    # Configuration sources
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to JSON configuration file containing TrainConfig fields.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config field (e.g. --set lr=1e-4 --set model_overrides.hidden_size=128).",
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        "--decision-train",
        dest="decision_train",
        default=None,
        help="Path to training decision JSONL dataset.",
    )
    parser.add_argument(
        "--val-data",
        "--decision-val",
        dest="decision_val",
        default=None,
        help="Path to validation decision JSONL dataset.",
    )
    parser.add_argument(
        "--lm-data",
        default=None,
        help="Path to prepared token corpus directory (containing train.bin and val.bin).",
    )

    # Training run configuration
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Target run directory for checkpoints, metrics, and manifest (default: runs/default).",
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "small", "base"],
        default=None,
        help="Model architecture preset (default: tiny).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training optimizer steps (default: 1000).",
    )
    parser.add_argument(
        "--batch-size",
        "--decision-batch-size",
        dest="decision_batch_size",
        type=int,
        default=None,
        help="Decision batch size per micro-step (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--optimizer",
        choices=list(OPTIMIZERS),
        default=None,
        help=f"Optimizer to use: {OPTIMIZERS} (default: adamw).",
    )
    parser.add_argument(
        "--schedule",
        choices=list(SCHEDULES),
        default=None,
        help=f"Learning rate schedule: {SCHEDULES} (default: cosine).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Warmup steps for learning rate schedule (default: 100).",
    )
    parser.add_argument(
        "--order-consistency",
        "--order-consistency-coef",
        dest="order_consistency_coef",
        type=float,
        default=None,
        help="Loss weight for option-order permutation consistency objective.",
    )
    parser.add_argument(
        "--decision-loss-coef",
        type=float,
        default=None,
        help="Weight of decision loss objective (default: 1.0).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluation frequency in optimizer steps (default: 200).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Checkpoint save frequency in optimizer steps (default: 500).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Metric logging frequency in steps (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for data sampling and model initialization (default: 0).",
    )

    # Hardware & System
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        help="Compute device: auto, cpu, cuda, cuda:0, etc. (default: auto).",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16", "fp16"],
        default=None,
        help="Compute precision policy: auto, fp32, bf16, fp16 (default: auto).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint resume behavior: auto, never, or explicit checkpoint path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print summary without starting training.",
    )
    return parser


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Build and validate TrainConfig from CLI arguments and optional JSON config."""
    data: Dict[str, Any] = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)

    # Direct CLI flag mappings (CLI flags override JSON config file)
    direct_fields = [
        "decision_train",
        "decision_val",
        "lm_data",
        "run_dir",
        "preset",
        "max_steps",
        "decision_batch_size",
        "lr",
        "optimizer",
        "schedule",
        "warmup_steps",
        "order_consistency_coef",
        "decision_loss_coef",
        "eval_every",
        "save_every",
        "log_every",
        "seed",
        "device",
        "precision",
        "resume",
    ]
    for field_name in direct_fields:
        val = getattr(args, field_name, None)
        if val is not None:
            data[field_name] = val

    # Apply arbitrary key=value overrides
    if args.set:
        data = apply_overrides(data, args.set)

    cfg = TrainConfig.from_dict(data)
    cfg.validate()
    return cfg


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = build_config(args)
    except Exception as exc:
        sys.stderr.write(f"Configuration Error: {exc}\n")
        return 1

    if args.dry_run:
        print("RHEN Train Configuration (Dry Run):")
        print(json.dumps(cfg.__dict__, indent=2))
        return 0

    print("Starting RHEN training...")
    print(f"  Run Directory : {cfg.run_dir}")
    print(f"  Model Preset  : {cfg.preset}")
    print(f"  Max Steps     : {cfg.max_steps}")
    print(f"  Optimizer     : {cfg.optimizer} (lr={cfg.lr}, schedule={cfg.schedule})")
    print(f"  Decision Data : train={cfg.decision_train}, val={cfg.decision_val}")
    if cfg.order_consistency_coef > 0.0:
        print(f"  Order Invariance Coef: {cfg.order_consistency_coef}")

    try:
        trainer = Trainer(cfg)
        metrics = trainer.fit()
        print(f"\nTraining completed successfully! Run dir: {cfg.run_dir}")
        if metrics:
            print("Final Metrics Summary:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)):
                    print(f"  {k:<24}: {v:.4f}" if isinstance(v, float) else f"  {k:<24}: {v}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saved checkpoint safely.")
        return 130
    finally:
        distmod.cleanup()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
