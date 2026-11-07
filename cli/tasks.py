"""rhen-tasks: Generate synthetic decision benchmark sets.

Creates reproducible datasets with known ground truth across 6 task families:
- lookup: Direct value retrieval from state (tests grounding)
- routing: Multi-class rule-based decision routing (3 queues)
- approval: Binary transaction policy validation (noul)
- risk: Ordinal 5-tier exposure scoring (score)
- compare: Sensor value comparison (choice)
- checks: Multi-limit threshold excursion detection (multichoice)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from eos_research.tasks import FAMILIES, RULES, SPLITS, write_benchmark


def list_families() -> None:
    """Print available task families and their deterministic ground-truth rules."""
    print("RHEN Synthetic Benchmark Task Families:")
    print("=" * 60)
    for name, rule in RULES.items():
        print(f"  * {name:<12}: {rule}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhen-tasks",
        description="Generate synthetic decision benchmark sets with known ground truth.",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Target directory where <split>.jsonl and benchmark.json will be saved.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="flag_out_dir",
        default=None,
        help="Target directory (alternative to positional argument).",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=20_000,
        metavar="N",
        help="Number of in-distribution training examples (default: 20,000).",
    )
    parser.add_argument(
        "--val",
        type=int,
        default=1_000,
        metavar="N",
        help="Number of in-distribution validation examples (default: 1,000).",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=2_000,
        metavar="N",
        help="Number of in-distribution test examples (default: 2,000).",
    )
    parser.add_argument(
        "--ood",
        type=int,
        default=2_000,
        metavar="N",
        help="Number of out-of-distribution (OOD) test examples (default: 2,000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible generation (default: 0).",
    )
    parser.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help=f"Comma-separated subset of task families (default: {','.join(FAMILIES)}).",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="Display descriptions of all task families and exit.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode; suppress progress and summary output.",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_families:
        list_families()
        return 0

    target_dir = args.flag_out_dir or args.out_dir
    if not target_dir:
        parser.error("out_dir is required (e.g. `rhen-tasks data/decisions` or `--out-dir data/decisions`)")

    os.makedirs(target_dir, exist_ok=True)

    family_list = [f.strip() for f in args.families.split(",") if f.strip()]
    invalid = set(family_list) - set(FAMILIES)
    if invalid:
        sys.stderr.write(f"Error: Unknown task families: {sorted(invalid)}. Choose from: {list(FAMILIES)}\n")
        return 1

    if not args.quiet:
        print(f"Generating RHEN decision benchmark at '{target_dir}'...")
        print(f"  Seed: {args.seed}")
        print(f"  Families: {', '.join(family_list)}")
        print(f"  Splits: train={args.train}, val={args.val}, test={args.test}, test_ood={args.ood}")

    paths = write_benchmark(
        out_dir=target_dir,
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        n_ood=args.ood,
        seed=args.seed,
        families=family_list,
    )

    if not args.quiet:
        print("\nBenchmark files successfully created:")
        for split in SPLITS:
            path = paths.get(split)
            if path and os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024.0
                print(f"  * {split:<10}: {path} ({size_kb:.1f} KB)")
        meta_path = os.path.join(target_dir, "benchmark.json")
        if os.path.exists(meta_path):
            print(f"  * metadata  : {meta_path}")
        print("\nDone.")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
