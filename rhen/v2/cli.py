"""Command line entry points: ``rhen2-train`` (``main_train``) and ``rhen2-eval`` (``main_eval``)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Sequence


def _train_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rhen2-train", description="Train RHEN-2 (pretrain, decision or distill stage).")
    ap.add_argument("--config", required=True, help="path to a TrainConfig JSON file (configs/v2/*.json)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key with a JSON value; dotted keys reach into dicts and lists "
                         "(for example --set steps=20 --set model_overrides.d_model=128). Repeatable.")
    ap.add_argument("--dry-run", action="store_true", help="validate config and data, print the plan, do not train")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint in run_dir")
    ap.add_argument("--stop-at", type=int, default=None, help="stop (and checkpoint) after this step; for testing resume")
    return ap


def main_train(argv: Optional[Sequence[str]] = None) -> int:
    args = _train_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from rhen.v2.train import TrainConfig, run_training

    overrides: List[str] = list(args.overrides)
    if args.no_resume:
        overrides.append("resume=false")
    try:
        cfg = TrainConfig.load(args.config, overrides)
    except (ValueError, OSError) as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2
    result = run_training(cfg, dry_run=args.dry_run, stop_at=args.stop_at)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _eval_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rhen2-eval", description="Evaluate a RHEN-2 checkpoint (needs rhen/v2/eval.py).")
    ap.add_argument("--checkpoint", required=True, help="model .pt file, a step directory or a run directory")
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema", help="which weights of a step directory to load")
    ap.add_argument("--jsonl", action="append", default=[], help="labelled decision JSONL to evaluate on (repeatable)")
    ap.add_argument("--split", default="val", choices=("train", "val", "test", "test_ood"),
                    help="synthetic split used when no --jsonl is given")
    ap.add_argument("-n", "--num", type=int, default=64, help="synthetic examples when no --jsonl is given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    ap.add_argument("--batch-questions", type=int, default=64)
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    return ap


def main_eval(argv: Optional[Sequence[str]] = None) -> int:
    args = _eval_parser().parse_args(argv)
    try:
        from rhen.v2 import eval as v2eval
        evaluate_model, summarise = v2eval.evaluate_model, v2eval.summarise
    except (ImportError, AttributeError) as exc:
        print("rhen2-eval needs rhen/v2/eval.py with evaluate_model and summarise, which could not be imported "
              f"({exc}).", file=sys.stderr)
        return 3
    from rhen.v2.data.schema import read_jsonl
    from rhen.v2.data.synth import generate_tagged
    from rhen.v2.io import load_checkpoint
    from rhen.v2.packing import Rhen2Packer
    from rhen.v2.train import resolve_model_file

    path = resolve_model_file(args.checkpoint, prefer_ema=args.weights == "ema")
    if args.weights == "raw" and os.path.basename(path) == "model_ema.pt":
        path = os.path.join(os.path.dirname(path), "model.pt")
    model, extra = load_checkpoint(path)
    packer = Rhen2Packer(model.cfg)
    if args.jsonl:
        examples = [ex for p in args.jsonl for ex in read_jsonl(p)]
        families = ["jsonl"] * len(examples)
    else:
        tagged = generate_tagged(args.split, args.num, args.seed)
        families = [f for f, _ in tagged]
        examples = [e for _, e in tagged]
    records = evaluate_model(model, packer, examples, effort=args.effort, batch_questions=args.batch_questions,
                             families=families)
    result = {"checkpoint": path, "checkpoint_extra": extra, "effort": args.effort, "n_examples": len(examples),
              "metrics": summarise(records)}
    if extra.get("smoke_only"):
        result["notice"] = "checkpoint comes from an r2-tiny smoke run: plumbing only, not evidence about model quality"
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main_train())
