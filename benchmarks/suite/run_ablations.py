#!/usr/bin/env python3
"""Ablation runner for plan section 10.4 items 1-6 (T-703).

Every variant trains a fresh r2-tiny model on the same data with the same step budget and seeds,
using the small self-contained loop below (AdamW, the model's own loss), then evaluates accuracy,
ECE and per-example latency on a held-out split.

Ablation switches
  loops_fixed_1       LoopPlan(n_loops=1) in training and evaluation (vs R sampled)
  no_deep_supervision the loss is recomputed from the last loop only: the model's output with
                      ``per_loop=[per_loop[-1]]`` goes through ``compute_losses`` (earlier loops
                      still run with truncated backprop, they just get no loss term)
  no_truncation       plan.k_bptt = 1000, so backprop covers every loop
  no_reinjection / no_damping / no_sandwich_norm / no_set_layer / no_null / no_numeric
                      Rhen2Config flags;  no_scratch: n_scratch=0

Tiny CPU runs say nothing about 50M-scale conclusions: differences are usually within seed noise.
Item 7 (flat baseline) is not implemented here.

    python3 benchmarks/suite/run_ablations.py --steps 60 --seeds 0,1,2 --out-dir /tmp/abl
"""

from __future__ import annotations

import argparse
import dataclasses
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

from benchmarks.suite import run_suite  # noqa: E402

# name -> (config overrides, plan mode, deep supervision, k_bptt override)
VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "loops_fixed_1": {"fixed_loops": 1},
    "no_deep_supervision": {"deep": False},
    "no_truncation": {"k_bptt": 1000},
    "no_reinjection": {"cfg": {"use_reinjection": False}},
    "no_damping": {"cfg": {"use_damping": False}},
    "no_sandwich_norm": {"cfg": {"use_sandwich_norm": False}},
    "no_set_layer": {"cfg": {"use_set_layer": False}},
    "no_null": {"cfg": {"use_null": False}},
    "no_numeric": {"cfg": {"use_numeric": False}},
    "no_scratch": {"cfg": {"n_scratch": 0}},
}


def load_data(source: str, n_train: int, n_eval: int, seed: int):
    if source == "v2":
        tr = run_suite.load_synthetic("train", n_train, seed)[0]
        te = run_suite.load_synthetic("test", n_eval, seed)[0]
        return tr, te
    from rhen.research.tasks import generate_split
    from rhen.v2.data.schema import from_v1_example
    return ([from_v1_example(e) for e in generate_split("train", n_train, seed)],
            [from_v1_example(e) for e in generate_split("test", n_eval, seed)])


def train_and_eval(name: str, spec: Dict[str, Any], train_ex, eval_ex, steps: int, seed: int, lr: float,
                   preset_name: str = "r2-tiny", batch_examples: int = 16) -> Dict[str, Any]:
    import torch
    from rhen.v2 import eval as ev
    from rhen.v2.config import preset
    from rhen.v2.losses import compute_losses
    from rhen.v2.model import Rhen2Model
    from rhen.v2.packing import Rhen2Packer
    from rhen.v2.workspace import LoopPlan

    torch.manual_seed(seed)
    cfg = preset(preset_name, **spec.get("cfg", {}))
    model, pk = Rhen2Model(cfg), Rhen2Packer(cfg)
    rng = random.Random(seed)
    order = list(train_ex)
    rng.shuffle(order)
    batches = []
    for i in range(0, len(order), batch_examples):
        try:
            batches.append(pk.pack(order[i:i + batch_examples]))
        except ValueError:
            continue
    if not batches:
        raise RuntimeError("no trainable batches")
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    last_loss = float("nan")
    t0 = time.perf_counter()
    for step in range(steps):
        batch = batches[step % len(batches)]
        plan = LoopPlan.sample(cfg, gen)
        if "fixed_loops" in spec:
            plan = LoopPlan(n_loops=spec["fixed_loops"], k_bptt=cfg.k_bptt, min_loops=spec["fixed_loops"], train=True)
        if "k_bptt" in spec:
            plan.k_bptt = spec["k_bptt"]
        out = model(batch, plan)
        loss = out.loss
        if not spec.get("deep", True):
            loss, _ = compute_losses(dataclasses.replace(out, per_loop=[out.per_loop[-1]]), batch, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach())
    train_seconds = time.perf_counter() - t0

    eval_plan: Any = "medium"
    if "fixed_loops" in spec:
        eval_plan = LoopPlan(n_loops=spec["fixed_loops"], min_loops=spec["fixed_loops"])
    timings: List[Dict[str, float]] = []
    recs = ev.evaluate_model(model, pk, eval_ex, eval_plan, 64, None, timings)
    d = ev.discrete_metrics(recs)
    n_ex = max(len(eval_ex), 1)
    return {"variant": name, "seed": seed, "accuracy": d.get("accuracy"), "ece": d.get("ece_15bin"),
            "n_eval_questions": d.get("n", 0), "ms_per_example": 1000.0 * sum(t["seconds"] for t in timings) / n_ex,
            "final_train_loss": last_loss, "train_seconds": train_seconds}


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in dict.fromkeys(r["variant"] for r in runs):
        rows = [r for r in runs if r["variant"] == name]
        entry: Dict[str, Any] = {"n_seeds": len(rows)}
        for key in ("accuracy", "ece", "ms_per_example", "final_train_loss"):
            v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
            entry[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0} if v.size else None
        out[name] = entry
    return out


def markdown_table(agg: Dict[str, Dict[str, Any]], steps: int) -> str:
    def f(e, scale=1.0, nd=3):
        return "n/a" if e is None else f"{e['mean'] * scale:.{nd}f} +- {e['std'] * scale:.{nd}f}"
    lines = [f"Tiny-budget ablations ({steps} steps, r2-tiny). Differences within seed noise mean nothing; "
             "tiny CPU runs say nothing about 50M-scale conclusions.", "",
             "| variant | seeds | accuracy | ECE (15 bin) | ms / example |", "| :--- | ---: | ---: | ---: | ---: |"]
    for name, e in agg.items():
        lines.append(f"| {name} | {e['n_seeds']} | {f(e['accuracy'])} | {f(e['ece'])} | {f(e['ms_per_example'], nd=2)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--variants", default="all", help="comma list of: " + ",".join(VARIANTS))
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--data-source", choices=("v1", "v2"), default="v1",
                   help="v1: rhen.research.tasks (short states, fast); v2: rhen.v2.data.synth")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "ablations")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = []
    for seed in seeds:
        train_ex, eval_ex = load_data(args.data_source, args.n_train, args.n_eval, 1000 + seed)
        for name in names:
            runs.append(train_and_eval(name, VARIANTS[name], train_ex, eval_ex, args.steps, seed, args.lr))
            print(f"seed {seed} {name}: acc={runs[-1]['accuracy']} ece={runs[-1]['ece']}", file=sys.stderr)
    agg = aggregate(runs)
    results = {
        "benchmark": "rhen2-ablations", "trained": "tiny budget only", "model": "r2-tiny",
        "steps": args.steps, "seeds": seeds, "data_source": args.data_source,
        "n_train": args.n_train, "n_eval": args.n_eval, "lr": args.lr,
        "commit": run_suite._git(["rev-parse", "HEAD"]), "worktree_dirty": run_suite.git_dirty(),
        "environment": run_suite.environment(), "command": "python3 " + shlex.join(sys.argv),
        "variants": agg, "runs": runs,
        "caveat": "tiny CPU runs with few steps say nothing about 50M-scale conclusions; most differences are "
                  "inside seed noise. Plan item 7 (flat baseline) is not run here.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ablations.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table = markdown_table(agg, args.steps)
    (args.out_dir / "ablations.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
