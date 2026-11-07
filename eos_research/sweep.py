"""Run a grid of training configurations, with repeated seeds, and tabulate the results.

    python -m eos_research.sweep --config base.json --out runs/ablate_loops \\
        --grid model_overrides.max_recurrent_loops=2,4,8 --grid order_consistency_coef=0,0.1 \\
        --seeds 0,1,2

Every combination runs ``eos_research.train`` in its own process (so one failure does not stop
the sweep) under ``<out>/<name>/seed<k>``. Finished runs are skipped when the sweep is
restarted. ``summary.json`` and ``summary.csv`` report, per combination, the mean and standard
deviation over seeds of every final evaluation metric.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple

from eos_research.train import _flatten, _parse_value


def parse_grid(items: Sequence[str]) -> List[Tuple[str, List[Any]]]:
    grid = []
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or not values:
            raise ValueError(f"--grid expects key=v1,v2,..., got {item!r}")
        grid.append((key, [_parse_value(v) for v in values.split(",")]))
    return grid


def combo_name(combo: Sequence[Tuple[str, Any]]) -> str:
    if not combo:
        return "base"
    parts = [f"{key.split('.')[-1]}={value}" for key, value in combo]
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", "__".join(parts))


def _stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def run_sweep(config: str, out: str, grid: List[Tuple[str, List[Any]]], seeds: List[int], extra: Sequence[str]) -> Dict[str, Any]:
    keys = [k for k, _ in grid]
    combos = [list(zip(keys, values)) for values in itertools.product(*[v for _, v in grid])] or [[]]
    summary: Dict[str, Any] = {}
    for combo in combos:
        name = combo_name(combo)
        per_seed: List[Dict[str, float]] = []
        for seed in seeds:
            run_dir = os.path.join(out, name, f"seed{seed}")
            final_path = os.path.join(run_dir, "final_metrics.json")
            if not os.path.exists(final_path):
                cmd = [sys.executable, "-m", "eos_research.train", "--config", config,
                       "--set", f"run_dir={json.dumps(run_dir)}", "--set", f"seed={seed}"]
                for key, value in combo:
                    cmd += ["--set", f"{key}={json.dumps(value)}"]
                for item in extra:
                    cmd += ["--set", item]
                print(f"[sweep] {name} seed={seed}", flush=True)
                result = subprocess.run(cmd)
                if result.returncode != 0 or not os.path.exists(final_path):
                    print(f"[sweep] {name} seed={seed} failed (exit {result.returncode})", flush=True)
                    continue
            with open(final_path, "r", encoding="utf-8") as fh:
                flat: Dict[str, float] = {}
                _flatten("", json.load(fh).get("eval", {}), flat)
                per_seed.append(flat)
        metrics = sorted(set().union(*per_seed)) if per_seed else []
        summary[name] = {
            "overrides": dict(combo),
            "seeds_completed": len(per_seed),
            "metrics": {m: _stats([r[m] for r in per_seed if m in r]) for m in metrics},
        }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    all_metrics = sorted({m for entry in summary.values() for m in entry["metrics"]})
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "seeds"] + [f"{m}_{s}" for m in all_metrics for s in ("mean", "std")])
        for name, entry in summary.items():
            row = [name, entry["seeds_completed"]]
            for m in all_metrics:
                st = entry["metrics"].get(m)
                row += [st["mean"], st["std"]] if st else ["", ""]
            writer.writerow(row)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Grid/seed sweeps over eos_research.train.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="fixed override for every run")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    run_sweep(args.config, args.out, parse_grid(args.grid), seeds, args.set)
    print(f"wrote {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
