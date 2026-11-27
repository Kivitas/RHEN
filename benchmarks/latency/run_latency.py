#!/usr/bin/env python3
"""Cold and warm latency by state length, K and effort (plan 5.2, task T-702).

cold = encode the state and deliberate (what a first question on a new state costs).
warm = deliberate only, state memory cached (an additional question on the same state).
Packing is done once outside the timed region. Weights are random: latency does not depend on
training, but the numbers apply to this preset, hardware and thread count only.

    python3 benchmarks/latency/run_latency.py --preset r2-tiny --state-lengths 32,128 --ks 2,8 --out /tmp/lat.json
"""

from __future__ import annotations

import argparse
import json
import platform
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

# Design targets from plan section 5.2 (not measurements): state 1k tokens, 1 question x 8 options, medium effort.
TARGETS = {
    "label": "design targets from plan 5.2, not results",
    "r2-50m": {"cpu_8_cores_ms": 60, "t4_gpu_ms": 10},
    "r2-300m": {"t4_gpu_ms": 40},
    "r2-1b": {"a10_gpu_ms": 80},
    "r2-8b": {"a100_gpu_ms": 250},
    "warm_cache_fraction_of_cold": 0.30,
    "condition": "state 1k tokens, 1 question x 8 options, medium effort",
}


def hardware() -> Dict[str, Any]:
    import torch
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cpu_count": __import__("os").cpu_count(),
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def make_example(packer, n_state_tokens: int, k: int):
    from rhen.v2.packing import QuestionSpec, Rhen2Example
    words = " ".join(f"item{i % 97} {i % 13}" for i in range(n_state_tokens))
    text = words
    while True:
        n = len(packer.tok.tokenizer(text, add_special_tokens=False)["input_ids"])
        if n <= n_state_tokens or len(text) < 4:
            break
        text = text[: int(len(text) * n_state_tokens / n) - 1]
    q = QuestionSpec("choice", "Which option is correct?", [f"option {i}" for i in range(k)])
    return Rhen2Example(text, [q])


def _sync(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()


def stats(ms: List[float]) -> Dict[str, float]:
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "n": int(a.size)}


def measure(model, packer, device, n_state: int, k: int, effort: str, warmup: int, repeats: int) -> Dict[str, Any]:
    import torch
    from rhen.v2.workspace import LoopPlan
    batch = packer.pack([make_example(packer, n_state, k)]).to(device)
    plan = LoopPlan.effort(model.cfg, effort)
    cold, warm = [], []
    with torch.inference_mode():
        S = model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
        for _ in range(warmup):
            model.deliberate(model.encode_state(batch.state_ids, batch.state_mask, batch.state_num), batch, plan)
            model.deliberate(S, batch, plan)
        for _ in range(repeats):
            _sync(device)
            t = time.perf_counter()
            S2 = model.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
            model.deliberate(S2, batch, plan)
            _sync(device)
            cold.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            model.deliberate(S, batch, plan)
            _sync(device)
            warm.append((time.perf_counter() - t) * 1000)
    c, w = stats(cold), stats(warm)
    return {"state_tokens_requested": n_state, "state_tokens_actual": int(batch.state_mask.sum()), "k": k,
            "effort": effort, "cold": c, "warm": w, "warm_over_cold_p50": w["p50_ms"] / c["p50_ms"]}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="r2-tiny")
    p.add_argument("--state-lengths", default="64,256")
    p.add_argument("--ks", default="2,8")
    p.add_argument("--efforts", default="low,medium")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--devices", default="auto", help="'auto' (cpu, plus cuda if available), or a comma list")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results.json")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import torch
    from rhen.v2.config import preset
    from rhen.v2.model import Rhen2Model
    from rhen.v2.packing import Rhen2Packer

    args = parse_args(argv)
    devices = (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])) if args.devices == "auto" else args.devices.split(",")
    torch.manual_seed(0)
    cfg = preset(args.preset)
    packer = Rhen2Packer(cfg)
    results: Dict[str, Any] = {}
    skipped = []
    for dev in devices:
        device = torch.device(dev)
        model = Rhen2Model(cfg).to(device).eval()
        rows = []
        for n_state in (int(x) for x in args.state_lengths.split(",")):
            if n_state + 1 > cfg.max_state_tokens:
                skipped.append({"state_tokens": n_state, "reason": f"exceeds max_state_tokens={cfg.max_state_tokens}"})
                continue
            for k in (int(x) for x in args.ks.split(",")):
                for effort in args.efforts.split(","):
                    rows.append(measure(model, packer, device, n_state, k, effort, args.warmup, args.repeats))
        results[dev] = rows
    out = {
        "benchmark": "rhen2-latency", "preset": args.preset, "parameters": Rhen2Model(cfg).num_parameters(),
        "trained": False, "model": f"untrained {args.preset} (latency does not depend on weights)",
        "warmup": args.warmup, "repeats": args.repeats, "hardware": hardware(),
        "commit": run_suite._git(["rev-parse", "HEAD"]), "worktree_dirty": run_suite.git_dirty(),
        "command": "python3 " + shlex.join(sys.argv), "results": results, "skipped": skipped,
        "targets": TARGETS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
