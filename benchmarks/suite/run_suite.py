#!/usr/bin/env python3
"""RHEN-2 evaluation suite (plan section 10, T-701).

Evaluates a checkpoint (or an UNTRAINED preset model, recorded as ``trained: false``) on the
repo's synthetic ``test`` and ``test_ood`` splits plus any JSONL files given with
``--data NAME=PATH``, and writes one JSON result file in the banking77 conventions.

    python3 benchmarks/suite/run_suite.py --preset r2-tiny --n 200 --out /tmp/suite.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = Path(__file__).resolve().parent / "results.json"

REFERENCE_POINTS = {
    "label": "vendor-reported numbers copied from the plan (section 10.1); NOT reproduced here and "
             "not measured on the same files",
    "JEV": {"banking77_acc": 0.870, "clinc150_acc": 0.895, "sst2_acc": 0.964, "arc_acc": 0.988, "pooled_ece": 0.028},
    "Laya": {"banking77_acc": 0.425, "banking77_ece": 0.081, "latency_ms_T4": [33, 40]},
    "Microsoft-Decision-1": {"suite_acc_36_benchmarks": 0.835, "latency_ms_p50": 85},
}

DEFAULT_WEAK_PATTERNS = {
    "arithmetic": ("arith", "sum", "total", "math"),
    "counting": ("count",),
    "date_comparison": ("date",),
    "multi_hop": ("hop", "chain"),
}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty() -> Optional[bool]:
    s = _git(["status", "--porcelain", "--untracked-files=no"])
    return None if s is None else bool(s.splitlines())


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> Dict[str, Any]:
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": package_version("torch"),
            "transformers": package_version("transformers"), "platform": platform.platform()}


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# model and data
# ---------------------------------------------------------------------------

def build_model(preset_name: str, checkpoint: Optional[str]):
    import torch
    from rhen.v2.config import Rhen2Config, preset
    from rhen.v2.model import Rhen2Model

    if checkpoint is None:
        cfg = preset(preset_name)
        model = Rhen2Model(cfg)
        info = {"trained": False, "preset": preset_name, "checkpoint": None,
                "description": f"untrained {preset_name} model with random initialisation"}
    else:
        try:
            from rhen.v2.io import load_checkpoint  # type: ignore
            model, extra = load_checkpoint(checkpoint)
        except ImportError:
            blob = torch.load(checkpoint, weights_only=True)
            model = Rhen2Model(Rhen2Config.from_dict(blob["cfg"]))
            model.load_state_dict(blob["state_dict"])
            extra = {k: v for k, v in blob.items() if k not in ("cfg", "state_dict")}
        info = {"trained": True, "preset": preset_name, "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint), "extra": _jsonable(extra)}
    model.eval()
    info["parameters"] = model.num_parameters()
    info["config"] = model.cfg.to_dict()
    return model, info


def _jsonable(x: Any) -> Any:
    try:
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return str(x)


def load_synthetic(split: str, n: int, seed: int, source: str = "auto") -> Tuple[list, List[str], str]:
    """(examples, family per example, generator description). Prefers rhen.v2.data.synth."""
    try:
        if source == "v1":
            raise ImportError("v1 generator requested")
        from rhen.v2.data import synth  # type: ignore
        tagged = synth.generate_tagged(split, n, seed)
        return [e for _, e in tagged], [f for f, _ in tagged], "rhen.v2.data.synth.generate_tagged"
    except Exception:  # missing module or signature mismatch: fall back to the v1 generator
        from rhen.research.tasks import generate_split
        from rhen.v2.data.schema import from_v1_example
        v1 = generate_split(split, n, seed)
        return [from_v1_example(e) for e in v1], [e.task or "unknown" for e in v1], \
            "rhen.research.tasks.generate_split via from_v1_example"


def fit_examples(packer, examples, families):
    """Drop examples the packer rejects (state longer than the model's max_state_tokens, ...).
    Returns (examples, families, n_dropped)."""
    keep_e, keep_f = [], []
    for e, f in zip(examples, families):
        try:
            packer.pack([e])
        except ValueError:
            continue
        keep_e.append(e)
        keep_f.append(f)
    return keep_e, keep_f, len(examples) - len(keep_e)


def _family_of(ex: Any) -> str:
    for attr in ("family", "task"):
        if getattr(ex, attr, None):
            return str(getattr(ex, attr))
    meta = getattr(ex, "meta", None)
    if isinstance(meta, dict) and meta.get("family"):
        return str(meta["family"])
    return "unknown"


def load_jsonl_with_families(path: str):
    from rhen.v2.data.schema import example_from_record
    exs, fams = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                exs.append(example_from_record(rec))
                meta = rec.get("meta") or {}
                fams.append(str(meta.get("family") or meta.get("task") or "unknown"))
    return exs, fams


def weak_groups(families: Sequence[str], override: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    if override:
        return override
    seen = sorted(set(families))
    return {g: [f for f in seen if any(p in f.lower() for p in pats)] for g, pats in DEFAULT_WEAK_PATTERNS.items()}


# ---------------------------------------------------------------------------
# numpy baselines on choice questions
# ---------------------------------------------------------------------------

_TOK = re.compile(r"[a-z0-9]+")


def _doc(ex, q) -> str:
    return (ex.state if isinstance(ex.state, str) else json.dumps(ex.state, default=str)) + " || " + q.instructions


def _choice_items(examples):
    for ex in examples:
        for q in ex.questions:
            if q.type == "choice" and q.label is not None and q.label != "__null__":
                gold = q.options[q.label] if isinstance(q.label, int) else q.label
                yield ex, q, str(gold)


def run_baselines(train_examples, test_examples) -> Dict[str, Any]:
    """Majority class and TF-IDF nearest centroid, with classes = option texts, choice questions only.
    Both choose among the options offered for the question; an option never seen in training
    scores lowest (ties go to the first option)."""
    tr = list(_choice_items(train_examples))
    te = list(_choice_items(test_examples))
    if not tr or not te:
        return {"status": "skipped", "reason": "no labelled choice questions in train or test"}
    from rhen.v2.eval import accuracy_with_ci
    prior = Counter(g for _, _, g in tr)
    df: Counter = Counter()
    docs = [_TOK.findall(_doc(e, q).lower()) for e, q, _ in tr]
    for d in docs:
        df.update(set(d))
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 1))}
    n = len(docs)
    idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in vocab])

    def vec(tokens):
        v = np.zeros(len(vocab))
        for t in tokens:
            j = vocab.get(t)
            if j is not None:
                v[j] += 1
        v *= idf
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    cent: Dict[str, np.ndarray] = {}
    for d, (_, _, g) in zip(docs, tr):
        cent[g] = cent.get(g, 0) + vec(d)
    for g in cent:
        nrm = np.linalg.norm(cent[g])
        cent[g] = cent[g] / nrm if nrm else cent[g]
    maj, cen = [], []
    for e, q, gold in te:
        opts = [str(o) for o in q.options]
        maj.append(opts[int(np.argmax([prior.get(o, 0) for o in opts]))] == gold)
        v = vec(_TOK.findall(_doc(e, q).lower()))
        cen.append(opts[int(np.argmax([float(cent[o] @ v) if o in cent else -1.0 for o in opts]))] == gold)
    return {
        "scope": "choice questions only; classes are option texts; trained on the synthetic train split",
        "majority_class": accuracy_with_ci(maj),
        "tfidf_nearest_centroid": accuracy_with_ci(cen),
    }


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def latency_summary(timings: List[Dict[str, float]]) -> Dict[str, Any]:
    if not timings:
        return {"n_batches": 0}
    per_q = np.array([t["seconds"] / max(t["questions"], 1) for t in timings]) * 1000.0
    per_b = np.array([t["seconds"] for t in timings]) * 1000.0
    return {"n_batches": len(timings), "ms_per_question_p50": float(np.percentile(per_q, 50)),
            "ms_per_question_p95": float(np.percentile(per_q, 95)), "ms_per_batch_p50": float(np.percentile(per_b, 50)),
            "note": "model call only (no packing), per-batch time divided by questions in the batch"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def evaluate_split(model, packer, examples, families, args, conformal_calib=None, with_robustness=False,
                   train_examples=None) -> Dict[str, Any]:
    from rhen.v2 import eval as ev
    timings: List[Dict[str, float]] = []
    recs = ev.evaluate_model(model, packer, examples, args.effort, args.batch_questions, families, timings)
    tables = None
    if conformal_calib is not None:
        tables = {a: ev.fit_conformal_table(conformal_calib, a) for a in (0.05, 0.1)}
    out: Dict[str, Any] = {
        "n_examples": len(examples),
        "model": ev.summarise(recs, weak_groups(families, args.family_groups), tables),
        "latency": latency_summary(timings),
    }
    if with_robustness:
        out["robustness"] = ev.robustness_probes(model, packer, examples, args.effort, args.batch_questions, args.seed)
    if train_examples is not None:
        out["baselines"] = run_baselines(train_examples, examples)
    return out, recs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="r2-tiny")
    p.add_argument("--checkpoint", default=None, help="torch.save dict {'cfg', 'state_dict'} or io.load_checkpoint file")
    p.add_argument("--n", type=int, default=200, help="examples per synthetic split")
    p.add_argument("--n-train", type=int, default=400, help="synthetic train examples for the numpy baselines")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--effort", default="medium", choices=("low", "medium", "high"))
    p.add_argument("--batch-questions", type=int, default=64)
    p.add_argument("--data", action="append", default=[], metavar="NAME=PATH", help="extra JSONL evaluation file")
    p.add_argument("--train-data", action="append", default=[], metavar="NAME=PATH",
                   help="JSONL training file enabling the numpy baselines for --data NAME")
    p.add_argument("--family-groups", default=None, type=json.loads,
                   help='JSON {"arithmetic": ["family", ...], ...} for the weak-spot table')
    p.add_argument("--synthetic", choices=("auto", "v1"), default="auto",
                   help="auto: rhen.v2.data.synth if importable, else the v1 generator; v1: force the v1 generator")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    import torch
    from rhen.v2.packing import Rhen2Packer

    torch.manual_seed(args.seed)
    model, info = build_model(args.preset, args.checkpoint)
    packer = Rhen2Packer(model.cfg)
    train_ex, _, _ = load_synthetic("train", args.n_train, args.seed, args.synthetic)
    val_ex, val_fam, _ = load_synthetic("val", args.n, args.seed, args.synthetic)
    val_ex, val_fam, _ = fit_examples(packer, val_ex, val_fam)
    from rhen.v2 import eval as ev
    calib = ev.evaluate_model(model, packer, val_ex, args.effort, args.batch_questions, val_fam)

    splits: Dict[str, Any] = {}
    datasets: Dict[str, Any] = {}
    id_recs = None
    for name in ("test", "test_ood"):
        ex, fam, gen = load_synthetic(name, args.n, args.seed, args.synthetic)
        ex, fam, dropped = fit_examples(packer, ex, fam)
        res, recs = evaluate_split(model, packer, ex, fam, args, calib, not args.skip_robustness and name == "test", train_ex)
        splits[name] = res
        datasets[name] = {"source": gen, "n_examples": len(ex), "dropped_too_long": dropped, "seed": args.seed}
        if name == "test":
            id_recs = recs
        else:
            res["ood_auroc_vs_test"] = ev.ood_report(id_recs, recs)
    train_files = dict(s.split("=", 1) for s in args.train_data)
    for spec in args.data:
        name, path = spec.split("=", 1)
        ex, fam = load_jsonl_with_families(path)
        ex, fam, dropped = fit_examples(packer, ex, fam)
        tr = None
        if name in train_files:
            tr = load_jsonl_with_families(train_files[name])[0]
        res, _ = evaluate_split(model, packer, ex, fam, args, None, False, tr)
        splits[name] = res
        datasets[name] = {"source": "jsonl", "path": path, "sha256": file_sha256(path), "n_examples": len(ex), "dropped_too_long": dropped}

    all_lat = [s["latency"] for s in splits.values() if s["latency"].get("n_batches")]
    results = {
        "benchmark": "rhen2-suite",
        "task": "RHEN-2 typed decision evaluation (plan section 10)",
        "trained": info["trained"],
        "model": info,
        "dataset": {"splits": datasets, "conformal_calibration": "synthetic val split, quantiles fitted per "
                    "(type, k_bucket) group on the model's own predictions; nominal coverage is only meaningful "
                    "when val and test are exchangeable"},
        "sampling": {"n_per_split": args.n, "seed": args.seed, "effort": args.effort},
        "seed": args.seed,
        "commit": _git(["rev-parse", "HEAD"]),
        "worktree_dirty": git_dirty(),
        "environment": environment(),
        "command": "python3 " + shlex.join(sys.argv),
        "splits": splits,
        "latency": {"splits": {k: v["latency"] for k, v in splits.items()},
                    "ms_per_question_p50_mean": (float(np.mean([l["ms_per_question_p50"] for l in all_lat])) if all_lat else None)},
        "reference_points": REFERENCE_POINTS,
        "notes": ["untrained models score at chance; these numbers only show the harness works"] if not info["trained"] else [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
