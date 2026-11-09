#!/usr/bin/env python3
"""Reproducible Banking77 intent-classification evaluation.

Variants (definitions and caveats in README.md):

  majority_class             most frequent training label
  tfidf_nearest_centroid     TF-IDF (numpy) + cosine similarity to class centroids
  tfidf_logistic_regression  TF-IDF (numpy) + multinomial logistic regression,
                             full-batch gradient descent
  rhen_untrained             RhenEngine().choice() with the default configuration,
                             no checkpoint and no training

Only numpy is imported at module load. torch and rhen are imported only for the
RHEN variant, and --help exits before any network access.

Example:
    python3 benchmarks/banking77/run_eval.py --limit 1000 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

HF_DATASET = "PolyAI/banking77"
HF_CONFIG = "default"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}"
ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
ROWS_PAGE_SIZE = 100
SPLITS = ("train", "test")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rhen-benchmarks" / "banking77"
DEFAULT_OUTPUT = SCRIPT_DIR / "results.json"

Z95 = 1.959963984540054
ECE_BINS = 10
TFIDF_MIN_DF = 2
LOGREG_LR_GRID = (0.5, 2.0, 8.0, 32.0, 128.0)
LOGREG_EPOCHS = 300
LOGREG_L2 = 1e-4
LOGREG_VAL_FRACTION = 0.10
RHEN_QUESTION = "Which banking intent?"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _fetch_json(url: str, retries: int = 3) -> Dict[str, Any]:
    """GET a JSON document over HTTPS with simple retries."""
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last_exc}")


def _fetch_revision() -> Optional[str]:
    try:
        return _fetch_json(HF_API_URL).get("sha")
    except RuntimeError:
        return None


def _label_names_from_features(features: List[Dict[str, Any]]) -> List[str]:
    for feature in features:
        if feature.get("name") == "label":
            return list(feature["type"]["names"])
    raise RuntimeError("rows API response has no 'label' feature")


def _download_split_via_rows_api(split: str) -> Dict[str, Any]:
    """Page through the datasets-server rows API for one split (JSON, no pyarrow)."""
    rows: List[Dict[str, Any]] = []
    total: Optional[int] = None
    label_names: Optional[List[str]] = None
    offset = 0
    while total is None or offset < total:
        query = urllib.parse.urlencode({
            "dataset": HF_DATASET,
            "config": HF_CONFIG,
            "split": split,
            "offset": offset,
            "length": ROWS_PAGE_SIZE,
        })
        payload = _fetch_json(f"{ROWS_API_URL}?{query}")
        if total is None:
            total = int(payload["num_rows_total"])
            label_names = _label_names_from_features(payload["features"])
        page = payload.get("rows", [])
        if not page:
            break
        for item in page:
            if item["row_idx"] != len(rows):
                raise RuntimeError(f"{split}: non-contiguous row_idx {item['row_idx']} at {len(rows)}")
            rows.append({"text": item["row"]["text"], "label": int(item["row"]["label"])})
        offset += len(page)
    if total is None or len(rows) != total:
        raise RuntimeError(f"{split}: downloaded {len(rows)} rows, expected {total}")
    return {"num_rows_total": total, "label_names": label_names, "rows": rows}


def _load_via_rows_api(cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        cache_file = cache_dir / f"{HF_DATASET.replace('/', '__')}-{split}.json"
        if not cache_file.exists():
            blob = _download_split_via_rows_api(split)
            tmp = cache_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(blob, sort_keys=True, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_file)
        raw = cache_file.read_bytes()
        blob = json.loads(raw.decode("utf-8"))
        splits[split] = {
            "texts": [r["text"] for r in blob["rows"]],
            "labels": np.asarray([r["label"] for r in blob["rows"]], dtype=np.int64),
            "label_names": blob["label_names"],
            "file": cache_file.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(blob["rows"]),
            "num_rows_total": blob["num_rows_total"],
        }
    if splits["train"]["label_names"] != splits["test"]["label_names"]:
        raise RuntimeError("train and test label names differ")
    return {
        "loader": "datasets-server rows API (JSON, cached)",
        "label_names": splits["train"]["label_names"],
        "splits": splits,
    }


def _load_via_datasets_library() -> Optional[Dict[str, Any]]:
    try:
        import datasets  # type: ignore
    except ImportError:
        return None
    ds = datasets.load_dataset(HF_DATASET)
    label_names = list(ds["train"].features["label"].names)
    splits: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        part = ds[split]
        texts = [str(t) for t in part["text"]]
        labels = np.asarray([int(v) for v in part["label"]], dtype=np.int64)
        splits[split] = {
            "texts": texts,
            "labels": labels,
            "file": None,
            "sha256": None,
            "rows": len(texts),
            "num_rows_total": len(texts),
        }
    return {
        "loader": f"datasets {datasets.__version__} load_dataset",
        "label_names": label_names,
        "splits": splits,
    }


def load_banking77(cache_dir: Path, prefer_datasets_library: bool = True) -> Dict[str, Any]:
    """Return the official train and test splits of PolyAI/banking77.

    Uses the `datasets` library when it is installed, otherwise the datasets-server
    rows API. Either way the split sizes come from the dataset itself.
    """
    loaded = _load_via_datasets_library() if prefer_datasets_library else None
    if loaded is None:
        loaded = _load_via_rows_api(cache_dir)
    loaded["hf_dataset"] = HF_DATASET
    loaded["hf_revision"] = _fetch_revision()
    return loaded


def select_test_indices(n_total: int, limit: Optional[int], seed: int) -> np.ndarray:
    """Reproducible random subset of test indices (sorted, so original order is kept)."""
    if limit is None or limit >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=limit, replace=False))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = ECE_BINS) -> float:
    """ECE with equal-width bins over [0, 1] on the top-label confidence."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        raise ValueError("empty input")
    bins = np.minimum(np.floor(conf * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += (mask.sum() / conf.size) * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)


def summarize(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    seconds: float,
    probs: Optional[np.ndarray] = None,
    abstained: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    correct = np.asarray(y_pred) == np.asarray(y_true)
    n = int(correct.size)
    k = int(correct.sum())
    lo, hi = wilson_interval(k, n)
    out: Dict[str, Any] = {
        "status": "ok",
        "n": n,
        "correct": k,
        "top1_accuracy": k / n,
        "wilson95": [lo, hi],
        "ece_10bin": (
            expected_calibration_error(probs.max(axis=1), correct) if probs is not None else None
        ),
        "abstain_rate": None,
        "accuracy_on_answered": None,
        "seconds_per_example": seconds / n,
    }
    if abstained is not None:
        ab = np.asarray(abstained, dtype=bool)
        answered = ~ab
        out["abstain_rate"] = float(ab.mean())
        out["accuracy_on_answered"] = float(correct[answered].mean()) if answered.any() else None
    out.update(extra or {})
    return out


# ---------------------------------------------------------------------------
# Baselines (numpy only)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """Minimal smooth-idf TF-IDF with L2 normalisation. Vocabulary is fit on training text only."""

    def __init__(self, min_df: int = TFIDF_MIN_DF) -> None:
        self.min_df = min_df
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.zeros(0, dtype=np.float32)

    def fit(self, texts: Sequence[str]) -> "TfidfVectorizer":
        df: Counter = Counter()
        for text in texts:
            df.update(set(tokenize(text)))
        vocab = sorted(w for w, c in df.items() if c >= self.min_df)
        self.vocab_ = {w: i for i, w in enumerate(vocab)}
        n = len(texts)
        doc_freq = np.asarray([df[w] for w in vocab], dtype=np.float64)
        self.idf_ = (np.log((1.0 + n) / (1.0 + doc_freq)) + 1.0).astype(np.float32)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in tokenize(text):
                j = self.vocab_.get(word)
                if j is not None:
                    X[i, j] += 1.0
        X *= self.idf_[None, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms


def run_majority(train_labels: np.ndarray, test_labels: np.ndarray, n_classes: int) -> Dict[str, Any]:
    counts = np.bincount(train_labels, minlength=n_classes)
    majority = int(np.argmax(counts))  # ties resolve to the smallest label index
    prior = counts / counts.sum()
    t0 = time.perf_counter()
    preds = np.full(len(test_labels), majority, dtype=np.int64)
    probs = np.tile(prior, (len(test_labels), 1))
    seconds = time.perf_counter() - t0
    return summarize(
        test_labels, preds, seconds, probs=probs,
        extra={
            "majority_label_index": majority,
            "fit_seconds": 0.0,
            "notes": "probabilities are the training label prior",
        },
    )


def run_nearest_centroid(
    train_texts: Sequence[str], train_labels: np.ndarray,
    test_texts: Sequence[str], test_labels: np.ndarray, n_classes: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    vec = TfidfVectorizer().fit(train_texts)
    X_train = vec.transform(train_texts)
    onehot = np.eye(n_classes, dtype=np.float32)[train_labels]
    counts = onehot.sum(axis=0)
    centroids = (onehot.T @ X_train) / np.maximum(counts, 1.0)[:, None]
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test = vec.transform(test_texts)
    preds = (X_test @ centroids.T).argmax(axis=1)
    seconds = time.perf_counter() - t0
    return summarize(
        test_labels, preds, seconds,
        extra={
            "features": f"TF-IDF, vocabulary {len(vec.vocab_)} terms (min_df={TFIDF_MIN_DF}, train only)",
            "fit_seconds": fit_seconds,
            "notes": "no probability output, so no ECE and no abstain signal",
        },
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def fit_logreg(X: np.ndarray, y: np.ndarray, n_classes: int, lr: float, epochs: int, l2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Full-batch gradient descent on the mean cross-entropy with L2 on the weights."""
    n, d = X.shape
    Y = np.eye(n_classes, dtype=np.float32)[y]
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    for _ in range(epochs):
        P = _softmax(X @ W + b)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def run_logreg(
    train_texts: Sequence[str], train_labels: np.ndarray,
    test_texts: Sequence[str], test_labels: np.ndarray, n_classes: int, seed: int,
) -> Dict[str, Any]:
    """Pick the learning rate on a 10% validation slice of train, then refit on all of train."""
    t0 = time.perf_counter()
    vec = TfidfVectorizer().fit(train_texts)
    X_train = vec.transform(train_texts)

    perm = np.random.default_rng(seed + 1).permutation(len(train_labels))
    n_val = max(1, int(len(perm) * LOGREG_VAL_FRACTION))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    search = []
    best_lr, best_val = None, -1.0
    for lr in LOGREG_LR_GRID:
        W, b = fit_logreg(X_train[fit_idx], train_labels[fit_idx], n_classes, lr, LOGREG_EPOCHS, LOGREG_L2)
        val_acc = float(((X_train[val_idx] @ W + b).argmax(axis=1) == train_labels[val_idx]).mean())
        search.append({"lr": lr, "val_accuracy": val_acc})
        if val_acc > best_val:
            best_lr, best_val = lr, val_acc
    W, b = fit_logreg(X_train, train_labels, n_classes, best_lr, LOGREG_EPOCHS, LOGREG_L2)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test = vec.transform(test_texts)
    logits = X_test @ W + b
    probs = _softmax(logits)
    preds = logits.argmax(axis=1)
    seconds = time.perf_counter() - t0
    return summarize(
        test_labels, preds, seconds, probs=probs,
        extra={
            "features": f"TF-IDF, vocabulary {len(vec.vocab_)} terms (min_df={TFIDF_MIN_DF}, train only)",
            "fit_seconds": fit_seconds,
            "hyperparameters": {
                "optimizer": "full-batch gradient descent",
                "epochs": LOGREG_EPOCHS,
                "l2": LOGREG_L2,
                "lr_grid": list(LOGREG_LR_GRID),
                "lr_selected": best_lr,
                "validation": f"{LOGREG_VAL_FRACTION:.0%} of train, permutation seed+1",
                "validation_search": search,
            },
            "notes": "probabilities are the softmax of the fitted logits",
        },
    )


# ---------------------------------------------------------------------------
# RHEN variant (untrained, optional)
# ---------------------------------------------------------------------------

def run_rhen_untrained(texts: Sequence[str], label_names: List[str], test_labels: np.ndarray) -> Dict[str, Any]:
    """RhenEngine().choice() with the default configuration. Nothing is trained here."""
    try:
        import torch
    except Exception as exc:  # any import failure means the variant is skipped
        return {"status": "skipped", "reason": f"import torch failed: {type(exc).__name__}: {exc}"}
    # The script lives in benchmarks/banking77/, so the repository root is not on sys.path by default.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from rhen import RhenEngine
    except Exception as exc:
        return {"status": "skipped", "reason": f"import rhen failed: {type(exc).__name__}: {exc}"}
    try:
        engine = RhenEngine()
    except Exception as exc:
        return {"status": "skipped", "reason": f"RhenEngine() failed: {type(exc).__name__}: {exc}"}

    print(
        "RHEN variant: UNTRAINED RhenEngine (default configuration, no checkpoint, no training). "
        f"Chance level is 1/{len(label_names)} = {100.0 / len(label_names):.2f}%.",
        file=sys.stderr,
    )
    index = {name: i for i, name in enumerate(label_names)}
    n = len(texts)
    preds = np.full(n, -1, dtype=np.int64)
    probs = np.zeros((n, len(label_names)), dtype=np.float64)
    abstained = np.zeros(n, dtype=bool)
    has_verdict = False
    times = np.zeros(n, dtype=np.float64)
    done = 0
    try:
        for i, text in enumerate(texts):
            t0 = time.perf_counter()
            result = engine.choice(text, RHEN_QUESTION, label_names)
            times[i] = time.perf_counter() - t0
            print(f"[rhen] {i + 1}/{n} {times[i]:.1f}s", file=sys.stderr, flush=True)
            preds[i] = index.get(result.choice_label, -1)
            probs[i] = [float(result.probabilities.get(name, 0.0)) for name in label_names]
            verdict = getattr(result, "abstention_verdict", None)
            if verdict is not None:
                has_verdict = True
                abstained[i] = verdict.decision != "Answer"
            done = i + 1
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"choice() failed after {done} of {n} examples: {type(exc).__name__}: {exc}",
        }

    return summarize(
        test_labels, preds, float(times.sum()), probs=probs,
        abstained=abstained if has_verdict else None,
        extra={
            "trained": False,
            "model": "RhenEngine() default EOSConfig, randomly initialised, no checkpoint, no training",
            "question": RHEN_QUESTION,
            "options": "dataset label names, unmodified",
            "device": str(engine.device),
            "torch_version": torch.__version__,
            "notes": "expected near chance; this measures an untrained model, not a trained RHEN head",
        },
    )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=30, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> Optional[bool]:
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    if status is None:
        return None
    lines = [ln for ln in status.splitlines() if not ln.endswith("benchmarks/banking77/results.json")]
    return bool(lines)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.2f}%"


def markdown_table(variants: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| variant | status | n | top-1 | 95% Wilson CI | ECE (10 bins) | abstain rate | s/example |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, v in variants.items():
        if v["status"] != "ok":
            lines.append(f"| {name} | {v['status']}: {v.get('reason', '')} | | | | | | |")
            continue
        lo, hi = v["wilson95"]
        lines.append(
            f"| {name} | ok | {v['n']} | {_fmt_pct(v['top1_accuracy'])} | "
            f"[{_fmt_pct(lo)}, {_fmt_pct(hi)}] | {_fmt_pct(v['ece_10bin'])} | "
            f"{_fmt_pct(v['abstain_rate'])} | {v['seconds_per_example']:.6f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible Banking77 evaluation: numpy baselines and an untrained RHEN variant.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate a reproducible random subset of N test examples (default: all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for test subsampling and the logistic-regression validation split (default: 0)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help=f"where downloaded split files are cached (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="results JSON path (default: benchmarks/banking77/results.json)")
    parser.add_argument("--rhen-limit", type=int, default=None,
                        help="run the RHEN variant on only the first N of the evaluated examples "
                             "(default: all). One CPU call with 77 options takes minutes, so use this "
                             "for a small RHEN sample.")
    parser.add_argument("--skip-rhen", action="store_true",
                        help="do not run the RHEN variant even if torch and rhen import")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.rhen_limit is not None and args.rhen_limit < 1:
        parser.error("--rhen-limit must be a positive integer")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    data = load_banking77(args.cache_dir)
    label_names: List[str] = data["label_names"]
    n_classes = len(label_names)
    train = data["splits"]["train"]
    test = data["splits"]["test"]

    idx = select_test_indices(len(test["labels"]), args.limit, args.seed)
    test_texts = [test["texts"][i] for i in idx]
    test_labels = test["labels"][idx]
    print(
        f"Banking77: {len(train['labels'])} train, {len(test['labels'])} test, {n_classes} labels; "
        f"evaluating {len(test_labels)} test examples (seed={args.seed}, limit={args.limit})",
        file=sys.stderr,
    )

    variants: Dict[str, Dict[str, Any]] = {}
    variants["majority_class"] = run_majority(train["labels"], test_labels, n_classes)
    variants["tfidf_nearest_centroid"] = run_nearest_centroid(
        train["texts"], train["labels"], test_texts, test_labels, n_classes)
    variants["tfidf_logistic_regression"] = run_logreg(
        train["texts"], train["labels"], test_texts, test_labels, n_classes, args.seed)
    n_rhen = len(test_labels) if args.rhen_limit is None else min(args.rhen_limit, len(test_labels))
    if args.skip_rhen:
        variants["rhen_untrained"] = {"status": "skipped", "reason": "--skip-rhen given"}
    else:
        variants["rhen_untrained"] = run_rhen_untrained(
            test_texts[:n_rhen], label_names, test_labels[:n_rhen])

    results: Dict[str, Any] = {
        "benchmark": "banking77",
        "task": "English intent classification, 77 labels",
        "dataset": {
            "hf_dataset": data["hf_dataset"],
            "hf_revision": data["hf_revision"],
            "loader": data["loader"],
            "splits": {
                "train": {"rows": len(train["labels"]), "file": train["file"], "sha256": train["sha256"]},
                "test": {"rows": len(test["labels"]), "file": test["file"], "sha256": test["sha256"]},
            },
            "num_labels": n_classes,
        },
        "sampling": {
            "sampled": args.limit is not None and args.limit < len(test["labels"]),
            "limit": args.limit,
            "seed": args.seed,
            "test_examples_total": len(test["labels"]),
            "test_examples_evaluated": int(len(test_labels)),
            "rhen_limit": args.rhen_limit,
            "rhen_examples_evaluated": (
                0 if args.skip_rhen else n_rhen
            ),
            "method": "numpy default_rng(seed).choice without replacement, sorted",
        },
        "seed": args.seed,
        "commit": _git(["rev-parse", "HEAD"]),
        "worktree_dirty": _git_dirty(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "platform": platform.platform(),
        },
        "command": "python3 " + shlex.join(sys.argv),
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(markdown_table(variants))
    print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
