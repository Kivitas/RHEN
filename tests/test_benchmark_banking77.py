"""Offline tests for benchmarks/banking77/run_eval.py.

No network access and no torch: the synthetic data is generated in this file, and the
script only imports numpy at module load.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "banking77" / "run_eval.py"


@pytest.fixture(scope="module")
def run_eval():
    spec = importlib.util.spec_from_file_location("run_eval_banking77", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEYWORDS = ["zephyr", "quasar", "nebula", "vortex", "glacier", "lantern"]
FILLER = ["please help", "thanks a lot", "i need this", "can you check", "for my account"]


def make_synthetic(n: int = 40, n_labels: int = len(KEYWORDS)):
    """Six labels, each marked by its own keyword. Labels cycle so every label appears in both splits."""
    texts, labels = [], []
    for idx in range(n):
        y = idx % n_labels
        texts.append(f"{KEYWORDS[y]} {FILLER[idx % len(FILLER)]} {KEYWORDS[y]}")
        labels.append(y)
    return texts, np.asarray(labels, dtype=np.int64)


def test_wilson_interval_matches_hand_value_5_of_10(run_eval):
    # z = 1.959964, n = 10, p = 0.5: center = 0.5, half-width = z*sqrt(0.025 + z^2/400)/(1 + z^2/10)
    # = 0.263407, so the interval is about [0.236593, 0.763407].
    lo, hi = run_eval.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-12)


def test_wilson_interval_zero_successes_has_closed_form_upper_bound(run_eval):
    # For k = 0 the upper bound is z^2 / (n + z^2) = 3.841459 / 13.841459 = 0.277532.
    lo, hi = run_eval.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(3.841458820694124 / 13.841458820694124, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-4)


def test_wilson_interval_all_successes_is_symmetric(run_eval):
    lo, hi = run_eval.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12)
    assert lo == pytest.approx(1.0 - 0.277532, abs=1e-5)


def test_ece_hand_computed_two_bin_case(run_eval):
    # Five predictions at confidence 0.9 (bin 9) with 3/5 correct: gap 0.3, weight 0.5 -> 0.15.
    # Five predictions at confidence 0.3 (bin 3) with 1/5 correct: gap 0.1, weight 0.5 -> 0.05.
    conf = [0.9] * 5 + [0.3] * 5
    correct = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    assert run_eval.expected_calibration_error(conf, correct) == pytest.approx(0.2, abs=1e-12)


def test_ece_confidence_one_falls_in_last_bin(run_eval):
    # Confidence 1.0 must be counted in the last bin, not overflow to bin 10.
    # Two predictions at 1.0 with accuracy 0.5 -> ECE = |0.5 - 1.0| = 0.5.
    assert run_eval.expected_calibration_error([1.0, 1.0], [1, 0]) == pytest.approx(0.5, abs=1e-12)


def test_ece_is_zero_for_perfectly_calibrated_bin(run_eval):
    assert run_eval.expected_calibration_error([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.0, abs=1e-12)


def test_majority_baseline_predicts_majority_label(run_eval):
    train_labels = np.array([2, 2, 2, 0, 0, 1, 1, 5], dtype=np.int64)
    test_labels = np.array([2, 2, 0, 1, 5, 2], dtype=np.int64)
    out = run_eval.run_majority(train_labels, test_labels, n_classes=6)
    assert out["status"] == "ok"
    assert out["majority_label_index"] == 2
    assert out["correct"] == 3  # three test labels equal 2
    assert out["n"] == 6


def test_majority_baseline_breaks_ties_toward_smallest_label(run_eval):
    out = run_eval.run_majority(np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64), n_classes=2)
    assert out["majority_label_index"] == 0


def test_nearest_centroid_is_perfect_on_separable_synthetic_data(run_eval):
    texts, labels = make_synthetic()
    train_texts, test_texts = texts[:30], texts[30:]
    train_labels, test_labels = labels[:30], labels[30:]
    assert set(test_labels.tolist()) == set(range(len(KEYWORDS)))
    out = run_eval.run_nearest_centroid(train_texts, train_labels, test_texts, test_labels, n_classes=len(KEYWORDS))
    assert out["top1_accuracy"] == 1.0
    assert out["correct"] == len(test_texts)
    assert out["ece_10bin"] is None  # no probability output


def test_logistic_regression_outputs_probabilities_and_is_perfect_on_separable_data(run_eval):
    texts, labels = make_synthetic()
    out = run_eval.run_logreg(texts[:30], labels[:30], texts[30:], labels[30:], n_classes=len(KEYWORDS), seed=0)
    assert out["top1_accuracy"] == 1.0
    assert out["ece_10bin"] is not None and 0.0 <= out["ece_10bin"] <= 1.0
    assert out["hyperparameters"]["lr_selected"] in run_eval.LOGREG_LR_GRID


def test_tfidf_vocabulary_is_fit_on_training_text_only(run_eval):
    vec = run_eval.TfidfVectorizer(min_df=2).fit(["alpha beta", "alpha gamma", "delta"])
    assert set(vec.vocab_) == {"alpha"}  # beta, gamma, delta appear in fewer than two training docs
    X = vec.transform(["alpha beta", "delta"])
    assert X.shape == (2, 1)
    assert X[1, 0] == 0.0


def test_test_subsampling_is_reproducible_and_seed_dependent(run_eval):
    a = run_eval.select_test_indices(3080, 100, seed=0)
    b = run_eval.select_test_indices(3080, 100, seed=0)
    c = run_eval.select_test_indices(3080, 100, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 100 and len(set(a.tolist())) == 100
    assert np.all(np.diff(a) > 0)  # sorted, original order kept


def test_test_subsampling_without_limit_returns_every_example(run_eval):
    assert np.array_equal(run_eval.select_test_indices(7, None, seed=0), np.arange(7))
    assert np.array_equal(run_eval.select_test_indices(7, 50, seed=0), np.arange(7))


def test_help_runs_without_network_or_torch():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--limit" in proc.stdout
    assert "--seed" in proc.stdout
    assert "--skip-rhen" in proc.stdout
