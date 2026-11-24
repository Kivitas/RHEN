"""Tests for bank77corpus.py, configs/banking77_l4_gpu.json, and copula breach probability."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import bank77corpus
from rhen.decision_pipeline import DecisionPolicy, ForecastAwareDecider
from rhen.forecasting import Forecast, NaiveQuantileForecaster
from rhen.research.data import load_decision_jsonl
from rhen.research.train import TrainConfig
from rhen.rhen import ANSWER, AbstentionVerdict, ChoiceResult

REPO_ROOT = Path(__file__).resolve().parents[1]
L4_CONFIG_PATH = REPO_ROOT / "configs" / "banking77_l4_gpu.json"
ROW = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stub_choice(probabilities: dict[str, float]) -> ChoiceResult:
    labels = list(probabilities)
    return ChoiceResult(
        question="q",
        choice_index=0,
        choice_label=labels[0],
        probabilities=dict(probabilities),
        raw_probabilities=[probabilities[k] for k in labels],
        confidence=0.9,
        temperature=1.0,
        abstention_verdict=AbstentionVerdict(decision=ANSWER),
    )


class _StubEngine:
    def __init__(self, result: ChoiceResult) -> None:
        self.result = result

    def choice(self, state, question, options):
        return self.result


# ---------------------------------------------------------------------------
# 1. Banking77 corpus builder (synthetic offline tests across all modes)
# ---------------------------------------------------------------------------


def test_root_reexports_and_77_unique_intent_labels():
    assert len(bank77corpus.INTENT_LABELS) == 77
    assert len(set(bank77corpus.INTENT_LABELS)) == 77
    assert "activate_my_card" in bank77corpus.INTENT_LABELS
    assert "reverted_card_payment?" in bank77corpus.INTENT_LABELS
    assert callable(bank77corpus.load_banking77)
    assert callable(bank77corpus.build_corpus)
    assert callable(bank77corpus.main)


@pytest.mark.parametrize("mode", ["full77", "hard_negatives", "mixed"])
def test_build_corpus_synthetic_all_modes_and_load_decision_jsonl(tmp_path: Path, mode: str):
    out_dir = tmp_path / f"b77_{mode}"
    manifest = bank77corpus.build_corpus(
        out_dir=out_dir,
        mode=mode,
        num_options=16,
        val_fraction=0.2,
        holdout_ood_intents=7,
        permute_options=True,
        seed=42,
        synthetic=True,
        synthetic_train_per_class=5,
        synthetic_test_per_class=2,
    )

    # Expected files all exist and match manifest SHA-256 digests
    manifest_path = out_dir / "corpus_manifest.json"
    assert manifest_path.is_file()
    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert disk_manifest == manifest
    assert manifest["mode"] == mode
    assert manifest["num_labels_total"] == 77
    assert len(manifest["ood_intents"]) == 7
    assert len(manifest["id_intents"]) == 70
    assert "rhen-train --config configs/banking77_l4_gpu.json" in manifest["commands"]["train_l4_gpu"]
    assert "rhen-eval" in manifest["commands"]["eval_l4_gpu"]

    ood_set = set(manifest["ood_intents"])
    loaded = {}
    for split in ("train", "val", "test", "test_ood"):
        split_path = out_dir / f"{split}.jsonl"
        assert split_path.is_file()
        assert manifest["splits"][split]["sha256"] == _sha256(split_path)
        examples = load_decision_jsonl(str(split_path))
        assert len(examples) == manifest["splits"][split]["num_examples"]
        assert len(examples) > 0
        loaded[split] = examples

    # Held-out OOD intents never appear as ground-truth labels in train/val/test
    for split in ("train", "val", "test"):
        for ex in loaded[split]:
            choice_q = ex.questions[0]
            assert choice_q.primitive == "choice"
            true_label = choice_q.options[choice_q.label]
            assert true_label not in ood_set

    # Every test_ood example belongs to a held-out OOD intent and carries all 77 options
    for ex in loaded["test_ood"]:
        choice_q = ex.questions[0]
        assert len(choice_q.options) == 77
        assert choice_q.options[choice_q.label] in ood_set

    # Mode-specific checks on train.jsonl
    train_examples = loaded["train"]
    if mode == "full77":
        option_orders = set()
        for ex in train_examples:
            assert len(ex.questions) == 1
            q = ex.questions[0]
            assert q.primitive == "choice"
            assert len(q.options) == 77
            assert set(q.options) == set(bank77corpus.INTENT_LABELS)
            option_orders.add(tuple(q.options))
        # Deterministic option permutation produces diverse orderings across examples
        assert len(option_orders) > 1
    elif mode == "hard_negatives":
        for ex in train_examples:
            assert len(ex.questions) == 1
            q = ex.questions[0]
            assert q.primitive == "choice"
            assert len(q.options) == 16
            assert len(set(q.options)) == 16
        # By default val/test keep all 77 options
        assert all(len(ex.questions[0].options) == 77 for ex in loaded["val"])
        assert all(len(ex.questions[0].options) == 77 for ex in loaded["test"])
    elif mode == "mixed":
        choice_sizes = set()
        noul_labels = set()
        for ex in train_examples:
            assert len(ex.questions) == 3
            choice_q, pos_noul, neg_noul = ex.questions
            assert choice_q.primitive == "choice"
            choice_sizes.add(len(choice_q.options))
            assert pos_noul.primitive == "noul" and pos_noul.label is True
            assert neg_noul.primitive == "noul" and neg_noul.label is False
            assert pos_noul.question.startswith("Is the customer's banking intent '")
            assert neg_noul.question.startswith("Is the customer's banking intent '")
            noul_labels.add(pos_noul.label)
            noul_labels.add(neg_noul.label)
        assert choice_sizes == {16, 77}
        assert noul_labels == {True, False}


def test_build_corpus_zero_holdout_generates_off_domain_ood_queries(tmp_path: Path):
    out_dir = tmp_path / "b77_all77"
    manifest = bank77corpus.build_corpus(
        out_dir=out_dir,
        mode="full77",
        holdout_ood_intents=0,
        seed=7,
        synthetic=True,
        synthetic_train_per_class=4,
        synthetic_test_per_class=2,
    )
    assert manifest["ood_strategy"] == "out_of_domain_queries"
    assert manifest["ood_intents"] == []
    assert len(manifest["id_intents"]) == 77

    train_ex = load_decision_jsonl(str(out_dir / "train.jsonl"))
    val_ex = load_decision_jsonl(str(out_dir / "val.jsonl"))
    test_ex = load_decision_jsonl(str(out_dir / "test.jsonl"))
    ood_ex = load_decision_jsonl(str(out_dir / "test_ood.jsonl"))

    assert len(train_ex) + len(val_ex) == 77 * 4
    assert len(test_ex) == 77 * 2
    assert len(ood_ex) == len(bank77corpus.OOD_QUERIES)
    assert all(ex.questions[0].metadata.get("is_ood") is True for ex in ood_ex)


def test_build_corpus_hard_negatives_eval_match_k(tmp_path: Path):
    out_dir = tmp_path / "b77_hn_k"
    bank77corpus.build_corpus(
        out_dir=out_dir,
        mode="hard_negatives",
        num_options=12,
        eval_match_k=True,
        seed=3,
        synthetic=True,
        synthetic_train_per_class=4,
        synthetic_test_per_class=2,
    )
    for split in ("train", "val", "test"):
        exs = load_decision_jsonl(str(out_dir / f"{split}.jsonl"))
        assert all(len(ex.questions[0].options) == 12 for ex in exs)


def test_cli_root_bank77corpus_synthetic_runs_cleanly(tmp_path: Path):
    out_dir = tmp_path / "cli_smoke"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bank77corpus.py"),
            "--synthetic",
            "--out-dir",
            str(out_dir),
            "--mode",
            "mixed",
            "--synthetic-train-per-class",
            "3",
            "--synthetic-test-per-class",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "corpus_manifest.json" in proc.stdout
    assert "rhen-train --config configs/banking77_l4_gpu.json" in proc.stdout
    assert (out_dir / "corpus_manifest.json").is_file()


# ---------------------------------------------------------------------------
# 2. L4 GPU TrainConfig validation
# ---------------------------------------------------------------------------


def test_banking77_l4_gpu_config_validates_with_train_config():
    assert L4_CONFIG_PATH.is_file()
    raw = json.loads(L4_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = TrainConfig.from_dict(raw)
    cfg.validate()

    assert cfg.device == "cuda"
    assert cfg.precision == "bf16"
    assert cfg.preset == "small"
    assert cfg.decision_train == "data/banking77/train.jsonl"
    assert cfg.decision_val == "data/banking77/val.jsonl"
    assert cfg.decision_batch_size == 16
    assert cfg.grad_accum_steps == 2
    assert cfg.optimizer == "muon"
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.muon_lr == pytest.approx(0.02)
    assert cfg.schedule == "wsd"
    assert cfg.warmup_steps == 200
    assert cfg.max_steps == 3000
    assert cfg.order_consistency_coef == pytest.approx(0.1)
    assert cfg.eval_every == 250
    assert cfg.save_every == 500
    assert cfg.run_dir == "runs/banking77_l4"


# ---------------------------------------------------------------------------
# 3. Forecast.copula_breach_prob, Forecast.uncertainty(scale_floor), pipeline
# ---------------------------------------------------------------------------


def test_copula_breach_prob_respects_bounds_and_decreases_monotonically_with_rho():
    # 8-step horizon where each step has P(Y_t > 16.5) = 0.25 -> lower=0.25, upper=1.0
    fc = Forecast("demand", 8, [14.0] * 8, [ROW] * 8)
    thr = 16.5
    lower = fc.prob_exceeds(thr, step=None)
    upper = min(1.0, sum(fc.step_exceedance_probs(thr)))
    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(1.0)

    rhos = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
    probs = [fc.copula_breach_prob(thr, rho=r, num_samples=4000, seed=0) for r in rhos]

    for p in probs:
        assert lower <= p <= upper

    # Strictly higher positive autocorrelation yields monotonically non-increasing joint breach prob
    for earlier, later in zip(probs, probs[1:]):
        assert earlier >= later

    # At rho=0.0 (independence), theoretical P = 1 - (1 - 0.25)^8 = 0.89989
    assert probs[0] == pytest.approx(1.0 - (0.75 ** 8), abs=0.03)
    # At rho=0.99, close to comonotonic lower bound 0.25; at rho=1.0, exact lower bound
    assert probs[-1] < probs[0] - 0.35
    assert fc.copula_breach_prob(thr, rho=1.0, num_samples=100, seed=0) == pytest.approx(lower)


def test_copula_breach_prob_edge_cases_and_validation():
    fc_one = Forecast("one", 1, [14.0], [ROW])
    assert fc_one.copula_breach_prob(16.0) == pytest.approx(fc_one.prob_exceeds(16.0))

    fc_multi = Forecast("multi", 4, [14.0] * 4, [ROW] * 4)
    # Extreme thresholds: 0 and 1 exact
    assert fc_multi.copula_breach_prob(100.0) == 0.0
    assert fc_multi.copula_breach_prob(0.0) == 1.0
    # Deterministic with same seed
    p1 = fc_multi.copula_breach_prob(16.0, rho=0.7, num_samples=1000, seed=123)
    p2 = fc_multi.copula_breach_prob(16.0, rho=0.7, num_samples=1000, seed=123)
    assert p1 == p2

    # Validation errors
    with pytest.raises(ValueError):
        fc_multi.copula_breach_prob(16.0, rho=1.5)
    with pytest.raises(ValueError):
        fc_multi.copula_breach_prob(16.0, num_samples=0)
    with pytest.raises(ValueError):
        Forecast("no_q", 2, [1.0, 2.0]).copula_breach_prob(1.5)


def test_forecast_uncertainty_scale_floor():
    # Zero-crossing median: default floor produces huge value; scale_floor=1.0 keeps it bounded.
    zero_row = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    fc = Forecast("zero_cross", 1, [0.0], [zero_row])
    assert fc.uncertainty() == pytest.approx(8.0 / 1e-12)
    assert fc.uncertainty(scale_floor=1.0) == pytest.approx(8.0)
    assert fc.uncertainty(scale_floor=2.0) == pytest.approx(4.0)

    # Non-zero median preserves exact default behavior
    fc_normal = Forecast("demand", 1, [14.0], [ROW])
    assert fc_normal.uncertainty() == pytest.approx(8.0 / 14.0)
    assert fc_normal.uncertainty(scale_floor=1.0) == pytest.approx(8.0 / 14.0)

    with pytest.raises(ValueError):
        fc.uncertainty(scale_floor=0.0)
    with pytest.raises(ValueError):
        fc.uncertainty(scale_floor=-1.0)


def test_decision_pipeline_exposes_breach_prob_copula_alongside_bounds():
    engine = _StubEngine(_stub_choice({"restock": 0.85, "hold": 0.15}))
    policy = DecisionPolicy(utility=lambda o, f: 1.0, max_forecast_uncertainty=math.inf)
    decider = ForecastAwareDecider(engine, NaiveQuantileForecaster(), policy=policy)

    series = {
        "demand": [100.0, 103.0, 99.0, 104.0, 98.0, 105.0, 97.0, 106.0, 96.0, 107.0],
        "latency": [50.0, 52.0, 49.0, 53.0, 48.0, 54.0, 47.0, 55.0, 46.0, 56.0],
    }
    decision = decider.decide(
        state="Ops snapshot",
        question="Action?",
        options=["restock", "hold"],
        series=series,
        horizon=6,
        thresholds={"demand": 110.0, "latency": 58.0},
    )

    feats = decision.features
    for name in ("demand", "latency"):
        s = feats["series"][name]
        assert s["breach_prob_copula"] is not None
        assert s["breach_prob_step_max_lower"] <= s["breach_prob_copula"] <= s["breach_prob_any_step_upper"]

    assert feats["breach_prob_copula"] is not None
    assert feats["breach_prob_step_max_lower"] <= feats["breach_prob_copula"] <= feats["breach_prob_any_step_upper"]

    # Without thresholds, copula breach feature is None
    no_thr = decider.decide(
        state="Ops snapshot",
        question="Action?",
        options=["restock", "hold"],
        series={"demand": series["demand"]},
        horizon=4,
    )
    assert no_thr.features["breach_prob_copula"] is None
    assert no_thr.features["series"]["demand"]["breach_prob_copula"] is None
