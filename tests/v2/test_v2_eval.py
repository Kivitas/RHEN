"""Evaluation metrics, probes and the suite / ablation / latency scripts (T-701 to T-703)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rhen.v2 import eval as ev  # noqa: E402
from rhen.v2.config import preset  # noqa: E402
from rhen.v2.model import Rhen2Model  # noqa: E402
from rhen.v2.packing import NULL_LABEL, QuestionSpec, Rhen2Example, Rhen2Packer  # noqa: E402

STATE = "the red box holds 42 apples and the blue box holds 7 pears"


def six_type_examples():
    c0 = STATE.index("42")
    qs = [
        QuestionSpec("choice", "Which box holds apples?", ["red", "blue", "green"], label=0),
        QuestionSpec("choice", "Which colour is the car?", ["red", "blue"], label=NULL_LABEL),
        QuestionSpec("noul", "The blue box holds pears.", label=True),
        QuestionSpec("score", "How many apples?", ["none", "few", "many"], label=2),
        QuestionSpec("multi", "Which boxes exist?", ["red", "blue", "green"], label=[0, 1]),
        QuestionSpec("range", "Apples in the red box?", lo=0.0, hi=100.0, label=42.0),
        QuestionSpec("extract", "How many apples?", label=(c0, c0 + 2)),
    ]
    return [Rhen2Example(STATE, qs), Rhen2Example(STATE + " and more", qs[:3])]


class TestPureMetrics(unittest.TestCase):
    def test_wilson(self):
        lo, hi = ev.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(ev.wilson_interval(0, 10)[0], 0.0)

    def test_ece_equal_mass(self):
        self.assertAlmostEqual(ev.ece_equal_mass([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=2), 0.15)
        self.assertAlmostEqual(ev.ece_equal_mass([1.0, 1.0], [1, 1], 15), 0.0)

    def test_brier_nll(self):
        self.assertAlmostEqual(ev.brier_score([[0.7, 0.2, 0.1]], [0]), 0.09 + 0.04 + 0.01)
        self.assertAlmostEqual(ev.nll([[0.5, 0.5]], [1]), np.log(2))

    def test_auroc_with_ties(self):
        self.assertAlmostEqual(ev.auroc([2, 3], [1, 2]), 0.875)
        self.assertAlmostEqual(ev.auroc([1, 1], [1, 1]), 0.5)
        self.assertAlmostEqual(ev.auroc([5, 6], [1, 2]), 1.0)

    def test_selective(self):
        conf, corr = [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1]
        self.assertAlmostEqual(ev.aurc(conf, corr), (0 + 0 + 1 / 3 + 1 / 4) / 4)
        self.assertAlmostEqual(ev.accuracy_at_coverage(conf, corr, 0.5), 1.0)
        self.assertAlmostEqual(ev.accuracy_at_coverage(conf, corr, 0.75), 2 / 3)
        self.assertAlmostEqual(ev.coverage_at_error(conf, corr, 0.01), 0.5)

    def test_conformal_coverage(self):
        recs = [{"type": "choice", "k": 3, "label": 0, "probs": p} for p in
                ([0.6, 0.3, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05], [0.5, 0.4, 0.05, 0.05], [0.9, 0.05, 0.03, 0.02])]
        # nonconformity 1-p[label] = .4, .8, .5, .1; q=0.5 -> covers 3 of 4
        rep = ev.conformal_report(recs, {0.1: {("choice", 4): 0.5}})["0.1"]
        g = rep["groups"]["choice/k4"]
        self.assertEqual(g["n"], 4)
        self.assertAlmostEqual(g["coverage"], 0.75)
        self.assertAlmostEqual(g["mean_set_size"], 1.0)
        self.assertEqual(ev.conformal_quantile([0.1, 0.4, 0.5, 0.8], 0.5), 0.5)
        self.assertEqual(ev.conformal_quantile([0.1], 0.05), float("inf"))

    def test_token_f1(self):
        self.assertAlmostEqual(ev.token_f1("42 apples", "42"), 2 * 0.5 * 1 / 1.5)
        self.assertEqual(ev.token_f1("", ""), 1.0)
        self.assertEqual(ev.token_f1("a", ""), 0.0)


class TestModelEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = preset("r2-tiny")
        cls.model = Rhen2Model(cls.cfg).eval()
        cls.pk = Rhen2Packer(cls.cfg)
        cls.ex = six_type_examples()

    def test_all_types_and_null_labels(self):
        recs = ev.evaluate_model(self.model, self.pk, self.ex, families=["a", "b"])
        self.assertEqual(len(recs), 10)
        types = {r["type"] for r in recs}
        self.assertEqual(types, {"choice", "noul", "score", "multi", "range", "extract"})
        null = [r for r in recs if r["type"] == "choice" and r["label"] == r["k"]][0]
        self.assertEqual(null["correct"], int(np.argmax(null["probs"])) == null["k"])
        self.assertTrue(all(abs(sum(r["probs"]) - 1) < 1e-4 for r in recs if r["type"] in ev.DISCRETE))
        self.assertEqual([r["label"] for r in recs if r["type"] == "extract"], ["42"])
        s = ev.summarise(recs, {"weak": ["a"], "none": ["zzz"]})
        self.assertIn("range", s["per_type"])
        self.assertIn("band_10_90_coverage", s["per_type"]["range"])
        self.assertEqual(s["per_family"]["b"]["n_questions"], 3)
        self.assertEqual(s["weak_spots"]["none"]["n"], 0)
        self.assertIn("ece_15bin", s["pooled_discrete"])

    def test_shuffle_and_reorder_flip_rates_are_zero(self):
        res = ev.robustness_probes(self.model, self.pk, self.ex, paraphrase_fn=lambda t: "Please answer: " + t)
        self.assertGreater(res["option_shuffle"]["n"], 0)
        self.assertEqual(res["option_shuffle"]["flips"], 0)
        self.assertGreater(res["question_reorder"]["n"], 0)
        self.assertEqual(res["question_reorder"]["flips"], 0)
        self.assertIsNotNone(res["paraphrase"]["flip_rate"])
        self.assertIn("accuracy_drop", res["distractor"])

    def test_noul_consistency(self):
        a = Rhen2Example(STATE, [QuestionSpec("noul", "The red box holds apples.")])
        b = Rhen2Example(STATE, [QuestionSpec("noul", ev.default_negation("The red box holds apples."))])
        r = ev.noul_consistency(self.model, self.pk, [(a, b)])
        self.assertEqual(r["n"], 1)
        self.assertLess(r["mean_sum_residual"], 1e-4)
        self.assertGreaterEqual(r["mean_abs_diff"], 0.0)

    def test_ood_report(self):
        recs = ev.evaluate_model(self.model, self.pk, self.ex)
        rep = ev.ood_report(recs, recs)
        self.assertAlmostEqual(rep["energy"], 0.5)


class TestScripts(unittest.TestCase):
    def test_run_suite_untrained(self):
        from benchmarks.suite import run_suite
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            self.assertEqual(run_suite.main(["--synthetic", "v1", "--n", "12", "--n-train", "24", "--out", str(out)]), 0)
            data = json.loads(out.read_text())
        self.assertIs(data["trained"], False)
        for key in ("commit", "worktree_dirty", "seed", "environment", "command", "reference_points", "splits"):
            self.assertIn(key, data)
        self.assertIn("vendor-reported", data["reference_points"]["label"])
        t = data["splits"]["test"]
        self.assertEqual(len(t["model"]["pooled_discrete"]["wilson95"]), 2)
        self.assertEqual(t["robustness"]["option_shuffle"]["flips"], 0)
        self.assertIn("tfidf_nearest_centroid", t["baselines"])
        self.assertIn("ood_auroc_vs_test", data["splits"]["test_ood"])

    def test_run_ablations_tiny(self):
        from benchmarks.suite import run_ablations
        with tempfile.TemporaryDirectory() as d:
            argv = ["--steps", "3", "--seeds", "0", "--variants", "full,no_deep_supervision", "--n-train", "16",
                    "--n-eval", "8", "--out-dir", d]
            self.assertEqual(run_ablations.main(argv), 0)
            data = json.loads((Path(d) / "ablations.json").read_text())
            self.assertTrue((Path(d) / "ablations.md").read_text().startswith("Tiny-budget"))
        self.assertEqual(set(data["variants"]), {"full", "no_deep_supervision"})
        self.assertIn("mean", data["variants"]["full"]["accuracy"])

    def test_run_latency_tiny(self):
        from benchmarks.latency import run_latency
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "l.json"
            argv = ["--state-lengths", "16", "--ks", "2", "--efforts", "low", "--warmup", "1", "--repeats", "3",
                    "--devices", "cpu", "--out", str(out)]
            self.assertEqual(run_latency.main(argv), 0)
            data = json.loads(out.read_text())
        row = data["results"]["cpu"][0]
        self.assertGreater(row["cold"]["p50_ms"], 0)
        self.assertIn("targets", data)
        self.assertIs(data["trained"], False)


if __name__ == "__main__":
    unittest.main()
