import math
import os
import tempfile
import unittest

import numpy as np
import torch

from rhen.v2.calibrate import (
    Calibration, CalibrationMismatch, OnlineRecalibrator, clopper_pearson_upper,
    expected_calibration_error, fit_calibration, fit_isotonic, fit_tau_act,
    fit_temperatures, brier, nll, _softmax,
)
from rhen.v2.selective import (
    accuracy_at_coverage, aurc, coverage_at_error, cross_fit_predictions,
    risk_coverage_curve, selective_finetune,
)


def synth(n, k, t_true, scale, seed, qtype="choice", d=0, shift=0.0):
    """Logits z whose labels are drawn from softmax(z / t_true): true temperature t_true."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n, k + 1) * scale
    p = _softmax(z / t_true)
    u = rng.rand(n, 1)
    lab = (p.cumsum(axis=1) < u).sum(axis=1).clip(max=k)
    recs = []
    for i in range(n):
        r = {"type": qtype, "k": k, "logits": z[i].tolist(), "label_idx": int(lab[i] if lab[i] < k else -2)}
        if d:
            r["q_vec"] = (rng.randn(d) + shift).tolist()
        recs.append(r)
    return recs


def auroc(score, y):
    score, y = np.asarray(score), np.asarray(y)
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class TestMetricsAndBounds(unittest.TestCase):
    def test_clopper_pearson_known(self):
        # one-sided 95 %: 1 - 0.05 ** (1/100)
        self.assertAlmostEqual(clopper_pearson_upper(0, 100), 1 - 0.05 ** 0.01, places=5)
        self.assertAlmostEqual(clopper_pearson_upper(0, 100), 0.0295, places=4)
        # 5 of 100 one-sided 95 %: 0.95 quantile of Beta(6, 95) ~ 0.1023 (binomial cdf(5; 100, 0.1023) = 0.0499)
        self.assertAlmostEqual(clopper_pearson_upper(5, 100), 0.1023, places=3)
        self.assertEqual(clopper_pearson_upper(10, 10), 1.0)

    def test_ece_perfect_and_bad(self):
        rng = np.random.RandomState(0)
        conf = rng.uniform(0.5, 1.0, 20000)
        ok = (rng.rand(20000) < conf).astype(int)
        self.assertLess(expected_calibration_error(conf, ok), 0.02)
        self.assertGreater(expected_calibration_error(np.full(1000, 0.99), np.zeros(1000)), 0.9)
        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        self.assertAlmostEqual(nll(probs, np.array([0, 1])), -(math.log(0.7) + math.log(0.8)) / 2)
        self.assertAlmostEqual(brier(probs, np.array([0, 1])), 0.13)


class TestTemperature(unittest.TestCase):
    def test_recovers_true_temperature(self):
        for t_true in (0.6, 1.0, 2.5):
            recs = synth(8000, 4, t_true, 3.0, seed=1)
            temps, ns = fit_temperatures(recs)
            self.assertAlmostEqual(temps["choice"]["4"] / t_true, 1.0, delta=0.05)
            self.assertEqual(ns["choice"]["4"], 8000)

    def test_group_fallback_and_calibrated_flag(self):
        recs = synth(3000, 4, 2.0, 3.0, 2) + synth(150, 16, 0.7, 3.0, 3)
        cal = fit_calibration(recs, model_hash="m")
        self.assertEqual(cal.temperature("choice", 16), cal.temperature_table["choice"]["all"])
        self.assertNotEqual(cal.temperature("choice", 4), cal.temperature_table["choice"]["all"])
        self.assertEqual(cal.temperature("range", 4), 1.0)
        self.assertTrue(cal.calibrated)
        self.assertIsNotNone(cal.conformal_q("choice", 16))  # 150 >= 100
        small = fit_calibration(synth(50, 4, 1.0, 2.0, 4))
        self.assertFalse(small.calibrated)
        self.assertIsNone(small.conformal_q("choice", 4))
        self.assertFalse(Calibration().calibrated)


class TestConformal(unittest.TestCase):
    def test_group_coverage(self):
        alpha = 0.1
        cal_recs = synth(20000, 4, 2.0, 3.0, 10) + synth(20000, 16, 0.8, 1.5, 11)
        cal = fit_calibration(cal_recs, alpha=alpha)
        for k, t, sc, seed in ((4, 2.0, 3.0, 20), (16, 0.8, 1.5, 21)):
            test = synth(20000, k, t, sc, seed)
            hit, size = 0, 0
            for r in test:
                p = cal.apply("choice", k, r["logits"])
                s = cal.prediction_set("choice", k, p)
                lab = k if r["label_idx"] == -2 else r["label_idx"]
                hit += lab in s
                size += len(s)
            self.assertAlmostEqual(hit / len(test), 1 - alpha, delta=0.02)
        # different difficulty -> different set sizes
        self.assertNotEqual(cal.conformal_q("choice", 4), cal.conformal_q("choice", 16))


class TestIsotonic(unittest.TestCase):
    def test_monotone_and_improves_ece(self):
        # overconfident model: logits too sharp and NOT temperature-fixed (use raw softmax)
        train = synth(10000, 4, 2.0, 3.0, 5)
        test = synth(10000, 4, 2.0, 3.0, 6)
        temps, ns = {"choice": {"all": 1.0}}, {"choice": {"all": 10000}}
        iso = fit_isotonic(train, temps, ns)["choice"]
        self.assertTrue(all(b >= a for a, b in zip(iso["y"], iso["y"][1:])))
        cal = Calibration(isotonic_table={"choice": iso})
        ps = [cal.isotonic("choice", p) for p in np.linspace(0, 1, 200)]
        self.assertTrue(all(b >= a for a, b in zip(ps, ps[1:])))
        conf, ok = [], []
        for r in test:
            lab = 4 if r["label_idx"] == -2 else r["label_idx"]
            p = _softmax(np.array(r["logits"]))
            conf.append(p.max())
            ok.append(int(p.argmax() == lab))
        before = expected_calibration_error(np.array(conf), np.array(ok))
        after = expected_calibration_error(np.array([cal.isotonic("choice", c) for c in conf]), np.array(ok))
        self.assertLess(after, before / 2)


class TestTauAct(unittest.TestCase):
    def test_threshold_found_and_impossible(self):
        rng = np.random.RandomState(0)
        p = rng.uniform(0, 1, 20000)
        ok = (rng.rand(20000) < p).astype(int)
        t = fit_tau_act(p, ok, 0.05)
        self.assertIsNotNone(t)
        self.assertGreater(t, 0.9)
        sel = p >= t
        self.assertLessEqual(clopper_pearson_upper(int((1 - ok[sel]).sum()), int(sel.sum())), 0.05)
        # tighter tier -> higher or equal threshold, or impossible
        t2 = fit_tau_act(p, ok, 0.01)
        self.assertTrue(t2 is None or t2 >= t)
        # impossible: always wrong / too few points for the tolerance
        self.assertIsNone(fit_tau_act(np.full(50, 0.9), np.zeros(50), 0.05))
        self.assertIsNone(fit_tau_act(np.full(20, 0.99), np.ones(20), 0.001))


class TestCalibrationIO(unittest.TestCase):
    def test_round_trip_and_mismatch(self):
        recs = synth(1500, 4, 1.5, 2.0, 7, d=3)
        cal = fit_calibration(recs, model_hash="abc", data_hash="d1", delta_ev=0.1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            cal.save(path)
            back = Calibration.load(path, expected_model_hash="abc")
            self.assertEqual(back.to_dict(), cal.to_dict())
            self.assertEqual(back.temperature("choice", 4), cal.temperature("choice", 4))
            self.assertEqual(back.tau_act("low"), cal.tau_act("low"))
            for key in ("model_hash", "data_hash", "created", "temperature", "isotonic", "conformal_q",
                        "tau_act", "tau_review", "halt_threshold", "delta_ev", "ood", "n_calibration"):
                self.assertIn(key, cal.to_dict())
            with self.assertRaises(CalibrationMismatch):
                Calibration.load(path, expected_model_hash="other")


class TestOOD(unittest.TestCase):
    def test_in_distribution_rate_and_shift(self):
        d = 4
        cal = fit_calibration(synth(6000, 4, 1.5, 3.0, 30, d=d))
        test = synth(6000, 4, 1.5, 3.0, 31, d=d)
        t = cal.temperature("choice", 4)
        from rhen.v2.calibrate import ood_signals
        flagged = 0
        for r in test:
            e, pn = ood_signals(r["logits"], t)
            flagged += bool(cal.ood_flags("choice", 4, e, pn, r["q_vec"]))
        rate = flagged / len(test)
        self.assertLess(rate, 0.05)
        self.assertGreater(rate, 0.003)
        # far shift: flat logits (high energy / null mass) and shifted embeddings
        far = synth(500, 4, 1.5, 0.05, 32, d=d, shift=8.0)
        hits = 0
        for r in far:
            e, pn = ood_signals(r["logits"], t)
            hits += bool(cal.ood_flags("choice", 4, e, pn, r["q_vec"]))
        self.assertGreater(hits / len(far), 0.9)
        # flags name the signals
        e, pn = ood_signals(far[0]["logits"], t)
        flags = cal.ood_flags("choice", 4, e, pn, far[0]["q_vec"])
        self.assertEqual(flags[0], "OOD")
        self.assertIn("mahalanobis", flags)


class TestOnline(unittest.TestCase):
    def _cal(self):
        # alpha small so that 3 consecutive conformal misses are rare on stationary data
        return fit_calibration(synth(10000, 4, 1.5, 3.0, 40, d=2), alpha=0.01)

    def test_quiet_when_stationary(self):
        cal = self._cal()
        rec = OnlineRecalibrator(cal)
        alarms = []
        for r in synth(3000, 4, 1.5, 3.0, 41, d=2):
            alarms += rec.observe("choice", 4, r["logits"], r["label_idx"])
        self.assertEqual(alarms, [])
        self.assertTrue(rec.group_calibrated("choice", 4))
        self.assertEqual(rec.alarms, {})
        # temperature was refitted and stays within the 2x band of the offline value
        t_off = cal.temperature("choice", 4)
        self.assertLess(abs(rec.temperature("choice", 4) / t_off - 1), 0.2)

    def test_alarm_on_drift(self):
        cal = self._cal()
        rec = OnlineRecalibrator(cal)
        # drifted: labels follow a much hotter distribution (true T = 8)
        alarms = []
        for r in synth(1500, 4, 8.0, 3.0, 42, d=2):
            alarms += rec.observe("choice", 4, r["logits"], r["label_idx"])
        self.assertTrue(alarms)
        self.assertFalse(rec.group_calibrated("choice", 4))
        self.assertIn("ece", rec.alarms[("choice", "4")])
        self.assertTrue(cal.group_calibrated("choice", 4))  # offline file unchanged
        rec.reset_alarm("choice", 4)
        self.assertTrue(rec.group_calibrated("choice", 4))

    def test_ood_rate_alarm(self):
        cal = self._cal()
        rec = OnlineRecalibrator(cal)
        alarms = []
        for r in synth(600, 4, 1.5, 3.0, 43, d=2):
            alarms += rec.observe("choice", 4, r["logits"], r["label_idx"], ood_flag=True)
        self.assertIn("ood_rate", alarms)


class TestSelective(unittest.TestCase):
    def setUp(self):
        self._nt = torch.get_num_threads()
        torch.set_num_threads(1)  # tiny tensors: threading only adds contention

    def tearDown(self):
        torch.set_num_threads(self._nt)

    def test_metrics(self):
        conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        ok = np.array([1, 1, 1, 0, 0])
        cov, risk = risk_coverage_curve(conf, ok)
        np.testing.assert_allclose(cov, [0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(risk, [0, 0, 0, 0.25, 0.4])
        self.assertAlmostEqual(aurc(conf, ok), 0.13)
        self.assertAlmostEqual(accuracy_at_coverage(conf, ok, 0.6), 1.0)
        self.assertAlmostEqual(accuracy_at_coverage(conf, ok, 1.0), 0.6)
        self.assertAlmostEqual(coverage_at_error(conf, ok, 0.01), 0.6)
        self.assertEqual(coverage_at_error(conf, 1 - ok, 0.01), 0.0)
        # an informative score beats a random one
        rng = np.random.RandomState(0)
        y = (rng.rand(2000) < 0.7).astype(int)
        self.assertLess(aurc(y + rng.rand(2000) * 0.5, y), aurc(rng.rand(2000), y))

    def test_selective_finetune_unanswerable_half(self):
        torch.manual_seed(0)
        n, d = 4000, 8
        answerable = (torch.rand(n) < 0.5)
        x = torch.randn(n, d)
        x[:, 0] += 2.0 * (answerable.float() * 2 - 1)   # informative feature
        correct = answerable.float()                      # unanswerable -> always wrong
        head = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        selective_finetune(head, x[:3000], correct[:3000], steps=600, lr=1e-2, seed=0)
        with torch.no_grad():
            p = torch.sigmoid(head(x[3000:]).reshape(-1)).numpy()
        self.assertGreater(auroc(p, correct[3000:].numpy()), 0.9)

    def test_cross_fit_never_uses_own_fold(self):
        examples = list(range(103))
        fitted = []

        def fit_fn(train):
            fitted.append(set(train))
            return len(fitted) - 1

        def predict_fn(model, held):
            self.assertFalse(set(held) & fitted[model])
            return [((x,), x % 2 == 0) for x in held]

        out = cross_fit_predictions(examples, fit_fn, predict_fn, folds=5, seed=3)
        self.assertEqual(len(fitted), 5)
        self.assertEqual([f[0][0] for f in out], examples)  # aligned, every example predicted once
        self.assertEqual(out[4], ((4,), True))


if __name__ == "__main__":
    unittest.main()
