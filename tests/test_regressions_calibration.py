"""
Regression tests for calibration and monitoring honesty (meta_cognitive_monitor.py).

The decision-quality path must not invent labels, confidences or energies:
- ECE counts only labeled decisions (there is no default ground truth of 1.0).
- Batch analysis does not feed the ECE bins.
- Missing probabilities are reported (probabilities_missing), never fabricated as
  [c, 1 - c]; energy is computed only from explicit scores or logits.
- Self-correction attaches flags only; a stated confidence is never rewritten.
- Conformal sets report whether the engine was calibrated.
- The learned global temperature is gone.
"""

import asyncio
import copy
import unittest

import torch

from config import EOSConfig
from meta_cognitive_monitor import (
    AdaptiveConformalEngine,
    ConformalPredictionEngine,
    MetaCognitiveMonitor,
)

LABELED = [
    {"confidence": 0.95, "ground_truth": 1},
    {"confidence": 0.85, "ground_truth": 1},
    {"confidence": 0.70, "ground_truth": 0},
    {"confidence": 0.40, "ground_truth": 0},
    {"confidence": 0.10, "ground_truth": 0},
]


def _monitor() -> MetaCognitiveMonitor:
    return MetaCognitiveMonitor(EOSConfig())


class TestEceCountsOnlyLabeledDecisions(unittest.TestCase):

    def test_unlabeled_entries_leave_ece_unchanged(self):
        monitor = _monitor()
        labeled_only = monitor.compute_ece(LABELED, n_bins=10)
        mixed = LABELED + [
            {"confidence": 0.99},                        # no label at all
            {"confidence": 0.60, "ground_truth": None},  # None is not a label
            {"confidence": 0.20},
        ]
        self.assertAlmostEqual(monitor.compute_ece(mixed, n_bins=10), labeled_only, places=12)

    def test_no_usable_decisions_gives_zero(self):
        monitor = _monitor()
        self.assertEqual(monitor.compute_ece([]), 0.0)
        self.assertEqual(monitor.compute_ece([{"confidence": 0.9}, {"confidence": 0.1}]), 0.0)

    def test_missing_confidence_is_not_defaulted(self):
        monitor = _monitor()
        labeled_only = monitor.compute_ece(LABELED)
        self.assertAlmostEqual(monitor.compute_ece(LABELED + [{"ground_truth": 0}]), labeled_only, places=12)

    def test_single_labeled_decision_value(self):
        monitor = _monitor()
        # One decision: bin confidence 0.9, accuracy 0, so ECE = |0 - 0.9| = 0.9.
        self.assertAlmostEqual(monitor.compute_ece([{"confidence": 0.9, "ground_truth": 0}]), 0.9, places=12)

    def test_out_of_range_and_non_finite_confidences(self):
        monitor = _monitor()
        # A negative confidence is clamped into the lowest bin instead of wrapping into the top bin.
        self.assertEqual(monitor.compute_ece([{"confidence": -0.15, "ground_truth": 0}]), 0.0)
        self.assertEqual(monitor.compute_ece([{"confidence": 1.7, "ground_truth": 1}]), 0.0)
        # Non-finite confidences are not counted.
        labeled_only = monitor.compute_ece(LABELED)
        with_nan = LABELED + [{"confidence": float("nan"), "ground_truth": 1}]
        self.assertAlmostEqual(monitor.compute_ece(with_nan), labeled_only, places=12)


class TestBatchAnalysisLeavesCalibrationBinsAlone(unittest.TestCase):

    def test_analysis_does_not_change_ece_or_bins(self):
        monitor = _monitor()
        monitor.update_calibration_from_feedback_sync("d1", True, 0.9)
        monitor.update_calibration_from_feedback_sync("d2", False, 0.8)
        ece_before = monitor.compute_ece()
        bins_before = {k: list(v) for k, v in monitor.bin_predictions.items()}

        monitor.analyze_decision_quality_sync(
            [{"confidence": 0.2}, {"confidence": 0.55}, {"confidence": 0.95}], context=None
        )

        self.assertEqual(monitor.compute_ece(), ece_before)
        self.assertEqual(monitor.bin_predictions, bins_before)

    def test_analysis_keeps_distribution_and_recommendations(self):
        analysis = _monitor().analyze_decision_quality_sync(
            [{"confidence": 0.2}, {"confidence": 0.55}, {"confidence": 0.95}], context=None
        )
        self.assertEqual(analysis["total_decisions"], 3)
        self.assertEqual(set(analysis["confidence_distribution"]), {"0.2-0.3", "0.5-0.6", "0.9-1.0"})
        self.assertEqual(sum(v["count"] for v in analysis["confidence_distribution"].values()), 3)
        self.assertTrue(any("High aggregate uncertainty" in r for r in analysis["recommendations"]))

    def test_analysis_on_empty_bins_keeps_ece_zero(self):
        monitor = _monitor()
        monitor.analyze_decision_quality_sync([{"confidence": 0.99}, {"confidence": 0.01}], context=None)
        self.assertEqual(monitor.compute_ece(), 0.0)
        self.assertEqual(sum(len(v) for v in monitor.bin_predictions.values()), 0)


class TestMonitorWithoutProbabilities(unittest.TestCase):

    def test_missing_probabilities_are_reported_not_fabricated(self):
        out = _monitor().monitor_decision_sync({"mode": "choice", "confidence": 0.97}, context=None)
        self.assertIs(out["probabilities_missing"], True)
        self.assertNotIn("conformal_set", out)
        self.assertNotIn("adaptive_conformal_set", out)
        self.assertNotIn("energy_ood", out)
        self.assertEqual(out["uncertainty"], {})  # no [c, 1 - c] distribution was assumed
        self.assertEqual(out["confidence"], 0.97)
        self.assertNotIn("meta_cognitive_correction", out)

    def test_half_confidence_does_not_fabricate_ambiguity(self):
        # [c, 1 - c] = [0.5, 0.5] used to look like an ambiguous set and trigger escalation.
        out = _monitor().monitor_decision_sync({"confidence": 0.5}, context=None)
        self.assertIs(out["probabilities_missing"], True)
        self.assertNotIn("meta_cognitive_correction", out)

    def test_scores_without_probabilities_give_energy_but_no_conformal(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.9, "logits": [6.0, 1.0, 0.5, -2.0]}, context=None
        )
        self.assertIs(out["probabilities_missing"], True)
        self.assertNotIn("conformal_set", out)
        self.assertIn("energy_ood", out)

    def test_uncalibrated_conformal_ambiguity_does_not_escalate(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.5, "probabilities": [0.5, 0.5]}, context=None
        )
        self.assertIs(out["conformal_set"]["calibrated"], False)
        self.assertIs(out["conformal_set"]["is_ambiguous"], True)
        self.assertNotIn("conformal_ambiguity", out.get("meta_cognitive_correction", {}))

    def test_probabilities_only_gives_conformal_but_no_energy(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.9, "probabilities": [0.97, 0.03]}, context=None
        )
        self.assertNotIn("probabilities_missing", out)
        self.assertEqual(out["conformal_set"]["prediction_set"], [0])
        self.assertNotIn("energy_ood", out)

    def test_empty_probabilities_count_as_missing(self):
        out = _monitor().monitor_decision_sync({"confidence": 0.9, "probabilities": []}, context=None)
        self.assertIs(out["probabilities_missing"], True)
        self.assertNotIn("conformal_set", out)

    def test_results_fallback_uses_only_explicit_values(self):
        out = _monitor().monitor_decision_sync(
            {"mode": "decision", "results": [{"confidence": 0.9}]}, context=None
        )
        self.assertIs(out["probabilities_missing"], True)
        self.assertNotIn("conformal_set", out)

        explicit = _monitor().monitor_decision_sync(
            {"mode": "decision", "results": [{"probabilities": [0.97, 0.03]}]}, context=None
        )
        self.assertIn("conformal_set", explicit)
        self.assertNotIn("probabilities_missing", explicit)


class TestEnergyUsesExplicitScoresOnly(unittest.TestCase):

    def test_energy_is_computed_from_logits_not_probabilities(self):
        monitor = _monitor()
        logits = [-6.0, -7.0, -6.5, -8.0]
        probs = [0.6, 0.38, 0.02]
        out = monitor.monitor_decision_sync(
            {"confidence": 0.9, "probabilities": probs, "logits": logits}, context=None
        )
        self.assertAlmostEqual(out["energy_ood"]["energy"], monitor.energy_detector.compute_energy(logits), places=9)
        self.assertNotAlmostEqual(
            out["energy_ood"]["energy"], monitor.energy_detector.compute_energy(probs), places=3
        )
        self.assertIs(out["energy_ood"]["is_ood"], True)
        self.assertIs(out["meta_cognitive_correction"]["energy_ood"], True)

    def test_empty_logits_give_no_energy(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.9, "probabilities": [0.97, 0.03], "logits": []}, context=None
        )
        self.assertNotIn("energy_ood", out)

    def test_in_distribution_logits_are_not_flagged(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.9, "probabilities": [0.97, 0.03], "logits": [6.0, 1.0, 0.5, -2.0]},
            context=None,
        )
        self.assertIs(out["energy_ood"]["is_ood"], False)
        self.assertNotIn("energy_ood", out.get("meta_cognitive_correction", {}))


class TestTensorAndEscalationBehaviour(unittest.TestCase):

    def test_tensor_probabilities_are_converted_not_emptied(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.9, "probabilities": torch.tensor([0.97, 0.03])}, context=None
        )
        self.assertEqual(out["conformal_set"]["prediction_set"], [0])
        self.assertTrue(out["conformal_set"]["is_singleton"])
        self.assertIn("aleatoric_uncertainty", out["uncertainty"])

    def test_explicit_ambiguity_escalates_once_calibrated(self):
        monitor = _monitor()
        # Non-conformity 0.95 everywhere gives threshold 0.05, so [0.6, 0.38] is ambiguous.
        monitor.conformal_engine.calibrate([0.95] * 20)
        out = monitor.monitor_decision_sync(
            {"confidence": 0.9, "probabilities": [0.6, 0.38, 0.02]}, context=None
        )
        self.assertIs(out["conformal_set"]["calibrated"], True)
        self.assertEqual(out["conformal_set"]["prediction_set"], [0, 1])
        correction = out["meta_cognitive_correction"]
        self.assertIs(correction["recommend_escalation"], True)
        self.assertIs(correction["conformal_ambiguity"], True)

    def test_conformal_sets_report_uncalibrated_fallback(self):
        out = _monitor().monitor_decision_sync(
            {"confidence": 0.9, "probabilities": [0.97, 0.03]}, context=None
        )
        self.assertIs(out["conformal_set"]["calibrated"], False)


class TestSelfCorrectionNeverRewritesConfidence(unittest.TestCase):

    ANOMALOUS = {"mode": "choice", "confidence": 0.99, "probabilities": [0.34, 0.33, 0.33]}

    def test_anomaly_is_flagged_and_confidence_kept(self):
        monitor = _monitor()
        monitor.monitor_decision_sync(
            {"mode": "choice", "confidence": 0.3, "probabilities": [0.9, 0.1]}, context=None
        )
        out = monitor.monitor_decision_sync(dict(self.ANOMALOUS), context=None)
        self.assertEqual(out["confidence"], 0.99)
        meta = out["meta_cognitive_correction"]
        self.assertEqual(meta["anomaly_detected"], "high_entropy_high_confidence_conflict")
        for key in ("calibrated_confidence", "original_confidence", "calibration_factor"):
            self.assertNotIn(key, meta)

    def test_self_correct_sync_does_not_touch_the_decision(self):
        monitor = _monitor()
        decision = {"mode": "choice", "confidence": 0.99, "uncertainty": {"aleatoric": 0.95}}
        before = copy.deepcopy(decision)
        result = monitor.self_correct_sync(decision, context=None)
        self.assertTrue(result["applied"])
        self.assertEqual(result["corrections"]["anomaly_detected"], "high_entropy_high_confidence_conflict")
        self.assertNotIn("calibrated_confidence", result["corrections"])
        self.assertEqual(decision, before)

    def test_external_calibration_factor_is_ignored(self):
        monitor = _monitor()
        monitor.confidence_calibration["avg_conf_choice"] = 0.2  # would previously have rescaled confidence
        result = monitor.self_correct_sync({"mode": "choice", "confidence": 0.9}, context=None)
        self.assertFalse(result["applied"])
        self.assertEqual(result["corrections"], {})

    def test_async_monitor_keeps_confidence(self):
        monitor = _monitor()
        monitor.monitor_decision_sync(
            {"mode": "choice", "confidence": 0.3, "probabilities": [0.9, 0.1]}, context=None
        )
        out = asyncio.run(monitor.monitor_decision(dict(self.ANOMALOUS), context=None))
        self.assertEqual(out["confidence"], 0.99)
        self.assertIn("anomaly_detected", out["meta_cognitive_correction"])


class TestConformalCalibratedFlag(unittest.TestCase):

    def test_uncalibrated_engine_reports_fallback(self):
        engine = ConformalPredictionEngine(alpha=0.05)
        self.assertIs(engine.calibrated, False)
        result = engine.predict_set({"Deploy": 0.95, "Rollback": 0.03, "Canary": 0.02})
        self.assertIs(result["calibrated"], False)
        self.assertIsInstance(result["prediction_set"], list)

    def test_calibrate_sets_flag_and_predict_reports_it(self):
        engine = ConformalPredictionEngine(alpha=0.1)
        engine.calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertIs(engine.calibrated, True)
        self.assertIs(engine.predict_set({"A": 0.9, "B": 0.1})["calibrated"], True)

    def test_empty_calibration_keeps_flag_false(self):
        engine = ConformalPredictionEngine()
        engine.calibrate([])
        self.assertIs(engine.calibrated, False)
        self.assertIs(engine.predict_set([0.5, 0.5])["calibrated"], False)

    def test_existing_keys_are_preserved(self):
        result = ConformalPredictionEngine().predict_set([0.9, 0.1])
        for key in ("prediction_set", "set_size", "is_singleton", "is_ambiguous",
                    "is_empty", "coverage_guarantee", "threshold", "calibrated"):
            self.assertIn(key, result)

    def test_prediction_set_is_always_a_list(self):
        engine = ConformalPredictionEngine()
        for probs in ({"A": 0.9, "B": 0.1}, [0.9, 0.1], (0.9, 0.1), torch.tensor([0.9, 0.1])):
            self.assertIsInstance(engine.predict_set(probs)["prediction_set"], list)

    def test_tensor_input_is_not_silently_emptied(self):
        result = ConformalPredictionEngine().predict_set(torch.tensor([0.97, 0.03]))
        self.assertEqual(result["prediction_set"], [0])


class TestAdaptiveDriftRuleMatchesDocstring(unittest.TestCase):

    @staticmethod
    def _drift_flags(outcomes):
        engine = AdaptiveConformalEngine(alpha=0.1, eta=0.01)
        flags = []
        for covered in outcomes:
            record = engine.update(y_true="a", prediction_set=["a"] if covered else [])
            flags.append(record["drift_detected"])
        return flags

    def test_three_consecutive_misses_flag_drift(self):
        self.assertEqual(self._drift_flags([True, False, False, False]), [False, False, False, True])

    def test_rolling_error_above_2_5_alpha_flags_drift_from_eleventh_update(self):
        # Three misses in the first nine updates. The rolling window (previous updates only)
        # is first checked once ten updates are recorded: 3/10 = 0.30 > 2.5 * 0.1 = 0.25.
        flags = self._drift_flags([True, True, False, True, True, False, True, True, False, True, True])
        self.assertFalse(any(flags[:10]))
        self.assertTrue(flags[10])

    def test_rolling_error_at_or_below_2_5_alpha_does_not_flag(self):
        # Two misses in ten updates: 0.20 <= 0.25 (2.5 * alpha).
        flags = self._drift_flags([True, True, False, True, True, False, True, True, True, True, True])
        self.assertFalse(any(flags))


class TestGlobalTemperatureRemoved(unittest.TestCase):

    def test_attribute_does_not_exist(self):
        self.assertFalse(hasattr(_monitor(), "global_temperature"))

    def test_feedback_is_recorded_without_learning_a_temperature(self):
        monitor = _monitor()
        for i in range(12):
            asyncio.run(monitor.update_calibration_from_feedback(
                f"d{i}", actual_correctness=False, predicted_confidence=0.95
            ))
        self.assertFalse(hasattr(monitor, "global_temperature"))
        self.assertEqual(len(monitor.calibration_history), 12)
        self.assertEqual(len(monitor.bin_predictions[9]), 12)
        self.assertEqual(len(monitor.bin_accuracies[9]), 12)
        self.assertAlmostEqual(monitor.compute_ece(), 0.95, places=9)

    def test_calibration_report_has_no_temperature_and_no_factors(self):
        monitor = _monitor()
        monitor.update_calibration_from_feedback_sync("d0", True, 0.6)
        report = monitor.get_calibration_report()
        self.assertNotIn("global_temperature", report)
        self.assertIn("expected_calibration_error", report)
        self.assertEqual(report["calibration_factors"], {})


if __name__ == "__main__":
    unittest.main()
