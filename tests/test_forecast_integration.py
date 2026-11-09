"""
Predictive layer and forecast-aware decisions: forecast math, TimesFM adapter, policy gates,
approval gate, feedback ledger, and an end-to-end run with a real RhenEngine.

Covered:
1. Quantile interval, exceedance and uncertainty math on hand-computed cases.
2. The naive random-walk baseline on deterministic and hand-checked series.
3. The TimesFM adapter: licence gate, missing-package hint, output mapping (fake timesfm3 module).
4. DecisionPolicy: BLOCKED, ABSTAIN, NEEDS_APPROVAL and EXECUTE, each reached by a specific input.
5. The approval gate in ForecastAwareDecider.execute.
6. FeedbackLedger: pinball loss, coverage, Brier, MASE, calibration report and JSONL round trip.
7. ForecastAwareDecider.decide end to end with a real RhenEngine (needs torch).
8. A real TimesFM run, skipped unless RHEN_TIMESFM_RUN_REAL=1 and the checkpoint is cached.
"""

import importlib
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rhen.rhen import ABSTAIN, ANSWER, AbstentionVerdict, ChoiceResult
from rhen.forecasting import (
    QUANTILE_LEVELS,
    Forecast,
    NaiveQuantileForecaster,
    TimesFMForecaster,
    brier_score,
    interval_coverage,
    mase,
    pinball_loss,
)
from rhen.decision_pipeline import (
    DecisionGateError,
    DecisionPolicy,
    FeedbackLedger,
    FutureAwareDecision,
    ForecastAwareDecider,
    PolicyOutcome,
    PolicyStatus,
)

# Quantile row for a single step: values 10..18 at levels 0.1..0.9.
ROW = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]


def _choice(probabilities, confidence=0.9, verdict=ANSWER):
    labels = list(probabilities)
    return ChoiceResult(
        question="q",
        choice_index=0,
        choice_label=labels[0],
        probabilities=dict(probabilities),
        raw_probabilities=[probabilities[k] for k in labels],
        confidence=confidence,
        temperature=1.0,
        abstention_verdict=AbstentionVerdict(decision=verdict),
    )


class _StubEngine:
    """Returns a fixed ChoiceResult and records the augmented state it was given."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def choice(self, state, question, options):
        self.calls.append({"state": state, "question": question, "options": list(options)})
        return self.result


def _features(breach=None, uncertainty=0.1):
    # One breach figure stands for both bounds here; the tier is gated on the upper bound.
    return {
        "series": {},
        "breach_prob_step_max_lower": breach,
        "breach_prob_any_step_upper": breach,
        "uncertainty": uncertainty,
    }


def _make_decision(status, option="restock", breach=None, thresholds=None, forecast=None):
    """Hand-built FutureAwareDecision for ledger and gate tests."""
    fc = forecast if forecast is not None else Forecast("demand", 1, [14.0], [ROW])
    choice = _choice({"restock": 0.7, "hold": 0.3})
    outcome = PolicyOutcome(status, option, 0.7, ["test"], "low")
    return FutureAwareDecision(
        decision_id=f"fad-{os.urandom(8).hex()}",
        created_at="2026-01-01T00:00:00+00:00",
        question="q",
        options=["restock", "hold"],
        horizon=fc.horizon,
        thresholds=dict(thresholds or {}),
        forecasts={fc.name: fc},
        features=_features(breach=breach),
        augmented_state="state",
        choice=choice,
        outcome=outcome,
    )


# ---------------------------------------------------------------- 1. forecast math


class TestForecastMath(unittest.TestCase):
    def setUp(self):
        self.fc = Forecast("demand", 1, [14.0], [ROW])

    def test_80_and_50_percent_intervals_from_quantiles(self):
        lower, upper = self.fc.interval(0.8)
        self.assertEqual(lower, [10.0])
        self.assertEqual(upper, [18.0])
        # The 0.25 and 0.75 quantiles fall between stored levels and are interpolated.
        lower50, upper50 = self.fc.interval(0.5)
        self.assertAlmostEqual(lower50[0], 11.5)
        self.assertAlmostEqual(upper50[0], 16.5)

    def test_interval_rejects_levels_needing_untracked_tails(self):
        with self.assertRaises(ValueError):
            self.fc.interval(0.9)

    def test_prob_exceeds_at_a_step(self):
        self.assertAlmostEqual(self.fc.prob_exceeds(14.0, step=0), 0.5)
        self.assertAlmostEqual(self.fc.prob_exceeds(12.5, step=0), 0.65)
        self.assertAlmostEqual(self.fc.prob_exceeds(18.5, step=0), 0.05)
        # Below the 0.1 quantile the linear extrapolation exceeds 1 and is clamped.
        self.assertEqual(self.fc.prob_exceeds(9.0, step=0), 1.0)
        self.assertEqual(self.fc.prob_exceeds(8.0, step=0), 1.0)
        # Above the 0.9 quantile it is extrapolated past 1 and clamped to 0.
        self.assertEqual(self.fc.prob_exceeds(19.0, step=0), 0.0)

    def test_prob_exceeds_over_horizon_is_the_largest_step_probability(self):
        row2 = [v + 10.0 for v in ROW]  # step 2 median is 24
        fc = Forecast("demand", 2, [14.0, 24.0], [ROW, row2])
        self.assertAlmostEqual(fc.prob_exceeds(25.0, step=0), 0.0)
        self.assertAlmostEqual(fc.prob_exceeds(25.0, step=1), 0.4)
        # Without a step, prob_exceeds is the lower bound: the largest single-step probability.
        self.assertAlmostEqual(fc.prob_exceeds(25.0), 0.4)
        self.assertEqual(fc.step_exceedance_probs(25.0), [fc.prob_exceeds(25.0, step=0), fc.prob_exceeds(25.0, step=1)])

    def test_tied_lower_quantiles_resolve_to_the_highest_tied_level(self):
        # Quantiles 5 at levels 0.1, 0.2, 0.3. F(5) is the highest tied level, 0.3, so P(X > 5) = 0.7.
        fc = Forecast("demand", 1, [6.0], [[5, 5, 5, 6, 7, 8, 9, 10, 11]])
        self.assertAlmostEqual(fc.prob_exceeds(5.0, step=0), 0.70)
        # Strictly inside the segment from 5 (level 0.3) to 6 (level 0.4).
        self.assertAlmostEqual(fc.prob_exceeds(5.5, step=0), 0.65)

    def test_tied_interior_and_upper_quantiles_resolve_to_the_highest_tied_level(self):
        interior = Forecast("x", 1, [3.0], [[1, 2, 3, 3, 3, 4, 5, 6, 7]])
        self.assertAlmostEqual(interior.prob_exceeds(3.0, step=0), 0.50)  # F(3) = 0.5
        upper = Forecast("x", 1, [9.0], [[1, 2, 3, 4, 5, 6, 9, 9, 9]])
        self.assertAlmostEqual(upper.prob_exceeds(9.0, step=0), 0.10)  # F(9) = 0.9

    def test_constant_series_threshold_at_the_constant_is_a_point_mass(self):
        # Every quantile equals 5, so the forecast is the point 5. P(X > 5) = 0 by the point-mass
        # rule: no tail mass is added above a point with no width. Below the point it is 1.
        (fc,) = NaiveQuantileForecaster().predict([[5.0] * 10], 3)
        self.assertEqual(fc.quantiles[0], [5.0] * 9)
        self.assertEqual(fc.prob_exceeds(5.0, step=0), 0.0)
        self.assertEqual(fc.prob_exceeds(4.0, step=0), 1.0)
        self.assertEqual(fc.prob_exceeds(6.0, step=0), 0.0)

    def test_threshold_below_all_quantiles_is_one_and_above_all_is_zero(self):
        self.assertEqual(self.fc.prob_exceeds(0.0, step=0), 1.0)
        self.assertEqual(self.fc.prob_exceeds(100.0, step=0), 0.0)
        tied = Forecast("x", 1, [6.0], [[5, 5, 5, 6, 7, 8, 9, 10, 11]])
        self.assertEqual(tied.prob_exceeds(4.0, step=0), 1.0)
        self.assertEqual(tied.prob_exceeds(12.0, step=0), 0.0)

    def test_central_interval_levels_supported_range_is_zero_to_point_eight(self):
        self.fc.interval(0.8)  # the widest supported level
        self.fc.interval(0.01)
        with self.assertRaises(ValueError):
            self.fc.interval(0.81)

    def test_degenerate_forecast_is_a_point_mass(self):
        fc = Forecast("flat", 1, [110.0], [[110.0] * 9])
        self.assertEqual(fc.prob_exceeds(150.0, step=0), 0.0)
        self.assertEqual(fc.prob_exceeds(100.0, step=0), 1.0)

    def test_uncertainty_is_mean_relative_width_of_the_80_percent_band(self):
        # Width 8 over a median of 14.
        self.assertAlmostEqual(self.fc.uncertainty(), 8.0 / 14.0)

    def test_uncertainty_without_quantiles_fails_closed(self):
        self.assertTrue(math.isinf(Forecast("x", 1, [1.0]).uncertainty()))

    def test_forecast_validation(self):
        with self.assertRaises(ValueError):
            Forecast("x", 1, [1.0], [[1.0] * 8])  # wrong row width
        with self.assertRaises(ValueError):
            Forecast("x", 1, [1.0], [[3.0, 2.0] + [4.0] * 7])  # decreasing quantiles
        with self.assertRaises(ValueError):
            Forecast("x", 2, [1.0], None)  # mean length mismatch

    def test_forecast_round_trips_through_dict(self):
        clone = Forecast.from_dict(json.loads(json.dumps(self.fc.to_dict())))
        self.assertEqual(clone, self.fc)


class TestNaiveQuantileForecaster(unittest.TestCase):
    def test_deterministic_series_has_flat_median_and_zero_width_bands(self):
        (fc,) = NaiveQuantileForecaster().predict([list(range(10))], 3)
        self.assertEqual(fc.mean, [9.0, 9.0, 9.0])
        for row in fc.quantiles:
            self.assertEqual(row, [9.0] * 9)

    def test_alternating_series_matches_hand_computed_bands(self):
        # diffs are [1, -1, 1, -1, 1]; centred quantiles are [-2, -2, -1.6, -0.8, 0, 0, 0, 0, 0]
        # around the last value 1. Band half-widths scale with sqrt(h).
        (fc,) = NaiveQuantileForecaster().predict([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], 4)
        self.assertEqual(fc.mean, [1.0, 1.0, 1.0, 1.0])
        lower, upper = fc.interval(0.8)
        self.assertAlmostEqual(lower[0], -1.0)
        self.assertAlmostEqual(lower[3], -3.0)  # 1 + sqrt(4) * (-2)
        self.assertEqual(upper, [1.0, 1.0, 1.0, 1.0])
        widths = [u - l for l, u in zip(lower, upper)]
        self.assertEqual(widths, sorted(widths))  # bands never narrow with horizon

    def test_series_are_forecast_in_order(self):
        out = NaiveQuantileForecaster().predict([[1.0, 2.0, 3.0], [10.0, 10.0, 12.0]], 2)
        self.assertEqual([f.name for f in out], ["series[0]", "series[1]"])
        self.assertEqual(out[0].mean, [3.0, 3.0])
        self.assertEqual(out[1].mean, [12.0, 12.0])

    def test_rejects_non_finite_and_short_series(self):
        with self.assertRaises(ValueError):
            NaiveQuantileForecaster().predict([[1.0, float("nan")]], 2)
        with self.assertRaises(ValueError):
            NaiveQuantileForecaster().predict([[1.0]], 2)


# ---------------------------------------------------------------- 3. TimesFM adapter


class TestTimesFMAdapter(unittest.TestCase):
    def test_construction_without_license_acknowledgement_raises(self):
        with self.assertRaises(ValueError) as ctx:
            TimesFMForecaster()
        self.assertIn("non-commercial", str(ctx.exception))

    def test_truthy_non_bool_acknowledgement_is_refused(self):
        with self.assertRaises(ValueError):
            TimesFMForecaster(license_acknowledged="yes")

    def test_missing_timesfm_package_gives_install_hint(self):
        with mock.patch.dict(sys.modules, {"timesfm3": None}):
            with self.assertRaises(ImportError) as ctx:
                TimesFMForecaster(license_acknowledged=True)
        self.assertIn('pip install "timesfm[torch]>=3.0.2"', str(ctx.exception))

    def _fake_timesfm3(self, quantile_width=9):
        calls = {}

        class ModelConfig:
            def __init__(self, **kwargs):
                calls["config"] = kwargs

        class TimesFM3Evaluator:
            def __init__(self, config):
                calls["evaluator_config"] = config

            def predict_batch(self, contexts, horizon, return_quantiles, use_symmetric_averaging, **kwargs):
                calls["predict_batch"] = {
                    "n_contexts": len(contexts),
                    "horizon": horizon,
                    "return_quantiles": return_quantiles,
                    "use_symmetric_averaging": use_symmetric_averaging,
                }
                for i in range(len(contexts)):
                    base = float(i + 1)
                    point = np.arange(horizon, dtype=np.float32) + base
                    quant = np.stack([point + d for d in np.linspace(-2, 2, quantile_width)], axis=1)
                    yield SimpleNamespace(forecast=point, quantiles=quant)

        module = types.ModuleType("timesfm3")
        module.ModelConfig = ModelConfig
        module.TimesFM3Evaluator = TimesFM3Evaluator
        return module, calls

    def test_maps_timesfm_outputs_onto_forecasts(self):
        module, calls = self._fake_timesfm3()
        with mock.patch.dict(sys.modules, {"timesfm3": module}):
            fc = TimesFMForecaster(
                checkpoint_path="google/timesfm-3.0-pytorch",
                device="cpu",
                per_core_batch_size=8,
                license_acknowledged=True,
            )
            out = fc.predict([[1.0, 2.0, 3.0], [5.0, 4.0, 3.0]], horizon=3)

        self.assertEqual(calls["config"], {
            "checkpoint_path": "google/timesfm-3.0-pytorch",
            "per_core_batch_size": 8,
            "device": "cpu",
        })
        self.assertEqual(calls["predict_batch"], {
            "n_contexts": 2,
            "horizon": 3,
            "return_quantiles": True,
            "use_symmetric_averaging": False,
        })
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].horizon, 3)
        self.assertEqual(out[0].mean, [1.0, 2.0, 3.0])
        self.assertEqual(len(out[0].quantiles), 3)
        self.assertEqual(len(out[0].quantiles[0]), len(QUANTILE_LEVELS))
        # The median column is the point forecast: quantile index 4 is the 0.5 level.
        self.assertAlmostEqual(out[0].quantiles[1][4], out[0].mean[1], places=5)
        self.assertAlmostEqual(out[1].mean[0], 2.0, places=5)

    def test_rejects_quantile_output_with_wrong_width(self):
        module, _ = self._fake_timesfm3(quantile_width=10)
        with mock.patch.dict(sys.modules, {"timesfm3": module}):
            fc = TimesFMForecaster(license_acknowledged=True)
            with self.assertRaises(RuntimeError):
                fc.predict([[1.0, 2.0]], horizon=2)


# ---------------------------------------------------------------- 4. policy


class TestDecisionPolicy(unittest.TestCase):
    def test_blocked_when_every_option_violates_a_constraint(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0, constraints=[("never", lambda o, f: False)])
        out = policy.evaluate(_choice({"a": 0.7, "b": 0.3}), _features())
        self.assertIs(out.status, PolicyStatus.BLOCKED)
        self.assertIsNone(out.option)
        self.assertTrue(any("never" in r for r in out.reasons))

    def test_blocked_option_is_skipped_for_the_next_admissible_one(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0, constraints=[("not_a", lambda o, f: o != "a")])
        out = policy.evaluate(_choice({"a": 0.8, "b": 0.2}), _features())
        self.assertIs(out.status, PolicyStatus.EXECUTE)
        self.assertEqual(out.option, "b")

    def test_abstain_on_low_confidence(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0, min_confidence=0.6)
        out = policy.evaluate(_choice({"a": 0.5, "b": 0.5}, confidence=0.4), _features())
        self.assertIs(out.status, PolicyStatus.ABSTAIN)
        self.assertTrue(any("confidence" in r for r in out.reasons))

    def test_abstain_on_high_forecast_uncertainty(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0, max_forecast_uncertainty=0.5)
        out = policy.evaluate(_choice({"a": 0.9, "b": 0.1}), _features(uncertainty=0.9))
        self.assertIs(out.status, PolicyStatus.ABSTAIN)
        self.assertTrue(any("uncertainty" in r for r in out.reasons))

    def test_abstain_when_engine_abstains(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0)
        out = policy.evaluate(_choice({"a": 0.9, "b": 0.1}, verdict=ABSTAIN), _features())
        self.assertIs(out.status, PolicyStatus.ABSTAIN)
        self.assertTrue(any("engine" in r for r in out.reasons))

    def test_needs_approval_for_high_breach_risk(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0)
        out = policy.evaluate(_choice({"a": 0.9, "b": 0.1}), _features(breach=0.9))
        self.assertIs(out.status, PolicyStatus.NEEDS_APPROVAL)
        self.assertEqual(out.risk_tier, "critical")
        self.assertEqual(out.option, "a")

    def test_execute_for_low_breach_risk(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0)
        out = policy.evaluate(_choice({"a": 0.9, "b": 0.1}), _features(breach=0.05))
        self.assertIs(out.status, PolicyStatus.EXECUTE)
        self.assertEqual(out.risk_tier, "low")

    def test_selection_is_by_probability_weighted_utility(self):
        utilities = {"a": 1.0, "b": 3.0}
        policy = DecisionPolicy(utility=lambda o, f: utilities[o])
        out = policy.evaluate(_choice({"a": 0.6, "b": 0.4}), _features())
        self.assertEqual(out.option, "b")  # 0.4 * 3 = 1.2 beats 0.6 * 1 = 0.6
        self.assertAlmostEqual(out.expected_utility, 1.2)

    def test_negative_utility_is_rejected(self):
        # Without the guard, a = 0.9 * -1 = -0.9 and b = 0.1 * -0.5 = -0.05 would pick b.
        utilities = {"a": -1.0, "b": -0.5}
        policy = DecisionPolicy(utility=lambda o, f: utilities[o])
        with self.assertRaises(ValueError) as ctx:
            policy.evaluate(_choice({"a": 0.9, "b": 0.1}), _features())
        message = str(ctx.exception)
        self.assertIn("non-negative", message)
        self.assertIn("not a full cost model", message)

    def test_negative_utility_on_a_blocked_option_is_not_evaluated(self):
        policy = DecisionPolicy(
            utility=lambda o, f: -1.0 if o == "a" else 1.0,
            constraints=[("not_a", lambda o, f: o != "a")],
        )
        out = policy.evaluate(_choice({"a": 0.8, "b": 0.2}), _features())
        self.assertIs(out.status, PolicyStatus.EXECUTE)
        self.assertEqual(out.option, "b")

    def test_risk_tier_is_gated_on_the_upper_bound_not_the_lower(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0)
        # Lower bound 0.19 alone would be "low"; the upper bound 0.38 gives "medium".
        features = {"series": {}, "breach_prob_step_max_lower": 0.19, "breach_prob_any_step_upper": 0.38,
                    "uncertainty": 0.1}
        tier, reason = policy.risk_tier("a", features)
        self.assertEqual(tier, "medium")
        self.assertIn("upper bound 0.380", reason)
        self.assertIn("lower bound", reason)

    def test_tier_fn_can_force_approval_without_breach_threshold(self):
        policy = DecisionPolicy(utility=lambda o, f: 1.0, tier_fn=lambda o, f: "high")
        out = policy.evaluate(_choice({"a": 0.9, "b": 0.1}), _features())
        self.assertIs(out.status, PolicyStatus.NEEDS_APPROVAL)

    def test_policy_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            DecisionPolicy(utility=lambda o, f: 1.0, approval_required_tiers=("urgent",))
        with self.assertRaises(TypeError):
            DecisionPolicy(utility=1.0)
        with self.assertRaises(ValueError):
            DecisionPolicy(utility=lambda o, f: 1.0, tier_cutoffs=(0.8, 0.5, 0.2))


# ---------------------------------------------------------------- 5. decide and approval gate


SERIES = {"demand": [100.0, 103.0, 99.0, 104.0, 98.0, 105.0, 97.0, 106.0, 96.0, 107.0]}


class TestDecideAndApprovalGate(unittest.TestCase):
    def _decider(self, result, policy=None, ledger=None):
        engine = _StubEngine(result)
        decider = ForecastAwareDecider(engine, NaiveQuantileForecaster(), policy=policy, ledger=ledger)
        return decider, engine

    def _decide(self, decider, thresholds):
        return decider.decide(
            state="Warehouse snapshot.",
            question="What should we do?",
            options=["restock", "hold"],
            series=SERIES,
            horizon=4,
            thresholds=thresholds,
        )

    def test_augmented_state_carries_a_deterministic_forecast_summary(self):
        decider, engine = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        first = self._decide(decider, {"demand": 1000.0})
        second = self._decide(decider, {"demand": 1000.0})
        self.assertEqual(first.augmented_state, second.augmented_state)
        self.assertIn("Forecast summary (method=naive_random_walk, horizon=4)", first.augmented_state)
        self.assertIn(
            "P(demand > 1000 at some step) is between 0.000 (max single step, lower bound) "
            "and 0.000 (union bound, upper bound)",
            first.augmented_state,
        )
        self.assertNotIn("within horizon", first.augmented_state)
        self.assertEqual(engine.calls[0]["state"], first.augmented_state)
        self.assertEqual(engine.calls[0]["options"], ["restock", "hold"])
        self.assertEqual(first.forecasts["demand"].horizon, 4)
        self.assertEqual(first.forecasts["demand"].name, "demand")

    def test_policy_status_tracks_breach_risk_through_the_pipeline(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        safe = self._decide(decider, {"demand": 1000.0})
        self.assertIs(safe.outcome.status, PolicyStatus.EXECUTE)
        risky = self._decide(decider, {"demand": 0.0})
        self.assertIs(risky.outcome.status, PolicyStatus.NEEDS_APPROVAL)

    def test_executor_runs_for_execute_and_only_once(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 1000.0})
        executor = mock.Mock(return_value="done")
        self.assertEqual(decider.execute(decision, executor), "done")
        executor.assert_called_once()
        option, passed_decision = executor.call_args.args
        self.assertEqual(option, "restock")
        self.assertIs(passed_decision, decision)
        with self.assertRaises(DecisionGateError):
            decider.execute(decision, executor)

    def test_unapproved_needs_approval_decision_is_refused(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 0.0})
        executor = mock.Mock()
        with self.assertRaises(DecisionGateError):
            decider.execute(decision, executor)
        executor.assert_not_called()

    def test_approved_needs_approval_decision_executes_and_records_approver(self):
        ledger = FeedbackLedger()
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}), ledger=ledger)
        decision = self._decide(decider, {"demand": 0.0})
        approval = decider.approve(decision.decision_id, "  ops-lead ")
        self.assertEqual(approval["approver"], "ops-lead")
        record = ledger.get(decision.decision_id)
        self.assertEqual(record.approval["approver"], "ops-lead")
        self.assertIsNotNone(record.approval["approved_at"])
        executor = mock.Mock(return_value=1)
        decider.execute(decision, executor)
        executor.assert_called_once()
        self.assertEqual(ledger.get(decision.decision_id).executed_option, "restock")

    def test_setting_the_approval_attribute_does_not_authorise_execution(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 0.0})
        self.assertIs(decision.outcome.status, PolicyStatus.NEEDS_APPROVAL)
        forged = {"approver": "attacker", "approved_at": "2026-01-01T00:00:00+00:00"}
        # approval is a read-only view of the decider's registry, so assignment is refused.
        with self.assertRaises(AttributeError):
            decision.approval = forged
        # Writing the instance dict directly does not shadow the read-only property either.
        decision.__dict__["approval"] = forged
        self.assertIsNone(decision.approval)
        executor = mock.Mock()
        with self.assertRaises(DecisionGateError):
            decider.execute(decision, executor)
        executor.assert_not_called()

    def test_approve_writes_the_registry_and_execute_reads_it(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 0.0})
        approval = decider.approve(decision.decision_id, "ops-lead")
        self.assertEqual(decision.approval, approval)
        self.assertEqual(set(decision.approval), {"approver", "approved_at"})
        # The view is a copy: editing it does not change what execute() will check.
        decision.approval["approver"] = "someone-else"
        self.assertEqual(decision.approval["approver"], "ops-lead")
        executor = mock.Mock(return_value="ok")
        self.assertEqual(decider.execute(decision, executor), "ok")
        executor.assert_called_once()

    def test_breach_bounds_are_reported_and_the_tier_uses_the_upper_bound(self):
        # Per-step breach probability is 0.19 at every step (1 - F(17.1) with F from ROW).
        class FixedForecaster:
            name = "fixed"

            def __init__(self, forecast):
                self.forecast = forecast

            def predict(self, series_list, horizon):
                return [self.forecast for _ in series_list]

        policy = DecisionPolicy(utility=lambda o, f: 1.0, max_forecast_uncertainty=math.inf)
        ten_steps = Forecast("x", 10, [14.0] * 10, [ROW] * 10)
        engine = _StubEngine(_choice({"restock": 0.8, "hold": 0.2}))
        decider = ForecastAwareDecider(engine, FixedForecaster(ten_steps), policy=policy)
        decision = decider.decide("s", "q", ["restock", "hold"], {"x": [12.0, 13.0, 14.0]}, 10,
                                  thresholds={"x": 17.1})
        self.assertAlmostEqual(decision.features["breach_prob_step_max_lower"], 0.19)
        self.assertEqual(decision.features["breach_prob_any_step_upper"], 1.0)  # min(1, 10 * 0.19)
        self.assertNotEqual(decision.outcome.risk_tier, "low")
        self.assertEqual(decision.outcome.risk_tier, "critical")
        self.assertIs(decision.outcome.status, PolicyStatus.NEEDS_APPROVAL)
        tier_reason = next(r for r in decision.outcome.reasons if r.startswith("P(breach at some step)"))
        self.assertIn("is between 0.190 (lower bound, max single step)", tier_reason)
        self.assertIn("1.000 (upper bound, union bound)", tier_reason)
        # Two steps: upper bound 0.38 gives "medium" while the lower bound 0.19 alone would be "low".
        two_steps = Forecast("x", 2, [14.0] * 2, [ROW] * 2)
        decider2 = ForecastAwareDecider(_StubEngine(_choice({"restock": 0.8, "hold": 0.2})),
                                        FixedForecaster(two_steps), policy=policy)
        second = decider2.decide("s", "q", ["restock", "hold"], {"x": [12.0, 13.0, 14.0]}, 2,
                                 thresholds={"x": 17.1})
        self.assertAlmostEqual(second.features["breach_prob_any_step_upper"], 0.38)
        self.assertEqual(second.outcome.risk_tier, "medium")
        self.assertIn("P(x > 17.1 at some step)", second.augmented_state)

    def test_blocked_and_abstain_decisions_never_execute(self):
        blocked_policy = DecisionPolicy(utility=lambda o, f: 1.0, constraints=[("no", lambda o, f: False)])
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}), policy=blocked_policy)
        blocked = self._decide(decider, {"demand": 1000.0})
        self.assertIs(blocked.outcome.status, PolicyStatus.BLOCKED)
        abstain_decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}, confidence=0.1))
        abstained = self._decide(abstain_decider, {"demand": 1000.0})
        self.assertIs(abstained.outcome.status, PolicyStatus.ABSTAIN)
        for decision, d in ((blocked, decider), (abstained, abstain_decider)):
            executor = mock.Mock()
            with self.assertRaises(DecisionGateError):
                d.execute(decision, executor)
            executor.assert_not_called()

    def test_approve_refuses_decisions_that_do_not_need_approval(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 1000.0})
        with self.assertRaises(DecisionGateError):
            decider.approve(decision.decision_id, "ops-lead")

    def test_failed_executor_is_not_retried(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 1000.0})

        def boom(option, decision):
            raise RuntimeError("downstream down")

        with self.assertRaises(RuntimeError):
            decider.execute(decision, boom)
        self.assertIn("downstream down", decision.execution_error)
        with self.assertRaises(DecisionGateError):
            decider.execute(decision, mock.Mock())

    def test_decider_rejects_unknown_threshold_names_and_duplicate_options(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        with self.assertRaises(ValueError):
            self._decide(decider, {"revenue": 1.0})
        with self.assertRaises(ValueError):
            decider.decide("s", "q", ["a", "a"], SERIES, 2)

    def test_decider_requires_quantiles_from_the_forecaster(self):
        class PointOnlyForecaster:
            name = "point_only"

            def predict(self, series_list, horizon):
                return [Forecast("x", horizon, [1.0] * horizon) for _ in series_list]

        decider = ForecastAwareDecider(_StubEngine(_choice({"restock": 0.8, "hold": 0.2})), PointOnlyForecaster())
        with self.assertRaises(ValueError) as ctx:
            self._decide(decider, {})
        self.assertIn("point_only", str(ctx.exception))

    def test_approve_and_execute_refuse_decisions_from_another_decider(self):
        decider, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        other, _ = self._decider(_choice({"restock": 0.8, "hold": 0.2}))
        decision = self._decide(decider, {"demand": 0.0})
        with self.assertRaises(DecisionGateError):
            other.approve(decision.decision_id, "ops-lead")


# ---------------------------------------------------------------- 6. feedback ledger


class TestFeedbackLedger(unittest.TestCase):
    def test_pinball_loss_hand_computed(self):
        # Residuals d = 14 - q are [4, 3, 2, 1, 0, -1, -2, -3, -4]; losses sum to 4.0 over nine levels.
        self.assertAlmostEqual(pinball_loss([ROW], [14.0], QUANTILE_LEVELS), 4.0 / 9.0)

    def test_pinball_loss_for_an_outcome_above_the_band(self):
        # d = 19 - q is [9, 8, ..., 1]; the sum of tau * d over the nine levels is 16.5.
        self.assertAlmostEqual(pinball_loss([ROW], [19.0]), 16.5 / 9.0)

    def test_interval_coverage_hand_computed(self):
        fc = Forecast("demand", 1, [14.0], [ROW])
        self.assertEqual(interval_coverage(fc, [14.0], 0.8), 1.0)
        self.assertEqual(interval_coverage(fc, [19.0], 0.8), 0.0)
        self.assertEqual(interval_coverage(fc, [13.0], 0.5), 1.0)  # inside [11.5, 16.5]
        self.assertEqual(interval_coverage(fc, [17.0], 0.5), 0.0)

    def test_brier_score_hand_computed(self):
        self.assertAlmostEqual(brier_score(0.7, True), 0.09)
        self.assertAlmostEqual(brier_score(0.2, False), 0.04)
        self.assertAlmostEqual((brier_score(0.7, True) + brier_score(0.2, False)) / 2, 0.065)

    def test_mase_hand_computed(self):
        # In-sample naive scale is mean |diff| of [0..4] = 1.
        self.assertAlmostEqual(mase([9.0, 9.0, 9.0], [10.0, 10.0, 10.0], [0.0, 1.0, 2.0, 3.0, 4.0]), 1.0)
        self.assertAlmostEqual(mase([9.0, 9.0, 9.0], [11.0, 11.0, 11.0], [0.0, 1.0, 2.0, 3.0, 4.0]), 2.0)

    def test_record_outcome_matches_hand_computed_values(self):
        ledger = FeedbackLedger()
        decision = _make_decision(PolicyStatus.EXECUTE, thresholds={"demand": 12.0})
        ledger.record_decision(decision)
        outcome = ledger.record_outcome(decision.decision_id, {"demand": [14.0]})
        entry = outcome["series"]["demand"]
        self.assertAlmostEqual(entry["pinball_loss"], 4.0 / 9.0)
        self.assertEqual(entry["covered_80"], 1)
        self.assertEqual(entry["covered_50"], 1)
        # One step, so both bounds equal P(> 12) = 1 - 0.3.
        self.assertAlmostEqual(entry["predicted_breach_lower"], 0.7)
        self.assertAlmostEqual(entry["predicted_breach_upper"], 0.7)
        self.assertTrue(entry["realized_breach"])
        self.assertAlmostEqual(entry["brier_lower"], 0.09)
        self.assertAlmostEqual(entry["brier_upper"], 0.09)

    def test_outcome_can_only_be_recorded_once(self):
        ledger = FeedbackLedger()
        decision = _make_decision(PolicyStatus.EXECUTE)
        ledger.record_decision(decision)
        ledger.record_outcome(decision.decision_id, {"demand": [14.0]})
        with self.assertRaises(ValueError):
            ledger.record_outcome(decision.decision_id, {"demand": [14.0]})

    def test_calibration_report_aggregates_across_decisions(self):
        ledger = FeedbackLedger()
        first = _make_decision(PolicyStatus.EXECUTE)
        second = _make_decision(PolicyStatus.EXECUTE)
        ledger.record_decision(first)
        ledger.record_decision(second)
        ledger.record_outcome(first.decision_id, {"demand": [14.0]})   # inside the 80% band
        ledger.record_outcome(second.decision_id, {"demand": [19.0]})  # outside it
        report = ledger.calibration_report()
        series = report["series"]["demand"]
        self.assertEqual(report["decisions"], 2)
        self.assertEqual(report["decisions_with_outcomes"], 2)
        self.assertEqual(series["outcomes"], 2)
        self.assertAlmostEqual(series["coverage_80"], 0.5)
        self.assertAlmostEqual(series["mean_pinball_loss"], (4.0 / 9.0 + 16.5 / 9.0) / 2)
        self.assertEqual(series["target_coverage_80"], 0.8)
        self.assertIsNone(series["mean_brier_lower"])  # no thresholds on these decisions
        self.assertIsNone(series["mean_brier_upper"])

    def test_jsonl_round_trip_preserves_records_and_report(self):
        ledger = FeedbackLedger()
        scored = _make_decision(PolicyStatus.NEEDS_APPROVAL, thresholds={"demand": 12.0}, breach=0.7)
        ledger.record_decision(scored)
        ledger.record_approval(scored.decision_id, "ops-lead", "2026-01-02T00:00:00+00:00")
        ledger.record_outcome(scored.decision_id, {"demand": [14.0]})
        pending = _make_decision(PolicyStatus.ABSTAIN)
        ledger.record_decision(pending)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            ledger.to_jsonl(path)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            restored = FeedbackLedger.from_jsonl(path)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.get(scored.decision_id).to_dict(), ledger.get(scored.decision_id).to_dict())
        self.assertEqual(restored.calibration_report(), ledger.calibration_report())
        self.assertEqual(restored.get(pending.decision_id).status, "ABSTAIN")


# ---------------------------------------------------------------- 7. end to end with RhenEngine


def _torch_import_problem():
    try:
        importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on the sandbox
        return f"torch is not importable: {exc}"
    return None


class TestEndToEndWithRhenEngine(unittest.TestCase):
    def test_decide_runs_through_a_real_rhen_engine(self):
        problem = _torch_import_problem()
        if problem:
            self.skipTest(problem)
        from rhen.config import EOSConfig
        from rhen.rhen import RhenEngine

        config = EOSConfig(
            hidden_size=64,
            intermediate_size=128,
            num_pre_layers=1,
            num_recurrent_layers=1,
            num_post_layers=1,
            max_recurrent_loops=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        engine = RhenEngine(config=config)
        policy = DecisionPolicy(
            utility=lambda option, features: 1.0,
            min_confidence=0.0,
            max_forecast_uncertainty=math.inf,
        )
        ledger = FeedbackLedger()
        decider = ForecastAwareDecider(engine, NaiveQuantileForecaster(), policy=policy, ledger=ledger)

        decision = decider.decide(
            state="Warehouse snapshot: 40 units on hand, 2 suppliers.",
            question="Should we restock now?",
            options=["restock", "hold"],
            series=SERIES,
            horizon=6,
            thresholds={"demand": 1000.0},
        )

        self.assertIn("Forecast summary", decision.augmented_state)
        self.assertEqual(decision.forecasts["demand"].horizon, 6)
        self.assertEqual(set(decision.choice.probabilities), {"restock", "hold"})
        self.assertAlmostEqual(sum(decision.choice.probabilities.values()), 1.0, places=4)
        self.assertIn(decision.outcome.status, set(PolicyStatus))
        self.assertTrue(decision.outcome.reasons)
        self.assertEqual(len(ledger), 1)


# ---------------------------------------------------------------- 8. real TimesFM


def _timesfm_checkpoint_cached():
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub and os.environ.get("HF_HOME"):
        hub = os.path.join(os.environ["HF_HOME"], "hub")
    hub = Path(hub or Path.home() / ".cache" / "huggingface" / "hub")
    snapshots = hub / "models--google--timesfm-3.0-pytorch" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


@unittest.skipUnless(os.environ.get("RHEN_TIMESFM_RUN_REAL") == "1", "set RHEN_TIMESFM_RUN_REAL=1 to run the real TimesFM test")
@unittest.skipUnless(_timesfm_checkpoint_cached(), "google/timesfm-3.0-pytorch is not in the local Hugging Face cache")
class TestRealTimesFM(unittest.TestCase):
    def test_real_checkpoint_returns_well_formed_forecasts(self):
        forecaster = TimesFMForecaster(
            checkpoint_path="google/timesfm-3.0-pytorch",
            device="cpu",
            per_core_batch_size=8,
            license_acknowledged=True,
        )
        t = np.arange(96, dtype=np.float64)
        series = 50.0 + 5.0 * np.sin(2 * np.pi * t / 12.0)
        (fc,) = forecaster.predict([series.tolist()], horizon=12)
        self.assertEqual(fc.horizon, 12)
        self.assertEqual(len(fc.mean), 12)
        self.assertTrue(all(math.isfinite(v) for v in fc.mean))
        self.assertEqual(len(fc.quantiles), 12)
        self.assertEqual(len(fc.quantiles[0]), len(QUANTILE_LEVELS))


if __name__ == "__main__":
    unittest.main()
