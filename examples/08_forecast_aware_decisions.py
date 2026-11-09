"""
08. Forecast-Aware Decisions: Predictive Layer -> RHEN Choice -> Policy Gate -> Feedback
---------------------------------------------------------------------------------------
Demonstrates the forecast-aware decision pipeline:
1. Forecast a demand series with the dependency-free naive baseline (default).
2. Condition a RhenEngine choice on a deterministic forecast summary.
3. Gate the choice with a DecisionPolicy (constraints, confidence, uncertainty, risk tier).
4. Execute only when the gate allows it, with explicit approval for NEEDS_APPROVAL.
5. Record the realized outcome and print a calibration report.

Run:
    python examples/08_forecast_aware_decisions.py
    python examples/08_forecast_aware_decisions.py --timesfm --accept-noncommercial-license

The TimesFM 3.0 weights are licensed for non-commercial, non-production use only.
TimesFM is used only when both flags are passed, and the flag name states the restriction.
The RHEN engine here is randomly initialised (the repository ships no trained decision
weights), so the choice probabilities are illustrative only.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from rhen import (
    DecisionPolicy,
    FeedbackLedger,
    ForecastAwareDecider,
    NaiveQuantileForecaster,
    TimesFMForecaster,
)
from rhen.config import EOSConfig
from rhen.decision_pipeline import DecisionGateError
from rhen.rhen import RhenEngine

HISTORY = {
    "demand": [120.0, 131.0, 118.0, 125.0, 140.0, 137.0, 150.0, 144.0, 139.0, 152.0, 148.0, 160.0],
}
REALIZED = {"demand": [158.0, 171.0, 166.0, 169.0, 175.0, 162.0]}
THRESHOLDS = {"demand": 165.0}
HORIZON = 6
OPTIONS = ["restock_now", "hold"]
UTILITY = {"restock_now": 1.0, "hold": 0.8}


def build_engine() -> RhenEngine:
    torch.manual_seed(0)
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
    return RhenEngine(config=config)


def build_policy() -> DecisionPolicy:
    return DecisionPolicy(
        utility=lambda option, features: UTILITY[option],
        constraints=[
            # Restocking is blocked when the forecast median at the horizon end exceeds the budget line.
            ("restock_within_budget",
             lambda option, features: option != "restock_now"
             or features["series"]["demand"]["median_end"] <= 200.0),
        ],
        min_confidence=0.6,
        max_forecast_uncertainty=0.5,
        approval_required_tiers=("high", "critical"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesfm", action="store_true", help="use the TimesFM 3.0 forecaster (needs timesfm[torch])")
    parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        help="confirm your use of TimesFM 3.0 weights is non-commercial and non-production",
    )
    args = parser.parse_args()
    if args.accept_noncommercial_license and not args.timesfm:
        parser.error("--accept-noncommercial-license only applies together with --timesfm")
    if args.timesfm and not args.accept_noncommercial_license:
        parser.error(
            "TimesFM 3.0 weights are non-commercial only (timesfm-non-commercial-license-v1.0). "
            "Pass --accept-noncommercial-license to confirm your use is non-commercial and non-production."
        )

    print("=" * 70)
    print("RHEN Forecast-Aware Decisions")
    print("=" * 70)

    forecaster = TimesFMForecaster(license_acknowledged=True) if args.timesfm else NaiveQuantileForecaster()
    print(f"Forecaster: {forecaster.name}")
    print("Engine: randomly initialised RhenEngine (illustrative probabilities)")

    ledger = FeedbackLedger()
    decider = ForecastAwareDecider(build_engine(), forecaster, policy=build_policy(), ledger=ledger)

    print("\n--- 1. Forecast and conditioned decision ---")
    decision = decider.decide(
        state="Warehouse: 40 units on hand, lead time 5 days, weekly demand history attached.",
        question="Should we restock now?",
        options=OPTIONS,
        series=HISTORY,
        horizon=HORIZON,
        thresholds=THRESHOLDS,
    )
    forecast = decision.forecasts["demand"]
    lower, upper = forecast.interval(0.8)
    print(f"Median path: {[round(v, 1) for v in forecast.mean]}")
    print(f"80% band at horizon end: [{lower[-1]:.1f}, {upper[-1]:.1f}]")
    print(f"P(demand > {THRESHOLDS['demand']:g} within horizon) = {decision.features['max_breach_probability']:.3f}")
    print("Choice probabilities: " + ", ".join(f"{k}={v:.3f}" for k, v in decision.choice.probabilities.items()))

    print("\n--- 2. Policy gate ---")
    print(f"Status: {decision.outcome.status.value}  option: {decision.outcome.option}  "
          f"risk tier: {decision.outcome.risk_tier}")
    for reason in decision.outcome.reasons:
        print(f"  - {reason}")

    print("\n--- 3. Approval and execution ---")

    def restock(option, decided):
        print(f"  executor called with option={option!r} for {decided.decision_id}")
        return {"ordered": option == "restock_now"}

    if decision.outcome.status.value == "NEEDS_APPROVAL":
        approval = decider.approve(decision.decision_id, "ops-lead")
        print(f"  approved by {approval['approver']} at {approval['approved_at']}")
    try:
        result = decider.execute(decision, restock)
        print(f"  result: {result}")
    except DecisionGateError as exc:  # the gate refused: print why and continue to the feedback step
        print(f"  not executed: {exc}")

    print("\n--- 4. Feedback: realized outcome and calibration ---")
    ledger.record_outcome(decision.decision_id, REALIZED)
    report = ledger.calibration_report()
    series = report["series"]["demand"]
    print(f"Pinball loss: {series['mean_pinball_loss']:.2f}  "
          f"80% coverage: {series['coverage_80']:.2f} (target 0.80)  "
          f"Brier: {series['mean_brier']:.3f}")


if __name__ == "__main__":
    main()
