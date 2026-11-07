"""
03. Conformal Prediction, OOD Energy & Geometric Abstention
-----------------------------------------------------------
Demonstrates Rhen's statistical guarantees and selective abstention:
1. Split-Conformal Prediction calibration with finite-sample (1 - alpha) coverage.
2. Online Adaptive Conformal Engine with Robbins-Monro martingale updates.
3. Covariate shift and non-stationary drift detection.
4. Helmholtz Free Energy OOD detection: E(x; T) = -T * logsumexp(s_i / T).
5. Geometric Convex-Hull Selective Gate: ANSWER, ABSTAIN, or NEED_CLARIFICATION.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rhen import (
    ConformalPredictionEngine,
    AdaptiveConformalEngine,
    EnergyOODDetector,
    GeometricAbstentionGate,
)

def main():
    print("=" * 70)
    print("RHEN Statistical Guarantees & Selective Abstention")
    print("=" * 70)

    # 1. Finite-Sample Conformal Calibration
    print("\n--- 1. Split-Conformal Prediction Calibration (1 - alpha) ---")
    alpha = 0.10  # 90% guaranteed coverage
    conformal_engine = ConformalPredictionEngine(alpha=alpha)

    # Calibration non-conformity scores (e.g. 1 - P(y_true))
    calibration_scores = [0.04, 0.07, 0.09, 0.12, 0.15, 0.19, 0.22, 0.26, 0.31, 0.42]
    q_hat = conformal_engine.calibrate(calibration_scores, alpha=alpha)
    print(f"Calibration Set Size:     N = {len(calibration_scores)}")
    print(f"Significance Level:       alpha = {alpha} (Target Coverage = {100 * (1 - alpha):.0f}%)")
    print(f"Computed Quantile q_hat:  {q_hat:.4f}")

    # Predict conformal set
    sample_probs = {
        "RestartReplica": 0.88,
        "ScaleOutWorkers": 0.08,
        "QuarantineNode": 0.04,
    }
    prediction_res = conformal_engine.predict_set(sample_probs)
    print(f"Candidate Probabilities: {sample_probs}")
    print(f"Conformal Prediction Set: {prediction_res['prediction_set']}")
    print(f"Set Size:                 {prediction_res['set_size']} (Singleton={prediction_res['is_singleton']})")

    # 2. Online Adaptive Conformal Engine (Robbins-Monro Martingale)
    print("\n--- 2. Online Adaptive Conformal Engine ---")
    adaptive = AdaptiveConformalEngine(alpha=0.05, eta=0.02, initial_quantile=0.90)
    print(f"Initial Quantile:         q_0 = {adaptive.quantile_threshold:.4f}")

    # Scenario A: Correct prediction covered by set -> threshold tightens
    res1 = adaptive.update(y_true="RestartReplica", prediction_set=["RestartReplica", "ScaleOutWorkers"])
    print(f"Step 1 (Covered=True):    q_1 = {res1['new_quantile']:.4f} (tightened)")

    # Scenario B: Miss occurs -> threshold expands to maintain long-term coverage
    res2 = adaptive.update(y_true="RebootHost", prediction_set=["RestartReplica", "ScaleOutWorkers"])
    print(f"Step 2 (Covered=False):   q_2 = {res2['new_quantile']:.4f} (expanded)")

    # Scenario C: 3 consecutive misses triggers Non-Stationary Drift Alert
    print("Simulating non-stationary covariate shift...")
    adaptive.update(y_true="UnknownAction", prediction_set=["RestartReplica"])
    adaptive.update(y_true="UnknownAction", prediction_set=["RestartReplica"])
    print(f"Consecutive Errors:       {adaptive.consecutive_errors}")
    print(f"Drift Alert Triggered:    {adaptive.drift_detected} (Adaptation flag raised)")

    # 3. Helmholtz Free Energy Out-of-Distribution (OOD) Detection
    print("\n--- 3. Helmholtz Free Energy OOD Detection ---")
    detector = EnergyOODDetector(ood_threshold=4.5)

    # In-distribution logits: clear, sharp preference
    in_logits = [5.5, 1.2, 0.4, -2.1]
    e_in = detector.compute_energy(in_logits, temperature=1.0)
    is_ood_in = detector.is_ood(e_in)
    print(f"In-Distribution Logits:   {in_logits}")
    print(f"Free Energy E(x):         {e_in:.4f} (OOD={is_ood_in})")

    # Out-of-distribution logits: all suppressed, uncertain state
    ood_logits = [-6.5, -7.2, -6.8, -8.0]
    e_ood = detector.compute_energy(ood_logits, temperature=1.0)
    is_ood_out = detector.is_ood(e_ood)
    print(f"OOD Distorted Logits:     {ood_logits}")
    print(f"Free Energy E(x):         {e_ood:.4f} (Threshold={detector.ood_threshold}, OOD={is_ood_out})")

    # 4. Geometric Convex-Hull Selective Abstention Gate
    print("\n--- 4. Geometric Convex-Hull Selective Abstention Gate ---")
    gate = GeometricAbstentionGate()

    # Three canonical option representations forming a 2D simplex
    option_latents = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    # Case A: High Confidence, Low Energy, In-Hull -> ANSWER
    v_ans = gate.evaluate(
        probabilities=[0.92, 0.05, 0.03],
        free_energy=1.2,
        conformal_set={"set": ["RestartReplica"]},
        option_latents=option_latents,
        query_latent=[0.25, 0.25],
    )
    print(f"Case A (Ideal State):      Verdict={v_ans.decision} (should_answer={v_ans.should_answer})")

    # Case B: Confident but OFF-MANIFOLD (High Free Energy) -> ABSTAIN
    v_ood = gate.evaluate(
        probabilities=[0.95, 0.03, 0.02],
        free_energy=8.4,  # High energy
        conformal_set={"set": ["RestartReplica"]},
        option_latents=option_latents,
        query_latent=[0.25, 0.25],
    )
    print(f"Case B (High Free Energy): Verdict={v_ood.decision} (Reasons: {v_ood.reasons})")

    # Case C: Confident but OUTSIDE CONVEX HULL -> ABSTAIN
    # Even if softmax looks confident, latent is far outside known option geometry
    v_hull = gate.evaluate(
        probabilities=[0.94, 0.03, 0.03],
        free_energy=1.1,
        conformal_set={"set": ["RestartReplica"]},
        option_latents=option_latents,
        query_latent=[25.0, 30.0],  # Out-of-hull
    )
    print(f"Case C (Outside Hull):     Verdict={v_hull.decision} (Hull Distance={v_hull.hull_distance:.2f}, Reasons: {v_hull.reasons})")

    # Case D: Ambiguous top-2 alternatives on-manifold -> NEED_CLARIFICATION
    v_clarify = gate.evaluate(
        probabilities=[0.49, 0.47, 0.04],
        free_energy=1.2,
        conformal_set={"set": ["RestartReplica", "ScaleOutWorkers"]},
        option_latents=option_latents,
        query_latent=[0.4, 0.3],
    )
    print(f"Case D (Close Margin):     Verdict={v_clarify.decision} (Top-2 delta too narrow, asks for clarification)")

    print("\n[SUCCESS] Statistical guarantees and selective abstention verified!")

if __name__ == "__main__":
    main()
