"""
05. Hierarchical Routing, RLCD & Cognitive Extensions
-----------------------------------------------------
Demonstrates advanced cognitive mechanisms:
1. HierarchicalChoiceRouter: Coarse-to-fine routing for large option sets (77+ intents).
2. RLCDOptimizer: Reinforcement Learning for Calibrated Decisions (penalizing overconfidence).
3. DurationThresholdingHead (TDT): Dynamic recurrent loop reduction.
4. OODRegressionGuard: Preventing regression on held-out out-of-distribution benchmarks.
5. EvidenceVerificationLattice: Order-invariance and controlled verification protocol.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rhen import (
    HierarchicalChoiceRouter,
    RLCDOptimizer,
    DurationThresholdingHead,
    OODRegressionGuard,
    EvidenceVerificationLattice,
)

def main():
    print("=" * 70)
    print("RHEN Cognitive Extensions - Hierarchical Routing & RLCD")
    print("=" * 70)

    # 1. Hierarchical Coarse-to-Fine Choice (Banking77 Scale)
    print("\n--- 1. Hierarchical Choice Router (Large Action Spaces) ---")
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)

    # Simulate 77 realistic customer support intents (Banking77 dataset setting)
    intents = [
        f"card_issue_{i}" for i in range(15)
    ] + [
        f"transfer_wire_{i}" for i in range(15)
    ] + [
        f"account_auth_{i}" for i in range(15)
    ] + [
        f"loan_credit_{i}" for i in range(16)
    ] + [
        f"fraud_dispute_{i}" for i in range(16)
    ]
    print(f"Total Candidate Actions: K = {len(intents)} options")

    # Scorer that models preference for fraud_dispute_7 (index 68)
    target_intent = "fraud_dispute_7"
    def mock_intent_scorer(labels):
        return [
            10.0 if target_intent in lbl else float(len(lbl))
            for lbl in labels
        ]

    route_res = router.route(intents, score_fn=mock_intent_scorer)
    print(f"Decomposition Triggered: {route_res['decomposed']}")
    print(f"Total Options Scored:    {route_res['scored_options']} of {route_res['total_options']} "
          f"({100 * route_res['scored_options'] / route_res['total_options']:.1f}% scored)")
    print(f"Chosen Action Index:     {route_res['chosen_index']} ('{intents[route_res['chosen_index']]}')")

    # 2. RLCD: Reinforcement Learning for Calibrated Decisions
    print("\n--- 2. RLCD Calibration-Aware Reinforcement Learning ---")
    rlcd = RLCDOptimizer(seed=42, group_size=4, learning_rate=0.05)

    # Compute calibration-aware rewards for 3 cases:
    # Target true label = 0
    # A: Confident wrong ([0.05, 0.90, 0.05], chosen 1)
    # B: Unsure wrong    ([0.45, 0.50, 0.05], chosen 1)
    # C: Confident right ([0.92, 0.04, 0.04], chosen 0)
    r_conf_wrong = rlcd.compute_reward([0.05, 0.90, 0.05], 1, 0)
    r_unsure_wrong = rlcd.compute_reward([0.45, 0.50, 0.05], 1, 0)
    r_correct = rlcd.compute_reward([0.92, 0.04, 0.04], 0, 0)

    print(f"Case A (Confident Wrong):  Total Reward = {r_conf_wrong.total:+.4f} (Heavily penalised!)")
    print(f"Case B (Unsure Wrong):     Total Reward = {r_unsure_wrong.total:+.4f} (Mild penalty)")
    print(f"Case C (Confident Right):  Total Reward = {r_correct.total:+.4f} (Positive reward)")
    assert r_conf_wrong.total < r_unsure_wrong.total, "Confident errors must receive harsher penalties!"

    # Perform a Group-Relative Policy Optimization (GRPO) gradient step
    logits = torch.tensor([1.0, 3.2, 0.5], requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=rlcd.learning_rate)
    print(f"\nInitial Decision Logits:  {logits.detach().tolist()}")

    step_stats = rlcd.policy_gradient_step(logits, label_index=0, optimizer=optimizer)
    print(f"Updated Decision Logits:  {logits.detach().tolist()}")
    print(f"GRPO Loss:                {step_stats['loss']:.4f}")
    print(f"Gradient Norm:            {step_stats['grad_norm']:.4f}")
    print(f"Mean Advantage:           {step_stats['mean_reward']:.4f}")

    # 3. Decision Duration Thresholding (TDT)
    print("\n--- 3. Decision Duration Thresholding (TDT Loop Early-Exit) ---")
    hidden_dim = 128
    tdt_head = DurationThresholdingHead(hidden_size=hidden_dim, max_duration=6, threshold=0)

    # Simulate hidden state from decision branch
    mock_hidden = torch.randn(4, hidden_dim)
    plan = tdt_head.expected_loops(mock_hidden, min_loops=1)
    savings = tdt_head.compute_savings(mock_hidden)

    print("Loop Budget Allowed:      Max 6 recurrent iterations")
    print(f"Scheduled Loop Durations: {plan['durations']} per branch")
    print(f"Total Loops Needed:       {plan['expected_total_loops']} (vs naive {plan['naive_total_loops']})")
    print(f"Recurrent Compute Saved:  {savings * 100:.1f}% computation saved!")

    # 4. OOD Regression Guard
    print("\n--- 4. OOD Regression Guard ---")
    guard = OODRegressionGuard(tolerance=0.03, min_examples=5)
    held_out_labels = ["restart", "scale", "drain", "ignore", "notify", "escalate"]
    baseline_preds  = ["restart", "scale", "drain", "ignore", "notify", "escalate"]

    base_metric = guard.establish_baseline(baseline_preds, held_out_labels, name="v1.0-prod")
    print(f"Production Baseline Accuracy: {base_metric['accuracy']:.2%}")

    # Candidate model evaluation
    cand_preds = ["restart", "scale", "drain", "ignore", "restart", "escalate"]  # 1 mistake
    eval_res = guard.evaluate(cand_preds, held_out_labels, name="v1.1-candidate")
    print(f"Candidate Model Accuracy:    {eval_res['accuracy']:.2%} (Delta = {eval_res['delta']:+.2%})")
    print(f"Regression Guard Status:     Accepted = {eval_res['accepted']} (Regressed = {guard.has_regressed()})")

    # 5. Evidence Verification Lattice
    print("\n--- 5. Evidence Verification Lattice ---")
    lattice = EvidenceVerificationLattice(seed=123)

    # Critic check: Permutation Invariance check
    options = ["QuarantineHost", "RestartContainer", "NotifySlack"]
    perm_check = lattice.critique_permutation_invariance(
        predict_fn=lambda opts: min(opts),  # Invariant predictor
        options=options,
        label="NotifySlack",
    )
    print(f"Permutation Invariance Passed: {perm_check.passed}")

    # Verifier check: Controlled baseline verification
    controlled_verif = lattice.verify_with_control(candidate_accuracy=0.96, control_accuracy=0.62)
    print(f"Controlled Improvement Passed: {controlled_verif.passed}")

    print("\n[SUCCESS] Hierarchical Choice, RLCD & Cognitive Extensions fully operational!")

if __name__ == "__main__":
    main()
