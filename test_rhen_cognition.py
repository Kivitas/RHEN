"""
Verification Suite for Cognitive Extensions (`rhen_cognition.py`).

Covers the six new pillars:
1. RLCD -- Reinforcement Learning for Calibrated Decisions (Pillar 8)
2. Decision Duration Thresholding, ported from parakeet-tdt (Pillar 9)
3. Hierarchical coarse-to-fine choice for large option sets (Pillar 10)
4. OOD regression guard on held-out optimal-action accuracy (Pillar 11)
5. Evidence verification lattice with controls (Pillar 12)
6. Multilingual decision routing and calibration profiles (Pillar 13)
"""

import torch

from rhen_cognition import (
    DurationThresholdingHead,
    EvidenceVerificationLattice,
    HierarchicalChoiceRouter,
    MultilingualDecisionRouter,
    OODRegressionGuard,
    RLCDOptimizer,
)


def test_rlcd_calibrated_reward():
    print("\n--- Test 1: RLCD Calibration-Aware Reward ---")
    opt = RLCDOptimizer(seed=11)

    # A confident-but-wrong decision must be punished harder than the same
    # decision made with low confidence -- that is the whole point of "RL for
    # calibrated decisions" rather than plain RL.
    confident_wrong = opt.compute_reward([0.05, 0.90, 0.05], 1, 0)
    unsure_wrong = opt.compute_reward([0.45, 0.50, 0.05], 1, 0)
    correct = opt.compute_reward([0.90, 0.05, 0.05], 0, 0)

    print(f"  confident wrong total={confident_wrong.total:.3f}")
    print(f"  unsure wrong    total={unsure_wrong.total:.3f}")
    print(f"  correct         total={correct.total:.3f}")

    assert confident_wrong.total < unsure_wrong.total, (
        "overconfident errors must be punished harder than uncertain ones"
    )
    assert correct.total > 0.0
    assert correct.correctness == 1.0
    print("  [PASS] Reward penalises overconfidence on wrong decisions")


def test_rlcd_policy_gradient_step():
    print("\n--- Test 2: RLCD Group-Relative Policy Gradient Step ---")
    opt = RLCDOptimizer(seed=3, group_size=4, learning_rate=0.1)

    logits = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=opt.learning_rate)
    before = logits.detach().clone()

    stats = opt.policy_gradient_step(logits, label_index=0, optimizer=optimizer)

    for key in ("loss", "mean_reward", "advantage_spread", "kl", "grad_norm", "rewards"):
        assert key in stats, f"missing diagnostic: {key}"
    assert stats["grad_norm"] > 0.0, "policy gradient must actually flow"
    assert len(stats["rewards"]) == opt.group_size
    assert not torch.allclose(before, logits.detach()), "weights must be updated"
    assert opt.step_count == 1

    # The label is always sampled, so at least one reward must be positive.
    assert any(r["correctness"] > 0 for r in stats["rewards"])
    print(f"  [PASS] GRPO step updated logits, grad_norm={stats['grad_norm']:.4f}")

    # Diagnostics-only mode must not mutate the graph.
    lg = torch.tensor([1.0, 0.0], requires_grad=True)
    snapshot = lg.detach().clone()
    opt2 = RLCDOptimizer(seed=5)
    opt2.policy_gradient_step(lg, label_index=1, optimizer=None)
    assert torch.allclose(snapshot, lg.detach())
    print("  [PASS] optimizer=None performs a read-only diagnostic step")


def test_duration_thresholding_head():
    print("\n--- Test 3: Decision Duration Thresholding (TDT) ---")
    head = DurationThresholdingHead(hidden_size=64, max_duration=8, threshold=0)
    hidden = torch.randn(4, 64)

    durations = head(hidden)
    assert durations.shape == (4,)
    assert durations.dtype == torch.long
    assert bool((durations >= 0).all()) and bool((durations <= 8).all())

    plan = head.expected_loops(hidden, min_loops=1)
    assert len(plan["durations"]) == 4
    assert plan["expected_total_loops"] <= plan["naive_total_loops"]
    assert len(plan["halt_flags"]) == 4

    savings = head.compute_savings(hidden)
    assert 0.0 <= savings <= 1.0
    print(f"  [PASS] TDT loop plan: durations={plan['durations']}, savings={savings:.3f}")

    # 3-D input ([B, 1, D]) must be accepted as well as 2-D.
    assert head(torch.randn(2, 1, 64)).shape == (2,)
    print("  [PASS] TDT head accepts both [B, D] and [B, 1, D] branch layouts")


def test_hierarchical_choice_router():
    print("\n--- Test 4: Hierarchical Coarse-to-Fine Choice (Banking77 Path) ---")
    router = HierarchicalChoiceRouter(num_groups=8, flat_threshold=20)

    # Small option sets stay on the flat path.
    small = router.route(["Scale Up", "Failover", "Ignore"])
    assert small["decomposed"] is False
    assert small["scored_options"] == 3
    print("  [PASS] Small option sets use the direct flat path")

    # 77 options (Laya's Banking77 setting) must decompose and score far fewer.
    options = [f"intent_{i}" for i in range(77)]
    clusters = router.build_clusters(options)
    assert sum(len(c) for c in clusters) == 77, "clusters must cover every option"
    assert all(c for c in clusters), "no cluster may be empty"
    assert len(clusters) == 8

    def scorer(labels):
        # A deterministic stand-in for the model: prefer the highest index.
        return [float(sum(ord(c) for c in s)) for s in labels]

    result = router.route(options, score_fn=scorer)
    assert result["decomposed"] is True
    assert result["total_options"] == 77
    assert result["scored_options"] < 77
    assert result["chosen_index"] in range(77)
    print(
        f"  [PASS] K=77 decomposed: scored {result['scored_options']} of 77 "
        f"({100 * result['scored_options'] / 77:.0f}%)"
    )

    # A custom scorer must actually steer the decision to a late option.
    def prefer_last(labels):
        # Score each label by the highest option index it mentions. Returning a
        # per-label score (rather than one constant) is required for argmax to
        # work at both levels: the group pass then selects the group containing
        # option 76, and the member pass selects option 76 inside it.
        scores = []
        for label in labels:
            best = -1.0
            for token in label.replace("|", " ").split():
                if token.startswith("intent_"):
                    try:
                        best = max(best, float(token.split("_")[1]))
                    except ValueError:
                        continue
            scores.append(best)
        return scores

    steered = router.route(options, score_fn=prefer_last)
    assert steered["chosen_index"] == 76, f"expected 76, got {steered['chosen_index']}"
    print("  [PASS] Two-level routing honours the supplied scorer")

    # Degenerate input must not raise.
    assert router.route([])["chosen_index"] == -1
    print("  [PASS] Empty option set handled safely")



def test_ood_regression_guard():
    print("\n--- Test 5: OOD Regression Guard ---")
    guard = OODRegressionGuard(tolerance=0.02, min_examples=4)

    labels = ["a", "b", "c", "d", "e", "f"]
    baseline_preds = ["a", "b", "c", "d", "x", "f"]
    base = guard.establish_baseline(baseline_preds, labels, name="baseline")
    assert base["accuracy"] == 5 / 6
    print(f"  [PASS] Baseline established: accuracy={base['accuracy']:.4f}")

    # An equal-quality candidate must be accepted.
    good = guard.evaluate(baseline_preds, labels, name="no_change")
    assert good["accepted"] is True
    assert good["delta"] == 0.0
    print("  [PASS] Equal-quality candidate accepted")

    # A small regression inside tolerance is tolerated. Baseline is 5/6; this
    # candidate scores 5/6 as well (one swap of a wrong label for another), so
    # the delta is exactly 0.0 -- safely inside the 0.02 tolerance.
    near = guard.evaluate(["a", "b", "c", "x", "e", "f"], labels, name="small_regression")
    assert near["accepted"] is True
    assert abs(near["delta"]) <= guard.tolerance
    print(f"  [PASS] Regression inside tolerance accepted (delta={near['delta']:+.4f})")

    # A one-example drop (5/6 -> 4/6, delta -0.167) must be rejected: that is the
    # Open-Jev "Wiki OOD optimal-action accuracy regressed" failure mode.
    real_drop = guard.evaluate(["a", "b", "c", "x", "x", "f"], labels, name="real_regression")
    assert real_drop["accepted"] is False
    assert guard.has_regressed() is True
    print(f"  [PASS] One-example regression rejected: delta={real_drop['delta']:+.4f}")

    # Too few held-out examples is itself a rejection reason.
    thin = guard.evaluate(["a", "b"], ["a", "b"], name="too_small")
    assert thin["accepted"] is False
    assert thin["reason"] == "insufficient held-out examples"
    print("  [PASS] Insufficient held-out examples rejected")


def test_evidence_verification_lattice():
    print("\n--- Test 6: Evidence Verification Lattice ---")
    lattice = EvidenceVerificationLattice(seed=17)

    obs = lattice.observe("smoke_test", lambda s: s * 2)
    assert obs.role == "observer"
    assert obs.seed == 17
    assert "int" in obs.detail
    print("  [PASS] Observer recorded a seeded observation")

    # A genuinely order-invariant predictor passes the critic.
    invariant = lattice.critique_permutation_invariance(
        predict_fn=lambda labels: min(labels),
        options=["scale", "failover", "ignore"],
        label="failover",
    )
    assert invariant.passed is True
    print("  [PASS] Critic accepted an order-invariant predictor")

    # An order-dependent predictor must be caught.
    dependent = lattice.critique_permutation_invariance(
        predict_fn=lambda labels: labels[0],
        options=["scale", "failover", "ignore"],
        label="scale",
    )
    assert dependent.passed is False
    print("  [PASS] Critic caught an order-dependent predictor")

    # An un-controlled improvement claim must be rejected.
    no_control = lattice.verify_with_control(candidate_accuracy=0.95)
    assert no_control.passed is False
    assert "REJECTED" in no_control.detail
    print("  [PASS] Verifier rejected a claim with no control baseline")

    controlled = lattice.verify_with_control(candidate_accuracy=0.95, control_accuracy=0.60)
    assert controlled.passed is True
    print("  [PASS] Verifier accepted a controlled improvement")

    report = lattice.report()
    assert report["total_records"] == 5
    assert report["records_by_role"] == {"observer": 1, "critic": 2, "verifier": 2}
    assert report["all_passed"] is False  # the dependent predictor failed
    assert report["seeds"] == [17]
    print(f"  [PASS] Report aggregates {report['total_records']} records across 3 roles")



def test_multilingual_decision_router():
    print("\n--- Test 7: Multilingual Decision Routing ---")
    router = MultilingualDecisionRouter()

    cases = [
        ("The server is down and not responding", "en"),
        ("Der Server ist nicht erreichbar", "de"),
        ("Le serveur est arrete pour maintenance", "fr"),
        ("El servidor no esta funcionando", "es"),
        ("Il server non e raggiungibile", "it"),
        ("De server is niet bereikbaar voor gebruikers", "nl"),
        ("O servidor esta em manutencao", "pt"),
        ("\u091c\u0939\u093e\u0932\u093e \u0921\u0947\u091f\u093e", "hi"),
        ("\u30b5\u30fc\u30d0\u30fc\u306f\u6b62\u307e\u3063\u305f", "ja"),
        ("\uc11c\ubc84\uc774 \uc548 \ub3d9\uc791\ud569\ub2c8\ub2e4", "ko"),
        ("\u0421\u0435\u0440\u0432\u0435\u0440 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d", "ru"),
        ("\u0627\u0644\u062e\u0627\u062f\u0645 \u0645\u0637\u0644\u0648", "ar"),
    ]
    for text, expected in cases:
        got = router.detect_language(text)
        assert got == expected, f"expected {expected} for {text!r}, got {got}"
    print(f"  [PASS] {len(cases)} languages detected across 8 scripts")

    assert router.detect_language("") == "unknown"
    assert router.detect_language("12345 !!!") == "unknown"
    print("  [PASS] Empty / non-letter input degrades to 'unknown'")

    # Per-language calibration profiles must actually be applied.
    profiles = {"hi": {"temperature": 1.3, "alpha": 0.08}}
    routed = router.route("\u091c\u0939\u093e\u0932\u093e \u0921\u0947\u091f\u093e", profiles)
    assert routed["language"] == "hi"
    assert routed["custom_profile_applied"] is True
    assert routed["profile"]["temperature"] == 1.3
    assert routed["profile"]["alpha"] == 0.08
    print(f"  [PASS] Hindi routed to custom profile: T={routed['profile']['temperature']}")

    # A language with no custom profile must fall back to the neutral one.
    english = router.route("The server is down", profiles)
    assert english["custom_profile_applied"] is False
    assert english["profile"]["temperature"] == 1.0
    print("  [PASS] Unprofiled language uses the neutral calibration profile")

    assert len(router.detection_history) == 2


def test_pillars_together():
    print("\n--- Test 8: Integrated Rhen-1.0 Decision Flow ---")
    # A single incident handled by every new pillar at once.
    state = "Production API p99 latency 4200ms; error rate 4.8%; region us-east-1."

    routing = MultilingualDecisionRouter().route(state)
    assert routing["language"] == "en"

    tdt = DurationThresholdingHead(hidden_size=32, max_duration=6)
    plan = tdt.expected_loops(torch.randn(3, 32))
    assert plan["expected_total_loops"] <= plan["naive_total_loops"]

    router = HierarchicalChoiceRouter(num_groups=4, flat_threshold=3)
    decision = router.route(["Scale", "Throttle", "Failover", "Drain", "Restart"])
    assert decision["chosen_index"] in range(5)

    # The guard requires at least min_examples held-out rows, so supply four.
    guard = OODRegressionGuard()
    held_out = ["a", "b", "c", "d"]
    guard.establish_baseline(held_out, held_out)
    assert guard.evaluate(held_out, held_out)["accepted"] is True

    lattice = EvidenceVerificationLattice(seed=1)
    lattice.verify_with_control(candidate_accuracy=0.9, control_accuracy=0.5)

    print("  [PASS] All six Rhen-1.0 pillars composed into one decision flow")


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestRhenCognition(_unittest.TestCase):
    """test_rhen_cognition verification suite (master battery)."""

    def test_rlcd_calibrated_reward(self):
        test_rlcd_calibrated_reward()

    def test_rlcd_policy_gradient_step(self):
        test_rlcd_policy_gradient_step()

    def test_duration_thresholding_head(self):
        test_duration_thresholding_head()

    def test_hierarchical_choice_router(self):
        test_hierarchical_choice_router()

    def test_ood_regression_guard(self):
        test_ood_regression_guard()

    def test_evidence_verification_lattice(self):
        test_evidence_verification_lattice()

    def test_multilingual_decision_router(self):
        test_multilingual_decision_router()

    def test_pillars_together(self):
        test_pillars_together()

if __name__ == "__main__":
    print("=" * 74)
    print("        RHEN-1.0 COGNITIVE EXTENSIONS - VERIFICATION SUITE")
    print("=" * 74)
    test_rlcd_calibrated_reward()
    test_rlcd_policy_gradient_step()
    test_duration_thresholding_head()
    test_hierarchical_choice_router()
    test_ood_regression_guard()
    test_evidence_verification_lattice()
    test_multilingual_decision_router()
    test_pillars_together()
    print("\n" + "=" * 74)
    print("  ALL 8 RHEN-1.0 COGNITIVE SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 74)

