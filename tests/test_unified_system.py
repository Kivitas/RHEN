"""
EOS LLM & RHEN SYSTEM ONE - COMPREHENSIVE UNIFIED VERIFICATION SUITE.

Validates the core architectural pillars:
1. Baseline Rhen System One Primitives & Invariants (2D Mask, Packer, H=4 Pointer Head, Primitives, Multi-Task Losses)
2. Rhen decision layer & AdvancedRhenEngine (typed primitives and cascades on one evaluate() path)
3. Single decision routing (the three-layer pipeline was removed; generation requests are rejected)
4. Meta-Cognitive Monitoring (ECE 10-bin calibration, Platt scaling, Anomaly detection, Self-correction)
5. Agent Tools & Action Card Staging Engine (Preflight verification, Risk tiering, Lifecycle transitions)
6. Context Builder & UnifiedAIOrchestrator (JSON/YAML/CSV/Tabular, Entities, Multimodal, End-to-End Execution)
"""

import torch

from config import EOSConfig
from model import EOSModel, MultiHeadRhenPointerHead
from optim.rhen_mask import Rhen2DBranchMask
from rhen_packer import get_rhen_packer
from rhen import (
    RhenEngine, AdvancedRhenEngine,
    Noul, Choice, Score,
    NoulResult, ChoiceResult, ScoreResult, CascadeResult,
)
from rhen_loss import compute_rhen_multi_task_loss
from meta_cognitive_monitor import MetaCognitiveMonitor
from agent_tools import ActionCard, ActionStagingEngine
from context_builder import ContextBuilder
from unified_orchestrator import UnifiedAIOrchestrator


def run_suite_1_baseline_rhen():
    print("\n" + "=" * 70)
    print("SUITE 1: Baseline Rhen System One Invariants & Primitives")
    print("=" * 70)

    # 1. 2D Block-Causal Branch Mask
    state_len = 4
    branch_lens = [3, 2, 4]
    mask = Rhen2DBranchMask.create_mask(state_len, branch_lens).squeeze()
    total_len = state_len + sum(branch_lens)
    assert mask.shape == (total_len, total_len), f"Shape mismatch: {mask.shape}"
    assert torch.all(mask[:state_len, :state_len] == 0.0), "State prefix must attend bidirectionally"

    curr = state_len
    for b_idx, b_len in enumerate(branch_lens):
        b_end = curr + b_len
        assert torch.all(mask[curr:b_end, :state_len] == 0.0), f"Branch {b_idx} must attend to state prefix"
        branch_block = mask[curr:b_end, curr:b_end]
        tril_idx = torch.tril_indices(b_len, b_len, offset=0)
        triu_idx = torch.triu_indices(b_len, b_len, offset=1)
        assert torch.all(branch_block[tril_idx[0], tril_idx[1]] == 0.0), f"Branch {b_idx} lower triangle must be 0.0"
        assert torch.all(branch_block[triu_idx[0], triu_idx[1]] == float("-inf")), f"Branch {b_idx} upper triangle must be -inf"
        
        other_start = state_len
        for other_idx, other_len in enumerate(branch_lens):
            other_end = other_start + other_len
            if other_idx != b_idx:
                assert torch.all(mask[curr:b_end, other_start:other_end] == float("-inf")), \
                    f"Branch {b_idx} must be isolated from Branch {other_idx}"
            other_start = other_end
        curr = b_end
    print("  [PASS] 1.1 2D Block-Causal Branch Mask strictly isolates concurrent branches.")

    # 2. Delimiter Packaging
    packer = get_rhen_packer()
    state = {
        "user_id": 1042,
        "subscription": "pro_tier",
        "last_login_hours": 3,
        "failed_attempts": 0,
    }
    questions = [
        ("Is user verified?", ["Yes", "No"]),
        ("Upgrade path?", ["Enterprise", "Team", "None"]),
        ("Risk severity?", ["Level 0", "Level 1", "Level 2", "Level 3"]),
    ]
    packed = packer.pack(state, questions)
    assert packed.state_length > 0, "State length must be > 0"
    assert len(packed.branches) == 3, "Must have 3 branches"
    assert packed.input_ids.shape[1] == packed.state_length + sum(packed.branch_lengths)
    for b_idx, b_meta in enumerate(packed.branches):
        assert b_meta.decide_idx > 0, f"Branch {b_idx} decide_idx must be valid"
        assert len(b_meta.option_end_indices) == len(questions[b_idx][1]), \
            f"Branch {b_idx} option count mismatch"
        for opt_idx in b_meta.option_end_indices:
            assert opt_idx < b_meta.decide_idx, f"Option index {opt_idx} must precede decide token {b_meta.decide_idx}"
    print("  [PASS] 1.2 Delimiter Packaging accurately coordinates <|decide|> and <|/opt|> indices.")

    # 3. Multi-Head Factorized Pointer Head
    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    pointer_head = MultiHeadRhenPointerHead(hidden_size=560, num_heads=4, rank=128)
    q_repr = torch.randn(1, 1, 560)
    opt_reprs = torch.randn(1, 5, 560)
    details = pointer_head(q_repr, opt_reprs, return_details=True)
    assert details["probabilities"].shape == (1, 5), "Probabilities shape mismatch"
    assert torch.allclose(details["probabilities"].sum(dim=-1), torch.ones(1), atol=1e-5), \
        "Probabilities must sum to 1.0"
    assert details["head_scores"].shape == (1, 5, 4), "Head scores shape mismatch"
    assert details["temperature"].item() > 0.0, "Temperature must be positive"
    print(f"  [PASS] 1.3 Multi-Head Pointer Head: H=4 heads, rank=128, tau={details['temperature'].item():.3f}")

    # 4. Typed Primitives Execution
    model = EOSModel(config)
    engine = RhenEngine(model, packer.tokenizer)
    eval_state = "Telemetry: Reactor core temp=850C, coolant_flow=normal."
    noul_res = engine.noul(eval_state, "Is core stable?")
    assert isinstance(noul_res, NoulResult)
    choice_res = engine.choice(eval_state, "Primary recommendation", ["Monitor", "Throttle", "Scram"])
    assert isinstance(choice_res, ChoiceResult)
    score_res = engine.score(eval_state, "Safety Rating", min_score=1.0, max_score=5.0)
    assert isinstance(score_res, ScoreResult)
    print("  [PASS] 1.4 Typed Decision Primitives (Noul, Choice, Score) evaluated successfully.")

    # 5. Multi-Task Losses
    labels = [True, 1, 2.0]
    total_loss, breakdown = compute_rhen_multi_task_loss(model, packed, labels, torch.device("cpu"))
    assert total_loss.item() > 0.0, "Total loss must be positive"
    assert "b0_noul_bce" in breakdown
    assert "b1_choice_ce" in breakdown
    assert "b2_score_rps" in breakdown
    assert "rhen_ortho" in breakdown
    print(f"  [PASS] 1.5 Multi-Task Losses: Total={total_loss.item():.4f}, Ortho={breakdown.get('rhen_ortho', 0.0):.4f}")


def run_suite_2_enhanced_primitives():
    print("\n" + "=" * 70)
    print("SUITE 2: Rhen Decision Layer & AdvancedRhenEngine")
    print("=" * 70)

    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    engine = AdvancedRhenEngine(config)
    state = "Cluster health: node_1=OK, node_2=HIGH_CPU (98%), network=STABLE."
    questions = [
        Noul("Is node_2 healthy?"),
        Choice("Mitigation", ["Restart node_2", "Drain traffic", "Ignore alert"]),
        Score("Incident severity", min_score=0.0, max_score=10.0),
    ]

    # All typed primitives go through one evaluate() call on the single decision layer.
    results = engine.evaluate(state, questions)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    noul_res, choice_res, score_res = results
    assert isinstance(noul_res, NoulResult)
    assert isinstance(choice_res, ChoiceResult)
    assert isinstance(score_res, ScoreResult)
    assert choice_res.conformal_set is not None and choice_res.free_energy is not None
    print("  [PASS] 2.1 Typed primitives evaluated in one pass with conformal set and free energy.")

    # Cascade through the base RhenEngine implementation.
    cascade_stages = [
        Noul("Requires immediate intervention?"),
        Choice("Tier selection", ["Tier 1", "Tier 2", "Tier 3"]),
    ]
    cascade_res = engine.cascade(state, cascade_stages)
    assert isinstance(cascade_res, CascadeResult)
    print("  [PASS] 2.2 Cascade Decision Graph execution verified.")


def run_suite_4_meta_cognitive_monitor():
    print("\n" + "=" * 70)
    print("SUITE 4: Meta-Cognitive Monitoring & Calibration")
    print("=" * 70)

    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    monitor = MetaCognitiveMonitor(config)

    # 1. 10-Bin ECE calculation
    # Simulate well-calibrated vs miscalibrated data
    sample_decisions = [
        {"confidence": 0.95, "ground_truth": 1},
        {"confidence": 0.85, "ground_truth": 1},
        {"confidence": 0.70, "ground_truth": 0},
        {"confidence": 0.40, "ground_truth": 0},
        {"confidence": 0.10, "ground_truth": 0},
    ]
    ece = monitor.compute_ece(sample_decisions, n_bins=10)
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0, 1], got {ece}"
    print(f"  [PASS] 4.1 Expected Calibration Error (ECE) across 10 bins: {ece:.4f}")

    # 2. Platt Scaling & Temperature Calibration
    scaled_conf = monitor.apply_platt_scaling(0.65, temperature=1.2)
    assert 0.0 < scaled_conf < 1.0, f"Scaled confidence invalid: {scaled_conf}"
    print(f"  [PASS] 4.2 Platt Scaling & Temperature Adjustment: 0.65 -> {scaled_conf:.4f}")

    # 3. Uncertainty Decomposition
    unc = monitor.decompose_uncertainty([0.7, 0.2, 0.1])
    assert "aleatoric_uncertainty" in unc
    assert "epistemic_uncertainty" in unc
    print(f"  [PASS] 4.3 Epistemic vs. Aleatoric Decomposition: Aleatoric={unc['aleatoric_uncertainty']:.3f}, Epistemic={unc['epistemic_uncertainty']:.3f}")

    # 4. Anomaly Detection & Self-Correction
    anomaly_flag = monitor.detect_confidence_anomaly(confidence=0.99, probabilities=[0.34, 0.33, 0.33])
    assert anomaly_flag is True, "Entropy-confidence mismatch anomaly must be flagged"
    print("  [PASS] 4.4 Anomaly Detection correctly identified entropy-confidence divergence.")


def run_suite_5_agent_tools_action_staging():
    print("\n" + "=" * 70)
    print("SUITE 5: Agent Tools & Action Card Staging Engine")
    print("=" * 70)

    staging = ActionStagingEngine(
        default_granted_permissions=["read", "write", "execute", "query", "notify"],
        secret_key=b"unified-suite5-signing-key",
    )

    # 1. Stage an Action Card
    mock_decision = {"choice_label": "Isolate Subnet", "confidence": 0.90}
    card = staging.stage_action(
        decision_result=mock_decision,
        action_type="execute",
        payload={"subnet": "10.0.1.0/24", "action": "block"},
        persona="SecurityAgent",
        title="Quarantine Compromised Subnet",
    )
    assert isinstance(card, ActionCard)
    assert card.status == "staged"
    assert card.risk_tier in ["medium", "high", "low"]
    print(f"  [PASS] 5.1 Action Card staged: ID={card.id}, Title='{card.title}', Tier={card.risk_tier}")

    # 2. Preflight Verification
    verified, errors = staging.verify_action(card)
    assert verified, f"Verification failed with: {errors}"
    print("  [PASS] 5.2 Preflight verification checks passed (Permissions & Safety Thresholds).")

    # 3. Lifecycle Transitions: approval requires a passing preflight for this revision
    preflight = staging.preflight_verification(card.id)
    assert preflight["allowed"] is True
    approved = staging.approve_action(card.id, approver="SecurityAdmin")
    assert approved.status == "approved"
    
    executed = staging.execute_action(
        card.id,
        executor_fn=lambda p: {"status": "success", "blocked_cidr": p["subnet"]},
    )
    assert executed.status == "executed"
    assert executed.execution_result["status"] == "success"
    print("  [PASS] 5.3 Lifecycle transitions verified: staged -> approved -> executed.")

    # 4. Markdown Card Formatting
    md_str = executed.to_markdown()
    assert "### Action Card:" in md_str
    assert "EXECUTED" in md_str
    print("  [PASS] 5.4 Formatted Markdown Action Card rendered cleanly.")


def run_suite_6_context_builder_and_orchestrator():
    print("\n" + "=" * 70)
    print("SUITE 6: Context Builder & End-to-End UnifiedAIOrchestrator")
    print("=" * 70)

    # 1. Context Builder: Ingesting Tabular, JSON, YAML, Entities, Multimodal
    builder = ContextBuilder()
    builder.add_entity("Host", "srv-db-01", {"env": "prod", "region": "us-east-1", "role": "primary"})
    builder.add_json({"db_status": "degraded", "active_connections": 1420, "max_connections": 1500}, section_name="Database Telemetry")
    builder.add_tabular([
        {"query_id": "q101", "duration_ms": 4500, "user": "reporting_bot"},
        {"query_id": "q102", "duration_ms": 3800, "user": "analytics_worker"},
    ], section_name="Slow Query Log")
    builder.add_audio_descriptor("Fan resonance high-pitch oscillation", duration_sec=4.5, detected_events=["high_frequency_hum"])
    builder.add_vision_descriptor("Thermal imaging panel shows GPU-3 at 92C hotspot", resolution=(1920, 1080), objects_detected=["GPU-3 Hotspot"])

    state_str = builder.build_state_string()
    assert "--- ENTITIES ---" in state_str
    assert "srv-db-01" in state_str
    assert "--- TABULAR RECORDS ---" in state_str or "--- SLOW QUERY LOG ---" in state_str
    assert "<|audio|>" in state_str
    assert "<|vision|>" in state_str
    print("  [PASS] 6.1 Context Builder assembled heterogeneous state (Entities + JSON + Tabular + Audio + Vision).")

    # 2. UnifiedAIOrchestrator End-to-End Execution (Sync & Async)
    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    orchestrator = UnifiedAIOrchestrator(config)

    questions = [
        Choice("Primary bottleneck", ["Connection pool saturation", "Unoptimized reporting queries", "Thermal throttling"]),
        Noul("Should reporting queries be terminated immediately?"),
    ]

    result = orchestrator.unified_decide_sync(
        state=builder,
        questions=questions,
        auto_stage_actions=True,
    )

    assert result["routing"] == "single_decision_layer"
    assert "final_answers" in result
    assert len(result["final_answers"]) == 2
    assert "staged_actions" in result
    assert len(result["staged_actions"]) == 2
    assert "meta_cognitive" in result
    assert result["latency_ms"] > 0.0

    print(f"  [PASS] 6.2 Unified Orchestrator executed in {result['latency_ms']:.1f}ms with {len(result['staged_actions'])} auto-staged action cards.")
    print("  [PASS] 6.3 Meta-Cognitive integration confirmed on final decisions.")

    print("\n" + "=" * 70)
    print("  ALL VERIFICATION TEST SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestUnifiedSystem(_unittest.TestCase):
    """test_unified_system verification suite (master battery)."""

    def test_suite_1_baseline_rhen(self):
        run_suite_1_baseline_rhen()

    def test_suite_2_enhanced_primitives(self):
        run_suite_2_enhanced_primitives()

    def test_suite_4_meta_cognitive_monitor(self):
        run_suite_4_meta_cognitive_monitor()

    def test_suite_5_agent_tools_action_staging(self):
        run_suite_5_agent_tools_action_staging()

    def test_suite_6_context_builder_and_orchestrator(self):
        run_suite_6_context_builder_and_orchestrator()

if __name__ == "__main__":
    run_suite_1_baseline_rhen()
    run_suite_2_enhanced_primitives()
    run_suite_4_meta_cognitive_monitor()
    run_suite_5_agent_tools_action_staging()
    run_suite_6_context_builder_and_orchestrator()
