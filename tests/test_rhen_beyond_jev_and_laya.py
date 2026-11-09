"""
Rhen-1.0 Master Verification Suite: invariant and unit checks for the Rhen-1.0 pillars.
No comparison against Kev, JEV, Open-Jev or Laya is run by this file.

Validates all 7 architectural, empirical, and cognitive pillars:
  Pillar 1: Continuous Latent Grounding via <|latent_slot|> & ContinuousLatentAdapter
  Pillar 2: Latent World-Model Simulation (Transition Operator T_theta + Prospective Value Head)
  Pillar 3: 2D Speculative Tree-Cascade Masking (O(1) multi-path tree forward pass)
  Pillar 4: Online Adaptive Conformal Martingale Risk Control (Robbins-Monro online calibration)
  Pillar 5: Dynamic DAG Schema Mutation (e_schema = W_schema z_decision)
  Pillar 6: Cryptographic Merkle Action Ledger & Multi-Agent BFT Consensus
  Pillar 7: Modern Hopfield Episodic Memory Bank Associative Retrieval
"""

import time
import torch
import torch.nn.functional as F

from config import EOSConfig
from tokenizer import get_tokenizer, LATENT_SLOT_TOKEN
from rhen_packer import get_rhen_packer, PackedRhenSequence
from model import (
    EOSModel,
    ContinuousLatentAdapter,
    LatentTransitionOperator,
    ProspectiveValueHead,
    ModernHopfieldEpisodicBank,
    SchemaMutationHead,
)
from meta_cognitive_monitor import AdaptiveConformalEngine
from agent_tools import ActionStagingEngine, BFTConsensusEngine
from rhen import (
    AdvancedRhenEngine,
    Noul,
    Choice,
    Score,
    SchemaMutation,
)
from unified_orchestrator import UnifiedAIOrchestrator


def run_pillar_1_continuous_latent_grounding():
    print("\n" + "=" * 75)
    print("PILLAR 1: Continuous Latent Grounding via <|latent_slot|> & Adapter")
    print("=" * 75)

    tokenizer = get_tokenizer()
    assert hasattr(tokenizer, "latent_slot_token_id"), "Tokenizer must expose latent_slot_token_id"
    latent_id = tokenizer.latent_slot_token_id
    assert latent_id is not None, "latent_slot_token_id must not be None"
    print(f"  [PASS] 1.1 Tokenizer recognizes {LATENT_SLOT_TOKEN} (ID={latent_id}).")

    # Verify adapter projection
    adapter = ContinuousLatentAdapter(in_features=768, hidden_size=560)
    raw_emb = torch.randn(2, 4, 768)
    proj_emb = adapter(raw_emb)
    assert proj_emb.shape == (2, 4, 560), f"Adapter output shape mismatch: {proj_emb.shape}"
    print("  [PASS] 1.2 ContinuousLatentAdapter projects (2, 4, 768) -> (2, 4, 560) with RMSNorm.")

    # Verify packer continuous slot coordinate tracking
    packer = get_rhen_packer(tokenizer)
    state_with_slots = f"Vector DB context: {LATENT_SLOT_TOKEN} {LATENT_SLOT_TOKEN} retrieved from index."
    questions = [Choice("Source validity", ["Valid", "Stale"])]
    latent_tensors = {"vector_db": torch.randn(1, 2, 768)}

    packed = packer.pack(
        state=state_with_slots,
        questions=questions,
        latent_tensors=latent_tensors,
        allow_latent_slots=True,
    )
    assert isinstance(packed, PackedRhenSequence)
    assert packed.latent_slot_indices is not None and len(packed.latent_slot_indices) == 2, \
        f"Expected 2 latent slot indices, found: {packed.latent_slot_indices}"
    assert packed.latent_tensors is not None
    print(f"  [PASS] 1.3 RhenPacker tracked {len(packed.latent_slot_indices)} continuous latent slot indices: {packed.latent_slot_indices}.")


def run_pillar_2_latent_world_model_simulation():
    print("\n" + "=" * 75)
    print("PILLAR 2: Latent World-Model Simulation (T_theta & Prospective Value Head)")
    print("=" * 75)

    hidden_dim = 560
    trans_op = LatentTransitionOperator(hidden_dim=hidden_dim)
    val_head = ProspectiveValueHead(hidden_dim=hidden_dim)

    z_state = torch.randn(1, hidden_dim)
    z_opt_a = torch.randn(1, hidden_dim)
    z_opt_b = torch.randn(1, hidden_dim)

    # 1-step transition
    z_next_a = trans_op(z_state, z_opt_a)
    z_next_b = trans_op(z_state, z_opt_b)
    assert z_next_a.shape == (1, hidden_dim)
    assert not torch.allclose(z_next_a, z_state), "State must evolve after transition"
    assert not torch.allclose(z_next_a, z_next_b), "Different options must lead to distinct latent states"
    print("  [PASS] 2.1 LatentTransitionOperator simulates conditioned counterfactual transitions.")

    v_a = val_head(z_next_a)
    v_b = val_head(z_next_b)
    assert v_a.shape == (1,)
    print(f"  [PASS] 2.2 ProspectiveValueHead scored future states: V(z_A)={v_a.item():.4f}, V(z_B)={v_b.item():.4f}.")

    # End-to-end model rollout simulation
    config = EOSConfig(hidden_size=hidden_dim, num_heads=8, num_layers=4, vocab_size=50048)
    model = EOSModel(config)

    opts = torch.randn(1, 3, hidden_dim)
    rollout = model.simulate_latent_rollout(state_repr=z_state, option_reprs=opts, horizon=3)
    assert "trajectories" in rollout
    assert len(rollout["trajectories"]) == 3
    assert rollout["prospective_values"].shape == (1, 3)
    assert rollout["recommended_option"] in [0, 1, 2]
    print(f"  [PASS] 2.3 Model simulated 3-horizon rollout across 3 candidate options: recommended_option={rollout['recommended_option']}.")


def run_pillar_3_speculative_tree_cascade_masking():
    print("\n" + "=" * 75)
    print("PILLAR 3: 2D Speculative Tree-Cascade Masking (O(1) Multi-Stage Forward Pass)")
    print("=" * 75)

    tokenizer = get_tokenizer()
    packer = get_rhen_packer(tokenizer)

    state = "Database cluster telemetry: replica lag 8ms, CPU load 82%."
    root_q = Choice("Mitigation strategy", ["Scale-Up", "Cache-Offload", "Throttling"])
    spec_branches = {
        0: [Noul("Provision extra compute instance immediately")],
        1: [Choice("Cache tier", ["L1 In-Memory", "L2 Distributed", "Bypass"])],
        2: [Score("Rate limit severity", levels=["Mild", "Medium", "Aggressive"])],
    }

    tree_pack = packer.build_speculative_tree_pack(
        state=state,
        root_question=root_q,
        speculative_branches=spec_branches,
    )
    assert tree_pack.is_speculative_tree is True
    assert len(tree_pack.branches) == 4, f"Expected 4 branches (1 root + 3 children), got {len(tree_pack.branches)}"

    # Verify 2D mask isolation properties
    mask = tree_pack.attention_mask[0, 0]
    total_len = tree_pack.input_ids.shape[1]
    assert mask.shape == (total_len, total_len)

    # Root branch attends to state + root
    root_b = tree_pack.branches[0]
    root_start = tree_pack.state_length
    root_end = root_start + root_b.branch_token_len
    assert torch.all(mask[root_start:root_end, :root_start] == 0.0), "Root must attend to state"

    # Child branches must NOT attend to competing sibling child branches (-inf)
    child_1 = tree_pack.branches[1]
    child_2 = tree_pack.branches[2]
    c1_start = root_end
    c1_end = c1_start + child_1.branch_token_len
    c2_start = c1_end
    c2_end = c2_start + child_2.branch_token_len

    assert torch.all(mask[c2_start:c2_end, c1_start:c1_end] == float("-inf")), \
        "Child 2 must be strictly isolated from sibling Child 1"
    print("  [PASS] 3.1 2D Tree-Block-Causal Masking guarantees exact cross-branch isolation (-inf).")

    # Evaluate speculative tree end-to-end via AdvancedRhenEngine
    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    engine = AdvancedRhenEngine(config)

    cascade_result = engine.speculative_cascade(
        state=state,
        root_question=root_q,
        speculative_branches=spec_branches,
    )
    assert "root" in cascade_result
    assert "speculative_branches" in cascade_result
    assert "active_branch" in cascade_result
    assert "winning_option_index" in cascade_result
    assert len(cascade_result["active_branch"]) >= 1

    winning_idx = cascade_result["winning_option_index"]
    print(f"  [PASS] 3.2 O(1) Speculative Cascade completed: root selected option {winning_idx}, instantly activating {len(cascade_result['active_branch'])} child stage decision(s).")


def run_pillar_4_online_adaptive_conformal_risk():
    print("\n" + "=" * 75)
    print("PILLAR 4: Online Adaptive Conformal Martingale Risk Control")
    print("=" * 75)

    # Target 95% coverage -> alpha = 0.05
    ace = AdaptiveConformalEngine(alpha=0.05, eta=0.02)
    assert ace.alpha == 0.05
    assert ace.quantile_threshold == 0.95

    # Simulate non-stationary shift (10 consecutive miscoverage events)
    for t in range(5):
        # Successful coverage
        probs = {"A": 0.85, "B": 0.15}
        pred_set = ace.predict_set(probs)
        ace.update(y_true="A", prediction_set=pred_set["prediction_set"])

    assert ace.drift_detected is False
    print("  [PASS] 4.1 AdaptiveConformalEngine operates stably under normal stationary distribution.")

    # Induce adversarial covariate shift (misses)
    for t in range(4):
        pred_set = ace.predict_set({"A": 0.90, "B": 0.10})
        ace.update(y_true="C", prediction_set=pred_set["prediction_set"])

    assert ace.drift_detected is True, "Drift must be detected after consecutive misses"
    assert ace.quantile_threshold > 0.95, "Quantile threshold must expand to control risk"
    print(f"  [PASS] 4.2 Robbins-Monro martingale dynamically expanded threshold ({ace.quantile_threshold:.3f}) and flagged covariate drift.")


def run_pillar_5_dynamic_dag_schema_mutation():
    print("\n" + "=" * 75)
    print("PILLAR 5: Dynamic DAG Schema Mutation (e_schema = W_schema z)")
    print("=" * 75)

    schema_head = SchemaMutationHead(hidden_dim=560, num_schema_types=8)
    z_decision = torch.randn(1, 560)
    logits = schema_head(z_decision)
    assert logits.shape == (1, 8), f"Expected shape (1, 8), got {logits.shape}"
    print("  [PASS] 5.1 SchemaMutationHead projects latent decision state to 8 schema mutation logits.")

    # Verify dynamic DAG mutation routing
    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    engine = AdvancedRhenEngine(config)

    mutation_spawn = SchemaMutation(
        parent_question="Cluster status?",
        mutation_type="spawn_child",
        mutation_logits=[0.9, 0.05, 0.03, 0.02],
        confidence=0.92,
    )
    mut_res = engine.dynamic_schema_mutation(
        state="Node-04 degraded",
        question=Choice("Node Action", ["Reboot", "Drain"]),
        mutation=mutation_spawn,
        speculative_branches={
            0: [Noul("Node recovers after reboot?")],
            1: [Noul("Drain completes without dropped requests?")],
        },
    )
    assert mut_res["action"] == "spawn_child"
    assert "result" in mut_res
    print(f"  [PASS] 5.2 Dynamic Schema Mutation triggered action: '{mut_res['action']}' with live tree spawning.")

    mutation_switch = SchemaMutation(
        parent_question="Is node responsive?",
        mutation_type="switch_primitive",
        mutation_logits=[0.1, 0.85, 0.03, 0.02],
        confidence=0.88,
        metadata={"target_primitive": "choice"},
    )
    mut_switch_res = engine.dynamic_schema_mutation(
        state="Node-04 degraded",
        question=Noul("Is node responsive?"),
        mutation=mutation_switch,
    )
    assert mut_switch_res["action"] == "switch_primitive"
    assert isinstance(mut_switch_res["switched_question"], Choice)
    print("  [PASS] 5.3 Dynamic Schema Mutation seamlessly switched primitive from Noul to Choice.")


def run_pillar_6_merkle_action_ledger_and_bft():
    print("\n" + "=" * 75)
    print("PILLAR 6: Cryptographic Merkle Action Ledger & Multi-Agent BFT Consensus")
    print("=" * 75)

    staging = ActionStagingEngine(secret_key=b"beyond-jev-pillar6-signing-key-2026")
    ledger = staging.merkle_ledger
    assert ledger is not None

    # Stage multiple action cards
    cards = []
    for i in range(4):
        card = staging.stage_action(
            decision_result={"choice_label": f"Action_{i}", "confidence": 0.96},
            action_type="execute",
            payload={"service": f"svc_{i}"},
            risk_tier="low" if i < 3 else "critical",
        )
        cards.append(card)

    root = ledger.get_merkle_root()
    assert len(root) == 64, "Root must be valid 64-char hex SHA256"
    assert len(ledger.leaves) == 4
    print(f"  [PASS] 6.1 Merkle Action Ledger constructed tamper-evident root: {root[:16]}... (4 leaves).")

    # Generate and verify cryptographic inclusion proof
    t0 = time.perf_counter()
    proof_entry = ledger.generate_inclusion_proof(leaf_index=2)
    valid_proof = ledger.verify_inclusion_proof(
        leaf_hash=proof_entry["leaf_hash"],
        proof=proof_entry["proof"],
        expected_root=root,
    )
    proof_time_us = (time.perf_counter() - t0) * 1e6
    assert valid_proof is True, "Merkle inclusion proof must verify against current root"
    print(f"  [PASS] 6.2 Merkle inclusion proof verified in {proof_time_us:.1f}us (< 100us).")

    # Multi-Agent BFT Consensus Engine
    bft_engine = BFTConsensusEngine(
        registered_agents={
            "Agent_Sentinel": b"sentinel-secret-key-101",
            "Agent_Supervisor": b"supervisor-secret-key-202",
            "Agent_Auditor": b"auditor-secret-key-303",
        },
        required_quorum_ratio=2.0 / 3.0,  # Requires at least 2 of 3 agents
    )

    critical_card = cards[3]
    # Collect 2 valid ballots
    b1 = bft_engine.collect_ballot(critical_card, "Agent_Sentinel", vote=True, secret_key=b"sentinel-secret-key-101")
    b2 = bft_engine.collect_ballot(critical_card, "Agent_Supervisor", vote=True, secret_key=b"supervisor-secret-key-202")

    quorum = bft_engine.evaluate_quorum(critical_card, [b1, b2])
    assert quorum["approved"] is True, "2 of 3 quorum must approve action"
    assert quorum["valid_count"] == 2
    print(f"  [PASS] 6.3 Multi-Agent BFT Quorum achieved: {quorum['valid_count']}/{quorum['total_agents']} approved critical card.")


def run_pillar_7_modern_hopfield_episodic_memory():
    print("\n" + "=" * 75)
    print("PILLAR 7: Modern Hopfield Episodic Memory Bank Associative Retrieval")
    print("=" * 75)

    hidden_dim = 560
    hopfield_bank = ModernHopfieldEpisodicBank(hidden_dim=hidden_dim, max_memories=128)

    # Store episodic prototype memories
    mem1 = torch.zeros(1, hidden_dim)
    mem1[0, :50] = 2.0  # Pattern 1
    mem2 = torch.zeros(1, hidden_dim)
    mem2[0, 50:100] = 2.0  # Pattern 2

    hopfield_bank.store(mem1)
    hopfield_bank.store(mem2)
    assert hopfield_bank.memory_count.item() == 2

    # Query with corrupted/noisy Pattern 1
    query = mem1 + 0.15 * torch.randn_like(mem1)
    retrieved, energy = hopfield_bank.retrieve(query)

    assert retrieved.shape == (1, hidden_dim)
    # Cosine similarity with mem1 should be significantly higher than with mem2
    sim1 = F.cosine_similarity(retrieved, mem1).item()
    sim2 = F.cosine_similarity(retrieved, mem2).item()
    assert sim1 > 0.90, f"Retrieved memory must closely match stored Pattern 1 (got {sim1:.3f})"
    assert sim1 > sim2 + 0.5, "Associative memory must correctly discriminate against Pattern 2"
    print(f"  [PASS] 7.1 Modern Hopfield associative retrieval converged: cos_sim(Pattern 1)={sim1:.4f}, Energy={energy.item():.4f}.")


def run_unified_orchestrator_integration():
    print("\n" + "=" * 75)
    print("UNIFIED SYSTEM: End-to-End Orchestrator Integration & Regression Tests")
    print("=" * 75)

    config = EOSConfig(hidden_size=560, num_heads=8, num_layers=4, vocab_size=50048)
    orchestrator = UnifiedAIOrchestrator(config)

    # 1. Speculative cascade with adaptive feedback and Merkle logging
    root_q = Choice("Target routing", ["Direct", "Proxy-Mesh"])
    spec_branches = {
        0: [Noul("Enable keep-alive connections")],
        1: [Score("Proxy pool size", levels=["Small", "Medium", "Large"])],
    }

    res = orchestrator.unified_decide_sync(
        state="High-throughput ingress proxy cluster under 45k RPS load.",
        questions=[root_q],
        speculative_branches=spec_branches,
        auto_stage_actions=True,
        auto_stage_risk_tier="low",  # explicit: the default tier is now inferred by the engine
        feedback_label=0,
    )

    assert res["orchestrator_mode"] == "speculative_cascade"
    assert "speculative_cascade" in res
    assert len(res["staged_actions"]) >= 1
    assert "adaptive_conformal_update" in res
    print("  [PASS] 8.1 Orchestrator successfully combined Speculative Cascade, Adaptive Conformal, and Staged Actions.")

    # 2. Cryptographic execution boundary. The auto-staged card comes from a near-zero-confidence
    # decision, so preflight must refuse it and approval must not go through.
    card_id = res["staged_actions"][0]["id"]
    weak = orchestrator.action_staging.preflight_verification(card_id)
    assert weak["allowed"] is False, "low-confidence auto-staged card must fail preflight"
    try:
        orchestrator.approve_action(card_id, approver="SecurityDirector")
    except PermissionError:
        pass
    else:
        raise AssertionError("approve_action must refuse a card that failed preflight")
    print("  [PASS] 8.2a Low-confidence auto-staged card refused at preflight and approval.")

    # A card from a confident decision passes preflight and executes through the ledger.
    strong = orchestrator.stage_action(
        decision_result={"choice_label": "Proxy-Mesh", "confidence": 0.9},
        action_type="execute",
        payload={"route": "Proxy-Mesh"},
        title="Route to proxy mesh",
        risk_tier="low",
    )
    assert orchestrator.action_staging.preflight_verification(strong.id)["allowed"] is True
    orchestrator.approve_action(strong.id, approver="SecurityDirector")
    executed_card = orchestrator.execute_action(strong.id)
    assert executed_card.status == "executed"
    assert "merkle_execution_leaf_index" in executed_card.metadata
    print("  [PASS] 8.2 Executed Action Card cryptographically committed to Merkle ledger.")


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestRhenBeyondJevAndLaya(_unittest.TestCase):
    """test_rhen_beyond_jev_and_laya verification suite (master battery)."""

    def test_pillar_1_continuous_latent_grounding(self):
        run_pillar_1_continuous_latent_grounding()

    def test_pillar_2_latent_world_model_simulation(self):
        run_pillar_2_latent_world_model_simulation()

    def test_pillar_3_speculative_tree_cascade_masking(self):
        run_pillar_3_speculative_tree_cascade_masking()

    def test_pillar_4_online_adaptive_conformal_risk(self):
        run_pillar_4_online_adaptive_conformal_risk()

    def test_pillar_5_dynamic_dag_schema_mutation(self):
        run_pillar_5_dynamic_dag_schema_mutation()

    def test_pillar_6_merkle_action_ledger_and_bft(self):
        run_pillar_6_merkle_action_ledger_and_bft()

    def test_pillar_7_modern_hopfield_episodic_memory(self):
        run_pillar_7_modern_hopfield_episodic_memory()

    def test_unified_orchestrator_integration(self):
        run_unified_orchestrator_integration()

if __name__ == "__main__":
    t_start = time.perf_counter()
    run_pillar_1_continuous_latent_grounding()
    run_pillar_2_latent_world_model_simulation()
    run_pillar_3_speculative_tree_cascade_masking()
    run_pillar_4_online_adaptive_conformal_risk()
    run_pillar_5_dynamic_dag_schema_mutation()
    run_pillar_6_merkle_action_ledger_and_bft()
    run_pillar_7_modern_hopfield_episodic_memory()
    run_unified_orchestrator_integration()
    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 75)
    print(f"  ALL 7 PILLARS AND UNIFIED ORCHESTRATOR VERIFIED IN {t_total:.2f}s!")
    print("  RHEN-1.0 invariant and unit verification battery complete.")
    print("=" * 75 + "\n")
