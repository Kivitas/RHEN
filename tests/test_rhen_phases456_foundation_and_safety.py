"""
Verification Suite for Rhen Phases 4, 5 & 6:
Pillar 3: AI Foundation (Diff-MLA, SE-MoE Aux-Free Dynamic Balancing, SSM Bridge).
Pillar 4: Training Dynamics (DDPO Preference Loss, Option Permutation Invariance).
Pillar 5: Agentic Safety (HMAC-SHA256 Cryptographic Action Digests, Dual-Custody Consensus).
"""

import torch

from config import EOSConfig
from model import (
    EOSModel,
    MultiLatentAttention,
    SharedExpertFineGrainedMoE,
    StateSpaceMemoryHighway,
    MultiHeadRhenPointerHead,
)
from rhen_packer import get_rhen_packer
from rhen_loss import (
    compute_ddpo_loss,
    compute_permutation_invariance_loss,
    compute_rhen_multi_task_loss,
)
from agent_tools import ActionStagingEngine


def test_diff_mla_attention():
    print("\n--- Test 1: Differential Multi-Latent Attention (Diff-MLA) ---")
    config = EOSConfig(
        hidden_size=256,
        num_attention_heads=4,
        head_dim=64,
        kv_lora_rank=64,
        q_lora_rank=64,
        use_diff_mla=True,
    )
    assert getattr(config, "use_diff_mla", False) is True

    mla = MultiLatentAttention(config)
    assert hasattr(mla, "lambda_param")
    assert mla.lambda_param is not None

    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

    # Pre-generate RoPE cos/sin
    cos = torch.ones(seq_len, config.head_dim)
    sin = torch.zeros(seq_len, config.head_dim)

    attn_out, kv_cache = mla(hidden_states, cos=cos, sin=sin)
    assert attn_out.shape == (batch_size, seq_len, config.hidden_size)

    # Verify backward gradient flow through Diff-MLA lambda parameter
    loss = attn_out.sum()
    loss.backward()
    assert mla.lambda_param.grad is not None
    assert hidden_states.grad is not None
    print(f"  [PASS] Diff-MLA verified with learnable lambda={2.0 * torch.sigmoid(mla.lambda_param).mean().item():.4f}")


def test_se_moe_aux_free_balancing():
    print("\n--- Test 2: Shared-Expert MoE (SE-MoE) Auxiliary-Loss-Free Balancing ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_shared_experts=1,
        num_routed_experts=8,
        num_active_routed=2,
        use_aux_free_balancing=True,
    )
    moe = SharedExpertFineGrainedMoE(config)
    moe.train()

    assert hasattr(moe, "congestion_bias")
    assert hasattr(moe, "get_load_distribution")
    assert hasattr(moe, "reset_congestion_bias")

    # Run forward on synthetic tokens
    x = torch.randn(4, 16, config.hidden_size)
    out, rz_loss = moe(x)
    assert out.shape == x.shape
    assert rz_loss is not None

    # Verify load distribution
    load = moe.get_load_distribution()
    assert load.shape == (config.num_routed_experts,)
    assert torch.isclose(load.sum(), torch.tensor(1.0), atol=1e-5)

    # Congestion bias reset
    moe.reset_congestion_bias()
    assert torch.all(moe.congestion_bias == 0.0)
    print("  [PASS] SE-MoE verified: 1 shared + 8 routed experts, aux-free load balancing active.")


def test_ssm_memory_highway():
    print("\n--- Test 3: State-Space Memory Highway (SSM Bridge Across Loops) ---")
    ssm = StateSpaceMemoryHighway(hidden_size=256, state_dim=64)
    ssm.reset_state()
    assert ssm.memory_state is None

    batch_size = 2
    h1 = torch.randn(batch_size, 10, 256)
    out1 = ssm(h1, step_idx=1)
    assert ssm.memory_state is not None
    assert out1.shape == h1.shape

    # Step 2: Recalls and updates persistent memory state
    h2 = torch.randn(batch_size, 10, 256)
    out2 = ssm(h2, step_idx=2)
    assert ssm.memory_state is not None
    assert out2.shape == h2.shape

    # Reset
    ssm.reset_state()
    assert ssm.memory_state is None
    print("  [PASS] State-Space Memory Highway verified across recurrent loop passes.")


def test_ddpo_preference_loss():
    print("\n--- Test 4: Direct Decision Preference Optimization (DDPO) ---")
    # Positive preference margin (preferred score > rejected score)
    chosen_pos = torch.tensor([4.5, 3.2, 5.0])
    rejected_pos = torch.tensor([1.2, 0.5, 2.0])
    loss_small = compute_ddpo_loss(chosen_pos, rejected_pos, beta=0.1)

    # Inverted preference margin (preferred score < rejected score)
    chosen_neg = torch.tensor([1.2, 0.5, 2.0])
    rejected_neg = torch.tensor([4.5, 3.2, 5.0])
    loss_large = compute_ddpo_loss(chosen_neg, rejected_neg, beta=0.1)

    assert loss_small < loss_large
    assert loss_small > 0.0
    print(f"  [PASS] DDPO Loss calibrated: Positive margin loss={loss_small.item():.4f} < Inverted loss={loss_large.item():.4f}")


def test_permutation_invariance_loss():
    print("\n--- Test 5: Candidate Option Permutation Invariance Loss ---")
    pointer_head = MultiHeadRhenPointerHead(hidden_size=256, num_heads=4, rank=64)
    q_repr = torch.randn(1, 1, 256)
    opt_reprs = torch.randn(1, 4, 256)

    # Identical permutation -> KL == 0
    identity_perm = [0, 1, 2, 3]
    kl_zero = compute_permutation_invariance_loss(pointer_head, q_repr, opt_reprs, permutation=identity_perm)
    assert torch.isclose(kl_zero, torch.tensor(0.0), atol=1e-5)

    # Random permutation
    kl_perm = compute_permutation_invariance_loss(pointer_head, q_repr, opt_reprs)
    assert kl_perm >= 0.0
    print(f"  [PASS] Permutation Invariance Loss verified: Identity KL={kl_zero.item():.6f}, Permuted KL={kl_perm.item():.4f}")


def test_ddpo_and_perm_in_multitask_loss():
    print("\n--- Test 6: Multi-Task Loss with DDPO and Permutation Invariance ---")
    config = EOSConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        max_recurrent_loops=2,
    )
    model = EOSModel(config)
    model.train()

    packer = get_rhen_packer()
    state = {"system": "cluster-api", "status": "active"}
    questions = [
        ("Action choice", ["Scale Up", "Failover", "Ignore"], "choice", None),
    ]
    packed = packer.pack(state, questions)
    labels = [0]  # Target choice: 0

    total_loss, breakdown = compute_rhen_multi_task_loss(
        model=model,
        packed_sequence=packed,
        branch_labels=labels,
        device=torch.device("cpu"),
        ddpo_pairs=[(0, 1)],  # Preferred choice 0 over 1
        compute_perm_loss=True,
    )

    assert "rhen_ddpo" in breakdown
    assert "rhen_perm" in breakdown
    assert total_loss.requires_grad
    total_loss.backward()
    assert model.embed_tokens.weight.grad is not None
    print(f"  [PASS] Multi-task loss integrated with DDPO ({breakdown['rhen_ddpo']:.4f}) and Permutation loss ({breakdown['rhen_perm']:.4f})")


def test_cryptographic_action_cards():
    print("\n--- Test 7: Cryptographic Action Cards & HMAC-SHA256 Signatures ---")
    engine = ActionStagingEngine(secret_key=b"phases456-suite-signing-key")

    card = engine.stage_action(
        decision_result={"confidence": 0.88, "choice_label": "Quarantine Pod"},
        action_type="execute",
        payload={"pod_id": "pod-882", "zone": "us-west-2"},
        risk_tier="medium",
    )

    assert card.signature is not None
    assert len(card.signature) == 64  # SHA-256 hex string
    assert engine.verify_action_digest(card) is True

    # Tamper test: Altering payload breaks HMAC verification
    card.payload["pod_id"] = "pod-999-tampered"
    assert engine.verify_action_digest(card) is False
    print("  [PASS] HMAC-SHA256 signature generated and verified; payload tampering detected.")


def test_dual_custody_critical_actions():
    print("\n--- Test 8: Dual-Custody Multi-Agent Consensus for Critical Risk ---")
    engine = ActionStagingEngine(
        secret_key=b"phases456-suite-signing-key",
        registered_bft_agents={
            "Agent-Alpha": b"alpha-bft-secret-key-2026",
            "Agent-Beta": b"beta-bft-secret-key-2026",
            "Agent-Gamma": b"gamma-bft-secret-key-2026",
        },
    )
    # Co-signers must be registered out of band before they can sign.
    engine.register_cosigner_agent("SecurityAuditorAgent-07", b"security-auditor-secret-key-2026")

    # Stage critical action
    card = engine.stage_action(
        decision_result={"confidence": 0.96},
        action_type="execute",
        payload={"command": "DROP_DATABASE_CLUSTER"},
        risk_tier="critical",
        metadata={"conformal_set": {"is_singleton": True, "set_size": 1}},
    )

    # 1. Verification fails without co-signer
    preflight_single = engine.preflight_verification(card.id)
    assert preflight_single["allowed"] is False
    assert any("Dual-Custody" in c["name"] and not c["passed"] for c in preflight_single["checks"])

    # 2. Add secondary agent co-signature
    cosigned = engine.request_dual_custody_cosign(
        card_id=card.id,
        cosigner_agent_id="SecurityAuditorAgent-07",
        cosigner_key=b"security-auditor-secret-key-2026",
    )
    assert cosigned is True
    assert card.cosigner_agent_id == "SecurityAuditorAgent-07"
    assert card.cosigner_signature is not None

    # 3. A verified co-signature alone is not enough: the BFT quorum is mandatory too
    preflight_no_quorum = engine.preflight_verification(card.id)
    assert preflight_no_quorum["allowed"] is False
    assert any("BFT" in c["name"] and not c["passed"] for c in preflight_no_quorum["checks"])

    # 4. With a BFT quorum of pre-registered agents, verification passes
    for voter, key in (
        ("Agent-Alpha", b"alpha-bft-secret-key-2026"),
        ("Agent-Beta", b"beta-bft-secret-key-2026"),
        ("Agent-Gamma", b"gamma-bft-secret-key-2026"),
    ):
        engine.collect_bft_ballot(card.id, voter, True, key)
    preflight_dual = engine.preflight_verification(card.id)
    assert preflight_dual["allowed"] is True
    assert card.status == "approved"
    print("  [PASS] Dual-Custody consensus enforced: Critical action co-signed, quorum met, approved.")


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestRhenPhases456FoundationAndSafety(_unittest.TestCase):
    """test_rhen_phases456_foundation_and_safety verification suite (master battery)."""

    def test_diff_mla_attention(self):
        test_diff_mla_attention()

    def test_se_moe_aux_free_balancing(self):
        test_se_moe_aux_free_balancing()

    def test_ssm_memory_highway(self):
        test_ssm_memory_highway()

    def test_ddpo_preference_loss(self):
        test_ddpo_preference_loss()

    def test_permutation_invariance_loss(self):
        test_permutation_invariance_loss()

    def test_ddpo_and_perm_in_multitask_loss(self):
        test_ddpo_and_perm_in_multitask_loss()

    def test_cryptographic_action_cards(self):
        test_cryptographic_action_cards()

    def test_dual_custody_critical_actions(self):
        test_dual_custody_critical_actions()

if __name__ == "__main__":
    print("=" * 70)
    print("  RHEN PHASES 4, 5 & 6: FOUNDATION, PREFERENCE & SAFETY VERIFICATION")
    print("=" * 70)
    test_diff_mla_attention()
    test_se_moe_aux_free_balancing()
    test_ssm_memory_highway()
    test_ddpo_preference_loss()
    test_permutation_invariance_loss()
    test_ddpo_and_perm_in_multitask_loss()
    test_cryptographic_action_cards()
    test_dual_custody_critical_actions()
    print("\n" + "=" * 70)
    print("  ALL PHASES 4, 5 & 6 VERIFICATION SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70)
