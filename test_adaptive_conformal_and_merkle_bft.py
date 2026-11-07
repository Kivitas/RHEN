"""
Verification Suite for Pillar 4 (Online Adaptive Conformal Risk Control)
and Pillar 6 (Cryptographic Merkle Action Ledger & Multi-Agent BFT Consensus).
"""

import time
import math
import hashlib
import json
from config import EOSConfig
from meta_cognitive_monitor import AdaptiveConformalEngine, MetaCognitiveMonitor
from agent_tools import (
    ActionCard,
    ActionStagingEngine,
    MerkleActionLedger,
    BFTConsensusEngine,
)


def test_adaptive_conformal_engine():
    print("\n--- Test 1: AdaptiveConformalEngine (Pillar 4) ---")
    alpha = 0.05
    eta = 0.02
    engine = AdaptiveConformalEngine(alpha=alpha, eta=eta, initial_quantile=0.90)

    assert engine.alpha == alpha
    assert engine.eta == eta
    assert engine.quantile_threshold == 0.90
    assert engine.consecutive_errors == 0
    assert engine.drift_detected is False

    # 1. Prediction set evaluation
    probs = {"optA": 0.85, "optB": 0.12, "optC": 0.03}
    # threshold = 1.0 - 0.90 = 0.10
    pred = engine.predict_set(probs)
    assert "prediction_set" in pred
    assert "optA" in pred["prediction_set"]
    assert "optB" in pred["prediction_set"]
    assert "optC" not in pred["prediction_set"]
    assert pred["set_size"] == 2
    assert pred["is_singleton"] is False
    assert pred["is_ambiguous"] is True
    assert pred["is_empty"] is False
    assert pred["drift_detected"] is False
    print("  [PASS] Adaptive prediction set extracted correctly for dictionary probabilities.")

    # List probabilities
    list_probs = [0.05, 0.92, 0.03]
    pred_list = engine.predict_set(list_probs)
    assert pred_list["prediction_set"] == [1]
    assert pred_list["is_singleton"] is True
    print("  [PASS] Adaptive prediction set extracted correctly for sequence probabilities.")

    # 2. Robbins-Monro Update: Covered case
    # q_{t+1} = clamp(q_t + eta * (error_t - alpha), 0.0, 1.0)
    # err = 0.0 -> q_{t+1} = 0.90 + 0.02 * (0 - 0.05) = 0.90 - 0.001 = 0.899
    rec1 = engine.update(y_true="optA", prediction_set=["optA", "optB"])
    assert rec1["covered"] is True
    assert rec1["error"] == 0.0
    assert math.isclose(rec1["new_quantile"], 0.899, abs_tol=1e-5)
    assert engine.consecutive_errors == 0
    print(f"  [PASS] Robbins-Monro martingale update on covered sample: q={rec1['new_quantile']:.4f}")

    # 3. Robbins-Monro Update: Uncovered case
    # err = 1.0 -> q_{t+1} = 0.899 + 0.02 * (1 - 0.05) = 0.899 + 0.019 = 0.918
    rec2 = engine.update(y_true="optZ", prediction_set=["optA", "optB"])
    assert rec2["covered"] is False
    assert rec2["error"] == 1.0
    assert math.isclose(rec2["new_quantile"], 0.918, abs_tol=1e-5)
    assert engine.consecutive_errors == 1
    print(f"  [PASS] Quantile threshold expanded on error: q={rec2['new_quantile']:.4f}")

    # 4. Drift Detection: 3 consecutive misses
    engine.update(y_true="optZ", prediction_set=["optA"])
    assert engine.consecutive_errors == 2
    assert engine.drift_detected is False

    engine.update(y_true="optZ", prediction_set=["optA"])
    assert engine.consecutive_errors == 3
    assert engine.drift_detected is True
    print("  [PASS] Non-stationary covariate drift successfully detected on 3 consecutive errors.")


def test_meta_cognitive_monitor_adaptive_integration():
    print("\n--- Test 2: MetaCognitiveMonitor Adaptive Conformal Integration ---")
    config = EOSConfig()
    monitor = MetaCognitiveMonitor(config)

    assert hasattr(monitor, "adaptive_conformal")
    assert isinstance(monitor.adaptive_conformal, AdaptiveConformalEngine)
    assert monitor.adaptive_conformal.alpha == 0.02

    # Update through monitor helper
    probs = {"A": 0.90, "B": 0.08, "C": 0.02}
    record = monitor.update_adaptive_conformal(y_true="A", probabilities=probs)
    assert record["covered"] is True
    assert record["error"] == 0.0

    # Decision monitoring attaches adaptive_conformal_set
    decision_result = {
        "mode": "test_decision",
        "confidence": 0.99,
        "probabilities": [0.99, 0.01],
    }
    monitored = monitor.monitor_decision_sync(decision_result, context=None)
    assert "adaptive_conformal_set" in monitored
    assert "conformal_set" in monitored
    assert monitored["adaptive_conformal_set"]["is_singleton"] is True
    print("  [PASS] MetaCognitiveMonitor integrated with AdaptiveConformalEngine.")


def test_merkle_action_ledger():
    print("\n--- Test 3: MerkleActionLedger (Pillar 6) ---")
    ledger = MerkleActionLedger()
    assert len(ledger.get_merkle_root()) == 64

    cards = []
    for i in range(7):
        card = ActionCard(
            id=f"act-00{i}",
            title=f"Action Step {i}",
            risk_tier="high" if i % 2 == 0 else "medium",
            payload={"step": i, "server": f"worker-{i}.prod"},
            signature=f"hmac-sig-{i}" * 4,
            timestamp=1700000000.0 + i * 10,
        )
        cards.append(card)
        record = ledger.append_action(card)
        # Domain-separated leaf: sha256(0x00 || JSON[event, id, tier, status, signature, timestamp]).
        # append_action records the card's status, here "staged", as the "stage" event.
        leaf_data = json.dumps(
            ["stage", card.id, card.risk_tier, card.status, card.signature, repr(float(card.timestamp))],
            separators=(",", ":"),
        ).encode("ascii")
        expected_leaf = hashlib.sha256(b"\x00" + leaf_data).hexdigest()
        assert record["leaf_hash"] == expected_leaf
        assert record["leaf_index"] == i

    root = ledger.get_merkle_root()
    assert len(root) == 64
    assert len(ledger.leaves) == 7
    print(f"  [PASS] 7 Action Cards staged into MerkleActionLedger. Root: {root[:16]}...")

    # Inclusion proof verification and timing benchmark (< 0.1ms)
    for idx in range(len(cards)):
        proof_bundle = ledger.generate_inclusion_proof(idx)
        assert proof_bundle["leaf_index"] == idx
        assert proof_bundle["leaf_hash"] == ledger.leaves[idx]
        assert proof_bundle["merkle_root"] == root

        t0 = time.perf_counter()
        valid = ledger.verify_inclusion_proof(proof_bundle["leaf_hash"], proof_bundle, root)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert valid is True
        assert t_elapsed_ms < 0.1, f"Proof verification took {t_elapsed_ms:.4f}ms, expected < 0.1ms"

    print("  [PASS] Cryptographic inclusion proofs verified in < 0.1ms per leaf.")

    # Tampering tests
    tampered_leaf = "f" * 64
    assert ledger.verify_inclusion_proof(tampered_leaf, ledger.generate_inclusion_proof(0), root) is False
    assert ledger.verify_inclusion_proof(ledger.leaves[0], ledger.generate_inclusion_proof(0), "0" * 64) is False
    print("  [PASS] Merkle root tampering and leaf tampering strictly rejected.")


def test_bft_consensus_engine():
    print("\n--- Test 4: BFTConsensusEngine (Pillar 6) ---")
    registered_agents = {
        "Agent-1": b"secret-agent-key-1",
        "Agent-2": b"secret-agent-key-2",
        "Agent-3": b"secret-agent-key-3",
        "Agent-4": b"secret-agent-key-4",
    }
    bft = BFTConsensusEngine(registered_agents=registered_agents, required_quorum_ratio=0.67)

    card = ActionCard(
        id="act-critical-99",
        title="Emergency Gateway Drain",
        risk_tier="critical",
        payload={"gateway": "gw-01", "force": True},
        signature="initial-staging-signature-hash",
        timestamp=1700000500.0,
    )

    # Collect ballots
    b1 = bft.collect_ballot(card, "Agent-1", vote=True, secret_key=registered_agents["Agent-1"])
    b2 = bft.collect_ballot(card, "Agent-2", vote=True, secret_key=registered_agents["Agent-2"])
    b3 = bft.collect_ballot(card, "Agent-3", vote=True, secret_key=registered_agents["Agent-3"])
    b4_no = bft.collect_ballot(card, "Agent-4", vote=False, secret_key=registered_agents["Agent-4"])

    # Threshold for 4 agents at 0.67 is ceil(0.67 * 4) = 3
    # 2 affirmative votes fail quorum
    q_two = bft.evaluate_quorum(card, [b1, b2, b4_no])
    assert q_two["approved"] is False
    assert q_two["valid_count"] == 2
    assert q_two["quorum_threshold"] == 3
    assert q_two["certified_digest"] is None
    print("  [PASS] Quorum denied with 2/3 required affirmative votes.")

    # 3 affirmative votes pass quorum
    q_three = bft.evaluate_quorum(card, [b1, b2, b3, b4_no])
    assert q_three["approved"] is True
    assert q_three["valid_count"] == 3
    assert q_three["quorum_threshold"] == 3
    assert q_three["certified_digest"] is not None
    assert len(q_three["certified_digest"]) == 64
    print(f"  [PASS] BFT Quorum approved with 3/3 valid votes: cert={q_three['certified_digest'][:16]}...")

    # Byzantine attack: forged signature
    forged_ballot = dict(b1)
    forged_ballot["signature"] = "deadbeef" * 8
    q_forged = bft.evaluate_quorum(card, [forged_ballot, b2])
    assert q_forged["valid_count"] == 1
    assert q_forged["approved"] is False
    print("  [PASS] Forged HMAC signature cleanly detected and discarded.")

    # Byzantine attack: duplicate vote (Sybil replay)
    q_sybil = bft.evaluate_quorum(card, [b1, b1, b1])
    assert q_sybil["valid_count"] == 1
    assert q_sybil["approved"] is False
    print("  [PASS] Sybil duplicate voting prevented.")


def test_action_staging_engine_full_integration():
    print("\n--- Test 5: ActionStagingEngine Merkle & BFT Full Integration ---")
    engine = ActionStagingEngine(
        secret_key=b"adaptive-merkle-suite-signing-key",
        registered_bft_agents={
            "Agent-Alpha": b"alpha-bft-secret-key-2026",
            "Agent-Beta": b"beta-bft-secret-key-2026",
            "Agent-Gamma": b"gamma-bft-secret-key-2026",
        },
    )
    engine.register_cosigner_agent("SecurityAuditor", b"auditor-secret-key")

    assert isinstance(engine.merkle_ledger, MerkleActionLedger)
    assert isinstance(engine.bft_engine, BFTConsensusEngine)

    # 1. Staging auto-logs to Merkle Ledger
    card = engine.stage_action(
        decision_result={"confidence": 0.98},
        action_type="execute",
        payload={"action": "RESTART_CLUSTER"},
        risk_tier="critical",
        metadata={"conformal_set": {"is_singleton": True, "set_size": 1}},
    )

    assert "merkle_leaf_hash" in card.metadata
    assert "merkle_leaf_index" in card.metadata
    assert "merkle_root" in card.metadata
    assert engine.merkle_ledger.get_merkle_root() == card.metadata["merkle_root"]
    print("  [PASS] Action staging automatically logged to Merkle ledger.")

    # 2. Critical preflight: Fails without co-signer
    res1 = engine.preflight_verification(card.id)
    assert res1["allowed"] is False
    assert any("Dual-Custody" in c["name"] and not c["passed"] for c in res1["checks"])

    # 3. Add the pre-registered secondary co-signer
    assert engine.request_dual_custody_cosign(card.id, "SecurityAuditor", b"auditor-secret-key") is True

    # 4. Co-signed, but with no BFT ballots the quorum is mandatory and fails
    res2 = engine.preflight_verification(card.id)
    assert res2["allowed"] is False
    assert any("BFT" in c["name"] and not c["passed"] for c in res2["checks"])
    print("  [PASS] Missing BFT ballots fail the mandatory quorum even when co-signed.")

    # 5. When BFT ballots are collected into metadata, BFT quorum is enforced
    crit_bft_card = engine.stage_action(
        decision_result={"confidence": 0.99},
        action_type="execute",
        payload={"action": "PURGE_ENCRYPTION_KEYS"},
        risk_tier="critical",
        metadata={"conformal_set": {"is_singleton": True, "set_size": 1}},
    )
    assert engine.request_dual_custody_cosign(crit_bft_card.id, "SecurityAuditor", b"auditor-secret-key") is True

    # Collect 1 vote from Agent-Alpha (out of 3: Alpha, Beta, Gamma; threshold is 3 of 3 with ratio 0.67)
    engine.collect_bft_ballot(crit_bft_card.id, "Agent-Alpha", True, b"alpha-bft-secret-key-2026")
    res_bft1 = engine.preflight_verification(crit_bft_card.id)
    assert res_bft1["allowed"] is False
    assert any("BFT" in c["name"] and not c["passed"] for c in res_bft1["checks"])

    # Collect remaining votes to reach quorum
    engine.collect_bft_ballot(crit_bft_card.id, "Agent-Beta", True, b"beta-bft-secret-key-2026")
    engine.collect_bft_ballot(crit_bft_card.id, "Agent-Gamma", True, b"gamma-bft-secret-key-2026")

    res_bft_full = engine.preflight_verification(crit_bft_card.id)
    assert res_bft_full["allowed"] is True
    assert res_bft_full["bft_consensus"]["approved"] is True
    assert res_bft_full["merkle_root"] == engine.merkle_ledger.get_merkle_root()
    print("  [PASS] Full Critical Action Verification (Dual-Custody + BFT Quorum + Merkle Ledger) passed.")


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestAdaptiveConformalAndMerkleBft(_unittest.TestCase):
    """test_adaptive_conformal_and_merkle_bft verification suite (master battery)."""

    def test_adaptive_conformal_engine(self):
        test_adaptive_conformal_engine()

    def test_meta_cognitive_monitor_adaptive_integration(self):
        test_meta_cognitive_monitor_adaptive_integration()

    def test_merkle_action_ledger(self):
        test_merkle_action_ledger()

    def test_bft_consensus_engine(self):
        test_bft_consensus_engine()

    def test_action_staging_engine_full_integration(self):
        test_action_staging_engine_full_integration()

if __name__ == "__main__":
    print("=" * 70)
    print("  PILLARS 4 & 6: ADAPTIVE CONFORMAL & MERKLE BFT VERIFICATION")
    print("=" * 70)
    test_adaptive_conformal_engine()
    test_meta_cognitive_monitor_adaptive_integration()
    test_merkle_action_ledger()
    test_bft_consensus_engine()
    test_action_staging_engine_full_integration()
    print("\n" + "=" * 70)
    print("  ALL PILLAR 4 & PILLAR 6 SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70)
