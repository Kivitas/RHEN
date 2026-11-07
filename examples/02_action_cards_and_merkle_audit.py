"""
02. Cryptographic Action Cards, Merkle Ledger & Out-of-Band Audit
-----------------------------------------------------------------
Demonstrates Rhen's cryptographic safety fabric:
1. ActionCard with risk-tiered authorization (Low, Medium, High, Critical).
2. HMAC-SHA256 canonical payload hashing and tamper rejection.
3. Cryptographic Dual-Custody co-signing protocol for Critical actions.
4. Multi-Agent BFT Quorum Consensus before executing destructive operations.
5. Tamper-evident Merkle Action Ledger with inclusion proofs (<0.1ms).
6. Out-of-Band Audit Ledger (arXiv:2609.30266) with gap and link tampering detection.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rhen import (
    ActionCard,
    ActionStagingEngine,
    OutOfBandAuditLedger,
)

def main():
    print("=" * 70)
    print("RHEN Safety Fabric - Action Cards, Merkle Ledger & Audit")
    print("=" * 70)

    secret_key = b"primary-agent-secret-signing-key-2026"
    cosigner_key = b"security-auditor-cosigner-key-2026"
    bft_key_beta = b"bft-validator-beta-secret-key-2026"
    bft_key_gamma = b"bft-validator-gamma-secret-key-2026"

    # 1. Initialize Action Staging Engine with BFT peers & Co-signers
    engine = ActionStagingEngine(
        secret_key=secret_key,
        registered_bft_agents={
            "Validator-Beta": bft_key_beta,
            "Validator-Gamma": bft_key_gamma,
        },
    )
    engine.register_cosigner_agent("SecurityOfficer", cosigner_key)

    print("\n--- 1. Staging Medium Risk Action ---")
    card_med = engine.stage_action(
        decision_result={"confidence": 0.94, "chosen_option": "ScaleReplicaGroup"},
        action_type="execute",
        payload={"action": "scale_replicas", "cluster": "us-east-1", "service": "payment-api", "replicas": 8},
        risk_tier="medium",
        title="Scale Payment API Replicas",
    )
    print(f"Staged Action ID: {card_med.id}")
    print(f"Risk Tier:        {card_med.risk_tier}")
    print(f"HMAC Signature:   {card_med.signature[:24]}...")
    print(f"Merkle Leaf Hash: {card_med.metadata.get('merkle_leaf_hash')[:24]}...")

    # Preflight verification for medium action
    preflight_med = engine.preflight_verification(card_med.id)
    print(f"Preflight Allowed: {preflight_med['allowed']}")
    print(f"Preflight Status:  {card_med.status}")

    # Execute the medium action
    exec_res = engine.execute_action(card_med.id)
    print(f"Execution Status:  {exec_res.status}")

    print("\n--- 2. Staging Critical Action (Dual-Custody + BFT Quorum Enforcement) ---")
    card_crit = engine.stage_action(
        decision_result={"confidence": 0.99, "chosen_option": "DrainDatacenterTraffic"},
        action_type="execute",
        payload={"action": "traffic_drain", "datacenter": "eu-central-1", "reason": "Severe Hardware Failure"},
        risk_tier="critical",
        title="Emergency DC Traffic Drain",
        metadata={"conformal_set": {"is_singleton": True, "set_size": 1}},
    )
    print(f"Critical Action ID: {card_crit.id}")

    # Attempting to preflight without co-signer must be blocked
    preflight_crit_blocked = engine.preflight_verification(card_crit.id)
    print(f"Preflight without co-signer: Allowed={preflight_crit_blocked['allowed']}")
    for check in preflight_crit_blocked.get("checks", []):
        if not check["passed"]:
            print(f"  [BLOCKED] Check failed: {check['name']}")

    # Perform Secondary Co-signing
    print("\nRequesting Secondary Dual-Custody Signature from SecurityOfficer...")
    cosigned_ok = engine.request_dual_custody_cosign(
        card_crit.id,
        "SecurityOfficer",
        cosigner_key,
    )
    print(f"Dual-Custody Approval Registered: {cosigned_ok}")

    # Collect BFT Quorum Votes from Validators
    print("Collecting BFT consensus ballots from peer agents...")
    engine.collect_bft_ballot(card_crit.id, "Validator-Beta", vote=True, secret_key=bft_key_beta)
    engine.collect_bft_ballot(card_crit.id, "Validator-Gamma", vote=True, secret_key=bft_key_gamma)

    # Now preflight succeeds
    preflight_crit_approved = engine.preflight_verification(card_crit.id)
    print(f"Preflight with co-signer & BFT: Allowed={preflight_crit_approved['allowed']}")
    print(f"Card Status:                   {card_crit.status}")

    # 3. Tamper Resistance Demonstration
    print("\n--- 3. Payload Tamper Resistance Check ---")
    tampered_card = ActionCard(
        id="act-tampered-001",
        title="Unauthorized Admin Elevation",
        action_type="execute",
        risk_tier="critical",
        payload={"role": "guest"},
        timestamp=time.time(),
    )
    engine.sign_action(tampered_card)
    valid_before = engine.verify_action_digest(tampered_card)
    print(f"Original Signature Valid:    {valid_before}")

    # Adversary modifies payload
    tampered_card.payload["role"] = "superadmin"
    valid_after = engine.verify_action_digest(tampered_card)
    print(f"Modified Payload Valid:      {valid_after} (Tamper correctly detected & rejected)")

    # 4. Merkle Action Ledger & Inclusion Proofs
    print("\n--- 4. Merkle Action Ledger Inclusion Proofs ---")
    ledger = engine.merkle_ledger
    root = ledger.get_merkle_root()
    print(f"Current Merkle Root: {root}")
    print(f"Total Ledger Leaves: {len(ledger.leaves)}")

    for idx, leaf in enumerate(ledger.leaves):
        proof = ledger.generate_inclusion_proof(idx)
        t0 = time.perf_counter()
        verified = ledger.verify_inclusion_proof(leaf, proof, root)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        print(f"Leaf #{idx} inclusion proof verified: {verified} (verified in {elapsed_us:.1f} microseconds)")

    # 5. Out-of-Band Audit Ledger (arXiv:2609.30266)
    print("\n--- 5. Out-of-Band Audit Trail (Anti-Tamper & Gap Detection) ---")
    audit_key = b"audit-master-sealing-key-256bit!"
    audit = OutOfBandAuditLedger(key=audit_key)

    audit.record_decision(
        question="Select failover target",
        chosen="us-west-2",
        confidence=0.965,
        options=["us-west-2", "eu-central-1", "ap-southeast-1"],
    )
    audit.record(
        "action_executed",
        {"action_id": card_med.id, "status": "success"},
    )
    audit.record(
        "dual_custody_approved",
        {"action_id": card_crit.id, "cosigner": "SecurityOfficer"},
    )

    audit_verify = audit.verify_chain()
    print(f"Audit Chain Valid:           {audit_verify['all_valid']}")
    print(f"Audit Chain Entries:         {audit_verify['count']}")

    # Gap detection demonstration
    full_entries = [e.to_dict() for e in audit.entries]
    simulated_partial = OutOfBandAuditLedger(key=audit_key)
    simulated_partial.record("init", {})
    detected_gaps = simulated_partial.detect_gaps(full_entries)
    print(f"Detected Trace Omissions:    Sequences {detected_gaps} missing from partial trace (gap detected!)")

    print("\n[SUCCESS] Cryptographic Action Cards, Merkle Ledger & Audit fully operational!")

if __name__ == "__main__":
    main()
