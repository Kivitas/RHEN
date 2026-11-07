"""
Regression tests for the action-gate bypasses and the audit-ledger honesty fixes.

Each test reproduces one bypass and asserts that it is now refused. The action gate
(``agent_tools``) covers risk tiers, approval, execution, dual custody, signing keys,
canonical digests, BFT ballots and the Merkle ledger. The audit ledger (``rhen_verify``
and ``rhen_pillars``) covers mandatory keys, remote sealing and receiver-side verification.
"""

import hashlib
import json
import os
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

from agent_tools import (
    ACTION_TYPES,
    RISK_TIERS,
    ActionCard,
    ActionStagingEngine,
    BFTConsensusEngine,
    MerkleActionLedger,
    canonical_payload_json,
)
from rhen_pillars import PillarStack, RhenPillarConfig
from rhen_verify import (
    AuditSinkError,
    OutOfBandAuditLedger,
    SocketAuditSink,
    serve_audit_socket,
    verify_sealed_records,
)

SIGNING_KEY = b"regression-signing-key-2026"
ATTACKER_KEY = b"attacker-signing-key-2026"
LEGACY_DEFAULT_KEY = b"eos-rhen-system-master-key-2026"
COSIGNER = "SecurityAuditor"
COSIGNER_KEY = b"security-auditor-secret-key-2026"
BFT_KEYS = {
    "Agent-Alpha": b"alpha-bft-secret-key-2026",
    "Agent-Beta": b"beta-bft-secret-key-2026",
    "Agent-Gamma": b"gamma-bft-secret-key-2026",
}
AUDIT_KEY = b"32-bytes-of-real-entropy-here!!"


def make_engine(**overrides):
    options = {"secret_key": SIGNING_KEY, "registered_bft_agents": dict(BFT_KEYS)}
    options.update(overrides)
    return ActionStagingEngine(**options)


def stage_medium(engine, confidence=0.90, **kwargs):
    return engine.stage_action(
        {"confidence": confidence},
        action_type="execute",
        payload={"op": "restart-service"},
        risk_tier="medium",
        **kwargs,
    )


def stage_critical(engine, confidence=0.97, singleton=True):
    return engine.stage_action(
        {"confidence": confidence},
        action_type="execute",
        payload={"command": "DROP_DATABASE_CLUSTER"},
        risk_tier="critical",
        metadata={"conformal_set": {"is_singleton": singleton, "set_size": 1 if singleton else 2}},
    )


def collect_all_ballots(engine, card_id, vote=True):
    for voter, key in BFT_KEYS.items():
        engine.collect_bft_ballot(card_id, voter, vote, key)


def failing_check_names(result):
    return [check["name"] for check in result["checks"] if not check["passed"]]


def _frame(record):
    body = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _start_receiver(address, store_path, max_iterations, key=AUDIT_KEY):
    stop = threading.Event()
    thread = threading.Thread(
        target=serve_audit_socket,
        kwargs={
            "address": address,
            "store_path": store_path,
            "stop_event": stop,
            "max_iterations": max_iterations,
            "key": key,
        },
        daemon=True,
    )
    thread.start()
    host, _, port = address.rpartition(":")
    deadline = time.time() + 5
    while True:  # wait until the receiver accepts connections
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.05)
    return thread, stop


def _send(address, payload):
    host, _, port = address.rpartition(":")
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(payload)


class _CaptureSocketSink(SocketAuditSink):
    """A remote-sealing sink that keeps the frames in memory instead of sending them."""

    def __init__(self):
        super().__init__("127.0.0.1:9", reconnect_attempts=1)
        self.frames = []

    def emit(self, record):
        self.frames.append(json.loads(json.dumps(record, sort_keys=True)))


class TestStagingValidation(unittest.TestCase):
    """C1: tiers and action types are validated, and permission requirements only grow."""

    def test_unknown_tier_is_rejected_at_stage_time(self):
        # Exploit: 'Critical' fell through every tier check and passed at confidence 0.70.
        engine = make_engine()
        with self.assertRaises(ValueError):
            engine.stage_action({"confidence": 0.70}, risk_tier="Critical")
        self.assertEqual(engine.get_staged_cards(), [])

    def test_action_type_is_validated_case_sensitively(self):
        engine = make_engine()
        for bad in ("Execute", "drop_database", ""):
            with self.assertRaises(ValueError):
                engine.stage_action({"confidence": 0.9}, action_type=bad)
        self.assertEqual(set(ACTION_TYPES), {"execute", "query", "notify", "escalate", "stage"})
        self.assertEqual(set(RISK_TIERS), {"low", "medium", "high", "critical"})

    def test_caller_built_card_with_unknown_tier_is_refused(self):
        engine = make_engine()
        card = ActionCard(id="act-caller-1", title="t", risk_tier="Critical", confidence=0.70, payload={})
        with self.assertRaises(ValueError):
            engine.stage_action(card)
        ok, _ = engine.verify_action(card)
        self.assertFalse(ok, "an unrecognised tier must fail closed in preflight")

    def test_caller_permissions_are_added_never_removed(self):
        engine = make_engine()
        card = engine.stage_action(
            {"confidence": 0.97},
            action_type="execute",
            payload={"command": "X"},
            risk_tier="critical",
            required_permissions=["read"],
            metadata={"conformal_set": {"is_singleton": True}},
        )
        self.assertTrue({"execute_privileged", "write", "read"} <= set(card.required_permissions))
        ok, _ = engine.verify_action(card, granted_permissions=["read", "write"])
        self.assertFalse(ok, "dropping execute_privileged from the grant must fail the check")

    def test_empty_grant_list_grants_nothing(self):
        # Exploit: an empty list fell back to the full default grant set.
        engine = make_engine(default_granted_permissions=[])
        card = stage_medium(engine)
        result = engine.preflight_verification(card.id)
        self.assertFalse(result["allowed"])
        self.assertIn("Permission Verification", failing_check_names(result))

    def test_per_call_grants_can_only_narrow_the_engine_grant(self):
        engine = make_engine(default_granted_permissions=["read"])
        card = stage_medium(engine)
        ok, _ = engine.verify_action(card, granted_permissions=["read", "execute", "write"])
        self.assertFalse(ok, "a per-call grant must not widen the engine's own grant")

    def test_pre_signed_card_from_another_key_is_refused(self):
        foreign = stage_medium(make_engine(secret_key=ATTACKER_KEY))
        with self.assertRaises(ValueError):
            make_engine().stage_action(foreign)

    def test_unsigned_caller_card_is_normalised_before_signing(self):
        engine = make_engine()
        card = ActionCard(
            id="act-normalise-1", title="t", risk_tier="critical", confidence=0.9649999,
            payload={"x": 1}, required_permissions=["read"],
            metadata={"conformal_set": {"is_singleton": True}},
        )
        staged = engine.stage_action(card)
        self.assertEqual(staged.confidence, 0.964999, "confidence is truncated to the 6-decimal grid")
        self.assertTrue({"execute_privileged", "write", "read"} <= set(staged.required_permissions))
        self.assertTrue(engine.verify_action_digest(staged))

    def test_non_finite_or_out_of_range_confidence_is_refused(self):
        engine = make_engine()
        for bad in (float("nan"), float("inf"), 1.5, -0.1):
            with self.assertRaises(ValueError):
                engine.stage_action({"confidence": bad}, risk_tier="low")
        with self.assertRaises(TypeError):
            engine.stage_action({"confidence": "0.9"}, risk_tier="low")

    def test_critical_gates_are_mandatory_not_skipped(self):
        engine = make_engine()
        engine.register_cosigner_agent(COSIGNER, COSIGNER_KEY)

        # No conformal certificate at all: the singleton gate must fail, not be skipped.
        no_conformal = engine.stage_action(
            {"confidence": 0.97}, action_type="execute", payload={"command": "X"}, risk_tier="critical"
        )
        self.assertTrue(engine.request_dual_custody_cosign(no_conformal.id, COSIGNER, COSIGNER_KEY))
        collect_all_ballots(engine, no_conformal.id)
        result = engine.preflight_verification(no_conformal.id)
        self.assertFalse(result["allowed"])
        self.assertIn("Conformal Singleton Gate", failing_check_names(result))

        # No BFT ballots at all: the quorum check must fail, not be skipped.
        no_quorum = stage_critical(engine)
        self.assertTrue(engine.request_dual_custody_cosign(no_quorum.id, COSIGNER, COSIGNER_KEY))
        result = engine.preflight_verification(no_quorum.id)
        self.assertFalse(result["allowed"])
        self.assertTrue(any("BFT" in name for name in failing_check_names(result)))

    def test_critical_card_passes_only_with_every_gate_satisfied(self):
        engine = make_engine()
        engine.register_cosigner_agent(COSIGNER, COSIGNER_KEY)
        card = stage_critical(engine)
        self.assertTrue(engine.request_dual_custody_cosign(card.id, COSIGNER, COSIGNER_KEY))
        collect_all_ballots(engine, card.id)
        result = engine.preflight_verification(card.id)
        self.assertTrue(result["allowed"], failing_check_names(result))


class TestApprovalAndExecution(unittest.TestCase):
    """C2: approval needs a passing preflight, execution re-verifies, and ids are never reused."""

    def test_approve_without_preflight_raises(self):
        # Exploit: approve_action accepted any staged card.
        engine = make_engine()
        card = stage_medium(engine)
        with self.assertRaises(PermissionError):
            engine.approve_action(card.id, approver="Supervisor")
        self.assertEqual(card.status, "staged")

    def test_approve_after_failed_preflight_raises(self):
        engine = make_engine()
        card = stage_medium(engine, confidence=0.60)  # below the 0.65 medium floor
        self.assertFalse(engine.preflight_verification(card.id)["allowed"])
        with self.assertRaises(PermissionError):
            engine.approve_action(card.id, approver="Supervisor")

    def test_approve_after_passing_preflight_succeeds_and_is_recorded(self):
        engine = make_engine()
        card = stage_medium(engine)
        self.assertTrue(engine.preflight_verification(card.id)["allowed"])
        approved = engine.approve_action(card.id, approver="Supervisor")
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.metadata["approver"], "Supervisor")
        self.assertIn("approve", engine.merkle_ledger.events)

    def test_approve_unknown_id_returns_none(self):
        self.assertIsNone(make_engine().approve_action("act-does-not-exist"))

    def test_failing_preflight_revokes_an_earlier_approval(self):
        engine = make_engine()
        card = stage_medium(engine)
        self.assertTrue(engine.preflight_verification(card.id)["allowed"])
        engine.approve_action(card.id)
        card.payload["op"] = "drop-database"  # the approved card changed
        self.assertFalse(engine.preflight_verification(card.id)["allowed"])
        self.assertEqual(card.status, "staged")
        with self.assertRaises(PermissionError):
            engine.approve_action(card.id)

    def test_approval_is_bound_to_the_revision_that_passed(self):
        engine = make_engine()
        card = stage_medium(engine)
        self.assertTrue(engine.preflight_verification(card.id)["allowed"])
        card.payload["op"] = "drop-database"  # changed after the preflight passed
        with self.assertRaises(PermissionError):
            engine.approve_action(card.id)

    def test_execute_reverifies_after_confidence_is_lowered(self):
        # Exploit: a card approved at 0.90 could be executed after its confidence was lowered.
        engine = make_engine()
        card = stage_medium(engine, confidence=0.90)
        engine.preflight_verification(card.id)
        engine.approve_action(card.id)
        card.confidence = 0.10
        engine.sign_action(card)  # even a re-signed card must fail on the confidence floor
        calls = []
        result = engine.execute_action(card.id, executor_fn=lambda payload: calls.append(payload) or {"ok": True})
        self.assertEqual(result.status, "failed")
        self.assertEqual(calls, [], "the executor must not run for a card that no longer verifies")

    def test_execute_twice_raises_runtime_error(self):
        # Exploit: a second execute of the same id re-ran the side effect.
        engine = make_engine()
        card = stage_medium(engine)
        self.assertEqual(engine.execute_action(card.id).status, "executed")
        with self.assertRaises(RuntimeError):
            engine.execute_action(card.id)

    def test_restaging_an_executed_card_is_refused(self):
        engine = make_engine()
        card = stage_medium(engine)
        engine.execute_action(card.id)
        with self.assertRaises(ValueError):
            engine.stage_action(card)
        replay = ActionCard(
            id=card.id, title="replay", risk_tier="medium", action_type="execute",
            payload={"op": "restart-service"}, confidence=0.9,
        )
        with self.assertRaises(ValueError):
            engine.stage_action(replay)

    def test_restaging_a_rejected_card_is_refused(self):
        engine = make_engine()
        card = stage_medium(engine)
        engine.reject_action(card.id, reason="declined")
        self.assertEqual(engine.merkle_ledger.events[-1], "reject")
        with self.assertRaises(ValueError):
            engine.stage_action(card)

    def test_bypass_is_allowed_only_as_a_recorded_choice(self):
        engine = make_engine()
        card = stage_medium(engine, confidence=0.10)  # would fail verification
        result = engine.execute_action(card.id, bypass_verification=True)
        self.assertEqual(result.status, "executed")
        self.assertTrue(result.metadata["verification_bypassed"])

    def test_execute_unknown_id_raises_key_error(self):
        with self.assertRaises(KeyError):
            make_engine().execute_action("act-does-not-exist")

    def test_failed_execution_is_recorded_in_the_ledger(self):
        engine = make_engine()
        card = stage_medium(engine, confidence=0.10)
        engine.execute_action(card.id)  # verification fails and the card is retired
        self.assertEqual(engine.merkle_ledger.events[-1], "execute")
        self.assertEqual(card.status, "failed")


class TestDualCustody(unittest.TestCase):
    """C3: co-signers must be registered in advance, keyed correctly and separate from the persona."""

    def test_unregistered_cosigner_is_refused(self):
        engine = make_engine()
        card = stage_critical(engine)
        self.assertFalse(engine.request_dual_custody_cosign(card.id, "Intruder", ATTACKER_KEY))
        self.assertIsNone(card.cosigner_agent_id)
        result = engine.preflight_verification(card.id)
        self.assertTrue(any("Dual-Custody" in name for name in failing_check_names(result)))

    def test_mismatched_key_is_refused(self):
        engine = make_engine()
        engine.register_cosigner_agent(COSIGNER, COSIGNER_KEY)
        card = stage_critical(engine)
        self.assertFalse(engine.request_dual_custody_cosign(card.id, COSIGNER, ATTACKER_KEY))
        self.assertIsNone(card.cosigner_signature)

    def test_no_trust_on_first_use(self):
        # Exploit: the first key presented for an unknown id was stored and trusted.
        engine = make_engine()
        card = stage_critical(engine)
        engine.request_dual_custody_cosign(card.id, "Intruder", ATTACKER_KEY)
        self.assertNotIn("Intruder", engine.trusted_cosigners)

    def test_cosigner_must_differ_from_persona(self):
        engine = make_engine()
        engine.register_cosigner_agent("SecurityAgent", COSIGNER_KEY)
        card = engine.stage_action(
            {"confidence": 0.97}, action_type="execute", payload={"command": "X"},
            persona="SecurityAgent", risk_tier="critical",
            metadata={"conformal_set": {"is_singleton": True}},
        )
        self.assertFalse(engine.request_dual_custody_cosign(card.id, "SecurityAgent", COSIGNER_KEY))

    def test_registered_cosigner_with_correct_key_is_accepted(self):
        engine = make_engine()
        engine.register_cosigner_agent(COSIGNER, COSIGNER_KEY)
        card = stage_critical(engine)
        self.assertTrue(engine.request_dual_custody_cosign(card.id, COSIGNER, COSIGNER_KEY))
        self.assertTrue(engine.verify_cosigner_digest(card))

    def test_forged_cosignature_fields_fail_verification(self):
        engine = make_engine()
        engine.register_cosigner_agent(COSIGNER, COSIGNER_KEY)
        card = stage_critical(engine)
        card.cosigner_agent_id = COSIGNER
        card.cosigner_signature = engine.generate_cosign_digest(card, COSIGNER, ATTACKER_KEY)
        self.assertFalse(engine.verify_cosigner_digest(card))
        collect_all_ballots(engine, card.id)
        self.assertFalse(engine.preflight_verification(card.id)["allowed"])

    def test_cosignature_cannot_be_transplanted_to_another_card(self):
        engine = make_engine()
        engine.register_cosigner_agent(COSIGNER, COSIGNER_KEY)
        signed = stage_critical(engine)
        engine.request_dual_custody_cosign(signed.id, COSIGNER, COSIGNER_KEY)
        other = stage_critical(engine)
        other.cosigner_agent_id = signed.cosigner_agent_id
        other.cosigner_signature = signed.cosigner_signature
        self.assertFalse(engine.verify_cosigner_digest(other))


class TestKeysDigestsAndCanonicalForm(unittest.TestCase):
    """C4: no built-in key, digests cover every signed field, canonical JSON is strict."""

    def test_engine_without_key_or_environment_raises(self):
        # Exploit: a built-in default key let anyone who read the source forge cards.
        with mock.patch.dict(os.environ):
            os.environ.pop("EOS_ACTION_SIGNING_KEY", None)
            with self.assertRaises(ValueError):
                ActionStagingEngine()
            os.environ["EOS_ACTION_SIGNING_KEY"] = "too-short"
            with self.assertRaises(ValueError):
                ActionStagingEngine()
            os.environ["EOS_ACTION_SIGNING_KEY"] = "environment-signing-key-2026"
            engine = ActionStagingEngine()
            self.assertTrue(engine.verify_action_digest(stage_medium(engine)))

    def test_ephemeral_key_is_random_per_instance(self):
        first = ActionStagingEngine(ephemeral_key=True)
        second = ActionStagingEngine(ephemeral_key=True)
        self.assertNotEqual(first.secret_key, second.secret_key)
        self.assertFalse(second.verify_action_digest(stage_medium(first)))

    def test_card_forged_with_another_key_fails_verification(self):
        honest = make_engine()
        attacker = make_engine(secret_key=ATTACKER_KEY)
        forged = stage_medium(attacker, confidence=0.99)
        ok, _ = honest.verify_action(forged)
        self.assertFalse(ok)
        checks = {c["name"]: c["passed"] for c in forged.verification_checks}
        self.assertFalse(checks["Cryptographic Integrity"])

    def test_card_signed_with_the_old_hard_coded_key_fails(self):
        honest = make_engine()
        legacy = make_engine(secret_key=LEGACY_DEFAULT_KEY)
        forged = stage_medium(legacy, confidence=0.99)
        self.assertFalse(honest.verify_action_digest(forged))

    def test_changing_required_permissions_breaks_the_digest(self):
        # Exploit: required_permissions was not covered, so a requirement could be rewritten.
        engine = make_engine()
        card = stage_medium(engine)
        self.assertTrue(engine.verify_action_digest(card))
        card.required_permissions = ["read"]
        self.assertFalse(engine.verify_action_digest(card))
        card.required_permissions = ["execute", "write", "admin"]
        self.assertFalse(engine.verify_action_digest(card))

    def test_changing_action_type_title_or_persona_breaks_the_digest(self):
        engine = make_engine()
        for field, value in (("action_type", "notify"), ("title", "other"), ("persona", "Mallory")):
            card = stage_medium(engine)
            setattr(card, field, value)
            self.assertFalse(engine.verify_action_digest(card), field)

    def test_metadata_is_outside_the_digest_by_design(self):
        engine = make_engine()
        card = stage_medium(engine)
        card.metadata["note"] = "annotation"
        self.assertTrue(engine.verify_action_digest(card))

    def test_canonical_payload_rejects_non_json_data(self):
        # Exploit: sets were coerced with default=str, so two payloads could share a digest.
        for bad in ({"tags": {"a", "b"}}, {"raw": b"bytes"}, {1: "int-key"},
                    {"x": float("nan")}, {"obj": object()}):
            with self.assertRaises(TypeError):
                canonical_payload_json(bad)

    def test_canonical_payload_is_order_independent(self):
        self.assertEqual(
            canonical_payload_json({"b": 1, "a": {"y": 2, "x": 3}}),
            canonical_payload_json({"a": {"x": 3, "y": 2}, "b": 1}),
        )

    def test_staging_refuses_a_payload_with_a_set(self):
        engine = make_engine()
        with self.assertRaises(TypeError):
            engine.stage_action({"confidence": 0.9}, payload={"tags": {"a"}}, risk_tier="low")

    def test_subgrid_confidence_nudge_cannot_cross_a_gate(self):
        # Exploit: a change smaller than the 6-decimal digest grid kept the digest valid.
        engine = make_engine()
        card = stage_medium(engine, confidence=0.65)
        self.assertTrue(engine.verify_action_digest(card))
        card.confidence = 0.65 + 4e-7  # same six-decimal string, different value
        self.assertFalse(engine.verify_action_digest(card))

        critical = stage_critical(engine, confidence=0.9499996)
        self.assertEqual(critical.confidence, 0.949999, "staging truncates; it never rounds up")
        self.assertFalse(engine.preflight_verification(critical.id)["allowed"])

    def test_sub_floor_confidence_is_not_rounded_over_the_medium_floor(self):
        engine = make_engine()
        card = stage_medium(engine, confidence=0.6499996)
        self.assertFalse(engine.preflight_verification(card.id)["allowed"])

    def test_malformed_bft_ballots_never_raise(self):
        bft = BFTConsensusEngine(registered_agents=BFT_KEYS)
        card = ActionCard(id="act-bft-malformed", title="t", risk_tier="critical",
                          payload={"x": 1}, signature="sig-malformed")
        malformed = [
            None, "x", 3, [], {},
            {"voter_agent_id": ["Agent-Alpha"], "vote": True, "signature": "x", "card_id": card.id},
            {"voter_agent_id": "Agent-Alpha", "vote": "yes", "signature": "x", "card_id": card.id},
            {"voter_agent_id": "Agent-Alpha", "vote": True, "signature": b"raw", "card_id": card.id},
            {"voter_agent_id": "Agent-Alpha", "vote": True, "signature": "é" * 4, "card_id": card.id},
            {"voter_agent_id": "Agent-Alpha", "vote": True, "signature": None, "card_id": card.id},
            {"voter_agent_id": "Unknown", "vote": True, "signature": "0" * 64, "card_id": card.id},
        ]
        result = bft.evaluate_quorum(card, malformed)
        self.assertFalse(result["approved"])
        self.assertEqual(result["valid_count"], 0)
        for container in (None, "string", {"a": 1}):
            self.assertEqual(bft.evaluate_quorum(card, container)["valid_count"], 0)

    def test_bft_ballot_must_verify_under_the_registered_key(self):
        bft = BFTConsensusEngine(registered_agents=BFT_KEYS)
        card = ActionCard(id="act-bft-key", title="t", risk_tier="critical",
                          payload={"x": 1}, signature="sig-key")
        forged = bft.collect_ballot(card, "Agent-Alpha", True, ATTACKER_KEY)
        self.assertEqual(bft.evaluate_quorum(card, [forged])["valid_count"], 0)
        intruder = bft.collect_ballot(card, "Intruder", True, ATTACKER_KEY)
        self.assertEqual(bft.evaluate_quorum(card, [intruder])["valid_count"], 0)

    def test_quorum_ratio_must_be_a_fraction_in_zero_to_one(self):
        for ratio in (0, -0.5, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                BFTConsensusEngine(registered_agents=BFT_KEYS, required_quorum_ratio=ratio)

    def test_first_valid_ballot_per_voter_counts(self):
        bft = BFTConsensusEngine(registered_agents=BFT_KEYS)
        card = ActionCard(id="act-bft-twice", title="t", risk_tier="critical",
                          payload={"x": 1}, signature="sig-twice")
        no = bft.collect_ballot(card, "Agent-Alpha", False, BFT_KEYS["Agent-Alpha"])
        yes = bft.collect_ballot(card, "Agent-Alpha", True, BFT_KEYS["Agent-Alpha"])
        self.assertEqual(bft.evaluate_quorum(card, [no, yes])["valid_count"], 0)

    def test_quorum_is_unreachable_without_registered_agents(self):
        engine = make_engine(registered_bft_agents=None)
        card = stage_critical(engine)
        with self.assertRaises(ValueError):
            engine.collect_bft_ballot(card.id, "Agent-Alpha", True, BFT_KEYS["Agent-Alpha"])
        card.metadata["ballots"] = [
            engine.bft_engine.collect_ballot(card, "Agent-Alpha", True, BFT_KEYS["Agent-Alpha"])
        ]
        result = engine.preflight_verification(card.id)
        self.assertFalse(result["allowed"])
        self.assertFalse(result["bft_consensus"]["approved"])

    def test_collect_ballot_requires_the_registered_key(self):
        engine = make_engine()
        card = stage_critical(engine)
        with self.assertRaises(ValueError):
            engine.collect_bft_ballot(card.id, "Agent-Alpha", True, ATTACKER_KEY)
        with self.assertRaises(ValueError):
            engine.collect_bft_ballot(card.id, "Intruder", True, ATTACKER_KEY)


class TestMerkleLedger(unittest.TestCase):
    """C5: unsigned cards are refused, domain-separated hashing, odd leaves promoted, count committed."""

    @staticmethod
    def _signed_cards(count):
        engine = make_engine()
        return [stage_medium(engine) for _ in range(count)]

    def test_unsigned_card_is_rejected_by_the_ledger(self):
        # Exploit: an unsigned card could be appended and given a valid-looking leaf.
        ledger = MerkleActionLedger()
        with self.assertRaises(ValueError):
            ledger.append_action(ActionCard(id="act-unsigned", title="t"))
        self.assertEqual(ledger.leaves, [])

    def test_odd_leaf_inclusion_proofs_verify(self):
        cards = self._signed_cards(9)
        for count in range(1, 10):
            ledger = MerkleActionLedger()
            for card in cards[:count]:
                ledger.append_action(card)
            root = ledger.get_merkle_root()
            for index in range(count):
                bundle = ledger.generate_inclusion_proof(index)
                leaf = bundle["leaf_hash"]
                self.assertTrue(ledger.verify_inclusion_proof(leaf, bundle, root), (count, index))
                self.assertTrue(ledger.verify_inclusion_proof(leaf, bundle["proof"], root), (count, index))
                self.assertTrue(
                    ledger.verify_inclusion_proof(leaf, list(bundle["proof"]), root, leaf_count=count),
                    (count, index),
                )
                other = ("1" if leaf[0] == "0" else "0") + leaf[1:]
                self.assertFalse(ledger.verify_inclusion_proof(other, bundle, root))
                self.assertFalse(
                    ledger.verify_inclusion_proof(leaf, bundle["proof"], root, leaf_count=count + 1),
                    "a wrong leaf count must not verify",
                )

    def test_promoted_leaf_has_no_duplicated_sibling(self):
        cards = self._signed_cards(3)
        ledger = MerkleActionLedger()
        for card in cards:
            ledger.append_action(card)
        # The third leaf is promoted at level 0 (no sibling there) and paired at level 1 only.
        self.assertEqual(len(ledger.generate_inclusion_proof(2)["proof"]), 1)

    def test_duplicating_the_last_leaf_changes_the_root(self):
        cards = self._signed_cards(3)
        three = MerkleActionLedger()
        for card in cards:
            three.append_action(card)
        four = MerkleActionLedger()
        for card in cards + [cards[2]]:
            four.append_action(card)
        self.assertNotEqual(three.get_merkle_root(), four.get_merkle_root())

    def test_leaf_hashes_are_domain_separated(self):
        self.assertNotEqual(
            MerkleActionLedger._leaf_hash(b"payload"),
            hashlib.sha256(b"payload").hexdigest(),
        )

    def test_record_event_returns_the_leaf_and_rejects_unknown_events(self):
        ledger = MerkleActionLedger()
        card = self._signed_cards(1)[0]
        record = ledger.record_event(card, "approve")
        self.assertEqual(set(record), {"leaf_index", "leaf_hash", "merkle_root"})
        self.assertEqual(ledger.events, ["approve"])
        with self.assertRaises(ValueError):
            ledger.record_event(card, "delete")

    def test_malformed_proofs_return_false_and_never_raise(self):
        cards = self._signed_cards(3)
        ledger = MerkleActionLedger()
        for card in cards:
            ledger.append_action(card)
        root = ledger.get_merkle_root()
        leaf = ledger.leaves[0]
        for steps in (
            [{"hash": 5, "direction": "left"}],
            [{"hash": "zz", "direction": "right"}],
            ["not-a-step"],
            [{"hash": "a" * 64, "direction": "sideways"}],
        ):
            self.assertFalse(ledger.verify_inclusion_proof(leaf, steps, root, leaf_count=3))
        self.assertFalse(ledger.verify_inclusion_proof("zz", [], root, leaf_count=1))
        self.assertFalse(ledger.verify_inclusion_proof(leaf, [], "", leaf_count=1))


class TestAuditLedgerHonesty(unittest.TestCase):
    """C6: audit keys are mandatory, remote sealing keeps the key out of the agent, receiver seals."""

    def test_enabled_audit_without_a_key_raises(self):
        # Exploit: os.urandom(32) silently sealed records under a key nobody could later check.
        with self.assertRaises(ValueError):
            RhenPillarConfig(enable_audit=True)
        with self.assertRaises(ValueError):
            RhenPillarConfig(enable_audit=True, audit_key=b"too-short")

    def test_enabled_file_audit_with_a_key_seals_records(self):
        path = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        stack = PillarStack(RhenPillarConfig(
            enable_audit=True, audit_key=AUDIT_KEY, audit_sink="file", audit_path=path,
        ))
        try:
            entry = stack.authorize_action("Primary action", "Failover", 0.87)
            self.assertIsInstance(entry.mac, str)
            self.assertTrue(stack.audit.verify_chain()["all_valid"])
        finally:
            stack.close()

    def test_socket_config_refuses_an_agent_side_key(self):
        with self.assertRaises(ValueError):
            RhenPillarConfig(
                enable_audit=True, audit_sink="socket",
                audit_address="127.0.0.1:1", audit_key=AUDIT_KEY,
            )

    def test_remote_sealing_stack_holds_no_key(self):
        stack = PillarStack(RhenPillarConfig(
            enable_audit=True, audit_sink="socket", audit_address="127.0.0.1:1",
        ))
        try:
            self.assertTrue(stack.audit.remote_sealing)
            self.assertIsNone(getattr(stack.audit, "_OutOfBandAuditLedger__sealing_key"))
        finally:
            stack.close()

    def test_local_ledger_requires_an_explicit_key(self):
        with self.assertRaises(ValueError):
            OutOfBandAuditLedger()
        with self.assertRaises(ValueError):
            OutOfBandAuditLedger(key=b"short")

    def test_remote_ledger_refuses_a_key_argument(self):
        with self.assertRaises(ValueError):
            OutOfBandAuditLedger(key=AUDIT_KEY, sink=_CaptureSocketSink())

    def test_remote_ledger_emits_records_without_a_mac(self):
        sink = _CaptureSocketSink()
        ledger = OutOfBandAuditLedger(sink=sink)
        entry = ledger.record("decision", {"chosen": "Failover"})
        self.assertIsNone(entry.mac)
        self.assertEqual(len(sink.frames), 1)
        frame = sink.frames[0]
        self.assertNotIn("mac", frame)
        self.assertEqual(frame["entry_hash"], entry.entry_hash)
        self.assertEqual(frame["sequence"], 1)
        self.assertEqual(frame["prev_hash"], OutOfBandAuditLedger.GENESIS)

    def test_remote_ledger_reports_mac_as_unchecked(self):
        ledger = OutOfBandAuditLedger(sink=_CaptureSocketSink())
        entry = ledger.record("decision", {"chosen": "Failover"})
        result = ledger.verify(entry)
        self.assertTrue(result["valid"])
        self.assertIsNone(result["mac_valid"])

    def test_local_ledger_still_detects_a_rewritten_payload(self):
        ledger = OutOfBandAuditLedger(key=AUDIT_KEY)
        entry = ledger.record("decision", {"chosen": "Failover"})
        forged = entry.to_dict()
        forged["payload"]["chosen"] = "Scale"
        self.assertFalse(ledger.verify(forged)["mac_valid"])

    def test_failed_emit_does_not_consume_a_sequence_number(self):
        dead = SocketAuditSink("127.0.0.1:1", reconnect_attempts=1)
        ledger = OutOfBandAuditLedger(sink=dead)
        with self.assertRaises(AuditSinkError):
            ledger.record("decision", {"q": "primary"})
        self.assertEqual(ledger.entries, [])
        self.assertEqual(ledger.audit_failures, 1)

    def test_oversized_record_fails_closed_before_sending(self):
        sink = SocketAuditSink("127.0.0.1:1")
        with self.assertRaises(AuditSinkError):
            sink.emit({"payload": "x" * (2 * 1024 * 1024)})
        self.assertEqual(sink.dropped_records, 1)

    def test_receiver_requires_a_sealing_key(self):
        store = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        with self.assertRaises(ValueError):
            serve_audit_socket("127.0.0.1:1", store)
        with self.assertRaises(ValueError):
            serve_audit_socket("127.0.0.1:1", store, key=b"short")

    def test_receiver_rejects_tampered_frames_and_seals_valid_ones(self):
        # Exploit: a record altered after sealing (or with a forged hash) reached the store.
        address = f"127.0.0.1:{_free_port()}"
        store = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        producer_sink = _CaptureSocketSink()
        producer = OutOfBandAuditLedger(sink=producer_sink)
        producer.record("decision", {"chosen": "Failover"})
        producer.record("decision", {"chosen": "Scale"})
        good_first, good_second = producer_sink.frames

        tampered = json.loads(json.dumps(good_first))
        tampered["payload"]["chosen"] = "Shut down everything"
        forged_hash = dict(good_first, entry_hash="f" * 64)
        frames = [_frame(tampered), _frame(forged_hash), _frame(good_first), _frame(good_second)]

        thread, stop = _start_receiver(address, store, max_iterations=2)
        try:
            _send(address, b"".join(frames))
            thread.join(timeout=5)
        finally:
            stop.set()

        with open(store, "r", encoding="utf-8") as handle:
            stored = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual([r["payload"]["chosen"] for r in stored], ["Failover", "Scale"])
        self.assertTrue(all(isinstance(r.get("mac"), str) for r in stored))
        self.assertTrue(verify_sealed_records(stored, AUDIT_KEY)["all_valid"])
        self.assertFalse(verify_sealed_records(stored, b"another-key-0123456789")["all_valid"])

    def test_stored_records_detect_rewrites_and_gaps(self):
        local = OutOfBandAuditLedger(key=AUDIT_KEY)
        for index in range(4):
            local.record("step", {"i": index})
        records = [entry.to_dict() for entry in local.entries]
        self.assertTrue(verify_sealed_records(records, AUDIT_KEY)["all_valid"])

        rewritten = json.loads(json.dumps(records))
        rewritten[1]["payload"]["i"] = 99
        self.assertEqual(verify_sealed_records(rewritten, AUDIT_KEY)["first_invalid_index"], 1)

        gapped = records[:1] + records[2:]
        report = verify_sealed_records(gapped, AUDIT_KEY)
        self.assertFalse(report["all_valid"])
        self.assertEqual(report["gaps"], [2])

    def test_verification_requires_a_key_of_sufficient_length(self):
        with self.assertRaises(ValueError):
            verify_sealed_records([], b"short")


if __name__ == "__main__":
    unittest.main()
