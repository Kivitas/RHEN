"""
Agent Tools and Action Card Staging Primitives for EOS LLM.

Inspired by Laya command center and Open-Jev agent verification paradigms:
- System One decisions propose actions which are staged as structured Action Cards.
- Every Action Card undergoes risk tiering, permission checks, and preflight verification
  before execution boundaries are crossed.
- Every Action Card carries an HMAC-SHA256 digest over its signed fields (version tag,
  id, risk tier, action type, title, persona, sorted required permissions, canonical
  payload JSON, confidence and timestamp at 6 decimals). Any change to a signed field
  breaks verification. Metadata is deliberately outside the digest.
- There is no built-in signing key. It must be passed explicitly, generated per process
  (``ephemeral_key=True``) or supplied through ``EOS_ACTION_SIGNING_KEY``.
- Execution always re-runs preflight verification unless a caller explicitly bypasses it,
  and the bypass is recorded in the card's metadata.
- CRITICAL risk-tier transitions require three mandatory checks: dual-custody co-signing by
  a pre-registered secondary agent (distinct from the card's persona), a BFT quorum of
  pre-registered agents, and a singleton conformal set. Missing evidence fails each check.
- Lifecycle transitions (stage, approve, execute, reject) are recorded in a domain-separated
  Merkle ledger.
"""

import hashlib
import hmac
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union


def _robust_serialize_value(val: Any) -> Any:
    """Robust serializer handling torch.Tensor, numpy types, sets, dataclasses, dicts, lists."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    try:
        import torch
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return val.item()
            return val.detach().cpu().tolist()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    if isinstance(val, (set, frozenset)):
        try:
            return sorted(list(val))
        except TypeError:
            return list(val)
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return val.to_dict()
    if isinstance(val, dict):
        return {str(k): _robust_serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_robust_serialize_value(v) for v in val]
    return str(val)

RISK_TIERS: Tuple[str, ...] = ("low", "medium", "high", "critical")
ACTION_TYPES: Tuple[str, ...] = ("execute", "query", "notify", "escalate", "stage")

# Permissions every card of a tier must hold. Caller-supplied requirements are unioned
# on top of these and can never remove them.
_TIER_DEFAULT_PERMISSIONS: Dict[str, Tuple[str, ...]] = {
    "low": ("query", "read"),
    "medium": ("execute", "write"),
    "high": ("execute_privileged", "write"),
    "critical": ("execute_privileged", "write"),
}
_DEFAULT_GRANTED_PERMISSIONS: Tuple[str, ...] = (
    "read", "write", "execute", "query", "stage", "notify", "execute_privileged",
)

_DIGEST_VERSION = "eos-action-v2"
_ENV_SIGNING_KEY = "RHEN_ACTION_SIGNING_KEY"
_FALLBACK_ENV_SIGNING_KEY = "EOS_ACTION_SIGNING_KEY"
_MIN_KEY_BYTES = 16
_CONFIDENCE_SCALE = 1_000_000  # confidence is committed at 6 decimal places
_LEAF_DOMAIN = b"\x00"
_NODE_DOMAIN = b"\x01"
_ROOT_DOMAIN = b"\x02"
_LEDGER_EVENTS: Tuple[str, ...] = ("stage", "approve", "execute", "reject")
_STATUS_TO_LEDGER_EVENT: Dict[str, str] = {
    "staged": "stage",
    "approved": "approve",
    "executed": "execute",
    "failed": "execute",
    "rejected": "reject",
}

# Canonical JSON: sorted keys and no insignificant whitespace, so one logical payload
# always yields one byte string.
_CANONICAL_JSON_KWARGS = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False, "allow_nan": False}


def _require_canonical_json(value: Any, path: str = "payload", _stack: Optional[Set[int]] = None) -> None:
    """Raise TypeError unless ``value`` is plain JSON data (no sets, bytes, objects, cycles, NaN)."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path}: non-finite float is not canonical JSON")
        return
    if isinstance(value, (dict, list, tuple)):
        stack = _stack if _stack is not None else set()
        if id(value) in stack:
            raise TypeError(f"{path}: circular reference is not canonical JSON")
        stack.add(id(value))
        try:
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError(f"{path}: object keys must be str, not {type(key).__name__}")
                    _require_canonical_json(item, f"{path}.{key}", stack)
            else:
                for index, item in enumerate(value):
                    _require_canonical_json(item, f"{path}[{index}]", stack)
        finally:
            stack.discard(id(value))
        return
    raise TypeError(f"{path}: {type(value).__name__} is not canonical JSON data")


def canonical_payload_json(payload: Any) -> str:
    """Deterministic JSON text for an action payload (the form that enters the digest).

    Raises TypeError for anything that is not plain JSON data: sets, bytes, arbitrary
    objects, non-string keys, NaN or infinity. There is no ``default=str`` fallback,
    because a lossy rendering would let two different payloads share one digest.
    """
    _require_canonical_json(payload)
    return json.dumps(payload, **_CANONICAL_JSON_KWARGS)


def _key_bytes(key: Any) -> bytes:
    """Normalize a secret to bytes. Wrong types and empty secrets are refused."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("secret key must be bytes or str")
    if len(key) == 0:
        raise ValueError("secret key must not be empty")
    return bytes(key)


def _digest_equal(expected: str, actual: Any) -> bool:
    """Constant-time comparison of two hex digests. Never raises on malformed input."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    return hmac.compare_digest(expected.encode("utf-8"), actual.encode("utf-8"))


def _digest_bytes(value: Any) -> bytes:
    """Decode a 64-character hex digest to exactly 32 bytes, or raise ValueError."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected a 64-character hex digest")
    raw = bytes.fromhex(value)
    if len(raw) != 32:
        raise ValueError("expected a 64-character hex digest")
    return raw


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantise_confidence(value: float) -> float:
    """Truncate to the 6-decimal grid that the digest commits to.

    Truncation rather than rounding: a caller's 0.6499996 becomes 0.649999 and stays
    below the 0.65 medium floor instead of being promoted over it.
    """
    return math.floor(value * _CONFIDENCE_SCALE + 1e-9) / _CONFIDENCE_SCALE


def _on_confidence_grid(value: Any) -> bool:
    if not _is_number(value) or not math.isfinite(value):
        return False
    return _quantise_confidence(float(value)) == float(value)


def _staged_confidence(raw: Any) -> float:
    """Validate and quantise a confidence supplied at staging time."""
    if raw is None or isinstance(raw, (bool, str, bytes)):
        raise TypeError(f"confidence must be numeric, got {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be a finite value in [0, 1], got {raw!r}")
    return _quantise_confidence(value)


def _check_tier_and_type(risk_tier: Any, action_type: Any) -> None:
    if risk_tier not in RISK_TIERS:
        raise ValueError(f"risk_tier must be one of {RISK_TIERS} (case-sensitive), got {risk_tier!r}")
    if action_type not in ACTION_TYPES:
        raise ValueError(f"action_type must be one of {ACTION_TYPES} (case-sensitive), got {action_type!r}")


def _effective_permissions(risk_tier: str, requested: Any) -> List[str]:
    """Tier defaults UNION caller-supplied requirements. A caller can only add to them."""
    extra: Set[str] = set()
    if requested is not None:
        if not isinstance(requested, (list, tuple, set, frozenset)):
            raise ValueError("required_permissions must be a list of permission names")
        for permission in requested:
            if not isinstance(permission, str) or not permission:
                raise ValueError(f"invalid permission name: {permission!r}")
            extra.add(permission)
    return sorted(set(_TIER_DEFAULT_PERMISSIONS[risk_tier]) | extra)


@dataclass
class ActionCard:
    """Structured representation of a proposed agent action or tool invocation."""
    id: str
    title: str
    persona: str = "DefaultAgent"
    risk_tier: str = "medium"  # "low", "medium", "high", "critical"
    action_type: str = "execute"  # "execute", "query", "notify", "escalate", "stage"
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "staged"  # "staged", "approved", "rejected", "executed", "failed"
    confidence: float = 0.5
    required_permissions: List[str] = field(default_factory=list)
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None  # HMAC-SHA256 signature
    cosigner_agent_id: Optional[str] = None  # Dual-custody co-signer
    cosigner_signature: Optional[str] = None  # Co-signer HMAC-SHA256 signature

    def to_dict(self) -> Dict[str, Any]:
        """Convert action card to dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "persona": self.persona,
            "risk_tier": self.risk_tier,
            "action_type": self.action_type,
            "payload": _robust_serialize_value(self.payload),
            "status": self.status,
            "confidence": float(self.confidence),
            "required_permissions": _robust_serialize_value(self.required_permissions),
            "verification_checks": _robust_serialize_value(self.verification_checks),
            "execution_result": _robust_serialize_value(self.execution_result),
            "timestamp": float(self.timestamp),
            "metadata": _robust_serialize_value(self.metadata),
            "signature": self.signature,
            "cosigner_agent_id": self.cosigner_agent_id,
            "cosigner_signature": self.cosigner_signature,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize action card to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionCard":
        """Instantiate ActionCard from dictionary."""
        kwargs = dict(data)
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, data_str: str) -> "ActionCard":
        """Instantiate ActionCard from JSON string."""
        return cls.from_dict(json.loads(data_str))

    def to_markdown(self) -> str:
        """Render a formatted markdown card for display or audit trails."""
        badge = {
            "low": "[LOW RISK]",
            "medium": "[MEDIUM RISK]",
            "high": "[HIGH RISK]",
            "critical": "[CRITICAL RISK]",
        }.get(self.risk_tier, "[UNKNOWN RISK]")

        status_icon = {
            "staged": "[STAGED]",
            "approved": "[APPROVED]",
            "rejected": "[REJECTED]",
            "executed": "[EXECUTED]",
            "failed": "[FAILED]",
        }.get(self.status, f"[{self.status.upper()}]")

        lines = [
            f"### Action Card: `{self.title}`",
            f"- **Card ID**: `{self.id}`",
            f"- **Status**: **{status_icon}** | **Tier**: {badge}",
            f"- **Persona**: `{self.persona}` | **Type**: `{self.action_type}`",
            f"- **Confidence**: `{self.confidence * 100:.1f}%`",
            f"- **Required Permissions**: `{', '.join(self.required_permissions) or 'None'}`",
            f"- **HMAC Signature**: `{self.signature or 'UNSIGNED'}`",
        ]
        if self.cosigner_agent_id:
            lines.append(f"- **Dual-Custody Co-Signer**: `{self.cosigner_agent_id}` (Signature: `{self.cosigner_signature}`)")

        lines.extend([
            "- **Payload**:",
            "  ```json",
            f"  {self.payload}",
            "  ```",
        ])
        if self.verification_checks:
            lines.append("- **Verification Checks**:")
            for check in self.verification_checks:
                passed = "PASS" if check.get("passed", False) else "FAIL"
                lines.append(f"  - [{passed}] {check.get('name', 'Check')}: {check.get('message', '')}")

        if self.execution_result:
            lines.append(f"- **Execution Result**: `{self.execution_result}`")

        return "\n".join(lines)


class MerkleInclusionPath(list):
    """Sibling path of an inclusion proof: ``{"hash", "direction"}`` steps.

    The leaf count the root commits to travels with the path as ``leaf_count``, so the
    path can be checked against its root without any other context.
    """

    def __init__(self, steps: Sequence[Dict[str, str]] = (), leaf_count: Optional[int] = None) -> None:
        super().__init__(steps)
        self.leaf_count = leaf_count


class MerkleActionLedger:
    """
    Append-only Merkle ledger of action-card lifecycle events, with inclusion proofs.

    Hashing is domain separated: a leaf is ``sha256(0x00 || data)``, an interior node is
    ``sha256(0x01 || left || right)`` and the root is ``sha256(0x02 || leaf_count || top)``,
    so the root commits to the number of leaves. An odd node is promoted unchanged to the
    next level rather than duplicated. Each leaf commits to one lifecycle event (stage,
    approve, execute or reject), the card's id, tier, status, signature and timestamp.

    The ledger is tamper-evident for anyone who holds an earlier root. It lives in the
    engine's own process, so it does not by itself resist a compromised process; the
    out-of-band audit ledger in ``rhen_verify`` is the component for that threat.
    """

    def __init__(self):
        self.leaves: List[str] = []  # hex leaf hashes, in append order
        self.events: List[str] = []  # lifecycle event recorded by each leaf
        self.cards: List[ActionCard] = []
        self.tree_levels: List[List[str]] = []

    @staticmethod
    def _leaf_hash(data: bytes) -> str:
        return hashlib.sha256(_LEAF_DOMAIN + data).hexdigest()

    @staticmethod
    def _node_hash(left: Any, right: Any) -> str:
        return hashlib.sha256(_NODE_DOMAIN + _digest_bytes(left) + _digest_bytes(right)).hexdigest()

    @staticmethod
    def _root_hash(top: Any, leaf_count: int) -> str:
        return hashlib.sha256(_ROOT_DOMAIN + str(leaf_count).encode("ascii") + _digest_bytes(top)).hexdigest()

    def _rebuild_tree(self) -> None:
        """Reconstruct the tree levels from the leaves. Odd nodes are promoted, not duplicated."""
        if not self.leaves:
            self.tree_levels = []
            return

        level = list(self.leaves)
        levels = [level]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    next_level.append(self._node_hash(level[i], level[i + 1]))
                else:
                    next_level.append(level[i])
            levels.append(next_level)
            level = next_level
        self.tree_levels = levels

    def record_event(self, card: ActionCard, event: str) -> Dict[str, Any]:
        """Append one lifecycle event for ``card`` and return its leaf and the new root.

        Args:
            card: a signed ActionCard. Unsigned cards are refused.
            event: one of ``"stage"``, ``"approve"``, ``"execute"`` or ``"reject"``.

        Returns:
            ``{"leaf_index", "leaf_hash", "merkle_root"}``.
        """
        if event not in _LEDGER_EVENTS:
            raise ValueError(f"unknown ledger event {event!r}; expected one of {_LEDGER_EVENTS}")
        if not isinstance(card.signature, str) or not card.signature:
            raise ValueError("unsigned ActionCard cannot be recorded in the Merkle ledger")
        leaf_data = json.dumps(
            [event, card.id, card.risk_tier, card.status, card.signature, repr(float(card.timestamp))],
            separators=(",", ":"),
        ).encode("ascii")
        leaf_hash = self._leaf_hash(leaf_data)

        self.cards.append(card)
        self.events.append(event)
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
        return {
            "leaf_index": len(self.leaves) - 1,
            "leaf_hash": leaf_hash,
            "merkle_root": self.get_merkle_root(),
        }

    def append_action(self, card: ActionCard) -> Dict[str, Any]:
        """Record the card's current state. The ledger event is derived from its status."""
        event = _STATUS_TO_LEDGER_EVENT.get(card.status)
        if event is None:
            raise ValueError(f"no ledger event for ActionCard status {card.status!r}")
        return self.record_event(card, event)

    def get_merkle_root(self) -> str:
        """Committed root over all leaves. An empty ledger commits to zero leaves."""
        if not self.leaves:
            return hashlib.sha256(_ROOT_DOMAIN + b"0").hexdigest()
        return self._root_hash(self.tree_levels[-1][0], len(self.leaves))

    def generate_inclusion_proof(self, leaf_index: int) -> Dict[str, Any]:
        """Sibling path from a leaf to the root, plus the leaf count the root commits to."""
        leaf_count = len(self.leaves)
        if leaf_index < 0 or leaf_index >= leaf_count:
            raise IndexError(f"Leaf index {leaf_index} out of bounds for ledger with {leaf_count} leaves.")

        path = MerkleInclusionPath(leaf_count=leaf_count)
        idx = leaf_index
        for level in self.tree_levels[:-1]:
            if idx % 2 == 0:
                # Target is a left child; its sibling, if any, is on the right.
                if idx + 1 < len(level):
                    path.append({"hash": level[idx + 1], "direction": "right"})
                # Otherwise the node is unpaired and promoted unchanged: no sibling.
            else:
                # Target is a right child; its sibling is on the left.
                path.append({"hash": level[idx - 1], "direction": "left"})
            idx //= 2

        return {
            "leaf_index": leaf_index,
            "leaf_hash": self.leaves[leaf_index],
            "proof": path,
            "leaf_count": leaf_count,
            "merkle_root": self.get_merkle_root(),
        }

    def verify_inclusion_proof(
        self,
        leaf_hash: str,
        proof: Union[List[Dict[str, str]], Dict[str, Any]],
        expected_root: str,
        leaf_count: Optional[int] = None,
    ) -> bool:
        """Fold the sibling path up from ``leaf_hash`` and compare with ``expected_root``.

        The leaf count the root commits to is taken from the proof bundle, from the path
        itself, or from the ``leaf_count`` argument. Without one, the root cannot be checked
        and the proof is refused. Malformed input returns False and never raises.
        """
        if not isinstance(expected_root, str) or not expected_root:
            return False
        if isinstance(proof, dict):
            steps = proof.get("proof", [])
            count = proof.get("leaf_count", leaf_count)
        else:
            steps = proof
            count = leaf_count if leaf_count is not None else getattr(proof, "leaf_count", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return False

        try:
            _digest_bytes(leaf_hash)
            curr = leaf_hash
            for step in steps:
                if not isinstance(step, dict):
                    return False
                sibling = step.get("hash")
                direction = step.get("direction")
                if direction == "left":
                    curr = self._node_hash(sibling, curr)
                elif direction == "right":
                    curr = self._node_hash(curr, sibling)
                else:
                    return False
            computed = self._root_hash(curr, count)
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(computed.encode("ascii"), expected_root.encode("utf-8"))


class BFTConsensusEngine:
    """
    Multi-Agent Byzantine Fault Tolerant (BFT) Quorum Consensus for critical actions.
    Requires M-of-N independent agent signatures before granting execution authority.
    Only pre-registered agents can vote; with no registered agents no quorum is reachable.
    """

    def __init__(
        self,
        registered_agents: Optional[Dict[str, Union[bytes, str]]] = None,
        required_quorum_ratio: float = 0.67,
    ):
        ratio = float(required_quorum_ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("required_quorum_ratio must be in (0, 1]")
        self.registered_agents: Dict[str, bytes] = {}
        for agent_id, key in (registered_agents or {}).items():
            self.register_agent(agent_id, key)
        self.required_quorum_ratio = ratio

    def register_agent(self, agent_id: str, secret_key: Union[bytes, str]) -> None:
        """Register or update an agent's secret key for BFT quorum consensus."""
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("BFT agent id must be a non-empty string")
        self.registered_agents[agent_id] = _key_bytes(secret_key)

    def _generate_ballot_digest(
        self,
        card: ActionCard,
        voter_agent_id: str,
        vote: bool,
        secret_key: Union[bytes, str],
    ) -> str:
        """HMAC digest binding one voter's decision to one signed card revision."""
        material = json.dumps(
            [
                "bft_ballot",
                card.id,
                card.risk_tier,
                canonical_payload_json(card.payload),
                voter_agent_id,
                bool(vote),
                card.signature if isinstance(card.signature, str) else "",
            ],
            separators=(",", ":"),
        ).encode("ascii")
        return hmac.new(_key_bytes(secret_key), material, digestmod=hashlib.sha256).hexdigest()

    def collect_ballot(
        self,
        card: ActionCard,
        voter_agent_id: str,
        vote: bool,
        secret_key: Union[bytes, str],
    ) -> Dict[str, Any]:
        """Sign one voter's decision for this card revision. ``vote`` must be a bool."""
        if not isinstance(vote, bool):
            raise TypeError("vote must be a bool")
        sig = self._generate_ballot_digest(card, voter_agent_id, vote, secret_key)
        return {
            "card_id": card.id,
            "voter_agent_id": voter_agent_id,
            "vote": vote,
            "signature": sig,
            "timestamp": time.time(),
        }

    def _verified_ballot(self, card: ActionCard, ballot: Any) -> Optional[Tuple[str, bool]]:
        """Return ``(voter, vote)`` for a well-formed ballot signed by its registered key, else None."""
        if not isinstance(ballot, dict):
            return None
        voter_id = ballot.get("voter_agent_id")
        if not isinstance(voter_id, str) or voter_id not in self.registered_agents:
            return None
        vote = ballot.get("vote")
        signature = ballot.get("signature")
        if not isinstance(vote, bool) or not isinstance(signature, str):
            return None
        if ballot.get("card_id") != card.id:
            return None
        try:
            expected = self._generate_ballot_digest(card, voter_id, vote, self.registered_agents[voter_id])
        except (TypeError, ValueError):
            return None
        return (voter_id, vote) if _digest_equal(expected, signature) else None

    def evaluate_quorum(
        self,
        card: ActionCard,
        ballots: Any,
    ) -> Dict[str, Any]:
        """
        Counts distinct registered voters whose ballots verify and vote yes.

        Malformed or unverifiable ballots are ignored and never raise. The first valid ballot
        per voter counts, so repeating a vote has no effect. Quorum needs
        valid_count >= ceil(required_quorum_ratio * N) over the N registered agents.
        Returns approved, valid_count, quorum_threshold, certified_digest, total_agents, valid_voters.
        """
        n_total = len(self.registered_agents)
        quorum_threshold = max(1, math.ceil(self.required_quorum_ratio * n_total - 1e-6))
        affirmative_voters: List[str] = []
        seen_voters: Set[str] = set()

        for ballot in (ballots if isinstance(ballots, (list, tuple)) else []):
            parsed = self._verified_ballot(card, ballot)
            if parsed is None:
                continue
            voter_id, vote = parsed
            if voter_id in seen_voters:
                continue
            seen_voters.add(voter_id)
            if vote:
                affirmative_voters.append(voter_id)

        valid_count = len(affirmative_voters)
        approved = n_total > 0 and valid_count >= quorum_threshold

        certified_digest: Optional[str] = None
        if approved:
            cert_material = json.dumps(
                [
                    "bft_certified",
                    card.id,
                    card.signature if isinstance(card.signature, str) else "",
                    sorted(affirmative_voters),
                    valid_count,
                    quorum_threshold,
                ],
                separators=(",", ":"),
            ).encode("ascii")
            certified_digest = hashlib.sha256(cert_material).hexdigest()

        return {
            "approved": approved,
            "valid_count": valid_count,
            "quorum_threshold": quorum_threshold,
            "certified_digest": certified_digest,
            "total_agents": n_total,
            "valid_voters": affirmative_voters,
        }


class ActionStagingEngine:
    """Preflight verification and staging engine for Agent Action Cards."""

    # Default minimum confidence required per risk tier
    RISK_CONFIDENCE_THRESHOLDS = {
        "low": 0.40,
        "medium": 0.65,
        "high": 0.85,
        "critical": 0.95,
    }

    def __init__(
        self,
        default_granted_permissions: Optional[List[str]] = None,
        secret_key: Optional[bytes] = None,
        registered_bft_agents: Optional[Dict[str, bytes]] = None,
        ephemeral_key: bool = False,
    ):
        """
        Args:
            default_granted_permissions: permissions this engine grants. ``None`` selects the
                built-in set; an empty list grants nothing.
            secret_key: HMAC signing key. When omitted, ``ephemeral_key=True`` generates a random
                32-byte key for this instance; otherwise ``EOS_ACTION_SIGNING_KEY`` (at least 16
                bytes) is used. With none of these, construction raises ValueError. There is no
                built-in key.
            registered_bft_agents: BFT voters as ``{agent_id: key}``. ``None`` means no voters,
                so no critical card can reach quorum.
            ephemeral_key: generate a random per-instance signing key. Cards signed by one
                instance cannot be verified by another.
        """
        if default_granted_permissions is None:
            self.granted_permissions = set(_DEFAULT_GRANTED_PERMISSIONS)
        else:
            self.granted_permissions = set(default_granted_permissions)
        self.secret_key = self._resolve_signing_key(secret_key, ephemeral_key)
        self.staged_cards: Dict[str, ActionCard] = {}
        self.history: List[ActionCard] = []
        # Dual-custody registry: a co-signer must be registered here before it can sign.
        # Nothing is provisioned on first use.
        self.trusted_cosigners: Dict[str, bytes] = {}
        self._seen_ids: Set[str] = set()  # every id ever staged; ids are never reused
        self._executed_ids: Set[str] = set()
        # card id -> signature at its last passing preflight. Approval requires a match.
        self._preflight_passed: Dict[str, str] = {}

        # Cryptographic Merkle Action Ledger & Multi-Agent BFT Consensus
        self.merkle_ledger = MerkleActionLedger()
        self.bft_engine = BFTConsensusEngine(registered_agents=registered_bft_agents or {})

    @staticmethod
    def _resolve_signing_key(secret_key: Any, ephemeral_key: bool) -> bytes:
        if secret_key is not None:
            return _key_bytes(secret_key)
        if ephemeral_key:
            return os.urandom(32)
        raw = os.environ.get(_ENV_SIGNING_KEY, "") or os.environ.get(_FALLBACK_ENV_SIGNING_KEY, "")
        if not raw:
            raise ValueError(
                "ActionStagingEngine needs a signing key: pass secret_key=..., pass "
                f"ephemeral_key=True, or set {_ENV_SIGNING_KEY} (or {_FALLBACK_ENV_SIGNING_KEY}) (at least {_MIN_KEY_BYTES} bytes)"
            )
        key = raw.encode("utf-8")
        if len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"{_ENV_SIGNING_KEY} must be at least {_MIN_KEY_BYTES} bytes")
        return key

    def register_cosigner_agent(self, agent_id: str, cosigner_key: Union[bytes, str]) -> None:
        """Pre-register a secondary agent's key for dual-custody co-signing (out of band)."""
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("co-signer id must be a non-empty string")
        self.trusted_cosigners[agent_id] = _key_bytes(cosigner_key)

    @staticmethod
    def _digest_material(card: ActionCard) -> str:
        """Canonical JSON array of the signed fields. Metadata is excluded by design.

        Raises TypeError or ValueError when the card is not in canonical form, so callers
        treat such a card as unverifiable rather than comparing partial data.
        """
        for name in ("id", "risk_tier", "action_type", "title", "persona"):
            if not isinstance(getattr(card, name), str):
                raise TypeError(f"card.{name} must be a string")
        permissions = card.required_permissions
        if not isinstance(permissions, (list, tuple)) or not all(isinstance(p, str) for p in permissions):
            raise TypeError("card.required_permissions must be a list of strings")
        if not _on_confidence_grid(card.confidence):
            raise ValueError("card.confidence must be finite and on the 6-decimal grid")
        if not _is_number(card.timestamp) or not math.isfinite(card.timestamp):
            raise ValueError("card.timestamp must be a finite number")
        return json.dumps(
            [
                _DIGEST_VERSION,
                card.id,
                card.risk_tier,
                card.action_type,
                card.title,
                card.persona,
                sorted(permissions),
                canonical_payload_json(card.payload),
                f"{float(card.confidence):.6f}",
                f"{float(card.timestamp):.6f}",
            ],
            separators=(",", ":"),
        )

    def generate_action_digest(self, card: ActionCard, key: Optional[bytes] = None) -> str:
        """HMAC-SHA256 digest over an Action Card's signed fields.

        Covers the version tag, id, risk tier, action type, title, persona, sorted required
        permissions, canonical payload JSON, confidence (6 decimals) and timestamp (6 decimals).
        Each field is encoded inside one JSON array, so field boundaries cannot be shifted.
        Metadata is not covered. Raises ValueError or TypeError for non-canonical cards.
        """
        signing_key = self.secret_key if key is None else _key_bytes(key)
        material = self._digest_material(card).encode("ascii")
        return hmac.new(signing_key, material, digestmod=hashlib.sha256).hexdigest()

    def sign_action(self, card: ActionCard, key: Optional[bytes] = None) -> ActionCard:
        """Cryptographically sign an Action Card using HMAC-SHA256."""
        card.signature = self.generate_action_digest(card, key=key)
        return card

    def verify_action_digest(self, card: ActionCard, key: Optional[bytes] = None) -> bool:
        """True only when the card's signature matches its current signed fields."""
        if not isinstance(card.signature, str) or not card.signature:
            return False
        try:
            expected = self.generate_action_digest(card, key=key)
        except (TypeError, ValueError):
            return False
        return _digest_equal(expected, card.signature)

    def generate_cosign_digest(
        self,
        card: ActionCard,
        cosigner_agent_id: str,
        cosigner_key: Union[bytes, str],
    ) -> str:
        """Bind a secondary agent's co-signature to one card revision.

        The co-signature covers the primary digest, so a co-signed approval cannot be
        transplanted onto a mutated or re-staged card.
        """
        material = json.dumps(
            ["cosign", card.id, cosigner_agent_id, card.signature if isinstance(card.signature, str) else ""],
            separators=(",", ":"),
        ).encode("ascii")
        return hmac.new(_key_bytes(cosigner_key), material, digestmod=hashlib.sha256).hexdigest()

    def verify_cosigner_digest(self, card: ActionCard) -> bool:
        """Verify the co-signature against the pre-registered key.

        Separation of duties is enforced here too: a co-signer equal to the card's persona
        never verifies.
        """
        cosigner = card.cosigner_agent_id
        if not isinstance(cosigner, str) or not cosigner or not card.cosigner_signature:
            return False
        if cosigner == card.persona:
            return False
        registered_key = self.trusted_cosigners.get(cosigner)
        if registered_key is None:
            return False
        try:
            expected = self.generate_cosign_digest(card, cosigner, registered_key)
        except (TypeError, ValueError):
            return False
        return _digest_equal(expected, card.cosigner_signature)

    def request_dual_custody_cosign(
        self,
        card_id: str,
        cosigner_agent_id: str,
        cosigner_key: bytes,
    ) -> bool:
        """Attach a co-signature from a pre-registered secondary agent.

        Returns True only when the card is staged, the co-signer was registered with
        ``register_cosigner_agent``, the presented key equals the registered key, and the
        co-signer is not the card's persona. Otherwise nothing is attached and False is
        returned. No co-signer is ever provisioned on first use.
        """
        card = self.staged_cards.get(card_id)
        if card is None:
            return False
        if not isinstance(cosigner_agent_id, str) or not cosigner_agent_id:
            return False
        if cosigner_agent_id == card.persona:
            return False
        registered_key = self.trusted_cosigners.get(cosigner_agent_id)
        if registered_key is None:
            return False
        try:
            presented_key = _key_bytes(cosigner_key)
        except (TypeError, ValueError):
            return False
        if not hmac.compare_digest(presented_key, registered_key):
            return False
        card.cosigner_agent_id = cosigner_agent_id
        card.cosigner_signature = self.generate_cosign_digest(card, cosigner_agent_id, registered_key)
        return True

    def stage_action(
        self,
        decision_result: Any,
        action_type: str = "execute",
        payload: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        persona: str = "DefaultAgent",
        risk_tier: Optional[str] = None,
        required_permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionCard:
        """Stage a new Action Card derived from a decision result and record it in the Merkle ledger.

        Raises ValueError for an unknown (case-sensitive) risk tier or action type, a non-dict
        payload, a confidence that is not finite or lies outside [0, 1], or an id that was
        already staged. Raises TypeError for a payload that is not canonical JSON data.
        """
        if isinstance(decision_result, ActionCard):
            return self._stage_existing_card(decision_result)

        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES} (case-sensitive), got {action_type!r}")
        if risk_tier is not None and risk_tier not in RISK_TIERS:
            raise ValueError(f"risk_tier must be one of {RISK_TIERS} (case-sensitive), got {risk_tier!r}")
        if not isinstance(persona, str):
            raise TypeError("persona must be a string")
        payload = {} if payload is None else payload
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        canonical_payload_json(payload)  # TypeError for non-JSON data
        metadata = {} if metadata is None else dict(metadata)

        # Extract confidence
        raw_confidence: Any = 0.5
        if hasattr(decision_result, "confidence"):
            raw_confidence = decision_result.confidence
        elif isinstance(decision_result, dict) and "confidence" in decision_result:
            raw_confidence = decision_result["confidence"]
        confidence = _staged_confidence(raw_confidence)

        # Auto-infer title if missing
        if not title:
            if hasattr(decision_result, "choice_label") and decision_result.choice_label:
                title = f"Action: {decision_result.choice_label}"
            elif hasattr(decision_result, "decision"):
                title = f"Action for Decision: {decision_result.decision}"
            elif isinstance(decision_result, dict) and "choice_label" in decision_result:
                title = f"Action: {decision_result['choice_label']}"
            else:
                title = f"Action Invocation ({action_type})"
        if not isinstance(title, str):
            raise TypeError("title must be a string")

        # Infer risk tier based on action_type and confidence if not specified
        if risk_tier is None:
            if action_type in ("notify", "query", "stage"):
                risk_tier = "low"
            elif action_type == "execute":
                if confidence >= 0.85:
                    risk_tier = "medium"
                elif confidence >= 0.65:
                    risk_tier = "high"
                else:
                    risk_tier = "critical"
            else:
                risk_tier = "medium"

        card = ActionCard(
            id=f"act-{uuid.uuid4().hex}",
            title=title,
            persona=persona,
            risk_tier=risk_tier,
            action_type=action_type,
            payload=payload,
            status="staged",
            confidence=confidence,
            required_permissions=_effective_permissions(risk_tier, required_permissions),
            verification_checks=[],
            timestamp=time.time(),
            metadata=metadata,
        )
        self.sign_action(card)
        return self._commit_new_card(card)

    def _check_new_id(self, card_id: Any) -> None:
        if not isinstance(card_id, str) or not card_id:
            raise ValueError("ActionCard id must be a non-empty string")
        if card_id in self._seen_ids:
            raise ValueError(f"ActionCard id '{card_id}' was already staged; re-staging an existing id is refused")

    def _stage_existing_card(self, card: ActionCard) -> ActionCard:
        """Stage a caller-built ActionCard.

        A signed card is accepted only if this engine's key verifies its exact revision.
        An unsigned card is normalised (confidence quantised, tier defaults unioned into its
        permissions) and then signed here.
        """
        _check_tier_and_type(card.risk_tier, card.action_type)
        self._check_new_id(card.id)
        if not isinstance(card.payload, dict):
            raise ValueError("payload must be a dict")
        canonical_payload_json(card.payload)  # TypeError for non-JSON data
        if card.signature:
            if not self.verify_action_digest(card):
                raise ValueError("ActionCard signature does not verify under this engine's signing key")
        else:
            card.confidence = _staged_confidence(card.confidence)
            card.required_permissions = _effective_permissions(card.risk_tier, card.required_permissions)
            self.sign_action(card)
        card.status = "staged"
        card.execution_result = None
        card.verification_checks = []
        return self._commit_new_card(card)

    def _commit_new_card(self, card: ActionCard) -> ActionCard:
        """Record the stage event (this raises before any state changes if it cannot), then store."""
        entry = self.merkle_ledger.record_event(card, "stage")
        self.staged_cards[card.id] = card
        self._seen_ids.add(card.id)
        card.metadata["merkle_leaf_index"] = entry["leaf_index"]
        card.metadata["merkle_leaf_hash"] = entry["leaf_hash"]
        card.metadata["merkle_root"] = entry["merkle_root"]
        return card

    def verify_action(
        self,
        card: ActionCard,
        granted_permissions: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Run preflight checks against an Action Card and store them in ``card.verification_checks``.

        Permissions are the card's requirements plus its tier defaults, checked against the
        engine's granted set. A ``granted_permissions`` argument can only narrow that set.
        Critical cards also need dual custody, a BFT quorum, confidence >= 0.95 and a
        singleton conformal set. Each of these is mandatory: missing evidence fails the check.
        """
        checks: List[Dict[str, Any]] = []
        error_messages: List[str] = []

        def record(name: str, passed: bool, message: str, **extra: Any) -> None:
            checks.append({"name": name, "passed": bool(passed), "message": message, **extra})
            if not passed:
                error_messages.append(message)

        # 0. The tier and action type must be recognised. Anything else fails closed.
        tier_known = card.risk_tier in RISK_TIERS
        type_known = card.action_type in ACTION_TYPES
        record(
            "Risk Tier & Action Type Validation",
            tier_known and type_known,
            "Risk tier and action type are recognised" if tier_known and type_known
            else f"Unrecognised risk tier {card.risk_tier!r} or action type {card.action_type!r}",
        )

        # 1. Permission check: tier defaults UNION the card's own requirements.
        granted = set(self.granted_permissions)
        if granted_permissions is not None:
            granted &= set(granted_permissions)
        requested = card.required_permissions
        if isinstance(requested, (list, tuple)) and all(isinstance(p, str) for p in requested):
            required = set(_TIER_DEFAULT_PERMISSIONS.get(card.risk_tier, ())) | set(requested)
            missing = sorted(required - granted)
            record(
                "Permission Verification",
                not missing,
                "All required permissions granted" if not missing else f"Insufficient permissions: {missing}",
            )
        else:
            record("Permission Verification", False, "Required permissions are malformed")

        # 2. Confidence threshold for the tier. An unknown tier has no floor and fails.
        threshold = self.RISK_CONFIDENCE_THRESHOLDS.get(card.risk_tier)
        confidence_ok = (
            threshold is not None
            and _is_number(card.confidence)
            and math.isfinite(card.confidence)
            and card.confidence >= threshold
        )
        if threshold is None:
            confidence_msg = f"No confidence floor is defined for risk tier {card.risk_tier!r}"
        elif not _is_number(card.confidence):
            confidence_msg = "Confidence is not numeric"
        elif confidence_ok:
            confidence_msg = f"Confidence {card.confidence*100:.1f}% meets threshold {threshold*100:.1f}%"
        else:
            confidence_msg = (
                f"Confidence {card.confidence*100:.1f}% is below risk tier '{card.risk_tier}' "
                f"threshold {threshold*100:.1f}%"
            )
        record(f"Confidence Threshold Check ({str(card.risk_tier).upper()})", confidence_ok, confidence_msg)

        # 3. Payload must be canonical structured data.
        payload_ok = isinstance(card.payload, dict)
        if payload_ok:
            try:
                canonical_payload_json(card.payload)
            except TypeError:
                payload_ok = False
        record(
            "Payload Integrity",
            payload_ok,
            "Payload is canonical structured data" if payload_ok else "Payload is not canonical JSON data",
        )

        # 4. Cryptographic integrity over the signed fields.
        sig_ok = self.verify_action_digest(card)
        record(
            "Cryptographic Integrity",
            sig_ok,
            "Valid HMAC-SHA256 signature" if sig_ok
            else "Cryptographic integrity check failed: invalid or missing HMAC-SHA256 signature",
        )

        if card.risk_tier == "critical":
            # 5. Dual custody: a pre-registered co-signer, distinct from the persona, whose
            #    signature verifies over this exact revision.
            has_cosigner = bool(card.cosigner_signature and card.cosigner_agent_id)
            cosign_ok = self.verify_cosigner_digest(card)
            if cosign_ok:
                custody_msg = f"Co-signed and cryptographically verified by {card.cosigner_agent_id}"
            elif has_cosigner:
                custody_msg = (
                    f"Co-signature from {card.cosigner_agent_id} failed verification "
                    "against its registered key or the separation-of-duties rule"
                )
            else:
                custody_msg = "Critical action requires dual-custody secondary agent co-signature"
            record("Dual-Custody Multi-Agent Consensus", cosign_ok, custody_msg)

            # 6. BFT quorum: mandatory. Missing ballots fail the check.
            ballots = card.metadata.get("ballots")
            bft_eval = self.bft_engine.evaluate_quorum(card, ballots)
            bft_ok = bool(bft_eval.get("approved", False))
            if bft_ok:
                bft_msg = (
                    f"BFT quorum passed ({bft_eval['valid_count']}/{bft_eval['quorum_threshold']} valid votes)"
                )
            elif not isinstance(ballots, (list, tuple)) or len(ballots) == 0:
                bft_msg = "No BFT ballots collected; critical action requires quorum consensus"
            else:
                bft_msg = (
                    f"BFT quorum failed ({bft_eval['valid_count']}/{bft_eval['quorum_threshold']} valid votes required)"
                )
            record("Multi-Agent BFT Quorum Consensus", bft_ok, bft_msg, details=bft_eval)

            # 7. Critical confidence gate
            critical_floor = self.RISK_CONFIDENCE_THRESHOLDS["critical"]
            crit_conf_ok = (
                _is_number(card.confidence) and math.isfinite(card.confidence)
                and card.confidence >= critical_floor
            )
            record(
                "Critical Confidence Safety Gate",
                crit_conf_ok,
                f"Confidence {card.confidence:.4f} meets >= {critical_floor:.2f} requirement" if crit_conf_ok
                else f"Confidence is below the required {critical_floor:.2f} for the critical tier",
            )

            # 8. Conformal singleton gate: mandatory. A missing certificate fails the check.
            conformal_info = card.metadata.get("conformal_set")
            if conformal_info is None:
                singleton_ok = False
                singleton_msg = "No conformal set certificate attached; critical actions require a singleton set"
            else:
                singleton_ok = isinstance(conformal_info, dict) and conformal_info.get("is_singleton") is True
                singleton_msg = (
                    "Conformal set is singleton (unambiguous decision)" if singleton_ok
                    else "Conformal set is non-singleton or empty"
                )
            record("Conformal Singleton Gate", singleton_ok, singleton_msg)

        card.verification_checks = checks
        return (not error_messages), error_messages

    def preflight_verification(
        self,
        card_id: str,
        granted_permissions: Optional[List[str]] = None,
        ballots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run preflight verification on a staged Action Card by card ID.

        A pass sets the card to approved and binds that approval to the card's current
        signature. A failure revokes any earlier approval.
        """
        card = self.staged_cards.get(card_id)
        if card is None:
            raise KeyError(f"ActionCard with id '{card_id}' not found in staged pool.")

        if ballots is not None:
            card.metadata["ballots"] = list(ballots) if isinstance(ballots, (list, tuple)) else ballots

        allowed, errors = self.verify_action(card, granted_permissions=granted_permissions)
        if allowed:
            card.status = "approved"
            card.metadata["preflight_verified_at"] = time.time()
            self._preflight_passed[card.id] = card.signature
        else:
            self._preflight_passed.pop(card.id, None)
            if card.status == "approved":
                card.status = "staged"

        dual_custody_required = card.risk_tier == "critical"
        bft_result = None
        if dual_custody_required:
            bft_result = self.bft_engine.evaluate_quorum(card, card.metadata.get("ballots"))

        return {
            "card_id": card.id,
            "allowed": allowed,
            "checks": card.verification_checks,
            "errors": errors,
            "risk_tier": card.risk_tier,
            "signature_valid": self.verify_action_digest(card),
            "dual_custody_required": dual_custody_required,
            "cosigner_verified": self.verify_cosigner_digest(card) if dual_custody_required else None,
            "bft_consensus": bft_result,
            "merkle_root": self.merkle_ledger.get_merkle_root(),
        }

    def collect_bft_ballot(
        self,
        card_id: str,
        voter_agent_id: str,
        vote: bool,
        secret_key: Union[bytes, str],
    ) -> Dict[str, Any]:
        """Collect and register a BFT ballot for a staged card.

        The voter must be registered with the BFT engine and must present the registered key.
        """
        card = self.staged_cards.get(card_id)
        if card is None:
            raise KeyError(f"ActionCard with id '{card_id}' not found in staged pool.")
        registered_key = self.bft_engine.registered_agents.get(voter_agent_id)
        if registered_key is None:
            raise ValueError(f"BFT voter {voter_agent_id!r} is not registered")
        if not hmac.compare_digest(registered_key, _key_bytes(secret_key)):
            raise ValueError(f"secret key does not match the key registered for {voter_agent_id!r}")
        ballot = self.bft_engine.collect_ballot(card, voter_agent_id, vote, registered_key)
        if not isinstance(card.metadata.get("ballots"), list):
            card.metadata["ballots"] = []
        card.metadata["ballots"].append(ballot)
        return ballot

    def _retire(self, card: ActionCard) -> None:
        """Move a card out of the staged pool into history and drop any approval."""
        self.staged_cards.pop(card.id, None)
        self._preflight_passed.pop(card.id, None)
        self.history.append(card)

    def approve_action(self, card_id: str, approver: str = "SystemSupervisor") -> Optional[ActionCard]:
        """Approve a staged Action Card.

        Returns None for an unknown id. Raises PermissionError when the card has not passed
        preflight verification for its current revision, or no longer passes it.
        """
        card = self.staged_cards.get(card_id)
        if card is None:
            return None
        recorded = self._preflight_passed.get(card_id)
        if recorded is None or recorded != card.signature:
            raise PermissionError(f"ActionCard '{card_id}' has not passed preflight verification")
        allowed, errors = self.verify_action(card)
        if not allowed:
            self._preflight_passed.pop(card_id, None)
            card.status = "staged"
            raise PermissionError(f"ActionCard '{card_id}' no longer passes preflight verification: {errors}")
        card.status = "approved"
        card.metadata["approver"] = approver
        card.metadata["approved_at"] = time.time()
        self.merkle_ledger.record_event(card, "approve")
        return card

    def reject_action(self, card_id: str, reason: str = "User declined") -> Optional[ActionCard]:
        """Reject a staged Action Card. Returns None for an unknown id."""
        card = self.staged_cards.get(card_id)
        if card is None:
            return None
        card.status = "rejected"
        card.metadata["rejection_reason"] = reason
        card.metadata["rejected_at"] = time.time()
        self._retire(card)
        self.merkle_ledger.record_event(card, "reject")
        return card

    def execute_action(
        self,
        card_id: str,
        executor_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        bypass_verification: bool = False,
    ) -> ActionCard:
        """Execute a staged Action Card.

        Preflight verification is re-run on every execution, so a card changed after approval
        is refused. ``bypass_verification=True`` skips that re-run and is recorded in the card's
        metadata. Executing an id that has already run raises RuntimeError.
        """
        if card_id in self._executed_ids:
            raise RuntimeError(f"ActionCard '{card_id}' has already been executed")
        card = self.staged_cards.get(card_id)
        if card is None:
            raise KeyError(f"ActionCard with id '{card_id}' not found in staged pool.")

        if bypass_verification:
            card.metadata["verification_bypassed"] = True
        else:
            verified, errors = self.verify_action(card)
            if not verified:
                card.status = "failed"
                card.execution_result = {"error": "Verification failed", "details": errors}
                self._retire(card)
                self._record_execution_event(card)
                return card

        # The card leaves the staged pool before the executor runs, so a re-entrant call
        # cannot execute it a second time.
        self.staged_cards.pop(card_id, None)
        self._preflight_passed.pop(card_id, None)
        try:
            if executor_fn is not None:
                result = executor_fn(card.payload)
            else:
                result = {"status": "success", "executed_action": card.title, "payload": card.payload}
            card.status = "executed"
            card.execution_result = result
            self._executed_ids.add(card_id)
        except Exception as exc:
            card.status = "failed"
            card.execution_result = {"error": str(exc)}
        finally:
            if card.status not in ("executed", "failed"):
                card.status = "failed"
                card.execution_result = {"error": "execution interrupted"}
            self.history.append(card)
            self._record_execution_event(card)
        return card

    def _record_execution_event(self, card: ActionCard) -> None:
        """Record the card's execute event once and keep its leaf position on the card."""
        entry = self.merkle_ledger.record_event(card, "execute")
        card.metadata["merkle_execution_leaf_index"] = entry["leaf_index"]
        card.metadata["merkle_execution_leaf_hash"] = entry["leaf_hash"]
        card.metadata["merkle_execution_root"] = entry["merkle_root"]

    def get_staged_cards(self) -> List[ActionCard]:
        """Return all currently staged cards."""
        return list(self.staged_cards.values())

    def get_history(self) -> List[ActionCard]:
        """Return history of all processed cards."""
        return list(self.history)
