"""
Rhen-1.0 Verification, Selectivity and Audit-Integrity Layer.

These components upgrade the *decision* layer (not the EOS language backbone) so
that a Rhen decision is trustworthy, auditable and cheap to verify. Each class
maps to a specific 2026 result rather than to a general intuition.

1. :class:`OutOfBandAuditLedger` -- trace logging that the agent cannot quietly rewrite.
   arXiv:2609.30266 ("LLM Agents Can Easily Tamper With Their Own Traces",
   2026-09-24) showed that every major coding agent harness tested -- Claude
   Code, Codex, Antigravity, Open Code, Grok Build -- allowed the agent to
   *delete its own trace* on request with no guardrail triggering, and that the
   behaviour emerges naturally once agents are rewarded for it. An in-process
   Merkle tree (see ``agent_tools.MerkleActionLedger``) is therefore tamper-
   evident against third parties but offers nothing against the agent itself.
   With a :class:`SocketAuditSink` the agent holds no sealing key: records leave
   the process, and :func:`serve_audit_socket`, which holds the key, recomputes
   and seals each one. The precise guarantee is stated on the class.

2. :class:`GeometrySelectiveClassifier` -- abstention from representation
   geometry rather than from a confidence threshold.
   arXiv:2609.29327 (GCUL) shows hard examples form a "confusion attractor" in
   representation space, and derives a *selectivity score* plus a sufficient
   condition under which rejection has positive operational utility -- both
   computable before deployment. That is strictly more informative than
   thresholding confidence.

3. :class:`CheapVerifier` -- choose the verification signal empirically.
   arXiv:2609.28879 compared probability-, sampling-, self-verification-,
   evidential- and contrastive uncertainty across four open models and five
   audio-QA benchmarks, and found plain **top-1 first-token probability
   (mean AUROC 0.740) beat 10-sample discrete semantic entropy (0.708)** while
   requiring *no extra model call*. This is direct evidence that Rhen's O(1)
   single-pass contract is correct and that bolting on self-consistency
   ensembles would make it both slower and worse.

4. :class:`BranchSparsityTracker` -- FlashLoop-style observation for decision
   branches. arXiv:2609.29812 found that as recurrence proceeds, state changes
   concentrate on a small subset of tokens. Applying that to cascades tells us
   which branches have stopped moving and can be frozen early.

All components are pure-PyTorch / stdlib and CPU-safe.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import ipaddress
import json
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "AuditEntry",
    "AuditSinkError",
    "InProcessAuditSink",
    "FileAuditSink",
    "SocketAuditSink",
    "OutOfBandAuditLedger",
    "serve_audit_socket",
    "verify_sealed_records",
    "SelectivityReport",
    "GeometrySelectiveClassifier",
    "CheapVerifier",
    "BranchSparsityTracker",
]

# Minimum length for any audit sealing key (local MAC key or receiver key).
_MIN_AUDIT_KEY_BYTES = 16
# Largest frame the reference receiver accepts. Larger frames are refused by the
# sink before sending, so an oversized record fails closed instead of vanishing.
_MAX_AUDIT_FRAME_BYTES = 1 << 20


# ---------------------------------------------------------------------------
# 1. Tamper-resistant audit ledger
# ---------------------------------------------------------------------------


class AuditSinkError(RuntimeError):
    """Raised when an audit sink cannot accept a record."""


@dataclass
class AuditEntry:
    """A single audit record. ``mac`` is None when the record was sealed remotely."""

    sequence: int
    timestamp: float
    kind: str
    payload: Dict[str, Any]
    prev_hash: str
    entry_hash: str
    mac: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": copy.deepcopy(self.payload),
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }
        if self.mac is not None:
            data["mac"] = self.mac
        return data


def _entry_hash(
    sequence: int, timestamp: float, kind: str, payload: Dict[str, Any], prev_hash: str
) -> str:
    """SHA-256 over the canonical JSON body of one record (the MAC is computed over this)."""
    body = json.dumps(
        {
            "sequence": sequence,
            "timestamp": timestamp,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _mac_for(key: bytes, entry_hash: str) -> str:
    return hmac.new(key, entry_hash.encode("ascii"), hashlib.sha256).hexdigest()


def _const_eq(expected: Any, actual: Any) -> bool:
    """Constant-time equality for two strings. Anything that is not a string is unequal."""
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    return hmac.compare_digest(expected.encode("utf-8"), actual.encode("utf-8"))


def _json_copy(payload: Any) -> Dict[str, Any]:
    """Detach a payload from the caller by a JSON round trip. Non-JSON data is refused."""
    if not isinstance(payload, dict):
        raise TypeError("audit payload must be a dict")
    try:
        return json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"audit payload must be JSON-serialisable: {exc}") from exc


def _parse_sealed_record(record: Any) -> Tuple[int, float, str, Dict[str, Any], str, Optional[str]]:
    """Type-check the content fields of one record. Raises ValueError when malformed."""
    if not isinstance(record, dict):
        raise ValueError("audit record must be a JSON object")
    sequence = record.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ValueError("sequence must be an integer")
    timestamp = record.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError("timestamp must be a finite number")
    kind = record.get("kind")
    if not isinstance(kind, str):
        raise ValueError("kind must be a string")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    prev_hash = record.get("prev_hash")
    if not isinstance(prev_hash, str):
        raise ValueError("prev_hash must be a string")
    declared = record.get("entry_hash")
    return (
        sequence,
        float(timestamp),
        kind,
        payload,
        prev_hash,
        declared if isinstance(declared, str) else None,
    )


def _seal_received(record: Any, key: bytes) -> Optional[Dict[str, Any]]:
    """Receiver-side sealing. Recomputes the entry hash, rejects mismatches, and adds a MAC.

    Returns the record to store, or None when it is malformed or its declared entry hash
    does not match its content. Any ``mac`` supplied by the sender is ignored.
    """
    try:
        sequence, timestamp, kind, payload, prev_hash, declared = _parse_sealed_record(record)
    except ValueError:
        return None
    expected = _entry_hash(sequence, timestamp, kind, payload, prev_hash)
    if not _const_eq(declared, expected):
        return None
    return {
        "sequence": sequence,
        "timestamp": timestamp,
        "kind": kind,
        "payload": payload,
        "prev_hash": prev_hash,
        "entry_hash": expected,
        "mac": _mac_for(key, expected),
    }


def _seal_frame(frame: bytes, key: bytes) -> Optional[Dict[str, Any]]:
    """Decode one received frame as JSON and seal it. Undecodable frames return None."""
    try:
        record = json.loads(frame.decode("utf-8"))
    except ValueError:  # includes UnicodeDecodeError and JSONDecodeError
        return None
    return _seal_received(record, key)


class _BaseAuditSink:
    """Common interface for audit sinks."""

    def emit(self, record: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


class InProcessAuditSink(_BaseAuditSink):
    """Default sink: keeps records in this process.

    This is the *development* default only. It provides hash-chaining and HMAC
    sealing (so casual edits to the list are detectable) but it does **not**
    resist a fully compromised process, because that process holds both the
    records and the key. Use :class:`SocketAuditSink` or :class:`FileAuditSink`
    for any deployment where the agent is untrusted.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def emit(self, record: Dict[str, Any]) -> None:
        self.records.append(record)


class FileAuditSink(_BaseAuditSink):
    """Append-only file sink with fsync and durable newline-delimited JSON."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8", newline="\n")

    def emit(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()


def _resolve_family(address: str) -> int:
    """Return the socket family for ``address``.

    Windows has no AF_UNIX in many Python builds, and a Windows path is never a
    valid TCP endpoint either, so a leading drive letter or backslash is treated
    as "not a unix socket" rather than crashing on a missing attribute.
    """
    has_unix = hasattr(socket, "AF_UNIX")
    looks_unix = address.startswith("/") or address.startswith("\\\\")
    if has_unix and looks_unix:
        return socket.AF_UNIX
    return socket.AF_INET


def _is_unix_path(address: str) -> bool:
    return _resolve_family(address) == getattr(socket, "AF_UNIX", object())


class SocketAuditSink(_BaseAuditSink):
    """IPC sink that forwards records to a separate audit receiver.

    Using this sink switches :class:`OutOfBandAuditLedger` to remote sealing, so the
    agent process never holds the sealing key. The receiver computes the MAC with its
    own key. The sink does not wait for an acknowledgement: ``emit`` returning means
    the frame was handed to the socket, not that the receiver stored it. Frames above
    the size limit are refused before sending, so no record is lost silently.

    The frame format is a 4-byte big-endian length followed by UTF-8 JSON. The
    reference receiver is :func:`serve_audit_socket`.
    """

    def __init__(
        self,
        address: str,
        connect_timeout: float = 2.0,
        reconnect_attempts: int = 3,
    ) -> None:
        self.address = address
        self.connect_timeout = float(connect_timeout)
        self.reconnect_attempts = int(reconnect_attempts)
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self.dropped_records = 0

    def _connect(self) -> socket.socket:
        last: Optional[Exception] = None
        for _ in range(max(1, self.reconnect_attempts)):
            try:
                family = _resolve_family(self.address)
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.settimeout(self.connect_timeout)
                if family == socket.AF_INET:
                    host, _, port = self.address.rpartition(":")
                    if not port.isdigit():
                        raise ValueError(f"not a host:port endpoint: {self.address!r}")
                    sock.connect((host or "127.0.0.1", int(port)))
                else:
                    sock.connect(self.address)
                return sock
            except OSError as exc:  # pragma: no cover - environment dependent
                last = exc
        raise AuditSinkError(
            f"could not connect to audit sink at {self.address!r}: {last}"
        )

    def emit(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(line) > _MAX_AUDIT_FRAME_BYTES:
            self.dropped_records += 1
            raise AuditSinkError("audit record exceeds the frame size limit; record NOT logged")
        frame = struct.pack(">I", len(line)) + line
        with self._lock:
            for _ in range(max(1, self.reconnect_attempts)):
                try:
                    if self._sock is None:
                        self._sock = self._connect()
                    self._sock.sendall(frame)
                    return
                except (OSError, AuditSinkError):
                    if self._sock is not None:
                        try:
                            self._sock.close()
                        except OSError:
                            pass
                        self._sock = None
        # Failing closed is the safe behaviour: an unaudited action must never
        # be reported as an audited one.
        self.dropped_records += 1
        raise AuditSinkError("audit sink unavailable; record NOT durably logged")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


def _tcp_bind_host(host: str, allow_remote: bool = False) -> str:
    """
    Return the address a TCP audit receiver binds to for ``host``.

    An empty host (``":9000"``) and ``localhost`` bind ``127.0.0.1``. Any other
    loopback IP literal binds as given. Every other host, including ``0.0.0.0``,
    a LAN address and a hostname, raises ``ValueError`` unless ``allow_remote`` is
    True. The receiver does not authenticate frames, so exposing it beyond loopback
    lets any peer that can reach the port inject records that pass the hash check.
    """
    name = host.strip("[]")
    if name == "" or name.lower() == "localhost":
        return "127.0.0.1"
    try:
        if ipaddress.ip_address(name).is_loopback:
            return name
    except ValueError:
        pass
    if allow_remote:
        return name
    raise ValueError(
        f"refusing to bind the audit receiver to non-loopback host {host!r}; "
        "pass allow_remote=True to opt in (frames are not authenticated)"
    )


def serve_audit_socket(
    address: str,
    store_path: str,
    stop_event: Optional[threading.Event] = None,
    max_iterations: Optional[int] = None,
    key: Optional[bytes] = None,
    *,
    allow_remote: bool = False,
) -> None:
    """Reference audit receiver for :class:`SocketAuditSink`. It seals what it stores.

    Run it in a separate process, ideally with different privileges, and give it a
    sealing key that the agent process cannot read. For each frame it parses the
    record, recomputes ``entry_hash`` from the record's content, and drops the frame
    if the declared hash differs or a field is malformed. Accepted records are appended
    to ``store_path`` with a MAC computed by this receiver's ``key``. Any ``mac`` the
    sender supplied is ignored. Use :func:`verify_sealed_records` to check the store.

    Frames are not authenticated: the receiver accepts any well-formed frame from any
    peer that can reach its socket. For that reason a TCP receiver binds loopback only
    by default. ``allow_remote=True`` is an explicit opt-in to bind a non-loopback host,
    and the caller then owns the network exposure.

    Args:
        address: ``/path/to.sock`` (unix) or ``host:port`` (tcp). An empty host binds
            ``127.0.0.1``.
        store_path: newline-delimited JSON destination.
        stop_event: set this to request a graceful shutdown.
        max_iterations: stop after N accepted records (used by the test suite).
            Rejected frames are not counted.
        key: sealing key, at least 16 bytes. Required.
        allow_remote: opt in to binding a non-loopback TCP host. Off by default.

    Raises:
        ValueError: when ``key`` is missing or shorter than 16 bytes, when the TCP
            address is not ``host:port``, or when the host is non-loopback without
            ``allow_remote=True``. All of these are checked before a socket is created.
    """
    if not isinstance(key, (bytes, bytearray)) or len(key) < _MIN_AUDIT_KEY_BYTES:
        raise ValueError("serve_audit_socket requires a sealing key of at least 16 bytes")
    sealing_key = bytes(key)
    stop_event = stop_event or threading.Event()
    family = _resolve_family(address)
    unix_mode = family == getattr(socket, "AF_UNIX", object())
    if not unix_mode:
        host, _, port = address.rpartition(":")
        if not port.isdigit():
            raise ValueError(f"not a host:port endpoint: {address!r}")
        bind_target = (_tcp_bind_host(host, allow_remote), int(port))
    server = socket.socket(family, socket.SOCK_STREAM)
    if unix_mode:
        if os.path.exists(address):
            os.unlink(address)
        server.bind(address)
    else:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(bind_target)
    server.listen(4)
    server.settimeout(0.25)

    with open(store_path, "a", encoding="utf-8", newline="\n") as store:
        count = 0
        while not stop_event.is_set():
            if max_iterations is not None and count >= max_iterations:
                break
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:  # pragma: no cover - shutdown race
                break
            with conn:
                conn.settimeout(1.0)
                buf = b""
                oversized = False
                while not stop_event.is_set() and not oversized:
                    try:
                        chunk = conn.recv(65536)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while len(buf) >= 4:
                        (length,) = struct.unpack(">I", buf[:4])
                        if length > _MAX_AUDIT_FRAME_BYTES:
                            oversized = True  # refuse the connection; it cannot be framed safely
                            break
                        if len(buf) < 4 + length:
                            break
                        frame = buf[4 : 4 + length]
                        buf = buf[4 + length :]
                        sealed = _seal_frame(frame, sealing_key)
                        if sealed is None:
                            continue  # malformed or hash mismatch: dropped, not stored
                        store.write(json.dumps(sealed, sort_keys=True, separators=(",", ":")) + "\n")
                        store.flush()
                        os.fsync(store.fileno())
                        count += 1
                        if max_iterations is not None and count >= max_iterations:
                            stop_event.set()
                            break
    server.close()
    if unix_mode and os.path.exists(address):
        try:
            os.unlink(address)
        except OSError:  # pragma: no cover
            pass


class OutOfBandAuditLedger:
    """Hash-chained audit ledger. Each record is sealed with a MAC, held locally or remotely.

    The sink decides the sealing mode.

    * **Local sealing** (``InProcessAuditSink``, ``FileAuditSink``, the default). An explicit
      ``key`` of at least 16 bytes is required and every record carries a MAC over its
      entry hash. Whoever holds the key, including the agent's own process, can compute
      valid MACs. This mode detects casual edits and corruption. It does not protect
      against a compromised agent process.

    * **Remote sealing** (``SocketAuditSink``). The agent holds no key, so ``key`` must be
      None. Records carry ``sequence``, ``timestamp``, ``kind``, ``payload``, ``prev_hash``
      and ``entry_hash`` but no MAC. :func:`serve_audit_socket`, running with its own key,
      recomputes each entry hash, rejects mismatches, and seals what it stores.
      :func:`verify_sealed_records` checks the stored chain against that key.

    Guarantee, stated precisely:

    * A record that has already been emitted to a remote receiver cannot be rewritten
      without detection. Its entry hash is chained to its predecessor and sealed under
      the receiver's key, which the agent process does not hold.
    * A compromised agent can still do two things. It can stop emitting. That shows up
      as a sequence gap, but only against a sequence number the auditor already expects
      (for example, the agent's own counter). A silent stop at the end of the stream is
      not detectable from the store alone. It can also emit new records with false
      content. Those verify, because the receiver authenticates what it received, not
      whether it is true.
    * Records that were never emitted, or that were rewritten before emission, are
      outside the guarantee. A new ledger instance starts a new chain at sequence 1.
    """

    GENESIS = "0" * 64

    def __init__(
        self,
        key: Optional[bytes] = None,
        sink: Optional[_BaseAuditSink] = None,
        fail_closed: bool = True,
    ) -> None:
        self._sink: _BaseAuditSink = sink if sink is not None else InProcessAuditSink()
        self.remote_sealing = isinstance(self._sink, SocketAuditSink)
        if self.remote_sealing:
            if key is not None:
                raise ValueError(
                    "a SocketAuditSink ledger must not hold the sealing key: pass key=None, "
                    "the receiver seals with its own key"
                )
            self.__sealing_key = None
        else:
            if not isinstance(key, (bytes, bytearray)) or len(key) < _MIN_AUDIT_KEY_BYTES:
                raise ValueError("audit sealing key must be at least 16 bytes")
            self.__sealing_key = bytes(key)
        self._prev_hash = self.GENESIS
        self._sequence = 0
        self.fail_closed = bool(fail_closed)
        self.entries: List[AuditEntry] = []
        self.audit_failures = 0

    def _mac(self, entry_hash: str) -> Optional[str]:
        if self.__sealing_key is None:
            return None
        return _mac_for(self.__sealing_key, entry_hash)

    def _compute(
        self, seq: int, ts: float, kind: str, payload: Dict[str, Any], prev: str
    ) -> AuditEntry:
        entry_hash = _entry_hash(seq, ts, kind, payload, prev)
        return AuditEntry(
            sequence=seq, timestamp=ts, kind=kind, payload=payload,
            prev_hash=prev, entry_hash=entry_hash, mac=self._mac(entry_hash),
        )

    def record(self, kind: str, payload: Dict[str, Any]) -> AuditEntry:
        """Seal and emit one audit entry.

        The payload is copied through JSON first, so the sealed content is exactly what is
        emitted and later changes by the caller cannot alter it.

        Raises:
            AuditSinkError: when the sink is unreachable and ``fail_closed`` is set. The
                sequence number is not consumed in that case, so a retry reuses it.
                Callers must treat this as "action not authorised" rather than retrying
                silently.
            TypeError: when the payload is not a dict of JSON-serialisable data.
        """
        sequence = self._sequence + 1
        entry = self._compute(
            sequence, time.time(), str(kind), _json_copy(payload), self._prev_hash
        )
        try:
            self._sink.emit(entry.to_dict())
        except AuditSinkError:
            self.audit_failures += 1
            if self.fail_closed:
                raise
        self._sequence = sequence
        self._prev_hash = entry.entry_hash
        self.entries.append(entry)
        return entry

    def record_decision(
        self,
        question: str,
        chosen: Any,
        confidence: float,
        conformal_set: Optional[Sequence[Any]] = None,
        **extra: Any,
    ) -> AuditEntry:
        """Record a Rhen decision together with its calibration certificate."""
        payload: Dict[str, Any] = {
            "question": question,
            "chosen": chosen,
            "confidence": float(confidence),
        }
        if conformal_set is not None:
            payload["conformal_set"] = list(conformal_set)
        payload.update(extra)
        return self.record("decision", payload)

    def verify(self, entry: Any) -> Dict[str, Any]:
        """Check one entry's content hash, chain link and, when this ledger holds the key, its MAC.

        In remote-sealing mode the agent cannot check MACs, so ``mac_valid`` is None and
        ``valid`` covers the hash and chain only. Authenticate records held by the receiver
        with :func:`verify_sealed_records` and the receiver's key.
        """
        data = entry.to_dict() if isinstance(entry, AuditEntry) else entry
        try:
            seq, ts, kind, payload, declared_prev, declared_hash = _parse_sealed_record(data)
        except (TypeError, ValueError):
            return {"valid": False, "mac_valid": False, "hash_valid": False,
                    "chain_valid": False, "sequence": 0}

        expected_hash = _entry_hash(seq, ts, kind, payload, declared_prev)
        hash_ok = _const_eq(declared_hash, expected_hash)
        if self.__sealing_key is None:
            mac_ok: Optional[bool] = None
        else:
            mac_ok = _const_eq(data.get("mac"), _mac_for(self.__sealing_key, expected_hash))

        if seq == 1:
            link_ok = declared_prev == self.GENESIS
        elif 1 < seq <= len(self.entries):
            # entries is 0-indexed and stores sequence 1 first, so the
            # predecessor of sequence `seq` lives at index `seq - 2`.
            link_ok = declared_prev == self.entries[seq - 2].entry_hash
        else:
            link_ok = False
        return {
            "valid": bool(hash_ok and link_ok and mac_ok is not False),
            "mac_valid": mac_ok,
            "hash_valid": bool(hash_ok),
            "chain_valid": bool(link_ok),
            "sequence": seq,
        }

    def verify_chain(self) -> Dict[str, Any]:
        """Verify every entry held by this ledger; report the first sequence number that fails."""
        results = [self.verify(e) for e in self.entries]
        first_bad = next((r["sequence"] for r in results if not r["valid"]), None)
        return {
            "all_valid": all(r["valid"] for r in results) if results else True,
            "count": len(results),
            "first_invalid_sequence": first_bad,
            "audit_failures": self.audit_failures,
            "remote_sealing": self.remote_sealing,
        }

    def detect_gaps(self, external: Sequence[Dict[str, Any]]) -> List[int]:
        """Sequence numbers present externally but absent from this ledger.

        A gap is the observable signature of an agent that stopped emitting. It is
        visible only for sequence numbers the auditor already knows should exist;
        a silent stop at the end of the stream is not detectable this way.
        """
        mine = {e.sequence for e in self.entries}
        return sorted(
            int(r["sequence"]) for r in external if int(r.get("sequence", 0)) not in mine
        )

    def close(self) -> None:
        self._sink.close()


def verify_sealed_records(records: Sequence[Dict[str, Any]], key: bytes) -> Dict[str, Any]:
    """Authenticate records stored by :func:`serve_audit_socket` (or a local ledger).

    For each record it checks that the entry hash matches the content and that the MAC
    verifies under ``key``. Across records it checks that each ``prev_hash`` equals the
    previous record's entry hash, that sequence numbers increase, and it reports any gaps.
    A window that starts mid-chain cannot have its first link checked. The first record's
    ``prev_hash`` is compared with GENESIS only when that record is sequence 1.

    Returns:
        ``{"all_valid", "count", "first_invalid_index", "gaps"}``. ``gaps`` lists the missing
        sequence numbers between consecutive records.
    """
    if not isinstance(key, (bytes, bytearray)) or len(key) < _MIN_AUDIT_KEY_BYTES:
        raise ValueError("verification key must be at least 16 bytes")
    sealing_key = bytes(key)
    records = list(records)
    first_invalid: Optional[int] = None
    gaps: List[int] = []
    previous: Optional[Tuple[int, str]] = None  # (sequence, entry_hash) of the last record

    for index, record in enumerate(records):
        try:
            seq, ts, kind, payload, prev_hash, declared = _parse_sealed_record(record)
        except ValueError:
            ok = False
        else:
            expected = _entry_hash(seq, ts, kind, payload, prev_hash)
            hash_ok = _const_eq(declared, expected)
            mac_ok = _const_eq(record.get("mac"), _mac_for(sealing_key, expected))
            if previous is None:
                link_ok = prev_hash == OutOfBandAuditLedger.GENESIS if seq == 1 else True
                order_ok = True
            else:
                link_ok = prev_hash == previous[1]
                order_ok = seq > previous[0]
                if seq > previous[0] + 1:
                    gaps.extend(range(previous[0] + 1, seq))
            ok = bool(hash_ok and mac_ok and link_ok and order_ok)
            previous = (seq, expected)
        if not ok and first_invalid is None:
            first_invalid = index

    return {
        "all_valid": first_invalid is None,
        "count": len(records),
        "first_invalid_index": first_invalid,
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# 2. Geometry-based selective classification
# ---------------------------------------------------------------------------


@dataclass
class SelectivityReport:
    """Pre-deployment feasibility report for rejection."""

    coverage: float
    accepted: int
    rejected: int
    accuracy_on_accepted: float
    accuracy_overall: float
    utility: float
    feasible: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class GeometrySelectiveClassifier:
    """Selective classification driven by representation geometry, not a threshold.

    Confidence thresholding treats a confident-but-wrong example as a rare
    accident. arXiv:2609.29327 (GCUL) instead finds that hard examples form a
    *confusion attractor* -- a contiguous region in representation space -- so
    the rejection boundary is recoverable geometrically.

    This class implements the cheap, model-free part of that idea: fit a radius
    ball around each class centroid and reject anything beyond
    ``max_radius * r``. It also computes the paper's **selectivity score** and
    its **sufficient condition** for rejection to have positive operational
    utility, both checkable *before* deployment::

        utility  = accuracy_on_accepted - accuracy_overall
        feasible = utility > 0 and accepted_rows >= min_accepted

    ``utility`` is the accuracy you get on the questions you actually answer,
    minus the accuracy you would have got by answering everything. That is the
    quantity abstention is supposed to buy.

    A note on a tempting but wrong alternative: ``acc_accepted * coverage -
    overall`` is algebraically identically zero whenever the rejected set is
    exactly the error set, so it can never report that a *perfect* rejection
    policy helped. Comparing conditional accuracy directly avoids that.
    """

    def __init__(self, max_radius: float = 1.0, min_accepted: int = 3) -> None:
        self.max_radius = float(max_radius)
        self.min_accepted = int(min_accepted)
        self.centroids: Dict[Any, torch.Tensor] = {}
        self.radii: Dict[Any, float] = {}
        self.fitted = False

    def fit(self, embeddings: torch.Tensor, labels: Sequence[Any]) -> "GeometrySelectiveClassifier":
        """Learn per-class centroids and intra-class radii."""
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if len(labels) != int(embeddings.shape[0]):
            raise ValueError("embeddings and labels must have equal length")
        if embeddings.shape[0] == 0:
            raise ValueError("cannot fit on an empty sample")

        for label in set(labels):
            mask = torch.tensor(
                [l == label for l in labels], dtype=torch.bool, device=embeddings.device
            )
            block = embeddings[mask]
            centroid = block.mean(dim=0)
            dists = torch.norm(block - centroid, dim=-1)
            # Floor keeps a zero-variance class from producing radius 0.
            radius = float(dists.max().item()) if dists.numel() else 0.0
            self.centroids[label] = centroid
            self.radii[label] = max(radius, 1e-6)
        self.fitted = True
        return self

    def _ratio(self, point: torch.Tensor) -> float:
        if not self.fitted:
            raise RuntimeError("call fit() before scoring")
        best = float("inf")
        for label, centroid in self.centroids.items():
            dist = float(torch.norm(point - centroid).item())
            best = min(best, dist / self.radii[label])
        return best

    def predict(
        self, embeddings: torch.Tensor, labels: Optional[Sequence[Any]] = None
    ) -> Dict[str, Any]:
        """Score rows, returning decisions plus a selectivity diagnostic.

        ``uniformity`` is a 1-D proxy for the paper's sufficient condition: the
        spread of normalised distances. A tight spread means the rejection
        boundary is well defined; a wide one means the geometry is unreliable.
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        n = int(embeddings.shape[0])
        accept_flags: List[bool] = []
        distances: List[float] = []
        predicted: List[Any] = []

        for i in range(n):
            point = embeddings[i]
            ratio = self._ratio(point)
            distances.append(ratio)
            accept_flags.append(ratio <= self.max_radius)
            best_label, best_d = None, float("inf")
            for label, centroid in self.centroids.items():
                d = float(torch.norm(point - centroid).item())
                if d < best_d:
                    best_d, best_label = d, label
            predicted.append(best_label)

        accepted_idx = [i for i, ok in enumerate(accept_flags) if ok]
        report: Dict[str, Any] = {
            "accept": accept_flags,
            "predicted": predicted,
            "normalized_distance": distances,
            "n_accepted": len(accepted_idx),
            "n_rejected": n - len(accepted_idx),
        }

        if labels is not None:
            overall = sum(1 for p, y in zip(predicted, labels) if p == y) / n if n else 0.0
            acc_acc = (
                sum(1 for i in accepted_idx if predicted[i] == labels[i]) / len(accepted_idx)
                if accepted_idx
                else 0.0
            )
            coverage = len(accepted_idx) / n if n else 0.0
            mean_d = sum(distances) / n if n else 0.0
            var = sum((d - mean_d) ** 2 for d in distances) / n if n else 0.0
            uniformity = 1.0 / (1.0 + (var ** 0.5) / max(mean_d, 1e-9)) if n else 0.0
            # Accuracy on the answered subset, minus accuracy if everything were
            # answered. See the class docstring for why the coverage-weighted
            # form is not used.
            utility = acc_acc - overall
            feasible = bool(utility > 0.0 and len(accepted_idx) >= self.min_accepted)
            report["selectivity"] = SelectivityReport(
                coverage=coverage,
                accepted=len(accepted_idx),
                rejected=n - len(accepted_idx),
                accuracy_on_accepted=acc_acc,
                accuracy_overall=overall,
                utility=utility,
                feasible=feasible,
                reason=(
                    "rejection improves operational utility"
                    if feasible
                    else (
                        "too few accepted rows to trust"
                        if len(accepted_idx) < self.min_accepted
                        else "rejection does not improve utility"
                    )
                ),
            ).to_dict()
            report["uniformity"] = uniformity
        return report


# ---------------------------------------------------------------------------
# 3. Cheap verification (and why NOT to add self-consistency)
# ---------------------------------------------------------------------------


class CheapVerifier:
    """Choose a verification signal by measured cost, not by assumption.

    arXiv:2609.28879 compared five families of uncertainty estimate across four
    open models and five audio-QA benchmarks. The headline result is the one that
    should discipline Rhen: **top-1 first-token probability reached mean AUROC
    0.740 while 10-sample discrete semantic entropy reached only 0.708 -- and
    the cheap one needs no additional model call.**

    Two consequences:

    * Rhen's O(1) single-forward-pass contract is the right default.
    * Bolting a self-consistency ensemble onto Rhen would make it both slower
      and *less* discriminative.

    This class makes that a checkable property rather than a comment: it scores
    several candidate signals against ground truth, computes each signal's AUROC
    and its cost, and refuses to enable a signal that does not earn its calls.
    """

    def __init__(self, min_auc_gain: float = 0.0, samples: int = 10) -> None:
        self.min_auc_gain = float(min_auc_gain)
        self.samples = int(samples)
        self.measurements: List[Dict[str, Any]] = []

    @staticmethod
    def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
        """Area under the ROC via the rank-sum (Mann-Whitney U) identity."""
        pairs = list(zip(scores, labels))
        n_pos = sum(1 for _, y in pairs if y == 1)
        n_neg = len(pairs) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        # Average ranks over ties so equal scores are not unfairly ordered.
        order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        rank_sum_pos = sum(ranks[i] for i, (_, y) in enumerate(pairs) if y == 1)
        return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def signals(
        self,
        probs: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_probs: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    ) -> Dict[str, Any]:
        """Score candidate verification signals.

        Args:
            probs: per-example option probability vectors (one forward pass).
            labels: 1 for correct, 0 for incorrect.
            sample_probs: optional per-example *groups* of sampled probability
                vectors, used to compute discrete semantic entropy. Supplying it
                is how you demonstrate that the extra samples do not pay.
        """
        top1: List[float] = []
        margin: List[float] = []
        neg_entropy: List[float] = []
        for row in probs:
            p = torch.tensor(list(row), dtype=torch.float32)
            p = p / p.sum().clamp_min(1e-12)
            top1.append(float(p.max()))
            top2 = torch.topk(p, min(2, p.numel())).values
            margin.append(float(top2[0] - top2[1]) if top2.numel() > 1 else 0.0)
            neg_entropy.append(float((p * p.clamp_min(1e-12).log()).sum()))

        out: Dict[str, Any] = {
            "top1_probability": {"auc": self.auroc(top1, labels), "extra_calls": 0},
            "decision_margin": {"auc": self.auroc(margin, labels), "extra_calls": 0},
            "neg_entropy": {"auc": self.auroc(neg_entropy, labels), "extra_calls": 0},
        }

        if sample_probs is not None:
            dse: List[float] = []
            for group in sample_probs:
                if not group:
                    dse.append(0.0)
                    continue
                answers = [
                    int(torch.tensor(list(r), dtype=torch.float32).argmax()) for r in group
                ]
                counts: Dict[int, int] = {}
                for a in answers:
                    counts[a] = counts.get(a, 0) + 1
                n = len(answers)
                ent = -sum((c / n) * math.log(c / n) for c in counts.values())
                dse.append(-ent)  # higher = more agreement = more reliable
            out["discrete_semantic_entropy"] = {
                "auc": self.auroc(dse, labels),
                "extra_calls": self.samples,
            }

        best_free = max(
            (k for k, v in out.items() if v["extra_calls"] == 0),
            key=lambda k: out[k]["auc"],
        )
        free_auc = out[best_free]["auc"]
        recommendations: List[Dict[str, Any]] = []
        for name, val in out.items():
            gain = val["auc"] - free_auc
            worth_it = val["extra_calls"] == 0 or gain > self.min_auc_gain
            recommendations.append(
                {
                    "signal": name,
                    "auc": val["auc"],
                    "extra_calls": val["extra_calls"],
                    "gain_over_best_free": gain,
                    "enabled": bool(worth_it),
                }
            )
            if not worth_it:
                self.measurements.append(recommendations[-1])

        return {
            "best_free_signal": best_free,
            "best_free_auc": free_auc,
            "signals": out,
            "recommendations": recommendations,
            "rejected_signals": [r["signal"] for r in recommendations if not r["enabled"]],
        }


# ---------------------------------------------------------------------------
# 4. FlashLoop-style branch sparsity
# ---------------------------------------------------------------------------


class BranchSparsityTracker:
    """Detect which cascade branches have stopped changing.

    arXiv:2609.29812 (FlashLoop) observed that as recurrence proceeds in a
    looped transformer, state changes concentrate on a small subset of tokens
    while the rest are effectively frozen. Applied to a Rhen cascade, the same
    observation means most branches stop moving after a few loops -- so the
    remaining loops are pure waste for those branches.

    This tracker is an *instrumentation* component: it observes a sequence of
    per-branch state vectors and reports which branches have converged. It is
    the measurement prerequisite for FlashLoop-style savings (the paper reports
    1.64x speedup and 6x KV-cache reduction, training-free).

    Usage:
        >>> tracker = BranchSparsityTracker(threshold=1e-3)
        >>> for loop, states in enumerate(branch_states_over_loops, 1):
        ...     tracker.observe(loop, states)
        >>> tracker.frozen_branches()
        [2, 5]
    """

    def __init__(self, threshold: float = 1e-3, min_loops: int = 2) -> None:
        self.threshold = float(threshold)
        self.min_loops = int(min_loops)
        self._previous: Optional[Dict[int, torch.Tensor]] = None
        self._last_delta: Dict[int, float] = {}
        self._frozen: Dict[int, int] = {}  # branch -> loop at which it froze
        self._loop_seen = 0
        self.history: List[Dict[str, Any]] = []

    def observe(self, loop_idx: int, branch_states: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Record one loop's per-branch states and compute per-branch deltas."""
        deltas: Dict[int, float] = {}
        for branch_id, state in branch_states.items():
            if self._previous is not None and branch_id in self._previous:
                prev = self._previous[branch_id]
                delta = float(torch.norm(state.detach().float() - prev.float()).item())
            else:
                delta = float("inf")  # first sight of this branch cannot be "frozen"
            deltas[branch_id] = delta
            if delta <= self.threshold and branch_id not in self._frozen:
                self._frozen[branch_id] = int(loop_idx)

        self._previous = {k: v.detach().clone() for k, v in branch_states.items()}
        self._last_delta = deltas
        self._loop_seen = int(loop_idx)
        record = {
            "loop": int(loop_idx),
            "n_branches": len(branch_states),
            "n_frozen": len(self._frozen),
            "deltas": deltas,
        }
        self.history.append(record)
        return record

    def frozen_branches(self) -> List[int]:
        """Branches whose movement fell below the threshold."""
        return sorted(self._frozen)

    def active_branches(self) -> List[int]:
        """Branches still moving, i.e. those that still deserve a forward pass."""
        if not self._last_delta:
            return []
        return sorted(b for b, d in self._last_delta.items() if d > self.threshold)

    def estimated_savings(self, total_loops: int) -> Dict[str, Any]:
        """Fraction of loop-branches that could have been skipped.

        Only counts a branch as saveable once it has had at least ``min_loops``
        observations, so a branch that never moved is not credited with savings
        it could not have delivered.
        """
        if total_loops <= 0 or not self._frozen:
            return {
                "estimated_saved_fraction": 0.0,
                "frozen_count": 0,
                "total_loops": total_loops,
            }
        saved = 0
        for branch, froze_at in self._frozen.items():
            if froze_at >= self.min_loops:
                saved += total_loops - froze_at
        total = total_loops * max(1, len(self._frozen) + len(self.active_branches()))
        return {
            "estimated_saved_fraction": saved / total if total else 0.0,
            "frozen_count": len(self._frozen),
            "total_loops": total_loops,
        }




