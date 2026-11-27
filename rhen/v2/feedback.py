"""Outcome ledger, online recalibration bridge and drift alarms (plan 9.6, T-604).

``OutcomeLedger`` keeps decisions in a bounded ring buffer (``max_entries``; the oldest entry is
dropped, never an unbounded list).  Outcomes arrive later with ``decision_correct: bool | None``.
When an ``OnlineRecalibrator`` is attached, each labelled outcome that carries the decision's logits
is passed to ``OnlineRecalibrator.observe``, which refits temperature and conformal quantile every
500 outcomes per group and raises drift alarms; an alarmed group is no longer calibrated
(``group_calibrated`` is False), and the pipeline treats its decisions as ``calibrated=False``.

Label rule for ``observe`` (it needs the true option index): an explicit ``label_idx`` wins; else a
correct decision (``decision_correct=True``) means the truth was the predicted index; a wrong decision
with no ``label_idx`` cannot be observed (the truth is unknown), is counted in ``unobserved`` and does
not reach the recalibrator.  ``decision_correct=None`` records a pending/unknown outcome.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

from rhen.v2.packing import k_bucket


@dataclass
class DecisionEntry:
    decision_id: str
    qtype: str
    k: int
    status: str = ""
    answer: Any = None
    predicted_idx: Optional[int] = None
    confidence: float = 0.0
    calibrated: bool = False
    ood: bool = False
    logits: Optional[List[float]] = None      # pre-temperature logits, options then null last
    created_at: float = field(default_factory=time.time)
    decision_correct: Optional[bool] = None
    label_idx: Optional[int] = None
    outcome_at: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutcomeLedger:
    def __init__(self, recalibrator: Any = None, max_entries: int = 10000, max_alarms: int = 1000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.recalibrator = recalibrator
        self.max_entries = int(max_entries)
        self._entries: Deque[DecisionEntry] = deque(maxlen=self.max_entries)
        self._index: Dict[str, DecisionEntry] = {}
        self.alarm_log: Deque[Dict[str, Any]] = deque(maxlen=max_alarms)
        self.n_outcomes = 0
        self.n_correct = 0
        self.unobserved = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[DecisionEntry]:
        return list(self._entries)

    def get(self, decision_id: str) -> DecisionEntry:
        try:
            return self._index[decision_id]
        except KeyError:
            raise KeyError(f"unknown or evicted decision_id {decision_id!r}") from None

    # -- decisions ----------------------------------------------------------------------
    def record_decision(self, entry: DecisionEntry) -> DecisionEntry:
        if entry.decision_id in self._index:
            raise ValueError(f"decision {entry.decision_id!r} already recorded")
        if len(self._entries) == self._entries.maxlen:
            old = self._entries[0]
            self._index.pop(old.decision_id, None)
            self.evicted += 1
        self._entries.append(entry)
        self._index[entry.decision_id] = entry
        return entry

    # -- outcomes -----------------------------------------------------------------------
    def record_outcome(self, decision_id: str, decision_correct: Optional[bool] = None,
                       label_idx: Optional[int] = None) -> List[str]:
        """Attach an outcome; returns the drift alarms it raised (empty when none)."""
        e = self.get(decision_id)
        if e.outcome_at is not None:
            raise ValueError(f"decision {decision_id!r} already has an outcome")
        if decision_correct is not None and not isinstance(decision_correct, bool):
            raise TypeError("decision_correct must be bool or None")
        e.decision_correct, e.label_idx, e.outcome_at = decision_correct, label_idx, time.time()
        if decision_correct is None and label_idx is None:
            return []
        self.n_outcomes += 1
        if decision_correct:
            self.n_correct += 1
        if self.recalibrator is None or e.logits is None:
            return []
        lab = label_idx
        if lab is None and decision_correct is True:
            lab = e.predicted_idx
        if lab is None:
            self.unobserved += 1
            return []
        alarms = self.recalibrator.observe(e.qtype, e.k, e.logits, int(lab), ood_flag=e.ood)
        for a in alarms:
            self.alarm_log.append({"decision_id": decision_id, "qtype": e.qtype, "k": e.k, "alarm": a,
                                   "at": time.time()})
        return list(alarms)

    # -- state --------------------------------------------------------------------------
    def group_calibrated(self, qtype: str, k: int) -> bool:
        """False once a drift alarm is active for the group (True when no recalibrator is attached)."""
        if self.recalibrator is None:
            return True
        alarms = getattr(self.recalibrator, "alarms", {})
        return (qtype, str(k_bucket(k))) not in alarms

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._entries), "outcomes": self.n_outcomes, "correct": self.n_correct,
                "unobserved": self.unobserved, "evicted": self.evicted,
                "alarmed_groups": sorted(f"{t}/{b}" for t, b in getattr(self.recalibrator, "alarms", {}))}

    # -- JSONL --------------------------------------------------------------------------
    def to_jsonl(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], recalibrator: Any = None, max_entries: int = 10000) -> "OutcomeLedger":
        """Load records (the last ``max_entries`` are kept).  Loading does not replay outcomes into the recalibrator."""
        led = cls(recalibrator=recalibrator, max_entries=max_entries)
        names = set(DecisionEntry.__dataclass_fields__)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                e = DecisionEntry(**{k: v for k, v in d.items() if k in names})
                led.record_decision(e)
                if e.outcome_at is not None and (e.decision_correct is not None or e.label_idx is not None):
                    led.n_outcomes += 1
                    led.n_correct += 1 if e.decision_correct else 0
        return led


__all__ = ["DecisionEntry", "OutcomeLedger"]
