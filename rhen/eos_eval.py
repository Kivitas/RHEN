"""
Rhen-1.0 Measurement Infrastructure (Track 0).

This module exists because several of the architectural claims in EOS are
currently *unmeasured*, and adding more architecture on top of an unverified
premise is how projects get large and no better. Everything here is
instrumentation, not model change: it costs nothing at inference and can be run
against any checkpoint.

1. :class:`PreRegistration` -- fix a claim, its metric and its threshold
   *before* measuring. Derived from arXiv:2609.20538 ("Refuse, Decompose,
   Refresh"), which argues that an evaluation can be perfectly reproducible and
   still support the wrong claim, and prescribes three actions: **Refuse**
   (abstain when the reference stream cannot support a claim), **Decompose**
   (report protocol execution, operational false admission and structural
   hypotheses separately rather than as one PASS/FAIL), and **Refresh** (treat a
   distribution-shift alarm as a request to invalidate and recompute the
   reference map, not as evidence against the candidate).

2. :class:`ReferenceRelativeThreshold` -- the operational half of "Refresh".
   arXiv:2609.20538 found clean fault-null streams triggered alarms in 15/15,
   0/15 and 14/15 cases across three regimes, with only one regime matching
   the frozen detector reference. A fixed threshold is therefore only
   interpretable *relative to the stream it was calibrated on*. This class
   calibrates on a reference stream and re-derives thresholds per regime, rather
   than reusing one global number.

3. :class:`CostLedger` -- per-pillar cost accounting. A pillar that does not pay
   for its latency should be visible as such.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "PreRegistration",
    "ClaimResult",
    "ReferenceRelativeThreshold",
    "CostLedger",
    "PillarCost",
    "RunRecord",
]


# ---------------------------------------------------------------------------
# 1. Pre-registration (arXiv:2609.20538)
# ---------------------------------------------------------------------------


@dataclass
class ClaimResult:
    """Outcome of one pre-registered claim.

    Deliberately reports three things separately rather than a single verdict,
    per the Decompose step of arXiv:2609.20538.
    """

    claim_id: str
    protocol_executed: bool
    operational_result: Optional[bool]
    structural_verdict: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class PreRegistration:
    """Register claims before measuring them, then report them decomposed.

    Example:
        >>> reg = PreRegistration("step-size", metric="val_loss", threshold=0.0,
        ...                       direction="decrease", min_examples=2)
        >>> reg.record("step-size", protocol_executed=True, value=-0.05).to_dict()
        {'claim_id': 'step-size', 'protocol_executed': True, ...}
    """

    def __init__(
        self,
        claim_id: str,
        metric: str,
        threshold: float,
        direction: str = "increase",
        min_examples: int = 1,
        description: str = "",
    ) -> None:
        if direction not in ("increase", "decrease"):
            raise ValueError("direction must be 'increase' or 'decrease'")
        self.claim_id = str(claim_id)
        self.metric = str(metric)
        self.threshold = float(threshold)
        self.direction = direction
        self.min_examples = int(min_examples)
        self.description = description
        self.results: List[ClaimResult] = []
        self.refresh_count = 0

    def record(
        self,
        claim_id: str,
        protocol_executed: bool,
        value: Optional[float] = None,
        note: str = "",
    ) -> ClaimResult:
        """Record one observation of a registered claim."""
        if claim_id != self.claim_id:
            raise ValueError(
                f"claim {claim_id!r} was not registered (expected {self.claim_id!r})"
            )
        operational: Optional[bool] = None
        if value is not None and protocol_executed:
            if self.direction == "increase":
                operational = value > self.threshold
            else:
                operational = value < self.threshold
        result = ClaimResult(
            claim_id=self.claim_id,
            protocol_executed=bool(protocol_executed),
            operational_result=operational,
            structural_verdict=self._verdict(protocol_executed, operational),
            note=note,
        )
        self.results.append(result)
        return result

    def _verdict(self, executed: bool, operational: Optional[bool]) -> str:
        if not executed:
            # "Refuse": the protocol could not be run, so no claim is licensed.
            return "refused"
        if operational is None:
            return "inconclusive"
        return "supported" if operational else "not_supported"

    def summary(self) -> Dict[str, Any]:
        """Aggregate verdict, refusing to collapse a refusal into a failure."""
        n = len(self.results)
        if n == 0:
            return {
                "claim_id": self.claim_id,
                "n": 0,
                "verdict": "inconclusive",
                "note": "no observations recorded",
            }
        executed = sum(1 for r in self.results if r.protocol_executed)
        supported = sum(1 for r in self.results if r.operational_result is True)
        operational = sum(1 for r in self.results if r.operational_result is not None)
        enough = n >= self.min_examples

        if executed == 0:
            verdict = "refused"
        elif not enough:
            verdict = "inconclusive"
        elif supported / max(1, operational) >= 0.95:
            verdict = "supported"
        elif supported == 0:
            verdict = "not_supported"
        else:
            verdict = "mixed"

        return {
            "claim_id": self.claim_id,
            "metric": self.metric,
            "threshold": self.threshold,
            "direction": self.direction,
            "n": n,
            "protocol_executed": executed,
            "operational_results": operational,
            "supported": supported,
            "refusals": n - executed,
            "verdict": verdict,
        }

    def refresh(self, reason: str) -> Dict[str, Any]:
        """Invalidate the reference and re-baseline (the 'Refresh' action).

        Distribution-shift alarms should trigger a recomputation of the
        reference map, not be recorded as evidence against the candidate.
        """
        self.refresh_count += 1
        self.results.clear()
        return {
            "refreshed": True,
            "claim_id": self.claim_id,
            "reason": reason,
            "refresh_count": self.refresh_count,
            "note": "prior observations discarded; re-baseline the reference stream",
        }


# ---------------------------------------------------------------------------
# 2. Reference-relative thresholds (arXiv:2609.20538)
# ---------------------------------------------------------------------------


class ReferenceRelativeThreshold:
    """Calibrate a threshold on a reference stream, then re-derive per regime.

    arXiv:2609.20538's sharpest operational finding: across three demand
    regimes, *clean* fault-null streams triggered alarms in 15/15, 0/15 and
    14/15 cases, and only the middle regime matched the frozen detector
    reference. The conclusion is that a detector threshold is only meaningful
    **relative to the stream it was calibrated on**. A hard-coded global
    threshold -- like ``EnergyOODDetector(ood_threshold=4.5)`` in
    ``meta_cognitive_monitor`` -- produces false alarms in some regimes and
    misses real ones in others.

    This class makes that explicit: calibrate on a labelled reference stream,
    then compute a per-regime quantile so a candidate is compared against its
    *own* regime's null rather than a global constant.
    """

    def __init__(self, quantile: float = 0.95, min_reference: int = 8) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be strictly between 0 and 1")
        self.quantile = float(quantile)
        self.min_reference = int(min_reference)
        self.reference: Dict[str, List[float]] = {}

    def calibrate(self, regime: str, scores: Sequence[float]) -> Dict[str, Any]:
        """Record a labelled *null* (no-fault) stream for one regime."""
        vals = [float(s) for s in scores]
        if not vals:
            raise ValueError("cannot calibrate on an empty reference stream")
        self.reference.setdefault(str(regime), []).extend(vals)
        return {
            "regime": regime,
            "n_reference": len(self.reference[str(regime)]),
            "sufficient": len(self.reference[str(regime)]) >= self.min_reference,
        }

    def threshold(self, regime: str) -> Optional[float]:
        """Per-regime alarm threshold, or ``None`` when under-referenced."""
        vals = self.reference.get(str(regime))
        if not vals or len(vals) < self.min_reference:
            return None
        ordered = sorted(vals)
        idx = min(len(ordered) - 1, int(round(self.quantile * (len(ordered) - 1))))
        return ordered[idx]

    def is_anomalous(self, score: float, regime: str) -> Dict[str, Any]:
        """Compare a score against its own regime's null.

        An under-referenced regime yields ``admitted=False`` rather than a
        confident verdict -- abstention, not a fabricated pass.
        """
        thr = self.threshold(regime)
        if thr is None:
            return {
                "admitted": False,
                "verdict": "unadmitted",
                "reason": f"regime {regime!r} has fewer than "
                          f"{self.min_reference} reference samples",
                "threshold": None,
            }
        return {
            "admitted": bool(float(score) > thr),
            "verdict": "anomalous" if float(score) > thr else "within_reference",
            "threshold": thr,
            "reason": "reference-relative comparison",
        }

    def cross_regime_false_alarm_rate(
        self, scores_by_regime: Dict[str, Sequence[float]]
    ) -> Dict[str, Any]:
        """Measure whether a single global threshold would have been wrong.

        Compares the pooled-global threshold against per-regime thresholds and
        reports each regime's false-alarm rate under both -- quantifying exactly
        the failure arXiv:2609.20538 documents.
        """
        pooled: List[float] = []
        for vals in scores_by_regime.values():
            pooled.extend(float(v) for v in vals)
        out: Dict[str, Any] = {"per_regime": {}}
        if not pooled:
            return out

        ordered = sorted(pooled)
        gidx = min(len(ordered) - 1, int(round(self.quantile * (len(ordered) - 1))))
        global_thr = ordered[gidx]

        for regime, vals in scores_by_regime.items():
            scores = [float(v) for v in vals]
            per_thr = self.threshold(regime)
            out["per_regime"][regime] = {
                "n": len(scores),
                "global_threshold": global_thr,
                "global_false_alarm_rate": (
                    sum(1 for s in scores if s > global_thr) / len(scores) if scores else 0.0
                ),
                "relative_threshold": per_thr,
                "relative_false_alarm_rate": (
                    (sum(1 for s in scores if s > per_thr) / len(scores))
                    if per_thr is not None and scores
                    else None
                ),
            }
        return out


# ---------------------------------------------------------------------------
# 3. Per-pillar cost ledger
# ---------------------------------------------------------------------------


@dataclass
class PillarCost:
    """Latency and parameter accounting for one architectural pillar."""

    name: str
    n_calls: int = 0
    total_seconds: float = 0.0
    extra_params: int = 0
    measured_gain: Optional[float] = None
    note: str = ""

    @property
    def mean_ms(self) -> float:
        return (self.total_seconds / self.n_calls * 1000.0) if self.n_calls else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_calls": self.n_calls,
            "mean_ms": self.mean_ms,
            "extra_params": self.extra_params,
            "measured_gain": self.measured_gain,
            # None means "never measured", which is different from "measured
            # and worth nothing"; the report keeps those apart.
            "pays_for_itself": (
                None if self.measured_gain is None else self.measured_gain > 0.0
            ),
            "note": self.note,
        }


@dataclass
class RunRecord:
    """One timed invocation of a pillar."""

    name: str
    seconds: float
    ok: bool = True
    error: str = ""


class CostLedger:
    """Track what each pillar costs and whether it demonstrably pays.

    The point is accountability: a 13-pillar architecture in which most pillars
    are unmeasured is not a stronger design, it is an unexamined one. This
    ledger makes latency, parameter overhead and *measured* benefit visible per
    pillar so unproven pillars can be identified and removed.
    """

    def __init__(self) -> None:
        self.pillars: Dict[str, PillarCost] = {}
        self.runs: List[RunRecord] = []

    def register(
        self,
        name: str,
        extra_params: int = 0,
        measured_gain: Optional[float] = None,
        note: str = "",
    ) -> PillarCost:
        if name not in self.pillars:
            self.pillars[name] = PillarCost(
                name=name, extra_params=extra_params,
                measured_gain=measured_gain, note=note,
            )
        return self.pillars[name]

    def time(self, name: str) -> "_CostTimer":
        """Context manager timing one invocation of a pillar."""
        return _CostTimer(self, name)

    def _record(self, name: str, seconds: float, ok: bool, error: str = "") -> None:
        pillar = self.pillars.setdefault(name, PillarCost(name=name))
        pillar.n_calls += 1
        pillar.total_seconds += float(seconds)
        self.runs.append(RunRecord(name=name, seconds=float(seconds), ok=ok, error=error))

    def report(self) -> Dict[str, Any]:
        """Full cost/benefit table, separating unmeasured from unprofitable."""
        rows = [p.to_dict() for p in self.pillars.values()]
        measured = [r for r in rows if r["measured_gain"] is not None]
        unmeasured = [r["name"] for r in rows if r["measured_gain"] is None]
        unprofitable = [r["name"] for r in measured if r["measured_gain"] <= 0.0]
        return {
            "pillars": sorted(rows, key=lambda r: r["mean_ms"], reverse=True),
            "n_pillars": len(rows),
            "n_measured": len(measured),
            "unmeasured_pillars": unmeasured,
            "unprofitable_pillars": unprofitable,
            "total_ms": sum(r["mean_ms"] * r["n_calls"] for r in rows),
            "n_errors": sum(1 for r in self.runs if not r.ok),
        }

    def prune_candidates(self) -> List[str]:
        """Pillars to consider removing: unmeasured first, then unprofitable.

        The two are kept separate on purpose -- a pillar that was never measured
        is not yet known to be bad, and conflating them would license deleting
        things purely for lack of evidence.
        """
        rep = self.report()
        return rep["unprofitable_pillars"] + [
            n for n in rep["unmeasured_pillars"] if n not in rep["unprofitable_pillars"]
        ]


class _CostTimer:
    """Context manager returned by :meth:`CostLedger.time`."""

    def __init__(self, ledger: CostLedger, name: str) -> None:
        self.ledger = ledger
        self.name = name
        self.start = 0.0

    def __enter__(self) -> "_CostTimer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self.start
        ok = exc_type is None
        msg = f"{exc_type.__name__}: {exc}" if exc_type is not None else ""
        self.ledger._record(self.name, elapsed, ok, msg)
        # Never swallow the exception; the ledger only observes.
        return False





