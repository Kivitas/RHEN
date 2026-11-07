"""
Rhen-1.0 Pillar Wiring.

The Rhen-1.0 pillars in ``rhen_cognition.py`` and the Rhen-1.0 components in
``rhen_verify.py`` / ``rhen_credit.py`` were previously standalone utilities with
no automatic effect on model routing, recurrent halting, calibration or
training. This module is the missing integration layer: it binds them to a
``UnifiedAIOrchestrator`` so they actually gate decisions.

Design rules, all deliberate:

* **Fail-closed on the security path.** If the audit sink is unavailable, the
  action is *not* authorised. Silently degrading to an in-memory log would
  recreate exactly the weakness arXiv:2609.30266 documents.
* **Gates before side effects.** Verification and regression gating run *before*
  an action card is staged, not after.
* **Reference-relative, not fixed thresholds.** Per arXiv:2609.20538, alarm
  thresholds are re-derived per regime instead of reusing a global constant.
* **Search before RL.** The search baseline is recorded so that later RL gains
  are credited only with what search could not already deliver.
* **Every stage is toggleable.** ``RhenPillarConfig`` defaults to the current
  behaviour so that enabling this module changes nothing until a
  pillar is explicitly switched on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

try:
    from rhen.rhen_cognition import (
        DurationThresholdingHead,
        HierarchicalChoiceRouter,
        MultilingualDecisionRouter,
        OODRegressionGuard,
    )
    from rhen.rhen_credit import (
        CascadeSearch,
        HillSampler,
        SchemaContrastiveAdvantage,
        SegmentAdvantage,
        TrajectoryGraph,
        search_vs_rl,
    )
    from rhen.rhen_verify import (
        BranchSparsityTracker,
        CheapVerifier,
        FileAuditSink,
        GeometrySelectiveClassifier,
        InProcessAuditSink,
        OutOfBandAuditLedger,
        SocketAuditSink,
    )
    from rhen.eos_eval import CostLedger, PreRegistration, ReferenceRelativeThreshold
except ImportError:
    from rhen_cognition import (
        DurationThresholdingHead,
        HierarchicalChoiceRouter,
        MultilingualDecisionRouter,
        OODRegressionGuard,
    )
    from rhen_credit import (
        CascadeSearch,
        HillSampler,
        SchemaContrastiveAdvantage,
        SegmentAdvantage,
        TrajectoryGraph,
        search_vs_rl,
    )
    from rhen_verify import (
        BranchSparsityTracker,
        CheapVerifier,
        FileAuditSink,
        GeometrySelectiveClassifier,
        InProcessAuditSink,
        OutOfBandAuditLedger,
        SocketAuditSink,
    )
    from eos_eval import CostLedger, PreRegistration, ReferenceRelativeThreshold

__all__ = [
    "RhenPillarConfig",
    "PillarDecision",
    "PillarStack",
]


@dataclass
class RhenPillarConfig:
    """Per-pillar switches. All default to the behaviour without the pillar enabled."""

    # Language-aware calibration routing
    enable_language_calibration: bool = False
    calibration_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Hierarchical choice for large option sets
    enable_hierarchical_choice: bool = False
    choice_num_groups: int = 8
    choice_flat_threshold: int = 20

    # Adaptive recurrent depth per branch
    enable_duration_thresholding: bool = False
    tdt_max_duration: int = 8

    # OOD / regression gates
    enable_ood_guard: bool = False
    enable_reference_relative_thresholds: bool = False
    ood_regime: str = "default"

    # Audit integrity. With enable_audit=True:
    #   "memory" / "file": audit_key is required (>= 16 bytes) and seals locally.
    #   "socket": the receiver seals with its own key, so audit_key must stay None.
    enable_audit: bool = False
    audit_key: Optional[bytes] = None
    audit_sink: str = "memory"  # "memory" | "file" | "socket"
    audit_address: Optional[str] = None
    audit_path: Optional[str] = None
    audit_fail_closed: bool = True

    # Verification / selectivity
    enable_selectivity: bool = False
    selectivity_max_radius: float = 1.0
    enable_cheap_verifier: bool = False

    # Search-before-RL
    enable_search: bool = False
    search_samples: int = 8

    # Credit assignment
    enable_credit_assignment: bool = False

    # Instrumentation
    enable_cost_ledger: bool = True
    enable_branch_sparsity: bool = False

    def __post_init__(self) -> None:
        if self.audit_sink not in ("memory", "file", "socket"):
            raise ValueError(
                f"audit_sink must be 'memory', 'file' or 'socket', got {self.audit_sink!r}"
            )
        if not self.enable_audit:
            return
        if self.audit_sink == "socket":
            if not self.audit_address:
                raise ValueError("audit_sink='socket' requires audit_address")
            if self.audit_key is not None:
                raise ValueError(
                    "audit_sink='socket' seals on the receiver; leave audit_key=None so the "
                    "agent process never holds the sealing key"
                )
            return
        if self.audit_sink == "file" and not self.audit_path:
            raise ValueError("audit_sink='file' requires audit_path")
        key = self.audit_key
        if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
            raise ValueError("enable_audit=True requires an explicit audit_key of at least 16 bytes")


@dataclass
class PillarDecision:
    """Verdict returned by a pillar gate."""

    allowed: bool
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class PillarStack:
    """Binds the Rhen-1.0 pillars to an orchestrator.

    Every gate returns a :class:`PillarDecision` and never raises for an ordinary
    negative outcome -- a *rejected* decision is a normal result, not an error.
    The single exception is audit failure, which propagates (see
    :meth:`authorize_action`) because a failed audit must not be silently
    downgraded to a successful one.
    """

    def __init__(self, config: Optional[RhenPillarConfig] = None) -> None:
        self.config = config or RhenPillarConfig()
        cfg = self.config

        self.language_router = MultilingualDecisionRouter()
        self.choice_router = HierarchicalChoiceRouter(
            num_groups=cfg.choice_num_groups,
            flat_threshold=cfg.choice_flat_threshold,
        )
        self.ood_guard = OODRegressionGuard()
        self.selective = GeometrySelectiveClassifier(
            max_radius=cfg.selectivity_max_radius
        )
        self.cheap_verifier = CheapVerifier()
        self.search = CascadeSearch()
        self.hill = HillSampler()
        self.trajectory_graph = TrajectoryGraph()
        self.segment_adv = SegmentAdvantage()
        self.schema_adv = SchemaContrastiveAdvantage()
        self.tdt: Optional[DurationThresholdingHead] = None
        self.thresholds = ReferenceRelativeThreshold()
        self.sparsity = BranchSparsityTracker()
        self.costs = CostLedger()
        self.registrations: Dict[str, PreRegistration] = {}

        self.audit: Optional[OutOfBandAuditLedger] = None
        if cfg.enable_audit:
            # Remote sealing (socket sink): the agent holds no key. Local sinks use the
            # explicit key that RhenPillarConfig has already validated.
            self.audit = OutOfBandAuditLedger(
                key=None if cfg.audit_sink == "socket" else cfg.audit_key,
                sink=self._build_sink(),
                fail_closed=cfg.audit_fail_closed,
            )

        if cfg.enable_cost_ledger:
            for pillar in (
                "language_calibration",
                "hierarchical_choice",
                "duration_thresholding",
                "ood_guard",
                "audit_ledger",
                "selectivity",
                "search",
            ):
                self.costs.register(pillar, measured_gain=None)

    def _build_sink(self) -> Any:
        cfg = self.config
        if cfg.audit_sink == "socket":
            if not cfg.audit_address:
                raise ValueError("audit_sink='socket' requires audit_address")
            return SocketAuditSink(cfg.audit_address)
        if cfg.audit_sink == "file":
            if not cfg.audit_path:
                raise ValueError("audit_sink='file' requires audit_path")
            return FileAuditSink(cfg.audit_path)
        return InProcessAuditSink()

    # -- gates -------------------------------------------------------------

    def route_calibration(self, state_text: str) -> Dict[str, Any]:
        """Language-specific temperature/alpha for this state."""
        if not self.config.enable_language_calibration:
            return {
                "language": "default",
                "profile": {"temperature": 1.0, "alpha": 0.05},
                "applied": False,
            }
        with self.costs.time("language_calibration"):
            record = self.language_router.route(state_text, self.config.calibration_profiles)
        record["applied"] = True
        return record

    def route_choice(
        self, options: Sequence[str], score_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Coarse-to-fine option selection for large option sets."""
        if not self.config.enable_hierarchical_choice:
            return {
                "chosen_index": -1,
                "decomposed": False,
                "scored_options": 0,
                "total_options": len(options),
                "applied": False,
            }
        with self.costs.time("hierarchical_choice"):
            return self.choice_router.route(list(options), score_fn=score_fn)

    def plan_depth(self, hidden: Any) -> Dict[str, Any]:
        """Per-branch recurrent-loop budget (TDT)."""
        if not self.config.enable_duration_thresholding or hidden is None:
            return {"applied": False}
        if self.tdt is None:
            self.tdt = DurationThresholdingHead(
                hidden_size=int(hidden.shape[-1]),
                max_duration=self.config.tdt_max_duration,
            )
        with self.costs.time("duration_thresholding"):
            return self.tdt.expected_loops(hidden)

    def check_regression(
        self, predictions: Sequence[Any], labels: Sequence[Any], name: str = "candidate"
    ) -> PillarDecision:
        """OOD regression gate run before a change is accepted."""
        if not self.config.enable_ood_guard:
            return PillarDecision(True, "ood guard disabled", {"applied": False})
        with self.costs.time("ood_guard"):
            if self.ood_guard.baseline is None:
                return PillarDecision(
                    True,
                    "no baseline established yet",
                    {"applied": True, "verdict": "baseline_pending"},
                )
            rec = self.ood_guard.evaluate(predictions, labels, name=name)
        return PillarDecision(
            bool(rec["accepted"]), rec["reason"], {"applied": True, "record": rec}
        )

    def establish_baseline(
        self, predictions: Sequence[Any], labels: Sequence[Any], name: str = "baseline"
    ) -> Dict[str, Any]:
        return self.ood_guard.establish_baseline(predictions, labels, name=name)

    def threshold_verdict(self, score: float, regime: Optional[str] = None) -> Dict[str, Any]:
        """Per-regime alarm verdict; abstains when the regime is uncalibrated."""
        if not self.config.enable_reference_relative_thresholds:
            return {"admitted": True, "verdict": "disabled", "threshold": None}
        return self.thresholds.is_anomalous(score, regime or self.config.ood_regime)

    def calibrate_regime(self, regime: str, scores: Sequence[float]) -> Dict[str, Any]:
        return self.thresholds.calibrate(regime, scores)

    def authorize_action(
        self, question: str, chosen: Any, confidence: float, **extra: Any
    ) -> Any:
        """Record a decision in the out-of-band audit ledger.

        Records already emitted to a remote receiver cannot be rewritten without
        detection. A compromised agent can still stop emitting (visible as a gap) or
        emit false records, which verify. See :class:`OutOfBandAuditLedger`.

        Raises:
            AuditSinkError: when the sink is unreachable and the ledger is
                fail-closed. This is deliberate -- an action that could not be
                audited must not be reported as audited.
        """
        if self.audit is None:
            raise RuntimeError("audit ledger not enabled")
        with self.costs.time("audit_ledger"):
            return self.audit.record_decision(question, chosen, confidence, **extra)

    def search_baseline(self, sample_fn: Callable[[int], Tuple[Any, float]]) -> Dict[str, Any]:
        """Record the search-only ceiling any later RL result must beat."""
        if not self.config.enable_search:
            return {"applied": False, "best_score": None}
        with self.costs.time("search"):
            res = self.search.best_of_n(sample_fn, self.config.search_samples)
        return {
            "applied": True,
            "best_score": res["best_score"],
            "mean_score": res["mean_score"],
            "n": res["n"],
            "note": "any RL gain must exceed this search-only ceiling",
        }

    def rl_report(
        self, base_scores: Sequence[float], rl_scores: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        """Quantify how much of an RL gain search already explains."""
        return search_vs_rl(base_scores, rl_scores).to_dict()

    def register_claim(
        self,
        claim_id: str,
        metric: str,
        threshold: float,
        direction: str = "increase",
        min_examples: int = 1,
    ) -> PreRegistration:
        reg = PreRegistration(claim_id, metric, threshold, direction, min_examples)
        self.registrations[claim_id] = reg
        return reg

    def cost_report(self) -> Dict[str, Any]:
        return self.costs.report()

    def close(self) -> None:
        if self.audit is not None:
            self.audit.close()




