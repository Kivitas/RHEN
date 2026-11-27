"""Expected-loss policy with a reject option (plan section 9.4).

Replaces ``P(option) x utility`` (finding C6).  The caller gives, per question, either a full cost
matrix ``C[a][y]`` (cost of taking action ``a`` when the truth is ``y``; ``y`` ranges over the
option labels plus ``__null__``) or the shorthand ``cost_wrong`` / ``cost_abstain`` with
``cost_right = 0``::

    risk(a)       = sum_y P(y) * C[a][y]            P is the calibrated distribution, null included
    risk(abstain) = cost_abstain
    a*            = argmin over admissible a

Constraints remove inadmissible actions first.  Then: nothing admissible -> BLOCKED; ``a*`` is not
strictly cheaper than abstaining, or the Checker verdict is not CONFIRM -> ABSTAIN; the tier needs
approval -> NEEDS_APPROVAL; otherwise EXECUTE.  With the shorthand costs this is exactly
"act iff p_correct > 1 - cost_abstain / cost_wrong".

Defaults per tier come from the error tolerances of plan 8.2 (low 10%, medium 5%, high 1%,
critical 0.1%): ``cost_wrong = 1`` and ``cost_abstain = tolerance``, so the policy acts only when
the calibrated error probability is below the tier's tolerance.  No other number is invented here.

Fail closed (plan 9.2): when the decision is not calibrated, nothing above tier ``low`` may EXECUTE
(or be offered for approval); it is ABSTAIN.

The status enum is the v1 ``PolicyStatus`` (EXECUTE / NEEDS_APPROVAL / ABSTAIN / BLOCKED).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from rhen.decision_pipeline import DecisionPolicy, PolicyStatus

NULL = "__null__"
TIERS = ("low", "medium", "high", "critical")
TIER_TOLERANCE: Dict[str, float] = {"low": 0.10, "medium": 0.05, "high": 0.01, "critical": 0.001}
DEFAULT_APPROVAL_TIERS = ("high", "critical")

Constraint = Tuple[str, Callable[[str, Mapping[str, Any]], bool]]


# ---------------------------------------------------------------------------- costs


@dataclass
class CostSpec:
    """Either a full matrix ``matrix[action][truth]`` or ``cost_wrong`` (+ ``cost_abstain``).

    Entries missing from a matrix fall back to ``cost_wrong`` off the diagonal and ``cost_right`` on
    it when ``cost_wrong`` is given; otherwise they raise ``ValueError``.  ``cost_abstain`` and, with
    no matrix, ``cost_wrong`` default to the tier values (see module docstring).
    """

    matrix: Optional[Dict[str, Dict[str, float]]] = None
    cost_wrong: Optional[float] = None
    cost_abstain: Optional[float] = None
    cost_right: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cost_wrong", "cost_abstain", "cost_right"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(float(v)) or float(v) < 0):
                raise ValueError(f"{name} must be a finite non-negative number, got {v!r}")
        if self.matrix is not None:
            for a, row in self.matrix.items():
                for y, v in row.items():
                    if not math.isfinite(float(v)) or float(v) < 0:
                        raise ValueError(f"cost C[{a!r}][{y!r}] must be finite and non-negative")

    @classmethod
    def for_tier(cls, tier: str) -> "CostSpec":
        if tier not in TIER_TOLERANCE:
            raise ValueError(f"risk tier must be one of {TIERS}, got {tier!r}")
        return cls(cost_wrong=1.0, cost_abstain=TIER_TOLERANCE[tier])

    def resolve(self, tier: str) -> "CostSpec":
        """Fill unset fields from the tier defaults."""
        base = CostSpec.for_tier(tier)
        wrong = self.cost_wrong
        if wrong is None and self.matrix is None:
            wrong = base.cost_wrong
        return CostSpec(
            matrix=self.matrix, cost_wrong=wrong,
            cost_abstain=self.cost_abstain if self.cost_abstain is not None else base.cost_abstain,
            cost_right=float(self.cost_right),
        )

    def cost(self, action: str, truth: str) -> float:
        if self.matrix is not None:
            row = self.matrix.get(action)
            if row is not None and truth in row:
                return float(row[truth])
        if self.cost_wrong is None:
            raise ValueError(f"cost matrix has no entry C[{action!r}][{truth!r}] and no cost_wrong fallback")
        return float(self.cost_right) if action == truth else float(self.cost_wrong)

    def threshold(self) -> Optional[float]:
        """The shorthand act threshold ``1 - cost_abstain / cost_wrong`` (None for a full matrix)."""
        if self.matrix is not None or not self.cost_wrong:
            return None
        return 1.0 - float(self.cost_abstain) / float(self.cost_wrong)


def anchor_to_confidence(probs: Mapping[str, float], top: str, p_correct: float) -> Dict[str, float]:
    """Return a distribution whose ``top`` entry is ``p_correct`` and whose other entries keep their ratios.

    Used so the policy acts on the calibrated ``p_correct`` rather than on a raw softmax entry.
    """
    p_correct = min(max(float(p_correct), 0.0), 1.0)
    rest = {k: float(v) for k, v in probs.items() if k != top}
    z = sum(rest.values())
    out = {top: p_correct}
    for k, v in rest.items():
        out[k] = (1.0 - p_correct) * (v / z) if z > 0 else (1.0 - p_correct) / max(len(rest), 1)
    return out


# ---------------------------------------------------------------------------- outcome


@dataclass
class PolicyOutcome2:
    status: PolicyStatus
    option: Optional[str]
    risk_act: Optional[float]
    risk_abstain: Optional[float]
    risk_tier: str
    reasons: List[str] = field(default_factory=list)
    admissible: List[str] = field(default_factory=list)
    costs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value, "option": self.option, "risk_act": self.risk_act,
            "risk_abstain": self.risk_abstain, "risk_tier": self.risk_tier,
            "reasons": list(self.reasons), "admissible": list(self.admissible), "costs": dict(self.costs),
        }


# ---------------------------------------------------------------------------- policy


class DecisionPolicy2:
    def __init__(self, approval_required_tiers: Sequence[str] = DEFAULT_APPROVAL_TIERS,
                 constraints: Sequence[Constraint] = ()) -> None:
        tiers = tuple(approval_required_tiers)
        bad = [t for t in tiers if t not in TIERS]
        if bad:
            raise ValueError(f"approval_required_tiers must be drawn from {TIERS}, got {bad}")
        self.approval_required_tiers = tiers
        self.constraints: Tuple[Constraint, ...] = self._check_constraints(constraints)

    @staticmethod
    def _check_constraints(constraints: Sequence[Constraint]) -> Tuple[Constraint, ...]:
        checked: List[Constraint] = []
        for item in constraints or ():
            if not (isinstance(item, tuple) and len(item) == 2 and callable(item[1])):
                raise TypeError("each constraint must be a (name, predicate) pair; predicate(option, features) -> bool")
            checked.append((str(item[0]), item[1]))
        return tuple(checked)

    def _needs_approval(self, tier: str, approval: Union[None, bool, Sequence[str]]) -> bool:
        if approval is None:
            return tier in self.approval_required_tiers
        if isinstance(approval, bool):
            return approval or tier in self.approval_required_tiers
        return tier in tuple(approval)

    def evaluate(
        self,
        probabilities: Mapping[str, float],
        *,
        verdict: str = "CONFIRM",
        risk_tier: str = "low",
        calibrated: bool = True,
        costs: Optional[CostSpec] = None,
        constraints: Sequence[Constraint] = (),
        features: Optional[Mapping[str, Any]] = None,
        p_correct: Optional[float] = None,
        top_label: Optional[str] = None,
        approval: Union[None, bool, Sequence[str]] = None,
    ) -> PolicyOutcome2:
        if risk_tier not in TIERS:
            raise ValueError(f"risk_tier must be one of {TIERS}, got {risk_tier!r}")
        feats: Mapping[str, Any] = features if features is not None else {}
        spec = (costs if costs is not None else CostSpec()).resolve(risk_tier)
        probs = {str(k): float(v) for k, v in probabilities.items()}
        if any((not math.isfinite(v)) or v < 0 for v in probs.values()):
            raise ValueError("probabilities must be finite and non-negative")
        labels = [k for k in probs if k != NULL]
        reasons: List[str] = []
        costs_info = {"cost_abstain": spec.cost_abstain, "cost_wrong": spec.cost_wrong,
                      "cost_right": spec.cost_right, "matrix": spec.matrix is not None,
                      "act_threshold": spec.threshold()}

        # 1. constraints remove inadmissible actions first
        active = self._check_constraints(constraints) + self.constraints
        admissible: List[str] = []
        for a in labels:
            failed = [n for n, pred in active if not bool(pred(a, feats))]
            if failed:
                reasons.append(f"option {a!r} blocked by constraint(s): {', '.join(failed)}")
            else:
                admissible.append(a)
        if not admissible:
            reasons.append("no admissible action" if labels else "no actions were supplied")
            return PolicyOutcome2(PolicyStatus.BLOCKED, None, None, spec.cost_abstain, risk_tier, reasons, [], costs_info)

        # 2. expected loss
        if p_correct is not None and labels:
            top = top_label if top_label in probs else max(labels, key=lambda a: probs[a])
            probs = anchor_to_confidence(probs, top, p_correct)
        truths = list(probs)
        risks = {a: sum(probs[y] * spec.cost(a, y) for y in truths) for a in admissible}
        best = min(admissible, key=lambda a: (risks[a], -probs[a], labels.index(a)))
        risk_act, risk_abs = risks[best], float(spec.cost_abstain)
        reasons.append(f"best action {best!r}: risk {risk_act:.6g} vs abstain {risk_abs:.6g}")

        # 3. abstain conditions
        abstain: List[str] = []
        if not risk_act < risk_abs:
            abstain.append(f"abstaining is not riskier ({risk_abs:.6g}) than acting ({risk_act:.6g})")
        if verdict != "CONFIRM":
            abstain.append(f"Checker verdict is {verdict}")
        if not calibrated and risk_tier != "low":
            abstain.append(f"not calibrated: nothing above tier 'low' may act (tier {risk_tier!r})")
        if abstain:
            return PolicyOutcome2(PolicyStatus.ABSTAIN, best, risk_act, risk_abs, risk_tier, reasons + abstain,
                                  admissible, costs_info)

        # 4. approval, then execute
        if self._needs_approval(risk_tier, approval):
            reasons.append(f"risk tier {risk_tier!r} requires human approval")
            return PolicyOutcome2(PolicyStatus.NEEDS_APPROVAL, best, risk_act, risk_abs, risk_tier, reasons,
                                  admissible, costs_info)
        return PolicyOutcome2(PolicyStatus.EXECUTE, best, risk_act, risk_abs, risk_tier, reasons, admissible, costs_info)


# ---------------------------------------------------------------------------- cascades


@dataclass
class CascadeReport:
    value: float
    label: str
    stages: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "label": self.label, "stages": list(self.stages)}


def cascade_confidence(stage_confidences: Sequence[float],
                       whole_workflow_calibration: Optional[Callable[[float], float]] = None) -> CascadeReport:
    """Product of stage confidences (plan 9.4).  Calibrated parts do not compose into a calibrated whole,
    so the value is labelled ``uncalibrated_joint`` unless a calibration fitted on whole workflows is given
    (a callable mapping the product to a calibrated probability), then ``calibrated_joint``.
    """
    stages = [float(c) for c in stage_confidences]
    if not stages or any(not 0.0 <= c <= 1.0 for c in stages):
        raise ValueError("stage confidences must be a non-empty list of values in [0, 1]")
    prod = 1.0
    for c in stages:
        prod *= c
    if whole_workflow_calibration is None:
        return CascadeReport(prod, "uncalibrated_joint", stages)
    return CascadeReport(float(whole_workflow_calibration(prod)), "calibrated_joint", stages)


# ---------------------------------------------------------------------------- forecast hook (plan 9.5)


def forecast_cost_fn(base_cost_wrong: float = 1.0,
                     cost_abstain: Optional[float] = None) -> Callable[[Mapping[str, Any]], CostSpec]:
    """Default ``cost_fn`` for forecast-aware decisions: ``cost_wrong = base * (1 + breach_prob_any_step_upper)``.

    ``features`` is the forecast feature dict of ``rhen.decision_pipeline`` (a missing or None upper bound counts as
    0).  A wider breach risk makes a wrong action dearer, so the policy needs a higher calibrated confidence to act
    instead of applying a separate hard uncertainty threshold.  ``cost_abstain`` None keeps the tier default.
    """
    def fn(features: Mapping[str, Any]) -> CostSpec:
        upper = features.get("breach_prob_any_step_upper")
        upper = 0.0 if upper is None else min(max(float(upper), 0.0), 1.0)
        return CostSpec(cost_wrong=float(base_cost_wrong) * (1.0 + upper), cost_abstain=cost_abstain)
    return fn


# ---------------------------------------------------------------------------- v1 migration


class LegacyDecisionPolicy(DecisionPolicy):
    """v1 ``DecisionPolicy(utility=...)`` with a DeprecationWarning.  Behaviour is unchanged."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "DecisionPolicy(utility=...) ranks by P(option) x utility, a heuristic without costs. "
            "Use rhen.v2.policy.DecisionPolicy2 with a CostSpec (expected loss with a reject option); "
            "from_v1_policy() converts an existing policy.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def from_v1_policy(policy: DecisionPolicy, cost_wrong: float = 1.0) -> Tuple[DecisionPolicy2, CostSpec]:
    """Convert a v1 policy to ``(DecisionPolicy2, CostSpec)``.

    ``min_confidence`` c becomes ``cost_abstain = (1 - c) * cost_wrong`` (act iff p > c).  Constraints
    and approval tiers carry over; the utility function and the forecast-uncertainty cap have no
    expected-loss equivalent and are dropped (the caller should fold forecast uncertainty into
    ``cost_wrong``, plan 9.5).  Emits a DeprecationWarning.
    """
    warnings.warn("from_v1_policy: DecisionPolicy is deprecated; use rhen.v2.policy.DecisionPolicy2",
                  DeprecationWarning, stacklevel=2)
    new = DecisionPolicy2(approval_required_tiers=tuple(policy.approval_required_tiers),
                          constraints=tuple(policy.constraints))
    spec = CostSpec(cost_wrong=float(cost_wrong), cost_abstain=(1.0 - float(policy.min_confidence)) * float(cost_wrong))
    return new, spec


__all__ = [
    "CostSpec", "DecisionPolicy2", "PolicyOutcome2", "PolicyStatus", "CascadeReport", "cascade_confidence",
    "anchor_to_confidence", "forecast_cost_fn", "LegacyDecisionPolicy", "from_v1_policy", "TIER_TOLERANCE", "TIERS", "NULL",
]
