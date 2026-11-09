"""
Forecast-aware decisions for RHEN.

``ForecastAwareDecider.decide`` runs this pipeline:

1. Forecast every named series with a ``Forecaster`` (naive baseline or TimesFM).
2. Append a short, deterministic forecast summary to the state text.
3. Call ``RhenEngine.choice`` on the augmented state, so the decision is conditioned on the forecast.
4. Pass the choice probabilities and forecast features to a ``DecisionPolicy``, which applies
   constraints, utility, confidence and uncertainty gates, and risk tiers.
5. Record the decision in an optional ``FeedbackLedger``.

Execution is separate. ``approve`` records a human approver in a private registry on the
decider, keyed by decision_id. ``execute`` runs an executor only for EXECUTE decisions, or for
NEEDS_APPROVAL decisions with an entry in that registry. It reads nothing else on the decision
object. The policy never executes anything. The registry is an in-process guard, not a security
boundary against code that can modify the process.

Breach risk is reported as two bounds on P(the series exceeds its threshold at some step):
a lower bound (the largest single-step probability) and an upper bound (the union bound,
the sum of single-step probabilities capped at 1). The risk tier is gated on the upper bound.

Limits: the policy's rules are deterministic and tested, but the utility function, constraints
and thresholds are the caller's to set. RHEN's decision scorer has no trained weights in the
repository, so conditioning on the forecast is wired up but not shown to improve accuracy. The
ledger measures forecast calibration (pinball loss, interval coverage, Brier score) against
realized outcomes. It does not measure decision quality.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .agent_tools import RISK_TIERS
from .forecasting import (
    QUANTILE_LEVELS,
    Forecast,
    Forecaster,
    brier_score,
    pinball_loss,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps this module free of a runtime engine import
    from .rhen import ChoiceResult


class PolicyStatus(str, Enum):
    """Outcome of ``DecisionPolicy.evaluate``. The string values are the public status names."""

    EXECUTE = "EXECUTE"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    ABSTAIN = "ABSTAIN"
    BLOCKED = "BLOCKED"


class DecisionGateError(RuntimeError):
    """Raised when an approval or execution request is refused by the decision gate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unit_utility(option: str, features: Mapping[str, Any]) -> float:
    return 1.0


def _check_horizon(horizon: Any) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    return int(horizon)


def _engine_abstains(choice_result: Any) -> bool:
    return bool(getattr(choice_result, "should_abstain", False))


@dataclass
class PolicyOutcome:
    """Result of a policy evaluation.

    ``option`` is the admissible option with the highest expected utility, or None when
    every option is blocked. For ABSTAIN it still names that candidate, so the caller can
    log what the policy would have done.
    """

    status: PolicyStatus
    option: Optional[str]
    expected_utility: Optional[float]
    reasons: List[str] = field(default_factory=list)
    risk_tier: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "option": self.option,
            "expected_utility": self.expected_utility,
            "reasons": list(self.reasons),
            "risk_tier": self.risk_tier,
        }


class DecisionPolicy:
    """Deterministic gate between a choice result and any action.

    Rules, applied in order:

    1. Constraints. Each ``(name, predicate)`` is called as ``predicate(option, features)``.
       Any option that fails a predicate is blocked. No admissible option gives BLOCKED.
    2. Expected utility. Each admissible option scores ``P(option) * utility(option, features)``,
       where P is the engine's probability, not renormalised over the admissible set. The
       highest score is the candidate. Utilities must be non-negative gains. A negative
       utility raises ``ValueError``, because with a negative utility a low-probability option
       can outrank a likely one. This expected-utility form is a heuristic over the option
       probabilities, not a full cost model, so costs belong outside the policy (for example
       as constraints).
    3. Abstention. Confidence below ``min_confidence``, forecast uncertainty above
       ``max_forecast_uncertainty``, or the engine's own abstention verdict gives ABSTAIN.
    4. Risk tier. The tier is ``tier_fn(option, features)`` if given. Otherwise it is gated on
       the breach upper bound, ``features["breach_prob_any_step_upper"]``, which is the union
       bound on P(breach at some step): below ``tier_cutoffs[0]`` is "low", below
       ``tier_cutoffs[1]`` is "medium", below ``tier_cutoffs[2]`` is "high", and anything else
       is "critical". Gating on the upper bound means the tier is never understated. The lower
       bound, ``features["breach_prob_step_max_lower"]``, is reported in the reasons but does not
       set the tier. With no breach threshold supplied, the tier is "low" and the reason says so.
       A tier in ``approval_required_tiers`` gives NEEDS_APPROVAL.
    5. Otherwise EXECUTE.

    The policy never executes anything.
    """

    def __init__(
        self,
        utility: Callable[[str, Mapping[str, Any]], float],
        constraints: Sequence[Tuple[str, Callable[[str, Mapping[str, Any]], bool]]] = (),
        min_confidence: float = 0.6,
        max_forecast_uncertainty: float = 0.5,
        approval_required_tiers: Sequence[str] = ("high", "critical"),
        tier_fn: Optional[Callable[[str, Mapping[str, Any]], str]] = None,
        tier_cutoffs: Sequence[float] = (0.2, 0.5, 0.8),
    ) -> None:
        if not callable(utility):
            raise TypeError("utility must be callable as utility(option, features) -> float")
        self.utility = utility

        checked: List[Tuple[str, Callable[..., bool]]] = []
        for item in constraints:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError("each constraint must be a (name, predicate) pair")
            name, predicate = item
            if not callable(predicate):
                raise TypeError(f"constraint {name!r} predicate must be callable")
            checked.append((str(name), predicate))
        self.constraints = tuple(checked)

        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if float(max_forecast_uncertainty) < 0.0:
            raise ValueError("max_forecast_uncertainty must be >= 0")
        self.min_confidence = float(min_confidence)
        self.max_forecast_uncertainty = float(max_forecast_uncertainty)

        tiers = tuple(approval_required_tiers)
        unknown = [t for t in tiers if t not in RISK_TIERS]
        if unknown:
            raise ValueError(f"approval_required_tiers must be drawn from {RISK_TIERS}, got {unknown}")
        self.approval_required_tiers = tiers

        if tier_fn is not None and not callable(tier_fn):
            raise TypeError("tier_fn must be callable as tier_fn(option, features) -> tier")
        self.tier_fn = tier_fn

        cuts = tuple(float(c) for c in tier_cutoffs)
        if len(cuts) != 3 or not all(0.0 <= c <= 1.0 for c in cuts) or list(cuts) != sorted(cuts):
            raise ValueError("tier_cutoffs must be three non-decreasing values in [0, 1]")
        self.tier_cutoffs = cuts

    def risk_tier(self, option: str, features: Mapping[str, Any]) -> Tuple[str, str]:
        """Return ``(tier, reason)`` for one option."""
        if self.tier_fn is not None:
            tier = self.tier_fn(option, features)
            if tier not in RISK_TIERS:
                raise ValueError(f"tier_fn returned {tier!r}; expected one of {RISK_TIERS}")
            return tier, f"risk tier {tier!r} from tier_fn"
        upper = features.get("breach_prob_any_step_upper")
        if upper is None:
            return "low", "no breach threshold supplied; risk tier defaults to 'low'"
        lower = features.get("breach_prob_step_max_lower")
        lower_txt = "n/a" if lower is None else f"{float(lower):.3f}"
        bounds = (
            f"P(breach at some step) is between {lower_txt} (lower bound, max single step) and "
            f"{float(upper):.3f} (upper bound, union bound); the tier is gated on the upper bound"
        )
        for tier, cut in zip(("low", "medium", "high"), self.tier_cutoffs):
            if upper < cut:
                return tier, f"{bounds}, upper bound {upper:.3f} is below {cut}, risk tier {tier!r}"
        return "critical", (
            f"{bounds}, upper bound {upper:.3f} reaches {self.tier_cutoffs[2]}, risk tier 'critical'"
        )

    def evaluate(self, choice_result: Any, features: Mapping[str, Any]) -> PolicyOutcome:
        reasons: List[str] = []
        probabilities: Dict[str, float] = dict(choice_result.probabilities)

        admissible: List[str] = []
        for option in probabilities:
            failed = [name for name, predicate in self.constraints if not bool(predicate(option, features))]
            if failed:
                reasons.append(f"option {option!r} blocked by constraint(s): {', '.join(failed)}")
            else:
                admissible.append(option)

        if not admissible:
            if not probabilities:
                reasons.append("no options were supplied to the choice")
            return PolicyOutcome(PolicyStatus.BLOCKED, None, None, reasons, None)

        scored: List[Tuple[str, float, float]] = []
        for option in admissible:
            utility = float(self.utility(option, features))
            if not math.isfinite(utility):
                raise ValueError(f"utility of option {option!r} is not finite: {utility}")
            if utility < 0:
                raise ValueError(
                    f"utility of option {option!r} is negative ({utility:.6g}). Utilities must be "
                    "non-negative gains. The policy ranks options by P(option) * utility, a heuristic "
                    "over the engine's option probabilities and not a full cost model, so a negative "
                    "utility can let a low-probability option outrank a likely one. Represent costs "
                    "outside the policy, for example as constraints."
                )
            scored.append((option, probabilities[option] * utility, utility))

        best_option, best_eu, best_utility = max(scored, key=lambda item: item[1])
        tier, tier_reason = self.risk_tier(best_option, features)
        reasons.append(f"selected {best_option!r} with expected utility {best_eu:.4g} (utility {best_utility:.4g})")
        reasons.append(tier_reason)

        abstain_reasons: List[str] = []
        confidence = float(choice_result.confidence)
        if confidence < self.min_confidence:
            abstain_reasons.append(f"confidence {confidence:.3f} is below min_confidence {self.min_confidence}")
        uncertainty = float(features.get("uncertainty", math.inf))
        if not uncertainty <= self.max_forecast_uncertainty:
            abstain_reasons.append(
                f"forecast uncertainty {uncertainty:.3f} exceeds max_forecast_uncertainty {self.max_forecast_uncertainty}"
            )
        if _engine_abstains(choice_result):
            abstain_reasons.append("the engine's own abstention gate fired")
        if abstain_reasons:
            return PolicyOutcome(PolicyStatus.ABSTAIN, best_option, best_eu, reasons + abstain_reasons, tier)

        if tier in self.approval_required_tiers:
            reasons.append(f"risk tier {tier!r} requires human approval")
            return PolicyOutcome(PolicyStatus.NEEDS_APPROVAL, best_option, best_eu, reasons, tier)
        return PolicyOutcome(PolicyStatus.EXECUTE, best_option, best_eu, reasons, tier)


def _breach_bounds(step_probs: Sequence[float]) -> Tuple[float, float]:
    """Bounds on P(the path exceeds a threshold at some step), from per-step probabilities.

    Lower bound: the largest single-step probability, since the union is at least each member.
    Upper bound: the union bound, the sum of single-step probabilities capped at 1.
    The joint law of the steps is not modelled, so the true probability lies between them.
    """
    probs = [float(p) for p in step_probs]
    return max(probs), min(1.0, sum(probs))


def _build_features(
    forecasts: Mapping[str, Forecast],
    histories: Mapping[str, List[float]],
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    """Features the policy reads. Breach figures are bounds, not point probabilities.

    Per series: ``breach_prob_step_max_lower`` (max single-step probability) and
    ``breach_prob_any_step_upper`` (union bound, capped at 1). Top level: the same two bounds
    computed over every thresholded series and step together, so the upper bound covers any
    breach of any series. The risk tier is gated on ``breach_prob_any_step_upper``.
    """
    per_series: Dict[str, Dict[str, Any]] = {}
    all_step_probs: List[float] = []
    uncertainties: List[float] = []
    for name, fc in forecasts.items():
        last = histories[name][-1]
        lower, upper = fc.interval(0.8)
        end = fc.mean[-1]
        uncertainty = fc.uncertainty()
        entry: Dict[str, Any] = {
            "last_value": last,
            "median_end": end,
            "mean_path_change": end - last,
            "relative_path_change": (end - last) / max(abs(last), 1e-12),
            "interval_width_end": upper[-1] - lower[-1],
            "uncertainty": uncertainty,
            "threshold": None,
            "breach_prob_step_max_lower": None,
            "breach_prob_any_step_upper": None,
        }
        if name in thresholds:
            step_probs = fc.step_exceedance_probs(thresholds[name])
            series_lower, series_upper = _breach_bounds(step_probs)
            entry["threshold"] = thresholds[name]
            entry["breach_prob_step_max_lower"] = series_lower
            entry["breach_prob_any_step_upper"] = series_upper
            all_step_probs.extend(step_probs)
        per_series[name] = entry
        uncertainties.append(uncertainty)
    if all_step_probs:
        overall_lower, overall_upper = _breach_bounds(all_step_probs)
    else:
        overall_lower = overall_upper = None
    return {
        "series": per_series,
        "breach_prob_step_max_lower": overall_lower,
        "breach_prob_any_step_upper": overall_upper,
        "uncertainty": max(uncertainties),
    }


def _format_path(values: Sequence[float]) -> str:
    if len(values) <= 4:
        shown = [f"{v:.4g}" for v in values]
    else:
        shown = [f"{v:.4g}" for v in values[:3]] + ["..."] + [f"{values[-1]:.4g}"]
    return "[" + ", ".join(shown) + "]"


def _augmented_state(
    state: Any,
    forecasts: Mapping[str, Forecast],
    histories: Mapping[str, List[float]],
    thresholds: Mapping[str, float],
    method: str,
    horizon: int,
) -> str:
    base = state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)
    lines = [base, "", f"Forecast summary (method={method}, horizon={horizon}):"]
    for name, fc in forecasts.items():
        lower, upper = fc.interval(0.8)
        line = (
            f"- {name}: last={histories[name][-1]:.4g}; median path {_format_path(fc.mean)}; "
            f"80% band at horizon end [{lower[-1]:.4g}, {upper[-1]:.4g}]"
        )
        if name in thresholds:
            series_lower, series_upper = _breach_bounds(fc.step_exceedance_probs(thresholds[name]))
            line += (
                f"; P({name} > {thresholds[name]:.4g} at some step) is between {series_lower:.3f} "
                f"(max single step, lower bound) and {series_upper:.3f} (union bound, upper bound)"
            )
        lines.append(line)
    return "\n".join(lines)


@dataclass
class FutureAwareDecision:
    """One forecast-aware decision and everything that produced it."""

    decision_id: str
    created_at: str
    question: str
    options: List[str]
    horizon: int
    thresholds: Dict[str, float]
    forecasts: Dict[str, Forecast]
    features: Dict[str, Any]
    augmented_state: str
    choice: "ChoiceResult"
    outcome: PolicyOutcome
    executed_at: Optional[str] = None
    executed_option: Optional[str] = None
    execution_error: Optional[str] = None
    # The decider's private approval registry (decision_id -> approval record). Set by the
    # decider. ``approval`` reads from it. ``execute`` gates on the decider's own registry,
    # never on this reference or on any attribute of the decision.
    _approvals: Optional[Mapping[str, Dict[str, str]]] = field(default=None, repr=False, compare=False)

    @property
    def approval(self) -> Optional[Dict[str, str]]:
        """Read-only copy of the approval recorded for this decision, or None.

        Assigning to this attribute raises AttributeError. Approvals are written only by
        ``ForecastAwareDecider.approve``.
        """
        if self._approvals is None:
            return None
        entry = self._approvals.get(self.decision_id)
        return None if entry is None else dict(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "created_at": self.created_at,
            "question": self.question,
            "options": list(self.options),
            "horizon": self.horizon,
            "thresholds": dict(self.thresholds),
            "forecasts": {n: f.to_dict() for n, f in self.forecasts.items()},
            "features": self.features,
            "augmented_state": self.augmented_state,
            "choice": {
                "choice_label": self.choice.choice_label,
                "probabilities": dict(self.choice.probabilities),
                "confidence": float(self.choice.confidence),
            },
            "outcome": self.outcome.to_dict(),
            "approval": self.approval,
            "executed_at": self.executed_at,
            "executed_option": self.executed_option,
            "execution_error": self.execution_error,
        }


class ForecastAwareDecider:
    """Runs the forecast-aware decision pipeline around a ``RhenEngine``.

    Args:
        engine: Any object with ``choice(state, question, options)``, such as ``RhenEngine``.
        forecaster: Any ``Forecaster``, such as ``NaiveQuantileForecaster`` or ``TimesFMForecaster``.
        policy: A ``DecisionPolicy``. The default uses a constant utility of 1.0, which ranks
            options by engine probability alone. Set a real utility for real use.
        ledger: Optional ``FeedbackLedger``. Decisions are recorded in it when given.

    Approvals are kept in a private registry on this decider, keyed by decision_id, and written
    only by ``approve``. ``execute`` gates NEEDS_APPROVAL decisions on that registry alone. Setting
    an ``approval`` attribute on a decision does not authorise anything. This is an in-process
    guard, not a security boundary against code that can modify the process.
    """

    def __init__(
        self,
        engine: Any,
        forecaster: Forecaster,
        policy: Optional[DecisionPolicy] = None,
        ledger: Optional["FeedbackLedger"] = None,
    ) -> None:
        self.engine = engine
        self.forecaster = forecaster
        self.policy = policy if policy is not None else DecisionPolicy(utility=_unit_utility)
        self.ledger = ledger
        self._decisions: Dict[str, FutureAwareDecision] = {}
        self._approvals: Dict[str, Dict[str, str]] = {}

    def decide(
        self,
        state: Any,
        question: str,
        options: Sequence[str],
        series: Mapping[str, Sequence[float]],
        horizon: int,
        thresholds: Optional[Mapping[str, float]] = None,
    ) -> FutureAwareDecision:
        option_list = list(options)
        if not option_list or not all(isinstance(o, str) for o in option_list):
            raise ValueError("options must be a non-empty list of strings")
        if len(set(option_list)) != len(option_list):
            raise ValueError("options must be unique")
        if not series:
            raise ValueError("series must name at least one history")
        horizon = _check_horizon(horizon)

        names = list(series.keys())
        histories = {name: [float(v) for v in series[name]] for name in names}
        thresholds_map = {str(k): float(v) for k, v in (thresholds or {}).items()}
        unknown = [k for k in thresholds_map if k not in histories]
        if unknown:
            raise ValueError(f"thresholds name series that were not supplied: {unknown}")

        predicted = list(self.forecaster.predict([histories[n] for n in names], horizon))
        if len(predicted) != len(names):
            raise RuntimeError(f"forecaster returned {len(predicted)} forecasts for {len(names)} series")
        forecasts = {name: replace(fc, name=name) for name, fc in zip(names, predicted)}
        method = str(getattr(self.forecaster, "name", type(self.forecaster).__name__))
        missing_quantiles = [name for name, fc in forecasts.items() if fc.quantiles is None]
        if missing_quantiles:
            raise ValueError(
                f"forecaster {method!r} returned no quantiles for {missing_quantiles}; "
                "the decision pipeline needs quantiles for interval and breach-risk features"
            )

        features = _build_features(forecasts, histories, thresholds_map)
        augmented = _augmented_state(state, forecasts, histories, thresholds_map, method, horizon)

        choice = self.engine.choice(augmented, question, option_list)
        outcome = self.policy.evaluate(choice, features)

        decision = FutureAwareDecision(
            decision_id=f"fad-{uuid.uuid4().hex}",
            created_at=_utc_now(),
            question=question,
            options=option_list,
            horizon=horizon,
            thresholds=thresholds_map,
            forecasts=forecasts,
            features=features,
            augmented_state=augmented,
            choice=choice,
            outcome=outcome,
            _approvals=self._approvals,
        )
        self._decisions[decision.decision_id] = decision
        if self.ledger is not None:
            self.ledger.record_decision(decision)
        return decision

    def _lookup(self, decision_id: str) -> FutureAwareDecision:
        try:
            return self._decisions[decision_id]
        except KeyError:
            raise DecisionGateError(
                f"unknown decision_id {decision_id!r}; approve and execute act only on decisions made by this decider"
            ) from None

    def approve(self, decision_id: str, approver: str) -> Dict[str, str]:
        """Record an explicit human approval for a NEEDS_APPROVAL decision.

        The approval (approver and UTC timestamp) is written to this decider's private registry,
        and mirrored to the ledger when one is set. Returns a copy of the registry entry.
        """
        decision = self._lookup(decision_id)
        if not isinstance(approver, str) or not approver.strip():
            raise DecisionGateError("approver must be a non-empty name")
        if decision.outcome.status is not PolicyStatus.NEEDS_APPROVAL:
            raise DecisionGateError(
                f"decision {decision_id} has status {decision.outcome.status.value}; "
                "only NEEDS_APPROVAL decisions can be approved"
            )
        if decision_id in self._approvals:
            raise DecisionGateError(
                f"decision {decision_id} is already approved by {self._approvals[decision_id]['approver']!r}"
            )
        name = approver.strip()
        approved_at = _utc_now()
        if self.ledger is not None:
            self.ledger.record_approval(decision_id, name, approved_at)
        self._approvals[decision_id] = {"approver": name, "approved_at": approved_at}
        return dict(self._approvals[decision_id])

    def execute(self, decision: FutureAwareDecision, executor: Callable[[str, FutureAwareDecision], Any]) -> Any:
        """Run ``executor(option, decision)`` if the gate allows it.

        EXECUTE runs immediately. NEEDS_APPROVAL runs only when this decider's approval registry
        holds an entry for the decision, which ``approve`` writes. Nothing on the decision object
        is consulted for this check. BLOCKED and ABSTAIN are refused. Each decision executes at
        most once. The execution is marked before the executor runs, so a failed executor is not
        retried automatically. This is an in-process guard, not a security boundary.
        """
        if not isinstance(decision, FutureAwareDecision):
            raise TypeError("decision must be a FutureAwareDecision produced by this decider")
        if not callable(executor):
            raise TypeError("executor must be callable as executor(option, decision)")
        if decision.executed_at is not None:
            raise DecisionGateError(f"decision {decision.decision_id} has already been executed")
        status = decision.outcome.status
        if status is PolicyStatus.NEEDS_APPROVAL and decision.decision_id not in self._approvals:
            raise DecisionGateError(f"decision {decision.decision_id} needs human approval before execution")
        if status not in (PolicyStatus.EXECUTE, PolicyStatus.NEEDS_APPROVAL):
            raise DecisionGateError(
                f"decision {decision.decision_id} has status {status.value} and cannot be executed"
            )
        option = decision.outcome.option
        if option is None:
            raise DecisionGateError(f"decision {decision.decision_id} has no option to execute")

        executed_at = _utc_now()
        decision.executed_at = executed_at
        decision.executed_option = option
        try:
            result = executor(option, decision)
        except Exception as exc:
            decision.execution_error = repr(exc)
            if self.ledger is not None:
                self.ledger.record_execution(decision.decision_id, option, executed_at, error=repr(exc))
            raise
        if self.ledger is not None:
            self.ledger.record_execution(decision.decision_id, option, executed_at)
        return result


@dataclass
class DecisionRecord:
    """Ledger entry for one decision. Plain values only, so it serializes to JSON."""

    decision_id: str
    created_at: str
    question: str
    options: List[str]
    horizon: int
    thresholds: Dict[str, float]
    forecasts: Dict[str, Forecast]
    choice_label: str
    probabilities: Dict[str, float]
    confidence: float
    status: str
    option: Optional[str]
    expected_utility: Optional[float]
    risk_tier: Optional[str]
    features: Dict[str, Any]
    approval: Optional[Dict[str, str]] = None
    executed_option: Optional[str] = None
    executed_at: Optional[str] = None
    execution_error: Optional[str] = None
    outcome: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "created_at": self.created_at,
            "question": self.question,
            "options": list(self.options),
            "horizon": self.horizon,
            "thresholds": dict(self.thresholds),
            "forecasts": {n: f.to_dict() for n, f in self.forecasts.items()},
            "choice_label": self.choice_label,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "status": self.status,
            "option": self.option,
            "expected_utility": self.expected_utility,
            "risk_tier": self.risk_tier,
            "features": self.features,
            "approval": None if self.approval is None else dict(self.approval),
            "executed_option": self.executed_option,
            "executed_at": self.executed_at,
            "execution_error": self.execution_error,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionRecord":
        return cls(
            decision_id=str(data["decision_id"]),
            created_at=str(data["created_at"]),
            question=str(data["question"]),
            options=list(data["options"]),
            horizon=int(data["horizon"]),
            thresholds={str(k): float(v) for k, v in data["thresholds"].items()},
            forecasts={n: Forecast.from_dict(f) for n, f in data["forecasts"].items()},
            choice_label=str(data["choice_label"]),
            probabilities={str(k): float(v) for k, v in data["probabilities"].items()},
            confidence=float(data["confidence"]),
            status=str(data["status"]),
            option=data.get("option"),
            expected_utility=data.get("expected_utility"),
            risk_tier=data.get("risk_tier"),
            features=dict(data.get("features") or {}),
            approval=data.get("approval"),
            executed_option=data.get("executed_option"),
            executed_at=data.get("executed_at"),
            execution_error=data.get("execution_error"),
            outcome=data.get("outcome"),
        )


def _as_realized(values: Sequence[float], horizon: int, name: str) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if y.size != horizon:
        raise ValueError(f"realized values for {name!r} must have {horizon} entries, got {y.size}")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"realized values for {name!r} contain non-finite numbers")
    return y


class FeedbackLedger:
    """In-memory record of decisions and their realized outcomes, with JSONL persistence."""

    def __init__(self) -> None:
        self._records: List[DecisionRecord] = []
        self._index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[DecisionRecord]:
        return list(self._records)

    def get(self, decision_id: str) -> DecisionRecord:
        try:
            return self._records[self._index[decision_id]]
        except KeyError:
            raise KeyError(f"no ledger record for decision_id {decision_id!r}") from None

    def _add(self, record: DecisionRecord) -> DecisionRecord:
        if record.decision_id in self._index:
            raise ValueError(f"decision_id {record.decision_id!r} is already in the ledger")
        self._index[record.decision_id] = len(self._records)
        self._records.append(record)
        return record

    def record_decision(self, decision: FutureAwareDecision) -> DecisionRecord:
        outcome = decision.outcome
        return self._add(
            DecisionRecord(
                decision_id=decision.decision_id,
                created_at=decision.created_at,
                question=decision.question,
                options=list(decision.options),
                horizon=decision.horizon,
                thresholds=dict(decision.thresholds),
                forecasts=dict(decision.forecasts),
                choice_label=decision.choice.choice_label,
                probabilities={str(k): float(v) for k, v in decision.choice.probabilities.items()},
                confidence=float(decision.choice.confidence),
                status=outcome.status.value,
                option=outcome.option,
                expected_utility=outcome.expected_utility,
                risk_tier=outcome.risk_tier,
                features=json.loads(json.dumps(decision.features, default=str)),
            )
        )

    def record_approval(self, decision_id: str, approver: str, approved_at: str) -> None:
        record = self.get(decision_id)
        if record.approval is not None:
            raise DecisionGateError(f"decision {decision_id} is already approved")
        record.approval = {"approver": approver, "approved_at": approved_at}

    def record_execution(self, decision_id: str, option: str, executed_at: str, error: Optional[str] = None) -> None:
        record = self.get(decision_id)
        record.executed_option = option
        record.executed_at = executed_at
        record.execution_error = error

    def record_outcome(self, decision_id: str, realized_series: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
        """Score a decision against realized values for every forecast series.

        Per series, this records the pinball loss over the nine quantile levels, the number
        of steps inside the 80% and 50% bands, and, when a threshold was set, both bounds on
        the predicted probability of a breach at some step (``predicted_breach_lower`` and
        ``predicted_breach_upper``), the realized breach (any step above the threshold), and a
        Brier score under each bound (``brier_lower``, ``brier_upper``).
        """
        record = self.get(decision_id)
        if record.outcome is not None:
            raise ValueError(f"an outcome is already recorded for decision {decision_id}")
        missing = [n for n in record.forecasts if n not in realized_series]
        if missing:
            raise ValueError(f"realized_series is missing series: {missing}")

        per_series: Dict[str, Dict[str, Any]] = {}
        for name, fc in record.forecasts.items():
            y = _as_realized(realized_series[name], fc.horizon, name)
            entry: Dict[str, Any] = {"horizon": fc.horizon}
            if fc.quantiles is not None:
                lower80, upper80 = fc.interval(0.8)
                lower50, upper50 = fc.interval(0.5)
                entry["pinball_loss"] = pinball_loss(fc.quantiles, y, QUANTILE_LEVELS)
                entry["covered_80"] = int(sum(lo <= v <= hi for lo, hi, v in zip(lower80, upper80, y)))
                entry["covered_50"] = int(sum(lo <= v <= hi for lo, hi, v in zip(lower50, upper50, y)))
            threshold = record.thresholds.get(name)
            if threshold is not None and fc.quantiles is not None:
                lower, upper = _breach_bounds(fc.step_exceedance_probs(threshold))
                occurred = bool(np.any(y > threshold))
                entry.update(
                    threshold=threshold,
                    predicted_breach_lower=lower,
                    predicted_breach_upper=upper,
                    realized_breach=occurred,
                    brier_lower=brier_score(lower, occurred),
                    brier_upper=brier_score(upper, occurred),
                )
            per_series[name] = entry

        record.outcome = {"recorded_at": _utc_now(), "series": per_series}
        return record.outcome

    def calibration_report(self) -> Dict[str, Any]:
        """Aggregate calibration across every decision that has an outcome.

        Pinball loss and coverage are weighted by horizon steps. Coverage targets are 0.8 and
        0.5. Breach figures are averaged over outcomes with a threshold. The true breach
        probability lies between the lower and upper bounds, so each Brier score treats one
        bound as the forecast probability, and neither bound is a calibrated probability on its
        own. This is a calibration summary, not evidence that decisions were better.
        """
        scored = [r for r in self._records if r.outcome is not None]
        acc: Dict[str, Dict[str, float]] = {}
        for rec in scored:
            for name, entry in rec.outcome["series"].items():
                a = acc.setdefault(name, {
                    "outcomes": 0, "quantile_steps": 0, "pinball_sum": 0.0, "covered_80": 0, "covered_50": 0,
                    "breach_outcomes": 0, "brier_lower_sum": 0.0, "brier_upper_sum": 0.0,
                    "predicted_lower_sum": 0.0, "predicted_upper_sum": 0.0, "realized_breaches": 0,
                })
                a["outcomes"] += 1
                if "pinball_loss" in entry:
                    h = entry["horizon"]
                    a["quantile_steps"] += h
                    a["pinball_sum"] += entry["pinball_loss"] * h
                    a["covered_80"] += entry["covered_80"]
                    a["covered_50"] += entry["covered_50"]
                if "brier_lower" in entry:
                    a["breach_outcomes"] += 1
                    a["brier_lower_sum"] += entry["brier_lower"]
                    a["brier_upper_sum"] += entry["brier_upper"]
                    a["predicted_lower_sum"] += entry["predicted_breach_lower"]
                    a["predicted_upper_sum"] += entry["predicted_breach_upper"]
                    a["realized_breaches"] += int(entry["realized_breach"])

        series_report: Dict[str, Dict[str, Any]] = {}
        for name, a in acc.items():
            steps = a["quantile_steps"]
            nb = a["breach_outcomes"]
            series_report[name] = {
                "outcomes": int(a["outcomes"]),
                "quantile_steps": int(steps),
                "mean_pinball_loss": (a["pinball_sum"] / steps) if steps else None,
                "coverage_80": (a["covered_80"] / steps) if steps else None,
                "target_coverage_80": 0.8,
                "coverage_50": (a["covered_50"] / steps) if steps else None,
                "target_coverage_50": 0.5,
                "breach_outcomes": int(nb),
                "mean_brier_lower": (a["brier_lower_sum"] / nb) if nb else None,
                "mean_brier_upper": (a["brier_upper_sum"] / nb) if nb else None,
                "mean_predicted_breach_lower": (a["predicted_lower_sum"] / nb) if nb else None,
                "mean_predicted_breach_upper": (a["predicted_upper_sum"] / nb) if nb else None,
                "realized_breach_rate": (a["realized_breaches"] / nb) if nb else None,
            }
        status_counts: Dict[str, int] = {}
        for rec in self._records:
            status_counts[rec.status] = status_counts.get(rec.status, 0) + 1
        return {
            "decisions": len(self._records),
            "decisions_with_outcomes": len(scored),
            "status_counts": status_counts,
            "series": series_report,
        }

    def to_jsonl(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "FeedbackLedger":
        ledger = cls()
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    ledger._add(DecisionRecord.from_dict(json.loads(line)))
        return ledger


__all__ = [
    "PolicyStatus",
    "DecisionGateError",
    "PolicyOutcome",
    "DecisionPolicy",
    "FutureAwareDecision",
    "ForecastAwareDecider",
    "DecisionRecord",
    "FeedbackLedger",
]
