"""
Rhen Latent Decision Engine - Pythonic Typed SDK.

High-level interface for non-autoregressive parallel decisions over structured state:
- Noul: Binary semantic truth proposition evaluation (P in [0, 1])
- Choice: Categorical 1-of-K option selection with calibrated probability simplex
- Score: Continuous expected value estimation over ordered discrete spectrum
- MultiChoice: Multi-label selection above calibrated threshold
- Range: Continuous expected value and interval estimation
- Extract: Latent slot/span value extraction from state
- Cascade: Chained multi-stage conditional decision graphs
- RhenEngine: Main orchestration engine for O(1) state forward passes
"""

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import hashlib
import json
import logging
import math

import torch

try:
    from rhen.meta_cognitive_monitor import ConformalPredictionEngine, EnergyOODDetector
    from rhen.rhen_packer import PackedRhenSequence, get_rhen_packer
except ImportError:
    from meta_cognitive_monitor import ConformalPredictionEngine, EnergyOODDetector
    from rhen_packer import PackedRhenSequence, get_rhen_packer

logger = logging.getLogger(__name__)


def _robust_serialize_value(val: Any) -> Any:
    """Robust serializer handling torch.Tensor, numpy types, sets, dataclasses, dicts, lists."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.detach().cpu().tolist()
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


class ResultBase:
    """Base class providing serialization, deserialization, and abstention helpers for Rhen results."""

    @property
    def should_abstain(self) -> bool:
        """Whether the engine abstained or requested clarification on this result."""
        verdict = getattr(self, "abstention_verdict", None)
        if verdict is not None:
            return verdict.should_abstain
        probs = (
            getattr(self, "probabilities", None)
            or getattr(self, "raw_probabilities", None)
            or getattr(self, "level_probabilities", None)
        )
        gate = GeometricAbstentionGate()
        v = gate.evaluate(
            probabilities=probs,
            free_energy=getattr(self, "free_energy", None),
            conformal_set=getattr(self, "conformal_set", None),
        )
        return v.should_abstain

    @property
    def should_answer(self) -> bool:
        """Whether a decision may be returned to the caller."""
        return not self.should_abstain

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            d[f.name] = _robust_serialize_value(val)
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Reconstruct result from dictionary."""
        kwargs = dict(data)
        if "schema_mutation" in kwargs and isinstance(kwargs["schema_mutation"], dict):
            kwargs["schema_mutation"] = SchemaMutation.from_dict(kwargs["schema_mutation"])
        if "abstention_verdict" in kwargs and isinstance(kwargs["abstention_verdict"], dict):
            kwargs["abstention_verdict"] = AbstentionVerdict.from_dict(kwargs["abstention_verdict"])
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, data_str: str) -> Any:
        """Reconstruct result from JSON string."""
        return cls.from_dict(json.loads(data_str))

#: Schema mutation types, in the order of the schema head's logits.
SCHEMA_MUTATION_TYPES = (
    "spawn_child",
    "switch_primitive",
    "terminate",
    "route_tool",
    "refine",
    "verify",
    "escalate",
    "noop",
)


def _stable_seed(*parts: Any) -> int:
    """Process-independent 31-bit seed.

    ``hash()`` is salted per process (PYTHONHASHSEED), so seeding latents with it
    makes the same state produce different vectors in different runs.
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


@dataclass
class SchemaMutation:
    parent_question: str
    mutation_type: str  # one of SCHEMA_MUTATION_TYPES
    mutation_logits: List[float]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_question": self.parent_question,
            "mutation_type": self.mutation_type,
            "mutation_logits": _robust_serialize_value(self.mutation_logits),
            "confidence": float(self.confidence),
            "metadata": _robust_serialize_value(self.metadata) if self.metadata else None,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaMutation":
        return cls(
            parent_question=data["parent_question"],
            mutation_type=data["mutation_type"],
            mutation_logits=list(data.get("mutation_logits", [])),
            confidence=float(data.get("confidence", 0.0)),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, data_str: str) -> "SchemaMutation":
        return cls.from_dict(json.loads(data_str))


@dataclass
class LatentRolloutResult:
    state_summary: str
    candidate_options: List[str]
    horizon: int
    prospective_values: Dict[str, float]
    best_option: str
    best_value: float
    anticipated_trajectories: Optional[Dict[str, Any]] = None
    # How the values were produced. Only the calibrated-choice projection exists today.
    source: str = "calibrated_choice_projection"


@dataclass
class LatentSearchResult:
    """Result of a multi-step System Two latent beam search.

    ``fallbacks`` counts user-callback failures that were replaced by the explicit
    heuristic (each one is also logged). ``source`` is ``"heuristic"`` when no
    user callable produced a value, ``"callable"`` when every user-callable call
    succeeded, and ``"mixed"`` when some calls fell back and some succeeded.
    """
    best_trajectory: List[Any]
    best_action_sequence: List[Union[str, int]]
    prospective_value: float
    step_rewards: List[float]
    search_tree: Dict[str, Any]
    conformal_bound: Optional[float] = None
    epistemic_variance: Optional[float] = None
    fallbacks: int = 0
    source: str = "heuristic"


@dataclass
class Noul:
    """Binary semantic proposition (Yes/No)."""
    proposition: str

    def to_tuple(self) -> Tuple[str, List[str]]:
        return (self.proposition, ["Yes", "No"])


@dataclass
class NoulResult(ResultBase):
    """Result of a Noul binary evaluation."""
    proposition: str
    p_true: float
    decision: bool
    confidence: float
    temperature: float
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None

    def __bool__(self) -> bool:
        return self.decision


@dataclass
class Choice:
    """Categorical 1-of-K selection."""
    question: str
    options: List[str]

    def __post_init__(self):
        if not self.options or len(self.options) == 0:
            raise ValueError("Choice requires at least one option")

    def to_tuple(self) -> Tuple[str, List[str]]:
        return (self.question, self.options)


@dataclass
class ChoiceResult(ResultBase):
    """Result of a categorical Choice evaluation."""
    question: str
    choice_index: int
    choice_label: str
    probabilities: Dict[str, float]
    raw_probabilities: List[float]
    confidence: float
    temperature: float
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None


@dataclass
class MultiChoice:
    """Multi-label selection above calibrated threshold."""
    question: str
    options: List[str]
    threshold: float = 0.5

    def __post_init__(self):
        if not self.options or len(self.options) == 0:
            raise ValueError("Choice requires at least one option")

    def to_tuple(self):
        return (self.question, self.options, "multichoice", {"threshold": self.threshold})


@dataclass
class MultiChoiceResult(ResultBase):
    """Result of a MultiChoice evaluation."""
    question: str
    selected_indices: List[int]
    selected_options: List[str]
    probabilities: Dict[str, float]
    raw_probabilities: List[float]
    confidence: float
    temperature: float
    threshold: float = 0.5
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None


@dataclass
class Range:
    """Continuous expected value and interval estimation."""
    question: str
    min_val: float = 0.0
    max_val: float = 100.0

    def to_tuple(self):
        return (self.question, [str(self.min_val), str(self.max_val)], "range", {"min_val": self.min_val, "max_val": self.max_val})


@dataclass
class RangeResult(ResultBase):
    """Result of a Range evaluation."""
    question: str
    expected_value: float
    interval_low: float
    interval_high: float
    confidence_interval: float  # e.g. 0.95
    min_val: float
    max_val: float
    confidence: float
    temperature: float
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None


@dataclass
class Extract:
    """Latent slot/span value extraction from state."""
    slot_name: str
    query: str = ""

    def to_tuple(self):
        return (self.query or self.slot_name, [], "extract", {"slot_name": self.slot_name})


@dataclass
class ExtractResult(ResultBase):
    """Result of an Extract evaluation."""
    slot_name: str
    extracted_value: str
    start_idx: int
    end_idx: int
    confidence: float
    temperature: float
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None


@dataclass
class Score:
    """Continuous expected value over an ordered discrete level spectrum."""
    question: str
    levels: List[str] = field(default_factory=lambda: ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"])
    min_score: float = 0.0
    max_score: float = 10.0

    def to_tuple(self) -> Tuple[str, List[str]]:
        return (self.question, self.levels)


@dataclass
class ScoreResult(ResultBase):
    """Result of a Score evaluation."""
    question: str
    score: float
    min_score: float
    max_score: float
    level_probabilities: List[float]
    confidence: float
    temperature: float
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None


@dataclass
class Cascade:
    """Chained multi-stage conditional decision graph."""
    stages: List[Union[Noul, Choice, Score, MultiChoice, Range, Extract]]
    id: Optional[str] = None

    def to_tuple(self) -> Tuple[str, List[str]]:
        return (f"Cascade: {self.id or 'multi-stage'}", [f"Stage {i}" for i in range(len(self.stages))])


@dataclass
class CascadeResult(ResultBase):
    """Result of a multi-stage conditional decision cascade."""
    id: Optional[str]
    stages: List[Union[NoulResult, ChoiceResult, ScoreResult, MultiChoiceResult, RangeResult, ExtractResult]]
    decisions: Dict[str, Any]
    confidence: float
    temperature: float
    meta_reasoning_context: Optional[Dict] = None
    reasoning_confidence: Optional[float] = None
    alternative_hypotheses: List[Dict] = field(default_factory=list)
    risk_assessment: Optional[str] = None
    quality_score: Optional[float] = None
    conformal_set: Optional[Dict[str, Any]] = None
    free_energy: Optional[float] = None
    schema_mutation: Optional[SchemaMutation] = None
    abstention_verdict: Optional["AbstentionVerdict"] = None

    def __getitem__(self, key: str) -> Any:
        return self.decisions[key]

    def __len__(self) -> int:
        return len(self.stages)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CascadeResult":
        kwargs = dict(data)
        if "stages" in kwargs and isinstance(kwargs["stages"], list):
            reconstructed_stages = []
            for s in kwargs["stages"]:
                if isinstance(s, dict):
                    if "proposition" in s:
                        reconstructed_stages.append(NoulResult.from_dict(s))
                    elif "choice_index" in s or "choice_label" in s:
                        reconstructed_stages.append(ChoiceResult.from_dict(s))
                    elif "selected_indices" in s or "selected_options" in s:
                        reconstructed_stages.append(MultiChoiceResult.from_dict(s))
                    elif "level_probabilities" in s:
                        reconstructed_stages.append(ScoreResult.from_dict(s))
                    elif "expected_value" in s or "interval_low" in s:
                        reconstructed_stages.append(RangeResult.from_dict(s))
                    elif "slot_name" in s:
                        reconstructed_stages.append(ExtractResult.from_dict(s))
                    elif "stages" in s:
                        reconstructed_stages.append(CascadeResult.from_dict(s))
                    else:
                        reconstructed_stages.append(s)
                else:
                    reconstructed_stages.append(s)
            kwargs["stages"] = reconstructed_stages

        if "schema_mutation" in kwargs and isinstance(kwargs["schema_mutation"], dict):
            kwargs["schema_mutation"] = SchemaMutation.from_dict(kwargs["schema_mutation"])
        if "abstention_verdict" in kwargs and isinstance(kwargs["abstention_verdict"], dict):
            kwargs["abstention_verdict"] = AbstentionVerdict.from_dict(kwargs["abstention_verdict"])

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)


# --------------------------------------------------------------------------------------
# Unified Geometry-Based Selective Abstention
# --------------------------------------------------------------------------------------

#: Decision labels emitted by the abstention gate.
ANSWER = "Answer"
ABSTAIN = "Abstain"
NEED_CLARIFICATION = "NeedClarification"


@dataclass
class AbstentionVerdict:
    """Outcome of the unified geometry-based selective abstention gate.

    Attributes:
        decision: One of ``"Answer"``, ``"Abstain"`` or ``"NeedClarification"``.
        reasons: Human-readable names of every detector that fired.
        free_energy: Helmholtz free energy, when available.
        conformal_size: ``|C(S)|``, when a conformal set was supplied.
        hull_distance: Distance from the query latent to the candidate convex hull.
        max_probability: Largest calibrated option probability.
    """

    decision: str
    reasons: List[str] = field(default_factory=list)
    free_energy: Optional[float] = None
    conformal_size: Optional[int] = None
    hull_distance: Optional[float] = None
    max_probability: Optional[float] = None

    @property
    def should_answer(self) -> bool:
        """Whether a decision may be returned to the caller."""
        return self.decision == ANSWER

    @property
    def should_abstain(self) -> bool:
        """Whether the engine must refuse rather than guess."""
        return self.decision in (ABSTAIN, NEED_CLARIFICATION)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "free_energy": self.free_energy,
            "conformal_size": self.conformal_size,
            "hull_distance": self.hull_distance,
            "max_probability": self.max_probability,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AbstentionVerdict":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    @classmethod
    def from_json(cls, data_str: str) -> "AbstentionVerdict":
        return cls.from_dict(json.loads(data_str))


def _affine_min_norm_weights(points: torch.Tensor) -> torch.Tensor:
    """Weights ``alpha`` with ``sum(alpha) = 1`` minimising ``||points^T alpha||``.

    ``points`` is ``[m, D]``. The solution of the KKT system is unique for an
    affinely independent set; a singular system falls back to least squares.
    """
    m = points.shape[0]
    kkt = torch.zeros(m + 1, m + 1, dtype=points.dtype)
    kkt[:m, :m] = points @ points.transpose(0, 1)
    kkt[:m, m] = 1.0
    kkt[m, :m] = 1.0
    rhs = torch.zeros(m + 1, dtype=points.dtype)
    rhs[m] = 1.0
    try:
        sol = torch.linalg.solve(kkt, rhs)
    except RuntimeError:
        sol = torch.linalg.pinv(kkt) @ rhs
    return sol[:m]


def _min_norm_point(points: torch.Tensor, max_iter_scale: int = 20) -> torch.Tensor:
    """Exact minimum-norm point of ``conv(rows of points)`` (Wolfe, 1976).

    The minimum-norm point is the Euclidean projection of the origin onto the
    hull, so its norm is the distance from the origin to the hull. Wolfe's
    active-set method reaches it in finitely many steps, which makes it exact
    where projected gradient descent is only sublinear: a query inside the hull
    is recognised as inside (distance ~ 0) regardless of conditioning.
    """
    pts = points.to(torch.float64)
    norms = (pts * pts).sum(dim=1)
    tol = 1e-12 * max(1.0, float(norms.max()))
    start = int(torch.argmin(norms))
    corral = [start]
    lam = torch.ones(1, dtype=torch.float64)
    x = pts[start].clone()
    budget = max_iter_scale * (pts.shape[0] ** 2) + 100
    for _ in range(budget):
        # Optimality: <x, p_j> >= ||x||^2 for every point. Otherwise add the worst one.
        j = int(torch.argmin(pts @ x))
        if float(x @ x) - float(x @ pts[j]) <= tol or j in corral:
            break
        corral.append(j)
        lam = torch.cat([lam, lam.new_zeros(1)])
        # Minor cycle: move the corral's affine minimiser back into the simplex.
        for _inner in range(budget):
            alpha = _affine_min_norm_weights(pts[corral])
            if bool((alpha >= -1e-12).all()):
                lam = alpha.clamp(min=0.0)
                lam = lam / lam.sum()
                break
            neg = alpha < 0
            theta = float((lam[neg] / (lam[neg] - alpha[neg])).min())
            lam = lam + theta * (alpha - lam)
            keep = lam > 1e-12
            corral = [corral[i] for i in range(len(corral)) if bool(keep[i])]
            lam = lam[keep]
            lam = lam / lam.sum()
        x = (lam.unsqueeze(1) * pts[corral]).sum(dim=0)
    return x


class GeometricAbstentionGate:
    """Combines free energy, conformal sets and latent convex-hull geometry.

    A decision engine that always emits an answer is a hallucination engine. This
    gate is the refusal path, and it fires on the union of three *independent*
    failure detectors -- each covering a case the others structurally cannot see:

    ==================  =======================================================
    Detector            Catches
    ==================  =======================================================
    Free energy         A state far off-manifold: every answer is a guess.
    Conformal set       No candidate carries enough mass to commit to
                       (``|C(S)| = 0``).
    Convex hull         The query lies outside the hull of the candidate
                       latents -- the state supports *no* candidate, even when
                       the probabilities look confident.
    ==================  =======================================================

    The convex-hull detector is the one that matters most in practice: a softmax
    can be confidently wrong, because it always produces a distribution. Checking
    whether the query is actually *inside* the region the options span catches
    that specific failure.

    Args:
        free_energy_threshold: ``E`` above which the state is off-manifold.
        max_conformal_set_size: At or below this, the conformal set is treated as
            uninformative.
        hull_distance_threshold: Distance outside the hull that counts as
            unsupported.
        min_margin: Below this probability margin the top two options are
            treated as indistinguishable, which warrants clarification rather
            than abstention.
    """

    def __init__(
        self,
        free_energy_threshold: float = 4.5,
        max_conformal_set_size: int = 0,
        hull_distance_threshold: float = 1e-3,
        min_margin: float = 0.10,
    ) -> None:
        self.free_energy_threshold = float(free_energy_threshold)
        self.max_conformal_set_size = int(max_conformal_set_size)
        self.hull_distance_threshold = float(hull_distance_threshold)
        self.min_margin = float(min_margin)

    @staticmethod
    def _as_matrix(latents: Sequence[Sequence[float]]) -> torch.Tensor:
        return torch.as_tensor(latents, dtype=torch.float64)

    def convex_hull_distance(
        self,
        option_latents: Sequence[Sequence[float]],
        query_latent: Sequence[float],
    ) -> float:
        """Signed distance from ``query_latent`` to the candidates' convex hull.

        Returns ``<= 0`` when the query is inside the hull, and the (positive)
        distance to the closest point on the hull when it is outside.

        The distance from ``q`` to ``conv(z_1..z_K)`` equals the norm of the
        minimum-norm point of ``conv(z_i - q)``, which is computed exactly by
        Wolfe's active-set method. ``K`` is small (typically 2-8), so no Qhull
        construction is needed, and the result does not depend on grid resolution
        or on iteration counts: a query that is a convex combination of the
        candidates is always reported inside.

        Args:
            option_latents: ``[K, D]`` candidate option latents.
            query_latent: ``[D]`` query / state latent.

        Returns:
            Negative inside the hull, positive outside.
        """
        opts = self._as_matrix(option_latents)
        if opts.ndim != 2 or opts.shape[0] == 0:
            return 0.0

        q = torch.as_tensor(query_latent, dtype=torch.float64).reshape(-1)
        if q.numel() != opts.shape[1]:
            return 0.0

        best = float(torch.linalg.vector_norm(_min_norm_point(opts - q)))

        # Inside test: a distance of ~0 means q is a convex combination of the
        # candidates, so report it as inside (negative).
        return -best if best <= self.hull_distance_threshold else best

    def evaluate(
        self,
        probabilities: Union[Dict[str, float], List[float], None] = None,
        free_energy: Optional[float] = None,
        conformal_set: Optional[Dict[str, Any]] = None,
        option_latents: Optional[Sequence[Sequence[float]]] = None,
        query_latent: Optional[Sequence[float]] = None,
    ) -> AbstentionVerdict:
        """Run all applicable detectors and return the combined verdict.

        Args:
            probabilities: Option probabilities, as a dict or a list.
            free_energy: Helmholtz free energy for the branch.
            conformal_set: A conformal prediction set, as returned by
                :class:`ConformalPredictionEngine`.
            option_latents: ``[K, D]`` candidate option latents.
            query_latent: ``[D]`` query / state latent.

        Returns:
            An :class:`AbstentionVerdict`.
        """
        reasons: List[str] = []

        # Normalize probabilities to a list.
        if probabilities is None:
            probs: List[float] = []
        elif isinstance(probabilities, dict):
            probs = [float(p) for p in probabilities.values()]
        else:
            probs = [float(p) for p in probabilities]

        max_prob = max(probs) if probs else None
        margin = None
        if len(probs) >= 2:
            ordered = sorted(probs, reverse=True)
            margin = ordered[0] - ordered[1]

        # Detector 1: free energy (off-manifold state).
        energy_fired = False
        energy_value: Optional[float] = None
        if free_energy is not None:
            try:
                f = float(free_energy)
                energy_value = f
                # NaN energy is a detector failure, not a low-energy state.
                energy_fired = math.isnan(f) or f > self.free_energy_threshold
                if energy_fired:
                    reasons.append(f"free_energy {f:.3f} > {self.free_energy_threshold}")
            except (TypeError, ValueError):
                energy_fired = False

        # Detector 2: conformal set emptiness.
        conformal_size: Optional[int] = None
        conformal_fired = False
        if isinstance(conformal_set, dict) and conformal_set.get("calibrated") is False:
            # An uncalibrated set carries no coverage guarantee, so its size is not
            # evidence either way. Record why this detector is silent; do not fire.
            reasons.append("conformal uncalibrated")
        elif conformal_set is not None:
            if isinstance(conformal_set, dict):
                for key in ("set", "prediction_set", "options", "candidates"):
                    if key in conformal_set:
                        conformal_size = len(conformal_set[key])
                        break
            elif isinstance(conformal_set, (list, tuple, set)):
                conformal_size = len(conformal_set)
            if conformal_size is not None:
                conformal_fired = conformal_size <= self.max_conformal_set_size
                if conformal_fired:
                    reasons.append(f"empty conformal set |C(S)|={conformal_size}")

        # Detector 3: latent convex hull geometry.
        hull_distance: Optional[float] = None
        hull_fired = False
        if option_latents is not None and query_latent is not None:
            try:
                hull_distance = self.convex_hull_distance(option_latents, query_latent)
                hull_fired = hull_distance > self.hull_distance_threshold
                if hull_fired:
                    reasons.append(f"query outside option convex hull (d={hull_distance:.4f})")
            except (RuntimeError, ValueError, IndexError):
                hull_distance = None

        # Aggregation. A structurally unanswerable state (off-manifold, or with
        # no admissible candidate at all) is an Abstain. A state that is
        # on-manifold and has candidates, but where the top two are
        # indistinguishable, is a NeedClarification -- the answer probably
        # exists, the question is just under-specified.
        if energy_fired or hull_fired or conformal_fired:
            decision = ABSTAIN
        elif margin is not None and margin < self.min_margin:
            reasons.append(f"top-2 margin {margin:.4f} < {self.min_margin}")
            decision = NEED_CLARIFICATION
        else:
            decision = ANSWER

        return AbstentionVerdict(
            decision=decision,
            reasons=reasons,
            free_energy=energy_value,
            conformal_size=conformal_size,
            hull_distance=hull_distance,
            max_probability=max_prob,
        )


class LatentSearchEngine(torch.nn.Module):
    """
    Rhen System Two Latent Tree & Beam Search Engine.

    Performs multi-step non-autoregressive beam search directly in continuous latent space:
    At depth d, expands candidates z_{d+1} = T_theta(z_d, a), evaluates prospective value V(z_{d+1})
    and epistemic variance U(z_{d+1}). Prunes to top beam_width trajectories at each depth.
    Integrates step-wise Process Reward (PRM) scoring and conformal risk bounds.
    """

    def __init__(
        self,
        state_repr: Optional[Any] = None,
        candidate_options: Optional[Sequence[Union[str, int]]] = None,
        transition_operator: Optional[Callable] = None,
        prospective_value_head: Optional[Callable] = None,
        beam_width: int = 3,
        max_depth: int = 3,
        prm_scorer: Optional[Callable] = None,
        conformal_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.transition_operator = transition_operator
        self.prospective_value_head = prospective_value_head
        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.prm_scorer = prm_scorer
        self.conformal_alpha = float(conformal_alpha)

        self._action_embeds: Dict[str, torch.Tensor] = {}
        self._stored_state_repr = state_repr
        self._stored_candidate_options = candidate_options
        # Callback bookkeeping for the current search (reset by search()).
        self._callback_ok = 0
        self._callback_fallbacks = 0

    def _get_action_tensor(self, action: Union[str, int], d_model: int, device: torch.device) -> torch.Tensor:
        act_key = str(action)
        if act_key not in self._action_embeds or self._action_embeds[act_key].shape[-1] != d_model:
            g = torch.Generator(device="cpu").manual_seed(_stable_seed("action", act_key))
            vec = torch.randn(d_model, generator=g, dtype=torch.float32).to(device)
            vec = vec / (torch.linalg.vector_norm(vec) + 1e-8)
            self._action_embeds[act_key] = vec
        return self._action_embeds[act_key]

    def _default_transition(self, z: torch.Tensor, action: Union[str, int]) -> torch.Tensor:
        """Fallback latent transition z_{d+1} = T(z_d, a)."""
        d_model = z.shape[-1]
        act_vec = self._get_action_tensor(action, d_model, z.device)
        if z.ndim == 1:
            z_next = z + 0.1 * act_vec
        else:
            z_next = z + 0.1 * act_vec.unsqueeze(0)
        norm = torch.linalg.vector_norm(z_next, dim=-1, keepdim=True) + 1e-8
        return z_next / norm

    def _callback_failed(self, name: str, exc: BaseException) -> None:
        """Count a user-callback failure and log it; the caller uses the heuristic."""
        self._callback_fallbacks += 1
        logger.warning(
            "LatentSearchEngine: %s raised %s (%s); using the explicit heuristic for this call.",
            name, type(exc).__name__, exc,
        )

    def _step_transition(
        self,
        z: torch.Tensor,
        action: Union[str, int],
        transition_op: Optional[Callable],
    ) -> torch.Tensor:
        if transition_op is None:
            return self._default_transition(z, action)
        try:
            res = transition_op(z, action)
            if isinstance(res, torch.Tensor):
                out = res
            elif isinstance(res, (list, tuple)):
                out = torch.as_tensor(res, dtype=torch.float32, device=z.device)
            else:
                raise TypeError(f"transition_operator returned {type(res).__name__}")
        except Exception as exc:
            self._callback_failed("transition_operator", exc)
            return self._default_transition(z, action)
        self._callback_ok += 1
        return out

    def _evaluate_value_and_variance(
        self,
        z: torch.Tensor,
        value_head: Optional[Callable],
    ) -> Tuple[float, float]:
        """Evaluates prospective value V(z_{d+1}) and epistemic variance U(z_{d+1})."""
        if value_head is not None:
            try:
                val_out = value_head(z)
                if isinstance(val_out, tuple) and len(val_out) >= 2:
                    out = (float(val_out[0]), max(float(val_out[1]), 0.0))
                elif isinstance(val_out, torch.Tensor):
                    v = float(val_out.mean().item())
                    u = float(val_out.var().item()) if val_out.numel() > 1 else 0.01
                    out = (v, max(u, 0.0))
                elif isinstance(val_out, (int, float)):
                    out = (float(val_out), 0.01)
                else:
                    raise TypeError(f"prospective_value_head returned {type(val_out).__name__}")
            except Exception as exc:
                self._callback_failed("prospective_value_head", exc)
            else:
                self._callback_ok += 1
                return out

        z_flat = z.detach().reshape(-1)
        mean_val = float(torch.tanh(z_flat.mean()).item())
        var_val = float(torch.var(z_flat).item()) if z_flat.numel() > 1 else 0.01
        return mean_val, max(var_val, 1e-4)

    def _evaluate_prm_reward(
        self,
        z_curr: torch.Tensor,
        action: Union[str, int],
        z_next: torch.Tensor,
        val_next: float,
        prm_scorer: Optional[Callable],
    ) -> float:
        """Step-wise Process Reward Model (PRM) scoring."""
        if prm_scorer is not None:
            try:
                r = prm_scorer(z_curr, action, z_next)
                if isinstance(r, (int, float)):
                    out = float(r)
                elif isinstance(r, torch.Tensor):
                    out = float(r.mean().item())
                else:
                    raise TypeError(f"prm_scorer returned {type(r).__name__}")
            except Exception as exc:
                self._callback_failed("prm_scorer", exc)
            else:
                self._callback_ok += 1
                return out
        z_c_flat = z_curr.detach().reshape(-1)
        val_curr = float(torch.tanh(z_c_flat.mean()).item())
        return val_next - val_curr

    def search(
        self,
        state_repr: Any = None,
        candidate_options: Optional[Sequence[Union[str, int]]] = None,
        transition_operator: Optional[Callable] = None,
        prospective_value_head: Optional[Callable] = None,
        beam_width: Optional[int] = None,
        max_depth: Optional[int] = None,
        prm_scorer: Optional[Callable] = None,
        conformal_alpha: Optional[float] = None,
    ) -> LatentSearchResult:
        """Executes multi-step non-autoregressive latent beam search."""
        s_repr = state_repr if state_repr is not None else self._stored_state_repr
        c_opts = candidate_options if candidate_options is not None else self._stored_candidate_options

        if s_repr is None:
            raise ValueError("LatentSearchEngine.search needs a state_repr (none passed or stored)")
        if not c_opts:
            raise ValueError("LatentSearchEngine.search needs candidate_options (none passed or stored)")

        if isinstance(s_repr, torch.Tensor):
            z_init = s_repr
        elif isinstance(s_repr, (list, tuple)):
            z_init = torch.tensor(s_repr, dtype=torch.float32)
        else:
            # A text state has no latent encoding here: map it to a deterministic pseudo-latent.
            g = torch.Generator(device="cpu").manual_seed(_stable_seed("state", str(s_repr)))
            z_init = torch.randn(128, generator=g, dtype=torch.float32)

        options = [str(o) for o in c_opts]
        trans_op = transition_operator or self.transition_operator
        val_head = prospective_value_head or self.prospective_value_head
        prm = prm_scorer or self.prm_scorer
        user_supplied = any(f is not None for f in (trans_op, val_head, prm))
        self._callback_ok = 0
        self._callback_fallbacks = 0
        b_width = max(1, int(beam_width if beam_width is not None else self.beam_width))
        m_depth = max(1, int(max_depth if max_depth is not None else self.max_depth))
        c_alpha = float(conformal_alpha if conformal_alpha is not None else self.conformal_alpha)

        q_conformal = 1.645 if abs(c_alpha - 0.05) < 1e-4 else math.sqrt(max(0.1, -2.0 * math.log(c_alpha)))

        init_val, init_var = self._evaluate_value_and_variance(z_init, val_head)
        init_conf_bound = init_val - q_conformal * math.sqrt(init_var)

        beam: List[Dict[str, Any]] = [
            {
                "state": z_init,
                "trajectory": [z_init.detach().cpu().numpy().tolist() if isinstance(z_init, torch.Tensor) else list(z_init)],
                "action_sequence": [],
                "step_rewards": [],
                "cum_reward": 0.0,
                "prospective_value": init_val,
                "epistemic_variance": init_var,
                "conformal_bound": init_conf_bound,
                "total_score": init_conf_bound,
            }
        ]

        nodes_tree: List[Dict[str, Any]] = []
        gamma = 0.95

        for depth in range(1, m_depth + 1):
            candidates: List[Dict[str, Any]] = []
            for hyp in beam:
                z_curr = hyp["state"]
                if not isinstance(z_curr, torch.Tensor):
                    z_curr = torch.tensor(z_curr, dtype=torch.float32)

                for act in options:
                    z_next = self._step_transition(z_curr, act, trans_op)
                    val_next, var_next = self._evaluate_value_and_variance(z_next, val_head)
                    prm_r = self._evaluate_prm_reward(z_curr, act, z_next, val_next, prm)
                    cum_r = hyp["cum_reward"] + (gamma ** (depth - 1)) * prm_r

                    conf_bound = val_next - q_conformal * math.sqrt(var_next)
                    score = cum_r + (gamma ** depth) * val_next - 0.1 * var_next

                    z_list = z_next.detach().cpu().numpy().tolist() if isinstance(z_next, torch.Tensor) else list(z_next)
                    new_hyp = {
                        "state": z_next,
                        "trajectory": hyp["trajectory"] + [z_list],
                        "action_sequence": hyp["action_sequence"] + [act],
                        "step_rewards": hyp["step_rewards"] + [float(prm_r)],
                        "cum_reward": cum_r,
                        "prospective_value": float(val_next),
                        "epistemic_variance": float(var_next),
                        "conformal_bound": float(conf_bound),
                        "total_score": float(score),
                    }
                    candidates.append(new_hyp)
                    nodes_tree.append({
                        "depth": depth,
                        "parent_actions": hyp["action_sequence"],
                        "action": act,
                        "prospective_value": float(val_next),
                        "epistemic_variance": float(var_next),
                        "conformal_bound": float(conf_bound),
                        "step_reward": float(prm_r),
                        "score": float(score),
                    })

            if not candidates:
                break

            candidates.sort(key=lambda h: h["total_score"], reverse=True)
            beam = candidates[:b_width]

        best_hyp = beam[0] if beam else {
            "trajectory": [z_init.detach().cpu().numpy().tolist() if isinstance(z_init, torch.Tensor) else list(z_init)],
            "action_sequence": [],
            "prospective_value": init_val,
            "step_rewards": [],
            "conformal_bound": init_conf_bound,
            "epistemic_variance": init_var,
        }

        search_tree_dict = {
            "depth": m_depth,
            "beam_width": b_width,
            "total_expansions": len(nodes_tree),
            "nodes": nodes_tree,
            "best_trajectory_actions": best_hyp["action_sequence"],
            "best_prospective_value": float(best_hyp["prospective_value"]),
            "conformal_alpha": c_alpha,
        }

        return LatentSearchResult(
            best_trajectory=best_hyp["trajectory"],
            best_action_sequence=best_hyp["action_sequence"],
            prospective_value=float(best_hyp["prospective_value"]),
            step_rewards=[float(r) for r in best_hyp["step_rewards"]],
            search_tree=search_tree_dict,
            conformal_bound=float(best_hyp.get("conformal_bound", 0.0)),
            epistemic_variance=float(best_hyp.get("epistemic_variance", 0.0)),
            fallbacks=self._callback_fallbacks,
            source=self._search_source(user_supplied),
        )

    def _search_source(self, user_supplied: bool) -> str:
        """Provenance of the scores used by the last search (see LatentSearchResult)."""
        if not user_supplied:
            return "heuristic"
        if self._callback_fallbacks == 0:
            return "callable"
        if self._callback_ok == 0:
            return "heuristic"
        return "mixed"

    def forward(
        self,
        state_repr: Any = None,
        candidate_options: Optional[Sequence[Union[str, int]]] = None,
        transition_operator: Optional[Callable] = None,
        prospective_value_head: Optional[Callable] = None,
        beam_width: Optional[int] = None,
        max_depth: Optional[int] = None,
        prm_scorer: Optional[Callable] = None,
        conformal_alpha: Optional[float] = None,
    ) -> LatentSearchResult:
        return self.search(
            state_repr=state_repr,
            candidate_options=candidate_options,
            transition_operator=transition_operator,
            prospective_value_head=prospective_value_head,
            beam_width=beam_width,
            max_depth=max_depth,
            prm_scorer=prm_scorer,
            conformal_alpha=conformal_alpha,
        )


class RhenEngine:
    """
    Rhen Parallel Decision Engine.

    Executes non-autoregressive latent decisions over arbitrary states in a single
    EOS forward pass using 2D block-causal branch masking.
    """

    def __init__(self, config=None, model=None, tokenizer=None, device: Optional[torch.device] = None):
        if isinstance(config, torch.nn.Module):
            if model is not None and tokenizer is None:
                tokenizer = model
            model = config
            config = getattr(model, 'config', None)

        if model is None:
            if config is None:
                try:
                    from rhen.config import EOSConfig
                except ImportError:
                    from config import EOSConfig
                config = EOSConfig()
            try:
                from rhen.model import EOSModel
            except ImportError:
                from model import EOSModel
            self.model = EOSModel(config)
        else:
            self.model = model

        if tokenizer is None:
            try:
                from rhen.tokenizer import get_tokenizer
            except ImportError:
                from tokenizer import get_tokenizer
            self.tokenizer = get_tokenizer()
        else:
            self.tokenizer = tokenizer

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.conformal_engine = ConformalPredictionEngine(alpha=0.05)
        self.ood_detector = EnergyOODDetector(ood_threshold=4.5)
        self.packer = get_rhen_packer(self.tokenizer)

        # Unified geometry-based selective abstention gate.
        self.abstention_gate = GeometricAbstentionGate()

    # ---------------------------------------------------------------------------------
    # Unified Geometry-Based Selective Abstention
    # ---------------------------------------------------------------------------------

    def abstention_verdict(
        self,
        probabilities: Union[Dict[str, float], List[float], None],
        free_energy: Optional[float] = None,
        conformal_set: Optional[Dict[str, Any]] = None,
        option_latents: Optional[Sequence[Sequence[float]]] = None,
        query_latent: Optional[Sequence[float]] = None,
    ) -> "AbstentionVerdict":
        """Decide whether to answer, abstain, or ask for clarification.

        This is the single place where "I don't know" is decided. It combines three
        independent failure detectors, each catching a case the others miss:

        1. **Free energy** ``E > 4.5`` -- the state is far off the training
           manifold, so *every* answer would be a guess.
        2. **Conformal set** ``|C(S)| = 0`` -- the calibrated prediction set is
           empty, meaning no candidate carries enough mass to commit to.
        3. **Latent convex hull distance** -- the query falls outside the convex
           hull of the candidate option latents, i.e. the state supports no
           candidate at all. This is the detector that still fires when the
           probabilities look *confident* but are confidently extrapolating.

        ``NeedClarification`` is separated from ``Abstain`` because they call for
        different behaviour: an abstain means "no answer exists here", while a
        clarification means "an answer may exist but the question is ambiguous".
        """
        return self.abstention_gate.evaluate(
            probabilities=probabilities,
            free_energy=free_energy,
            conformal_set=conformal_set,
            option_latents=option_latents,
            query_latent=query_latent,
        )

    # ---------------------------------------------------------------------------------
    # Dynamic Primitive Morphism
    # ---------------------------------------------------------------------------------

    @staticmethod
    def choice_entropy(probabilities: Union[Dict[str, float], List[float]]) -> float:
        """Shannon entropy ``H(p)`` in nats."""
        if isinstance(probabilities, dict):
            probs = [float(p) for p in probabilities.values()]
        else:
            probs = [float(p) for p in probabilities]
        probs = [p for p in probs if p > 0.0]
        if not probs:
            return 0.0
        return float(-sum(p * math.log(p) for p in probs))

    def should_morph(
        self,
        probabilities: Union[Dict[str, float], List[float]],
        entropy_ratio_threshold: float = 0.85,
    ) -> Tuple[bool, float, float]:
        """Whether a categorical ``Choice`` has become too diffuse to commit to.

        When the ``Choice`` distribution is nearly uniform, the categorical
        readout is not reporting genuine uncertainty between discrete options --
        it is reporting that the *question itself* has a continuous answer. A
        5-way choice at exactly ``p = 0.2`` each has ``H(p) = ln 5``; the model has
        no basis to prefer any single option, and forcing it to pick one produces
        a confident-looking arbitrary answer.

        Args:
            probabilities: The ``Choice`` distribution.
            entropy_ratio_threshold: ``H(p) / ln K`` above which the primitive
                should morph. ``0.85`` is the declared default.

        Returns:
            ``(should_morph, entropy, entropy_ratio)``.
        """
        probs = (
            [float(p) for p in probabilities.values()]
            if isinstance(probabilities, dict)
            else [float(p) for p in probabilities]
        )
        k = len(probs)
        if k < 2:
            return False, 0.0, 0.0
        h = self.choice_entropy(probs)
        max_h = math.log(k)
        ratio = h / max_h if max_h > 0 else 0.0
        return ratio >= float(entropy_ratio_threshold), h, ratio

    def morph_primitive(
        self,
        question: str,
        options: List[str],
        probabilities: Union[Dict[str, float], List[float]],
        entropy_ratio_threshold: float = 0.85,
        min_score: float = 0.0,
        max_score: float = 10.0,
    ) -> Dict[str, Any]:
        """Morph a diffuse ``Choice`` into an ordered ``Score`` or bounded ``Range``.

        A near-uniform categorical distribution is re-interpreted as a
        *gradation* over the options: the expected option index becomes an
        expected value on a continuous scale.

        * **``Score``** for a merely diffuse distribution -- the options become
          ordered levels, the natural reading when the question has a continuum.
        * **``Range``** for a *fully* uniform one, since at ``H(p) = ln K`` the
          model has literally no ordinal information and a bounded interval is
          the only defensible answer.

        The result reports ``morphed_from`` and the triggering entropy, so a
        downstream consumer can tell the categorical answer was not trustworthy.

        Args:
            question: The original question text.
            options: The candidate options, treated as ordered levels.
            probabilities: The diffuse ``Choice`` distribution.
            entropy_ratio_threshold: ``H(p) / ln K`` trigger threshold.
            min_score: Lower bound of the morphed scale.
            max_score: Upper bound of the morphed scale.

        Returns:
            A dict describing the morphed primitive.
        """
        probs = (
            [float(p) for p in probabilities.values()]
            if isinstance(probabilities, dict)
            else [float(p) for p in probabilities]
        )
        k = len(probs)
        morph, entropy, ratio = self.should_morph(probs, entropy_ratio_threshold)

        if not morph:
            return {
                "morphed": False,
                "primitive_type": "choice",
                "entropy": entropy,
                "entropy_ratio": ratio,
                "reason": "categorical distribution is decisive",
            }

        # Expected option index -> normalized [0, 1] -> the requested scale.
        expected_idx = sum(p * i for p, i in zip(probs, range(k))) if k else 0.0
        norm = expected_idx / max(k - 1, 1)
        expected_value = min_score + norm * (max_score - min_score)

        # A fully uniform distribution means "no ordinal preference at all";
        # a bounded interval is then more honest than any single number.
        target = "range" if ratio >= 0.99 else "score"
        # Interval width grows with how far past the threshold the entropy is.
        span = max(1.0 - entropy_ratio_threshold, 1e-6)
        half_width = (max_score - min_score) * 0.5 * (ratio - entropy_ratio_threshold) / span

        return {
            "morphed": True,
            "primitive_type": target,
            "from_primitive": "choice",
            "question": question,
            "entropy": entropy,
            "entropy_ratio": ratio,
            "levels": list(options),
            "level_probabilities": probs,
            "expected_value": expected_value,
            "normalized_expected_value": norm,
            "min_value": min_score,
            "max_value": max_score,
            "confidence_interval": (
                max(min_score, expected_value - half_width),
                min(max_score, expected_value + half_width),
            ),
            "reason": (
                f"H(p)={entropy:.4f} is {ratio:.1%} of ln({k})={math.log(k):.4f}; the "
                f"categorical readout is not decisive, so the decision was re-expressed "
                f"as a continuous {'interval' if target == 'range' else 'score'}"
            ),
        }

    # ---------------------------------------------------------------------------------
    # Non-Autoregressive Latent Beam Search
    # ---------------------------------------------------------------------------------

    def simulate_latent_beam_search(
        self,
        state: Any,
        question: str,
        options: List[str],
        beam_width: int = 4,
        horizon: int = 2,
        length_norm: float = 0.0,
        max_expansions: int = 64,
    ) -> Dict[str, Any]:
        """Expand candidate action sequences directly in continuous latent space.

        Autoregressive search over actions is expensive: each expansion needs a
        model forward pass, and the beam must be extended one discrete action at a
        time. Because Rhen is *non-autoregressive*, every candidate action is
        already scored in a single parallel pass, so a "step" of the search is a
        batched latent transition rather than a token-generation round.

        Hypotheses are ranked by a length-normalised score::

            score(sequence) = (1/h) * sum_t log p(z_t | z_{t-1}, o)

        so sequences of different depths stay comparable. Candidates are scored
        through the model's latent transition operator when the model exposes one.
        Otherwise they are ranked by the calibrated ``Choice`` probabilities, and the
        returned ``source`` says which was used. Model errors propagate.

        This never generates text: it only moves vectors.

        Timing note: ``expansion_ms`` measures the search itself, which is the
        figure the sub-5 ms budget refers to (it is three orders of magnitude
        under budget in practice). ``elapsed_ms`` additionally includes the single
        non-autoregressive forward pass that scores the candidates, so it is
        dominated by the model and grows with sequence length -- it is reported
        separately rather than conflated with the search cost.

        Args:
            state: The structured state.
            question: The decision question.
            options: Candidate actions to search over.
            beam_width: Number of hypotheses kept at each depth.
            horizon: Number of latent steps to expand.
            length_norm: Depth divisor for length normalisation (0 disables it).
            max_expansions: Hard cap on total expanded hypotheses, bounding cost.

        Returns:
            A dict with the ranked ``hypotheses``, ``best_option``, the measured
            ``expansion_ms`` / ``elapsed_ms``, and the ``source`` actually used.
        """
        import time as _time

        started = _time.perf_counter()
        opts = list(options)
        if not opts:
            return {
                "hypotheses": [], "best_option": "", "best_score": 0.0,
                "horizon": horizon, "expansions": 0, "expansion_ms": 0.0,
                "expansion_within_5ms": True, "elapsed_ms": 0.0, "source": "empty",
            }

        # Latent-space scoring, when the model exposes the transition operator.
        # Errors from the model propagate; only a missing entry point (None) falls back.
        latent_scores = self._latent_option_values(state, question, opts, horizon)

        if latent_scores is not None and len(latent_scores) == len(opts):
            source = "latent_transition"
            base = latent_scores
        else:
            source = "calibrated_choice"
            res = self.choice(state, question, opts)
            probs = [float(p) for p in res.raw_probabilities]
            base = {o: math.log(max(p, 1e-12)) for o, p in zip(opts, probs)}

        # Beam expansion. Each hypothesis accumulates a per-step log-probability;
        # a wider beam is a strictly larger search, never a different one.
        beam: List[Dict[str, Any]] = [
            {"option": o, "logp": s, "sequence": [o], "steps": 0}
            for o, s in base.items()
        ]
        expansions = 0

        # Time the *expansion* separately from scoring the options. The < 5 ms
        # budget in the design refers to the search itself; the dominant cost of a
        # whole call is the single non-autoregressive forward pass that scores the
        # candidates, which any decision needs regardless of beam width.
        expand_started = _time.perf_counter()

        depth = max(1, int(horizon))
        for _step in range(1, depth):
            if expansions >= max_expansions:
                break
            candidates: List[Dict[str, Any]] = []
            for hyp in beam:
                # In latent space the transition is deterministic given the option,
                # so a hypothesis' score accumulates the next option's contribution.
                # That determinism is exactly what makes this non-autoregressive.
                for o, s in base.items():
                    if expansions >= max_expansions:
                        break
                    candidates.append(
                        {
                            "option": o,
                            "logp": hyp["logp"] + s,
                            "sequence": hyp["sequence"] + [o],
                            "steps": hyp["steps"] + 1,
                        }
                    )
                    expansions += 1
            if not candidates:
                break
            candidates.sort(key=lambda h: h["logp"], reverse=True)
            beam = candidates[: max(1, int(beam_width))]

        expand_ms = (_time.perf_counter() - expand_started) * 1000.0

        hypotheses = []
        for hyp in beam:
            denom = hyp["steps"] if (length_norm and hyp["steps"] > 0) else 1
            hypotheses.append(
                {
                    "sequence": list(hyp["sequence"]),
                    "terminal_option": hyp["option"],
                    "logp": hyp["logp"],
                    "score": hyp["logp"] / denom,
                    "depth": hyp["steps"] + 1,
                }
            )
        hypotheses.sort(key=lambda h: h["score"], reverse=True)

        best = hypotheses[0] if hypotheses else None
        elapsed_ms = (_time.perf_counter() - started) * 1000.0

        return {
            "hypotheses": hypotheses,
            "best_option": best["terminal_option"] if best else opts[0],
            "best_score": best["score"] if best else 0.0,
            "horizon": depth,
            "beam_width": int(beam_width),
            "expansions": expansions,
            # `expansion_ms` is the search itself and is the figure the < 5 ms
            # budget refers to. `elapsed_ms` additionally includes the single
            # forward pass that scores the candidates, so it is dominated by the
            # model and is reported separately rather than conflated.
            "expansion_ms": expand_ms,
            "expansion_within_5ms": expand_ms < 5.0,
            "elapsed_ms": elapsed_ms,
            "source": source,
        }

    def _latent_option_values(
        self,
        state: Any,
        question: str,
        options: List[str],
        horizon: int,
    ) -> Optional[Dict[str, float]]:
        """Prospective value per option via the model's latent operator.

        Returns ``None`` when the model does not expose the required entry point,
        so the caller can fall back to calibrated option probabilities.
        """
        if not hasattr(self.model, "simulate_latent_rollout"):
            return None

        packer = self.packer or get_rhen_packer(self.tokenizer)
        packed = packer.pack(
            state=state,
            questions_and_options=[(question, list(options))],
            device=self.device,
        )
        if not isinstance(packed, PackedRhenSequence):
            return None

        details = self.model.forward_rhen(packed)[0]
        opt_layers = details.get("opt_layers", None)
        if not opt_layers:
            return None

        pointer_head = self.model.recurrent_core.rhen.pointer_head
        opts = pointer_head.fuse_layers(
            [o if o.ndim == 3 else o.unsqueeze(0) for o in opt_layers]
        )
        z_init = opts.mean(dim=1)          # [B, D] pooled state latent

        out = self.model.simulate_latent_rollout(
            state_repr=z_init, option_reprs=opts[0], horizon=max(1, int(horizon))
        )
        values = out.get("prospective_values", None)
        if values is None:
            return None

        row = values[0] if values.ndim > 1 else values
        # Detach before converting to Python floats: these are inference-time
        # scores, and `float()` on a grad-tracking tensor warns (and would retain
        # the whole graph for a value we only ever read).
        row = row.detach().reshape(-1).tolist()
        return {opt: float(row[i]) for i, opt in enumerate(options) if i < len(row)}

    def _extract_conformal_and_energy(
        self,
        raw_res: Dict[str, Any],
        probabilities: Union[List[float], Dict[str, float], None] = None,
        scores: Optional[List[float]] = None,
        temperature: float = 1.0,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """
        Extracts conformal prediction set and free energy from raw branch output,
        or queries ConformalPredictionEngine and EnergyOODDetector.
        """
        conformal_set = raw_res.get("conformal_set")
        if conformal_set is None and self.conformal_engine is not None and probabilities is not None:
            conformal_set = self.conformal_engine.predict_set(probabilities)

        free_energy = raw_res.get("free_energy")
        if free_energy is not None:
            try:
                free_energy = float(free_energy)
            except (ValueError, TypeError):
                free_energy = None
        elif self.ood_detector is not None:
            # Energy comes only from explicit raw scores or logits. Never from log(probabilities):
            # at temperature 1, logsumexp(log p) = log(sum p) = 0 for normalized p, so that
            # energy carries no information and the OOD detector could never fire.
            # No explicit scores or logits means no energy (None).
            target_scores = scores if scores is not None else raw_res.get("scores")
            if target_scores is None:
                target_scores = raw_res.get("logits")
            if target_scores is not None and len(target_scores) > 0:
                free_energy = float(self.ood_detector.compute_energy(target_scores, temperature=temperature))
            else:
                free_energy = None

        return conformal_set, free_energy

    def _extract_schema_mutation(
        self,
        raw_res: Union[Dict[str, Any], Any],
        question_text: str = "",
    ) -> Optional[SchemaMutation]:
        """Extracts the dynamic schema mutation from a raw branch result.

        ``schema_logits`` carry one logit per entry of ``SCHEMA_MUTATION_TYPES``. The
        mutation is the argmax over all of them. A logit vector of the wrong width,
        or an unknown mutation type, raises: it is never collapsed onto one type.
        """
        if not isinstance(raw_res, dict):
            return None
        if "schema_mutation" in raw_res and raw_res["schema_mutation"] is not None:
            sm = raw_res["schema_mutation"]
            if isinstance(sm, SchemaMutation):
                return sm
            if isinstance(sm, dict):
                mut_type = sm.get("mutation_type")
                if mut_type not in SCHEMA_MUTATION_TYPES:
                    raise ValueError(
                        f"schema_mutation.mutation_type must be one of {SCHEMA_MUTATION_TYPES}, got {mut_type!r}"
                    )
                return SchemaMutation(
                    parent_question=sm.get("parent_question", question_text),
                    mutation_type=mut_type,
                    mutation_logits=sm.get("mutation_logits", []),
                    confidence=float(sm.get("confidence", 0.5)),
                    metadata=sm.get("metadata"),
                )
        if "schema_logits" in raw_res and raw_res["schema_logits"] is not None:
            raw_logits = raw_res["schema_logits"]
            if isinstance(raw_logits, torch.Tensor):
                logits_list = raw_logits.detach().cpu().reshape(-1).tolist()
            elif isinstance(raw_logits, (list, tuple)):
                logits_list = [float(x) for x in raw_logits]
            else:
                logits_list = []

            if logits_list:
                if len(logits_list) != len(SCHEMA_MUTATION_TYPES):
                    raise ValueError(
                        f"schema_logits must have {len(SCHEMA_MUTATION_TYPES)} entries "
                        f"(one per schema mutation type), got {len(logits_list)}"
                    )
                max_l = max(logits_list)
                exps = [math.exp(l - max_l) for l in logits_list]
                sum_exps = sum(exps)
                probs = [e / max(sum_exps, 1e-12) for e in exps]
                best_idx = max(range(len(logits_list)), key=lambda i: logits_list[i])
                return SchemaMutation(
                    parent_question=question_text,
                    mutation_type=SCHEMA_MUTATION_TYPES[best_idx],
                    mutation_logits=logits_list,
                    confidence=probs[best_idx],
                    metadata=raw_res.get("schema_metadata"),
                )
        return None

    def simulate_latent_outcomes(
        self,
        state: Any,
        question: str,
        options: List[str],
        horizon: int = 2,
    ) -> LatentRolloutResult:
        """
        Projects the calibrated ``Choice`` distribution forward over the horizon.

        This is a heuristic counterfactual projection, not a learned latent rollout:
        each option's calibrated probability is discounted per step. The result's
        ``source`` is ``"calibrated_choice_projection"`` to say so. Model errors propagate.
        """
        choice_res = self.choice(state, question, options)
        probs = choice_res.probabilities
        conf = choice_res.confidence

        prospective_values: Dict[str, float] = {}
        anticipated_trajectories: Dict[str, Any] = {}

        gamma = 0.95
        for opt in options:
            p = float(probs.get(opt, 0.0))
            val = 0.0
            traj = []
            for t in range(1, horizon + 1):
                discount = gamma ** (t - 1)
                step_val = p * (1.0 - (t - 1) / max(horizon * 2, 1)) + (conf / max(horizon, 1))
                val += discount * step_val
                traj.append({
                    "horizon_step": t,
                    "projected_value": round(p * discount, 4),
                    "confidence_decay": round(conf * (0.98 ** (t - 1)), 4),
                })
            prospective_values[opt] = round(val / max(sum(gamma ** (t - 1) for t in range(1, horizon + 1)), 1e-6), 4)
            anticipated_trajectories[opt] = traj

        if prospective_values:
            best_opt = max(prospective_values, key=lambda k: prospective_values[k])
            best_val = prospective_values[best_opt]
        else:
            best_opt = options[0] if options else ""
            best_val = 0.0

        state_summary = str(state)
        if len(state_summary) > 200:
            state_summary = state_summary[:197] + "..."

        return LatentRolloutResult(
            state_summary=state_summary,
            candidate_options=options,
            horizon=horizon,
            prospective_values=prospective_values,
            best_option=best_opt,
            best_value=best_val,
            anticipated_trajectories=anticipated_trajectories,
            source="calibrated_choice_projection",
        )

    def latent_search(
        self,
        state: Any,
        root_question: Any,
        candidate_branches: Sequence[Union[str, int]],
        beam_width: int = 3,
        max_depth: int = 3,
        transition_operator: Optional[Callable] = None,
        prospective_value_head: Optional[Callable] = None,
        prm_scorer: Optional[Callable] = None,
        conformal_alpha: float = 0.05,
        **kwargs: Any,
    ) -> LatentSearchResult:
        """
        Performs multi-step non-autoregressive latent beam search over decision branches.

        Extracts continuous state representations via model packing/forward passes or latent
        projections, expands candidates in pure latent space, scores step-wise Process Rewards (PRM),
        and computes calibrated conformal risk bounds.
        """
        state_repr = None
        if isinstance(state, torch.Tensor):
            state_repr = state
        elif hasattr(self.model, "forward_rhen"):
            packer = self.packer or get_rhen_packer(self.tokenizer)
            packed = packer.pack(
                state=state,
                questions_and_options=[(str(root_question), [str(c) for c in candidate_branches])],
                device=self.device,
            )
            details = self.model.forward_rhen(packed)[0]
            opt_layers = details.get("opt_layers", None)
            if opt_layers:
                rec_core = getattr(self.model, "recurrent_core", None)
                rhen_mod = getattr(rec_core, "rhen", None) if rec_core is not None else None
                if rhen_mod is not None and hasattr(rhen_mod, "pointer_head"):
                    opts = rhen_mod.pointer_head.fuse_layers([o if o.ndim == 3 else o.unsqueeze(0) for o in opt_layers])
                    state_repr = opts.mean(dim=1)

        if state_repr is None:
            # No latent for this state: use a deterministic pseudo-latent (sha256 seed).
            g = torch.Generator(device="cpu").manual_seed(_stable_seed("state", str(state), str(root_question)))
            state_repr = torch.randn(128, generator=g, dtype=torch.float32).to(self.device)

        # Transition, value and PRM come only from the caller's callables. Without
        # them, LatentSearchEngine applies its explicit heuristics and reports
        # source == "heuristic". No model-backed closures are substituted.
        engine = LatentSearchEngine(
            transition_operator=transition_operator,
            prospective_value_head=prospective_value_head,
            beam_width=beam_width,
            max_depth=max_depth,
            prm_scorer=prm_scorer,
            conformal_alpha=conformal_alpha,
        )
        return engine.search(
            state_repr=state_repr,
            candidate_options=candidate_branches,
            beam_width=beam_width,
            max_depth=max_depth,
        )

    def speculative_cascade(
        self,
        state: Any,
        root_question: Any,
        speculative_branches: Dict[int, List[Any]],
        latent_tensors: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes a 2D speculative tree cascade in a single O(1) state forward pass.

        Constructs a tree-block-causal mask isolating alternative child branches,
        evaluates root and all speculative child paths simultaneously, and returns
        the unpacked tree hierarchy and active decision path.
        """
        packer = getattr(self, "packer", None) or get_rhen_packer(self.tokenizer)

        packed = packer.build_speculative_tree_pack(
            state=state,
            root_question=root_question,
            speculative_branches=speculative_branches,
            latent_tensors=latent_tensors,
            device=self.device,
        )

        if hasattr(self.model, "evaluate_rhen"):
            raw_results = self.model.evaluate_rhen(
                packed_or_state=packed,
                tokenizer=self.tokenizer,
            )
        elif hasattr(self.model, "forward_rhen"):
            raw_results = self.model.forward_rhen(
                packed_or_state=packed,
                tokenizer=self.tokenizer,
            )
        else:
            raise NotImplementedError("Model does not expose evaluate_rhen or forward_rhen for packed sequences.")

        if isinstance(raw_results, dict):
            if "all_results" in raw_results:
                raw_results = raw_results["all_results"]
            else:
                flat = [raw_results.get("root_result", {})]
                for k in sorted(raw_results.get("speculative_results", {}).keys()):
                    flat.extend(raw_results["speculative_results"][k])
                raw_results = flat
        elif not isinstance(raw_results, list):
            raw_results = [raw_results]

        unpacked_results: List[Any] = []
        for b_idx, (branch_meta, raw_res) in enumerate(zip(packed.branches, raw_results)):
            if not isinstance(raw_res, dict):
                # A bare value carries no probabilities or confidence; inventing them would
                # report certainty the model never expressed.
                raise TypeError(
                    f"speculative_cascade branch {b_idx} returned {type(raw_res).__name__}, expected a dict"
                )
            prim = getattr(branch_meta, "primitive_type", "choice")
            meta = getattr(branch_meta, "metadata", None) or {}
            sm = self._extract_schema_mutation(raw_res, branch_meta.question_text)

            if prim == "noul":
                p_true = raw_res["probabilities"][0] if "probabilities" in raw_res else 0.5
                noul_probs = {"Yes": p_true, "No": 1.0 - p_true}
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=noul_probs, scores=raw_res.get("scores"), temperature=raw_res.get("temperature", 1.0)
                )
                res_obj = NoulResult(
                    proposition=branch_meta.question_text,
                    p_true=p_true,
                    decision=bool(p_true >= 0.5),
                    confidence=abs(p_true - 0.5) * 2.0,
                    temperature=raw_res.get("temperature", 1.0),
                    conformal_set=conf_set,
                    free_energy=energy,
                    schema_mutation=sm,
                )
            elif prim == "score":
                probs = raw_res.get("probabilities", [0.2] * 5)
                num_levels = len(probs)
                min_s = float(meta.get("min_score", 0.0))
                max_s = float(meta.get("max_score", 10.0))
                norm_weights = [i / max(num_levels - 1, 1) for i in range(num_levels)]
                expected_norm = sum(p * w for p, w in zip(probs, norm_weights))
                expected_score = min_s + expected_norm * (max_s - min_s)
                conf = float(raw_res.get("confidence", 0.5))
                level_dict = {f"Level {k}": p for k, p in enumerate(probs)}
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=level_dict, scores=raw_res.get("scores"), temperature=raw_res.get("temperature", 1.0)
                )
                res_obj = ScoreResult(
                    question=branch_meta.question_text,
                    score=expected_score,
                    min_score=min_s,
                    max_score=max_s,
                    level_probabilities=probs,
                    confidence=conf,
                    temperature=raw_res.get("temperature", 1.0),
                    conformal_set=conf_set,
                    free_energy=energy,
                    schema_mutation=sm,
                )
            elif prim == "multichoice":
                raw_probs = raw_res.get("probabilities", [])
                prob_dict = {opt: p for opt, p in zip(branch_meta.options, raw_probs)}
                thresh = float(meta.get("threshold", 0.5))
                selected_indices = raw_res.get("selected_indices", [i for i, p in enumerate(raw_probs) if p >= thresh])
                selected_options = raw_res.get("selected_options", [branch_meta.options[i] for i in selected_indices if i < len(branch_meta.options)])
                conf = float(raw_res.get("confidence", max(raw_probs) if raw_probs else 0.5))
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=prob_dict, scores=raw_res.get("scores"), temperature=float(raw_res.get("temperature", 1.0))
                )
                res_obj = MultiChoiceResult(
                    question=branch_meta.question_text,
                    selected_indices=selected_indices,
                    selected_options=selected_options,
                    probabilities=prob_dict,
                    raw_probabilities=raw_probs,
                    confidence=conf,
                    temperature=float(raw_res.get("temperature", 1.0)),
                    threshold=thresh,
                    conformal_set=conf_set,
                    free_energy=energy,
                    schema_mutation=sm,
                )
            elif prim == "range":
                expected_val = float(raw_res.get("expected_value", 50.0))
                int_low = float(raw_res.get("interval_low", 40.0))
                int_high = float(raw_res.get("interval_high", 60.0))
                min_v = float(meta.get("min_val", 0.0))
                max_v = float(meta.get("max_val", 100.0))
                conf = float(raw_res.get("confidence", 0.5))
                # A range has no categorical distribution, so no conformal set or energy is
                # derived from synthetic probabilities: only the model's own values are used.
                conf_set = raw_res.get("conformal_set")
                energy = raw_res.get("free_energy")
                energy = None if energy is None else float(energy)
                res_obj = RangeResult(
                    question=branch_meta.question_text,
                    expected_value=expected_val,
                    interval_low=int_low,
                    interval_high=int_high,
                    confidence_interval=0.95,
                    min_val=min_v,
                    max_val=max_v,
                    confidence=conf,
                    temperature=float(raw_res.get("temperature", 1.0)),
                    conformal_set=conf_set,
                    free_energy=energy,
                    schema_mutation=sm,
                )
            elif prim == "extract":
                extracted_val = raw_res.get("extracted_value") or ""
                # No extracted value means no evidence: empty value, zero confidence.
                conf = float(raw_res.get("confidence", 0.0)) if extracted_val else 0.0
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=[conf, 1.0 - conf], scores=raw_res.get("scores"), temperature=float(raw_res.get("temperature", 1.0))
                )
                res_obj = ExtractResult(
                    slot_name=meta.get("slot_name", branch_meta.question_text),
                    extracted_value=extracted_val,
                    start_idx=int(raw_res.get("start_idx", 0)),
                    end_idx=int(raw_res.get("end_idx", 0)),
                    confidence=conf,
                    temperature=float(raw_res.get("temperature", 1.0)),
                    conformal_set=conf_set,
                    free_energy=energy,
                    schema_mutation=sm,
                )
            else:
                # Default ChoiceResult
                probs_list = raw_res.get("probabilities", [])
                prob_dict = {opt: p for opt, p in zip(branch_meta.options, probs_list)}
                typesafe_conf = raw_res.get("typesafe_confidence", None)
                if typesafe_conf is None:
                    K = max(len(branch_meta.options), 2)
                    p_max = float(raw_res.get("confidence", 0.5))
                    typesafe_conf = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=prob_dict, scores=raw_res.get("scores"), temperature=raw_res.get("temperature", 1.0)
                )
                res_obj = ChoiceResult(
                    question=branch_meta.question_text,
                    choice_index=raw_res.get("chosen_index", 0),
                    choice_label=raw_res.get("chosen_option", branch_meta.options[0] if branch_meta.options else ""),
                    probabilities=prob_dict,
                    raw_probabilities=probs_list,
                    confidence=typesafe_conf,
                    temperature=raw_res.get("temperature", 1.0),
                    conformal_set=conf_set,
                    free_energy=energy,
                    schema_mutation=sm,
                )
            unpacked_results.append(res_obj)

        root_result = unpacked_results[0]
        unpacked_speculative: Dict[int, List[Any]] = {}

        for b_idx in range(1, len(packed.branches)):
            b_meta = packed.branches[b_idx]
            cond_k = b_meta.condition_option_idx
            try:
                cond_int = int(cond_k) if cond_k is not None else 0
            except (ValueError, TypeError):
                cond_int = cond_k
            unpacked_speculative.setdefault(cond_int, []).append(unpacked_results[b_idx])

        # Identify winning root option
        if isinstance(root_result, ChoiceResult):
            winning_opt_idx = root_result.choice_index
        elif isinstance(root_result, NoulResult):
            winning_opt_idx = 0 if root_result.decision else 1
        elif isinstance(root_result, dict):
            winning_opt_idx = root_result.get("chosen_index", 0)
        else:
            winning_opt_idx = getattr(root_result, "choice_index", getattr(root_result, "chosen_index", 0))

        active_branch = unpacked_speculative.get(winning_opt_idx, [])

        return {
            "root": root_result,
            "speculative_branches": unpacked_speculative,
            "active_branch": active_branch,
            "winning_option_index": winning_opt_idx,
            "all_results": unpacked_results,
            "packed_sequence": packed,
        }

    def noul(self, state: Any, proposition: str) -> NoulResult:
        """Evaluates a binary semantic proposition."""
        return self.evaluate(state, [Noul(proposition=proposition)])[0]

    def choice(self, state: Any, question: str, options: List[str]) -> ChoiceResult:
        """Evaluates a categorical choice."""
        return self.evaluate(state, [Choice(question=question, options=options)])[0]

    def score(
        self,
        state: Any,
        question: str,
        levels: Optional[List[str]] = None,
        min_score: float = 0.0,
        max_score: float = 10.0,
    ) -> ScoreResult:
        """Evaluates an expected continuous score."""
        return self.evaluate(
            state,
            [Score(
                question=question,
                levels=levels if levels is not None else ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"],
                min_score=min_score,
                max_score=max_score,
            )],
        )[0]

    def multichoice(
        self,
        state: Any,
        question: str,
        options: List[str],
        threshold: float = 0.5,
    ) -> MultiChoiceResult:
        """Evaluates a multi-choice selection where multiple options can exceed threshold."""
        return self.evaluate(state, [MultiChoice(question=question, options=options, threshold=threshold)])[0]

    def range(
        self,
        state: Any,
        question: str,
        min_val: float = 0.0,
        max_val: float = 100.0,
    ) -> RangeResult:
        """Evaluates a continuous expected value and interval range."""
        return self.evaluate(state, [Range(question=question, min_val=min_val, max_val=max_val)])[0]

    def extract(
        self,
        state: Any,
        slot_name: str,
        query: str = "",
    ) -> ExtractResult:
        """Extracts a structured value for a given slot from state."""
        return self.evaluate(state, [Extract(slot_name=slot_name, query=query)])[0]

    def evaluate(
        self,
        state: Any,
        questions: List[Union[Noul, Choice, Score, MultiChoice, Range, Extract, Cascade, Tuple[str, List[str]]]],
        delta_state: Optional[Any] = None,
        is_delta: bool = False,
        latent_tensors: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Union[NoulResult, ChoiceResult, ScoreResult, MultiChoiceResult, RangeResult, ExtractResult, CascadeResult, Dict[str, Any]]]:
        """
        Evaluates a batch of mixed decision primitives concurrently over a single state
        in one forward pass ($O(1)$ state compute). Supports state deltas via pack_delta
        and continuous feature grounding via latent_tensors.

        Args:
            state: Any structured state object (dict, list, str, JSON, or PackedRhenSequence).
            questions: List of Noul, Choice, Score, MultiChoice, Range, Extract, Cascade, or raw tuple primitives.
            delta_state: Optional delta state object for incremental state evaluation.
            is_delta: Boolean flag indicating if state represents a delta against base_state.
            latent_tensors: Optional dictionary of continuous feature tensors to ground into latent slots.
            **kwargs: Optional additional packaging arguments (e.g. base_state).

        Returns:
            List of typed results (NoulResult, ChoiceResult, ScoreResult, MultiChoiceResult, RangeResult, ExtractResult, CascadeResult, or dict).
        """
        tuples = []
        types = []
        for q in questions:
            if isinstance(q, Noul):
                tuples.append(q.to_tuple())
                types.append(("noul", q))
            elif isinstance(q, Choice):
                tuples.append(q.to_tuple())
                types.append(("choice", q))
            elif isinstance(q, Score):
                tuples.append(q.to_tuple())
                types.append(("score", q))
            elif isinstance(q, MultiChoice):
                tuples.append(q.to_tuple())
                types.append(("multichoice", q))
            elif isinstance(q, Range):
                tuples.append(q.to_tuple())
                types.append(("range", q))
            elif isinstance(q, Extract):
                tuples.append(q.to_tuple())
                types.append(("extract", q))
            elif isinstance(q, Cascade):
                tuples.append(q.to_tuple())
                types.append(("cascade", q))
            elif isinstance(q, tuple):
                tuples.append(q)
                types.append(("raw", q))
            else:
                raise TypeError(f"Unsupported question specification: {type(q)}")

        packer = getattr(self, "packer", None) or get_rhen_packer(self.tokenizer)

        # Support state deltas and continuous latent grounding
        packed_input = None
        if isinstance(state, PackedRhenSequence):
            packed_input = state
        elif delta_state is not None and packer is not None:
            packed_input = packer.pack_delta(
                base_state=state,
                delta_state=delta_state,
                questions=tuples,
                device=self.device,
                latent_tensors=latent_tensors,
            )
        elif is_delta and packer is not None:
            base_state = kwargs.get("base_state", "")
            packed_input = packer.pack_delta(
                base_state=base_state,
                delta_state=state,
                questions=tuples,
                device=self.device,
                latent_tensors=latent_tensors,
            )
        elif packer is not None and (latent_tensors is not None or not isinstance(state, str)):
            packed_input = packer.pack(
                state=state,
                questions_and_options=tuples,
                latent_tensors=latent_tensors,
                device=self.device,
            )
        else:
            packed_input = state

        # Errors from the model or packer propagate: there is no retry with altered questions.
        raw_results = self.model.evaluate_rhen(
            packed_or_state=packed_input,
            questions_and_options=tuples,
            tokenizer=self.tokenizer,
        )

        typed_results = []
        for (q_type, q_obj), raw_res in zip(types, raw_results):
            if q_type == "noul":
                p_true = raw_res["probabilities"][0]
                noul_probs = {"Yes": p_true, "No": 1.0 - p_true}
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=noul_probs, scores=raw_res.get("scores"), temperature=raw_res.get("temperature", 1.0)
                )
                sm = self._extract_schema_mutation(raw_res, q_obj.proposition)
                verdict = self.abstention_verdict(probabilities=noul_probs, free_energy=energy, conformal_set=conf_set)
                typed_results.append(
                    NoulResult(
                        proposition=q_obj.proposition,
                        p_true=p_true,
                        decision=bool(p_true >= 0.5),
                        confidence=abs(p_true - 0.5) * 2.0,
                        temperature=raw_res["temperature"],
                        conformal_set=conf_set,
                        free_energy=energy,
                        schema_mutation=sm,
                        abstention_verdict=verdict,
                    )
                )
            elif q_type == "choice":
                prob_dict = {opt: p for opt, p in zip(q_obj.options, raw_res["probabilities"])}
                typesafe_conf = raw_res.get("typesafe_confidence", None)
                if typesafe_conf is None:
                    K = max(len(q_obj.options), 2)
                    p_max = float(raw_res["confidence"])
                    typesafe_conf = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=prob_dict, scores=raw_res.get("scores"), temperature=raw_res.get("temperature", 1.0)
                )
                sm = self._extract_schema_mutation(raw_res, q_obj.question)
                verdict = self.abstention_verdict(probabilities=prob_dict, free_energy=energy, conformal_set=conf_set)
                typed_results.append(
                    ChoiceResult(
                        question=q_obj.question,
                        choice_index=raw_res["chosen_index"],
                        choice_label=raw_res["chosen_option"],
                        probabilities=prob_dict,
                        raw_probabilities=raw_res["probabilities"],
                        confidence=typesafe_conf,
                        temperature=raw_res["temperature"],
                        conformal_set=conf_set,
                        free_energy=energy,
                        schema_mutation=sm,
                        abstention_verdict=verdict,
                    )
                )
            elif q_type == "score":
                probs = raw_res["probabilities"]
                num_levels = len(probs)
                norm_weights = [i / max(num_levels - 1, 1) for i in range(num_levels)]
                expected_norm = sum(p * w for p, w in zip(probs, norm_weights))
                expected_score = q_obj.min_score + expected_norm * (q_obj.max_score - q_obj.min_score)

                # Modal ordinal dispersion confidence
                k_mode = int(max(range(num_levels), key=lambda i: probs[i])) if probs else 0
                disp_mode = sum(p * abs(k - k_mode) for k, p in enumerate(probs))
                disp_uniform = sum(abs(k - k_mode) for k in range(num_levels)) / max(num_levels, 1)
                score_conf = max(0.0, min(1.0, 1.0 - (disp_mode / max(disp_uniform, 1e-6))))

                level_dict = {f"Level {k}": p for k, p in enumerate(probs)}
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=level_dict, scores=raw_res.get("scores"), temperature=raw_res.get("temperature", 1.0)
                )
                sm = self._extract_schema_mutation(raw_res, q_obj.question)
                verdict = self.abstention_verdict(probabilities=level_dict, free_energy=energy, conformal_set=conf_set)

                typed_results.append(
                    ScoreResult(
                        question=q_obj.question,
                        score=expected_score,
                        min_score=q_obj.min_score,
                        max_score=q_obj.max_score,
                        level_probabilities=probs,
                        confidence=score_conf,
                        temperature=raw_res["temperature"],
                        conformal_set=conf_set,
                        free_energy=energy,
                        schema_mutation=sm,
                        abstention_verdict=verdict,
                    )
                )
            elif q_type == "multichoice":
                raw_probs = raw_res.get("probabilities", [])
                prob_dict = {opt: p for opt, p in zip(q_obj.options, raw_probs)}
                selected_indices = raw_res.get(
                    "selected_indices",
                    [i for i, p in enumerate(raw_probs) if p >= q_obj.threshold]
                )
                selected_options = raw_res.get(
                    "selected_options",
                    [q_obj.options[i] for i in selected_indices if i < len(q_obj.options)]
                )
                conf = raw_res.get("confidence")
                if conf is None:
                    conf = max(raw_probs) if raw_probs else 0.0

                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=prob_dict, scores=raw_res.get("scores"), temperature=float(raw_res.get("temperature", 1.0))
                )
                sm = self._extract_schema_mutation(raw_res, q_obj.question)
                verdict = self.abstention_verdict(probabilities=prob_dict, free_energy=energy, conformal_set=conf_set)

                typed_results.append(
                    MultiChoiceResult(
                        question=q_obj.question,
                        selected_indices=selected_indices,
                        selected_options=selected_options,
                        probabilities=prob_dict,
                        raw_probabilities=raw_probs,
                        confidence=float(conf),
                        temperature=float(raw_res.get("temperature", 1.0)),
                        threshold=q_obj.threshold,
                        conformal_set=conf_set,
                        free_energy=energy,
                        schema_mutation=sm,
                        abstention_verdict=verdict,
                    )
                )
            elif q_type == "range":
                probs = raw_res.get("probabilities", [0.5, 0.5])
                if len(probs) >= 2 and sum(probs[:2]) > 0:
                    p_high = probs[1] / sum(probs[:2])
                else:
                    p_high = 0.5
                default_expected = q_obj.min_val + p_high * (q_obj.max_val - q_obj.min_val)
                expected_val = float(raw_res.get("expected_value", default_expected))
                conf = float(raw_res.get("confidence", abs(p_high - 0.5) * 2.0))
                ci = float(raw_res.get("confidence_interval", 0.95))
                span = q_obj.max_val - q_obj.min_val
                half_w = span * (1.0 - conf) * 0.5 * ci
                int_low = float(raw_res.get("interval_low", max(q_obj.min_val, expected_val - half_w)))
                int_high = float(raw_res.get("interval_high", min(q_obj.max_val, expected_val + half_w)))

                # A range has no categorical distribution, so no conformal set or energy is
                # derived from synthetic probabilities or expected values: only the model's
                # own values are used, and the free energy is None unless the model gives one.
                conf_set = raw_res.get("conformal_set")
                energy = raw_res.get("free_energy")
                energy = None if energy is None else float(energy)
                sm = self._extract_schema_mutation(raw_res, q_obj.question)
                verdict = self.abstention_verdict(probabilities=None, free_energy=energy, conformal_set=conf_set)

                typed_results.append(
                    RangeResult(
                        question=q_obj.question,
                        expected_value=expected_val,
                        interval_low=int_low,
                        interval_high=int_high,
                        confidence_interval=ci,
                        min_val=q_obj.min_val,
                        max_val=q_obj.max_val,
                        confidence=conf,
                        temperature=float(raw_res.get("temperature", 1.0)),
                        conformal_set=conf_set,
                        free_energy=energy,
                        schema_mutation=sm,
                        abstention_verdict=verdict,
                    )
                )
            elif q_type == "extract":
                extracted_val = raw_res.get("extracted_value", None)
                start_i = raw_res.get("start_idx", None)
                end_i = raw_res.get("end_idx", None)
                conf = raw_res.get("confidence", None)

                if not extracted_val:
                    # The model returned no value: report an empty value with zero confidence.
                    # No regex or structured-state lookup stands in for the model.
                    extracted_val = ""
                    start_i = 0
                    end_i = 0
                    conf = 0.0
                elif conf is None:
                    conf = 0.0

                if start_i is None or end_i is None:
                    state_str = str(state)
                    start_i = state_str.find(extracted_val)
                    if start_i == -1:
                        start_i = 0
                    end_i = start_i + len(extracted_val)

                extract_probs = [float(conf), max(0.0, 1.0 - float(conf))]
                conf_set, energy = self._extract_conformal_and_energy(
                    raw_res, probabilities=extract_probs, scores=raw_res.get("scores"), temperature=float(raw_res.get("temperature", 1.0))
                )
                sm = self._extract_schema_mutation(raw_res, q_obj.slot_name)
                verdict = self.abstention_verdict(probabilities=extract_probs, free_energy=energy, conformal_set=conf_set)

                typed_results.append(
                    ExtractResult(
                        slot_name=q_obj.slot_name,
                        extracted_value=extracted_val,
                        start_idx=start_i,
                        end_idx=end_i,
                        confidence=float(conf),
                        temperature=float(raw_res.get("temperature", 1.0)),
                        conformal_set=conf_set,
                        free_energy=energy,
                        schema_mutation=sm,
                        abstention_verdict=verdict,
                    )
                )
            elif q_type == "cascade":
                # Cascade primitives are evaluated as a composite branch.
                cascade_result = self._evaluate_cascade(q_obj, state, delta_state=delta_state, is_delta=is_delta, latent_tensors=latent_tensors, **kwargs)
                typed_results.append(cascade_result)
            else:
                typed_results.append(raw_res)

        return typed_results

    def _evaluate_cascade(
        self,
        cascade: Cascade,
        state: Any,
        delta_state: Optional[Any] = None,
        is_delta: bool = False,
        latent_tensors: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CascadeResult:
        """
        Executes a multi-stage decision cascade where subsequent stages can condition
        on prior stage outputs.
        """
        context = {}
        stage_results = []
        for stage in cascade.stages:
            res = self.evaluate(state, [stage], delta_state=delta_state, is_delta=is_delta, latent_tensors=latent_tensors, **kwargs)[0]
            stage_results.append(res)
            if isinstance(res, NoulResult):
                context[res.proposition] = res.decision
            elif isinstance(res, ChoiceResult):
                context[res.question] = res.choice_label
            elif isinstance(res, ScoreResult):
                context[res.question] = res.score
            elif isinstance(res, MultiChoiceResult):
                context[res.question] = res.selected_options
            elif isinstance(res, RangeResult):
                context[res.question] = res.expected_value
            elif isinstance(res, ExtractResult):
                context[res.slot_name] = res.extracted_value

        confidences = [res.confidence for res in stage_results if hasattr(res, "confidence")]
        temperatures = [res.temperature for res in stage_results if hasattr(res, "temperature")]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        temperature = sum(temperatures) / len(temperatures) if temperatures else 0.0

        stage_energies = [getattr(s, "free_energy", None) for s in stage_results if getattr(s, "free_energy", None) is not None]
        free_energy = max(stage_energies) if stage_energies else None

        stage_mutations = [getattr(s, "schema_mutation", None) for s in stage_results if getattr(s, "schema_mutation", None) is not None]
        schema_mutation = stage_mutations[-1] if stage_mutations else None

        stage_conformal = [getattr(s, "conformal_set", None) for s in stage_results if getattr(s, "conformal_set", None) is not None]
        if stage_conformal:
            is_ambiguous = any(c.get("is_ambiguous", False) for c in stage_conformal)
            is_empty = any(c.get("is_empty", False) for c in stage_conformal)
            conformal_set = {
                "is_ambiguous": is_ambiguous,
                "is_singleton": not is_ambiguous and not is_empty,
                "is_empty": is_empty,
                "stage_conformal_sets": stage_conformal,
            }
        else:
            conformal_set = None

        any_abstained = any(getattr(s, "should_abstain", False) for s in stage_results)
        cascade_decision = ABSTAIN if any_abstained else ANSWER
        reasons = [
            f"stage {idx}: {r}"
            for idx, s in enumerate(stage_results)
            if hasattr(s, "abstention_verdict") and s.abstention_verdict
            for r in s.abstention_verdict.reasons
        ]
        cascade_verdict = AbstentionVerdict(
            decision=cascade_decision,
            reasons=reasons,
            free_energy=free_energy,
        )

        return CascadeResult(
            id=cascade.id,
            stages=stage_results,
            decisions=context,
            confidence=confidence,
            temperature=temperature,
            conformal_set=conformal_set,
            free_energy=free_energy,
            schema_mutation=schema_mutation,
            abstention_verdict=cascade_verdict,
        )

    def cascade(
        self,
        state: Any,
        stages: Union[
            List[Callable[[Dict[str, Any]], Union[Noul, Choice, Score, MultiChoice, Range, Extract]]],
            List[Union[Noul, Choice, Score, MultiChoice, Range, Extract]],
            Cascade,
        ],
    ) -> Union[List[Union[NoulResult, ChoiceResult, ScoreResult, MultiChoiceResult, RangeResult, ExtractResult]], CascadeResult]:
        """
        Executes a multi-stage decision cascade where subsequent stages can condition
        on prior stage outputs.

        Args:
            state: Any structured state object.
            stages: Either callable stages (functions of context) or explicit primitive list.

        Returns:
            List of typed results for callable stages, or CascadeResult for explicit list.
        """
        if isinstance(stages, Cascade):
            return self._evaluate_cascade(stages, state)

        if callable(stages[0]):
            context = {}
            stage_results = []
            for stage_fn in stages:
                question = stage_fn(context)
                res = self.evaluate(state, [question])[0]
                stage_results.append(res)
                if isinstance(res, NoulResult):
                    context[res.proposition] = res.decision
                elif isinstance(res, ChoiceResult):
                    context[res.question] = res.choice_label
                elif isinstance(res, ScoreResult):
                    context[res.question] = res.score
                elif isinstance(res, MultiChoiceResult):
                    context[res.question] = res.selected_options
                elif isinstance(res, RangeResult):
                    context[res.question] = res.expected_value
                elif isinstance(res, ExtractResult):
                    context[res.slot_name] = res.extracted_value
            return stage_results

        return self._evaluate_cascade(Cascade(stages=stages), state)


class AdvancedRhenEngine(RhenEngine):
    """RhenEngine with OOD-calibrated latent search and speculative / schema operations."""

    def __init__(self, config=None, model=None, tokenizer=None, device=None):
        if isinstance(config, torch.nn.Module):
            if model is not None and tokenizer is None:
                tokenizer = model
            model = config
            config = getattr(model, 'config', None)
        elif config is not None:
            if model is None:
                try:
                    from rhen.model import EOSModel
                except ImportError:
                    from model import EOSModel
                model = EOSModel(config)
        elif model is None:
            try:
                from rhen.config import EOSConfig
                from rhen.model import EOSModel
            except ImportError:
                from config import EOSConfig
                from model import EOSModel
            config = EOSConfig()
            model = EOSModel(config)
        super().__init__(config=config, model=model, tokenizer=tokenizer, device=device)
        self.config = config

    def simulate_latent_outcomes(
        self,
        state: Any,
        question: str,
        options: List[str],
        horizon: int = 2,
    ) -> LatentRolloutResult:
        """Calibrated-choice horizon projection (see ``RhenEngine.simulate_latent_outcomes``)."""
        return super().simulate_latent_outcomes(
            state=state,
            question=question,
            options=options,
            horizon=horizon,
        )

    def latent_search(
        self,
        state: Any,
        root_question: Any,
        candidate_branches: Sequence[Union[str, int]],
        beam_width: int = 3,
        max_depth: int = 3,
        transition_operator: Optional[Callable] = None,
        prospective_value_head: Optional[Callable] = None,
        prm_scorer: Optional[Callable] = None,
        conformal_alpha: float = 0.05,
        **kwargs: Any,
    ) -> LatentSearchResult:
        """Latent search with an OOD-calibrated budget.

        A state whose explicit free energy exceeds the OOD threshold is treated as
        off-manifold and gets a wider and deeper search (at least beam 4, depth 4).
        Otherwise, or when the model gives no explicit energy, the caller's budget
        is used unchanged.
        """
        calibrated_beam = beam_width
        calibrated_depth = max_depth

        if self.ood_detector is not None:
            choice_res = self.choice(state, str(root_question), [str(c) for c in candidate_branches])
            # Use only the model's explicit energy (from its raw scores). Energy derived from
            # the normalized choice probabilities would be identically 0 and never fire.
            energy = choice_res.free_energy
            if energy is not None and energy > getattr(self.ood_detector, "ood_threshold", 4.5):
                calibrated_beam = max(beam_width, 4)
                calibrated_depth = max(max_depth, 4)

        return super().latent_search(
            state=state,
            root_question=root_question,
            candidate_branches=candidate_branches,
            beam_width=calibrated_beam,
            max_depth=calibrated_depth,
            transition_operator=transition_operator,
            prospective_value_head=prospective_value_head,
            prm_scorer=prm_scorer,
            conformal_alpha=conformal_alpha,
            **kwargs,
        )

    def speculative_cascade(
        self,
        state: Any,
        root_question: Any,
        speculative_branches: Dict[int, List[Any]],
        latent_tensors: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Expose speculative_cascade for O(1) multi-path tree execution."""
        return super().speculative_cascade(
            state=state,
            root_question=root_question,
            speculative_branches=speculative_branches,
            latent_tensors=latent_tensors,
            **kwargs,
        )

    def dynamic_schema_mutation(
        self,
        state: Any = None,
        question: Optional[Any] = None,
        mutation: Optional[SchemaMutation] = None,
        latent_tensors: Optional[Dict[str, Any]] = None,
        state_or_result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes the dynamic schema mutation carried by a decision, or predicted for a question:

        - ``spawn_child``: expands the decision tree. Requires explicit
          ``speculative_branches``, and ``root_options`` when the question is a string.
        - ``switch_primitive``: re-evaluates the question as another primitive.
        - ``route_tool``: reports the tool route for action-card staging.
        - ``terminate``: halts further cascade progression.
        - ``noop``: no structural change.
        - ``refine`` / ``verify`` / ``escalate``: named by the schema, reported back
          with ``status='not_executed'`` for the caller to act on.

        With no schema mutation available, returns ``action='none'`` and
        ``status='no_mutation'``. No default mutation or option set is invented.
        """
        target = state if state is not None else state_or_result
        if question is None:
            # Called with a single typed result (ChoiceResult, NoulResult, etc.): the
            # result's own schema mutation is the evidence. The state is not recoverable
            # from a result, so it must be passed explicitly for actions that re-evaluate.
            if mutation is None:
                mutation = getattr(target, "schema_mutation", None)
            question = getattr(target, "question", None) or getattr(target, "proposition", None)
            state = kwargs.get("state")
        else:
            state = target
            if mutation is None:
                eval_q = Noul(proposition=question) if isinstance(question, str) else question
                res = self.evaluate(state, [eval_q], latent_tensors=latent_tensors, **kwargs)[0]
                mutation = getattr(res, "schema_mutation", None)

        if mutation is None:
            return {"action": "none", "mutation": None, "status": "no_mutation"}
        if mutation.mutation_type not in SCHEMA_MUTATION_TYPES:
            raise ValueError(f"unknown schema mutation type {mutation.mutation_type!r}")

        mut_type = mutation.mutation_type
        if mut_type in ("spawn_child", "switch_primitive") and (state is None or question is None):
            raise ValueError(f"{mut_type} re-evaluates a decision: pass the state and the question explicitly")

        if mut_type == "spawn_child":
            child_branches = kwargs.get("speculative_branches")
            if not child_branches:
                raise ValueError("spawn_child requires explicit speculative_branches")
            if isinstance(question, str):
                root_options = kwargs.get("root_options")
                if not root_options:
                    raise ValueError("spawn_child on a string question requires root_options (or pass a Choice)")
                root_q = Choice(question=question, options=[str(o) for o in root_options])
            else:
                root_q = question
            cascade_res = self.speculative_cascade(
                state=state,
                root_question=root_q,
                speculative_branches=child_branches,
                latent_tensors=latent_tensors,
            )
            return {
                "action": "spawn_child",
                "mutation": mutation,
                "result": cascade_res,
            }
        if mut_type == "switch_primitive":
            target_prim = (mutation.metadata or {}).get("target_primitive", "choice")
            if target_prim == "choice" and isinstance(question, Noul):
                new_q = Choice(question=question.proposition, options=["Yes", "No", "Uncertain"])
            elif target_prim == "noul" and isinstance(question, Choice):
                new_q = Noul(proposition=question.question)
            elif isinstance(question, str):
                new_q = Choice(question=question, options=["Yes", "No"]) if target_prim == "choice" else Noul(proposition=question)
            else:
                new_q = question
            new_res = self.evaluate(state, [new_q], latent_tensors=latent_tensors, **kwargs)[0]
            return {
                "action": "switch_primitive",
                "mutation": mutation,
                "switched_question": new_q,
                "result": new_res,
            }
        if mut_type == "route_tool":
            return {
                "action": "route_tool",
                "mutation": mutation,
                "route_target": (mutation.metadata or {}).get("tool", "ActionCardStaging"),
            }
        if mut_type == "terminate":
            return {
                "action": "terminate",
                "mutation": mutation,
                "status": "terminated",
            }
        if mut_type == "noop":
            return {"action": "none", "mutation": mutation, "status": "noop"}
        # refine / verify / escalate: returned for the caller; nothing is executed here.
        return {"action": mut_type, "mutation": mutation, "status": "not_executed"}

    apply_schema_mutation = dynamic_schema_mutation
    mutate_schema = dynamic_schema_mutation


RhenEngine.decide = RhenEngine.evaluate


# --------------------------------------------------------------------------------------
# Top-Level Ergonomic API (rhen.decide & rhen.evaluate)
# --------------------------------------------------------------------------------------

_DEFAULT_ENGINE: Optional[RhenEngine] = None


class RhenAbstentionError(RuntimeError):
    """Raised when on_abstain='raise' and a decision is abstained or requires clarification."""
    def __init__(self, message: str, verdict: Optional[AbstentionVerdict] = None, result: Optional[Any] = None):
        super().__init__(message)
        self.verdict = verdict
        self.result = result


def get_default_engine() -> RhenEngine:
    """Return process-level cached default RhenEngine instance."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = RhenEngine()
    return _DEFAULT_ENGINE


def reset_default_engine() -> None:
    """Reset the process-level cached default RhenEngine instance."""
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = None


def decide(
    state: Any,
    questions: Union[Any, List[Any]],
    on_abstain: str = "return",
    engine: Optional[RhenEngine] = None,
    **kwargs: Any,
) -> Any:
    """
    High-level ergonomic entrypoint for non-autoregressive decision making with Rhen.

    Args:
        state: Structured state (dict, list, JSON string, or raw text). Automatically
               converted to formatted state representation via ContextBuilder.from_data.
        questions: Single decision primitive (e.g. Choice, Noul, Score) or list of primitives.
        on_abstain: Handling policy when an abstention occurs:
                    - 'return': Return the result object as normal (default).
                    - 'raise': Raise RhenAbstentionError if decision is abstained.
                    - 'verdict': Return the AbstentionVerdict object on abstention.
        engine: Optional explicit RhenEngine instance. If None, uses process-level cached default engine.
        **kwargs: Additional parameters passed to engine.evaluate.

    Returns:
        A single result object if a single question was provided, or a list of results if a list was provided.
    """
    if isinstance(questions, (list, tuple)):
        single_input = False
        q_list = list(questions)
    else:
        single_input = True
        q_list = [questions]

    # Validate upfront
    for q in q_list:
        if isinstance(q, Choice):
            if not q.options or len(q.options) == 0:
                raise ValueError("Choice requires at least one option")
        elif isinstance(q, MultiChoice):
            if not q.options or len(q.options) == 0:
                raise ValueError("Choice requires at least one option")
        elif isinstance(q, tuple) and len(q) >= 2 and isinstance(q[1], (list, tuple)):
            if len(q[1]) == 0:
                raise ValueError("Choice requires at least one option")

    if on_abstain not in ("return", "raise", "verdict"):
        raise ValueError(f"Invalid on_abstain parameter: {on_abstain!r}. Must be 'return', 'raise', or 'verdict'")

    # State normalization via ContextBuilder
    try:
        from rhen.context_builder import ContextBuilder
    except ImportError:
        from context_builder import ContextBuilder

    if isinstance(state, (dict, list)):
        normalized_state = ContextBuilder.from_data(state).build_state_string()
    elif isinstance(state, str) and not isinstance(state, PackedRhenSequence):
        stripped = state.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            try:
                json.loads(stripped)
                normalized_state = ContextBuilder.from_data(state).build_state_string()
            except Exception:
                normalized_state = state
        else:
            normalized_state = state
    else:
        normalized_state = state

    eng = engine if engine is not None else get_default_engine()
    results = eng.evaluate(normalized_state, q_list, **kwargs)

    processed = []
    for res in results:
        is_abstained = getattr(res, "should_abstain", False)
        if is_abstained:
            if on_abstain == "raise":
                v = getattr(res, "abstention_verdict", None)
                reasons = v.reasons if v and v.reasons else ["Decision abstained by selective gate"]
                raise RhenAbstentionError(f"Decision abstained: {reasons}", verdict=v, result=res)
            elif on_abstain == "verdict":
                v = getattr(res, "abstention_verdict", None)
                if v is None:
                    probs = (
                        getattr(res, "probabilities", None)
                        or getattr(res, "raw_probabilities", None)
                        or getattr(res, "level_probabilities", None)
                    )
                    v = eng.abstention_verdict(
                        probabilities=probs,
                        free_energy=getattr(res, "free_energy", None),
                        conformal_set=getattr(res, "conformal_set", None),
                    )
                processed.append(v)
                continue
        processed.append(res)

    if single_input:
        return processed[0]
    return processed


evaluate = decide



