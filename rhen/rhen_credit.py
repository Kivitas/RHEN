"""
Rhen-1.0 Credit Assignment and Search layer.

``rhen_cognition.RLCDOptimizer`` normalises advantage *within a group sampled
from one state*, which is correct for a single decision but wrong for a
cascade. This module fixes that and adds the search step that should precede
any policy-gradient training.

1. :class:`TrajectoryGraph` -- step-level credit by Bellman iteration.
   arXiv:2609.28963 (GRAFT) shows GRPO's group-normalised advantage is
   *systematically biased at the step level*, because a failed trajectory can
   still contain valuable steps. GRAFT grafts all rollouts into a trajectory
   graph, recovers node state-values by Bellman iteration, and assigns credit to
   each edge by the node value difference. The paper reports consistent gains
   over GRPO on multi-turn agentic benchmarks.

2. :class:`SegmentAdvantage` -- segment-level advantage correction.
   arXiv:2609.27093 (HISPO) finds that for long-form reasoning the right RLVR
   granularity sits *between* token-level and sequence-level: token-level
   credit is too noisy, sequence-level is too coarse. A Rhen cascade stage is
   exactly such a segment.

3. :class:`SchemaContrastiveAdvantage` -- advantages computed against
   *per-schema* baselines. arXiv:2609.29230 (EAGER) reports that plain group
   baselines suffer advantage collapse under sparse binary rewards, and that
   contrastive advantage estimation over schemas fixes it, beating prior RL
   baselines on seven structured-extraction datasets. Rhen decisions are
   overwhelmingly binary (Noul is Yes/No), so this is the dominant regime.

4. :class:`CascadeSearch` -- search before RL.
   arXiv:2609.01274 derives a Budgeted Operating-Point Transition Rule under
   which a large share of measured RL gain is a *sampling-efficiency shift*
   toward points the base policy already reaches under search. The practical
   reading: run search first, then train, and only count the training gain that
   search could not already deliver. arXiv:2609.25510 (Hill Sampling)
   reinforces this -- repeatedly conditioning on the best verified solution so
   far beat both evolutionary search and test-time weight updates.

All components are pure-PyTorch, dependency-free and CPU-safe.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "TrajectoryNode",
    "TrajectoryGraph",
    "SegmentAdvantage",
    "SchemaContrastiveAdvantage",
    "CascadeSearch",
    "HillSampler",
    "SearchVsRLReport",
    "fit_boptr",
    "search_vs_rl",
]


# ---------------------------------------------------------------------------
# 1. GRAFT: step-level credit via Bellman iteration on a trajectory graph
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryNode:
    """One decision state visited during rollout."""

    node_id: str
    outcome_reward: float = 0.0
    value: float = 0.0
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    visits: int = 0
    terminal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class TrajectoryGraph:
    """Step-level credit assignment (arXiv:2609.28963, GRAFT).

    Rollouts are grafted into a directed graph keyed by state. Node values are
    recovered by value iteration, and each edge receives credit equal to the
    difference between its head and tail values::

        Q(s_t, a) ~= V(s_{t+1}) - V(s_t)

    The point is that a step inside an ultimately-failed trajectory can still
    receive positive credit when it moved toward a higher-value state, which
    trajectory-level normalisation cannot express.
    """

    def __init__(self, gamma: float = 0.99, iterations: int = 32, tol: float = 1e-6) -> None:
        self.gamma = float(gamma)
        self.iterations = int(iterations)
        self.tol = float(tol)
        self.nodes: Dict[str, TrajectoryNode] = {}
        self.edges: List[Tuple[str, str, float]] = []  # (tail, head, step_reward)

    def add_rollout(
        self,
        states: Sequence[str],
        step_rewards: Sequence[float],
        terminal_reward: float = 0.0,
    ) -> None:
        """Graft one trajectory: a list of state keys plus per-step rewards."""
        if len(states) != len(step_rewards) + 1:
            raise ValueError("expected len(states) == len(step_rewards) + 1")
        for s in states:
            if s not in self.nodes:
                self.nodes[s] = TrajectoryNode(node_id=s)
            self.nodes[s].visits += 1
        for i, reward in enumerate(step_rewards):
            tail, head = states[i], states[i + 1]
            self.nodes[tail].children.append(head)
            self.nodes[head].parents.append(tail)
            self.edges.append((tail, head, float(reward)))
        last = states[-1]
        self.nodes[last].terminal = True
        self.nodes[last].outcome_reward += float(terminal_reward)

    def solve(self) -> Dict[str, float]:
        """Value iteration over the graph; returns the node value map.

        Uses the standard expected-return Bellman operator -- each node takes
        the *mean* backup over its outgoing edges -- rather than accumulating
        backups. Accumulating would diverge on cycles, and a shared-prefix
        rollout graph always contains them.
        """
        if not self.nodes:
            return {}
        outgoing: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for tail, head, reward in self.edges:
            outgoing[tail].append((head, float(reward)))

        values: Dict[str, float] = {
            nid: (n.outcome_reward if n.terminal else 0.0)
            for nid, n in self.nodes.items()
        }
        for _ in range(self.iterations):
            nxt: Dict[str, float] = {}
            for nid, node in self.nodes.items():
                edges = outgoing.get(nid, [])
                if not edges:
                    nxt[nid] = node.outcome_reward if node.terminal else values[nid]
                    continue
                backup = sum(
                    reward + self.gamma * values[head] for head, reward in edges
                ) / len(edges)
                # A terminal node keeps its observed outcome as an anchor.
                nxt[nid] = backup if not node.terminal else 0.5 * backup + 0.5 * node.outcome_reward
            delta = max(abs(nxt[k] - values[k]) for k in values)
            values = nxt
            if delta < self.tol:
                break
        for nid, node in self.nodes.items():
            node.value = values[nid]
        return values

    def edge_credit(self) -> List[Dict[str, Any]]:
        """Assign credit to every step as the difference of endpoint values."""
        values = self.solve()
        out: List[Dict[str, Any]] = []
        for tail, head, reward in self.edges:
            out.append(
                {
                    "tail": tail,
                    "head": head,
                    "immediate_reward": reward,
                    "value_delta": values[head] - values[tail],
                    "credit": values[head] - values[tail],
                }
            )
        return out

    def step_level_advantage(
        self, discount: Optional[float] = None
    ) -> Dict[str, float]:
        """Per-tail-state mean advantage, comparable to a group baseline.

        This is the drop-in replacement for the group-normalised advantage that
        ``RLCDOptimizer`` currently uses at the *sequence* level.
        """
        gamma = self.gamma if discount is None else float(discount)
        buckets: Dict[str, List[float]] = defaultdict(list)
        for tail, head, reward in self.edges:
            buckets[tail].append(reward + gamma * self.nodes[head].value)
        result: Dict[str, float] = {}
        for state, vals in buckets.items():
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            result[state] = mean / (std + 1e-8) if len(vals) > 1 else mean
        return result


# ---------------------------------------------------------------------------
# 2. Segment-level advantage (HISPO granularity)
# ---------------------------------------------------------------------------


class SegmentAdvantage:
    """Segment-level advantage correction (arXiv:2609.27093, HISPO).

    For long-form rollouts, token-level credit is too noisy and
    sequence-level credit is too coarse. HISPO places the correction *between*
    the two and reports gains over both GRPO and GSPO, including +3.75 Acc@8 on
    AIME25 over GRPO.

    For a Rhen cascade the natural segment is one **stage**: stage 1's outcome
    should not be credited or blamed for stage 3's result, but the whole
    sequence should not be flattened either.
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = True) -> None:
        self.gamma = float(gamma)
        self.normalize = bool(normalize)

    def segment_returns(
        self, rewards: Sequence[float], terminal_bonus: float = 0.0
    ) -> List[float]:
        """Discounted return *from* each segment onwards."""
        n = len(rewards)
        out = [0.0] * n
        running = float(terminal_bonus)
        for i in range(n - 1, -1, -1):
            running = float(rewards[i]) + self.gamma * running
            out[i] = running
        return out

    def baseline(self, segment_rewards: Sequence[Sequence[float]]) -> Dict[int, float]:
        """Per-segment-index baseline averaged across rollouts (GSPO-style)."""
        if not segment_rewards:
            return {}
        width = max(len(r) for r in segment_rewards)
        out: Dict[int, float] = {}
        for i in range(width):
            vals = [r[i] for r in segment_rewards if i < len(r)]
            if vals:
                out[i] = sum(vals) / len(vals)
        return out

    def advantage(
        self, rewards: Sequence[float], terminal_bonus: float = 0.0
    ) -> List[float]:
        """Subtract this rollout's own mean (a self-consistent baseline)."""
        returns = self.segment_returns(rewards, terminal_bonus)
        if not self.normalize or len(returns) < 2:
            return returns
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = var ** 0.5
        if std < 1e-8:
            return [0.0 for _ in returns]
        return [(r - mean) / std for r in returns]


# ---------------------------------------------------------------------------
# 3. Schema-contrastive advantage (EAGER)
# ---------------------------------------------------------------------------


class SchemaContrastiveAdvantage:
    """Contrastive advantage estimation across decision schemas.

    arXiv:2609.29230 (EAGER) reports advantage collapse under sparse binary
    rewards, and shows that *Schema-Contrastive Advantage Estimation* -- which
    baselines each schema against the batch rather than against samples of the
    same schema -- fixes it, outperforming prompting, SFT and prior RL on seven
    structured-extraction datasets.

    This matters disproportionately for Rhen because its primitives are mostly
    binary (Noul is Yes/No). Under a single global baseline, an all-negative
    Noul batch produces zero advantage and therefore no gradient at all, even
    though the relative confidence ordering still carries information.

    Schemas are supplied as free-form strings (``"noul"``, ``"choice:3"``, ...).
    """

    def __init__(self, clip: float = 5.0, min_schema_count: int = 2) -> None:
        self.clip = float(clip)
        self.min_schema_count = int(min_schema_count)
        self.fallback_history: List[int] = []

    def _clip(self, value: float) -> float:
        return max(-self.clip, min(self.clip, value))

    def advantage(
        self, rewards: Sequence[float], schemas: Sequence[str]
    ) -> List[float]:
        """Per-sample advantage, baselined within its own schema when possible."""
        n = len(rewards)
        if n != len(schemas):
            raise ValueError("rewards and schemas must have equal length")
        if n == 0:
            return []

        by_schema: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(schemas):
            by_schema[str(s)].append(i)

        batch_mean = sum(float(r) for r in rewards) / n
        out: List[float] = [0.0] * n
        for schema, idxs in by_schema.items():
            vals = [float(rewards[i]) for i in idxs]
            # A schema seen too rarely has no usable local baseline; fall back
            # to the batch mean and record that we did so.
            if len(vals) < self.min_schema_count:
                self.fallback_history.append(len(vals))
                for i, v in zip(idxs, vals):
                    out[i] = self._clip(v - batch_mean)
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            for i, v in zip(idxs, vals):
                out[i] = self._clip((v - mean) / std) if std > 1e-8 else 0.0
        return out

    def diagnose(self, rewards: Sequence[float], schemas: Sequence[str]) -> Dict[str, Any]:
        """Report whether a batch would have collapsed under a global baseline.

        ``collapsed`` is True when a naive single-baseline scheme yields no
        gradient signal because every reward in the batch is identical -- the
        failure mode EAGER targets. ``rescued`` shows whether the contrastive
        estimator still recovers usable signal from that same batch.
        """
        n = len(rewards)
        distinct = len(set(float(r) for r in rewards))
        mean = sum(float(r) for r in rewards) / n if n else 0.0
        naive_std = (
            (sum((float(r) - mean) ** 2 for r in rewards) / n) ** 0.5 if n else 0.0
        )
        per_schema = self.advantage(rewards, schemas)
        c_mean = sum(per_schema) / n if n else 0.0
        contrastive_std = (
            (sum((a - c_mean) ** 2 for a in per_schema) / n) ** 0.5 if n else 0.0
        )
        collapsed = naive_std < 1e-8
        return {
            "batch_size": n,
            "n_schemas": len(set(schemas)),
            "distinct_rewards": distinct,
            "naive_global_std": naive_std,
            "collapsed": collapsed,
            "contrastive_std": contrastive_std,
            "rescued": bool(collapsed and contrastive_std > 1e-8),
        }


# ---------------------------------------------------------------------------
# 4. Search before RL (BOPTR + Hill Sampling)
# ---------------------------------------------------------------------------


@dataclass
class SearchVsRLReport:
    """How much of a policy's apparent RL gain search already explains."""

    best_single: float
    best_search: float
    rl_checkpoint: Optional[float]
    unexplained_by_search: Optional[float]
    boptr_alpha: Optional[float]
    boptr_beta: Optional[float]
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class CascadeSearch:
    """Best-of-N and beam search over a decision cascade.

    arXiv:2609.01274 studies RLVR gains through a Unified Decoding Framework and
    derives a **Budgeted Operating-Point Transition Rule**::

        N_base ~= alpha * N_rl ** beta

    i.e. much of a policy's measured RL gain is a shift in *sampling
    efficiency* toward points the base policy can already reach under search.
    The practical consequence for Rhen: measure the search-only baseline
    **first**, so any later RL gain is credited only with what search could not
    already deliver.
    """

    def __init__(self, prune_below: float = 0.05) -> None:
        self.prune_below = float(prune_below)
        self.evaluations = 0

    def best_of_n(
        self,
        sample_fn: Callable[[int], Tuple[Any, float]],
        n: int,
    ) -> Dict[str, Any]:
        """Evaluate ``n`` candidates, returning the best and the whole trace."""
        if n <= 0:
            raise ValueError("n must be positive")
        best_item: Any = None
        best_score = -math.inf
        scores: List[float] = []
        items: List[Any] = []
        for i in range(n):
            item, score = sample_fn(i)
            self.evaluations += 1
            scores.append(float(score))
            items.append(item)
            if score > best_score:
                best_score, best_item = float(score), item
        return {
            "best_item": best_item,
            "best_score": best_score,
            "scores": scores,
            "items": items,
            "n": n,
            "mean_score": sum(scores) / n,
        }

    def beam(
        self,
        expand_fn: Callable[[Any], List[Tuple[Any, float]]],
        root: Any,
        width: int = 3,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Beam search over cascade stages, keeping the ``width`` best partials."""
        beam: List[Tuple[Any, float, List[Any]]] = [(root, 0.0, [root])]
        best: Optional[Tuple[Any, float, List[Any]]] = None
        for _ in range(max(0, depth)):
            candidates: List[Tuple[Any, float, List[Any]]] = []
            for node, score, path in beam:
                for child, child_score in expand_fn(node):
                    # Cumulative scoring; child_score is the incremental value
                    # of taking that branch.
                    candidates.append(
                        (child, score + float(child_score), path + [child])
                    )
            if not candidates:
                break
            candidates.sort(key=lambda t: t[1], reverse=True)
            beam = candidates[:width]
            if best is None or beam[0][1] > best[1]:
                best = beam[0]
        final = best or beam[0]
        return {
            "best_item": final[0],
            "best_score": final[1],
            "path": final[2],
            "beam_size": len(beam),
        }


class HillSampler:
    """Hill Sampling: condition each new candidate on the best one so far.

    arXiv:2609.25510 reports that repeatedly sampling candidate edits from a
    *frozen* model, keeping the best, and conditioning all subsequent samples on
    that best, beat both evolutionary search harnesses and test-time parameter
    updates. A striking secondary finding: applying evolution strategies
    directly to the weights at test time performed *worse than setting the
    learning rate to zero* -- the useful part was the random perturbation, not
    the learning. The lesson is to spend budget on conditioning, not on weight
    updates.
    """

    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self.history: List[Dict[str, Any]] = []

    def run(
        self,
        propose_fn: Callable[[Any], List[Tuple[Any, float]]],
        verify_fn: Callable[[Any], bool],
        initial: Any,
        rounds: int = 4,
        keep_alternatives: int = 0,
    ) -> Dict[str, Any]:
        """Iterate propose-verify-accept, always conditioning on the champion.

        Args:
            propose_fn: given the current champion, return ``[(candidate, score)]``.
            verify_fn: predicate deciding whether a candidate is valid; an
                unverified candidate is never accepted regardless of score.
            initial: starting champion.
            rounds: number of proposal rounds.
            keep_alternatives: how many runners-up to retain for diversity.

        ``verify_fn`` is what separates this from naive best-of-N: a high-scoring
        but unverified candidate is discarded, which is the mechanism the
        EvidenceVerificationLattice supplies.
        """
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        champion = initial
        champion_score = -math.inf
        alternatives: List[Tuple[Any, float]] = []
        rejected = 0

        for r in range(rounds):
            candidates = propose_fn(champion)
            round_best: Any = None
            round_score = -math.inf
            for cand, score in candidates:
                if not verify_fn(cand):
                    rejected += 1
                    continue
                if score > round_score:
                    round_score, round_best = float(score), cand
            improved = round_best is not None and round_score > champion_score
            if improved:
                champion, champion_score = round_best, round_score
            if keep_alternatives and round_best is not None:
                alternatives.append((round_best, round_score))
            self.history.append(
                {
                    "round": r + 1,
                    "improved": bool(improved),
                    "best_score": round_score,
                    "champion_score": champion_score,
                    "rejected_unverified": rejected,
                }
            )

        alternatives.sort(key=lambda t: t[1], reverse=True)
        return {
            "champion": champion,
            "champion_score": champion_score,
            "rounds": rounds,
            "rejected_unverified": rejected,
            "alternatives": [a for a, _ in alternatives[:keep_alternatives]],
            "history": self.history,
        }


def fit_boptr(
    base_scores: Sequence[float],
    rl_scores: Sequence[float],
) -> Dict[str, Any]:
    """Fit ``N_base ~ alpha * N_rl ** beta`` in log space (arXiv:2609.01274).

    ``base_scores[i]`` is the best score reachable by the base policy with
    ``i + 1`` samples; ``rl_scores[j]`` is the RL policy's best at ``j + 1``.

    With fewer than two usable points the fit is refused rather than guessed: a
    two-point fit always looks perfect and would manufacture a spurious claim.
    """
    if len(base_scores) < 2 or len(rl_scores) < 2:
        return {
            "fitted": False,
            "reason": "need >= 2 points on each curve to fit two parameters",
        }
    xs = [math.log(j + 1.0) for j in range(len(rl_scores))]
    ys = [math.log(max(float(s), 1e-9)) for s in base_scores[: len(xs)]]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    if var_x < 1e-12:
        return {"fitted": False, "reason": "degenerate rl sample grid"}
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    beta = cov / var_x
    alpha = math.exp(mean_y - beta * mean_x)
    residuals = [ys[i] - (math.log(alpha) + beta * xs[i]) for i in range(n)]
    return {
        "fitted": True,
        "alpha": alpha,
        "beta": beta,
        "rmse": (sum(r * r for r in residuals) / n) ** 0.5,
        "n_points": n,
    }


def search_vs_rl(
    base_scores: Sequence[float],
    rl_scores: Optional[Sequence[float]] = None,
) -> SearchVsRLReport:
    """Quantify how much of an RL gain is really just better sampling.

    Returns the search-only ceiling, the RL ceiling, and the residual genuinely
    attributable to training.
    """
    best_single = float(base_scores[0]) if base_scores else 0.0
    best_search = float(max(base_scores)) if base_scores else 0.0
    rl_ckpt = float(max(rl_scores)) if rl_scores else None
    fit = fit_boptr(base_scores, rl_scores or [])

    unexplained: Optional[float] = None
    if rl_ckpt is not None:
        unexplained = rl_ckpt - best_search
        if unexplained <= 0.01:
            interp = (
                "search alone accounts for the entire RL gain; training added "
                "nothing beyond allocating compute to search"
            )
        elif unexplained < 0.5 * max(rl_ckpt - best_single, 1e-9):
            interp = "majority of the RL gain is explained by sampling efficiency"
        else:
            interp = "substantial residual gain attributable to training"
    else:
        interp = (
            "no RL checkpoint supplied; report this search ceiling as the "
            "baseline any future RL result must beat"
        )

    return SearchVsRLReport(
        best_single=best_single,
        best_search=best_search,
        rl_checkpoint=rl_ckpt,
        unexplained_by_search=unexplained,
        boptr_alpha=fit.get("alpha"),
        boptr_beta=fit.get("beta"),
        interpretation=interp,
    )








