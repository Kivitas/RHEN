"""
Cognitive Extensions.

Six new capabilities layered on top of the foundational pillars, each derived from a
specific, documented weakness in the current competitive field (JEV / Open-Jev /
Laya / TypeSafe):

1. :class:`RLCDOptimizer` -- Reinforcement Learning for Calibrated Decisions.
   Laya's model card describes RLCD-style calibration. Rhen previously shipped
   supervised objectives only (DDPO + conformal). This adds a group-relative
   policy gradient (GRPO-style) over the Rhen decision head with a
   *calibration-aware reward*, so confidence is optimised jointly with
   correctness.

2. :class:`DurationThresholdingHead` -- Decision Duration Thresholding (TDT).
   NVIDIA's parakeet-tdt predicts, alongside each output, *how many* outputs to
   emit before stopping, and terminates when the predicted duration falls below a
   threshold. Rhen ports this to decision branches: a learned per-branch loop
   budget means easy questions halt early and only hard ones pay for extra
   recurrent depth.

3. :class:`HierarchicalChoiceRouter` -- two-level coarse-to-fine choice.
   Laya's published Banking77 figure is 0.425 on 77 labels, against Jev's 0.870
   on 72 labels; the label sets differ, so the two are not directly comparable.
   This router clusters K options into G groups by text similarity, selects the
   group in one pass using a medoid label, then discriminates inside it.

4. :class:`OODRegressionGuard` -- a regression gate on out-of-distribution
   optimal-action accuracy. Open-Jev openly reports that "Wiki OOD optimal-action
   accuracy regressed for 9B and 27B on 18 examples each". This guard tracks a
   held-out scoreboard and refuses to accept a change that degrades it.

5. :class:`EvidenceVerificationLattice` -- observer / critic / verifier roles with
   reproducible seeds, held-out scenarios and recorded controls, following the
   loftwah JEV framework's insistence on "enough evidence to distinguish an
   improvement from a prettier screenshot".

6. :class:`MultilingualDecisionRouter` -- script/language detection and routing.
   A heuristic over about 13 language tags (stopword and script rules), with no
   LLM call. It is not equivalent to Laya's 100+ language coverage and has no
   accuracy evaluation in this repo.

All components are pure-PyTorch and CPU-safe; none require network access or
external model weights. They are standalone utilities: nothing here is wired into
`EOSModel` yet, so instantiating the model does not allocate these parameters.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "RLCDReward",
    "RLCDOptimizer",
    "DurationThresholdingHead",
    "HierarchicalChoiceRouter",
    "OODRegressionGuard",
    "EvidenceRecord",
    "EvidenceVerificationLattice",
    "MultilingualDecisionRouter",
]


# ===========================================================================
# Pillar 8: RLCD -- Reinforcement Learning for Calibrated Decisions
# ===========================================================================


@dataclass
class RLCDReward:
    """
    Decomposed reward for a single decision.

    Attributes:
        total: scalar reward handed to the policy gradient.
        correctness: +1 / -1 depending on whether the sampled option matched
            the label.
        calibration: penalty on the Brier term ``(p - y)^2`` of the sampled
            option, so high confidence on a wrong answer is punished hardest.
        margin: reward for a decisive *and* correct answer, encouraging the
            policy to sharpen on positives rather than flattening everywhere.
    """

    total: float
    correctness: float
    calibration: float
    margin: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "correctness": self.correctness,
            "calibration": self.calibration,
            "margin": self.margin,
        }


class RLCDOptimizer:
    """
    Group-relative policy gradient (GRPO) for calibrated decisions.

    For each state we sample ``group_size`` candidate options from the decision
    distribution, score each with a calibration-aware reward, then normalise the
    rewards within the group to obtain a zero-mean advantage. The policy update
    is the standard clipped surrogate::

        A_i = (r_i - mean(r)) / (std(r) + eps)
        L   = -mean( min( r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i ) )

    A group baseline (rather than a learned value network) keeps the optimiser
    lightweight, which matters for an EOSModel (see EOSModel.count_parameters()) intended to run on
    commodity CPUs.

    References (not verified by the author of this comment): Laya / ConvAI
    arXiv:2503.23303 and arXiv:2510.01237, and the RLCD description in the
    Open-Jev README.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
        calibration_weight: float = 1.0,
        margin_weight: float = 0.25,
        group_size: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.clip_eps = float(clip_eps)
        self.kl_coef = float(kl_coef)
        self.calibration_weight = float(calibration_weight)
        self.margin_weight = float(margin_weight)
        self.group_size = int(group_size)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.step_count = 0
        self.reward_history: List[float] = []

    def compute_reward(
        self,
        probabilities: Sequence[float],
        sampled_index: int,
        label_index: int,
    ) -> RLCDReward:
        """
        Calibration-aware reward for one sampled decision.

        ``calibration`` is the Brier contribution ``(p - y)^2`` of the sampled
        option. Weighting it negatively is what makes this "RL for *calibrated*
        decisions" rather than plain RL: a policy assigning 0.99 to a wrong
        option is punished far harder than one assigning 0.51.
        """
        probs = torch.as_tensor(list(probabilities), dtype=torch.float32)
        if probs.numel() == 0:
            return RLCDReward(0.0, 0.0, 0.0, 0.0)
        probs = probs / probs.sum().clamp_min(1e-12)

        idx = int(min(max(sampled_index, 0), probs.numel() - 1))
        p = float(probs[idx])
        correct = 1.0 if idx == int(label_index) else -1.0
        one_hot = 1.0 if idx == int(label_index) else 0.0
        calibration = (p - one_hot) ** 2

        best = float(probs.max())
        margin = max(0.0, best - p) if correct > 0 else 0.0

        total = correct - self.calibration_weight * calibration + self.margin_weight * margin
        return RLCDReward(
            total=float(total),
            correctness=float(correct),
            calibration=float(calibration),
            margin=float(margin),
        )


    def policy_gradient_step(
        self,
        logits: torch.Tensor,
        label_index: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Perform one GRPO update on a single decision distribution.

        Args:
            logits: 1-D tensor of raw option scores (ideally requires grad).
            label_index: index of the ground-truth option.
            optimizer: optimiser whose graph includes ``logits``. When ``None``
                the method computes diagnostics only and mutates nothing.

        Returns:
            Dict with mean reward, advantage spread, KL, gradient norm and the
            per-sample reward breakdown.
        """
        if logits.dim() != 1:
            raise ValueError("policy_gradient_step expects a 1-D logits tensor")

        probs = F.softmax(logits, dim=-1)
        k = int(probs.numel())
        if k == 0:
            raise ValueError("cannot run RLCD on an empty option set")

        # Sample a group of candidate indices.
        group = torch.randint(
            low=0, high=k, size=(self.group_size,), generator=self.generator
        )
        if int(label_index) not in group.tolist():
            # Always include the label so the advantage signal is non-degenerate.
            group[0] = int(label_index)

        # Score the group.
        rewards: List[RLCDReward] = []
        log_probs = torch.log(probs.clamp_min(1e-12))
        for g in group.tolist():
            rewards.append(self.compute_reward(probs.detach().tolist(), g, label_index))

        reward_tensor = torch.tensor([r.total for r in rewards], dtype=torch.float32)
        mean_r = reward_tensor.mean()
        std_r = reward_tensor.std(unbiased=False)
        advantages = (reward_tensor - mean_r) / (std_r + 1e-6)

        # Clipped surrogate objective over the sampled group.
        ratio = torch.exp(log_probs[group] - log_probs[group].detach())
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        # KL penalty toward the pre-update policy.
        with torch.no_grad():
            ref_probs = probs.detach()
        kl = torch.sum(
            ref_probs
            * (torch.log(ref_probs.clamp_min(1e-12)) - torch.log(probs.clamp_min(1e-12)))
        )
        loss = policy_loss + self.kl_coef * kl

        grad_norm = 0.0
        if optimizer is not None and logits.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            if logits.grad is not None:
                grad_norm = float(logits.grad.norm())
            optimizer.step()

        self.step_count += 1
        self.reward_history.append(float(mean_r))

        return {
            "loss": float(loss.detach()),
            "mean_reward": float(mean_r),
            "advantage_spread": float(std_r),
            "kl": float(kl.detach()),
            "grad_norm": grad_norm,
            "sampled_indices": group.tolist(),
            "rewards": [r.to_dict() for r in rewards],
        }



# ===========================================================================
# Pillar 9: Decision Duration Thresholding (TDT)
# ===========================================================================


class DurationThresholdingHead(torch.nn.Module):
    """
    Port of NVIDIA parakeet's Token Duration Thresholding to Rhen decision
    branches.

    parakeet-tdt emits, at every decoder step, both a token *and* the number of
    additional tokens it expects to produce; decoding stops as soon as the
    predicted duration falls to zero. This head does the same for decisions: for
    each question branch it predicts how many extra recurrent loops are still
    needed, and the loop controller halts early once the predicted duration drops
    to (or below) the threshold.

    Consequence for Rhen: a cascade of K stages no longer costs K x max_loops.
    Trivial questions halt after minimum depth; only genuinely ambiguous branches
    pay for the full recurrent budget.

    Reference: ``nvidia/parakeet-tdt-0.6b-v3``, ``moondream/parakeet-ultra``.
    """

    def __init__(
        self,
        hidden_size: int,
        max_duration: int = 8,
        threshold: int = 0,
    ) -> None:
        super().__init__()
        self.max_duration = int(max_duration)
        self.threshold = int(threshold)
        self.duration_proj = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict a non-negative duration (in recurrent loops) per branch.

        Args:
            hidden: ``[B, D]`` or ``[B, 1, D]`` branch representations.

        Returns:
            ``[B]`` integer loop budgets clamped to ``[threshold, max_duration]``.
        """
        z = hidden.squeeze(1) if hidden.dim() == 3 else hidden
        if z.dim() == 1:
            z = z.unsqueeze(0)
        raw = self.duration_proj(z).squeeze(-1)
        durations = torch.round(torch.sigmoid(raw) * self.max_duration)
        return durations.clamp(min=self.threshold, max=self.max_duration).long()

    def should_halt(self, duration: int) -> bool:
        """True when the branch has converged and may stop early."""
        return int(duration) <= self.threshold

    def expected_loops(
        self,
        hidden: torch.Tensor,
        min_loops: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute a halting plan for a batch of branches.

        Returns:
            Dict with per-branch durations, the derived loop budget, halt flags,
            and the expected total loops (the quantity the TDT speedup is
            measured against).
        """
        durations = self.forward(hidden)
        loop_budget = (durations + int(min_loops)).clamp(min=1, max=self.max_duration)
        return {
            "durations": durations.tolist(),
            "loop_budget": loop_budget.tolist(),
            "halt_flags": [self.should_halt(int(d)) for d in durations.tolist()],
            "expected_total_loops": float(loop_budget.sum().item()),
            "naive_total_loops": float(
                int(durations.numel()) * int(min_loops) * self.max_duration
            ),
        }

    def compute_savings(self, hidden: torch.Tensor, min_loops: int = 1) -> float:
        """Fraction of loop iterations saved versus always running to max depth."""
        plan = self.expected_loops(hidden, min_loops=min_loops)
        naive = plan["naive_total_loops"]
        if naive <= 0:
            return 0.0
        return 1.0 - (plan["expected_total_loops"] / naive)



# ===========================================================================
# Pillar 10: Hierarchical Coarse-to-Fine Choice
# ===========================================================================


def _char_trigrams(text: Any) -> frozenset:
    """Set of lowercased character trigrams, padded so every string has at least one."""
    padded = f"  {' '.join(str(text).lower().split())} "
    return frozenset(padded[i:i + 3] for i in range(len(padded) - 2))


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two trigram sets, in [0, 1]."""
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _balanced_clusters(
    options: Sequence[Any], num_groups: int, max_iterations: int = 10
) -> Tuple[List[List[int]], List[int]]:
    """
    Deterministic, balanced k-medoids over character-trigram similarity.

    Returns ``(clusters, medoids)``. ``clusters[j]`` holds the sorted option
    indices of group ``j`` and ``medoids[j]`` is the member of that group with
    the highest total similarity to the other members. Steps:

    1. Seed ``G`` medoids by farthest-point traversal from option 0 (ties go to
       the lower index).
    2. Assign options to medoids in order of decreasing similarity, subject to a
       capacity of ``ceil(K / G)`` per group. Each medoid is pinned to its own
       group first, so no group is ever empty and no group exceeds capacity.
    3. Recompute each group's medoid and reassign, until stable or
       ``max_iterations`` is reached.
    """
    k = len(options)
    if k == 0:
        return [], []
    g = min(max(1, int(num_groups)), k)
    cap = -(-k // g)  # ceil(k / g)
    feats = [_char_trigrams(o) for o in options]

    # 1. Farthest-point seeding.
    seeds = [0]
    nearest = [_jaccard(feats[i], feats[0]) for i in range(k)]
    while len(seeds) < g:
        chosen = set(seeds)
        nxt = min((i for i in range(k) if i not in chosen), key=lambda i: (nearest[i], i))
        seeds.append(nxt)
        nearest = [max(nearest[i], _jaccard(feats[i], feats[nxt])) for i in range(k)]

    def assign(medoids: List[int]) -> List[List[int]]:
        clusters: List[List[int]] = [[m] for m in medoids]
        owner = [-1] * k
        for j, m in enumerate(medoids):
            owner[m] = j
        pairs = sorted(
            (-_jaccard(feats[i], feats[m]), i, j)
            for j, m in enumerate(medoids)
            for i in range(k)
        )
        for _neg_sim, i, j in pairs:
            if owner[i] == -1 and len(clusters[j]) < cap:
                clusters[j].append(i)
                owner[i] = j
        return [sorted(c) for c in clusters]

    def medoid_of(cluster: List[int]) -> int:
        return max(
            cluster,
            key=lambda m: (sum(_jaccard(feats[m], feats[n]) for n in cluster), -m),
        )

    # 2-3. Assign, then refine medoids until the partition is stable.
    medoids = [medoid_of(c) for c in assign(seeds)]
    clusters = assign(medoids)
    for _ in range(max_iterations):
        refined = [medoid_of(c) for c in clusters]
        if refined == medoids:
            break
        medoids = refined
        clusters = assign(medoids)
    return clusters, [medoid_of(c) for c in clusters]


#: Number of option-list partitions a HierarchicalChoiceRouter keeps (LRU).
_PARTITION_CACHE_SIZE = 8


class HierarchicalChoiceRouter:
    """
    Two-level choice decomposition for large option sets.

    Motivation, as reported by the Laya authors (figures not verified by the
    author of this comment): Laya reports 0.425 on Banking77 with 77 labels,
    against Jev's 0.870 on 72 labels, and attributes the gap to a flat head
    token budget that leaves only a few tokens per candidate at 77 options. It
    suggests a coarse-to-fine hierarchy for large option sets.

    This router implements that coarse-to-fine approach. Options are partitioned
    into ``num_groups`` clusters by text similarity. Level 1 scores one medoid
    label per group. Level 2 scores the members of every group whose medoid is
    close to the best group: a group is expanded when its medoid score is at
    least ``best - EXPAND_MARGIN * |best|``, and at most ``MAX_EXPANDED_GROUPS``
    groups are expanded, the highest-scoring first. A single medoid label can
    misrepresent its group, so the margin lets a target survive a medoid that
    ranks below the best group.

    Bounds on work, with ``|group| <= ceil(K / G)``:

    * level 1 scores exactly ``G`` labels;
    * level 2 scores at most ``MAX_EXPANDED_GROUPS * ceil(K / G)`` options;
    * so ``scored_options <= G + MAX_EXPANDED_GROUPS * ceil(K / G)``
      (28 for K=77, G=8, MAX_EXPANDED_GROUPS=2), against K for the flat path.

    Partitions are cached per instance for the most recent option lists
    (``_PARTITION_CACHE_SIZE`` entries), keyed on ``num_groups`` and the string
    form of each option, since the partition depends on nothing else.

    Reference (not verified by the author of this comment):
    https://laya.convaiinnovations.com/ (Banking77 discussion).
    """

    #: Relative margin for expanding a group: groups scoring at least
    #: ``best - EXPAND_MARGIN * |best|`` at level 1 are candidates for level 2.
    EXPAND_MARGIN: float = 0.1
    #: Upper bound on the number of groups expanded at level 2.
    MAX_EXPANDED_GROUPS: int = 2

    def __init__(
        self,
        num_groups: int = 8,
        flat_threshold: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self.num_groups = max(2, int(num_groups))
        self.flat_threshold = int(flat_threshold)
        self.seed = seed
        self._partition_cache: "OrderedDict[Tuple[int, Tuple[str, ...]], Tuple[List[List[int]], List[int]]]" = (
            OrderedDict()
        )

    def _partition(self, options: Sequence[str]) -> Tuple[List[List[int]], List[int]]:
        """
        Return ``(clusters, medoids)`` for ``options``, computing them at most once per option list.

        The result is a copy, so callers may mutate it without corrupting the cache.
        """
        key = (self.num_groups, tuple(str(o) for o in options))
        cache = self._partition_cache
        if key in cache:
            cache.move_to_end(key)
            clusters, medoids = cache[key]
        else:
            clusters, medoids = _balanced_clusters(options, self.num_groups)
            cache[key] = (clusters, medoids)
            while len(cache) > _PARTITION_CACHE_SIZE:
                cache.popitem(last=False)
        return [list(c) for c in clusters], list(medoids)

    def build_clusters(self, options: Sequence[str]) -> List[List[int]]:
        """
        Deterministically partition option indices into ``num_groups`` clusters.

        Partitioning is by text similarity (character-trigram Jaccard), not by
        position, so related options share a group wherever they sit in the
        list. Clusters are balanced to at most ``ceil(K / G)`` members, which
        bounds the number of options scored in the second pass, and every
        cluster is non-empty. The procedure is dependency-free and has no
        randomness, so the same option list always yields the same partition,
        which keeps the routing auditable.
        """
        clusters, _ = self._partition(options)
        return clusters

    def should_decompose(self, num_options: int) -> bool:
        """Hierarchical routing only pays off beyond the flat-budget threshold."""
        return int(num_options) > self.flat_threshold

    def route(
        self,
        options: Sequence[str],
        score_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Select an option via coarse-to-fine scoring.

        Args:
            options: candidate option labels.
            score_fn: optional callable. Given a list of option labels it must
                return a list of scores (higher is better). When omitted the
                router falls back to a deterministic lexical heuristic, keeping
                the class unit-testable without a model.

        Returns:
            Dict with the winning label/index, group structure, how many options
            were actually scored, and whether the two-level path was taken.
        """
        k = len(options)
        if k == 0:
            return {
                "chosen_index": -1,
                "chosen_option": None,
                "decomposed": False,
                "scored_options": 0,
                "total_options": 0,
                "clusters": [],
            }

        scorer = score_fn if score_fn is not None else self._lexical_score

        # Small option sets take the direct flat path: decomposition would only
        # add overhead and reduce accuracy.
        if not self.should_decompose(k):
            scores = list(scorer(list(options)))
            best = int(max(range(k), key=lambda i: scores[i]))
            return {
                "chosen_index": best,
                "chosen_option": options[best],
                "decomposed": False,
                "scored_options": k,
                "total_options": k,
                "clusters": [],
                "scores": scores,
            }

        # Level 1: score the groups. Each group is represented by its medoid
        # option, the member most similar to the rest of its group, so the
        # label reflects the whole group rather than its first few members.
        clusters, medoids = self._partition(options)
        group_labels = [options[m] for m in medoids]
        group_scores = list(scorer(group_labels))
        best_group = int(max(range(len(clusters)), key=lambda i: group_scores[i]))

        # Expand the best group, plus any group whose medoid score is within
        # EXPAND_MARGIN (relative to the best score) of it, up to
        # MAX_EXPANDED_GROUPS groups in total, highest score first. The best
        # group always qualifies, so at least one group is expanded.
        best_score = group_scores[best_group]
        threshold = best_score - self.EXPAND_MARGIN * abs(best_score)
        cap = max(1, int(self.MAX_EXPANDED_GROUPS))
        ranked = sorted(range(len(clusters)), key=lambda i: (-group_scores[i], i))
        expanded = [i for i in ranked if group_scores[i] >= threshold][:cap]

        # Level 2: score the union of the expanded groups' members, once each.
        scored_indices = sorted({i for g in expanded for i in clusters[g]})
        member_labels = [options[i] for i in scored_indices]
        member_scores = list(scorer(member_labels))
        best_local = int(max(range(len(scored_indices)), key=lambda i: member_scores[i]))
        chosen = scored_indices[best_local]

        return {
            "chosen_index": int(chosen),
            "chosen_option": options[chosen],
            "decomposed": True,
            "scored_options": len(clusters) + len(scored_indices),
            "total_options": k,
            "clusters": clusters,
            "winning_cluster": best_group,
            "expanded_clusters": expanded,
            "scored_indices": scored_indices,
            "group_scores": group_scores,
            "member_scores": member_scores,
        }

    @staticmethod
    def _lexical_score(options: Sequence[str]) -> List[float]:
        """
        Deterministic fallback scorer (no model required).

        Ranks by token overlap between each option and the concatenation of all
        options, which behaves like a crude centroid-relevance signal and makes
        the routing path testable in isolation.
        """
        if not options:
            return []
        corpus_tokens = set(" ".join(options).lower().split())
        scores: List[float] = []
        for opt in options:
            tokens = set(str(opt).lower().split())
            if not tokens:
                scores.append(0.0)
            else:
                scores.append(len(tokens & corpus_tokens) / len(tokens))
        return scores



# ===========================================================================
# Pillar 11: OOD Regression Guard
# ===========================================================================


class OODRegressionGuard:
    """
    Held-out out-of-distribution regression gate.

    Open-Jev's README states plainly: "Wiki OOD optimal-action accuracy regressed
    for 9B and 27B on 18 examples each. Do not treat that pilot as broad task
    competence." A regression the authors themselves report is exactly the
    failure mode this guard exists to prevent.

    The guard keeps a reference scoreboard and refuses any candidate change whose
    held-out optimal-action accuracy drops by more than ``tolerance``. Because
    the gate scores against held-out data rather than the tuning split, it cannot
    be satisfied by overfitting the visible examples.
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        min_examples: int = 4,
        seed: int = 0,
    ) -> None:
        self.tolerance = float(tolerance)
        self.min_examples = int(min_examples)
        self.seed = int(seed)
        self.baseline: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def optimal_action_accuracy(
        predictions: Sequence[Any],
        labels: Sequence[Any],
    ) -> float:
        """Fraction of examples where the top-1 action matches the label."""
        if not labels:
            return 0.0
        correct = sum(1 for p, y in zip(predictions, labels) if p == y)
        return correct / len(labels)

    def establish_baseline(
        self,
        predictions: Sequence[Any],
        labels: Sequence[Any],
        name: str = "baseline",
    ) -> Dict[str, Any]:
        """Record the reference scoreboard future candidates must not degrade."""
        accuracy = self.optimal_action_accuracy(predictions, labels)
        self.baseline = {
            "name": name,
            "accuracy": accuracy,
            "num_examples": len(labels),
        }
        return dict(self.baseline)

    def evaluate(
        self,
        predictions: Sequence[Any],
        labels: Sequence[Any],
        name: str = "candidate",
    ) -> Dict[str, Any]:
        """
        Score a candidate and decide whether it is admissible.

        Returns:
            Dict with candidate accuracy, the delta against baseline, and an
            ``accepted`` flag that is ``False`` on regression beyond tolerance.
        """
        if self.baseline is None:
            raise RuntimeError("establish_baseline() must be called before evaluate()")

        accuracy = self.optimal_action_accuracy(predictions, labels)
        delta = accuracy - self.baseline["accuracy"]
        enough = len(labels) >= self.min_examples
        accepted = bool(delta >= -self.tolerance) and enough

        record = {
            "name": name,
            "accuracy": accuracy,
            "baseline_accuracy": self.baseline["accuracy"],
            "delta": delta,
            "num_examples": len(labels),
            "sufficient_examples": enough,
            "accepted": accepted,
            "reason": (
                "insufficient held-out examples"
                if not enough
                else ("regression beyond tolerance" if not accepted else "ok")
            ),
        }
        self.history.append(record)
        return record

    def has_regressed(self) -> bool:
        """True if the most recent evaluation was rejected."""
        return bool(self.history and not self.history[-1]["accepted"])



# ===========================================================================
# Pillar 12: Evidence Verification Lattice
# ===========================================================================


@dataclass
class EvidenceRecord:
    """
    A single auditable observation backing (or contradicting) a claim.

    Mirrors the loftwah JEV framework's requirement for "enough evidence to
    distinguish an improvement from a prettier screenshot": every record carries
    its seed, scenario identity, role and verdict, so a reviewer can reproduce
    the exact condition that produced it.
    """

    scenario: str
    role: str
    passed: bool
    detail: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "role": self.role,
            "passed": self.passed,
            "detail": self.detail,
            "seed": self.seed,
        }


class EvidenceVerificationLattice:
    """
    Observer / critic / verifier lattice with recorded controls.

    Roles:
        * ``observer``  -- records what the system actually did.
        * ``critic``    -- attacks the result (invariance checks).
        * ``verifier``  -- adjudicates, requiring an explicit control baseline
          so a "better" score cannot be claimed without something to compare to.

    The loftwah JEV framework is explicit that "The strongest bot is not a novice
    model, and equal win rates do not automatically mean good balance", and that
    results must use "reproducible seeds, multiple policies, held-out scenarios
    and existing bots as controls". This class makes that discipline executable.
    """

    ROLES = ("observer", "critic", "verifier")

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.records: List[EvidenceRecord] = []

    def observe(
        self,
        scenario: str,
        decision_fn: Any,
        seed: Optional[int] = None,
    ) -> EvidenceRecord:
        """Run ``decision_fn(seed)`` under a recorded seed and observe its output."""
        s = int(seed if seed is not None else self.seed)
        result = decision_fn(s)
        record = EvidenceRecord(
            scenario=scenario,
            role="observer",
            passed=True,
            detail=f"observed result type={type(result).__name__}",
            seed=s,
        )
        self.records.append(record)
        return record

    def critique_permutation_invariance(
        self,
        predict_fn: Any,
        options: Sequence[str],
        label: str,
        seed: Optional[int] = None,
    ) -> EvidenceRecord:
        """
        Verify the decision is invariant to candidate presentation order.

        ``predict_fn(labels)`` must return the chosen label. This is the
        permutation-invariance property Rhen trains for (``L_perm``); here it is
        checked as *evidence* rather than assumed.
        """
        s = int(seed if seed is not None else self.seed)
        forward = predict_fn(list(options))
        reversed_choice = predict_fn(list(reversed(options)))
        passed = forward == reversed_choice == label
        record = EvidenceRecord(
            scenario="permutation_invariance",
            role="critic",
            passed=passed,
            detail=(
                f"forward={forward!r} reversed={reversed_choice!r} expected={label!r}"
            ),
            seed=s,
        )
        self.records.append(record)
        return record

    def verify_with_control(
        self,
        candidate_accuracy: float,
        control_accuracy: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> EvidenceRecord:
        """
        Adjudicate a claimed improvement against an explicit control.

        A candidate with no control cannot be verified -- mirroring the JEV
        framework's requirement for "existing bots as controls" before any
        quality claim is credible.
        """
        s = int(seed if seed is not None else self.seed)
        if control_accuracy is None:
            passed = False
            detail = "REJECTED: no control baseline supplied"
        else:
            passed = candidate_accuracy > control_accuracy
            detail = (
                f"candidate={candidate_accuracy:.4f} control={control_accuracy:.4f} "
                f"delta={candidate_accuracy - control_accuracy:+.4f}"
            )
        record = EvidenceRecord(
            scenario="control_comparison",
            role="verifier",
            passed=passed,
            detail=detail,
            seed=s,
        )
        self.records.append(record)
        return record

    def report(self) -> Dict[str, Any]:
        """Aggregate the lattice into a reproducibility report."""
        by_role: Dict[str, int] = {role: 0 for role in self.ROLES}
        passed_by_role: Dict[str, int] = {role: 0 for role in self.ROLES}
        for r in self.records:
            by_role[r.role] = by_role.get(r.role, 0) + 1
            if r.passed:
                passed_by_role[r.role] = passed_by_role.get(r.role, 0) + 1
        return {
            "total_records": len(self.records),
            "records_by_role": by_role,
            "passed_by_role": passed_by_role,
            "all_passed": (
                all(r.passed for r in self.records) if self.records else False
            ),
            "seeds": sorted({r.seed for r in self.records}),
            "records": [r.to_dict() for r in self.records],
        }



# ===========================================================================
# Pillar 13: Multilingual Decision Routing
# ===========================================================================


class MultilingualDecisionRouter:
    """
    Script/language detection and decision routing, without an LLM call.

    This is a heuristic over about 13 language tags (de, fr, pt, es, it, nl, en,
    hi, ja, ko, ru, ar, zh) plus script-family tags for other scripts. Unicode
    script analysis and a compact stopword signature select the language and
    the matching calibration profile. It has no accuracy evaluation in this
    repo and is not equivalent to Laya's 100+ language coverage claim.

    Why it matters for decisions: conformal sets and temperatures fitted on
    English do not transfer. Each detected language selects its own calibration
    profile, so a Hindi ticket is not scored with English thresholds.
    """

    # Compact script ranges; deliberately coarse (script family, not language).
    SCRIPT_RANGES: List[Tuple[int, int, str]] = [
        (0x0400, 0x04FF, "cyrillic"),
        (0x0590, 0x05FF, "hebrew"),
        (0x0600, 0x06FF, "arabic"),
        (0x0700, 0x074F, "syriac"),
        (0x0900, 0x097F, "devanagari"),
        (0x0980, 0x09FF, "bengali"),
        (0x0A00, 0x0A7F, "gurmukhi"),
        (0x0B00, 0x0B7F, "oriya"),
        (0x0B80, 0x0BFF, "tamil"),
        (0x0C00, 0x0C7F, "telugu"),
        (0x0C80, 0x0CFF, "kannada"),
        (0x0D00, 0x0D7F, "malayalam"),
        (0x0E00, 0x0E7F, "thai"),
        (0x0F00, 0x0FFF, "tibetan"),
        (0x10A0, 0x10FF, "georgian"),
        (0x1100, 0x11FF, "hangul"),
        (0x3040, 0x309F, "hiragana"),
        (0x30A0, 0x30FF, "katakana"),
        (0x4E00, 0x9FFF, "han"),
        (0xAC00, 0xD7AF, "hangul"),
    ]

    def __init__(self) -> None:
        self.detection_history: List[Dict[str, Any]] = []

    def detect_script(self, text: str) -> Dict[str, Any]:
        """
        Classify the dominant writing system of ``text``.

        Returns:
            Dict with ``script``, ``confidence`` (share of letters attributable
            to the winning script) and ``char_counts``.
        """
        if not text:
            return {"script": "unknown", "confidence": 0.0, "char_counts": {}}

        counts: Counter = Counter()
        letters = 0
        for ch in text:
            if not ch.isalpha():
                continue
            cp = ord(ch)
            if cp < 0x0250:  # Latin / Greek / Cyrillic block
                if cp < 0x0370:
                    counts["latin"] += 1
                elif cp < 0x0400:
                    counts["greek"] += 1
                else:
                    counts["cyrillic"] += 1
                letters += 1
                continue
            for lo, hi, name in self.SCRIPT_RANGES:
                if lo <= cp <= hi:
                    counts[name] += 1
                    letters += 1
                    break

        if letters == 0:
            return {"script": "unknown", "confidence": 0.0, "char_counts": {}}

        script, count = counts.most_common(1)[0]
        return {
            "script": script,
            "confidence": count / letters,
            "char_counts": dict(counts),
        }

    def detect_language(self, text: str) -> str:
        """
        Map a script detection onto a coarse language tag.

        Latin-script languages are disambiguated by a small stopword signature;
        otherwise the script name is returned as the tag.
        """
        script = self.detect_script(text)["script"]

        if script == "latin":
            tokens = {t.strip(".,!?;:").lower() for t in text.split()}
            # Order matters: the more distinctive signatures are tested first so
            # that short, frequently-overlapping words ("is" in both English and
            # Dutch) can only decide the result once a stronger signal exists.
            if tokens & {"der", "die", "das", "und", "nicht", "einen"}:
                return "de"
            if tokens & {"le", "les", "une", "est", "des", "pour", "avec"}:
                return "fr"
            # Portuguese is tested before Spanish because short function words
            # ("para", "com", "as") overlap heavily between the two languages.
            if tokens & {"uma", "nao", "dos", "como", "pelo", "manutencao"}:
                return "pt"
            if tokens & {"los", "las", "que", "por", "para", "con", "esta", "servidor"}:
                return "es"
            if tokens & {"che", "per", "non", "sono", "della", "gli"}:
                return "it"
            if tokens & {"het", "een", "niet", "van", "zijn", "voor"}:
                return "nl"
            return "en"
        if script == "han":
            return "zh"
        if script in ("hiragana", "katakana"):
            return "ja"
        if script == "hangul":
            return "ko"
        if script == "cyrillic":
            return "ru"
        if script == "arabic":
            return "ar"
        if script == "devanagari":
            return "hi"
        return script

    def route(
        self,
        text: str,
        calibration_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Route a decision by detected language and return its calibration profile.

        Args:
            text: the state/question text.
            calibration_profiles: optional per-language overrides, e.g.
                ``{"hi": {"temperature": 1.15, "alpha": 0.05}}``.

        Returns:
            Dict with the detected language/script, the recommended temperature
            and alpha for that language, and whether a custom profile applied.
        """
        info = self.detect_script(text)
        language = self.detect_language(text)
        confidence = float(info["confidence"])

        # A low-confidence detection should not silently inherit a confident
        # language's calibration; fall back to the neutral profile instead.
        profile_key = language if confidence >= 0.5 else "default"

        profile: Dict[str, float] = {"temperature": 1.0, "alpha": 0.05}
        profiles = calibration_profiles or {}
        custom = profile_key in profiles
        if custom:
            profile.update(profiles[profile_key])

        record = {
            "language": language,
            "script": info["script"],
            "detection_confidence": confidence,
            "profile": profile,
            "custom_profile_applied": custom,
        }
        self.detection_history.append(record)
        return record

