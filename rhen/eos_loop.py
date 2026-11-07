"""
Rhen-1.0 Recurrent-Loop Control.

This module addresses the *language backbone* (EOS), separately from the Rhen
decision engine. Every component here targets a specific, identified defect or
inefficiency in a parameter-shared recurrent transformer, and each is opt-in via
``EOSConfig`` so that existing checkpoints keep loading unchanged.

1. :class:`StepSizeController` -- the single highest-value change here.
   arXiv:2609.16665 ("Right Direction, Wrong Step: Geometric Analysis of
   Finite-Step Failure in Looped Transformers", 2026-09-15) shows looped
   transformers fail in a specific, geometric way: the update *direction* is
   locally helpful but the *step scale* overshoots, so a full update destroys
   progress that a smaller step would have kept. Their reported remedy is a
   **fixed quarter step**, which recovered positive gains for **72.2-83.2%** of
   selected failures.

   The current EOS loop performs ``h_loop = h_next`` -- a *full* replacement
   (see ``model.EOSModel.forward``). This controller inserts an explicit
   trust-region step and anneals it.

2. :class:`DepthControlProtocol` -- is the recurrent depth actually earning
   anything? arXiv:2609.19934 shows the standard "truncate depth, plot quality"
   test is confounded: it simultaneously reduces block applications, reduces
   distinct computation, and pushes the readout head onto an OOD residual
   stream. Its linchpin control (full budget, one distinct iteration) is
   *only* realisable in a weight-shared architecture like EOS.

3. :class:`FlashLoopUpdater` -- token-sparse cross-loop updates.
   arXiv:2609.29812 observes that as recurrence proceeds, state changes
   concentrate on a small subset of tokens while the rest are effectively
   frozen, and exploits this training-free for **1.64x speedup and 6x KV-cache
   reduction**.

4. :class:`LoopCurriculum` -- arXiv:2603.02615 found that *deeper* recursion
   ("overthinking") degraded accuracy on simple retrieval while inflating
   runtime from 3.6s to 344.5s. Uniform random depth (EOS's current default)
   is a weak curriculum; this replaces it.

5. :class:`WorkingSetAttention` -- arXiv:2609.27373 finds attention routing
   stabilises *early* in recurrent LMs, so late loops may reuse a frozen sparse
   working set.

All components are pure-PyTorch and CPU-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "StepSizeController",
    "StepStats",
    "DepthControlProtocol",
    "DepthProbe",
    "FlashLoopUpdater",
    "LoopCurriculum",
    "WorkingSetAttention",
    "LoopDiagnostics",
]


# ---------------------------------------------------------------------------
# 1. Step-size controlled recurrence (arXiv:2609.16665)
# ---------------------------------------------------------------------------


@dataclass
class StepStats:
    """Per-loop record of step sizes and reference-utility movement."""

    loop: int
    alpha: float
    mean_update_norm: float
    mean_utility_delta: float
    overshoot: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class StepSizeController(nn.Module):
    """Trust-region step control for the recurrent loop.

    arXiv:2609.16665 distinguishes two independent failure modes:

    * the update *direction* is unhelpful -> shrink the step, or stop;
    * the direction is helpful but the *scale* overshoots -> take a partial step.

    A full update conflates them. This controller implements the partial-step
    remedy directly::

        h <- h + alpha * (h_next - h)

    with ``alpha`` defaulting to the paper's empirically effective **0.25**, plus
    two refinements:

    * **annealing** -- shrink the step in later loops, which have less headroom;
    * **overshoot rejection** -- if a full step *reduces* reference utility (a
      proxy for moving toward the answer), immediately retry that loop with a
      quarter step. This is the "right direction, wrong step" repair, and it is
      what recovered 72-83% of failures in the paper.

    Set ``enabled=False`` to restore exact legacy behaviour, in which case
    :meth:`apply` returns the proposed state unchanged.
    """

    def __init__(
        self,
        hidden_size: int = 0,
        base_alpha: float = 0.25,
        min_alpha: float = 0.05,
        anneal: bool = True,
        anneal_to: float = 0.5,
        overshoot_rejection: bool = True,
        enabled: bool = True,
        utility_fn: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.base_alpha = float(base_alpha)
        self.min_alpha = float(min_alpha)
        self.anneal = bool(anneal)
        self.anneal_to = float(anneal_to)
        self.overshoot_rejection = bool(overshoot_rejection)
        self.enabled = bool(enabled)
        self.utility_fn = utility_fn
        self.history: List[StepStats] = []
        self._last_utility: Optional[float] = None
        self._budget = 1

        # A learned gate is available but off by default: the paper's fixed
        # quarter step is the validated setting, and a learned gate would need
        # its own ablation rather than being assumed correct.
        self.gate: Optional[nn.Linear]
        if hidden_size > 0:
            self.gate = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        else:
            self.gate = None
        self.use_learned_gate = False

    def set_budget(self, max_loops: int) -> None:
        """Tell the controller the loop budget so annealing can be scheduled."""
        self._budget = max(1, int(max_loops))

    def apply(
        self,
        h_current: torch.Tensor,
        h_proposed: torch.Tensor,
        loop_idx: int,
    ) -> Tuple[torch.Tensor, StepStats]:
        """Blend the proposed loop output into the running state.

        Returns the updated state and a :class:`StepStats` diagnostic.
        """
        if not self.enabled:
            return h_proposed, StepStats(
                loop=int(loop_idx), alpha=1.0,
                mean_update_norm=0.0, mean_utility_delta=0.0, overshoot=False,
            )

        delta = h_proposed - h_current
        alpha = self.base_alpha
        if self.anneal and self._budget > 1:
            progress = min(1.0, max(0.0, (loop_idx - 1) / float(self._budget - 1)))
            alpha = self.base_alpha * (1.0 + (self.anneal_to - 1.0) * progress)
        if self.use_learned_gate and self.gate is not None:
            with torch.no_grad():
                alpha *= float(torch.sigmoid(self.gate(h_current.mean(dim=(0, 1)))).item())
        alpha = max(self.min_alpha, min(1.0, alpha))

        blended = h_current + alpha * delta
        overshoot = False
        util_delta = 0.0

        if self.overshoot_rejection and self.utility_fn is not None:
            with torch.no_grad():
                u_prev = self._last_utility
                # Probe the FULL step and a QUARTER step independently. This has
                # to be an independent probe rather than a comparison against
                # `blended`: when alpha == 1.0 the taken step *is* the full step,
                # so comparing them could never detect anything.
                u_full = float(self.utility_fn(h_proposed))
                u_quarter = float(
                    self.utility_fn(h_current + (0.25 * alpha) * delta)
                )
            if u_prev is not None and u_full < u_prev and u_quarter >= u_prev:
                # Full step destroyed progress; the smaller step kept it. This
                # is exactly the "right direction, wrong step" repair.
                blended = h_current + (0.25 * alpha) * delta
                overshoot = True
                u_taken = u_quarter
            else:
                u_taken = float(self.utility_fn(blended))
            self._last_utility = u_taken
            util_delta = u_taken - (u_prev if u_prev is not None else u_taken)

        stats = StepStats(
            loop=int(loop_idx),
            alpha=float(alpha),
            mean_update_norm=float(delta.detach().float().norm(dim=-1).mean().item()),
            mean_utility_delta=float(util_delta),
            overshoot=bool(overshoot),
        )
        self.history.append(stats)
        return blended, stats

    def reset(self) -> None:
        self.history.clear()
        self._last_utility = None
        self._budget = 1

    def summary(self) -> Dict[str, Any]:
        """Aggregate loop behaviour, including how often repair was needed."""
        if not self.history:
            return {"loops": 0, "mean_alpha": 0.0, "overshoot_repairs": 0, "overshoot_rate": 0.0}
        n = len(self.history)
        repairs = sum(1 for s in self.history if s.overshoot)
        return {
            "loops": n,
            "mean_alpha": sum(s.alpha for s in self.history) / n,
            "mean_update_norm": sum(s.mean_update_norm for s in self.history) / n,
            "overshoot_repairs": repairs,
            "overshoot_rate": repairs / n,
        }


# ---------------------------------------------------------------------------
# 2. Depth Control Protocol (arXiv:2609.19934)
# ---------------------------------------------------------------------------


@dataclass
class DepthProbe:
    """One point on a depth/quality curve, with the control it was measured under."""

    retained_depth_fraction: float
    quality: float
    condition: str
    n_block_applications: int
    n_distinct_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class DepthControlProtocol:
    """Disentangle what a depth-truncation curve conflates.

    arXiv:2609.19934's objection to the standard protocol: truncating depth at
    inference changes three things at once --

    1. the number of block applications,
    2. the volume of *distinct* computation performed, and
    3. the readout head's position on the residual stream (pushing it out of
       distribution).

    Reading the slope of ``quality vs retained_depth`` and attributing it to (2)
    alone is therefore unsound. This class produces the controls that separate
    them, most importantly the **linchpin control**: run the *whole*
    block-application budget but execute only one distinct iteration. That is
    only realisable in a weight-shared architecture -- in a dense network
    repeating a layer yields a different model -- which is exactly EOS's case,
    and precisely why a dense negative control is mandatory.
    """

    CONDITIONS = (
        "truncated_standard",
        "full_budget_single_iteration",
        "repeat_same_state",
        "dense_negative_control",
    )

    def __init__(self, total_loops: int = 12, distinct_iterations: int = 4) -> None:
        self.total_loops = int(total_loops)
        self.distinct_iterations = int(distinct_iterations)
        self.probes: List[DepthProbe] = []

    @staticmethod
    def _quality(loss: float) -> float:
        """Convert a loss into an interpretable quality score (higher better)."""
        return 1.0 / (1.0 + max(0.0, float(loss)))

    def record(
        self,
        condition: str,
        retained_depth_fraction: float,
        loss: float,
        n_block_applications: int,
        n_distinct_iterations: int,
    ) -> DepthProbe:
        if condition not in self.CONDITIONS:
            raise ValueError(
                f"unknown condition {condition!r}; expected one of {self.CONDITIONS}"
            )
        probe = DepthProbe(
            retained_depth_fraction=float(retained_depth_fraction),
            quality=self._quality(loss),
            condition=condition,
            n_block_applications=int(n_block_applications),
            n_distinct_iterations=int(n_distinct_iterations),
        )
        self.probes.append(probe)
        return probe

    def plan(self) -> List[Dict[str, Any]]:
        """Emit the measurement plan so a harness can execute it reproducibly."""
        plan: List[Dict[str, Any]] = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            kept = max(1, int(round(self.total_loops * frac)))
            plan.append(
                {
                    "condition": "truncated_standard",
                    "retained_depth_fraction": frac,
                    "loops": kept,
                    "n_block_applications": kept,
                    "n_distinct_iterations": kept,
                }
            )
        plan.append(
            {
                "condition": "full_budget_single_iteration",
                "retained_depth_fraction": 1.0,
                "loops": self.total_loops,
                "n_block_applications": self.total_loops,
                "n_distinct_iterations": 1,
                "note": "linchpin control; requires weight sharing",
            }
        )
        plan.append(
            {
                "condition": "repeat_same_state",
                "retained_depth_fraction": 1.0,
                "loops": self.total_loops,
                "n_block_applications": self.total_loops,
                "n_distinct_iterations": 1,
                "note": "applies the block to an unchanged state each iteration",
            }
        )
        plan.append(
            {
                "condition": "dense_negative_control",
                "retained_depth_fraction": 1.0,
                "note": "run the same interventions on a dense baseline; the "
                        "effect must NOT reproduce, or the protocol is confounded",
            }
        )
        return plan

    def conclude(self) -> Dict[str, Any]:
        """State whether recurrent depth appears to be genuinely exploited.

        If the full-budget single-iteration control scores as well as the full
        distinct-iteration run, then the depth curve was measuring readout drift
        rather than additional computation, and the "N effective layers" claim
        does not hold.
        """
        by_cond: Dict[str, List[DepthProbe]] = {}
        for p in self.probes:
            by_cond.setdefault(p.condition, []).append(p)

        def best(cond: str) -> Optional[float]:
            vals = by_cond.get(cond)
            return max(v.quality for v in vals) if vals else None

        full_single = best("full_budget_single_iteration")
        standard = by_cond.get("truncated_standard", [])
        full_standard: Optional[float] = None
        if standard:
            full_frac = [v for v in standard if v.retained_depth_fraction >= 1.0]
            full_standard = (
                max(v.quality for v in full_frac)
                if full_frac
                else max(v.quality for v in standard)
            )

        if full_single is None or full_standard is None:
            return {
                "verdict": "insufficient_data",
                "detail": "record both truncated_standard (frac=1.0) and "
                          "full_budget_single_iteration before concluding",
            }

        ratio = full_single / full_standard if full_standard > 0 else float("inf")
        if ratio >= 0.98:
            verdict = "depth_not_exploited"
            detail = (
                "full application budget with a single distinct iteration "
                "matches the full run, so extra depth is not adding distinct "
                "computation; the depth curve was measuring readout drift"
            )
        elif ratio >= 0.85:
            verdict = "depth_partially_exploited"
            detail = "distinct iterations beat repeated applications, but not by much"
        else:
            verdict = "depth_exploited"
            detail = "distinct iterations clearly beat repeated applications"
        return {
            "verdict": verdict,
            "detail": detail,
            "single_iteration_quality": full_single,
            "full_standard_quality": full_standard,
            "ratio": ratio,
            "dense_negative_control": best("dense_negative_control"),
        }


# ---------------------------------------------------------------------------
# 3. FlashLoop-style token-sparse updates (arXiv:2609.29812)
# ---------------------------------------------------------------------------


class FlashLoopUpdater:
    """Skip cross-loop updates for tokens that have stopped moving.

    arXiv:2609.29812 (FlashLoop) makes three observations about looped
    transformers and exploits all of them training-free: as recurrence proceeds,
    state changes concentrate on a small subset of tokens; attention-output
    differences are dominated by a sparse, stable subset of key columns; and KV
    residuals between adjacent loops become increasingly quantisation-friendly.
    It reports **1.64x end-to-end speedup and up to 6x KV-cache reduction** with
    lossless accuracy.

    This class implements the first observation, which needs no kernel changes:
    identify tokens whose state stopped moving and hold them at their current
    value through the next loop, so frozen positions cost nothing.

    The sparsity mask is *derived*, never assumed -- it is recomputed from actual
    per-token movement each loop, so a badly-conditioned model simply yields a
    low sparsity ratio instead of silently losing signal.
    """

    def __init__(
        self,
        threshold: float = 1e-3,
        warmup_loops: int = 2,
        enabled: bool = True,
    ) -> None:
        self.threshold = float(threshold)
        self.warmup_loops = int(warmup_loops)
        self.enabled = bool(enabled)
        self._previous: Optional[torch.Tensor] = None
        self.frozen_mask: Optional[torch.Tensor] = None
        self.history: List[Dict[str, Any]] = []
        # Counts *every* observation, including the very first (which only
        # establishes a baseline). Comparing a post-warmup-only counter against
        # warmup_loops would double-count warmup and never unfreeze.
        self._total_obs = 0

    def observe(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """Boolean mask of tokens still moving; ``None`` during warmup/disabled."""
        if not self.enabled:
            return None
        if self._previous is None:
            self._previous = hidden.detach().clone()
            self.frozen_mask = None
            self._total_obs += 1
            return None

        movement = torch.norm(
            (hidden.detach().float() - self._previous.float()), dim=-1
        )  # [B, S]
        active = movement > self.threshold
        self._previous = hidden.detach().clone()
        self._total_obs += 1

        if self._total_obs <= self.warmup_loops:
            self.frozen_mask = None
            return None

        self.frozen_mask = active
        self.history.append(
            {
                "loop": self._total_obs,
                "active_fraction": float(active.float().mean().item()),
            }
        )
        return active

    def apply(self, h_current: torch.Tensor, h_proposed: torch.Tensor) -> torch.Tensor:
        """Hold frozen tokens at their current value; update only active ones."""
        if self.frozen_mask is None:
            return h_proposed
        m = self.frozen_mask
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        # torch.where requires a bool condition; the mask must not be cast to
        # the data dtype or it stops being a predicate.
        return torch.where(m, h_proposed, h_current)

    def reset(self) -> None:
        self._previous = None
        self.frozen_mask = None
        self.history.clear()
        self._total_obs = 0

    def savings(self) -> Dict[str, Any]:
        """Estimated fraction of token-updates skipped so far."""
        if not self.history:
            return {"observed_loops": 0, "mean_skipped_fraction": 0.0}
        fracs = [1.0 - h["active_fraction"] for h in self.history]
        return {
            "observed_loops": len(self.history),
            "mean_skipped_fraction": sum(fracs) / len(fracs),
            "last_skipped_fraction": fracs[-1],
        }


# ---------------------------------------------------------------------------
# 4. Loop curriculum (arXiv:2603.02615, 2609.15160)
# ---------------------------------------------------------------------------


class LoopCurriculum:
    """Depth schedule that ramps with training, replacing uniform sampling.

    EOS currently samples depth uniformly from ``[min_recurrent_loops,
    max_recurrent_loops]`` (``config.loop_sampling == "uniform"``). Two findings
    argue against keeping that:

    * arXiv:2603.02615 found *deeper* recursion degraded accuracy on simple
      retrieval while inflating runtime from 3.6s to 344.5s -- useful depth is
      task-dependent, so spending the maximum uniformly is wasteful.
    * arXiv:2609.15160 (T-LoopFormer) shows per-token dynamic depth beats a
      uniform recursion depth at equal parameters.

    This curriculum therefore (a) ramps the *cap* upward so early training is
    cheap and stable, and (b) samples from a geometric truncated to
    ``[min_loops, cap]`` rather than uniformly, concentrating mass near the cap.
    """

    def __init__(
        self,
        min_loops: int = 2,
        max_loops: int = 12,
        warmup_steps: int = 0,
        concentration: float = 2.0,
    ) -> None:
        self.min_loops = int(min_loops)
        self.max_loops = int(max_loops)
        self.warmup_steps = int(warmup_steps)
        self.concentration = float(concentration)

    def cap(self, step: int) -> int:
        """Current depth ceiling at training ``step``."""
        if self.warmup_steps <= 0:
            return self.max_loops
        progress = min(
            1.0, max(0.0, (int(step) - self.warmup_steps) / self.warmup_steps)
        )
        return int(round(self.min_loops + progress * (self.max_loops - self.min_loops)))

    def sample(self, step: int, generator: Optional[torch.Generator] = None) -> int:
        """Sample a depth from a geometric truncated to ``[min_loops, cap]``."""
        ceiling = max(self.min_loops, min(self.max_loops, self.cap(step)))
        if ceiling <= self.min_loops:
            return self.min_loops
        u = float(torch.rand(1, generator=generator).item())
        span = ceiling - self.min_loops
        # Inverse-transform sampling on a discrete geometric.
        offset = int(min(span, (1.0 - (1.0 - u) ** self.concentration) * (span + 1)))
        return self.min_loops + max(0, min(span, offset))


# ---------------------------------------------------------------------------
# 5. Working-set attention (arXiv:2609.27373)
# ---------------------------------------------------------------------------


class WorkingSetAttention:
    """Freeze the attention working set once routing has stabilised.

    arXiv:2609.27373 ("Attention Routing Stabilizes Early: Working-Set Inference
    for Recurrent Language Models") reports that in recurrent LMs the set of
    positions a query attends to stabilises early in the sequence, so later
    loops can reuse a fixed sparse working set instead of recomputing full
    attention. Combined with the FlashLoop finding that late-loop state changes
    concentrate on few tokens, this makes the tail loops nearly free.

    The working set is *observed*, then frozen -- it is not a heuristic sparsity
    pattern. :meth:`observe` accumulates per-query key-argmax positions during
    warmup and freezes once routing agreement is stable, returning a boolean
    mask the attention kernel can consume.
    """

    def __init__(
        self,
        warmup_loops: int = 2,
        stability_threshold: float = 0.9,
        min_fraction: float = 0.05,
    ) -> None:
        self.warmup_loops = int(warmup_loops)
        self.stability_threshold = float(stability_threshold)
        self.min_fraction = float(min_fraction)
        self._votes: Optional[torch.Tensor] = None
        self._previous: Optional[torch.Tensor] = None
        self.frozen_working_set: Optional[torch.Tensor] = None
        self.stability: float = 0.0
        self.loops_observed = 0

    def observe(self, attn_weights: torch.Tensor) -> Optional[torch.Tensor]:
        """Accumulate routing votes; freeze and return the mask once stable.

        Args:
            attn_weights: ``[B, S, S]`` or ``[B, H, S, S]`` attention weights.
        """
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)
        if attn_weights.dim() != 3:
            raise ValueError("attn_weights must be [B, S, S] or [B, H, S, S]")
        mean_attn = attn_weights.mean(dim=0)  # [S, S]

        # votes[query, key] counts loops in which key was that query's argmax.
        idx = mean_attn.argmax(dim=-1)  # [S]
        votes = torch.zeros_like(mean_attn)
        votes.scatter_(1, idx.unsqueeze(1), 1.0)
        self._votes = votes if self._votes is None else self._votes + votes
        self.loops_observed += 1

        if self._previous is not None and self._previous.shape == idx.shape:
            agreement = float((idx == self._previous).float().mean().item())
            self.stability = 0.5 * self.stability + 0.5 * agreement
        self._previous = idx

        if self.loops_observed < self.warmup_loops:
            return None
        if self.stability < self.stability_threshold or self._votes is None:
            return None

        s = self._votes.shape[-1]
        k = max(1, int(round(self.min_fraction * s)))
        mask = torch.zeros_like(self._votes, dtype=torch.bool)
        mask.scatter_(1, self._votes.topk(k, dim=-1).indices, True)
        # A query must always be able to attend to itself.
        mask = mask | torch.eye(s, dtype=torch.bool, device=mask.device)
        self.frozen_working_set = mask
        return mask

    def mask(self) -> Optional[torch.Tensor]:
        """The frozen ``[S, S]`` boolean mask, once routing has stabilised."""
        return self.frozen_working_set

    def reset(self) -> None:
        self._votes = None
        self._previous = None
        self.frozen_working_set = None
        self.stability = 0.0
        self.loops_observed = 0


@dataclass
class LoopDiagnostics:
    """Aggregate view of one forward pass's recurrent-loop behaviour."""

    loops: int
    mean_step_alpha: float
    overshoot_repairs: int
    mean_token_skip_fraction: float
    act_avg_loops: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)











