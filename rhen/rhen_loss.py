"""
Rhen Calibrated Multi-Task Decision Losses.

This module owns every Rhen training objective. It was extracted from the former
``train.py`` so that loss definitions live beside the decision engine (``rhen.py``)
instead of inside a heavyweight training script, and so verification suites can
import the losses without pulling in an optimizer / dataloader / checkpoint stack.

Objectives implemented here
---------------------------
1. ``Noul``        : Binary Cross-Entropy + Bernoulli Brier        (w: ``lambda_noul``)
2. ``Choice``      : Cross-Entropy + calibrated Brier              (w: ``lambda_choice``)
3. ``Score``       : Ranked Probability Score (RPS)                (w: ``lambda_score``)
4. ``MultiChoice`` : per-option BCE + Brier                        (w: ``lambda_noul``)
5. ``Range``       : Beta-parameterized MSE on the normalized interval
6. ``Extract``     : Start/End span cross-entropy
7. Slot orthogonality : Frobenius-norm penalty against slot collapse
8. ``L_DDPO``      : Direct Decision Preference Optimization
                     ``L_DDPO = -log sigma(beta * (s(q, o+) - s(q, o-)))``
9. ``L_perm``      : Candidate Option Permutation Invariance
                     ``L_perm = KL( P(O) || pi^-1( P(pi(O)) ) )`` on the pointer head only,
                     which scores options independently, so it is zero by construction.
10. ``L_order``    : Option-order consistency at the trunk level: the question is packed again
                     with its options reordered, the model is rerun, and the symmetric KL to the
                     original decision (per option identity) is penalised
                     (``compute_option_order_consistency_loss``).

Total objective (Pillar 4)::

    L_total = L_LM + L_mtp
              + lambda_choice * (L_CE + 0.5 * L_Brier)
              + lambda_noul   * (L_BCE + 0.5 * L_Brier_Bernoulli)
              + lambda_score  * L_RPS
              + L_range + L_extract
              + L_ortho + lambda_perm * L_perm + lambda_ddpo * L_DDPO

``lambda_noul`` weights the whole independent-Bernoulli family (binary ``Noul`` and
multi-label ``MultiChoice``), matching the original formulation.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:  # ``model`` does not import this module; keep the import optional anyway.
    try:
        from rhen.model import EOSModel
    except ImportError:
        from model import EOSModel
except ImportError:  # pragma: no cover - exercised only in partial installs
    EOSModel = Any


# --------------------------------------------------------------------------------------
# Hyper-parameter defaults
# --------------------------------------------------------------------------------------

#: Inverse temperature of the implicit Bradley-Terry preference model in L_DDPO.
DEFAULT_DDPO_BETA: float = 0.1

#: Numerical floor applied to both distributions inside the permutation KL.
DEFAULT_PERM_EPS: float = 1e-8

#: Mixing weight of the Brier term relative to the (log-)likelihood term.
DEFAULT_BRIER_WEIGHT: float = 0.5


# --------------------------------------------------------------------------------------
# L_DDPO - Direct Decision Preference Optimization
# --------------------------------------------------------------------------------------

def compute_ddpo_loss(
    chosen_scores: Union[torch.Tensor, List[float]],
    rejected_scores: Union[torch.Tensor, List[float]],
    beta: float = DEFAULT_DDPO_BETA,
    margin: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Direct Decision Preference Optimization (DDPO) loss.

    Aligns the Rhen latent pointer head directly on chosen / rejected option
    pairs, bypassing the variance of RL rollout:

        L_DDPO = -log sigmoid( beta * ( s(q, o+) - s(q, o-) - margin ) )

    Higher ``beta`` sharpens the preference margin (inverse temperature of the
    implicit Bradley-Terry preference model). ``margin`` requires the preferred
    option to win by at least that amount.

    Args:
        chosen_scores: Per-sample score s(q, o+) of the preferred option.
        rejected_scores: Per-sample score s(q, o-) of the dispreferred option.
        beta: Inverse-temperature scaling of the preference margin.
        margin: Minimum required score separation (default 0.0).
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    """
    if not torch.is_tensor(chosen_scores):
        chosen_scores = torch.as_tensor(chosen_scores, dtype=torch.float32)
    if not torch.is_tensor(rejected_scores):
        rejected_scores = torch.as_tensor(rejected_scores, dtype=torch.float32)

    chosen = chosen_scores.float().reshape(-1)
    rejected = rejected_scores.float().reshape(-1)
    if chosen.numel() != rejected.numel():
        raise ValueError(
            f"DDPO requires matching chosen/rejected shapes, got {tuple(chosen.shape)} "
            f"vs {tuple(rejected.shape)}"
        )

    logits = float(beta) * (chosen - rejected - float(margin))
    loss = -F.logsigmoid(logits)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


# --------------------------------------------------------------------------------------
# L_perm - Candidate Option Permutation Invariance
# --------------------------------------------------------------------------------------

def compute_permutation_invariance_loss(
    pointer_head,
    q_repr: torch.Tensor,
    option_reprs: torch.Tensor,
    permutation: Optional[List[int]] = None,
    kv_latent: Optional[torch.Tensor] = None,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Candidate Option Permutation Invariance Loss (L_perm).

    One-directional KL between the pointer distribution and its permuted counterpart:

        L_perm = KL( P(O) || pi^-1( P(pi(O)) ) )

    where ``pi`` re-orders the candidate options, ``P(pi(O))`` is the pointer
    distribution computed over the permuted option order and ``pi^-1`` maps that
    distribution back onto the original index space. When the head is fully
    order-invariant the two distributions coincide and the loss is exactly 0.

    Note: the pointer head scores each option independently, so it is order-invariant by
    construction and this term is zero for the head. It does not measure position bias in
    the trunk representations that feed the head.

    Args:
        pointer_head: ``MultiHeadRhenPointerHead`` instance (or any callable
            returning a ``"probabilities"`` key when ``return_details=True``).
        q_repr: ``[B, 1, H]`` (or ``[B, H]`` / list of layer tensors) query.
        option_reprs: ``[B, K, H]`` (or ``[K, H]`` / list of layer tensors) options.
        permutation: Candidate index permutation (defaults to a random draw).
        kv_latent: Optional ``[B, S, H]`` MLA Key/Value latent for DLHP probing.
        reduction: ``"batchmean"``, ``"sum"`` or ``"none"``.
    """
    # Normalize option representations to [B, K, H] to match the pointer head.
    if isinstance(option_reprs, list):
        option_reprs = pointer_head.fuse_layers(
            [o.unsqueeze(0) if o.ndim == 2 else o for o in option_reprs]
        )
    elif option_reprs.ndim == 2:
        option_reprs = option_reprs.unsqueeze(0)

    num_options = int(option_reprs.shape[1])
    if num_options < 2:
        return torch.zeros((), device=option_reprs.device, dtype=torch.float32)

    if permutation is None:
        perm = torch.randperm(num_options, device=option_reprs.device).tolist()
    else:
        perm = [int(p) for p in permutation]

    if sorted(perm) != list(range(num_options)):
        raise ValueError(f"permutation must be a bijection of range({num_options}), got {perm}")

    inverse = [0] * num_options
    for src, dst in enumerate(perm):
        inverse[dst] = src

    perm_idx = torch.as_tensor(perm, device=option_reprs.device, dtype=torch.long)
    inv_idx = torch.as_tensor(inverse, device=option_reprs.device, dtype=torch.long)

    p_original = pointer_head(
        q_repr, option_reprs, kv_latent=kv_latent, return_details=True
    )["probabilities"]
    p_permuted = pointer_head(
        q_repr,
        option_reprs.index_select(1, perm_idx),
        kv_latent=kv_latent,
        return_details=True,
    )["probabilities"].index_select(1, inv_idx)

    p = p_original.clamp_min(DEFAULT_PERM_EPS)
    q = p_permuted.clamp_min(DEFAULT_PERM_EPS)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    # KL divergence is non-negative; float32 rounding can push tiny values slightly below zero.
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).clamp_min(0.0)

    if reduction == "none":
        return kl
    if reduction == "sum":
        return kl.sum()
    return kl.mean()


# --------------------------------------------------------------------------------------
# Decision curriculum loading (JSONL)
# --------------------------------------------------------------------------------------

def load_rhen_curriculum(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL decision-curriculum file for auxiliary Rhen loss training.

    Each line must be a JSON object containing:
      - ``state``: state payload (dict or string) fed to the Rhen packer.
      - ``questions``: list of ``[question, options, primitive_type, metadata]``.
      - ``labels``: list of per-branch targets
        (``int`` choice, ``bool`` noul, ``float`` score, ``list`` multichoice,
        ``[start, end]`` extract, ``{"value": float}`` range).

    Optional keys: ``ddpo_pairs`` (``[[chosen_idx, rejected_idx], ...]``),
    ``compute_perm_loss`` (bool) and the ``lambda_*`` per-item overrides.
    """
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_no} of '{path}': {exc}"
                ) from exc
            if "state" not in obj or "questions" not in obj or "labels" not in obj:
                raise ValueError(
                    f"Line {line_no} of '{path}' must define 'state', 'questions' and 'labels'."
                )
            items.append(obj)
    return items


def build_rhen_curriculum_provider(
    path: str,
    tokenizer=None,
    repeat: bool = True,
    cache: bool = True,
) -> Callable[[int], Optional[Dict[str, Any]]]:
    """Build a per-step provider closure over a JSONL decision curriculum.

    The closure cycles through the curriculum, packing each entry with the Rhen
    delimiter packer (results are memoized because packing is deterministic for a
    fixed state/question triple). Pass the resulting callable to the auxiliary
    Rhen training step to obtain ``{"packed_sequence", "branch_labels", ...}``.
    """
    try:
        from rhen.rhen_packer import get_rhen_packer
    except ImportError:
        from rhen_packer import get_rhen_packer

    items = load_rhen_curriculum(path)
    if not items:
        raise ValueError(f"Decision curriculum '{path}' contains no usable entries.")

    packer = get_rhen_packer(tokenizer)
    packed_cache: Dict[int, Any] = {}

    def provider(step: int) -> Optional[Dict[str, Any]]:
        idx = (int(step) - 1) % len(items)
        if not repeat and int(step) > len(items):
            return None
        item = items[idx]

        if cache and idx in packed_cache:
            packed = packed_cache[idx]
        else:
            packed = packer.pack(
                item["state"],
                [tuple(q) for q in item.get("questions", [])],
            )
            if cache:
                packed_cache[idx] = packed

        spec: Dict[str, Any] = {
            "packed_sequence": packed,
            "branch_labels": item["labels"],
            "ddpo_pairs": item.get("ddpo_pairs"),
            "compute_perm_loss": bool(item.get("compute_perm_loss", False)),
        }
        for lam in ("lambda_choice", "lambda_noul", "lambda_score", "lambda_ddpo", "lambda_perm"):
            if lam in item:
                spec[lam] = item[lam]
        return spec

    return provider


# --------------------------------------------------------------------------------------
# The multi-task objective
# --------------------------------------------------------------------------------------

def _ordinal_cdf_target(num_levels: int, label: float, device: torch.device) -> torch.Tensor:
    """Cumulative target ``F(k) = P(level <= k)`` for an ordinal label on ``0 .. num_levels - 1``.

    An integer label is a point mass at that level, so ``F(k)`` is 0 below it and 1 from it
    upwards. A fractional label splits its mass linearly between the two neighbouring levels,
    which gives ``F(k) = clamp(k + 1 - label, 0, 1)``. The label is clamped to the scale first.
    """
    norm_target = min(max(float(label), 0.0), float(num_levels - 1))
    levels = torch.arange(num_levels, device=device, dtype=torch.float32)
    return torch.clamp(levels + 1.0 - norm_target, 0.0, 1.0)


def compute_rhen_multi_task_loss(
    model: "EOSModel",
    packed_sequence,
    branch_labels: List[Union[int, float, bool]],
    device: torch.device,
    ddpo_pairs: Optional[List[Tuple[int, int]]] = None,
    compute_perm_loss: bool = False,
    ddpo_beta: float = DEFAULT_DDPO_BETA,
    lambda_choice: float = 1.0,
    lambda_noul: float = 1.0,
    lambda_score: float = 1.0,
    lambda_ddpo: float = 1.0,
    lambda_perm: float = 1.0,
    perm_samples: int = 1,
    precomputed_results: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the exact multi-task Rhen loss over a packed decision sequence.

    Dispatch is driven by the *label* type first (bool -> Noul, int -> Choice,
    float -> Score, list/set -> MultiChoice, tuple -> Extract, dict with ``value``
    -> Range) with the packed branch's ``primitive_type`` used as a tie-breaker.

    ``ddpo_pairs`` holds ``(chosen_idx, rejected_idx)`` candidate index tuples
    resolved against the first branch exposing a per-option score vector.
    ``compute_perm_loss`` requires ``forward_rhen`` (not the ``@torch.no_grad``
    ``evaluate_rhen`` fallback) to retain the pointer graph.

    ``precomputed_results`` reuses the output of an earlier ``forward_rhen`` call on the same
    ``packed_sequence`` (for example one that also feeds another objective), so the decision
    forward pass is not run twice.
    """
    if precomputed_results is not None:
        results = precomputed_results
    elif hasattr(model, "forward_rhen"):
        results = model.forward_rhen(packed_sequence)
    else:
        results = model.evaluate_rhen(packed_sequence)

    total_loss = torch.zeros((), device=device)
    loss_breakdown: Dict[str, float] = {}

    for branch_idx, (branch_meta, label) in enumerate(zip(packed_sequence.branches, branch_labels)):
        res = results[branch_idx]
        probs = res.get("probs_tensor", None)
        if probs is not None:
            probs = probs[0].to(device)
        else:
            probs = torch.tensor(res["probabilities"], device=device, dtype=torch.float32)

        if getattr(branch_meta, "primitive_type", "choice") == "range" or (
            isinstance(label, dict) and "value" in label
        ):
            target_val = float(label if not isinstance(label, dict) else label.get("value", 0.0))
            min_v = float(res.get("min_val", 0.0))
            max_v = float(res.get("max_val", 100.0))
            norm_target = min(
                max((target_val - min_v) / max(max_v - min_v, 1e-6), 0.0), 1.0
            )

            alpha = res.get("alpha_tensor", None)
            beta = res.get("beta_tensor", None)
            if alpha is not None and beta is not None:
                mu_norm = alpha / (alpha + beta)
                loss_range = (mu_norm[0, 0] - norm_target).pow(2)
            else:
                pred_val = float(res.get("expected_value", (min_v + max_v) / 2.0))
                loss_range = torch.tensor(
                    ((pred_val - target_val) / max(max_v - min_v, 1e-6)) ** 2, device=device
                )
            total_loss = total_loss + loss_range
            loss_breakdown[f"b{branch_idx}_range_mse"] = loss_range.item()

        elif getattr(branch_meta, "primitive_type", "choice") == "extract" or (
            isinstance(label, tuple) and len(label) == 2
        ):
            start_logits = res.get("start_logits", None)
            end_logits = res.get("end_logits", None)
            tgt_start, tgt_end = label if isinstance(label, tuple) else (0, 0)
            if start_logits is not None and end_logits is not None:
                t_s = torch.tensor([tgt_start], device=device, dtype=torch.long)
                t_e = torch.tensor([tgt_end], device=device, dtype=torch.long)
                loss_span = F.cross_entropy(start_logits, t_s) + F.cross_entropy(end_logits, t_e)
            else:
                loss_span = torch.zeros((), device=device)
            total_loss = total_loss + loss_span
            loss_breakdown[f"b{branch_idx}_extract_span"] = loss_span.item()

        elif isinstance(label, (list, set)) or getattr(branch_meta, "primitive_type", "choice") == "multichoice":
            target_indices = label if isinstance(label, (list, set)) else []
            target_vec = torch.zeros(len(probs), device=device, dtype=torch.float32)
            for idx in target_indices:
                if 0 <= idx < len(probs):
                    target_vec[idx] = 1.0
            p_clamped = probs.clamp(1e-6, 1.0 - 1e-6)
            bce_mc = F.binary_cross_entropy(p_clamped, target_vec)
            brier_mc = torch.mean((probs - target_vec) ** 2)
            total_loss = total_loss + lambda_noul * (bce_mc + DEFAULT_BRIER_WEIGHT * brier_mc)
            loss_breakdown[f"b{branch_idx}_multichoice_bce"] = bce_mc.item()
            loss_breakdown[f"b{branch_idx}_multichoice_brier"] = brier_mc.item()

        elif isinstance(label, bool):
            target = torch.tensor([1.0 if label else 0.0], device=device, dtype=torch.float32)
            p_true = probs[0:1].clamp(1e-6, 1.0 - 1e-6)
            loss_noul = F.binary_cross_entropy(p_true, target)
            brier_noul = (probs[0] - target[0]).pow(2)
            total_loss = total_loss + lambda_noul * (loss_noul + DEFAULT_BRIER_WEIGHT * brier_noul)
            loss_breakdown[f"b{branch_idx}_noul_bce"] = loss_noul.item()
            loss_breakdown[f"b{branch_idx}_noul_brier"] = brier_noul.item()

        elif isinstance(label, int):
            logits = res.get("logits_tensor", None)
            if logits is not None:
                log_probs = F.log_softmax(logits[0].to(device), dim=-1)
            else:
                log_probs = torch.log(probs.clamp(min=1e-8))

            target_idx = torch.tensor([label], device=device, dtype=torch.long)
            loss_choice = F.nll_loss(log_probs.unsqueeze(0), target_idx)

            # Calibrated Brier score: (1/K) * sum((p_k - y_k)^2)
            target_one_hot = F.one_hot(target_idx, num_classes=len(probs)).float()[0]
            brier_loss = torch.mean((probs - target_one_hot) ** 2)

            total_loss = total_loss + lambda_choice * (loss_choice + DEFAULT_BRIER_WEIGHT * brier_loss)
            loss_breakdown[f"b{branch_idx}_choice_ce"] = loss_choice.item()
            loss_breakdown[f"b{branch_idx}_choice_brier"] = brier_loss.item()

        elif isinstance(label, float):
            num_levels = len(probs)
            true_cdf = _ordinal_cdf_target(num_levels, label, device)
            pred_cdf = torch.cumsum(probs, dim=-1)
            rps_loss = torch.mean((pred_cdf[:-1] - true_cdf[:-1]) ** 2)
            total_loss = total_loss + lambda_score * rps_loss
            loss_breakdown[f"b{branch_idx}_score_rps"] = rps_loss.item()

    # 7. Slot orthogonality - prevents latent decision-slot collapse.
    rhen_engine = getattr(getattr(model, "recurrent_core", None), "rhen", None)
    if rhen_engine is not None:
        ortho_loss = rhen_engine.compute_ortho_loss() * rhen_engine.ortho_loss_coef
        total_loss = total_loss + ortho_loss
        loss_breakdown["rhen_ortho"] = ortho_loss.item()

    pointer_head = getattr(rhen_engine, "pointer_head", None)

    # 8. Direct Decision Preference Optimization (DDPO) on latent pointer scores.
    #    Preference pairs reference candidate indices inside the first branch that
    #    exposes a per-option score vector (choice / noul / score / multichoice).
    if ddpo_pairs:
        score_matrix = None
        for res in results:
            candidate = res.get("scores_tensor", None)
            if candidate is not None and candidate.ndim == 2 and candidate.shape[-1] > 1:
                score_matrix = candidate.to(device)
                break
        if score_matrix is None:
            # Fall back to log-probabilities so DDPO still receives a usable ranking signal.
            for res in results:
                probs_t = res.get("probs_tensor", None)
                if probs_t is not None and probs_t.ndim == 2 and probs_t.shape[-1] > 1:
                    score_matrix = torch.log(probs_t.to(device).clamp_min(1e-8))
                    break

        if score_matrix is not None:
            num_candidates = int(score_matrix.shape[-1])
            pair_losses = []
            for pair in ddpo_pairs:
                if len(pair) != 2:
                    continue
                chosen_idx, rejected_idx = int(pair[0]), int(pair[1])
                if chosen_idx == rejected_idx:
                    continue
                if not (0 <= chosen_idx < num_candidates and 0 <= rejected_idx < num_candidates):
                    continue
                pair_losses.append(
                    compute_ddpo_loss(
                        score_matrix[:, chosen_idx],
                        score_matrix[:, rejected_idx],
                        beta=ddpo_beta,
                        reduction="mean",
                    )
                )
            if pair_losses:
                ddpo_loss = torch.stack(pair_losses).mean()
                total_loss = total_loss + lambda_ddpo * ddpo_loss
                loss_breakdown["rhen_ddpo"] = ddpo_loss.item()
            else:
                loss_breakdown["rhen_ddpo"] = 0.0
        else:
            loss_breakdown["rhen_ddpo"] = 0.0

    # 9. Candidate Option Permutation Invariance (L_perm) on the live pointer head.
    if compute_perm_loss and pointer_head is not None:
        perm_losses = []
        for res in results:
            q_layers = res.get("q_layers", None)
            opt_layers = res.get("opt_layers", None)
            if q_layers is None or opt_layers is None:
                continue
            num_opts = int(opt_layers[0].shape[1])
            if num_opts < 2:
                continue
            for _ in range(max(1, int(perm_samples))):
                perm_losses.append(
                    compute_permutation_invariance_loss(
                        pointer_head,
                        q_layers,
                        opt_layers,
                        permutation=None,
                        kv_latent=res.get("kv_state_latent", None),
                    )
                )
        if perm_losses:
            perm_loss = torch.stack(perm_losses).mean()
            total_loss = total_loss + lambda_perm * perm_loss
            loss_breakdown["rhen_perm"] = perm_loss.item()
        else:
            loss_breakdown["rhen_perm"] = 0.0

    loss_breakdown["total_rhen_loss"] = total_loss.item()
    return total_loss, loss_breakdown


# --------------------------------------------------------------------------------------
# Option-order consistency (trunk level)
# --------------------------------------------------------------------------------------

#: Primitives whose options are unordered alternatives. Score levels are ordinal, so a
#: reordering changes their meaning and they are excluded.
ORDER_FREE_PRIMITIVES = ("noul", "choice", "multichoice")


def sample_option_permutation(num_options: int, generator: Optional[torch.Generator] = None) -> List[int]:
    """A random non-identity permutation of ``range(num_options)`` (identity only when n < 2)."""
    if num_options < 2:
        return list(range(num_options))
    perm = torch.randperm(num_options, generator=generator).tolist()
    if perm == list(range(num_options)):
        perm = perm[1:] + perm[:1]
    return perm


def option_order_consistency(
    probs_original: torch.Tensor,
    probs_permuted: torch.Tensor,
    permutation: List[int],
    primitive: str = "choice",
    eps: float = DEFAULT_PERM_EPS,
) -> torch.Tensor:
    """Symmetric KL between a decision and the decision on reordered options, per option identity.

    ``probs_permuted[j]`` is the probability of original option ``permutation[j]``. Choice and
    noul compare the categorical distributions; multichoice compares each option's independent
    Bernoulli probability. Zero exactly when the decision ignores presentation order.
    """
    p = probs_original.reshape(-1).float()
    q_perm = probs_permuted.reshape(-1).float()
    if p.numel() != len(permutation) or q_perm.numel() != len(permutation):
        raise ValueError("probability vectors and permutation must have the same length")
    index = torch.as_tensor(permutation, device=q_perm.device, dtype=torch.long)
    q = torch.empty_like(q_perm).scatter(0, index, q_perm)  # q[permutation[j]] = q_perm[j]
    if primitive == "multichoice":
        p, q = p.clamp(eps, 1.0 - eps), q.clamp(eps, 1.0 - eps)
        kl_pq = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
        kl_qp = q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()
        return 0.5 * (kl_pq + kl_qp).mean()
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p, q = p / p.sum(), q / q.sum()
    return 0.5 * ((p * (p / q).log()).sum() + (q * (q / p).log()).sum())


def compute_option_order_consistency_loss(
    model: "EOSModel",
    state: Any,
    questions: List[Tuple[str, List[str], str, Optional[Dict[str, Any]]]],
    original_results: Optional[List[Dict[str, Any]]] = None,
    generator: Optional[torch.Generator] = None,
    tokenizer=None,
    num_loops: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Penalise position bias in the trunk by re-presenting every question with reordered options.

    Unlike ``compute_permutation_invariance_loss`` (which permutes options *after* the trunk
    and is zero by construction), this packs the options in a new order, runs the full model
    again and compares the decisions option by option. ``questions`` are packer 4-tuples
    ``(question, options, primitive, metadata)``; ``original_results`` reuses an existing
    ``forward_rhen`` output for the original order. Returns the mean over eligible questions.
    """
    try:
        from rhen.rhen_packer import get_rhen_packer
    except ImportError:
        from rhen_packer import get_rhen_packer

    packer = get_rhen_packer(tokenizer)
    if original_results is None:
        original_results = model.forward_rhen(packer.pack(state=state, questions_and_options=questions), num_loops=num_loops)

    permutations: List[List[int]] = []
    permuted_questions = []
    for question, options, primitive, meta in questions:
        perm = sample_option_permutation(len(options), generator) if primitive in ORDER_FREE_PRIMITIVES else list(range(len(options)))
        permutations.append(perm)
        permuted_questions.append((question, [options[i] for i in perm], primitive, meta))

    eligible = [
        i for i, (q, perm) in enumerate(zip(questions, permutations))
        if q[2] in ORDER_FREE_PRIMITIVES and perm != list(range(len(perm)))
    ]
    device = next(model.parameters()).device
    if not eligible:
        return torch.zeros((), device=device), {"order_consistency": 0.0, "order_consistency_n": 0}

    permuted_results = model.forward_rhen(packer.pack(state=state, questions_and_options=permuted_questions), num_loops=num_loops)
    terms = [
        option_order_consistency(
            original_results[i]["probs_tensor"][0], permuted_results[i]["probs_tensor"][0], permutations[i], questions[i][2]
        )
        for i in eligible
    ]
    loss = torch.stack(terms).mean()
    return loss, {"order_consistency": float(loss.item()), "order_consistency_n": len(terms)}


__all__ = [
    "DEFAULT_DDPO_BETA",
    "DEFAULT_PERM_EPS",
    "DEFAULT_BRIER_WEIGHT",
    "compute_ddpo_loss",
    "compute_permutation_invariance_loss",
    "compute_rhen_multi_task_loss",
    "ORDER_FREE_PRIMITIVES",
    "sample_option_permutation",
    "option_order_consistency",
    "compute_option_order_consistency_loss",
    "load_rhen_curriculum",
    "build_rhen_curriculum_provider",
]
