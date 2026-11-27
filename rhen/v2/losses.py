"""Training objectives for RHEN-2. Every task loss is a strictly proper scoring rule.

    L = L_task(last loop) + 0.5 * L_task(each earlier supervised loop)   (deep supervision)
      + 0.1 * L_halt + 0.05 * L_null_margin

choice / noul: log loss + 0.5 Brier. score: ranked probability score + null term + 0.5 log
loss. multi: per-option BCE + 0.5 Brier. range: pinball loss over the nine quantiles.
extract: cross-entropy on start and end. Soft labels (teacher distributions) replace one-hot
targets for the discrete types.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from rhen.v2.config import QUANTILE_LEVELS, TYPE_INDEX, Rhen2Config
from rhen.v2.packing import IGNORE, NULL_INDEX, Rhen2Batch
from rhen.v2.readouts import NEG, ReadoutOut

CHOICE, NOUL, SCORE, MULTI, RANGE, EXTRACT = (TYPE_INDEX[t] for t in ("choice", "noul", "score", "multi", "range", "extract"))
EPS = 1e-9


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _targets(batch: Rhen2Batch, k: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Target distribution [N, K+1] (null last) and a mask of rows that have a discrete label."""
    idx = batch.label_idx.reshape(n)
    y = torch.zeros(n, k + 1, device=idx.device)
    has = idx >= 0
    y[has, idx[has].clamp_max(k - 1)] = 1.0
    is_null = idx == NULL_INDEX
    y[is_null, k] = 1.0
    labelled = has | is_null
    if batch.has_soft is not None:
        soft = batch.has_soft.reshape(n)
        y = torch.where(soft[:, None], batch.label_soft.reshape(n, k + 1), y)
        labelled = labelled | soft
    return y, labelled


def task_losses(ro: ReadoutOut, batch: Rhen2Batch, cfg: Rhen2Config, with_spans=None) -> Dict[str, torch.Tensor]:
    n = ro.logits.shape[0]
    k = ro.logits.shape[1] - 1
    qt = batch.q_type.reshape(n)
    valid = batch.q_valid.reshape(n)
    opt_valid = batch.opt_valid.reshape(n, k)
    y, labelled = _targets(batch, k, n)
    if not cfg.use_null:                                  # a null target is meaningless without a null node
        labelled = labelled & (y[:, k] == 0)
    out: Dict[str, torch.Tensor] = {}

    # choice and noul
    p = ro.probs
    ce = -(y * p.clamp_min(EPS).log()).sum(-1)
    brier = ((p - y) ** 2).sum(-1)
    row = valid & labelled & ((qt == CHOICE) | (qt == NOUL))
    out["choice"] = _mean(ce + 0.5 * brier, row)

    # score
    ps = ro.score_probs
    n_levels = opt_valid.sum(-1).clamp_min(1)
    cum_p = ps[:, :k].cumsum(-1)
    cum_y = y[:, :k].cumsum(-1)
    in_levels = opt_valid.to(ps.dtype)
    rps = (((cum_p - cum_y) ** 2) * in_levels).sum(-1) / (n_levels - 1).clamp_min(1)
    null_term = (ps[:, k] - y[:, k]) ** 2
    ce_s = -(y * ps.clamp_min(EPS).log()).sum(-1)
    row = valid & labelled & (qt == SCORE)
    out["score"] = _mean(rps + null_term + 0.5 * ce_s, row)

    # multi-label
    if batch.has_multi is not None:
        t = batch.label_multi.reshape(n, k)
        logit = ro.multi_logit.clamp_min(NEG / 1e6)
        bce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
        pm = torch.sigmoid(logit)
        per = ((bce + 0.5 * (pm - t) ** 2) * in_levels).sum(-1) / opt_valid.sum(-1).clamp_min(1)
        row = valid & batch.has_multi.reshape(n) & (qt == MULTI)
        out["multi"] = _mean(per, row)

    # range
    if batch.has_range is not None:
        lo, hi = batch.range_lo.reshape(n), batch.range_hi.reshape(n)
        target = ((batch.label_range.reshape(n) - lo) / (hi - lo).clamp_min(1e-9)).clamp(0.0, 1.0)
        tau = torch.tensor(QUANTILE_LEVELS, device=target.device)
        diff = target[:, None] - ro.quantiles
        pin = torch.maximum(tau * diff, (tau - 1.0) * diff).mean(-1)
        row = valid & batch.has_range.reshape(n) & (qt == RANGE)
        out["range"] = _mean(pin, row)

    # null margin: on answerable choice / noul questions the gold option must beat "cannot tell"
    idx = batch.label_idx.reshape(n)
    gold = idx.clamp(min=0, max=k - 1)
    s_gold = ro.logits.gather(1, gold[:, None]).squeeze(1)
    hinge = F.relu(1.0 - (s_gold - ro.logits[:, k]))
    row = valid & (idx >= 0) & ((qt == CHOICE) | (qt == NOUL)) & cfg.use_null
    out["null_margin"] = _mean(hinge, row)

    # halting: should the model stop here? target is whether the argmax is already correct
    pr = torch.where((qt == SCORE)[:, None], ro.score_probs, ro.probs)
    pred = pr.argmax(-1)
    gold_idx = torch.where(idx == NULL_INDEX, torch.full_like(idx, k), idx)
    correct = (pred == gold_idx).to(ro.halt_logit.dtype)
    row = valid & (idx != IGNORE) & ((qt == CHOICE) | (qt == NOUL) | (qt == SCORE))
    out["halt"] = _mean(F.binary_cross_entropy_with_logits(ro.halt_logit, correct, reduction="none"), row)
    return out


def span_loss(out, batch: Rhen2Batch) -> torch.Tensor:
    n = out.span_start.shape[0]
    qt = batch.q_type.reshape(n)
    s, e = batch.span_start.reshape(n), batch.span_end.reshape(n)
    row = batch.q_valid.reshape(n) & (qt == EXTRACT) & (s >= 0)
    ce = F.cross_entropy(out.span_start, s.clamp_min(0), reduction="none") + F.cross_entropy(
        out.span_end, e.clamp_min(0), reduction="none"
    )
    return _mean(ce, row)


def compute_losses(out, batch: Rhen2Batch, cfg: Rhen2Config) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Total loss and a dict of float parts for logging."""
    zero = out.q_vec.new_zeros(())
    per_loop = out.per_loop
    last = task_losses(per_loop[-1], batch, cfg)
    task = last["choice"] + last["score"] + last.get("multi", zero) + last.get("range", zero)
    deep = zero
    for ro in per_loop[:-1]:
        part = task_losses(ro, batch, cfg)
        deep = deep + 0.5 * (part["choice"] + part["score"] + part.get("multi", zero) + part.get("range", zero))
    halt = torch.stack([task_losses(ro, batch, cfg)["halt"] for ro in per_loop]).mean() if len(per_loop) > 1 else last["halt"]
    spans = span_loss(out, batch)
    total = task + deep + spans + 0.1 * halt + 0.05 * last["null_margin"]
    parts = {
        "loss": float(total.detach()), "choice": float(last["choice"].detach()),
        "score": float(last["score"].detach()), "multi": float(last.get("multi", zero).detach()),
        "range": float(last.get("range", zero).detach()), "extract": float(spans.detach()),
        "halt": float(halt.detach()), "null_margin": float(last["null_margin"].detach()),
        "deep": float(deep.detach()),
    }
    return total, parts
