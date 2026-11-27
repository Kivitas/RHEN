"""Flat baseline for plan section 10.4 item 7: one bidirectional encoder over the whole sequence.

This is the layout that Laya-style models use, built from the same pieces as RHEN-2 so the
comparison is about the architecture and not about the parts:

    [ state ; question ; marker_1 option_1 tokens ; marker_2 option_2 tokens ; ... ]

* ONE stack of ``EncoderBlock`` (global attention, RoPE over the concatenated sequence, pre-norm
  SwiGLU), the same factorised embedding, segment embedding, question-type embedding and numeric
  channel as ``Rhen2Model``.
* Each option is read at its marker position (``option_read="marker"``) or mean pooled over its
  marker and tokens (``option_read="mean"``). The question is read at its first token (the
  ``CTRL_TYPE_x`` control id acts as the question marker). The null node is a learned vector.
* The same ``Readout`` heads as RHEN-2 (choice / noul / score / multi / range / halt / spans),
  so ``compute_losses`` and ``rhen.v2.eval`` work unchanged.
* NO deliberation loops, NO option-set layer, NO state cache. ``loops`` is accepted and ignored.
* Each question is encoded in its OWN copy of the sequence (the state is repeated once per
  question), as Laya does, so questions stay independent. Cost: a batch with M questions per
  example runs M sequences of length Ls + Lq + K * (1 + Lo) through every layer; RHEN-2 reads the
  state once and spends the loops on the small question and option workspace only.

Structural consequences (documented, not tuned away): option order is visible through RoPE
positions and through attention between options, so predictions are NOT exactly invariant to
option order; and nothing is cached across questions on the same state.

``matched_flat_config`` picks the layer count (and, only if the layer grid cannot reach 3 percent,
the FFN width) so the parameter count equals the RHEN-2 model's. ``flops_estimate`` gives matmul
FLOPs per example for either model so a report can compare compute as well as parameters.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rhen.v2.config import TYPE_INDEX, Rhen2Config, count_parameters
from rhen.v2.layers import EncoderBlock, FactorisedEmbedding, NumericChannel, RMSNorm, Rotary
from rhen.v2.model import Rhen2Output
from rhen.v2.packing import Rhen2Batch
from rhen.v2.readouts import Readout

SCORE = TYPE_INDEX["score"]
PARAM_TOLERANCE = 0.03


class FlatBaselineModel(nn.Module):
    """Single-encoder baseline with the same ``forward(batch, loops=None)`` interface as ``Rhen2Model``.

    Uses ``cfg.n_state_layers`` as its number of encoder layers; the workspace fields
    (``n_core_layers``, ``n_set_layers``, loops, scratch) are ignored.
    """

    def __init__(self, cfg: Rhen2Config, option_read: str = "marker"):
        super().__init__()
        if option_read not in ("marker", "mean"):
            raise ValueError("option_read must be 'marker' or 'mean'")
        self.cfg = cfg
        self.option_read = option_read
        d = cfg.d_model
        self.embed = FactorisedEmbedding(cfg.vocab_size, cfg.d_embed, d)
        self.segment = nn.Embedding(3, d)               # state, question, option
        self.qtype = nn.Embedding(len(TYPE_INDEX), d)
        self.numeric = NumericChannel(d) if cfg.use_numeric else None
        self.marker = nn.Parameter(torch.zeros(d))      # option marker token
        self.null_vec = nn.Parameter(torch.zeros(d))    # learned "cannot tell" vector
        self.blocks = nn.ModuleList(
            EncoderBlock(d, cfg.n_heads, cfg.ffn_hidden, cfg.norm_eps) for _ in range(cfg.n_state_layers)
        )
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.rotary = Rotary(cfg.head_dim, cfg.rope_theta)
        self.readout = Readout(cfg)
        self.apply(self._init)
        scale = (2 * max(cfg.n_state_layers, 1)) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("ffn.down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std * scale)
        with torch.no_grad():
            self.marker.normal_(0.0, cfg.init_std)
            self.null_vec.normal_(0.0, cfg.init_std)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        if any(p.dtype in (torch.bfloat16, torch.float16) for p in self.readout.parameters()):
            self.readout.float()
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def forward(self, batch: Rhen2Batch, loops: Optional[Any] = None) -> Rhen2Output:
        cfg = self.cfg
        b, m, lq = batch.q_ids.shape
        k, lo = batch.opt_ids.shape[2], batch.opt_ids.shape[3]
        n = b * m
        ls = batch.state_ids.shape[1]
        d = cfg.d_model
        dev = batch.state_ids.device

        xs = self.embed(batch.state_ids) + self.segment.weight[0]
        if self.numeric is not None:
            xs = xs + self.numeric(batch.state_num.to(xs.dtype)).to(xs.dtype)
        xq = self.embed(batch.q_ids) + self.segment.weight[1] + self.qtype(batch.q_type)[:, :, None]
        if self.numeric is not None:
            xq = xq + self.numeric(batch.q_num.to(xq.dtype)).to(xq.dtype)
        xo = self.embed(batch.opt_ids) + self.segment.weight[2]
        marker = (self.marker + self.segment.weight[2]).to(xo.dtype)
        xo = torch.cat([marker.expand(b, m, k, 1, d), xo], dim=3)                      # [B, M, K, 1 + Lo, d]
        mo = torch.cat([batch.opt_valid[..., None], batch.opt_mask & batch.opt_valid[..., None]], dim=3)

        state_mask = batch.state_mask
        xs_q = xs[:, None].expand(b, m, ls, d).reshape(n, ls, d)                      # state repeated per question
        sm_q = state_mask[:, None].expand(b, m, ls).reshape(n, ls)
        x = torch.cat([xs_q, xq.reshape(n, lq, d), xo.reshape(n, k * (1 + lo), d)], dim=1)
        mask = torch.cat([sm_q, batch.q_mask.reshape(n, lq), mo.reshape(n, k * (1 + lo))], dim=1)
        length = x.shape[1]

        # RoPE positions run over the real tokens of the concatenated sequence, not over padding.
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        cos, sin = self.rotary(length, dev)
        rope = (cos[pos][:, None], sin[pos][:, None])                                   # [N, 1, L, hd]
        for block in self.blocks:
            x = block(x, key_mask=mask, rope=rope)
        h = self.norm(x)

        S = h[:, :ls]
        q_vec = h[:, ls]
        opt = h[:, ls + lq:].reshape(n, k, 1 + lo, d)
        if self.option_read == "marker":
            u = opt[:, :, 0]
        else:
            w = mo.reshape(n, k, 1 + lo).to(opt.dtype)[..., None]
            u = (opt * w).sum(2) / w.sum(2).clamp_min(1.0)
        opt_valid = batch.opt_valid.reshape(n, k)
        null = self.null_vec.expand(n, d)

        with torch.autocast(device_type=h.device.type, enabled=False):
            final = self.readout(q_vec.float(), u.float(), null.float(), opt_valid)
            s_proj = self.readout.project_state_for_spans(S.float())
            start, end = self.readout.span_logits(q_vec.float(), s_proj, sm_q, group=1)

        q_type = batch.q_type.reshape(n)
        probs = torch.where((q_type == SCORE)[:, None], final.score_probs, final.probs)
        lo_r = batch.range_lo.reshape(n, 1)
        hi_r = batch.range_hi.reshape(n, 1)
        out = Rhen2Output(
            final=final, per_loop=[final], halts=[final.halt_logit],
            loops_run=torch.ones(n, dtype=torch.long, device=dev),
            probs=probs, multi_prob=torch.sigmoid(final.multi_logit),
            quantiles=lo_r + (hi_r - lo_r) * final.quantiles,
            span_start=start, span_end=end, evidence=None, q_vec=q_vec,
            batch_size=b, max_questions=m,
        )
        if batch.label_idx is not None:
            from rhen.v2.losses import compute_losses

            out.loss, parts = compute_losses(out, batch, cfg)
            out.loss_parts = parts
        return out


# ---------------------------------------------------------------------------
# Parameter accounting and matching
# ---------------------------------------------------------------------------

def _block_params(cfg: Rhen2Config, ffn_hidden: Optional[int] = None) -> int:
    d, hd = cfg.d_model, cfg.head_dim
    h = cfg.ffn_hidden if ffn_hidden is None else ffn_hidden
    return 4 * d * d + 2 * hd + 3 * d * h + 2 * d


def _fixed_params(cfg: Rhen2Config) -> int:
    """Everything except the encoder blocks: embeddings, numeric channel, final norm, markers, readout."""
    parts = count_parameters(cfg)
    return parts["embedding"] + parts["numeric"] + parts["readout"] + cfg.d_model + 2 * cfg.d_model


def count_flat_parameters(cfg: Rhen2Config) -> int:
    """Exact parameter count of ``FlatBaselineModel(cfg)`` (uses ``cfg.n_state_layers``)."""
    return _fixed_params(cfg) + cfg.n_state_layers * _block_params(cfg)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(b, 1.0)


DEFAULT_REF = {"n_state": 128, "n_q": 24, "k": 4, "opt_tokens": 4}


def matched_flat_config(
    cfg: Rhen2Config,
    target_params: Optional[int] = None,
    target_flops: Optional[float] = None,
    ref: Optional[Dict[str, int]] = None,
    max_layers: int = 96,
) -> Rhen2Config:
    """Flat-baseline config matched to a RHEN-2 config.

    ``target_params`` (default: ``count_parameters(cfg)['total']``) picks ``n_state_layers`` at the
    given width; if no layer count lands within 3 percent, the FFN width is solved so the count
    is met to within a few parameters (the layer count whose FFN multiplier is closest to the
    RHEN-2 default is used). ``target_flops`` instead picks the layer count whose flat FLOPs per
    question at the reference lengths ``ref`` (keys n_state, n_q, k, opt_tokens) is closest to the
    target; width and FFN stay as in ``cfg``.
    """
    if target_flops is not None:
        r = dict(DEFAULT_REF, **(ref or {}))
        best = min(range(1, max_layers + 1), key=lambda L: abs(
            _flat_question_flops(dataclasses.replace(cfg, n_state_layers=L), r["n_state"], r["n_q"], r["k"],
                                 r["k"] * r["opt_tokens"], r["k"] * r["opt_tokens"] ** 2) - target_flops))
        return dataclasses.replace(cfg, n_state_layers=best)

    target = int(count_parameters(cfg)["total"] if target_params is None else target_params)
    fixed = _fixed_params(cfg)
    if target <= fixed + _block_params(cfg):
        raise ValueError(f"target_params={target} leaves no room for an encoder layer (fixed part is {fixed})")
    best_l = min(range(1, max_layers + 1), key=lambda L: abs(fixed + L * _block_params(cfg) - target))
    if _rel(fixed + best_l * _block_params(cfg), target) <= PARAM_TOLERANCE:
        return dataclasses.replace(cfg, n_state_layers=best_l)

    d, hd = cfg.d_model, cfg.head_dim
    base_mult = cfg.ffn_hidden / d
    candidates = []
    for L in range(1, max_layers + 1):
        h = (target - fixed - L * (4 * d * d + 2 * hd + 2 * d)) // (3 * d * L)
        if h >= d:
            candidates.append((abs(h / d - base_mult), L, int(h)))
    if not candidates:
        raise ValueError("no layer count can reach the target parameter count")
    _, L, h = min(candidates)
    return dataclasses.replace(cfg, n_state_layers=L, ffn_mult=(h + 0.25) / d)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------
# Matmul FLOPs only (2 * m * n * k per product); norms, softmax, RoPE and elementwise ops are ignored.
# One self-attention transformer layer on T tokens with hidden width d and FFN width h:
#     q,k,v,o projections   8 T d^2
#     scores + weighted sum 4 T^2 d          (QK^T and AV, all heads)
#     SwiGLU FFN            6 T d h
# Cross-attention from Tq query tokens onto Tk key tokens: 4 Tq d^2 (q, o) + 4 Tk d^2 (k, v, only when
# the keys are not cached) + 4 Tq Tk d. Embedding up-projection 2 T d_e d per token; numeric channel
# 2 (d + d^2 / 4) per state or question token; readout heads: 2 d dr (K + 2) plus the span projection
# 2 Ls d dr and span scores 4 Ls dr.
#
# flat model, per question:  Ls + Lq + K + To tokens through n_state_layers layers (one sequence per
#     question; the state is re-encoded for every question).
# workspace model, per example: state encoder once; per layer, the state keys/values once; then per
#     question and per loop (R = cfg.r_mean): question tokens (Lq + n_scratch) with self- and
#     cross-attention onto the state, their key/values for the option cross-attention, option tokens
#     To with self-attention inside each option (sum of Lo_i^2) and cross-attention onto
#     [state ; question], and the option-set layer on K + 2 tokens.

def _readout_flops(cfg: Rhen2Config, ls: float, k: float) -> float:
    d, dr = cfg.d_model, cfg.d_readout
    return 2.0 * d * dr * (k + 2) + 2.0 * ls * d * dr + 4.0 * ls * dr + 4.0 * d * dr


def _numeric_flops(cfg: Rhen2Config, tokens: float) -> float:
    d = cfg.d_model
    return 2.0 * tokens * (d + d * d / 4.0) if cfg.use_numeric else 0.0


def _flat_question_flops(cfg: Rhen2Config, ls: float, lq: float, k: float, opt_tokens: float, opt_sq: float = 0.0) -> float:
    d, h = cfg.d_model, cfg.ffn_hidden
    t = ls + lq + k + opt_tokens
    layer = 8.0 * t * d * d + 4.0 * t * t * d + 6.0 * t * d * h
    return (cfg.n_state_layers * layer + 2.0 * t * cfg.d_embed * d + _numeric_flops(cfg, ls + lq)
            + _readout_flops(cfg, ls, k))


def _workspace_question_flops(cfg: Rhen2Config, ls: float, lq: float, k: float, opt_tokens: float, opt_sq: float) -> float:
    d, h = cfg.d_model, cfg.ffn_hidden
    tq = lq + cfg.n_scratch
    per_layer = (
        8.0 * tq * d * d + 4.0 * tq * tq * d              # question self-attention
        + 4.0 * tq * d * d + 4.0 * tq * ls * d            # question cross-attention onto the state (q, o projections)
        + 6.0 * tq * d * h                                # question FFN
        + 4.0 * tq * d * d                                # key/values of the question for the options
        + 8.0 * opt_tokens * d * d + 4.0 * opt_sq * d     # option self-attention, inside each option
        + 4.0 * opt_tokens * d * d + 4.0 * opt_tokens * (ls + tq) * d   # option cross-attention onto [state ; question]
        + 6.0 * opt_tokens * d * h                        # option FFN
    )
    setl = (k + 2) * 8.0 * d * d + 4.0 * (k + 2) ** 2 * d + 6.0 * (k + 2) * d * h
    loop = cfg.n_core_layers * per_layer + (cfg.n_set_layers * setl if cfg.use_set_layer else 0.0)
    return cfg.r_mean * loop + 2.0 * (lq + opt_tokens) * cfg.d_embed * d + _numeric_flops(cfg, lq) + _readout_flops(cfg, ls, k) * cfg.r_mean


def _workspace_example_flops(cfg: Rhen2Config, ls: float) -> float:
    d, h = cfg.d_model, cfg.ffn_hidden
    enc = cfg.n_state_layers * (8.0 * ls * d * d + 4.0 * ls * ls * d + 6.0 * ls * d * h)
    kv = cfg.n_core_layers * 4.0 * ls * d * d              # state key/values, computed once and cached across loops
    return enc + kv + 2.0 * ls * cfg.d_embed * d + _numeric_flops(cfg, ls)


@torch.no_grad()
def flops_estimate(model: nn.Module, batch: Rhen2Batch) -> float:
    """Matmul FLOPs per example (batch average) for ``FlatBaselineModel`` or ``Rhen2Model``.

    Uses the real (unpadded) token counts of the batch, counts only valid questions, and for the
    workspace model assumes ``cfg.r_mean`` loops. See the formula notes above this function.
    """
    cfg = model.cfg
    flat = isinstance(model, FlatBaselineModel)
    ls = batch.state_mask.sum(-1).double()                                  # [B]
    lq = batch.q_mask.sum(-1).double()                                      # [B, M]
    k = batch.opt_valid.sum(-1).double()                                    # [B, M]
    lens = batch.opt_mask.sum(-1).double()                                  # [B, M, K]
    opt_tokens = lens.sum(-1)
    opt_sq = (lens ** 2).sum(-1)
    valid = batch.q_valid
    total = 0.0
    for bi in range(batch.batch_size):
        n_state = float(ls[bi])
        if not flat:
            total += _workspace_example_flops(cfg, n_state)
        for mi in range(batch.max_questions):
            if not bool(valid[bi, mi]):
                continue
            args = (cfg, n_state, float(lq[bi, mi]), float(k[bi, mi]), float(opt_tokens[bi, mi]), float(opt_sq[bi, mi]))
            total += _flat_question_flops(*args) if flat else _workspace_question_flops(*args)
    return total / max(batch.batch_size, 1)


__all__ = [
    "FlatBaselineModel", "matched_flat_config", "flops_estimate", "count_flat_parameters",
]
