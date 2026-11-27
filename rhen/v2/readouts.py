"""Typed readouts for RHEN-2. All of them consume pooled question / option / null vectors.

A "null" node (cannot tell) is part of every discrete question. Its score starts high
(``null_bias_init``), so an option has to accumulate evidence to beat it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhen.v2.config import NUM_QUANTILES, TYPE_INDEX, Rhen2Config
from rhen.v2.layers import RMSNorm

NEG = -1e9


@dataclass
class ReadoutOut:
    logits: torch.Tensor        # [N, K+1] choice-style scores, null last, padded options at NEG
    probs: torch.Tensor         # [N, K+1] choice / noul softmax over options + null (null last)
    score_probs: torch.Tensor   # [N, K+1] ordered levels through a cumulative link + null (null last)
    multi_logit: torch.Tensor   # [N, K]   score minus null score, for multi-label questions
    quantiles: torch.Tensor     # [N, 9]   range questions, normalised to [0, 1]
    halt_logit: torch.Tensor    # [N]

    def where(self, mask: torch.Tensor, other: "ReadoutOut") -> "ReadoutOut":
        """Rows where ``mask`` is True come from ``self``, the others from ``other``."""
        def pick(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.where(mask.view(-1, *([1] * (a.dim() - 1))), a, b)
        return ReadoutOut(*(pick(getattr(self, f), getattr(other, f)) for f in
                            ("logits", "probs", "score_probs", "multi_logit", "quantiles", "halt_logit")))


class Readout(nn.Module):
    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        d, dr = cfg.d_model, cfg.d_readout
        self.cfg = cfg
        self.wq = nn.Linear(d, dr, bias=False)
        self.wo = nn.Linear(d, dr, bias=False)
        self.wn = nn.Linear(d, dr, bias=False)
        self.nq, self.no, self.nn_ = RMSNorm(dr, cfg.norm_eps), RMSNorm(dr, cfg.norm_eps), RMSNorm(dr, cfg.norm_eps)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.null_bias = nn.Parameter(torch.tensor(float(cfg.null_bias_init)))
        # score: cumulative link
        self.thr = nn.Linear(d, 1)
        self.loc1 = nn.Linear(d, d // 2)
        self.loc2 = nn.Linear(d // 2, 1)
        # range: 10 bins -> 9 monotone quantiles
        self.rg1 = nn.Linear(d, dr)
        self.rg2 = nn.Linear(dr, NUM_QUANTILES + 1)
        # halting
        self.h1 = nn.Linear(d, d // 4)
        self.h2 = nn.Linear(d // 4, 1)
        # spans
        self.sq = nn.Linear(d, dr, bias=False)
        self.eq = nn.Linear(d, dr, bias=False)
        self.st = nn.Linear(d, dr, bias=False)
        self.sn = RMSNorm(dr, cfg.norm_eps)

    # -- discrete types ------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,             # [N, d]
        u: torch.Tensor,             # [N, K, d]
        null: torch.Tensor,          # [N, d]
        opt_valid: torch.Tensor,     # [N, K]
        temperature: Optional[torch.Tensor] = None,   # [N] or None
        null_delta: float = 0.0,
    ) -> ReadoutOut:
        scale = self.log_scale.exp()
        qn = self.nq(self.wq(q))
        # RMS-normalised vectors have a dot product that grows with their width, so divide by it.
        width = float(self.cfg.d_readout)
        s = scale * torch.einsum("nr,nkr->nk", qn, self.no(self.wo(u))) / width
        s_null = scale * (qn * self.nn_(self.wn(null))).sum(-1, keepdim=True) / width + self.null_bias + null_delta
        logits = torch.cat([s.masked_fill(~opt_valid, NEG), s_null], dim=-1)
        if not self.cfg.use_null:
            logits = torch.cat([logits[:, :-1], torch.full_like(s_null, NEG)], dim=-1)
        t = 1.0 if temperature is None else temperature.clamp_min(1e-3).unsqueeze(-1)
        choice_probs = (logits / t).softmax(dim=-1)

        # score: ordered levels through a cumulative link, null mass from the choice distribution
        k = u.shape[1]
        n_valid = opt_valid.sum(-1, keepdim=True)
        theta = F.softplus(self.thr(u).squeeze(-1)).cumsum(dim=-1)
        g = self.loc2(F.silu(self.loc1(q)))
        cdf = torch.sigmoid(theta - g)
        idx = torch.arange(k, device=u.device)[None]
        cdf = torch.where(idx >= n_valid - 1, torch.ones_like(cdf), cdf)
        levels = (cdf - F.pad(cdf[:, :-1], (1, 0))).clamp_min(0.0)
        p_null = choice_probs[:, -1:]
        score_probs = torch.cat([levels * (1.0 - p_null), p_null], dim=-1)

        # range: monotone quantiles inside [0, 1]
        bins = self.rg2(F.silu(self.rg1(q))).softmax(dim=-1)
        quantiles = bins.cumsum(dim=-1)[:, :NUM_QUANTILES]

        halt = self.h2(F.silu(self.h1(q))).squeeze(-1)
        return ReadoutOut(
            logits=logits,
            probs=choice_probs,
            score_probs=score_probs,
            multi_logit=(s - s_null).masked_fill(~opt_valid, NEG),
            quantiles=quantiles,
            halt_logit=halt,
        )

    # -- spans ---------------------------------------------------------------
    def project_state_for_spans(self, S: torch.Tensor) -> torch.Tensor:
        return self.sn(self.st(S))

    def span_logits(self, q: torch.Tensor, S_proj: torch.Tensor, state_mask: torch.Tensor, group: int):
        """``S_proj`` [G, Ls, dr] shared by ``group`` questions each; ``q`` [N, d] with N = G * group."""
        scale = 1.0 / math.sqrt(self.cfg.d_readout)
        g, ls, dr = S_proj.shape
        qs = self.sn(self.sq(q)).view(g, group, dr)
        qe = self.sn(self.eq(q)).view(g, group, dr)
        start = torch.einsum("gmr,glr->gml", qs, S_proj) * scale
        end = torch.einsum("gmr,glr->gml", qe, S_proj) * scale
        mask = state_mask[:, None, :]
        start = start.masked_fill(~mask, NEG).reshape(g * group, ls)
        end = end.masked_fill(~mask, NEG).reshape(g * group, ls)
        return start, end


def decode_span(start: torch.Tensor, end: torch.Tensor, max_span: int):
    """Best (i, j) with i <= j <= i + max_span maximising start_i + end_j. Position 0 is the null span."""
    n, length = start.shape
    best = torch.full((n,), -float("inf"), device=start.device)
    bi = torch.zeros(n, dtype=torch.long, device=start.device)
    bj = torch.zeros(n, dtype=torch.long, device=start.device)
    for off in range(max_span + 1):
        if off >= length:
            break
        sc = start[:, : length - off] + end[:, off:]
        if off > 0:
            sc = sc.clone()
            sc[:, 0] = -float("inf")     # a real span never starts on the null position
        val, arg = sc.max(dim=-1)
        better = val > best
        best = torch.where(better, val, best)
        bi = torch.where(better, arg, bi)
        bj = torch.where(better, arg + off, bj)
    return bi, bj, best


SCORE = TYPE_INDEX["score"]
