"""Building blocks for RHEN-2: norms, rotary positions, attention, SwiGLU, embeddings.

Every attention call takes boolean masks where True means "may attend". A query row must
have at least one True key; callers guarantee it (position 0 of every group is always real).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    """Plain RoPE. Positions are ``arange(length)`` within each token group."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, length: int, device: torch.device, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        # Always float32: bf16 cos/sin lose position information beyond a few hundred tokens.
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` is [N, H, L, hd]; ``cos``/``sin`` are [L, hd]. Computed in float32, returned in ``x.dtype``."""
    xf = x.float()
    return (xf * cos.float() + _rotate_half(xf) * sin.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Multi-head attention with QK-norm. Used for self-attention and for cross-attention."""

    def __init__(self, d: int, heads: int, eps: float = 1e-6):
        super().__init__()
        self.d, self.heads, self.hd = d, heads, d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.hd, eps)
        self.k_norm = RMSNorm(self.hd, eps)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        n, l, _ = x.shape
        return x.view(n, l, self.heads, self.hd).transpose(1, 2)

    def project_kv(self, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[N, L, d] -> k, v of shape [N, H, L, hd] (k is QK-normed)."""
        return self.k_norm(self._split(self.k_proj(mem))), self._split(self.v_proj(mem))

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_norm(self._split(self.q_proj(x)))

    @staticmethod
    def _combine(key_mask: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :]
        if attn_mask is not None:
            mask = attn_mask if mask is None else (mask & attn_mask)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        per_group: int = 1,
    ) -> torch.Tensor:
        """Self-attention when ``kv`` is None, otherwise cross-attention onto ``kv``.

        ``per_group > 1`` (cross-attention only) folds ``per_group`` consecutive query rows into
        one row of ``per_group * L`` queries that share one memory, so the memory is neither
        copied nor re-projected for each of them.
        """
        n, l, _ = x.shape
        q = self._project_q(x)
        if kv is None:
            k, v = self.project_kv(x)
            if rope is not None:
                cos, sin = rope
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            mask = self._combine(key_mask, attn_mask)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return self.o_proj(out.transpose(1, 2).reshape(n, l, self.d))
        k, v = kv
        if per_group > 1:
            g = n // per_group
            q = q.view(g, per_group, self.heads, l, self.hd).permute(0, 2, 1, 3, 4).reshape(g, self.heads, per_group * l, self.hd)
        mask = self._combine(key_mask, attn_mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if per_group > 1:
            g = n // per_group
            out = out.view(g, self.heads, per_group, l, self.hd).permute(0, 2, 3, 1, 4).reshape(n, l, self.d)
            return self.o_proj(out)
        return self.o_proj(out.transpose(1, 2).reshape(n, l, self.d))

    @torch.no_grad()
    def attention_mass(
        self,
        x: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        key_mask: torch.Tensor,
        query_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention weights averaged over heads and over valid query tokens: [N, Lk]."""
        q = self._project_q(x)
        k, _ = kv
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.hd)
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        weights = scores.softmax(dim=-1).mean(dim=1)                      # [N, Lq, Lk]
        w = query_weight.float().unsqueeze(-1)
        return (weights * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


class EncoderBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU block (state encoder and option-set layer)."""

    def __init__(self, d: int, heads: int, ffn_hidden: int, eps: float = 1e-6):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d, eps), RMSNorm(d, eps)
        self.attn = Attention(d, heads, eps)
        self.ffn = SwiGLU(d, ffn_hidden)

    def forward(self, x, key_mask=None, attn_mask=None, rope=None):
        x = x + self.attn(self.n1(x), key_mask=key_mask, attn_mask=attn_mask, rope=rope)
        return x + self.ffn(self.n2(x))


class FactorisedEmbedding(nn.Module):
    def __init__(self, vocab: int, d_embed: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(vocab, d_embed)
        self.up = nn.Linear(d_embed, d_model, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.up(self.table(ids))


class NumericChannel(nn.Module):
    """Maps 4 numeric features per token (zeros for non-number tokens) into the embedding space."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(4, d_model // 4, bias=False)
        self.fc2 = nn.Linear(d_model // 4, d_model, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(feats)))


class AttnPool(nn.Module):
    """Learned attention pooling over the token axis. ``x`` [N, L, d], ``mask`` [N, L]."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1).float().masked_fill(~mask, float("-inf"))
        weights = logits.softmax(dim=-1).to(x.dtype)
        return torch.einsum("nl,nld->nd", weights, x)
