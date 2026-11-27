"""State encoder: a bidirectional transformer that reads the state once.

Its output depends only on the state tokens, so it can be cached by a hash of the state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rhen.v2.config import Rhen2Config
from rhen.v2.layers import EncoderBlock, RMSNorm, Rotary


class StateEncoder(nn.Module):
    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.ffn_hidden, cfg.norm_eps) for _ in range(cfg.n_state_layers)
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.rotary = Rotary(cfg.head_dim, cfg.rope_theta)

    def is_global(self, layer: int) -> bool:
        every = self.cfg.global_every
        return every == 0 or (layer + 1) % every == 0

    def _local_mask(self, length: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        near = (idx[:, None] - idx[None, :]).abs() <= self.cfg.local_window // 2
        near[:, 0] = True            # position 0 is always real, so padded rows never lose every key
        return near[None, None]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` [B, Ls, d], ``mask`` [B, Ls] bool -> [B, Ls, d]."""
        length = x.shape[1]
        rope = self.rotary(length, x.device, x.dtype)
        local = None
        for i, block in enumerate(self.blocks):
            if self.is_global(i):
                x = block(x, key_mask=mask, rope=rope)
            else:
                if local is None:
                    local = self._local_mask(length, x.device)
                x = block(x, key_mask=mask, attn_mask=local, rope=rope)
        return self.norm(x)
