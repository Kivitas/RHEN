"""Exponential moving average of model weights with warmup decay ``min(decay, (1 + n) / (10 + n))``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def current_decay(self) -> float:
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        self.num_updates += 1
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().to(s.dtype), alpha=1.0 - d)
            else:
                s.copy_(v)

    @contextmanager
    def swap_into(self, model: nn.Module):
        """Evaluate with the EMA weights; the training weights are restored on exit, even on error."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
