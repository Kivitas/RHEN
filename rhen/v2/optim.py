"""Optimizer groups and schedule (plan section 5.1).

* no weight decay on norms, biases, embeddings, scalars and null / scratch / rank / mask vectors;
* hidden matrices (state encoder, core, set layer) get ``lr * base_width / d_model`` (muP style);
* embedding / input projections, readouts and pooling heads keep the base lr.

Each group carries ``lr_mult`` so a scheduler can set ``group["lr"] = peak_lr * group["lr_mult"]``
(see ``set_lr``).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

_HIDDEN_PREFIXES = ("encoder.blocks.", "workspace.stack.", "workspace.set_layer.blocks.")


def _no_decay(name: str, p: nn.Parameter, embedding_params: set) -> bool:
    if p.ndim < 2 or name in embedding_params:
        return True
    return any(t in name for t in ("norm", "null", "scratch", "rank", "mask_memory"))


def build_param_groups(model: nn.Module, lr: float, weight_decay: float = 0.05, base_width: int = 512) -> List[Dict]:
    d_model = model.cfg.d_model
    emb = {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, nn.Embedding)}
    buckets: Dict[str, List[nn.Parameter]] = {"hidden_decay": [], "input_decay": [], "head_decay": [], "no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _no_decay(name, p, emb):
            buckets["no_decay"].append(p)
        elif name.startswith(_HIDDEN_PREFIXES):
            buckets["hidden_decay"].append(p)
        elif name.startswith(("embed.", "numeric.")):
            buckets["input_decay"].append(p)
        else:
            buckets["head_decay"].append(p)
    mult = {"hidden_decay": base_width / d_model}
    groups = []
    for key, params in buckets.items():
        if not params:
            continue
        m = mult.get(key, 1.0)
        groups.append({"name": key, "params": params, "lr": lr * m, "lr_mult": m,
                       "weight_decay": 0.0 if key == "no_decay" else weight_decay})
    return groups


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.02, min_ratio: float = 0.1) -> float:
    """Linear warmup to ``peak``, then cosine decay to ``peak * min_ratio`` at ``step == total``."""
    warm = max(1, int(round(total * warmup_frac)))
    if step < warm:
        return peak * (step + 1) / warm
    t = min(1.0, (step - warm) / max(1, total - warm))
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)
