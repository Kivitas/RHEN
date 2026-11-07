"""
Hybrid Muon + AdamW optimization for EOS.

Parameter partitioning:
  - Group 1 (Muon):     2D weight matrices from Linear layers (attention projections,
                        SwiGLU/MoE FFN layers, router). Updates are orthogonalized via a
                        quintic Newton-Schulz iteration, giving near-unit singular values
                        => every direction of the matrix learns at the same effective rate.
  - Group 2 (AdamW):    everything else -- embeddings (incl. tied LM head & LoRA step
                        adapters), RMSNorm gains, FiLM tables, biases, scalar gates.

Step order per training step (strict coupling):
  1. loss.backward()                     (raw gradients)
  2. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)   # GLOBAL, both groups
  3. muon.step()
  4. adamw.step()

Reference: Jordan, "Muon: MomentUm Orthogonalized by Newton-Schulz" (2024);
Moonlight/muP-style RMS scaling keeps Muon LRs comparable to AdamW's.
"""

import torch
from typing import List, Tuple


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximates the orthogonalization UV^T of G via a quintic Newton-Schulz
    iteration (B = bA + cA^2 polynomial on X X^T). Runs in bf16 on CUDA for speed;
    converges to unit singular values within a few iterations."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


def muon_update_scale(rows: int, cols: int) -> float:
    """RMS scale applied to a Newton-Schulz-orthogonalized Muon update.

    Matches the reference Muon implementation: tall matrices (rows > cols) get
    sqrt(rows / cols); wide matrices (rows <= cols) get 1.0.
    """
    return max(1.0, rows / cols) ** 0.5


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalized updates for 2D matrices."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Muon lr must be positive, got {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            nesterov, ns_steps, wd = group["nesterov"], group["ns_steps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Correct heavy-ball momentum: buf = momentum*buf + g
                buf.mul_(momentum).add_(g)
                # Nesterov look-ahead: update = g + momentum*buf
                update = g.add(buf, alpha=momentum) if nesterov else buf

                # Orthogonalize the momentum update, then apply the reference
                # Muon scale (tall matrices only; wide matrices are unscaled).
                u = zeropower_via_newtonschulz5(update, steps=ns_steps)
                alpha = lr * muon_update_scale(p.size(-2), p.size(-1))

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled weight decay
                p.add_(u.to(p.dtype).view_as(p), alpha=-alpha)

        return loss


def split_parameters(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Partitions trainable parameters into (muon_2d, adamw_decayed, adamw_no_decay).

    Muon gets only true Linear 2D matrices. Embedding tables (token embeddings,
    tied LM head, LoRA step adapters) always stay with AdamW regardless of rank,
    together with norms/biases/scalars (no weight decay).
    """
    embedding_param_ids = set()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            embedding_param_ids.update(id(p) for p in m.parameters(recurse=False))

    muon_params: List[torch.nn.Parameter] = []
    adamw_decay: List[torch.nn.Parameter] = []
    adamw_no_decay: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and id(p) not in embedding_param_ids:
            muon_params.append(p)
        elif p.ndim < 2 or name.endswith(".bias"):
            adamw_no_decay.append(p)
        else:
            adamw_decay.append(p)

    return muon_params, adamw_decay, adamw_no_decay


def build_muon_adamw(
    model: torch.nn.Module,
    adamw_lr: float,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    muon_weight_decay: float = 0.0,
    adamw_betas=(0.9, 0.95),
    adamw_weight_decay: float = 0.1,
) -> Tuple[Muon, torch.optim.AdamW, dict]:
    """Builds the (Muon, AdamW) pair. Returns stats for logging."""
    muon_params, adamw_decay, adamw_no_decay = split_parameters(model)

    fused_ok = next(model.parameters()).is_cuda
    try:
        adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": adamw_weight_decay},
                {"params": adamw_no_decay, "weight_decay": 0.0},
            ],
            lr=adamw_lr,
            betas=adamw_betas,
            fused=fused_ok,
        )
    except (TypeError, RuntimeError):
        adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": adamw_weight_decay},
                {"params": adamw_no_decay, "weight_decay": 0.0},
            ],
            lr=adamw_lr,
            betas=adamw_betas,
        )

    muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                nesterov=True, ns_steps=muon_ns_steps, weight_decay=muon_weight_decay)

    stats = {
        "muon_params": sum(p.numel() for p in muon_params),
        "muon_tensors": len(muon_params),
        "adamw_params": sum(p.numel() for p in adamw_decay) + sum(p.numel() for p in adamw_no_decay),
        "adamw_tensors": len(adamw_decay) + len(adamw_no_decay),
    }
    return muon, adamw, stats
