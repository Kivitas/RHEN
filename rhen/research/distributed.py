"""Single- and multi-process execution: torchrun/SLURM environments, DDP and FSDP.

Everything degrades to a no-op on one process, so the same training script runs on a laptop
CPU, one GPU or NPU, or a multi-node cluster. The collective backend follows the device
(NCCL for CUDA/ROCm, HCCL for Ascend NPUs, XCCL for Intel XPUs, Gloo for CPU and MPS).
"""

from __future__ import annotations

import datetime
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from eos_research import device as devmod

STRATEGIES = ("none", "ddp", "fsdp")


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    backend: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def init_distributed(
    device_spec: str = "auto",
    backend: Optional[str] = None,
    timeout_minutes: int = 30,
) -> DistributedContext:
    """Read the launcher environment (torchrun or SLURM) and start the process group if needed."""
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    rank = _env_int("RANK", "SLURM_PROCID", default=0)
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)

    device = devmod.resolve_device(device_spec, local_rank=local_rank)
    devmod.set_current_device(device)
    ctx = DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, device=device)
    if world_size > 1:
        ctx.backend = backend or devmod.collective_backend(device)
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            dist.init_process_group(
                backend=ctx.backend,
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(minutes=timeout_minutes),
            )
    return ctx


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    """Mean of ``tensor`` across ranks (a copy; the input is not modified)."""
    if not ctx.enabled:
        return tensor.clone()
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / ctx.world_size


def broadcast_object(obj: Any, ctx: DistributedContext, src: int = 0) -> Any:
    if not ctx.enabled:
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def wrap_model(module: nn.Module, ctx: DistributedContext, strategy: str = "ddp") -> nn.Module:
    """Wrap ``module`` for data-parallel training; returns it unchanged on one process.

    DDP is built with ``broadcast_buffers=False`` because the MoE step-mode load counters are
    per-rank buffers that the trainer averages explicitly, and with
    ``find_unused_parameters=True`` because routed experts and decision-only heads do not
    receive gradients on every step. FSDP shards every ``TransformerLayer``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy == "none" or not ctx.enabled:
        return module
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [ctx.device.index] if ctx.device.type in ("cuda", "npu", "xpu") else None
        return DDP(module, device_ids=device_ids, broadcast_buffers=False, find_unused_parameters=True)

    import functools

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from model import TransformerLayer

    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerLayer})
    return FSDP(module, auto_wrap_policy=policy, device_id=ctx.device, use_orig_params=True)


def unwrap(module: nn.Module) -> nn.Module:
    """The model inside DDP/FSDP/torch.compile wrappers."""
    while True:
        inner = getattr(module, "module", None) or getattr(module, "_orig_mod", None)
        if inner is None or inner is module:
            return module
        module = inner


def no_sync(module: nn.Module, active: bool):
    """Skip gradient all-reduce on non-final gradient-accumulation micro-steps (DDP and FSDP)."""
    if active and hasattr(module, "no_sync"):
        return module.no_sync()
    return nullcontext()


def is_fsdp(module: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return False
    return isinstance(module, FSDP)
