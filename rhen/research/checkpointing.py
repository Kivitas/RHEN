"""Training checkpoints with exact resume.

A checkpoint directory ``<run_dir>/checkpoints/step_00001000/`` holds:

- ``model.pt``: the model state dict (canonical key layout, gathered on rank 0 under FSDP);
- ``model_config.json``: the ``EOSConfig`` needed to rebuild the model;
- ``optimizers.pt``: optimizer, scheduler and grad-scaler state;
- ``rng_rank<k>.pt``: every random generator of rank ``k``;
- ``trainer_state.json``: the step and the training configuration.

``<run_dir>/checkpoints/latest`` names the newest complete checkpoint. It is written last,
so an interrupted save never becomes the resume point.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from typing import Any, Dict, List, Optional

import torch

from rhen.research import distributed as distmod
from rhen.research.manifest import write_json

LATEST = "latest"

logger = logging.getLogger(__name__)


def checkpoint_dir(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, "checkpoints", f"step_{step:08d}")


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    for name in ("cuda", "npu", "xpu"):
        module = getattr(torch, name, None)
        if module is not None and hasattr(module, "is_available") and module.is_available():
            state[name] = module.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    for name in ("cuda", "npu", "xpu"):
        module = getattr(torch, name, None)
        if name in state and module is not None and module.is_available():
            module.set_rng_state_all(state[name])


def model_state_dict(wrapped: torch.nn.Module, model: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
    """Full state dict of ``model`` (rank 0 only under FSDP; None on other ranks)."""
    if distmod.is_fsdp(wrapped):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, cfg):
            full = wrapped.state_dict()
        prefix = "model."
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in full.items()} if full else None
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_checkpoint(
    run_dir: str,
    step: int,
    wrapped: torch.nn.Module,
    model: torch.nn.Module,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[Any],
    scaler: Any,
    trainer_state: Dict[str, Any],
    ctx: distmod.DistributedContext,
    keep_last: int = 3,
) -> str:
    path = checkpoint_dir(run_dir, step)
    if ctx.is_main:
        os.makedirs(path, exist_ok=True)
    distmod.barrier(ctx)

    state = model_state_dict(wrapped, model)
    if distmod.is_fsdp(wrapped):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        optim_states = [FSDP.optim_state_dict(wrapped, opt) for opt in optimizers]
    else:
        optim_states = [opt.state_dict() for opt in optimizers]

    if ctx.is_main:
        torch.save(state, os.path.join(path, "model.pt"))
        torch.save(
            {
                "optimizers": optim_states,
                "schedulers": [s.state_dict() for s in schedulers],
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            os.path.join(path, "optimizers.pt"),
        )
        write_json(os.path.join(path, "model_config.json"), model.config.to_dict())
        write_json(os.path.join(path, "trainer_state.json"), dict(trainer_state, step=step))
    torch.save(_rng_state(), os.path.join(path, f"rng_rank{ctx.rank}.pt"))
    distmod.barrier(ctx)

    if ctx.is_main:
        with open(os.path.join(run_dir, "checkpoints", LATEST + ".tmp"), "w", encoding="utf-8") as fh:
            fh.write(os.path.basename(path))
        os.replace(os.path.join(run_dir, "checkpoints", LATEST + ".tmp"), os.path.join(run_dir, "checkpoints", LATEST))
        _prune(run_dir, keep_last)
    distmod.barrier(ctx)
    return path


def _prune(run_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    root = os.path.join(run_dir, "checkpoints")
    steps = sorted(d for d in os.listdir(root) if d.startswith("step_"))
    for old in steps[:-keep_last]:
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)


def latest_checkpoint(run_dir: str) -> Optional[str]:
    pointer = os.path.join(run_dir, "checkpoints", LATEST)
    if not os.path.exists(pointer):
        return None
    with open(pointer, "r", encoding="utf-8") as fh:
        name = fh.read().strip()
    path = os.path.join(run_dir, "checkpoints", name)
    return path if os.path.isdir(path) else None


def load_model_weights(model: torch.nn.Module, path: str, strict: bool = True) -> None:
    state = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=strict)


def load_checkpoint(
    path: str,
    wrapped: torch.nn.Module,
    model: torch.nn.Module,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[Any],
    scaler: Any,
    ctx: distmod.DistributedContext,
) -> Dict[str, Any]:
    """Restore everything saved by ``save_checkpoint``; returns the trainer state.

    Loading a checkpoint runs deserialization, so checkpoints must come from a trusted
    source: a crafted ``.pt`` file can execute code when it is loaded. ``model.pt`` and
    ``optimizers.pt`` are read with ``weights_only=True``. ``rng_rank<k>.pt`` is read with
    ``weights_only=False`` because it holds a NumPy RNG state, and a WARNING is logged
    each time that happens.
    """
    if distmod.is_fsdp(wrapped):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT, cfg):
            wrapped.load_state_dict({"model." + k: v for k, v in state.items()})
    else:
        load_model_weights(model, path)

    extra = torch.load(os.path.join(path, "optimizers.pt"), map_location="cpu", weights_only=True)
    for opt, opt_state in zip(optimizers, extra["optimizers"]):
        if distmod.is_fsdp(wrapped):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            opt_state = FSDP.optim_state_dict_to_load(wrapped, opt, opt_state)
        opt.load_state_dict(opt_state)
    for sched, sched_state in zip(schedulers, extra["schedulers"]):
        sched.load_state_dict(sched_state)
    if scaler is not None and extra.get("scaler") is not None:
        scaler.load_state_dict(extra["scaler"])

    rng_path = os.path.join(path, f"rng_rank{ctx.rank}.pt")
    if os.path.exists(rng_path):
        logger.warning(
            "Loading %s with weights_only=False (NumPy RNG state needs full unpickling); "
            "only resume from checkpoints produced by a trusted run",
            rng_path,
        )
        _set_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    with open(os.path.join(path, "trainer_state.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_from_checkpoint(path: str, device: Optional[torch.device] = None):
    """Rebuild an ``EOSModel`` from a checkpoint directory (for evaluation and inference)."""
    from config import EOSConfig
    from model import EOSModel

    with open(os.path.join(path, "model_config.json"), "r", encoding="utf-8") as fh:
        cfg_dict = json.load(fh)
    for key in ("model_type", "transformers_version", "architectures"):
        cfg_dict.pop(key, None)
    config = EOSConfig(**cfg_dict)
    model = EOSModel(config)
    load_model_weights(model, path)
    if device is not None:
        model = model.to(device)
    return model.eval()
