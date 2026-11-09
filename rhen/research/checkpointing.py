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

import base64
import json
import logging
import os
import pickle
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


#: Marker for the plain-type RNG format written by ``_rng_state``.
RNG_FORMAT = "rhen-rng-v2"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _rng_state() -> Dict[str, Any]:
    """
    Every generator's state as plain Python types (dict, list, int, float, str, None).

    Nothing here needs pickle globals, so the file loads under ``weights_only=True``.
    Byte-valued generator states (torch and accelerator RNGs) are stored as base64.
    """
    py_version, py_internal, py_gauss = random.getstate()
    state: Dict[str, Any] = {
        "format": RNG_FORMAT,
        "python": {"version": int(py_version), "internal": [int(x) for x in py_internal],
                   "gauss_next": py_gauss},
        "torch": _b64(bytes(torch.get_rng_state().tolist())),
    }
    try:
        import numpy as np

        name, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = {
            "bit_generator": str(name),
            "keys": [int(x) for x in keys],
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }
    except ImportError:
        pass
    for name in ("cuda", "npu", "xpu"):
        module = getattr(torch, name, None)
        if module is not None and hasattr(module, "is_available") and module.is_available():
            state[name] = [_b64(bytes(t.tolist())) for t in module.get_rng_state_all()]
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    """Restore a state written by ``_rng_state`` (or the legacy object format)."""
    if state.get("format") != RNG_FORMAT:
        _set_rng_state_legacy(state)
        return
    py = state["python"]
    random.setstate((py["version"], tuple(py["internal"]), py["gauss_next"]))
    torch.set_rng_state(torch.frombuffer(bytearray(_unb64(state["torch"])), dtype=torch.uint8).clone())
    if "numpy" in state:
        import numpy as np

        npy = state["numpy"]
        np.random.set_state((
            npy["bit_generator"],
            np.array(npy["keys"], dtype=np.uint32),
            npy["pos"],
            npy["has_gauss"],
            npy["cached_gaussian"],
        ))
    for name in ("cuda", "npu", "xpu"):
        module = getattr(torch, name, None)
        if name in state and module is not None and module.is_available():
            module.set_rng_state_all([
                torch.frombuffer(bytearray(_unb64(s)), dtype=torch.uint8).clone() for s in state[name]
            ])


def _set_rng_state_legacy(state: Dict[str, Any]) -> None:
    """Restore the pre-v2 layout, which stored Python and NumPy state objects directly."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    for name in ("cuda", "npu", "xpu"):
        module = getattr(torch, name, None)
        if name in state and module is not None and module.is_available():
            module.set_rng_state_all(state[name])


def _load_rng_file(rng_path: str, allow_unsafe_legacy_rng: bool = False) -> Dict[str, Any]:
    """
    Read an ``rng_rank<k>.pt`` file with ``weights_only=True``.

    Files in the pre-v2 layout hold NumPy objects, which ``weights_only=True`` refuses. Those
    are NOT loaded by default: loading them with ``weights_only=False`` executes code from a
    crafted file. Pass ``allow_unsafe_legacy_rng=True`` to load one, and only do so for a
    checkpoint from a trusted source. The load logs a WARNING when it takes that path.
    """
    try:
        return torch.load(rng_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if not allow_unsafe_legacy_rng:
            raise RuntimeError(
                f"RNG state file {rng_path} is in the old checkpoint layout and cannot be read "
                "with weights_only=True. Loading it executes code from the file. Pass "
                "allow_unsafe_legacy_rng=True to load_checkpoint only if the checkpoint comes "
                "from a trusted source, or re-save it with the current format."
            ) from exc
        logger.warning(
            "Loading legacy RNG state %s with weights_only=False because "
            "allow_unsafe_legacy_rng=True. This executes code from the file; only resume from a "
            "trusted source.",
            rng_path,
        )
        return torch.load(rng_path, map_location="cpu", weights_only=False)


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
    allow_unsafe_legacy_rng: bool = False,
) -> Dict[str, Any]:
    """Restore everything saved by ``save_checkpoint``; returns the trainer state.

    Loading a checkpoint runs deserialization, so checkpoints must come from a trusted
    source: a crafted ``.pt`` file can execute code when it is loaded. ``model.pt`` and
    ``optimizers.pt`` and ``rng_rank<k>.pt`` are read with ``weights_only=True``. The RNG
    file is written in plain types since format v2. An RNG file in the old layout is refused
    unless ``allow_unsafe_legacy_rng=True``; see ``_load_rng_file``.
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
        _set_rng_state(_load_rng_file(rng_path, allow_unsafe_legacy_rng=allow_unsafe_legacy_rng))
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
