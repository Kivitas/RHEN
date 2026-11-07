"""Device, precision and reproducibility helpers for every supported accelerator.

One code path serves CPU, NVIDIA/AMD GPUs (``cuda``, which also covers ROCm builds), Ascend
NPUs (``npu``, through the ``torch_npu`` plug-in), Intel GPUs (``xpu``) and Apple silicon
(``mps``). Callers ask for ``"auto"`` or a device type, and get back a concrete
``torch.device`` plus a precision policy that the device actually supports.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

DEVICE_TYPES = ("cuda", "npu", "xpu", "mps", "cpu")
# Preference order for "auto": the fastest accelerator that is present wins.
AUTO_ORDER = ("cuda", "npu", "xpu", "mps", "cpu")
PRECISIONS = ("auto", "fp32", "bf16", "fp16")

_COLLECTIVE_BACKENDS = {"cuda": "nccl", "npu": "hccl", "xpu": "xccl", "mps": "gloo", "cpu": "gloo"}


def _import_torch_npu() -> bool:
    """Ascend support lives in the separate ``torch_npu`` package; import it on demand."""
    if hasattr(torch, "npu"):
        return True
    try:
        importlib.import_module("torch_npu")  # importing it registers torch.npu
    except ImportError:
        return False
    return hasattr(torch, "npu")


def is_available(device_type: str) -> bool:
    if device_type == "cpu":
        return True
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "npu":
        return _import_torch_npu() and torch.npu.is_available()
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    raise ValueError(f"unknown device type {device_type!r}; expected one of {DEVICE_TYPES}")


def available_device_types() -> List[str]:
    return [d for d in AUTO_ORDER if is_available(d)]


def _device_module(device_type: str):
    """The ``torch.<type>`` namespace (``torch.cuda``, ``torch.npu``, ...) or None for CPU."""
    if device_type == "cpu":
        return None
    if device_type == "npu":
        _import_torch_npu()
    return getattr(torch, device_type, None)


def device_count(device_type: str) -> int:
    if device_type in ("cpu", "mps"):
        return 1 if is_available(device_type) else 0
    module = _device_module(device_type)
    return int(module.device_count()) if module is not None and is_available(device_type) else 0


def resolve_device(spec: str = "auto", local_rank: int = 0) -> torch.device:
    """Turn ``"auto"``, a device type (``"npu"``) or an explicit device (``"cuda:1"``) into a device.

    With a bare accelerator type, ``local_rank`` selects the device index, which is what a
    ``torchrun`` worker needs. Asking for an accelerator that is not present raises instead of
    silently falling back to the CPU.
    """
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        spec = next(d for d in AUTO_ORDER if is_available(d))
    device_type, _, index = spec.partition(":")
    if device_type not in DEVICE_TYPES:
        raise ValueError(f"unknown device {spec!r}; expected 'auto' or one of {DEVICE_TYPES}")
    if not is_available(device_type):
        hint = " (install torch_npu for Ascend NPUs)" if device_type == "npu" else ""
        raise RuntimeError(
            f"device type {device_type!r} is not available here{hint}; "
            f"available: {available_device_types()}"
        )
    if device_type in ("cpu", "mps"):
        return torch.device(device_type)
    idx = int(index) if index else int(local_rank)
    count = device_count(device_type)
    if idx >= count:
        raise RuntimeError(f"{device_type}:{idx} requested but only {count} device(s) are visible")
    return torch.device(device_type, idx)


def set_current_device(device: torch.device) -> None:
    module = _device_module(device.type)
    if module is not None and device.index is not None and hasattr(module, "set_device"):
        module.set_device(device)


def bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type in ("npu", "xpu", "cpu"):
        return True
    if device.type == "mps":
        # bf16 kernels on MPS need macOS 14+ and a recent torch; be conservative.
        return False
    return False


@dataclass(frozen=True)
class PrecisionPolicy:
    """How training runs numerically on a device.

    Parameters and optimizer state stay in float32; ``autocast_dtype`` (when set) is the
    compute dtype inside ``torch.autocast``. ``use_grad_scaler`` is True only for float16,
    whose narrow exponent range needs loss scaling.
    """

    name: str
    autocast_dtype: Optional[torch.dtype]
    use_grad_scaler: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "autocast_dtype": str(self.autocast_dtype) if self.autocast_dtype is not None else None,
            "use_grad_scaler": self.use_grad_scaler,
        }


def resolve_precision(precision: str, device: torch.device) -> PrecisionPolicy:
    """``"auto"`` picks bf16 on accelerators that support it and fp32 elsewhere."""
    precision = (precision or "auto").lower()
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "auto":
        precision = "bf16" if device.type in ("cuda", "npu", "xpu") and bf16_supported(device) else "fp32"
    if precision == "fp32":
        return PrecisionPolicy("fp32", None, False)
    if precision == "bf16":
        if not bf16_supported(device):
            raise RuntimeError(f"bf16 is not supported on {device}; use fp16 or fp32")
        return PrecisionPolicy("bf16", torch.bfloat16, False)
    if device.type == "cpu":
        raise RuntimeError("fp16 autocast is not supported for training on CPU; use bf16 or fp32")
    return PrecisionPolicy("fp16", torch.float16, True)


def autocast(device: torch.device, policy: PrecisionPolicy):
    """``torch.autocast`` for the device, or a no-op context for fp32."""
    if policy.autocast_dtype is None:
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=device.type, dtype=policy.autocast_dtype)
    except RuntimeError as exc:  # e.g. an older torch without autocast for this device
        warnings.warn(f"autocast is unavailable on {device} ({exc}); running in fp32", RuntimeWarning)
        return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, policy: PrecisionPolicy):
    """A gradient scaler for fp16, or None. Uses the device-generic ``torch.amp`` API."""
    if not policy.use_grad_scaler:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type)
        except (TypeError, RuntimeError):
            pass
    if device.type == "npu" and _import_torch_npu():
        return torch.npu.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def collective_backend(device: torch.device) -> str:
    """The ``torch.distributed`` backend that matches the device (HCCL on Ascend, XCCL on Intel)."""
    return _COLLECTIVE_BACKENDS[device.type]


def synchronize(device: torch.device) -> None:
    module = _device_module(device.type)
    if module is not None and hasattr(module, "synchronize"):
        module.synchronize()


def memory_stats(device: torch.device) -> Dict[str, float]:
    """Peak and current allocated memory in GiB, when the device reports it."""
    module = _device_module(device.type)
    if module is None or not hasattr(module, "max_memory_allocated"):
        return {}
    gib = float(1 << 30)
    try:
        return {
            "mem_allocated_gib": module.memory_allocated(device) / gib,
            "mem_peak_gib": module.max_memory_allocated(device) / gib,
        }
    except (RuntimeError, TypeError):
        return {}


def seed_everything(seed: int, rank: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy (if present) and torch on every device; returns the per-rank seed.

    ``deterministic=True`` asks for deterministic kernels (with warnings where none exist),
    which makes runs repeatable on one machine at some speed cost.
    """
    rank_seed = int(seed) + 1000 * int(rank)
    random.seed(rank_seed)
    try:
        import numpy as np

        np.random.seed(rank_seed % (2 ** 32))
    except ImportError:
        pass
    torch.manual_seed(rank_seed)  # also seeds CUDA, NPU, XPU and MPS generators
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return rank_seed


def describe_environment(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Versions and hardware, recorded in every run manifest."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "available_devices": available_device_types(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda or torch.version.hip
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if "npu" in info["available_devices"]:
        info["npu_devices"] = device_count("npu")
    if device is not None:
        info["device"] = str(device)
    return info
