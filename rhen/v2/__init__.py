"""RHEN-2.  Public names are exported lazily (PEP 562) so ``import rhen.v2`` stays cheap."""

from __future__ import annotations

import importlib
from typing import Any, List

_LAZY = {
    "DecisionPipeline": "rhen.v2.pipeline",
    "Rhen2Model": "rhen.v2.model",
    "Rhen2Config": "rhen.v2.config",
    "V2Backend": "rhen.v2.compat",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY))
