"""Dataset converters. Every module in this package registers its converters on import.

``load_all()`` imports every sibling module, so adding a dataset means adding one file that
calls ``register(DatasetConverter(...))``; nothing else needs editing.
"""

from __future__ import annotations

import importlib
import pkgutil

from rhen.v2.data.converters.base import REGISTRY, DatasetConverter, convert_rows, register


def load_all() -> dict:
    for info in pkgutil.iter_modules(__path__):
        if info.name != "base":
            importlib.import_module(f"{__name__}.{info.name}")
    return REGISTRY


__all__ = ["REGISTRY", "DatasetConverter", "convert_rows", "register", "load_all"]
