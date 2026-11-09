"""Backward-compatibility alias: ``eos_research`` is the rhen.research package.

The canonical package replaces this one in sys.modules, so ``import eos_research``
returns the same object. Submodules are registered under the old names first;
otherwise Python would execute the canonical files a second time under the old
name and produce a different module object.
"""
import importlib
import sys

_pkg = importlib.import_module("rhen.research")
for _name in ['calibration', 'checkpointing', 'data', 'device', 'distributed', 'evaluate', 'manifest', 'metrics', 'prepare', 'presets', 'sweep', 'tasks', 'train']:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"rhen.research.{_name}")
sys.modules[__name__] = _pkg
