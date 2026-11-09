"""Backward-compatibility alias: ``optim`` is the rhen.optim package.

The canonical package replaces this one in sys.modules, so ``import optim``
returns the same object. Submodules are registered under the old names first;
otherwise Python would execute the canonical files a second time under the old
name and produce a different module object.
"""
import importlib
import sys

_pkg = importlib.import_module("rhen.optim")
for _name in ['advanced_loops', 'checkpoint', 'fused_swiglu', 'grouped_gemm', 'mup', 'quantized_kv', 'rhen_mask', 'ring_mla']:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"rhen.optim.{_name}")
sys.modules[__name__] = _pkg
