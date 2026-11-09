"""Shim forwarding to rhen.optim.quantized_kv."""
import sys
import rhen.optim.quantized_kv as _mod

sys.modules[__name__] = _mod
