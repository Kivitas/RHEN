"""Shim forwarding to rhen.optim.fused_swiglu."""
import sys
import rhen.optim.fused_swiglu as _mod

sys.modules[__name__] = _mod
