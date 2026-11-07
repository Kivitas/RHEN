"""Shim forwarding to rhen.optimizers."""
import sys
import rhen.optimizers as _mod

sys.modules[__name__] = _mod
