"""Shim forwarding to rhen.optim.rhen_mask."""
import sys
import rhen.optim.rhen_mask as _mod

sys.modules[__name__] = _mod
