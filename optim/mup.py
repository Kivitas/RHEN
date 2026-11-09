"""Shim forwarding to rhen.optim.mup."""
import sys
import rhen.optim.mup as _mod

sys.modules[__name__] = _mod
