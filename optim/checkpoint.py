"""Shim forwarding to rhen.optim.checkpoint."""
import sys
import rhen.optim.checkpoint as _mod

sys.modules[__name__] = _mod
