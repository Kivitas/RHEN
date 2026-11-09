"""Shim forwarding to rhen.optim.advanced_loops."""
import sys
import rhen.optim.advanced_loops as _mod

sys.modules[__name__] = _mod
