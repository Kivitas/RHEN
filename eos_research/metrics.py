"""Shim forwarding to rhen.research.metrics."""
import sys
import rhen.research.metrics as _mod

sys.modules[__name__] = _mod
