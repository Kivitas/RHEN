"""Shim forwarding to rhen.research.train."""
import sys
import rhen.research.train as _mod

sys.modules[__name__] = _mod
