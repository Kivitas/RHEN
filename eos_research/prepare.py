"""Shim forwarding to rhen.research.prepare."""
import sys
import rhen.research.prepare as _mod

sys.modules[__name__] = _mod
