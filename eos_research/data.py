"""Shim forwarding to rhen.research.data."""
import sys
import rhen.research.data as _mod

sys.modules[__name__] = _mod
