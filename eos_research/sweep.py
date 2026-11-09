"""Shim forwarding to rhen.research.sweep."""
import sys
import rhen.research.sweep as _mod

sys.modules[__name__] = _mod
