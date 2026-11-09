"""Shim forwarding to rhen.research.evaluate."""
import sys
import rhen.research.evaluate as _mod

sys.modules[__name__] = _mod
