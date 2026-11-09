"""Shim forwarding to rhen.research.device."""
import sys
import rhen.research.device as _mod

sys.modules[__name__] = _mod
