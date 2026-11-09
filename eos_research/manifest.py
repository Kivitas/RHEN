"""Shim forwarding to rhen.research.manifest."""
import sys
import rhen.research.manifest as _mod

sys.modules[__name__] = _mod
