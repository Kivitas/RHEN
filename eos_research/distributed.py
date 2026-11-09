"""Shim forwarding to rhen.research.distributed."""
import sys
import rhen.research.distributed as _mod

sys.modules[__name__] = _mod
