"""Shim forwarding to rhen.research.checkpointing."""
import sys
import rhen.research.checkpointing as _mod

sys.modules[__name__] = _mod
