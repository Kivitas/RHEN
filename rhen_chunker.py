"""Shim forwarding to rhen.rhen_chunker."""
import sys
import rhen.rhen_chunker as _mod

sys.modules[__name__] = _mod
