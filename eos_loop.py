"""Shim forwarding to rhen.eos_loop."""
import sys
import rhen.eos_loop as _mod

sys.modules[__name__] = _mod
