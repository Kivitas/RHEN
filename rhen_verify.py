"""Shim forwarding to rhen.rhen_verify."""
import sys
import rhen.rhen_verify as _mod

sys.modules[__name__] = _mod
