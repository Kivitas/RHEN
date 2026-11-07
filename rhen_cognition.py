"""Shim forwarding to rhen.rhen_cognition."""
import sys
import rhen.rhen_cognition as _mod

sys.modules[__name__] = _mod
