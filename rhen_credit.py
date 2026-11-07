"""Shim forwarding to rhen.rhen_credit."""
import sys
import rhen.rhen_credit as _mod

sys.modules[__name__] = _mod
