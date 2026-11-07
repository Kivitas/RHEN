"""Shim forwarding to rhen package."""
import sys
import rhen as _mod

sys.modules[__name__] = _mod
