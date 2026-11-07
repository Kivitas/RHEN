"""Shim forwarding to rhen.model."""
import sys
import rhen.model as _mod

sys.modules[__name__] = _mod
