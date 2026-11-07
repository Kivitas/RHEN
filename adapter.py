"""Shim forwarding to rhen.adapter."""
import sys
import rhen.adapter as _mod

sys.modules[__name__] = _mod
