"""Shim forwarding to rhen.config."""
import sys
import rhen.config as _mod

sys.modules[__name__] = _mod
