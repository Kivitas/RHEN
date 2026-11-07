"""Shim forwarding to rhen.rhen_pillars."""
import sys
import rhen.rhen_pillars as _mod

sys.modules[__name__] = _mod
