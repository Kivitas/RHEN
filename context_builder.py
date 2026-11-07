"""Shim forwarding to rhen.context_builder."""
import sys
import rhen.context_builder as _mod

sys.modules[__name__] = _mod
