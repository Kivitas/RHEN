"""Shim forwarding to rhen.meta_cognitive_monitor."""
import sys
import rhen.meta_cognitive_monitor as _mod

sys.modules[__name__] = _mod
