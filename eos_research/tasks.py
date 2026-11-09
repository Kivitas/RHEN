"""Shim forwarding to rhen.research.tasks."""
import sys
import rhen.research.tasks as _mod

sys.modules[__name__] = _mod
