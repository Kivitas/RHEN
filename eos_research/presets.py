"""Shim forwarding to rhen.research.presets."""
import sys
import rhen.research.presets as _mod

sys.modules[__name__] = _mod
