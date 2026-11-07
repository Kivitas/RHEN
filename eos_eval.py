"""Shim forwarding to rhen.eos_eval."""
import sys
import rhen.eos_eval as _mod

sys.modules[__name__] = _mod
