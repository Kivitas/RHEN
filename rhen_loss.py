"""Shim forwarding to rhen.rhen_loss."""
import sys
import rhen.rhen_loss as _mod

sys.modules[__name__] = _mod
