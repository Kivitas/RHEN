"""Shim forwarding to rhen.optim.ring_mla."""
import sys
import rhen.optim.ring_mla as _mod

sys.modules[__name__] = _mod
