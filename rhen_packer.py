"""Shim forwarding to rhen.rhen_packer."""
import sys
import rhen.rhen_packer as _mod

sys.modules[__name__] = _mod
