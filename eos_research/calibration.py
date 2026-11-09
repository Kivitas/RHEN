"""Shim forwarding to rhen.research.calibration."""
import sys
import rhen.research.calibration as _mod

sys.modules[__name__] = _mod
