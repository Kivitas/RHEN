"""Shim forwarding to rhen.optim.grouped_gemm."""
import sys
import rhen.optim.grouped_gemm as _mod

sys.modules[__name__] = _mod
