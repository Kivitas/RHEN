"""Shim forwarding to rhen.tokenizer."""
import sys
import rhen.tokenizer as _mod

sys.modules[__name__] = _mod
