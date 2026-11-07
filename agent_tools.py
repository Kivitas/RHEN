"""Shim forwarding to rhen.agent_tools."""
import sys
import rhen.agent_tools as _mod

sys.modules[__name__] = _mod
