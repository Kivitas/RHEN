"""Pytest collection policy for the EOS & Rhen master battery.

Since Phase A2, each script-style suite defines a
``unittest.TestCase`` wrapper (inserted just above its ``if __name__ ==
"__main__"`` block) that mirrors the legacy runner's call order. The wrapper is
the canonical pytest/unittest surface; the bare module-level ``test_*``
functions remain the script runner's internals.

Without this policy pytest would collect *both* and run every assertion twice
(e.g. ``test_rhen_chunking`` showed 24 items instead of 12). When a module
defines a TestCase wrapper, this policy collects only the wrapper classes.
"""

from __future__ import annotations

import unittest


def _is_unittest_item(item) -> bool:
    """True for pytest items that run a unittest.TestCase method.

    pytest wraps TestCase methods in ``TestCaseFunction``, which is *not* an
    ``unittest.TestCase`` subclass, so we check the bound class instead.
    """
    cls = getattr(item, "cls", None)
    return isinstance(cls, type) and issubclass(cls, unittest.TestCase)


def pytest_collection_modifyitems(config, items):
    module_of = {}
    for item in items:
        module_of.setdefault(item.module.__name__, []).append(item)

    # One TestCase class per suite file, keyed by module name.
    wrapper_classes = {
        name for name, members in module_of.items()
        if any(_is_unittest_item(m) for m in members)
    }

    keep, drop = [], []
    for item in items:
        if item.module.__name__ in wrapper_classes and not _is_unittest_item(item):
            drop.append(item)
        else:
            keep.append(item)

    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
