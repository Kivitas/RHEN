"""Regression guards for the test runner, the CI matrix and the contributor docs.

run_tests.py must discover every root-level test_*.py as a suite, and its
--list and banner output must stay pure ASCII (Windows consoles garble
non-ASCII). ci.yml must keep the Windows and Python 3.10 legs and an all-files
pyflakes lint. CONTRIBUTING.md must keep a sequential numbered list and no
hard-coded suite counts.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "tests"
    else Path(__file__).resolve().parent
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_tests  # noqa: E402  (after the sys.path tweak above)

# Matches hard-coded counts such as "14 suites" or "13 verification suites".
SUITE_COUNT_RE = re.compile(r"\b1[34]\b[^\n]{0,20}\bsuites?\b", re.IGNORECASE)

DISCOVERY_FILES = {
    "test_known_b.py": "pass\n",
    "test_known_a.py": "pass\n",
    "test_zeta_new.py": "pass\n",
    "test_alpha_new.py": "pass\n",
    "helper.py": "pass\n",
    "notes_test.py": "pass\n",
    "sub/test_nested.py": "pass\n",
}
DISCOVERY_HINTS = {"test_known_b.py": 2.0, "test_known_a.py": 20.0}


@contextlib.contextmanager
def fake_repo(files, hints):
    """Write ``files`` into a temp dir and point run_tests at it.

    ``run_tests.ROOT`` and ``run_tests.SUITE_RUNTIMES`` are patched for the
    duration, so discovery and suite subprocesses see only this layout.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, body in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        with mock.patch.object(run_tests, "ROOT", root):
            with mock.patch.object(run_tests, "SUITE_RUNTIMES", dict(hints)):
                yield root


def _args(suite=None, fast=False, slow=False):
    return argparse.Namespace(suite=suite, fast=fast, slow=slow, list_only=False)


def _root_test_files_by_glob():
    """Independent of run_tests: a plain glob over the real repo root."""
    pattern = os.path.join(str(ROOT), "test_*.py")
    return {os.path.basename(p) for p in glob.glob(pattern) if os.path.isfile(p)}


def _run_list():
    return subprocess.run(
        [sys.executable, str(ROOT / "run_tests.py"), "--list"],
        capture_output=True,
        cwd=str(ROOT),
        timeout=60,
    )


class DiscoveryTests(unittest.TestCase):
    def test_discovers_every_root_test_file(self):
        self.assertEqual(set(run_tests.discover_suites()), _root_test_files_by_glob())

    def test_known_suites_first_in_registered_order_then_unknown_sorted(self):
        with fake_repo(DISCOVERY_FILES, DISCOVERY_HINTS):
            self.assertEqual(
                run_tests.discover_suites(),
                ["test_known_b.py", "test_known_a.py",
                 "test_alpha_new.py", "test_zeta_new.py"],
            )

    def test_missing_registered_suite_is_an_error(self):
        with fake_repo({"test_real.py": "pass\n"}, {"test_gone.py": 1.0}):
            with self.assertRaises(SystemExit):
                run_tests.discover_suites()

    def test_unregistered_suites_default_to_fast(self):
        with fake_repo(DISCOVERY_FILES, DISCOVERY_HINTS):
            fast = run_tests.select_suites(_args(fast=True))
            slow = run_tests.select_suites(_args(slow=True))
        self.assertEqual(
            fast, ["test_known_b.py", "test_alpha_new.py", "test_zeta_new.py"])
        self.assertEqual(slow, ["test_known_a.py"])

    def test_suite_flag_accepts_unregistered_root_suite(self):
        with fake_repo(DISCOVERY_FILES, DISCOVERY_HINTS):
            self.assertEqual(
                run_tests.select_suites(_args(suite="test_alpha_new")),
                ["test_alpha_new.py"])

    def test_suite_flag_rejects_non_suites(self):
        with fake_repo(DISCOVERY_FILES, DISCOVERY_HINTS):
            for bad in ("helper.py", "notes_test.py", "sub/test_nested.py",
                        "run_tests.py", "test_gone"):
                with self.subTest(bad=bad):
                    with self.assertRaises(SystemExit):
                        run_tests.select_suites(_args(suite=bad))


class RunnerExitCodeAndOutputTests(unittest.TestCase):
    FILES = {
        "test_ok.py": "raise SystemExit(0)\n",
        "test_broken.py": "raise SystemExit(3)\n",
    }

    def _main(self, argv):
        out = io.StringIO()
        with fake_repo(self.FILES, {}), contextlib.redirect_stdout(out):
            rc = run_tests.main(argv)
        return rc, out.getvalue()

    def test_passing_suite_exits_zero_with_ascii_output(self):
        rc, out = self._main(["--suite", "test_ok.py"])
        self.assertEqual(rc, 0)
        self.assertTrue(out.isascii(), out)

    def test_failing_suite_exits_one(self):
        rc, out = self._main(["--suite", "test_broken.py"])
        self.assertEqual(rc, 1)
        self.assertIn("FAILED SUITES", out)
        self.assertTrue(out.isascii(), out)

    def test_full_run_lists_every_suite_and_fails_if_any_fails(self):
        rc, out = self._main([])
        self.assertEqual(rc, 1)
        self.assertIn("test_ok.py", out)
        self.assertIn("test_broken.py", out)


class ListOutputTests(unittest.TestCase):
    def test_list_output_is_pure_ascii(self):
        proc = _run_list()
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("ascii", "replace"))
        for stream, data in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            bad = sorted({b for b in data if b > 0x7F})
            self.assertEqual(bad, [], f"non-ASCII bytes in --list {stream}: {bad}")

    def test_list_shows_every_discovered_suite(self):
        text = _run_list().stdout.decode("ascii")
        for name in run_tests.discover_suites():
            self.assertIn(name, text)


class CiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8")

    def test_matrix_covers_windows(self):
        self.assertIn("windows-latest", self.text)

    def test_matrix_covers_python_3_10_as_a_string(self):
        # Unquoted 3.10 is parsed by YAML as the float 3.1.
        self.assertRegex(self.text, r"""python-version:\s*\[\s*["']3\.10["']""")

    def test_lint_covers_all_tracked_python_files(self):
        self.assertIn("python -m pyflakes $(git ls-files '*.py')", self.text)

    def test_battery_artifact_name_is_unique_per_leg(self):
        line = next((ln for ln in self.text.splitlines()
                     if "name: battery-log-" in ln), "")
        self.assertIn("matrix.os", line)
        self.assertIn("matrix.python-version", line)

    def test_no_hard_coded_suite_counts(self):
        self.assertNotRegex(self.text, SUITE_COUNT_RE)


class ContributingDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    def test_numbered_guidelines_are_sequential(self):
        numbers = [int(m.group(1))
                   for m in re.finditer(r"^(\d+)\. \*\*", self.text, re.MULTILINE)]
        self.assertTrue(numbers)
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_no_hard_coded_suite_counts(self):
        self.assertNotRegex(self.text, SUITE_COUNT_RE)

    def test_lint_command_matches_ci(self):
        self.assertIn("python -m pyflakes $(git ls-files '*.py')", self.text)


if __name__ == "__main__":
    unittest.main()
