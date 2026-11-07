#!/usr/bin/env python
"""Master verification battery runner for the EOS & Rhen test suites.

Replaces the inline `python -c "subprocess.run(...)"` battery that used to be
embedded in CI and the README. Every test_*.py file in the repository root is a
suite, discovered from disk, so a new suite needs no registry edit. Each suite
runs as an isolated subprocess and the battery fails if any suite exits
non-zero. The runner adds:

  * ``--fast``    only suites under the 15s runtime budget (quick feedback loop)
  * ``--slow``    only suites at or over the budget (the heavyweight gates)
  * ``--suite``   run one specific suite by file name
  * ``--list``    print the discovered suites with classifications and exit
  * per-suite and total wall-clock timing, printed as a summary table

Timing hints in SUITE_RUNTIMES are optional. Registered suites run first, in
registered order; unregistered suites follow, sorted by name, and count as
'fast' until a hint is added.

Usage:
    python run_tests.py              # full battery, CI order
    python run_tests.py --fast       # quick feedback loop
    python run_tests.py --slow       # heavyweight gates only
    python run_tests.py --suite test_rhen_pillars.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Optional timing hints in CI order (matches the historical inline battery and
# .github/workflows/ci.yml). Times are per-suite wall time on the CI-like CPU
# runner. A suite without a hint still runs and is classed 'fast'.
# Budget: 15s (see README "Test Modernization" note).
SUITE_RUNTIMES: dict[str, float] = {
    "test_rhen_master_verification_matrix.py": 17.9,
    "test_rhen_phases456_foundation_and_safety.py": 9.7,
    "test_rhen_phase2_interlayer_and_conformal.py": 11.1,
    "test_rhen_phase1_primitives.py": 15.6,
    "test_rhen_system_one.py": 18.8,
    "test_rhen_brain_enhancements.py": 12.8,
    "test_unified_system.py": 41.4,
    "test_adaptive_conformal_and_merkle_bft.py": 4.9,
    "test_rhen_beyond_jev_and_laya.py": 33.3,
    "test_rhen_cognition.py": 7.0,
    "test_rhen_pillars.py": 32.3,
    "test_rhen_chunking.py": 2.6,
    "test_eos_recurrent_enhancements.py": 11.6,
    # Phase B1: nested config groups + flat-signature round-trip contract.
    "test_config_roundtrip.py": 1.0,
}

FAST_BUDGET_SECONDS = 15.0

ROOT = Path(__file__).resolve().parent


def _classify(runtime: float) -> str:
    return "slow" if runtime >= FAST_BUDGET_SECONDS else "fast"


def _suite_class(suite: str) -> str:
    """Unregistered suites have no timing hint, so they count as 'fast'."""
    runtime = SUITE_RUNTIMES.get(suite)
    return "fast" if runtime is None else _classify(runtime)


def discover_suites() -> list[str]:
    """Every test_*.py in the repo root is a suite; subdirectories are not scanned.

    Registered suites come first, in SUITE_RUNTIMES order; unregistered suites
    follow, sorted by name. A registered name missing on disk is an error,
    because its timing hint is then stale.
    """
    missing = [name for name in SUITE_RUNTIMES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"run_tests.py: missing suite files: {missing}")
    on_disk = {p.name for p in ROOT.glob("test_*.py") if p.is_file()}
    registered = list(SUITE_RUNTIMES)
    return registered + sorted(on_disk - set(registered))


def select_suites(args: argparse.Namespace) -> list[str]:
    suites = discover_suites()
    if args.suite:
        wanted = args.suite if args.suite.endswith(".py") else f"{args.suite}.py"
        if wanted not in suites:
            raise SystemExit(
                f"run_tests.py: unknown suite(s): {[wanted]}. Known: {suites}"
            )
        return [wanted]
    if args.fast:
        return [s for s in suites if _suite_class(s) == "fast"]
    if args.slow:
        return [s for s in suites if _suite_class(s) == "slow"]
    return suites


def run_suite(suite: str) -> tuple[bool, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / suite)],
        cwd=str(ROOT),
        # UTF-8 stdout: a suite that prints non-ASCII must not crash on a Windows
        # pipe (cp1252) and be reported as a failure.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return proc.returncode == 0, time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast", action="store_true",
        help=f"only suites under {FAST_BUDGET_SECONDS:g}s nominal runtime",
    )
    parser.add_argument(
        "--slow", action="store_true",
        help=f"only suites at or over {FAST_BUDGET_SECONDS:g}s nominal runtime",
    )
    parser.add_argument(
        "--suite", metavar="NAME",
        help="run a single suite by file name (with or without .py)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="print the discovered suites and exit",
    )
    args = parser.parse_args(argv)

    if args.list_only:
        suites = discover_suites()
        print(f"{'suite':<52} {'class':<6} {'~runtime':>9}")
        for name in suites:
            runtime = SUITE_RUNTIMES.get(name)
            shown = "-" if runtime is None else f"{runtime:.1f}s"
            print(f"{name:<52} {_suite_class(name):<6} {shown:>9}")
        hinted = sum(1 for name in suites if name in SUITE_RUNTIMES)
        print(f"\n{len(suites)} suites discovered ({hinted} with timing hints)")
        return 0

    suites = select_suites(args)
    label = "fast" if args.fast else "slow" if args.slow else args.suite or "full"

    print("=" * 74)
    print(f"  EOS & RHEN MASTER VERIFICATION BATTERY - {label.upper()} "
          f"({len(suites)} suite{'s' if len(suites) != 1 else ''})")
    print("=" * 74)

    results: list[tuple[str, bool, float]] = []
    for suite in suites:
        ok, elapsed = run_suite(suite)
        results.append((suite, ok, elapsed))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {elapsed:7.2f}s  {suite}", flush=True)

    total = sum(t for _, _, t in results)
    failures = [name for name, ok, _ in results if not ok]

    print("\n" + "-" * 74)
    print(f"  {'SUITE':<52} {'STATUS':<7} {'WALL':>9}")
    for name, ok, elapsed in results:
        print(f"  {name:<52} {'PASS' if ok else 'FAIL':<7} {elapsed:>7.2f}s")
    print("-" * 74)
    print(f"  TOTAL: {total:.2f}s   PASSED: {len(results) - len(failures)}/{len(results)}")
    print("=" * 74)

    if failures:
        print("\nFAILED SUITES:")
        for name in failures:
            print(f"  - {name}")
        return 1

    print(f"\nALL {len(results)}/{len(results)} SUITES ({label}) PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
