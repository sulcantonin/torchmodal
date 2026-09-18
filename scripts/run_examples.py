#!/usr/bin/env python3
"""Smoke-test every script in examples/.

A reviewer will run these, so CI runs them too. Each script must exit 0 and
print something; a script that runs but produces no output is treated as a
failure, since every example here is meant to demonstrate a result.

The benchmark scripts honour the ``MLNN_*_SMOKE`` environment variables and
are given a longer budget. Nothing is skipped: an example that cannot be run
in CI is an example that will break without anyone noticing.

**Committed artefacts are restored afterwards.** Several benchmarks write into
``examples/coloring_results/``, ``examples/sudoku_results/`` and
``examples/coloring_solve.json``, which are tracked and hold *full*-mode
results — 10 graphs per tier, 5 seeds, 2000 epochs. A smoke run would
overwrite them with far weaker numbers, so anything this script dirties is
checked back out at the end and reported.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# Scripts that need a longer budget even in smoke mode.
SLOW = {
    "graph_coloring_benchmark.py",
    "sudoku_benchmark.py",
    "baseline_comparison.py",
    "regen_coloring_figs.py",
}
FAST_TIMEOUT = 300
SLOW_TIMEOUT = 1800


#: Tracked paths the benchmarks write into. Restored after the run.
ARTEFACTS = [
    "examples/coloring_results",
    "examples/sudoku_results",
    "examples/coloring_solve.json",
]


def _restore_artefacts() -> list[str]:
    """Check out any tracked artefact the run modified, and say which."""
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--"] + ARTEFACTS,
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []          # not a git checkout; nothing to restore
    changed = [p for p in dirty if p.startswith("examples/")]
    if changed:
        subprocess.run(
            ["git", "checkout", "--"] + ARTEFACTS,
            cwd=ROOT, capture_output=True, check=False,
        )
    return changed


def main() -> int:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MLNN_COLOR_SMOKE", "1")
    env.setdefault("MLNN_SUDOKU_SMOKE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    scripts = sorted(p for p in EXAMPLES.glob("*.py"))
    if not scripts:
        print("no examples found — that is itself a failure")
        return 1

    failures: list[tuple[str, str]] = []
    print(f"running {len(scripts)} examples\n")

    for path in scripts:
        name = path.name
        budget = SLOW_TIMEOUT if name in SLOW else FAST_TIMEOUT
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=ROOT, env=env, capture_output=True,
                text=True, timeout=budget,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT  {name:<38} (> {budget}s)")
            failures.append((name, f"timed out after {budget}s"))
            continue

        elapsed = time.time() - start
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            print(f"  FAIL     {name:<38} {elapsed:6.1f}s  {tail}")
            failures.append((name, tail))
        elif not proc.stdout.strip():
            print(f"  SILENT   {name:<38} {elapsed:6.1f}s")
            failures.append((name, "exited 0 but printed nothing"))
        else:
            print(f"  ok       {name:<38} {elapsed:6.1f}s")

    restored = _restore_artefacts()
    if restored:
        print(
            f"\nrestored {len(restored)} committed artefact(s) the smoke run "
            f"overwrote: {', '.join(sorted(set(restored)))}"
        )

    print()
    if failures:
        print(f"{len(failures)} of {len(scripts)} examples failed:")
        for name, why in failures:
            print(f"  - {name}: {why}")
        return 1
    print(f"all {len(scripts)} examples ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
