#!/usr/bin/env python3
"""Execute every notebook in examples/notebooks/ end to end.

The notebooks carry Colab badges, so a reader's first contact with this
library may be running one. A notebook that raises halfway is worse than no
notebook. The pip-install cell is stripped, since the checkout is already
importable in CI.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "examples" / "notebooks"


def main() -> int:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        print("no notebooks found")
        return 1

    failures = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nb.cells = [
            c for c in nb.cells
            if not (c.cell_type == "code" and "pip install" in "".join(c.source))
        ]
        client = NotebookClient(
            nb, timeout=1800, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
            print(f"  ok    {path.name}")
        except Exception as exc:  # noqa: BLE001 - report any execution error
            print(f"  FAIL  {path.name}: {type(exc).__name__}")
            failures.append(path.name)

    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {', '.join(failures)}")
        return 1
    print(f"\nall {len(paths)} notebooks executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
