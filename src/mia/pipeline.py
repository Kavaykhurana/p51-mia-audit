"""Execute notebooks 00-09 in order, saving each one's outputs in place. Used by notebooks/run_all.ipynb.

Run directly with:  python -m mia.pipeline   (from the repo root, with src/ on PYTHONPATH)
"""
from __future__ import annotations

import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from .config import ROOT

NOTEBOOKS_DIR = ROOT / "notebooks"


def pipeline_notebooks() -> list:
    return sorted(NOTEBOOKS_DIR.glob("0[0-9]_*.ipynb"))


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def run_notebook(path) -> None:
    """Execute one notebook top to bottom (no time limit) and write its outputs back into the file."""
    nb = nbformat.read(path, as_version=4)
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    t0 = time.perf_counter()

    def on_start(cell, cell_index):
        first_line = next((ln for ln in cell.source.splitlines() if ln.strip() and not ln.startswith("#")), "")
        print(f"    cell {code_cells.index(cell_index) + 1}/{len(code_cells)}  {first_line[:70]}", flush=True)

    client = NotebookClient(nb, timeout=None, kernel_name="python3", on_cell_execute=on_start,
                            resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)  # keep outputs (and any error) so the notebook can be opened and read
    print(f"    done in {_clock(time.perf_counter() - t0)}", flush=True)


def run_all() -> bool:
    t0 = time.perf_counter()
    notebooks = pipeline_notebooks()
    for i, path in enumerate(notebooks, 1):
        print(f"[{i}/{len(notebooks)}] {path.name}", flush=True)
        try:
            run_notebook(path)
        except CellExecutionError as e:
            print(f"\nFAILED in {path.name}. Open that notebook to see the error, fix it, then run this again "
                  f"(finished work is skipped).\n{str(e)[-1500:]}", flush=True)
            return False
    print(f"\nAll {len(notebooks)} notebooks finished in {_clock(time.perf_counter() - t0)}.", flush=True)
    return True


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
