#!/usr/bin/env python3
"""Build WB2 (statistical learning). Cells live in wb2_cells_a.py (§1–5) and wb2_cells_b.py (§6–9)."""
from nbkit import run
from wb2_cells_a import CELLS as A
from wb2_cells_b import CELLS as B

if __name__ == "__main__":
    raise SystemExit(run(A + B, "wb2", "02_statistical_learning"))
