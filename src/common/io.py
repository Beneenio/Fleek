"""Data loading — the single place the workbook is read.

Every downstream module (classify, rank, clean, outbound, city_rank) imports the
two loaders below and never opens the .xlsx itself. That keeps the data source
behind one seam: swapping the messy scrape for a CSV export, a DuckDB table, or a
Postgres query at 30,000 rows is a change to *this file only*, not a grep across
the whole pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

# repo_root/src/common/io.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WORKBOOK = DATA_DIR / "Fleek_-_Physical_Store_Acquisition_-_Pipeline_Data.xlsx"

PART1_SHEET = "Part1_Manchester_scrape"
PART2_SHEET = "Part2_leads_and_customers"


def _load_sheet(sheet: str, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    workbook = Path(path) if path is not None else WORKBOOK
    if not workbook.exists():
        raise FileNotFoundError(
            f"Workbook not found at {workbook}. Expected the case-study xlsx in "
            f"{DATA_DIR}/ — see the README for setup."
        )
    return pd.read_excel(workbook, sheet_name=sheet)


def load_part1_df(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Part 1 — the raw 121-row Manchester Google Maps scrape (one row per place)."""
    return _load_sheet(PART1_SHEET, path)


def load_part2_df(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Part 2 — the 206-row leads-and-customers book across every city and stage."""
    return _load_sheet(PART2_SHEET, path)
