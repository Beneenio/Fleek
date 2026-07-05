"""Messy-date parsing for Part 2's `last_contact_date` / `last_purchase_date`.

The sheet mixes five observed formats and leaves some blank:

    2026-04-28      ISO                     %Y-%m-%d
    2026/05/21      ISO, slash              %Y/%m/%d
    19/06/2026      day-first, slash        %d/%m/%Y
    June 4 2026     long month, no comma    %B %d %Y
    14 May          day + month, NO year    %d %b   (year inferred)

We try the known formats first (fast, unambiguous), then fall back to a
day-first dateutil parse so an unseen sixth format degrades to a best-effort
date instead of crashing the pipeline at scale. Blanks map to ``None``.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Optional

import pandas as pd
from dateutil import parser as _dateutil_parser

# Ordered most-specific / most-common first. Formats without %Y get the year
# filled from ``default_year`` (the year-less "14 May" style dates).
KNOWN_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%B %d %Y",
    "%d %b",
)

# The workbook is a mid-2026 snapshot; year-less dates ("14 May") are recent.
DEFAULT_YEAR = 2026

_BLANK_TOKENS = {"", "nan", "nat", "none", "null", "-"}


def parse_messy_date(value, default_year: int = DEFAULT_YEAR) -> Optional[dt.date]:
    """Parse one messy cell into a ``date``, or ``None`` if blank/unparseable."""
    # Already a real date/datetime (openpyxl parses some ISO cells for us).
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    s = str(value).strip()
    if s.lower() in _BLANK_TOKENS:
        return None

    for fmt in KNOWN_FORMATS:
        try:
            parsed = dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=default_year)
        return parsed.date()

    # Unknown format: best-effort, day-first (UK data). Year-less inputs inherit
    # ``default_year`` via the default anchor.
    try:
        anchor = dt.datetime(default_year, 1, 1)
        return _dateutil_parser.parse(s, dayfirst=True, default=anchor).date()
    except (ValueError, OverflowError, TypeError):
        return None


def parse_date_series(series: pd.Series, default_year: int = DEFAULT_YEAR) -> pd.Series:
    """Vectorised-friendly wrapper: parse a whole column to ``date``/``None``."""
    return series.map(lambda v: parse_messy_date(v, default_year=default_year))
