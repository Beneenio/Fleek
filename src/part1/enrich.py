"""Enrichment interface for Part 1 — STUBBED, but wired end-to-end.

The brief explicitly asks us to push past what's in the scrape: Instagram (do they
have one, how big/active), their own website, storefront size, whether they run
multiple locations, how busy Google says they get. That signal genuinely sharpens
a visit list — a shop with 40k engaged IG followers and three locations is a very
different trip from a quiet single unit.

Rather than build (and rate-limit-manage) four scrapers for a half-day tool, we
define the *interface* and a `provider` seam, and ship a stub that returns all-None
so the enrichment bonus is a wired, unit-tested **0** today. Swapping in the real
thing is: implement a `provider(row) -> EnrichmentSignals` and pass it in — no
change to ranking. The tests exercise the bonus with a fixture provider, proving
the wiring works the day the real data shows up.

Real wiring (documented, not built):
- instagram_*: resolve handle from name+city via IG Graph API / a maps enrichment
  vendor, read follower count and last-post recency.
- num_locations: group the multi-city scrape by brand/owner, or Places "chain" data.
- storefront_size: Places photos / Street View frontage width heuristic.
- popularity: Google Places "popular times" busyness.
All are network calls, so at 30k rows they'd run in a bounded-concurrency batch and
be cached — see the scale notes in the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd


@dataclass
class EnrichmentSignals:
    instagram_followers: Optional[int] = None
    instagram_active: Optional[bool] = None   # posted within ~30 days
    num_locations: Optional[int] = None       # >1 => multi-site operator
    storefront_size: Optional[str] = None      # "small" | "medium" | "large"
    popularity: Optional[float] = None         # Google busyness, 0..1
    source: str = "stub"


# A provider takes a row (Series/dict-like) and returns EnrichmentSignals. This is
# the single seam the real IG/website/Places wiring plugs into.
Provider = Callable[[object], EnrichmentSignals]


def _stub_provider(row) -> EnrichmentSignals:
    """No network calls: everything unknown, so the enrichment bonus is 0."""
    return EnrichmentSignals(source="stub")


def enrich_shop(row, provider: Optional[Provider] = None) -> EnrichmentSignals:
    return (provider or _stub_provider)(row)


def enrich_frame(df: pd.DataFrame, provider: Optional[Provider] = None,
                 prefix: str = "enr_") -> pd.DataFrame:
    """Attach enrichment columns. Stub leaves them empty; a real provider fills them."""
    signals: List[EnrichmentSignals] = [enrich_shop(row, provider)
                                        for _, row in df.iterrows()]
    out = df.copy()
    out[f"{prefix}instagram_followers"] = [s.instagram_followers for s in signals]
    out[f"{prefix}instagram_active"] = [s.instagram_active for s in signals]
    out[f"{prefix}num_locations"] = [s.num_locations for s in signals]
    out[f"{prefix}storefront_size"] = [s.storefront_size for s in signals]
    out[f"{prefix}popularity"] = [s.popularity for s in signals]
    out[f"{prefix}source"] = [s.source for s in signals]
    return out
