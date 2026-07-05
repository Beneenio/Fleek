"""Explainable ranking of genuine Manchester vintage shops for a day of visits.

The classifier is the *gate* (only genuine shops get here). Ranking then answers
"which are worth my day, and why", from signals the brief calls out — quality,
size/appetite, price positioning, and how clustered they are so you can walk
between them — plus an enrichment bonus that is wired but 0 today (see enrich.py).

Every shop gets a signed, weighted score in 0–100 and a plain-English `reason`, so
"why shop A beats shop B" is inspectable rather than asserted.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Weights sum to 1.0. Enrichment carries real weight but contributes 0 under the
# stub, so today's scores top out ~85 and the *ordering* is unaffected — the
# headroom is deliberate and documented.
WEIGHTS: Dict[str, float] = {
    "rating": 0.30,      # quality
    "reviews": 0.25,     # size / footfall / appetite
    "price": 0.10,       # positioning (mid/upmarket buys more than pure budget)
    "density": 0.20,     # walkable cluster => efficient day
    "enrichment": 0.15,  # IG/site/multi-location/busyness (stubbed -> 0)
}

# neighbours within 500m at which the density signal saturates
DENSITY_SATURATION = 4
# review count at which the footfall signal saturates (~log scale)
REVIEW_SATURATION = 800

_PRICE_SCORE = {"£": 0.30, "££": 1.00, "£££": 0.85}


def rating_score(rating) -> float:
    r = pd.to_numeric(rating, errors="coerce")
    if pd.isna(r):
        return 0.5
    return float(np.clip((r - 3.0) / 2.0, 0.0, 1.0))  # 3★->0, 5★->1


def reviews_score(count) -> float:
    c = pd.to_numeric(count, errors="coerce")
    if pd.isna(c) or c <= 0:
        return 0.0
    return float(np.clip(math.log10(c + 1) / math.log10(REVIEW_SATURATION), 0.0, 1.0))


def price_score(price_level) -> float:
    if price_level is None or (isinstance(price_level, float) and math.isnan(price_level)):
        return 0.5
    return _PRICE_SCORE.get(str(price_level).strip(), 0.5)


def density_score(neighbours) -> float:
    n = pd.to_numeric(neighbours, errors="coerce")
    if pd.isna(n):
        return 0.0
    return float(np.clip(n / DENSITY_SATURATION, 0.0, 1.0))


def enrichment_score(row, prefix: str = "enr_") -> Tuple[float, list]:
    """0..1 from enrichment signals, plus notes. All-None (stub) -> 0."""
    score = 0.0
    notes = []
    followers = row.get(f"{prefix}instagram_followers")
    if followers is not None and not pd.isna(followers) and followers > 0:
        f = float(np.clip(math.log10(followers + 1) / math.log10(50000), 0.0, 1.0))
        score += 0.4 * f
        notes.append(f"IG {int(followers):,} followers")
    if bool(row.get(f"{prefix}instagram_active")):
        score += 0.2
        notes.append("active IG")
    locs = row.get(f"{prefix}num_locations")
    if locs is not None and not pd.isna(locs) and locs > 1:
        score += 0.2
        notes.append(f"{int(locs)} locations")
    size = row.get(f"{prefix}storefront_size")
    if isinstance(size, str) and size.lower() == "large":
        score += 0.1
        notes.append("large storefront")
    pop = row.get(f"{prefix}popularity")
    if pop is not None and not pd.isna(pop):
        score += 0.1 * float(np.clip(pop, 0.0, 1.0))
        notes.append("busy on Google")
    return float(np.clip(score, 0.0, 1.0)), notes


def _reason(row, components: Dict[str, float]) -> str:
    parts = []

    r = pd.to_numeric(row.get("rating"), errors="coerce")
    rc = pd.to_numeric(row.get("review_count"), errors="coerce")
    if not pd.isna(r):
        band = ("excellent" if r >= 4.5 else "strong" if r >= 4.0
                else "solid" if r >= 3.7 else "modest")
        rc_txt = f"{int(rc):,} reviews" if not pd.isna(rc) else "few reviews"
        foot = (" (established, high footfall)" if not pd.isna(rc) and rc >= 500
                else " (well-reviewed)" if not pd.isna(rc) and rc >= 200 else "")
        parts.append(f"{band} {r:.1f}★ on {rc_txt}{foot}")

    pl = row.get("price_level")
    price_txt = {"£": "budget (£)", "££": "mid-range (££)", "£££": "upmarket (£££)"}
    if isinstance(pl, str) and pl.strip() in price_txt:
        parts.append(price_txt[pl.strip()])

    neigh = pd.to_numeric(row.get("neighbours"), errors="coerce")
    if not pd.isna(neigh):
        if neigh >= 3:
            parts.append(f"dense cluster — {int(neigh)} vintage shops within 500m, easy to combine")
        elif neigh >= 1:
            parts.append(f"near {int(neigh)} other vintage shop(s)")
        else:
            parts.append("standalone (a small detour)")

    enr_notes = row.get("_enr_notes")
    if enr_notes:
        parts.append("enrichment: " + ", ".join(enr_notes))
    else:
        parts.append("enrichment not yet wired (stub)")

    sig = str(row.get("vintage_signals") or "")
    pos = [s for s in sig.split(" | ") if s.startswith("+review")]
    if pos:
        parts.append("genuine: " + pos[0].split(":", 1)[1].strip())

    return " · ".join(parts)


def rank_frame(df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
    """Score and sort genuine shops. Expects classifier + enrichment + `neighbours`
    columns already present. Returns a copy sorted by descending score."""
    w = weights or WEIGHTS
    out = df.copy().reset_index(drop=True)

    out["s_rating"] = out["rating"].map(rating_score)
    out["s_reviews"] = out["review_count"].map(reviews_score)
    out["s_price"] = out["price_level"].map(price_score)
    out["s_density"] = out["neighbours"].map(density_score) if "neighbours" in out else 0.0

    enr = [enrichment_score(row) for _, row in out.iterrows()]
    out["s_enrichment"] = [e[0] for e in enr]
    out["_enr_notes"] = [e[1] for e in enr]

    out["score"] = (
        w["rating"] * out["s_rating"]
        + w["reviews"] * out["s_reviews"]
        + w["price"] * out["s_price"]
        + w["density"] * out["s_density"]
        + w["enrichment"] * out["s_enrichment"]
    ) * 100.0
    out["score"] = out["score"].round(1)

    out["reason"] = [_reason(row, w) for _, row in out.iterrows()]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out = out.drop(columns=["_enr_notes"])
    return out
