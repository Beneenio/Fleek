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
import re
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

# Weights sum to 1.0. Enrichment now carries real offline signal (stock appetite &
# storefront size from review text, real IG/spend where a shop is already in the CRM,
# website presence) — see enrich.py. Live IG-activity / multi-location / Google
# busyness slot into the same weight when a network provider fills them.
WEIGHTS: Dict[str, float] = {
    "rating": 0.30,      # quality
    "reviews": 0.25,     # size / footfall / appetite
    "price": 0.10,       # positioning (mid/upmarket buys more than pure budget)
    "density": 0.20,     # walkable cluster => efficient day
    "enrichment": 0.15,  # stock appetite / storefront size / real IG+spend / website
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
    """0..1 from enrichment signals, plus notes. No signals -> 0.

    Weighted toward what matters to a wholesale supplier: stock appetite (how much a
    shop buys) and real spend/reach lead; storefront size and website presence are
    tie-breakers. IG activity / multi-location / Google busyness are live-scrape
    extensions (see enrich.py) and contribute only when a network provider fills them.
    """
    score = 0.0
    notes = []

    appetite = row.get(f"{prefix}stock_appetite")
    if appetite is not None and not pd.isna(appetite):
        score += 0.35 * float(np.clip(appetite, 0.0, 1.0))
        if appetite >= 0.66:
            notes.append("high stock turnover")
        elif appetite > 0:
            notes.append("moves stock")

    spend = row.get(f"{prefix}est_monthly_spend")
    if spend is not None and not pd.isna(spend) and spend > 0:
        s = float(np.clip(math.log10(spend + 1) / math.log10(3000), 0.0, 1.0))
        score += 0.20 * s
        notes.append(f"known ~£{int(spend):,}/mo spend")

    followers = row.get(f"{prefix}instagram_followers")
    if followers is not None and not pd.isna(followers) and followers > 0:
        f = float(np.clip(math.log10(followers + 1) / math.log10(50000), 0.0, 1.0))
        score += 0.20 * f
        notes.append(f"IG {int(followers):,} followers")

    size = row.get(f"{prefix}storefront_size")
    if isinstance(size, str) and size.lower() == "large":
        score += 0.10
        notes.append("large storefront")

    if bool(row.get(f"{prefix}has_website")):
        score += 0.05
        notes.append("has website")

    if bool(row.get(f"{prefix}instagram_active")):
        score += 0.05
        notes.append("active IG")
    locs = row.get(f"{prefix}num_locations")
    if locs is not None and not pd.isna(locs) and locs > 1:
        score += 0.05
        notes.append(f"{int(locs)} locations")
    pop = row.get(f"{prefix}popularity")
    if pop is not None and not pd.isna(pop):
        score += 0.05 * float(np.clip(pop, 0.0, 1.0))
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


# --- Pairwise explainer: "why shop A beats shop B" ---------------------------
# The brief asks us to be able to show *why* A beats B, not just assert it. The
# ranked frame already carries every `s_<component>` sub-score, so the gap between
# two shops decomposes exactly into weighted per-component contributions that sum
# to the headline score difference.

_COMPONENT_LABELS = {
    "rating": "quality (rating)",
    "reviews": "size / footfall (reviews)",
    "price": "price positioning",
    "density": "walkable cluster",
    "enrichment": "enrichment (appetite / IG / spend)",
}

ShopKey = Union[int, str]


def _find_shop(ranked: pd.DataFrame, key: ShopKey) -> pd.Series:
    """Locate one shop by rank (int) or by name (exact, else unique substring)."""
    if isinstance(key, (int, np.integer)) or (isinstance(key, str) and key.strip().isdigit()):
        match = ranked[ranked["rank"] == int(key)]
    else:
        names = ranked["place_name"].astype(str).str.lower()
        s = str(key).strip().lower()
        match = ranked[names == s]
        if match.empty:
            match = ranked[names.str.contains(re.escape(s), regex=True)]
    if match.empty:
        raise KeyError(f"no shop matching {key!r}")
    if len(match) > 1:
        raise KeyError(f"{key!r} is ambiguous — matches {list(match['place_name'])}")
    return match.iloc[0]


def pair_breakdown(ranked: pd.DataFrame, a: ShopKey, b: ShopKey,
                   weights: Dict[str, float] = None) -> pd.DataFrame:
    """Component-by-component score breakdown between shops ``a`` and ``b``.

    Each shop can be named or given by rank. Returns one row per scoring component
    plus a TOTAL row, with each shop's *weighted points* (out of 100) and the signed
    ``delta`` (a − b). The deltas sum to the headline score gap, so the ranking is
    inspectable rather than asserted.
    """
    w = weights or WEIGHTS
    ra, rb = _find_shop(ranked, a), _find_shop(ranked, b)

    rows = []
    for comp, weight in w.items():
        pa = weight * float(ra[f"s_{comp}"]) * 100.0
        pb = weight * float(rb[f"s_{comp}"]) * 100.0
        rows.append({
            "component": _COMPONENT_LABELS.get(comp, comp),
            "weight": weight,
            f"{ra['place_name']}": round(pa, 1),
            f"{rb['place_name']}": round(pb, 1),
            "delta": round(pa - pb, 1),
        })
    rows.append({
        "component": "TOTAL", "weight": sum(w.values()),
        f"{ra['place_name']}": round(float(ra["score"]), 1),
        f"{rb['place_name']}": round(float(rb["score"]), 1),
        "delta": round(float(ra["score"]) - float(rb["score"]), 1),
    })
    return pd.DataFrame(rows)


def explain_pair(ranked: pd.DataFrame, a: ShopKey, b: ShopKey,
                 weights: Dict[str, float] = None) -> str:
    """Plain-English 'why A beats B' summary built from :func:`pair_breakdown`.

    Names the winner, the size of the gap, and which components drove it (largest
    absolute deltas first), so you can defend any pairwise call in the visit list.
    """
    w = weights or WEIGHTS
    ra, rb = _find_shop(ranked, a), _find_shop(ranked, b)
    na, nb = ra["place_name"], rb["place_name"]
    gap = float(ra["score"]) - float(rb["score"])

    bd = pair_breakdown(ranked, a, b, w)
    comps = bd[bd["component"] != "TOTAL"].copy()
    comps["abs"] = comps["delta"].abs()
    comps = comps.sort_values("abs", ascending=False)

    if gap == 0:
        head = f"{na} (#{int(ra['rank'])}) and {nb} (#{int(rb['rank'])}) tie at {ra['score']:.1f}."
    else:
        winner, loser, mag = (na, nb, gap) if gap > 0 else (nb, na, -gap)
        head = (f"{winner} beats {loser} by {mag:.1f} pts "
                f"({na} {ra['score']:.1f} @#{int(ra['rank'])} vs "
                f"{nb} {rb['score']:.1f} @#{int(rb['rank'])}).")

    lines = [head, "Component contributions (a − b):"]
    for _, r in comps.iterrows():
        d = r["delta"]
        if d == 0:
            arrow, who = "=", "level"
        else:
            arrow, who = ("▲", na) if d > 0 else ("▼", nb)
        lines.append(f"  {arrow} {r['component']:<34} {d:+5.1f}  "
                     f"({na} {r[na]:.1f} vs {nb} {r[nb]:.1f}) — favours {who}")
    return "\n".join(lines)
