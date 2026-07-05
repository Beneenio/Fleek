"""City prioritisation: which city should Fleek go after next, and why.

Reuses the SAME vintage classifier as Part 1 (on `store_name` +
`google_maps_category` + `note_bio`) so the messy book's non-vintage noise
(antiques, bric-a-brac, furniture, charity) is filtered out *before* we count
"genuine shops per city" — otherwise a city full of antique shops would look like
a great vintage-clothing opportunity. Online resellers carry no address/city, so a
city ranking is naturally about the located physical shops.

Each city is scored on the signals the brief calls out — density of genuine shops,
spend potential, stage mix (warm pipeline vs cold whitespace vs won proof vs
lost/churned headwind) — into a weighted, explainable score with a plain-English
reason per city. See docs/explanations/06-city-ranking.md.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.common.vintage_classify import classify_frame

# Weights: positives sum to 0.95, headwind is a 0.10 penalty.
CITY_WEIGHTS: Dict[str, float] = {
    "density": 0.30,      # count of genuine shops (the core opportunity size)
    "spend": 0.25,        # total monthly spend potential
    "warm": 0.15,         # active pipeline (replied/in convo/negotiating/meeting)
    "won": 0.10,          # proof it already converts in that city
    "whitespace": 0.15,   # never-contacted share = room to grow
    "headwind": 0.10,     # lost/churned/skeptical share = penalty
}

_WARM_STAGES = {"replied", "in_conversation", "negotiating", "meeting_booked"}
_HEADWIND_STAGES = {"lost", "churned", "skeptical"}


def _share(series: pd.Series, members) -> float:
    if len(series) == 0:
        return 0.0
    if isinstance(members, set):
        return float(series.isin(members).mean())
    return float((series == members).mean())


def aggregate_cities(genuine: pd.DataFrame) -> pd.DataFrame:
    """One row per city with the raw signals (pre-scoring)."""
    rows = []
    for city, g in genuine.groupby("city"):
        spend = pd.to_numeric(g["est_monthly_spend_gbp"], errors="coerce")
        stages = g["stage"]
        won = int((stages == "won").sum())
        lost = int((stages == "lost").sum())
        rows.append({
            "city": city,
            "country": g["country"].dropna().iloc[0] if g["country"].notna().any() else None,
            "n_genuine": len(g),
            "total_spend_gbp": float(spend.fillna(0).sum()),
            "avg_spend_gbp": float(spend.mean()) if spend.notna().any() else 0.0,
            "warm_share": _share(stages, _WARM_STAGES),
            "won_share": _share(stages, "won"),
            "whitespace_share": _share(stages, "never_contacted"),
            "headwind_share": _share(stages, _HEADWIND_STAGES),
            "n_won": won,
            "n_lost": lost,
            "win_rate": (won / (won + lost)) if (won + lost) > 0 else None,
        })
    return pd.DataFrame(rows)


def _reason(row) -> str:
    parts = [f"{int(row['n_genuine'])} genuine shops"]
    parts.append(f"£{row['total_spend_gbp']:,.0f}/mo potential (avg £{row['avg_spend_gbp']:,.0f})")
    parts.append(f"{row['warm_share']*100:.0f}% warm pipeline")
    parts.append(f"{row['whitespace_share']*100:.0f}% never contacted (whitespace)")
    if row["won_share"] > 0:
        wr = f", win-rate {row['win_rate']*100:.0f}%" if row["win_rate"] is not None else ""
        parts.append(f"{row['won_share']*100:.0f}% already won{wr}")
    if row["headwind_share"] > 0:
        parts.append(f"{row['headwind_share']*100:.0f}% lost/churned/skeptical (headwind)")
    return " · ".join(parts)


def rank_cities(deduped: pd.DataFrame, include_ambiguous: bool = False,
                min_shops: int = 1, weights: Dict[str, float] = None) -> pd.DataFrame:
    """Filter to genuine located vintage shops, aggregate per city, score & sort."""
    w = weights or CITY_WEIGHTS

    classified = classify_frame(deduped, name_col="store_name",
                                category_col="google_maps_category", review_col="note_bio")
    labels = ["genuine", "ambiguous"] if include_ambiguous else ["genuine"]
    genuine = classified[classified["vintage_label"].isin(labels) &
                         classified["city"].notna()].copy()

    agg = aggregate_cities(genuine)
    agg = agg[agg["n_genuine"] >= min_shops].reset_index(drop=True)
    if agg.empty:
        return agg

    # normalise the two count-scale signals against the strongest city
    agg["density_n"] = agg["n_genuine"] / agg["n_genuine"].max()
    agg["spend_n"] = agg["total_spend_gbp"] / agg["total_spend_gbp"].max()

    agg["score"] = (
        w["density"] * agg["density_n"]
        + w["spend"] * agg["spend_n"]
        + w["warm"] * agg["warm_share"]
        + w["won"] * agg["won_share"]
        + w["whitespace"] * agg["whitespace_share"]
        - w["headwind"] * agg["headwind_share"]
    ).clip(lower=0) * 100.0
    agg["score"] = agg["score"].round(1)

    agg["reason"] = [_reason(r) for _, r in agg.iterrows()]
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)
    return agg
