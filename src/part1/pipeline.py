"""Part 1 pipeline: raw scrape -> ranked Manchester visit list + a walking route.

    load -> classify/filter -> enrich(stub) -> cluster(density+zones) -> rank -> write

Outputs (under outputs/):
- part1_ranked_manchester.csv  — full ranked genuine list with score components + reason
- part1_ranked_manchester.md   — readable top list + a suggested walking route for the
  densest zone (the natural anchor for a day of visits)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.io import REPO_ROOT, load_part1_df, load_part2_df
from src.common.vintage_classify import classify_frame
from src.part1.enrich import enrich_frame, Provider
from src.part1.cluster import assign_clusters, nearest_neighbour_route
from src.part1.rank import rank_frame

OUTPUT_DIR = REPO_ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "part1_ranked_manchester.csv"
MD_PATH = OUTPUT_DIR / "part1_ranked_manchester.md"

# Columns surfaced in the CSV (keep it readable, not the whole internal frame).
_CSV_COLS = [
    "rank", "score", "place_name", "maps_category", "rating", "review_count",
    "price_level", "cluster", "neighbours",
    "enr_stock_appetite", "enr_storefront_size", "enr_instagram_followers",
    "enr_est_monthly_spend", "enr_has_website", "enr_source",
    "full_address", "postcode", "website", "phone", "vintage_confidence", "reason",
]


def run(path: Optional[str] = None, provider: Optional[Provider] = None,
        include_ambiguous: bool = False, radius_m: float = 500.0,
        write: bool = True) -> pd.DataFrame:
    raw = load_part1_df(path)

    classified = classify_frame(raw)
    labels = ["genuine", "ambiguous"] if include_ambiguous else ["genuine"]
    genuine = classified[classified["vintage_label"].isin(labels)].copy()

    # Offline enrichment (review-text stock/size + website presence), joined to the
    # Part2 CRM book for real IG followers + spend where a scraped shop is already a
    # known lead. Missing book (or a custom `provider`) degrades gracefully.
    crm_df = None
    if provider is None:
        try:
            crm_df = load_part2_df()
        except FileNotFoundError:
            crm_df = None
    enriched = enrich_frame(genuine, provider=provider, crm_df=crm_df)

    cluster, neighbours = assign_clusters(enriched, radius_m=radius_m)
    enriched["cluster"] = cluster
    enriched["neighbours"] = neighbours

    ranked = rank_frame(enriched)

    if write:
        _write_outputs(ranked, radius_m=radius_m)
    return ranked


def suggested_route(ranked: pd.DataFrame, zone: int = 0):
    """Nearest-neighbour walking order through one zone, starting at its top shop."""
    zone_df = ranked[ranked["cluster"] == zone]
    if zone_df.empty:
        return [], 0.0, zone_df
    start = zone_df.sort_values("score", ascending=False).index[0]
    order, metres = nearest_neighbour_route(zone_df, start_index=start)
    return order, metres, zone_df.loc[order]


def _write_outputs(ranked: pd.DataFrame, radius_m: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [c for c in _CSV_COLS if c in ranked.columns]
    ranked[cols].to_csv(CSV_PATH, index=False)

    order, metres, route_df = suggested_route(ranked, zone=0)
    lines = []
    lines.append("# Manchester — ranked vintage-shop visit list\n")
    lines.append(f"_{len(ranked)} genuine vintage clothing shops, "
                 f"ranked for a day of visits. Walkable zones = shops within "
                 f"{int(radius_m)}m of each other._\n")

    lines.append("## Top 15\n")
    lines.append("| # | Score | Shop | Rating | Reviews | Price | Zone | Why |")
    lines.append("|---|------|------|--------|---------|-------|------|-----|")
    for _, r in ranked.head(15).iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['score']:.1f} | {r['place_name']} | "
            f"{r['rating']}★ | {int(r['review_count'])} | "
            f"{r['price_level'] if isinstance(r['price_level'], str) else '—'} | "
            f"{int(r['cluster'])} | {r['reason']} |"
        )

    if order:
        lines.append("\n## Suggested walking route — densest zone (zone 0)\n")
        lines.append(f"_{len(order)} shops, ~{metres/1000:.1f} km on foot, "
                     f"starting at the top-ranked shop in the zone._\n")
        for i, idx in enumerate(order, 1):
            r = ranked.loc[idx]
            lines.append(f"{i}. **{r['place_name']}** — score {r['score']:.1f}, "
                         f"{r['rating']}★ ({int(r['review_count'])} reviews)")

    MD_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
