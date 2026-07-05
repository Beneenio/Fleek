"""City-ranking tests: the weighted score reacts correctly to synthetic fixtures,
and non-vintage noise is filtered out before counting shops per city."""
import datetime as dt

import pandas as pd

from src.common.io import load_part2_df
from src.part2.clean import clean_and_dedupe
from src.part2.city_rank import rank_cities, aggregate_cities

_GENUINE_BIO = "Best vintage denim, huge racks of Carhartt and Dickies, band tees"
_ANTIQUE_BIO = "Lovely antiques, china and bric-a-brac, no clothes really"


def _row(city, bio=_GENUINE_BIO, category="Vintage clothing store",
         stage="contacted", spend=1000, country="UK", name=None):
    return {
        "store_name": name or f"{city} Vintage",
        "google_maps_category": category,
        "note_bio": bio,
        "city": city, "country": country,
        "stage": stage, "est_monthly_spend_gbp": spend,
    }


def _frame(rows):
    return pd.DataFrame(rows)


def test_more_genuine_shops_scores_higher():
    rows = [_row("Big", name=f"Big Vintage {i}") for i in range(5)]
    rows += [_row("Small")]
    ranked = rank_cities(_frame(rows))
    assert ranked.iloc[0]["city"] == "Big"
    assert ranked.set_index("city").loc["Big", "score"] > \
           ranked.set_index("city").loc["Small", "score"]


def test_higher_spend_lifts_score():
    rows = [_row("Rich", spend=9000), _row("Poor", spend=100)]
    ranked = rank_cities(_frame(rows)).set_index("city")
    assert ranked.loc["Rich", "score"] > ranked.loc["Poor", "score"]


def test_warm_pipeline_beats_headwind():
    # same size + spend; one city is warm, the other is lost/churned.
    warm = [_row("Warm", stage="in_conversation", name=f"W{i}") for i in range(3)]
    cold = [_row("Cold", stage="lost", name=f"C{i}") for i in range(3)]
    ranked = rank_cities(_frame(warm + cold)).set_index("city")
    assert ranked.loc["Warm", "score"] > ranked.loc["Cold", "score"]


def test_non_vintage_noise_is_filtered_out():
    rows = [_row("Vintageville")]
    # a whole city of antiques/bric-a-brac should not count as vintage opportunity
    rows += [_row("Antiquetown", bio=_ANTIQUE_BIO, category="Antiques store",
                  name=f"Antiques {i}") for i in range(4)]
    ranked = rank_cities(_frame(rows))
    assert "Antiquetown" not in set(ranked["city"])
    assert "Vintageville" in set(ranked["city"])


def test_aggregate_computes_shares_and_win_rate():
    rows = [
        _row("X", stage="won"), _row("X", stage="lost"),
        _row("X", stage="never_contacted"), _row("X", stage="in_conversation"),
    ]
    from src.common.vintage_classify import classify_frame
    g = classify_frame(_frame(rows), name_col="store_name",
                       category_col="google_maps_category", review_col="note_bio")
    agg = aggregate_cities(g[g.vintage_label == "genuine"]).set_index("city")
    assert agg.loc["X", "n_genuine"] == 4
    assert agg.loc["X", "win_rate"] == 0.5          # 1 won / (1 won + 1 lost)
    assert abs(agg.loc["X", "whitespace_share"] - 0.25) < 1e-9
    assert abs(agg.loc["X", "warm_share"] - 0.25) < 1e-9


def test_min_shops_filter():
    rows = [_row("Big", name=f"B{i}") for i in range(3)] + [_row("Tiny")]
    ranked = rank_cities(_frame(rows), min_shops=2)
    assert "Tiny" not in set(ranked["city"]) and "Big" in set(ranked["city"])


def test_real_sheet_london_leads_and_output_shape():
    deduped, _, _ = clean_and_dedupe(load_part2_df(), as_of=dt.date(2026, 7, 5))
    ranked = rank_cities(deduped)
    for col in ("rank", "city", "n_genuine", "total_spend_gbp", "score", "reason"):
        assert col in ranked.columns
    assert ranked["score"].is_monotonic_decreasing
    # London has by far the most genuine shops + spend -> should top the list
    assert ranked.iloc[0]["city"] == "London"
    # sanity: every ranked city has at least one genuine shop and a reason
    assert (ranked["n_genuine"] >= 1).all()
    assert ranked["reason"].str.len().gt(0).all()
