"""Part 1 tests: haversine/cluster geometry, enrichment wiring, rank sanity."""
import pandas as pd

from src.part1.cluster import haversine_km, assign_clusters, nearest_neighbour_route
from src.part1.enrich import EnrichmentSignals
from src.part1.rank import rank_frame, price_score, reviews_score
from src.part1.pipeline import run


def test_haversine_known_distance():
    # Manchester Piccadilly -> MediaCityUK is ~3.9 km
    d = haversine_km(53.4775, -2.2310, 53.4720, -2.2970)
    assert 3.5 < d < 4.5


def test_clusters_group_nearby_and_count_neighbours():
    # three shops ~100m apart, one ~5km away
    df = pd.DataFrame({
        "lat": [53.4800, 53.4805, 53.4810, 53.5200],
        "lng": [-2.2400, -2.2405, -2.2410, -2.2000],
        "place_name": list("ABCD"),
    })
    cluster, neighbours = assign_clusters(df, radius_m=500)
    assert cluster.iloc[0] == cluster.iloc[1] == cluster.iloc[2]   # A,B,C together
    assert cluster.iloc[3] != cluster.iloc[0]                       # D on its own
    assert neighbours.iloc[0] == 2 and neighbours.iloc[3] == 0


def test_route_orders_and_measures():
    df = pd.DataFrame({
        "lat": [53.4800, 53.4805, 53.4810],
        "lng": [-2.2400, -2.2405, -2.2410],
    }, index=["A", "B", "C"])
    order, metres = nearest_neighbour_route(df, start_index="A")
    assert order[0] == "A" and set(order) == {"A", "B", "C"}
    assert metres > 0


def test_price_and_reviews_scores_monotonic():
    assert price_score("££") > price_score("£")
    assert reviews_score(800) > reviews_score(50) > reviews_score(0)


def _fixture_provider(row):
    if row.get("place_name") == "Revival Denim":
        return EnrichmentSignals(instagram_followers=42000, instagram_active=True,
                                 num_locations=3, storefront_size="large",
                                 popularity=0.8, source="fixture")
    return EnrichmentSignals(source="stub")


def test_enrichment_is_wired_zero_under_stub_and_lifts_with_data():
    base = run(write=False)
    assert (base["s_enrichment"] == 0).all()          # stub contributes nothing

    enriched = run(write=False, provider=_fixture_provider)
    base_rank = int(base.set_index("place_name").loc["Revival Denim", "rank"])
    enr_rank = int(enriched.set_index("place_name").loc["Revival Denim", "rank"])
    assert enr_rank < base_rank                        # enrichment moved it up
    assert enriched.set_index("place_name").loc["Revival Denim", "s_enrichment"] > 0


def test_pipeline_ranks_only_genuine_and_is_sorted():
    ranked = run(write=False)
    assert len(ranked) == 34
    assert ranked["vintage_is_genuine"].all()
    assert ranked["score"].is_monotonic_decreasing
    # a known strong shop should be near the top
    top10 = set(ranked.head(10)["place_name"])
    assert "Second Rail" in top10 or "Second Corner" in top10
