"""Streamlit UI tying both parts together.

    streamlit run src/app.py

Tab 1 — Manchester ranked visit list + map + per-shop reasoning.
Tab 2 — City prioritisation chart + lead browser (draft + the brief behind it)
        + cleaning stats.

The heavy lifting lives in the pipelines; this is a thin, cached view over them.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# allow `streamlit run src/app.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.part1.pipeline import run as run_part1, suggested_route
from src.part2.pipeline import run as run_part2
from src.part2.outbound import get_client

st.set_page_config(page_title="Fleek — Physical Store Acquisition", layout="wide")

AS_OF = dt.date(2026, 7, 5)  # the workbook is a mid-2026 snapshot


@st.cache_data(show_spinner=False)
def part1_ranked(include_ambiguous: bool):
    return run_part1(include_ambiguous=include_ambiguous, write=False)


@st.cache_data(show_spinner=False)
def part2_all(use_llm: bool):
    deduped, drafts, city, stats, merge_log = run_part2(
        as_of=AS_OF, use_llm=use_llm, write=False)
    return deduped, drafts, city, stats, merge_log


st.title("Fleek — Physical Store Acquisition")
st.caption("Filter → rank vintage shops to visit (Part 1); clean → draft outbound → "
           "rank cities (Part 2).")

tab1, tab2 = st.tabs(["🧭 Part 1 — Manchester visits", "📣 Part 2 — Outreach & cities"])

# ---------------------------------------------------------------- Part 1 -----
with tab1:
    include_amb = st.toggle("Include 'ambiguous' shops (uncertain, no clothing tell)",
                            value=False,
                            help="The classifier's middle ground — secondhand shops "
                                 "with no clothing-specific signal.")
    ranked = part1_ranked(include_amb)

    c1, c2, c3 = st.columns(3)
    c1.metric("Genuine shops ranked", len(ranked))
    c2.metric("Walkable zones", int(ranked["cluster"].nunique()))
    top = ranked.iloc[0]
    c3.metric("Top pick", top["place_name"], f"score {top['score']:.0f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Where they are")
        plot = ranked.copy()
        plot["size"] = pd.to_numeric(plot["review_count"], errors="coerce").fillna(1).clip(lower=1)
        fig = px.scatter_map(
            plot, lat="lat", lon="lng", color="score", size="size",
            hover_name="place_name",
            hover_data={"rating": True, "review_count": True, "cluster": True,
                        "lat": False, "lng": False, "size": False},
            color_continuous_scale="Viridis", size_max=22, zoom=11,
            map_style="carto-positron", height=520)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Ranked visit list")
        st.dataframe(
            ranked[["rank", "score", "place_name", "rating", "review_count",
                    "price_level", "cluster"]],
            hide_index=True, height=520, width="stretch")

    st.subheader("Why this shop?")
    pick = st.selectbox("Pick a shop", ranked["place_name"].tolist(), key="p1pick")
    row = ranked[ranked["place_name"] == pick].iloc[0]
    st.markdown(f"**#{int(row['rank'])} · {pick}** — score **{row['score']:.1f}**  \n"
                f"{row['reason']}")
    comp = {
        "rating": row["s_rating"], "reviews": row["s_reviews"],
        "price": row["s_price"], "density": row["s_density"],
        "enrichment": row["s_enrichment"],
    }
    cfig = px.bar(x=list(comp.values()), y=list(comp.keys()), orientation="h",
                  range_x=[0, 1], labels={"x": "sub-score (0–1)", "y": ""}, height=240)
    cfig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(cfig, use_container_width=True)

    order, metres, _ = suggested_route(ranked, zone=0)
    if order:
        names = " → ".join(ranked.loc[i, "place_name"] for i in order)
        st.info(f"**Suggested walking route (densest zone, ~{metres/1000:.1f} km):** {names}")

# ---------------------------------------------------------------- Part 2 -----
with tab2:
    has_key = get_client() is not None
    use_llm = st.toggle(
        "Draft with Claude (needs ANTHROPIC_API_KEY)", value=has_key, disabled=not has_key,
        help="Off → labelled deterministic templates (same archetype logic, no API key).")
    deduped, drafts, city, stats, merge_log = part2_all(use_llm and has_key)

    st.subheader("Cleaning — done before anything else holds up")
    m = st.columns(5)
    m[0].metric("Rows in → out", f"{stats['rows_in']} → {stats['rows_out']}")
    m[1].metric("Duplicates merged", stats["duplicates_removed"])
    m[2].metric("Physical / online",
                f"{stats['channel_distribution'].get('physical', 0)} / "
                f"{stats['channel_distribution'].get('online', 0)}")
    m[3].metric("Channel label blank", f"{stats['channel_label_blank']}/{stats['rows_in']}")
    m[4].metric("Drafting mode", stats.get("drafting_mode", "template"))

    st.subheader("Which city next?")
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        cfig = px.bar(city.sort_values("score"), x="score", y="city", orientation="h",
                      color="score", color_continuous_scale="Viridis",
                      labels={"score": "priority score", "city": ""}, height=420)
        cfig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
        st.plotly_chart(cfig, use_container_width=True)
    with cc2:
        st.dataframe(
            city[["rank", "city", "country", "n_genuine", "total_spend_gbp",
                  "score", "reason"]],
            hide_index=True, height=420, width="stretch")

    st.subheader("Lead browser — draft + the brief behind it")
    # Only verified stores that actually receive a message belong in the browser;
    # online resellers and non-vintage shops (skipped_*) are excluded here.
    messaged = drafts[~drafts["draft_source"].astype(str).str.startswith("skipped")]
    skipped_n = len(drafts) - len(messaged)
    st.caption(f"{len(messaged)} verified leads receiving a message "
               f"({skipped_n} online/non-vintage leads excluded from outreach).")
    order_stage = ["never_contacted", "contacted", "replied", "in_conversation",
                   "negotiating", "meeting_booked", "skeptical", "won", "churned", "lost"]
    fcol1, fcol2 = st.columns(2)
    arch_filter = fcol1.multiselect("Filter by archetype",
                                    sorted(messaged["archetype"].unique()),
                                    default=[])
    view = messaged if not arch_filter else messaged[messaged["archetype"].isin(arch_filter)]
    view = view.reset_index(drop=True)
    view["label"] = (view["store_name"].astype(str) + " — " + view["city"].astype(str)
                     + " (" + view["archetype"] + ")")
    pick2 = fcol2.selectbox("Pick a lead", view["label"].tolist(), key="p2pick")
    lead = view[view["label"] == pick2].iloc[0]

    dcol, bcol = st.columns([3, 2])
    with dcol:
        st.markdown(f"**Archetype:** `{lead['archetype']}`  ·  "
                    f"**source:** `{lead['draft_source']}`")
        st.markdown(f"**Subject:** {lead['draft_subject']}")
        st.write(lead["draft_body"])
    with bcol:
        st.markdown("**The brief that drove it**")
        brief = json.loads(lead["brief_json"])
        st.json({k: v for k, v in brief.items() if v not in (None, "")})
