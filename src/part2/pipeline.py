"""Part 2 pipeline: messy book -> cleaned/deduped -> outbound drafts -> city ranking.

    load -> clean/dedupe -> (classify channel is part of clean) -> outbound drafts
         -> city ranking -> write outputs

Outputs (under outputs/):
- part2_cleaned_leads.csv    — deduped book with canonical stage, channel, dates
- part2_outbound_drafts.csv  — per-lead archetype + drafted subject/body + brief
- part2_outbound_sample.md   — one draft per archetype (for the Loom), tone side-by-side
- part2_city_ranking.csv/.md — ranked cities with per-city reasoning + cleaning stats
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.io import REPO_ROOT, load_part2_df
from src.part2.clean import clean_and_dedupe
from src.part2.outbound import generate_drafts, get_client
from src.part2.city_rank import rank_cities

OUTPUT_DIR = REPO_ROOT / "outputs"

_SAMPLE_ARCHETYPES = ["cold_intro", "follow_up", "in_conversation", "skeptical",
                      "active_customer", "win_back", "lost"]


def run(path: Optional[str] = None, as_of: Optional[dt.date] = None,
        use_llm: bool = True, only_due: bool = False, write: bool = True):
    """Returns (deduped, drafts, city_ranking, stats, merge_log)."""
    as_of = as_of or dt.date.today()
    raw = load_part2_df(path)

    deduped, merge_log, stats = clean_and_dedupe(raw, as_of=as_of)

    client = get_client() if use_llm else None
    stats["drafting_mode"] = "claude" if client is not None else "template"
    drafts = generate_drafts(deduped, client=client, only_due=only_due)

    city_ranking = rank_cities(deduped)

    if write:
        _write_outputs(deduped, drafts, city_ranking, stats, merge_log)
    return deduped, drafts, city_ranking, stats, merge_log


def _write_outputs(deduped, drafts, city_ranking, stats, merge_log) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lead_cols = ["lead_id", "store_name", "city", "country", "channel", "stage",
                 "days_since_contact", "days_since_purchase", "est_monthly_spend_gbp",
                 "note_bio", "note_context"]
    deduped[[c for c in lead_cols if c in deduped.columns]].to_csv(
        OUTPUT_DIR / "part2_cleaned_leads.csv", index=False)

    draft_cols = ["store_name", "city", "stage", "channel", "archetype",
                  "draft_source", "draft_subject", "draft_body"]
    drafts[[c for c in draft_cols if c in drafts.columns]].to_csv(
        OUTPUT_DIR / "part2_outbound_drafts.csv", index=False)

    city_ranking.to_csv(OUTPUT_DIR / "part2_city_ranking.csv", index=False)

    _write_sample_md(drafts, stats)
    _write_city_md(city_ranking, stats, merge_log)


def _write_sample_md(drafts: pd.DataFrame, stats: dict) -> None:
    lines = ["# Outbound — one draft per archetype\n",
             f"_Drafting mode: **{stats.get('drafting_mode')}** "
             f"(set ANTHROPIC_API_KEY for Claude-written copy)._\n",
             "Tone changes with the lead's situation — cold intro vs skeptical "
             "objection vs win-back vs active-customer check-in.\n"]
    for arch in _SAMPLE_ARCHETYPES:
        sub = drafts[drafts["archetype"] == arch]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(f"## {arch}  ·  {r['store_name']} ({r.get('city','?')}, "
                     f"stage={r['stage']})\n")
        lines.append(f"**Subject:** {r['draft_subject']}\n")
        lines.append(f"{r['draft_body']}\n")
    (OUTPUT_DIR / "part2_outbound_sample.md").write_text("\n".join(lines) + "\n")


def _write_city_md(city: pd.DataFrame, stats: dict, merge_log: pd.DataFrame) -> None:
    lines = ["# City prioritisation — where to go after next\n"]
    lines.append("## Cleaning summary\n")
    lines.append(f"- Rows in: **{stats['rows_in']}** → out: **{stats['rows_out']}** "
                 f"({stats['duplicates_removed']} duplicates merged across "
                 f"{stats['merge_groups']} groups)")
    lines.append(f"- Channel: {stats['channel_distribution']} "
                 f"(inferred from data — `lead_channel_label` blank "
                 f"{stats['channel_label_blank']}/{stats['rows_in']})")
    lines.append(f"- Stage mix: {stats['stage_distribution']}\n")

    lines.append("## Ranked cities\n")
    lines.append("| # | City | Country | Genuine shops | £/mo potential | Score | Why |")
    lines.append("|---|------|---------|---------------|----------------|-------|-----|")
    for _, r in city.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['city']} | {r['country'] or '—'} | "
            f"{int(r['n_genuine'])} | £{r['total_spend_gbp']:,.0f} | {r['score']:.1f} | "
            f"{r['reason']} |")
    (OUTPUT_DIR / "part2_city_ranking.md").write_text("\n".join(lines) + "\n")
