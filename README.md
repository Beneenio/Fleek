# Fleek — Physical Store Acquisition

A lean GTM tool for Fleek's physical-store motion, built for the take-home case study.
Two parts, one repo:

- **Part 1 — Which shops do I visit?** Filter a messy 121-row Manchester Google Maps
  scrape down to *genuine* vintage clothing shops, then produce a ranked, explainable
  one-day visit list.
- **Part 2 — How do I reach them, and where next?** Clean a messy 206-row leads &
  customers book, draft personalised outbound per lead stage, and rank cities to go
  after next — all explainable.

> Half-day build, deliberately not over-built: the core done well, the scale and
> enrichment story explained rather than gold-plated.

## Status

Scaffold in place. Pipeline modules land commit-by-commit — see
[`docs/explanations/`](docs/explanations/) for a plain-English note per commit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional — enables real Claude-drafted outbound (falls back to templates without it)
cp .env.example .env   # then paste your ANTHROPIC_API_KEY
```

The case-study workbook lives at
`data/Fleek_-_Physical_Store_Acquisition_-_Pipeline_Data.xlsx`.

## Run

```bash
python run_part1.py        # -> ranked Manchester visit list (outputs/)
python run_part2.py        # -> cleaned book, outbound drafts, city ranking (outputs/)
streamlit run src/app.py   # interactive tour of both parts
pytest                     # classifier / date-parser / dedupe / ranking tests
```

_(Entrypoints and UI are added in later commits — see build order below.)_

## How it's built

| Piece | Where |
| --- | --- |
| Data loading (one seam over the source) | `src/common/io.py` |
| Messy-date parsing (5 formats + fallback) | `src/common/dates.py` |
| Shared "is this a genuine vintage clothing shop?" classifier | `src/common/vintage_classify.py` |
| Part 1: cluster → rank → enrich (stub) → pipeline | `src/part1/` |
| Part 2: clean/dedupe → outbound → city ranking | `src/part2/` |
| Streamlit UI | `src/app.py` |

### Design notes

_To be finalised in the last commit: classifier design, ranking weights, outbound
archetypes, city-ranking logic, and the 200 → 30,000-row scale story._

## Build order

1. Scaffold — structure, `common/io.py`, `common/dates.py`, README skeleton.
2. Shared vintage classifier + tests.
3. Part 1 — cluster, rank, enrich (stub), pipeline.
4. Part 2 — cleaning (stages, dedupe, dates, channel).
5. Part 2 — outbound (archetypes + Claude + template fallback).
6. Part 2 — city ranking.
7. Streamlit app.
8. README finalise + sample outputs.
