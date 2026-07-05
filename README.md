# Fleek — Physical Store Acquisition

A lean GTM tool for Fleek's physical-store motion, built for the take-home case study.
Two parts, one repo:

- **Part 1 — Which shops do I visit?** Filter a messy 121-row Manchester Google Maps
  scrape down to *genuine* vintage clothing shops, then produce a ranked, explainable
  one-day visit list (with a walking route).
- **Part 2 — How do I reach them, and where next?** Clean a messy 206-row leads &
  customers book, draft personalised outbound per lead stage, and rank cities to go
  after next — all explainable.

> Half-day build, deliberately not over-built: the core done well, the scale and
> enrichment story explained rather than gold-plated.

A plain-English note accompanies every commit under
[`docs/explanations/`](docs/explanations/) — written so you can read one and explain
that commit to someone else. Sample outputs are in [`outputs/samples/`](outputs/samples/).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional — enables real Claude-drafted outbound (falls back to templates without it)
cp .env.example .env      # then paste your ANTHROPIC_API_KEY
```

The case-study workbook lives at
`data/Fleek_-_Physical_Store_Acquisition_-_Pipeline_Data.xlsx`.

## Run

```bash
python run_part1.py                 # ranked Manchester visit list + walking route -> outputs/
python run_part2.py --as-of 2026-07-05   # cleaned book, outbound drafts, city ranking -> outputs/
streamlit run src/app.py            # interactive tour of both parts (map, drafts, city chart)
pytest                              # 102 tests: classifier / dates / dedupe / ranking / outbound
```

`run_part2.py` uses Claude when `ANTHROPIC_API_KEY` is set, otherwise labelled
deterministic templates (`--no-llm` forces templates; `--only-due` drafts only leads
due a touch).

## What the tool decides, and why

### Genuine vintage classifier — [`src/common/vintage_classify.py`](src/common/vintage_classify.py)

Shared by **both** parts. Scores a place from **review text (strong) + category
(weak prior) + name (weakest)** into `genuine` / `ambiguous` / `not_vintage`, with a
confidence, a signed score, and the list of fired signals.

- **Multiple signals, not a keyword filter.** A positive *name* ("Vintage Wines",
  "Vintage Barber", "Retro Games") can't outvote a wine/barber/games review — that's
  what kills the name-traps. A genuine denim review carries a shop filed under a
  generic "Clothing store" / "Boutique". Both directions are tested.
- **Lexicon-based, not a lookup of this sheet's strings** — so it transfers to a real
  scrape instead of overfitting to the synthetic templates (and has something to say
  in the scale story).
- **A real middle ground.** "Decent secondhand finds" has no clothing tell, so it's
  `ambiguous`, not a coin-flip. `no clothes` / `not clothing` force `not_vintage`.
- On the real scrape: **34 genuine / 6 ambiguous / 81 not-vintage**.

### Part 1 ranking — [`src/part1/`](src/part1/)

Classifier gates (genuine only); ranking sorts on the signals the brief names, each
normalised then weighted into a 0–100 score with a plain-English reason:

| signal | weight | why |
| --- | --- | --- |
| rating | 0.30 | quality |
| log(review_count) | 0.25 | size / footfall / appetite |
| price positioning | 0.10 | mid/upmarket buys more than pure budget |
| cluster density | 0.20 | walkable zone → efficient day |
| enrichment | 0.15 | stock appetite / storefront size / real IG + spend / website |

Shops are grouped into **walkable zones** (haversine, single-linkage, 500m) and the
densest zone gets a nearest-neighbour **walking route**.

**Enrichment — pushed past the tab, honestly.** The brief grades *how far you push
enrichment*. The catch: this workbook is synthetic, so the `website` domains don't
resolve and live IG/Street-View scraping would no-op. So rather than fake it, we mine
the enrichment that's genuinely present and, for a *bulk buyer* like Fleek, the most
decision-relevant of all — **how much stock a shop moves** ([`enrich.py`](src/part1/enrich.py)):

- **Stock appetite & storefront size from `top_review` text** — reviews like "three
  floors — massive shop" or "they get huge deliveries weekly" reveal a shop that
  *buys* a lot. Matched on transferable size/turnover vocabulary (not this file's
  literal templates), so it carries to a real scrape.
- **CRM join** — 3 Manchester shops in the scrape already sit in the Part2 book, so we
  pull their **real IG follower counts (14k–29k) and estimated monthly spend** in
  directly, keyed by name + city. (This lifts *Patina Store* and *Golden Era Goods*.)
- **Website presence** — a mild maturity signal, presence only (no fetch).

The **network `provider` seam is preserved**: point it at a real city with live
domains, implement `provider(row) -> EnrichmentSignals` filling IG activity /
multi-location / Google busyness, and ranking is unchanged. A `_stub_provider`
(all-None → enrichment 0) is kept as a control and is unit-tested to prove ranking is
unaffected when enrichment is absent.

Top of the list: **Second Rail** (4.8★, 684 reviews, 7 shops within 500m), **Second
Corner**, **Analog Archive**.

### Part 2 cleaning — [`src/part2/clean.py`](src/part2/clean.py)

Nothing downstream holds up until this runs, so it's done unprompted:

- **Stages** — ~39 spellings → 10 funnel stages (explicit map + keyword fallback for
  unseen spellings, so a 40th spelling degrades gracefully instead of dropping a lead).
- **Channel** — inferred from the *data* (address/geo → physical; reseller scrape
  fields / IG-only → online), **not** `lead_channel_label` (blank 86/206, sometimes
  wrong). Clean split: 130 physical / 62 online.
- **Dates** — five formats parsed; future-dated cells dropped as corrupt (no negative
  recency leaking into copy).
- **Dedupe** — blocked candidate generation (IG handle / address / name-prefix+city)
  + fuzzy name match **plus a corroborating signal**, so "Reclaimed Interiors" in
  London and LA stay separate while true duplicates merge (206 → 192, no cross-city
  merges). Keeps the most-complete row and the freshest stage.

### Part 2 outbound — [`src/part2/outbound.py`](src/part2/outbound.py)

Hybrid: **deterministic** archetype + brief selection (rule-based, tested — the LLM
never picks the strategy), then **Claude** (`claude-opus-4-8`) drafts the prose. Tone
changes with the situation — cold intro vs a skeptical owner (the draft answers the
*specific* objection found in the notes: price-vs-wholesalers, "for small resellers",
bad past experience, wants-volume) vs win-back (acknowledges the time gap) vs
active-customer check-in. **No API key → labelled deterministic templates**, so it
runs end-to-end either way.

### Part 2 city ranking — [`src/part2/city_rank.py`](src/part2/city_rank.py)

Reuses the shared classifier to filter the book to genuine, located vintage shops
(so a city of antiques doesn't read as a vintage opportunity), then scores each city
on density (0.30) + spend potential (0.25) + warm pipeline (0.15) + won (0.10) +
whitespace (0.15) − headwind (0.10), with a per-city reason.

Result: **London #1** (44 shops, £131k/mo, but 20% headwind), **Brighton #2** (small
but 86% warm pipeline — most receptive), **Manchester #3** (100% whitespace — pure
greenfield). London is the biggest prize; the reasoning shows *why*, and where the
warmer / more untapped bets are.

## Building it so it wouldn't fall over at 30,000 rows

- **One data seam** — [`src/common/io.py`](src/common/io.py) is the only place the
  workbook is read; swapping the source to CSV / DuckDB / Postgres is a one-file change.
- **Vectorised pandas throughout**; the only per-row Python loop is the LLM draft step.
- **Dedupe uses blocking keys** (IG handle / address / name+city), never an O(n²)
  all-pairs compare; clustering would swap its all-pairs matrix for a
  `DBSCAN(metric="haversine")` / geohash grid at scale (same function signature).
- **LLM drafting** runs under a bounded `ThreadPoolExecutor`, and `due_for_touch`
  redrafts only leads actually due a touch, not the whole table each run.
- **Sourcing at scale** (debrief): the Manchester scrape would be *generated* per city
  on demand — Google Places / OpenStreetMap for the shop list, then the same
  enrichment interface (IG, site, Places details) on a schedule — feeding the same
  classifier and ranker unchanged.

## Layout

```
src/common/     io.py · dates.py · vintage_classify.py   (shared)
src/part1/      cluster.py · rank.py · enrich.py · pipeline.py
src/part2/      clean.py · outbound.py · city_rank.py · pipeline.py
src/app.py      Streamlit UI
run_part1.py    run_part2.py                              (CLIs)
tests/          102 tests over the real sheet + synthetic fixtures
docs/explanations/   one plain-English note per commit
outputs/samples/     a committed sample of each report
```
