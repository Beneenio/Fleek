# 03 — Part 1 ranking

## What this commit adds

The full Part 1 pipeline — from the raw scrape to a ranked, explainable Manchester
visit list plus a walking route:

- `src/part1/enrich.py` — **real offline enrichment**: stock appetite & storefront
  size mined from `top_review` text, a **CRM join** to the Part2 book for real IG
  followers + monthly spend on shops we already know, and website presence — plus a
  preserved `provider` seam for live IG/multi-location/Google-busyness scraping.
- `src/part1/cluster.py` — haversine distances, single-linkage **walkable zones**
  (shops within 500m), per-shop neighbour counts (density signal), and a
  nearest-neighbour **route** with total walking distance.
- `src/part1/rank.py` — weighted, explainable score (0–100) + a plain-English
  `reason` per shop.
- `src/part1/pipeline.py` + `run_part1.py` — `load → classify/filter → enrich(stub)
  → cluster → rank → write`, emitting `outputs/part1_ranked_manchester.csv` / `.md`.
- `tests/test_part1.py` — geometry, enrichment wiring, and rank sanity.

On the real data it ranks the **34 genuine** shops; top of the list is Second Rail
(4.8★, 684 reviews, 7 shops within 500m) and Second Corner (4.7★, 772 reviews,
upmarket).

## Why it's built this way

- **Classifier gates, ranking sorts.** Only genuine shops reach the ranker, so the
  score is about "worth my day", not "is it real". The two concerns stay separate
  and independently testable.
- **Signals the brief names, each normalised then weighted** — rating (quality,
  0.30), log-scaled review count (footfall/appetite, 0.25), price positioning
  (0.10), cluster density (walkable → efficient day, 0.20), enrichment (0.15).
  Weights are one dict at the top of `rank.py`, easy to defend and tune in the
  debrief.
- **Density is a real signal, not decoration.** A shop with several genuine
  neighbours earns rank *and* seeds the day plan — you can walk between them. That's
  why clustering feeds both the score and the route.
- **Enrichment pushed past the tab — honestly.** The workbook is synthetic (fake
  domains), so live IG/site scraping would no-op; instead we mine the signal that *is*
  present and matters most to a bulk buyer — **how much stock a shop moves** (review
  text: "three floors", "huge deliveries weekly") — and **join the scrape to our CRM
  book** to pull real IG followers (14k–29k) and monthly spend for the 3 shops already
  in it. A `_stub_provider` (all-None → 0) is kept as a tested control, and the
  network `provider` seam still lets a real-city run fill live signals with no ranking
  change.
- **Explainable end to end.** Each row carries its component sub-scores and a reason
  string ("excellent 4.8★ on 684 reviews · mid-range (££) · dense cluster — 7 within
  500m · genuine: sportswear, football shirt"), so "why A beats B" is inspectable.

## How to see it work

```bash
python run_part1.py                 # top 10 + suggested route, writes outputs/
python run_part1.py --ambiguous     # also include the ambiguous middle ground
python -m pytest tests/test_part1.py -q
open outputs/part1_ranked_manchester.md
```

Confirm: known junk (charity/antiques/name-traps) is absent, known genuine shops
(Second Rail, Nomad Racks, Revival Denim, Golden Era Goods…) are ranked with
sensible reasoning, and the route lists the densest zone in walking order.
