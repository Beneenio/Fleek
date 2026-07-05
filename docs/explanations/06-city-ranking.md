# 06 — City ranking

## What this commit adds

`src/part2/city_rank.py` (+ `tests/test_city_rank.py`, 7 tests), plus the Part 2
runner that ties cleaning → outbound → city ranking together
(`src/part2/pipeline.py`, `run_part2.py`).

`rank_cities(deduped)` filters the book to **genuine, located vintage shops** using
the shared classifier, aggregates per city, and scores each on a weighted,
explainable formula:

- **density** (count of genuine shops) 0.30
- **spend** (total £/mo potential) 0.25
- **warm** (active-pipeline share) 0.15
- **won** (already-converted share) 0.10
- **whitespace** (never-contacted share) 0.15
- **headwind** (lost/churned/skeptical share) −0.10

On the real book it ranks **London #1** (44 shops, £131k/mo, but 20% headwind),
**Brighton #2** (small but 86% warm pipeline), **Manchester #3** (100% whitespace —
pure greenfield) — each with a plain-English reason.

## Why it's built this way

- **Reuses the Part 1 classifier — filter before you count.** The book is full of
  non-vintage noise (antiques, bric-a-brac, furniture, charity). If you counted
  every "shop" per city, a town full of antique dealers would look like a vintage
  goldmine. Running the same lexicon classifier on `store_name` +
  `google_maps_category` + `note_bio` strips that out first, so "genuine shops per
  city" means what it says. One classifier, both parts — that's the systems answer.
- **Stage mix, not just a headcount.** The brief asks for density *and* spend
  potential *and* stage mix (warm replies vs all-cold, how many won/lost). So the
  score rewards a warm pipeline and untapped whitespace, credits cities that already
  convert, and penalises churn/loss headwind. The reason string exposes all of it,
  so "why this city first" is defensible — London is the biggest prize, but you can
  see Brighton is the *warmest* and Manchester is *pure whitespace*.
- **Density and spend are normalised to the strongest city**, so the two count-scale
  signals are comparable and one whale-spend city can't dominate on an absolute
  number alone.
- **Located shops only.** Online resellers have no address/city (channel inference
  already separates them), so a *city* ranking is naturally about the physical shops
  you'd actually go visit — which is the physical-store motion this role owns.

## How to see it work

```bash
python -m pytest tests/test_city_rank.py -q     # 7 passed

python run_part2.py --as-of 2026-07-05          # prints the ranked city table
# writes outputs/part2_city_ranking.csv/.md, cleaned leads, and outbound drafts
open outputs/part2_city_ranking.md
```

Confirm: cities of pure antiques are absent, London leads on density+spend, and the
reason column explains each city's warm/whitespace/headwind mix.
