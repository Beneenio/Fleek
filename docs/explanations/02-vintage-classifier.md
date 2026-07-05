# 02 — Vintage classifier

## What this commit adds

`src/common/vintage_classify.py` — the shared "is this genuinely a vintage
clothing shop?" scorer used by **both** parts (Part 1 on `top_review`, Part 2 on
`notes`), plus `tests/test_vintage_classify.py` (29 tests built from real rows).

Given a place's name, category, and representative review, it returns:

- `label` — `genuine` / `ambiguous` / `not_vintage`
- `is_genuine` — bool
- `confidence` — 0–1
- `score` — signed evidence total (explainable)
- `signals` — the human-readable list of what fired ("+review vintage-clothing
  signal: denim, carhartt", "name-trap: 'vintage' in name not backed by
  review/category", …)

On the real Manchester scrape it splits 121 rows into **34 genuine / 6 ambiguous /
81 not-vintage** — in line with the brief's "~70% isn't a vintage clothing shop".

## Why it's built this way

- **Multiple signals, weighted — not a keyword filter.** The brief warns that a
  filter on "vintage" both lets name-traps through and misses genuine shops under
  generic categories. So review text is the strong signal, category a weak prior,
  and the name the weakest. A positive *name* ("Vintage Wines", "Vintage Barber")
  can't outvote a wine/barber review — which is exactly what kills the traps. And a
  genuine denim review carries a shop filed under plain "Clothing store" or
  "Boutique". Both directions are tested.
- **Lexicon-based, not a lookup of this sheet's strings.** Matching the literal
  templates in this synthetic file would score perfectly here and uselessly on a
  real Google scrape — and would leave nothing to say in the "sourcing at scale"
  debrief. The lexicon is transferable vintage-clothing vocabulary (denim, band
  tees, Carhartt/Dickies, streetwear, Y2K, reworked/archive) vs. non-clothing
  vocabulary (bric-a-brac, antiques, homeware, vinyl, costume, barber…).
- **A real middle ground.** Reviews like "decent secondhand finds" are genuinely
  ambiguous — secondhand, but no clothing tell. Forcing them to yes/no would be
  dishonest, so they get their own `ambiguous` label and the ranking layer decides
  whether to include them, rather than the classifier pretending to be sure.
- **Hard disqualifiers.** "no clothes" / "not clothing" in a review forces
  `not_vintage` regardless of an inviting name or category.
- **Explainable by construction.** Every result ships the fired-signal list, so
  "why shop A beats shop B" is inspectable — which is what the ranking and the
  debrief need.

## How to see it work

```bash
python -m pytest tests/test_vintage_classify.py -q      # 29 passed

python -c "from src.common.io import load_part1_df; \
from src.common.vintage_classify import classify_frame; \
df=classify_frame(load_part1_df()); \
print(df['vintage_label'].value_counts())"
# genuine 34 / not_vintage 81 / ambiguous 6

python -c "from src.common.vintage_classify import classify as c; \
r=c('Vintage Barber Co','Barber shop','Best barber in town, proper old-school cuts'); \
print(r.label, r.signals)"
# not_vintage  [... 'name-trap: 'vintage' in name not backed by review/category']
```
