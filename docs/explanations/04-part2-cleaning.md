# 04 — Part 2 cleaning

## What this commit adds

`src/part2/clean.py` (+ `tests/test_clean.py`, 32 tests) — the cleaning layer both
Part 2 outputs depend on. It turns the messy book into something outbound and city
ranking can trust:

- **Stage canonicalisation** — the ~39 observed spellings fold into 10 funnel
  stages via an explicit map, with a keyword fallback for unseen spellings.
- **Channel inference** — physical vs online from the *data* (address/geo → physical;
  reseller scrape fields / IG-only → online), with a confidence and a
  `channel_label_disagrees` flag. Not from `lead_channel_label`.
- **Dates** — both date columns parsed to real dates, plus `days_since_contact` /
  `days_since_purchase` against an `as_of` date.
- **Notes split** — `"what they sell | rep note"` → `note_bio` + `note_context`.
- **Blocked fuzzy dedupe** — merges true duplicates, keeps the most-complete row,
  logs every merge.

On the real sheet: **206 → 192 rows** (14 duplicates merged), **130 physical / 62
online**, every stage mapped (no "unknown"), and **no merge group spans two cities**.

## Why it's built this way

- **Explicit map + fallback, not one or the other.** The explicit dict is exact and
  auditable for the spellings we've actually seen; the keyword fallback means the
  40th spelling at 30k rows degrades to a sensible bucket instead of crashing or
  silently dropping the lead.
- **Channel is inferred, because the label can't be trusted.** `lead_channel_label`
  is blank **86 of 206 times** and the brief says don't trust it. The data tells the
  truth: a row with an address is a physical shop; a row with `items_listed` /
  `sell_through_rate` (Depop/Vinted/Whatnot scrape fields) and no address is an
  online reseller. That split is clean (144 vs 62 in the raw, zero overlap), so we
  key off it and only *use the label to flag disagreements*.
- **Dedupe needs a corroborating signal, or it over-merges.** "Reclaimed Interiors"
  appears in London *and* Los Angeles — same name, same IG handle, different city
  and address. Name-similarity alone would wrongly merge them. So a match requires a
  fuzzy name hit **plus** one of: same address, same IG-handle *and* city, or geo
  within 60m *and* city. Result: the two LA rows merge, the London one stays — and
  no merge group in the whole sheet crosses a city boundary.
- **Blocking, not all-pairs.** Candidate pairs come from small blocks keyed on IG
  handle, normalised address, and name-prefix+city — so we never do an O(n²)
  comparison. At 30k rows the blocks stay small and the work stays near-linear.
- **Merge keeps the most-complete row and the *freshest* stage.** A duplicate might
  carry a staler "new lead" alongside a newer "Closed Lost"; we keep the stage from
  the most recently contacted row and the most recent dates, so merging never rolls
  a lead backwards.

## How to see it work

```bash
python -m pytest tests/test_clean.py -q

python -c "import datetime as dt; from src.common.io import load_part2_df; \
from src.part2.clean import clean_and_dedupe; \
_,log,stats = clean_and_dedupe(load_part2_df(), as_of=dt.date(2026,7,5)); \
print(stats); print(log[['kept_store_name','merged_count','stages_seen','cities']].to_string())"
# rows 206 -> 192, online 62, label blank 86, and every merge group is single-city
```
