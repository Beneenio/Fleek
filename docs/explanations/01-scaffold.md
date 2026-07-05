# 01 — Scaffold

## What this commit adds

The skeleton of the repo: directory layout (`src/common`, `src/part1`, `src/part2`,
`tests`, `outputs`, `docs/explanations`), `requirements.txt`, `.env.example`,
`.gitignore`, a README skeleton, and the two foundation modules everything else
builds on:

- **`src/common/io.py`** — `load_part1_df()` / `load_part2_df()`, the *only* place
  the workbook is opened.
- **`src/common/dates.py`** — `parse_messy_date()`, which turns the five date
  formats in the sheet (and blanks) into real `date` objects.

## Why it's built this way

- **One data seam (`io.py`).** Downstream code never touches the `.xlsx`. When the
  input grows from a 200-row scrape to 30,000 rows out of a database, the swap is
  contained to one file instead of rippling through every module — that's the
  "wouldn't fall over at 30k rows" requirement, handled structurally up front.
- **Format-first date parsing, with a fallback.** We try the five *known* formats
  explicitly (fast and unambiguous), then fall back to a day-first `dateutil` parse
  so an unseen sixth format degrades to a best-effort date rather than crashing the
  run. Year-less dates like `14 May` inherit a sensible default year. Blanks become
  `None` (never-contacted / no purchase), not a crash.
- **Template fallback signalled from the start.** `.env.example` documents that the
  tool runs end-to-end without an API key — outbound just uses labelled templates —
  so a reviewer can clone and run it immediately.

## How to see it work

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -c "from src.common.io import load_part1_df, load_part2_df; \
print('part1', load_part1_df().shape, '| part2', load_part2_df().shape)"
# -> part1 (121, 14) | part2 (206, 24)

python -c "from src.common.dates import parse_messy_date as p; \
print(p('2026-04-28'), p('14 May'), p('June 4 2026'), p('19/06/2026'), \
p('2026/05/21'), p(''), p(None))"
# -> 2026-04-28 2026-05-14 2026-06-04 2026-06-19 2026-05-21 None None
```
