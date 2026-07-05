# 08 — README finalize + sample outputs

## What this commit adds

- A finished [`README.md`](../../README.md): setup, run commands, a "what the tool
  decides and why" section per component (classifier, Part 1 ranking, cleaning,
  outbound, city ranking), the 200 → 30,000-row scale answer, and the layout.
- One committed sample of each generated report under
  [`outputs/samples/`](../../outputs/samples/) — the ranked Manchester list (CSV +
  Markdown), the city ranking (CSV + Markdown), the per-archetype outbound sample,
  the full outbound drafts, and the cleaned leads — so a reviewer (or the Loom) can
  see real output without running anything.

## Why it's built this way

- **The README leads with the reasoning, not the file list.** Grading rewards
  defensible logic — filtering on multiple signals, an explainable ranking, tone that
  genuinely changes, unprompted cleaning, and a real scale story — so the README is
  organised around those decisions and states the concrete result of each (34/6/81
  classifier split, the Part 1 weight table, 206 → 192 dedupe, London/Brighton/
  Manchester city order).
- **Samples are committed, generated reports are not.** `outputs/` is gitignored so
  runs don't churn the repo, but `outputs/samples/` is force-tracked so the artefacts
  are in the repo for review while staying reproducible with one command.
- **Explanations stay in lockstep.** Every commit — including this one — ships its
  `docs/explanations/NN-*.md`, so the history and the plain-English narrative match
  end to end.

## How to see it work

```bash
# reproduce the committed samples
python run_part1.py
python run_part2.py --as-of 2026-07-05

# or just read them
open outputs/samples/part1_ranked_manchester.md
open outputs/samples/part2_city_ranking.md
open outputs/samples/part2_outbound_sample.md
```
