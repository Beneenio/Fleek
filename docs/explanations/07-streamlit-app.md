# 07 — Streamlit app

## What this commit adds

`src/app.py` — a small Streamlit UI that ties both parts together for the Loom:

- **Tab 1 — Manchester visits.** Metrics (genuine shops, walkable zones, top pick),
  a map (`plotly.scatter_map`, OpenStreetMap tiles — no token) of the genuine shops
  coloured by score and sized by review count, the ranked visit list, a per-shop
  "why this shop?" panel with the reason string + a sub-score breakdown, and the
  suggested walking route for the densest zone. A toggle folds in the classifier's
  `ambiguous` middle ground.
- **Tab 2 — Outreach & cities.** Cleaning stats up top (rows in→out, duplicates
  merged, physical/online split, how often the channel label was blank), the city
  ranking as a bar chart + table with reasons, and a **lead browser**: pick a lead,
  see its drafted subject/body next to the structured brief that produced it. A
  toggle switches drafting between Claude and the deterministic templates (disabled
  when no API key is set).

## Why it's built this way

- **Thin view over the pipelines.** The app calls `run_part1` / `run_part2` and
  renders — no business logic lives here. Everything shown (scores, reasons, drafts,
  city ranking) is the same code `pytest` and the CLIs exercise, so the UI can't
  drift from the tool.
- **Cached runs.** `@st.cache_data` wraps each pipeline call so toggling a control
  re-renders instantly instead of re-running classification/dedupe/drafting every
  interaction.
- **The brief sits next to the draft.** The whole point of Part 2 is that tone
  changes with the situation — showing the archetype and the structured brief beside
  the generated message makes the "why this message" inspectable, not magic.
- **Cleaning is surfaced, not hidden.** The messy-data handling is a graded signal,
  so the stats panel puts the dedupe count and the "label blank 86/206" fact front
  and centre.
- **No map token needed.** `scatter_map` with `carto-positron`/OpenStreetMap tiles
  renders without a Mapbox token, so it works on a fresh clone.

## How to see it work

```bash
streamlit run src/app.py
```

Click through both tabs: confirm the map renders, picking a shop shows its reasoning
and sub-scores, the city bar chart matches the CLI, and the lead browser shows a
draft alongside its brief. Verified headless via Streamlit's `AppTest` (script runs
with zero exceptions, both tabs populate, selections re-render cleanly).
