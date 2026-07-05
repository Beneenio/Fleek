# 05 — Part 2 outbound

## What this commit adds

`src/part2/outbound.py` (+ `tests/test_outbound.py`, 18 tests) — personalised
outbound that genuinely changes tone with the lead's situation:

- **Archetype selection** — cleaned stage (+ purchase recency + objection) → one of
  `cold_intro`, `follow_up`, `in_conversation`, `skeptical`, `active_customer`,
  `win_back`, `lost`.
- **Objection extraction** — for skeptical leads, the rep note in `note_context` is
  classified (`price_vs_wholesalers`, `thinks_for_small_resellers`,
  `bad_past_experience`, `wants_volume`, `wants_proof`, …).
- **Structured brief** — what they sell, city, days since touch/order, spend
  potential, channel, objection.
- **Drafting** — brief + an archetype-specific system prompt → `claude-opus-4-8`,
  returning subject + body as structured JSON. **No key → labelled deterministic
  template** so the tool runs end-to-end regardless.

## Why it's built this way

- **Hybrid, not "let the LLM decide".** The strategy — which archetype, which
  objection, what the next step is — is deterministic and unit-tested. The LLM only
  turns a fixed brief into natural prose. That's what makes the tone differences
  defensible and reproducible instead of a black box.
- **Tone is driven by the situation, not a template with the name swapped.** A
  cold intro is short and curious; a skeptical reply is empathetic and answers the
  *specific* objection found in the notes ("compares to local wholesalers" →
  "Fleek isn't about being the cheapest…"); a win-back acknowledges the exact time
  gap; an active-customer note is a casual check-in, not a pitch. The tests assert
  the copy actually differs and that the skeptical draft addresses its objection.
- **Runs without an API key.** The brief warns candidates may not pay for AI. The
  template fallback is labelled `[template]` and reproduces the same
  archetype/objection logic, so a reviewer can clone and run it and still see the
  tone change — and a transient API error falls back rather than crashing the run.
- **Objection handling is split in two.** `_OBJECTION_ANSWER` is an *instruction*
  that steers Claude; `_OBJECTION_TEMPLATE_LINE` is *customer-facing prose* for the
  template — so the fallback never leaks "acknowledge the past issue…" into a real
  message.
- **Scale.** Drafting is the only per-row LLM step; `generate_drafts` runs it under a
  bounded `ThreadPoolExecutor`, and `due_for_touch` lets a real run redraft only the
  leads actually due a touch (cold leads always; others on a cadence; won/lost
  excluded) rather than the whole 30k-row table every run.

## How to see it work

```bash
python -m pytest tests/test_outbound.py -q          # 18 passed (no network)

# With a key: cp .env.example .env, add ANTHROPIC_API_KEY -> Claude drafts.
# Without one: templates, still tone-differentiated.
python -c "import datetime as dt; from src.common.io import load_part2_df; \
from src.part2.clean import clean_and_dedupe; \
from src.part2.outbound import generate_drafts, get_client; \
d,_,_=clean_and_dedupe(load_part2_df(), as_of=dt.date(2026,7,5)); \
out=generate_drafts(d, client=get_client()); \
print(out['archetype'].value_counts().to_dict())"
```
