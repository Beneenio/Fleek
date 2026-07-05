#!/usr/bin/env python
"""Part 2 CLI: clean the book, draft outbound, rank cities.

    python run_part2.py                 # write outputs/, print summary + city table
    python run_part2.py --no-llm        # force deterministic templates (skip Claude)
    python run_part2.py --only-due      # draft only leads currently due a touch
"""
import argparse
import datetime as dt
import json

from src.part2.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean leads, draft outbound, rank cities.")
    ap.add_argument("--no-llm", action="store_true",
                    help="force deterministic templates even if an API key is set")
    ap.add_argument("--only-due", action="store_true",
                    help="draft only leads currently due a touch")
    ap.add_argument("--as-of", type=str, default=None,
                    help="reference date YYYY-MM-DD for recency (default: today)")
    args = ap.parse_args()

    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    deduped, drafts, city, stats, merge_log = run(
        as_of=as_of, use_llm=not args.no_llm, only_due=args.only_due)

    print("\n=== Cleaning ===")
    print(f"  {stats['rows_in']} -> {stats['rows_out']} rows "
          f"({stats['duplicates_removed']} duplicates merged)")
    print(f"  channel: {stats['channel_distribution']} "
          f"(label blank {stats['channel_label_blank']}/{stats['rows_in']})")
    print(f"  drafting mode: {stats['drafting_mode']}  |  drafts: {len(drafts)}")

    print("\n=== Outbound archetypes ===")
    print(f"  {drafts['archetype'].value_counts().to_dict()}")

    print("\n=== City ranking ===")
    for _, r in city.iterrows():
        print(f"  {int(r['rank']):2d}. [{r['score']:5.1f}] {r['city']:<12} "
              f"{int(r['n_genuine']):2d} shops, £{r['total_spend_gbp']:,.0f}/mo")
        print(f"       {r['reason']}")

    print("\nWrote outputs/part2_cleaned_leads.csv, part2_outbound_drafts.csv, "
          "part2_outbound_sample.md, part2_city_ranking.csv/.md")


if __name__ == "__main__":
    main()
