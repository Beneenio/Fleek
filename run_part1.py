#!/usr/bin/env python
"""Part 1 CLI: build the ranked Manchester vintage-shop visit list.

    python run_part1.py                 # write outputs/, print top 10 + route
    python run_part1.py --ambiguous     # also include the 'ambiguous' middle ground
"""
import argparse

from src.part1.pipeline import run, suggested_route


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank Manchester vintage shops to visit.")
    ap.add_argument("--ambiguous", action="store_true",
                    help="include shops the classifier flags as ambiguous")
    ap.add_argument("--radius", type=float, default=500.0,
                    help="walkable-cluster radius in metres (default 500)")
    ap.add_argument("--top", type=int, default=10, help="how many to print")
    args = ap.parse_args()

    ranked = run(include_ambiguous=args.ambiguous, radius_m=args.radius)

    print(f"\nGenuine vintage shops ranked: {len(ranked)}")
    print(f"Wrote outputs/part1_ranked_manchester.csv and .md\n")
    print(f"Top {args.top}:")
    for _, r in ranked.head(args.top).iterrows():
        print(f"  {int(r['rank']):2d}. [{r['score']:5.1f}] {r['place_name']:<22} "
              f"{r['rating']}★/{int(r['review_count'])}  zone {int(r['cluster'])}")
        print(f"       {r['reason']}")

    order, metres, _ = suggested_route(ranked, zone=0)
    if order:
        names = " -> ".join(ranked.loc[i, "place_name"] for i in order)
        print(f"\nSuggested route (zone 0, ~{metres/1000:.1f} km): {names}")


if __name__ == "__main__":
    main()
