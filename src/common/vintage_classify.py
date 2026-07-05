"""Shared "is this genuinely a vintage clothing shop?" classifier.

Used by BOTH parts:
- Part 1 ranks Manchester shops — classify on ``top_review`` + ``maps_category``.
- Part 2 filters leads before counting genuine shops per city — classify on
  ``notes`` + ``google_maps_category``.

Design (see docs/explanations/02-vintage-classifier.md for the full rationale):

- **Lexicon-based, not a lookup of this sheet's exact review strings.** Matching on
  the literal templates in this synthetic file would score 100% here and 0% on a
  real Google scrape, and would have nothing to say in the "sourcing at scale"
  debrief. Instead we score on *transferable* vintage-clothing vocabulary.
- **Review text is the strongest signal; category a weak prior; name weakest.**
  The brief is explicit that ``maps_category`` is noisy and a keyword filter on
  "vintage" fails. So a positive *name* ("Vintage Wines", "Vintage Barber") is only
  a weak nudge and is overridden by contradicting review text — that's what defeats
  the name-traps. Conversely a genuine shop filed under "Clothing store" / "Boutique"
  still wins on its review.
- **Three-way output, not binary.** Reviews like "decent secondhand finds" carry no
  clothing-specific tell, so we return ``genuine`` / ``ambiguous`` / ``not_vintage``
  rather than forcing a coin-flip. Ranking can decide whether to include the middle.
- **Explainable.** Every result carries the list of fired signals and a signed
  score, so "why shop A beats shop B" is inspectable, not a black box.

Matching is case-insensitive substring matching over curated, specific terms — fast
enough to run per-row, and vectorisable with compiled term-union regexes if profiling
ever shows it matters at 30k rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

# --- Review lexicons (the primary signal) ----------------------------------

# Clothing-specific vintage vocabulary. Deliberately transferable to real scrapes,
# not the literal template strings in this workbook.
POSITIVE_STRONG = (
    "vintage clothing", "vintage denim", "vintage streetwear", "retro clothing",
    "curated vintage", "reworked", "archive designer", "band tee", "band tees",
    "band t-shirt", "denim", "levi", "carhartt", "dickies", "streetwear",
    "sportswear", "football shirt", "workwear", "military surplus", "y2k",
    "grunge", "nike", "adidas", "ralph",
)

# On-theme but not clothing-specific — the genuinely ambiguous middle ground.
POSITIVE_WEAK = (
    "secondhand", "second-hand", "second hand", "pre-loved", "preloved",
    "some clothes", "curated", "thrift",
)

# Non-clothing retail: antiques, homeware, records, hospitality, services, etc.
NEGATIVE_STRONG = (
    "bric-a-brac", "bric a brac", "curio", "antique", "china", "collectable",
    "collectible", "furniture", "homeware", "home goods", "interiors",
    "mid-century", "restoration", "decor", "vinyl", "record", "bookshop",
    "book", "costume", "fancy dress", "wine", "spirits", "coffee", "cake",
    "cafe", "tea room", "barber", "tattoo", "console", "video game", "games",
    "phone", "electrical", "pawn",
)

# General-jumble tells — a browse/rummage shop, not a clothing shop.
NEGATIVE_WEAK = (
    "bit of everything", "rummage", "browse than a shop", "jumble",
)

# Explicit disqualifiers — reviewer says outright there are no clothes.
HARD_NEGATIVE = (
    "no clothes", "not clothing", "no clothing", "not clothes",
)

# --- Category priors (weak) --------------------------------------------------

CLOTHING_CATEGORIES = frozenset({
    "vintage clothing store", "used clothing store", "retro clothing shop",
    "clothing store", "vintage store", "thrift store", "boutique",
})
NONCLOTHING_CATEGORIES = frozenset({
    "charity shop", "antique store", "antiques store", "antiques & collectibles",
    "furniture store", "homeware store", "home goods store", "record shop",
    "wine shop", "cafe", "barber shop", "tattoo studio", "video game store",
    "pawn shop", "used book store", "costume shop",
})
# Everything else (notably "second hand shop") is a neutral 0.

# --- Name lexicons (weakest) -------------------------------------------------

NAME_POSITIVE = (
    "vintage", "retro", "thrift", "denim", "reworked", "thread", "garment",
    "wardrobe", "closet", "racks", "rail", "worn", "rewind", "revival",
    "retrograde", "preloved",
)
NAME_NEGATIVE = (
    "charity", "oxfam", "barnardo", "sue ryder", "british heart", "antique",
    "furniture", "homeware", "record", "wine", "barber", "tattoo", "costume",
    "electrical", "pawn", "cash converters", "bookworm", "bazaar", "curio",
    "collectable",
)
# Terms that make a "vintage/retro" name a *potential* trap.
NAME_TRAP_TERMS = ("vintage", "retro")

# --- Scoring config ----------------------------------------------------------

GENUINE_THRESHOLD = 2.0
AMBIGUOUS_THRESHOLD = 0.5


@dataclass
class ClassificationResult:
    is_genuine: bool          # True only when label == "genuine"
    label: str                # "genuine" | "ambiguous" | "not_vintage"
    confidence: float         # 0..1, how sure we are of the label
    score: float              # signed evidence score (explainable)
    signals: List[str] = field(default_factory=list)  # human-readable fired signals


def _norm(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().casefold()


def _hits(text: str, terms) -> List[str]:
    """Return the lexicon terms (in lexicon order) that appear in ``text``."""
    return [t for t in terms if t in text]


def _confidence(score: float, label: str, hard: bool) -> float:
    if hard:
        return 0.97
    if label == "ambiguous":
        return 0.40
    if label == "genuine":
        return round(min(0.98, 0.55 + (score - GENUINE_THRESHOLD) * 0.10), 2)
    return round(min(0.98, 0.55 + (AMBIGUOUS_THRESHOLD - score) * 0.10), 2)


def classify(name=None, category=None, review=None) -> ClassificationResult:
    """Classify one place from its name, category, and representative review text."""
    n, c, r = _norm(name), _norm(category), _norm(review)

    hard = _hits(r, HARD_NEGATIVE)
    strong_pos = _hits(r, POSITIVE_STRONG)
    # Don't double-count a weak term that's a substring of a strong hit
    # (e.g. "curated" inside "curated vintage").
    weak_pos = [t for t in _hits(r, POSITIVE_WEAK)
                if not any(t in sp for sp in strong_pos)]
    strong_neg = _hits(r, NEGATIVE_STRONG)
    weak_neg = _hits(r, NEGATIVE_WEAK)

    pos_score = min(len(strong_pos), 3) * 2.0 + min(len(weak_pos), 2) * 0.5
    neg_score = -(min(len(strong_neg), 4) * 2.0 + min(len(weak_neg), 2) * 1.0)

    if c in CLOTHING_CATEGORIES:
        cat_score = 1.0
    elif c in NONCLOTHING_CATEGORIES:
        cat_score = -1.0
    else:
        cat_score = 0.0

    name_pos = _hits(n, NAME_POSITIVE)
    name_neg = _hits(n, NAME_NEGATIVE)
    name_score = max(-1.0, min(1.0, 0.5 * len(name_pos) - 0.5 * len(name_neg)))

    score = pos_score + neg_score + cat_score + name_score

    if hard:
        label = "not_vintage"
    elif score >= GENUINE_THRESHOLD:
        label = "genuine"
    elif score >= AMBIGUOUS_THRESHOLD:
        label = "ambiguous"
    else:
        label = "not_vintage"

    # --- explainability -----------------------------------------------------
    signals: List[str] = []
    if strong_pos:
        signals.append(f"+review vintage-clothing signal: {', '.join(strong_pos)}")
    if weak_pos:
        signals.append(f"~review on-theme (no clothing tell): {', '.join(weak_pos)}")
    if hard:
        signals.append(f"-review explicit disqualifier: {', '.join(hard)}")
    if strong_neg:
        signals.append(f"-review non-clothing signal: {', '.join(strong_neg)}")
    if weak_neg:
        signals.append(f"-review general-jumble signal: {', '.join(weak_neg)}")
    if cat_score > 0:
        signals.append(f"~category clothing prior: {c}")
    elif cat_score < 0:
        signals.append(f"~category non-clothing prior: {c}")
    if name_pos:
        signals.append(f"~name hint: {', '.join(name_pos)}")
    if name_neg:
        signals.append(f"~name non-clothing hint: {', '.join(name_neg)}")

    trap = [t for t in NAME_TRAP_TERMS if t in n]
    if trap and label != "genuine":
        signals.append(
            f"name-trap: '{'/'.join(trap)}' in name not backed by review/category"
        )

    return ClassificationResult(
        is_genuine=(label == "genuine"),
        label=label,
        confidence=_confidence(score, label, bool(hard)),
        score=round(score, 2),
        signals=signals,
    )


def classify_frame(
    df: pd.DataFrame,
    *,
    name_col: str = "place_name",
    category_col: str = "maps_category",
    review_col: str = "top_review",
    prefix: str = "vintage_",
) -> pd.DataFrame:
    """Add classifier columns to a copy of ``df``.

    Part 1 uses the defaults; Part 2 passes ``name_col="store_name"``,
    ``category_col="google_maps_category"``, ``review_col="notes"``.
    """
    def _row(row) -> ClassificationResult:
        return classify(
            row.get(name_col) if name_col else None,
            row.get(category_col) if category_col else None,
            row.get(review_col) if review_col else None,
        )

    results = [_row(row) for _, row in df.iterrows()]
    out = df.copy()
    out[f"{prefix}label"] = [res.label for res in results]
    out[f"{prefix}is_genuine"] = [res.is_genuine for res in results]
    out[f"{prefix}confidence"] = [res.confidence for res in results]
    out[f"{prefix}score"] = [res.score for res in results]
    out[f"{prefix}signals"] = [" | ".join(res.signals) for res in results]
    return out
