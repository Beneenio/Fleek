"""Part 2 cleaning: stages, channel, dates, notes, and blocked fuzzy dedupe.

The book is a Maps-scrape-meets-CRM-dump: stage names written ~39 ways, dates in
five formats, an unreliable channel label, duplicates (renamed/UPPERCASE/different
stage), and free-text notes that mix a bio with a rep note. Nothing downstream —
outbound or city ranking — holds up until this is tidied, so we do it unprompted.

What this produces (see docs/explanations/04-part2-cleaning.md):

- `stage` — canonical funnel stage from an explicit map over the observed spellings,
  with a keyword fallback so an unseen 40th spelling degrades gracefully.
- `channel` (+ confidence, + `channel_label_disagrees`) — inferred from the DATA
  (address/geo => physical; reseller scrape fields / IG-only => online), NOT from
  the `lead_channel_label`, which is blank 86/206 times and sometimes wrong.
- `last_contact_date` / `last_purchase_date` parsed to real dates, plus
  `days_since_contact` / `days_since_purchase` against an `as_of` date.
- `note_bio` / `note_context` — the "what they sell | rep note" split, feeding both
  the classifier (bio) and outbound personalisation (context).
- deduped rows: blocked candidate generation + fuzzy name match + a corroborating
  signal (address, or IG-handle+city, or geo+city) so we merge true duplicates
  without collapsing two same-named shops in different cities. Most-complete row is
  kept, gaps filled from its duplicates, with a merge log.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz

from src.common.dates import parse_messy_date

# --- Stage canonicalisation --------------------------------------------------

# Explicit map over every spelling observed in the sheet. Fast and unambiguous;
# the fallback below catches anything unseen.
STAGE_MAP: Dict[str, str] = {
    "not contacted": "never_contacted", "new": "never_contacted",
    "new lead": "never_contacted", "new - inbound": "never_contacted",
    "1st touch sent": "contacted", "contacted": "contacted", "emailed": "contacted",
    "replied": "replied", "responded": "replied",
    "in conversation": "in_conversation", "in convo": "in_conversation",
    "negotiating": "negotiating", "sent pricing": "negotiating",
    "quote sent": "negotiating",
    "visit booked": "meeting_booked", "visiting": "meeting_booked",
    "demo booked": "meeting_booked", "meeting set": "meeting_booked",
    "trial pending": "meeting_booked",
    "no fit": "skeptical", "not interested": "skeptical",
    "closed won": "won", "closed - won": "won", "won": "won",
    "churned": "churned", "lapsed": "churned", "dormant": "churned",
    "stopped buying": "churned",
    "closed lost": "lost", "closed - lost": "lost", "lost": "lost",
}

# Funnel order for freshness/advancement tie-breaks when merging duplicates.
STAGE_RANK: Dict[str, int] = {
    "never_contacted": 0, "contacted": 1, "replied": 2, "in_conversation": 3,
    "negotiating": 4, "meeting_booked": 5, "skeptical": 6, "won": 7,
    "churned": 8, "lost": 9, "unknown": -1,
}


def canonicalize_stage(raw) -> str:
    """Map a messy stage string to a canonical funnel stage."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "never_contacted"
    key = re.sub(r"\s+", " ", str(raw).strip().lower())
    if key in STAGE_MAP:
        return STAGE_MAP[key]
    # Fallback keyword normaliser for unseen spellings.
    k = re.sub(r"[^a-z ]+", " ", key)
    if "won" in k:
        return "won"
    if "lost" in k:
        return "lost"
    if any(t in k for t in ("churn", "lapsed", "dormant", "stopped")):
        return "churned"
    if "not interested" in k or "no fit" in k or "not a fit" in k:
        return "skeptical"
    if any(t in k for t in ("negotiat", "pricing", "quote")):
        return "negotiating"
    if "convo" in k or "conversation" in k:
        return "in_conversation"
    if "repl" in k or "respond" in k:
        return "replied"
    if any(t in k for t in ("visit", "meeting", "demo", "booked", "trial", "call")):
        return "meeting_booked"
    if any(t in k for t in ("contact", "email", "sent", "touch", "reach")):
        return "contacted"
    if any(t in k for t in ("new", "inbound", "prospect")):
        return "never_contacted"
    return "unknown"


# --- Channel inference -------------------------------------------------------

_PHYSICAL_LABELS = {"physical", "store", "shop"}
_ONLINE_LABELS = {"depop", "ig", "reseller", "vinted", "whatnot"}


def infer_channel(row) -> Tuple[str, float, bool]:
    """Infer physical vs online from the data, not the (unreliable) label.

    Returns (channel, confidence, disagrees_with_label).
    """
    has_addr = _nonblank(row.get("address")) or (
        pd.notna(row.get("lat")) and pd.notna(row.get("lng")))
    has_reseller_fields = pd.notna(row.get("items_listed")) or pd.notna(
        row.get("sell_through_rate"))

    if has_addr:
        channel, conf = "physical", 0.9
    elif has_reseller_fields:
        channel, conf = "online", 0.9
    elif _nonblank(row.get("instagram_handle")):
        channel, conf = "online", 0.6   # IG-only, no address => reseller
    else:
        channel, conf = "online", 0.4   # weak default

    disagrees = False
    label = row.get("lead_channel_label")
    if _nonblank(label):
        lab = str(label).strip().lower()
        label_channel = ("physical" if lab in _PHYSICAL_LABELS
                         else "online" if lab in _ONLINE_LABELS else None)
        if label_channel is not None and label_channel != channel:
            disagrees = True
    return channel, conf, disagrees


# --- Notes split -------------------------------------------------------------

def split_notes(notes) -> Tuple[Optional[str], Optional[str]]:
    """"bio | rep note" -> (bio, context). Single-part notes are treated as bio."""
    if not _nonblank(notes):
        return None, None
    parts = [p.strip() for p in str(notes).split("|")]
    parts = [p for p in parts if p]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " | ".join(parts[1:])


# --- Row-level cleaning ------------------------------------------------------

def _nonblank(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    return str(v).strip() != ""


def _valid_past_date(d: Optional[dt.date], as_of: dt.date) -> Optional[dt.date]:
    """Drop dates after ``as_of``. A last-contact/last-purchase date in the
    "future" relative to the snapshot is corrupt (typo, wrong year) and would
    otherwise produce a negative ``days_since_*`` that leaks into messages."""
    if d is None or (isinstance(d, float) and pd.isna(d)):
        return None
    return d if d <= as_of else None


def _days_since(d: Optional[dt.date], as_of: dt.date) -> Optional[int]:
    if d is None:
        return None
    return (as_of - d).days


def clean_frame(df: pd.DataFrame, as_of: Optional[dt.date] = None) -> pd.DataFrame:
    """Add canonical stage, channel, parsed dates, and note split (pre-dedupe)."""
    as_of = as_of or dt.date.today()
    out = df.copy()

    out["stage"] = out["lead_stage"].map(canonicalize_stage)

    ch = [infer_channel(row) for _, row in out.iterrows()]
    out["channel"] = [c[0] for c in ch]
    out["channel_confidence"] = [c[1] for c in ch]
    out["channel_label_disagrees"] = [c[2] for c in ch]

    out["last_contact_date"] = out["last_contact_date"].map(parse_messy_date).map(
        lambda d: _valid_past_date(d, as_of))
    out["last_purchase_date"] = out["last_purchase_date"].map(parse_messy_date).map(
        lambda d: _valid_past_date(d, as_of))
    out["days_since_contact"] = out["last_contact_date"].map(
        lambda d: _days_since(d, as_of))
    out["days_since_purchase"] = out["last_purchase_date"].map(
        lambda d: _days_since(d, as_of))

    notes = [split_notes(n) for n in out.get("notes", pd.Series([None] * len(out)))]
    out["note_bio"] = [n[0] for n in notes]
    out["note_context"] = [n[1] for n in notes]
    return out


# --- Dedupe ------------------------------------------------------------------

def _norm_name(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip() if _nonblank(s) else ""


def _norm_addr(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip() if _nonblank(s) else ""


def _norm_handle(s) -> str:
    return str(s).lower().lstrip("@").strip() if _nonblank(s) else ""


NAME_MATCH_THRESHOLD = 88
GEO_MATCH_METRES = 60.0


def _is_duplicate(a: pd.Series, b: pd.Series) -> bool:
    """Same shop? Fuzzy name match PLUS a corroborating identity signal, so two
    same-named shops in different cities don't get merged."""
    if fuzz.token_sort_ratio(_norm_name(a["store_name"]),
                             _norm_name(b["store_name"])) < NAME_MATCH_THRESHOLD:
        return False
    aa, ba = _norm_addr(a.get("address")), _norm_addr(b.get("address"))
    if aa and aa == ba:
        return True
    ah, bh = _norm_handle(a.get("instagram_handle")), _norm_handle(b.get("instagram_handle"))
    same_city = _norm_name(a.get("city")) == _norm_name(b.get("city")) and _nonblank(a.get("city"))
    if ah and ah == bh and same_city:
        return True
    if same_city and all(pd.notna(a.get(k)) and pd.notna(b.get(k)) for k in ("lat", "lng")):
        from src.part1.cluster import haversine_km
        if haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]) * 1000.0 <= GEO_MATCH_METRES:
            return True
    return False


def _candidate_blocks(df: pd.DataFrame) -> List[List[int]]:
    """Blocking: group rows that *might* be duplicates so we never compare all
    pairs. Keys: IG handle, normalised address, and name-prefix+city."""
    blocks: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for pos, (_, row) in enumerate(df.iterrows()):
        h = _norm_handle(row.get("instagram_handle"))
        if h:
            blocks[("ig", h)].append(pos)
        a = _norm_addr(row.get("address"))
        if a:
            blocks[("addr", a)].append(pos)
        nm = _norm_name(row.get("store_name"))
        if nm:
            blocks[("nm", nm[:8] + "|" + _norm_name(row.get("city")))].append(pos)
    return [idxs for idxs in blocks.values() if len(idxs) > 1]


def _merge_group(rows: pd.DataFrame) -> pd.Series:
    """Keep the most-complete row, fill its gaps from the rest, take the freshest
    stage and the most-recent dates."""
    completeness = rows.notna().sum(axis=1)
    base = rows.loc[completeness.idxmax()].copy()
    for _, other in rows.iterrows():
        for col in rows.columns:
            if not _nonblank(base.get(col)) and _nonblank(other.get(col)):
                base[col] = other[col]
    # freshest stage = stage of the most recently contacted duplicate
    dated = rows.dropna(subset=["last_contact_date"])
    if not dated.empty:
        base["stage"] = dated.loc[dated["last_contact_date"].idxmax(), "stage"]
    else:
        base["stage"] = max(rows["stage"], key=lambda s: STAGE_RANK.get(s, -1))
    for dcol in ("last_contact_date", "last_purchase_date"):
        vals = [d for d in rows[dcol] if d is not None and not pd.isna(d)]
        if vals:
            base[dcol] = max(vals)
    return base


def dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (deduped_df, merge_log). Expects a cleaned frame (has `stage`)."""
    work = df.reset_index(drop=True)
    n = len(work)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for block in _candidate_blocks(work):
        for x in range(len(block)):
            for y in range(x + 1, len(block)):
                i, j = block[x], block[y]
                if find(i) != find(j) and _is_duplicate(work.iloc[i], work.iloc[j]):
                    union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged_rows = []
    log_entries = []
    for root, members in groups.items():
        if len(members) == 1:
            merged_rows.append(work.iloc[members[0]])
        else:
            sub = work.iloc[members]
            merged_rows.append(_merge_group(sub))
            log_entries.append({
                "kept_store_name": work.iloc[members[0]]["store_name"],
                "merged_count": len(members),
                "lead_ids": ", ".join(str(x) for x in sub["lead_id"].tolist()),
                "stages_seen": ", ".join(sorted(set(sub["stage"].tolist()))),
                "cities": ", ".join(sorted(set(str(c) for c in sub["city"].tolist()))),
            })

    deduped = pd.DataFrame(merged_rows).reset_index(drop=True)
    merge_log = pd.DataFrame(log_entries)
    return deduped, merge_log


def clean_and_dedupe(df: pd.DataFrame, as_of: Optional[dt.date] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Full Part 2 clean: returns (deduped_df, merge_log, stats)."""
    cleaned = clean_frame(df, as_of=as_of)
    deduped, merge_log = dedupe(cleaned)
    stats = {
        "rows_in": len(df),
        "rows_out": len(deduped),
        "duplicates_removed": len(df) - len(deduped),
        "merge_groups": len(merge_log),
        "stage_distribution": deduped["stage"].value_counts().to_dict(),
        "channel_distribution": deduped["channel"].value_counts().to_dict(),
        "channel_label_disagreements": int(cleaned["channel_label_disagrees"].sum()),
        "channel_label_blank": int(cleaned["lead_channel_label"].isna().sum()),
    }
    return deduped, merge_log, stats
