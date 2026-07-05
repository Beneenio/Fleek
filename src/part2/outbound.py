"""Personalised outbound: deterministic archetype/brief selection, then Claude drafts.

The brief is explicit that tone must genuinely change with the lead's situation — a
cold intro should read nothing like a win-back, a reply to a skeptical owner who
thinks Fleek is "for small resellers", or a check-in with an active customer. So
this is a **hybrid**:

1. **Deterministic** — from the cleaned stage (+ purchase recency + the objection in
   the notes) we pick an *archetype* and build a structured per-lead *brief*. This is
   rule-based and testable; the LLM never decides the strategy.
2. **Generative** — the brief + an archetype-specific system prompt go to Claude
   (`claude-opus-4-8`) to draft natural subject + body. If no `ANTHROPIC_API_KEY` is
   set (or the call fails), we fall back to a **labelled deterministic template** so
   the tool still runs end-to-end and the tone difference is still visible.

Scale: drafting is the only per-row LLM step. `generate_drafts` uses a bounded
ThreadPoolExecutor, and `due_for_touch` lets a real run redraft only leads actually
due a touch rather than the whole table each time.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass

MODEL = "claude-opus-4-8"

# --- Stage -> archetype ------------------------------------------------------
# active_customer vs win_back for "won" is decided by purchase recency below.
_STAGE_TO_ARCHETYPE = {
    "never_contacted": "cold_intro",
    "contacted": "follow_up",
    "replied": "follow_up",
    "in_conversation": "in_conversation",
    "negotiating": "in_conversation",
    "meeting_booked": "in_conversation",
    "skeptical": "skeptical",
    "won": "active_customer",     # may become win_back if the purchase is stale
    "churned": "win_back",
    "lost": "lost",
}

# A "won" customer with no purchase in this many days is treated as a win-back.
LAPSED_PURCHASE_DAYS = 120
# Default "due a touch" cadence for cold/follow-up leads.
TOUCH_CADENCE_DAYS = 14


def select_archetype(row) -> str:
    stage = row.get("stage", "never_contacted")
    archetype = _STAGE_TO_ARCHETYPE.get(stage, "follow_up")
    if archetype == "active_customer":
        dsp = row.get("days_since_purchase")
        # A missing, negative (corrupt/future-dated) or stale purchase -> win-back.
        if dsp is None or pd.isna(dsp) or dsp < 0 or dsp > LAPSED_PURCHASE_DAYS:
            return "win_back"
    return archetype


# --- Objection extraction (skeptical leads) ---------------------------------
# The notes on "no fit"/"not interested" rows carry the real objection.
_OBJECTION_RULES = [
    ("price_vs_wholesalers", ("wholesaler", "wholesalers", "price-sensitive",
                              "price sensitive", "compares", "cheaper", "too expensive")),
    ("thinks_for_small_resellers", ("small reseller", "small resellers", "too cheap",
                                    "not for us", "isn't for us", "for resellers")),
    ("bad_past_experience", ("tried us", "wasn't happy", "was not happy", "sizing",
                             "bad experience", "churned before", "last time")),
    ("wants_volume", ("wants volume", "volume", "bulk only", "second shop")),
    ("wants_proof", ("see the app", "before committing", "wants to see", "proof",
                     "demo", "sample")),
]


def extract_objection(note_context) -> Dict[str, Optional[str]]:
    if note_context is None or (isinstance(note_context, float) and pd.isna(note_context)):
        return {"type": "generic", "raw": None}
    text = str(note_context).lower()
    for otype, keys in _OBJECTION_RULES:
        if any(k in text for k in keys):
            return {"type": otype, "raw": str(note_context)}
    return {"type": "generic", "raw": str(note_context)}


# Customer-facing lines the template fallback drops straight into the body.
_OBJECTION_TEMPLATE_LINE = {
    "price_vs_wholesalers":
        "I know price matters — Fleek isn't about being the cheapest, it's hand-picked "
        "vintage bought at volume so it actually sells through at a good margin.",
    "thinks_for_small_resellers":
        "Quick myth-bust: Fleek isn't just for small Depop resellers — we supply "
        "established shops buying 100+ pieces at a time.",
    "bad_past_experience":
        "I know it didn't land last time — a lot has changed since, and I'd rather earn it "
        "back with a small low-risk order than talk you into anything.",
    "wants_volume":
        "If it's volume you're after, that's exactly our strength — consistent bulk sourcing, "
        "enough to stock a second shop.",
    "wants_proof":
        "No need to take my word for it — I can show you the live catalogue or send a sample "
        "box first.",
    "generic":
        "I'd rather show you than sell you — happy to walk through exactly how it'd work.",
}

# How to answer each objection — steers Claude's drafting (an instruction, not prose).
_OBJECTION_ANSWER = {
    "price_vs_wholesalers":
        "Acknowledge price sensitivity honestly; position Fleek as curated, hand-picked "
        "vintage bought at volume (better margins on sell-through), not a race to the bottom.",
    "thinks_for_small_resellers":
        "Correct the misconception: Fleek supplies established physical shops buying 100+ "
        "pieces wholesale, not just small Depop resellers.",
    "bad_past_experience":
        "Acknowledge the past issue specifically, note it's improved, and offer a low-risk "
        "trial order rather than a hard sell.",
    "wants_volume":
        "Lead with volume: consistent bulk sourcing and the ability to fill a second shop.",
    "wants_proof":
        "Offer proof — a look at the live catalogue/app or a sample box — before any commitment.",
    "generic":
        "Be empathetic, address their hesitation directly, and offer a low-pressure next step.",
}


# --- Brief -------------------------------------------------------------------
@dataclass
class Draft:
    archetype: str
    subject: str
    body: str
    source: str  # "claude" | "template"


def build_brief(row) -> Dict:
    archetype = select_archetype(row)
    objection = extract_objection(row.get("note_context")) if archetype == "skeptical" else \
        {"type": None, "raw": None}
    return {
        "store_name": _clean(row.get("store_name")),
        "owner_name": _clean(row.get("owner_name")),
        "city": _clean(row.get("city")),
        "sells": _clean(row.get("note_bio")),
        "channel": _clean(row.get("channel")),
        "stage": _clean(row.get("stage")),
        "archetype": archetype,
        # Day-counts and spend can't be negative; a negative value is corrupt
        # (e.g. a future-dated purchase) so we drop it rather than render it.
        "days_since_contact": _num(row.get("days_since_contact"), minimum=0),
        "days_since_purchase": _num(row.get("days_since_purchase"), minimum=0),
        "monthly_spend_gbp": _num(row.get("est_monthly_spend_gbp"), minimum=0),
        "objection_type": objection["type"],
        "objection_note": objection["raw"],
    }


# Cells that are technically non-blank but carry no real value. We must never
# drop these into a message ("Hi N/A," / "loved the nan").
_PLACEHOLDER_TOKENS = {"nan", "nat", "none", "null", "n/a", "na", "-", "--",
                       "unknown", "tbd", "?", "."}


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() in _PLACEHOLDER_TOKENS:
        return None
    return s


def _num(v, *, minimum: Optional[float] = None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        f = round(float(v), 1)
    except (TypeError, ValueError):
        return None
    if minimum is not None and f < minimum:
        return None  # out-of-range -> treat as unknown, never render bad data
    return f


# --- Claude drafting ---------------------------------------------------------
FLEEK_CONTEXT = (
    "You write outbound for Fleek, a B2B marketplace for secondhand & vintage clothing. "
    "Physical vintage shops buy in bulk (100+ pieces at a time) and are relationship buyers "
    "— one-and-done selling doesn't work. Keep messages short (subject + 3-6 sentence body), "
    "specific to what the shop sells, human, and never salesy or full of buzzwords. "
    "Return ONLY the message."
)

_ARCHETYPE_TONE = {
    "cold_intro":
        "COLD INTRO to a shop that's never heard from us. Short, curious, professional. "
        "Reference what they actually sell. One line on why Fleek fits a shop like theirs. "
        "Soft CTA (a quick call or dropping by) — no pressure, no discount talk.",
    "follow_up":
        "FOLLOW-UP after an earlier unanswered touch. Brief, low-friction nudge. Add one new "
        "concrete bit of value. Make saying yes easy. Don't guilt-trip about silence.",
    "in_conversation":
        "Lead is WARM / mid-conversation. Keep momentum, reference where things are, and "
        "propose one clear next step (pricing, a visit, or a first trial order).",
    "skeptical":
        "Lead is SKEPTICAL and pushed back. Empathetic, not defensive. Directly address their "
        "specific objection. Offer a low-risk next step. Absolutely no hard sell.",
    "active_customer":
        "Existing ACTIVE customer. Casual relationship check-in, NOT a pitch. Ask how recent "
        "orders have landed and what they're after next. Warm, peer-to-peer register.",
    "win_back":
        "WIN-BACK of a lapsed/churned customer. Warm and low-pressure. Acknowledge the time "
        "gap, note what's improved, make re-opening the door easy.",
    "lost":
        "Lead was marked LOST. Respectful, brief, no pressure. Leave the door open for later.",
}


def _system_prompt(archetype: str) -> str:
    tone = _ARCHETYPE_TONE.get(archetype, _ARCHETYPE_TONE["follow_up"])
    return f"{FLEEK_CONTEXT}\n\nThis message is a: {tone}"


def _brief_text(brief: Dict) -> str:
    lines = [f"Shop: {brief['store_name'] or 'unknown'}"]
    if brief["owner_name"]:
        lines.append(f"Owner: {brief['owner_name']}")
    if brief["city"]:
        lines.append(f"City: {brief['city']}")
    if brief["sells"]:
        lines.append(f"What they sell: {brief['sells']}")
    if brief["channel"]:
        lines.append(f"Channel: {brief['channel']} (physical shop = in-person/email; online = IG DM)")
    if brief["monthly_spend_gbp"]:
        lines.append(f"Estimated monthly spend potential: £{brief['monthly_spend_gbp']:.0f}")
    if brief["days_since_contact"] is not None:
        lines.append(f"Days since we last reached out: {brief['days_since_contact']:.0f}")
    if brief["days_since_purchase"] is not None:
        lines.append(f"Days since their last order: {brief['days_since_purchase']:.0f}")
    if brief["objection_type"]:
        lines.append(f"Their objection: {brief['objection_note'] or brief['objection_type']}")
        lines.append("How to handle it: " + _OBJECTION_ANSWER.get(brief["objection_type"],
                                                                  _OBJECTION_ANSWER["generic"]))
    return "\n".join(lines)


_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["subject", "body"],
    "additionalProperties": False,
}


def get_client():
    """Return an Anthropic client if a key is configured, else None (template mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def _draft_with_claude(brief: Dict, client) -> Draft:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=700,
        system=_system_prompt(brief["archetype"]),
        messages=[{"role": "user", "content": _brief_text(brief)}],
        output_config={"format": {"type": "json_schema", "schema": _JSON_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return Draft(brief["archetype"], data["subject"].strip(), data["body"].strip(), "claude")


# --- Deterministic template fallback ----------------------------------------
# Global copy rules applied to every template message (subject + body):
#   1. No em/en dashes anywhere. Spaced dashes read as a pause -> comma; a bare
#      dash -> hyphen. (Regular hyphens like "myth-bust" are left untouched.)
#   2. The greeting addresses the owner by first name only, never the full name.
def _no_dashes(text: str) -> str:
    return (text.replace(" — ", ", ").replace(" – ", ", ")
                .replace("—", "-").replace("–", "-"))


def _first_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.split()[0]


# CRM/meta notes that describe the *account*, not the stock. These must never be
# quoted back as "what they sell"; a bio matching one falls back to neutral copy.
_BIO_META_TOKENS = ("ig shop", "dm open", "dms open", "depop", "vinted",
                    "whatnot", "fast seller", "good margins", "live sales",
                    "sells on", "reseller")


def _sells_phrase(bio: Optional[str]) -> Optional[str]:
    """A short, lower-cased observation of what a shop sells, drawn from the raw
    bio — or None when the bio is CRM/meta noise rather than a description of
    stock. Trims trailing asides and descriptor suffixes so the result reads as
    an observation, not a copy-paste of the data."""
    if not bio:
        return None
    low = bio.strip().lower()
    if any(tok in low for tok in _BIO_META_TOKENS):
        return None
    # Drop a trailing dash aside, e.g. "vintage streetwear — Nike, Adidas, Ralph".
    for dash in ("—", "–", " - "):
        if dash in low:
            low = low.split(dash)[0].strip()
    # "... sportswear specialist" -> "... sportswear".
    for suf in (" specialist", " specialists"):
        if low.endswith(suf):
            low = low[: -len(suf)].strip()
    return low or None


def _visit_cta(brief: Dict) -> Optional[str]:
    """An in-person visit CTA, but only where there's a real store to drop into:
    a physical shop with a known city. Online leads never reach here (they're
    excluded from outreach) and a missing city returns None so the caller falls
    back to a remote CTA instead of writing 'I'm in your city'."""
    city = brief["city"]
    if not city or is_online_business(brief):
        return None
    return f"I'm in {city} next Thursday so I'll swing by to say hi."


def _draft_template(brief: Dict) -> Draft:
    shop = brief["store_name"] or "there"
    owner = _first_name(brief["owner_name"]) or "there"
    sells = _sells_phrase(brief["sells"])  # None when the bio isn't real stock
    visit = _visit_cta(brief)              # None when there's no store to visit
    a = brief["archetype"]

    if a == "cold_intro":
        obs = f"loved the {sells}" if sells else "love what you're building"
        close = visit or "Would you be open to a quick call?"
        subject = f"Fleek x {shop} — sourcing vintage in bulk"
        body = (f"Hi {owner}, I came across {shop} and {obs}. "
                f"Fleek helps vintage shops like yours source hand-picked stock in bulk "
                f"(100+ pieces at a time) without the usual minimums. {close}")
    elif a == "follow_up":
        what = f"sourcing {sells}" if sells else "sourcing stock"
        subject = f"Following up — Fleek for {shop}"
        body = (f"Hi {owner}, circling back on my note about {what} "
                f"through Fleek. Happy to send a short look at what's landing this week "
                f"if useful — no pressure either way.")
    elif a == "in_conversation":
        pricing = f"bulk pricing on {sells}" if sells else "bulk pricing"
        close = (f"{visit} We can lock in the details then." if visit
                 else "Which is easier for you?")
        subject = f"Next step for {shop}"
        body = (f"Hi {owner}, good chatting. To keep things moving I can put together "
                f"{pricing} or set up a small first order whenever you're ready. {close}")
    elif a == "skeptical":
        line = _OBJECTION_TEMPLATE_LINE.get(brief["objection_type"],
                                            _OBJECTION_TEMPLATE_LINE["generic"])
        # Soft, opt-in visit for a skeptic — friendly, never a pitch.
        close = (f"No hard sell. {visit}" if visit
                 else "No hard sell — happy to let the stock speak for itself whenever suits.")
        subject = f"Fair point — a quick thought for {shop}"
        body = f"Hi {owner}, totally hear you. {line} {close}"
    elif a == "active_customer":
        more = f"more of the {sells} you're after" if sells else "more of what's working for you"
        tail = f" {visit}" if visit else ""
        subject = f"How's it going, {shop}?"
        body = (f"Hi {owner}, just checking in — how did the last order land in-store? "
                f"Let me know what's selling and I'll line up {more}.{tail}")
    elif a == "win_back":
        tail = f" {visit}" if visit else ""
        subject = f"Been a minute — what's new at Fleek"
        body = (f"Hi {owner}, it's been a while since we last worked together. We've made "
                f"some real changes to how sourcing works, and I think it'd suit {shop} "
                f"again. Would you be open to taking another look? No commitment.{tail}")
    else:  # lost
        src = f"sourcing {sells} in bulk" if sells else "sourcing stock in bulk"
        subject = f"Leaving the door open — {shop}"
        body = (f"Hi {owner}, no worries that the timing wasn't right. If {src} "
                f"becomes useful down the line, I'm one message away. "
                f"Wishing {shop} a great season.")

    return Draft(a, _no_dashes(f"[template] {subject}"), _no_dashes(body), "template")


def is_online_business(brief: Dict) -> bool:
    """Part 2 targets physical stores. Online resellers (IG/Depop-only, no
    storefront) are out of scope for outreach and must never be drafted a message."""
    ch = brief.get("channel")
    return isinstance(ch, str) and ch.strip().lower() == "online"


def generate_message(brief: Dict, client=None) -> Draft:
    """Draft one message. Uses Claude if a client is available; else a template.
    Online businesses are excluded from outreach and returned as a skipped Draft."""
    if is_online_business(brief):
        return Draft(brief["archetype"], "",
                     "(Online business, excluded from physical-store outreach.)",
                     "skipped_online")
    if client is not None:
        try:
            return _draft_with_claude(brief, client)
        except Exception:
            pass  # fall back rather than crash the pipeline
    return _draft_template(brief)


# --- Batch over a frame ------------------------------------------------------
def due_for_touch(row, cadence_days: int = TOUCH_CADENCE_DAYS) -> bool:
    """Would a real run redraft this lead now? Cold leads always; others on cadence."""
    stage = row.get("stage")
    if stage in ("won", "lost"):
        return False
    dsc = row.get("days_since_contact")
    if dsc is None or pd.isna(dsc):
        return True  # never contacted / no date -> due
    return dsc >= cadence_days


def generate_drafts(df: pd.DataFrame, client=None, max_workers: int = 8,
                    only_due: bool = False) -> pd.DataFrame:
    """Add archetype + draft columns. `client=None` uses templates (no API key)."""
    work = df.copy().reset_index(drop=True)
    if only_due:
        work = work[work.apply(due_for_touch, axis=1)].reset_index(drop=True)

    briefs = [build_brief(row) for _, row in work.iterrows()]

    def _one(brief):
        return generate_message(brief, client)

    if client is not None and len(briefs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            drafts = list(pool.map(_one, briefs))
    else:
        drafts = [_one(b) for b in briefs]

    work["archetype"] = [d.archetype for d in drafts]
    work["draft_subject"] = [d.subject for d in drafts]
    work["draft_body"] = [d.body for d in drafts]
    work["draft_source"] = [d.source for d in drafts]
    work["brief_json"] = [json.dumps(b, default=str) for b in briefs]
    return work
