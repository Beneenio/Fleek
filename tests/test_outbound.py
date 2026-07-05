"""Outbound tests: archetype selection, objection extraction, brief, templates.

These cover the deterministic layer and the template fallback (no network). The
Claude path is exercised with a fake client so we never hit the API in tests.
"""
import pandas as pd
import pytest

from src.part2.outbound import (
    select_archetype, extract_objection, build_brief,
    generate_message, generate_drafts, due_for_touch, Draft,
)


def _row(**kw):
    base = dict(stage="never_contacted", store_name="Test Shop", owner_name="Alex",
                city="Leeds", note_bio="Vintage denim & band tees", note_context=None,
                channel="physical", days_since_contact=None, days_since_purchase=None,
                est_monthly_spend_gbp=500)
    base.update(kw)
    return pd.Series(base)


@pytest.mark.parametrize("stage,expected", [
    ("never_contacted", "cold_intro"),
    ("contacted", "follow_up"),
    ("replied", "follow_up"),
    ("in_conversation", "in_conversation"),
    ("negotiating", "in_conversation"),
    ("skeptical", "skeptical"),
    ("churned", "win_back"),
    ("lost", "lost"),
])
def test_stage_maps_to_archetype(stage, expected):
    assert select_archetype(_row(stage=stage)) == expected


def test_won_recent_is_active_customer_but_stale_is_winback():
    assert select_archetype(_row(stage="won", days_since_purchase=20)) == "active_customer"
    assert select_archetype(_row(stage="won", days_since_purchase=400)) == "win_back"
    assert select_archetype(_row(stage="won", days_since_purchase=None)) == "win_back"


@pytest.mark.parametrize("note,expected", [
    ("Price-sensitive, compares to local wholesalers.", "price_vs_wholesalers"),
    ("Thinks Fleek is for small resellers", "thinks_for_small_resellers"),
    ("Tried us in 2023, wasn't happy with the sizing mix.", "bad_past_experience"),
    ("Second shop opening soon, wants volume.", "wants_volume"),
    ("Keen but wants to see the app before committing.", "wants_proof"),
    ("Left a voicemail, no callback yet.", "generic"),
    (None, "generic"),
])
def test_objection_extraction(note, expected):
    assert extract_objection(note)["type"] == expected


def test_brief_includes_objection_only_for_skeptical():
    skeptical = build_brief(_row(stage="skeptical",
                                 note_context="compares to local wholesalers"))
    assert skeptical["archetype"] == "skeptical"
    assert skeptical["objection_type"] == "price_vs_wholesalers"
    cold = build_brief(_row(stage="never_contacted", note_context="compares to wholesalers"))
    assert cold["objection_type"] is None


def test_template_tone_differs_by_archetype():
    cold = generate_message(build_brief(_row(stage="never_contacted")))
    winback = generate_message(build_brief(_row(stage="churned", days_since_purchase=300)))
    assert cold.source == "template" and winback.source == "template"
    assert cold.archetype == "cold_intro" and winback.archetype == "win_back"
    # genuinely different copy, not one template with a name swapped
    assert cold.body != winback.body
    assert "while" in winback.body.lower()  # win-back acknowledges the gap


def test_skeptical_template_addresses_the_specific_objection():
    d = generate_message(build_brief(_row(stage="skeptical",
                                          note_context="compares to local wholesalers")))
    assert d.archetype == "skeptical"
    assert "cheapest" in d.body.lower() or "margin" in d.body.lower()  # price objection answered


class _FakeBlock:
    type = "text"
    def __init__(self, text): self.text = text


class _FakeResp:
    def __init__(self, text): self.content = [_FakeBlock(text)]


class _FakeClient:
    """Stands in for anthropic.Anthropic — records the call, returns canned JSON."""
    def __init__(self): self.calls = []
    class _Messages:
        def __init__(self, outer): self.outer = outer
        def create(self, **kw):
            self.outer.calls.append(kw)
            return _FakeResp('{"subject": "Hi", "body": "Drafted by Claude."}')
    @property
    def messages(self): return _FakeClient._Messages(self)


def test_claude_path_used_when_client_present():
    client = _FakeClient()
    d = generate_message(build_brief(_row(stage="never_contacted")), client=client)
    assert d.source == "claude"
    assert d.body == "Drafted by Claude."
    # correct model + structured-output request shape
    assert client.calls[0]["model"] == "claude-opus-4-8"
    assert "json_schema" in str(client.calls[0]["output_config"])


def test_claude_failure_falls_back_to_template():
    class Boom:
        @property
        def messages(self):
            class M:
                def create(self, **kw): raise RuntimeError("api down")
            return M()
    d = generate_message(build_brief(_row(stage="never_contacted")), client=Boom())
    assert d.source == "template"  # never crashes the pipeline


def test_brief_drops_negative_daycounts_and_spend():
    # Future-dated purchase/contact -> negative days; these are corrupt, not usable.
    b = build_brief(_row(stage="churned", days_since_purchase=-44,
                         days_since_contact=-5, est_monthly_spend_gbp=-10))
    assert b["days_since_purchase"] is None
    assert b["days_since_contact"] is None
    assert b["monthly_spend_gbp"] is None


def test_winback_message_never_renders_negative_gap():
    d = generate_message(build_brief(_row(stage="churned", days_since_purchase=-44)))
    assert d.archetype == "win_back"
    assert "-44" not in d.body and "-" not in d.body.split("while")[1][:12]
    assert "while" in d.body.lower()  # still acknowledges the gap, just no bad number


def test_won_with_corrupt_future_purchase_becomes_winback():
    # Negative days_since_purchase (future-dated) must not read as an active customer.
    assert select_archetype(_row(stage="won", days_since_purchase=-30)) == "win_back"


def test_placeholder_strings_are_treated_as_missing():
    b = build_brief(_row(stage="never_contacted", owner_name="N/A", city="unknown",
                         store_name="nan"))
    assert b["owner_name"] is None and b["city"] is None and b["store_name"] is None
    # and they don't leak into the drafted copy
    body = generate_message(b).body
    for junk in ("N/A", "unknown", "nan"):
        assert junk not in body


def test_due_for_touch():
    assert due_for_touch(_row(stage="never_contacted")) is True       # no date -> due
    assert due_for_touch(_row(stage="contacted", days_since_contact=30)) is True
    assert due_for_touch(_row(stage="contacted", days_since_contact=2)) is False
    assert due_for_touch(_row(stage="won", days_since_contact=1)) is False  # customers excluded


def test_generate_drafts_adds_columns():
    df = pd.DataFrame([_row(stage="never_contacted"), _row(stage="skeptical",
                       note_context="wants volume")])
    out = generate_drafts(df, client=None)
    for col in ("archetype", "draft_subject", "draft_body", "draft_source", "brief_json"):
        assert col in out.columns
    assert set(out["draft_source"]) == {"template"}
