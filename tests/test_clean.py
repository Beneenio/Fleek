"""Part 2 cleaning tests: stage canonicalisation, channel, notes, and dedupe."""
import datetime as dt

import pandas as pd
import pytest

from src.common.io import load_part2_df
from src.part2.clean import (
    canonicalize_stage, infer_channel, split_notes,
    clean_frame, dedupe, clean_and_dedupe,
)


# --- Stage canonicalisation -------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("Closed Won", "won"), ("closed - won", "won"), ("WON", "won"),
    ("not contacted", "never_contacted"), ("NEW", "never_contacted"),
    ("in convo", "in_conversation"), ("Negotiating", "negotiating"),
    ("sent pricing", "negotiating"), ("no fit", "skeptical"),
    ("not interested", "skeptical"), ("Churned", "churned"),
    ("lapsed", "churned"), ("dormant", "churned"), ("stopped buying", "churned"),
    ("Visit booked", "meeting_booked"), ("closed - lost", "lost"),
])
def test_explicit_stage_map(raw, expected):
    assert canonicalize_stage(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("won deal!", "won"), ("gone cold - lapsed", "churned"),
    ("pricing negotiation", "negotiating"), ("booked a call", "meeting_booked"),
    ("brand new prospect", "never_contacted"), ("following up via email", "contacted"),
])
def test_fallback_normaliser_handles_unseen_spellings(raw, expected):
    assert canonicalize_stage(raw) == expected


def test_stage_unknown_and_blank():
    assert canonicalize_stage("xyzzy") == "unknown"
    assert canonicalize_stage(None) == "never_contacted"
    assert canonicalize_stage(float("nan")) == "never_contacted"


def test_all_sheet_stages_map_without_unknown():
    df = load_part2_df()
    canon = df["lead_stage"].map(canonicalize_stage)
    assert "unknown" not in set(canon)  # explicit map covers every observed spelling


# --- Channel inference ------------------------------------------------------
def test_channel_from_address_is_physical():
    row = pd.Series({"address": "12 High St", "lat": 53.4, "lng": -2.2,
                     "items_listed": None, "instagram_handle": "@x",
                     "lead_channel_label": None})
    channel, conf, _ = infer_channel(row)
    assert channel == "physical" and conf >= 0.8


def test_channel_from_reseller_fields_is_online():
    row = pd.Series({"address": None, "lat": None, "lng": None,
                     "items_listed": 340, "sell_through_rate": 0.7,
                     "instagram_handle": "@shop", "lead_channel_label": None})
    channel, _, _ = infer_channel(row)
    assert channel == "online"


def test_channel_ignores_but_flags_disagreeing_label():
    row = pd.Series({"address": "9 Market St", "lat": 51.5, "lng": -0.1,
                     "items_listed": None, "instagram_handle": None,
                     "lead_channel_label": "depop"})  # label says online, data says physical
    channel, _, disagrees = infer_channel(row)
    assert channel == "physical" and disagrees is True


# --- Notes split ------------------------------------------------------------
def test_split_notes():
    bio, ctx = split_notes("Vintage workwear, carhartt | Price-sensitive, compares to wholesalers")
    assert "carhartt" in bio and "wholesalers" in ctx
    assert split_notes("just a bio") == ("just a bio", None)
    assert split_notes(None) == (None, None)


# --- Dates ------------------------------------------------------------------
def test_dates_parsed_with_days_since():
    df = clean_frame(load_part2_df(), as_of=dt.date(2026, 7, 5))
    assert df["last_contact_date"].notna().sum() > 100
    # days_since must be non-negative where a date exists
    ds = df["days_since_contact"].dropna()
    assert (ds >= 0).all()


# --- Dedupe -----------------------------------------------------------------
def test_dedupe_merges_same_shop_same_address():
    # two rows, same shop (UPPERCASE + different stage), same address -> 1
    df = pd.DataFrame([
        {"lead_id": "A", "store_name": "Nomad Room", "city": "Bristol",
         "address": "116 Broadway", "instagram_handle": "@nomadroom",
         "lat": 51.4, "lng": -2.5, "items_listed": None, "sell_through_rate": None,
         "lead_stage": "Churned", "last_contact_date": "May 4 2026",
         "last_purchase_date": None, "lead_channel_label": None, "notes": None},
        {"lead_id": "B", "store_name": "NOMAD ROOM", "city": "Bristol",
         "address": "116 Broadway", "instagram_handle": "@nomadroom",
         "lat": 51.4, "lng": -2.5, "items_listed": None, "sell_through_rate": None,
         "lead_stage": "Closed Lost", "last_contact_date": "2026/05/20",
         "last_purchase_date": None, "lead_channel_label": None, "notes": None},
    ])
    cleaned = clean_frame(df)
    deduped, log = dedupe(cleaned)
    assert len(deduped) == 1
    assert len(log) == 1
    # freshest contact (2026-05-20 -> Closed Lost) wins the stage
    assert deduped.iloc[0]["stage"] == "lost"


def test_dedupe_does_not_merge_same_name_different_city():
    df = pd.DataFrame([
        {"lead_id": "A", "store_name": "Reclaimed Interiors", "city": "London",
         "address": "130 Union St", "instagram_handle": "@reclaimedinteriors",
         "lat": 51.5, "lng": -0.1, "items_listed": None, "sell_through_rate": None,
         "lead_stage": "contacted", "last_contact_date": None,
         "last_purchase_date": None, "lead_channel_label": None, "notes": None},
        {"lead_id": "B", "store_name": "RECLAIMED INTERIORS", "city": "Los Angeles",
         "address": "108 Church Rd", "instagram_handle": "@reclaimedinteriors",
         "lat": 34.0, "lng": -118.2, "items_listed": None, "sell_through_rate": None,
         "lead_stage": "no fit", "last_contact_date": None,
         "last_purchase_date": None, "lead_channel_label": None, "notes": None},
    ])
    deduped, log = dedupe(clean_frame(df))
    assert len(deduped) == 2  # different city + address => NOT merged
    assert log.empty


def test_clean_and_dedupe_on_real_sheet():
    deduped, log, stats = clean_and_dedupe(load_part2_df(), as_of=dt.date(2026, 7, 5))
    assert stats["rows_in"] == 206
    assert stats["duplicates_removed"] > 0
    assert stats["channel_distribution"]["online"] == 62   # brief: ~62 online resellers
    assert stats["channel_label_blank"] == 86              # why we can't trust the label
    # no merge group spans multiple cities (no over-merging)
    for cities in log["cities"]:
        assert len(set(cities.split(", "))) == 1
