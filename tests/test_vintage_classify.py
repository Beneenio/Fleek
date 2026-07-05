"""Tests for the shared vintage classifier, using real rows from the sheet.

Cases are the actual (name, category, review) triples from
``Part1_Manchester_scrape`` — genuine shops, "vintage" name-traps, reverse-traps
(nice category, non-clothing review), the ambiguous middle ground, and hard
disqualifiers — so the tests assert behaviour on the data we actually filter, not
on invented strings.
"""
import pytest

from src.common.io import load_part1_df
from src.common.vintage_classify import classify, classify_frame


# --- Genuine shops, including some filed under generic categories -----------
GENUINE = [
    ("Nomad Racks", "Vintage clothing store",
     "Great reworked pieces and archive designer, always something new in"),
    ("Golden Era Goods", "Vintage clothing store",
     "Best vintage denim in the Northern Quarter, huge racks of Carhartt and Dickies"),
    ("Revival Denim", "Vintage clothing store",
     "Best vintage denim in the Northern Quarter, huge racks of Carhartt and Dickies"),
    ("Moth Goods", "Thrift store",
     "Best vintage denim in the Northern Quarter, huge racks of Carhartt and Dickies"),
    ("Loop Vintage", "Thrift store",
     "Retro football shirts and sportswear heaven, they get huge deliveries weekly"),
    ("Mercer Racks", "Used clothing store",
     "Tucked away but worth it, one-off Y2K and grunge pieces, big turnover of stock"),
    # genuine shops hiding under generic categories — review must override:
    ("Bygone Store", "Clothing store",
     "Best vintage denim in the Northern Quarter, huge racks of Carhartt and Dickies"),
    ("Heirloom Closet", "Boutique",
     "Three floors of curated vintage — massive shop, staff really know their stuff"),
    ("Second Corner", "Boutique",
     "Proper vintage streetwear — Nike, Adidas, Ralph everywhere. Owner is lovely"),
]

# --- "vintage"/"retro" in the NAME but not a clothing shop ------------------
NAME_TRAPS = [
    ("Vintage Wines & Spirits", "Wine shop",
     "Great little wine shop, nice selection of natural wines"),
    ("Vintage Barber Co", "Barber shop",
     "Best barber in town, proper old-school cuts"),
    ("Vintage Ink Tattoo", "Tattoo studio",
     "Class tattoo studio, really talented artists"),
    ("The Vintage Tea Rooms", "Cafe",
     "Lovely spot for coffee and cake in a vintage setting"),
    ("Vintage Interiors", "Home goods store",
     "Gorgeous homeware and interiors, lots of vintage-style decor"),
    ("Vintage Home Co", "Homeware store",
     "Gorgeous homeware and interiors, lots of vintage-style decor"),
    ("Retro Games Exchange", "Video game store",
     "Retro games paradise, loads of old consoles"),
]

# --- Non-clothing junk (antiques / charity / furniture / records / etc.) ----
JUNK = [
    ("Helping Hand Charity", "Charity shop",
     "Standard charity shop — some clothes but mostly homeware and books"),
    ("The Bookworm", "Used book store",
     "Brilliant second-hand bookshop, could get lost in here"),
    ("Eastern Bloc Records", "Record shop",
     "Best vinyl selection in Manchester, great for rare records"),
    ("Reclaimed Furniture", "Furniture store",
     "Great for reclaimed furniture and homeware, not clothing though"),
    # reverse-trap: appealing "Boutique" category, but the review is furniture:
    ("Salvage Loft", "Boutique",
     "Beautiful antique furniture and mid-century pieces, lovely restoration work"),
    ("The Attic Antiques", "Antiques store",
     "Aladdin's cave of antiques, china and collectables, no clothes really"),
]

# --- Genuinely ambiguous: secondhand, but no clothing-specific tell ---------
AMBIGUOUS = [
    ("Thread Store", "Second hand shop", "Friendly place, decent secondhand finds"),
    ("Nomad Finds", "Second hand shop", "Friendly place, decent secondhand finds"),
    ("Bygone Depot", "Boutique",
     "Small but beautifully curated, prices a little steep but great quality"),
]


@pytest.mark.parametrize("name,category,review", GENUINE)
def test_genuine_shops_pass(name, category, review):
    res = classify(name, category, review)
    assert res.is_genuine is True
    assert res.label == "genuine"
    assert res.confidence >= 0.5
    assert res.signals  # explainable


@pytest.mark.parametrize("name,category,review", NAME_TRAPS)
def test_name_traps_rejected_and_flagged(name, category, review):
    res = classify(name, category, review)
    assert res.is_genuine is False
    assert res.label == "not_vintage"
    # the explainability must call out that the vintage/retro name was a trap
    assert any("name-trap" in s for s in res.signals), res.signals


@pytest.mark.parametrize("name,category,review", JUNK)
def test_junk_rejected(name, category, review):
    res = classify(name, category, review)
    assert res.is_genuine is False
    assert res.label == "not_vintage"


@pytest.mark.parametrize("name,category,review", AMBIGUOUS)
def test_ambiguous_is_middle_ground(name, category, review):
    res = classify(name, category, review)
    assert res.label == "ambiguous"
    assert res.is_genuine is False


def test_hard_negative_overrides_positive_name_and_category():
    # Even a clothing category + a vintage-y name can't survive "no clothes".
    res = classify("Vintage Thread Co", "Vintage clothing store",
                   "Aladdin's cave of antiques, china and collectables, no clothes really")
    assert res.label == "not_vintage"
    assert any("disqualifier" in s for s in res.signals)


def test_review_overrides_category_both_directions():
    # good review beats a non-clothing category ...
    hidden = classify("Bygone Store", "Clothing store",
                      "Best vintage denim, huge racks of Carhartt and Dickies")
    assert hidden.is_genuine
    # ... and a bad review beats a clothing-ish category.
    reverse = classify("Salvage Loft", "Boutique",
                       "Beautiful antique furniture and mid-century pieces")
    assert not reverse.is_genuine


def test_blank_and_nan_inputs_do_not_crash():
    assert classify(None, None, None).label in {"not_vintage", "ambiguous"}
    assert classify("", "", "").is_genuine is False
    assert classify(float("nan"), float("nan"), float("nan")).is_genuine is False


def test_classify_frame_on_real_sheet():
    df = classify_frame(load_part1_df())
    for col in ("vintage_label", "vintage_is_genuine", "vintage_confidence",
                "vintage_score", "vintage_signals"):
        assert col in df.columns
    counts = df["vintage_label"].value_counts()
    # brief says ~70% of the scrape is NOT a vintage clothing shop — so genuine
    # should be a clear minority, and the junk should dominate.
    assert counts.get("genuine", 0) < len(df) * 0.5
    assert counts.get("not_vintage", 0) > len(df) * 0.4
    # a couple of anchor rows by name
    by_name = df.set_index("place_name")["vintage_is_genuine"]
    assert bool(by_name.get("Revival Denim")) is True
    assert bool(by_name.get("Vintage Barber Co")) is False
