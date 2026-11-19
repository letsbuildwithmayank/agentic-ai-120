"""Tests for field-level scoring. Orders are built by hand: no API."""

from agentic_ai_120.order_models import Order, OrderItem
from agentic_ai_120.order_scoring import parse_items, parse_modifiers, score_order


def item(name, qty=1, modifiers=None):
    """Short way to build an OrderItem in the tests."""
    return OrderItem(name=name, qty=qty, modifiers=modifiers or [])


def make_row(items="", modifiers="", unavailable="", time="", notes=""):
    return {
        "expected_items": items,
        "expected_modifiers": modifiers,
        "expected_unavailable": unavailable,
        "expected_time": time,
        "expected_notes": notes,
    }


def test_parse_helpers():
    assert parse_items("naan:2;butter chicken:1") == {
        ("naan", 2),
        ("butter chicken", 1),
    }
    assert parse_modifiers("coke:no ice") == {("coke", "no ice")}
    assert parse_items("") == set()


def test_perfect_order_scores_all_ones():
    row = make_row(
        items="butter chicken:2;naan:4",
        modifiers="butter chicken:extra spicy",
        time="20:00",
        notes="guard",
    )
    order = Order(
        items=[item("butter chicken", 2, ["extra spicy"]), item("naan", 4)],
        delivery_time="20:00",
        notes="Give it to the guard",
    )
    assert score_order(row, order) == {
        "items": 1,
        "modifiers": 1,
        "unavailable": 1,
        "delivery_time": 1,
        "notes": 1,
    }


def test_wrong_qty_only_loses_the_items_field():
    row = make_row(items="naan:4")
    scores = score_order(row, Order(items=[item("naan", 3)]))
    assert scores["items"] == 0
    assert scores["modifiers"] == 1  # the other fields still count


def test_missing_modifier():
    row = make_row(items="coke:1", modifiers="coke:no ice")
    assert score_order(row, Order(items=[item("coke")]))["modifiers"] == 0


def test_unavailable_items():
    row = make_row(items="coke:2", unavailable="pizza")
    order = Order(items=[item("coke", 2)], unavailable_items=["Pizza (2)"])
    assert score_order(row, order)["unavailable"] == 1


def test_extra_note_when_none_expected_fails():
    row = make_row(items="naan:1")
    assert (
        score_order(row, Order(items=[item("naan")], notes="deliver fast"))["notes"]
        == 0
    )


def test_time_must_match_exactly():
    row = make_row(items="naan:1", time="20:00")
    assert (
        score_order(row, Order(items=[item("naan")], delivery_time="08:00"))[
            "delivery_time"
        ]
        == 0
    )


def test_failed_extraction_scores_zero():
    assert score_order(make_row(items="naan:1"), None) == {
        "items": 0,
        "modifiers": 0,
        "unavailable": 0,
        "delivery_time": 0,
        "notes": 0,
    }
