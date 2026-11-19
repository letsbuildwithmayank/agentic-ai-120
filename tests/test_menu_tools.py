"""Tests for the 3 menu tools. No API, no internet: plain function calls."""

import pytest

from agentic_ai_120.menu_tools import calc_bill, get_price, search_menu


def test_search_finds_both_biryanis():
    result = search_menu("biryani")
    ids = [item["item_id"] for item in result["results"]]
    assert ids == ["veg-biryani", "chicken-biryani"]


def test_search_ignores_case_and_spaces():
    result = search_menu("  NAAN ")
    assert result["results"][0]["item_id"] == "naan"


def test_search_unknown_item_gives_message_not_error():
    result = search_menu("pizza")
    assert result["results"] == []
    assert "not on the menu" in result["message"]


def test_search_result_is_limited():
    result = search_menu("")  # empty query matches all 8 items
    assert len(result["results"]) == 5  # MAX_RESULTS
    assert "Showing 5 of 8" in result["message"]


def test_search_returns_copies():
    result = search_menu("naan")
    result["results"][0]["price_inr"] = 0  # change the result...
    assert get_price("naan")["price_inr"] == 40  # ...the menu stays the same


@pytest.mark.parametrize(
    ("item_id", "price"), [("naan", 40), ("lassi", 60), ("butter-chicken", 340)]
)
def test_get_price(item_id, price):
    assert get_price(item_id)["price_inr"] == price


def test_get_price_unknown_id_returns_helpful_error():
    result = get_price("pizza-99")
    assert "pizza-99" in result["error"]
    assert "search_menu" in result["error"]  # tells the model what to do next


def test_calc_bill_adds_every_line():
    result = calc_bill(
        [{"item_id": "butter-chicken", "qty": 2}, {"item_id": "naan", "qty": 4}]
    )
    assert [line["line_total"] for line in result["lines"]] == [680, 160]
    assert result["total_inr"] == 840


def test_calc_bill_with_unknown_item_returns_error_and_no_total():
    result = calc_bill([{"item_id": "naan", "qty": 2}, {"item_id": "pizza", "qty": 1}])
    assert "pizza" in result["error"]
    assert "total_inr" not in result  # no half-correct bill


def test_calc_bill_empty_order_is_zero():
    assert calc_bill([]) == {"lines": [], "total_inr": 0}
