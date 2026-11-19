"""Tests for the Order models: what passes, what fails, and the error messages. No API."""

import pytest
from pydantic import ValidationError

from agentic_ai_120.order_models import Order, OrderItem


def test_good_order_passes():
    order = Order(
        items=[OrderItem(name="butter chicken", qty=2, modifiers=["extra spicy"])],
        delivery_time="20:00",
        notes="give it to the guard",
    )
    assert order.items[0].qty == 2
    assert order.items[0].modifiers == ["extra spicy"]


def test_name_is_cleaned():
    assert (
        OrderItem(name="  Butter Chicken ", qty=1, modifiers=[]).name
        == "butter chicken"
    )


def test_qty_is_required():
    with pytest.raises(ValidationError) as error_info:
        OrderItem(name="naan", modifiers=[])  # the model skipped qty
    assert "qty" in str(
        error_info.value
    )  # the re-ask tells the model which field is missing


def test_qty_zero_fails():
    with pytest.raises(ValidationError):
        OrderItem(name="naan", qty=0, modifiers=[])


def test_dish_not_on_menu_fails_with_a_helpful_message():
    with pytest.raises(ValidationError) as error_info:
        OrderItem(name="pizza", qty=1, modifiers=[])
    assert "not on the menu" in str(error_info.value)
    assert "unavailable_items" in str(error_info.value)  # tells the model how to fix it


def test_unknown_modifier_fails():
    with pytest.raises(ValidationError):
        OrderItem(name="naan", qty=1, modifiers=["extra cheese"])


def test_modifier_in_unavailable_items_fails():
    with pytest.raises(ValidationError) as error_info:
        Order(unavailable_items=["less oil"])  # Day 8 run 1, row 24
    assert "is a modifier, not a dish" in str(error_info.value)


def test_real_unavailable_dish_passes():
    assert Order(unavailable_items=["pizza"]).unavailable_items == ["pizza"]


@pytest.mark.parametrize("time", ["20:00", "09:30", "23:59"])
def test_good_times_pass(time):
    assert Order(delivery_time=time).delivery_time == time


@pytest.mark.parametrize("time", ["8 pm", "8:00", "24:00", "20:60"])
def test_bad_times_fail(time):
    with pytest.raises(ValidationError):
        Order(delivery_time=time)


def test_empty_order_is_valid():
    order = Order()
    assert order.items == []
    assert order.unavailable_items == []
    assert order.delivery_time is None
    assert order.notes is None
