"""Pydantic models for one restaurant order: the shape AND the business rules.

Schema-valid:   right fields and types (qty is a number >= 1, time looks like "20:30")
Business-valid: every dish is really on our menu, and modifiers stay with their dish

When a check fails, Pydantic raises a ValidationError with a clear message.
Instructor sends that message back to the model and asks again (the re-ask loop).
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

from agentic_ai_120.menu_tools import MENU

MENU_NAMES = [item["name"] for item in MENU]

# The only modifiers the kitchen understands. The model must map the customer's words onto these.
Modifier = Literal[
    "extra spicy",
    "less spicy",
    "no onion",
    "no garlic",
    "extra butter",
    "less oil",
    "no ice",
]
MODIFIERS = list(get_args(Modifier))


class OrderItem(BaseModel):
    """One dish in the order."""

    name: str = Field(description=f"Exact menu name, one of: {', '.join(MENU_NAMES)}")
    # No defaults on purpose: a field with a default is optional in the schema, so the model may
    # skip it and the default fills in silently (Day 8 run 1: every qty came back as 1).
    qty: int = Field(
        ge=1,
        description="How many. Always write it: 1 if the customer didn't say a number.",
    )
    modifiers: list[Modifier] = Field(
        description="Changes asked for on THIS dish. [] if none."
    )

    @field_validator("name")
    @classmethod
    def must_be_on_menu(cls, value):
        name = value.strip().lower()
        if name not in MENU_NAMES:
            raise ValueError(
                f"'{value}' is not on the menu. Use one of {MENU_NAMES}, "
                "or put it in unavailable_items if the customer asked for something we don't sell."
            )
        return name


class Order(BaseModel):
    """Everything the kitchen and the rider need from one WhatsApp message."""

    items: list[OrderItem] = Field(
        default_factory=list, description="Dishes that are on the menu"
    )
    unavailable_items: list[str] = Field(
        default_factory=list,
        description="Dishes or drinks the customer asked for that are NOT on the menu. Never modifiers.",
    )
    delivery_time: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Clock time in 24-hour HH:MM, e.g. '20:30'. null if no clock time was given.",
    )
    notes: str | None = Field(
        default=None,
        description="Instructions for the rider or kitchen that are not modifiers, e.g. 'give it to the guard'. null if none.",
    )

    @field_validator("unavailable_items")
    @classmethod
    def no_modifiers_here(cls, values):
        for value in values:
            if value.strip().lower() in MODIFIERS:
                raise ValueError(
                    f"'{value}' is a modifier, not a dish. Put it in the modifiers of the dish it belongs to."
                )
        return values
