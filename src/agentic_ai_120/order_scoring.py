"""Field-level scoring: compare one extracted Order with the expected answer from messages.csv.

Each message gets 5 separate scores (0 or 1), one per field, instead of one
"whole order right or wrong". That shows WHICH field the model gets wrong.
"""

FIELDS = ["items", "modifiers", "unavailable", "delivery_time", "notes"]


def parse_items(text):
    """'naan:2;butter chicken:1' -> {('naan', 2), ('butter chicken', 1)}"""
    result = set()
    for piece in text.split(";"):
        if piece.strip():
            name, qty = piece.split(":")
            result.add((name.strip().lower(), int(qty)))
    return result


def parse_modifiers(text):
    """'paneer tikka:no onion;coke:no ice' -> {('paneer tikka', 'no onion'), ('coke', 'no ice')}"""
    result = set()
    for piece in text.split(";"):
        if piece.strip():
            name, modifier = piece.split(":")
            result.add((name.strip().lower(), modifier.strip().lower()))
    return result


def parse_words(text):
    """'pasta;garlic bread' -> ['pasta', 'garlic bread']"""
    return [word.strip().lower() for word in text.split(";") if word.strip()]


def score_order(row, order):
    """row: dict from messages.csv. order: an Order, or None if extraction failed. Returns 5 scores."""
    if order is None:
        return {field: 0 for field in FIELDS}

    items = {(item.name, item.qty) for item in order.items}
    modifiers = {
        (item.name, modifier) for item in order.items for modifier in item.modifiers
    }

    expected_unavailable = parse_words(row["expected_unavailable"])
    unavailable_text = " ".join(order.unavailable_items).lower()
    if expected_unavailable:
        unavailable_ok = all(word in unavailable_text for word in expected_unavailable)
    else:
        unavailable_ok = not order.unavailable_items

    expected_note = row["expected_notes"].strip().lower()
    if expected_note:
        notes_ok = (
            expected_note in (order.notes or "").lower()
        )  # the key word must be in the note
    else:
        notes_ok = not order.notes

    return {
        "items": int(items == parse_items(row["expected_items"])),
        "modifiers": int(modifiers == parse_modifiers(row["expected_modifiers"])),
        "unavailable": int(unavailable_ok),
        "delivery_time": int(
            (order.delivery_time or "") == row["expected_time"].strip()
        ),
        "notes": int(notes_ok),
    }
