"""Score one model answer against the expected answer from messages.csv."""

import json


def parse_expected_items(text):
    """Turn 'naan:2;lassi:1' into {('naan', 2), ('lassi', 1)}. Empty text gives an empty set."""
    items = set()
    if not text or not text.strip():
        return items  # no items expected (complaint, question, cancel, other)

    for piece in text.split(";"):  # "naan:2;lassi:1" -> ["naan:2", "lassi:1"]
        name, qty = piece.split(":")  # "naan:2" -> "naan", "2"
        items.add((name.strip().lower(), int(qty)))
    return items


def output_to_dict(text):
    """Read the model's text as JSON. Return the dict, or None if it isn't valid JSON or isn't a dict."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError, TypeError:
        return None  # extra text, code fences, or no text at all -> unusable

    if not isinstance(data, dict):
        return None  # valid JSON but the wrong shape, like [1, 2, 3]
    return data


def items_to_set(items):
    """Turn [{'name': ' Naan ', 'qty': 2}] into {('naan', 2)}.

    Names are trimmed and lowercased. qty must convert to an int.
    Returns None if the shape is wrong (not a list, missing name or qty, bad qty).
    """
    if not isinstance(items, list):
        return None

    result = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        if "name" not in item or "qty" not in item:
            return None
        try:
            qty = int(item["qty"])  # 2 and "2" both become 2
        except ValueError, TypeError:
            return None  # "two" or None can't be a quantity
        name = str(item["name"]).strip().lower()
        result.add((name, qty))
    return result


def score_row(row, output_text):
    """Score one model answer.

    row: one row from messages.csv as a dict (uses 'expected_intent' and 'expected_items').
    output_text: the model's raw answer.
    Returns {'valid_json': 0/1, 'intent_match': 0/1, 'items_match': 0/1, 'score': average rounded to 2 places}.
    """
    data = output_to_dict(output_text)
    if data is None:
        # The kitchen system can't read it at all, so everything fails
        return {"valid_json": 0, "intent_match": 0, "items_match": 0, "score": 0.0}

    valid_json = 1

    expected_intent = row["expected_intent"].strip().lower()
    output_intent = str(data.get("intent", "")).strip().lower()
    intent_match = 1 if output_intent == expected_intent else 0

    expected_items = parse_expected_items(row["expected_items"])
    output_items = items_to_set(data.get("items"))
    items_match = (
        1 if output_items == expected_items else 0
    )  # sets: order doesn't matter

    score = round((valid_json + intent_match + items_match) / 3, 2)
    return {
        "valid_json": valid_json,
        "intent_match": intent_match,
        "items_match": items_match,
        "score": score,
    }
