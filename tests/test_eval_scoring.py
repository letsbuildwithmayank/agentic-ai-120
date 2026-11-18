"""Tests for eval_scoring: checks how one model output is scored against the expected answer."""

from agentic_ai_120.eval_scoring import (
    items_to_set,
    output_to_dict,
    parse_expected_items,
    score_row,
)

# ---------- parse_expected_items ----------


def test_parse_expected_items_two_items():
    assert parse_expected_items("naan:2;lassi:1") == {("naan", 2), ("lassi", 1)}


def test_parse_expected_items_name_with_space():
    assert parse_expected_items("dal makhani:2") == {("dal makhani", 2)}


def test_parse_expected_items_empty():
    assert parse_expected_items("") == set()


# ---------- output_to_dict ----------


def test_output_to_dict_valid_json():
    assert output_to_dict('{"intent": "order", "items": []}') == {
        "intent": "order",
        "items": [],
    }


def test_output_to_dict_text_before_json_is_invalid():
    assert output_to_dict('Sure! {"intent": "order", "items": []}') is None


def test_output_to_dict_code_fences_are_invalid():
    assert output_to_dict('```json\n{"intent": "order", "items": []}\n```') is None


def test_output_to_dict_not_a_dict_is_invalid():
    assert output_to_dict("[1, 2, 3]") is None


# ---------- items_to_set ----------


def test_items_to_set_normalizes_names():
    assert items_to_set([{"name": " Naan ", "qty": 2}]) == {("naan", 2)}


def test_items_to_set_order_does_not_matter():
    a = items_to_set([{"name": "naan", "qty": 2}, {"name": "coke", "qty": 1}])
    b = items_to_set([{"name": "coke", "qty": 1}, {"name": "naan", "qty": 2}])
    assert a == b


def test_items_to_set_numeric_string_qty_is_ok():
    assert items_to_set([{"name": "naan", "qty": "2"}]) == {("naan", 2)}


def test_items_to_set_bad_qty_returns_none():
    assert items_to_set([{"name": "naan", "qty": "two"}]) is None


def test_items_to_set_missing_qty_returns_none():
    assert items_to_set([{"name": "naan"}]) is None


# ---------- score_row ----------


def test_score_row_perfect():
    row = {"expected_intent": "order", "expected_items": "naan:2"}
    result = score_row(
        row, '{"intent": "order", "items": [{"name": "naan", "qty": 2}]}'
    )
    assert result == {
        "valid_json": 1,
        "intent_match": 1,
        "items_match": 1,
        "score": 1.0,
    }


def test_score_row_wrong_intent():
    row = {"expected_intent": "question", "expected_items": ""}
    result = score_row(row, '{"intent": "complaint", "items": []}')
    assert result == {
        "valid_json": 1,
        "intent_match": 0,
        "items_match": 1,
        "score": 0.67,
    }


def test_score_row_intent_case_and_spaces_ignored():
    row = {"expected_intent": "complaint", "expected_items": ""}
    result = score_row(row, '{"intent": " Complaint ", "items": []}')
    assert result["intent_match"] == 1


def test_score_row_invalid_json_scores_zero():
    row = {"expected_intent": "order", "expected_items": "naan:2"}
    result = score_row(row, 'Sure! {"intent": "order"}')
    assert result == {
        "valid_json": 0,
        "intent_match": 0,
        "items_match": 0,
        "score": 0.0,
    }


def test_score_row_missing_items_key():
    row = {"expected_intent": "other", "expected_items": ""}
    result = score_row(row, '{"intent": "other"}')
    assert result == {
        "valid_json": 1,
        "intent_match": 1,
        "items_match": 0,
        "score": 0.67,
    }
