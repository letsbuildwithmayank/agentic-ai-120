"""Tests for the tool loop. A fake chat() plays the model by a script: no API, no internet."""

import json
from types import SimpleNamespace

from agentic_ai_120 import tool_loop
from agentic_ai_120.tool_loop import run_tool


def test_run_tool_good_call():
    assert run_tool("get_price", '{"item_id": "naan"}')["price_inr"] == 40


def test_run_tool_unknown_tool():
    assert "Unknown tool 'get_pizza'" in run_tool("get_pizza", "{}")["error"]


def test_run_tool_broken_json():
    assert "not valid JSON" in run_tool("get_price", '{"item_id": naan')["error"]


def test_run_tool_missing_argument():
    assert "item_id" in run_tool("get_price", "{}")["error"]


def test_run_tool_qty_zero_is_caught_by_pydantic():
    error = run_tool("calc_bill", '{"items": [{"item_id": "naan", "qty": 0}]}')["error"]
    assert "items.0.qty" in error  # tells the model exactly which field is wrong


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def fake_model(replies, sent_messages):
    """A fake chat(): returns the scripted replies one by one and saves what it was sent."""

    def fake_chat(messages, **kwargs):
        sent_messages.append(
            json.loads(json.dumps(messages))
        )  # a copy of this round's messages
        return replies.pop(0), "fake/model"

    return fake_chat


def test_parallel_calls_then_answer(monkeypatch):
    replies = [
        SimpleNamespace(
            content=None,
            tool_calls=[
                tool_call("call_A", "get_price", '{"item_id": "naan"}'),
                tool_call("call_B", "get_price", '{"item_id": "lassi"}'),
            ],
        ),
        SimpleNamespace(content="Naan ₹40, lassi ₹60.", tool_calls=None),
    ]
    sent = []
    monkeypatch.setattr(tool_loop, "chat", fake_model(replies, sent))

    answer, trace = tool_loop.run_agent("naan aur lassi ka price?")

    assert answer == "Naan ₹40, lassi ₹60."
    tool_messages = [m for m in sent[1] if m["role"] == "tool"]  # what round 2 received
    assert [m["tool_call_id"] for m in tool_messages] == ["call_A", "call_B"]
    assert (
        json.loads(tool_messages[0]["content"])["price_inr"] == 40
    )  # call_A got naan's price
    assert [step["type"] for step in trace] == ["tool", "tool", "answer"]


def test_error_goes_back_to_the_model(monkeypatch):
    replies = [
        SimpleNamespace(
            content=None,
            tool_calls=[tool_call("call_A", "get_price", '{"item_id": "pizza-99"}')],
        ),
        SimpleNamespace(content="Sorry, pizza is not on our menu.", tool_calls=None),
    ]
    sent = []
    monkeypatch.setattr(tool_loop, "chat", fake_model(replies, sent))

    answer, _ = tool_loop.run_agent("pizza kitne ka hai?")

    tool_message = sent[1][-1]
    assert (
        "Unknown item_id 'pizza-99'" in tool_message["content"]
    )  # an observation, not a crash
    assert answer == "Sorry, pizza is not on our menu."


def test_stops_after_max_rounds(monkeypatch):
    stuck = SimpleNamespace(
        content=None,
        tool_calls=[tool_call("call_X", "search_menu", '{"query": "naan"}')],
    )
    final = SimpleNamespace(content="Naan is ₹40.", tool_calls=None)
    replies = [stuck, stuck, stuck, final]  # 3 stuck rounds, then the forced answer
    sent = []
    monkeypatch.setattr(tool_loop, "chat", fake_model(replies, sent))

    answer, trace = tool_loop.run_agent("naan?", max_rounds=3)

    assert len(sent) == 4  # 3 normal rounds + 1 forced answer, never more
    assert trace[-1]["type"] == "forced_answer"
    assert answer == "Naan is ₹40."
