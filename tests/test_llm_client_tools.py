"""Tests for chat(..., tools=...). A fake LiteLLM records what it was sent: no API, no internet."""

from types import SimpleNamespace

import pytest

from agentic_ai_120 import llm_client

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Price of one item.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
MESSAGES = [{"role": "user", "content": "naan kitne ka hai?"}]


def make_tool_call_response():
    """Looks like a LiteLLM response where the model asks for get_price("naan")."""
    tool_call = SimpleNamespace(
        id="call_A",
        type="function",
        function=SimpleNamespace(name="get_price", arguments='{"item_id": "naan"}'),
    )
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def sent(monkeypatch):
    """Patch LiteLLM with a fake that saves every argument it gets. Returns that list."""
    received = []

    def fake_completion(**kwargs):
        received.append(kwargs)
        return make_tool_call_response()

    monkeypatch.setattr(llm_client, "completion", fake_completion)
    return received


def test_tools_are_sent_and_whole_message_comes_back(sent, tmp_path):
    message, model_used = llm_client.chat(
        MESSAGES, model="fake/model", tools=TOOLS, db_path=tmp_path / "t.db"
    )

    assert sent[0]["tools"] == TOOLS
    assert (
        "tool_choice" not in sent[0]
    )  # not set -> not sent -> provider default ("auto")
    assert model_used == "fake/model"
    assert message.tool_calls[0].id == "call_A"
    assert message.tool_calls[0].function.name == "get_price"


def test_tool_choice_is_sent_when_set(sent, tmp_path):
    llm_client.chat(
        MESSAGES,
        model="fake/model",
        tools=TOOLS,
        tool_choice="required",
        db_path=tmp_path / "t.db",
    )
    assert sent[0]["tool_choice"] == "required"


def test_without_tools_nothing_changes(sent, tmp_path):
    llm_client.chat(MESSAGES, model="fake/model", db_path=tmp_path / "t.db")
    assert "tools" not in sent[0]
    assert "tool_choice" not in sent[0]


def test_stream_with_tools_is_refused(sent, tmp_path):
    with pytest.raises(ValueError):
        llm_client.chat(
            MESSAGES,
            model="fake/model",
            stream=True,
            tools=TOOLS,
            db_path=tmp_path / "t.db",
        )
    assert sent == []  # refused before any call was made
