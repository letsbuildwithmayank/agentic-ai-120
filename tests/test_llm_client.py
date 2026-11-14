"""Tests for llm_client.chat, with a fake LiteLLM (no internet, no cost)."""

import sqlite3
import time
from types import SimpleNamespace

import pytest

from agentic_ai_120 import llm_client
from agentic_ai_120.retry import APIError

FIRST, SECOND, THIRD = llm_client.DEFAULT_MODELS
MESSAGES = [{"role": "user", "content": "Hi"}]


class FakeLiteLLMError(Exception):
    """Looks like a LiteLLM error: it carries an HTTP status code."""

    def __init__(self, status_code):
        super().__init__(f"fake error {status_code}")
        self.status_code = status_code


def make_response(text):
    """Build an object shaped like a LiteLLM response: response.choices[0].message.content."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Replace LiteLLM with a script per model. Returns (script, calls, db_path)."""
    script = {}  # model -> list of things to do on each call: text or FakeLiteLLMError
    calls = []  # every model that got called, in order

    def fake_completion(model, messages, timeout, **kwargs):
        calls.append(model)
        result = script[model].pop(0)
        if isinstance(result, Exception):
            raise result
        return make_response(result)

    def fake_sleep(seconds):
        pass  # don't really wait in tests

    monkeypatch.setattr(llm_client, "completion", fake_completion)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    db_path = tmp_path / "test_calls.db"  # a fresh, temporary database for each test
    return script, calls, db_path


def test_first_model_answers(fake_llm):
    script, calls, db_path = fake_llm
    script[FIRST] = ["Hello!"]

    text, used = llm_client.chat(MESSAGES, db_path=db_path)

    assert text == "Hello!"
    assert used == FIRST
    assert calls == [FIRST]


def test_retries_same_model_then_answers(fake_llm):
    script, calls, db_path = fake_llm
    script[FIRST] = [FakeLiteLLMError(503), "Hello!"]

    text, used = llm_client.chat(MESSAGES, db_path=db_path)

    assert text == "Hello!"
    assert calls == [FIRST, FIRST]  # retried on the same model


def test_falls_back_when_first_model_keeps_failing(fake_llm):
    script, calls, db_path = fake_llm
    script[FIRST] = [FakeLiteLLMError(503)] * 3  # 1 try + 2 retries, all fail
    script[SECOND] = ["From the backup"]

    text, used = llm_client.chat(MESSAGES, max_retries=2, db_path=db_path)

    assert text == "From the backup"
    assert used == SECOND
    assert calls == [FIRST, FIRST, FIRST, SECOND]


def test_401_is_not_retried_but_falls_back(fake_llm):
    script, calls, db_path = fake_llm
    script[FIRST] = [FakeLiteLLMError(401)]
    script[SECOND] = ["Hello!"]

    text, used = llm_client.chat(MESSAGES, db_path=db_path)

    assert used == SECOND
    assert calls == [FIRST, SECOND]  # only ONE call to the first model


def test_all_models_fail_raises_api_error(fake_llm):
    script, calls, db_path = fake_llm
    for model in llm_client.DEFAULT_MODELS:
        script[model] = [FakeLiteLLMError(503)] * 3

    with pytest.raises(APIError) as error_info:
        llm_client.chat(MESSAGES, max_retries=2, db_path=db_path)

    assert error_info.value.code == 503
    assert len(calls) == 9  # 3 models x 3 tries


def test_every_call_is_logged(fake_llm):
    script, calls, db_path = fake_llm
    script[FIRST] = [FakeLiteLLMError(503), "Hello!"]

    llm_client.chat(MESSAGES, db_path=db_path)

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT model, success, error_code FROM calls ORDER BY id"
    ).fetchall()
    connection.close()

    assert rows == [(FIRST, 0, 503), (FIRST, 1, None)]
