"""Tests for call_with_retries, using fake API calls (no internet, no cost)."""

import time

import pytest

from agentic_ai_120.retry import APIError, call_with_retries


def make_fake_api(results):
    """Build a fake API: each call takes the next item; errors are raised, answers returned."""
    calls = {"count": 0}

    def fake_api():
        result = results[calls["count"]]
        calls["count"] += 1
        if isinstance(result, APIError):
            raise result
        return result

    return fake_api, calls


@pytest.fixture
def waits(monkeypatch):
    """Replace time.sleep so tests don't really wait; record each wait instead."""
    recorded = []

    def fake_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    return recorded


def test_success_on_first_try(waits):
    # Arrange
    fake_api, calls = make_fake_api(["Hello!"])
    # Act
    answer = call_with_retries(fake_api)
    # Assert
    assert answer == "Hello!"
    assert calls["count"] == 1
    assert waits == []  # never waited


def test_retries_then_succeeds(waits):
    fake_api, calls = make_fake_api([APIError(503), APIError(503), "Hello!"])

    answer = call_with_retries(fake_api)

    assert answer == "Hello!"
    assert calls["count"] == 3
    assert len(waits) == 2  # waited before the 2nd and 3rd try


def test_401_is_not_retried(waits):
    fake_api, calls = make_fake_api([APIError(401), "Hello!"])

    with pytest.raises(APIError) as error_info:
        call_with_retries(fake_api)

    assert error_info.value.code == 401  # it is exactly the 401 error
    assert calls["count"] == 1  # gave up straight away
    assert waits == []


def test_gives_up_after_max_retries(waits):
    fake_api, calls = make_fake_api([APIError(503)] * 10)

    with pytest.raises(APIError) as error_info:
        call_with_retries(fake_api, max_retries=3)

    assert error_info.value.code == 503  # the last error is passed on
    assert calls["count"] == 4  # 1 first try + 3 retries


def test_backoff_doubles_with_jitter(waits):
    fake_api, calls = make_fake_api(
        [APIError(429), APIError(429), APIError(429), "Hello!"]
    )

    answer = call_with_retries(fake_api, base=1.0)

    assert answer == "Hello!"
    assert calls["count"] == 4  # 3 errors + 1 success
    # Expected waits: 1, 2, 4 seconds, each plus 0-1 second of jitter
    assert 1 <= waits[0] < 2
    assert 2 <= waits[1] < 3
    assert 4 <= waits[2] < 5


@pytest.mark.parametrize("code", [408, 429, 500, 503, 504])
def test_every_retryable_code_is_retried(waits, code):
    fake_api, calls = make_fake_api([APIError(code), "Hello!"])

    answer = call_with_retries(fake_api)

    assert answer == "Hello!"
    assert calls["count"] == 2
