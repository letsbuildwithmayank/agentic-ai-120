"""Retry an LLM call with exponential backoff and jitter."""

import random
import time

# HTTP codes worth retrying: timeouts, rate limits and server errors
RETRYABLE_CODES = [408, 429, 500, 503, 504]


class APIError(Exception):
    """An error from an LLM provider, carrying its HTTP status code."""

    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code = code


def call_with_retries(make_call, max_retries=3, base=1.0, max_wait=30.0):
    """Run make_call(); on a retryable error, wait longer each time and try again."""
    for attempt in range(max_retries + 1):
        try:
            return make_call()
        except APIError as error:
            if error.code not in RETRYABLE_CODES:
                raise  # 401, 404...: retrying won't help, give up now
            if attempt == max_retries:
                raise  # out of retries, let the caller try a fallback

            wait = base * (2**attempt)  # 1, 2, 4, 8...
            wait = min(wait, max_wait)  # never wait too long
            wait += random.uniform(0, 1)  # jitter
            time.sleep(wait)
