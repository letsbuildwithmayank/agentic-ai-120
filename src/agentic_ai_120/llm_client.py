"""One reusable LLM client for every project.

How a call flows:

    chat()                   -> which model? (fallback chain)
      call_with_retries()    -> how many tries? (retry + backoff, from retry.py)
        _call_once()         -> one real LiteLLM call, logged to SQLite,
                                with errors turned into APIError(code)
"""

import sqlite3
import time
from datetime import UTC, datetime

import litellm
from litellm import completion

from agentic_ai_120.retry import APIError, call_with_retries

litellm.suppress_debug_info = True  # hide LiteLLM's help messages

# Fallback chain, tried in this order. Reorder it using the call log!
DEFAULT_MODELS = [
    "groq/openai/gpt-oss-20b",
    "gemini/gemini-3.8-flash",
    "openrouter/google/gemma-4-31b-it:free",
]

DB_PATH = "llm_calls.db"


def init_db(db_path=DB_PATH):
    """Create the log table if it doesn't exist yet."""
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    TEXT,
            model         TEXT,
            success       INTEGER,
            error_code    INTEGER,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            latency_ms    REAL,
            cost_usd      REAL
        )
        """
    )
    connection.commit()
    connection.close()


def log_call(
    model,
    success,
    error_code,
    input_tokens,
    output_tokens,
    latency_ms,
    cost_usd,
    db_path=DB_PATH,
):
    """Add one row: what happened in one LLM call."""
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO calls
            (created_at, model, success, error_code, input_tokens, output_tokens, latency_ms, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            model,
            int(success),
            error_code,
            input_tokens,
            output_tokens,
            latency_ms,
            cost_usd,
        ),
    )
    connection.commit()
    connection.close()


def _completion_args(
    model, messages, timeout, temperature, stream=False, tools=None, tool_choice=None
):
    """Build the arguments for LiteLLM. Optional ones are only sent when they're set."""
    args = {"model": model, "messages": messages, "timeout": timeout}
    if temperature is not None:
        args["temperature"] = temperature
    if stream:
        args["stream"] = True
    if tools is not None:
        args["tools"] = tools
        if tool_choice is not None:
            args["tool_choice"] = tool_choice
    return args


def _get_cost_usd(response):
    """Ask LiteLLM what a call cost in USD. Unknown or free models -> None."""
    try:
        return litellm.completion_cost(completion_response=response)
    except Exception:  # any pricing problem just means "cost unknown"
        return None


def _fail(error, model, start, db_path):
    """Log a failed call and raise it as APIError(code). Errors without a code are bugs: re-raise them."""
    latency_ms = (time.perf_counter() - start) * 1000
    code = getattr(error, "status_code", None)
    if code is None:
        raise error
    log_call(model, False, code, None, None, latency_ms, None, db_path)
    raise APIError(code, str(error)[:200]) from error


def _call_once(
    model, messages, timeout, temperature, db_path, tools=None, tool_choice=None
):
    """Make ONE normal (non-streaming) call and log it."""
    start = time.perf_counter()
    try:
        response = completion(
            **_completion_args(
                model,
                messages,
                timeout,
                temperature,
                tools=tools,
                tool_choice=tool_choice,
            )
        )
    except Exception as error:  # _fail() re-raises anything that isn't an API error
        _fail(error, model, start, db_path)

    latency_ms = (time.perf_counter() - start) * 1000
    log_call(
        model,
        True,
        None,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        latency_ms,
        _get_cost_usd(response),
        db_path,
    )
    return response


def _open_stream(model, messages, timeout, temperature, db_path):
    """Start a streaming call. Returns (stream, start_time)."""
    start = time.perf_counter()
    try:
        stream = completion(
            **_completion_args(model, messages, timeout, temperature, stream=True)
        )
    except Exception as error:  # _fail() re-raises anything that isn't an API error
        _fail(error, model, start, db_path)
    return stream, start


def _stream_pieces(model, messages, stream, start, db_path):
    """Yield text pieces as they arrive, then log the whole call."""
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece

    latency_ms = (time.perf_counter() - start) * 1000
    full_response = litellm.stream_chunk_builder(chunks, messages=messages)
    log_call(
        model,
        True,
        None,
        full_response.usage.prompt_tokens,
        full_response.usage.completion_tokens,
        latency_ms,
        _get_cost_usd(full_response),
        db_path,
    )


def chat(
    messages,
    model=None,
    stream=False,
    max_retries=2,
    timeout=30,
    temperature=None,
    tools=None,
    tool_choice=None,
    db_path=DB_PATH,
):
    """Send messages to an LLM, with retries, a fallback chain and a call log.

    model=None  -> try DEFAULT_MODELS in order; model="..." -> use only that model.
    temperature=None -> the provider's default; 0 -> most stable answers (good for evals).
    stream=False -> returns (answer_text, model_used)
    stream=True  -> returns (generator_of_text_pieces, model_used)
    tools=[...]  -> returns (message, model_used): message.content is the text (or None),
                    message.tool_calls is the list of tool requests (or None).
                    tool_choice: "auto" (default), "required" or "none".
    """
    if stream and tools is not None:
        raise ValueError("stream=True with tools is not supported yet")

    init_db(db_path)
    if model is not None:
        models = [model]
    else:
        models = DEFAULT_MODELS
    last_error = None
    for current_model in models:
        # The default value saves a COPY of current_model right now,
        # so this function never sees a later loop value (Ruff B023).
        def make_call(model_for_this_try=current_model):
            if stream:
                return _open_stream(
                    model_for_this_try, messages, timeout, temperature, db_path
                )
            return _call_once(
                model_for_this_try,
                messages,
                timeout,
                temperature,
                db_path,
                tools,
                tool_choice,
            )

        try:
            result = call_with_retries(make_call, max_retries=max_retries)
        except APIError as error:
            last_error = error
            continue  # this model failed even after retries: try the next one

        if stream:
            opened_stream, start = result
            pieces = _stream_pieces(
                current_model, messages, opened_stream, start, db_path
            )
            return pieces, current_model
        if tools is not None:
            return result.choices[
                0
            ].message, current_model  # the whole message: text AND tool calls
        return result.choices[0].message.content, current_model

    raise last_error  # every model failed
