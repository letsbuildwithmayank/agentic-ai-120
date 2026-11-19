"""Same questions, two versions of the tool descriptions: good vs bad.

Only the description texts change. The tool names, the parameters, the model
and the questions stay the same, so any difference comes from the descriptions.
Saves results.csv (one row per question) and traces.jsonl (every step).
"""

import csv
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from agentic_ai_120.menu_tool_schemas import TOOLS
from agentic_ai_120.retry import APIError
from agentic_ai_120.tool_loop import run_agent

load_dotenv()  # API keys from .env

HERE = Path(__file__).parent
PAUSE_SECONDS = 5  # stay under Groq's tokens-per-minute limit

# Same names and parameters as TOOLS, but vague descriptions:
# no "when to use", no item_id format, and "the dish" could mean a name or an id
BAD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Search.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "text"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Price of a dish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "the dish"}
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_bill",
            "description": "Bill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "list",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_id": {"type": "string", "description": "id"},
                                "qty": {"type": "integer", "description": "number"},
                            },
                            "required": ["item_id", "qty"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
]

VERSIONS = {"good": TOOLS, "bad": BAD_TOOLS}

# Every right answer must contain all of these (numbers are checked as whole numbers: "40" doesn't match "240")
QUESTIONS = [
    {"question": "naan kitne ka hai?", "must_contain": ["40"]},
    {
        "question": "2 butter chicken aur 4 naan ka bill kitna hoga?",
        "must_contain": ["840"],
    },
    {
        "question": "ek panner tikka aur do lassi ka total batao",
        "must_contain": ["400"],
    },
    {"question": "3 coke aur 2 daal makhni ka bill", "must_contain": ["590"]},
    {"question": "5 veg biryani ka bill kitna?", "must_contain": ["1200"]},
    {
        "question": "lassi aur coke mein sasta kaunsa hai?",
        "must_contain": ["coke", "50"],
    },
    {
        "question": "sabse sasti biryani kaunsi hai?",
        "must_contain": ["veg biryani", "240"],
    },
]


def contains(answer, expected):
    """True if the answer has the expected word or number. '₹1,200' counts as 1200."""
    text = answer.lower().replace(",", "")
    if expected.isdigit():
        return re.search(rf"(?<!\d){expected}(?!\d)", text) is not None
    return expected.lower() in text


def summarize(trace):
    """Count what happened in one run: tool calls, tool errors, rounds, forced stop."""
    tool_steps = [step for step in trace if step["type"] == "tool"]
    errors = [step for step in tool_steps if "error" in step["result"]]
    return {
        "tool_calls": len(tool_steps),
        "tool_errors": len(errors),
        "rounds": trace[-1]["round"],
        "forced_stop": int(trace[-1]["type"] == "forced_answer"),
    }


def main():
    results = []
    with open(HERE / "traces.jsonl", "w", encoding="utf-8") as trace_file:
        for version, tools in VERSIONS.items():
            print(f"\n=========== {version} descriptions ===========")
            for item in QUESTIONS:
                try:
                    answer, trace = run_agent(item["question"], tools=tools)
                    api_error = ""
                except APIError as error:
                    answer, trace, api_error = (
                        "",
                        [{"round": 0, "type": "api_error"}],
                        f"API error {error.code}",
                    )
                    print(f"  {api_error}")

                correct = int(
                    not api_error
                    and all(
                        contains(answer or "", word) for word in item["must_contain"]
                    )
                )
                stats = (
                    summarize(trace)
                    if not api_error
                    else {
                        "tool_calls": 0,
                        "tool_errors": 0,
                        "rounds": 0,
                        "forced_stop": 0,
                    }
                )
                results.append(
                    {
                        "version": version,
                        "question": item["question"],
                        "answer": answer,
                        "correct": correct,
                        **stats,
                        "api_error": api_error,
                    }
                )
                trace_file.write(
                    json.dumps(
                        {
                            "version": version,
                            "question": item["question"],
                            "trace": trace,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                time.sleep(PAUSE_SECONDS)

    with open(HERE / "results.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== good vs bad descriptions (API errors left out) ===")
    print(
        f"{'version':<9}{'correct':>9}{'tool calls':>12}{'tool errors':>13}{'rounds':>8}"
    )
    for version in VERSIONS:
        rows = [r for r in results if r["version"] == version and not r["api_error"]]
        if not rows:
            print(f"{version:<9}  no successful runs")
            continue
        correct = sum(r["correct"] for r in rows)
        calls = sum(r["tool_calls"] for r in rows)
        errors = sum(r["tool_errors"] for r in rows)
        rounds = sum(r["rounds"] for r in rows) / len(rows)
        print(
            f"{version:<9}{f'{correct}/{len(rows)}':>9}{calls:>12}{errors:>13}{rounds:>8.1f}"
        )


if __name__ == "__main__":
    main()
