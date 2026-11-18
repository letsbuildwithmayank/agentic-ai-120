"""Run every prompt version on every test message, score each answer, and save results.csv."""

import csv
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from agentic_ai_120.eval_scoring import score_row
from agentic_ai_120.llm_client import chat
from agentic_ai_120.retry import APIError

load_dotenv()  # API keys from .env

HERE = Path(__file__).parent  # the prompt_harness folder, wherever you run this from
EVAL_MODEL = "groq/openai/gpt-oss-20b"  # ONE fixed model, so only the prompt changes
PAUSE_SECONDS = (
    8  # long prompts use many tokens; pause to stay under the tokens-per-minute limit
)
MAX_RETRIES = 4  # waits about 1, 2, 4, 8 seconds before giving up on a call


def find_prompt_versions():
    """Every .md file in prompts/ is one version: v1.md -> 'v1', v2_draft.md -> 'v2_draft'."""
    return sorted(path.stem for path in (HERE / "prompts").glob("*.md"))


def clean_message(text):
    """Remove <message> tags a customer might type, so they can't close our tag early."""
    return re.sub(r"</?message>", "", text, flags=re.IGNORECASE)


def load_rows():
    """Read messages.csv as a list of dicts, cleaning spaces and a possible BOM in the header."""
    with open(HERE / "messages.csv", encoding="utf-8-sig", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            clean_row = {}
            for key, value in row.items():
                if key is None:
                    continue  # extra cells with no header
                clean_row[key.strip()] = (value or "").strip()
            rows.append(clean_row)
        return rows


def run_one(system_prompt, row):
    """Ask the model about one message. Returns (output_text, model_used, error)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"<message>{clean_message(row['message'])}</message>",
        },
    ]
    try:
        output_text, model_used = chat(
            messages, model=EVAL_MODEL, temperature=0, max_retries=MAX_RETRIES
        )
    except APIError as error:
        return (
            "",
            EVAL_MODEL,
            f"API error {error.code}",
        )  # recorded, and kept out of the score below
    return output_text, model_used, ""


def main():
    versions = find_prompt_versions()
    rows = load_rows()
    print(
        f"Versions: {versions} | Messages: {len(rows)} | Calls: {len(versions) * len(rows)}"
    )
    results = []

    for version in versions:
        system_prompt = (HERE / "prompts" / f"{version}.md").read_text(encoding="utf-8")
        print(f"\n=== {version} ===")

        for row in rows:
            output_text, model_used, error = run_one(system_prompt, row)
            scores = score_row(row, output_text)
            results.append(
                {
                    "prompt_version": version,
                    "id": row["id"],
                    "message": row["message"],
                    "expected_intent": row["expected_intent"],
                    "expected_items": row["expected_items"],
                    "output_text": output_text,
                    "model_used": model_used,
                    "error": error,
                    **scores,  # adds valid_json, intent_match, items_match, score
                }
            )
            print(f"  #{row['id']:>2}  score {scores['score']:.2f}  {error}")
            time.sleep(PAUSE_SECONDS)

    with open(HERE / "results.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Average score per version (API errors excluded) ===")
    for version in versions:
        version_rows = [r for r in results if r["prompt_version"] == version]
        scored = [r["score"] for r in version_rows if not r["error"]]
        errors = len(version_rows) - len(scored)
        if scored:
            average = sum(scored) / len(scored)
            print(
                f"  {version}: {average:.0%}  ({len(scored)} scored, {errors} API errors)"
            )
        else:
            print(f"  {version}: no successful calls ({errors} API errors)")


if __name__ == "__main__":
    main()
