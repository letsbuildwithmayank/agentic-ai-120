"""Extract an Order from every message in messages.csv and report field-level accuracy.

Saves results.csv (one row per message, one 0/1 score per field) and prints:
accuracy per field, overall field accuracy (target: 90% or more), and the rows that lost points.
"""

import csv
import time
from pathlib import Path

from dotenv import load_dotenv

from agentic_ai_120.extractor import extract_order
from agentic_ai_120.order_scoring import FIELDS, score_order

load_dotenv()  # API keys from .env

HERE = Path(__file__).parent
PAUSE_SECONDS = 8  # stay under Groq's tokens-per-minute limit (4 gave 429s in run 1)
TARGET = 0.90


def load_rows():
    """Read messages.csv as a list of dicts, cleaning spaces and a possible BOM in the header."""
    with open(HERE / "messages.csv", encoding="utf-8-sig", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {
                    key.strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
            )
        return rows


def main():
    rows = load_rows()
    results = []
    print(f"Messages: {len(rows)}")

    for row in rows:
        try:
            order, reasks = extract_order(row["message"])
            extracted, error = order.model_dump_json(), ""
        except (
            Exception
        ) as error_object:  # never validated, or an API error: score 0, but keep going
            order, reasks, extracted = None, 0, ""
            error = f"{type(error_object).__name__}: {str(error_object)[:120]}"

        scores = score_order(row, order)
        results.append(
            {
                "id": row["id"],
                "message": row["message"],
                **scores,
                "reasks": reasks,
                "extracted": extracted,
                "error": error,
            }
        )
        lost = [field for field in FIELDS if scores[field] == 0]
        print(
            f"  #{row['id']:>2}  {sum(scores.values())}/{len(FIELDS)}  re-asks {reasks}  "
            f"{'lost: ' + ', '.join(lost) if lost else ''}  {error}"
        )
        time.sleep(PAUSE_SECONDS)

    with open(HERE / "results.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Accuracy per field ===")
    for field in FIELDS:
        right = sum(result[field] for result in results)
        print(f"  {field:<14} {right}/{len(results)}  ({right / len(results):.0%})")

    total_right = sum(result[field] for result in results for field in FIELDS)
    total = len(results) * len(FIELDS)
    accuracy = total_right / total
    print(
        f"\nField-level accuracy: {total_right}/{total} = {accuracy:.1%}  (target {TARGET:.0%}: "
        f"{'PASS' if accuracy >= TARGET else 'NOT YET'})"
    )
    print(
        f"Whole order fully right: {sum(all(r[f] for f in FIELDS) for r in results)}/{len(results)}"
    )
    print(
        f"Re-asks in total: {sum(result['reasks'] for result in results)}  |  "
        f"Failed extractions: {sum(1 for result in results if result['error'])}"
    )


if __name__ == "__main__":
    main()
