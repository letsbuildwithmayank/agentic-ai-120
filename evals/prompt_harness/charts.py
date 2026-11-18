"""Read results.csv, print the score tables and save chart.png.

Left chart:  average score per prompt version.
Right chart: the three checks (valid JSON, intent, items) per prompt version.
Rows with an API error are left out everywhere.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent

CHECKS = ["valid_json", "intent_match", "items_match"]
CHECK_NAMES = ["Valid JSON", "Intent correct", "Items correct"]
LABELS = ["order", "complaint", "question", "cancel", "other"]

# Blue, orange, aqua: checked to stay different for colour-blind readers
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def load_results():
    """Read results.csv as a list of dicts."""
    with open(HERE / "results.csv", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def rows_of(results, version):
    """All rows of one prompt version, without the API errors."""
    rows = []
    for row in results:
        if row["prompt_version"] == version and not row["error"]:
            rows.append(row)
    return rows


def percent(rows, column):
    """Average of one column (0/1 checks or the score), as a percentage."""
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += float(row[column])
    return total / len(rows) * 100


def print_checks_table(results, versions):
    print("\n=== Score and checks per version ===")
    print(f"{'version':<10}{'rows':>6}{'score':>8}{'json':>8}{'intent':>8}{'items':>8}")
    for version in versions:
        rows = rows_of(results, version)
        line = f"{version:<10}{len(rows):>6}"
        for column in ["score", *CHECKS]:
            line += f"{percent(rows, column):>7.0f}%"
        print(line)


def print_label_table(results, versions):
    """Intent correct for each expected label: which kind of message does each prompt get wrong?"""
    print("\n=== Intent correct, per expected label ===")
    print(f"{'label':<11}" + "".join(f"{version:>10}" for version in versions))
    for label in LABELS:
        line = f"{label:<11}"
        for version in versions:
            label_rows = [
                row
                for row in rows_of(results, version)
                if row["expected_intent"] == label
            ]
            right = sum(int(row["intent_match"]) for row in label_rows)
            line += f"{right}/{len(label_rows)}".rjust(10)
        print(line)


def print_failed_rows(results, versions):
    """Every row that lost points: this is where error analysis starts."""
    print("\n=== Rows that lost points ===")
    for version in versions:
        failed = [row for row in rows_of(results, version) if float(row["score"]) < 1]
        print(f"{version}: {len(failed)} rows")
        for row in failed:
            print(
                f"   #{row['id']:>2}  score {float(row['score']):.2f}  "
                f"expected {row['expected_intent']:<9}  {row['message'][:45]}"
            )


def style(axis, title):
    """Same clean look for both charts: light grid, no box, 0-100% axis."""
    axis.set_facecolor(BACKGROUND)
    axis.set_title(
        title, loc="left", fontsize=13, fontweight="bold", color=TEXT, pad=14
    )
    axis.set_ylim(0, 110)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.grid(axis="y", color=GRID, linewidth=1)
    axis.set_axisbelow(True)  # grid lines behind the bars
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=MUTED, length=0, labelsize=11)


def draw_chart(results, versions):
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=BACKGROUND)

    # Left: one bar per version, with its score written on top
    scores = [percent(rows_of(results, version), "score") for version in versions]
    bars = left.bar(versions, scores, width=0.55, color=COLORS[0])
    for bar, score in zip(bars, scores):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            score + 2,
            f"{score:.0f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=TEXT,
        )
    style(left, "Average score per prompt version")

    # Right: three bars per version, one for each check
    width = 0.26
    for number, check in enumerate(CHECKS):
        values = [percent(rows_of(results, version), check) for version in versions]
        positions = [x + (number - 1) * width for x in range(len(versions))]
        right.bar(
            positions,
            values,
            width=width,
            color=COLORS[number],
            label=CHECK_NAMES[number],
            edgecolor=BACKGROUND,
            linewidth=2,
        )  # thin gap between the bars
    right.set_xticks(range(len(versions)))
    right.set_xticklabels(versions)
    style(right, "Each check per prompt version")
    right.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
        fontsize=11,
        labelcolor=TEXT,
    )

    messages = len(rows_of(results, versions[0]))
    model = results[0]["model_used"]
    figure.text(
        0.01,
        0.012,
        f"One run · {messages} test messages · {model} · temperature 0",
        fontsize=10,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(HERE / "chart.png", dpi=150, facecolor=BACKGROUND)
    print(f"\nSaved {HERE / 'chart.png'}")


def main():
    results = load_results()
    versions = sorted({row["prompt_version"] for row in results})
    print_checks_table(results, versions)
    print_label_table(results, versions)
    print_failed_rows(results, versions)
    draw_chart(results, versions)


if __name__ == "__main__":
    main()
