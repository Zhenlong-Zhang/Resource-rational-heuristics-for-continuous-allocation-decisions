"""Summarize baseline experiments result folders.

This helper reads completed or partially completed result folders and writes a
compact Markdown summary. It is intentionally lightweight: it uses only the
Python standard library so it can run locally or on Hoffman2 without extra setup.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


BEHAVIOR_FILE = "targeted_regime_rr_behavior_profiles.csv"
FINAL_CHOICE_FILE = "targeted_regime_final_choice_comparison.csv"
BEHAVIOR_CANDIDATE_FILE = "targeted_regime_behavior_candidates.csv"
FINAL_CHOICE_CANDIDATE_FILE = "targeted_regime_final_choice_candidates.csv"
STATUS_FILE = "parallel_run_status.csv"
SUMMARY_FILE = "parallel_summary.md"

PARAMETER_COLUMNS = [
    "grid_name",
    "grid_index",
    "environment",
    "mu_need",
    "sigma_need",
    "sigma_sample",
    "total_time",
    "sample_time_cost",
    "lambda_shortfall",
    "utility_exponent",
    "prior_sample_count",
    "prior_sample_count_1",
    "prior_sample_count_2",
    "learning_per_unit_of_tutoring",
    "delta_learning_per_unit_tutoring",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def format_number(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def compact_params(row: dict[str, str]) -> str:
    parts: list[str] = []
    for column in PARAMETER_COLUMNS:
        value = row.get(column)
        if value not in (None, ""):
            parts.append(f"{column}={value}")
    return ", ".join(parts) if parts else "no parameter columns found"


def count_statuses(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        return "no status file"
    counts = Counter(row.get("status", "unknown") for row in rows)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def candidate_counts(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        return "none"
    counts = Counter(row.get("candidate_type", "unknown") for row in rows)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def top_rows(
    rows: Sequence[dict[str, str]],
    *,
    key_columns: Sequence[str],
    limit: int,
    reverse: bool = True,
) -> list[dict[str, str]]:
    def score(row: dict[str, str]) -> tuple[float, ...]:
        return tuple(as_float(row, column) for column in key_columns)

    return sorted(rows, key=score, reverse=reverse)[:limit]


def behavior_line(row: dict[str, str]) -> str:
    metrics = {
        "near_50_50": as_float(row, "near_equal_allocation_rate"),
        "equal_outcome": as_float(row, "equal_outcome_rate"),
        "abs_from_50_50": as_float(row, "mean_abs_allocation_from_equal"),
        "equal_outcome_gap": as_float(row, "mean_equal_outcome_allocation_gap"),
        "utility": as_float(row, "mean_utility"),
        "ci95": as_float(row, "mean_utility_ci95"),
        "samples": as_float(row, "mean_sample_count"),
    }
    metric_text = ", ".join(f"{key}={format_number(value)}" for key, value in metrics.items())
    return f"- {metric_text}; {compact_params(row)}"


def final_choice_line(row: dict[str, str]) -> str:
    metrics = {
        "match": as_float(row, "final_choice_match_rate"),
        "mean_abs_gap": as_float(row, "mean_abs_allocation_gap"),
        "rmse_gap": as_float(row, "rmse_allocation_gap"),
        "utility_gap_rr_minus_heuristic": as_float(row, "utility_gap_rr_minus_heuristic"),
    }
    metric_text = ", ".join(f"{key}={format_number(value)}" for key, value in metrics.items())
    heuristic = row.get("heuristic", "unknown")
    return f"- heuristic={heuristic}, {metric_text}; {compact_params(row)}"


def summarize_folder(folder: Path, *, top_n: int) -> list[str]:
    status_rows = read_csv(folder / STATUS_FILE)
    behavior_rows = read_csv(folder / BEHAVIOR_FILE)
    final_choice_rows = read_csv(folder / FINAL_CHOICE_FILE)
    behavior_candidates = read_csv(folder / BEHAVIOR_CANDIDATE_FILE)
    final_choice_candidates = read_csv(folder / FINAL_CHOICE_CANDIDATE_FILE)

    lines = [
        f"## {folder}",
        "",
        f"- status: {count_statuses(status_rows)}",
        f"- has_parallel_summary: {(folder / SUMMARY_FILE).exists()}",
        f"- behavior_rows: {len(behavior_rows)}",
        f"- final_choice_rows: {len(final_choice_rows)}",
        f"- behavior_candidate_counts: {candidate_counts(behavior_candidates)}",
        f"- final_choice_candidate_counts: {candidate_counts(final_choice_candidates)}",
        "",
    ]

    if behavior_rows:
        lines.extend(
            [
                "### Top Near-50/50 Behavior Rows",
                "",
                *[
                    behavior_line(row)
                    for row in top_rows(
                        behavior_rows,
                        key_columns=["near_equal_allocation_rate", "equal_outcome_rate"],
                        limit=top_n,
                    )
                ],
                "",
                "### Top Equal-Outcome Rows Distinct From 50/50",
                "",
                *[
                    behavior_line(row)
                    for row in top_rows(
                        behavior_rows,
                        key_columns=["equal_outcome_rate", "mean_abs_allocation_from_equal"],
                        limit=top_n,
                    )
                ],
                "",
            ]
        )

    if final_choice_rows:
        lines.extend(
            [
                "### Closest Final-Choice Heuristic Rows",
                "",
            ]
        )
        closest = sorted(
            final_choice_rows,
            key=lambda row: (
                as_float(row, "final_choice_match_rate"),
                -as_float(row, "mean_abs_allocation_gap"),
            ),
            reverse=True,
        )[:top_n]
        lines.extend([final_choice_line(row) for row in closest])
        lines.append("")

    return lines


def summarize(folders: Iterable[Path], *, top_n: int) -> str:
    lines = [
        "# baseline experiments Result Summary",
        "",
        "This file is generated by `scripts/summarize_results.py`.",
        "It summarizes result folders; it does not rerun simulations.",
        "",
    ]
    for folder in folders:
        lines.extend(summarize_folder(folder, top_n=top_n))
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path, help="Result folders to summarize.")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of top rows per section.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = summarize(args.folders, top_n=args.top_n)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
