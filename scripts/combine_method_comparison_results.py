from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine checkpointed approximation-method task outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    summary_rows: List[Dict[str, str]] = []
    episode_rows: List[Dict[str, str]] = []
    for path in sorted(input_dir.glob("tasks/methods/*/*/rr_approximation_methods_comparison.csv")):
        rows = read_rows(path)
        for row in rows:
            if args.require_complete and str(row.get("complete", "")) not in {"1.0", "1", "True", "true"}:
                continue
            summary_rows.append(row)
    for path in sorted(input_dir.glob("tasks/methods/*/*/rr_approximation_method_episode_results.csv")):
        episode_rows.extend(read_rows(path))

    best_by_environment: Dict[str, float] = {}
    for row in summary_rows:
        environment = row["environment"]
        mean_utility = float(row["mean_utility"])
        best_by_environment[environment] = max(
            mean_utility,
            best_by_environment.get(environment, float("-inf")),
        )
    for row in summary_rows:
        best = best_by_environment.get(row["environment"])
        if best is not None:
            row["regret_vs_best_rr_approximation"] = str(best - float(row["mean_utility"]))

    write_rows(output_dir / "rr_approximation_methods_comparison.csv", summary_rows)
    write_rows(output_dir / "rr_approximation_method_episode_results.csv", episode_rows)
    status_rows = [
        {
            "summary_rows": len(summary_rows),
            "episode_rows": len(episode_rows),
            "input_dir": str(input_dir),
            "require_complete": args.require_complete,
        }
    ]
    write_rows(output_dir / "method_comparison_combine_status.csv", status_rows)
    print(
        "Combined "
        f"{len(summary_rows)} method summaries and {len(episode_rows)} episode rows into {output_dir}"
    )


if __name__ == "__main__":
    main()
