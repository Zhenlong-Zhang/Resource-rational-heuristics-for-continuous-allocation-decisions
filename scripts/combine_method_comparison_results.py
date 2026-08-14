from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine checkpointed approximation-method task outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--task-manifest", default="jobs/method_comparison_approx_methods_tasks.tsv")
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


def mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def method_family(policy: str) -> str:
    if policy.startswith("discretized_dp"):
        return "discretized_dp"
    if policy.startswith("myopic_voi"):
        return "myopic_voi"
    if policy.startswith("blinkered"):
        return "blinkered"
    return policy


def is_canonical_config(row: Dict[str, str]) -> bool:
    family = row.get("method_family") or method_family(row.get("policy", ""))
    if family in {"myopic_voi", "blinkered"}:
        return True
    return (
        family == "discretized_dp"
        and float(row.get("policy_max_samples") or 0.0) == 10.0
        and float(row.get("policy_mean_grid_size") or 0.0) == 50.0
        and float(row.get("policy_observation_branches") or 0.0) == 7.0
    )


def parse_task_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            while len(parts) < 6:
                parts.append("")
            rows.append(
                {
                    "manifest_line": str(line_number),
                    "environment": parts[0],
                    "policy_arg": parts[1],
                    "policy": parts[2],
                    "dp_max_samples": parts[3],
                    "dp_mean_grid_size": parts[4],
                    "dp_observation_branches": parts[5],
                }
            )
    return rows


def complete_flag(row: Dict[str, str]) -> bool:
    return str(row.get("complete", "")) in {"1.0", "1", "True", "true"}


def validate_against_manifest(
    manifest_rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, str]],
    require_complete: bool,
) -> List[Dict[str, str]]:
    by_key: Dict[Tuple[str, str], Dict[str, str]] = {
        (row.get("environment", ""), row.get("policy", "")): row
        for row in summary_rows
    }
    status_rows: List[Dict[str, str]] = []
    for task in manifest_rows:
        key = (task["environment"], task["policy"])
        row = by_key.get(key)
        status = "ok"
        reason = ""
        if row is None:
            status = "missing"
            reason = "summary CSV missing"
        elif not complete_flag(row):
            status = "incomplete"
            reason = f"completed={row.get('completed_episodes')} target={row.get('target_episodes')}"
        elif row.get("completed_episodes") != row.get("target_episodes"):
            status = "episode_mismatch"
            reason = f"completed={row.get('completed_episodes')} target={row.get('target_episodes')}"
        status_rows.append(
            {
                **task,
                "status": status,
                "reason": reason,
                "completed_episodes": "" if row is None else row.get("completed_episodes", ""),
                "target_episodes": "" if row is None else row.get("target_episodes", ""),
                "complete": "" if row is None else row.get("complete", ""),
            }
        )
    if require_complete:
        failures = [row for row in status_rows if row["status"] != "ok"]
        if failures:
            preview = "; ".join(
                f"{row['environment']}/{row['policy']}={row['status']}" for row in failures[:10]
            )
            raise RuntimeError(
                f"Manifest completeness check failed for {len(failures)} tasks: {preview}"
            )
    return status_rows


def enrich_summary_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    enriched: List[Dict[str, str]] = []
    for row in rows:
        row = dict(row)
        row["method_family"] = method_family(row.get("policy", ""))
        row["config_id"] = row.get("policy", "")
        row["canonical_config"] = "1.0" if is_canonical_config(row) else "0.0"
        enriched.append(row)
    return enriched


def episode_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (row.get("environment", ""), row.get("episode_index", ""))


def build_common_randomness_rows(episode_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in episode_rows:
        grouped.setdefault(episode_key(row), []).append(row)
    check_rows: List[Dict[str, str]] = []
    for (environment, episode_index), rows in sorted(grouped.items()):
        fingerprints = {row.get("episode_fingerprint", "") for row in rows}
        true_state_pairs = {
            (row.get("true_need_1", ""), row.get("true_need_2", ""))
            for row in rows
        }
        stream_pairs = {
            (row.get("observation_stream_hash_1", ""), row.get("observation_stream_hash_2", ""))
            for row in rows
        }
        check_rows.append(
            {
                "environment": environment,
                "episode_index": episode_index,
                "n_policy_rows": len(rows),
                "unique_episode_fingerprints": len(fingerprints),
                "unique_true_states": len(true_state_pairs),
                "unique_observation_stream_pairs": len(stream_pairs),
                "common_randomness_ok": 1.0
                if len(fingerprints) == len(true_state_pairs) == len(stream_pairs) == 1
                else 0.0,
            }
        )
    return check_rows


def pairwise_rows(summary_rows: List[Dict[str, str]], episode_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    episodes_by_env_policy: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = {}
    for row in episode_rows:
        key = (row.get("environment", ""), row.get("policy", ""))
        episodes_by_env_policy.setdefault(key, {})[row.get("episode_index", "")] = row

    summaries_by_environment: Dict[str, List[Dict[str, str]]] = {}
    for row in summary_rows:
        if complete_flag(row):
            summaries_by_environment.setdefault(row["environment"], []).append(row)

    rows: List[Dict[str, str]] = []
    for environment, summaries in sorted(summaries_by_environment.items()):
        for summary_a, summary_b in itertools.combinations(sorted(summaries, key=lambda row: row["policy"]), 2):
            policy_a = summary_a["policy"]
            policy_b = summary_b["policy"]
            episodes_a = episodes_by_env_policy.get((environment, policy_a), {})
            episodes_b = episodes_by_env_policy.get((environment, policy_b), {})
            common_indices = sorted(set(episodes_a) & set(episodes_b), key=lambda value: int(value))
            utility_diffs: List[float] = []
            sample_diffs: List[float] = []
            sample_1_diffs: List[float] = []
            sample_2_diffs: List[float] = []
            allocation_diffs: List[float] = []
            fingerprint_mismatches = 0
            for episode_index in common_indices:
                row_a = episodes_a[episode_index]
                row_b = episodes_b[episode_index]
                if row_a.get("episode_fingerprint") != row_b.get("episode_fingerprint"):
                    fingerprint_mismatches += 1
                utility_diffs.append(float(row_a["realized_utility"]) - float(row_b["realized_utility"]))
                sample_diffs.append(float(row_a["sample_count"]) - float(row_b["sample_count"]))
                sample_1_diffs.append(float(row_a["sample_1_count"]) - float(row_b["sample_1_count"]))
                sample_2_diffs.append(float(row_a["sample_2_count"]) - float(row_b["sample_2_count"]))
                allocation_diffs.append(float(row_a["allocation_to_person1"]) - float(row_b["allocation_to_person1"]))
            sample_time_cost = float(summary_a.get("sample_time_cost") or 1.0)
            utility_ci = ci95(utility_diffs)
            rows.append(
                {
                    "environment": environment,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "method_family_a": summary_a.get("method_family", method_family(policy_a)),
                    "method_family_b": summary_b.get("method_family", method_family(policy_b)),
                    "canonical_pair": 1.0
                    if summary_a.get("canonical_config") == "1.0" and summary_b.get("canonical_config") == "1.0"
                    else 0.0,
                    "n_paired_episodes": len(common_indices),
                    "fingerprint_mismatch_count": fingerprint_mismatches,
                    "mean_utility_difference_a_minus_b": mean(utility_diffs),
                    "utility_difference_ci95": utility_ci,
                    "abs_mean_utility_difference": abs(mean(utility_diffs)),
                    "utility_ci95_gt_sample_time_cost": 1.0 if utility_ci > sample_time_cost else 0.0,
                    "mean_sample_count_difference_a_minus_b": mean(sample_diffs),
                    "sample_count_difference_ci95": ci95(sample_diffs),
                    "mean_sample_1_count_difference_a_minus_b": mean(sample_1_diffs),
                    "mean_sample_2_count_difference_a_minus_b": mean(sample_2_diffs),
                    "mean_allocation_difference_a_minus_b": mean(allocation_diffs),
                    "allocation_difference_ci95": ci95(allocation_diffs),
                    "sample_time_cost": sample_time_cost,
                }
            )
    return rows


def action_pattern_rows(episode_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    counts: Dict[Tuple[str, str, str], int] = {}
    totals: Dict[Tuple[str, str], int] = {}
    for row in episode_rows:
        key = (row.get("environment", ""), row.get("policy", ""))
        pattern_key = (key[0], key[1], row.get("action_sequence", ""))
        counts[pattern_key] = counts.get(pattern_key, 0) + 1
        totals[key] = totals.get(key, 0) + 1
    rows = []
    for (environment, policy, action_sequence), count in sorted(counts.items()):
        total = totals[(environment, policy)]
        rows.append(
            {
                "environment": environment,
                "policy": policy,
                "method_family": method_family(policy),
                "action_sequence": action_sequence,
                "count": count,
                "rate": count / total if total else math.nan,
            }
        )
    return rows


def family_rows(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in summary_rows:
        if complete_flag(row):
            grouped.setdefault((row["environment"], row["method_family"]), []).append(row)
    for (environment, family), group_rows in sorted(grouped.items()):
        best = max(group_rows, key=lambda row: float(row["mean_utility"]))
        canonical = [row for row in group_rows if row.get("canonical_config") == "1.0"]
        rows.append(
            {
                "environment": environment,
                "method_family": family,
                "n_configs": len(group_rows),
                "best_config_id": best["config_id"],
                "best_mean_utility": best["mean_utility"],
                "best_mean_utility_ci95": best.get("mean_utility_ci95", ""),
                "best_mean_sample_count": best.get("mean_sample_count", ""),
                "canonical_config_id": canonical[0]["config_id"] if canonical else "",
                "canonical_mean_utility": canonical[0]["mean_utility"] if canonical else "",
                "canonical_mean_utility_ci95": canonical[0].get("mean_utility_ci95", "") if canonical else "",
                "canonical_mean_sample_count": canonical[0].get("mean_sample_count", "") if canonical else "",
            }
        )
    return rows


def write_markdown_summary(
    path: Path,
    summary_rows: List[Dict[str, str]],
    pair_rows: List[Dict[str, str]],
    manifest_status_rows: List[Dict[str, str]],
    common_rows: List[Dict[str, str]],
) -> None:
    manifest_failures = [row for row in manifest_status_rows if row.get("status") != "ok"]
    common_failures = [row for row in common_rows if str(row.get("common_randomness_ok")) not in {"1.0", "1"}]
    canonical_pairs = [row for row in pair_rows if str(row.get("canonical_pair")) in {"1.0", "1"}]
    wide_pairs = [
        row for row in canonical_pairs
        if str(row.get("utility_ci95_gt_sample_time_cost")) in {"1.0", "1"}
    ]
    lines = [
        "# MethodComparison Approximation-Method Comparison Summary",
        "",
        "This file is generated by `scripts/combine_method_comparison_results.py`.",
        "",
        "## Completeness",
        "",
        f"- Method/config summaries: `{len(summary_rows)}`",
        f"- Manifest failures: `{len(manifest_failures)}`",
        f"- Common-randomness failures: `{len(common_failures)}`",
        "",
        "## Paired Comparison",
        "",
        f"- Pairwise comparison rows: `{len(pair_rows)}`",
        f"- Canonical pair rows: `{len(canonical_pairs)}`",
        f"- Canonical rows with paired utility CI95 greater than sample_time_cost: `{len(wide_pairs)}`",
        "",
        "Interpretation guardrail: use `method_pairwise_comparison.csv` for method differences. "
        "The per-policy `mean_utility_ci95` values are useful, but paired differences are the safer answer "
        "because all methods share the same episode fingerprints.",
        "",
        "## DP Reporting Guardrail",
        "",
        "Use `method_family_comparison.csv` to distinguish canonical DP from the best tuned DP configuration. "
        "Do not interpret post-hoc best tuned DP as a single pre-registered method.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    manifest_rows = parse_task_manifest(Path(args.task_manifest))
    if args.require_complete and not manifest_rows:
        raise RuntimeError(
            f"--require-complete needs a task manifest, but {args.task_manifest} was not found or was empty"
        )

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
    summary_rows = enrich_summary_rows(summary_rows)
    manifest_status_rows = validate_against_manifest(
        manifest_rows,
        summary_rows,
        args.require_complete,
    ) if manifest_rows else []
    complete_keys = {
        (row.get("environment", ""), row.get("policy", ""))
        for row in summary_rows
        if complete_flag(row)
    }
    if args.require_complete:
        episode_rows = [
            row for row in episode_rows
            if (row.get("environment", ""), row.get("policy", "")) in complete_keys
        ]
        if not summary_rows:
            raise RuntimeError("No complete method-comparison summary rows found")
        if not episode_rows:
            raise RuntimeError("No complete method-comparison episode rows found")

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

    common_rows = build_common_randomness_rows(episode_rows)
    pair_rows = pairwise_rows(summary_rows, episode_rows)
    pattern_rows = action_pattern_rows(episode_rows)
    family_summary_rows = family_rows(summary_rows)

    write_rows(output_dir / "rr_approximation_methods_comparison.csv", summary_rows)
    write_rows(output_dir / "rr_approximation_method_episode_results.csv", episode_rows)
    write_rows(output_dir / "method_comparison_manifest_status.csv", manifest_status_rows)
    write_rows(output_dir / "method_common_randomness_check.csv", common_rows)
    write_rows(output_dir / "method_pairwise_comparison.csv", pair_rows)
    write_rows(output_dir / "method_action_pattern_counts.csv", pattern_rows)
    write_rows(output_dir / "method_family_comparison.csv", family_summary_rows)
    status_rows = [
        {
            "summary_rows": len(summary_rows),
            "episode_rows": len(episode_rows),
            "manifest_rows": len(manifest_rows),
            "manifest_failures": len([row for row in manifest_status_rows if row.get("status") != "ok"]),
            "common_randomness_failures": len([
                row for row in common_rows
                if str(row.get("common_randomness_ok")) not in {"1.0", "1"}
            ]),
            "pairwise_rows": len(pair_rows),
            "input_dir": str(input_dir),
            "require_complete": args.require_complete,
        }
    ]
    write_rows(output_dir / "method_comparison_combine_status.csv", status_rows)
    write_markdown_summary(
        output_dir / "method_comparison_summary.md",
        summary_rows,
        pair_rows,
        manifest_status_rows,
        common_rows,
    )
    if args.require_complete and status_rows[0]["common_randomness_failures"]:
        raise RuntimeError("Common-randomness check failed in combined episode outputs")
    print(
        "Combined "
        f"{len(summary_rows)} method summaries and {len(episode_rows)} episode rows into {output_dir}"
    )
    if manifest_rows:
        print(f"Checked {len(manifest_rows)} manifest tasks")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
