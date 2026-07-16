#!/usr/bin/env python3
"""Create reproducible Round 3/4 analysis tables and a professor-facing report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


R3_SUMMARY = "rr_approximation_methods_comparison.csv"
R3_EPISODES = "rr_approximation_method_episode_results.csv"
R4_SUMMARY = "r4_diagnostic_environment_summary.csv"
R4_PROFILES = "r4_diagnostic_policy_profiles.csv"
R4_CANDIDATES = "r4_diagnostic_manual_advantage_candidates.csv"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def ci95_half_width(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def paired_summary(values: Sequence[float]) -> dict[str, float]:
    center = mean(values)
    half_width = ci95_half_width(values)
    return {
        "mean_difference": center,
        "ci95_half_width": half_width,
        "ci95_lower": center - half_width,
        "ci95_upper": center + half_width,
        "positive_rate": mean([value > 0 for value in values]),
    }


def f(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def canonical_r3_policies(summary_rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    by_environment: dict[str, dict[str, str]] = defaultdict(dict)
    for row in summary_rows:
        if float(row["canonical_config"]) == 1.0:
            by_environment[row["environment"]][row["method_family"]] = row["policy"]
    required = {"myopic_voi", "blinkered", "discretized_dp"}
    for environment, policies in by_environment.items():
        if set(policies) != required:
            raise RuntimeError(f"Canonical methods are incomplete for {environment}: {policies}")
    return dict(by_environment)


def analyze_r3(r3_dir: Path, output_dir: Path) -> dict[str, object]:
    _, summaries = read_rows(r3_dir / R3_SUMMARY)
    policies = canonical_r3_policies(summaries)
    canonical_names = {name for values in policies.values() for name in values.values()}

    episodes: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    total_episode_rows = 0
    with (r3_dir / R3_EPISODES).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total_episode_rows += 1
            if row["policy"] in canonical_names:
                episodes[(row["environment"], int(row["episode_index"]))][row["policy"]] = row

    pairs = (
        ("myopic_minus_blinkered", "myopic_voi", "blinkered"),
        ("dp_minus_myopic", "discretized_dp", "myopic_voi"),
        ("dp_minus_blinkered", "discretized_dp", "blinkered"),
    )
    paired_rows: list[dict[str, object]] = []
    common_randomness_mismatches = 0
    for environment in sorted(policies):
        environment_episodes = [
            episode for (name, _), episode in episodes.items() if name == environment
        ]
        if len(environment_episodes) != 1200:
            raise RuntimeError(f"Expected 1200 paired episodes for {environment}")
        expected_names = set(policies[environment].values())
        for episode in environment_episodes:
            if set(episode) != expected_names:
                raise RuntimeError(f"Canonical episode coverage is incomplete for {environment}")
            fingerprints = {
                (
                    row["episode_fingerprint"],
                    row["observation_stream_hash_1"],
                    row["observation_stream_hash_2"],
                )
                for row in episode.values()
            }
            common_randomness_mismatches += len(fingerprints) != 1

        for label, left_family, right_family in pairs:
            left = policies[environment][left_family]
            right = policies[environment][right_family]
            utility_differences = [
                f(episode[left], "realized_utility") - f(episode[right], "realized_utility")
                for episode in environment_episodes
            ]
            sample_differences = [
                f(episode[left], "sample_count") - f(episode[right], "sample_count")
                for episode in environment_episodes
            ]
            utility = paired_summary(utility_differences)
            samples = paired_summary(sample_differences)
            paired_rows.append(
                {
                    "environment": environment,
                    "comparison": label,
                    "left_policy": left,
                    "right_policy": right,
                    "n_episodes": len(environment_episodes),
                    "mean_utility_difference": utility["mean_difference"],
                    "utility_difference_ci95_half_width": utility["ci95_half_width"],
                    "utility_difference_ci95_lower": utility["ci95_lower"],
                    "utility_difference_ci95_upper": utility["ci95_upper"],
                    "left_utility_win_rate": utility["positive_rate"],
                    "mean_sample_count_difference": samples["mean_difference"],
                    "sample_count_difference_ci95_half_width": samples["ci95_half_width"],
                    "sample_count_difference_ci95_lower": samples["ci95_lower"],
                    "sample_count_difference_ci95_upper": samples["ci95_upper"],
                }
            )

    paired_fields = list(paired_rows[0])
    write_rows(output_dir / "r3_paired_method_comparisons.csv", paired_fields, paired_rows)

    canonical_summaries = [row for row in summaries if float(row["canonical_config"]) == 1.0]
    sensitivity_rows: list[dict[str, object]] = []
    for environment in sorted(policies):
        environment_rows = [row for row in summaries if row["environment"] == environment]
        canonical_dp = next(
            row for row in environment_rows
            if row["method_family"] == "discretized_dp" and float(row["canonical_config"]) == 1.0
        )
        myopic = next(row for row in environment_rows if row["method_family"] == "myopic_voi")
        best_dp = max(
            (row for row in environment_rows if row["method_family"] == "discretized_dp"),
            key=lambda row: f(row, "mean_utility"),
        )
        sensitivity_rows.append(
            {
                "environment": environment,
                "canonical_dp_policy": canonical_dp["policy"],
                "canonical_dp_mean_utility": f(canonical_dp, "mean_utility"),
                "myopic_mean_utility": f(myopic, "mean_utility"),
                "canonical_dp_minus_myopic": f(canonical_dp, "mean_utility") - f(myopic, "mean_utility"),
                "best_observed_dp_policy": best_dp["policy"],
                "best_observed_dp_mean_utility": f(best_dp, "mean_utility"),
                "best_observed_dp_minus_myopic": f(best_dp, "mean_utility") - f(myopic, "mean_utility"),
                "best_observed_dp_minus_canonical_dp": f(best_dp, "mean_utility") - f(canonical_dp, "mean_utility"),
            }
        )
    write_rows(
        output_dir / "r3_dp_sensitivity.csv",
        list(sensitivity_rows[0]),
        sensitivity_rows,
    )

    by_comparison: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in paired_rows:
        by_comparison[str(row["comparison"])].append(row)
    comparison_overview = {}
    for label, rows in by_comparison.items():
        significant = [
            row for row in rows
            if float(row["utility_difference_ci95_lower"]) > 0
            or float(row["utility_difference_ci95_upper"]) < 0
        ]
        sample_significant = [
            row for row in rows
            if float(row["sample_count_difference_ci95_lower"]) > 0
            or float(row["sample_count_difference_ci95_upper"]) < 0
        ]
        comparison_overview[label] = {
            "environments": len(rows),
            "utility_ci_excludes_zero": len(significant),
            "left_ci_above_zero": sum(
                float(row["utility_difference_ci95_lower"]) > 0 for row in rows
            ),
            "left_ci_below_zero": sum(
                float(row["utility_difference_ci95_upper"]) < 0 for row in rows
            ),
            "mean_of_environment_differences": mean(
                [float(row["mean_utility_difference"]) for row in rows]
            ),
            "max_paired_ci95_half_width": max(
                float(row["utility_difference_ci95_half_width"]) for row in rows
            ),
            "mean_sample_count_difference": mean(
                [float(row["mean_sample_count_difference"]) for row in rows]
            ),
            "sample_count_ci_excludes_zero": len(sample_significant),
        }

    overview = {
        "environments": len(policies),
        "method_summary_rows": len(summaries),
        "episode_rows": total_episode_rows,
        "episodes_per_environment_method": 1200,
        "canonical_method_rows": len(canonical_summaries),
        "common_randomness_mismatches": common_randomness_mismatches,
        "max_marginal_mean_utility_ci95_half_width": max(
            f(row, "mean_utility_ci95") for row in canonical_summaries
        ),
        "max_paired_utility_ci95_half_width": max(
            float(row["utility_difference_ci95_half_width"]) for row in paired_rows
        ),
        "canonical_dp_point_estimate_below_myopic_environments": sum(
            float(row["canonical_dp_minus_myopic"]) < 0 for row in sensitivity_rows
        ),
        "best_observed_dp_above_myopic_environments": sum(
            float(row["best_observed_dp_minus_myopic"]) > 0 for row in sensitivity_rows
        ),
        "comparisons": comparison_overview,
    }
    return overview


def analyze_r4(r4_dir: Path, output_dir: Path) -> dict[str, object]:
    _, rows = read_rows(r4_dir / R4_SUMMARY)
    _, profiles = read_rows(r4_dir / R4_PROFILES)
    _, candidates = read_rows(r4_dir / R4_CANDIDATES)
    if len(rows) != 972 or len(profiles) != 2916:
        raise RuntimeError("R4 result does not contain 972 environments and three profiles each")

    strongest_rr = max(rows, key=lambda row: f(row, "rr_true_equal_outcome_rate"))
    largest_manual_advantage = max(
        rows, key=lambda row: f(row, "manual_active_minus_equal_split_utility")
    )
    manual_losses = [row for row in rows if f(row, "manual_active_minus_equal_split_utility") <= 0]
    profile_by_key = {(row["environment"], row["policy"]): row for row in profiles}
    manual_profiles = [
        row for row in profiles if row["policy"] == "manual_active_search_equal_outcome"
    ]
    rr_profiles = [row for row in profiles if row["policy"] == "myopic_voi"]

    selected_rows = []
    for label, row in (
        ("strongest_rr_true_equal_outcome", strongest_rr),
        ("largest_manual_utility_advantage", largest_manual_advantage),
    ):
        rr = profile_by_key[(row["environment"], "myopic_voi")]
        manual = profile_by_key[(row["environment"], "manual_active_search_equal_outcome")]
        equal_split = profile_by_key[(row["environment"], "manual_equal_split")]
        selected_rows.append(
            {
                "selection": label,
                **row,
                "mu_need": rr["mu_need"],
                "total_time": rr["total_time"],
                "learning_per_unit_of_tutoring": rr["learning_per_unit_of_tutoring"],
                "sigma_need": rr["sigma_need"],
                "sigma_sample": rr["sigma_sample"],
                "sample_time_cost": rr["sample_time_cost"],
                "utility_exponent": rr["utility_exponent"],
                "manual_mean_utility": manual["mean_utility"],
                "equal_split_mean_utility": equal_split["mean_utility"],
                "rr_mean_utility": rr["mean_utility"],
            }
        )
    write_rows(
        output_dir / "r4_representative_environments.csv",
        list(selected_rows[0]),
        selected_rows,
    )

    overview = {
        "environments": len(rows),
        "profiles": len(profiles),
        "episodes_per_environment_policy": int(float(profiles[0]["n_episodes"])),
        "manual_candidate_rule_count": len(candidates),
        "manual_utility_above_equal_split_count": sum(
            f(row, "manual_active_minus_equal_split_utility") > 0 for row in rows
        ),
        "manual_positive_true_gap_reduction_count": sum(
            f(row, "manual_active_true_gap_reduction_vs_equal_split") > 0 for row in rows
        ),
        "rr_utility_above_manual_count": sum(
            f(row, "rr_minus_manual_active_utility") > 0 for row in rows
        ),
        "rr_samples_above_one_count": sum(f(row, "rr_mean_sample_count") > 1 for row in rows),
        "rr_true_equal_outcome_rate_ge_0_9_count": sum(
            f(row, "rr_true_equal_outcome_rate") >= 0.9 for row in rows
        ),
        "rr_closer_to_true_equal_rate_gt_0_5_count": sum(
            f(row, "rr_closer_to_true_equal_rate") > 0.5 for row in rows
        ),
        "mean_manual_minus_equal_split_utility": mean(
            [f(row, "manual_active_minus_equal_split_utility") for row in rows]
        ),
        "mean_rr_minus_manual_utility": mean(
            [f(row, "rr_minus_manual_active_utility") for row in rows]
        ),
        "mean_rr_minus_equal_split_utility": mean(
            [
                f(row, "rr_minus_manual_active_utility")
                + f(row, "manual_active_minus_equal_split_utility")
                for row in rows
            ]
        ),
        "mean_manual_sample_count": mean(
            [f(row, "manual_active_mean_sample_count") for row in rows]
        ),
        "mean_rr_sample_count": mean([f(row, "rr_mean_sample_count") for row in rows]),
        "mean_manual_true_equal_outcome_rate": mean(
            [f(row, "manual_active_true_equal_outcome_rate") for row in rows]
        ),
        "mean_rr_true_equal_outcome_rate": mean(
            [f(row, "rr_true_equal_outcome_rate") for row in rows]
        ),
        "mean_manual_outcome_distance_to_true_equal": mean(
            [f(row, "mean_outcome_distance_to_true_equal") for row in manual_profiles]
        ),
        "mean_rr_outcome_distance_to_true_equal": mean(
            [f(row, "mean_outcome_distance_to_true_equal") for row in rr_profiles]
        ),
        "maximum_rr_true_equal_outcome_rate": f(strongest_rr, "rr_true_equal_outcome_rate"),
        "mean_manual_closer_to_true_equal_rate": mean(
            [f(row, "manual_active_closer_to_true_equal_rate") for row in rows]
        ),
        "mean_rr_closer_to_true_equal_rate": mean(
            [f(row, "rr_closer_to_true_equal_rate") for row in rows]
        ),
        "manual_nonpositive_utility_advantage_count": len(manual_losses),
        "manual_loss_mu_need_counts": Counter(
            profile_by_key[(row["environment"], "myopic_voi")]["mu_need"]
            for row in manual_losses
        ),
        "manual_loss_total_time_counts": Counter(
            profile_by_key[(row["environment"], "myopic_voi")]["total_time"]
            for row in manual_losses
        ),
        "strongest_rr_environment": selected_rows[0],
        "largest_manual_advantage_environment": selected_rows[1],
    }
    return overview


def pct(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_report(r3: Mapping[str, object], r4: Mapping[str, object]) -> str:
    comparisons = r3["comparisons"]
    strongest = r4["strongest_rr_environment"]
    r4_n = int(r4["environments"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resource-Rational Allocation: Computational Results</title>
  <style>
    :root {{ --ink:#17211f; --muted:#5b6864; --paper:#fbf7ef; --card:#fffdf8; --line:#d9cdbd; --green:#1f6b4b; --amber:#ad6b16; --blue:#245f8f; --red:#8d3f2d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:radial-gradient(circle at top left,rgba(31,107,75,.13),transparent 30rem),radial-gradient(circle at top right,rgba(36,95,143,.1),transparent 28rem),var(--paper); font-family:Georgia,"Times New Roman",serif; line-height:1.55; }}
    main {{ max-width:1060px; margin:auto; padding:42px 22px 64px; }}
    header,.card,.note,table {{ border:1px solid var(--line); background:rgba(255,253,248,.94); }}
    header {{ padding:32px; border-radius:22px; box-shadow:0 18px 50px rgba(23,33,31,.07); }}
    h1 {{ margin:0 0 10px; font-size:clamp(2rem,5vw,3.5rem); line-height:1; letter-spacing:-.04em; }}
    h2 {{ margin:38px 0 14px; font-size:1.65rem; }}
    h3 {{ margin:0 0 8px; font-size:1.12rem; }}
    p {{ margin:0 0 12px; }} .muted {{ color:var(--muted); }}
    .pills {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }}
    .pill {{ padding:6px 11px; border:1px solid var(--line); border-radius:999px; background:#fffaf1; color:var(--muted); font-size:.9rem; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .card {{ padding:18px; border-radius:16px; }}
    .number {{ display:block; color:var(--blue); font-size:1.9rem; font-weight:bold; line-height:1.1; }}
    .tag {{ display:inline-block; margin-bottom:9px; padding:3px 9px; border-radius:999px; color:white; background:var(--green); font:12px ui-monospace,monospace; }}
    .tag.amber {{ background:var(--amber); }} .tag.red {{ background:var(--red); }}
    table {{ width:100%; border-collapse:collapse; border-radius:14px; display:block; overflow-x:auto; }}
    th,td {{ padding:11px 13px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#efe4d2; font-size:.88rem; }} tr:last-child td {{ border-bottom:0; }}
    .note {{ padding:17px 19px; border-left:5px solid var(--amber); border-radius:12px; }}
    a {{ color:var(--blue); text-decoration:none; border-bottom:1px solid #9bb6cb; }}
    code {{ font: .9em ui-monospace,SFMono-Regular,Menlo,monospace; background:#f5edde; padding:2px 5px; border-radius:5px; }}
    footer {{ margin-top:38px; color:var(--muted); font-size:.9rem; }}
    @media(max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} header {{ padding:23px; }} }}
  </style>
</head>
<body><main>
  <header>
    <h1>Resource-Rational Allocation</h1>
    <p class="muted">Round 3 approximation precision and Round 4 diagnostic active-search results.</p>
    <div class="pills"><span class="pill">Hoffman2</span><span class="pill">1,200 episodes per condition</span><span class="pill">Common true states</span><span class="pill">Common observation streams</span><span class="pill">Validated complete runs</span></div>
  </header>

  <h2>Key Results</h2>
  <div class="grid">
    <article class="card"><span class="tag">R3 precision</span><span class="number">{fmt(float(r3['max_paired_utility_ci95_half_width']))}</span><p>Maximum paired 95% CI half-width across canonical method comparisons, below one utility unit.</p></article>
    <article class="card"><span class="tag">R4 environment test</span><span class="number">{r4['manual_utility_above_equal_split_count']}/{r4_n}</span><p>Environments where manual active search produced higher utility than equal split ({pct(int(r4['manual_utility_above_equal_split_count']), r4_n)}).</p></article>
    <article class="card"><span class="tag red">R4 discovery test</span><span class="number">0/{r4_n}</span><p>Environments where RR reached the strict true equal-outcome threshold of 0.90.</p></article>
  </div>

  <h2>Round 3: Approximation Methods</h2>
  <p>The completed comparison contains {r3['episode_rows']:,} episode rows, {r3['method_summary_rows']} method summaries, and {r3['environments']} environments. All canonical comparisons use the same 1,200 true states and observation streams; fingerprint mismatches: {r3['common_randomness_mismatches']}. The largest marginal mean-utility CI half-width is {fmt(float(r3['max_marginal_mean_utility_ci95_half_width']))}; pairing reduces the maximum difference-CI half-width to {fmt(float(r3['max_paired_utility_ci95_half_width']))}. Sampling consumes time rather than a fixed utility penalty, so one sample does not have one constant utility-unit cost.</p>
  <table><thead><tr><th>Paired comparison</th><th>Mean utility difference</th><th>Paired 95% CI excludes zero</th><th>Mean sample-count difference</th><th>Maximum utility CI half-width</th></tr></thead><tbody>
    <tr><td>Myopic VOI minus blinkered</td><td>{fmt(float(comparisons['myopic_minus_blinkered']['mean_of_environment_differences']))}</td><td>{comparisons['myopic_minus_blinkered']['left_ci_above_zero']} above / {comparisons['myopic_minus_blinkered']['left_ci_below_zero']} below</td><td>{fmt(float(comparisons['myopic_minus_blinkered']['mean_sample_count_difference']))} ({comparisons['myopic_minus_blinkered']['sample_count_ci_excludes_zero']}/14 CIs exclude zero)</td><td>{fmt(float(comparisons['myopic_minus_blinkered']['max_paired_ci95_half_width']))}</td></tr>
    <tr><td>DP minus myopic VOI</td><td>{fmt(float(comparisons['dp_minus_myopic']['mean_of_environment_differences']))}</td><td>{comparisons['dp_minus_myopic']['left_ci_above_zero']} above / {comparisons['dp_minus_myopic']['left_ci_below_zero']} below</td><td>{fmt(float(comparisons['dp_minus_myopic']['mean_sample_count_difference']))} ({comparisons['dp_minus_myopic']['sample_count_ci_excludes_zero']}/14 CIs exclude zero)</td><td>{fmt(float(comparisons['dp_minus_myopic']['max_paired_ci95_half_width']))}</td></tr>
    <tr><td>DP minus blinkered</td><td>{fmt(float(comparisons['dp_minus_blinkered']['mean_of_environment_differences']))}</td><td>{comparisons['dp_minus_blinkered']['left_ci_above_zero']} above / {comparisons['dp_minus_blinkered']['left_ci_below_zero']} below</td><td>{fmt(float(comparisons['dp_minus_blinkered']['mean_sample_count_difference']))} ({comparisons['dp_minus_blinkered']['sample_count_ci_excludes_zero']}/14 CIs exclude zero)</td><td>{fmt(float(comparisons['dp_minus_blinkered']['max_paired_ci95_half_width']))}</td></tr>
  </tbody></table>
  <p class="note"><strong>Interpretation.</strong> The 10x run makes method differences much more precise, but the canonical DP point estimate remains below myopic VOI in {r3['canonical_dp_point_estimate_below_myopic_environments']} of 14 environments. At least one tested DP configuration exceeds myopic VOI in all 14 environments, but selecting the best of 48 DP settings on the same episodes is exploratory and upward-biased. This supports approximation sensitivity, not a claim that DP is exactly optimal.</p>

  <h2>Round 4: Diagnostic Active Search</h2>
  <p>The diagnostic grid contains {r4_n} environments and three policies per environment: RR/myopic VOI, manual active-search equal outcome, and equal split. The manual policy used six observations in every environment. It had higher utility than equal split in {r4['manual_utility_above_equal_split_count']}/{r4_n} environments, and {r4['manual_candidate_rule_count']}/{r4_n} passed the predefined strict candidate rule. RR had higher utility than the manual policy in {r4['rr_utility_above_manual_count']}/{r4_n} environments.</p>
  <div class="grid">
    <article class="card"><span class="number">{fmt(float(r4['mean_manual_minus_equal_split_utility']))}</span><p>Mean manual active-search utility advantage over equal split.</p></article>
    <article class="card"><span class="number">{fmt(float(r4['mean_rr_minus_manual_utility']))}</span><p>Mean RR utility advantage over manual active search.</p></article>
    <article class="card"><span class="number">{fmt(float(r4['mean_rr_sample_count']))}</span><p>Mean RR sample count, compared with {fmt(float(r4['mean_manual_sample_count']))} for the manual policy.</p></article>
  </div>
  <table><thead><tr><th>Behavior</th><th>Manual active search</th><th>RR approximation</th></tr></thead><tbody>
    <tr><td>Mean true equal-outcome rate</td><td>{fmt(float(r4['mean_manual_true_equal_outcome_rate']))}</td><td>{fmt(float(r4['mean_rr_true_equal_outcome_rate']))}</td></tr>
    <tr><td>Mean closer-to-true-equal rate</td><td>{fmt(float(r4['mean_manual_closer_to_true_equal_rate']))}</td><td>{fmt(float(r4['mean_rr_closer_to_true_equal_rate']))}</td></tr>
    <tr><td>Mean outcome distance to true equal outcome</td><td>{fmt(float(r4['mean_manual_outcome_distance_to_true_equal']))}</td><td>{fmt(float(r4['mean_rr_outcome_distance_to_true_equal']))}</td></tr>
    <tr><td>Environments above 0.90 true equal outcome</td><td>Diagnostic benchmark</td><td>{r4['rr_true_equal_outcome_rate_ge_0_9_count']} / {r4_n}</td></tr>
    <tr><td>Environments closer to true equal outcome than equal split in most episodes</td><td>Not the primary selection test</td><td>{r4['rr_closer_to_true_equal_rate_gt_0_5_count']} / {r4_n}</td></tr>
  </tbody></table>
  <p class="note"><strong>Where the manual policy did not help.</strong> Manual active search failed to outperform equal split in {r4['manual_nonpositive_utility_advantage_count']}/{r4_n} environments. These failures were concentrated in high-need and scarce-time cases: {r4['manual_loss_mu_need_counts']['55.0']}/87 had <code>mu_need = 55</code>, and {r4['manual_loss_total_time_counts']['80.0']}/87 had <code>total_time = 80</code>.</p>

  <h2>Strongest RR Equal-Outcome Case</h2>
  <p>The highest RR true equal-outcome rate was <strong>{fmt(float(strongest['rr_true_equal_outcome_rate']))}</strong> at grid index {int(float(strongest['grid_index']))}. RR sampled {fmt(float(strongest['rr_mean_sample_count']))} times on average and was closer to true equal outcome than equal split in {fmt(float(strongest['rr_closer_to_true_equal_rate']))} of episodes. In this environment, manual active search exceeded equal split by {fmt(float(strongest['manual_active_minus_equal_split_utility']))} utility units, while RR exceeded the manual policy by {fmt(float(strongest['rr_minus_manual_active_utility']))}.</p>

  <h2>Conclusion</h2>
  <p><strong>The diagnostic environment construction succeeded:</strong> manual active search improved utility over equal split in {r4['manual_utility_above_equal_split_count']} environments and reduced the true outcome gap in all {r4_n}.</p>
  <p><strong>The current RR approximation did not reproduce the manual policy's strong equal-outcome behavior:</strong> it sampled actively and achieved higher utility, but its maximum true equal-outcome rate was {fmt(float(r4['maximum_rr_true_equal_outcome_rate']))}. The present evidence therefore does not yet establish a discovered multi-step resource-rational equal-outcome heuristic.</p>
  <p class="note"><strong>Next decision.</strong> Determine whether this behavioral difference reflects the myopic approximation missing a multi-step strategy or the utilitarian objective favoring a higher-utility allocation that is not equal outcome. A targeted comparison with stronger non-myopic solvers in the diagnostic environments is the most direct next test.</p>

  <h2>Supporting Data</h2>
  <p><a href="supporting_data/r3_paired_method_comparisons.csv">R3 paired comparisons</a> · <a href="supporting_data/r3_dp_sensitivity.csv">R3 DP sensitivity</a> · <a href="supporting_data/r4_representative_environments.csv">R4 representative environments</a> · <a href="supporting_data/analysis_summary.json">Analysis summary</a></p>
  <p class="muted">The supporting folder also includes the complete 700-row R3 method summary, the complete R4 aggregate tables, and run-validation evidence. The 840,000-row R3 episode file is retained in the project result archive rather than duplicated in this professor-facing folder.</p>
  <footer>Expected average utility is the performance criterion. Information-acquisition and outcome-equality measures are behavioral diagnostics. RR refers to the current approximation, not an exact continuous-state optimal policy.</footer>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-dir", type=Path, required=True)
    parser.add_argument("--r4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supporting_dir = args.output_dir / "supporting_data"
    supporting_dir.mkdir(parents=True, exist_ok=True)
    r3 = analyze_r3(args.r3_dir, supporting_dir)
    r4 = analyze_r4(args.r4_dir, supporting_dir)
    for source, destination in (
        (args.r3_dir / R3_SUMMARY, supporting_dir / R3_SUMMARY),
        (
            args.r3_dir / "method_comparison_combine_status.csv",
            supporting_dir / "method_comparison_combine_status.csv",
        ),
        (
            args.r3_dir / "r3_episode_array_progress.json",
            supporting_dir / "r3_episode_array_progress.json",
        ),
        (
            args.r3_dir / "recovery" / "stage_evidence_recovery.json",
            supporting_dir / "r3_stage_evidence_recovery.json",
        ),
        (args.r4_dir / R4_SUMMARY, supporting_dir / R4_SUMMARY),
        (args.r4_dir / R4_PROFILES, supporting_dir / R4_PROFILES),
        (args.r4_dir / R4_CANDIDATES, supporting_dir / R4_CANDIDATES),
        (args.r4_dir / "r4_array_progress.json", supporting_dir / "r4_array_progress.json"),
        (args.r4_dir / "r4_array_manifest.json", supporting_dir / "r4_array_manifest.json"),
    ):
        if source.is_file():
            shutil.copy2(source, destination)
    summary = {"round3": r3, "round4": r4}
    (supporting_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(render_report(r3, r4), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
