#!/usr/bin/env python3
"""Build a validated active-search evaluation report from collected results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Mapping, Sequence


TRUE_THRESHOLD = 0.80
CLOSER_THRESHOLD = 0.80
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRODUCIBILITY_NOTEBOOK = (
    PROJECT_ROOT / "notebooks" / "round_05" / "reproduce_round_05.ipynb"
)
REPOSITORY_URL = (
    "https://github.com/Zhenlong-Zhang/"
    "Resource-rational-heuristics-for-continuous-allocation-decisions"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(row: Mapping[str, object], key: str) -> float:
    return float(row[key])


def validate_completed(
    result_dir: Path,
    *,
    family: str,
    tasks: int,
    episodes: int,
    summaries: int,
) -> dict[str, object]:
    completed = load_json(result_dir / "COMPLETED.json")
    expected = {
        "family": family,
        "task_count": tasks,
        "episode_row_count": episodes,
        "summary_row_count": summaries,
    }
    mismatches = {
        key: (completed.get(key), value)
        for key, value in expected.items()
        if completed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Incomplete or unexpected result in {result_dir}: {mismatches}")
    return completed


def oracle_strata(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    definitions = {
        "both_positive": (
            "oracle_both_positive_rate",
            "oracle_true_equal_outcome_rate_given_both_positive",
            "oracle_closer_rate_given_both_positive",
        ),
        "mixed_sign": (
            "oracle_mixed_sign_rate",
            "oracle_true_equal_outcome_rate_given_mixed_sign",
            "oracle_closer_rate_given_mixed_sign",
        ),
        "both_negative": (
            "oracle_both_negative_rate",
            "oracle_true_equal_outcome_rate_given_both_negative",
            "oracle_closer_rate_given_both_negative",
        ),
    }
    result: dict[str, dict[str, float | int]] = {}
    for name, (share_field, true_field, closer_field) in definitions.items():
        count = 0.0
        true_count = 0.0
        closer_count = 0.0
        for row in rows:
            stratum_count = f(row, "n_episodes") * f(row, share_field)
            count += stratum_count
            true_rate = f(row, true_field)
            closer_rate = f(row, closer_field)
            if not math.isnan(true_rate):
                true_count += stratum_count * true_rate
            if not math.isnan(closer_rate):
                closer_count += stratum_count * closer_rate
        result[name] = {
            "episodes": round(count),
            "true_equal_outcome_rate": true_count / count if count else math.nan,
            "closer_rate": closer_count / count if count else math.nan,
        }
    return result


def active_row(row: Mapping[str, object]) -> bool:
    return (
        f(row, "mean_sample_count") > 1.0
        and f(row, "mean_abs_allocation_from_equal") >= 0.05
    )


def validate_confirmation_manifests(
    discovery_manifest: Mapping[str, object],
    confirmation_manifest: Mapping[str, object],
) -> None:
    if discovery_manifest.get("seed_namespace_offset") == confirmation_manifest.get(
        "seed_namespace_offset"
    ):
        raise RuntimeError("Discovery and confirmation seed namespaces must differ")
    if confirmation_manifest.get("episodes_per_environment") != 1200:
        raise RuntimeError("Confirmation must use 1,200 episodes per environment")
    if confirmation_manifest.get("observation_draws") != 500:
        raise RuntimeError("Confirmation must use 500 VOI observation draws")
    environments = list(confirmation_manifest.get("environments", []))
    if len(environments) != 12:
        raise RuntimeError("Confirmation manifest must contain 12 environments")
    for environment in environments:
        config = dict(environment["config"])
        if config.get("prior_sample_count_1") != 0 or config.get(
            "prior_sample_count_2"
        ) != 0:
            raise RuntimeError("Active-search confirmation requires zero prior samples")


def confirmation_overview(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 confirmation environments, found {len(rows)}")
    if any(int(f(row, "n_episodes")) != 1200 for row in rows):
        raise RuntimeError("Every confirmation environment must contain 1,200 episodes")
    point_joint = [
        row
        for row in rows
        if f(row, "true_equal_outcome_rate") >= TRUE_THRESHOLD
        and f(row, "closer_to_true_equal_outcome_than_equal_split_rate")
        >= CLOSER_THRESHOLD
        and active_row(row)
    ]
    confirmed = [
        row
        for row in rows
        if f(row, "true_equal_outcome_one_sided_95_low") >= TRUE_THRESHOLD
        and f(row, "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low")
        >= CLOSER_THRESHOLD
        and active_row(row)
    ]
    best = max(
        rows,
        key=lambda row: min(
            f(row, "true_equal_outcome_one_sided_95_low"),
            f(row, "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low"),
        ),
    )
    return {
        "environment_count": len(rows),
        "point_joint_count": len(point_joint),
        "confirmed_count": len(confirmed),
        "best_environment": str(best["environment"]),
        "best_true_rate": f(best, "true_equal_outcome_rate"),
        "best_true_low": f(best, "true_equal_outcome_one_sided_95_low"),
        "best_closer_rate": f(
            best, "closer_to_true_equal_outcome_than_equal_split_rate"
        ),
        "best_closer_low": f(
            best,
            "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low",
        ),
        "best_mean_samples": f(best, "mean_sample_count"),
        "best_allocation_distance": f(best, "mean_abs_allocation_from_equal"),
    }


def short_environment(value: str) -> str:
    if value.startswith("confirmation_"):
        return value.split("_active_search_six", 1)[0].replace("confirmation_", "Candidate ")
    return value


def confirmation_table(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    table = []
    for row in rows:
        table.append(
            {
                "environment": row["environment"],
                "n_episodes": int(f(row, "n_episodes")),
                "true_equal_outcome_rate": f(row, "true_equal_outcome_rate"),
                "true_equal_outcome_one_sided_95_low": f(
                    row, "true_equal_outcome_one_sided_95_low"
                ),
                "closer_rate": f(
                    row, "closer_to_true_equal_outcome_than_equal_split_rate"
                ),
                "closer_one_sided_95_low": f(
                    row,
                    "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low",
                ),
                "mean_sample_count": f(row, "mean_sample_count"),
                "mean_abs_allocation_from_equal": f(
                    row, "mean_abs_allocation_from_equal"
                ),
                "point_joint_active_search": int(
                    f(row, "true_equal_outcome_rate") >= TRUE_THRESHOLD
                    and f(row, "closer_to_true_equal_outcome_than_equal_split_rate")
                    >= CLOSER_THRESHOLD
                    and active_row(row)
                ),
                "confirmed_joint_active_search": int(
                    f(row, "true_equal_outcome_one_sided_95_low") >= TRUE_THRESHOLD
                    and f(
                        row,
                        "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low",
                    )
                    >= CLOSER_THRESHOLD
                    and active_row(row)
                ),
            }
        )
    return table


def write_confirmation_svg(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    width = 980
    left, right, top, bottom = 170, 45, 72, 78
    row_height = 42
    height = top + bottom + row_height * len(rows)
    plot_width = width - left - right

    def xp(value: float) -> float:
        bounded = min(max(value, 0.70), 0.90)
        return left + (bounded - 0.70) / 0.20 * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="27" font-family="Georgia, serif" font-size="22" fill="#19352f">Independent active-search confirmation</text>',
        f'<text x="{left}" y="49" font-family="Arial, sans-serif" font-size="12" fill="#63716b">Points are rates; horizontal segments extend to one-sided 95% Wilson lower bounds.</text>',
        f'<line x1="{xp(TRUE_THRESHOLD):.2f}" y1="{top-16}" x2="{xp(TRUE_THRESHOLD):.2f}" y2="{height-bottom+10}" stroke="#9d6b32" stroke-dasharray="5 5"/>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height + row_height / 2
        true_rate = f(row, "true_equal_outcome_rate")
        true_low = f(row, "true_equal_outcome_one_sided_95_low")
        closer_rate = f(row, "closer_to_true_equal_outcome_than_equal_split_rate")
        closer_low = f(
            row, "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low"
        )
        parts.extend(
            [
                f'<text x="{left-14}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#33443e">{html.escape(short_environment(str(row["environment"])))}</text>',
                f'<line x1="{xp(true_low):.2f}" y1="{y-7:.2f}" x2="{xp(true_rate):.2f}" y2="{y-7:.2f}" stroke="#1f6b4b" stroke-width="2"/>',
                f'<circle cx="{xp(true_rate):.2f}" cy="{y-7:.2f}" r="4" fill="#1f6b4b"/>',
                f'<line x1="{xp(closer_low):.2f}" y1="{y+7:.2f}" x2="{xp(closer_rate):.2f}" y2="{y+7:.2f}" stroke="#245f8f" stroke-width="2"/>',
                f'<circle cx="{xp(closer_rate):.2f}" cy="{y+7:.2f}" r="4" fill="#245f8f"/>',
            ]
        )
    for tick in (0.70, 0.75, 0.80, 0.85, 0.90):
        parts.append(
            f'<text x="{xp(tick):.2f}" y="{height-38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#56635e">{tick:.2f}</text>'
        )
    parts.extend(
        [
            f'<circle cx="{left}" cy="{height-17}" r="4" fill="#1f6b4b"/><text x="{left+10}" y="{height-13}" font-family="Arial, sans-serif" font-size="11" fill="#33443e">True equal-outcome rate</text>',
            f'<circle cx="{left+190}" cy="{height-17}" r="4" fill="#245f8f"/><text x="{left+200}" y="{height-13}" font-family="Arial, sans-serif" font-size="11" fill="#33443e">Closer than equal split</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def report_html(summary: Mapping[str, object]) -> str:
    oracle = summary["oracle"]
    fixed = summary["fixed_budget"]
    discovery = summary["discovery"]
    confirmation = summary["confirmation"]
    solver = summary["solver"]
    if confirmation["confirmed_count"]:
        next_decision = (
            "The prespecified active-search regime is confirmed. The next decision is "
            "whether this result is stable enough across nearby environments to support "
            "experiment design."
        )
    else:
        next_decision = (
            "The active-search pattern replicated at the point-estimate level but remains "
            "too close to the 0.80 boundary for the prespecified confidence rule. A larger "
            "independent precision run on frozen candidates is the direct next test. A "
            "stronger solver is warranted only if a material utility or behavioral gap to "
            "the manual active-search benchmark remains."
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resource-Rational Allocation: Active-Search Evaluation Results</title>
<style>
:root{{--ink:#17211f;--muted:#5b6864;--paper:#fbf7ef;--card:#fffdf8;--line:#d9cdbd;--green:#1f6b4b;--amber:#a96515;--blue:#245f8f;--red:#8d3f2d}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:radial-gradient(circle at top left,rgba(31,107,75,.12),transparent 30rem),var(--paper);font-family:Georgia,"Times New Roman",serif;line-height:1.55}}
main{{max-width:1040px;margin:auto;padding:42px 22px 64px}} header,.card,.note,table{{border:1px solid var(--line);background:rgba(255,253,248,.95)}} header{{padding:30px;border-radius:22px}} h1{{margin:0 0 8px;font-size:clamp(2rem,5vw,3.4rem);line-height:1;letter-spacing:-.04em}} h2{{margin:36px 0 13px;font-size:1.55rem}} p{{margin:0 0 12px}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}} .card{{padding:18px;border-radius:16px}} .number{{display:block;color:var(--blue);font-size:1.9rem;font-weight:bold;line-height:1.1}} .tag{{display:inline-block;margin-bottom:8px;padding:3px 9px;border-radius:999px;color:white;background:var(--green);font:12px ui-monospace,monospace}} .tag.amber{{background:var(--amber)}} .tag.red{{background:var(--red)}}
table{{width:100%;border-collapse:collapse;border-radius:14px;display:block;overflow-x:auto}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}} th{{background:#efe4d2;font-size:.88rem}} .note{{padding:16px 18px;border-left:5px solid var(--amber);border-radius:12px}} img{{display:block;width:100%;margin:14px 0;border:1px solid var(--line);border-radius:14px;background:#fbfaf7}} a{{color:var(--blue);text-decoration:none;border-bottom:1px solid #9bb6cb}} footer{{margin-top:36px;color:var(--muted);font-size:.9rem}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><h1>Resource-Rational Allocation</h1><p class="muted">Active-search evaluation: objective diagnosis, information value, independent active-search confirmation, and a frozen non-myopic solver check.</p></header>
<h2>Main Results</h2><div class="grid">
<article class="card"><span class="tag">Objective</span><span class="number">{oracle['joint_candidates']}/108</span><p>Full-information environments satisfying the joint 0.8/0.8 equal-outcome criterion.</p></article>
<article class="card"><span class="tag">Information value</span><span class="number">{fixed['useful_anchor_count']}/4</span><p>Oracle anchors where six observations significantly outperformed four.</p></article>
<article class="card"><span class="tag {'red' if confirmation['confirmed_count'] == 0 else ''}">Confirmation</span><span class="number">{confirmation['confirmed_count']}/12</span><p>Independent candidates whose two one-sided 95% lower bounds both exceed 0.80.</p></article>
</div>
<h2>1. Does The Utility Objective Favor Equal Outcome?</h2>
<p>Yes, but only in a specific outcome-sign regime. The kink-aware full-information oracle evaluated 108 environments and 129,600 episodes. Numerical optimality violations were at floating-point scale, and {oracle['joint_candidates']} environments met the joint behavioral rule.</p>
<table><thead><tr><th>Oracle outcome stratum</th><th>Episodes</th><th>True equal-outcome rate</th><th>Closer than equal split</th></tr></thead><tbody>
<tr><td>Both recipients positive</td><td>{oracle['strata']['both_positive']['episodes']:,}</td><td>{fmt(oracle['strata']['both_positive']['true_equal_outcome_rate'])}</td><td>{fmt(oracle['strata']['both_positive']['closer_rate'])}</td></tr>
<tr><td>Mixed sign</td><td>{oracle['strata']['mixed_sign']['episodes']:,}</td><td>{fmt(oracle['strata']['mixed_sign']['true_equal_outcome_rate'])}</td><td>{fmt(oracle['strata']['mixed_sign']['closer_rate'])}</td></tr>
<tr><td>Both negative</td><td>{oracle['strata']['both_negative']['episodes']:,}</td><td>{fmt(oracle['strata']['both_negative']['true_equal_outcome_rate'])}</td><td>{fmt(oracle['strata']['both_negative']['closer_rate'])}</td></tr>
</tbody></table>
<p class="note"><strong>Interpretation.</strong> The unchanged utilitarian objective can favor true equal outcome when both recipients can be brought above need. The earlier failure is therefore not solely an objective mismatch.</p>
<h2>2. Is Repeated Information Valuable?</h2>
<p>Yes in two high-variability oracle-compatible anchors. Moving from four to six balanced observations increased utility by {fmt(fixed['anchor_1_gain'])} (95% CI [{fmt(fixed['anchor_1_low'])}, {fmt(fixed['anchor_1_high'])}]) and {fmt(fixed['anchor_3_gain'])} (95% CI [{fmt(fixed['anchor_3_low'])}, {fmt(fixed['anchor_3_high'])}]). Lower sampling time increased adaptive sampling, but sampling cost alone did not close the behavioral gap.</p>
<img src="figures/active_search_fixed_budget_utility.svg" alt="Utility by fixed observation budget">
<h2>3. Did Active-Search Equal Outcome Replicate?</h2>
<p>The 324-environment discovery grid produced {discovery['candidate_count']} candidates under the frozen point-estimate rule. They used {fmt(discovery['sample_min'],2)}-{fmt(discovery['sample_max'],2)} online samples on average and chose allocations about {fmt(discovery['allocation_distance'],3)} away from 50/50.</p>
<p>The independent confirmation used 1,200 new episodes per candidate. {confirmation['point_joint_count']}/12 still met the point-estimate rule, while {confirmation['confirmed_count']}/12 met the stricter rule requiring both one-sided 95% Wilson lower bounds to exceed 0.80. The strongest minimum lower bound combined true-equal-outcome {fmt(confirmation['best_true_rate'])} (lower bound {fmt(confirmation['best_true_low'])}) with closer-than-equal-split {fmt(confirmation['best_closer_rate'])} (lower bound {fmt(confirmation['best_closer_low'])}).</p>
<img src="figures/active_search_confirmation_rates.svg" alt="Independent confirmation rates and one-sided lower bounds">
<h2>4. Did The Frozen DP Recover The Target Strategy?</h2>
<p>No. The paired held-out comparison used identical true states and observation streams for myopic VOI and the prespecified small-horizon DP; common-randomness mismatches: {solver['mismatches']}. DP improved mean utility significantly in {solver['positive_ci_count']}/3 environments, but satisfied the active-search joint confirmation rule in {solver['confirmed_count']}/3.</p>
<img src="figures/active_search_solver_paired_utility.svg" alt="Paired DP minus myopic utility differences">
<h2>Conclusion</h2>
<p>The full-information objective supports equal outcome in both-positive regimes, and at least six observations can be useful. The broad search found adaptive, unequal-allocation policies, and {confirmation['point_joint_count']}/12 independently retained both point estimates at or above 0.80. However, {confirmation['confirmed_count']}/12 passed the stricter two-lower-bound rule, so the point-level replication and the confirmatory claim must remain distinct.</p>
<p class="note"><strong>Next decision.</strong> {next_decision}</p>
<h2>Reproducibility</h2>
<p>The complete model, simulation pipeline, reproducible workflow, and tests are available in the <a href="{REPOSITORY_URL}">GitHub repository</a>. The included notebook is a lightweight interface to the same source code; it does not duplicate the model implementation.</p>
<h2>Supporting Data</h2><p><a href="supporting_data/active_search_confirmation_comparison.csv">Confirmation table</a> · <a href="supporting_data/active_search_joint_candidates_discovery.csv">Discovery candidates</a> · <a href="supporting_data/active_search_solver_comparison.csv">Solver comparison</a> · <a href="supporting_data/active_search_report_summary.json">Report summary</a></p>
<footer>Expected average utility remains the performance criterion. Outcome-equality and information-acquisition measures are behavioral diagnostics. RR refers to the evaluated approximation, not an exact continuous-state optimal policy.</footer>
</main></body></html>"""


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    validate_completed(
        args.oracle_dir,
        family="oracle",
        tasks=1296,
        episodes=129600,
        summaries=108,
    )
    discovery_completed = validate_completed(
        args.discovery_dir,
        family="six_sample",
        tasks=7776,
        episodes=38880,
        summaries=324,
    )
    confirmation_completed = validate_completed(
        args.confirmation_dir,
        family="custom_rr",
        tasks=1440,
        episodes=14400,
        summaries=12,
    )
    validate_confirmation_manifests(
        load_json(args.discovery_dir / "active_search_manifest.json"),
        load_json(args.confirmation_dir / "active_search_manifest.json"),
    )

    oracle_analysis = load_json(args.oracle_analysis_dir / "active_search_oracle_analysis.json")
    strata = oracle_strata(
        read_csv(args.oracle_analysis_dir / "active_search_oracle_sign_summary.csv")
    )
    fixed_rows = read_csv(args.formal_dir / "active_search_fixed_budget_evidence.csv")
    useful = [row for row in fixed_rows if f(row, "six_observations_useful") >= 0.5]
    if len(useful) != 2:
        raise RuntimeError(f"Expected two useful six-observation anchors, found {len(useful)}")
    discovery_rows = read_csv(args.discovery_dir / "active_search_joint_candidates.csv")
    if len(discovery_rows) != int(discovery_completed["candidate_count"]):
        raise RuntimeError("Discovery candidate count disagrees with completion metadata")
    confirmation_rows = read_csv(
        args.confirmation_dir / "active_search_rr_environment_summary.csv"
    )
    solver_rows = read_csv(args.solver_dir / "active_search_solver_comparison.csv")
    if len(solver_rows) != 3:
        raise RuntimeError(f"Expected three held-out solver rows, found {len(solver_rows)}")

    confirmation = confirmation_overview(confirmation_rows)
    table = confirmation_table(confirmation_rows)
    positive_solver_rows = [
        row
        for row in solver_rows
        if f(row, "paired_dp_minus_myopic_utility_ci95_low") > 0.0
    ]
    summary: dict[str, object] = {
        "oracle": {
            "joint_candidates": int(oracle_analysis["joint_oracle_candidate_count"]),
            "max_optimality_violation": float(
                oracle_analysis["max_grid_optimality_violation"]
            ),
            "strata": strata,
        },
        "fixed_budget": {
            "useful_anchor_count": len(useful),
            "anchor_1_gain": f(useful[0], "paired_mean_utility_gain"),
            "anchor_1_low": f(useful[0], "paired_utility_ci95_low"),
            "anchor_1_high": f(useful[0], "paired_utility_ci95_high"),
            "anchor_3_gain": f(useful[1], "paired_mean_utility_gain"),
            "anchor_3_low": f(useful[1], "paired_utility_ci95_low"),
            "anchor_3_high": f(useful[1], "paired_utility_ci95_high"),
        },
        "discovery": {
            "candidate_count": int(discovery_completed["candidate_count"]),
            "sample_min": min(f(row, "mean_sample_count") for row in discovery_rows),
            "sample_max": max(f(row, "mean_sample_count") for row in discovery_rows),
            "allocation_distance": sum(
                f(row, "mean_abs_allocation_from_equal") for row in discovery_rows
            )
            / len(discovery_rows),
        },
        "confirmation": confirmation,
        "solver": {
            "mismatches": sum(
                int(f(row, "common_randomness_mismatch_count")) for row in solver_rows
            ),
            "positive_ci_count": len(positive_solver_rows),
            "confirmed_count": sum(
                int(f(row, "dp_active_search_joint_confirmed")) for row in solver_rows
            ),
        },
        "provenance": {
            "discovery_commit": discovery_completed["git_commit"],
            "confirmation_commit": confirmation_completed["git_commit"],
        },
    }

    figures = args.output_dir / "figures"
    supporting = args.output_dir / "supporting_data"
    reproducibility = args.output_dir / "reproducibility"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    supporting.mkdir(parents=True, exist_ok=True)
    reproducibility.mkdir(parents=True, exist_ok=True)
    write_confirmation_svg(figures / "active_search_confirmation_rates.svg", confirmation_rows)
    write_csv(supporting / "active_search_confirmation_comparison.csv", table)
    copy_if_present(
        args.formal_dir / "figures" / "active_search_fixed_budget_utility.svg",
        figures / "active_search_fixed_budget_utility.svg",
    )
    copy_if_present(
        args.solver_dir / "figures" / "active_search_solver_paired_utility.svg",
        figures / "active_search_solver_paired_utility.svg",
    )
    for source, destination in (
        (
            args.discovery_dir / "active_search_joint_candidates.csv",
            supporting / "active_search_joint_candidates_discovery.csv",
        ),
        (
            args.confirmation_dir / "active_search_rr_environment_summary.csv",
            supporting / "active_search_confirmation_environment_summary.csv",
        ),
        (
            args.solver_dir / "active_search_solver_comparison.csv",
            supporting / "active_search_solver_comparison.csv",
        ),
        (
            args.oracle_analysis_dir / "active_search_oracle_sign_summary.csv",
            supporting / "active_search_oracle_sign_summary.csv",
        ),
        (
            args.formal_dir / "active_search_fixed_budget_evidence.csv",
            supporting / "active_search_fixed_budget_evidence.csv",
        ),
        (args.oracle_dir / "COMPLETED.json", supporting / "active_search_oracle_completed.json"),
        (
            args.discovery_dir / "COMPLETED.json",
            supporting / "active_search_discovery_completed.json",
        ),
        (
            args.confirmation_dir / "COMPLETED.json",
            supporting / "active_search_confirmation_completed.json",
        ),
    ):
        copy_if_present(source, destination)
    (supporting / "active_search_report_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(
        report_html(summary), encoding="utf-8"
    )
    copy_if_present(
        REPRODUCIBILITY_NOTEBOOK,
        reproducibility / "reproduce_round_05.ipynb",
    )
    (args.output_dir / "README.md").write_text(
        "# Active-Search Evaluation Results\n\n"
        "Open `index.html` for the concise results summary. The `supporting_data/` "
        "folder contains the corresponding aggregate evidence and completion "
        "metadata. Full episode-level archives are intentionally not duplicated "
        "in this professor-facing package.\n\n"
        "## Reproduce\n\n"
        f"- Repository: {REPOSITORY_URL}\n"
        "- Notebook: `reproducibility/reproduce_round_05.ipynb`\n"
        "- The notebook calls the shared source and workflow scripts; it does not "
        "contain a separate model implementation.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--oracle-analysis-dir", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_report(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
