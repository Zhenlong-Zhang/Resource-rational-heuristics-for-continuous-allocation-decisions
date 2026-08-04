#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.r5 import (  # noqa: E402
    summarize_r5_oracle_map,
    summarize_r5_rr_environments,
)
from src.experiments.sweeps import build_r5_oat_configs, build_r5_sampling_cost_configs  # noqa: E402
from src.mdp.meta_mdp import EnvironmentConfig  # noqa: E402


EXISTING_OAT_FEATURES = (
    "mu_need",
    "total_time",
    "learning_per_unit_of_tutoring",
    "sigma_need",
    "sigma_sample",
    "sample_time_cost",
    "utility_exponent",
)

FORMAL_OAT_FEATURES = (
    "sigma_sample",
    "total_time",
    "sigma_need",
    "utility_exponent",
    "learning_per_unit_of_tutoring",
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def environment_configs(manifest: Mapping[str, object]) -> Dict[str, EnvironmentConfig]:
    return {
        str(spec["environment"]): EnvironmentConfig(**spec["config"])
        for spec in manifest["environments"]
    }


def config_payload(
    configs: Iterable[tuple[str, EnvironmentConfig]],
    source: str,
    selection_rule: str,
) -> Dict[str, object]:
    return {
        "source": source,
        "selection_rule": selection_rule,
        "configs": [
            {"environment": environment, "config": config.__dict__}
            for environment, config in configs
        ],
    }


def repair_legacy_oracle_rows(rows: List[Dict[str, str]]) -> None:
    """Add post-grid diagnostics to rows produced before schema version 2."""

    for row in rows:
        raw_equal = float(row.get("raw_true_equal_outcome_regret", row["true_equal_outcome_regret"]))
        raw_split = float(row.get("raw_equal_split_regret", row["equal_split_regret"]))
        row["raw_true_equal_outcome_regret"] = str(raw_equal)
        row["raw_equal_split_regret"] = str(raw_split)
        row["true_equal_outcome_regret"] = str(max(0.0, raw_equal))
        row["equal_split_regret"] = str(max(0.0, raw_split))
        row["oracle_grid_optimality_violation"] = str(max(0.0, -raw_equal, -raw_split))


def _float(row: Mapping[str, object], field: str) -> float:
    return float(row[field])


def select_existing_rr_anchor(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Select one deterministic exploratory anchor from existing RR summaries."""

    if not rows:
        raise RuntimeError("No RR rows are available for existing-result analysis")
    return max(
        rows,
        key=lambda row: (
            min(
                _float(row, "true_equal_outcome_rate"),
                _float(row, "closer_to_true_equal_outcome_than_equal_split_rate"),
            ),
            _float(row, "true_equal_outcome_rate"),
            _float(row, "closer_to_true_equal_outcome_than_equal_split_rate"),
            _float(row, "mean_sample_count"),
            str(row["environment"]),
        ),
    )


def build_existing_oat_slices(
    rows: Sequence[Mapping[str, object]],
    anchor: Mapping[str, object],
    features: Sequence[str] = EXISTING_OAT_FEATURES,
) -> List[Dict[str, object]]:
    """Extract rows that differ from an anchor in one listed feature only."""

    slices: List[Dict[str, object]] = []
    for feature in features:
        other_features = [name for name in features if name != feature]
        matches = [
            row
            for row in rows
            if all(
                math.isclose(_float(row, name), _float(anchor, name), rel_tol=0.0, abs_tol=1e-12)
                for name in other_features
            )
        ]
        for row in sorted(matches, key=lambda item: _float(item, feature)):
            true_rate = _float(row, "true_equal_outcome_rate")
            closer_rate = _float(row, "closer_to_true_equal_outcome_than_equal_split_rate")
            slices.append(
                {
                    "anchor_environment": anchor["environment"],
                    "feature": feature,
                    "feature_value": _float(row, feature),
                    "environment": row["environment"],
                    "true_equal_outcome_rate": true_rate,
                    "closer_to_true_equal_outcome_than_equal_split_rate": closer_rate,
                    "joint_min_rate": min(true_rate, closer_rate),
                    "r5_joint_0_8_0_8": 1.0 if true_rate >= 0.8 and closer_rate >= 0.8 else 0.0,
                    "mean_sample_count": _float(row, "mean_sample_count"),
                    "mean_utility": _float(row, "mean_utility"),
                    "sample_time_cost": _float(row, "sample_time_cost"),
                    "sample_time_cost_percent": (
                        100.0 * _float(row, "sample_time_cost") / _float(row, "total_time")
                    ),
                }
            )
    return slices


def write_existing_oat_svg(path: Path, feature: str, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a dependency-free exploratory OAT plot for one feature."""

    width, height = 760, 420
    left, right, top, bottom = 78, 80, 55, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    ordered = sorted(rows, key=lambda row: _float(row, "feature_value"))
    xs = [_float(row, "feature_value") for row in ordered]
    x_min, x_max = min(xs), max(xs)
    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    sample_max = max(1.0, max(_float(row, "mean_sample_count") for row in ordered))

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def rate_y(value: float) -> float:
        return top + (1.0 - value) * plot_height

    def sample_y(value: float) -> float:
        return top + (1.0 - value / sample_max) * plot_height

    true_points = " ".join(
        f"{x_position(_float(row, 'feature_value')):.2f},{rate_y(_float(row, 'true_equal_outcome_rate')):.2f}"
        for row in ordered
    )
    closer_points = " ".join(
        f"{x_position(_float(row, 'feature_value')):.2f},{rate_y(_float(row, 'closer_to_true_equal_outcome_than_equal_split_rate')):.2f}"
        for row in ordered
    )
    sample_points = " ".join(
        f"{x_position(_float(row, 'feature_value')):.2f},{sample_y(_float(row, 'mean_sample_count')):.2f}"
        for row in ordered
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="28" font-family="Georgia, serif" font-size="20" fill="#19352f">Existing-result OAT: {html.escape(feature)}</text>',
        f'<text x="{left}" y="46" font-family="Arial, sans-serif" font-size="11" fill="#68756f">Exploratory slice; all other displayed environment parameters held at the selected anchor.</text>',
    ]
    for tick in range(6):
        rate = tick / 5.0
        y = rate_y(rate)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d9dfdb" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#56635e">{rate:.1f}</text>')
        sample_value = rate * sample_max
        parts.append(f'<text x="{width-right+12}" y="{y+4:.2f}" font-family="Arial, sans-serif" font-size="10" fill="#56635e">{sample_value:.1f}</text>')
    threshold_y = rate_y(0.8)
    parts.append(f'<line x1="{left}" y1="{threshold_y:.2f}" x2="{width-right}" y2="{threshold_y:.2f}" stroke="#9d6b32" stroke-width="1.5" stroke-dasharray="5 5"/>')
    for value in xs:
        x = x_position(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}" stroke="#eef1ef" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_height+22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#56635e">{value:g}</text>')
    parts.extend([
        f'<polyline points="{true_points}" fill="none" stroke="#137a68" stroke-width="3"/>',
        f'<polyline points="{closer_points}" fill="none" stroke="#d47432" stroke-width="3"/>',
        f'<polyline points="{sample_points}" fill="none" stroke="#315f8d" stroke-width="2.5" stroke-dasharray="5 4"/>',
    ])
    for row in ordered:
        x = x_position(_float(row, "feature_value"))
        parts.append(f'<circle cx="{x:.2f}" cy="{rate_y(_float(row, "true_equal_outcome_rate")):.2f}" r="4" fill="#137a68"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{rate_y(_float(row, "closer_to_true_equal_outcome_than_equal_split_rate")):.2f}" r="4" fill="#d47432"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{sample_y(_float(row, "mean_sample_count")):.2f}" r="3.5" fill="#315f8d"/>')
    legend_y = height - 20
    parts.extend([
        f'<text x="{left}" y="{legend_y}" font-family="Arial, sans-serif" font-size="11" fill="#137a68">True equal-outcome rate</text>',
        f'<text x="{left+180}" y="{legend_y}" font-family="Arial, sans-serif" font-size="11" fill="#d47432">Closer-to-equal-outcome rate</text>',
        f'<text x="{left+390}" y="{legend_y}" font-family="Arial, sans-serif" font-size="11" fill="#315f8d">Mean samples (right axis)</text>',
        f'<text x="18" y="{top+plot_height/2:.2f}" transform="rotate(-90 18 {top+plot_height/2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#56635e">Behavioral rate</text>',
        f'<text x="{width-14}" y="{top+plot_height/2:.2f}" transform="rotate(90 {width-14} {top+plot_height/2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#56635e">Mean online samples</text>',
        f'<text x="{left+plot_width/2:.2f}" y="{height-42}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#56635e">{html.escape(feature)}</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _validate_summary_environments(
    rows: Sequence[Mapping[str, object]],
    configs: Mapping[str, EnvironmentConfig],
    source: str,
) -> None:
    observed = {str(row["environment"]) for row in rows}
    expected = set(configs)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"{source} summary/manifest mismatch: missing={missing}, extra={extra}")


def _formal_sweep_row(
    row: Mapping[str, object],
    source: str,
    anchor: str,
    feature: str,
    feature_value: float,
) -> Dict[str, object]:
    fields = (
        "n_episodes",
        "mean_utility",
        "mean_utility_ci95",
        "mean_sample_count",
        "mean_abs_allocation_from_equal",
        "sample_time_cost",
        "sample_time_cost_percent",
        "true_equal_outcome_rate",
        "true_equal_outcome_ci95_low",
        "true_equal_outcome_ci95_high",
        "closer_to_true_equal_outcome_than_equal_split_rate",
        "closer_to_true_equal_outcome_than_equal_split_ci95_low",
        "closer_to_true_equal_outcome_than_equal_split_ci95_high",
        "sample_count_at_least_6_rate",
        "r5_joint_discovery_candidate",
    )
    result: Dict[str, object] = {
        "source": source,
        "anchor": anchor,
        "feature": feature,
        "feature_value": feature_value,
        "environment": row["environment"],
    }
    for field in fields:
        result[field] = row.get(field, "")
    true_rate = _float(row, "true_equal_outcome_rate")
    closer_rate = _float(row, "closer_to_true_equal_outcome_than_equal_split_rate")
    result["joint_min_rate"] = min(true_rate, closer_rate)
    result["r5_joint_0_8_0_8"] = 1.0 if true_rate >= 0.8 and closer_rate >= 0.8 else 0.0
    return result


def build_formal_sweep_rows(
    sampling_rows: Sequence[Mapping[str, object]],
    sampling_manifest: Mapping[str, object],
    oat_rows: Sequence[Mapping[str, object]],
    oat_manifest: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Join formal collector summaries to frozen anchor/feature identities."""

    sampling_configs = environment_configs(sampling_manifest)
    oat_configs = environment_configs(oat_manifest)
    _validate_summary_environments(sampling_rows, sampling_configs, "sampling")
    _validate_summary_environments(oat_rows, oat_configs, "oat")
    result: List[Dict[str, object]] = []
    for row in sampling_rows:
        environment = str(row["environment"])
        marker = "_sample_pct="
        if marker not in environment:
            raise RuntimeError(f"Sampling environment lacks {marker}: {environment}")
        anchor, raw_value = environment.rsplit(marker, 1)
        value = float(raw_value)
        result.append(_formal_sweep_row(row, "sampling_cost", anchor, "sample_time_cost_percent", value))
    for row in oat_rows:
        environment = str(row["environment"])
        matched = False
        for feature in FORMAL_OAT_FEATURES:
            prefix = f"{feature}_"
            marker = f"_{feature}="
            if environment.startswith(prefix) and marker in environment[len(prefix):]:
                body = environment[len(prefix):]
                anchor, raw_value = body.rsplit(marker, 1)
                result.append(_formal_sweep_row(row, "oat", anchor, feature, float(raw_value)))
                matched = True
                break
        if not matched:
            raise RuntimeError(f"Cannot recover OAT feature identity: {environment}")
    return sorted(
        result,
        key=lambda row: (str(row["source"]), str(row["feature"]), str(row["anchor"]), float(row["feature_value"])),
    )


def fixed_budget_evidence(
    rows: Sequence[Mapping[str, object]],
    target_budget: int = 6,
) -> List[Dict[str, object]]:
    """Summarize whether the paired utility curve is still increasing at six samples."""

    evidence: List[Dict[str, object]] = []
    environments = sorted({str(row["environment"]) for row in rows})
    for environment in environments:
        matches = [
            row
            for row in rows
            if str(row["environment"]) == environment
            and int(float(row["sampling_budget_total"])) == target_budget
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one budget-{target_budget} row for {environment}")
        row = matches[0]
        previous = int(float(row["previous_sampling_budget_total"]))
        delta = _float(row, "mean_incremental_utility_vs_previous_budget")
        half_width = _float(row, "paired_incremental_utility_ci95")
        evidence.append(
            {
                "environment": environment,
                "previous_sampling_budget_total": previous,
                "sampling_budget_total": target_budget,
                "paired_mean_utility_gain": delta,
                "paired_utility_ci95_low": delta - half_width,
                "paired_utility_ci95_high": delta + half_width,
                "six_observations_useful": 1.0 if previous == 4 and delta - half_width > 0.0 else 0.0,
                "true_equal_outcome_rate_at_6": _float(row, "true_equal_outcome_rate"),
                "closer_rate_at_6": _float(
                    row, "closer_to_true_equal_outcome_than_equal_split_rate"
                ),
            }
        )
    return evidence


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = float(statistics.mean(values))
    if len(values) <= 1:
        return mean, 0.0
    return mean, float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def _rows_by_episode(
    rows: Sequence[Mapping[str, object]],
    source: str,
) -> Dict[tuple[str, int], Mapping[str, object]]:
    indexed: Dict[tuple[str, int], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row["environment"]), int(float(row["episode_index"])))
        if key in indexed:
            raise RuntimeError(f"Duplicate {source} episode key: {key}")
        indexed[key] = row
    return indexed


def validate_solver_comparison_inputs(
    myopic_rows: Sequence[Mapping[str, object]],
    dp_rows: Sequence[Mapping[str, object]],
    myopic_manifest: Mapping[str, object],
    dp_manifest: Mapping[str, object],
) -> List[tuple[Mapping[str, object], Mapping[str, object]]]:
    """Require exact held-out episodes before comparing the two frozen solvers."""

    manifest_fields = (
        "family",
        "episodes_per_environment",
        "allocation_tolerance",
        "seed_namespace_offset",
        "configs_source_hash",
        "git_commit",
    )
    for field in manifest_fields:
        if myopic_manifest.get(field) != dp_manifest.get(field):
            raise RuntimeError(f"Solver manifests disagree on {field}")
    if myopic_manifest.get("family") != "custom_rr":
        raise RuntimeError("Solver comparison requires custom_rr manifests")
    if dict(myopic_manifest.get("rr_policy", {})).get("name") != "myopic_voi":
        raise RuntimeError("The myopic manifest does not freeze myopic_voi")
    if dict(dp_manifest.get("rr_policy", {})).get("name") != "discretized_dp":
        raise RuntimeError("The DP manifest does not freeze discretized_dp")
    if environment_configs(myopic_manifest) != environment_configs(dp_manifest):
        raise RuntimeError("Solver manifests contain different environment configurations")

    myopic_index = _rows_by_episode(myopic_rows, "myopic")
    dp_index = _rows_by_episode(dp_rows, "DP")
    if set(myopic_index) != set(dp_index):
        missing_dp = sorted(set(myopic_index) - set(dp_index))[:10]
        missing_myopic = sorted(set(dp_index) - set(myopic_index))[:10]
        raise RuntimeError(
            "Solver episode keys differ: "
            f"missing_dp={missing_dp}, missing_myopic={missing_myopic}"
        )

    pairs = []
    for key in sorted(myopic_index):
        myopic = myopic_index[key]
        dp = dp_index[key]
        for field in (
            "need_1",
            "need_2",
            "episode_fingerprint",
            "observation_stream_hash_1",
            "observation_stream_hash_2",
        ):
            if field not in myopic or field not in dp:
                raise RuntimeError(f"Missing paired-audit field {field} for {key}")
            if str(myopic[field]) != str(dp[field]):
                raise RuntimeError(f"Common-randomness mismatch in {field} for {key}")
        pairs.append((myopic, dp))
    return pairs


def build_solver_comparison_rows(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> List[Dict[str, object]]:
    """Summarize held-out DP versus myopic VOI with paired uncertainty."""

    environments = sorted({str(myopic["environment"]) for myopic, _ in pairs})
    rows: List[Dict[str, object]] = []
    for environment in environments:
        environment_pairs = [
            pair for pair in pairs if str(pair[0]["environment"]) == environment
        ]
        myopic_summary = summarize_r5_rr_environments(
            [pair[0] for pair in environment_pairs]
        )[0]
        dp_summary = summarize_r5_rr_environments(
            [pair[1] for pair in environment_pairs]
        )[0]
        utility_delta, utility_delta_ci95 = _mean_ci95(
            [
                _float(dp, "realized_utility") - _float(myopic, "realized_utility")
                for myopic, dp in environment_pairs
            ]
        )
        sample_delta, sample_delta_ci95 = _mean_ci95(
            [
                _float(dp, "online_sample_count")
                - _float(myopic, "online_sample_count")
                for myopic, dp in environment_pairs
            ]
        )
        row: Dict[str, object] = {
            "environment": environment,
            "n_paired_episodes": len(environment_pairs),
            "common_randomness_mismatch_count": 0,
            "paired_dp_minus_myopic_mean_utility": utility_delta,
            "paired_dp_minus_myopic_utility_ci95": utility_delta_ci95,
            "paired_dp_minus_myopic_utility_ci95_low": utility_delta - utility_delta_ci95,
            "paired_dp_minus_myopic_utility_ci95_high": utility_delta + utility_delta_ci95,
            "paired_dp_minus_myopic_mean_samples": sample_delta,
            "paired_dp_minus_myopic_samples_ci95": sample_delta_ci95,
        }
        for prefix, summary in (("myopic", myopic_summary), ("dp", dp_summary)):
            for field in (
                "mean_utility",
                "mean_utility_ci95",
                "mean_sample_count",
                "mean_abs_allocation_from_equal",
                "true_equal_outcome_rate",
                "true_equal_outcome_one_sided_95_low",
                "closer_to_true_equal_outcome_than_equal_split_rate",
                "closer_to_true_equal_outcome_than_equal_split_one_sided_95_low",
                "sample_count_at_least_6_rate",
                "r5_joint_confirmed",
            ):
                row[f"{prefix}_{field}"] = summary[field]
        row["dp_active_search_joint_confirmed"] = 1.0 if (
            float(dp_summary["r5_joint_confirmed"]) >= 0.5
            and float(dp_summary["mean_sample_count"]) > 1.0
            and float(dp_summary["mean_abs_allocation_from_equal"]) >= 0.05
        ) else 0.0
        rows.append(row)
    return rows


def write_solver_comparison_svg(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot paired DP-minus-myopic utility differences with 95% intervals."""

    if not rows:
        raise RuntimeError("No solver-comparison rows are available")
    width = 900
    left, right, top, bottom = 260, 70, 60, 65
    row_height = 78
    height = top + bottom + row_height * len(rows)
    low = min(_float(row, "paired_dp_minus_myopic_utility_ci95_low") for row in rows)
    high = max(_float(row, "paired_dp_minus_myopic_utility_ci95_high") for row in rows)
    bound = max(abs(low), abs(high), 1e-9) * 1.12
    plot_width = width - left - right

    def xp(value: float) -> float:
        return left + (value + bound) / (2.0 * bound) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="28" font-family="Georgia, serif" font-size="21" fill="#19352f">Frozen DP versus myopic VOI</text>',
        f'<text x="{left}" y="47" font-family="Arial, sans-serif" font-size="11" fill="#68756f">Paired mean realized-utility difference (DP minus myopic) with 95% confidence intervals.</text>',
        f'<line x1="{xp(0):.2f}" y1="{top-8}" x2="{xp(0):.2f}" y2="{height-bottom+8}" stroke="#9d6b32" stroke-dasharray="5 5"/>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height + row_height / 2
        mean = _float(row, "paired_dp_minus_myopic_mean_utility")
        lower = _float(row, "paired_dp_minus_myopic_utility_ci95_low")
        upper = _float(row, "paired_dp_minus_myopic_utility_ci95_high")
        parts.extend(
            [
                f'<text x="{left-14}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#33443e">{html.escape(str(row["environment"]))}</text>',
                f'<line x1="{xp(lower):.2f}" y1="{y:.2f}" x2="{xp(upper):.2f}" y2="{y:.2f}" stroke="#315f8d" stroke-width="2.5"/>',
                f'<circle cx="{xp(mean):.2f}" cy="{y:.2f}" r="5" fill="#137a68"/>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left}" y="{height-18}" font-family="Arial, sans-serif" font-size="10" fill="#56635e">Myopic higher</text>',
            f'<text x="{width-right}" y="{height-18}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#56635e">DP higher</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze_solver_comparison(args) -> None:
    myopic_rows = read_csv(Path(args.myopic_episodes))
    dp_rows = read_csv(Path(args.dp_episodes))
    myopic_manifest = load_manifest(Path(args.myopic_manifest))
    dp_manifest = load_manifest(Path(args.dp_manifest))
    pairs = validate_solver_comparison_inputs(
        myopic_rows,
        dp_rows,
        myopic_manifest,
        dp_manifest,
    )
    rows = build_solver_comparison_rows(pairs)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "r5_solver_comparison.csv", rows)
    write_solver_comparison_svg(
        output_dir / "figures" / "r5_solver_paired_utility.svg",
        rows,
    )
    summary = {
        "environment_count": len(rows),
        "paired_episode_count": len(pairs),
        "common_randomness_mismatch_count": 0,
        "dp_active_search_joint_confirmed_count": sum(
            float(row["dp_active_search_joint_confirmed"]) >= 0.5 for row in rows
        ),
        "dp_utility_ci_above_zero_count": sum(
            _float(row, "paired_dp_minus_myopic_utility_ci95_low") > 0.0 for row in rows
        ),
        "dp_utility_ci_below_zero_count": sum(
            _float(row, "paired_dp_minus_myopic_utility_ci95_high") < 0.0 for row in rows
        ),
        "myopic_manifest_hash": myopic_manifest["manifest_hash"],
        "dp_manifest_hash": dp_manifest["manifest_hash"],
        "configs_source_hash": myopic_manifest["configs_source_hash"],
        "seed_namespace_offset": myopic_manifest["seed_namespace_offset"],
    }
    write_json(output_dir / "r5_solver_comparison_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def _polyline(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_formal_feature_svg(
    path: Path,
    feature: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write four small-multiple panels for one formal sweep feature."""

    anchors = sorted({str(row["anchor"]) for row in rows})
    if not anchors:
        raise RuntimeError(f"No rows available for {feature}")
    width, panel_height = 900, 245
    height = 70 + panel_height * len(anchors)
    left, right, panel_top, panel_bottom = 76, 80, 30, 45
    plot_width = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="30" font-family="Georgia, serif" font-size="22" fill="#19352f">Formal sweep: {html.escape(feature)}</text>',
        f'<text x="{left}" y="50" font-family="Arial, sans-serif" font-size="11" fill="#68756f">Rates use the left axis; mean online samples use the right axis. Dashed horizontal line marks 0.8.</text>',
    ]
    for panel_index, anchor in enumerate(anchors):
        panel_rows = sorted(
            [row for row in rows if str(row["anchor"]) == anchor],
            key=lambda row: float(row["feature_value"]),
        )
        xs = [float(row["feature_value"]) for row in panel_rows]
        x_min, x_max = min(xs), max(xs)
        if math.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        sample_max = max(1.0, max(_float(row, "mean_sample_count") for row in panel_rows))
        top = 65 + panel_index * panel_height + panel_top
        plot_height = panel_height - panel_top - panel_bottom

        def xp(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * plot_width

        def rate_y(value: float) -> float:
            return top + (1.0 - value) * plot_height

        def sample_y(value: float) -> float:
            return top + (1.0 - value / sample_max) * plot_height

        parts.append(f'<text x="{left}" y="{top-12}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#33443e">{html.escape(anchor)}</text>')
        for tick in range(6):
            value = tick / 5.0
            y = rate_y(value)
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d9dfdb"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="9" fill="#56635e">{value:.1f}</text>')
            parts.append(f'<text x="{width-right+10}" y="{y+4:.2f}" font-family="Arial, sans-serif" font-size="9" fill="#56635e">{value*sample_max:.1f}</text>')
        threshold = rate_y(0.8)
        parts.append(f'<line x1="{left}" y1="{threshold:.2f}" x2="{width-right}" y2="{threshold:.2f}" stroke="#9d6b32" stroke-dasharray="5 5"/>')
        true_points = [(xp(float(row["feature_value"])), rate_y(_float(row, "true_equal_outcome_rate"))) for row in panel_rows]
        closer_points = [(xp(float(row["feature_value"])), rate_y(_float(row, "closer_to_true_equal_outcome_than_equal_split_rate"))) for row in panel_rows]
        sample_points = [(xp(float(row["feature_value"])), sample_y(_float(row, "mean_sample_count"))) for row in panel_rows]
        parts.extend([
            f'<polyline points="{_polyline(true_points)}" fill="none" stroke="#137a68" stroke-width="3"/>',
            f'<polyline points="{_polyline(closer_points)}" fill="none" stroke="#d47432" stroke-width="3"/>',
            f'<polyline points="{_polyline(sample_points)}" fill="none" stroke="#315f8d" stroke-width="2.5" stroke-dasharray="5 4"/>',
        ])
        for value in xs:
            x = xp(value)
            parts.append(f'<text x="{x:.2f}" y="{top+plot_height+18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#56635e">{value:g}</text>')
    parts.extend([
        f'<text x="{left}" y="{height-10}" font-family="Arial, sans-serif" font-size="11" fill="#137a68">True equal-outcome rate</text>',
        f'<text x="{left+185}" y="{height-10}" font-family="Arial, sans-serif" font-size="11" fill="#d47432">Closer rate</text>',
        f'<text x="{left+285}" y="{height-10}" font-family="Arial, sans-serif" font-size="11" fill="#315f8d">Mean samples</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_fixed_budget_svg(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write paired fixed-information utility curves with 95% CI bars."""

    anchors = sorted({str(row["environment"]) for row in rows})
    width, panel_height = 900, 230
    height = 70 + panel_height * len(anchors)
    left, right = 82, 45
    plot_width = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="30" font-family="Georgia, serif" font-size="22" fill="#19352f">Fixed information-budget utility curves</text>',
        f'<text x="{left}" y="50" font-family="Arial, sans-serif" font-size="11" fill="#68756f">Points are mean realized utility; vertical bars are mean-utility 95% CIs. Six observations are highlighted.</text>',
    ]
    for panel_index, anchor in enumerate(anchors):
        panel_rows = sorted(
            [row for row in rows if str(row["environment"]) == anchor],
            key=lambda row: float(row["sampling_budget_total"]),
        )
        budgets = [float(row["sampling_budget_total"]) for row in panel_rows]
        means = [_float(row, "mean_utility") for row in panel_rows]
        half_widths = [_float(row, "mean_utility_ci95") for row in panel_rows]
        y_min = min(mean-half for mean, half in zip(means, half_widths))
        y_max = max(mean+half for mean, half in zip(means, half_widths))
        padding = max(1e-6, (y_max-y_min)*0.12)
        y_min -= padding
        y_max += padding
        top = 80 + panel_index * panel_height
        plot_height = panel_height - 58

        def xp(value: float) -> float:
            return left + value / max(budgets) * plot_width if max(budgets) else left

        def yp(value: float) -> float:
            return top + (y_max-value)/(y_max-y_min)*plot_height

        parts.append(f'<text x="{left}" y="{top-10}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#33443e">{html.escape(anchor)}</text>')
        for tick in range(5):
            value = y_min + tick*(y_max-y_min)/4
            y = yp(value)
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d9dfdb"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="9" fill="#56635e">{value:.2f}</text>')
        points = [(xp(budget), yp(mean)) for budget, mean in zip(budgets, means)]
        parts.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="#137a68" stroke-width="3"/>')
        for budget, mean, half in zip(budgets, means, half_widths):
            x = xp(budget)
            color = "#c56a2d" if budget == 6 else "#137a68"
            parts.append(f'<line x1="{x:.2f}" y1="{yp(mean-half):.2f}" x2="{x:.2f}" y2="{yp(mean+half):.2f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{yp(mean):.2f}" r="4" fill="{color}"/>')
            parts.append(f'<text x="{x:.2f}" y="{top+plot_height+18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#56635e">{budget:g}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze_formal_summaries(args) -> None:
    sampling_rows = read_csv(Path(args.sampling_summary))
    oat_rows = read_csv(Path(args.oat_summary))
    fixed_rows = read_csv(Path(args.fixed_budget_summary))
    sweep_rows = build_formal_sweep_rows(
        sampling_rows,
        load_manifest(Path(args.sampling_manifest)),
        oat_rows,
        load_manifest(Path(args.oat_manifest)),
    )
    fixed_evidence = fixed_budget_evidence(fixed_rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "r5_formal_sweep_long.csv", sweep_rows)
    write_csv(output_dir / "r5_fixed_budget_evidence.csv", fixed_evidence)
    for feature in ("sample_time_cost_percent",) + FORMAL_OAT_FEATURES:
        feature_rows = [row for row in sweep_rows if row["feature"] == feature]
        if feature_rows:
            write_formal_feature_svg(
                output_dir / "figures" / f"r5_formal_{feature}.svg",
                feature,
                feature_rows,
            )
    write_fixed_budget_svg(output_dir / "figures" / "r5_fixed_budget_utility.svg", fixed_rows)
    joint_rows = [row for row in sweep_rows if float(row["r5_joint_0_8_0_8"]) >= 0.5]
    best = max(sweep_rows, key=lambda row: float(row["joint_min_rate"]))
    result = {
        "sampling_environment_count": len(sampling_rows),
        "oat_environment_count": len(oat_rows),
        "joint_0_8_0_8_row_count": len(joint_rows),
        "best_joint_environment": best["environment"],
        "best_joint_min_rate": float(best["joint_min_rate"]),
        "fixed_budget_anchor_count": len(fixed_evidence),
        "six_observation_value_anchor_count": sum(
            float(row["six_observations_useful"]) >= 0.5 for row in fixed_evidence
        ),
    }
    write_json(output_dir / "r5_formal_summary.json", result)
    print(json.dumps(result, sort_keys=True))


def analyze_existing(args) -> None:
    rows = [
        row
        for row in read_csv(Path(args.profiles))
        if row.get("policy") == args.policy
    ]
    anchor = select_existing_rr_anchor(rows)
    slices = build_existing_oat_slices(rows, anchor)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "r5_existing_oat_slices.csv", slices)
    for feature in EXISTING_OAT_FEATURES:
        feature_rows = [row for row in slices if row["feature"] == feature]
        if len(feature_rows) > 1:
            write_existing_oat_svg(
                output_dir / f"r5_existing_oat_{feature}.svg",
                feature,
                feature_rows,
            )

    candidate_count = sum(
        _float(row, "true_equal_outcome_rate") >= 0.8
        and _float(row, "closer_to_true_equal_outcome_than_equal_split_rate") >= 0.8
        for row in rows
    )
    summary = {
        "source": Path(args.profiles).name,
        "policy": args.policy,
        "environment_count": len(rows),
        "joint_0_8_0_8_environment_count": candidate_count,
        "anchor_selection": "maximize the smaller of the two behavioral rates, with deterministic tie-breakers",
        "anchor_environment": anchor["environment"],
        "anchor_true_equal_outcome_rate": _float(anchor, "true_equal_outcome_rate"),
        "anchor_closer_rate": _float(
            anchor, "closer_to_true_equal_outcome_than_equal_split_rate"
        ),
        "anchor_mean_sample_count": _float(anchor, "mean_sample_count"),
        "oat_row_count": len(slices),
        "interpretation_status": "exploratory reanalysis of existing results",
    }
    write_json(output_dir / "r5_existing_reanalysis_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def select_oracle_anchors(
    summaries: Sequence[Mapping[str, object]],
    configs: Mapping[str, EnvironmentConfig],
    count: int,
) -> List[tuple[str, EnvironmentConfig]]:
    eligible = [
        row
        for row in summaries
        if float(row["r5_joint_oracle_candidate"]) >= 0.5
        and float(row["exact_true_equal_outcome_feasibility_rate"]) >= 0.95
        and float(row["negative_need_either_rate"]) <= 0.05
    ]
    if not eligible:
        raise RuntimeError("No oracle-compatible anchor satisfies the frozen filters")

    rankings = (
        lambda row: float(row["mean_equal_split_regret"]),
        lambda row: min(
            float(row["oracle_true_equal_outcome_rate"]),
            float(row["oracle_closer_to_true_equal_than_equal_split_rate"]),
        ),
        lambda row: float(configs[str(row["environment"])].sigma_need),
        lambda row: -float(row["negative_need_either_rate"]),
    )
    selected: List[Mapping[str, object]] = []
    for ranking in rankings:
        for row in sorted(eligible, key=ranking, reverse=True):
            if row not in selected:
                selected.append(row)
                break
    for row in sorted(
        eligible,
        key=lambda item: (
            min(
                float(item["oracle_true_equal_outcome_rate"]),
                float(item["oracle_closer_to_true_equal_than_equal_split_rate"]),
            ),
            float(item["mean_equal_split_regret"]),
        ),
        reverse=True,
    ):
        if row not in selected:
            selected.append(row)
        if len(selected) >= count:
            break
    return [
        (f"anchor_{index + 1}_{row['environment']}", configs[str(row["environment"])])
        for index, row in enumerate(selected[:count])
    ]


def analyze_oracle(args) -> None:
    rows = read_csv(Path(args.episodes))
    repair_legacy_oracle_rows(rows)
    summaries = summarize_r5_oracle_map(rows)
    manifest = load_manifest(Path(args.manifest))
    configs = environment_configs(manifest)
    anchors = select_oracle_anchors(summaries, configs, args.anchor_count)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "r5_oracle_sign_summary.csv", summaries)
    write_json(
        output_dir / "r5_oracle_anchors.json",
        config_payload(
            anchors,
            source=str(Path(args.episodes).name),
            selection_rule=(
                "joint oracle 0.8/0.8; feasibility >= 0.95; negative-need rate <= 0.05; "
                "deterministic diversity rankings"
            ),
        ),
    )

    sampling = build_r5_sampling_cost_configs(anchors)
    write_json(
        output_dir / "r5_sampling_cost_sweep_configs.json",
        config_payload(sampling, "r5_oracle_anchors.json", "prespecified sample-cost percentages"),
    )

    oat_values = {
        "sigma_sample": [5.0, 10.0, 20.0, 30.0, 40.0, 60.0],
        "total_time": [120.0, 140.0, 160.0, 180.0, 200.0],
        "sigma_need": [10.0, 20.0, 30.0, 40.0, 60.0],
        "utility_exponent": [0.25, 0.35, 0.5, 0.75],
        "learning_per_unit_of_tutoring": [0.75, 1.0, 1.25],
    }
    oat_configs: List[tuple[str, EnvironmentConfig]] = []
    for anchor_name, anchor in anchors:
        for feature, _, name, config in build_r5_oat_configs(anchor_name, anchor, oat_values):
            oat_configs.append((f"{feature}_{name}", config))
    write_json(
        output_dir / "r5_oat_sweep_configs.json",
        config_payload(oat_configs, "r5_oracle_anchors.json", "one factor at a time"),
    )

    candidates = [row for row in summaries if float(row["r5_joint_oracle_candidate"]) >= 0.5]
    result = {
        "environment_count": len(summaries),
        "joint_oracle_candidate_count": len(candidates),
        "selected_anchor_count": len(anchors),
        "sampling_sweep_environment_count": len(sampling),
        "oat_environment_count": len(oat_configs),
        "max_grid_optimality_violation": max(
            float(row["max_oracle_grid_optimality_violation"]) for row in summaries
        ),
    }
    write_json(output_dir / "r5_oracle_analysis.json", result)
    print(json.dumps(result, sort_keys=True))


def select_confirmation(args) -> None:
    summaries = read_csv(Path(args.summary))
    manifest = load_manifest(Path(args.manifest))
    configs = environment_configs(manifest)
    eligible = [
        row
        for row in summaries
        if float(row["r5_joint_discovery_candidate"]) >= 0.5
        and float(row["mean_sample_count"]) > 1.0
        and float(row["mean_abs_allocation_from_equal"]) >= 0.05
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            min(
                float(row["true_equal_outcome_rate"]),
                float(row["closer_to_true_equal_outcome_than_equal_split_rate"]),
            ),
            float(row["sample_count_at_least_6_rate"]),
            float(row["mean_sample_count"]),
        ),
        reverse=True,
    )
    selected = ranked[: args.max_candidates]
    frozen = [
        (f"confirmation_{index + 1}_{row['environment']}", configs[row["environment"]])
        for index, row in enumerate(selected)
    ]
    output = Path(args.output)
    payload = config_payload(
        frozen,
        source=str(Path(args.summary).name),
        selection_rule=(
            "joint discovery 0.8/0.8; mean samples > 1; mean absolute allocation "
            "distance from equal split >= 0.05; deterministic rank"
        ),
    )
    payload["discovery_rows"] = selected
    write_json(output, payload)
    print(json.dumps({"eligible": len(eligible), "selected": len(selected)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze and freeze R5 discovery outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    oracle = subparsers.add_parser("oracle")
    oracle.add_argument("--episodes", required=True)
    oracle.add_argument("--manifest", required=True)
    oracle.add_argument("--output-dir", required=True)
    oracle.add_argument("--anchor-count", type=int, default=4)
    oracle.set_defaults(function=analyze_oracle)

    confirmation = subparsers.add_parser("select-confirmation")
    confirmation.add_argument("--summary", required=True)
    confirmation.add_argument("--manifest", required=True)
    confirmation.add_argument("--output", required=True)
    confirmation.add_argument("--max-candidates", type=int, default=12)
    confirmation.set_defaults(function=select_confirmation)

    existing = subparsers.add_parser("reanalyze-existing")
    existing.add_argument("--profiles", required=True)
    existing.add_argument("--output-dir", required=True)
    existing.add_argument("--policy", default="myopic_voi")
    existing.set_defaults(function=analyze_existing)

    formal = subparsers.add_parser("formal-summaries")
    formal.add_argument("--sampling-summary", required=True)
    formal.add_argument("--sampling-manifest", required=True)
    formal.add_argument("--oat-summary", required=True)
    formal.add_argument("--oat-manifest", required=True)
    formal.add_argument("--fixed-budget-summary", required=True)
    formal.add_argument("--output-dir", required=True)
    formal.set_defaults(function=analyze_formal_summaries)

    solver = subparsers.add_parser("solver-comparison")
    solver.add_argument("--myopic-episodes", required=True)
    solver.add_argument("--myopic-manifest", required=True)
    solver.add_argument("--dp-episodes", required=True)
    solver.add_argument("--dp-manifest", required=True)
    solver.add_argument("--output-dir", required=True)
    solver.set_defaults(function=analyze_solver_comparison)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
