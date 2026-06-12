from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.compare import ENVIRONMENT_LIBRARY  # noqa: E402
from src.experiments.dp_diagnostics import run_dp_sensitivity_analysis  # noqa: E402
from src.experiments.randomization import build_evaluation_episodes  # noqa: E402
from src.experiments.regimes import (  # noqa: E402
    compare_policy_behavior_profiles,
    compare_rr_approximation_methods,
    compare_rr_information_acquisition_to_heuristics,
    compare_rr_to_heuristics_by_final_choice,
)
from src.experiments.settings import (  # noqa: E402
    SERVER_EVALUATION_SETTINGS,
    SERIOUS_LOCAL_EVALUATION_SETTINGS,
    SMOKE_EVALUATION_SETTINGS,
    EvaluationSettings,
    build_rr_approximation_policies_from_settings,
    build_rr_policy_from_settings,
    settings_with_overrides,
)
from src.experiments.sweeps import (  # noqa: E402
    ONE_DIMENSIONAL_SWEEP_VALUES,
    build_all_one_dimensional_sweep_configs,
    build_positive_and_near_zero_utility_configs,
    identify_final_choice_regime_candidates,
    identify_rr_behavior_regime_candidates,
    run_one_dimensional_final_choice_sweeps,
    run_one_dimensional_rr_behavior_sweeps,
)
from src.mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig  # noqa: E402
from src.solvers.gauss_hermite import expected_terminal_utility_gauss_hermite, normal_expectation_1d  # noqa: E402


PRESETS: Dict[str, EvaluationSettings] = {
    "smoke": SMOKE_EVALUATION_SETTINGS,
    "serious": SERIOUS_LOCAL_EVALUATION_SETTINGS,
    "server": SERVER_EVALUATION_SETTINGS,
}


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_sections(value: str) -> set[str]:
    sections = {part.strip() for part in value.split(",") if part.strip()}
    if not sections or "all" in sections:
        return {"step7", "sweeps", "regimes", "dp", "gh"}
    return sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Round 2 resource-rational allocation results.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--output-dir", default="results/round2_current")
    parser.add_argument("--sections", default="all", help="Comma-separated: step7,sweeps,regimes,dp,gh or all.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--voi-samples", type=int, default=None)
    parser.add_argument("--blinkered-samples", type=int, default=None)
    parser.add_argument("--common-observations", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--observations-per-person", type=int, default=None)
    parser.add_argument("--allocation-grid-size", type=int, default=None)
    parser.add_argument("--expected-utility-draws", type=int, default=None)
    parser.add_argument("--terminal-integration", choices=["monte_carlo", "gauss_hermite"], default=None)
    parser.add_argument("--gauss-hermite-order", type=int, default=15)
    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--sweep-feature", action="append", default=[])
    parser.add_argument("--max-sweep-values-per-feature", type=int, default=None)
    parser.add_argument("--dp-max-samples-values", default="2,4,6,10")
    parser.add_argument("--dp-mean-grid-sizes", default="7,11,21,50")
    parser.add_argument("--dp-observation-branches", default="3,5")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> EvaluationSettings:
    settings = PRESETS[args.preset]
    if args.common_observations == "on":
        common_observations = True
    elif args.common_observations == "off":
        common_observations = False
    else:
        common_observations = None
    return settings_with_overrides(
        settings,
        n_episodes=args.episodes,
        rr_observation_draws=args.voi_samples,
        blinkered_observation_draws=args.blinkered_samples,
        use_common_observation_streams=common_observations,
        observations_per_person=args.observations_per_person,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    ensure_dir(path.parent)
    resolved_fieldnames: List[str] = list(fieldnames or [])
    for row in rows:
        for key in row.keys():
            if key not in resolved_fieldnames:
                resolved_fieldnames.append(key)
    if not resolved_fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_fieldnames(source_rows: Sequence[Dict[str, object]]) -> List[str]:
    """Keep candidate CSVs readable even when no regimes pass thresholds."""
    fieldnames = ["candidate_type"]
    for row in source_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _color(value: float, min_value: float, max_value: float) -> str:
    if math.isnan(value):
        return "#eeeeee"
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = min(1.0, max(0.0, ratio))
    red = int(240 * ratio + 40 * (1 - ratio))
    green = int(80 * ratio + 180 * (1 - ratio))
    blue = int(70 * ratio + 220 * (1 - ratio))
    return f"#{red:02x}{green:02x}{blue:02x}"


def write_heatmap_svg(
    path: Path,
    rows: Sequence[Dict[str, object]],
    x_key: str,
    y_key: str,
    value_key: str,
    title: str,
    max_labels: int = 40,
) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    xs = list(dict.fromkeys(str(row[x_key]) for row in rows))[:max_labels]
    ys = list(dict.fromkeys(str(row[y_key]) for row in rows))[:max_labels]
    values = {
        (str(row[x_key]), str(row[y_key])): _as_float(row[value_key])
        for row in rows
        if str(row[x_key]) in xs and str(row[y_key]) in ys
    }
    numeric_values = [value for value in values.values() if not math.isnan(value)]
    min_value = min(numeric_values) if numeric_values else 0.0
    max_value = max(numeric_values) if numeric_values else 1.0
    cell = 26
    left = 190
    top = 70
    width = left + cell * len(xs) + 40
    height = top + cell * len(ys) + 80
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Helvetica,Arial,sans-serif;font-size:11px}.title{font-size:16px;font-weight:bold}</style>",
        f"<text class='title' x='20' y='28'>{title}</text>",
        f"<text x='20' y='48'>metric: {value_key}</text>",
    ]
    for x_index, x_label in enumerate(xs):
        x = left + x_index * cell + 4
        parts.append(f"<text x='{x}' y='{top - 8}' transform='rotate(-45 {x},{top - 8})'>{x_label}</text>")
    for y_index, y_label in enumerate(ys):
        y = top + y_index * cell + 17
        parts.append(f"<text x='10' y='{y}'>{y_label}</text>")
    for x_index, x_label in enumerate(xs):
        for y_index, y_label in enumerate(ys):
            value = values.get((x_label, y_label), math.nan)
            color = _color(value, min_value, max_value)
            x = left + x_index * cell
            y = top + y_index * cell
            parts.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' fill='{color}' stroke='white'/>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_analysis_environments(args: argparse.Namespace) -> Dict[str, EnvironmentConfig]:
    configs = dict(ENVIRONMENT_LIBRARY)
    configs.update(dict(build_positive_and_near_zero_utility_configs()))
    configs = {name: apply_environment_overrides(config, args) for name, config in configs.items()}
    if args.environment:
        missing = sorted(set(args.environment) - set(configs))
        if missing:
            raise ValueError(f"Unknown environments: {missing}")
        return {name: configs[name] for name in args.environment}
    return configs


def apply_environment_overrides(config: EnvironmentConfig, args: argparse.Namespace) -> EnvironmentConfig:
    updates = {}
    if args.allocation_grid_size is not None:
        updates["allocation_grid_size"] = args.allocation_grid_size
    if args.expected_utility_draws is not None:
        updates["expected_utility_draws"] = args.expected_utility_draws
    if args.terminal_integration is not None:
        updates["expected_utility_method"] = args.terminal_integration
    if args.gauss_hermite_order is not None:
        updates["gauss_hermite_order"] = args.gauss_hermite_order
    return replace(config, **updates)


def build_sweep_configs_for_args(args: argparse.Namespace) -> List[tuple[str, float, str, EnvironmentConfig]]:
    values = {
        feature: list(feature_values)
        for feature, feature_values in ONE_DIMENSIONAL_SWEEP_VALUES.items()
        if not args.sweep_feature or feature in args.sweep_feature
    }
    if args.max_sweep_values_per_feature is not None:
        values = {
            feature: feature_values[: args.max_sweep_values_per_feature]
            for feature, feature_values in values.items()
        }
    configs = build_all_one_dimensional_sweep_configs(values)
    return [
        (feature, value, environment_name, apply_environment_overrides(config, args))
        for feature, value, environment_name, config in configs
    ]


def run_step7_outputs(
    output_dir: Path,
    environments: Dict[str, EnvironmentConfig],
    settings: EvaluationSettings,
) -> Dict[str, List[Dict[str, object]]]:
    rr_policy = build_rr_policy_from_settings(settings)
    approximation_policies = build_rr_approximation_policies_from_settings(settings)
    final_choice_rows: List[Dict[str, object]] = []
    info_rows: List[Dict[str, object]] = []
    behavior_rows: List[Dict[str, object]] = []
    approximation_rows: List[Dict[str, object]] = []

    for environment_name, config in environments.items():
        episodes = build_evaluation_episodes(
            config=config,
            n_episodes=settings.n_episodes,
            include_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        final_choice_rows.extend(
            compare_rr_to_heuristics_by_final_choice(
                environment_name=environment_name,
                config=config,
                n_episodes=settings.n_episodes,
                rr_policy=rr_policy,
                evaluation_episodes=episodes,
                use_common_observation_streams=settings.use_common_observation_streams,
                observations_per_person=settings.observations_per_person,
            )
        )
        info_rows.extend(
            compare_rr_information_acquisition_to_heuristics(
                environment_name=environment_name,
                config=config,
                n_episodes=settings.n_episodes,
                rr_policy=rr_policy,
                evaluation_episodes=episodes,
                use_common_observation_streams=settings.use_common_observation_streams,
                observations_per_person=settings.observations_per_person,
            )
        )
        behavior_rows.extend(
            compare_policy_behavior_profiles(
                environment_name=environment_name,
                config=config,
                n_episodes=settings.n_episodes,
                rr_policy=rr_policy,
                evaluation_episodes=episodes,
                use_common_observation_streams=settings.use_common_observation_streams,
                observations_per_person=settings.observations_per_person,
            )
        )
        approximation_rows.extend(
            compare_rr_approximation_methods(
                environment_name=environment_name,
                config=config,
                n_episodes=settings.n_episodes,
                policies=approximation_policies,
                evaluation_episodes=episodes,
                use_common_observation_streams=settings.use_common_observation_streams,
                observations_per_person=settings.observations_per_person,
            )
        )

    write_csv(output_dir / "step7_final_choice_comparison.csv", final_choice_rows)
    write_csv(output_dir / "step7_information_acquisition_comparison.csv", info_rows)
    write_csv(output_dir / "step7_behavior_profiles.csv", behavior_rows)
    write_csv(output_dir / "rr_approximation_methods_comparison.csv", approximation_rows)
    return {
        "final_choice": final_choice_rows,
        "information_acquisition": info_rows,
        "behavior_profiles": behavior_rows,
        "approximation_methods": approximation_rows,
    }


def run_sweep_outputs(
    output_dir: Path,
    sweep_configs: List[tuple[str, float, str, EnvironmentConfig]],
    settings: EvaluationSettings,
) -> Dict[str, List[Dict[str, object]]]:
    final_choice_rows = run_one_dimensional_final_choice_sweeps(
        sweep_configs=sweep_configs,
        settings=settings,
    )
    behavior_rows = run_one_dimensional_rr_behavior_sweeps(
        sweep_configs=sweep_configs,
        settings=settings,
    )
    final_candidates = identify_final_choice_regime_candidates(final_choice_rows)
    behavior_candidates = identify_rr_behavior_regime_candidates(behavior_rows)
    write_csv(output_dir / "sweep_final_choice_comparison.csv", final_choice_rows)
    write_csv(output_dir / "sweep_rr_behavior_profiles.csv", behavior_rows)
    write_csv(
        output_dir / "sweep_final_choice_candidates.csv",
        final_candidates,
        fieldnames=candidate_fieldnames(final_choice_rows),
    )
    write_csv(
        output_dir / "sweep_behavior_candidates.csv",
        behavior_candidates,
        fieldnames=candidate_fieldnames(behavior_rows),
    )
    return {
        "sweep_final_choice": final_choice_rows,
        "sweep_behavior": behavior_rows,
        "sweep_final_choice_candidates": final_candidates,
        "sweep_behavior_candidates": behavior_candidates,
    }


def run_dp_outputs(
    output_dir: Path,
    environments: Dict[str, EnvironmentConfig],
    settings: EvaluationSettings,
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    max_samples_values = parse_int_list(args.dp_max_samples_values)
    mean_grid_sizes = parse_int_list(args.dp_mean_grid_sizes)
    observation_branches = parse_int_list(args.dp_observation_branches)
    for environment_name, config in environments.items():
        rows.extend(
            run_dp_sensitivity_analysis(
                environment_name=environment_name,
                config=config,
                settings=settings,
                max_samples_values=max_samples_values,
                mean_grid_sizes=mean_grid_sizes,
                observation_branches_values=observation_branches,
            )
        )
    write_csv(output_dir / "dp_sensitivity_analysis.csv", rows)
    return {"dp_sensitivity": rows}


def run_gauss_hermite_outputs(
    output_dir: Path,
    environments: Dict[str, EnvironmentConfig],
    order: int,
) -> Dict[str, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = [
        {
            "check": "standard_normal_second_moment",
            "gauss_hermite_order": order,
            "estimated_value": normal_expectation_1d(0.0, 1.0, lambda value: value * value, order=order),
            "expected_value": 1.0,
        }
    ]
    for environment_name, config in environments.items():
        gh_config = replace(config, expected_utility_method="monte_carlo", random_seed=(config.random_seed or 0) + 909)
        mdp = ContinuousAllocationMetaMDP(gh_config)
        belief = mdp.initial_belief()
        for allocation in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mc_value = mdp.expected_terminal_utility(belief, allocation)
            gh_value = expected_terminal_utility_gauss_hermite(mdp, belief, allocation, order=order)
            rows.append(
                {
                    "check": "terminal_utility_comparison",
                    "environment": environment_name,
                    "allocation_to_person1": allocation,
                    "monte_carlo_value": mc_value,
                    "gauss_hermite_value": gh_value,
                    "gh_minus_mc": gh_value - mc_value,
                    "gauss_hermite_order": order,
                }
            )
    write_csv(output_dir / "gauss_hermite_diagnostics.csv", rows)
    return {"gauss_hermite": rows}


def write_figures(output_dir: Path, result_sets: Dict[str, List[Dict[str, object]]]) -> None:
    figures_dir = output_dir / "figures"
    if "final_choice" in result_sets:
        rows = result_sets["final_choice"]
        write_heatmap_svg(
            figures_dir / "step7_final_choice_match_rate_heatmap.svg",
            rows,
            x_key="environment",
            y_key="heuristic",
            value_key="final_choice_match_rate",
            title="Final-choice tolerance match rate",
        )
        write_heatmap_svg(
            figures_dir / "step7_mean_abs_allocation_gap_heatmap.svg",
            rows,
            x_key="environment",
            y_key="heuristic",
            value_key="mean_abs_allocation_gap",
            title="Mean absolute allocation gap",
        )
        write_heatmap_svg(
            figures_dir / "step7_rmse_allocation_gap_heatmap.svg",
            rows,
            x_key="environment",
            y_key="heuristic",
            value_key="rmse_allocation_gap",
            title="RMSE allocation gap",
        )
    if "behavior_profiles" in result_sets:
        rows = result_sets["behavior_profiles"]
        write_heatmap_svg(
            figures_dir / "step7_behavior_near_equal_allocation_rate.svg",
            rows,
            x_key="environment",
            y_key="policy",
            value_key="near_equal_allocation_rate",
            title="Near-50/50 allocation rate",
        )
        write_heatmap_svg(
            figures_dir / "step7_behavior_equal_outcome_rate.svg",
            rows,
            x_key="environment",
            y_key="policy",
            value_key="equal_outcome_rate",
            title="Equal-outcome allocation rate",
        )
        write_heatmap_svg(
            figures_dir / "step7_behavior_mean_sample_count.svg",
            rows,
            x_key="environment",
            y_key="policy",
            value_key="mean_sample_count",
            title="Mean sample count",
        )
    if "approximation_methods" in result_sets:
        rows = result_sets["approximation_methods"]
        write_heatmap_svg(
            figures_dir / "rr_approximation_method_regret_heatmap.svg",
            rows,
            x_key="environment",
            y_key="policy",
            value_key="regret_vs_best_rr_approximation",
            title="RR approximation regret vs best method",
        )
    if "sweep_behavior" in result_sets:
        rows = result_sets["sweep_behavior"]
        write_heatmap_svg(
            figures_dir / "sweep_near_equal_allocation_rate.svg",
            rows,
            x_key="sweep_value",
            y_key="sweep_feature",
            value_key="near_equal_allocation_rate",
            title="RR near-50/50 rate across one-dimensional sweeps",
        )
        write_heatmap_svg(
            figures_dir / "sweep_equal_outcome_rate.svg",
            rows,
            x_key="sweep_value",
            y_key="sweep_feature",
            value_key="equal_outcome_rate",
            title="RR equal-outcome rate across one-dimensional sweeps",
        )


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    settings: EvaluationSettings,
    result_sets: Dict[str, List[Dict[str, object]]],
) -> None:
    lines = [
        "# Round 2 Results Summary",
        "",
        "These outputs are generated from the current codebase. Treat smoke/local runs as preliminary until the server-scale configuration is run.",
        "",
        "## Run Settings",
        "",
        f"- preset: `{args.preset}`",
        f"- n_episodes: `{settings.n_episodes}`",
        f"- rr_observation_draws: `{settings.rr_observation_draws}`",
        f"- blinkered_observation_draws: `{settings.blinkered_observation_draws}`",
        f"- common_observation_streams: `{settings.use_common_observation_streams}`",
        f"- observations_per_person: `{settings.observations_per_person}`",
        f"- sections: `{args.sections}`",
        "",
        "## Generated Tables",
        "",
    ]
    for name, rows in sorted(result_sets.items()):
        lines.append(f"- `{name}`: {len(rows)} rows")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Expected average utility is the performance criterion.",
            "- Final-choice distances and information-acquisition metrics are diagnostics.",
            "- Do not over-interpret mean utility differences when confidence intervals overlap.",
            "- The one-dimensional sweeps are designed to identify candidate regimes for 50/50 splits and equal-outcome choices before experiment design.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = settings_from_args(args)
    sections = parse_sections(args.sections)
    output_dir = PROJECT_ROOT / args.output_dir
    ensure_dir(output_dir)

    environments = build_analysis_environments(args)
    result_sets: Dict[str, List[Dict[str, object]]] = {}

    if "step7" in sections:
        result_sets.update(run_step7_outputs(output_dir, environments, settings))
    if "sweeps" in sections:
        sweep_configs = build_sweep_configs_for_args(args)
        result_sets.update(run_sweep_outputs(output_dir, sweep_configs, settings))
    if "regimes" in sections and "sweeps" not in sections:
        sweep_configs = build_sweep_configs_for_args(args)
        sweep_results = run_sweep_outputs(output_dir, sweep_configs, settings)
        result_sets.update(
            {
                key: value
                for key, value in sweep_results.items()
                if "candidate" in key
            }
        )
    if "dp" in sections:
        result_sets.update(run_dp_outputs(output_dir, environments, settings, args))
    if "gh" in sections:
        result_sets.update(run_gauss_hermite_outputs(output_dir, environments, args.gauss_hermite_order))

    write_figures(output_dir, result_sets)
    write_summary(output_dir, args, settings, result_sets)
    print(f"Generated results in {output_dir}")


if __name__ == "__main__":
    main()
