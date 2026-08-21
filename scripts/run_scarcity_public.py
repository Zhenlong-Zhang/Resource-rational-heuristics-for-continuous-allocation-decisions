"""Portable R6 scarcity runner.

This runner calls the shared scientific implementation directly. It has no scheduler,
cluster, provenance, or server-path layer. ``smoke`` is a wiring check; ``serious`` uses
the frozen public settings and can be run on an ordinary Python environment with enough
time and memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

# Allow direct execution from a clean repository checkout without requiring editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.scarcity import (
    SCARCITY_CONFIRMATION_EPISODES,
    SCARCITY_DEVELOPMENT_EPISODES,
    SCARCITY_ORACLE_GRID_SIZE,
    build_confirmation_descriptors,
    build_development_descriptors,
    build_gaussian_oracle_descriptors,
    evaluate_gaussian_oracle_descriptor,
    evaluate_metalevel_descriptor,
    object_level_stop_decision,
    select_confirmation_targets,
    select_gaussian_oracle_anchors,
    summarize_gaussian_oracle_rows,
    summarize_metalevel_rows,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _run_object(
    output_dir: Path,
    *,
    episodes: int,
    descriptor_limit: int | None,
    oracle_grid_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    descriptors = [dict(row) for row in build_gaussian_oracle_descriptors()]
    if descriptor_limit is not None:
        descriptors = descriptors[:descriptor_limit]
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        rows.extend(
            evaluate_gaussian_oracle_descriptor(
                descriptor,
                n_episodes=episodes,
                oracle_grid_size=oracle_grid_size,
            )
        )
    summaries = summarize_gaussian_oracle_rows(rows)
    anchors = select_gaussian_oracle_anchors(summaries)
    gate = object_level_stop_decision(summaries, anchors)
    _write_rows(output_dir / "scarcity_oracle_rows.csv", rows)
    _write_rows(output_dir / "scarcity_oracle_summary.csv", summaries)
    _write_json(output_dir / "scarcity_object_gate.json", gate)
    _write_json(output_dir / "scarcity_selected_anchors.json", anchors)
    return descriptors, anchors, gate


def _run_metalevel(
    output_dir: Path,
    *,
    stage: str,
    descriptors: Sequence[Mapping[str, object]],
    episodes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for descriptor in descriptors:
        rows.extend(
            evaluate_metalevel_descriptor(
                descriptor,
                stage=stage,
                n_episodes=episodes,
            )
        )
    summaries = summarize_metalevel_rows(rows)
    _write_rows(output_dir / f"scarcity_{stage}_episode_rows.csv", rows)
    _write_rows(output_dir / f"scarcity_{stage}_summary.csv", summaries)
    return summaries


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "serious":
        object_episodes = SCARCITY_DEVELOPMENT_EPISODES
        development_episodes = SCARCITY_DEVELOPMENT_EPISODES
        confirmation_episodes = SCARCITY_CONFIRMATION_EPISODES
        descriptor_limit = None
        development_limit = None
    else:
        object_episodes = args.smoke_episodes
        development_episodes = args.smoke_episodes
        confirmation_episodes = args.smoke_episodes
        descriptor_limit = args.smoke_object_descriptors
        development_limit = args.smoke_development_descriptors

    descriptors, anchors, gate = _run_object(
        output_dir,
        episodes=object_episodes,
        descriptor_limit=descriptor_limit,
        oracle_grid_size=args.oracle_grid_size,
    )
    if bool(gate.get("stop_metalevel")):
        _write_rows(output_dir / "scarcity_development_episode_rows.csv", [])
        _write_rows(output_dir / "scarcity_development_summary.csv", [])
        _write_rows(output_dir / "scarcity_confirmation_episode_rows.csv", [])
        _write_rows(output_dir / "scarcity_confirmation_summary.csv", [])
        _write_json(
            output_dir / "scarcity_confirmation_selection.json",
            {
                "schema": "confirmation_selection_v1",
                "metalevel_stopped": True,
                "targets": [],
                "contrasts": [],
            },
        )
        _write_json(
            output_dir / "scarcity_run_metadata.json",
            {"mode": args.mode, "metalevel_stopped": True, "object_gate": gate},
        )
        return

    development_descriptors = build_development_descriptors(anchors, descriptors)
    if development_limit is not None:
        development_descriptors = development_descriptors[:development_limit]
    development_summaries = _run_metalevel(
        output_dir,
        stage="development",
        descriptors=development_descriptors,
        episodes=development_episodes,
    )
    selection = select_confirmation_targets(development_summaries)
    _write_json(output_dir / "scarcity_confirmation_selection.json", selection)
    confirmation_descriptors = build_confirmation_descriptors(
        selection,
        development_descriptors,
    )
    confirmation_summaries = _run_metalevel(
        output_dir,
        stage="confirmation",
        descriptors=confirmation_descriptors,
        episodes=confirmation_episodes,
    )
    _write_json(
        output_dir / "scarcity_run_metadata.json",
        {
            "mode": args.mode,
            "metalevel_stopped": False,
            "object_gate": gate,
            "development_environment_count": len(development_descriptors),
            "confirmation_environment_count": len(confirmation_descriptors),
            "confirmation_summary_count": len(confirmation_summaries),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "serious"), default="smoke")
    parser.add_argument("--output-dir", default="results/scarcity_public")
    parser.add_argument("--oracle-grid-size", type=int, default=SCARCITY_ORACLE_GRID_SIZE)
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--smoke-object-descriptors", type=int, default=2)
    parser.add_argument("--smoke-development-descriptors", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
