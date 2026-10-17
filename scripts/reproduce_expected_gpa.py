#!/usr/bin/env python3
"""Expected-GPA targets, person-level inference and conditional power references."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.design_validation.bounded_task import load_design
from src.design_validation.budget_horizon import BudgetTask
from src.design_validation.terminal_time import analysis_templates, terminal_summary
from src.tutoring_analysis import conditional_power, nominal_inference
from src.tutoring_analysis.expected_outcomes import trial_row


def templates():
    design = load_design(ROOT / "configs/tutoring_conditions.json")
    timing = json.loads((ROOT / "configs/tutoring_terminal_time_v01.json").read_text())
    return {c.template_id: c for c in analysis_templates(design, timing)
            if c.condition_id != "C1" or c.gap_offset == 1.}


def convert_trials(records, configs):
    """Recompute targets from declared trial histories, never trust cached metrics."""
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty array of synthetic trials is required")
    rows = []
    for record in records:
        if record.get("humanObservations") is not False:
            raise ValueError("Only explicitly synthetic trial histories are supported")
        config = configs[record["templateId"]]
        reverse = record.get("sideReversed", False)
        if not isinstance(reverse, bool):
            raise ValueError("sideReversed must be boolean")
        if reverse:
            config = replace(config, prior_centres=config.prior_centres[::-1],
                             prior_spreads=config.prior_spreads[::-1])
        row = trial_row(config, record["paid"], record["shareIndex"],
                        participant_id=record["participantId"],
                        session_id=record.get("sessionId"), main_index=record["mainIndex"])
        for field in ("recordPresent", "endpointPresent"):
            value = record.get(field, True)
            if not isinstance(value, bool):
                raise ValueError(field + " must be boolean")
            row[field] = value
        rows.append(row)
    return rows


def target_panel(configs):
    records = []
    for key, config in sorted(configs.items()):
        task = BudgetTask(config)
        summary = terminal_summary(config, *task.initial_logs, config.total_millihours)
        for macro in ("ES", "EO_expected_GPA", "OPT_expected_GPA"):
            index = summary["macros"][macro]["index"]
            row = trial_row(config, [], index, participant_id=key, main_index=1)
            records.append({"templateId": key, "conditionId": config.condition_id,
                            "macro": macro, "metrics": row,
                            "terminal": summary["macros"][macro]})
    return {"schema": "expected-gpa-initial-targets-v1", "templates": len(configs),
            "records": records, "scope": "deterministic_initial_target_geometry",
            "policyEvaluation": False, "humanObservations": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("targets", "analyze", "power"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", action="append", help="Optional target-panel subset")
    parser.add_argument("--trials", type=Path, help="Synthetic raw-trial JSON array for analyze")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new")
    if (args.stage == "analyze") != (args.trials is not None):
        parser.error("Only analyze requires --trials")
    if args.template and args.stage != "targets":
        parser.error("--template applies only to targets")
    outputs = {}
    if args.stage == "power":
        data = conditional_power.report()
        outputs["conditional_power.md"] = conditional_power.markdown(data)
    else:
        configs = templates()
        if args.stage == "targets":
            if args.template:
                if len(set(args.template)) != len(args.template) or not set(args.template) <= configs.keys():
                    parser.error("Unknown or duplicate template")
                configs = {k: configs[k] for k in args.template}
            data = target_panel(configs)
        else:
            records = json.loads(args.trials.read_text())
            rows = convert_trials(records, configs)
            spec = json.loads((ROOT / "configs/expected_gpa_analysis.json").read_text())
            data = nominal_inference.analyze(rows, spec)
            outputs["metrics.json"] = json.dumps(rows, indent=2, allow_nan=False) + "\n"
    outputs["result.json"] = json.dumps(data, indent=2, allow_nan=False) + "\n"
    outputs["completion.json"] = json.dumps({"complete": True, "stage": args.stage,
        "formalCalibrationReproduced": False, "humanValidation": False}) + "\n"
    args.output.mkdir(parents=True)
    for name, body in outputs.items():
        (args.output / name).write_text(body, encoding="utf-8")
    print(json.dumps({"complete": True, "stage": args.stage, "output": str(args.output)}))


if __name__ == "__main__":
    main()
