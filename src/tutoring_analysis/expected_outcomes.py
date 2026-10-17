"""Finite expected-GPA allocation diagnostics, separate from utility maximization.

Distances use allocation percentage points on the native 1001-choice grid.
The expected-GPA target uses posterior beliefs, not hidden realized outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from ..design_validation.bounded_task import normalize_log_weights
from ..design_validation.budget_horizon import BudgetTask
from ..design_validation.terminal_time import terminal_summary

ROW_SCHEMA = "expected-outcome-metrics-v1"
PROFILE = "finite-budget-only-v1"
ANALYSIS_ID = "person-level-nominal-v1"

def digest(value):
    # Cross-language semantic SHA: JS has one number type; encode all JSON
    # numbers as binary64, including validated integer fields, before JSON.
    # On-disk file hashes remain SHA256 of the untouched bytes, separately.
    def numbers(v):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError("Nonfinite receipt number")
            if isinstance(v, int) and abs(v) > 2 ** 53 - 1:
                raise ValueError("Large integer receipt identities must be decimal strings")
            return {"number_binary64": struct.pack(">d", float(v)).hex()}
        if isinstance(v, dict):
            return {str(k): numbers(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [numbers(x) for x in v]
        raise TypeError("Non-JSON receipt value")
    return hashlib.sha256(json.dumps(numbers(value), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def binary_payload(value):
    """Unrounded binary64 identity; bool/int/string retain distinct JSON types."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite semantic float")
        return {"binary64": struct.pack(">d", value).hex()}
    if isinstance(value, dict):
        return {str(k): binary_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [binary_payload(v) for v in value]
    return value



def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def integer(value, label, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid integer " + label)
    return value



def allocation_metrics(row, j):
    budget = row["remainingIntegerBudget"]
    empty = dict.fromkeys(("share", "D_ES_pp", "D_EGPA_pp", "R_EGPA_pp", "D_surplus_pp", "R_surplus_pp", "L_lower_mean_share"))
    empty.update({"allToLowerMatch": None, "meetLowerFirstMatch": None, "lowerMeanTie": None})
    if budget == 0:
        if j is not None:
            raise ValueError("Zero-budget share must be null")
        return {**empty, "shareIndex": None, "choiceAvailable": False, "choiceReason": "zero_budget"}
    if j is None:
        return {**empty, "shareIndex": None, "choiceAvailable": False, "choiceReason": "uncommitted_source_state"}
    integer(j, "shareIndex", 0, 1000)
    eo, old = row["jEGPA"], row["jSurplus"]
    integer(eo, "expected-GPA target", 0, 1000)
    integer(old, "raw-surplus target", 0, 1000)
    ma, mb = row["terminalMeans"]
    tie = abs(ma - mb) <= 1e-9 * max(1., abs(ma), abs(mb))
    lower = 0. if tie else (j - 500) / 1000 * (1 if ma < mb else -1)
    return {"shareIndex": j, "share": j / 1000, "D_ES_pp": abs(j - 500) / 10,
            "D_EGPA_pp": abs(j - eo) / 10, "R_EGPA_pp": (abs(j - 500) - abs(j - eo)) / 10,
            "D_surplus_pp": abs(j - old) / 10, "R_surplus_pp": (abs(j - 500) - abs(j - old)) / 10,
            "L_lower_mean_share": lower, "lowerMeanTie": tie,
            "allToLowerMatch": None if row.get("jAL") is None else j == row["jAL"],
            "meetLowerFirstMatch": None if row.get("jML") is None else j == row["jML"],
            "choiceAvailable": True, "choiceReason": None}



def _logs(values):
    if not isinstance(values, (list, tuple)) or len(values) != 100 or not all(finite(v) for v in values):
        raise ValueError("Exactly 100 finite log weights required")
    return list(values)


def finite_mean(logs):
    logs = _logs(logs)
    peak = max(logs)
    weights = [math.exp(x - peak) for x in logs]
    return math.fsum((i + 1) * w for i, w in enumerate(weights)) / math.fsum(weights)



def trial_row(config, paid, share_index, *, participant_id, main_index, session_id=None):
    """Recompute a synthetic trial's beliefs and targets from its paid history."""
    if not isinstance(participant_id, str) or not participant_id:
        raise ValueError("Nonempty participant identity required")
    integer(main_index, "main index", 1, 10)
    session_id = participant_id if session_id is None else session_id
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Nonempty session identity required")
    if (config.allocation_denominator != 1000 or config.alpha != .5
            or config.loss_multiplier != 2. or config.gpa_baseline != 3.5
            or config.gpa_scale != .03
            or config.max_assessments != config.total_millihours // config.cost_millihours):
        raise ValueError("The declared budget-only finite GPA model is required")
    if config.condition_id == "C1" and not math.isclose(
            abs(config.prior_centres[1] - config.prior_centres[0]), 1., abs_tol=1e-12):
        raise ValueError("C1 uses the working one-minute total centre gap")
    task = BudgetTask(config)
    initial = task.initial_logs
    logs = [x.copy() for x in initial]
    history = []
    for event in paid:
        student = integer(event["student"], "student", 0, 1)
        report = integer(event["report"], "report", 1, 100)
        logs[student] = normalize_log_weights(logs[student] + task.log_likelihood[:, report - 1])
        history.append({"student": student, "report": report})
    b0, cost = config.total_millihours, config.cost_millihours
    if len(history) > b0 // cost:
        raise ValueError("Paid history exceeds the affordable horizon")
    budget = b0 - cost * len(history)
    target = terminal_summary(config, *logs, budget)
    means0 = [finite_mean(x.tolist()) for x in initial]
    means = [finite_mean(x.tolist()) for x in logs]
    row = {
        "metric_schema": ROW_SCHEMA, "rowValidated": True, "profileId": PROFILE,
        "analysisId": ANALYSIS_ID, "participantId": participant_id,
        "sessionId": session_id, "conditionId": config.condition_id,
        "mainIndex": main_index, "templateId": config.template_id,
        "scale": config.scale, "gapOffset": config.gap_offset,
        "initialIntegerBudget": b0, "paidIntegerCost": cost,
        "remainingIntegerBudget": budget, "paidCount": len(history),
        "paid": history, "paidHistoryValid": True, "Q": int(bool(history)),
        "initialMeans": means0, "terminalMeans": means,
        "rawInitialGap": means0[0] - means0[1],
        "x0": (means0[0] - means0[1]) / (b0 / 1000),
        "xT": None if budget == 0 else (means[0] - means[1]) / (budget / 1000),
        "jEGPA": target["macros"]["EO_expected_GPA"]["index"],
        "jSurplus": target["macros"]["EO_raw_surplus"]["index"],
        "jAL": target["macros"]["AL"]["index"],
        "jML": target["macros"]["ML"]["index"],
        "EO_tied_indices": target["EO_tied_indices"],
        "raw_EO_minimum_GPA_gap": target["raw_EO_minimum_GPA_gap"],
        "EGPA_selected_combined_GPA": target["macros"]["EO_expected_GPA"]["expected_combined_GPA"],
        "recordPresent": True, "endpointPresent": True,
        "humanObservations": False, "origin": "synthetic_design",
    }
    row.update(allocation_metrics(row, share_index))
    return row
