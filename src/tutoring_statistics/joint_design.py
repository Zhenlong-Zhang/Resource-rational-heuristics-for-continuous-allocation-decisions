"""joint design: accepted numerical definitions, portable source projection."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from copy import deepcopy
from statistics import NormalDist
import copy
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from .finite_means import canonical_mean as frozen_mean

import numpy as np
import scipy
from scipy import stats, special, optimize
from scipy.stats import beta as beta_distribution, binom, nct, t as t_distribution
from scipy.optimize import brentq
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, stdtrit



PROFILE = "pilot-r9-design-v1"

FLOWS = (None, "r10-prepractice-v1")

CONDITIONS = ("C1", "C2", "C3", "C4", "C5")

FIELDS = (
    "participant", "session", "condition", "profileId", "flowVersion",
    "mainIndex", "template", "scale", "gap", "sideReversed",
    "meanAInitial", "meanBInitial", "meanATerminal", "meanBTerminal",
    "shareIndex", "share", "initialBudgetMs", "remainingBudgetMs",
)

SCOPE = {
    "candidateOnly": True,
    "scientificallyAdopted": False,
    "humanInferenceReleased": False,
    "population": "complete_ten_positive_defined_choices_equal_participants",
    "needTimepoint": "initial_candidate",
    "allocationUnit": "percentage_of_remaining_resource",
    "needUnit": "declared_source_resource_unit_not_display_rounding",
    "budgetFieldUnit": "integer_thousandths_of_source_resource_unit",
}

def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def _integer(value, minimum, maximum=None):
    return (isinstance(value, int) and not isinstance(value, bool) and value >= minimum
            and (maximum is None or value <= maximum))

def _budget_integer(value):
    if not _finite(value):
        raise ValueError("Nonfinite source budget")
    scaled = value * 1000
    rounded = round(scaled)
    if not math.isclose(scaled, rounded, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Source budget is not integer thousandths")
    return rounded

def adapt_normalized_records(normalized_records):
    """Read exact finite means from successful normalizer output, without optimization.

    This preserves source metadata but does not certify the origin of arbitrary
    JSON claiming to be normalized. The CLI must call the actual normalizer on
    raw records before invoking this adapter. No human inference is released.
    """
    from .finite_means import finite_mean

    if not isinstance(normalized_records, list):
        raise ValueError("A list of actual normalized records is required")
    rows, sources, recipes = [], [], set()
    seen_people, seen_sessions = set(), set()
    for normalized in normalized_records:
        if (not isinstance(normalized, dict) or normalized.get("schema") != "tutoring_normalized_v1"
                or normalized.get("errors") != [] or not isinstance(normalized.get("trials"), list)):
            raise ValueError("Only successful normalized records with no errors are accepted")
        meta = normalized.get("meta", {})
        required = ("participantId", "sessionId", "conditionId", "profileId",
                    "resolvedRecipeVersion", "resolvedRecipeFingerprint", "instructionVersion", "checkVersion")
        if any(not isinstance(meta.get(k), str) or not meta[k] for k in required):
            raise ValueError("Missing normalized scientific/teaching identity")
        if meta["participantId"] in seen_people or meta["sessionId"] in seen_sessions:
            raise ValueError("Duplicate participant/session source records cannot be merged by this candidate adapter")
        seen_people.add(meta["participantId"])
        seen_sessions.add(meta["sessionId"])
        flow = meta.get("participantFlowVersion")
        recipe = (meta["resolvedRecipeVersion"], meta["resolvedRecipeFingerprint"])
        recipes.add(recipe)
        main_count = sum(t["definition"]["stage"] == "main" for t in normalized["trials"])
        sources.append({**{k: meta[k] for k in required}, "flowVersion": flow,
                        "source": normalized.get("source"), "flags": list(normalized.get("flags", [])),
                        "reportedStatus": meta.get("status"), "reportedPhase": meta.get("phase"),
                        "normalizedMainRows": main_count,
                        "emptyHistoryReason": "no_normalized_main_trials" if main_count == 0 else None})
        for trial in normalized["trials"]:
            d, outcome = trial["definition"], trial["result"]
            if d["stage"] != "main":
                continue
            if any(trial.get(k) != meta[k] for k in ("participantId", "sessionId", "conditionId", "profileId")):
                raise ValueError("Normalized trial/record identity mismatch")
            initial, terminal = trial["initialLogs"], trial["terminalLogs"]
            if len(initial) != 2 or len(terminal) != 2:
                raise ValueError("Two ordered finite recipient beliefs are required")
            m0, mt = [finite_mean(logs) for logs in initial], [finite_mean(logs) for logs in terminal]
            remaining, j = outcome["remainingMillihours"], outcome["shareIndex"]
            rows.append({
                "participant": meta["participantId"], "session": meta["sessionId"],
                "condition": meta["conditionId"], "profileId": meta["profileId"], "flowVersion": flow,
                "mainIndex": d["stageIndex"], "template": d["templateId"], "scale": d["scale"],
                "gap": d["gap"], "sideReversed": d["sideReversed"],
                "meanAInitial": m0[0], "meanBInitial": m0[1],
                "meanATerminal": mt[0], "meanBTerminal": mt[1],
                "shareIndex": j, "share": None if j is None else j / 1000,
                "initialBudgetMs": d["budgetMillihours"], "remainingBudgetMs": remaining,
            })
    if len(recipes) > 1:
        raise ValueError("Mixed normalized scientific recipes cannot be pooled")
    # Check metadata even for failed/incomplete sessions with no main rows.
    if any(source["profileId"] != PROFILE or source["flowVersion"] not in FLOWS for source in sources):
        raise ValueError("Unsupported normalized scientific profile/flow")
    if len({(s["flowVersion"], s["instructionVersion"], s["checkVersion"]) for s in sources}) > 1:
        raise ValueError("Mixed normalized teaching versions cannot be pooled")
    payload = {"rows": rows, "sourceRecords": sources,
               "metadata": {"scope": dict(SCOPE), "inputRoute": "actual_normalizer_exact_finite_beliefs",
                            "originVerifiedByAdapter": False}}
    _validated_rows(payload)
    return payload

def _validated_rows(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Candidate input must be an object containing rows")
    rows = []
    for original in payload["rows"]:
        if not isinstance(original, dict) or any(key not in original for key in FIELDS):
            raise ValueError("Missing canonical row fields")
        row = dict(original)
        if any(not isinstance(row[key], str) or not row[key]
               for key in ("participant", "session", "template")):
            raise ValueError("Nonempty participant/session/template required")
        if row["condition"] not in CONDITIONS or row["profileId"] != PROFILE:
            raise ValueError("Unsupported condition or scientific profile")
        if row["flowVersion"] not in FLOWS:
            raise ValueError("Unsupported flow version")
        if not _integer(row["mainIndex"], 1, 10):
            raise ValueError("mainIndex must be an integer in 1..10")
        if not isinstance(row["sideReversed"], bool):
            raise ValueError("sideReversed must be boolean")
        if not _finite(row["scale"]) or row["scale"] <= 0 or not _finite(row["gap"]):
            raise ValueError("Invalid recipe scale/gap")
        for key in ("meanAInitial", "meanBInitial", "meanATerminal", "meanBTerminal"):
            if not _finite(row[key]) or not 1 <= row[key] <= 100:
                raise ValueError("Finite need means must be in source support [1,100]")
        b0, bt = row["initialBudgetMs"], row["remainingBudgetMs"]
        if not _integer(b0, 1) or not _integer(bt, 0, b0):
            raise ValueError("Invalid integer source budgets")
        j, share = row["shareIndex"], row["share"]
        if j is None or share is None:
            if j is not None or share is not None:
                raise ValueError("share and shareIndex must be jointly missing")
        elif (bt == 0 or not _integer(j, 0, 1000) or not _finite(share)
              or not math.isclose(share, j / 1000, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Inconsistent share/grid or nonpositive budget")
        reasons = row.get("sourceExclusionReasons", [])
        if not isinstance(reasons, list) or any(not isinstance(s, str) or not s for s in reasons):
            raise ValueError("Invalid source exclusion reasons")
        rows.append(row)
    if len({row["flowVersion"] for row in rows}) > 1:
        raise ValueError("Mixed teaching-flow versions cannot be pooled")
    return rows

def lower_need_scores(row):
    """Unrounded signed L and its percentage representation for one valid row."""
    if row["share"] is None or row["remainingBudgetMs"] <= 0:
        return {"centeredShare": None, "percent": None, "tie": None,
                "reason": "undefined_positive_budget_choice"}
    a, b = row["meanATerminal"], row["meanBTerminal"]
    tie = abs(a - b) <= 1e-9 * max(1., abs(a), abs(b))
    # Match the accepted grid arithmetic, rather than subtracting .5 after a
    # separately rounded share conversion.
    centered = 0. if tie else (row["shareIndex"] - 500) / 1000 * (1 if a < b else -1)
    return {"centeredShare": centered, "percent": 50. + 100. * centered,
            "tie": tie, "reason": None}

def prepare_population(payload):
    """Validate rows and quarantine conflicting/incomplete people, never merge them."""
    rows = _validated_rows(payload)
    groups, session_people = defaultdict(list), defaultdict(set)
    for row in rows:
        groups[row["participant"]].append(row)
        session_people[row["session"]].add(row["participant"])
    participants, eligible_rows, history_rows = [], [], []
    for participant, own in sorted(groups.items()):
        own = sorted(own, key=lambda row: row["mainIndex"])
        reasons = set()
        sessions, conditions = {r["session"] for r in own}, {r["condition"] for r in own}
        if len(sessions) != 1:
            reasons.add("participant_multiple_sessions")
        if len(conditions) != 1:
            reasons.add("participant_multiple_conditions")
        if any(len(session_people[s]) != 1 for s in sessions):
            reasons.add("session_multiple_participants")
        indices = [r["mainIndex"] for r in own]
        if len(set(indices)) != len(indices):
            reasons.add("duplicate_main_index")
        if indices != list(range(1, 11)):
            reasons.add("incomplete_ten_main_history")
        for row in own:
            reasons.update(row.get("sourceExclusionReasons", []))
        history_ok = not reasons
        if history_ok:
            history_rows.extend(own)
        if any(r["remainingBudgetMs"] <= 0 or r["share"] is None for r in own):
            reasons.add("incomplete_ten_positive_defined_choices")
        eligible = not reasons
        scores = [lower_need_scores(r) for r in own] if eligible else []
        record = {
            "participant": participant, "session": next(iter(sessions)) if len(sessions) == 1 else None,
            "condition": next(iter(conditions)) if len(conditions) == 1 else None,
            "eligible": eligible, "completeHistory": history_ok, "reasons": sorted(reasons),
            "observedRows": len(own), "meanLowerPercent": None, "meanCenteredLowerShare": None,
            "directPercentMeanDiagnostic": None,
            "meanOperation": "frozen_ordered_float64_L_mean_then_percent_display",
            "tieTrials": sum(s["tie"] for s in scores) if eligible else None,
        }
        if eligible:
            # Preserve the historical estimator, including its operation order.
            # Mean percent may round to exactly 50 even when this centered mean
            # is nonzero. Never recover the inferential value by subtracting 50.
            centered = float(frozen_mean([s["centeredShare"] for s in scores]))
            record["meanCenteredLowerShare"] = centered
            record["meanLowerPercent"] = 50. + 100. * centered
            record["directPercentMeanDiagnostic"] = float(frozen_mean([s["percent"] for s in scores]))
            eligible_rows.extend(own)
        participants.append(record)
    return {"rows": rows, "participants": participants, "eligibleRows": eligible_rows,
            "completeHistoryRows": history_rows,
            "eligibleParticipants": [p for p in participants if p["eligible"]],
            "scope": dict(SCOPE)}
