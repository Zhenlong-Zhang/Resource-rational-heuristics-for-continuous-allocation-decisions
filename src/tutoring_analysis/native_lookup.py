"""Source-derived finite lookup and independent numerical reconstruction.

Synthetic controls reuse ancestor reports; they are not selectable engine recipes.
Content validation replaces private execution approvals, not scientific checks.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import subprocess

from . import nominal_inference as inference
from . import native_reconstruction as I
from .expected_outcomes import (ROW_SCHEMA, PROFILE, ANALYSIS_ID, allocation_metrics,
                                binary_payload, digest, finite, integer, finite_mean, _logs)
from .native_calibration import ARMS, validate_calibration
from ..design_validation.bounded_task import TaskConfig
from ..design_validation.terminal_time import terminal_summary

SOURCE_SCHEMA = "native-source-states-v1"
LOOKUP_SCHEMA = "native-calibration-lookup-v1"
SOURCE_FILES = ("native_source.cjs", "native_engine.cjs", "native_normalizer.cjs")


def _target_content():
    root = Path(__file__).resolve().parents[1]
    files = ("tutoring_analysis/expected_outcomes.py", "design_validation/bounded_task.py",
             "design_validation/budget_horizon.py", "design_validation/terminal_time.py")
    return {"schema": "native-target-content-v1",
            "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}


def _config(definition, identity):
    common = identity["common"]
    utility, gpa = common["utility"], common["gpa"]
    if (utility["gain_exponent"] != .5 or utility["loss_exponent"] != .5
            or utility["loss_multiplier"] != 2 or gpa["baseline"] != 3.5
            or gpa["utility_scale"] != .03 or gpa.get("clipping")
            or common["allocation"]["share_grid_denominator"] != 1000):
        raise ValueError("Unsupported payoff semantics")
    return TaskConfig(condition_id=identity["meta"]["conditionId"], template_id=definition["templateId"],
                      scale=definition["scale"], gap_offset=definition["gap"],
                      prior_centres=tuple(definition["priorCentres"]), prior_spreads=tuple(definition["priorSpreads"]),
                      report_width=definition["reportWidth"], total_millihours=definition["budgetMillihours"],
                      cost_millihours=definition["costMillihours"], repeat_count_per_30=0,
                      max_assessments=definition["budgetMillihours"] // definition["costMillihours"],
                      allocation_denominator=1000, alpha=.5, loss_multiplier=2., gpa_baseline=3.5, gpa_scale=.03)


def _source_meta(arm, variant):
    return {"conditionId": arm, "scheduleVariant": variant,
            "participantId": "synthetic-lookup-" + arm + "-v" + str(variant),
            "sessionId": "synthetic-source-" + arm + "-v" + str(variant),
            "seed": "r9-calibration-lookup-v1:" + arm + ":variant:" + str(variant)}


def _prefix(definition, main, count):
    b0, cost = definition["budgetMillihours"], definition["costMillihours"]
    if count in (0, 1, 3):
        return {0: [], 1: [0], 3: [0, 1, 0]}[count]
    if main != 1 or b0 % cost or count != b0 // cost:
        raise ValueError("Invalid exact-exhaustion branch")
    return [i % 2 for i in range(count)]


def _paid(definition, prefix, streams):
    counters, paid = [0, 0], []
    for student in prefix:
        paid.append({"student": student, "report": streams[student][counters[student]]})
        counters[student] += 1
    return paid


def _expected_indices(schedules, controls):
    keys = set()
    for (arm, variant), definitions in schedules.items():
        for main in range(1, 11):
            d = definitions[main + 3]
            prefixes = [0, 1, 3]
            if main == 1 and d["budgetMillihours"] % d["costMillihours"] == 0:
                prefixes.append(d["budgetMillihours"] // d["costMillihours"])
            for population in ["working"] + (["gap_" + str(g) for g in controls] if arm == "C1" else []):
                keys.update((population, arm, variant, main, p) for p in prefixes)
    return keys


def validate_source(source, analysis, calibration):
    """Replay every transition witness and independently reconstruct finite inputs."""
    inference.validate_spec(analysis)
    validate_calibration(calibration)
    if (source.get("schema") != SOURCE_SCHEMA or source.get("humanObservations") is not False
            or source.get("syntheticOnly") is not True
            or source.get("analysisSpecHash") != digest(analysis)
            or source.get("calibrationSpecHash") != digest(calibration)
            or source.get("designHash") != digest(source["design"])
            or source.get("sourceReceiptHash") != digest(source["sourceReceipt"])):
        raise ValueError("Source input/content binding mismatch")
    folder = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    content = source["sourceReceipt"]
    if (source.get("sourceFiles") != files or content.get("files") != files
            or content.get("schema") != "native-source-content-v1"
            or content.get("designHash") != source["designHash"]
            or content.get("common") != source["design"]["common"]):
        raise ValueError("Source bytes/design changed")
    # Reconstruct full gate and trial histories rather than trusting a status flag.
    process = subprocess.run(["node", str(folder / "native_source.cjs"), "--verify-stdin"],
                             input=json.dumps({"source": source, "analysis": analysis, "calibration": calibration}),
                             text=True, capture_output=True, check=False)
    if process.returncode:
        raise ValueError("Native source reconstruction failed: " + process.stderr[-2000:])
    check = json.loads(process.stdout)
    if check.get("sourceStatesHash") != digest(source):
        raise ValueError("Reconstructed source content identity differs")
    replay = _Replay()
    schedules = {(a, v): I.schedule(_source_meta(a, v)) for a in ARMS for v in range(9)}
    streams, seen = {}, set()
    for state in source["states"]:
        arm, variant, main, count = (state[k] for k in ("conditionId", "scheduleVariant", "mainIndex", "prefixLength"))
        integer(variant, "variant", 0, 8)
        integer(main, "main index", 1, 10)
        integer(count, "prefix")
        ix = ("working", arm, variant, main, count)
        replay.require(ix not in seen, "Duplicate source state")
        seen.add(ix)
        d = schedules[(arm, variant)][main + 3]
        replay.near(state["definition"], d, "Independent source schedule")
        key = (arm, variant, main)
        if key not in streams:
            streams[key] = I.scenario(_source_meta(arm, variant), d)[1]
        paid = _paid(d, _prefix(d, main, count), streams[key])
        replay.near(state["paid"], paid, "Independent fixed report stream")
        initial, terminal = replay.logs(d, paid)
        replay.near(state["initialLogs"], initial, "Independent finite prior")
        replay.near(state["terminalLogs"], terminal, "Independent finite posterior")
        replay.near(state, {"seed": _source_meta(arm, variant)["seed"],
                           "remainingMillihours": d["budgetMillihours"] - count * d["costMillihours"],
                           "logicalToDisplayed": ["B", "A"] if d["sideReversed"] else ["A", "B"],
                           "enginePhase": "allocation" if count > 3 else "trial",
                           "prefixKind": "exact_exhaustion" if count > 3 else "fixed"}, "Source accounting")
    expected = {ix for ix in _expected_indices(schedules, []) if ix[0] == "working"}
    replay.require(seen == expected, "Incomplete/extra source roster")
    return {"states": len(seen), "checks": replay.checks, "maximumNumericalError": replay.maximum_error}


def project(value):
    target = min(1., max(0., value)) * 1000
    lower, upper = math.floor(target), math.ceil(target)
    best = min(abs(lower - target), abs(upper - target))
    return min((j for j in {lower, upper} if abs(j - target) <= best + 1e-12),
               key=lambda j: (abs(j - 500), j))


class _Replay:
    def __init__(self):
        self.checks, self.maximum_error = 0, 0.
        self.sources = {}
        self.tables = {}
        self.replayed_states = self.control_states = self.exhausted_states = 0
        self.baseline_trials = 0

    def require(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError(label)

    def near(self, actual, expected, label, tolerance=1e-10):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(expected) <= set(actual), label + ":fields")
            for key, value in expected.items():
                self.near(actual[key], value, label + ":" + str(key), tolerance)
        elif isinstance(expected, (list, tuple)):
            self.require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ":length")
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.near(a, b, label + ":" + str(i), tolerance)
        elif isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
            self.require(type(actual) is type(expected) and actual == expected, label + ":exact")
        else:
            self.require(not isinstance(actual, bool) and isinstance(actual, (int, float)) and math.isfinite(actual), label + ":finite")
            error = abs(actual - expected)
            self.maximum_error = max(self.maximum_error, error)
            self.require(error <= tolerance, label + ":numerical")

    def logs(self, definition, paid):
        order = [1, 0] if definition["sideReversed"] else [0, 1]
        initial = [I.prior(definition["priorCentres"][j], definition["priorSpreads"][j]) for j in order]
        terminal = copy.deepcopy(initial)
        for event in paid:
            self.require(type(event["student"]) is int and event["student"] in (0, 1) and
                         type(event["report"]) is int and 1 <= event["report"] <= 100, "Finite ordered paid event")
            j = event["student"]
            terminal[j] = I.update(terminal[j], event["report"], definition["reportWidth"])
        return initial, terminal

    def targets(self, logs, budget):
        import numpy as np
        if budget not in self.tables:
            # Integer need subtraction BEFORE conversion/division is mandatory.
            numerator = np.arange(1001, dtype=np.int64)[:, None] * budget - np.arange(1, 101, dtype=np.int64)[None, :] * 1000000
            surplus = numerator / 1000000
            self.tables[budget] = np.where(surplus >= 0, np.sqrt(np.maximum(surplus, 0)), -2 * np.sqrt(np.maximum(-surplus, 0)))
            if len(self.tables) > 64:
                del self.tables[next(iter(self.tables))]
        table = self.tables[budget]
        probs = [np.asarray(I.probabilities(logs[j]), dtype=float) for j in (0, 1)]
        ua, ub = table @ probs[0], table[::-1] @ probs[1]
        ga, gb = 3.5 + .03 * ua, 3.5 + .03 * ub
        total, gap = ga + gb, np.abs(ga - gb)
        minimum = float(gap.min())
        primary = np.flatnonzero(gap <= minimum + 1e-12 * max(1., abs(minimum)))
        maximum = float(total[primary].max())
        ties = primary[total[primary] >= maximum - 1e-12 * max(1., abs(maximum))].tolist()
        selected = min(ties, key=lambda j: (abs(j - 500), j))
        means = [I.mean(p) for p in probs]
        tied_means = abs(means[0] - means[1]) <= 1e-9 * max(1., abs(means[0]), abs(means[1]))
        B = budget / 1000
        old = None if not budget else project((B + means[0] - means[1]) / (2 * B))
        lower = None if not budget else 500 if tied_means else 1000 if means[0] < means[1] else 0
        meet = None if not budget else 500 if tied_means else project(min(B, means[0]) / B if means[0] < means[1] else 1 - min(B, means[1]) / B)
        return {"jEGPA": None if not budget else selected, "jSurplus": old, "jAL": lower, "jML": meet,
                "DeltaNumerator": None if not budget else abs(selected - 500), "EO_tied_indices": ties,
                "raw_EO_minimum_GPA_gap": minimum, "EO_selected_gap_excess": max(0., float(gap[selected]) - minimum),
                "EGPA_selected_combined_GPA": float(total[selected])}, means


def _normalized(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    maximum = values.max()
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def _independent_logs(d, paid):
    """Direct finite likelihood construction; not engine/target update helper."""
    import numpy as np
    n = np.arange(1., 101.)
    order = [1, 0] if d["sideReversed"] else [0, 1]
    initial = [_normalized(-.5 * ((n - d["priorCentres"][i]) / d["priorSpreads"][i]) ** 2) for i in order]
    raw = -.5 * ((n[:, None] - n[None, :]) / d["reportWidth"]) ** 2
    # Row=latent need, column=report; normalization is latent-dependent.
    peak = raw.max(axis=1, keepdims=True)
    likelihood = raw - (peak + np.log(np.exp(raw - peak).sum(axis=1, keepdims=True)))
    logs = [x.copy() for x in initial]
    for e in paid:
        logs[e["student"]] = _normalized(logs[e["student"]] + likelihood[:, e["report"] - 1])
    return initial, logs


def _independent_utility_table(budget):
    """Native integer surplus numerator BEFORE division, including exact kinks."""
    import numpy as np
    needs = np.arange(1, 101, dtype=np.int64)
    surplus_numerator = np.arange(1001, dtype=np.int64)[:, None] * budget - needs[None, :] * 1000000
    surplus = surplus_numerator.astype(float) / 1000000
    return np.where(surplus >= 0, np.maximum(surplus, 0) ** .5, -2 * np.maximum(-surplus, 0) ** .5)


def _independent_target(logs, budget):
    """Independent native-int allocation enumeration, without target helper."""
    import numpy as np
    if budget == 0:
        return None, [], None
    values = _independent_utility_table(budget)
    probs = [np.exp(_normalized(p)) for p in logs]
    ga, gb = 3.5 + .03 * (values @ probs[0]), 3.5 + .03 * (values[::-1] @ probs[1])
    gap, total = abs(ga - gb), ga + gb
    minimum = float(gap.min())
    primary = np.flatnonzero(gap <= minimum + 1e-12 * max(1., abs(minimum)))
    best = float(total[primary].max())
    ties = primary[total[primary] >= best - 1e-12 * max(1., abs(best))]
    selected = int(min(ties, key=lambda j: (abs(int(j) - 500), int(j))))
    return selected, ties.tolist(), minimum


def _synthetic_control(state, gap):
    """Independent finite fixture, NOT a selectable/acknowledged engine recipe."""
    result = copy.deepcopy(state)
    d = result["definition"]
    d["priorCentres"] = [50 * d["scale"] - gap / 2, 50 * d["scale"] + gap / 2]
    d["c1TotalCentreGapMinutes"] = gap
    recipe = "synthetic-C1-total-gap-" + str(gap)
    result["synthetic_recipe_id"] = recipe
    initial, terminal = _independent_logs(d, result["paid"])
    result["initialLogs"], result["terminalLogs"] = [x.tolist() for x in initial], [x.tolist() for x in terminal]
    result["engineReplayClaim"] = False
    result["reportStreamScope"] = "fixed_ancestor_reports;not_redrawn_under_control_prior"
    result["semanticFingerprint"] = digest(binary_payload({"recipe": recipe, "definition": d, "paid": result["paid"],
        "initialLogs": result["initialLogs"], "terminalLogs": result["terminalLogs"], "budget": result["remainingMillihours"]}))
    return result


def _person_slope_expectation(states, moments, frozen_plan):
    """Finite 2^10 expectation with analytic latent moments, before response draws.

    Retains the actual binary64 production projection, rather than snapping its
    expectation to the ideal algebraic zero. This numerical candidate still
    requires independent exact derivation; it is not a machine certificate.
    """
    import numpy as np
    x = np.asarray([s["metrics"]["x0"] for s in states])
    dummy = [{**s["metrics"], "share": .5} for s in states]
    support = inference._slope(dummy, "x0")
    if support["status"] != "available":
        return None
    weighted, probabilities = [], []
    plan = [struct.unpack(">d", bytes.fromhex(bits))[0] for bits in frozen_plan["weightBinary64"]]
    for pattern in itertools.product((0, 1), repeat=10):
        polynomial = np.array([1.])
        for y, xi in zip(pattern, x):
            polynomial = np.convolve(polynomial, np.array([.5, xi if y else -xi]))
        probability = math.fsum(float(c) * moments[k] for k, c in enumerate(polynomial))
        # No clipping of negative probabilities or draw-based normalization.
        if probability < -1e-14:
            raise ValueError("Analytic pattern probability is negative")
        value = float(inference.canonical_projection(plan, pattern))
        weighted.append(probability * value)
        probabilities.append(probability)
    mass = math.fsum(probabilities)
    if abs(mass - 1.) > 1e-10:
        raise ValueError("Analytic pattern probability mass failed")
    return {"nativeFloatingExpectation": math.fsum(weighted), "probabilityMass": mass,
            "idealAlgebraExpectation": moments[1], "patterns": 1024, "machineCertified": False}


def _native_R_expectation(schedules, law):
    """Independent 2^10 paired-metric expectation under an analytic p law."""
    from scipy.special import betaln
    if law.startswith("symmetric_beta_"):
        rho = float(law.rsplit("_", 1)[1]); shape = .5 * (1 / rho - 1)
        # Exact same floating mass for complementary patterns, justified by
        # the symmetric analytic law, not by recentering observed responses.
        masses = [math.exp(float(betaln(shape + min(k, 10 - k), shape + max(k, 10 - k)) - betaln(shape, shape))) for k in range(11)]
    else:
        if law == "selected_uniform":
            moments = [(.9 ** (k + 1) - .7 ** (k + 1)) / ((k + 1) * (.9 - .7)) for k in range(11)]
        elif law == "centered_beta":
            base, scale = .5 - .25 * (.5 / 5.5), .25
            moments = [math.fsum(math.comb(k, j) * base ** (k - j) * scale ** j
                       * math.prod((.5 + q) / (5.5 + q) for q in range(j)) for j in range(k + 1)) for k in range(11)]
        else:
            raise ValueError("Unknown fixed R truth law")
        masses = [math.fsum((-1) ** j * math.comb(10 - k, j) * moments[k + j] for j in range(11 - k)) for k in range(11)]
    if any(p < 0 for p in masses) or abs(math.fsum(math.comb(10, k) * masses[k] for k in range(11)) - 1.) > 1e-10:
        raise ValueError("Independent R pattern probability law failed")
    variant_expectations = []
    for schedule in schedules:
        delta = [s["metrics"]["DeltaNumerator"] / 10 for s in schedule]
        terms = []
        for pattern in itertools.product((0, 1), repeat=10):
            scalar_person = float(inference.canonical_mean([d if y else -d for d, y in zip(delta, pattern)]))
            terms.append(masses[sum(pattern)] * scalar_person)
        variant_expectations.append(math.fsum(terms))
    return {"nativeFloatingExpectation": math.fsum(variant_expectations) / len(variant_expectations),
            "perVariant": variant_expectations, "law": law, "status": "floating_candidate_not_exact_truth", "machineCertified": False}



def _trial_metrics(trial, identity, target_hash):
    if identity.get("valid") is not True:
        raise ValueError("Unvalidated identity")
    d = trial["definition"]
    if d.get("stage") != "main":
        raise ValueError("Practice cannot enter candidate metrics")
    integer(d["stageIndex"], "main index", 1, 10)
    b0 = integer(d["budgetMillihours"], "initial budget", 1)
    cost = integer(d["costMillihours"], "paid cost", 1)
    paid = trial.get("paid")
    if paid is None:
        paid = [{"student": e["student"], "report": e["report"]} for e in trial["events"] if e["type"] == "assessment"]
    for e in paid:
        integer(e["student"], "paid recipient", 0, 1)
        integer(e["report"], "paid report", 1, 100)
    if len(paid) > b0 // cost:
        raise ValueError("Paid history exceeds full budget horizon")
    result = trial["result"]
    bt = integer(result["remainingMillihours"], "remaining budget")
    if bt != b0 - cost * len(paid):
        raise ValueError("Integer paid-budget identity failed")
    initial = [_logs(v) for v in trial["initialLogs"]]
    terminal = [_logs(v) for v in trial["terminalLogs"]]
    if len(initial) != 2 or len(terminal) != 2:
        raise ValueError("Two ordered recipients required")
    m0, mt = [finite_mean(v) for v in initial], [finite_mean(v) for v in terminal]
    summary = terminal_summary(_config(d, identity), *terminal, bt)
    eo = summary["macros"]["EO_expected_GPA"]["index"]
    old = summary["macros"]["EO_raw_surplus"]["index"]
    meta = identity["meta"]
    row = {"metric_schema": ROW_SCHEMA, "analysisId": identity["analysisId"],
           "analysisSpecHash": identity["analysisSpecHash"], "sourceReceiptHash": identity["sourceReceiptHash"],
           "targetReceiptHash": target_hash, "profileId": PROFILE,
           "humanObservations": False, "origin": "synthetic_validation",
           "resolvedRecipeFingerprint": meta.get("resolvedRecipeFingerprint"), "guideFixtureHash": meta.get("guideFixtureHash"),
           "designFingerprint": meta.get("designFingerprint"), "semanticFingerprint": trial.get("semanticFingerprint"),
           "participantId": meta["participantId"], "sessionId": meta["sessionId"], "conditionId": meta["conditionId"],
           "mainIndex": d["stageIndex"], "templateId": d["templateId"], "scale": d["scale"], "gapOffset": d["gap"],
           "sideReversed": d["sideReversed"], "gapConvention": d.get("gapConvention"),
           "c1TotalCentreGapMinutes": d.get("c1TotalCentreGapMinutes"), "initialIntegerBudget": b0, "paidIntegerCost": cost,
           "remainingIntegerBudget": bt, "paidCount": len(paid), "paid": copy.deepcopy(paid),
           "Q": int(bool(paid)), "paidHistoryValid": True, "initialMeans": m0, "terminalMeans": mt,
           "rawInitialGap": m0[0] - m0[1], "x0": (m0[0] - m0[1]) / (b0 / 1000),
           "xT": None if bt == 0 else (mt[0] - mt[1]) / (bt / 1000), "jEGPA": eo, "jSurplus": old,
           "jAL": summary["macros"]["AL"]["index"], "jML": summary["macros"]["ML"]["index"],
           "DeltaNumerator": None if eo is None else abs(eo - 500),
           "EO_tied_indices": summary["EO_tied_indices"], "raw_EO_minimum_GPA_gap": summary["raw_EO_minimum_GPA_gap"],
           "EO_selected_gap_excess": summary["EO_selected_gap_excess"],
           "EGPA_selected_combined_GPA": summary["macros"]["EO_expected_GPA"]["expected_combined_GPA"],
           "rowValidated": True, "recordPresent": True, "endpointPresent": True}
    row.update(allocation_metrics(row, result.get("shareIndex")))
    return row


def build_lookup(source_states, analysis_spec, calibration_spec):
    import numpy as np
    inference.validate_spec(analysis_spec)
    validate_calibration(calibration_spec)
    validate_source(source_states, analysis_spec, calibration_spec)
    source_hash = digest(source_states)
    target_api_receipt = _target_content()
    states, index = {}, {}
    for source in source_states["states"]:
        fixtures = [("working", source)]
        if source["conditionId"] == "C1":
            fixtures += [("gap_" + str(g), _synthetic_control(source, g)) for g in calibration_spec["C1SyntheticGapControls"]]
        for population, state in fixtures:
            d, paid = state["definition"], state["paid"]
            if d.get("gapConvention") != source_states["sourceReceipt"]["identity"]["resolvedRecipe"]["gapConvention"]:
                raise ValueError("Unbound definition gap convention")
            initial, terminal = _independent_logs(d, paid)
            log_error = max(float(np.max(abs(np.asarray(a) - b))) for a, b in zip(state["initialLogs"] + state["terminalLogs"], initial + terminal))
            if log_error > 1e-10:
                raise ValueError("Independent finite source-log replay failed")
            meta = {**state["meta"], "conditionId": state["conditionId"],
                    "participantId": "source-lookup", "sessionId": "source-lookup"}
            identity = {"valid": True, "meta": meta, "common": state["common"], "analysisId": analysis_spec["analysisId"],
                        "analysisSpecHash": digest(analysis_spec), "sourceReceiptHash": source_states["sourceReceiptHash"]}
            trial = {"definition": d, "initialLogs": state["initialLogs"], "terminalLogs": state["terminalLogs"],
                     "paid": paid, "result": {"remainingMillihours": state["remainingMillihours"], "shareIndex": None},
                     "semanticFingerprint": state["semanticFingerprint"]}
            metrics = _trial_metrics(trial, identity, digest(target_api_receipt))
            metrics.update({"syntheticRecipeId": state.get("synthetic_recipe_id"),
                            "syntheticRecipeHash": None if population == "working" else digest(binary_payload(d)),
                            "nativeProfileClaim": population == "working",
                            "reportStreamScope": state.get("reportStreamScope", "fixed_engine_stream")})
            independent_j, independent_ties, independent_min = _independent_target(terminal, state["remainingMillihours"])
            if metrics["jEGPA"] != independent_j or (independent_j is not None and metrics["EO_tied_indices"] != independent_ties):
                raise ValueError("Independent target selection/tie class mismatch")
            if independent_min is not None and abs(metrics["raw_EO_minimum_GPA_gap"] - independent_min) > 1e-10:
                raise ValueError("Independent raw target minimum mismatch")
            semantic = {"lookupSchema": "native-calibration-lookup-v1", "sourceStatesHash": source_hash,
                        "sourceReceiptHash": source_states["sourceReceiptHash"], "targetReceiptHash": digest(target_api_receipt),
                        "syntheticRecipe": state.get("synthetic_recipe_id"), "population": population,
                        "conditionId": state["conditionId"], "scheduleVariant": state["scheduleVariant"],
                        "definition": d, "logicalToDisplayed": state["logicalToDisplayed"], "paid": paid,
                        "remainingIntegerBudget": state["remainingMillihours"], "initialLogs": state["initialLogs"], "terminalLogs": state["terminalLogs"]}
            key = digest(binary_payload(semantic))
            ix = (population, state["conditionId"], state["scheduleVariant"], state["mainIndex"], state["prefixLength"])
            if ix in index or key in states:
                raise ValueError("Duplicate state identity; no implicit merge allowed")
            states[key] = {"metrics": metrics, "semantic": semantic, "binary64Payload": binary_payload(semantic),
                           "independentLogMaximumError": log_error, "sourceSnapshotSHA256": state["stateSnapshotSHA256"],
                           "engineReplayClaim": population == "working"}
            index[ix] = key
    required = [("working", c, v, i, p) for c in ARMS for v in range(9) for i in range(1, 11) for p in (0, 1, 3)]
    if any(ix not in index for ix in required):
        raise ValueError("Partial lookup is not released")
    lookup_index = {"|".join(map(str, k)): v for k, v in index.items()}
    geometry, slope_candidates, R_candidates = {}, {}, {}
    global_M = max(abs(states[index[("working", arm, v, i, (0, 1, 3)[(i - 1) % 3])]]["metrics"]["x0"])
                   for arm in ARMS for v in range(9) for i in range(1, 11))
    if global_M <= 0:
        raise ValueError("Frozen cross-arm slope-generator M must be positive")
    for population in ("working", "gap_0", "gap_0.2", "gap_0.5", "gap_1"):
        for arm in ARMS if population == "working" else ("C1",):
            schedules = [[states[index[(population, arm, v, i, (0, 1, 3)[(i - 1) % 3])]] for i in range(1, 11)] for v in range(9)]
            all_rows = [s["metrics"] for schedule in schedules for s in schedule]
            arm_max = max(abs(r["x0"]) for r in all_rows)
            M = global_M
            supported = [v for v, schedule in enumerate(schedules)
                         if inference._slope([{**s["metrics"], "share": .5} for s in schedule], "x0")["status"] == "available"]
            geo_key = population + "|" + arm
            geometry[geo_key] = {"D_numerator_sum": sum(r["DeltaNumerator"] for r in all_rows), "D_denominator": 900,
                                 "D": sum(r["DeltaNumerator"] for r in all_rows) / 900,
                                 "M": M, "armMaximumAbsX0": arm_max, "supportedVariants": supported,
                                 "supportedVariantWeights": {str(v): 1 / len(supported) for v in supported},
                                 "assignedVariants": list(range(9)), "nativeTieRows": sum(abs(r["terminalMeans"][0] - r["terminalMeans"][1]) <= 1e-9 * max(1., *map(abs, r["terminalMeans"])) for r in all_rows)}
            projection_plans = {}
            for v in supported:
                metrics = [{**s["metrics"], "share": .5} for s in schedules[v]]
                plan = inference._projection_plan(metrics, "x0")
                payload = {"schema": "frozen-projection-plan-v1", "predictorBinary64": [struct.pack(">d", r["x0"]).hex() for r in metrics],
                           "weightBinary64": [struct.pack(">d", float(w)).hex() for w in plan["weights"]],
                           "operationOrder": "ordered_float64_multiply_then_add_mainIndex_1_to_10",
                           "diagnostics": {k: v for k, v in plan.items() if k != "weights"}}
                projection_plans[str(v)] = {**payload, "planHash": digest(payload)}
            geometry[geo_key]["projectionPlans"] = projection_plans
            geometry[geo_key]["nativeTieComponentStatus"] = "available_native_schedule" if geometry[geo_key]["nativeTieRows"] and population == "working" else "unavailable_native_schedule;artificial_ties_numerical_only"
            R_candidates[geo_key] = {law: _native_R_expectation(schedules, law) for law in
                ("symmetric_beta_0.3", "symmetric_beta_0.6", "centered_beta", "selected_uniform")}
            if not supported:
                slope_candidates[geo_key] = {"status": "structurally_unavailable"}
                continue
            beta_mean = .5 / 5.5
            central_beta = [math.fsum(math.comb(k, j) * (-beta_mean) ** (k - j) * math.prod((.5 + q) / (5.5 + q) for q in range(j)) for j in range(k + 1)) for k in range(11)]
            laws = {
                "zero": [1.] + [0.] * 10,
                "symmetric_uniform": [( .25 / M) ** k / (k + 1) if k % 2 == 0 else 0. for k in range(11)],
                "centered_beta": [(.25 / M) ** k * central_beta[k] for k in range(11)],
                "centered_beta_mirror": [(-.25 / M) ** k * central_beta[k] for k in range(11)],
                "selected_uniform": [((.125 / M) ** (k + 1) - (.0625 / M) ** (k + 1)) / ((k + 1) * (.0625 / M)) for k in range(11)]}
            slope_candidates[geo_key] = {}
            for law, moments in laws.items():
                expectations = [_person_slope_expectation(schedules[v], moments, projection_plans[str(v)]) for v in supported]
                slope_candidates[geo_key][law] = {"idealAlgebraExpectation": moments[1],
                    "nativeFloatingExpectation": math.fsum(e["nativeFloatingExpectation"] for e in expectations) / len(expectations),
                    "perVariant": dict(zip(map(str, supported), expectations)), "status": "floating_candidate_not_exact_truth"}
    arithmetic_maps = {"schema": "native-endpoint-arithmetic-v1",
        "personMean": "ordered_float64_add_mainIndex_1_to_10_then_float64_divide_by_10",
        "initialProjection": "frozen_weightBinary64;ordered_float64_multiply_then_add_mainIndex_1_to_10",
        "nativeExpectation": "analytic_pattern_probability_times_same_native_endpoint;math_fsum;no_zero_snapping"}
    body = {"schema": "native-calibration-lookup-v1", "states": states, "index": lookup_index, "arithmeticMaps": arithmetic_maps,
            "geometry": geometry, "nativeSlopeTruthCandidates": slope_candidates, "nativeRTruthCandidates": R_candidates}
    lookup_hash = digest(body)
    lookup = {**body, "receipt": {"schema": "native-lookup-receipt-v1", "lookupHash": lookup_hash,
            "status": "validated", "sourceStatesHash": source_hash,
            "sourceReceiptHash": source_states["sourceReceiptHash"], "targetReceiptHash": digest(target_api_receipt),
            "analysisSpecHash": digest(analysis_spec), "calibrationSpecHash": digest(calibration_spec),
            "sourceReceipt": source_states["sourceReceipt"], "targetContent": target_api_receipt,
            "geometry": geometry, "arithmeticMaps": arithmetic_maps,
            "nativeSlopeTruthCandidates": slope_candidates, "nativeRTruthCandidates": R_candidates, "machineCertifiedArithmetic": False,
            "prefixCycle": [0, 1, 3], "variants": list(range(9)), "humanObservations": False}}
    validate_lookup(lookup, analysis_spec, calibration_spec)
    return lookup


def validate_lookup(lookup, analysis, calibration):
    """Independently check every finite state, target, index and projection plan.

    Hash consistency alone is not numerical validation. Schedules and streams
    are reconstructed separately from the JS engine and producer kernels.
    """
    inference.validate_spec(analysis)
    validate_calibration(calibration)
    replay = _Replay()
    receipt = lookup["receipt"]
    body = {k: v for k, v in lookup.items() if k != "receipt"}
    replay.require(lookup.get("schema") == LOOKUP_SCHEMA and receipt.get("schema") == "native-lookup-receipt-v1"
                   and receipt.get("status") == "validated" and receipt.get("humanObservations") is False,
                   "Public synthetic lookup schema")
    replay.require(receipt.get("lookupHash") == digest(body), "Lookup content hash mismatch")
    replay.require(receipt.get("analysisSpecHash") == digest(analysis)
                   and receipt.get("calibrationSpecHash") == digest(calibration), "Lookup specification binding")
    source_content = receipt["sourceReceipt"]
    replay.require(source_content.get("schema") == "native-source-content-v1"
                   and source_content.get("humanObservations") is False
                   and receipt["sourceReceiptHash"] == digest(source_content), "Source content binding")
    folder = Path(__file__).resolve().parent
    source_files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    replay.require(source_content["files"] == source_files, "Native source content changed")
    replay.require(receipt["targetContent"] == _target_content()
                   and receipt["targetReceiptHash"] == digest(receipt["targetContent"]), "Target content changed")
    for field in ("geometry", "arithmeticMaps", "nativeSlopeTruthCandidates", "nativeRTruthCandidates"):
        replay.require(receipt[field] == lookup[field], "Lookup header/body mismatch: " + field)
    replay.near(lookup["arithmeticMaps"], {
        "schema": "native-endpoint-arithmetic-v1",
        "personMean": "ordered_float64_add_mainIndex_1_to_10_then_float64_divide_by_10",
        "initialProjection": "frozen_weightBinary64;ordered_float64_multiply_then_add_mainIndex_1_to_10",
        "nativeExpectation": "analytic_pattern_probability_times_same_native_endpoint;math_fsum;no_zero_snapping"},
        "Frozen endpoint arithmetic")
    schedules = {(a, v): I.schedule(_source_meta(a, v)) for a in ARMS for v in range(9)}
    streams, seen = {}, set()
    for index, key in lookup["index"].items():
        population, arm, variant, main, count = index.split("|")
        variant, main, count = map(int, (variant, main, count))
        integer(variant, "variant", 0, 8)
        integer(main, "main index", 1, 10)
        integer(count, "prefix")
        ix = (population, arm, variant, main, count)
        replay.require(ix not in seen, "Duplicate lookup index")
        seen.add(ix)
        ancestor = schedules[(arm, variant)][main + 3]
        stream_key = (arm, variant, main)
        if stream_key not in streams:
            streams[stream_key] = I.scenario(_source_meta(arm, variant), ancestor)[1]
        paid = _paid(ancestor, _prefix(ancestor, main, count), streams[stream_key])
        d = copy.deepcopy(ancestor)
        control = population != "working"
        if control:
            replay.require(arm == "C1" and population in ("gap_0", "gap_0.2", "gap_0.5", "gap_1"),
                           "Invalid conditional population")
            gap = float(population.split("_")[1])
            d["priorCentres"] = [50 * d["scale"] - gap / 2, 50 * d["scale"] + gap / 2]
            d["c1TotalCentreGapMinutes"] = gap
        initial, terminal = replay.logs(d, paid)
        budget = d["budgetMillihours"] - count * d["costMillihours"]
        state = lookup["states"][key]
        sem = state["semantic"]
        recipe = "synthetic-C1-total-gap-" + population.split("_")[1] if control else None
        replay.near(sem, {"lookupSchema": LOOKUP_SCHEMA, "sourceStatesHash": receipt["sourceStatesHash"],
                    "sourceReceiptHash": receipt["sourceReceiptHash"], "targetReceiptHash": receipt["targetReceiptHash"],
                    "population": population, "conditionId": arm, "scheduleVariant": variant,
                    "definition": d, "logicalToDisplayed": ["B", "A"] if d["sideReversed"] else ["A", "B"],
                    "paid": paid, "remainingIntegerBudget": budget, "initialLogs": initial,
                    "terminalLogs": terminal, "syntheticRecipe": recipe}, "Independent semantic reconstruction")
        replay.require(state["binary64Payload"] == binary_payload(sem)
                       and key == digest(binary_payload(sem)), "Binary64 state identity mismatch")
        replay.require(state["engineReplayClaim"] is (not control), "Conditional recipe is not an engine replay")
        means0 = [I.mean(I.probabilities(p)) for p in initial]
        target, means = replay.targets(terminal, budget)
        row = state["metrics"]
        replay.near(row, {"metric_schema": ROW_SCHEMA, "profileId": PROFILE, "analysisId": ANALYSIS_ID,
                    "analysisSpecHash": digest(analysis), "sourceReceiptHash": receipt["sourceReceiptHash"],
                    "targetReceiptHash": receipt["targetReceiptHash"], "conditionId": arm, "mainIndex": main,
                    "templateId": d["templateId"], "scale": d["scale"], "gapOffset": d["gap"],
                    "sideReversed": d["sideReversed"], "gapConvention": d["gapConvention"],
                    "initialIntegerBudget": d["budgetMillihours"], "paidIntegerCost": d["costMillihours"],
                    "remainingIntegerBudget": budget, "paidCount": count, "paid": paid, "Q": int(count > 0),
                    "initialMeans": means0, "terminalMeans": means,
                    "rawInitialGap": means0[0] - means0[1],
                    "x0": (means0[0] - means0[1]) / (d["budgetMillihours"] / 1000),
                    "xT": None if budget == 0 else (means[0] - means[1]) / (budget / 1000),
                    "rowValidated": True, "recordPresent": True, "endpointPresent": True,
                    "paidHistoryValid": True, "humanObservations": False, "origin": "synthetic_validation",
                    "shareIndex": None, "choiceAvailable": False,
                    "choiceReason": "zero_budget" if budget == 0 else "uncommitted_source_state",
                    "syntheticRecipeId": recipe, "nativeProfileClaim": not control,
                    "syntheticRecipeHash": digest(binary_payload(sem["definition"])) if control else None,
                    "reportStreamScope": "fixed_ancestor_reports;not_redrawn_under_control_prior" if control else "fixed_engine_stream",
                    **target}, "Independent finite metrics")
        for name in ("share", "D_ES_pp", "D_EGPA_pp", "R_EGPA_pp", "D_surplus_pp", "R_surplus_pp", "L_lower_mean_share"):
            replay.require(row[name] is None, "Uncommitted choice must remain undefined")
    replay.require(seen == _expected_indices(schedules, calibration["C1SyntheticGapControls"]),
                   "Complete lookup/control/exhaustion roster required")
    replay.require(len(lookup["states"]) == len(seen) and set(lookup["states"]) == set(lookup["index"].values()),
                   "Orphan or merged lookup state")
    from .native_truth import Oracle
    oracle = Oracle(lookup)
    geometry = lookup["geometry"]
    expected_geometry = {"working|" + a for a in ARMS} | {"gap_" + str(g) + "|C1" for g in calibration["C1SyntheticGapControls"]}
    replay.require(set(geometry) == expected_geometry, "Complete geometry roster")
    maximum = max(abs(r["x0"]) for a in ARMS for v in range(9) for r in oracle.schedule("working", a, v))
    for identity, geo in geometry.items():
        population, arm = identity.split("|")
        rows = [r for v in range(9) for r in oracle.schedule(population, arm, v)]
        numerator = sum(abs(r["jEGPA"] - 500) for r in rows)
        replay.near(geo, {"D_numerator_sum": numerator, "D_denominator": 900, "D": numerator / 900,
                         "M": maximum, "armMaximumAbsX0": max(abs(r["x0"]) for r in rows),
                         "assignedVariants": list(range(9))}, "Independent geometry")
        for variant in range(9):
            oracle.plan(population, arm, variant)
        supported = geo["supportedVariants"]
        replay.require(supported == sorted(set(supported)), "Ordered unique eligible variants")
        replay.near(geo["supportedVariantWeights"], {str(v): 1 / len(supported) for v in supported},
                    "Conditional eligible-variant law")
        replay.require(set(geo["projectionPlans"]) == set(map(str, supported)), "Projection-plan roster")
    return {"schema": "native-lookup-check-v1", "lookupHash": receipt["lookupHash"],
            "checks": replay.checks + oracle.checks, "states": len(seen),
            "maximumNumericalError": replay.maximum_error,
            "maximumCoefficientError": oracle.maximum_coefficient_error, "humanObservations": False}
