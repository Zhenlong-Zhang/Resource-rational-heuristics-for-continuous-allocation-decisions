"""Independent recorded-input projection and Student arithmetic.

The public full generator is shared solely as an input-law primitive. Person
projection, support, Student arithmetic and decisions are independently rebuilt.
No producer estimator or summary is used as the numerical oracle.
"""
from __future__ import annotations

import io
import json
import math
import struct
import zipfile
from pathlib import Path

import numpy as np

from . import calibration as cal
from .paired_recipe import (ORIGINAL_RUNTIME, REFERENCE_RUNTIME, fixed_cell,
                            fresh_directory, read_bytes, runtime, scope,
                            selected_batches, write_json)


BATCHES = (0, 39)
SIZE = 250
PEOPLE = 20
ROUNDS = 10
ABS_TOL = 1e-10
REL_TOL = 1e-9
ALPHA = .05
IDS = [f"C4:{i:02d}" for i in range(PEOPLE)]
ARRAY_NAMES = {
    "y", "x", "records", "personKeep", "endpoint_q", "endpoint_R",
    "endpoint_L", "endpoint_initial", "expectedY", "scale", "gap",
}
CASE_NAMES = {
    "batch", "index", "replicationId", "peopleIds", "vectorSlopes",
    "legacyRawSlopes", "projectionSlopes", "qualifications", "legacy",
    "legacyVector", "currentSameInput", "currentProjection", "decisions",
}
DECISION_NAMES = {
    "legacyLE", "currentLT", "projectionLT", "legacyCovered",
    "currentCovered", "projectionCovered", "legacyBoundary",
    "currentBoundary", "projectionBoundary",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def decimal(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON decimal")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=decimal)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def bits(value):
    return struct.pack(">d", float(value)).hex()


class Audit:
    def __init__(self):
        self.numeric_checks = 0
        self.exact_checks = 0
        self.errors = {}

    def exact(self, actual, expected, label):
        require(type(actual) is type(expected) and actual == expected,
                f"Exact mismatch: {label}")
        if type(expected) is dict:
            for key in expected:
                self.exact(actual[key], expected[key], label + "/" + str(key))
        elif type(expected) is list:
            for index, value in enumerate(expected):
                self.exact(actual[index], value, label + "/" + str(index))
        elif type(expected) is float:
            require(bits(actual) == bits(expected), f"Binary64 mismatch: {label}")
        self.exact_checks += 1

    def close(self, actual, expected, family, label):
        if expected is None:
            self.exact(actual, None, label)
            return
        require(finite_number(actual) and finite_number(expected),
                f"Nonfinite/non-numeric field: {label}")
        error = abs(actual - expected)
        allowed = ABS_TOL + REL_TOL * abs(expected)
        require(error <= allowed, f"Numerical mismatch: {label}: {error} > {allowed}")
        row = self.errors.setdefault(family, {
            "checks": 0, "exactNonzeroDifferences": 0,
            "maximumAbsoluteError": 0., "maximumRelativeError": 0.,
            "maximumToleranceFraction": 0.,
        })
        row["checks"] += 1
        row["exactNonzeroDifferences"] += int(bits(actual) != bits(expected))
        row["maximumAbsoluteError"] = max(row["maximumAbsoluteError"], error)
        relative = error / abs(expected) if expected else None
        if relative is not None:
            row["maximumRelativeError"] = max(row["maximumRelativeError"], relative)
        row["maximumToleranceFraction"] = max(row["maximumToleranceFraction"], error / allowed)
        self.numeric_checks += 1


def derive_projection(x, y, valid, keep):
    """Centered OLS formula, independently of both least-squares implementations."""
    require(type(keep) is bool and len(x) == len(y) == len(valid) == ROUNDS,
            "Malformed person projection")
    require(all(type(v) is bool for v in valid), "Nonboolean validity mask")
    require(all(finite_number(v) for v in x + y), "Nonfinite raw person inputs")
    xs = [v for v, ok in zip(x, valid) if ok]
    ys = [v for v, ok in zip(y, valid) if ok]
    n = len(xs)
    mean_x = math.fsum(xs) / n if n else 0.
    centered = [v - mean_x for v in xs]
    sxx = math.fsum(v * v for v in centered)
    levels = len(set(xs))
    detail = {
        "observations": n, "predictorLevels": levels, "Sxx": sxx,
        "predictorMinimum": min(xs) if n else None,
        "predictorMaximum": max(xs) if n else None,
        "predictorSpan": max(xs) - min(xs) if n else None,
        "model": "unadjusted",
    }
    if not keep or n < 3 or levels < 2 or sxx <= 1e-12:
        return {**detail, "status": "unavailable", "reason": "independent_support_failure", "value": None}
    weights = [v / sxx for v in centered]
    detail.update(rank=2, columns=2, residualDf=n - 2)
    value = math.fsum(w * v for w, v in zip(weights, ys))
    return {**detail, "status": "available", "reason": None, "value": value,
            "allocationChangeOverSpan": value * detail["predictorSpan"], "weights": weights}


def ordered_projection(np, weights, values):
    """Independent transcription of the declared binary64 operation order."""
    require(len(weights) == len(values) == ROUNDS, "Bad ordered projection shape")
    total = np.float64(0.)
    for weight, value in zip(weights, values):
        total = np.add(total, np.multiply(np.float64(weight), np.float64(value)))
    return float(total)


def student(values, tail, stdtr, stdtrit):
    """One independent arm; no production inference routine is called."""
    require(tail == "less", "Diagnostic tail must remain less")
    require(type(values) is list and all(finite_number(v) for v in values),
            "Malformed independent Student inputs")
    n = len(values)
    result = {"n": n, "mean": None, "variance": None, "effect": None,
              "SE": None, "df": None, "statistic": None, "p": None,
              "CI": None, "status": "unavailable"}
    if n < 2:
        return result
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    se2 = variance / n
    result.update(mean=mean, variance=variance, effect=mean)
    if not math.isfinite(se2) or se2 <= 0:
        return result
    df = float(n - 1)
    se = math.sqrt(se2)
    statistic = mean / se
    probability = float(stdtr(df, statistic))
    radius = float(stdtrit(df, .975)) * se
    require(all(math.isfinite(v) for v in (mean, variance, se, df, statistic, probability, radius)),
            "Nonfinite independent Student output")
    require(0 <= probability <= 1 and radius >= 0, "Invalid Student uncertainty")
    return {**result, "SE": se, "df": df, "statistic": statistic,
            "p": probability, "CI": [mean - radius, mean + radius], "status": "available"}


def decisions(legacy, current, projection):
    def reject(result, inclusive):
        return result["status"] == "available" and (result["p"] <= ALPHA if inclusive else result["p"] < ALPHA)

    def covered(result):
        return result["status"] == "available" and result["CI"][0] <= 0 <= result["CI"][1]

    def boundary(result):
        return result["status"] == "available" and result["p"] == ALPHA

    return {
        "legacyLE": reject(legacy, True), "currentLT": reject(current, False),
        "projectionLT": reject(projection, False), "legacyCovered": covered(legacy),
        "currentCovered": covered(current), "projectionCovered": covered(projection),
        "legacyBoundary": boundary(legacy), "currentBoundary": boundary(current),
        "projectionBoundary": boundary(projection),
    }


def transport_arrays(raw, np):
    require(type(raw) is bytes and 0 < len(raw) <= 4 * 1024 * 1024,
            "Malformed/oversized NPZ transport")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) == len(ARRAY_NAMES)
                and {m.filename for m in members} == {f"{k}.npy" for k in ARRAY_NAMES}
                and sum(m.file_size for m in members) <= 4 * 1024 * 1024,
                "NPZ member roster/size mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        require(set(archive.files) == ARRAY_NAMES, "NPZ array roster mismatch")
        arrays = {key: archive[key] for key in ARRAY_NAMES}
    shapes = {"y": (SIZE, PEOPLE, ROUNDS), "x": (PEOPLE, ROUNDS),
              "records": (SIZE, PEOPLE, ROUNDS), "personKeep": (SIZE, PEOPLE),
              "expectedY": (PEOPLE, ROUNDS), "scale": (ROUNDS,), "gap": (ROUNDS,)}
    for key in ("endpoint_q", "endpoint_R", "endpoint_L", "endpoint_initial"):
        shapes[key] = (SIZE, PEOPLE, ROUNDS)
    for key, value in arrays.items():
        require(value.shape == shapes[key], f"Raw shape mismatch: {key}")
        boolean = key in {"records", "personKeep", "endpoint_q", "endpoint_R", "endpoint_L", "endpoint_initial"}
        require(value.dtype == np.dtype(bool if boolean else np.float64), f"Raw dtype mismatch: {key}")
        require(boolean or bool(np.isfinite(value).all()), f"Nonfinite raw array: {key}")
    return arrays


def compare_raw(audit, actual, expected, np, batch):
    for key in ARRAY_NAMES:
        value = expected[key]
        require(actual[key].shape == value.shape and actual[key].dtype == value.dtype
                and actual[key].tobytes(order="C") == value.tobytes(order="C"),
                f"Full regenerated raw input mismatch: batch{batch}/{key}")
        audit.exact_checks += int(value.size)


def compare_legacy(audit, actual, oracle, label):
    require(type(actual) is dict and set(actual) == {
        "status", "reason", "weights", "side", "confidence", "method",
        "independentUnit", "weighting", "groups", "estimate", "standardError",
        "df", "statistic", "pValue", "interval",
    }, label + "/legacy fields")
    audit.exact(actual["status"], oracle["status"], label + "/status")
    audit.exact(actual["reason"], None, label + "/reason")
    audit.exact(actual["weights"], {"C4": 1}, label + "/weights")
    audit.exact(actual["side"], "less", label + "/side")
    audit.exact(actual["confidence"], .95, label + "/confidence")
    audit.exact(actual["method"], "one_sample_t", label + "/method")
    audit.exact(actual["independentUnit"], "participant", label + "/independentUnit")
    audit.exact(actual["weighting"], "equal participant within each condition; fixed condition weights", label + "/weighting")
    require(set(actual["groups"]) == {"C4"}, label + "/group roster")
    arm = actual["groups"]["C4"]
    require(type(arm) is dict and set(arm) == {"participants", "mean", "variance"},
            label + "/group fields")
    audit.exact(arm["participants"], PEOPLE, label + "/participants")
    for key in ("mean", "variance"):
        audit.close(arm[key], oracle[key], "legacyStudent", label + "/" + key)
    for key, ref in (("estimate", "effect"), ("standardError", "SE"), ("df", "df"),
                     ("statistic", "statistic"), ("pValue", "p")):
        audit.close(actual[key], oracle[ref], "legacyStudent", label + "/" + key)
    require(0 <= actual["pValue"] <= 1 and actual["standardError"] > 0 and actual["df"] > 0,
            label + "/invalid uncertainty")
    require(type(actual["interval"]) is list and len(actual["interval"]) == 2, label + "/interval shape")
    for index in range(2):
        audit.close(actual["interval"][index], oracle["CI"][index], "legacyStudent", f"{label}/CI{index}")
    require(actual["interval"][0] <= actual["interval"][1], label + "/reversed interval")


def compare_current(audit, actual, oracle, family, label):
    require(type(actual) is dict and set(actual) == {
        "testId", "endpointId", "weights", "tail", "null", "alpha", "confidence",
        "primary", "requiredArmCounts", "method", "status", "reason", "effect",
        "SE", "df", "statistic", "p", "CI", "armMeans", "candidateNominalReject",
    }, label + "/current fields")
    for key, expected in (("testId", "H05_initial_C4"), ("endpointId", "initial_slope"),
                          ("weights", {"C4": 1}), ("tail", "less"), ("null", 0),
                          ("alpha", .05), ("confidence", .95), ("primary", True),
                          ("requiredArmCounts", {"C4": PEOPLE}), ("method", "Student"),
                          ("status", "available"), ("reason", None)):
        audit.exact(actual[key], expected, label + "/" + key)
    for key in ("effect", "SE", "df", "statistic", "p"):
        audit.close(actual[key], oracle[key], family, label + "/" + key)
    require(0 <= actual["p"] <= 1 and actual["SE"] > 0 and actual["df"] > 0,
            label + "/invalid uncertainty")
    require(set(actual["armMeans"]) == {"C4"}, label + "/armMeans roster")
    audit.close(actual["armMeans"]["C4"], oracle["mean"], family, label + "/armMean")
    require(type(actual["CI"]) is list and len(actual["CI"]) == 2, label + "/CI shape")
    for index in range(2):
        audit.close(actual["CI"][index], oracle["CI"][index], family, f"{label}/CI{index}")
    require(actual["CI"][0] <= actual["CI"][1], label + "/reversed interval")
    audit.exact(actual["candidateNominalReject"], actual["p"] < ALPHA, label + "/candidateNominalReject")


def compare_vector(audit, actual, oracle, label):
    expected_names = {"estimate", "se", "df", "statistic", "p", "low", "high",
                      "available", "reason", "truth", "selectedTruth", "null", "selectedNull"}
    require(set(actual) == expected_names, label + "/vector fields")
    for key, ref in (("estimate", "effect"), ("se", "SE"), ("df", "df"), ("statistic", "statistic"), ("p", "p")):
        audit.close(actual[key], oracle[ref], "legacyVectorStudent", label + "/" + key)
    for key, index in (("low", 0), ("high", 1)):
        audit.close(actual[key], oracle["CI"][index], "legacyVectorStudent", label + "/" + key)
    require(0 <= actual["p"] <= 1 and actual["se"] > 0 and actual["df"] > 0
            and actual["low"] <= actual["high"], label + "/invalid vector uncertainty")
    for key, expected in (("available", True), ("reason", 0), ("null", True), ("selectedNull", True)):
        audit.exact(actual[key], expected, label + "/" + key)
    for key in ("truth", "selectedTruth"):
        audit.exact(actual[key], 0., label + "/" + key)


def contrast_view(value, legacy=False):
    return {"status": value["status"], "p": value["pValue" if legacy else "p"],
            "CI": value["interval" if legacy else "CI"]}


def self_checks(np, stdtr, stdtrit):
    results = []

    def negative(name, call):
        try:
            call()
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
            results.append({"name": name, "status": "PASS"})
        else:
            raise ValueError("Expected independent negative rejection: " + name)

    negative("duplicate_json", lambda: strict_json(b'{"a":1,"a":2}'))
    negative("nonfinite_json", lambda: strict_json(b'{"a":NaN}'))
    negative("overflow_json_decimal", lambda: strict_json(b'{"a":1e999}'))
    negative("malformed_npz_transport", lambda: transport_arrays(b"not-an-npz", np))
    negative("changed_tail", lambda: student([0., 1.], "greater", stdtr, stdtrit))
    negative("nonfinite_person_input", lambda: student([0., float("nan")], "less", stdtr, stdtrit))
    no_support = derive_projection([0.] * ROUNDS, [1.] * ROUNDS, [True] * ROUNDS, True)
    require(no_support["status"] == "unavailable", "Constant predictor was admitted")
    results.append({"name": "constant_predictor_support", "status": "PASS"})
    no_se = student([1.] * PEOPLE, "less", stdtr, stdtrit)
    require(no_se["status"] == "unavailable" and no_se["p"] is None, "Zero-SE was admitted")
    results.append({"name": "zero_standard_error", "status": "PASS"})
    boundary = {"status": "available", "p": ALPHA, "CI": [-1., 1.]}
    flags = decisions(boundary, boundary, boundary)
    require(flags["legacyLE"] and not flags["currentLT"] and not flags["projectionLT"]
            and flags["legacyBoundary"] and flags["currentBoundary"] and flags["projectionBoundary"],
            "Strict/inclusive boundary conventions collapsed")
    results.append({"name": "exact_alpha_boundary", "status": "PASS"})
    wrong = Audit()
    negative("stale_numeric_output", lambda: wrong.close(1., 0., "fixture", "stale"))
    negative("integer_not_boolean", lambda: wrong.exact(1, True, "typed boolean"))
    return {"tests": len(results), "passed": len(results), "skipped": 0, "cases": results}


def validate_summary_header(summary, require_full=False):
    """Reject false full-scope claims before any numerical work."""
    require(type(summary) is dict, "Malformed producer summary")
    schema = summary.get("schema")
    require(schema in ("paired-gamma-diagnostic-v1", "historical-gamma-diagnostic-v01"),
            "Unknown producer schema")
    names = {
        "schema", "scope", "cases", "auditOrigin", "testResults", "batchSummaries",
        "originalRuntime", "observedRuntime", "aggregateDifferences",
        "elapsedEntryScienceSeconds", "limitations", "typeIRepairClaim",
        "historicalFailuresChanged", "formalDispositionsChanged",
    }
    if schema == "paired-gamma-diagnostic-v1":
        names |= {"selectedComplete", "fixed500Complete", "referenceRuntime"}
    require(set(summary) == names, "Producer summary field roster")
    require(type(summary["scope"]) is dict, "Malformed producer scope")
    batches = selected_batches(summary["scope"]["batches"])
    audit = Audit()
    audit.exact(summary["scope"], scope(batches), "Frozen scope")
    audit.exact(summary["cases"], SIZE * len(batches), "Declared case census")
    if schema == "historical-gamma-diagnostic-v01":
        require(batches == BATCHES, "Original schema requires both complete batches")
    else:
        audit.exact(summary["selectedComplete"], True, "Selected completion")
        audit.exact(summary["fixed500Complete"], batches == BATCHES, "Fixed500 completion")
        audit.exact(summary["referenceRuntime"], REFERENCE_RUNTIME, "Reference runtime")
    require(not require_full or batches == BATCHES, "Full fixed500 packet required")
    audit.exact(summary["originalRuntime"], ORIGINAL_RUNTIME, "Earlier calibration runtime")
    observed = summary["observedRuntime"]
    require(type(observed) is dict and set(observed) == {"python", "numpy", "scipy"}
            and all(type(v) is str and v.strip() for v in observed.values()), "Malformed runtime")
    require(finite_number(summary["elapsedEntryScienceSeconds"])
            and summary["elapsedEntryScienceSeconds"] >= 0, "Malformed elapsed time")
    require(type(summary["limitations"]) is list and bool(summary["limitations"])
            and all(type(v) is str and v.strip() for v in summary["limitations"]),
            "Missing diagnostic limitations")
    require(type(summary["aggregateDifferences"]) is dict
            and set(summary["aggregateDifferences"]) == {
                "maxAbs", "nonzeroCaseCounts", "decisionCounts", "sameInputRejectDiscordances"},
            "Aggregate field roster")
    return batches


def _read_packet(config, producer, output=None, require_full=False):
    from scipy.special import stdtr, stdtrit

    producer = Path(producer)
    require(producer.is_dir(), "Input packet must be a directory")
    if output is not None:
        output = Path(output)
        require(not output.exists() and not output.is_symlink(), "Output must be a new directory")
        require(not output.resolve().is_relative_to(producer.resolve()),
                "Reader output cannot modify the input packet")
    producer_summary = strict_json(read_bytes(producer / "summary.json"))
    batches = validate_summary_header(producer_summary, require_full)
    cell = fixed_cell(config)
    expected_files = {f"batch_{b}.npz" for b in batches} | {f"cases_{b}.json" for b in batches}
    actual_files = {p.name for pattern in ("batch_*", "cases_*") for p in producer.glob(pattern)}
    require(actual_files == expected_files, "Missing, extra or aliased science files")
    audit = Audit()
    test_results = self_checks(np, stdtr, stdtrit)
    plan = strict_json(read_bytes(producer / "plan.json"))
    require(set(plan) == {"kind", "weights", "diagnostics"}, "Projection plan fields changed")
    audit.exact(plan["kind"], "diagnostic_unadjusted_same_raw_schedule_not_native_plan", "plan kind")
    require(type(plan["weights"]) is list and len(plan["weights"]) == PEOPLE
            and type(plan["diagnostics"]) is list and len(plan["diagnostics"]) == PEOPLE,
            "Projection plan people roster")
    batch_rows, checked_cases = [], []
    aggregate = {name: 0 for name in DECISION_NAMES}
    aggregate.update({"sameInputRejectionDifferences": 0, "projectionRejectionDifferences": 0,
                      "sameInputCoverageDifferences": 0, "projectionCoverageDifferences": 0,
                      "sameInputPExactNonzeroDifferences": 0, "projectionPExactNonzeroDifferences": 0,
                      "projectionPeopleExactNonzeroDifferences": 0,
                      "maximumSameInputAbsolutePDifference": 0.,
                      "maximumProjectionAbsolutePDifference": 0.,
                      "maximumRawOLSAbsoluteDifference": 0.,
                      "maximumProjectionPersonAbsoluteDifference": 0.})
    producer_names = ("legacyCurrentEffect", "legacyCurrentSE", "legacyCurrentDF",
                      "legacyCurrentStatistic", "legacyCurrentP", "legacyCurrentCILow",
                      "legacyCurrentCIHigh", "vectorRawSlope", "vectorProjectionSlope",
                      "rawProjectionSlope")
    producer_maxima = {name: 0. for name in producer_names}
    producer_nonzero = {name: 0 for name in producer_names}
    for batch in batches:
        arrays = transport_arrays(read_bytes(producer / f"batch_{batch}.npz"), np)
        generated = cal.generate_batch(cell, batch, SIZE)
        require(set(generated["raw"]) == {"C1", "C2", "C3", "C4", "C5"}, "Full generator condition roster")
        raw = generated["raw"]["C4"]
        expected = {key: raw[key] for key in ("y", "x", "records", "personKeep", "expectedY", "scale", "gap")}
        expected.update({"endpoint_" + key: raw["endpointMasks"][key] for key in ("q", "R", "L", "initial")})
        compare_raw(audit, arrays, expected, np, batch)
        require(bool(arrays["records"].all()) and bool(arrays["personKeep"].all())
                and all(bool(arrays["endpoint_" + key].all()) for key in ("q", "R", "L", "initial")),
                "Unexpected eligibility/missingness in fixed full-retention law")
        del generated, raw, expected
        cases = strict_json(read_bytes(producer / f"cases_{batch}.json"))
        require(type(cases) is list and len(cases) == SIZE, "Case count/shape changed")
        counts = {name: 0 for name in DECISION_NAMES}
        for index, case in enumerate(cases):
            require(type(case) is dict and set(case) == CASE_NAMES, "Case field roster changed")
            for name, expected_value in (("batch", batch), ("index", index), ("replicationId", batch * SIZE + index)):
                audit.exact(case[name], expected_value, name)
            audit.exact(case["peopleIds"], IDS, "Person IDs/order")
            for key in ("vectorSlopes", "legacyRawSlopes", "projectionSlopes", "qualifications"):
                require(type(case[key]) is list and len(case[key]) == PEOPLE, "Person vector shape: " + key)
            rebuilt_vector, rebuilt_projection = [], []
            for person in range(PEOPLE):
                xs = arrays["x"][person].tolist()
                ys = arrays["y"][index, person].tolist()
                records = arrays["records"][index, person].tolist()
                valid = (arrays["records"][index, person] & arrays["endpoint_initial"][index, person]).tolist()
                keep = bool(arrays["personKeep"][index, person]) and all(records)
                projection = derive_projection(xs, ys, valid, keep)
                require(projection["status"] == "available", "Unexpected unavailable fixed C4 person")
                qualification = case["qualifications"][person]
                require(type(qualification) is dict and set(qualification) == set(projection) - {"weights"},
                        "Frozen qualification field roster")
                for key in ("observations", "predictorLevels", "rank", "columns", "residualDf", "model", "status", "reason"):
                    audit.exact(qualification[key], projection[key], f"qualification/{key}")
                for key in ("Sxx", "predictorMinimum", "predictorMaximum", "predictorSpan", "value", "allocationChangeOverSpan"):
                    audit.close(qualification[key], projection[key], "rawOLS", f"qualification/{key}")
                audit.exact(case["legacyRawSlopes"][person], qualification["value"], "Raw OLS arm/value")
                audit.close(case["legacyRawSlopes"][person], projection["value"], "rawOLS", "raw OLS slope")
                audit.close(case["vectorSlopes"][person], projection["value"], "vectorProjection", "vector person slope")
                valid_x = np.asarray(xs, dtype=np.float64)
                centered = valid_x - np.sum(valid_x) / ROUNDS
                vector_value = float(np.sum((centered / np.sum(centered ** 2)) * np.asarray(ys, dtype=np.float64)))
                audit.exact(case["vectorSlopes"][person], vector_value, "Regenerated vector projection")
                rebuilt_vector.append(vector_value)
                weights = plan["weights"][person]
                require(type(weights) is list and len(weights) == ROUNDS, "Plan weight shape")
                for position, weight in enumerate(weights):
                    audit.close(weight, projection["weights"][position], "projectionWeights", "Independent centered weight")
                diagnostic = plan["diagnostics"][person]
                require(set(diagnostic) == {"observations", "predictorLevels", "Sxx", "rank", "residualDf"},
                        "Plan support fields changed")
                for key in ("observations", "predictorLevels", "rank", "residualDf"):
                    audit.exact(diagnostic[key], projection[key], "Plan support/" + key)
                audit.close(diagnostic["Sxx"], projection["Sxx"], "projectionSupport", "Plan Sxx")
                ordered = ordered_projection(np, weights, ys)
                audit.exact(case["projectionSlopes"][person], ordered, "Ordered canonical projection")
                rebuilt_projection.append(ordered)
                audit.close(ordered, projection["value"], "canonicalProjection", "Canonical/analytic projection")
                aggregate["maximumRawOLSAbsoluteDifference"] = max(aggregate["maximumRawOLSAbsoluteDifference"], abs(case["legacyRawSlopes"][person] - projection["value"]))
                delta = abs(case["projectionSlopes"][person] - case["vectorSlopes"][person])
                aggregate["maximumProjectionPersonAbsoluteDifference"] = max(aggregate["maximumProjectionPersonAbsoluteDifference"], delta)
                aggregate["projectionPeopleExactNonzeroDifferences"] += int(bits(case["projectionSlopes"][person]) != bits(case["vectorSlopes"][person]))
            shared = student(rebuilt_vector, "less", stdtr, stdtrit)
            projected = student(rebuilt_projection, "less", stdtr, stdtrit)
            require(shared["status"] == projected["status"] == "available", "Unexpected fixed-law zero SE")
            label = f"batch{batch}/case{index}"
            compare_legacy(audit, case["legacy"], shared, label + "/legacy")
            compare_vector(audit, case["legacyVector"], shared, label + "/legacyVector")
            compare_current(audit, case["currentSameInput"], shared, "currentSameStudent", label + "/sameInput")
            compare_current(audit, case["currentProjection"], projected, "currentProjectionStudent", label + "/currentProjection")
            actual_decisions = decisions(contrast_view(case["legacy"], True),
                                         contrast_view(case["currentSameInput"]), contrast_view(case["currentProjection"]))
            oracle_decisions = decisions(shared, shared, projected)
            require(type(case["decisions"]) is dict and set(case["decisions"]) == DECISION_NAMES,
                    "Decision fields changed")
            for name in DECISION_NAMES:
                audit.exact(case["decisions"][name], actual_decisions[name], label + "/actual " + name)
                audit.exact(actual_decisions[name], oracle_decisions[name], label + "/independent " + name)
                counts[name] += int(actual_decisions[name])
                aggregate[name] += int(actual_decisions[name])
            old_p, same_p, projection_p = case["legacy"]["pValue"], case["currentSameInput"]["p"], case["currentProjection"]["p"]
            aggregate["sameInputPExactNonzeroDifferences"] += int(bits(old_p) != bits(same_p))
            aggregate["projectionPExactNonzeroDifferences"] += int(bits(same_p) != bits(projection_p))
            aggregate["maximumSameInputAbsolutePDifference"] = max(aggregate["maximumSameInputAbsolutePDifference"], abs(old_p - same_p))
            aggregate["maximumProjectionAbsolutePDifference"] = max(aggregate["maximumProjectionAbsolutePDifference"], abs(same_p - projection_p))
            for name, left, right in (("sameInputRejectionDifferences", "legacyLE", "currentLT"),
                                      ("projectionRejectionDifferences", "currentLT", "projectionLT"),
                                      ("sameInputCoverageDifferences", "legacyCovered", "currentCovered"),
                                      ("projectionCoverageDifferences", "currentCovered", "projectionCovered")):
                aggregate[name] += int(actual_decisions[left] != actual_decisions[right])
            old, current = case["legacy"], case["currentSameInput"]
            delta_fields = {
                "legacyCurrentEffect": abs(old["estimate"] - current["effect"]),
                "legacyCurrentSE": abs(old["standardError"] - current["SE"]),
                "legacyCurrentDF": abs(old["df"] - current["df"]),
                "legacyCurrentStatistic": abs(old["statistic"] - current["statistic"]),
                "legacyCurrentP": abs(old["pValue"] - current["p"]),
                "legacyCurrentCILow": abs(old["interval"][0] - current["CI"][0]),
                "legacyCurrentCIHigh": abs(old["interval"][1] - current["CI"][1]),
                "vectorRawSlope": max(abs(v - r) for v, r in zip(case["vectorSlopes"], case["legacyRawSlopes"])),
                "vectorProjectionSlope": max(abs(v - p) for v, p in zip(case["vectorSlopes"], case["projectionSlopes"])),
                "rawProjectionSlope": max(abs(r - p) for r, p in zip(case["legacyRawSlopes"], case["projectionSlopes"])),
            }
            for name, value in delta_fields.items():
                producer_maxima[name] = max(producer_maxima[name], value)
                producer_nonzero[name] += int(value != 0.)
            checked_cases.append({"batch": batch, "index": index, "replicationId": batch * SIZE + index,
                                  "sameInputOracle": shared, "projectionOracle": projected,
                                  "decisions": actual_decisions})
        batch_rows.append({"batch": batch, "cases": SIZE, "fullRawInputComparison": "EXACT",
                           "independentStudentComparison": "MATCH_WITH_DECLARED_ROUNDOFF", "decisions": counts})
    require(len(checked_cases) == SIZE * len(batches), "Incomplete independent case census")
    audit.exact(producer_summary["auditOrigin"], "historical_abstract_summary_audit", "Producer summary origin")
    audit.exact(producer_summary["originalRuntime"],
                {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.16.3"}, "Original runtime")
    audit.exact(producer_summary["typeIRepairClaim"], False, "No Type-I repair")
    audit.exact(producer_summary["historicalFailuresChanged"], 0, "Historical failures unchanged")
    audit.exact(producer_summary["formalDispositionsChanged"], False, "Formal dispositions unchanged")
    producer_tests = producer_summary["testResults"]
    public_tests = [
        "duplicate_json_key", "nonfinite_json", "overflow_json_decimal",
        "changed_designated_tail", "duplicate_independent_person",
        "both_zero_SE_unavailable", "constant_predictor_unavailable",
        "exact_alpha_boundary_conventions",
    ]
    original_tests = [
        "duplicate_json_key", "nonfinite_json", "wrong_raw_source_pin",
        "noncanonical_input_alias", "changed_designated_tail",
        "duplicate_independent_person", "both_zero_SE_unavailable",
        "constant_predictor_unavailable", "bound_python_sources_compile_in_job",
        "exact_alpha_boundary_conventions",
    ]
    expected_tests = original_tests if producer_summary["schema"] == "historical-gamma-diagnostic-v01" else public_tests
    audit.exact(producer_tests, {"passed": len(expected_tests), "failed": 0,
                                "skipped": 0, "names": expected_tests}, "Producer self-check record")
    audit.exact(strict_json(read_bytes(producer / "tests.json")), producer_tests, "Producer tests file")
    paired = producer_summary["aggregateDifferences"]
    audit.exact(paired["decisionCounts"], {name: aggregate[name] for name in DECISION_NAMES}, "Producer aggregate decisions")
    audit.exact(paired["sameInputRejectDiscordances"], aggregate["sameInputRejectionDifferences"], "Producer discordances")
    audit.exact(paired["nonzeroCaseCounts"], producer_nonzero, "Producer exact nonzero differences")
    audit.exact(paired["maxAbs"], producer_maxima, "Producer maximum differences")
    require(type(producer_summary["batchSummaries"]) is list and len(producer_summary["batchSummaries"]) == len(batches),
            "Producer batch summary census")
    for observed, independent in zip(producer_summary["batchSummaries"], batch_rows):
        require(type(observed) is dict and set(observed) == {
            "batch", "cases", "decisionCounts", "sameInputRejectDiscordances", "seconds"},
            "Batch summary field roster")
        audit.exact(observed["batch"], independent["batch"], "Producer batch identity")
        audit.exact(observed["cases"], SIZE, "Producer batch cases")
        audit.exact(observed["decisionCounts"], independent["decisions"], "Producer batch decisions")
        start = sum(1 for case in checked_cases if case["batch"] == independent["batch"]
                    and case["decisions"]["legacyLE"] != case["decisions"]["currentLT"])
        audit.exact(observed["sameInputRejectDiscordances"], start, "Producer batch discordances")
        require(finite_number(observed["seconds"]) and observed["seconds"] >= 0, "Invalid batch timing")
    result = {
        "schema": "paired-gamma-readback-v1", "scope": producer_summary["scope"],
        "auditOrigin": "historical_abstract_summary_audit", "cases": len(checked_cases),
        "selectedComplete": True, "fixed500Complete": batches == BATCHES,
        "allRawInputsReconstructed": True, "generatorReuse": "public_full_generator_input_law_primitive_only",
        "independentStudentChecks": 4 * len(checked_cases), "testResults": test_results,
        "batchSummaries": batch_rows, "aggregateDifferences": aggregate,
        "actualMaxErrors": audit.errors, "numericFieldChecks": audit.numeric_checks,
        "exactStructuralAndRawChecks": audit.exact_checks,
        "numericalTolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
        "oldBoundary": "p<=.05", "currentBoundary": "p<.05",
        "originalRuntime": producer_summary["originalRuntime"], "observedRuntime": runtime(),
        "referenceRuntime": REFERENCE_RUNTIME, "referenceRuntimeMatches": runtime() == REFERENCE_RUNTIME,
        "typeICalibrationAccepted": False, "historicalFailureResolved": False,
        "historicalBitReplayCertified": False, "additionalIndependentHistoricalReplications": 0,
        "formalResultsChanged": False,
        "scientificReadback": "MATCH_WITH_DECLARED_ROUNDOFF",
        "operationalProvenanceAuthenticated": False,
        "limitations": [
            "Fixed first/last250-shaped batches of original declared law, not500 independent additions to old10000.",
            "Shared bound generator reconstructs inputs; person projection and Student numerical oracle are independently written.",
            "Observed current runtime does not certify original-runtime realization bits; old aggregate agreement is not bit replay.",
            "No500-case Type-I/coverage PASS or CP criterion; unchanged skewness assumptions remain unless a separately explained correction exists.",
            "No estimator replacement, FWER, historical111 resolution, native-trial validity or human-study approval.",
        ],
    }

    if output is not None:
        output = fresh_directory(output)
        write_json(output / "independent_cases.json", checked_cases)
        write_json(output / "independent_tests.json", test_results)
        write_json(output / "summary.json", result)
    return result


def read_packet(config, producer, output=None, require_full=False):
    """Check all selected science, including existing original full packets.

    Original operational receipts and pre-adoption flags are neither changed
    nor authenticated. An optional output is created only after validation.
    """
    try:
        return _read_packet(config, producer, output, require_full)
    except (KeyError, IndexError, TypeError, OverflowError, OSError,
            zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("Malformed or incomplete diagnostic packet: " + str(exc)) from exc
