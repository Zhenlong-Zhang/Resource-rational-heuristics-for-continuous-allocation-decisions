"""Candidate composition of externally supplied difference/sum p values.

This module neither computes linear-contrast p values nor establishes their
validity. Conditional on valid one-sided input p values, the two IUT branches
and Bonferroni combination test |beta_C1| < |beta_reference|. Holm adjustment
then uses the original fixed four-comparison family. No model, covariance, df,
guard, data provenance, fit or random draw is inferred from these inputs.
"""

import math


COMPARISON_IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
P_KEYS = ("D_less", "D_greater", "S_less", "S_greater")
ALPHA = 0.05
METHOD_ID = "h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1"
_RECORD_KEYS = frozenset(("comparisonID", "status", "p", "reason"))
_STATUSES = ("available", "statistical_unavailable", "technical_unresolved")


def _probability(value):
    """Accept only finite built-in probabilities, without coercion or clipping."""
    if type(value) not in (int, float):
        raise ValueError("Probability must be a built-in int or float, excluding bool")
    # Check bounds before conversion so an arbitrarily large integer is rejected
    # as an input error instead of raising OverflowError in float()/isfinite().
    if value < 0 or value > 1:
        raise ValueError("Probability must lie in [0, 1]")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Probability must be finite")
    return value


def compose_comparison(record):
    """Validate and combine one of the four fixed scientific comparisons.

    The input must have exactly comparisonID/status/p/reason. For an available
    comparison p contains D_less, D_greater, S_less and S_greater. Otherwise p is
    None and reason is a nonempty string. Opposite-tail p values need not sum to
    one. Statistical unavailability is a completed non-discovery; technical
    uncertainty remains unknown. Neither state is inferred from the numbers.
    """
    if type(record) is not dict or set(record) != _RECORD_KEYS:
        raise ValueError("Comparison record must contain exactly comparisonID/status/p/reason")
    comparison_id = record["comparisonID"]
    if type(comparison_id) is not str or comparison_id not in COMPARISON_IDS:
        raise ValueError("Unknown comparisonID")
    status = record["status"]
    if type(status) is not str or status not in _STATUSES:
        raise ValueError("Unknown comparison status")
    reason = record["reason"]
    if status == "available":
        if reason is not None:
            raise ValueError("Available comparison requires reason=None")
        values = record["p"]
        if type(values) is not dict or set(values) != set(P_KEYS):
            raise ValueError("Available comparison requires exactly four D/S tail p values")
        input_p = {key: _probability(values[key]) for key in P_KEYS}
        positive = max(input_p["D_less"], input_p["S_greater"])
        negative = max(input_p["D_greater"], input_p["S_less"])
        orientation_p = {"reference_positive": positive, "reference_negative": negative}
        magnitude_p = min(1.0, 2.0 * min(positive, negative))
        nominal_reject = magnitude_p < ALPHA
    else:
        if record["p"] is not None:
            raise ValueError("Unavailable or unresolved comparison requires p=None")
        if type(reason) is not str or not reason.strip():
            raise ValueError("Unavailable or unresolved comparison requires a nonempty reason")
        input_p = None
        orientation_p = None
        magnitude_p = None
        nominal_reject = False if status == "statistical_unavailable" else None
    return {
        "comparisonID": comparison_id,
        "status": status,
        "reason": reason,
        "inputP": input_p,
        "orientationP": orientation_p,
        "pMagnitude": magnitude_p,
        "nominalReject": nominal_reject,
    }


def _holm_four(p_values):
    """Monotone Holm adjusted p values, with all four positions retained."""
    adjusted = [None] * 4
    running = 0.0
    for rank, index in enumerate(sorted(range(4), key=lambda index: p_values[index])):
        running = max(running, min(1.0, (4 - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compose_family(records):
    """Compose the exact fixed four-item family; return a new result object.

    Statistical-unavailable comparisons retain a Holm padding value of one, but
    their public raw and adjusted p values remain None. Technical-unresolved input
    makes the family result unknown; it is never silently padded as a non-rejection.
    Available comparisons can reject without all four comparisons being available.
    """
    if type(records) not in (list, tuple) or len(records) != 4:
        raise ValueError("Family must contain exactly four comparison records")
    tests = {}
    for record in records:
        result = compose_comparison(record)
        comparison_id = result["comparisonID"]
        if comparison_id in tests:
            raise ValueError("Duplicate comparisonID")
        tests[comparison_id] = result
    if set(tests) != set(COMPARISON_IDS):
        raise ValueError("Family must contain the fixed four comparisonIDs")
    tests = {comparison_id: tests[comparison_id] for comparison_id in COMPARISON_IDS}
    available = sum(test["status"] == "available" for test in tests.values())
    unavailable = sum(test["status"] == "statistical_unavailable" for test in tests.values())
    unresolved = sum(test["status"] == "technical_unresolved" for test in tests.values())
    if unresolved:
        for test in tests.values():
            test.update(holmPaddingP=None, holmP=None, holmReject=None)
        any_rejects = None
    else:
        padded = [
            test["pMagnitude"] if test["status"] == "available" else 1.0
            for test in tests.values()
        ]
        adjusted = _holm_four(padded)
        for index, test in enumerate(tests.values()):
            is_available = test["status"] == "available"
            test["holmPaddingP"] = padded[index]
            test["holmP"] = adjusted[index] if is_available else None
            test["holmReject"] = is_available and adjusted[index] < ALPHA
        any_rejects = any(test["holmReject"] for test in tests.values())
    return {
        "schema": "h10-magnitude-family-candidate-v1",
        "methodID": METHOD_ID,
        "scope": "p_value_composition_only",
        "scientificAdoption": False,
        "constituentValidityEstablished": False,
        "validityCondition": "Error control is conditional on valid constituent one-sided p values.",
        "familySize": 4,
        "alpha": ALPHA,
        "comparisonIDs": list(COMPARISON_IDS),
        "status": "technical_unresolved" if unresolved else "complete",
        "holmComputed": not bool(unresolved),
        "completeFamilyAvailable": available == 4,
        "counts": {
            "planned": 4,
            "technicalCompleted": 4 - unresolved,
            "available": available,
            "statisticalUnavailable": unavailable,
            "technicalUnresolved": unresolved,
        },
        "tests": tests,
        "anyMagnitudeRejects": any_rejects,
    }
