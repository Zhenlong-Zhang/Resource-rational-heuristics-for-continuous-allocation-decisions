"""Fixed native-state marginal calibration with resumable 250-attempt batches.

Normal, native finite-state and selection panels retain distinct populations.
Exact rational truths, not rounded displays, determine coverage and null status.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import struct
import sys

from . import nominal_inference as inference
from .expected_outcomes import allocation_metrics, binary_payload, digest, finite, integer

ARMS = ("C1", "C2", "C3", "C4", "C5")
FAMILY_ENDPOINT = {"Q": "Q", "R": "R_EGPA_pp", "L": "L_lower_mean_share", "initial": "initial_slope"}


def requests(analysis_spec, calibration_spec):
    inference.validate_spec(analysis_spec); validate_calibration(calibration_spec)
    result = []
    for spec in calibration_spec["cells"]:
        for subcase in spec["subcases"]:
            cell = {**spec, "subcase": subcase, "analysisSpec": analysis_spec, "calibrationSpec": calibration_spec}
            if spec["id"] == "B07":
                cell["population"] = _population(cell)
            result.append(cell)
    return result


def validate_calibration(spec):
    if (spec.get("schema") != "native-calibration-spec-v1" or spec.get("attemptsPerSubcase") != 10000
            or spec.get("batchSize") != 250 or spec.get("choicePrefixCycle") != [0, 1, 3]
            or spec.get("C1SyntheticGapControls") != [0, .2, .5, 1]):
        raise ValueError("Frozen calibration contract changed")
    expected = {**{f"A{i:02d}": 20261011100 + i for i in range(1, 7)},
                **{f"B{i:02d}": 20261011110 + i for i in range(1, 13)},
                **{f"C{i:02d}": 20261011130 + i for i in range(1, 5)}}
    if {c["id"]: c["seed"] for c in spec["cells"]} != expected:
        raise ValueError("Fixed cell/seed roster changed")
    return spec


def _native_Q_expectation(q, rho):
    from scipy.stats import binom, betabinom
    if rho == 0:
        probabilities = binom.pmf(range(11), 10, q)
    else:
        k = 1 / rho - 1
        probabilities = betabinom.pmf(range(11), 10, q * k, (1 - q) * k)
    return math.fsum((k / 10) * float(p) for k, p in enumerate(probabilities))


def _specs_for(cell, analysis):
    family = cell.get("family", cell.get("subcase"))
    if cell["layer"] == "A":
        return analysis["claims"][:2] if cell["id"] in ("A03", "A04") or cell.get("subcase") == "separate_pooled_reference" else analysis["claims"]
    if cell["id"] == "B07":
        return analysis["diagnostics"]
    return [c for c in analysis["claims"] if c["endpoint"] == FAMILY_ENDPOINT[family]]


def analytic_truth(cell_spec, lookup_receipt):
    """No response draws; independently identified population and test subset."""
    cell = cell_spec
    analysis = cell["analysisSpec"]
    specs = _specs_for(cell, analysis)
    family = cell.get("family", cell.get("subcase"))
    population = cell.get("population", "working")
    targets = []
    for spec in specs:
        ideal, native = 0., 0.
        original = 0.
        classification = "boundary_null"
        if cell["id"] == "C04":
            ideal = native = original = None
            classification = "not_derived_diagnostic_only"
        elif cell["layer"] == "A":
            sign = 1 if spec["tail"] == "greater" else -1
            if cell["id"] == "A05":
                ideal = native = .3 * sign
                classification = "correct_direction_alternative"
            elif cell["id"] == "A06":
                if cell["subcase"] == "opposite_direction":
                    ideal = native = -.3 * sign
                    classification = "composite_null_opposite_direction"
                elif spec["id"] in ("H01_Q_C3_minus_C2", "H03_initial_C2", "H06_R_EGPA_C2"):
                    ideal = native = .3 * sign
                    classification = "correct_direction_alternative"
            original = ideal
        elif family == "Q" and cell["id"] != "C03":
            arm_means = {}
            for arm in spec["weights"]:
                rho = cell.get("armOverrides", {}).get(arm, {}).get("rho", cell.get("rho", .3))
                arm_means[arm] = _native_Q_expectation(cell.get("q", .3), rho)
            native = math.fsum(spec["weights"][a] * arm_means[a] for a in arm_means)
            if native != 0:
                classification = "native_floating_nonzero_not_boundary_null"
        elif family == "initial":
            law = ("selected_uniform" if cell["id"] == "C03" else "centered_beta_mirror" if cell["id"] == "B12" and cell["subcase"] == "sign_mirror"
                   else "centered_beta" if cell["id"] == "B12" else "symmetric_uniform" if cell["id"] in ("B10", "C01", "C02") else "zero")
            values, ideals = {}, {}
            for arm in spec["weights"]:
                candidate = lookup_receipt["nativeSlopeTruthCandidates"][population + "|" + arm]
                if candidate.get("status") == "structurally_unavailable":
                    values[arm] = ideals[arm] = None
                else:
                    truth = candidate[law]
                    values[arm], ideals[arm] = truth["nativeFloatingExpectation"], truth["idealAlgebraExpectation"]
            if any(v is None for v in values.values()):
                native = ideal = None
                classification = "structurally_unavailable"
            else:
                native = math.fsum(spec["weights"][a] * values[a] for a in values)
                ideal = math.fsum(spec["weights"][a] * ideals[a] for a in ideals)
                classification = "selected_nonzero_effect" if cell["id"] == "C03" else ("boundary_null" if native == 0 else "native_floating_nonzero_not_boundary_null")
        elif family == "R":
            law = "selected_uniform" if cell["id"] == "C03" else "centered_beta" if cell["id"] == "B11" else "symmetric_beta_" + str(cell.get("rho", .3))
            native = math.fsum(w * lookup_receipt["nativeRTruthCandidates"][population + "|" + a][law]["nativeFloatingExpectation"] for a, w in spec["weights"].items())
            if cell["id"] == "C03":
                p = .1 + .8 * (1 + .75) / 2
                ideal = math.fsum(w * (2 * p - 1) * lookup_receipt["geometry"][population + "|" + a]["D"] for a, w in spec["weights"].items())
                classification = "selected_nonzero_effect"
            elif native != 0:
                classification = "native_floating_nonzero_not_boundary_null"
        # Q selected arm means change together, but designated contrasts stay 0.
        targets.append({"testId": spec["id"], "endpointId": spec["endpoint"], "originalPopulationTruth": original,
                        "selectedIdealAlgebraTruth": ideal, "selectedNativeFloatingTruth": native,
                        "classification": classification, "typeIEligible": classification in ("boundary_null", "composite_null_opposite_direction"),
                        "coverageEligible": native is not None, "primaryNullUnchanged": 0})
    return {"schema": "native-analytic-truth-v1", "cellId": cell["id"], "subcase": cell["subcase"],
            "lookupHash": lookup_receipt["lookupHash"], "targets": targets,
            "population": "original_generated" if cell["id"] not in ("C01", "C02", "C03", "C04") else "eligible_selected_complete_ten",
            "ownLatentCutoff": .75 if cell["id"] in ("C03", "C04") else None,
            "QSelectedArmMean": .1 + .8 * (1 + .75) / 2 if cell["id"] == "C03" and family == "Q" else None,
            "nativeTruthAcceptance": "requires_independent_truth_receipt", "humanObservations": False}


def _seed(cell, batch):
    # B12 is a paired sign mirror, not an independent replication stream.
    subcase = "forward" if cell["id"] == "B12" else cell["subcase"]
    value = f'{cell["seed"]}|{cell["id"]}|{subcase}|batch:{batch}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:16], "big")


def _population(cell):
    return cell.get("subcase", "working") if cell["id"] == "B07" and cell["subcase"].startswith("gap_") else "working"


def _schedule(lookup, population, arm, variant, prefix_override=None):
    result = []
    for i in range(1, 11):
        prefix = prefix_override if prefix_override is not None else (0, 1, 3)[(i - 1) % 3]
        key = lookup["index"]["|".join(map(str, (population, arm, variant, i, prefix)))]
        result.append((key, lookup["states"][key]["metrics"]))
    return result


def _beta_probability(rng, shape, q, rho):
    import numpy as np
    if rho == 0:
        return np.full(shape, q)
    k = 1 / rho - 1
    return rng.beta(q * k, (1 - q) * k, size=shape)


def _normal_batch(cell, specs, rng, count):
    import numpy as np
    endpoints = {arm: {} for arm in ARMS}
    n_default = cell["n"]
    for arm in ARMS:
        other = arm in ("C2", "C4") and cell["id"] in ("A03", "A04")
        if cell["subcase"] == "variance_count_mirror":
            other = not other
        n = cell.get("otherN", n_default) if other else n_default
        sd = cell.get("otherSd", cell["sd"]) if other else cell["sd"]
        shared = rng.normal(size=(count, n))
        for endpoint in ("Q", "initial_slope", "R_EGPA_pp", "L_lower_mean_share"):
            innovation = rng.normal(size=(count, n))
            noise = (math.sqrt(.5) * shared + math.sqrt(.5) * innovation) if cell["id"] == "A02" else innovation
            shift = 0.
            direction = 1 if cell["subcase"] != "opposite_direction" else -1
            alternatives = cell["id"] in ("A05", "A06")
            mixed = cell["subcase"] == "mixed_H01_H03_H06"
            if alternatives:
                if endpoint == "Q":
                    shift = (.3 if arm == "C3" else 0) if mixed else (.15 if arm in ("C3", "C5") else -.15 if arm in ("C2", "C4") else 0) * direction
                elif endpoint == "initial_slope":
                    shift = (.3 if arm == "C2" else 0) if mixed else (.3 if arm in ("C2", "C3") else -.3 if arm == "C4" else 0) * direction
                elif endpoint == "R_EGPA_pp":
                    shift = (.3 if arm == "C2" else 0) if mixed else (.3 if arm in ("C2", "C3") else 0) * direction
                elif endpoint == "L_lower_mean_share" and not mixed:
                    shift = (.3 if arm in ("C4", "C5") else 0) * direction
            endpoints[arm][endpoint] = shift + sd * noise
            if not np.isfinite(endpoints[arm][endpoint]).all():
                raise ValueError("Unexpected nonfinite abstract endpoint")
    return endpoints, None


def _native_batch(cell, specs, lookup, rng, count):
    import numpy as np
    family, population = cell.get("family", cell["subcase"]), _population(cell)
    if cell["layer"] == "C":
        family = cell["subcase"]
    endpoint = FAMILY_ENDPOINT[family]
    output, audit = {}, {}
    for arm in ARMS if cell["id"] != "B07" else ("C1",):
        overrides = cell.get("armOverrides", {}).get(arm, {})
        n = overrides.get("n", cell["n"])
        variants = rng.integers(0, 9, size=(count, n))
        U = rng.uniform(size=(count, n))
        selected = np.ones((count, n), dtype=bool)
        if cell["id"] == "C01":
            selected &= rng.uniform(size=(count, n)) < .8
        if cell["id"] == "C03":
            selected &= U > .75
        positive_cycle = cell["id"] != "C04" and family != "Q"
        tables = [_schedule(lookup, population, arm, v, None if positive_cycle else 0) for v in range(9)]
        x = np.asarray([[r["x0"] for _, r in schedule] for schedule in tables])[variants]
        M = lookup["geometry"][population + "|" + arm]["M"]
        if family == "initial":
            if cell["id"] in ("C03", "C04"):
                beta = .25 * (U - .5) / M if M > 0 else np.zeros_like(U)
            elif cell["id"] in ("B10", "C01", "C02"):
                beta = rng.uniform(-.25 / M, .25 / M, size=(count, n)) if M > 0 else np.zeros_like(U)
            elif cell["id"] == "B12":
                u = rng.beta(.5, 5, size=(count, n)); beta = .25 * (u - .5 / 5.5) / M if M > 0 else np.zeros_like(U)
                if cell["subcase"] == "sign_mirror":
                    beta = -beta
            else:
                beta = np.zeros_like(U)
            probability = .5 + beta[:, :, None] * x
        else:
            if cell["id"] in ("C03", "C04"):
                p = .1 + .8 * U
            elif cell["id"] == "B11":
                p = .5 + .25 * (rng.beta(.5, 5, size=(count, n)) - .5 / 5.5)
            else:
                q = cell.get("q", .3 if family == "Q" else .5)
                rho = overrides.get("rho", cell.get("rho", .3))
                p = _beta_probability(rng, (count, n), q, rho)
            probability = np.broadcast_to(p[:, :, None], (count, n, 10))
        if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
            raise ValueError("Prospective probability bound failed; no clipping allowed")
        response_uniform = rng.uniform(size=(count, n, 10))
        if cell["id"] == "B12" and cell["subcase"] == "sign_mirror":
            response_uniform = 1 - response_uniform
        response = response_uniform < probability
        record_present = np.ones((count, n, 10), dtype=bool)
        endpoint_present = np.ones_like(record_present)
        if cell["id"] == "C01":
            record_present &= rng.uniform(size=record_present.shape) < .98
        if cell["id"] == "C02":
            endpoint_present &= rng.uniform(size=record_present.shape) >= .1
        # Build native j/index tables; targets were evaluated once, not per draw.
        j_eo = np.asarray([[r["jEGPA"] for _, r in schedule] for schedule in tables], dtype=int)[variants]
        means = np.asarray([[r["terminalMeans"] for _, r in schedule] for schedule in tables])[variants]
        shares = np.where(response, j_eo, 500) if family == "R" else response.astype(int) * 1000
        if family == "Q":
            shares = np.full(response.shape, 500, dtype=int)
        exhausted = np.zeros((count, n), dtype=bool)
        if cell["id"] == "C04":
            divisible = np.asarray([schedule[0][1]["initialIntegerBudget"] % schedule[0][1]["paidIntegerCost"] == 0 for schedule in tables])[variants]
            exhausted = (U > .75) & divisible
        eligible = selected & record_present.all(axis=2)
        if family != "Q":
            eligible &= endpoint_present.all(axis=2) & ~exhausted
        if family == "Q":
            values = response.astype(float)
            values[:, :, 0] = np.where(exhausted, 1., values[:, :, 0])
            person_values = inference.canonical_mean(values)
        elif family == "R":
            values = (abs(shares - 500) - abs(shares - j_eo)) / 10
            person_values = inference.canonical_mean(values)
        elif family == "L":
            diff = means[:, :, :, 0] - means[:, :, :, 1]
            tied = abs(diff) <= 1e-9 * np.maximum(1., abs(means).max(axis=3))
            values = np.where(tied, 0., (shares - 500) / 1000 * np.where(diff < 0, 1, -1))
            person_values = inference.canonical_mean(values)
        else:
            supported_variants = lookup["geometry"][population + "|" + arm]["supportedVariants"]
            eligible &= np.isin(variants, supported_variants)
            # Fixed linear projection matrix for each schedule. Scalar API parity
            # is audited at frozen replicate/person positions, not presumed.
            weights = []
            for variant in range(9):
                plan = lookup["geometry"][population + "|" + arm]["projectionPlans"].get(str(variant))
                if plan is None:
                    weights.append(np.zeros(10))
                else:
                    weights.append([struct.unpack(">d", bytes.fromhex(bits))[0] for bits in plan["weightBinary64"]])
            person_values = inference.canonical_projection(np.asarray(weights)[variants], shares / 1000)
        if (eligible & ~np.isfinite(person_values)).any():
            raise ValueError("Unexpected nonfinite supported native endpoint")
        person_values = np.where(eligible, person_values, np.nan)
        output[arm] = {endpoint: person_values}
        audit[arm] = {"variants": variants, "shares": shares, "response": response, "recordPresent": record_present,
                      "endpointPresent": endpoint_present, "selected": selected, "exhausted": exhausted,
                      "tables": tables, "family": family, "population": population}
    return output, audit


def _vector_contrast(endpoints, spec, pooled=False):
    import numpy as np
    from scipy.stats import t
    arrays = [endpoints[a][spec["endpoint"]] for a in spec["weights"]]
    weights = np.asarray(list(spec["weights"].values()))
    counts = np.asarray([np.isfinite(a).sum(axis=1) for a in arrays])
    sums = np.asarray([np.where(np.isfinite(a), a, 0).sum(axis=1) for a in arrays])
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.asarray([np.where(np.isfinite(a), (a - m[:, None]) ** 2, 0).sum(axis=1) for a, m in zip(arrays, means)])
    variances = np.divide(variances, counts - 1, out=np.zeros_like(variances), where=counts > 1)
    effect = (weights[:, None] * means).sum(axis=0)
    if pooled:
        if len(arrays) != 2 or list(spec["weights"].values()) != [1, -1]:
            raise ValueError("Pooled reference is separate balanced two-arm only")
        if not (counts[0] == counts[1]).all():
            raise ValueError("Separate pooled reference requires balanced eligible counts")
        df = counts.sum(axis=0) - 2
        sp2 = ((counts - 1) * variances).sum(axis=0) / df
        se2 = sp2 * (1 / counts).sum(axis=0)
    else:
        parts = np.divide(weights[:, None] ** 2 * variances, counts, out=np.zeros_like(variances), where=counts > 0)
        se2 = parts.sum(axis=0)
        denominator = np.divide(parts ** 2, counts - 1, out=np.zeros_like(parts), where=counts > 1).sum(axis=0)
        df = np.divide(se2 ** 2, denominator, out=np.full_like(se2, np.nan), where=denominator > 0)
    available = (counts >= 2).all(axis=0) & np.isfinite(se2) & (se2 > 0) & np.isfinite(df) & (df > 0)
    se = np.sqrt(se2)
    statistic = np.divide(effect, se, out=np.full_like(effect, np.nan), where=available)
    p = t.sf(statistic, df) if spec["tail"] == "greater" else t.cdf(statistic, df) if spec["tail"] == "less" else 2 * t.sf(abs(statistic), df)
    radius = t.ppf(.975, df) * se
    available &= np.isfinite(p) & np.isfinite(radius)
    unexpected = (counts >= 2).all(axis=0) & (~np.isfinite(effect) | ~np.isfinite(se2)
                 | ((se2 > 0) & (~np.isfinite(df) | ~np.isfinite(p) | ~np.isfinite(radius))))
    return {"effect": effect, "SE": se, "df": df, "statistic": statistic, "p": p,
            "low": effect - radius, "high": effect + radius, "available": available,
            "reject": available & (p < .05), "counts": counts, "unexpectedNumericalFailure": unexpected}


def _scalar_pooled_reference(people, spec):
    """Independent scalar arithmetic for the separate Normal reference only."""
    from scipy.stats import t
    arms = list(spec["weights"])
    if len(arms) != 2 or list(spec["weights"].values()) != [1, -1]:
        raise ValueError("Separate balanced pooled reference only")
    values = [[p["endpoints"][spec["endpoint"]]["value"] for p in people
               if p["conditionId"] == arm and p["endpoints"][spec["endpoint"]]["status"] == "available"] for arm in arms]
    n = len(values[0])
    if n != len(values[1]) or n < 2:
        raise ValueError("Separate pooled reference requires balanced counts at least two")
    means = [math.fsum(xs) / n for xs in values]
    pooled = math.fsum(math.fsum((v - mean) ** 2 for v in xs) for xs, mean in zip(values, means)) / (2 * n - 2)
    effect, se2, df = means[0] - means[1], pooled * (2 / n), 2 * n - 2
    if se2 <= 0 or not finite(se2):
        return {"status": "unavailable", "candidateNominalReject": False}
    se = math.sqrt(se2); statistic = effect / se
    p = float(t.sf(statistic, df))
    return {"status": "available", "effect": effect, "SE": se, "df": df, "statistic": statistic,
            "p": p, "candidateNominalReject": p < .05}


def _audit(cell, endpoints, audit, lookup, analysis, origin, offsets, specs):
    import numpy as np
    audits = []
    for offset in offsets:
        if audit is None:
            people = []
            for arm, per_endpoint in endpoints.items():
                n = next(iter(per_endpoint.values())).shape[1]
                for person in range(n):
                    people.append({"participantId": f"abstract-{arm}-{person}", "conditionId": arm,
                                   "endpoints": {e: {"status": "available", "value": float(a[offset, person])} for e, a in per_endpoint.items()}})
        else:
            draws = []
            for arm, info in audit.items():
                for person, variant in enumerate(info["variants"][offset]):
                    if not info["selected"][offset, person]:
                        continue
                    schedule = info["tables"][int(variant)]
                    for i, (key, row) in enumerate(schedule):
                        if info["family"] == "Q":
                            prefix = int(info["response"][offset, person, i])
                            key = lookup["index"]["|".join(map(str, (info["population"], arm, int(variant), i + 1, prefix)))]
                        j = int(info["shares"][offset, person, i])
                        if i == 0 and info["exhausted"][offset, person]:
                            prefix = row["initialIntegerBudget"] // row["paidIntegerCost"]
                            key = lookup["index"]["|".join(map(str, (info["population"], arm, int(variant), 1, prefix)))]
                            j = None
                        draws.append({"stateKey": key, "participantId": f"native-{arm}-{person}", "sessionId": f"native-{arm}-{person}",
                                      "shareIndex": j, "recordPresent": bool(info["recordPresent"][offset, person, i]),
                                      "endpointPresent": bool(info["endpointPresent"][offset, person, i])})
            rows = make_synthetic_rows(lookup, draws, origin)
            people = inference.participant_endpoints(rows, analysis)
            # Explicit scalar support/value readback for first/middle/last people.
            for arm, per_endpoint in endpoints.items():
                a = next(iter(per_endpoint.values())); n = a.shape[1]
                family_endpoint = FAMILY_ENDPOINT[audit[arm]["family"]]
                by_id = {p["participantId"]: p for p in people}
                for person in sorted({0, n // 2, n - 1}):
                    value = a[offset, person]
                    p = by_id.get(f"native-{arm}-{person}")
                    scalar = None if p is None else p["endpoints"][family_endpoint]
                    supported = scalar is not None and scalar["status"] == "available"
                    if supported != bool(np.isfinite(value)):
                        raise ValueError("Scalar/vector person availability disagreement")
                    if supported and struct.pack(">d", float(value)) != struct.pack(">d", scalar["value"]):
                        raise ValueError("Scalar/vector person endpoint binary64-map disagreement")
        for spec in specs:
            pooled = cell["subcase"] == "separate_pooled_reference"
            # Reference is a separately labelled audit, never a fallback for
            # selected Student/Welch. It is not silently left unaudited.
            scalar = _scalar_pooled_reference(people, spec) if pooled else inference.test_contrast(people, spec)
            vector = _vector_contrast(endpoints, spec, pooled)
            available = bool(vector["available"][offset])
            if available != (scalar["status"] == "available"):
                raise ValueError("Scalar/vector test availability disagreement")
            if available:
                for name in ("effect", "SE", "df", "statistic", "p"):
                    relative = 1e-9 if name == "effect" and spec["endpoint"] == "initial_slope" and abs(scalar[name]) > 1 else 0
                    if not math.isclose(float(vector[name][offset]), scalar[name], rel_tol=relative, abs_tol=1e-10):
                        raise ValueError("Scalar/vector test parity failed: " + name)
                if bool(vector["reject"][offset]) != scalar["candidateNominalReject"]:
                    raise ValueError("Scalar/vector decision parity failed")
            audits.append({"offset": offset, "testId": spec["id"], "available": available, "status": "PASS",
                           "route": "separate_pooled_reference" if pooled else "selected_Student_or_Welch"})
    return audits


def _cp(successes, trials, confidence=.95):
    from scipy.stats import beta
    if trials == 0:
        return None
    alpha = 1 - confidence
    return [0. if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1)),
            1. if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))]


def _exact_ratio(value):
    """Reduced, canonical decimal-string rational; never a rounded display zero."""
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("Exact native expectation requires numerator/denominator strings")
    numerator, denominator = value["numerator"], value["denominator"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise ValueError("Rational identities must be decimal strings")
    n, d = int(numerator), int(denominator)
    if str(n) != numerator or str(d) != denominator or d <= 0:
        raise ValueError("Noncanonical rational identity")
    ratio = Fraction(n, d)
    if ratio.numerator != n or ratio.denominator != d or not finite(float(ratio)):
        raise ValueError("Rational must be reduced with finite display conversion")
    return ratio


def _projection_hashes(cell, spec, lookup):
    if cell["layer"] == "A" or spec["endpoint"] != "initial_slope":
        return {}
    population = cell.get("population", "working")
    result = {}
    for arm in spec["weights"]:
        geometry = lookup["geometry"][population + "|" + arm]
        result[arm] = {str(v): geometry["projectionPlans"][str(v)]["planHash"] for v in geometry["supportedVariants"]}
    return result


def _validate_truth_bundle(cell, lookup, bundle):
    """Candidate identity and independent exact truth are different objects."""
    if bundle.get("schema") != "native-truth-bundle-v1":
        raise ValueError("Separate producer candidate and independent truth required")
    candidate, accepted = bundle["candidate"], bundle["accepted"]
    if analytic_truth(cell, lookup["receipt"]) != candidate:
        raise ValueError("Source-bound producer candidate differs before draws")
    expected_header = {"schema": "native-accepted-native-truth-v1", "cellId": cell["id"], "subcase": cell["subcase"],
        "lookupHash": lookup["receipt"]["lookupHash"], "analysisSpecHash": digest(cell["analysisSpec"]),
        "calibrationSpecHash": digest(cell["calibrationSpec"]), "candidateCellHash": digest(candidate),
        "arithmeticMapHash": digest(lookup["arithmeticMaps"])}
    if any(accepted.get(k) != v for k, v in expected_header.items()):
        raise ValueError("Independent truth cell/map/source identity mismatch")
    specs = _specs_for(cell, cell["analysisSpec"])
    if [t.get("testId") for t in accepted.get("targets", [])] != [s["id"] for s in specs]:
        raise ValueError("Independent truth target subset/order changed")
    law = ("declared_decimal_ideal_normal_reference_law" if cell["layer"] == "A" else "declared_decimal_ideal_probability_law") + ";not_literal_finite_PRNG_or_probability_expression_rounding"
    for spec, proposed, truth in zip(specs, candidate["targets"], accepted["targets"]):
        if (truth.get("endpointId") != spec["endpoint"] or truth.get("candidateTargetHash") != digest(proposed)
                or truth.get("projectionPlanHashes") != _projection_hashes(cell, spec, lookup)
                or truth.get("probabilityLawScope") != law):
            raise ValueError("Independent truth target/law/frozen-plan mismatch")
        if "idealAlgebraTruth" not in truth or (truth["idealAlgebraTruth"] is not None and not finite(truth["idealAlgebraTruth"])):
            raise ValueError("Separate finite/null ideal algebra truth required")
        status = truth.get("truthStatus")
        if status in ("numerically_unresolved", "diagnostic_not_derived"):
            if (truth.get("exactNativeExpectation") is not None or truth.get("numericalValue") is not None
                    or truth.get("typeIEligible") is not False or truth.get("coverageEligible") is not False):
                raise ValueError("Unresolved/diagnostic truth cannot certify Type I or coverage")
            expected_class = "not_derived_diagnostic_only" if status == "diagnostic_not_derived" else "numerically_unresolved"
            if truth.get("classification") != expected_class or (status == "diagnostic_not_derived") != (cell["id"] == "C04"):
                raise ValueError("Diagnostic/unresolved truth classification mismatch")
            continue
        expected_status = ("exact_rational_abstract_reference_mean_under_declared_ideal_normal_law" if cell["layer"] == "A"
                           else "exact_rational_native_endpoint_under_declared_ideal_probability_law")
        if status != expected_status or cell["id"] == "C04":
            raise ValueError("Unreleased independent truth status")
        ratio = _exact_ratio(truth.get("exactNativeExpectation"))
        if not finite(truth.get("numericalValue")) or struct.pack(">d", truth["numericalValue"]) != struct.pack(">d", float(ratio)):
            raise ValueError("Truth display is not the declared nearest-binary64 rational conversion")
        if cell["layer"] == "A":
            # Layer A's abstract mean is declared exactly, unlike native-map
            # integration candidates. Validate magnitude BEFORE zero/null logic.
            sign = 1 if spec["tail"] == "greater" else -1
            expected = Fraction(0)
            if cell["id"] == "A05":
                expected = Fraction(3, 10) * sign
            elif cell["id"] == "A06":
                if cell["subcase"] == "opposite_direction":
                    expected = -Fraction(3, 10) * sign
                elif spec["id"] in ("H01_Q_C3_minus_C2", "H03_initial_C2", "H06_R_EGPA_C2"):
                    expected = Fraction(3, 10) * sign
            if ratio != expected or truth["idealAlgebraTruth"] != float(expected):
                raise ValueError("Abstract reference truth differs from its exact declared mean")
        if ratio == 0:
            classification, type_i = "boundary_null", True
        elif cell["layer"] == "A":
            classification = proposed["classification"]
            sign = 1 if spec["tail"] == "greater" else -1
            if classification not in ("correct_direction_alternative", "composite_null_opposite_direction") or ((ratio * sign > 0) != (classification == "correct_direction_alternative")):
                raise ValueError("Abstract reference exact truth changes the declared direction")
            type_i = classification == "composite_null_opposite_direction"
        else:
            classification, type_i = ("selected_nonzero_effect" if cell["id"] == "C03" else "exact_native_nonzero_not_boundary_null"), False
        if truth.get("classification") != classification or truth.get("typeIEligible") is not type_i or truth.get("coverageEligible") is not True:
            raise ValueError("Exact rational null/coverage classification mismatch")
    return candidate, accepted


def _coverage_mask(result, truth):
    """Exact closed-CI coverage; rounded display truth never decides a boundary."""
    import numpy as np
    covered = np.zeros(result["available"].shape, dtype=bool)
    if not truth["coverageEligible"]:
        return covered
    actual = _exact_ratio(truth["exactNativeExpectation"])
    for offset in np.flatnonzero(result["available"]):
        lower = Fraction.from_float(float(result["low"][offset]))
        upper = Fraction.from_float(float(result["high"][offset]))
        covered[offset] = lower <= actual <= upper
    return covered


def _generated_arm_size(cell, arm):
    """Declared pre-selection size; mirror the frozen generator shape rules only."""
    if cell["layer"] == "A":
        other = arm in ("C2", "C4") and cell["id"] in ("A03", "A04")
        if cell["subcase"] == "variance_count_mirror":
            other = not other
        size = cell.get("otherN", cell["n"]) if other else cell["n"]
    else:
        size = cell.get("armOverrides", {}).get(arm, {}).get("n", cell["n"])
    return integer(size, "declared generated arm size", 1)


def _validate_completed_batch(record, cell, specs, truth_by_id, binding_hash, cell_hash):
    """Reject malformed fresh/resumed summaries, without changing their contents.

    These necessary schema/count/audit checks do not establish authenticity or
    reconstruct per-replication denominators. External raw hashes and independent
    scalar replay remain required alongside the genuine execution dependency.
    """
    fields = {"schema", "bindingHash", "cellHash", "batchId", "seed", "attempts",
              "status", "targets", "scalarAudits", "contentHash"}
    if type(record) is not dict or set(record) != fields:
        raise ValueError("Completed batch has an unsupported field schema")
    if (record["schema"] != "native-calibration-completed-batch-v1" or record["status"] != "complete"
            or record["bindingHash"] != binding_hash or record["cellHash"] != cell_hash):
        raise ValueError("Incompatible/incomplete completed batch")
    batch_id = integer(record["batchId"], "completed batch ID", 0, 39)
    integer(record["attempts"], "completed batch attempts", 250, 250)
    if record["seed"] != str(_seed(cell, batch_id)):
        raise ValueError("Completed batch seed changed")
    target_ids = [spec["id"] for spec in specs]
    if type(record["targets"]) is not dict or set(record["targets"]) != set(target_ids):
        raise ValueError("Completed batch target roster changed")
    target_fields = {"attempts", "available", "rejections", "coveredAvailableIntervals",
                     "requiredArmCountsMinimum", "requiredArmCountsMaximum", "unexpectedNumericalFailures"}
    for spec in specs:
        target = record["targets"][spec["id"]]
        if type(target) is not dict or set(target) != target_fields:
            raise ValueError("Completed target has an unsupported field schema")
        integer(target["attempts"], "completed target attempts", 250, 250)
        available = integer(target["available"], "completed target available", 0, 250)
        integer(target["rejections"], "completed target rejections", 0, available)
        integer(target["unexpectedNumericalFailures"], "completed target numerical failures", 0, 0)
        coverage_eligible = truth_by_id[spec["id"]]["coverageEligible"]
        if type(coverage_eligible) is not bool:
            raise ValueError("Completed target needs a literal accepted coverage flag")
        if coverage_eligible:
            integer(target["coveredAvailableIntervals"], "completed target coverage", 0, available)
        elif target["coveredAvailableIntervals"] is not None:
            raise ValueError("Ineligible completed target coverage must be null")
        minima, maxima = target["requiredArmCountsMinimum"], target["requiredArmCountsMaximum"]
        arms = list(spec["weights"])
        if (type(minima) is not list or type(maxima) is not list
                or len(minima) != len(arms) or len(maxima) != len(arms)):
            raise ValueError("Completed target arm-count array shape changed")
        for arm, minimum, maximum in zip(arms, minima, maxima):
            size = _generated_arm_size(cell, arm)
            integer(minimum, "completed arm minimum", 0, size)
            integer(maximum, "completed arm maximum", minimum, size)
            if cell["layer"] == "A" and (minimum != size or maximum != size):
                raise ValueError("Abstract Normal completed arm counts must equal generated size")
            if (available > 0 and maximum < 2) or (available == 250 and minimum < 2):
                raise ValueError("Completed target availability contradicts arm counts")
    audits = record["scalarAudits"]
    audited_batches = batch_id in (0, 39)
    offsets = (0, 1, 17, 249) if audited_batches else ()
    if type(audits) is not list or len(audits) != len(offsets) * len(specs):
        raise ValueError("Completed batch scalar-audit roster changed")
    route = "separate_pooled_reference" if cell["subcase"] == "separate_pooled_reference" else "selected_Student_or_Welch"
    audit_counts = {test_id: [0, 0] for test_id in target_ids}
    for audit, (offset, spec) in zip(audits, ((offset, spec) for offset in offsets for spec in specs)):
        if type(audit) is not dict or set(audit) != {"offset", "testId", "available", "status", "route"}:
            raise ValueError("Completed scalar audit has an unsupported field schema")
        integer(audit["offset"], "completed scalar-audit offset", 0, 249)
        if (audit["offset"] != offset or audit["testId"] != spec["id"]
                or type(audit["available"]) is not bool or audit["status"] != "PASS" or audit["route"] != route):
            raise ValueError("Completed scalar audit order/identity/status changed")
        audit_counts[spec["id"]][0 if audit["available"] else 1] += 1
    for test_id, (audited_available, audited_unavailable) in audit_counts.items():
        available = record["targets"][test_id]["available"]
        if available < audited_available or 250 - available < audited_unavailable:
            raise ValueError("Completed scalar audits contradict target availability")
    if record["contentHash"] != digest({k: v for k, v in record.items() if k != "contentHash"}):
        raise ValueError("Completed batch content changed")
    return record


def _pending_batch_ids(cell, records):
    """Validate every declared request before membership or batch-limit filtering."""
    requested = cell.get("onlyBatchIds", list(range(40)))
    if type(requested) is not list:
        raise ValueError("Requested batch IDs must be a literal list")
    requested = [integer(value, "requested batch ID", 0, 39) for value in requested]
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate requested batch")
    pending = [batch_id for batch_id in requested if batch_id not in records]
    return pending[:cell.get("batchLimit", len(pending))]


def run_cell(cell_spec, lookup_bundle, truth_manifest, execution_binding, completed_batches, prepared=None):
    """Return completed immutable batch records and per-target pointwise summary.

    Third object is a source-bound candidate plus separate independent exact
    truth bundle. Calling CLI checkpoints each returned batch durably.
    Use `batchLimit`/`onlyBatchIds` in the cell request for bounded resumable work;
    those select fixed IDs without changing their seeds or the 10,000-attempt goal.
    """
    import numpy as np
    from scipy.stats import nct, t
    cell = copy.deepcopy(cell_spec)
    analysis, calibration = cell["analysisSpec"], cell["calibrationSpec"]
    inference.validate_spec(analysis); validate_calibration(calibration)
    declared = [c for c in requests(analysis, calibration) if c["id"] == cell.get("id") and c["subcase"] == cell.get("subcase")]
    if len(declared) != 1 or {k: v for k, v in cell.items() if k not in ("onlyBatchIds", "batchLimit")} != declared[0]:
        raise ValueError("Cell/seed/law/count differs from the prospectively frozen request")
    if "batchLimit" in cell:
        integer(cell["batchLimit"], "fixed batch limit", 0, 40)
    _check_execution(lookup_bundle, execution_binding, analysis, calibration, truth_manifest, prepared)
    lookup = copy.copy(lookup_bundle)
    lookup["receipt"] = dict(lookup_bundle["receipt"])
    candidate_truth, accepted_truth = _validate_truth_bundle(cell, lookup, truth_manifest)
    specs = _specs_for(cell, analysis)
    records = {}
    binding_hash = digest(execution_binding)
    cell_hash = digest({k: v for k, v in cell.items() if k not in ("onlyBatchIds", "batchLimit")})
    for old in completed_batches or []:
        _validate_completed_batch(old, cell, specs,
                                  {t["testId"]: t for t in accepted_truth["targets"]}, binding_hash, cell_hash)
        batch_id = old["batchId"]
        if batch_id in records:
            raise ValueError("Duplicate resume batch")
        records[batch_id] = old
    pending = _pending_batch_ids(cell, records)
    origin = {"schema": "native-synthetic-origin-v1", "origin": "synthetic_validation", "generatorId": cell["id"] + ":" + cell["subcase"],
              "generatorHash": cell_hash, "sourceReceiptHash": lookup["receipt"]["sourceReceiptHash"],
              "analysisSpecHash": digest(analysis), "lookupHash": lookup["receipt"]["lookupHash"], "humanObservations": False}
    truth_by_id = {x["testId"]: x for x in accepted_truth["targets"]}
    candidate_by_id = {x["testId"]: x for x in candidate_truth["targets"]}
    for batch_id in pending:
        rng = np.random.default_rng(_seed(cell, batch_id))
        endpoints, audit = (_normal_batch(cell, specs, rng, 250) if cell["layer"] == "A" else _native_batch(cell, specs, lookup, rng, 250))
        summaries = {}
        for spec in specs:
            result = _vector_contrast(endpoints, spec, cell["subcase"] == "separate_pooled_reference")
            if result["unexpectedNumericalFailure"].any():
                raise ValueError("Unexpected numerical failures in fixed attempted batch: " + spec["id"])
            truth = truth_by_id[spec["id"]]
            coverage = _coverage_mask(result, truth)
            summaries[spec["id"]] = {"attempts": 250, "available": int(result["available"].sum()), "rejections": int(result["reject"].sum()),
                                      "coveredAvailableIntervals": int(coverage.sum()) if truth["coverageEligible"] else None,
                                      "requiredArmCountsMinimum": result["counts"].min(axis=1).tolist(),
                                      "requiredArmCountsMaximum": result["counts"].max(axis=1).tolist(),
                                      "unexpectedNumericalFailures": 0}
        audits = _audit(cell, endpoints, audit, lookup, analysis, origin, [0, 1, 17, 249], specs) if batch_id in (0, 39) else []
        _check_execution(lookup_bundle, execution_binding, analysis, calibration, truth_manifest, prepared)
        record = {"schema": "native-calibration-completed-batch-v1", "bindingHash": binding_hash, "cellHash": cell_hash,
                  "batchId": batch_id, "seed": str(_seed(cell, batch_id)), "attempts": 250, "status": "complete", "targets": summaries, "scalarAudits": audits}
        record["contentHash"] = digest(record)
        _validate_completed_batch(record, cell, specs, truth_by_id, binding_hash, cell_hash)
        records[batch_id] = record
    total = 250 * len(records)
    targets = []
    for spec in specs:
        truth = truth_by_id[spec["id"]]
        values = [r["targets"][spec["id"]] for r in records.values()]
        available = sum(v["available"] for v in values)
        reject = sum(v["rejections"] for v in values)
        covered = None if not truth["coverageEligible"] else sum(v["coveredAvailableIntervals"] for v in values)
        reject_ci, coverage_ci = _cp(reject, total), None if covered is None else _cp(covered, available)
        failures = []
        if truth["typeIEligible"] and reject_ci is not None and reject_ci[1] > .06:
            failures.append("pointwise_null_rejection_CP_upper_above_.06")
        if truth["coverageEligible"] and coverage_ci is not None and coverage_ci[0] < .93:
            failures.append("pointwise_available_coverage_CP_lower_below_.93")
        reference = None
        if cell["id"] == "A05" and (len(spec["weights"]) == 1 or cell["subcase"] == "separate_pooled_reference"):
            n = cell["n"]; df = n - 1 if len(spec["weights"]) == 1 else 2 * n - 2
            noncentrality = .3 * math.sqrt(n if len(spec["weights"]) == 1 else n / 2)
            reference = float(nct.sf(t.ppf(.95, df), df, noncentrality))
            if total and abs(reject / total - reference) > .02:
                failures.append("identical_normal_reference_power_difference_above_.02")
        diagnostic = truth["truthStatus"] == "diagnostic_not_derived"
        unresolved = truth["truthStatus"] == "numerically_unresolved"
        nonboundary_native = truth["classification"] == "exact_native_nonzero_not_boundary_null"
        status = "DIAGNOSTIC_ONLY" if diagnostic else "UNAVAILABLE" if available == 0 or unresolved or nonboundary_native else "MIXED" if failures else "PASS"
        type_i_status = "UNAVAILABLE" if unresolved else "not_a_declared_null" if not truth["typeIEligible"] else "UNAVAILABLE" if available == 0 else "MIXED" if reject_ci[1] > .06 else "PASS"
        coverage_status = "UNAVAILABLE" if unresolved else "not_derived" if covered is None else "UNAVAILABLE" if coverage_ci is None else "MIXED" if coverage_ci[0] < .93 else "PASS"
        candidate = candidate_by_id[spec["id"]]
        discrepancy = None if truth["numericalValue"] is None or candidate["selectedNativeFloatingTruth"] is None else truth["numericalValue"] - candidate["selectedNativeFloatingTruth"]
        targets.append({"testId": spec["id"], "truth": truth, "producerCandidate": candidate,
                        "candidateMinusIndependentDisplayDifference": None if discrepancy is None else -discrepancy,
                        "producerCandidateStatus": "floating_integration_only_not_accepted_truth",
                        "coverageComparison": "exact_rational_vs_Fraction_from_binary64_closed_CI",
                        "attempts": total, "available": available, "rejections": reject,
                        "allAttemptRejectionRate": None if total == 0 else reject / total,
                        "availableConditionalRejectionRate": None if not available else reject / available,
                        "rejectionCP95": reject_ci, "coveredAvailableIntervals": covered,
                        "availableCoverageRate": None if covered is None or not available else covered / available,
                        "availableCoverageCP95": coverage_ci, "allAttemptIntervalSuccessRate": None if covered is None or total == 0 else covered / total,
                        "exactNormalReferencePower": reference, "status": status if total == 10000 else "PROVISIONAL", "failures": failures,
                        "provisionalCriterionStatus": None if total == 10000 else status,
                        "criteriaFinal": total == 10000,
                        "typeICalibrationStatus": type_i_status if total == 10000 else "PROVISIONAL",
                        "coverageCalibrationStatus": coverage_status if total == 10000 else "PROVISIONAL",
                        "provisionalTypeICriterionStatus": None if total == 10000 else type_i_status,
                        "provisionalCoverageCriterionStatus": None if total == 10000 else coverage_status,
                        "primary": spec["primary"]})
    return {"schema": "native-cell-calibration-v1", "cellId": cell["id"], "subcase": cell["subcase"],
            "bindingHash": binding_hash, "cellHash": cell_hash, "completedBatches": [records[i] for i in sorted(records)],
            "candidateCellHash": digest(candidate_truth), "acceptedTruthCellHash": digest(accepted_truth),
            "independentTruthReceiptHash": execution_binding["nativeTruthReceiptHash"],
            "plannedAttempts": 10000, "completedAttempts": total, "complete": len(records) == 40, "targets": targets,
            "pointwiseOnly": True, "globalValidityClaim": None, "humanObservations": False,
            "historicalEvidence": "Earlier unmatched calibration/selection failures are not overturned."}

def numerical_identity():
    """Portable code/version identity, without host paths or scheduler metadata."""
    import numpy as np
    import scipy
    base = Path(__file__).parents[1]
    files = tuple("tutoring_analysis/" + name for name in
                  ("native_calibration.py", "native_lookup.py", "native_truth.py",
                   "native_reconstruction.py", "native_source.cjs", "native_engine.cjs",
                   "native_normalizer.cjs", "nominal_inference.py", "expected_outcomes.py")) + (
                       "design_validation/bounded_task.py", "design_validation/budget_horizon.py",
                       "design_validation/terminal_time.py")
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sourceSHA256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                             for name in files}}


def execution_binding(lookup, truths, analysis, calibration):
    """Bind actual public inputs; this is not a historical approval receipt."""
    wanted = {c["id"] + ":" + c["subcase"] for c in requests(analysis, calibration)}
    if set(truths) != wanted:
        raise ValueError("Complete 40-cell truth roster required")
    candidates = {key: value["candidate"] for key, value in truths.items()}
    accepted = {key: value["accepted"] for key, value in truths.items()}
    return {"schema": "native-calibration-binding-v1",
            "lookupHash": lookup["receipt"]["lookupHash"],
            "analysisSpecHash": digest(analysis), "calibrationSpecHash": digest(calibration),
            "candidateManifestHash": digest(candidates),
            "acceptedTruthManifestHash": digest(accepted),
            "candidateManifestHashes": {key: digest(value) for key, value in candidates.items()},
            "acceptedTruthManifestHashes": {key: digest(value) for key, value in accepted.items()},
            "nativeTruthReceiptHash": digest(accepted),
            "numericalIdentity": numerical_identity(), "humanObservations": False}


def _validate_inputs(lookup, binding, analysis, calibration, full=True):
    inference.validate_spec(analysis)
    validate_calibration(calibration)
    receipt = lookup.get("receipt", {})
    if (lookup.get("schema") != "native-calibration-lookup-v1"
            or receipt.get("status") != "validated"
            or receipt.get("humanObservations") is not False
            or binding.get("schema") != "native-calibration-binding-v1"
            or binding.get("humanObservations") is not False):
        raise ValueError("Validated public finite lookup and synthetic binding required")
    for name, expected in (("lookupHash", receipt.get("lookupHash")),
                           ("analysisSpecHash", digest(analysis)),
                           ("calibrationSpecHash", digest(calibration))):
        if binding.get(name) != expected or receipt.get(name) != expected:
            raise ValueError("Lookup/configuration identity differs: " + name)
    if full and digest({k: v for k, v in lookup.items() if k != "receipt"}) != receipt["lookupHash"]:
        raise ValueError("Lookup content changed")
    if full and binding.get("numericalIdentity") != numerical_identity():
        raise ValueError("Numerical code/runtime identity changed")


def _check_execution(lookup, binding, analysis, calibration, truth, prepared=None):
    if prepared is None:
        _validate_inputs(lookup, binding, analysis, calibration)
    else:
        prepared.check(lookup, binding, analysis, calibration)
    key = truth["candidate"]["cellId"] + ":" + truth["candidate"]["subcase"]
    if (binding.get("candidateManifestHashes", {}).get(key) != digest(truth["candidate"])
            or binding.get("acceptedTruthManifestHashes", {}).get(key) != digest(truth["accepted"])):
        raise ValueError("Candidate or independent truth content changed")


def make_synthetic_rows(lookup, draws, origin):
    """Prepared audit rows; full lookup content is checked at session boundaries."""
    receipt = lookup["receipt"]
    if (origin.get("schema") != "native-synthetic-origin-v1"
            or origin.get("origin") != "synthetic_validation"
            or origin.get("humanObservations") is not False
            or origin.get("lookupHash") != receipt["lookupHash"]
            or origin.get("analysisSpecHash") != receipt["analysisSpecHash"]
            or origin.get("sourceReceiptHash") != receipt["sourceReceiptHash"]):
        raise ValueError("Synthetic audit origin differs from the bound lookup")
    origin_hash = digest(origin)
    rows = []
    for draw in draws:
        base = lookup["states"].get(draw["stateKey"])
        if base is None:
            raise ValueError("Unknown lookup state")
        row = copy.deepcopy(base["metrics"])
        population = base["semantic"]["population"]
        variant = base["semantic"]["scheduleVariant"]
        plan = lookup["geometry"][population + "|" + row["conditionId"]]["projectionPlans"].get(str(variant))
        if plan is not None:
            row["initialProjectionPlan"] = copy.deepcopy(plan)
        for name in ("participantId", "sessionId"):
            if not isinstance(draw.get(name), str) or not draw[name]:
                raise ValueError("Missing synthetic person/session identity")
            row[name] = draw[name]
        for name in ("recordPresent", "endpointPresent"):
            if type(draw.get(name, True)) is not bool:
                raise ValueError("Literal record/endpoint availability required")
        row.update(allocation_metrics(row, draw.get("shareIndex")))
        row.update({"origin": origin["origin"], "originReceiptHash": origin_hash,
                    "lookupHash": receipt["lookupHash"], "stateKey": draw["stateKey"],
                    "generatorId": origin["generatorId"],
                    "recordPresent": draw.get("recordPresent", True),
                    "endpointPresent": draw.get("endpointPresent", True),
                    "humanObservations": False})
        rows.append(row)
    return rows


class PreparedCalibration:
    """Own a validated lookup once; never cache draws, endpoints or statistics.

    Trusted numerical code borrows this snapshot read-only. Admission/final
    fingerprints detect persistent changes, not transient mutate-and-revert.
    """
    def __init__(self, lookup, truths, analysis, calibration):
        self.closed = True
        self.pid = os.getpid()
        self.lookup = copy.deepcopy(lookup)
        self.truths = copy.deepcopy(truths)
        self.analysis = copy.deepcopy(analysis)
        self.calibration = copy.deepcopy(calibration)
        self.binding = execution_binding(self.lookup, self.truths, self.analysis, self.calibration)
        _validate_inputs(self.lookup, self.binding, self.analysis, self.calibration)
        self.cells = requests(self.analysis, self.calibration)
        for cell in self.cells:
            key = cell["id"] + ":" + cell["subcase"]
            _validate_truth_bundle(cell, self.lookup, self.truths[key])
        self.baseline = digest(binary_payload((self.lookup, self.truths, self.analysis,
                                              self.calibration, self.binding)))
        self.small = digest(binary_payload((self.lookup["receipt"], self.binding,
                                           self.analysis, self.calibration)))
        self.closed = False

    def check(self, lookup, binding, analysis, calibration):
        if self.closed or self.pid != os.getpid() or lookup is not self.lookup:
            raise ValueError("Closed/foreign prepared calibration")
        if digest(binary_payload((lookup["receipt"], binding, analysis, calibration))) != self.small:
            raise ValueError("Prepared input identity changed")

    def run(self, cell, completed=()):
        key = cell["id"] + ":" + cell["subcase"]
        return run_cell(cell, self.lookup, self.truths[key], self.binding, completed, prepared=self)

    def verify(self):
        """Validate content before durably accepting a completed batch."""
        self.check(self.lookup, self.binding, self.analysis, self.calibration)
        if digest(binary_payload((self.lookup, self.truths, self.analysis,
                                  self.calibration, self.binding))) != self.baseline:
            raise ValueError("Prepared numerical content changed")
        _validate_inputs(self.lookup, self.binding, self.analysis, self.calibration)

    def close(self):
        if self.closed:
            raise ValueError("Prepared calibration is already closed")
        try:
            self.verify()
        finally:
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            self.close()
        else:
            self.closed = True

    def __getstate__(self):
        raise TypeError("Prepared calibration is process-local")
