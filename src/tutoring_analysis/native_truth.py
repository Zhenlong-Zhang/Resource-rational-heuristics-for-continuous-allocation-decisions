"""Independent exact native expectations under declared ideal probability laws.

Pattern endpoints round after each binary64 operation. Probability integration
uses exact rationals, not the producer's floating truth candidates or PRNG draws.
"""
from fractions import Fraction as F
import itertools
import math
import struct

from .expected_outcomes import digest

SCOPE = "declared_decimal_ideal_probability_law;not_literal_finite_PRNG_or_probability_expression_rounding"
NORMAL_SCOPE = "declared_decimal_ideal_normal_reference_law;not_literal_finite_PRNG_or_probability_expression_rounding"
PATTERNS = tuple(itertools.product((0, 1), repeat=10))
ARMS = ("C1", "C2", "C3", "C4", "C5")
COEFFICIENT_ABSOLUTE_TOLERANCE = 1e-10
COEFFICIENT_RELATIVE_TOLERANCE = 1e-9


def decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Finite declared decimal required")
    return F(str(value))


def binary(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Finite binary64 input required")
    return F.from_float(float(value))


def unpack(bits):
    if not isinstance(bits, str) or len(bits) != 16:
        raise ValueError("Exactly sixteen hexadecimal binary64 digits required")
    value = struct.unpack(">d", bytes.fromhex(bits))[0]
    if not math.isfinite(value):
        raise ValueError("Nonfinite binary64 input")
    return value


def ratio(value):
    return None if value is None else {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def rounded(value):
    """Nearest binary64 conversion of an exact rational operation result."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite native arithmetic result")
    return F.from_float(result)


def native_mean(values):
    total = F(0)
    for value in values:
        total = rounded(total + value)
    return rounded(total / len(values))


def native_projection(weights, pattern):
    total = F(0)
    for weight, response in zip(weights, pattern):
        product = rounded(weight * response)
        total = rounded(total + product)
    return total


def polynomial_product(factors):
    coefficients = [F(1)]
    for a, b in factors:
        new = [F(0)] * (len(coefficients) + 1)
        for k, old in enumerate(coefficients):
            new[k] += old * a
            new[k + 1] += old * b
        coefficients = new
    return coefficients


def beta_moments(a, b, degree=10):
    result = [F(1)]
    for k in range(degree):
        result.append(result[-1] * (a + k) / (a + b + k))
    return result


def affine_moments(base, scale, moments):
    return [sum((F(math.comb(k, j)) * base ** (k - j) * scale ** j * moments[j]
                 for j in range(k + 1)), F(0)) for k in range(len(moments))]


def uniform_moments(low, high):
    if low == high:
        return [low ** k for k in range(11)]
    return [(high ** (k + 1) - low ** (k + 1)) / ((k + 1) * (high - low)) for k in range(11)]


def bernoulli_pattern_mass(k, moments):
    return sum((F((-1) ** j * math.comb(10 - k, j)) * moments[k + j]
                for j in range(11 - k)), F(0))


def slope_moments(law, M):
    if M <= 0:
        raise ValueError("Positive frozen slope-generator M required")
    if law == "zero":
        return [F(1)] + [F(0)] * 10
    if law == "symmetric_uniform":
        return uniform_moments(-1 / (4 * M), 1 / (4 * M))
    if law == "selected_uniform":
        return uniform_moments(1 / (16 * M), 1 / (8 * M))
    if law in ("centered_beta", "centered_beta_mirror"):
        sign = -1 if law.endswith("_mirror") else 1
        return affine_moments(-F(sign, 44) / M, F(sign, 4) / M, beta_moments(F(1, 2), F(5)))
    raise ValueError("Unknown fixed slope law")


def probability_moments(law, q=None, rho=None):
    if law == "selected_uniform":
        return uniform_moments(F(7, 10), F(9, 10))
    if law == "centered_beta":
        return affine_moments(F(1, 2) - F(1, 44), F(1, 4), beta_moments(F(1, 2), F(5)))
    if law == "beta":
        q, rho = decimal(q), decimal(rho)
        if not 0 < q < 1 or not 0 <= rho < 1:
            raise ValueError("Fixed beta/Bernoulli law out of bounds")
        if rho == 0:
            return [q ** k for k in range(11)]
        concentration = 1 / rho - 1
        return beta_moments(q * concentration, (1 - q) * concentration)
    raise ValueError("Unknown fixed probability law")


class Oracle:
    def __init__(self, lookup):
        self.lookup = lookup
        self.checks = 0
        self.maximum_coefficient_error = 0.
        self.cache = {}
        self.verified_plans = {}

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def schedule(self, population, arm, variant):
        rows = []
        for i in range(1, 11):
            key = self.lookup["index"]["|".join(map(str, (population, arm, variant, i, (0, 1, 3)[(i - 1) % 3])))]
            row = self.lookup["states"][key]["metrics"]
            self.require(row["conditionId"] == arm and row["mainIndex"] == i and row["remainingIntegerBudget"] > 0,
                         "Wrong positive-budget cycle row")
            rows.append(row)
        return rows

    def plan(self, population, arm, variant):
        identity = (population, arm, variant)
        if identity in self.verified_plans:
            return self.verified_plans[identity]
        geo = self.lookup["geometry"][population + "|" + arm]
        rows = self.schedule(population, arm, variant)
        x = [binary(r["x0"]) for r in rows]
        centre = sum(x, F(0)) / 10
        sxx = sum(((v - centre) ** 2 for v in x), F(0))
        supported = len(set(x)) >= 2 and sxx > F(1, 10 ** 12)
        self.require((variant in geo["supportedVariants"]) == supported, "Slope-support mismatch")
        if not supported:
            self.require(str(variant) not in geo["projectionPlans"], "Unsupported projection plan present")
            self.verified_plans[identity] = None
            return None
        plan = geo["projectionPlans"][str(variant)]
        self.require(plan["schema"] == "frozen-projection-plan-v1" and
                     plan["operationOrder"] == "ordered_float64_multiply_then_add_mainIndex_1_to_10",
                     "Wrong frozen projection arithmetic")
        self.require(plan["planHash"] == digest({k: v for k, v in plan.items() if k != "planHash"}), "Plan SHA mismatch")
        self.require(plan["predictorBinary64"] == [struct.pack(">d", float(v)).hex() for v in x], "Plan predictor bits mismatch")
        self.require(len(plan["weightBinary64"]) == 10, "Ten projection coefficients required")
        weights = [binary(unpack(b)) for b in plan["weightBinary64"]]
        for actual, predictor in zip(weights, x):
            exact_OLS = (predictor - centre) / sxx
            error = abs(float(actual - exact_OLS))
            self.maximum_coefficient_error = max(self.maximum_coefficient_error, error)
            self.require(error <= COEFFICIENT_ABSOLUTE_TOLERANCE + COEFFICIENT_RELATIVE_TOLERANCE * abs(float(exact_OLS)),
                         "Frozen coefficient fails independent centered-OLS check")
        self.verified_plans[identity] = weights
        return weights

    def expectation(self, family, population, arm, law, q=None, rho=None):
        identity = (family, population, arm, law, str(q), str(rho))
        if identity in self.cache:
            return self.cache[identity]
        if family == "Q":
            moments = probability_moments(law, q, rho)
            masses = [bernoulli_pattern_mass(k, moments) for k in range(11)]
            self.require(all(p >= 0 for p in masses) and sum((math.comb(10, k) * p for k, p in enumerate(masses)), F(0)) == 1,
                         "Exact Q probability mass failed")
            answer = sum((rounded(F(k, 10)) * math.comb(10, k) * p for k, p in enumerate(masses)), F(0))
            result = {"value": answer, "ideal": moments[1], "projectionPlanHashes": {}}
            self.cache[identity] = result
            return result
        geo = self.lookup["geometry"][population + "|" + arm]
        variants = geo["supportedVariants"] if family == "initial" else range(9)
        self.require(len(set(variants)) == len(variants) and all(v in range(9) for v in variants), "Invalid eligible variants")
        if not variants:
            result = {"value": None, "ideal": None, "projectionPlanHashes": {}}
            self.cache[identity] = result
            return result
        moments = slope_moments(law, binary(geo["M"])) if family == "initial" else probability_moments(law, q, rho)
        exchangeable = None if family == "initial" else [bernoulli_pattern_mass(k, moments) for k in range(11)]
        if exchangeable is not None:
            self.require(all(p >= 0 for p in exchangeable) and sum((math.comb(10, k) * p for k, p in enumerate(exchangeable)), F(0)) == 1,
                         "Exact paired-endpoint law mass failed")
        values, plan_hashes, ideal_variants = [], {}, []
        for variant in variants:
            rows = self.schedule(population, arm, variant)
            weights = self.plan(population, arm, variant) if family == "initial" else None
            if family == "initial":
                self.require(weights is not None, "Supported variant lacks projection plan")
                plan_hashes[str(variant)] = geo["projectionPlans"][str(variant)]["planHash"]
            accumulated, mass = F(0), F(0)
            for pattern in PATTERNS:
                if family == "initial":
                    coefficients = polynomial_product([(F(1, 2), binary(row["x0"]) if y else -binary(row["x0"]))
                                                       for row, y in zip(rows, pattern)])
                    probability = sum((c * m for c, m in zip(coefficients, moments)), F(0))
                    endpoint = native_projection(weights, pattern)
                elif family == "R":
                    probability = exchangeable[sum(pattern)]
                    terms = [rounded(F(abs(row["jEGPA"] - 500), 10)) * (1 if y else -1) for row, y in zip(rows, pattern)]
                    endpoint = native_mean(terms)
                elif family == "L":
                    probability = exchangeable[sum(pattern)]
                    terms = []
                    for row, y in zip(rows, pattern):
                        a, b = row["terminalMeans"]
                        tied = abs(a - b) <= 1e-9 * max(1., abs(a), abs(b))
                        terms.append(F(0) if tied else F(1 if y else -1, 2) * (1 if a < b else -1))
                    endpoint = native_mean(terms)
                else:
                    raise ValueError("Unknown native endpoint family")
                self.require(probability >= 0, "Negative exact pattern probability")
                mass += probability
                accumulated += probability * endpoint
            self.require(mass == 1, "Exact native pattern mass is not one")
            values.append(accumulated)
            if family == "R":
                D = sum((F(abs(r["jEGPA"] - 500), 10) for r in rows), F(0)) / 10
                ideal_variants.append((2 * moments[1] - 1) * D)
            elif family == "L":
                ideal_variants.append(F(0))
        result = {"value": sum(values, F(0)) / len(values),
                  "ideal": moments[1] if family == "initial" else sum(ideal_variants, F(0)) / len(ideal_variants),
                  "perVariant": [ratio(v) for v in values], "projectionPlanHashes": plan_hashes}
        self.cache[identity] = result
        return result


def selected_specs(cell, analysis):
    if cell["layer"] == "A":
        return analysis["claims"][:2] if cell["id"] in ("A03", "A04") or cell["subcase"] == "separate_pooled_reference" else analysis["claims"]
    if cell["id"] == "B07":
        return analysis["diagnostics"]
    endpoint = {"Q": "Q", "R": "R_EGPA_pp", "L": "L_lower_mean_share", "initial": "initial_slope"}[cell.get("family", cell["subcase"])]
    return [s for s in analysis["claims"] if s["endpoint"] == endpoint]


def derive_target(oracle, cell, spec):
    if cell["id"] == "C04":
        plans = {}
        if spec["endpoint"] == "initial_slope":
            for arm in spec["weights"]:
                geo = oracle.lookup["geometry"]["working|" + arm]
                plans[arm] = {str(v): geo["projectionPlans"][str(v)]["planHash"] for v in geo["supportedVariants"]}
        return None, None, plans, "diagnostic_not_derived", "not_derived_diagnostic_only"
    if cell["layer"] == "A":
        sign = 1 if spec["tail"] == "greater" else -1
        value = F(0)
        if cell["id"] == "A05":
            value = F(3, 10) * sign
        elif cell["id"] == "A06":
            if cell["subcase"] == "opposite_direction":
                value = -F(3, 10) * sign
            elif spec["id"] in ("H01_Q_C3_minus_C2", "H03_initial_C2", "H06_R_EGPA_C2"):
                value = F(3, 10) * sign
        classification = "boundary_null" if value == 0 else "composite_null_opposite_direction" if value * sign < 0 else "correct_direction_alternative"
        return value, value, {}, "exact_rational_abstract_reference_mean_under_declared_ideal_normal_law", classification
    family = cell.get("family", cell["subcase"])
    population = cell["subcase"] if cell["id"] == "B07" and cell["subcase"].startswith("gap_") else "working"
    if family == "initial":
        law = ("selected_uniform" if cell["id"] == "C03" else
               "centered_beta_mirror" if cell["id"] == "B12" and cell["subcase"] == "sign_mirror" else
               "centered_beta" if cell["id"] == "B12" else
               "symmetric_uniform" if cell["id"] in ("B10", "C01", "C02") else "zero")
    else:
        law = "selected_uniform" if cell["id"] == "C03" else "centered_beta" if cell["id"] == "B11" else "beta"
    values, ideals, plans = [], [], {}
    for arm, weight in spec["weights"].items():
        overrides = cell.get("armOverrides", {}).get(arm, {})
        result = oracle.expectation(family, population, arm, law,
                                    overrides.get("q", cell.get("q", .3 if family == "Q" else .5)),
                                    overrides.get("rho", cell.get("rho", .3)))
        if family == "initial":
            plans[arm] = result["projectionPlanHashes"]
        if result["value"] is None:
            return None, None, plans, "numerically_unresolved", "numerically_unresolved"
        values.append(decimal(weight) * result["value"])
        ideals.append(decimal(weight) * result["ideal"])
    value, ideal = sum(values, F(0)), sum(ideals, F(0))
    classification = ("boundary_null" if value == 0 else "selected_nonzero_effect" if cell["id"] == "C03" else
                      "exact_native_nonzero_not_boundary_null")
    return value, ideal, plans, "exact_rational_native_endpoint_under_declared_ideal_probability_law", classification



def build_truths(lookup, analysis, calibration):
    """Return complete cell-keyed candidate/exact bundles, with no approval claim."""
    from .native_calibration import requests, analytic_truth
    from .native_lookup import validate_lookup

    validate_lookup(lookup, analysis, calibration)
    lookup_hash = lookup["receipt"]["lookupHash"]
    input_identity = digest({"lookup": lookup, "analysis": analysis, "calibration": calibration})
    arithmetic_hash = digest(lookup["arithmeticMaps"])
    oracle = Oracle(lookup)
    bundles = {}
    for cell in requests(analysis, calibration):
        key = cell["id"] + ":" + cell["subcase"]
        candidate = analytic_truth(cell, lookup["receipt"])
        specs = selected_specs(cell, analysis)
        oracle.require(candidate["cellId"] == cell["id"] and candidate["subcase"] == cell["subcase"]
                       and candidate["lookupHash"] == lookup_hash, "Candidate cell identity")
        oracle.require([x["testId"] for x in candidate["targets"]] == [s["id"] for s in specs],
                       "Candidate target roster")
        targets = []
        for spec, candidate_target in zip(specs, candidate["targets"]):
            oracle.require(candidate_target["endpointId"] == spec["endpoint"], "Candidate endpoint identity")
            value, ideal, plans, status, classification = derive_target(oracle, cell, spec)
            targets.append({
                "testId": spec["id"], "endpointId": spec["endpoint"],
                "candidateTargetHash": digest(candidate_target), "projectionPlanHashes": plans,
                "probabilityLawScope": NORMAL_SCOPE if cell["layer"] == "A" else SCOPE,
                "exactNativeExpectation": ratio(value), "numericalValue": None if value is None else float(value),
                "idealAlgebraTruth": None if ideal is None else float(ideal), "exactIdealAlgebraTruth": ratio(ideal),
                "truthStatus": status, "classification": classification,
                "typeIEligible": classification in ("boundary_null", "composite_null_opposite_direction"),
                "coverageEligible": value is not None})
        accepted = {"schema": "native-accepted-native-truth-v1", "cellId": cell["id"], "subcase": cell["subcase"],
                    "lookupHash": lookup_hash, "analysisSpecHash": digest(analysis),
                    "calibrationSpecHash": digest(calibration), "candidateCellHash": digest(candidate),
                    "arithmeticMapHash": arithmetic_hash, "targets": targets}
        if key in bundles:
            raise ValueError("Duplicate truth cell")
        bundles[key] = {"schema": "native-truth-bundle-v1", "candidate": candidate, "accepted": accepted}
    oracle.require(len(bundles) == 40, "Complete forty-subcase truth roster required")
    oracle.require(input_identity == digest({"lookup": lookup, "analysis": analysis, "calibration": calibration}),
                   "Truth inputs changed during derivation")
    return bundles
