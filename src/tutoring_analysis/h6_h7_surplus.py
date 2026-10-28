"""Prospective surplus-primary roles for unchanged public nominal results.

Mapping computes no endpoints or statistical tests. The separate historical
fixture route feeds the recovered algebra recipe to the existing public metric
and inference generators. Content identity is not scientific acceptance, human
validation or calibration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from .expected_outcomes import digest
from .nominal_inference import validate_spec as validate_historical_spec


VERSION = "r11-h6h7-surplus-prospective-v1"
POPULATION = "equal_people_complete_ten;Q_includes_exhaustion;choice_all_positive_defined"


def specification():
    """The adopted two-claim definition, with a public-input envelope."""
    return {
        "schema": "h6-h7-surplus-public-spec-v1", "analysisId": VERSION,
        "scientificAdoption": "prospective_surplus_primary_egpa_secondary",
        "executionScope": "saved_public_synthetic_result_mapping_only",
        "referenceState": "actual_terminal_posterior_and_remaining_budget",
        "target": "native_projected_equal_posterior_mean_surplus",
        "pairedEndpoint": "(abs(j-500)-abs(j-jSurplus))/10",
        "personAggregation": "ordered_binary64_mainIndex_1_to_10_equal_person_weight",
        "population": POPULATION, "minimumPeoplePerArm": 2,
        "null": 0, "tail": "greater", "alpha": .05, "confidence": .95,
        "method": "Student", "noSupportingRescue": True,
        "noGlobalErrorGuarantee": True, "humanAdapterReleased": False,
        "humanDirectionEstablished": False, "humanPowerEstablished": False,
        "historicalAnalysisId": "person-level-nominal-v1",
        "historicalSchema": "person-level-nominal-result-v1",
        "claims": [
            {"id": "H06_R_SURPLUS_C2", "condition": "C2",
             "endpoint": "R_surplus_pp", "primary": True,
             "historicalNumericalTestId": "S_surplus_C2",
             "predecessorClaimId": "H06_R_EGPA_C2", "secondaryId": "S_R_EGPA_C2"},
            {"id": "H07_R_SURPLUS_C3", "condition": "C3",
             "endpoint": "R_surplus_pp", "primary": True,
             "historicalNumericalTestId": "S_surplus_C3",
             "predecessorClaimId": "H07_R_EGPA_C3", "secondaryId": "S_R_EGPA_C3"},
        ],
    }


def _require(value, label):
    if not value:
        raise ValueError(label)


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _same(actual, expected, label):
    # JSON comparison distinguishes bool/int aliases, unlike Python equality.
    _require(_encoded(actual) == _encoded(expected), label)


def validate_spec(spec):
    _same(spec, specification(), "Changed prospective H6/H7 specification")
    return spec


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _population_counts(analysis):
    counts = {(arm, endpoint): 0 for arm in ("C2", "C3")
              for endpoint in ("R_surplus_pp", "R_EGPA_pp")}
    seen = set()
    _require(type(analysis.get("people")) is list, "Missing historical people")
    for person in analysis["people"]:
        pid = person.get("participantId")
        _require(type(pid) is str and bool(pid) and pid not in seen,
                 "Duplicate/missing historical person")
        seen.add(pid)
        arm = person.get("conditionId")
        if arm not in ("C2", "C3"):
            continue
        eligibility = person["eligibility"]
        statuses = []
        for endpoint in ("R_surplus_pp", "R_EGPA_pp"):
            value = person["endpoints"][endpoint]
            status = value.get("status")
            _require(status in ("available", "unavailable"), "Unknown endpoint status")
            statuses.append(status)
            if status == "available":
                _require(eligibility.get("completeChoiceTen") is True
                         and eligibility.get("completePaidTen") is True
                         and eligibility.get("quarantineReasons") == []
                         and type(eligibility.get("recordedRows")) is int
                         and eligibility["recordedRows"] == 10
                         and type(eligibility.get("presentRows")) is int
                         and eligibility["presentRows"] == 10
                         and type(value.get("observations")) is int
                         and value["observations"] == 10
                         and type(person.get("sessionId")) is str
                         and bool(person["sessionId"])
                         and _finite(value.get("value")) and value.get("reason") is None,
                         "Available endpoint violates complete-person population")
                counts[(arm, endpoint)] += 1
            else:
                _require(value.get("value") is None and type(value.get("reason")) is str
                         and bool(value["reason"]), "Malformed unavailable person endpoint")
        _require(statuses[0] == statuses[1], "Historical endpoint eligibility differs")
    return counts


def _test(record, arm, endpoint, historical_id, primary, count):
    expected = {"testId": historical_id, "endpointId": endpoint,
                "weights": {arm: 1}, "tail": "greater", "null": 0,
                "alpha": .05, "confidence": .95, "primary": primary,
                "method": "Student", "personPopulation": POPULATION,
                "namespace": "primary" if primary else "supporting_descriptive",
                "requiredArmCounts": {arm: count}}
    for key, value in expected.items():
        _same(record.get(key), value, "Incompatible historical test: " + key)
    if record.get("status") == "available":
        _require(count >= 2 and all(_finite(record.get(k)) for k in ("effect", "SE", "df", "statistic", "p"))
                 and record["SE"] > 0 and record["df"] > 0 and 0 <= record["p"] <= 1
                 and type(record.get("CI")) is list and len(record["CI"]) == 2
                 and all(_finite(x) for x in record["CI"])
                 and record["CI"][0] <= record["CI"][1]
                 and record.get("reason") is None, "Invalid available historical uncertainty")
        _same(record.get("candidateNominalReject"), record["p"] < .05, "Changed nominal decision")
    else:
        _require(record.get("status") == "unavailable"
                 and record.get("reason") in ("fewer_than_two_eligible_people_in_required_arm",
                                              "zero_or_nonfinite_aggregate_SE", "nonfinite_test_uncertainty")
                 and record.get("p") is None and record.get("CI") is None
                 and record.get("candidateNominalReject") is False,
                 "Unknown/malformed unavailable historical test")


def map_saved_analysis(analysis, spec, historical_spec):
    """Reference four existing tests, preserving each complete historical record.

    historical_spec is the unchanged public expected_gpa_analysis.json, not a
    replacement roster with new primary IDs. The analysis is not authenticated
    as an accepted scientific population merely because its schema is admitted.
    """
    validate_spec(spec)
    validate_historical_spec(historical_spec)
    for key, value in {"population": POPULATION, "noSupportingRescue": True,
                       "noGlobalErrorGuarantee": True}.items():
        _same(historical_spec.get(key), value, "Changed historical policy: " + key)
    expected_supporting = [
        {"id": "S_surplus_" + arm, "endpoint": "R_surplus_pp",
         "weights": {arm: 1}, "tail": "greater"} for arm in ("C2", "C3")]
    actual_supporting = [s for s in historical_spec.get("supporting", [])
                         if s.get("id") in ("S_surplus_C2", "S_surplus_C3")]
    _same(actual_supporting, expected_supporting, "Changed historical surplus tests")
    for key, value in {"schema": "person-level-nominal-result-v1",
                       "analysisId": "person-level-nominal-v1",
                       "analysisSpecHash": digest(historical_spec),
                       "humanObservations": False, "globalErrorGuarantee": None,
                       "scope": "fixed_person_level_statistics_not_full_calibration_or_human_validation"}.items():
        _same(analysis.get(key), value, "Incompatible historical analysis: " + key)
    counts = _population_counts(analysis)
    _require(type(analysis.get("tests")) is list, "Missing historical tests")
    by_id = {}
    for result in analysis["tests"]:
        key = result.get("testId")
        _require(type(key) is str and bool(key) and key not in by_id,
                 "Duplicate/missing historical test identity")
        by_id[key] = result
    mappings = []
    for claim in spec["claims"]:
        arm = claim["condition"]
        for role, endpoint, test_id, claim_id, historical_primary in (
                ("primary", "R_surplus_pp", claim["historicalNumericalTestId"], claim["id"], False),
                ("secondary", "R_EGPA_pp", claim["predecessorClaimId"], claim["secondaryId"], True)):
            _require(test_id in by_id, "Missing required historical test: " + test_id)
            historical = by_id[test_id]
            _test(historical, arm, endpoint, test_id, historical_primary, counts[(arm, endpoint)])
            mappings.append({"prospectiveClaimId": claim_id, "prospectiveRole": role,
                             "endpointId": endpoint, "condition": arm,
                             "predecessorClaimId": claim["predecessorClaimId"],
                             "historicalTestId": test_id, "historicalResult": deepcopy(historical),
                             "historicalResultRecomputed": False,
                             "scope": "synthetic_software_compatibility_only"})
    return {"schema": "h6-h7-surplus-public-mapping-v1", "analysisId": VERSION,
            "scientificAdoption": spec["scientificAdoption"], "compatibilityOnly": True,
            "historicalAnalysisHash": hashlib.sha256(_encoded(analysis).encode()).hexdigest(),
            "historicalAnalysisId": analysis["analysisId"],
            "historicalAnalysisSpecHash": analysis["analysisSpecHash"],
            "sourceAuthentication": "content_identity_only_not_scientific_acceptance",
            "mappings": mappings, "noSupportingRescue": True, "noGlobalErrorGuarantee": True,
            "humanObservations": False, "humanAdapterReleased": False,
            "humanDirectionEstablished": False, "humanPowerEstablished": False,
            "newInputs": 0, "newPersonEndpoints": 0, "newTests": 0,
            "historicalResultsModified": False}


def example_trials():
    """Small deterministic wiring example, not observed or simulated evidence.

    Numerical targets and statistics must be produced by reproduce_expected_gpa;
    this function supplies raw synthetic histories only and uses no random seed.
    """
    return [{"templateId": arm + "_s1_g0", "participantId": arm + "_example_" + str(person),
             "sessionId": arm + "_session_" + str(person), "mainIndex": index,
             "paid": [], "shareIndex": share, "humanObservations": False}
            for arm in ("C2", "C3") for person, share in enumerate((0, 1000))
            for index in range(1, 11)]


def historical_fixture_specification():
    """Frozen recipe for the original deterministic 25-person API fixture."""
    return {
        "schema": "h6-h7-algebra-fixture-spec-v1",
        "recipeId": "independent-deterministic-algebra-v1",
        "scope": "nonengine_algebra_fixture;not_native_or_forward_generated",
        "arms": ["C1", "C2", "C3", "C4", "C5"],
        "peoplePerArm": 5, "mainIndices": list(range(1, 11)),
        "people": 25, "rows": 250, "stochasticResponseDraws": False,
        "initialIntegerBudget": 100000, "paidIntegerCost": 1000,
        "gap": "0 if C5 else (main-5.5)*(1+person/10)",
        "paidCount": "(0,1,3)[(main+person)%3]",
        "terminalGap": "gap+(-1)**(main+person)*(person+1)/5",
        "means": "[50+gap,50];[50+terminalGap,50]",
        "jEGPA": "500+((main+armIndex)%5-2)*7",
        "jSurplus": "jEGPA+3",
        "shareIndex": "500+(main-5)*(person+1)*(3 if armIndex%2==0 else -3)+2*person",
        "initialProjection": "exact_rational_centered_OLS_weights_rounded_once_to_binary64",
        "referenceEnvironment": {"python": "3.10.9", "numpy": "1.23.5", "scipy": "1.10.0"},
    }


def historical_fixture_rows(recipe):
    """Recreate the original algebra inputs, not synthetic native sessions.

    The fixed jEGPA/jSurplus values are algebra-test inputs, not targets to be
    recomputed from beliefs. Public schema names replace only transport identity.
    """
    from fractions import Fraction
    import struct
    from .expected_outcomes import ANALYSIS_ID, PROFILE, ROW_SCHEMA, allocation_metrics

    _same(recipe, historical_fixture_specification(), "Changed historical algebra recipe")
    rows = []
    for arm_index, arm in enumerate(recipe["arms"]):
        for person in range(5):
            person_rows = []
            for main in range(1, 11):
                gap = 0. if arm == "C5" else (main - 5.5) * (1 + person / 10)
                count = (0, 1, 3)[(main + person) % 3]
                budget = 100000 - count * 1000
                terminal_gap = gap + (-1) ** (main + person) * (person + 1) / 5
                initial = [50. + gap, 50.]
                terminal = [50. + terminal_gap, 50.]
                eo = 500 + ((main + arm_index) % 5 - 2) * 7
                j = 500 + (main - 5) * (person + 1) * (3 if arm_index % 2 == 0 else -3) + 2 * person
                row = {"metric_schema": ROW_SCHEMA, "rowValidated": True, "analysisId": ANALYSIS_ID,
                       "profileId": PROFILE, "participantId": arm + "-algebra-" + str(person),
                       "sessionId": arm + "-algebra-" + str(person), "conditionId": arm, "mainIndex": main,
                       "scale": 1., "gapOffset": 0., "initialIntegerBudget": 100000, "paidIntegerCost": 1000,
                       "remainingIntegerBudget": budget, "paidCount": count, "paidHistoryValid": True,
                       "Q": int(count > 0), "initialMeans": initial, "terminalMeans": terminal,
                       "rawInitialGap": initial[0] - initial[1], "x0": (initial[0] - initial[1]) / 100,
                       "xT": (terminal[0] - terminal[1]) / (budget / 1000),
                       "jEGPA": eo, "jSurplus": eo + 3, "jAL": 1000 if terminal_gap < 0 else 0,
                       "jML": 413, "recordPresent": True, "endpointPresent": True,
                       "humanObservations": False, "origin": "synthetic_validation",
                       "fixtureScope": recipe["scope"]}
                row.update(allocation_metrics(row, j))
                person_rows.append(row)
            x = [Fraction.from_float(float(row["x0"])) for row in person_rows]
            centre = sum(x, Fraction(0)) / len(x)
            sxx = sum(((value - centre) ** 2 for value in x), Fraction(0))
            if len(x) >= 3 and len(set(x)) >= 2 and sxx > Fraction(1, 10 ** 12):
                weights = [float((value - centre) / sxx) for value in x]
                payload = {"schema": "frozen-projection-plan-v1",
                           "predictorBinary64": [struct.pack(">d", row["x0"]).hex() for row in person_rows],
                           "weightBinary64": [struct.pack(">d", weight).hex() for weight in weights],
                           "operationOrder": "ordered_float64_multiply_then_add_mainIndex_1_to_10",
                           "diagnostics": {"fixtureScope": "independent_closed_form_centered_OLS"}}
                plan = {**payload, "planHash": digest(payload)}
                for row in person_rows:
                    row["initialProjectionPlan"] = plan
            rows.extend(person_rows)
    return rows


def reproduce_historical_fixture(recipe, historical_spec):
    """Run the unchanged public inference generator on the recovered fixture.

    This reproduces deterministic API-test numbers, not the original private
    metadata bytes, execution identity, or independent acceptance receipt.
    """
    from .nominal_inference import analyze

    validate_historical_spec(historical_spec)
    rows = historical_fixture_rows(recipe)
    result = analyze(rows, historical_spec)
    result["fixtureProvenance"] = {
        "recipeId": recipe["recipeId"], "recipeHash": digest(recipe),
        "scope": recipe["scope"], "people": 25, "rows": 250,
        "nativeTaskHistories": False, "historicalExecutionIdentityReused": False,
    }
    return {"rows": rows, "analysis": result, "recipe": deepcopy(recipe)}
