"""Independent, nonnumerical check of the prospective H6/H7 mapping.

This module deliberately does not import the mapper. Only the existing public
spec validator and semantic-digest primitive are shared; no inference is run.
"""
from __future__ import annotations

import hashlib
import json
import math

from .expected_outcomes import digest
from .nominal_inference import validate_spec as validate_historical_spec


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _equal(actual, expected, label):
    _require(_json(actual) == _json(expected), "Independent mismatch: " + label)


def _policy():
    # Independent literal: never derive the expected roles from mapper output.
    return dict(schema="h6-h7-surplus-public-spec-v1",
        analysisId="r11-h6h7-surplus-prospective-v1",
        scientificAdoption="prospective_surplus_primary_egpa_secondary",
        executionScope="saved_public_synthetic_result_mapping_only",
        referenceState="actual_terminal_posterior_and_remaining_budget",
        target="native_projected_equal_posterior_mean_surplus",
        pairedEndpoint="(abs(j-500)-abs(j-jSurplus))/10",
        personAggregation="ordered_binary64_mainIndex_1_to_10_equal_person_weight",
        population="equal_people_complete_ten;Q_includes_exhaustion;choice_all_positive_defined",
        minimumPeoplePerArm=2, null=0, tail="greater", alpha=.05, confidence=.95,
        method="Student", noSupportingRescue=True, noGlobalErrorGuarantee=True,
        humanAdapterReleased=False, humanDirectionEstablished=False, humanPowerEstablished=False,
        historicalAnalysisId="person-level-nominal-v1",
        historicalSchema="person-level-nominal-result-v1",
        claims=[dict(id="H06_R_SURPLUS_C2", condition="C2", endpoint="R_surplus_pp", primary=True,
                     historicalNumericalTestId="S_surplus_C2", predecessorClaimId="H06_R_EGPA_C2",
                     secondaryId="S_R_EGPA_C2"),
                dict(id="H07_R_SURPLUS_C3", condition="C3", endpoint="R_surplus_pp", primary=True,
                     historicalNumericalTestId="S_surplus_C3", predecessorClaimId="H07_R_EGPA_C3",
                     secondaryId="S_R_EGPA_C3")])


def check_mapping(analysis, spec, historical_spec, produced):
    """Check identities and exact old payloads, not statistical validity."""
    policy = _policy()
    _equal(spec, policy, "prospective specification")
    validate_historical_spec(historical_spec)
    for key, value in dict(population=policy["population"], noSupportingRescue=True,
                           noGlobalErrorGuarantee=True).items():
        _equal(historical_spec.get(key), value, "historical policy " + key)
    for arm in ("C2", "C3"):
        old_specs = [s for s in historical_spec.get("supporting", []) if s.get("id") == "S_surplus_" + arm]
        _equal(old_specs, [dict(id="S_surplus_" + arm, endpoint="R_surplus_pp",
                               weights={arm: 1}, tail="greater")], "old surplus specification")
    for key, value in dict(schema="person-level-nominal-result-v1", analysisId="person-level-nominal-v1",
                           analysisSpecHash=digest(historical_spec), humanObservations=False,
                           globalErrorGuarantee=None,
                           scope="fixed_person_level_statistics_not_full_calibration_or_human_validation").items():
        _equal(analysis.get(key), value, "historical analysis " + key)
    _require(type(analysis.get("people")) is list and type(analysis.get("tests")) is list,
             "Missing person/test lists")
    ids = [p.get("participantId") for p in analysis["people"]]
    _require(all(type(pid) is str and pid for pid in ids) and len(ids) == len(set(ids)),
             "Duplicate/missing person identity")
    test_ids = [t.get("testId") for t in analysis["tests"]]
    _require(all(type(tid) is str and tid for tid in test_ids) and len(test_ids) == len(set(test_ids)),
             "Duplicate/missing test identity")
    tests = {t["testId"]: t for t in analysis["tests"]}
    mappings = []
    for arm, primary_id, secondary_id, old_primary in (
            ("C2", "H06_R_SURPLUS_C2", "S_R_EGPA_C2", "H06_R_EGPA_C2"),
            ("C3", "H07_R_SURPLUS_C3", "S_R_EGPA_C3", "H07_R_EGPA_C3")):
        people = [p for p in analysis["people"] if p.get("conditionId") == arm]
        for person in people:
            endpoints = person["endpoints"]
            _equal(endpoints["R_surplus_pp"]["status"], endpoints["R_EGPA_pp"]["status"],
                   "shared eligibility")
        for role, endpoint, historical_id, new_id, historical_primary in (
                ("primary", "R_surplus_pp", "S_surplus_" + arm, primary_id, False),
                ("secondary", "R_EGPA_pp", old_primary, secondary_id, True)):
            count = 0
            for person in people:
                endpoint_value = person["endpoints"][endpoint]
                if endpoint_value.get("status") == "available":
                    eligibility = person["eligibility"]
                    for key, value in dict(recordedRows=10, presentRows=10, completePaidTen=True,
                                           completeChoiceTen=True, quarantineReasons=[]).items():
                        _equal(eligibility.get(key), value, "eligibility " + key)
                    _equal(endpoint_value.get("observations"), 10, "ten observations")
                    _require(type(person.get("sessionId")) is str and person["sessionId"]
                             and type(endpoint_value.get("value")) in (int, float)
                             and math.isfinite(endpoint_value["value"])
                             and endpoint_value.get("reason") is None, "Invalid available endpoint")
                    count += 1
                else:
                    _require(endpoint_value.get("status") == "unavailable"
                             and endpoint_value.get("value") is None
                             and type(endpoint_value.get("reason")) is str and endpoint_value["reason"],
                             "Invalid unavailable endpoint")
            _require(historical_id in tests, "Missing historical numerical test")
            value = tests[historical_id]
            for key, expected in dict(testId=historical_id, endpointId=endpoint, weights={arm: 1},
                    primary=historical_primary, tail="greater", null=0, alpha=.05, confidence=.95,
                    method="Student", personPopulation=policy["population"], requiredArmCounts={arm: count},
                    namespace="primary" if historical_primary else "supporting_descriptive").items():
                _equal(value.get(key), expected, "historical test " + key)
            if value.get("status") == "available":
                numbers = [value.get(key) for key in ("effect", "SE", "df", "statistic", "p")]
                _require(count >= 2 and all(type(v) in (int, float) and math.isfinite(v) for v in numbers),
                         "Invalid available uncertainty")
                ci = value.get("CI")
                _require(value["SE"] > 0 and value["df"] > 0 and 0 <= value["p"] <= 1
                         and type(ci) is list and len(ci) == 2
                         and all(type(v) in (int, float) and math.isfinite(v) for v in ci)
                         and ci[0] <= ci[1] and value.get("reason") is None, "Malformed interval")
                _equal(value.get("candidateNominalReject"), value["p"] < .05, "nominal decision")
            else:
                _require(value.get("status") == "unavailable" and value.get("reason") in (
                    "fewer_than_two_eligible_people_in_required_arm", "zero_or_nonfinite_aggregate_SE",
                    "nonfinite_test_uncertainty") and value.get("p") is None and value.get("CI") is None
                    and value.get("candidateNominalReject") is False, "Malformed unavailable test")
            mappings.append(dict(prospectiveClaimId=new_id, prospectiveRole=role, endpointId=endpoint,
                condition=arm, predecessorClaimId=old_primary, historicalTestId=historical_id,
                historicalResult=value, historicalResultRecomputed=False,
                scope="synthetic_software_compatibility_only"))
    expected = dict(schema="h6-h7-surplus-public-mapping-v1", analysisId=policy["analysisId"],
        scientificAdoption=policy["scientificAdoption"], compatibilityOnly=True,
        historicalAnalysisHash=hashlib.sha256(_json(analysis).encode()).hexdigest(),
        historicalAnalysisId=analysis["analysisId"], historicalAnalysisSpecHash=analysis["analysisSpecHash"],
        sourceAuthentication="content_identity_only_not_scientific_acceptance", mappings=mappings,
        noSupportingRescue=True, noGlobalErrorGuarantee=True, humanObservations=False,
        humanAdapterReleased=False, humanDirectionEstablished=False, humanPowerEstablished=False,
        newInputs=0, newPersonEndpoints=0, newTests=0, historicalResultsModified=False)
    _equal(produced, expected, "complete mapping")
    return dict(schema="h6-h7-surplus-public-check-v1", passed=True,
        historicalTestsReferenced=4, historicalAnalysisHash=expected["historicalAnalysisHash"],
        mappingHash=hashlib.sha256(_json(produced).encode()).hexdigest(),
        compatibilityOnly=True, newInputs=0, newPersonEndpoints=0, newTests=0,
        historicalResultsModified=False, humanAdapterReleased=False,
        humanDirectionEstablished=False, humanPowerEstablished=False,
        scope="exact_mapping_not_independent_statistical_recalculation_or_acceptance")
