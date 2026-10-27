"""Manually specified composition fixtures; no fit or simulation."""

import copy

import pytest

from src.tutoring_statistics.magnitude_composition import (
    COMPARISON_IDS,
    compose_comparison,
    compose_family,
)


def available(comparison_id, d_less, d_greater, s_less, s_greater):
    return {
        "comparisonID": comparison_id,
        "status": "available",
        "p": {"D_less": d_less, "D_greater": d_greater, "S_less": s_less, "S_greater": s_greater},
        "reason": None,
    }


def magnitude_fixture(comparison_id, p):
    # The manual fixture has q_positive=p/2, q_negative=1, hence pMagnitude=p.
    # This specifies input probabilities; it does not estimate valid model p's.
    return available(comparison_id, p / 2, 1, 1, p / 2)


def unavailable(comparison_id, status="statistical_unavailable"):
    return {"comparisonID": comparison_id, "status": status, "p": None, "reason": "external status"}


def test_two_orientations_and_intersection_evidence():
    positive = available(COMPARISON_IDS[0], .005, .99, .98, .01)
    negative = available(COMPARISON_IDS[0], .9, .01, .005, .95)
    p_result = compose_comparison(positive)
    n_result = compose_comparison(negative)
    assert p_result["orientationP"] == {"reference_positive": .01, "reference_negative": .99}
    assert n_result["orientationP"] == {"reference_positive": .95, "reference_negative": .01}
    assert p_result["pMagnitude"] == pytest.approx(.02)
    assert n_result["pMagnitude"] == pytest.approx(.02)
    # Simultaneous sign reversal swaps the two prespecified regions, not tails
    # selected from observed coefficient signs.
    flipped = available(COMPARISON_IDS[0], .99, .005, .01, .98)
    assert compose_comparison(flipped)["pMagnitude"] == p_result["pMagnitude"]
    assert compose_comparison(available(COMPARISON_IDS[0], .001, .999, .1, .9))["pMagnitude"] == 1
    # Both tails can be conservative; complement-to-one is not an API premise.
    assert compose_comparison(available(COMPARISON_IDS[0], .03, .99, .99, .03))["pMagnitude"] == .06


def test_manual_zero_boundaries_endpoints_and_strict_alpha():
    # These are artificial p-value boundary fixtures, not computed probabilities
    # or evidence that a real coefficient is exactly zero.
    for tails in ((.5, .5, 0, 1), (0, 1, .5, .5), (.5, .5, .5, .5)):
        result = compose_comparison(available(COMPARISON_IDS[0], *tails))
        assert result["pMagnitude"] == 1
        assert result["nominalReject"] is False
    zero = compose_comparison(available(COMPARISON_IDS[0], 0, 1, 1, 0))
    assert zero["pMagnitude"] == 0
    assert zero["nominalReject"] is True
    boundary = compose_comparison(magnitude_fixture(COMPARISON_IDS[0], .05))
    assert boundary["pMagnitude"] == .05
    assert boundary["nominalReject"] is False
    ones = compose_comparison(available(COMPARISON_IDS[0], 1, 1, 1, 1))
    assert ones["pMagnitude"] == 1


def test_fixed_four_holm_manual_values_order_and_ties():
    records = [magnitude_fixture(key, p) for key, p in zip(COMPARISON_IDS, (.008, .016, .03, .08))]
    result = compose_family(records)
    assert [item["holmP"] for item in result["tests"].values()] == pytest.approx([.032, .048, .06, .08])
    assert [item["holmReject"] for item in result["tests"].values()] == [True, True, False, False]
    assert result["anyMagnitudeRejects"] is True
    assert result["comparisonIDs"] == list(COMPARISON_IDS)
    assert compose_family(tuple(reversed(records))) == result
    tied = compose_family([magnitude_fixture(key, .01) for key in COMPARISON_IDS])
    assert [item["holmP"] for item in tied["tests"].values()] == pytest.approx([.04] * 4)
    threshold = compose_family([magnitude_fixture(key, .0125) for key in COMPARISON_IDS])
    assert [item["holmP"] for item in threshold["tests"].values()] == [.05] * 4
    assert threshold["anyMagnitudeRejects"] is False


def test_statistical_unavailability_pads_four_without_a4_gate():
    records = [magnitude_fixture(COMPARISON_IDS[0], .02)] + [unavailable(key) for key in COMPARISON_IDS[1:]]
    result = compose_family(records)
    assert result["tests"][COMPARISON_IDS[0]]["holmP"] == .08
    assert result["anyMagnitudeRejects"] is False
    assert result["completeFamilyAvailable"] is False
    assert result["counts"] == {"planned": 4, "technicalCompleted": 4, "available": 1,
                                "statisticalUnavailable": 3, "technicalUnresolved": 0}
    for key in COMPARISON_IDS[1:]:
        test = result["tests"][key]
        assert test["inputP"] is None and test["pMagnitude"] is None and test["holmP"] is None
        assert test["holmPaddingP"] == 1 and test["holmReject"] is False
    records[0] = magnitude_fixture(COMPARISON_IDS[0], .01)
    assert compose_family(records)["anyMagnitudeRejects"] is True
    all_unavailable = compose_family([unavailable(key) for key in COMPARISON_IDS])
    assert all_unavailable["anyMagnitudeRejects"] is False
    assert all_unavailable["status"] == "complete" and all_unavailable["holmComputed"] is True
    assert all(item["pMagnitude"] is None for item in all_unavailable["tests"].values())


def test_technical_unknown_retains_unknown_family_and_known_p():
    records = [magnitude_fixture(COMPARISON_IDS[0], .001),
               unavailable(COMPARISON_IDS[1]),
               unavailable(COMPARISON_IDS[2], "technical_unresolved"),
               magnitude_fixture(COMPARISON_IDS[3], .4)]
    result = compose_family(records)
    assert result["status"] == "technical_unresolved" and result["holmComputed"] is False
    assert result["anyMagnitudeRejects"] is None
    assert result["counts"] == {"planned": 4, "technicalCompleted": 3, "available": 2,
                                "statisticalUnavailable": 1, "technicalUnresolved": 1}
    assert result["tests"][COMPARISON_IDS[0]]["pMagnitude"] == .001
    assert result["tests"][COMPARISON_IDS[0]]["nominalReject"] is True
    assert result["tests"][COMPARISON_IDS[1]]["nominalReject"] is False
    assert result["tests"][COMPARISON_IDS[2]]["nominalReject"] is None
    for test in result["tests"].values():
        assert test["holmPaddingP"] is None and test["holmP"] is None and test["holmReject"] is None


def test_reject_nonfinite_out_of_range_or_illegal_p_values():
    class FloatSubclass(float):
        pass

    for bad in (float("nan"), float("inf"), -float("inf"), -.001, 1.001, True, False,
                "0.01", None, 1j, [], {}, FloatSubclass(.01), 10**1000):
        record = magnitude_fixture(COMPARISON_IDS[0], .02)
        record["p"]["D_less"] = bad
        with pytest.raises(ValueError):
            compose_comparison(record)
    for bad_p in (None, [], {}, {"D_less": .01},
                  {"D_less": .01, "D_greater": .99, "S_less": .99, "S_greater": .01, "df": 10}):
        record = magnitude_fixture(COMPARISON_IDS[0], .02)
        record["p"] = bad_p
        with pytest.raises(ValueError):
            compose_comparison(record)


def test_reject_structural_errors_and_implicit_missingness():
    base = magnitude_fixture(COMPARISON_IDS[0], .02)
    malformed = [None, [], {}, {**base, "df": 10}, {key: value for key, value in base.items() if key != "p"},
                 {**base, "comparisonID": "C2-C1:A"}, {**base, "comparisonID": None},
                 {**base, "status": "missing"}, {**base, "status": True}, {**base, "reason": "unexpected"}]
    for status in ("statistical_unavailable", "technical_unresolved"):
        malformed.extend([{**unavailable(COMPARISON_IDS[0], status), "reason": reason} for reason in (None, "", "  ", 2)])
        malformed.append({**unavailable(COMPARISON_IDS[0], status), "p": base["p"]})
    for record in malformed:
        with pytest.raises(ValueError):
            compose_comparison(record)
    good = [magnitude_fixture(key, .02) for key in COMPARISON_IDS]
    for records in (None, {}, [], good[:3], good + [good[0]], [good[0], good[0], good[2], good[3]], iter(good)):
        with pytest.raises(ValueError):
            compose_family(records)


def test_pure_outputs_and_explicit_candidate_validity_boundary():
    assert COMPARISON_IDS == ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
    records = [magnitude_fixture(key, .02) for key in COMPARISON_IDS]
    before = copy.deepcopy(records)
    result = compose_family(records)
    assert records == before
    assert result["schema"] == "h10-magnitude-family-candidate-v1"
    assert result["methodID"] == "h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1"
    assert result["scope"] == "p_value_composition_only"
    assert result["scientificAdoption"] is False and result["constituentValidityEstablished"] is False
    assert result["familySize"] == 4 and result["alpha"] == .05
    result["tests"][COMPARISON_IDS[0]]["inputP"]["D_less"] = .8
    result["comparisonIDs"].reverse()
    assert records == before
    later = compose_family(records)
    assert later["tests"][COMPARISON_IDS[0]]["inputP"]["D_less"] == .01
    assert later["comparisonIDs"] == list(COMPARISON_IDS)
