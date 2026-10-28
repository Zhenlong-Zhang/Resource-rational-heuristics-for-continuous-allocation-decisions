"""Eight named slope contrasts for an existing mixed-model fit.

Mathematical definitions retain their original scope; software agreement is not
population calibration. No sampling or fitting is performed by these helpers.
"""
import math

NAMES = ("(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
         "conditionC1:need_A", "conditionC3:need_A",
         "conditionC1:need_B", "conditionC3:need_B")
IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
LINEAR_IDS = tuple(key + ":" + kind for key in IDS for kind in ("D", "S"))
ARMS = ("C1", "C2", "C3")
TARGET = dict(caseID="h10pd1_initial_amount_s1_l10_recruit193_mcar08", repID=1, ordinal=32)
FIT_KEY = TARGET["caseID"] + "__r0001"
NAMESPACE = "h10-power-design-v1-20261020"
METHOD = "participant_CR2_Satterthwaite_v1"
AGGREGATE_SHA = "408003c741653dd27108202381cb09e4e0df17cb36eb0bb0fef90bcf178a4f38"
INDEPENDENT_SHA = "2c2f4f34ec6c0c45abf8c79c103d60ec470fe0b1b58fffae3b97cb180d05c66a"
MANIFEST_SHA = "0953299b5d0afadbf84e99aec706c49eee5126b16c1cf0ab40a28038f1426e69"
FROZEN = {
    "composition": "c3e9ec74d2b3490c24edf537d1ae5c2f924fe6d20cda6319648b960217edc7d1",
    "mathKernel": "8fe009853b77bc8c0f87f8a5f2f1f2fb3b87a06619cc9cbc6a4481b8e6e2fd0d",
    "rExtractor": "f84999365e043e0a0118dccc83ba48015b75bb11a823021ef819836c54007336",
    "sourceTransport": "be7b6557ffd158aab9d318c74c8d0782ba5825cd9c6c30b48b3c5d171f65ba42",
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def exact(actual, expected, label):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), label + " keys")
        for key in expected:
            exact(actual[key], expected[key], label + "." + key)
    elif type(expected) in (list, tuple):
        require(type(actual) is type(expected) and len(actual) == len(expected), label + " shape")
        for index, value in enumerate(expected):
            exact(actual[index], value, label + "[" + str(index) + "]")
    else:
        require(type(actual) is type(expected) and actual == expected, "Changed " + label)


def near(actual, expected, label):
    """Original accepted continuous scale; no coercion of categories or counts."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), label + " keys")
        for key in expected:
            near(actual[key], expected[key], label + "." + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), label + " shape")
        for index, value in enumerate(expected):
            near(actual[index], value, label + "[" + str(index) + "]")
    elif finite(expected):
        require(finite(actual) and math.isclose(actual, expected, abs_tol=2e-9, rel_tol=2e-8),
                "Changed continuous " + label)
    else:
        exact(actual, expected, label)



def eight_contrasts(names):
    require(tuple(names) == NAMES, "Changed nine named coefficient order")
    positions = {name: i for i, name in enumerate(names)}

    def slope(arm, need):
        vector = [0] * len(names)
        vector[positions["need_" + need]] = 1
        if arm != "C2":
            vector[positions["condition" + arm + ":need_" + need]] = 1
        return vector

    result = []
    for key in IDS:
        other, need = key[3:5], key[-1]
        left, right = slope("C1", need), slope(other, need)
        result.extend(([a - b for a, b in zip(left, right)], [a + b for a, b in zip(left, right)]))
    return result


def reference_rows(estimates, standard_errors, dfs, L, stats):
    """Convert a complete finite numerical result; invalid math raises technical U.

    This adapter has no evidence-producing branch which can classify a new
    nonpositive/NaN variance or df as statistical unavailability.
    """
    require(len(estimates) == len(standard_errors) == len(dfs) == len(L) == 8, "Incomplete eight references")
    output = {}
    for key, estimate, se, df, vector in zip(LINEAR_IDS, estimates, standard_errors, dfs, L):
        require(finite(estimate) and finite(se) and finite(df) and se > 0 and df > 0,
                "Undefined/incomplete eight-L numerical result")
        statistic = float(estimate / se)
        critical = float(stats.t.ppf(.975, df))
        less, greater = float(stats.t.cdf(statistic, df)), float(stats.t.sf(statistic, df))
        two = float(2 * stats.t.sf(abs(statistic), df))
        require(all(finite(v) for v in (statistic, critical, less, greater, two))
                and critical > 0 and all(0 <= p <= 1 for p in (less, greater, two)),
                "Nonfinite or invalid t reference")
        output[key] = dict(linearID=key, status="available", reason=None, L=list(vector),
            coefficientNames=list(NAMES), estimate=float(estimate), SE=float(se), df=float(df),
            statistic=statistic, lessP=less, greaterP=greater, twoSidedP=two,
            CI=[float(estimate-critical*se), float(estimate+critical*se)])
    return output


def compose_references(references, composition):
    require(type(references) is dict and set(references) == set(LINEAR_IDS), "Changed eight linear IDs")
    rows = []
    for key in IDS:
        d, s = references[key + ":D"], references[key + ":S"]
        require(d["linearID"] == key + ":D" and s["linearID"] == key + ":S", "Mislabeled linear reference")
        states = (d["status"], s["status"])
        require(all(v in ("available", "statistical_unavailable", "technical_unresolved") for v in states),
                "Unknown linear-reference state")
        status = ("technical_unresolved" if "technical_unresolved" in states else
                  "statistical_unavailable" if "statistical_unavailable" in states else "available")
        p = dict(D_less=d["lessP"], D_greater=d["greaterP"], S_less=s["lessP"], S_greater=s["greaterP"]) if status == "available" else None
        reasons = [r["reason"] for r in (d, s) if r["status"] != "available"]
        require(all(type(v) is str and v for v in reasons), "Missing unresolved/unavailable reason")
        rows.append(dict(comparisonID=key, status=status, p=p, reason="; ".join(reasons) if reasons else None))
    return composition.compose_family(rows)


def unavailable_method(status, reason, composition):
    require(status in ("technical_unresolved", "statistical_unavailable") and type(reason) is str and reason,
            "Invalid explicit method state")
    refs = {key: dict(linearID=key, status=status, reason=reason, L=list(vector),
            coefficientNames=list(NAMES), estimate=None, SE=None, df=None, statistic=None,
            lessP=None, greaterP=None, twoSidedP=None, CI=None)
            for key, vector in zip(LINEAR_IDS, eight_contrasts(NAMES))}
    return dict(status=status, covariance=None, contrastCovariance=None, references=refs,
                composition=compose_references(refs, composition))



def validate_population(original, spec, generation, design=None):
    """Exact identities/counts; numerical X/y is checked in the same R/export domain."""
    counts, masks = spec["populationCounts"], spec["retentionMask"]
    require(type(counts) is dict and set(counts) == set(ARMS) and set(masks) == set(ARMS)
            and spec["recruitedCounts"] == {a: 193 for a in ARMS}, "Wrong actual population")
    for arm in ARMS:
        require(type(counts[arm]) is int and 0 <= counts[arm] <= 193
                and type(masks[arm]) is list and len(masks[arm]) == 193
                and all(type(v) is bool for v in masks[arm]) and sum(masks[arm]) == counts[arm],
                "Changed whole-person retention")
    rows = original["rows"]
    require(type(rows) is list and len(rows) == 10 * sum(counts.values()), "Changed ten-trial row count")
    persons, labels = {}, []
    for row in rows:
        require(type(row) is dict and set(row) == {"participant", "condition", "need_A", "need_B", "y"}
                and type(row["participant"]) is str and row["participant"]
                and row["condition"] in ARMS and all(finite(row[k]) for k in ("need_A", "need_B", "y")),
                "Invalid saved request row")
        person = row["participant"]
        if person not in persons:
            persons[person] = [row["condition"], 0]
        require(persons[person][0] == row["condition"], "Participant spans arms")
        persons[person][1] += 1
        labels.append(person)
    require(all(value[1] == 10 for value in persons.values())
            and {arm: sum(v[0] == arm for v in persons.values()) for arm in ARMS} == counts,
            "Changed participant cluster sizes/counts")
    # The original generator stores retained IDs in original row-first order.
    exact(list(persons), generation["retainedParticipantIDs"], "retained participant roster")
    if design is not None:
        require(set(design) == {"X", "y", "participant", "condition"}
                and len(design["X"]) == len(design["y"]) == len(rows)
                and all(type(v) is list and len(v) == 9 and all(finite(x) for x in v) for v in design["X"])
                and all(finite(v) for v in design["y"]), "Invalid transported actual design")
        exact(design["participant"], labels, "actual participant order")
        exact(design["condition"], [row["condition"] for row in rows], "actual arm order")
    return dict(counts)


def guard_components(raw):
    require(raw["status"] == "fitted", "Saved fitted baseline required")
    diag = raw["diagnostics"]
    require(type(diag["converged"]) is bool and type(diag["singular"]) is bool
            and type(diag["usableCandidateInference"]) is bool, "Invalid original guard categories")
    values = lambda v: v if type(v) is list else [v]
    gradient, eigen = diag.get("gradient"), values(diag.get("hessianEigenvalues"))
    coefficients, covariance = raw["coefficients"], raw["vcov"]
    require(raw["coefficientNames"] == list(NAMES) and set(coefficients) == set(NAMES)
            and type(covariance) is list and len(covariance) == 9
            and all(type(row) is list and len(row) == 9 for row in covariance), "Wrong original model shape")
    result = dict(converged=diag["converged"], nonsingular=not diag["singular"],
        hessianOK=bool(eigen) and all(finite(v) for v in eigen) and min(eigen) > 0,
        gradientOK=gradient is not None and all(finite(v) for v in values(gradient)),
        finiteBeta=all(finite(v) for v in coefficients.values()),
        finiteModelV=all(finite(v) for row in covariance for v in row))
    require(all(result.values()) == diag["usableCandidateInference"], "Inconsistent original full guard")
    return result


def validate_extraction(extracted, transported, history, *, fit_key=FIT_KEY):
    spec, base = history["spec"], history["baseline"]
    exact(extracted["coefficientNames"], list(NAMES), "actual coefficient names")
    require(extracted["fitKey"] == fit_key and extracted["formula"] ==
            "y~condition*need_A+condition*need_B+(1|participant)"
            and extracted["REML"] is True and extracted["optimizer"] == "nloptwrap"
            and extracted["verified"] is True and extracted["requestDescriptor"] == spec["request"]
            and extracted["frameRowCount"] == extracted["requestRowCount"] == len(history["request"]["rows"]),
            "Changed actual model/frame")
    exact(extracted["verifiedFields"], ["participant", "condition", "need_A", "need_B", "y"], "verified frame fields")
    require(transported["request"] == spec["request"] and transported["fitArtifact"] == base["fitArtifact"]
            and transported["exactMatrixInR"] is True and transported["exactResponseInR"] is True
            and transported["oneULPMatrixMutationRejected"] is True, "Missing same-R exact evidence")
    # Both arrays are from the same jsonlite digits=NA domain. Never compare
    # Python-parsed original need/y to these serialized floating values.
    exact(extracted["design"], transported["design"], "same-export actual X/y")
    counts = validate_population(history["request"], spec, history["generation"], extracted["design"])
    raw = base["raw"]
    near(extracted["coefficients"], [raw["coefficients"][name] for name in NAMES], "actual model beta")
    near(extracted["fixedVcov"], raw["vcov"], "actual model covariance")
    near(extracted["residualVariance"], raw["residualVariance"], "actual residual variance")
    near(extracted["randomCovariance"], raw["randomCovariance"], "actual random covariance")
    for key in ("converged", "singular", "usableCandidateInference"):
        exact(extracted["diagnostics"][key], raw["diagnostics"][key], "actual guard " + key)
    for key in ("gradient", "hessianEigenvalues"):
        near(extracted["diagnostics"][key], raw["diagnostics"][key], "actual guard " + key)
    # A supplied baseline RDS can be checked before a separate candidate
    # record exists. Its CR2 inference is still recomputed independently.
    if history.get("candidate") is None:
        return counts
    candidate = history["candidate"]["rawCandidate"]
    exact(candidate["coefficientNames"], list(NAMES), "candidate coefficient names")
    require(candidate["status"] == "fitted" and candidate["method"] == METHOD, "Wrong historical candidate method")
    exact(candidate["diagnostics"], raw["diagnostics"], "candidate original guard")
    metadata = history["candidate"]["candidateMetadata"]
    require(metadata["target"] == "fitted_scaled_marginal_Theta" and metadata["inverseVariance"] is True
            and metadata["cluster"] == "participant" and metadata["residual"] == "y-Xbeta"
            and metadata["priorWeights"] is False and metadata["frameVerified"] is True
            and metadata["frameRowCount"] == len(history["request"]["rows"])
            and metadata["requestDescriptor"] == spec["request"], "Wrong historical CR2 definition")
    near([candidate["coefficients"][key] for key in NAMES], extracted["coefficients"], "candidate beta")
    near(history["candidate"]["candidateMetadata"]["modelBasedVcov"], extracted["fixedVcov"], "candidate saved model covariance")
    return counts


def check_old_differences(references, raw, label):
    require(set(raw["references"]) == set(IDS), "Changed old difference roster")
    for key in IDS:
        current, old = references[key + ":D"], raw["references"][key]
        require(old["coefficientNames"] == list(NAMES) and old["L"] == current["L"], "Changed old difference L")
        for field in ("estimate", "SE", "df", "statistic", "CI"):
            near(current[field], old[field], label + "." + key + "." + field)
        p_key = "twoSidedP" if label.startswith("baseline") else "packageTwoSidedP"
        near(current["twoSidedP"], old[p_key], label + "." + key + ".twoSidedP")


def calculate_methods(extracted, kernel, composition):
    """Only core/complete-reference computation failures become technical U."""
    import numpy as np
    from scipy import stats
    L = eight_contrasts(extracted["coefficientNames"])
    actual = extracted["design"]
    model = extracted["fixedVcov"]
    beta = extracted["coefficients"]
    result, errors = {}, []
    for method in ("baseline", "candidate"):
        phase = method + "_eight_L"
        try:
            if method == "baseline":
                cache = extracted["originalSatterthwaite"]
                df = kernel.baseline_degrees_of_freedom(model, cache["covarianceJacobians"],
                                                       cache["varianceParameterCovariance"], L)
                matrix = np.asarray(L, float)
                full = np.asarray(model, float)
                contrast = matrix @ full @ matrix.T
                refs = reference_rows((matrix @ np.asarray(beta, float)).tolist(),
                                      np.sqrt(np.diag(contrast)).tolist(), df, L, stats)
                output = dict(covariance=full.tolist(), contrastCovariance=contrast.tolist())
            else:
                output = kernel.cr2_reconstruct(actual["X"], actual["y"], actual["participant"],
                    beta, extracted["theta"], model, extracted["residualVariance"], L,
                    expected_cluster_count=len(set(actual["participant"])), expected_rows_per_cluster=10)
                refs = reference_rows(output["estimates"], output["SE"], output["df"], L, stats)
            result[method] = dict(status="complete", covariance=output["covariance"],
                contrastCovariance=output["contrastCovariance"], references=refs,
                composition=compose_references(refs, composition))
            if method == "candidate":
                result[method]["dfTrace"] = output["dfTrace"]
                result[method]["dfSquaredNorm"] = output["dfSquaredNorm"]
        except (ValueError, KeyError, TypeError, ArithmeticError, np.linalg.LinAlgError) as error:
            detail = dict(phase=phase, type=type(error).__name__, message=str(error))
            errors.append(detail)
            result[method] = unavailable_method("technical_unresolved", phase + ": " + str(error), composition)
    return result, errors
