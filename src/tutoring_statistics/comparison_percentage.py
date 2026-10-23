"""Synthetic H8/H9 percentage reporting using the original ordered-L Student kernel.

The displayed mean is not recentered for inference. Percentage units transform
effect, SE and CI only; the frozen status, t, df, p and decision are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from scipy.stats import t as student_t
from src.tutoring_analysis.nominal_inference import test_contrast as frozen_contrast
from .joint_design import SCOPE, PROFILE, prepare_population, lower_need_scores, adapt_normalized_records, _finite


def one_sample_student(values, *, null=0., tail="greater"):
    """Independent direct-value diagnostic, never the H8/H9 inferential route."""
    if tail not in ("greater", "less") or not _finite(null):
        raise ValueError("Finite null and prespecified one-sided tail required")
    if any(not _finite(v) for v in values):
        raise ValueError("Missing/nonfinite values must be handled by population rules")
    n = len(values)
    mean = math.fsum(values) / n if n else None
    result = dict(status="unavailable", reason=None, n=n, mean=mean,
        effect=None if mean is None else mean - null, null=null, tail=tail,
        SE=None, statistic=None, df=None, p=None, meanCI95=None, effectCI95=None,
        candidateReject=False, method="independent_person_one_sample_Student", alpha=.05)
    if n < 2:
        result["reason"] = "fewer_than_two_eligible_people"
        return result
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    if not math.isfinite(variance) or variance <= 0:
        result["reason"] = "zero_or_nonfinite_SE"
        return result
    se, df = math.sqrt(variance / n), n - 1
    statistic = (mean - null) / se
    p = float(student_t.sf(statistic, df) if tail == "greater" else student_t.cdf(statistic, df))
    radius = float(student_t.ppf(.975, df)) * se
    result.update(status="available", SE=se, statistic=statistic, df=df, p=p,
        meanCI95=[mean - radius, mean + radius], effectCI95=[mean - null - radius, mean - null + radius],
        candidateReject=p < .05)
    return result


def _percent_from_frozen(reference, condition):
    effect, ci = reference["effect"], reference["CI"]
    mean = None if effect is None else 50. + 100. * effect
    percent_effect = None if effect is None else 100. * effect
    return dict(status=reference["status"], reason=reference["reason"], sourceStatus=reference["status"],
        sourceReason=reference["reason"], n=reference["requiredArmCounts"][condition], mean=mean,
        effect=percent_effect, null=50., tail=reference["tail"],
        SE=None if reference["SE"] is None else 100. * reference["SE"],
        statistic=reference["statistic"], df=reference["df"], p=reference["p"],
        meanCI95=None if ci is None else [50. + 100. * x for x in ci],
        effectCI95=None if ci is None else [100. * x for x in ci],
        candidateReject=reference["candidateNominalReject"], alpha=reference["alpha"],
        method="frozen_ordered_L_mean_and_Student_then_percent_reporting", sourceMethod=reference["method"],
        displayRoundingErasesEffect=effect is not None and effect != 0 and mean == 50.,
        displaySubtractionDiffersFromEffect=mean is not None and mean - 50. != percent_effect,
        inferenceUsesDisplaySubtraction=False)


def _equivalence(percent, reference, condition):
    checks = dict(status=percent["status"] == reference["status"], reason=percent["reason"] == reference["reason"],
        sourceReason=percent["sourceReason"] == reference["reason"], n=percent["n"] == reference["requiredArmCounts"][condition],
        decision=percent["candidateReject"] == reference["candidateNominalReject"],
        **{k: percent[k] == reference[k] for k in ("statistic", "df", "p")})
    differences = {}
    if percent["status"] == reference["status"] == "available":
        expected = dict(effect=100 * reference["effect"], SE=100 * reference["SE"],
                        **{k: reference[k] for k in ("statistic", "df", "p")})
        for key, target in expected.items():
            differences[key] = abs(percent[key] - target)
            checks[key] = percent[key] == target
        for i in (0, 1):
            checks["meanCI95_" + str(i)] = percent["meanCI95"][i] == 50 + 100 * reference["CI"][i]
            checks["effectCI95_" + str(i)] = percent["effectCI95"][i] == 100 * reference["CI"][i]
    return dict(status="pass" if all(checks.values()) else "fail", checks=checks, absoluteDifferences=differences,
        scope="frozen_estimator_preserved_then_units_transformed_not_calibration", fractionReference=reference)


def _direct_percent_diagnostic(people, percent):
    direct = one_sample_student([p["directPercentMeanDiagnostic"] for p in people], null=50.)
    if direct["status"] == percent["status"] == "available":
        agrees = all(math.isclose(direct[k], percent[k], rel_tol=1e-11, abs_tol=1e-12)
                     for k in ("effect", "SE", "statistic", "df", "p"))
        agrees = agrees and direct["candidateReject"] == percent["candidateReject"]
        status = "consistent_within_diagnostic_tolerance" if agrees else "numerical_representation_difference"
    elif direct["status"] == percent["status"] == "unavailable":
        status = "both_unavailable_not_an_equivalence_certificate"
    else:
        status = "numerical_representation_changes_availability"
    return dict(status=status, result=direct, route="ordered_mean_of_trial_percent_then_independent_direct_percent_Student",
        participantArithmetic=[dict(participant=p["participant"], frozenCenteredMean=p["meanCenteredLowerShare"],
            transformedMeanPercent=p["meanLowerPercent"], directMeanPercent=p["directPercentMeanDiagnostic"],
            directCenteredPercent=p["directPercentMeanDiagnostic"] - 50.,
            scaledFrozenCenteredMean=100. * p["meanCenteredLowerShare"],
            directSubtractionErasesFrozenMean=p["meanCenteredLowerShare"] != 0 and p["directPercentMeanDiagnostic"] == 50.)
            for p in people], controlsMainInference=False,
        interpretation="Direct displayed-percent recomputation is diagnostic, not an alternative estimator.")


def percentage_analysis(payload):
    population = prepare_population(payload)
    trials = [dict(participant=r["participant"], mainIndex=r["mainIndex"], condition=r["condition"],
                   **lower_need_scores(r)) for r in population["rows"]]
    result = dict(population=population, trials=trials, equivalence={}, directPercentDiagnostics={}, scope=dict(SCOPE))
    for hypothesis, condition in (("H8", "C4"), ("H9", "C5")):
        people = [p for p in population["eligibleParticipants"] if p["condition"] == condition]
        source = [dict(participantId=p["participant"], conditionId=condition,
            endpoints={"L_lower_mean_share": dict(status="available", value=p["meanCenteredLowerShare"])}) for p in people]
        reference = frozen_contrast(source, dict(id="H08_lower_C4" if hypothesis == "H8" else "H09_lower_C5",
            endpoint="L_lower_mean_share", weights={condition: 1}, tail="greater", primary=False))
        percent = _percent_from_frozen(reference, condition)
        result[hypothesis] = dict(percent, condition=condition, unit="percentage_points",
                                 participantIds=[p["participant"] for p in people])
        result["equivalence"][hypothesis] = _equivalence(percent, reference, condition)
        result["directPercentDiagnostics"][hypothesis] = _direct_percent_diagnostic(people, percent)
    return result


def example_payload(*, cancellation=False):
    """Deterministic software fixture, not a simulated human population."""
    rows = []
    for arm in ("C4", "C5"):
        for person, share in enumerate((600, 700), 1):
            participant = f"percentage-{arm}-{person}"
            for trial in range(1, 11):
                j = ((600, 700, 200)[trial - 1] if person == 1 and trial <= 3 else 500) if cancellation else share
                rows.append(dict(participant=participant, session=participant + "-session", condition=arm,
                    profileId=PROFILE, flowVersion="r10-prepractice-v1", mainIndex=trial,
                    template="percentage-software-example", scale=1., gap=0., sideReversed=False,
                    meanAInitial=50., meanBInitial=50., meanATerminal=40., meanBTerminal=60.,
                    shareIndex=j, share=j / 1000, initialBudgetMs=100000, remainingBudgetMs=100000))
    return dict(origin="synthetic_software_fixture", rows=rows)


def synthetic_input(data, *, design=None, node="node"):
    if data.get("origin") != "synthetic_software_fixture":
        raise ValueError("Only explicitly labelled synthetic software fixtures are accepted")
    if "records" not in data:
        return data
    if data.get("schema") != "tutoring-statistics-fixture-records-v1" or design is None:
        raise ValueError("Raw/native records need the synthetic fixture schema and matching design")
    normalized = []
    normalizer = Path(__file__).with_name("panel_normalizer.cjs")
    for raw in data["records"]:
        process = subprocess.run([node, str(normalizer)], input=json.dumps(dict(data=raw, design=design), allow_nan=False),
                                 text=True, capture_output=True, check=True)
        checked = json.loads(process.stdout)
        if checked.get("ok") is not True or checked["normalized"].get("errors"):
            raise ValueError("Normalizer rejected the synthetic fixture")
        normalized.append(checked["normalized"])
    return adapt_normalized_records(normalized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Explicit synthetic canonical rows or raw/native record bundle")
    inputs.add_argument("--example", choices=("ordinary", "cancellation"))
    parser.add_argument("--design", type=Path, help="Matching source design required for raw/native records")
    parser.add_argument("--node", default="node")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design-only", action="store_true", help="Explicit no-fit software route (always enabled)")
    args = parser.parse_args(argv)
    from .reproduction import read, write
    from .comparison_runner import output_root
    output = output_root(args.output)
    data = read(args.input) if args.input else example_payload(cancellation=args.example == "cancellation")
    payload = synthetic_input(data, design=read(args.design) if args.design else None, node=args.node)
    result = dict(schema="synthetic-percentage-reporting-v1", scope="software_candidate_validation_only",
        scientificAdoption=False, powerEstablished=False, fittingRequested=False,
        percentage=percentage_analysis(payload))
    write(output, result)
    print(json.dumps(dict(output=str(output), fittingRequested=False, scientificAdoption=False)))


if __name__ == "__main__":
    main()
