"""Compute eight slope contrasts from a bound existing fit; never fit or sample."""
from pathlib import Path

from . import independent_cluster as kernel
from . import magnitude_composition as composition
from . import saved_fit_magnitude as math_core
from .saved_fit_inputs import open_input, write_new
from .saved_fit_runtime import ROOT, sources, runtime, run_r


def run(input_path, output_path, rscript):
    output = Path(output_path).resolve()
    math_core.require(output.parent.is_dir() and not output.exists(), "Use a new output file")
    ctx = open_input(input_path)
    observed, history = ctx["observed"], ctx["history"]
    source_pins = sources(observed)
    counts = math_core.validate_population(history["request"], history["spec"], history["generation"])
    guard = math_core.guard_components(history["baseline"]["raw"])
    environment = runtime(rscript, observed, output)
    artifacts, methods, errors, actual = {}, {}, [], None
    if not all(guard.values()):
        methods = {method: math_core.unavailable_method("statistical_unavailable",
                   "accepted_original_guard_false", composition) for method in ("baseline", "candidate")}
    else:
        pins = ctx["transportInputs"]
        extracted, artifacts["rds"] = run_r(environment["Rscript"], ROOT / "independent_cluster_rds.R",
            dict(schema="h10-inference-independent-rds-request-v1", fits=[dict(fitKey=math_core.FIT_KEY,
                 requestPath=pins["request"]["path"], rdsPath=pins["fitArtifact"]["path"])]),
            output.with_suffix(".rds.json"), observed, runtime_binding=environment)
        transported, artifacts["transport"] = run_r(environment["Rscript"], ROOT / "source_transport.R",
            dict(schema="h10-power-source-transport-request-v1", fits=[dict(fitKey=math_core.FIT_KEY, **pins)]),
            output.with_suffix(".transport.json"), observed, runtime_binding=environment)
        math_core.require(extracted["schema"] == "h10-inference-independent-rds-v1"
            and transported["schema"] == "h10-power-source-transport-v1"
            and set(extracted["fits"]) == set(transported["fits"]) == {math_core.FIT_KEY}
            and extracted["runtime"]["R"] == transported["runtime"]["R"] == "4.2.2"
            and transported["newScientificInputs"] == transported["newFitInvocations"]
            == transported["newCandidateCalculations"] == 0, "Changed extraction identity")
        for name, versions in (("lme4", ("1.1-35.5", "1.1.35.5")), ("lmerTest", ("3.1-3", "3.1.3"))):
            math_core.require(extracted["runtime"][name] in versions, "Changed extraction package")
        actual = extracted["fits"][math_core.FIT_KEY]
        math_core.validate_extraction(actual, transported["fits"][math_core.FIT_KEY], history)
        methods, errors = math_core.calculate_methods(actual, kernel, composition)
        for method, raw in (("baseline", history["baseline"]["raw"]),
                            ("candidate", history["candidate"]["rawCandidate"])):
            if methods[method]["status"] == "complete":
                math_core.check_old_differences(methods[method]["references"], raw, method + " saved D")
                math_core.near(methods[method]["covariance"], raw["vcov"], method + " saved covariance")
    observed.fresh()
    result = dict(schema="saved-magnitude-result-v1", input=ctx["inputPin"], kind=ctx["input"]["kind"],
        target=math_core.TARGET, fitKey=math_core.FIT_KEY, transportInputs=ctx["transportInputs"],
        actualK=counts, coefficientNames=list(math_core.NAMES), linearIDs=list(math_core.LINEAR_IDS),
        L=math_core.eight_contrasts(math_core.NAMES), originalGuard=guard,
        actualRdsInspected=actual is not None, sameRExactAndSameExportVerified=actual is not None,
        coefficients=actual["coefficients"] if actual else None, methods=methods, technicalErrors=errors,
        artifacts=artifacts, runtime=environment, sources=source_pins,
        computation=dict(newFits=0, newDraws=0, rdsReadToolInvocations=2 if actual else 0,
                         completedEightLMethods=sum(v["status"] == "complete" for v in methods.values())),
        status="technical_unresolved" if errors else "complete", scientificAdoption=False,
        constituentValidityEstablished=False, independentEightLVerified=False,
        observedInputs=dict(observed.files), sourceClosure=True)
    write_new(output, result)
    return result
