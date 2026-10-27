"""Fixed software fixtures for saved-fit contrasts. No RDS, optimizer or fit.

Synthetic arrays and stub kernels test the adapter seam; they do not verify
CR2 mathematics or validate the constituent p values. The real one-fit entry
and independent package implementation are separate required evidence.
"""
import copy
import math
import os
from types import SimpleNamespace

import pytest


import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from src.tutoring_statistics import magnitude_composition as composition
from src.tutoring_statistics import saved_fit_magnitude as adapter
from src.tutoring_statistics import saved_fit_inputs as inputs


EXPECTED_L = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 2, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 2, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 2, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, 0, 2, 0, 0, 1, 1],
]


def population_fixture():
    persons = [("C1:p1", "C1"), ("C1:p2", "C1"), ("C2:p1", "C2"), ("C3:p1", "C3")]
    rows = [dict(participant=p, condition=a, need_A=2., need_B=3., y=5.)
            for p, a in persons for _ in range(10)]
    counts = dict(C1=2, C2=1, C3=1)
    spec = dict(populationCounts=counts, recruitedCounts={a: 193 for a in adapter.ARMS},
                retentionMask={a: [True]*counts[a] + [False]*(193-counts[a]) for a in adapter.ARMS})
    generation = dict(retainedParticipantIDs=[p for p, _ in persons])
    design = dict(X=[[1., float(r["condition"] == "C1"), float(r["condition"] == "C3"), 2., 3.,
                      2. if r["condition"] == "C1" else 0., 2. if r["condition"] == "C3" else 0.,
                      3. if r["condition"] == "C1" else 0., 3. if r["condition"] == "C3" else 0.] for r in rows],
                  y=[5.]*40, participant=[r["participant"] for r in rows], condition=[r["condition"] for r in rows])
    return dict(model="H10", reference="C2", rows=rows), spec, generation, design


def test_literal_eight_named_vectors_and_order_rejections():
    assert adapter.eight_contrasts(adapter.NAMES) == EXPECTED_L
    assert adapter.LINEAR_IDS == (
        "C1-C2:A:D", "C1-C2:A:S", "C1-C2:B:D", "C1-C2:B:S",
        "C1-C3:A:D", "C1-C3:A:S", "C1-C3:B:D", "C1-C3:B:S")
    for names in (list(reversed(adapter.NAMES)), adapter.NAMES[:-1], adapter.NAMES + ("extra",),
                  ("bad",) + adapter.NAMES[1:]):
        with pytest.raises(ValueError, match="coefficient order"):
            adapter.eight_contrasts(names)


def test_full_covariance_cross_terms_and_eight_df_forwarding():
    # Manually specified model: Var(A)=1, Var(C1:A)=1, Cov=.25.
    # Var(D12A)=1; Var(S12A)=4+1+4*.25=6, not Var(D) or SE addition.
    model = np.eye(9)
    model[3, 5] = model[5, 3] = .25
    _, _, _, design = population_fixture()
    calls = {}
    dfs = [2., 3., 4., 5., 6., 7., 8., 9.]

    def baseline(V, J, Q, L):
        calls["baseline_L"] = copy.deepcopy(L)
        assert V == model.tolist() and J == ["saved_J"] and Q == ["saved_Q"]
        return list(dfs)

    def cr2(X, y, people, beta, theta, V, sigma2, L, **kwargs):
        calls["candidate_L"], calls["population"] = copy.deepcopy(L), kwargs
        assert X == design["X"] and people == design["participant"] and theta == .5 and sigma2 == 1.
        return dict(covariance=np.eye(9).tolist(), contrastCovariance=np.diag([1, 5, 1, 5, 2, 6, 2, 6]).tolist(),
                    estimates=[0.]*8, SE=[1., math.sqrt(5), 1., math.sqrt(5), math.sqrt(2), math.sqrt(6), math.sqrt(2), math.sqrt(6)],
                    df=[value+10 for value in dfs], dfTrace=[1.]*8, dfSquaredNorm=[1.]*8)

    extracted = dict(coefficientNames=list(adapter.NAMES), coefficients=[0.]*9,
        fixedVcov=model.tolist(), design=design, theta=.5, residualVariance=1.,
        originalSatterthwaite=dict(covarianceJacobians=["saved_J"], varianceParameterCovariance=["saved_Q"]))
    result, errors = adapter.calculate_methods(extracted,
        SimpleNamespace(baseline_degrees_of_freedom=baseline, cr2_reconstruct=cr2), composition)
    assert errors == []
    assert calls == dict(baseline_L=EXPECTED_L, candidate_L=EXPECTED_L,
                        population=dict(expected_cluster_count=4, expected_rows_per_cluster=10))
    assert result["baseline"]["references"]["C1-C2:A:D"]["SE"] == 1.
    assert result["baseline"]["references"]["C1-C2:A:S"]["SE"] == pytest.approx(math.sqrt(6))
    assert [v["df"] for v in result["baseline"]["references"].values()] == dfs
    assert [v["df"] for v in result["candidate"]["references"].values()] == [v+10 for v in dfs]
    assert extracted["originalSatterthwaite"]["covarianceJacobians"] == ["saved_J"]


def test_each_reference_own_t_tails_df_and_old_reference_protocol():
    refs = adapter.reference_rows([1.]*8, [1.]*8, [1., 2., 1., 2., 1., 2., 1., 2.], EXPECTED_L, stats)
    # Exact known Cauchy tail and df2 analytic CDF at t=1.
    assert refs["C1-C2:A:D"]["lessP"] == pytest.approx(.75)
    assert refs["C1-C2:A:D"]["greaterP"] == pytest.approx(.25)
    assert refs["C1-C2:A:S"]["lessP"] == pytest.approx(.5 + 1/(2*math.sqrt(3)))
    assert refs["C1-C2:A:S"]["CI"] != refs["C1-C2:A:D"]["CI"]
    old = {"references": {}}
    for key in adapter.IDS:
        row = refs[key + ":D"]
        old["references"][key] = {name: copy.deepcopy(row[name]) for name in
            ("coefficientNames", "L", "estimate", "SE", "df", "statistic", "CI")}
        old["references"][key]["packageTwoSidedP"] = row["twoSidedP"]
    adapter.check_old_differences(refs, old, "candidate historical D")
    old["references"][adapter.IDS[0]]["df"] += 1
    with pytest.raises(ValueError, match="continuous"):
        adapter.check_old_differences(refs, old, "candidate historical D")
    for bad in (0, -1, float("nan"), float("inf"), None):
        with pytest.raises(ValueError, match="numerical result"):
            adapter.reference_rows([0.]*8, [1.]*8, [bad]+[1.]*7, EXPECTED_L, stats)


def test_numerical_core_exception_remains_technical_unknown():
    _, _, _, design = population_fixture()

    def fails(*args, **kwargs):
        raise ValueError("Nonpositive Satterthwaite denominator")

    extracted = dict(coefficientNames=list(adapter.NAMES), coefficients=[0.]*9,
        fixedVcov=np.eye(9).tolist(), design=design, theta=.5, residualVariance=1.,
        originalSatterthwaite=dict(covarianceJacobians=[], varianceParameterCovariance=[]))
    methods, errors = adapter.calculate_methods(extracted,
        SimpleNamespace(baseline_degrees_of_freedom=fails, cr2_reconstruct=fails), composition)
    assert len(errors) == 2
    for method in methods.values():
        assert method["status"] == "technical_unresolved"
        assert method["composition"]["anyMagnitudeRejects"] is None
        assert method["composition"]["counts"]["statisticalUnavailable"] == 0
        assert method["composition"]["counts"]["technicalUnresolved"] == 4


def test_variable_population_exact_labels_masks_and_ten_rows():
    request, spec, generation, design = population_fixture()
    assert adapter.validate_population(request, spec, generation, design) == dict(C1=2, C2=1, C3=1)
    changed = copy.deepcopy(spec)
    changed["retentionMask"]["C1"][0] = False
    with pytest.raises(ValueError, match="retention"):
        adapter.validate_population(request, changed, generation)
    changed = copy.deepcopy(request)
    changed["rows"].pop()
    with pytest.raises(ValueError, match="row count"):
        adapter.validate_population(changed, spec, generation)
    changed = copy.deepcopy(request)
    changed["rows"][0]["condition"] = "C2"
    with pytest.raises(ValueError, match="spans arms"):
        adapter.validate_population(changed, spec, generation)
    changed = copy.deepcopy(design)
    changed["participant"][0] = "wrong"
    with pytest.raises(ValueError, match="participant order"):
        adapter.validate_population(request, spec, generation, changed)
    changed = copy.deepcopy(generation)
    changed["retainedParticipantIDs"] = list(reversed(changed["retainedParticipantIDs"]))
    with pytest.raises(ValueError, match="roster"):
        adapter.validate_population(request, spec, changed)


def test_same_export_zero_tolerance_without_rejecting_source_transport_rounding():
    request, spec, generation, design = population_fixture()
    pin = dict(path="/fixed/request.json", sha256="a"*64)
    fit = dict(path="/fixed/fit.rds", sha256="b"*64)
    spec["request"] = pin
    diag = dict(converged=True, singular=False, gradient=[], hessianEigenvalues=[1.], usableCandidateInference=True)
    raw = dict(status="fitted", coefficients={k: 0. for k in adapter.NAMES},
        coefficientNames=list(adapter.NAMES), vcov=np.eye(9).tolist(), residualVariance=1.,
        randomCovariance=[[.25]], diagnostics=diag)
    metadata = dict(target="fitted_scaled_marginal_Theta", inverseVariance=True,
        cluster="participant", residual="y-Xbeta", priorWeights=False, frameVerified=True,
        frameRowCount=40, requestDescriptor=pin, modelBasedVcov=raw["vcov"])
    candidate = dict(rawCandidate=dict(raw, method=adapter.METHOD), candidateMetadata=metadata)
    extracted = dict(coefficientNames=list(adapter.NAMES), fitKey=adapter.FIT_KEY,
        formula="y~condition*need_A+condition*need_B+(1|participant)", REML=True, optimizer="nloptwrap",
        verified=True, requestDescriptor=pin, frameRowCount=40, requestRowCount=40,
        verifiedFields=["participant", "condition", "need_A", "need_B", "y"], design=copy.deepcopy(design),
        coefficients=[0.]*9, fixedVcov=raw["vcov"], residualVariance=1., randomCovariance=[[.25]], diagnostics=diag)
    # Transport-representation fixture: original Python input and both R exports
    # can differ; the real pinned R tool, not this fixture, proves same-R identity.
    extracted["design"]["y"][0] = math.nextafter(5., math.inf)
    transported = dict(request=pin, fitArtifact=fit, exactMatrixInR=True, exactResponseInR=True,
                       oneULPMatrixMutationRejected=True, design=copy.deepcopy(extracted["design"]))
    history = dict(spec=spec, baseline=dict(raw=raw, fitArtifact=fit), candidate=candidate,
                   request=request, generation=generation)
    assert adapter.validate_extraction(extracted, transported, history) == spec["populationCounts"]
    transported["design"]["X"][0][0] = math.nextafter(1., math.inf)
    with pytest.raises(ValueError, match="same-export"):
        adapter.validate_extraction(extracted, transported, history)
    transported["design"] = copy.deepcopy(extracted["design"])
    transported["exactMatrixInR"] = False
    with pytest.raises(ValueError, match="same-R"):
        adapter.validate_extraction(extracted, transported, history)


def test_original_guard_and_explicit_status_no_all_available_gate():
    raw = dict(status="fitted", coefficientNames=list(adapter.NAMES),
        coefficients={k: 0. for k in adapter.NAMES}, vcov=np.eye(9).tolist(),
        diagnostics=dict(converged=True, singular=False, gradient=[],
                         hessianEigenvalues=[1.], usableCandidateInference=True))
    assert all(adapter.guard_components(raw).values())  # R all(numeric(0)) is true.
    raw["diagnostics"].update(singular=True, usableCandidateInference=False)
    assert adapter.guard_components(raw)["nonsingular"] is False
    result = adapter.unavailable_method("statistical_unavailable", "accepted_original_guard_false", composition)
    assert result["composition"]["anyMagnitudeRejects"] is False
    refs = result["references"]
    for suffix, less, greater in (("D", .001, .999), ("S", .999, .001)):
        refs["C1-C2:A:" + suffix].update(status="available", reason=None, lessP=less, greaterP=greater)
    family = adapter.compose_references(refs, composition)
    assert family["completeFamilyAvailable"] is False and family["anyMagnitudeRejects"] is True
    refs["C1-C3:B:S"].update(status="technical_unresolved", reason="missing cache")
    assert adapter.compose_references(refs, composition)["anyMagnitudeRejects"] is None
    raw["diagnostics"]["usableCandidateInference"] = True
    with pytest.raises(ValueError, match="full guard"):
        adapter.guard_components(raw)


def test_bound_scope_membership_duplicate_json_and_no_overwrite(tmp_path):
    manifest = dict(path="/accepted/manifest.json", sha256=adapter.MANIFEST_SHA)
    a = dict(path="/accepted/aggregate.json", sha256=adapter.AGGREGATE_SHA)
    i = dict(path="/accepted/independent.json", sha256=adapter.INDEPENDENT_SHA)
    other = dict(path="/accepted/block.json", sha256="a"*64)
    values = {
        a["path"]: dict(schema="h10-power-design-aggregate-v1", complete=True, sourceClosure=True,
            expectedTotal=80000, presentTotal=80000, completedTotal=80000, missingOrdinals=[],
            manifest=manifest, independentBlocks=[other]),
        i["path"]: dict(schema="h10-power-design-independent-aggregate-v1", aggregate=a,
            passed=True, complete=True, sourceClosure=True, manifest=manifest, independentBlocks=[other]),
    }
    observed = SimpleNamespace(read=lambda pin: values[pin["path"]])
    request = dict(target=dict(adapter.TARGET), aggregate=a, independentAggregate=i,
                   blockProof=dict(path="/unaccepted/block.json", sha256="b"*64))
    with pytest.raises(ValueError, match="Nonmember block"):
        inputs.admit_history(request, observed)
    request["target"]["repID"] = 2
    with pytest.raises(ValueError, match="target"):
        inputs.admit_history(request, observed)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        inputs.load(duplicate)
    output = tmp_path / "result.json"
    inputs.write_new(output, {"manual": True})
    before = output.read_bytes()
    with pytest.raises(ValueError, match="New absolute output"):
        inputs.write_new(output, {"manual": False})
    assert output.read_bytes() == before
    pins = inputs.Observations()
    saved = pins.observe(output)
    output.write_text('{"manual":false}')
    with pytest.raises(ValueError, match="Changed bound"):
        pins.read(saved)
    with pytest.raises(ValueError, match="closure"):
        pins.fresh()
