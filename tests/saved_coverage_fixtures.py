"""Artificial saved-file interface fixtures, never historical or fitted evidence.

All numbers are fixed constants. The opaque fit bytes below are NOT an RDS
model and are used only to test byte admission, which never deserializes them.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/tutoring_statistics"
CELL = "h10pd1_initial_amount_s1_l10_recruit193_mcar08"
KEY = "C1-C2:B"
IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
NAMES = ("(Intercept)", "conditionC1", "conditionC3", "need_A", "need_B",
         "conditionC1:need_A", "conditionC3:need_A", "conditionC1:need_B", "conditionC3:need_B")
ARMS = ("C1", "C2", "C3")
TAILS = ("literal", "sign_consistent")
PUBLIC = "conditional-statistics-reproduction-v1"
METHOD = "participant_CR2_Satterthwaite_v1"


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return pin(path)


def cell_template():
    return dict(caseID=CELL, cellIndex=33, view="initial_amount", scenarioIndex=1, multiplier=1.,
        populationRegime="recruit193_mcar08", recruitedPerArm=193, eligibilityProbability=.8,
        fixedEligiblePerArm=None, originalCaseIndex=7, role="new_population_control",
        delta={key: 0. for key in IDS}, trueNullIDs={tail: list(IDS) for tail in TAILS},
        parameters=dict(alpha=60., sdU=8., sigmaByArm={arm: 12. for arm in ARMS},
                        slopes={arm: dict(A=.25, B=-.25) for arm in ARMS}, deltaMagnitude=.15))


def case(rep=1):
    return dict(cell_template(), repID=rep, retentionSeed=802940200+rep,
                designSeed=902940200+rep, responseSeed=1002940200+rep)


def moment_cell():
    stat = dict(n=2000, missing=0, mean=0., SD=2., min=-2., max=2.)
    diag = dict(estimate=dict(stat), baselineSE=dict(stat, mean=3.), candidateSE=dict(stat, mean=3.))
    coverage = dict(counts=dict(A=2000, CIAvailable=2000, C=2000), ciUnavailableAmongAvailable=0)
    for name in ("coverageAll", "coverageAvailable", "availabilityCompleted"):
        coverage[name] = dict(numerator=2000, denominator=2000, estimate=1.)
    return dict(caseID=CELL, cellIndex=33, cell=cell_template(), M=2000, T=2000, present=2000,
        complete=True, technicalMissing=0, missingRepIDs=[], unresolvedRepIDs=[], originalGuardCounts={"true": 2000},
        contrastDiagnostics={KEY: diag}, baseline=dict(coverage={KEY: deepcopy(coverage)}),
        candidate=dict(coverage={KEY: deepcopy(coverage)}))


def scalar_rows():
    rows = []
    for rep in range(1, 2001):
        value = dict(estimate=(-1.)**rep, SE=1., df=10., lower=-2., upper=2., covered=True,
                     belowTruth=False, aboveTruth=False, z=(-1.)**rep)
        rows.append(dict(repID=rep, populationCounts=dict(C1=149, C2=156, C3=147),
                         baseline=dict(value), candidate=dict(value)))
    return rows


def public_manifest(directory):
    # The test binds real installed source bytes, not invented producer files.
    sources = [pin(p) for p in sorted(SOURCE.rglob("*")) if p.is_file() and p.suffix in (".py", ".R", ".cjs", ".json")]
    native = [save(directory / (name+".json"), {"fixtureOnly": True})
              for name in ("resolved_design", "initial_panel", "h10_canonical")]
    value = dict(schema=PUBLIC, family="power", expectedTotal=80000,
        identity="new_public_reproduction", legacyPolicy="original_input_seed_semantics_not_fresh_grid_redraw",
        sources=save(directory / "source_snapshot.json", dict(files=sources)),
        driver=pin(SOURCE / "reproduction.py"), rDriver=pin(SOURCE / "kernel_entry.R"),
        environment=pin(SOURCE / "data/numerical_environment.json"), nativeEntry=pin(SOURCE / "native_inputs.py"),
        nativeIdentity=pin(SOURCE / "data/native_identity.json"), native=native,
        numericalRuntime=dict(python="3.10.9", executable=pin(sys.executable), Rscript=pin(sys.executable)))
    return save(directory / "manifest.json", value)


def public_member(directory, manifest, rep=1, estimate=0.):
    directory.mkdir(parents=True, exist_ok=True)
    scientific = case(rep)
    key = f"{CELL}__r{rep:04d}"
    recruited = [dict(participant=f"{arm}_{i}", condition=arm) for arm in ARMS for i in range(193)]
    selected = [f"{arm}_0" for arm in ARMS]
    rows = [dict(participant=p, condition=p[:2], need_A=float(j), need_B=float(j*j), y=float(j+1))
            for p in selected for j in range(10)]
    actual = save(directory / "actual_request.json", dict(model="H10", reference="C2", rows=rows))
    generation = dict(schema="h10-power-design-generation-v1", namespace="h10-power-design-v1-20261020",
        case=scientific, populationCounts=dict(C1=1, C2=1, C3=1), recruitedCounts={arm: 193 for arm in ARMS},
        retentionMask={arm: [True]+[False]*192 for arm in ARMS}, kernelGeneration=dict(people=recruited),
        recruitedParticipantIDs=[p["participant"] for p in recruited], retainedParticipantIDs=selected,
        retainedRowIndices=[10*i+j for i in (0, 193, 386) for j in range(10)],
        responseTransform="none", eligibility="independent_whole_person_MCAR_complete_ten_not_trial_dropout",
        execution=dict(fixtureOnly=True))
    gen_pin = save(directory / "generation.json", generation)
    spec = dict(ordinal=40*(rep-1)+32, caseID=CELL, repID=rep, fitKey=key, case=scientific,
        request=actual, generation=gen_pin, populationCounts=dict(C1=1, C2=1, C3=1), origin=None,
        manifest=manifest, execution=generation["execution"], support=dict(fitEligible=True))
    spec_pin = save(directory / "spec.json", spec)
    coefficients = {name: (estimate if name == "conditionC1:need_B" else 0.) for name in NAMES}
    covariance = [[float(i == j) for j in range(9)] for i in range(9)]
    guard = dict(converged=True, singular=False, gradient=0., hessianEigenvalues=1., usableCandidateInference=True,
                 convergenceMessages=[], warnings=[])
    raw = dict(status="fitted", diagnostics=guard, coefficients=coefficients, coefficientNames=list(NAMES), vcov=covariance,
               references={KEY: dict(estimate=estimate, SE=1., df=10., CI=[-2., 2.])})
    raw_pin = save(directory / "raw.json", raw)
    fit_path = directory / "fit.rds"
    fit_path.write_bytes(b"fixed opaque non-RDS fixture, never deserialized")
    fit_pin = pin(fit_path)
    base = dict(schema="h10-calibration-production-fit-v1", caseID=CELL, repID=rep, fitKey=key,
        manifestSha256=manifest["sha256"], request=actual, generation=gen_pin, raw=raw,
        rawArtifact=raw_pin, fitArtifact=fit_pin)
    base_pin = save(directory / "baseline.json", base)
    environment = json.loads((SOURCE / "data/numerical_environment.json").read_text())["statistical_reference"]
    runtime = dict(R=environment["R"], **environment["R_packages"])
    candidate = dict(schema="h10-inference-record-v1", caseID=CELL, repID=rep, fitKey=key,
        manifest=manifest, baseline=base_pin, request=actual, generation=gen_pin, fitArtifact=fit_pin,
        method=METHOD, technicalCompleted=True, technicalError=None, newScientificInputs=0, newFitInvocations=0,
        rawCandidate=dict(raw, method=METHOD), candidateMetadata=dict(frameVerified=True, frameRowCount=30,
                                                                  priorWeights=False, cluster="participant", inverseVariance=True,
                                                                  target="fitted_scaled_marginal_Theta", residual="y-Xbeta",
                                                                  modelBasedVcov=covariance,
                                                                  requestDescriptor=actual, runtime=runtime))
    cand_pin = save(directory / "candidate.json", candidate)
    matrix = [[1., float(r["condition"] == "C1"), float(r["condition"] == "C3"), r["need_A"], r["need_B"],
               float(r["condition"] == "C1")*r["need_A"], float(r["condition"] == "C3")*r["need_A"],
               float(r["condition"] == "C1")*r["need_B"], float(r["condition"] == "C3")*r["need_B"]] for r in rows]
    design = dict(X=matrix, y=[r["y"] for r in rows],
                  participant=[r["participant"] for r in rows], condition=[r["condition"] for r in rows])
    extraction = dict(schema="h10-inference-independent-rds-v1", runtime=dict(runtime, lme4="1.1.35.5", lmerTest="3.1.3"), fits={key:
        dict(fitKey=key, requestDescriptor=actual, verified=True, frameRowCount=30, requestRowCount=30, design=design,
             diagnostics=dict(converged=True, singular=False, gradient=0., hessianEigenvalues=1., usableCandidateInference=True,
                              optimizerMessages=[]), coefficientNames=list(NAMES), coefficients=list(coefficients.values()), fixedVcov=covariance)})
    extraction_pin = save(directory / "independent_rds.json", extraction)
    transport = dict(schema="h10-power-source-transport-v1", runtime=runtime, newScientificInputs=0,
        newFitInvocations=0, newCandidateCalculations=0, fits={key: dict(request=actual, fitArtifact=fit_pin,
            exactMatrixInR=True, exactResponseInR=True, oneULPMatrixMutationRejected=True, design=design)})
    transport_pin = save(directory / "source_transport.json", transport)
    branches = [dict(tailMode=mode, tests={KEY: dict(status="available_candidate")},
                    coverage={KEY: dict(A=1, CIAvailable=1, C=1, covered=True, truth=0.)}) for mode in TAILS]
    paired = [dict(tailMode=mode, contrasts={KEY: dict(estimate=estimate, baselineSE=1., candidateSE=1.,
        baselineDf=10., candidateDf=10., baselineCovered=True, candidateCovered=True,
        baselineAvailable=True, candidateAvailable=True)}) for mode in TAILS]
    event = dict(ordinal=spec["ordinal"], caseID=CELL, repID=rep, fitKey=key, origin="new_population_input",
        _powerCompact=True, technicalCompleted=True, technicalError=None, diagnostics=dict(usableCandidateInference=True),
        populationCounts=spec["populationCounts"], baselineBranches=deepcopy(branches), candidateBranches=deepcopy(branches), paired=paired)
    commitment = dict(schema=PUBLIC, manifest=manifest, event=event, artifacts=[spec_pin, actual, gen_pin, base_pin,
        cand_pin, fit_pin, raw_pin, extraction_pin, transport_pin])
    refresh_event(commitment)
    return commitment


def refresh_event(value):
    value["eventSha256"] = hashlib.sha256(json.dumps(value["event"], sort_keys=True, allow_nan=False,
                                                   separators=(",", ":")).encode()).hexdigest()


def reader_request(directory):
    return save(directory / "reader_request.json", dict(relocations=[]))["path"]


def rebound_guard(commitment, field, value):
    """Rebind all affected files so the negative test reaches semantic checks."""
    changed = deepcopy(commitment)
    pins = changed["artifacts"]
    baseline = json.loads(Path(pins[3]["path"]).read_text())
    candidate = json.loads(Path(pins[4]["path"]).read_text())
    extraction = json.loads(Path(pins[7]["path"]).read_text())
    baseline["raw"]["diagnostics"][field] = value
    pins[6] = save(Path(pins[6]["path"]), baseline["raw"])
    baseline["rawArtifact"] = pins[6]
    pins[3] = save(Path(pins[3]["path"]), baseline)
    candidate["baseline"] = pins[3]
    candidate["rawCandidate"]["diagnostics"] = deepcopy(baseline["raw"]["diagnostics"])
    pins[4] = save(Path(pins[4]["path"]), candidate)
    extraction["fits"][changed["event"]["fitKey"]]["diagnostics"][field] = value
    pins[7] = save(Path(pins[7]["path"]), extraction)
    refresh_event(changed)
    return changed
