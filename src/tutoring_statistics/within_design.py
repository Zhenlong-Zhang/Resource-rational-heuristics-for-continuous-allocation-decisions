"""Two closed Gaussian studies with distinct immutable populations and laws."""
import copy

ARMS = ('C1', 'C2', 'C3')
COMPARISONS = ('C1-C2:A', 'C1-C2:B', 'C1-C3:A', 'C1-C3:B')
WITHIN_METHOD = 'h10_within_participant_gaussian_t_IUT2_Holm4_candidate_v1'
COVERAGE_NAMESPACE = 'r11-h10-within-cr2-validation-v1-20261028'
COVERAGE_BASES = dict(retention=2002610280, design=2012610280, response=2022610280)
MIDPOINT_NAMESPACE = 'r11-h10-midpoint-within-power-v1-20261104'
MIDPOINT_CASE = 'h10_midpoint_initial_amount_recruit193_mcar08'
MIDPOINT_BASES = dict(retention=2032611040, design=2042611040, response=2052611040)
MIDPOINT_RATE = 'rho_geometric_0.001_0.01'
MIDPOINT_PRICE = .0031622776601683794
# The selected synthetic scenario used these stored decimal values, not a new refit.
# The separate descriptive mapping checks their source-derived parity.
MIDPOINT_SLOPES = dict(C1=dict(A=.599999999999946, B=.600000000000064),
    C2=dict(A=1.20896595372047, B=.097267348859979),
    C3=dict(A=1.52846405654907, B=.586772857425495))
MIDPOINT_LAW = dict(rateID=MIDPOINT_RATE, rho=MIDPOINT_PRICE, methodID=WITHIN_METHOD,
    recruitedPerArm=193, retentionProbability=.8, rowsPerParticipant=10,
    center=[50., 50.], centerMean=60., randomInterceptSD=8., residualSD=12.,
    responseTransform='none',
    libraryLaw='uniform_accepted_mapping_midpoint_whole_person_v0_to_v8_numeric_main1_to10',
    slopes=MIDPOINT_SLOPES)


def require(value, message):
    if not value:
        raise ValueError(message)


def coverage_cases():
    result = []
    for index, suffix in ((1, 's1'), (2, 's5_l20')):
        slopes = {a: dict(A=.25, B=-.25) for a in ARMS}
        if index == 2:
            slopes['C2'] = dict(A=.55, B=-.55)
            slopes['C3'] = dict(A=.55, B=-.55)
        result.append(dict(caseID='h10wcv1_initial_amount_'+suffix+'_recruit193_mcar08',
            caseIndex=index, view='initial_amount', N=193, recruitedPerArm=193,
            populationRegime='recruit193_mcar08', eligibilityProbability=.8,
            fixedEligiblePerArm=None, parameters=dict(alpha=60., sdU=8.,
                sigmaByArm={a: 12. for a in ARMS}, slopes=slopes)))
    return result


def coverage_case(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 12129, 'Unplanned coverage ordinal')
    index = 1 if ordinal < 10000 else 2
    rep = ordinal+1 if index == 1 else ordinal-9999
    case = coverage_cases()[index-1]
    case.update(repID=rep, **{role+'Seed': base+100000*index+rep
                             for role, base in COVERAGE_BASES.items()})
    return case


def coverage_manifest():
    return dict(mode='fresh_within_validation', B={'1': 10000, '2': 2129},
        cases=coverage_cases(), expectedOrdinals=list(range(12129)), expectedTotal=12129,
        namespace=COVERAGE_NAMESPACE,
        method='h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1',
        upstreamMethod='participant_CR2_Satterthwaite_v1')


def midpoint_case(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 4612, 'Unplanned midpoint ordinal')
    return dict(ordinal=ordinal, repID=ordinal+1, namespace=MIDPOINT_NAMESPACE,
        caseID=MIDPOINT_CASE, methodID=WITHIN_METHOD, law=copy.deepcopy(MIDPOINT_LAW),
        seeds={role: base+ordinal+1 for role, base in MIDPOINT_BASES.items()})


def plan(study):
    require(study in ('coverage', 'midpoint'), 'Unknown closed study')
    if study == 'coverage':
        return dict(study=study, inputPlan=coverage_manifest(), expectedTotal=12129,
            expectedOrdinals=list(range(12129)), withinMethod=WITHIN_METHOD,
            scientificScope='fixed_Gaussian_laws_not_human_adoption')
    return dict(study=study, namespace=MIDPOINT_NAMESPACE, expectedTotal=4612,
        expectedOrdinals=list(range(4612)), law=copy.deepcopy(MIDPOINT_LAW),
        seedBases=dict(MIDPOINT_BASES), epsilon=.02, confidence=.95,
        scientificScope='one_fixed_synthetic_midpoint_not_human_power')
