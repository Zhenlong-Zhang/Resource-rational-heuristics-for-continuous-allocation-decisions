"""Independent design-sign, Student-tail and paired checks of saved references.

No producer import, estimator replay or I/O. Exact pairwise differences and a
regularized-beta tail provide independent routes to the common-t decisions.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import math

ARMS = ('C1', 'C2', 'C3')
NEEDS = ('A', 'B')
SLOPE_NAMES = tuple((arm, need) for arm in ARMS for need in NEEDS)
COMPARISONS = ('C1-C2:A', 'C1-C2:B', 'C1-C3:A', 'C1-C3:B')
REFS = tuple(f'{key}:{suffix}' for key in COMPARISONS for suffix in ('D', 'S'))
ORIGINAL_METHOD = 'h10_within_participant_gaussian_t_IUT2_Holm4_candidate_v1'
METHOD = 'h10_within_common_t_sign_congruence_IUT2fallback_Holm4_candidate_v1'
NAMESPACE = 'r11-h10-within-common-t-saved-development-v1-20261104'
STATES = ('available', 'statistical_unavailable', 'technical_unresolved')
TOTALS = {1: 10000, 2: 2129}
TABLE = ('00', '01', '10', '11')
SIGNS = {'negative': -1, 'zero': 0, 'positive': 1, 'uncertified': None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@lru_cache(maxsize=64)
def schedule_crossproducts(rows):
    """Ten times centered Gram, exactly over the input binary64 values."""
    rational = tuple(tuple(Fraction(float(value)) for value in row) for row in rows)
    aa, ab, bb = Fraction(0), Fraction(0), Fraction(0)
    for (a1, b1), (a2, b2) in combinations(rational, 2):
        da, db = a1-a2, b1-b2
        aa += da*da
        ab += da*db
        bb += db*db
    return aa, ab, bb


def predictor_rows(actual):
    """Project the authenticated request before design-only certification."""
    require(type(actual) is dict and set(actual) == {'model', 'reference', 'rows'}
            and actual['model'] == 'H10' and actual['reference'] == 'C2'
            and type(actual['rows']) is list, 'Changed accepted native request')
    fields = ('participant', 'condition', 'need_A', 'need_B')
    for row in actual['rows']:
        require(type(row) is dict and set(row) == {*fields, 'y'} and finite(row['y']),
                'Wrong accepted native row shape')
        yield tuple(row[field] for field in fields)


def exact_design(predictors, counts):
    """Use only X and participant membership to certify the design."""
    require(set(counts) == set(ARMS)
            and all(type(count) is int and 0 <= count <= 193 for count in counts.values()),
            'Changed accepted retained counts')
    grouped = {}
    for participant, condition, need_a, need_b in predictors:
        require(type(participant) is str and participant and condition in ARMS
                and finite(need_a) and finite(need_b), 'Wrong accepted participant design row')
        arm, design = grouped.setdefault(participant, (condition, []))
        require(arm == condition, 'Participant crosses native arms')
        design.append((need_a, need_b))
    schedules = {arm: Counter() for arm in ARMS}
    for arm, rows in grouped.values():
        require(len(rows) == 10, 'Expected ten complete rows per retained participant')
        schedules[arm][tuple(rows)] += 1
    require({arm: sum(schedules[arm].values()) for arm in ARMS} == counts, 'Different actual retained K')
    blocks = {}
    for arm, multiplicities in schedules.items():
        aa, ab, bb = Fraction(0), Fraction(0), Fraction(0)
        for schedule, count in multiplicities.items():
            sa, sab, sb = schedule_crossproducts(schedule)
            aa += count*sa
            ab += count*sab
            bb += count*sb
        determinant = aa*bb-ab*ab
        require(aa >= 0 and bb >= 0 and determinant >= 0, 'Invalid exact within design crossproducts')
        blocks[arm] = (aa, ab, bb, determinant, len(multiplicities))
    return blocks


def variance_signs(blocks):
    """Cross-multiply positive determinants to certify l_D C l_S'."""
    signs = {}
    for comparison in COMPARISONS:
        pair, need = comparison.split(':')
        left, right = pair.split('-')
        lhs, rhs = blocks[left], blocks[right]
        if lhs[3] == 0 or rhs[3] == 0:
            signs[comparison] = None
            continue
        diagonal_cofactor = 2 if need == 'A' else 0
        numerator = lhs[diagonal_cofactor]*rhs[3]-rhs[diagonal_cofactor]*lhs[3]
        signs[comparison] = (numerator > 0)-(numerator < 0)
    return signs


def check_saved_information(blocks, residual_variance, covariance):
    """Check covariance against X, but never use it to select the sign."""
    require(type(covariance) is list and len(covariance) == 6
            and all(type(row) is list and len(row) == 6 for row in covariance),
            'Changed saved six-slope covariance dimensions')
    for index, arm in enumerate(ARMS):
        aa, ab, bb, determinant, _ = blocks[arm]
        scaled = ((bb, -ab), (-ab, aa))
        for left in range(2):
            for right in range(6):
                expected = (residual_variance*float(10*scaled[left][right % 2]/determinant)
                            if right//2 == index else 0.)
                close_number(covariance[2*index+left][right], expected,
                             'Saved covariance disagrees with actual centered design information')


def student_upper(df, statistic):
    from scipy.special import betainc
    require(type(df) is int and df > 0 and finite(statistic) and statistic >= 0, 'Invalid common-t tail input')
    x = (math.sqrt(df)/math.hypot(math.sqrt(df), statistic))**2
    return .5*float(betainc(.5*df, .5, x))


def magnitude_from_t(d, s, df, sign, original):
    require(type(df) is int and df > 0 and finite(d) and finite(s) and finite(original) and 0 <= original <= 1
            and (sign is None or type(sign) is int and sign in (-1, 0, 1)), 'Invalid comparison inputs')
    if not (d < 0 < s or s < 0 < d):
        return 'same_or_zero_sign', 1.
    if sign is None or sign < 0:
        return 'original_iut2_fallback', original
    return 'certified_nonnegative_common_t', student_upper(df, min(abs(d), abs(s)))


def holm_flags(probabilities, states):
    require(set(probabilities) == set(states) == set(COMPARISONS)
            and all(state in STATES for state in states.values()), 'Changed fixed Holm family')
    if any(state == 'technical_unresolved' for state in states.values()):
        return dict.fromkeys(COMPARISONS), dict.fromkeys(COMPARISONS)
    padded = {key: probabilities[key] if states[key] == 'available' else 1. for key in COMPARISONS}
    require(all(finite(p) and 0 <= p <= 1 for p in padded.values()), 'Invalid available probability')
    ranked = sorted(COMPARISONS, key=padded.__getitem__)
    adjusted, rejected = {}, {}
    running = 0.
    for rank, key in enumerate(ranked):
        running = max(running, min(1., (len(COMPARISONS)-rank)*padded[key]))
        adjusted[key] = running if states[key] == 'available' else None
        rejected[key] = states[key] == 'available' and running < .05
    return adjusted, rejected


def family_events(flags, true_null):
    require(set(flags) == set(true_null) == set(COMPARISONS)
            and all(type(value) is bool for value in true_null.values())
            and all(value is None or type(value) is bool for value in flags.values()), 'Invalid family event inputs')
    if any(flag is None for flag in flags.values()):
        require(all(flag is None for flag in flags.values()), 'Partly classified technical Holm family')
        return dict(R=None, F=None, D=None)
    false_event = any(flags[key] for key in COMPARISONS if true_null[key]) if any(true_null.values()) else None
    true_event = any(flags[key] for key in COMPARISONS if not true_null[key]) if not all(true_null.values()) else None
    return dict(R=any(flags.values()), F=false_event, D=true_event)


def close_number(actual, expected, message):
    require(finite(actual) and finite(expected) and math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-9), message)


def compare(actual, expected, label):
    if type(expected) is dict:
        require(type(actual) is dict and actual.keys() == expected.keys(), label+' keys')
        for key, value in expected.items():
            compare(actual[key], value, f'{label}.{key}')
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), label+' length')
        for index, value in enumerate(expected):
            compare(actual[index], value, f'{label}[{index}]')
    elif type(expected) is float:
        close_number(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label)


def rational_pin(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def certificate_from_blocks(blocks, counts):
    arm_gram = {}
    for arm, (aa, ab, bb, determinant, distinct) in blocks.items():
        arm_gram[arm] = dict(AA=rational_pin(aa/10), AB=rational_pin(ab/10), BB=rational_pin(bb/10),
                            determinant=rational_pin(determinant/100), participants=counts[arm], distinctDesigns=distinct)
    differences = {}
    for key in COMPARISONS:
        pair, need = key.split(':')
        left, right = pair.split('-')
        lhs, rhs = blocks[left], blocks[right]
        cofactor = 2 if need == 'A' else 0
        if lhs[3] and rhs[3]:
            numerator = 10*(lhs[cofactor]*rhs[3]-rhs[cofactor]*lhs[3])
            differences[key] = rational_pin(numerator/(lhs[3]*rhs[3]))
        else:
            differences[key] = None
    return dict(schema='h10-common-t-design-certificate-v1', arithmetic='exact_binary64_rational',
                responseUsed=False, residualScaleUsed=False, originalGuardUsed=False, actualK=counts,
                rowCount=10*sum(counts.values()), armGram=arm_gram, covarianceDifferences=differences,
                signs=variance_signs(blocks), fullRank=all(blocks[arm][3] > 0 for arm in ARMS))


def within_vector(comparison, kind):
    pair, need = comparison.split(':')
    left, right = pair.split('-')
    require(kind in ('D', 'S') and need in NEEDS, 'Invalid within contrast')
    vector = [0]*len(SLOPE_NAMES)
    vector[SLOPE_NAMES.index((left, need))] = 1
    vector[SLOPE_NAMES.index((right, need))] = -1 if kind == 'D' else 1
    return vector


def saved_reference(actual, counts, case, stage, accepted):
    """Check saved references and independently derive new comparison decisions."""
    require(stage['conditionedOnOriginalGuard'] is False and stage['truthEvaluation']['usedByEstimator'] is False
            and accepted['methods']['within']['conditionedOnOriginalGuard'] is False,
            'The original REML guard must not select a within result')
    value = stage['result']; previous = accepted['methods']['within']; state = value['status']
    require(state in STATES and state == previous['status'] and value['reason'] == previous['reason']
            and value['methodID'] == ORIGINAL_METHOD and value['design']['actualK'] == counts == accepted['actualK']
            and value['coefficientNames'] == [f'{arm}:{need}' for arm, need in SLOPE_NAMES]
            and set(value['references']) == set(previous['references']) == set(REFS),
            'Changed saved within estimator identity')
    slopes = case['parameters']['slopes']
    nulls, truth = {}, {}
    for comparison in COMPARISONS:
        pair, need = comparison.split(':'); left, right = pair.split('-')
        a, b = slopes[left][need], slopes[right][need]
        nulls[comparison] = abs(a) >= abs(b)
        truth[f'{comparison}:D'], truth[f'{comparison}:S'] = a-b, a+b
    require(type(case['caseIndex']) is int and case['caseIndex'] in (1, 2)
            and nulls == dict.fromkeys(COMPARISONS, case['caseIndex'] == 1)
            and accepted['magnitudeTrueNull'] == stage['truthEvaluation']['magnitudeTrueNull'] == nulls
            and all(type(v) is bool for v in stage['truthEvaluation']['magnitudeTrueNull'].values())
            and all(type(v) is bool for v in accepted['magnitudeTrueNull'].values()), 'Changed frozen S1/S5 magnitude nulls')
    df = value['df']
    if state == 'available':
        require(type(df) is int and df > 0 and df == 9*sum(counts.values())-6
                and value['design']['rank'] == 6 and value['design']['projectedDimension'] == df+6
                and finite(value['residualVariance']) and value['residualVariance'] > 0,
                'Original available references lack the common residual scale/df')
    else:
        require(df is None, 'Unavailable original result claims a residual df')
    for key in REFS:
        comparison, kind = key.rsplit(':', 1)
        original, reviewed = value['references'][key], previous['references'][key]
        evaluation = stage['truthEvaluation']['references'][key]
        require(original['linearID'] == key and original['L'] == reviewed['L'] == within_vector(comparison, kind)
                and original['status'] == reviewed['status'] == evaluation['referenceStatus'] == state
                and evaluation['truth'] == reviewed['truth'] == truth[key]
                and evaluation['covered'] is reviewed['covered']
                and evaluation['endpointUnresolved'] is reviewed['coverageNumericallyUnresolved'],
                f'Changed canonical within contrast, truth or interval disposition for {key}')
        if state == 'available':
            require(type(original['df']) is int and type(reviewed['df']) is int
                    and original['df'] == reviewed['df'] == df and finite(original['SE']) and original['SE'] > 0,
                    f'Changed shared df or valid SE for {key}')
            for name in ('estimate', 'SE', 'statistic', 'lessP', 'greaterP', 'twoSidedP'):
                close_number(original[name], reviewed[name], f'Saved scalar proof mismatch for {key}/{name}')
            require(len(original['CI']) == len(reviewed['CI']) == 2
                    and original['CI'][0] < original['CI'][1], f'Changed original interval for {key}')
            for source, received in zip(original['CI'], reviewed['CI']):
                close_number(source, received, f'Changed original interval endpoint for {key}')
            close_number(original['statistic'], original['estimate']/original['SE'], f'Changed t statistic for {key}')
        else:
            require(all(original[name] is None for name in ('estimate', 'statistic', 'df', 'SE', 'CI')),
                    f'Unavailable contrast retains numerical test input for {key}')
    family = value['composition']
    require(family['familySize'] == 4 and family['alpha'] == .05 and family['comparisonIDs'] == list(COMPARISONS)
            and set(family['tests']) == set(previous['composition']['tests']) == set(COMPARISONS),
            'Changed original fixed Holm family')
    states = dict.fromkeys(COMPARISONS, state)
    old_ps, new_ps, branches = {}, {}, {}
    blocks = exact_design(predictor_rows(actual), counts)
    certificate = certificate_from_blocks(blocks, counts)
    signs = certificate['signs']
    if state == 'available':
        require(certificate['fullRank'], 'Available within fit has singular exact design')
        check_saved_information(blocks, value['residualVariance'], value['covariance'])
    for comparison in COMPARISONS:
        old, old_proof = family['tests'][comparison], previous['composition']['tests'][comparison]
        require(old['comparisonID'] == old_proof['comparisonID'] == comparison
                and old['status'] == old_proof['status'] == state and old['holmReject'] is old_proof['holmReject'],
                'Changed original paired Holm status or decision')
        old_ps[comparison] = old['pMagnitude']
        if state == 'available':
            d, s = (value['references'][f'{comparison}:{kind}'] for kind in ('D', 'S'))
            old_formula = min(1., 2*min(max(d['lessP'], s['greaterP']), max(d['greaterP'], s['lessP'])))
            close_number(old['pMagnitude'], old_formula, 'Changed original two-orientation IUT2 probability')
            close_number(old['pMagnitude'], old_proof['pMagnitude'], 'Changed original magnitude probability in proof')
            branches[comparison], new_ps[comparison] = magnitude_from_t(
                d['statistic'], s['statistic'], df, signs[comparison], old['pMagnitude'])
        else:
            require(old['pMagnitude'] is old_proof['pMagnitude'] is None, 'Unavailable original comparison has p')
            branches[comparison], new_ps[comparison] = state, None
    old_adjusted, old_flags = holm_flags(old_ps, states)
    new_adjusted, new_flags = holm_flags(new_ps, states)
    for comparison in COMPARISONS:
        test = family['tests'][comparison]
        require(test['holmReject'] is old_flags[comparison], 'Original Holm decision differs')
        if state == 'available':
            close_number(test['holmP'], old_adjusted[comparison], 'Original Holm adjusted probability differs')
        else:
            require(test['holmP'] is None and (test['holmPaddingP'] == 1 if state == 'statistical_unavailable'
                    else test['holmPaddingP'] is None), 'Wrong old statistical padding or technical unknown')
    analysis = dict(status=state, reason=value['reason'], truth=nulls, actualK=counts,
                    signs=signs, branches=branches, oldP=old_ps, newP=new_ps,
                    oldAdjusted=old_adjusted, newAdjusted=new_adjusted,
                    oldHolm=old_flags, newHolm=new_flags,
                    oldEvents=family_events(old_flags, nulls), newEvents=family_events(new_flags, nulls))
    return dict(certificate=certificate, analysis=analysis)


def calculate(data):
    """Public saved-input seam; all historical admission is external."""
    case, ordinal = data['case'], data['ordinal']
    require(type(ordinal) is int and 0 <= ordinal < 12129, 'Foreign saved ordinal')
    index = 1 if ordinal < 10000 else 2
    require(type(case['caseIndex']) is int and case['caseIndex'] == index
            and type(case['repID']) is int and case['repID'] == ordinal-(0 if index == 1 else 10000)+1
            and case['caseID'] == 'h10wcv1_initial_amount_'+('s1' if index == 1 else 's5_l20')+'_recruit193_mcar08',
            'Changed original case/rep identity')
    for arm in ARMS:
        magnitude = .25 if index == 1 or arm == 'C1' else .55
        require(case['parameters']['slopes'][arm] == dict(A=magnitude, B=-magnitude), 'Changed frozen slopes')
    return saved_reference(data['actual'], data['counts'], case, data['within'], data['accepted'])


def pair_cell(left, right):
    require(left is None or type(left) is bool, 'Invalid original decision')
    require(right is None or type(right) is bool, 'Invalid candidate decision')
    return 'unknown' if left is None or right is None else f'{int(left)}{int(right)}'


def proportion(successes, size):
    from scipy.special import betaincinv
    require(type(successes) is int and type(size) is int and 0 <= successes <= size,
            'Invalid independent descriptive denominator')
    if not size:
        return dict(count=successes, denominator=size, rate=None, CP95=None, MCSE=None)
    rate = successes/size
    lower = float(betaincinv(successes, size-successes+1, .025)) if successes else 0.
    upper = float(betaincinv(successes+1, size-successes, .975)) if successes < size else 1.
    require(0 <= lower <= upper <= 1, 'Invalid independent CP interval')
    return dict(count=successes, denominator=size, rate=rate, CP95=[lower, upper],
                MCSE=math.sqrt(rate*(1-rate)/size))


def paired(cells):
    table = {key: cells[key] for key in TABLE}
    require(all(type(v) is int and v >= 0 for v in table.values()), 'Invalid independent paired counts')
    known = sum(table.values())
    gains, losses = table['01'], table['10']
    # Integer arithmetic keeps exactly zero variance at concordant boundaries.
    variance_numerator = known*(gains+losses)-(gains-losses)**2
    require(variance_numerator >= 0, 'Invalid independent paired variance')
    return dict(Q=known, table=table, delta=(gains-losses)/known if known else None,
                pairedMCSE=math.sqrt(variance_numerator/(known**2*(known-1))) if known > 1 else None)


def marginals(cells):
    return cells['10']+cells['11'], cells['01']+cells['11']


class CaseCounts:
    """Reduce supplied scalars independently without retaining member objects."""
    def __init__(self, index):
        self.total = TOTALS[index]
        self.null = index == 1
        self.events = ('R', 'F') if self.null else ('R', 'D')
        self.inapplicable = 'D' if self.null else 'F'
        self.states = Counter(); self.reasons = Counter(); self.endpoints = Counter()
        self.cells = {key: Counter() for key in (*COMPARISONS, *self.events)}
        self.available = {key: Counter() for key in self.cells}
        self.signs = {key: Counter() for key in COMPARISONS}
        self.branches = {key: Counter() for key in COMPARISONS}

    def add(self, analysis, endpoints):
        state, reason = analysis['status'], analysis['reason']
        require(state in STATES and (reason is None or type(reason) is str)
                and set(endpoints) == set(REFS) and all(type(v) is bool for v in endpoints.values()),
                'Changed original availability or endpoint status')
        require(analysis['truth'] == dict.fromkeys(COMPARISONS, self.null)
                and all(type(flag) is bool for flag in analysis['truth'].values()), 'Changed original magnitude truth')
        self.states[state] += 1
        if reason:
            self.reasons[reason] += 1
        self.endpoints.update(key for key, value in endpoints.items() if value)
        for key in COMPARISONS:
            left, right = analysis['oldHolm'][key], analysis['newHolm'][key]
            self.add_decision(key, left, right, state)
            branch = analysis['branches'][key]
            require(branch in ('same_or_zero_sign', 'original_iut2_fallback', 'certified_nonnegative_common_t')
                    if state == 'available' else branch == state, 'Changed comparison branch')
            self.branches[key][branch] += 1
            sign = analysis['signs'][key]
            require(sign is None or type(sign) is int and sign in (-1, 0, 1), 'Changed exact sign state')
            if state == 'available':
                self.signs[key][sign] += 1
        for event_name, flags_name in (('oldEvents', 'oldHolm'), ('newEvents', 'newHolm')):
            values, flags = analysis[event_name], analysis[flags_name]
            require(set(values) == {'R', 'F', 'D'} and set(flags) == set(COMPARISONS)
                    and values[self.inapplicable] is None, 'Changed event applicability')
            expected = None if state == 'technical_unresolved' else any(flags.values())
            require(all(values[key] is expected for key in self.events), 'Changed same-mask family event')
        for key in self.events:
            self.add_decision(key, analysis['oldEvents'][key], analysis['newEvents'][key], state)

    def add_decision(self, key, left, right, state):
        cell = pair_cell(left, right)
        require((left is right is None) if state == 'technical_unresolved' else cell != 'unknown',
                'Technical unknown became a known paired classification')
        require(state != 'statistical_unavailable' or cell == '00', 'Statistical padding became a discovery')
        self.cells[key][cell] += 1
        if state == 'available':
            self.available[key][cell] += 1

    def report(self):
        present, available = sum(self.states.values()), self.states['available']
        comparisons, events = {}, {self.inapplicable: None}
        for key in COMPARISONS:
            cells, available_cells = self.cells[key], self.available[key]
            old, new = marginals(cells)
            old_available, new_available = marginals(available_cells)
            comparisons[key] = dict(Aj=available, T=present-cells['unknown'], unknown=cells['unknown'],
                paired=paired(cells), signCounts={label: self.signs[key][sign] for label, sign in SIGNS.items()},
                branchCounts=dict(self.branches[key]), oldRejected=old, newRejected=new,
                availableOld=proportion(old_available, available), availableNew=proportion(new_available, available))
        for key in self.events:
            cells, available_cells = self.cells[key], self.available[key]
            summary = paired(cells)
            known = summary['Q']
            old, new = marginals(cells)
            old_available, new_available = marginals(available_cells)
            unknown = self.total-known
            events[key] = dict(summary, unknown=cells['unknown'], old=proportion(old, known),
                new=proportion(new, known), availableOld=proportion(old_available, available),
                availableNew=proportion(new_available, available),
                fullOld=proportion(old, self.total) if not unknown else None,
                fullNew=proportion(new, self.total) if not unknown else None,
                plannedOldBounds=[old/self.total, (old+unknown)/self.total],
                plannedNewBounds=[new/self.total, (new+unknown)/self.total])
        return dict(M=self.total, present=present, missing=self.total-present,
                    T=present-self.states['technical_unresolved'], A4=available,
                    statisticalUnavailable=self.states['statistical_unavailable'],
                    technicalUnknown=self.states['technical_unresolved'], reasons=dict(self.reasons),
                    endpointUnknown={key: self.endpoints[key] for key in REFS},
                    comparisons=comparisons, events=events)


def report(rows):
    """Independent exactly-once scalar reduction; no provenance is inferred."""
    cases = {index: CaseCounts(index) for index in TOTALS}
    ordinals = set()
    for row in rows:
        ordinal = row['ordinal']
        require(type(ordinal) is int and 0 <= ordinal < sum(TOTALS.values()) and ordinal not in ordinals,
                'Duplicate or foreign independently received member')
        index = 1 if ordinal < TOTALS[1] else 2
        first = 0 if index == 1 else TOTALS[1]
        require(type(row['caseIndex']) is int and row['caseIndex'] == index
                and type(row['repID']) is int and row['repID'] == ordinal-first+1
                and row['caseID'] == 'h10wcv1_initial_amount_'+('s1' if index == 1 else 's5_l20')+'_recruit193_mcar08',
                'Changed fixed case and replicate membership')
        cases[index].add(row['analysis'], row['endpointUnknown'])
        ordinals.add(ordinal)
    reports = {str(index): counts.report() for index, counts in cases.items()}
    return dict(schema='h10-common-t-saved-statistics-v1', cases=reports, retrospectiveDevelopmentOnly=True,
                scientificAdoption=False, coverageRecomputed=False, linearIntervalsUnchanged=True,
                fullMeasurement=ordinals == set(range(sum(TOTALS.values()))))


def check_no_original_losses(summary):
    for item in summary['cases'].values():
        require(all(value['paired']['table']['10'] == 0 for value in item['comparisons'].values())
                and all(value is None or value['table']['10'] == 0 for value in item['events'].values()),
                'Original known rejection loss requires investigation before final acceptance')
