"""Independent scalar midpoint reduction; no producer reducer is called."""
import math
from collections import Counter

M = 4612
ARMS = ('C1', 'C2', 'C3')
COMPARISONS = ('C1-C2:A', 'C1-C2:B', 'C1-C3:A', 'C1-C3:B')
REFS = tuple(f'{comparison}:{suffix}' for comparison in COMPARISONS for suffix in ('D', 'S'))
STATES = ('available', 'statistical_unavailable', 'technical_unresolved')
TRUE_NULL = dict(zip(COMPARISONS, (False, True, False, True)))
CASE = 'h10_midpoint_initial_amount_recruit193_mcar08'
METHOD = 'h10_within_participant_gaussian_t_IUT2_Holm4_candidate_v1'
NAMESPACE = 'r11-h10-midpoint-within-power-v1-20261104'
SEED_BASES = {'retention': 2032611040, 'design': 2042611040, 'response': 2052611040}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def compare(actual, expected, label='final report'):
    if type(expected) is dict:
        require(type(actual) is dict and actual.keys() == expected.keys(), f'{label} keys')
        for key, value in expected.items():
            compare(actual[key], value, f'{label}.{key}')
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), f'{label} length')
        for index, value in enumerate(expected):
            compare(actual[index], value, f'{label}[{index}]')
    elif type(expected) is float:
        require(numeric(actual) and math.isclose(actual, expected, abs_tol=2e-9, rel_tol=2e-8),
                f'{label} continuous value')
    else:
        require(type(actual) is type(expected) and actual == expected, f'{label} exact value')


def exact_cp(successes, total):
    from scipy.special import betaincinv

    require(type(successes) is int and type(total) is int and 0 <= successes <= total,
            'Invalid binomial counts')
    if total == 0:
        return None
    lower = float(betaincinv(successes, total-successes+1, .025)) if successes else 0.
    upper = float(betaincinv(successes+1, total-successes, .975)) if successes < total else 1.
    require(0 <= lower <= upper <= 1, 'Invalid exact binomial interval')
    return [lower, upper]


def proportion(successes, total):
    interval = exact_cp(successes, total)
    rate = successes/total if total else None
    return dict(denominator=total, count=successes, rate=rate, CP95=interval,
                MCSE=math.sqrt(rate*(1-rate)/total) if total else None)

def linear_truths(slopes):
    result = {}
    for comparison in COMPARISONS:
        pair, need = comparison.split(':')
        first, second = pair.split('-')
        left, right = slopes[first][need], slopes[second][need]
        require((abs(left) >= abs(right)) is TRUE_NULL[comparison], 'Changed mixed-null truth')
        result[f'{comparison}:D'] = left-right
        result[f'{comparison}:S'] = left+right
    return result


def classified_event(event, truths):
    """Check the accepted event algebra while retaining the original decisions."""
    ordinal = event['ordinal']
    require(type(ordinal) is int and 0 <= ordinal < M and type(event['repID']) is int
            and event['repID'] == ordinal+1 and event['caseID'] == CASE
            and event['originalGuardUsedForSelection'] is False, 'Wrong scientific event identity')
    state, reason, retained = event['methodStatus'], event['methodReason'], event['actualK']
    require(state in STATES and (reason is None if state == 'available' else type(reason) is str and bool(reason))
            and retained.keys() == set(ARMS) and all(type(k) is int and 0 <= k <= 193 for k in retained.values()),
            'Wrong method disposition or actual K')

    evaluation = event['evaluation']
    require(evaluation['usedByEstimator'] is False and evaluation['magnitudeTrueNull'].keys() == TRUE_NULL.keys()
            and all(evaluation['magnitudeTrueNull'][k] is truth for k, truth in TRUE_NULL.items())
            and evaluation['references'].keys() == event['references'].keys() == set(REFS),
            'Wrong fixed truth evaluation or reference set')
    references = {}
    for key in REFS:
        reference, outcome = event['references'][key], evaluation['references'][key]
        covered, endpoint = outcome['covered'], outcome['endpointUnresolved']
        require(reference['linearID'] == key and reference['status'] == outcome['referenceStatus'] == state
                and reference['reason'] == reason and numeric(outcome['truth']) and outcome['truth'] == truths[key]
                and type(endpoint) is bool, f'Changed reference disposition or truth for {key}')
        if state == 'available':
            interval = reference['CI']
            require(type(interval) is list and len(interval) == 2 and all(map(numeric, interval))
                    and interval[0] < interval[1], f'Invalid accepted interval for {key}')
            require((covered is None if endpoint else covered is (interval[0] < truths[key] < interval[1])),
                    f'Wrong resolved coverage for {key}')
        else:
            require(reference['CI'] is None and covered is None and not endpoint,
                    f'Unavailable reference has coverage for {key}')
        references[key] = dict(status=state, reason=reason, covered=covered, endpointUnknown=endpoint)

    family = event['composition']
    known = state != 'technical_unresolved'
    available = int(state == 'available') * 4
    require(family['familySize'] == 4 and family['alpha'] == .05
            and family['comparisonIDs'] == list(COMPARISONS) and family['tests'].keys() == TRUE_NULL.keys()
            and family['status'] == ('complete' if known else 'technical_unresolved')
            and family['holmComputed'] is known and event['technicalCompleted'] is known
            and type(event['availableComparisons']) is int and event['availableComparisons'] == available
            and family['completeFamilyAvailable'] is event['A4'] is (available == 4),
            'Changed fixed family or availability')
    rejected = {}
    for key in COMPARISONS:
        test = family['tests'][key]
        adjusted, padded, decision = test['holmP'], test['holmPaddingP'], test['holmReject']
        require(test['comparisonID'] == key and test['status'] == state, f'Changed comparison identity for {key}')
        if not known:
            require(adjusted is padded is decision is None, 'Technical unknown entered Holm as a nonrejection')
        elif state == 'statistical_unavailable':
            require(adjusted is test['pMagnitude'] is None and numeric(padded) and padded == 1 and decision is False,
                    'Changed statistical-unavailable Holm padding')
        else:
            require(numeric(adjusted) and numeric(padded) and 0 <= padded <= adjusted <= 1
                    and padded == test['pMagnitude'] and decision is (adjusted < .05),
                    'Changed accepted strict Holm decision')
        rejected[key] = decision

    values = dict.fromkeys(('R', 'F', 'D'))
    if known:
        values = dict(R=any(rejected.values()), F=any(rejected[k] for k in COMPARISONS if TRUE_NULL[k]),
                      D=any(rejected[k] for k in COMPARISONS if not TRUE_NULL[k]))
    require(family['anyMagnitudeRejects'] is values['R'] and all(event[k] is v for k, v in values.items()),
            'Changed separate R/F/D events')
    return dict(ordinal=ordinal, actualK=retained, methodStatus=state, methodReason=reason,
                references=references, comparisonStatuses={key: state for key in COMPARISONS},
                A4=available == 4, **values)

def classify_primary(successes, unknown, total=M):
    require(all(type(value) is int for value in (successes, unknown, total))
            and total > 0 and 0 <= successes <= total-unknown <= total, 'Invalid fixed-denominator D counts')
    if unknown:
        return 'unresolved'
    if 50*successes >= 41*total:
        return 'supports_at_least_80_percent'
    if 50*successes < 39*total:
        return 'below_80_percent'
    return 'unresolved'


def reduce_rows(rows, total=M):
    """Count accepted classifications, keeping absent members in the planned denominator."""
    seen = set()
    method_states, method_reasons, available_comparisons = Counter(), Counter(), Counter()
    event_states = {key: Counter() for key in ('R', 'F', 'D')}
    reference_states = {key: Counter() for key in REFS}
    reference_reasons = {key: Counter() for key in REFS}
    k_values = {arm: [] for arm in ARMS}
    complete_available = 0
    for row in rows:
        ordinal = row['ordinal']
        require(type(ordinal) is int and 0 <= ordinal < total and ordinal not in seen,
                'Duplicate or out-of-range received ordinal')
        seen.add(ordinal)
        method_states[row['methodStatus']] += 1
        if row['methodReason'] is not None:
            method_reasons[row['methodReason']] += 1
        complete_available += int(row['A4'])
        for arm in ARMS:
            k_values[arm].append(row['actualK'][arm])
        for key in COMPARISONS:
            available_comparisons[key] += int(row['comparisonStatuses'][key] == 'available')
        for key, counts in event_states.items():
            counts[row[key]] += 1
        for key, counts in reference_states.items():
            ref = row['references'][key]
            counts[ref['status']] += 1
            counts['covered'] += int(ref['covered'] is True)
            counts['coverageKnown'] += int(ref['covered'] is not None)
            counts['endpointUnknown'] += int(ref['endpointUnknown'])
            if ref['reason'] is not None:
                reference_reasons[key][ref['reason']] += 1

    present = len(seen)
    missing = total-present
    counts = dict(M=total, present=present, received=present, missing=missing,
                  T=present-method_states['technical_unresolved'], A4=complete_available,
                  Aj=dict(available_comparisons), statisticalUnavailable=method_states['statistical_unavailable'],
                  technicalUnknown=method_states['technical_unresolved'])
    for key in COMPARISONS:
        counts['Aj'].setdefault(key, 0)
    retained = {arm: dict(n=len(values), sum=sum(values), minimum=min(values, default=None),
                          maximum=max(values, default=None)) for arm, values in k_values.items()}
    events = {}
    for key, cells in event_states.items():
        true, false = cells[True], cells[False]
        known = true+false
        unknown = total-known
        conditional = proportion(true, known)
        events[key] = dict(true=true, false=false, unknown=unknown, conditional=conditional,
                           full=conditional if known == total else None,
                           possibleFraction=[true/total, (true+unknown)/total])
    references = {}
    for key, cells in reference_states.items():
        resolved = cells['coverageKnown']
        conditional = proportion(cells['covered'], resolved)
        references[key] = dict(available=cells['available'], coverageKnown=resolved, covered=cells['covered'],
                               endpointUnknown=cells['endpointUnknown'],
                               statisticalUnavailable=cells['statistical_unavailable'],
                               technicalUnknown=cells['technical_unresolved'], missing=missing,
                               reasons=dict(reference_reasons[key]), conditional=conditional,
                               full=conditional if resolved == total else None)
    discoveries = events['D']
    unknown = discoveries['unknown']
    estimate = discoveries['full']['rate'] if unknown == 0 else None
    band = [max(0., estimate-.02), min(1., estimate+.02)] if estimate is not None else None
    primary = dict(estimand='P(at_least_one_true_alternative_rejected)', known=total-unknown,
                   true=discoveries['true'], false=discoveries['false'], unknown=unknown,
                   pHat=estimate, hoeffdingBand=band, epsilon=.02, confidence=.95, threshold=.8,
                   fullFlowEligible=unknown == 0,
                   decision=classify_primary(discoveries['true'], unknown, total),
                   possibleFraction=discoveries['possibleFraction'])
    return dict(counts=counts, reasons=dict(method_reasons), events=events, references=references,
                retainedK=retained, primary=primary)
