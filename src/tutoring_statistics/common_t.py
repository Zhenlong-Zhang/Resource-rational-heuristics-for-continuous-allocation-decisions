"""Fixed saved-input common-t comparison, not calibration or coverage repair.

Reuse the original within-participant Gaussian/common-scale references. Only
predictors and membership select the exact design-sign branch. This module
does no input generation, estimation, source admission or file I/O.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache
import math

METHOD = 'h10_within_common_t_sign_congruence_IUT2fallback_Holm4_candidate_v1'
NAMESPACE = 'r11-h10-within-common-t-saved-development-v1-20261104'
ORIGINAL_METHOD = 'h10_within_participant_gaussian_t_IUT2_Holm4_candidate_v1'
ARMS = ('C1', 'C2', 'C3')
COMPARISONS = ('C1-C2:A', 'C1-C2:B', 'C1-C3:A', 'C1-C3:B')
REFS = tuple(c+':'+k for c in COMPARISONS for k in ('D', 'S'))
STATES = ('available', 'statistical_unavailable', 'technical_unresolved')
TOTALS = {1: 10000, 2: 2129}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode_fraction(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


@lru_cache(maxsize=64)
def centered_gram(schedule):
    # Exact binary inputs, not rounded decimal values or an estimated covariance.
    x = [(Fraction(float(a)), Fraction(float(b))) for a, b in schedule]
    n = len(x); require(n == 10, 'Only complete native ten-trial designs')
    sa = sum((a for a, _ in x), Fraction(0)); sb = sum((b for _, b in x), Fraction(0))
    return (sum((a*a for a, _ in x), Fraction(0))-sa*sa/n,
            sum((a*b for a, b in x), Fraction(0))-sa*sb/n,
            sum((b*b for _, b in x), Fraction(0))-sb*sb/n)


def design_certificate(design_rows, counts):
    require(type(design_rows) is list, 'Design-only row list required')
    require(set(counts) == set(ARMS) and all(type(n) is int and 0 <= n <= 193 for n in counts.values()),
            'Invalid retained counts')
    people = {}
    for row in design_rows:
        require(type(row) is dict and set(row) == {'participant', 'condition', 'need_A', 'need_B'}
                and type(row['participant']) is str and row['participant'] and row['condition'] in ARMS
                and all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ('need_A', 'need_B')),
                'Design certificate rejects outcome or nonfinite fields')
        arm, values = people.setdefault(row['participant'], (row['condition'], []))
        require(arm == row['condition'], 'Cross-arm design')
        values.append((row['need_A'], row['need_B']))
    schedules = {a: Counter() for a in ARMS}
    for arm, values in people.values():
        require(len(values) == 10, 'Incomplete design group')
        schedules[arm][tuple(values)] += 1
    require({a: sum(schedules[a].values()) for a in ARMS} == counts, 'Actual design K mismatch')
    grams = {}; encoded = {}
    for arm in ARMS:
        aa = ab = bb = Fraction(0)
        for signature, count in schedules[arm].items():
            a, cross, b = centered_gram(signature)
            aa += count*a; ab += count*cross; bb += count*b
        determinant = aa*bb-ab*ab
        require(aa >= 0 and bb >= 0 and determinant >= 0, 'Invalid exact Gram')
        grams[arm] = (aa, ab, bb, determinant)
        encoded[arm] = dict(AA=encode_fraction(aa), AB=encode_fraction(ab), BB=encode_fraction(bb),
                            determinant=encode_fraction(determinant), participants=counts[arm],
                            distinctDesigns=len(schedules[arm]))
    signs = {}; differences = {}
    for key in COMPARISONS:
        pair, need = key.split(':'); left, right = pair.split('-')
        a, b = grams[left], grams[right]
        if a[3] <= 0 or b[3] <= 0:
            signs[key] = None; differences[key] = None
        else:
            cofactor = 2 if need == 'A' else 0
            difference = a[cofactor]/a[3]-b[cofactor]/b[3]
            signs[key] = (difference > 0)-(difference < 0)
            differences[key] = encode_fraction(difference)
    return dict(schema='h10-common-t-design-certificate-v1', arithmetic='exact_binary64_rational',
                formula='sum_xx_minus_outer_sum_over_ten', responseUsed=False, residualScaleUsed=False,
                originalGuardUsed=False, actualK=counts, rowCount=len(design_rows),
                armGram=encoded, covarianceDifferences=differences, signs=signs,
                fullRank=all(grams[a][3] > 0 for a in ARMS))


def holm(probabilities, states):
    require(set(probabilities) == set(states) == set(COMPARISONS)
            and all(s in STATES for s in states.values()), 'Changed fixed Holm family')
    if 'technical_unresolved' in states.values():
        return dict.fromkeys(COMPARISONS), dict.fromkeys(COMPARISONS)
    padded = {k: probabilities[k] if states[k] == 'available' else 1. for k in COMPARISONS}
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in padded.values()),
            'Invalid fixed-family probability')
    order = sorted(COMPARISONS, key=lambda k: (padded[k], COMPARISONS.index(k)))
    running = 0.; adjusted = {}; flags = {}
    for i, key in enumerate(order):
        running = min(1., max(running, (4-i)*padded[key]))
        adjusted[key] = running if states[key] == 'available' else None
        flags[key] = running < .05 if states[key] == 'available' else False
    return adjusted, flags


def events(flags, nulls):
    require(set(flags) == set(nulls) == set(COMPARISONS)
            and all(type(v) is bool for v in nulls.values())
            and all(v is None or type(v) is bool for v in flags.values()), 'Invalid family event')
    if any(v is None for v in flags.values()):
        require(all(v is None for v in flags.values()), 'Partly classified technical family')
        return dict(R=None, F=None, D=None)
    return dict(R=any(flags.values()),
                F=any(flags[k] for k in COMPARISONS if nulls[k]) if any(nulls.values()) else None,
                D=any(flags[k] for k in COMPARISONS if not nulls[k]) if not all(nulls.values()) else None)


def validate_reference_identity(data):
    value = data['within']['result']; case = data['case']; state = value['status']
    require(value['methodID'] == ORIGINAL_METHOD and state in STATES
            and value['coefficientNames'] == [arm+':'+need for arm in ARMS for need in ('A', 'B')]
            and set(value['references']) == set(REFS) and data['within']['conditionedOnOriginalGuard'] is False
            and data['within']['truthEvaluation']['usedByEstimator'] is False, 'Wrong unchanged within reference scope')
    require(type(case['caseIndex']) is int and case['caseIndex'] in (1, 2)
            and case['caseID'] == 'h10wcv1_initial_amount_'+('s1' if case['caseIndex'] == 1 else 's5_l20')+'_recruit193_mcar08'
            and type(case['repID']) is int and 1 <= case['repID'] <= TOTALS[case['caseIndex']]
            and type(data['ordinal']) is int
            and data['ordinal'] == (case['repID']-1 if case['caseIndex'] == 1 else 9999+case['repID']),
            'Wrong fixed cohort/case/rep identity')
    require(value['design']['actualK'] == data['counts'], 'Changed saved within counts')
    for arm in ARMS:
        magnitude = .25 if case['caseIndex'] == 1 or arm == 'C1' else .55
        require(case['parameters']['slopes'][arm] == dict(A=magnitude, B=-magnitude), 'Changed fixed signed law')
    for comparison in COMPARISONS:
        pair, need = comparison.split(':'); left, right = pair.split('-')
        a = case['parameters']['slopes'][left][need]; b = case['parameters']['slopes'][right][need]
        for kind, sign in (('D', -1.), ('S', 1.)):
            key = comparison+':'+kind; reference = value['references'][key]
            vector = [1. if arm == left and axis == need else sign if arm == right and axis == need else 0.
                      for arm in ARMS for axis in ('A', 'B')]
            require(reference['linearID'] == key and reference['L'] == vector and reference['status'] == state
                    and data['within']['truthEvaluation']['references'][key]['truth'] == a+sign*b,
                    'Changed exact within L/canonical truth')
            if state == 'available':
                require(type(value['df']) is int and value['df'] > 0
                        and type(reference['df']) is int and reference['df'] == value['df']
                        and type(reference['SE']) in (int, float) and math.isfinite(reference['SE']) and reference['SE'] > 0
                        and all(type(reference[k]) in (int, float) and math.isfinite(reference[k])
                                for k in ('estimate', 'statistic', 'lessP', 'greaterP'))
                        and math.isclose(reference['statistic'], reference['estimate']/reference['SE'], rel_tol=2e-8, abs_tol=2e-9),
                        'Invalid saved common-t reference')
            else:
                require(all(reference[k] is None for k in ('estimate', 'SE', 'df', 'statistic', 'CI')),
                        'Unavailable original reference with usable numbers')
    if state != 'available':
        require(value['df'] is None, 'Unavailable result claims residual df')
    family = value['composition']
    require(family['familySize'] == 4 and family['alpha'] == .05
            and family['comparisonIDs'] == list(COMPARISONS) and set(family['tests']) == set(COMPARISONS),
            'Changed original fixed family')
    return True


def candidate_probability(d, s, df, sign, old_probability):
    from scipy.special import stdtr

    require(type(df) is int and df > 0 and (sign is None or type(sign) is int and sign in (-1, 0, 1))
            and all(type(v) in (int, float) and math.isfinite(v) for v in (d, s, old_probability))
            and 0 <= old_probability <= 1, 'Invalid candidate common-t inputs')
    if not (d < 0 < s or s < 0 < d):
        return 'same_or_zero_sign', 1.
    if sign is None or sign < 0:
        return 'original_iut2_fallback', old_probability
    result = float(stdtr(df, -min(abs(d), abs(s))))
    require(math.isfinite(result) and 0 <= result <= 1, 'Invalid new Student tail')
    return 'certified_nonnegative_common_t', result


def calculate(data):
    """Compare saved references; admission/ancestry belong to the caller."""
    validate_reference_identity(data)
    value = data['within']['result']; state = value['status']; refs = value['references']
    actual = data['actual']
    require(type(actual) is dict and set(actual) == {'model', 'reference', 'rows'}
            and actual['model'] == 'H10' and actual['reference'] == 'C2'
            and type(actual['rows']) is list, 'Wrong original request')
    for row in actual['rows']:
        require(type(row) is dict and set(row) == {'participant', 'condition', 'need_A', 'need_B', 'y'}
                and type(row['y']) in (int, float) and math.isfinite(row['y']), 'Wrong native row shape')
    design_rows = [{k: row[k] for k in ('participant', 'condition', 'need_A', 'need_B')} for row in actual['rows']]
    certificate = design_certificate(design_rows, data['counts'])
    nulls = data['within']['truthEvaluation']['magnitudeTrueNull']
    require(nulls == dict.fromkeys(COMPARISONS, data['case']['caseIndex'] == 1)
            and all(type(v) is bool for v in nulls.values()), 'Changed fixed truth mask')
    states = dict.fromkeys(COMPARISONS, state); old_p = {}; new_p = {}; branches = {}
    df = value['df']
    if state == 'available':
        require(type(df) is int and df == 9*sum(data['counts'].values())-6 and df > 0
                and value['design']['rank'] == 6 and certificate['fullRank']
                and type(value['residualVariance']) in (int, float) and math.isfinite(value['residualVariance'])
                and value['residualVariance'] > 0, 'Common-scale/design identity unavailable')
        require(all(refs[k]['df'] == df and refs[k]['status'] == 'available' for k in REFS),
                'Different or unavailable component df')
    for key in COMPARISONS:
        old_p[key] = value['composition']['tests'][key]['pMagnitude']
        require(value['composition']['tests'][key]['status'] == state, 'Changed original comparison status')
        if state != 'available':
            require(old_p[key] is None, 'Unavailable old comparison with p')
            new_p[key] = None; branches[key] = state; continue
        d, s = (refs[key+':'+kind]['statistic'] for kind in ('D', 'S'))
        branches[key], new_p[key] = candidate_probability(d, s, df, certificate['signs'][key], old_p[key])
    old_adjusted, old_flags = holm(old_p, states)
    for key in COMPARISONS:
        test = value['composition']['tests'][key]
        require(old_flags[key] is test['holmReject'], 'Accepted original Holm differs')
        if old_adjusted[key] is not None:
            require(math.isclose(old_adjusted[key], test['holmP'], abs_tol=2e-9, rel_tol=2e-8),
                    'Accepted original adjusted p differs')
        else:
            require(test['holmP'] is None, 'Unavailable adjusted p must remain None')
    adjusted, flags = holm(new_p, states)
    analysis = dict(status=state, reason=value['reason'], truth=nulls, actualK=data['counts'],
                    signs=certificate['signs'], branches=branches, oldP=old_p, newP=new_p,
                    oldAdjusted=old_adjusted, newAdjusted=adjusted, oldHolm=old_flags, newHolm=flags,
                    oldEvents=events(old_flags, nulls), newEvents=events(flags, nulls))
    return dict(certificate=certificate, analysis=analysis)


def proportion(count, denominator):
    from scipy.stats import beta
    require(type(count) is int and type(denominator) is int and 0 <= count <= denominator,
            'Invalid descriptive counts')
    if denominator == 0:
        return dict(count=count, denominator=0, rate=None, CP95=None, MCSE=None)
    rate = count/denominator
    interval = [float(beta.ppf(.025, count, denominator-count+1)) if count else 0.,
                float(beta.ppf(.975, count+1, denominator-count)) if count < denominator else 1.]
    return dict(count=count, denominator=denominator, rate=rate, CP95=interval,
                MCSE=math.sqrt(rate*(1-rate)/denominator))


def paired_table(pairs):
    pairs = list(pairs)
    table = {key: 0 for key in ('00', '01', '10', '11')}
    for left, right in pairs:
        require(type(left) is bool and type(right) is bool, 'Paired classification required')
        table[str(int(left))+str(int(right))] += 1
    q = len(pairs)
    delta = (table['01']-table['10'])/q if q else None
    variance = ((table['01']+table['10'])/q-delta*delta)/(q-1) if q > 1 else None
    require(variance is None or variance >= 0, 'Invalid paired variance')
    return dict(Q=q, table=table, delta=delta, pairedMCSE=math.sqrt(variance) if variance is not None else None)


def validate_rows(rows):
    seen = set()
    for row in rows:
        ordinal = row['ordinal']; index = row['caseIndex']
        require(type(ordinal) is int and 0 <= ordinal < 12129 and ordinal not in seen,
                'Duplicate or foreign scientific ordinal')
        expected = 1 if ordinal < 10000 else 2
        require(type(index) is int and index == expected and type(row['repID']) is int
                and row['repID'] == ordinal-(0 if index == 1 else 10000)+1
                and row['caseID'] == 'h10wcv1_initial_amount_'+('s1' if index == 1 else 's5_l20')+'_recruit193_mcar08',
                'Changed fixed case/rep identity')
        value = row['analysis']; state = value['status']
        require(state in STATES and (value['reason'] is None or type(value['reason']) is str)
                and value['truth'] == dict.fromkeys(COMPARISONS, index == 1)
                and all(type(v) is bool for v in value['truth'].values())
                and set(row['endpointUnknown']) == set(REFS)
                and all(type(v) is bool for v in row['endpointUnknown'].values()), 'Changed scalar disposition')
        for side in ('old', 'new'):
            flags = value[side+'Holm']
            require(value[side+'Events'] == events(flags, value['truth']), 'Changed same-mask events')
            require(all(v is None for v in flags.values()) if state == 'technical_unresolved'
                    else all(type(v) is bool for v in flags.values()), 'Changed technical classification')
            if state == 'statistical_unavailable':
                require(not any(flags.values()), 'Statistical padding became a discovery')
        for key in COMPARISONS:
            sign = value['signs'][key]
            require(sign is None or type(sign) is int and sign in (-1, 0, 1), 'Invalid design sign')
            require(value['branches'][key] in ('same_or_zero_sign', 'original_iut2_fallback',
                    'certified_nonnegative_common_t') if state == 'available'
                    else value['branches'][key] == state, 'Changed branch disposition')
        seen.add(ordinal)
    return seen


def report(rows):
    """Reduce supplied paired scalars, not their source authentication."""
    rows = list(rows)
    validate_rows(rows)
    result = {}
    for index, total in ((1, 10000), (2, 2129)):
        values = [row['analysis'] for row in rows if row['caseIndex'] == index]
        present = len(values); complete = present == total
        statuses = Counter(v['status'] for v in values)
        item = dict(M=total, present=present, missing=total-present,
                    T=sum(v['status'] != 'technical_unresolved' for v in values),
                    A4=statuses['available'], statisticalUnavailable=statuses['statistical_unavailable'],
                    technicalUnknown=statuses['technical_unresolved'], reasons=dict(Counter(v['reason'] for v in values if v['reason'])),
                    endpointUnknown={key: sum(row['endpointUnknown'][key] for row in rows if row['caseIndex'] == index)
                                     for key in REFS}, comparisons={}, events={})
        for comparison in COMPARISONS:
            available = [v for v in values if v['status'] == 'available']
            pairs = [(v['oldHolm'][comparison], v['newHolm'][comparison]) for v in values
                     if v['oldHolm'][comparison] is not None and v['newHolm'][comparison] is not None]
            item['comparisons'][comparison] = dict(Aj=len(available), T=len(pairs), unknown=present-len(pairs),
                paired=paired_table(pairs),
                signCounts={label: sum(v['signs'][comparison] == sign for v in available)
                            for label, sign in (('negative', -1), ('zero', 0), ('positive', 1), ('uncertified', None))},
                branchCounts=dict(Counter(v['branches'][comparison] for v in values)),
                oldRejected=sum(v['oldHolm'][comparison] is True for v in values),
                newRejected=sum(v['newHolm'][comparison] is True for v in values),
                availableOld=proportion(sum(v['oldHolm'][comparison] is True for v in available), len(available)),
                availableNew=proportion(sum(v['newHolm'][comparison] is True for v in available), len(available)))
        for event in ('R', 'F', 'D'):
            if (event == 'D' and index == 1) or (event == 'F' and index == 2):
                item['events'][event] = None; continue
            known = [v for v in values if v['oldEvents'][event] is not None and v['newEvents'][event] is not None]
            paired = paired_table([(v['oldEvents'][event], v['newEvents'][event]) for v in known])
            old_count = sum(v['oldEvents'][event] is True for v in known)
            new_count = sum(v['newEvents'][event] is True for v in known)
            full = complete and len(known) == total
            available = [v for v in known if v['status'] == 'available']
            item['events'][event] = dict(**paired, unknown=present-len(known),
                old=proportion(old_count, len(known)), new=proportion(new_count, len(known)),
                availableOld=proportion(sum(v['oldEvents'][event] is True for v in available), len(available)),
                availableNew=proportion(sum(v['newEvents'][event] is True for v in available), len(available)),
                fullOld=proportion(old_count, total) if full else None,
                fullNew=proportion(new_count, total) if full else None,
                plannedOldBounds=[old_count/total, (old_count+total-len(known))/total],
                plannedNewBounds=[new_count/total, (new_count+total-len(known))/total])
        result[str(index)] = item
    return dict(schema='h10-common-t-saved-statistics-v1', cases=result,
                retrospectiveDevelopmentOnly=True, scientificAdoption=False,
                coverageRecomputed=False, linearIntervalsUnchanged=True,
                fullMeasurement=all(result[str(i)]['present'] == m for i, m in ((1, 10000), (2, 2129))))
