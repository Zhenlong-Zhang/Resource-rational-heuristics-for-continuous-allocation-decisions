"""Accepted independent QR/tail checks, generalized only to supplied fixed slopes.

The separate caller enforces each study's null mask. Shared composition is
checked by the explicit independent IUT/Holm arithmetic below.
"""
from decimal import Decimal, localcontext
import math

ARMS = ("C1", "C2", "C3")
NAMES = tuple(arm+":"+need for arm in ARMS for need in ("A", "B"))
COMPARISONS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
STATES = ("available", "statistical_unavailable", "technical_unresolved")
METHOD = "h10_within_participant_gaussian_t_IUT2_Holm4_candidate_v1"
SEEDS = dict(retention=2032611040, design=2042611040, response=2052611040)
RECRUITS = 193

def require(value, message):
    if not value:
        raise ValueError(message)


def near(left, right, label):
    require(type(left) is type(right) or
            (type(left) in (int, float) and type(right) in (int, float)),
            'Different value type: ' + label)
    if type(left) in (float, int) and type(right) in (float, int):
        if type(left) is type(right) is int:
            require(left == right, 'Different integer: ' + label)
        else:
            require(math.isfinite(left) and math.isfinite(right) and
                    math.isclose(left, right, rel_tol=2e-8, abs_tol=2e-9),
                    'Different numerical result: ' + label)
    elif type(left) is dict:
        require(left.keys() == right.keys(), 'Different fields: ' + label)
        for key in left:
            near(left[key], right[key], f'{label}.{key}')
    elif type(left) in (list, tuple):
        require(len(left) == len(right), 'Different list length: ' + label)
        for index, (a, b) in enumerate(zip(left, right)):
            near(a, b, f'{label}[{index}]')
    else:
        require(left == right, 'Different value: ' + label)


def precision_readback(count):
    """Verify the two neighboring exponential bounds without reusing log/ceil."""
    require(type(count) is int and count > 1, 'Wrong fixed precision count')
    with localcontext() as ctx:
        ctx.prec = 60
        target = Decimal('.05')
        scale = 2 * Decimal('.02') ** 2
        selected = 2 * (-scale * count).exp()
        preceding = 2 * (-scale * (count - 1)).exp()
        require(target - selected > Decimal('1e-40') and preceding - target > Decimal('1e-40'),
                'Fixed B fails the independent adjacent-bound check')
        return dict(B=count, previousB=count-1, upperFailureBound=str(selected),
                    previousUpperFailureBound=str(preceding), epsilon=.02, alphaMC=.05)

def targets(slopes):
    references, nulls = {}, {}
    require(set(slopes) == set(ARMS) and
            all(set(slopes[arm]) == {'A', 'B'} for arm in ARMS),
            'Wrong midpoint coefficient roster')
    for comparison in COMPARISONS:
        arms, need = comparison.split(':')
        left, right = arms.split('-')
        a, b = slopes[left][need], slopes[right][need]
        require(all(type(x) is float and math.isfinite(x) for x in (a, b)),
                'Invalid exact signed midpoint coefficient')
        nulls[comparison] = abs(a) >= abs(b)
        for kind, sign in (('D', -1.), ('S', 1.)):
            vector = [0.] * 6
            vector[NAMES.index(left+':'+need)] = 1.
            vector[NAMES.index(right+':'+need)] = sign
            references[comparison+':'+kind] = dict(L=vector, truth=a+sign*b)
    return references, nulls


def replay_input(library, slopes, namespace, rep_id, actual, generation):
    """Reconstruct all three PCG64 streams and the retained response rows."""
    import numpy as np

    seeds = {role: base + rep_id for role, base in SEEDS.items()}
    rng = {role: np.random.Generator(np.random.PCG64(seed)) for role, seed in seeds.items()}
    uniforms = rng['retention'].random((len(ARMS), RECRUITS))
    indices = rng['design'].integers(0, 9, size=(len(ARMS), RECRUITS), dtype=np.int64)
    random_intercepts = rng['response'].standard_normal(len(ARMS)*RECRUITS)
    trial_noise = rng['response'].standard_normal((len(ARMS)*RECRUITS, 10))
    mask = uniforms < .8
    counts = {arm: int(sum(mask[arm_index])) for arm_index, arm in enumerate(ARMS)}
    expected_rows, retained = [], []
    for arm_index, arm in enumerate(ARMS):
        for recruit_index in range(RECRUITS):
            if not mask[arm_index, recruit_index]:
                continue
            person_index = arm_index*RECRUITS + recruit_index
            participant = f'{namespace}:r{rep_id}:{arm}:p{recruit_index+1:03d}'
            retained.append(participant)
            schedule = library[arm][int(indices[arm_index, recruit_index])]
            for trial_index, trial in enumerate(schedule):
                a, b = trial['need_A'], trial['need_B']
                mean = 60. + slopes[arm]['A']*(a-50.) + slopes[arm]['B']*(b-50.)
                y = mean + 8.*float(random_intercepts[person_index]) + 12.*float(trial_noise[person_index, trial_index])
                expected_rows.append(dict(participant=participant, condition=arm, need_A=a, need_B=b, y=y))
    expected_generation = dict(scheduleIndices=indices.tolist(), retentionUniforms=uniforms.tolist(),
        retentionMask={arm: mask[i].tolist() for i, arm in enumerate(ARMS)},
        interceptZ=random_intercepts.tolist(), residualZ=trial_noise.tolist(),
        actualK=counts, retainedParticipantIDs=retained, responseTransform='none',
        rowOrder='C1,C2,C3;person1..193;numeric_main1..10')
    require(generation == expected_generation
            and actual == dict(model='H10', reference='C2', rows=expected_rows)
            and len(expected_rows) == 10*sum(counts.values()),
            'Independent PCG64 schedule, eligibility or centered response differs')
    return counts


def fitted_reference(qr, rows, counts):
    """The accepted QR body needs at least one row. An empty sample is design-U."""
    if not rows:
        require(counts == dict.fromkeys(ARMS, 0), 'Inconsistent empty sample')
        return dict(status='statistical_unavailable', reason='within_design_missing_arm',
                    counts=counts)
    try:
        return qr.calculate(rows)
    except (ValueError, ArithmeticError) as exc:
        return dict(status='technical_unresolved', reason=str(exc))


def family_events(family, records, nulls):
    """Check the fixed IUT/Holm operations and distinguish errors from discoveries."""
    tests = family['tests']
    require(family['familySize'] == 4 and family['alpha'] == .05
            and family['comparisonIDs'] == list(COMPARISONS) and set(tests) == set(COMPARISONS),
            'Changed Holm4 family')
    if any(record['status'] == 'technical_unresolved' for record in records):
        require(family['anyMagnitudeRejects'] is None and
                all(test['holmReject'] is None and test['holmPaddingP'] is None for test in tests.values()),
                'Technical unknown was counted as a nonrejection')
        return dict(R=None, F=None, D=None)
    raw = {}
    for record in records:
        key, p = record['comparisonID'], record['p']
        if p is None:
            raw[key] = 1.
            require(tests[key]['pMagnitude'] is None, 'Unavailable raw p-value must be absent')
        else:
            routes = max(p['D_less'], p['S_greater']), max(p['D_greater'], p['S_less'])
            raw[key] = min(1., 2*min(routes))
            near(tests[key]['pMagnitude'], raw[key], 'Independent two-orientation magnitude p')
        near(tests[key]['holmPaddingP'], raw[key], 'Fixed four-family padding')
    largest = 0.
    rejected = []
    for index, key in enumerate(sorted(COMPARISONS, key=lambda k: (raw[k], COMPARISONS.index(k)))):
        largest = max(largest, min(1., (4-index)*raw[key]))
        is_available = tests[key]['status'] == 'available'
        if is_available:
            near(tests[key]['holmP'], largest, 'Independent Holm adjusted p')
        else:
            require(tests[key]['holmP'] is None, 'Unavailable adjusted p-value must be absent')
        reject = is_available and largest < .05
        require(tests[key]['holmReject'] is reject, 'Different strict Holm decision')
        if reject:
            rejected.append(key)
    require(family['anyMagnitudeRejects'] is bool(rejected), 'Different Holm-any decision')
    return dict(R=bool(rejected), F=any(nulls[key] for key in rejected),
                D=any(not nulls[key] for key in rejected))


def check_projection(value, evaluation, reference, slopes, counts, compose):
    expected, nulls = targets(slopes)
    state = value['status']
    require(state in STATES and value['methodID'] == METHOD
            and value['coefficientNames'] == list(NAMES)
            and value['assumptions']['responseDependentGuardConditioning'] is False
            and value['design']['actualK'] == counts
            and set(value['references']) == set(evaluation['references']) == set(expected)
            and evaluation['usedByEstimator'] is False and evaluation['magnitudeTrueNull'] == nulls,
            'Changed within method, guard or truth identity')
    if state != 'technical_unresolved':
        require(reference['status'] == state and reference['counts'] == counts,
                'Independent within applicability differs')
    if state == 'available':
        for field, independent in (('beta', reference['beta']),
                                   ('covariance', reference['covariance']),
                                   ('RSS', reference['rss']),
                                   ('residualVariance', reference['rss']/reference['df'])):
            near(value[field], independent, 'Independent QR ' + field)
        require(value['reason'] is None and value['df'] == reference['df'] == 9*sum(counts.values())-6
                and value['design']['rank'] == reference['rank'] == 6
                and value['design']['projectedDimension'] == reference['transformedRows'],
                'Changed projected rank or residual degrees of freedom')
    else:
        require(type(value['reason']) is str and value['reason'] and
                all(value[key] is None for key in ('beta', 'covariance', 'RSS', 'df', 'residualVariance')),
                'Unavailable estimator supplied a result')
        if state == 'statistical_unavailable':
            require(value['reason'] == reference['reason'], 'Different design-unavailable reason')

    checked = {}
    for key, target in expected.items():
        original = value['references'][key]
        event = evaluation['references'][key]
        require(original['L'] == target['L'] and original['linearID'] == key and
                original['status'] == state and original['reason'] == value['reason'] and
                event['truth'] == target['truth'] and event['referenceStatus'] == state,
                'Changed named linear reference, truth or status')
        if state == 'available':
            independent = reference['references'][key]
            require(independent['L'] == target['L'], 'Different independent linear vector')
            for field in ('estimate', 'SE', 'df', 'statistic', 'lessP', 'greaterP', 'CI'):
                near(original[field], independent[field], f'Independent QR {key}.{field}')
            near(original['twoSidedP'], min(1., 2*min(independent['lessP'], independent['greaterP'])),
                 'Independent two-sided p ' + key)
            low, high = independent['CI']
            truth = target['truth']
            ambiguous = any(math.isclose(truth, x, rel_tol=2e-8, abs_tol=2e-9) for x in (low, high))
            covered = None if ambiguous else low < truth < high
        else:
            require(all(original[field] is None for field in
                        ('estimate', 'SE', 'df', 'statistic', 'lessP', 'greaterP', 'twoSidedP', 'CI')),
                    'Unavailable linear reference supplied an estimate')
            independent, ambiguous, covered = {}, False, None
        require(event['endpointUnresolved'] is ambiguous and event['covered'] is covered,
                'Independent endpoint classification differs')
        checked[key] = dict(independent, **target, status=state, covered=covered,
                            endpointUnresolved=ambiguous)

    records = []
    for key in COMPARISONS:
        d, s = checked[key+':D'], checked[key+':S']
        p = None if state != 'available' else dict(D_less=d['lessP'], D_greater=d['greaterP'],
                                                   S_less=s['lessP'], S_greater=s['greaterP'])
        records.append(dict(comparisonID=key, status=state, p=p, reason=value['reason']))
    family = compose(records)
    events = family_events(family, records, nulls)
    near(value['composition'], family, 'Independent QR tails with frozen Holm4')
    return dict(status=state, reason=value['reason'], references=checked, family=family,
                magnitudeTrueNull=nulls, events=events, independentReferenceStatus=reference['status'])
