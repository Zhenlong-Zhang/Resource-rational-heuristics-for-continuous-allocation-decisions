"""Independent geometry, prime-factor logs and original n/n-1 inequalities.

No producer arithmetic imports and no finite-law or matrix reconstruction.
Certificates establish conservative design-conditional bounds, not confidence
intervals from observed outcomes or authorization to execute a policy plan.
"""
from fractions import Fraction as Q

TARGETS = ('gap', 'partial_low')
EPSILON = Q(1, 10)
ROOT_COUNT = 90


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ratio(value):
    require(type(value) is list and len(value) == 2 and all(type(x) is str for x in value),
            'Independent exact rational shape')
    result = Q(int(value[0]), int(value[1]))
    require(value == [str(result.numerator), str(result.denominator)],
            'Independent canonical rational')
    return result


def encoded_ratio(value):
    value = Q(value)
    return [str(value.numerator), str(value.denominator)]


def prime_log_interval(number, terms):
    """ln(80), ln(800) from primes 2 and 5, with positive rational remainders."""
    require(number in (80, 800) and type(terms) is int and terms > 0, 'Fixed reference logarithms')
    remaining = number
    lower, upper = Q(0), Q(0)
    for prime in (2, 5):
        exponent = 0
        while remaining % prime == 0:
            exponent += 1
            remaining //= prime
        z = Q(prime - 1, prime + 1)
        power, partial = z, Q(0)
        for index in range(terms):
            partial += 2 * power / (2 * index + 1)
            power *= z * z
        remainder = 2 * power / ((2 * terms + 1) * (1 - z * z))
        lower += exponent * partial
        upper += exponent * (partial + remainder)
    require(remaining == 1 and 0 < lower < upper, 'Reference logarithm factorization/enclosure')
    return lower, upper


def geometry(designs, summary, model):
    """Read certified inverses and root ranges; do not reconstruct/invert G."""
    require(len(designs) == 9, 'Nine saved schedules')
    support_bounds, inverse_diagonal = [], Q(0)
    for variant in range(9):
        value = designs[variant][model]
        require(value['status'] == 'available', 'Saved design must be full rank')
        rows = value['rootRanges']
        require([row['rootKey'] for row in rows] == [['C5', variant, main] for main in range(1, 11)],
                'Independent exact fixed-root membership')
        inverse_diagonal += ratio(value['inverse'][1][1])
        for row in rows:
            if row['weightMinimum'] is None:
                require(row['weightMaximum'] is None and row['singlePrice'] == [['0', '1'], ['0', '1']],
                        'All-tie root contributes exactly zero')
                support_bounds.append((Q(0), Q(0), Q(0)))
                continue
            left, right = ratio(row['weightMinimum']), ratio(row['weightMaximum'])
            require(left <= right, 'Ordered saved signed weights')
            lower = left if left < 0 else Q(0)
            upper = right if right > 0 else Q(0)
            require(row['singlePrice'] == [encoded_ratio(lower), encoded_ratio(upper)],
                    'Saved uncentered root range')
            support_bounds.append((lower, upper, max(abs(left), abs(right))))
    require(len(support_bounds) == ROOT_COUNT and inverse_diagonal > 0, 'Complete nondegenerate target')
    saved_rows = [row for row in summary['ranges'] if row['model'] == model]
    require(len(saved_rows) == 1, 'Unique saved target summary')
    saved = saved_rows[0]
    lower_mean = sum((row[0] for row in support_bounds), Q(0)) / ROOT_COUNT
    upper_mean = sum((row[1] for row in support_bounds), Q(0)) / ROOT_COUNT
    require(saved['singlePriceRange'] == [encoded_ratio(lower_mean), encoded_ratio(upper_mean)],
            'Independent saved complete-replicate range')
    require([x['replicates'] for x in saved['zeroVarianceFloors']] == [16, 32, 64, 128, 256]
            and all(x['status'] == 'unattainable' for x in saved['zeroVarianceFloors']),
            'Preserve failed original five-look rule')
    radii = [row[2] for row in support_bounds]
    second_ranges = sum((radius * radius for radius in radii), Q(0))
    variance_bound = Q(5, 2) * inverse_diagonal
    maximum_range = max(radii)
    require(upper_mean > lower_mean and second_ranges > 0 and maximum_range > 0,
            'Nondegenerate known design quantities')
    return dict(width=upper_mean-lower_mean, radii=radii, sumSquaredRanges=second_ranges,
                varianceSumUpper=variance_bound, maximumRange=maximum_range)


def check_integer_certificate(item, kind, design):
    """Check original tail inequalities directly at n and n-1, not ceil code."""
    count = item['replicates']
    require(type(count) is int and count > 0 and item['previousReplicates'] == count-1,
            'Independent integer/previous count')
    if kind == 'originalEBNecessaryOnly':
        argument, offset = 800, 1
        left_factor = 3 * EPSILON
        right_factor = 7 * design['width']
        meaning = 'necessary_zero_variance_floor_only_not_sufficient_empirical_Bernstein'
    elif kind == 'fixedLookHoeffdingSufficient':
        argument, offset = 80, 0
        left_factor = 2 * ROOT_COUNT**2 * EPSILON**2
        right_factor = design['sumSquaredRanges']
        meaning = 'sufficient_root_independent_fixed_look_Hoeffding'
    else:
        require(kind == 'fixedLookBernsteinSufficient', 'Independent recognized bound')
        argument, offset = 80, 0
        left_factor = ROOT_COUNT**2 * EPSILON**2
        right_factor = 2 * design['varianceSumUpper'] + Q(2, 3) * ROOT_COUNT * EPSILON * design['maximumRange']
        meaning = 'sufficient_root_independent_fixed_look_Bernstein_variance_envelope'
    require(count > offset and item['integerOffset'] == offset and item['logArgument'] == argument,
            'Preserve old versus candidate look/error allocation')
    require(ratio(item['coefficient']) == right_factor/left_factor,
            'Independent coefficient/constants')
    published_lower, published_upper = [ratio(x) for x in item['logBounds']]
    require(0 < published_lower < published_upper, 'Published logarithm order')
    for terms in (256, 512, 1024, 2048, 4096, 8192):
        log_lower, log_upper = prime_log_interval(argument, terms)
        enclosure = published_lower <= log_lower and log_upper <= published_upper
        passes = left_factor * (count-offset) >= right_factor * log_upper
        previous_fails = left_factor * (count-1-offset) < right_factor * log_lower
        if enclosure and passes and previous_fails:
            if kind == 'fixedLookBernsteinSufficient':
                require(ROOT_COUNT * count * EPSILON > design['maximumRange'] * log_upper / 3,
                        'Positive Bernstein square-root rearrangement')
            require(item['previousFailsLower'] is True and item['currentPassesUpper'] is True,
                    'Reported certificate flags')
            return dict(bound=kind, replicates=count, previousReplicates=count-1,
                        independentPrimeSeriesTerms=terms,
                        independentLogBounds=[encoded_ratio(log_lower), encoded_ratio(log_upper)],
                        currentPasses=True, previousFails=True, publishedEnclosureCertified=True, meaning=meaning)
    raise ArithmeticError('Independent logarithm/integer certificate unresolved; no acceptance')


def verify_plan(plan, designs, summary):
    require(plan['precision'] == ['1', '10'] and plan['alpha'] == ['1', '20']
            and plan['activeJ'] == 2 and plan['activeTargets'] == list(TARGETS)
            and plan['inferenceLooks'] == 1, 'Candidate single-look family is explicit')
    require(plan['requiredIndependence'] == 'root_times_rep_under_declared_PRNG_model'
            and plan['estimator'] == 'new_fixed_control_variate_same_population_target_not_old_T_samplewise',
            'Assumption and new-estimator boundary')
    require([row['model'] for row in plan['rows']] == list(TARGETS), 'Exact two target rows')
    methods = ('fixedLookHoeffdingSufficient', 'fixedLookBernsteinSufficient')
    counts = {name: [] for name in methods}
    old_necessary = []
    certificates = []
    for model, row in zip(TARGETS, plan['rows']):
        known = geometry(designs, summary, model)
        require(row['oldWidth'] == encoded_ratio(known['width'])
                and row['centeredRootRadii'] == [encoded_ratio(x) for x in known['radii']]
                and row['centeredSumSquaredRanges'] == encoded_ratio(known['sumSquaredRanges'])
                and row['centeredMaxRange'] == encoded_ratio(known['maximumRange'])
                and row['centeredVarianceSumUpper'] == encoded_ratio(known['varianceSumUpper']),
                'Independent known-design and control-range transcription')
        for kind in ('originalEBNecessaryOnly', *methods):
            certificate = check_integer_certificate(row[kind], kind, known)
            certificate['model'] = model
            certificates.append(certificate)
            if kind in methods:
                counts[kind].append(certificate['replicates'])
            else:
                old_necessary.append(certificate['replicates'])
    family_counts = {name: max(values) for name, values in counts.items()}
    require(plan['originalEBNecessaryFullFamilyReplicates'] == max(old_necessary),
            'Original full-family floor count is necessary only')
    require(plan['candidateFullFamilyReplicates'] == family_counts, 'Complete simultaneous two-target count')
    best = min(family_counts.values())
    chosen = ('fixedLookBernsteinSufficient' if family_counts['fixedLookBernsteinSufficient'] == best
              else 'fixedLookHoeffdingSufficient')
    require(plan['recommendedMethod'] == chosen and plan['fixedReplicates'] == best
            and plan['fixedLowPriceEndpoints'] == ROOT_COUNT * best,
            'Only design-based fixed-family recommendation')
    require(plan['selectionBasis'] == 'accepted_design_only_min_sufficient_R_tie_Bernstein_before_any_outcome'
            and plan['selectionCreatesNoExtraOutcomeMultiplicity'] is True
            and plan['withinPrior256ReplicateLimit'] is (best <= 256)
            and plan['beyond256RequiresNewExplicitDomainRegistration'] is (best > 256)
            and plan['policyReleaseAuthorized'] is False, 'Prospective selection/domain/authorization')
    return dict(passed=True, boundaryCertificates=certificates,
                candidateFullFamilyReplicates=family_counts, recommendedMethod=chosen,
                fixedReplicates=best, requiredIndependenceIsAssumption=True,
                originalRuleUnchanged=True, futurePlanNotAdopted=True, policyDrawsReleased=False)
