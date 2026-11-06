"""Isolated within-participant H10 estimator; no I/O, RNG, REML or RDS.

The six slopes have the original meaning only under the common within/between
slope mean model. Exact t references additionally require independent Gaussian
equal-variance trial errors and no response-dependent selection. Historical
REML guard values and simulation truth are deliberately not inputs here.
"""
from __future__ import annotations

import math

ARMS = ('C1', 'C2', 'C3')
NAMES = tuple(arm + ':' + need for arm in ARMS for need in ('A', 'B'))
COMPARISONS = ('C1-C2:A', 'C1-C2:B', 'C1-C3:A', 'C1-C3:B')
LINEAR_IDS = tuple(comparison + ':' + kind for comparison in COMPARISONS for kind in ('D', 'S'))
METHOD_ID = 'h10_within_participant_gaussian_t_IUT2_Holm4_candidate_v1'
ASSUMPTIONS = dict(commonWithinBetweenSlopes=True, independentGaussianTrialErrors=True,
    equalUnknownTrialVariance=True, exogenousDesignAndWholePersonRetention=True,
    responseDependentGuardConditioning=False, randomInterceptDistributionRequired=False,
    humanApplicabilityEstablished=False, scientificAdoption=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def linear_vectors():
    vectors = {}
    for comparison in COMPARISONS:
        pair, need = comparison.split(':')
        left, right = pair.split('-')
        for kind, sign in (('D', -1.), ('S', 1.)):
            vector = [0.] * 6
            vector[NAMES.index(left + ':' + need)] = 1.
            vector[NAMES.index(right + ':' + need)] = sign
            vectors[comparison + ':' + kind] = vector
    return vectors


def validate_rows(actual):
    """Validate the saved native five-field rows, without altering any number."""
    require(type(actual) is dict and set(actual) == {'model', 'reference', 'rows'}
        and actual['model'] == 'H10' and actual['reference'] == 'C2', 'Wrong original request schema')
    require(type(actual['rows']) is list, 'Rows must be a list')
    groups = {}
    for row in actual['rows']:
        require(type(row) is dict and set(row) == {'participant', 'condition', 'need_A', 'need_B', 'y'},
            'Wrong saved row fields')
        require(type(row['participant']) is str and bool(row['participant'])
            and row['condition'] in ARMS, 'Invalid participant or condition')
        require(all(type(row[key]) in (int, float) and math.isfinite(row[key])
            for key in ('need_A', 'need_B', 'y')), 'Finite numeric row required; bool is not a response')
        group = groups.setdefault(row['participant'], dict(arm=row['condition'], rows=[]))
        require(group['arm'] == row['condition'], 'Participant crosses conditions')
        group['rows'].append(row)
    require(all(len(value['rows']) == 10 for value in groups.values()), 'Exactly ten rows per participant required')
    return groups


def unavailable(status, reason, design, compose_family):
    require(status in ('statistical_unavailable', 'technical_unresolved'), 'Unknown unavailable state')
    refs = {key: dict(linearID=key, L=vector, status=status, reason=reason,
        estimate=None, SE=None, df=None, statistic=None, lessP=None, greaterP=None,
        twoSidedP=None, CI=None) for key, vector in linear_vectors().items()}
    records = [dict(comparisonID=key, status=status, p=None, reason=reason) for key in COMPARISONS]
    return dict(schema='h10-within-projection-method-v1', methodID=METHOD_ID,
        status=status, reason=reason, coefficientNames=list(NAMES), beta=None, covariance=None,
        RSS=None, residualVariance=None, df=None, design=design, references=refs,
        composition=compose_family(records), assumptions=dict(ASSUMPTIONS))


def estimate_within(actual, compose_family):
    """Centered/scaled SVD estimator, independently checked by Helmert + QR.

    All six named slopes must be identified. This is a design requirement, not
    an observed REML or all-four-discoveries gate. No truth enters this function.
    """
    import numpy as np
    from scipy.special import stdtr, stdtrit

    groups = validate_rows(actual)
    counts = {arm: sum(value['arm'] == arm for value in groups.values()) for arm in ARMS}
    n, g = len(actual['rows']), len(groups)
    design = dict(rowCount=n, participantCount=g, actualK=counts, rowsPerParticipant=10,
        projectedDimension=n-g, requiredRank=6, rank=None, scaledSingularValues=[], rankTolerance=None)
    if not all(counts.values()):
        return unavailable('statistical_unavailable', 'within_design_missing_arm', design, compose_family)
    x = np.zeros((n, 6), dtype=float)
    y = np.empty(n, dtype=float)
    offset = 0
    for value in groups.values():
        values = np.asarray([[row['need_A'], row['need_B'], row['y']] for row in value['rows']], dtype=float)
        centered = values - values.mean(axis=0)
        column = 2 * ARMS.index(value['arm'])
        x[offset:offset+10, column:column+2] = centered[:, :2]
        y[offset:offset+10] = centered[:, 2]
        offset += 10
    scale = np.linalg.norm(x, axis=0)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(scale)):
        return unavailable('technical_unresolved', 'nonfinite_projection_arithmetic', design, compose_family)
    if np.any(scale == 0):
        return unavailable('statistical_unavailable', 'within_design_constant_slope_column', design, compose_family)
    try:
        u, singular, vt = np.linalg.svd(x / scale, full_matrices=False)
    except np.linalg.LinAlgError:
        return unavailable('technical_unresolved', 'within_svd_failed', design, compose_family)
    # Use the projected dimension, not the ten redundant centered rows, for
    # the declared resolution. Both implementations use this same rank policy.
    tolerance = max(n-g, 6) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    design.update(rank=rank, scaledSingularValues=singular.tolist(), rankTolerance=float(tolerance))
    if np.any((singular > .5*tolerance) & (singular < 2*tolerance)):
        return unavailable('technical_unresolved', 'within_rank_boundary_unresolved', design, compose_family)
    if rank != 6:
        return unavailable('statistical_unavailable', 'within_design_rank_deficient', design, compose_family)
    df = n-g-rank
    if df <= 0:
        return unavailable('statistical_unavailable', 'no_residual_degrees_of_freedom', design, compose_family)
    beta = (vt.T @ ((u.T @ y) / singular)) / scale
    residual = y-x @ beta
    rss = float(residual @ residual)
    if not math.isfinite(rss) or rss <= 0:
        return unavailable('technical_unresolved', 'nonpositive_or_nonfinite_residual_variance', design, compose_family)
    variance = rss / df
    information_inverse = ((vt.T / singular**2) @ vt) / np.outer(scale, scale)
    covariance = variance * information_inverse
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(covariance)):
        return unavailable('technical_unresolved', 'nonfinite_slope_or_covariance', design, compose_family)
    critical = float(stdtrit(df, .975))
    if not math.isfinite(critical) or critical <= 0:
        return unavailable('technical_unresolved', 'invalid_t_critical_value', design, compose_family)
    refs = {}
    for key, vector in linear_vectors().items():
        l = np.asarray(vector)
        estimate = float(l @ beta)
        v = float(l @ covariance @ l)
        if not math.isfinite(v) or v <= 0:
            return unavailable('technical_unresolved', 'invalid_linear_variance', design, compose_family)
        se = math.sqrt(v)
        statistic = estimate / se
        less, greater = float(stdtr(df, statistic)), float(stdtr(df, -statistic))
        interval = [estimate-critical*se, estimate+critical*se]
        if not all(math.isfinite(value) for value in [estimate, se, statistic, *interval, less, greater]) \
                or not 0 <= less <= 1 or not 0 <= greater <= 1 or not interval[0] < interval[1]:
            return unavailable('technical_unresolved', 'invalid_linear_reference', design, compose_family)
        refs[key] = dict(linearID=key, L=vector, status='available', reason=None, estimate=estimate,
            SE=se, df=df, statistic=statistic, lessP=less, greaterP=greater,
            twoSidedP=min(1., 2*min(less, greater)), CI=interval)
    records = [dict(comparisonID=key, status='available', reason=None,
        p=dict(D_less=refs[key+':D']['lessP'], D_greater=refs[key+':D']['greaterP'],
               S_less=refs[key+':S']['lessP'], S_greater=refs[key+':S']['greaterP'])) for key in COMPARISONS]
    return dict(schema='h10-within-projection-method-v1', methodID=METHOD_ID, status='available', reason=None,
        coefficientNames=list(NAMES), beta=beta.tolist(), covariance=covariance.tolist(),
        RSS=rss, residualVariance=variance, df=df, design=design, references=refs,
        composition=compose_family(records), assumptions=dict(ASSUMPTIONS))


def evaluate_truth(within, slopes):
    """Simulation evaluation only; never feeds estimates, widths, or p values."""
    require(set(slopes) == set(ARMS) and all(set(slopes[arm]) == {'A', 'B'} for arm in ARMS), 'Wrong truth shape')
    truth_vector = [slopes[arm][need] for arm in ARMS for need in ('A', 'B')]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in truth_vector), 'Nonfinite truth')
    refs = {}
    for key, vector in linear_vectors().items():
        truth = sum(a*b for a, b in zip(vector, truth_vector))
        ref = within['references'][key]
        ambiguous = ref['status'] == 'available' and any(math.isclose(truth, end, abs_tol=2e-9, rel_tol=2e-8) for end in ref['CI'])
        covered = (ref['CI'][0] < truth < ref['CI'][1]) if ref['status'] == 'available' and not ambiguous else None
        refs[key] = dict(truth=truth, covered=covered, endpointUnresolved=ambiguous, referenceStatus=ref['status'])
    nulls = {}
    for key in COMPARISONS:
        pair, need = key.split(':'); left, right = pair.split('-')
        nulls[key] = abs(slopes[left][need]) >= abs(slopes[right][need])
    return dict(schema='h10-within-projection-truth-evaluation-v1', usedByEstimator=False,
        references=refs, magnitudeTrueNull=nulls)
