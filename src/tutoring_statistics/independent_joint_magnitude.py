"""Independent named covariance mapping and joint-family composition.

Nonnegative correlations use an independently evaluated normal tail. The
negative-correlation supremum remains shared and is explicitly not independent.
"""
import math
from .saved_fit_magnitude import NAMES, IDS, LINEAR_IDS, require, near as equal
from .joint_magnitude import METHOD_ID as METHOD
from .sign_congruence import METHOD_ID as CORE_METHOD

BASELINE_METHOD = "h10_GaussianBoundarySup_ModelBased_Holm4_diagnostic_v1"


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid scalar")
    return float(value)


def independent_method(method, covariance_source, negative_core, counts):
    references = method['references']
    require(set(references) == set(LINEAR_IDS), 'Wrong eight references')
    require(method['status'] in ('complete', 'statistical_unavailable', 'technical_unresolved'), 'Unknown source method status')
    matrix = method['contrastCovariance']
    if matrix is not None:
        require(type(matrix) is list and len(matrix) == 8 and all(type(row) is list and len(row) == 8 for row in matrix),
                'Wrong contrast covariance shape')
        for i in range(8):
            for j in range(8):
                equal(number(matrix[i][j]), number(matrix[j][i]), 'symmetric saved covariance')
    tests = {}
    for j, key in enumerate(IDS):
        pair = [references[key+':D'], references[key+':S']]
        require([ref['linearID'] for ref in pair] == [key+':D', key+':S'], 'Changed D/S labels')
        states = [ref['status'] for ref in pair]
        require(all(state in ('available', 'statistical_unavailable', 'technical_unresolved') for state in states), 'Unknown linear state')
        status = ('technical_unresolved' if 'technical_unresolved' in states else
                  'statistical_unavailable' if 'statistical_unavailable' in states else 'available')
        reasons = [ref['reason'] for ref in pair if ref['status'] != 'available']
        require(all(type(reason) is str and reason for reason in reasons), 'Missing source failure reason')
        detail = None
        probability = None
        if status == 'available':
            require(method['status'] == 'complete' and matrix is not None, 'Missing complete covariance')
            coefficient_covariance = method['covariance']
            require(type(coefficient_covariance) is list and len(coefficient_covariance) == 9
                    and all(type(row) is list and len(row) == 9 for row in coefficient_covariance), 'Wrong named coefficient covariance')
            for i in range(9):
                for k in range(9):
                    equal(number(coefficient_covariance[i][k]), number(coefficient_covariance[k][i]), 'coefficient covariance symmetry')
            # Reconstruct named condition slopes directly, not via the old
            # eight-L adapter or the new wrapper's eight-dimensional matrix.
            need_index = 3 if key[-1] == 'A' else 4
            left = [0.0]*9; right = [0.0]*9
            left[need_index] = right[need_index] = 1.0
            left[5 if key[-1] == 'A' else 7] = 1.0
            if key[3:5] == 'C3':
                right[6 if key[-1] == 'A' else 8] = 1.0
            vectors = [[a-b for a, b in zip(left, right)], [a+b for a, b in zip(left, right)]]
            for ref, vector in zip(pair, vectors):
                require(ref['coefficientNames'] == list(NAMES) and len(ref['L']) == 9
                        and all(number(value) == expected for value, expected in zip(ref['L'], vector)),
                        'Changed named D/S projection')
            projected = [[math.fsum(vectors[a][i]*number(coefficient_covariance[i][k])*vectors[b][k]
                                   for i in range(9) for k in range(9)) for b in range(2)] for a in range(2)]
            for a in range(2):
                for b in range(2):
                    equal(number(matrix[2*j+a][2*j+b]), projected[a][b], 'named covariance projection')
            counts['covarianceMappingChecks'] += 1
            for index, ref in zip((2*j, 2*j+1), pair):
                se = number(ref['SE'])
                require(se > 0 and number(ref['df']) > 0 and ref['reason'] is None, 'Invalid accepted linear inference')
                equal(number(matrix[index][index]), se*se, 'saved SE/covariance diagonal')
            d, s = (number(ref['estimate']) for ref in pair)
            vd, vs, cov = number(matrix[2*j][2*j]), number(matrix[2*j+1][2*j+1]), number(matrix[2*j][2*j+1])
            require(vd > 0 and vs > 0, 'Nonpositive pair variance')
            rho = cov/math.sqrt(vd)/math.sqrt(vs)
            require(math.isfinite(rho) and -1.0 < rho < 1.0, 'Non-SPD D/S pair')
            # Compare signs without multiplying two possibly large estimates.
            opposite = (d < 0.0 < s) or (s < 0.0 < d)
            distance = min(abs(d)/math.sqrt(vd), abs(s)/math.sqrt(vs))
            if rho < 0:
                counts['negativeCorrelationComparisons'].append(dict(comparisonID=key, rho=rho))
                detail = negative_core(d, s, vd, vs, cov, covariance_basis='estimated_asymptotic')
                probability = detail['pGaussianReference']
            else:
                counts['analyticNonnegativeComparisons'] += 1
                probability = .5*math.erfc(distance/math.sqrt(2.0)) if opposite else 1.0
                bracket = dict(lower=probability, upper=probability, gap=0.0, intervals=0,
                    boundaryEvaluations=0, tailStart=None, numericalStatus='analytic', quadratureCertificate=False) if opposite else None
                detail = dict(methodID=CORE_METHOD, covarianceBasis='estimated_asymptotic',
                    scientificAdoption=False, finiteSampleValidityEstablished=False, rho=rho,
                    standardizedDistance=distance, oppositeSigns=opposite,
                    pGaussianReference=probability, referenceBracket=bracket)
        else:
            for ref in pair:
                if ref['status'] != 'available':
                    require(all(ref[field] is None for field in ('estimate', 'SE', 'df')), 'Unavailable inference is not zero')
        tests[key] = dict(status=status, reason='; '.join(reasons) if reasons else None,
                          pJoint=probability, reference=detail, holmP=None, rejected=None)
    unknown = method['status'] == 'technical_unresolved' or any(item['status'] == 'technical_unresolved' for item in tests.values())
    if method['status'] == 'statistical_unavailable':
        require(matrix is None and all(item['status'] == 'statistical_unavailable' for item in tests.values()),
                'Changed original statistical-unavailable state')
    if not unknown:
        order = sorted(IDS, key=lambda key: tests[key]['pJoint'] if tests[key]['status'] == 'available' else 1.0)
        adjusted = 0.0
        for rank, key in enumerate(order):
            available = tests[key]['status'] == 'available'
            p = tests[key]['pJoint'] if available else 1.0
            adjusted = max(adjusted, min(1.0, (4-rank)*p))
            tests[key]['holmP'] = adjusted if available else None
            tests[key]['rejected'] = bool(available and adjusted < .05)
    return dict(methodID=METHOD if covariance_source == 'CR2' else BASELINE_METHOD,
        covarianceSource=covariance_source, scope='estimated_covariance_joint_Gaussian_candidate',
        scientificAdoption=False, finiteSampleValidityEstablished=False,
        originalStatus=method['status'], technicalCompleted=not unknown,
        status='technical_unresolved' if unknown else 'complete', familySize=4, alpha=.05,
        comparisonIDs=list(IDS), availableCount=sum(item['status'] == 'available' for item in tests.values()),
        tests=tests, anyRejects=None if unknown else any(item['rejected'] for item in tests.values()),
        originalConfidenceIntervalsChanged=False)
