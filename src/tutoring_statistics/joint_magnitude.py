"""New H10 joint-normal reference on already authenticated saved contrasts.

This adapter performs no fit, CR2 reconstruction, df substitution, CI change,
or provenance admission. Its caller must authenticate the accepted eight-L
object and original guard. Estimated covariance gives an asymptotic candidate,
not finite-sample or human-study adoption.
"""
import math

from .magnitude_composition import COMPARISON_IDS, _holm_four
from .sign_congruence import gaussian_comparison

METHOD_ID = "h10_GaussianBoundarySup_CR2PlugIn_Holm4_candidate_v1"
REFERENCE_IDS = tuple(key + ':' + kind for key in COMPARISON_IDS for kind in ('D', 'S'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite saved scalar')
    return float(value)


def compose_saved_method(method, *, covariance_source='CR2'):
    """Compute the fixed four-item family; never convert technical U to p=1."""
    _require(covariance_source in ('CR2', 'model_based_diagnostic'), 'Unknown covariance source')
    _require(type(method) is dict and method.get('status') in
             ('complete', 'statistical_unavailable', 'technical_unresolved'), 'Unknown original status')
    refs = method['references']
    _require(type(refs) is dict and set(refs) == set(REFERENCE_IDS), 'Wrong eight named references')
    matrix = method['contrastCovariance']
    if matrix is not None:
        _require(type(matrix) is list and len(matrix) == 8
                 and all(type(row) is list and len(row) == 8 for row in matrix), 'Expected eight-L covariance')
        for i in range(8):
            for j in range(8):
                x, y = _number(matrix[i][j]), _number(matrix[j][i])
                _require(math.isclose(x, y, abs_tol=2e-9, rel_tol=2e-8), 'Asymmetric saved covariance')
    tests = {}
    for j, key in enumerate(COMPARISON_IDS):
        d, s = refs[key+':D'], refs[key+':S']
        _require(d['linearID'] == key+':D' and s['linearID'] == key+':S', 'Mislabeled contrast covariance order')
        states = (d['status'], s['status'])
        _require(all(v in ('available', 'statistical_unavailable', 'technical_unresolved') for v in states),
                 'Unknown linear status')
        status = ('technical_unresolved' if 'technical_unresolved' in states else
                  'statistical_unavailable' if 'statistical_unavailable' in states else 'available')
        reasons = [ref['reason'] for ref in (d, s) if ref['status'] != 'available']
        _require(all(type(r) is str and r for r in reasons), 'Missing typed failure reason')
        if status == 'available':
            _require(method['status'] == 'complete' and matrix is not None, 'Available reference without complete covariance')
            for index, ref in ((2*j, d), (2*j+1, s)):
                se = _number(ref['SE']); _require(se > 0 and _number(ref['df']) > 0, 'Invalid original reference')
                _require(math.isclose(matrix[index][index], se*se, abs_tol=2e-9, rel_tol=2e-8),
                         'Covariance diagonal differs from authenticated SE')
                _require(ref['reason'] is None, 'Available reference with failure reason')
            detail = gaussian_comparison(d['estimate'], s['estimate'], matrix[2*j][2*j],
                matrix[2*j+1][2*j+1], matrix[2*j][2*j+1], covariance_basis='estimated_asymptotic')
            p = detail['pGaussianReference']
        else:
            detail, p = None, None
            for ref in (d, s):
                if ref['status'] != 'available':
                    _require(all(ref.get(k) is None for k in ('estimate', 'SE', 'df')), 'Unavailable inference is not numeric zero')
        tests[key] = dict(status=status, reason='; '.join(reasons) if reasons else None,
                          pJoint=p, reference=detail, holmP=None, rejected=None)
    technical = method['status'] == 'technical_unresolved' or any(t['status']=='technical_unresolved' for t in tests.values())
    if method['status'] == 'statistical_unavailable':
        _require(all(t['status']=='statistical_unavailable' for t in tests.values()) and matrix is None,
                 'Inconsistent original guard-unavailable method')
    if not technical:
        padded = [tests[key]['pJoint'] if tests[key]['status']=='available' else 1.0 for key in COMPARISON_IDS]
        adjusted = _holm_four(padded)
        for key, p in zip(COMPARISON_IDS, adjusted):
            available = tests[key]['status']=='available'
            tests[key]['holmP'] = p if available else None
            tests[key]['rejected'] = available and p < .05
    return dict(methodID=METHOD_ID if covariance_source=='CR2' else
                'h10_GaussianBoundarySup_ModelBased_Holm4_diagnostic_v1', covarianceSource=covariance_source,
                scope='estimated_covariance_joint_Gaussian_candidate',
                scientificAdoption=False, finiteSampleValidityEstablished=False,
                originalStatus=method['status'], technicalCompleted=not technical,
                status='technical_unresolved' if technical else 'complete',
                familySize=4, alpha=.05, comparisonIDs=list(COMPARISON_IDS),
                availableCount=sum(t['status']=='available' for t in tests.values()),
                tests=tests, anyRejects=None if technical else any(t['rejected'] for t in tests.values()),
                originalConfidenceIntervalsChanged=False)
