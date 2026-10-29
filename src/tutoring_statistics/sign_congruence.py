"""Gaussian reference for the H10 absolute-magnitude composite null.

D = beta_C1 - beta_reference, S = beta_C1 + beta_reference; H0: D*S >= 0.
This is a development reference, not an adopted human-data procedure. Known
Gaussian covariance gives the oracle law. Plugging in a CR2 covariance requires
uniform joint CLT/consistency conditions and separate finite-sample assessment;
the marginal Satterthwaite degrees of freedom do not define a joint t law.

Reference: Miller, Molinari & Stoye, arXiv:2405.11759v4, Theorem 2.1/Prop. 2.1.
The negative-correlation supremum is bounded over the entire nonnegative axis,
not estimated by a local optimizer or a grid without a tail bound. QUADPACK's
error estimates are numerical diagnostics, not formal interval certificates.
"""

import heapq
import math
import warnings

from scipy.integrate import IntegrationWarning, quad
from scipy.special import ndtr, ndtri


METHOD_ID = "h10_DS_Gaussian_full_null_supremum_development_v1"
DEFAULT_TOLERANCE = 1e-7


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number, excluding bool")
    return float(value)


def _corner(c, rho, offset, error_target):
    """P(X>c, Y<offset), evaluated without cancellation of normal CDFs."""
    sd = math.sqrt((1.0 - rho) * (1.0 + rho))
    q = float(ndtr(-c))
    # z = P(X>x) changes the normal-tail integral into a finite interval.
    def integrand(z):
        if z == 0.0:
            return 1.0 if rho < 0.0 else 0.0
        x = -float(ndtri(z))
        return float(ndtr((offset - rho * x) / sd))

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        value, error = quad(integrand, 0.0, q, epsabs=error_target,
                            epsrel=1e-11, limit=300)
    if not math.isfinite(value + error) or error > error_target * 4:
        raise ArithmeticError("Gaussian corner integration failed numerical admission")
    # Error enlargement is a declared numerical safeguard, not a proof that
    # the library's estimated quadrature error bounds its true floating error.
    margin = max(8.0 * error, error_target)
    return max(0.0, value - margin), min(q, value + margin)


def gaussian_null_supremum(c, rho, *, tolerance=DEFAULT_TOLERANCE,
                           max_intervals=32768):
    """Return an enclosing numerical bracket for sup-null P(R_c).

R_c consists of both opposite-sign corners beyond c. The full Gaussian null
reduces to means (0,t), t>=0. For rho>=0 the supremum is sf(c); for rho=-1 it
is 2*sf(c). Otherwise A(t)=P(X>c,Y<-c-t) decreases, B(t)=P(X>c,Y<t-c)
increases, so A(l)+B(u) bounds every t in [l,u]. For t>=T the bound is
sf(c)+sf(c+T). The returned upper, including when resolution is capped, is
used for decisions. Singular endpoint handling is oracle-only, not admission
of a singular estimated covariance into the feasible comparison function.
"""
    c, rho = _finite(c, "c"), _finite(rho, "rho")
    tolerance = _finite(tolerance, "tolerance")
    if c < 0 or not -1 <= rho <= 1 or not 1e-10 <= tolerance <= 1e-3:
        raise ValueError("Invalid c, rho or numerical tolerance")
    if type(max_intervals) is not int or max_intervals < 1:
        raise ValueError("max_intervals must be a positive integer")
    q = float(ndtr(-c))
    if rho >= 0 or rho == -1 or q == 0:
        p = q if rho >= 0 else min(1.0, 2.0 * q)
        return dict(lower=p, upper=p, gap=0.0, intervals=0,
                    boundaryEvaluations=0, tailStart=None,
                    numericalStatus="analytic", quadratureCertificate=False)
    target = tolerance / 128.0
    tail_start = max(1.0, -float(ndtri(tolerance / 8.0)) - c)
    tail_upper = min(2.0 * q, q + float(ndtr(-c - tail_start)))
    cache = {}

    def evaluate(t):
        if t not in cache:
            a = _corner(c, rho, -c - t, target)
            b = _corner(c, rho, t - c, target)
            cache[t] = (a, b)
        return cache[t]

    def interval(l, u):
        a, b = evaluate(l), evaluate(u)
        upper = min(2.0 * q, a[0][1] + b[1][1])
        return (-upper, l, u)

    left, right = evaluate(0.0), evaluate(tail_start)
    lower = max(q, left[0][0] + left[1][0], right[0][0] + right[1][0])
    heap = [interval(0.0, tail_start)]
    while True:
        upper = max(lower, tail_upper, -heap[0][0])
        if upper - lower <= tolerance or len(heap) >= max_intervals:
            break
        _, l, u = heapq.heappop(heap)
        mid = (l + u) / 2.0
        if mid == l or mid == u:
            raise ArithmeticError("Boundary refinement exhausted floating-point resolution")
        middle = evaluate(mid)
        lower = max(lower, middle[0][0] + middle[1][0])
        heapq.heappush(heap, interval(l, mid))
        heapq.heappush(heap, interval(mid, u))
    return dict(lower=lower, upper=upper, gap=upper - lower,
                intervals=len(heap), boundaryEvaluations=len(cache),
                tailStart=tail_start,
                numericalStatus="resolved" if upper - lower <= tolerance else "resolution_capped",
                quadratureCertificate=False)


def gaussian_comparison(d, s, variance_d, variance_s, covariance_ds,
                        *, covariance_basis, tolerance=DEFAULT_TOLERANCE):
    """One comparison only; the caller preserves guards and the four-item Holm.

covariance_basis must explicitly distinguish a known Gaussian oracle from an
estimated asymptotic reference. No supplied covariance can establish the latter
conditions by itself. Zero/same-sign estimates produce p=1. Invalid covariance
is rejected, never silently clipped into a positive-semidefinite matrix.
"""
    d, s = _finite(d, "D"), _finite(s, "S")
    vd, vs = _finite(variance_d, "variance D"), _finite(variance_s, "variance S")
    cov = _finite(covariance_ds, "covariance D,S")
    if covariance_basis not in ("known_gaussian", "estimated_asymptotic"):
        raise ValueError("Explicit covariance_basis required")
    if vd <= 0 or vs <= 0:
        raise ValueError("Both marginal variances must be positive")
    rho = cov / math.sqrt(vd) / math.sqrt(vs)
    if not math.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("Comparison requires a nonsingular covariance matrix")
    opposite = (d < 0 < s) or (s < 0 < d)
    c = min(abs(d) / math.sqrt(vd), abs(s) / math.sqrt(vs))
    bracket = gaussian_null_supremum(c, rho, tolerance=tolerance) if opposite else None
    return dict(methodID=METHOD_ID, covarianceBasis=covariance_basis,
                scientificAdoption=False, finiteSampleValidityEstablished=False,
                rho=rho, standardizedDistance=c, oppositeSigns=opposite,
                pGaussianReference=bracket["upper"] if opposite else 1.0,
                referenceBracket=bracket)
