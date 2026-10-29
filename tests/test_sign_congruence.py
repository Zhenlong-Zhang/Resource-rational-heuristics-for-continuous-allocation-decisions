"""Deterministic oracle checks; no data simulation, fitting, or human validation."""
import math
import unittest

from scipy.integrate import quad
from scipy.special import ndtr

from src.tutoring_statistics.sign_congruence import (
    gaussian_comparison, gaussian_null_supremum,
)


def independent_rejection(c, rho, u, v):
    """Integrate on X directly, independently of producer's probability scale."""
    sd = math.sqrt(1 - rho * rho)
    normalizer = math.sqrt(2 * math.pi)
    density = lambda x: math.exp(-x*x/2) / normalizer
    positive = quad(lambda x: density(x) * ndtr((-c-v-rho*x)/sd),
                    c-u, math.inf, epsabs=2e-11, limit=300)[0]
    negative = quad(lambda x: density(x) * ndtr((v-c+rho*x)/sd),
                    -math.inf, -c-u, epsabs=2e-11, limit=300)[0]
    return positive + negative


class GaussianReferenceTests(unittest.TestCase):
    def test_analytic_positive_and_singular_limits(self):
        for c in (0.0, 0.025, 1.6448536269514722, 2.241402727604947, 3.5):
            q = 0.5 * math.erfc(c / math.sqrt(2))
            for rho in (0.0, 0.5, 0.95, 1.0):
                out = gaussian_null_supremum(c, rho)
                self.assertAlmostEqual(out['upper'], q, places=14)
                self.assertEqual(out['gap'], 0)
            self.assertAlmostEqual(gaussian_null_supremum(c, -1)['upper'], 2*q, places=14)

    def test_external_published_calibration_values(self):
        # Miller et al. arXiv2405.11759v4 Table2, alpha=.05.
        # Rounded published c values: allow their rounding plus our p tolerance.
        for rho, critical in ((-.95, 1.81773816), (-.90, 1.74893328),
                              (-.80, 1.64878158), (-.75, 1.64488267)):
            result = gaussian_null_supremum(critical, rho)
            self.assertEqual(result['numericalStatus'], 'resolved')
            self.assertLessEqual(result['gap'], 1e-7)
            self.assertLessEqual(result['lower'], .05 + 2e-7)
            self.assertGreaterEqual(result['upper'], .05 - 2e-7)

    def test_full_null_witnesses_and_axis_reduction(self):
        for rho in (-.90, -.5, .9):
            c = 2.0
            bound = gaussian_null_supremum(c, rho)
            for u, v in ((0, 0), (0, .5), (0, 2), (0, 6), (.5, 1), (2, 2)):
                direct = independent_rejection(c, rho, u, v)
                self.assertLessEqual(direct, bound['upper'] + 2e-9)
                axis = independent_rejection(c, rho, 0, abs(v-u))
                self.assertLessEqual(direct, axis + 2e-9)

    def test_finite_nonzero_axis_cannot_be_omitted(self):
        c, rho = 1.6448536269514722, -.75
        at_origin = independent_rejection(c, rho, 0, 0)
        # This witness is selected from the analytic boundary problem, not data.
        result = gaussian_null_supremum(c, rho)
        self.assertGreater(result['lower'], max(at_origin, float(ndtr(-c))) + 1e-7)

    def test_conservative_resolution_cap(self):
        out = gaussian_null_supremum(1.7, -.9, max_intervals=1)
        self.assertEqual(out['numericalStatus'], 'resolution_capped')
        self.assertGreaterEqual(out['upper'], independent_rejection(1.7, -.9, 0, 0))
        self.assertGreaterEqual(out['upper'], out['lower'])

    def test_symmetry_and_unit_invariance(self):
        base = gaussian_comparison(-3, 2.5, 1, 1, .7,
                                   covariance_basis='known_gaussian')
        for args in ((3, -2.5, 1, 1, .7), (2.5, -3, 1, 1, .7),
                     (-6, 7.5, 4, 9, 4.2)):
            out = gaussian_comparison(*args, covariance_basis='estimated_asymptotic')
            self.assertAlmostEqual(out['pGaussianReference'], base['pGaussianReference'], places=14)
            self.assertFalse(out['scientificAdoption'])
            self.assertFalse(out['finiteSampleValidityEstablished'])

    def test_null_signs_and_invalid_covariance(self):
        for d, s in ((0, 3), (-3, 0), (2, 3), (-2, -3)):
            out = gaussian_comparison(d, s, 1, 1, 0,
                                      covariance_basis='known_gaussian')
            self.assertEqual(out['pGaussianReference'], 1)
        for vals in ((1, 1, 1), (1, 1, 1.1), (0, 1, 0), (1, -1, 0)):
            with self.assertRaises(ValueError):
                gaussian_comparison(-2, 3, *vals, covariance_basis='known_gaussian')
        for c, rho in ((-1, 0), (True, 0), (1, float('nan')), (1, 1.01)):
            with self.assertRaises(ValueError):
                gaussian_null_supremum(c, rho)


if __name__ == '__main__':
    unittest.main()
