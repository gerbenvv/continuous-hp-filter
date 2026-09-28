import unittest
from unittest import mock

import numpy as np

import hpspline
from hpspline import CHPSmoother, hermite_basis


class BandedLinearAlgebraTest(unittest.TestCase):
    def test_element_stiffness_is_exact_integral(self):
        nodes, weights = np.polynomial.legendre.leggauss(4)
        second = hermite_basis((nodes + 1) / 2, 2)

        integral = (second * weights[:, None] / 2).T @ second

        self.assertTrue(np.allclose(integral, hpspline.ELEMENT_STIFFNESS))

    def test_selected_inverse_matches_dense_inverse(self):
        rng = np.random.default_rng(2)
        size, u = 30, hpspline.BANDWIDTH

        # A random symmetric positive definite banded matrix.
        dense = np.zeros((size, size))

        for d in range(1, u + 1):
            values = rng.uniform(-1, 1, size - d)
            dense += np.diag(values, d) + np.diag(values, -d)

        dense += np.diag(np.full(size, 2.0 * u + 1.0))

        band = np.zeros((u + 1, size))

        for d in range(u + 1):
            band[u - d, d:] = np.diagonal(dense, d)

        inverse = np.linalg.inv(dense)
        selected = hpspline.selected_inverse_banded(hpspline.cholesky_banded(band))

        for d in range(u + 1):
            self.assertTrue(np.allclose(selected[u - d, d:], np.diagonal(inverse, d)))

    @unittest.skipIf(hpspline.SCIPY_LINALG is None, "SciPy is not installed.")
    def test_pure_python_fallback_matches_scipy(self):
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0, 1, 60))
        y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(x.size)
        w = rng.uniform(0.5, 2.0, x.size)

        with_scipy = CHPSmoother(0.05, m=30).fit(x, y, w)

        with mock.patch.object(hpspline, "SCIPY_LINALG", None):
            without_scipy = CHPSmoother(0.05, m=30).fit(x, y, w)
            samples_without_scipy = without_scipy.sample(3, rng=1)

        self.assertTrue(np.allclose(with_scipy.theta_, without_scipy.theta_, atol=1e-12))
        self.assertTrue(np.allclose(with_scipy.sample(3, rng=1), samples_without_scipy, atol=1e-10))
