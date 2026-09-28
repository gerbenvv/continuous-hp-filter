import unittest

import numpy as np

from hpspline import HPSpline, hermite_basis

try:
    from scipy.interpolate import make_smoothing_spline
except ImportError:
    make_smoothing_spline = None


def make_data(
    n: int = 60, seed: int = 1, irregular: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns noisy observations of a smooth test function, with random weights."""

    rng = np.random.default_rng(seed)

    x = np.sort(rng.uniform(0, 1, n)) if irregular else np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.3 * np.cos(8 * np.pi * x) + 4 * x + 0.1 * rng.standard_normal(n)
    w = rng.uniform(0.5, 2.0, n)

    return x, y, w


def get_design_matrix(smoother: HPSpline, x: np.ndarray) -> np.ndarray:
    """Returns the dense matrix mapping the parameters to the function values at `x`."""

    index, z = smoother._locate(x)
    design = np.zeros((x.size, 2 * smoother.m_))

    for r in range(4):
        design[np.arange(x.size), 2 * index + r] = hermite_basis(z)[:, r]

    return design


def solve_dense_reference(
    smoother: HPSpline, x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solves the same problem densely, integrating `f''^2` with Gauss-Legendre quadrature."""

    m, dt = smoother.m_, smoother.dt_
    w = w / np.sum(w)
    design = get_design_matrix(smoother, x)

    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes = (nodes + 1) / 2
    weights = weights / 2

    second = hermite_basis(nodes, 2) / dt**2
    element = dt * (second * weights[:, None]).T @ second
    penalty = np.zeros((2 * m, 2 * m))

    for j in range(m - 1):
        penalty[2 * j : 2 * j + 4, 2 * j : 2 * j + 4] += element

    matrix = design.T @ (w[:, None] * design) + smoother.alpha_ * penalty
    rhs = design.T @ (w * y)

    return np.linalg.solve(matrix, rhs), matrix


class HPSplineTest(unittest.TestCase):
    def test_matches_dense_reference(self):
        x, y, w = make_data()
        smoother = HPSpline(0.05, m=40).fit(x, y, w)

        theta, _ = solve_dense_reference(smoother, x, y, w)

        self.assertTrue(np.allclose(smoother.theta_, theta, atol=1e-9))

    def test_gradient_vanishes(self):
        x, y, w = make_data()
        smoother = HPSpline(0.05, m=30).fit(x, y, w)

        rng = np.random.default_rng(0)
        base = smoother.loss()
        epsilon = 1e-4

        for _ in range(5):
            direction = rng.standard_normal(smoother.theta_.size)
            direction /= np.linalg.norm(direction)

            forward = smoother.loss(smoother.theta_ + epsilon * direction)
            backward = smoother.loss(smoother.theta_ - epsilon * direction)
            slope = (forward - backward) / (2 * epsilon)

            self.assertLess(abs(slope), 1e-7)
            self.assertGreater(smoother.loss(smoother.theta_ + 1e-2 * direction), base)

    def test_penalty_formula_matches_quadrature(self):
        x, y, w = make_data()
        smoother = HPSpline(0.05, m=30).fit(x, y, w)

        fine = np.linspace(smoother.t0_, smoother.t0_ + (smoother.m_ - 1) * smoother.dt_, 200001)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        numeric = integrate(smoother(fine, 2) ** 2, fine)

        self.assertTrue(np.isclose(smoother.penalty(), numeric, rtol=1e-4))

    @unittest.skipIf(make_smoothing_spline is None, "SciPy is not installed.")
    def test_equals_exact_smoothing_spline_when_data_on_knots(self):
        x, y, w = make_data(n=41, irregular=False)
        fine = np.linspace(0, 1, 2001)

        # Knots exactly at the data.
        smoother = HPSpline(0.08, m=41).fit(x, y, w)
        exact = make_smoothing_spline(x, y, w=w / w.sum(), lam=smoother.alpha_)

        self.assertLess(np.max(np.abs(smoother(fine) - exact(fine))), 1e-8)

        # Extra knots between the data do not change the exact solution either.
        smoother = HPSpline(0.08, m=161).fit(x, y, w)

        self.assertLess(np.max(np.abs(smoother(fine) - exact(fine))), 1e-8)

    @unittest.skipIf(make_smoothing_spline is None, "SciPy is not installed.")
    def test_converges_to_exact_smoothing_spline_off_grid(self):
        x, y, w = make_data(n=80)
        fine = np.linspace(x.min(), x.max(), 3001)

        exact = None
        errors = []

        for m in (11, 21, 41, 81, 161, 321):
            smoother = HPSpline(0.05, m=m).fit(x, y, w)

            if exact is None:
                exact = make_smoothing_spline(x, y, w=w / w.sum(), lam=smoother.alpha_)

            errors.append(np.max(np.abs(smoother(fine) - exact(fine))))

        errors = np.array(errors)

        self.assertTrue(np.all(np.diff(errors) < 0))
        self.assertLess(errors[-1], 1e-4)

    def test_scale_invariance(self):
        x, y, w = make_data()
        reference = HPSpline(0.05, m=50).fit(x, y, w)

        for scale in (1e-3, 7.0, 1e4):
            scaled = HPSpline(0.05 * scale, m=50).fit(x * scale, y, w)
            _, h, k = scaled.knots

            self.assertTrue(np.allclose(h, reference.knots[1], atol=1e-8))
            self.assertTrue(np.allclose(k * scale, reference.knots[2], atol=1e-6))

    def test_covariance_band_and_edf_match_dense(self):
        x, y, w = make_data()
        smoother = HPSpline(0.03, m=40).fit(x, y, w)

        _, matrix = solve_dense_reference(smoother, x, y, w)
        inverse = np.linalg.inv(matrix)

        band = smoother.covariance_band
        u = band.shape[0] - 1

        for d in range(u + 1):
            self.assertTrue(
                np.allclose(band[u - d, d:], np.diagonal(inverse, d), rtol=1e-8, atol=1e-12)
            )

        design = get_design_matrix(smoother, x)
        hat = design @ inverse @ design.T @ np.diag(w / w.sum())

        self.assertTrue(np.isclose(smoother.edf, np.trace(hat)))

    def test_edf_limits(self):
        x, y, _ = make_data(n=30, irregular=False)

        # A very smooth fit is a straight line; a very rough one interpolates.
        self.assertLess(abs(HPSpline(3.0, m=30).fit(x, y).edf - 2.0), 1e-3)
        self.assertLess(abs(HPSpline(1e-5, m=30).fit(x, y).edf - 30.0), 1e-3)

    def test_posterior_samples_match_std(self):
        x, y, _ = make_data(n=40)
        smoother = HPSpline(0.1, m=20).fit(x, y)

        grid = np.linspace(0, 1, 7)
        samples = smoother.evaluate(smoother.sample(20000, rng=3), grid)
        tolerance = 4 * smoother.std(grid).max() / np.sqrt(20000) * 3

        self.assertTrue(np.allclose(samples.std(axis=0), smoother.std(grid), rtol=0.03))
        self.assertTrue(np.allclose(samples.mean(axis=0), smoother(grid), atol=tolerance))

    def test_known_sigma_coverage(self):
        rng = np.random.default_rng(5)
        grid = np.linspace(0.05, 0.95, 50)
        hits = []

        for _ in range(60):
            x = np.sort(rng.uniform(0, 1, 200))
            y = np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(x.size)

            smoother = HPSpline(0.04, m=100).fit(x, y, sigma=0.2)
            error = np.abs(smoother(grid) - np.sin(2 * np.pi * grid))

            hits.append(error < 2 * smoother.std(grid))

        # Bayesian bands have (approximately) nominal coverage averaged over the function.
        self.assertTrue(0.85 < np.mean(hits) <= 1.0)
