import unittest

import numpy as np

from chp import hp_filter, hp_from_lambda, lambda_from_hp


class HPFilterTest(unittest.TestCase):
    def test_matches_dense_solution(self):
        rng = np.random.default_rng(1)
        y = np.cumsum(rng.standard_normal(50))

        difference = np.diff(np.eye(50), 2, axis=0)
        dense = np.linalg.solve(np.eye(50) + 1600.0 * difference.T @ difference, y)

        self.assertTrue(np.allclose(hp_filter(y, 1600.0), dense))

    def test_short_series_is_returned_unchanged(self):
        y = np.array([1.0, 2.0])

        self.assertTrue(np.array_equal(hp_filter(y, 1600.0), y))

    def test_lambda_conversion_follows_fourth_power_rule(self):
        # Quarterly 1600 corresponds to 6.25 for annual and 129600 for monthly data.
        lam = lambda_from_hp(1600.0, 0.25)

        self.assertAlmostEqual(hp_from_lambda(lam, 1.0), 6.25)
        self.assertAlmostEqual(hp_from_lambda(lam, 1.0 / 12.0), 129600.0)
