import unittest

import numpy as np

from hpspline import select_lambda


class SelectLambdaTest(unittest.TestCase):
    def test_choice_is_reasonable(self):
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0, 1, 150))
        y = np.sin(2 * np.pi * x) + 0.3 * np.cos(8 * np.pi * x) + 0.1 * rng.standard_normal(x.size)

        best, lams, scores = select_lambda(x, y)

        self.assertTrue(0.005 < best < 0.2)
        self.assertEqual(lams.shape, scores.shape)
