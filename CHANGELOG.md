# Changelog

## 1.0.0 (2026-09-28)

The first release of `hpspline`, the Python implementation of the continuous Hodrick–Prescott
filter: cubic smoothing splines on a uniform Hermite grid, for irregular and weighted data, with
linear-time Bayesian inference.

- `CHPSmoother`: fits in `O(n + m)` by banded Cholesky, and evaluates the function and its
  derivatives anywhere.
- Exact pointwise posterior standard deviations by selected inversion, posterior samples, effective
  degrees of freedom, a noise estimate and generalized cross-validation.
- `select_lambda` chooses the bandwidth by generalized cross-validation.
- `hp_filter`, `lambda_from_hp` and `hp_from_lambda` relate the bandwidth to the classical
  Hodrick–Prescott filter.
- NumPy only, with SciPy's LAPACK banded routines used when installed (`pip install hpspline[fast]`).
