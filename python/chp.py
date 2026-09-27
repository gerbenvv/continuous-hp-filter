"""Continuous Hodrick-Prescott (CHP) smoother.

Fits a C^1 piecewise cubic Hermite function `f` on a uniform, data-independent knot grid
`t_1 < t_2 < ... < t_m` by minimizing

```
J(f) = sum_i w_i (f(x_i) - y_i)^2 + (lambda^4 / L) * integral_{t_1}^{t_m} f''(x)^2 dx,
```

where the weights `w_i` sum to one and `L` is a normalization length (by default the domain
length `t_m - t_1`). The unknowns are the height `h_j` and slope `k_j` at every knot. `J` is
quadratic in these, so the minimizer is the solution of a symmetric positive definite linear
system with half-bandwidth 3 (7 diagonals), solved in O(n + m) time.

The same system is the posterior precision of a Gaussian model (integrated Wiener process
prior, Gaussian noise), which gives pointwise credible bands, posterior samples, effective
degrees of freedom and generalized cross-validation, all in O(n + m) as well.

Only NumPy is required. If SciPy is installed, its LAPACK banded routines are used for speed;
the results are identical.

Internally the slopes are stored scaled by the knot spacing, `p_j = dt * k_j`, which makes the
system well conditioned and independent of the x-axis scale. The parameter vector is
`theta = [h_1, p_1, h_2, p_2, ..., h_m, p_m]`.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np

# SciPy's LAPACK routines, used for speed when SciPy is installed.
SCIPY_LINALG: ModuleType | None

try:
    import scipy.linalg as SCIPY_LINALG
except ImportError:
    SCIPY_LINALG = None

# Number of diagonals above the main diagonal of the system matrix.
BANDWIDTH: int = 3

# Integral over [0, 1] of `phi''(z) phi''(z)^T` for the Hermite basis (the Euler-Bernoulli beam
# element stiffness). With `p = dt * k`, the integral of `f''(x)^2` over one element equals
# `theta_e^T K theta_e / dt^3`.
ELEMENT_STIFFNESS: np.ndarray = np.array(
    [
        [12.0, 6.0, -12.0, 6.0],
        [6.0, 4.0, -6.0, 2.0],
        [-12.0, -6.0, 12.0, -6.0],
        [6.0, 2.0, -6.0, 4.0],
    ]
)


def hermite_basis(z: Any, nu: int = 0) -> np.ndarray:
    """Returns the cubic Hermite basis on the unit interval, or its `nu`-th derivative in `z`.

    The basis functions are ordered as `[h_j, p_j, h_j+1, p_j+1]`.

    Args:
        z: Position(s) in the unit interval.
        nu: Order of the derivative.

    Returns:
        An array of shape `(..., 4)`.
    """

    z = np.asarray(z, dtype=np.float64)

    if nu == 0:
        one_minus_z = 1.0 - z

        return np.stack(
            [
                one_minus_z * one_minus_z * (1.0 + 2.0 * z),
                z * one_minus_z * one_minus_z,
                z * z * (3.0 - 2.0 * z),
                z * z * (z - 1.0),
            ],
            axis=-1,
        )

    if nu == 1:
        return np.stack(
            [
                6.0 * z * (z - 1.0),
                (1.0 - z) * (1.0 - 3.0 * z),
                6.0 * z * (1.0 - z),
                z * (3.0 * z - 2.0),
            ],
            axis=-1,
        )

    if nu == 2:
        return np.stack(
            [
                12.0 * z - 6.0,
                6.0 * z - 4.0,
                6.0 - 12.0 * z,
                6.0 * z - 2.0,
            ],
            axis=-1,
        )

    if nu == 3:
        ones = np.ones_like(z)

        return np.stack([12.0 * ones, 6.0 * ones, -12.0 * ones, 6.0 * ones], axis=-1)

    return np.zeros(np.shape(z) + (4,))


# The banded linear algebra below works on symmetric positive definite matrices stored in LAPACK
# upper form: `band[u + i - j, j] = a[i, j]` for `j >= i`, with `u = BANDWIDTH`. The Cholesky
# factor `U` (with `a = U^T U`) is stored the same way.


def cholesky_banded(band: np.ndarray) -> np.ndarray:
    """Returns the upper Cholesky factor of a symmetric positive definite banded matrix.

    Args:
        band: The matrix in LAPACK upper banded storage.

    Raises:
        np.linalg.LinAlgError: If the matrix is not positive definite.
    """

    if SCIPY_LINALG is not None:
        return SCIPY_LINALG.cholesky_banded(band, lower=False, check_finite=False)

    u = band.shape[0] - 1
    size = band.shape[1]
    a = [list(map(float, row)) for row in band]

    for j in range(size):
        # Off-diagonal entries of column j: U[i, j] for i in [j - u, j).
        for i in range(max(0, j - u), j):
            total = a[u + i - j][j]

            for k in range(max(0, j - u), i):
                total -= a[u + k - i][i] * a[u + k - j][j]

            a[u + i - j][j] = total / a[u][i]

        # Diagonal entry.
        total = a[u][j]

        for k in range(max(0, j - u), j):
            value = a[u + k - j][j]
            total -= value * value

        if total <= 0.0:
            raise np.linalg.LinAlgError("Matrix is not positive definite.")

        a[u][j] = math.sqrt(total)

    return np.array(a)


def cho_solve_banded(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves `(U^T U) x = rhs` given the upper banded Cholesky factor `U`.

    Args:
        factor: The Cholesky factor in LAPACK upper banded storage.
        rhs: Right-hand side, a vector or a matrix with one column per system.
    """

    if SCIPY_LINALG is not None:
        return SCIPY_LINALG.cho_solve_banded((factor, False), rhs, check_finite=False)

    u = factor.shape[0] - 1
    size = factor.shape[1]
    x = np.array(rhs, dtype=np.float64)

    single = x.ndim == 1
    if single:
        x = x[:, None]

    # Forward substitution with U^T.
    for i in range(size):
        for k in range(max(0, i - u), i):
            x[i] -= factor[u + k - i, i] * x[k]

        x[i] /= factor[u, i]

    # Backward substitution with U.
    for i in range(size - 1, -1, -1):
        for k in range(i + 1, min(size, i + u + 1)):
            x[i] -= factor[u + i - k, k] * x[k]

        x[i] /= factor[u, i]

    return x[:, 0] if single else x


def solve_upper_banded(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves `U x = rhs` for an upper banded triangular `U` (used for sampling).

    Args:
        factor: The upper triangular matrix in LAPACK upper banded storage.
        rhs: Right-hand side, a vector or a matrix with one column per system.
    """

    u = factor.shape[0] - 1

    if SCIPY_LINALG is not None:
        return SCIPY_LINALG.solve_banded((0, u), factor, rhs, check_finite=False)

    size = factor.shape[1]
    x = np.array(rhs, dtype=np.float64)

    for i in range(size - 1, -1, -1):
        for k in range(i + 1, min(size, i + u + 1)):
            x[i] -= factor[u + i - k, k] * x[k]

        x[i] /= factor[u, i]

    return x


def selected_inverse_banded(factor: np.ndarray) -> np.ndarray:
    """Returns the entries of `a^-1` inside the band of `a = U^T U`.

    Uses the Takahashi recursion, which runs in O(size * u^2) instead of the O(size^2) of a full
    inverse.

    Args:
        factor: The upper Cholesky factor `U` in LAPACK upper banded storage.

    Returns:
        The band of `a^-1` in the same storage.
    """

    u = factor.shape[0] - 1
    size = factor.shape[1]
    upper = [list(map(float, row)) for row in factor]
    inverse = [[0.0] * size for _ in range(u + 1)]

    def get_inverse(i: int, j: int) -> float:
        # The entry `a^-1[i, j]` for `|i - j| <= u`, which must already be computed.
        if i > j:
            i, j = j, i

        return inverse[u + i - j][j]

    for i in range(size - 1, -1, -1):
        inverse_diagonal = 1.0 / upper[u][i]
        end = min(size, i + u + 1)

        # Off-diagonal entries of row i, from far to near.
        for j in range(end - 1, i, -1):
            total = 0.0

            for k in range(i + 1, end):
                total += upper[u + i - k][k] * get_inverse(k, j)

            inverse[u + i - j][j] = -inverse_diagonal * total

        # Diagonal entry.
        total = 0.0

        for k in range(i + 1, end):
            total += upper[u + i - k][k] * inverse[u + i - k][k]

        inverse[u][i] = inverse_diagonal * (inverse_diagonal - total)

    return np.array(inverse)


class CHPSmoother:
    """Continuous Hodrick-Prescott smoother.

    Args:
        lam: Smoothness as a length on the x-axis. For evenly spread data the smoother acts as a
            low-pass filter with gain `1 / (1 + (lam * omega)^4)`; features shorter than about
            `2 * pi * lam` are smoothed away. Scaling the x-axis by `s` and `lam` by `s` gives the
            same fit.
        m: Number of knots. Mutually exclusive with `dt`.
        dt: Knot spacing. The upper bound is extended to fit a whole number of intervals.
        bounds: Domain `[t_1, t_m]`. Either entry may be `None` to use the data minimum or
            maximum.
        normalization: The length `L` in `lam^4 / L`: `"domain"` for the domain length,
            `"data"` for the data extent (so that padding the domain does not change the fit),
            or a fixed number.
        knots_per_lambda: When neither `m` nor `dt` is given, the spacing is
            `dt = lam / knots_per_lambda`.
        min_knots: Lower limit on the automatically chosen number of knots.
        max_knots: Upper limit on the automatically chosen number of knots.
    """

    def __init__(
        self,
        lam: float,
        m: int | None = None,
        dt: float | None = None,
        bounds: tuple[float | None, float | None] | None = None,
        normalization: str | float = "domain",
        knots_per_lambda: float = 8.0,
        min_knots: int = 32,
        max_knots: int = 100_000,
    ):
        if lam <= 0:
            raise ValueError("lam must be positive.")

        if m is not None and dt is not None:
            raise ValueError("Give either m or dt, not both.")

        self.lam = float(lam)
        self.m = m
        self.dt = dt
        self.bounds = bounds
        self.normalization = normalization
        self.knots_per_lambda = knots_per_lambda
        self.min_knots = min_knots
        self.max_knots = max_knots

    def _setup_grid(self, x: np.ndarray) -> None:
        lower, upper = (None, None) if self.bounds is None else self.bounds
        lower = float(np.min(x)) if lower is None else float(lower)
        upper = float(np.max(x)) if upper is None else float(upper)

        if not lower < upper:
            raise ValueError("Incorrect bounds: need lower < upper (and at least two distinct x).")

        if self.dt is not None:
            if self.dt <= 0:
                raise ValueError("dt must be positive.")

            dt = float(self.dt)
            m = int(math.ceil((upper - lower) / dt - 1e-9)) + 1
            m = max(m, 2)
            upper = lower + (m - 1) * dt
        else:
            if self.m is not None:
                m = int(self.m)
            else:
                m = int(math.ceil((upper - lower) * self.knots_per_lambda / self.lam)) + 1
                m = min(max(m, self.min_knots), self.max_knots)

            if m < 2:
                raise ValueError("Need at least two knots.")

            dt = (upper - lower) / (m - 1)

        if self.normalization == "domain":
            length = upper - lower
        elif self.normalization == "data":
            length = float(np.max(x) - np.min(x))

            if length <= 0:
                raise ValueError("normalization='data' needs at least two distinct x.")
        else:
            length = float(self.normalization)

        self.t0_ = lower
        self.m_ = m
        self.dt_ = dt
        self.length_ = length
        self.alpha_ = self.lam**4 / length

    def _locate(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        position = (np.asarray(x, dtype=np.float64) - self.t0_) / self.dt_
        index = np.clip(np.floor(position).astype(np.int64), 0, self.m_ - 2)

        return index, position - index

    def fit(
        self,
        x: Any,
        y: Any,
        w: Any | None = None,
        sigma: Any | None = None,
    ) -> "CHPSmoother":
        """Fits the smoother to points `(x_i, y_i)`.

        Args:
            x: Positions of the observations.
            y: Values of the observations.
            w: Relative weights, normalized to sum one.
            sigma: Alternatively, the known standard deviation of every observation. Then
                `w = 1 / sigma^2` and the posterior uncertainty uses the known noise level
                instead of an estimate.

        Returns:
            The fitted smoother itself.
        """

        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        if x.shape != y.shape or x.size == 0:
            raise ValueError("x and y must be non-empty and of equal length.")

        if w is not None and sigma is not None:
            raise ValueError("Give either w or sigma, not both.")

        if sigma is not None:
            sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), x.shape)
            precision = 1.0 / sigma**2
            self.total_precision_ = float(np.sum(precision))
            w = precision / self.total_precision_
        else:
            self.total_precision_ = None

            if w is None:
                w = np.ones_like(x)
            else:
                w = np.broadcast_to(np.asarray(w, dtype=np.float64), x.shape)

            if np.any(w < 0) or not np.sum(w) > 0:
                raise ValueError("Weights must be non-negative and not all zero.")

            w = w / np.sum(w)

        self._setup_grid(x)
        self.x_, self.y_, self.w_ = x, y, w
        self.index_, self.z_ = self._locate(x)

        band, rhs = self._assemble()

        try:
            self.cholesky_ = cholesky_banded(band)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "System is singular: the data must contain at least two distinct x with positive "
                "weight."
            ) from error

        self.theta_ = cho_solve_banded(self.cholesky_, rhs)
        self._covariance_band = None

        return self

    def _assemble(self) -> tuple[np.ndarray, np.ndarray]:
        m, size, u = self.m_, 2 * self.m_, BANDWIDTH
        phi = hermite_basis(self.z_)
        base = 2 * self.index_

        # Data term: `sum_i w_i phi_i phi_i^T` and `sum_i w_i y_i phi_i`.
        band = np.zeros((u + 1, size))
        rhs = np.zeros(size)

        for r in range(4):
            weighted = self.w_ * phi[:, r]
            rhs += np.bincount(base + r, weights=weighted * self.y_, minlength=size)

            for c in range(r, 4):
                band[u - (c - r)] += np.bincount(
                    base + c, weights=weighted * phi[:, c], minlength=size
                )

        # Curvature term: `alpha / dt^3` times the sum of `K` over the elements.
        coefficient = self.alpha_ / self.dt_**3
        element_base = 2 * np.arange(m - 1)

        for r in range(4):
            for c in range(r, 4):
                band[u - (c - r), element_base + c] += coefficient * ELEMENT_STIFFNESS[r, c]

        return band, rhs

    def _design(self, x: Any, nu: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the rows `phi(x)` (for derivative `nu`) and the parameter indices they multiply.

        Beyond the end knots the rows describe the natural extension, a straight line with the end
        slope.
        """

        x = np.asarray(x, dtype=np.float64)
        index, z = self._locate(x)
        phi = hermite_basis(z, nu) / self.dt_**nu
        columns = 2 * index[..., None] + np.arange(4)

        below, above = z < 0.0, z > 1.0

        if np.any(below) or np.any(above):
            offset = np.where(below, z, z - 1.0)

            if nu == 0:
                line = np.stack([np.ones_like(z), offset], axis=-1)
            elif nu == 1:
                line = np.stack([np.zeros_like(z), np.full_like(z, 1.0 / self.dt_)], axis=-1)
            else:
                line = np.zeros(z.shape + (2,))

            zeros = np.zeros_like(line)
            phi = np.where(below[..., None], np.concatenate([line, zeros], axis=-1), phi)
            phi = np.where(above[..., None], np.concatenate([zeros, line], axis=-1), phi)

        return phi, columns, index, z

    def __call__(self, x: Any, nu: int = 0) -> np.ndarray:
        """Evaluates `f` (`nu = 0`) or its derivatives (`nu = 1, 2, 3`).

        Beyond the domain the function continues linearly.
        """

        phi, columns, _, _ = self._design(x, nu)

        return np.sum(phi * self.theta_[columns], axis=-1)

    @property
    def knots(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The knot positions `t`, heights `h` and slopes `k`."""

        t = self.t0_ + self.dt_ * np.arange(self.m_)

        return t, self.theta_[0::2].copy(), self.theta_[1::2] / self.dt_

    def penalty(self, theta: np.ndarray | None = None) -> float:
        """Returns the integral of `f''(x)^2` over the domain."""

        theta = self.theta_ if theta is None else theta
        h, p = theta[0::2], theta[1::2]

        # With `a = p_j - (h_j+1 - h_j)` and `b = -p_j+1 + (h_j+1 - h_j)` the integral is
        # `4 (a^2 - a b + b^2) / dt^3` per element.
        dh = np.diff(h)
        a = p[:-1] - dh
        b = -p[1:] + dh

        return 4.0 * np.sum(a * a - a * b + b * b) / self.dt_**3

    def residuals(self) -> np.ndarray:
        """Returns the residuals `y_i - f(x_i)` of the fitted data."""

        return self.y_ - self(self.x_)

    def loss(self, theta: np.ndarray | None = None) -> float:
        """Returns the objective `J`, for the fitted parameters or the given `theta`."""

        theta = self.theta_ if theta is None else theta
        phi, columns, _, _ = self._design(self.x_)
        fitted = np.sum(phi * theta[columns], axis=-1)

        return float(np.dot(self.w_, (fitted - self.y_) ** 2) + self.alpha_ * self.penalty(theta))

    @property
    def covariance_band(self) -> np.ndarray:
        """The band of `A^-1` (with `A` the system matrix), computed on first use in O(m)."""

        if self._covariance_band is None:
            self._covariance_band = selected_inverse_banded(self.cholesky_)

        return self._covariance_band

    def _quadratic_form(self, phi: np.ndarray, columns: np.ndarray) -> np.ndarray:
        # Computes `phi^T A^-1 phi` for 4-vectors on consecutive parameters, which always lie
        # within the band.
        covariance, u = self.covariance_band, BANDWIDTH
        result = np.zeros(phi.shape[:-1])

        for r in range(4):
            for c in range(4):
                i, j = (r, c) if r <= c else (c, r)
                entries = covariance[u + i - j, columns[..., j]]
                result += phi[..., r] * phi[..., c] * entries

        return result

    @property
    def edf(self) -> float:
        """The effective degrees of freedom, the trace of the hat matrix."""

        phi, columns, _, _ = self._design(self.x_)

        return float(np.sum(self.w_ * self._quadratic_form(phi, columns)))

    @property
    def noise_variance(self) -> float:
        """The variance of an observation with average weight.

        Known if `sigma` was given, otherwise estimated as `n * sum(w r^2) / (n - edf)`.
        """

        n = self.x_.size

        if self.total_precision_ is not None:
            return n / self.total_precision_

        return n * float(np.dot(self.w_, self.residuals() ** 2)) / max(n - self.edf, 1e-12)

    @property
    def posterior_scale(self) -> float:
        """The factor `s` such that the posterior covariance of `theta` is `s * A^-1`."""

        if self.total_precision_ is not None:
            return 1.0 / self.total_precision_

        return self.noise_variance / self.x_.size

    def std(self, x: Any, nu: int = 0, predictive: bool = False) -> np.ndarray:
        """Returns the pointwise posterior standard deviation of `f` (or its derivative `nu`).

        Args:
            x: Position(s) to evaluate at.
            nu: Order of the derivative.
            predictive: If `True`, adds the noise of a new observation with average weight.
        """

        x = np.asarray(x, dtype=np.float64)
        phi, columns, _, _ = self._design(x, nu)
        variance = self.posterior_scale * self._quadratic_form(phi, columns)

        if predictive:
            variance = variance + self.noise_variance

        return np.sqrt(np.maximum(variance, 0.0))

    def sample(self, size: int = 1, rng: Any = None) -> np.ndarray:
        """Draws parameter vectors `theta` from the Gaussian posterior.

        Args:
            size: Number of samples.
            rng: Seed or generator passed to `np.random.default_rng`.

        Returns:
            An array of shape `(size, 2m)`.
        """

        rng = np.random.default_rng(rng)
        noise = rng.standard_normal((2 * self.m_, size))
        deviation = solve_upper_banded(self.cholesky_, noise) * math.sqrt(self.posterior_scale)

        return (self.theta_[:, None] + deviation).T

    def evaluate(self, theta: np.ndarray, x: Any, nu: int = 0) -> np.ndarray:
        """Evaluates the function defined by an arbitrary parameter vector (e.g. a sample)."""

        phi, columns, _, _ = self._design(np.asarray(x, dtype=np.float64), nu)

        return np.sum(phi * np.asarray(theta)[..., columns], axis=-1)

    def gcv(self) -> float:
        """Returns the generalized cross-validation score (lower is better)."""

        n = self.x_.size

        return n * float(np.dot(self.w_, self.residuals() ** 2)) / (1.0 - self.edf / n) ** 2


def select_lambda(
    x: Any,
    y: Any,
    w: Any | None = None,
    lams: Any | None = None,
    refine: bool = True,
    **kwargs: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Chooses `lam` by generalized cross-validation.

    Evaluates a log-spaced grid, then refines with golden-section search on `log(lam)`.

    Args:
        x: Positions of the observations.
        y: Values of the observations.
        w: Relative weights.
        lams: Candidate values; by default 25 log-spaced values relative to the data extent.
        refine: Whether to refine the best grid value with golden-section search.
        **kwargs: Passed to `CHPSmoother`.

    Returns:
        The best `lam`, the candidate values and their GCV scores.
    """

    x = np.asarray(x, dtype=np.float64)
    extent = float(np.max(x) - np.min(x))

    if lams is None:
        lams = extent * np.logspace(-3.5, 0.5, 25)

    lams = np.asarray(lams, dtype=np.float64)

    def get_score(lam: float) -> float:
        return CHPSmoother(lam, **kwargs).fit(x, y, w).gcv()

    scores = np.array([get_score(lam) for lam in lams])
    best = int(np.argmin(scores))
    best_lam = float(lams[best])

    if refine and 0 < best < len(lams) - 1:
        a, b = math.log(lams[best - 1]), math.log(lams[best + 1])
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        score_c, score_d = get_score(math.exp(c)), get_score(math.exp(d))

        for _ in range(30):
            if score_c < score_d:
                b, d, score_d = d, c, score_c
                c = b - ratio * (b - a)
                score_c = get_score(math.exp(c))
            else:
                a, c, score_c = c, d, score_d
                d = a + ratio * (b - a)
                score_d = get_score(math.exp(d))

            if b - a < 1e-4:
                break

        best_lam = math.exp((a + b) / 2.0)

    return best_lam, lams, scores


def lambda_from_hp(lam_hp: float, spacing: float) -> float:
    """Returns the CHP length scale equivalent to a classic HP parameter at the given spacing."""

    return spacing * lam_hp**0.25


def hp_from_lambda(lam: float, spacing: float) -> float:
    """Returns the classic HP parameter equivalent to a CHP length scale at the given spacing."""

    return (lam / spacing) ** 4


def hp_filter(y: Any, lam_hp: float) -> np.ndarray:
    """Returns the classic Hodrick-Prescott / Whittaker trend.

    Minimizes `sum (h - y)^2 + lam_hp * sum (second difference of h)^2`.

    Args:
        y: Equally spaced observations.
        lam_hp: The classic HP smoothing parameter.
    """

    y = np.asarray(y, dtype=np.float64)
    n = y.size

    if n < 3:
        return y.copy()

    # The matrix `I + lam_hp D^T D` is pentadiagonal; it is stored in upper banded form with
    # `u = 2`, padded to `u = BANDWIDTH`.
    band = np.zeros((BANDWIDTH + 1, n))
    stencil = np.array([1.0, -2.0, 1.0])

    for r in range(3):
        for c in range(r, 3):
            band[BANDWIDTH - (c - r), np.arange(n - 2) + c] += lam_hp * stencil[r] * stencil[c]

    band[BANDWIDTH] += 1.0

    return cho_solve_banded(cholesky_banded(band), y)
