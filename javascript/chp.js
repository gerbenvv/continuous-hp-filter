/*
 * Continuous Hodrick-Prescott (CHP) smoother.
 *
 * Fits a C^1 piecewise cubic Hermite function f on a uniform, data-independent knot grid by
 * minimizing
 *
 *     J(f) = sum_i w_i (f(x_i) - y_i)^2 + (lambda^4 / L) * integral f''(x)^2 dx,
 *
 * with weights normalized to sum one and L the domain length (or the data extent). The normal
 * equations form a symmetric positive definite system with 7 diagonals, solved in O(n + m). The
 * same matrix is the posterior precision of a Gaussian model, which gives credible bands, posterior
 * samples, effective degrees of freedom and GCV in O(n + m).
 *
 * Slopes are stored scaled by the knot spacing, p_j = dt * k_j. The parameter vector is
 * theta = [h_1, p_1, h_2, p_2, ..., h_m, p_m].
 *
 * Works as a browser global (window.CHP) and as a CommonJS / Node module.
 */

(function (root, factory) {
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.CHP = factory();
    }
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    // Number of diagonals above the main diagonal.
    const BANDWIDTH = 3;

    // Integral over [0, 1] of phi''(z) phi''(z)^T (the Euler-Bernoulli beam element stiffness).
    const ELEMENT_STIFFNESS = [
        [12, 6, -12, 6],
        [6, 4, -6, 2],
        [-12, -6, 12, -6],
        [6, 2, -6, 4],
    ];

    /**
     * Returns the cubic Hermite basis on [0, 1], or its derivative nu, ordered as
     * [h_j, p_j, h_j+1, p_j+1].
     */
    function hermiteBasis(z, nu, output) {
        output = output || new Float64Array(4);

        if (nu === 0) {
            const oneMinusZ = 1 - z;

            output[0] = oneMinusZ * oneMinusZ * (1 + 2 * z);
            output[1] = z * oneMinusZ * oneMinusZ;
            output[2] = z * z * (3 - 2 * z);
            output[3] = z * z * (z - 1);
        } else if (nu === 1) {
            output[0] = 6 * z * (z - 1);
            output[1] = (1 - z) * (1 - 3 * z);
            output[2] = 6 * z * (1 - z);
            output[3] = z * (3 * z - 2);
        } else if (nu === 2) {
            output[0] = 12 * z - 6;
            output[1] = 6 * z - 4;
            output[2] = 6 - 12 * z;
            output[3] = 6 * z - 2;
        } else if (nu === 3) {
            output[0] = 12;
            output[1] = 6;
            output[2] = -12;
            output[3] = 6;
        } else {
            output.fill(0);
        }

        return output;
    }

    // The banded linear algebra below works on symmetric positive definite matrices. A matrix of
    // size N with bandwidth u is stored as u + 1 rows (LAPACK upper form):
    // band[u + i - j][j] = a[i][j] for j >= i. The Cholesky factor U (a = U^T U) is stored likewise.

    /**
     * Returns an all-zero band for a matrix of the given size and bandwidth.
     */
    function createBand(size, u) {
        const band = [];

        for (let d = 0; d <= u; ++d) {
            band.push(new Float64Array(size));
        }

        return band;
    }

    /**
     * Returns the upper Cholesky factor of a symmetric positive definite banded matrix.
     */
    function choleskyBanded(band) {
        const u = band.length - 1;
        const size = band[u].length;
        const a = band.map((row) => Float64Array.from(row));

        for (let j = 0; j < size; ++j) {
            const start = Math.max(0, j - u);

            // Off-diagonal entries of column j.
            for (let i = start; i < j; ++i) {
                let total = a[u + i - j][j];

                for (let k = start; k < i; ++k) {
                    total -= a[u + k - i][i] * a[u + k - j][j];
                }

                a[u + i - j][j] = total / a[u][i];
            }

            // Diagonal entry.
            let total = a[u][j];

            for (let k = start; k < j; ++k) {
                const value = a[u + k - j][j];
                total -= value * value;
            }

            if (!(total > 0)) {
                throw new Error(
                    'System is singular: need at least two distinct x with positive weight.'
                );
            }

            a[u][j] = Math.sqrt(total);
        }

        return a;
    }

    /**
     * Solves (U^T U) x = rhs given the upper banded Cholesky factor U.
     */
    function choSolveBanded(factor, rhs) {
        const u = factor.length - 1;
        const size = factor[u].length;
        const x = Float64Array.from(rhs);

        // Forward substitution with U^T.
        for (let i = 0; i < size; ++i) {
            let total = x[i];

            for (let k = Math.max(0, i - u); k < i; ++k) {
                total -= factor[u + k - i][i] * x[k];
            }

            x[i] = total / factor[u][i];
        }

        // Backward substitution with U.
        solveUpperInPlace(factor, x);

        return x;
    }

    /**
     * Solves U x = rhs in place for an upper banded triangular U, with x holding rhs on entry.
     */
    function solveUpperInPlace(factor, x) {
        const u = factor.length - 1;
        const size = factor[u].length;

        for (let i = size - 1; i >= 0; --i) {
            const end = Math.min(size, i + u + 1);
            let total = x[i];

            for (let k = i + 1; k < end; ++k) {
                total -= factor[u + i - k][k] * x[k];
            }

            x[i] = total / factor[u][i];
        }

        return x;
    }

    /**
     * Returns the entries of a^-1 inside the band of a = U^T U, using the Takahashi recursion in
     * O(size * u^2).
     */
    function selectedInverseBanded(factor) {
        const u = factor.length - 1;
        const size = factor[u].length;
        const inverse = createBand(size, u);

        for (let i = size - 1; i >= 0; --i) {
            const inverseDiagonal = 1 / factor[u][i];
            const end = Math.min(size, i + u + 1);

            // Off-diagonal entries of row i, from far to near.
            for (let j = end - 1; j > i; --j) {
                let total = 0;

                for (let k = i + 1; k < end; ++k) {
                    const entry = k <= j ? inverse[u + k - j][j] : inverse[u + j - k][k];
                    total += factor[u + i - k][k] * entry;
                }

                inverse[u + i - j][j] = -inverseDiagonal * total;
            }

            // Diagonal entry.
            let total = 0;

            for (let k = i + 1; k < end; ++k) {
                total += factor[u + i - k][k] * inverse[u + i - k][k];
            }

            inverse[u][i] = inverseDiagonal * (inverseDiagonal - total);
        }

        return inverse;
    }

    /**
     * Returns a standard normal deviate (Box-Muller) from a uniform generator.
     */
    function gaussian(random) {
        let uniform = 0;

        while (uniform === 0) {
            uniform = random();
        }

        return Math.sqrt(-2 * Math.log(uniform)) * Math.cos(2 * Math.PI * random());
    }

    /**
     * Returns a small seedable uniform generator (mulberry32) for reproducible samples.
     */
    function seededRandom(seed) {
        let state = seed >>> 0;

        return function () {
            state = (state + 0x6d2b79f5) >>> 0;

            let t = state;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    /**
     * Returns the minimum and maximum of an array of numbers.
     */
    function getRange(values) {
        let minimum = Infinity;
        let maximum = -Infinity;

        for (const value of values) {
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
        }

        return [minimum, maximum];
    }

    class CHPSmoother {
        /**
         * Creates a smoother.
         *
         * lambda: smoothness as a length on the x-axis (features shorter than about 2 pi lambda
         * are smoothed away). Options: m or dt (knot count or spacing), bounds [lower, upper]
         * (null entries use the data), normalization ('domain', 'data' or a number),
         * knotsPerLambda, minKnots and maxKnots.
         */
        constructor(lambda, options = {}) {
            if (!(lambda > 0)) {
                throw new Error('lambda must be positive.');
            }

            if (options.m != null && options.dt != null) {
                throw new Error('Give either m or dt, not both.');
            }

            this.lambda = lambda;
            this.options = Object.assign(
                {
                    m: null,
                    dt: null,
                    bounds: null,
                    normalization: 'domain',
                    knotsPerLambda: 8,
                    minKnots: 32,
                    maxKnots: 100000,
                },
                options
            );
        }

        setupGrid(x) {
            const options = this.options;
            const [dataMin, dataMax] = getRange(x);

            let lower = options.bounds && options.bounds[0] != null ? options.bounds[0] : dataMin;
            let upper = options.bounds && options.bounds[1] != null ? options.bounds[1] : dataMax;

            if (!(lower < upper)) {
                throw new Error(
                    'Incorrect bounds: need lower < upper (and at least two distinct x).'
                );
            }

            let m;
            let dt;

            if (options.dt != null) {
                if (!(options.dt > 0)) {
                    throw new Error('dt must be positive.');
                }

                dt = options.dt;
                m = Math.max(2, Math.ceil((upper - lower) / dt - 1e-9) + 1);
                upper = lower + (m - 1) * dt;
            } else {
                if (options.m != null) {
                    m = options.m;
                } else {
                    m = Math.ceil(((upper - lower) * options.knotsPerLambda) / this.lambda) + 1;
                    m = Math.min(Math.max(m, options.minKnots), options.maxKnots);
                }

                if (m < 2) {
                    throw new Error('Need at least two knots.');
                }

                dt = (upper - lower) / (m - 1);
            }

            let length;

            if (options.normalization === 'domain') {
                length = upper - lower;
            } else if (options.normalization === 'data') {
                length = dataMax - dataMin;

                if (!(length > 0)) {
                    throw new Error("normalization='data' needs at least two distinct x.");
                }
            } else {
                length = options.normalization;
            }

            this.t0 = lower;
            this.m = m;
            this.dt = dt;
            this.length = length;
            this.alpha = Math.pow(this.lambda, 4) / length;
        }

        locate(x) {
            const position = (x - this.t0) / this.dt;
            const index = Math.min(Math.max(Math.floor(position), 0), this.m - 2);

            return [index, position - index];
        }

        /**
         * Fits the smoother to points (x_i, y_i).
         *
         * options.w: relative weights. options.sigma: known standard deviations (a number or an
         * array); then w = 1 / sigma^2 and the uncertainty uses the known noise level.
         */
        fit(x, y, options = {}) {
            const n = x.length;

            if (n === 0 || y.length !== n) {
                throw new Error('x and y must be non-empty and of equal length.');
            }

            let w;

            if (options.sigma != null) {
                const sigma = options.sigma;
                w = new Float64Array(n);

                for (let i = 0; i < n; ++i) {
                    const deviation = typeof sigma === 'number' ? sigma : sigma[i];
                    w[i] = 1 / (deviation * deviation);
                }

                this.totalPrecision = w.reduce((a, b) => a + b, 0);
            } else {
                this.totalPrecision = null;
                w = options.w ? Float64Array.from(options.w) : new Float64Array(n).fill(1);
            }

            const sum = w.reduce((a, b) => a + b, 0);

            if (w.some((value) => value < 0) || !(sum > 0)) {
                throw new Error('Weights must be non-negative and not all zero.');
            }

            for (let i = 0; i < n; ++i) {
                w[i] /= sum;
            }

            this.setupGrid(x);
            this.x = Float64Array.from(x);
            this.y = Float64Array.from(y);
            this.w = w;
            this.index = new Int32Array(n);
            this.z = new Float64Array(n);

            for (let i = 0; i < n; ++i) {
                [this.index[i], this.z[i]] = this.locate(x[i]);
            }

            const [band, rhs] = this.assemble();

            this.factor = choleskyBanded(band);
            this.theta = choSolveBanded(this.factor, rhs);
            this.covariance = null;
            this.cachedEdf = null;

            return this;
        }

        assemble() {
            const u = BANDWIDTH;
            const size = 2 * this.m;
            const band = createBand(size, u);
            const rhs = new Float64Array(size);
            const phi = new Float64Array(4);

            // Data term.
            for (let i = 0; i < this.x.length; ++i) {
                hermiteBasis(this.z[i], 0, phi);

                const base = 2 * this.index[i];
                const w = this.w[i];

                for (let r = 0; r < 4; ++r) {
                    const weighted = w * phi[r];
                    rhs[base + r] += weighted * this.y[i];

                    for (let c = r; c < 4; ++c) {
                        band[u - (c - r)][base + c] += weighted * phi[c];
                    }
                }
            }

            // Curvature term.
            const coefficient = this.alpha / Math.pow(this.dt, 3);

            for (let j = 0; j < this.m - 1; ++j) {
                const base = 2 * j;

                for (let r = 0; r < 4; ++r) {
                    for (let c = r; c < 4; ++c) {
                        band[u - (c - r)][base + c] += coefficient * ELEMENT_STIFFNESS[r][c];
                    }
                }
            }

            return [band, rhs];
        }

        /**
         * Returns the row phi(x) for derivative nu and the index of its first parameter. Beyond
         * the end knots the row describes the natural extension, a straight line with the end
         * slope.
         */
        designRow(x, nu = 0) {
            const [index, z] = this.locate(x);
            const base = 2 * index;

            if (z < 0 || z > 1) {
                const phi = new Float64Array(4);
                const at = z < 0 ? 0 : 2;

                if (nu === 0) {
                    phi[at] = 1;
                    phi[at + 1] = z < 0 ? z : z - 1;
                } else if (nu === 1) {
                    phi[at + 1] = 1 / this.dt;
                }

                return [base, phi];
            }

            const phi = hermiteBasis(z, nu);
            const scale = Math.pow(this.dt, -nu);

            for (let r = 0; r < 4; ++r) {
                phi[r] *= scale;
            }

            return [base, phi];
        }

        /**
         * Returns the value (nu = 0) or derivative nu at x of the function defined by theta.
         */
        evaluateTheta(theta, x, nu = 0) {
            const [base, phi] = this.designRow(x, nu);

            return (
                phi[0] * theta[base] +
                phi[1] * theta[base + 1] +
                phi[2] * theta[base + 2] +
                phi[3] * theta[base + 3]
            );
        }

        /**
         * Returns the value (nu = 0) or derivative nu of the fitted function at x.
         */
        evaluate(x, nu = 0) {
            return this.evaluateTheta(this.theta, x, nu);
        }

        /**
         * Returns the knot positions t, heights h and slopes k.
         */
        knots() {
            const t = [];
            const h = [];
            const k = [];

            for (let j = 0; j < this.m; ++j) {
                t.push(this.t0 + j * this.dt);
                h.push(this.theta[2 * j]);
                k.push(this.theta[2 * j + 1] / this.dt);
            }

            return { t, h, k };
        }

        /**
         * Returns the integral of f''(x)^2 over the domain.
         */
        penalty(theta = this.theta) {
            let total = 0;

            for (let j = 0; j < this.m - 1; ++j) {
                const dh = theta[2 * j + 2] - theta[2 * j];
                const a = theta[2 * j + 1] - dh;
                const b = -theta[2 * j + 3] + dh;

                total += a * a - a * b + b * b;
            }

            return (4 * total) / Math.pow(this.dt, 3);
        }

        weightedResidualSquares() {
            let total = 0;

            for (let i = 0; i < this.x.length; ++i) {
                const residual = this.y[i] - this.evaluate(this.x[i]);
                total += this.w[i] * residual * residual;
            }

            return total;
        }

        /**
         * Returns the objective J.
         */
        loss() {
            return this.weightedResidualSquares() + this.alpha * this.penalty();
        }

        /**
         * Returns phi^T A^-1 phi for the 4 parameters of the element containing x.
         */
        quadraticForm(x, nu = 0) {
            if (!this.covariance) {
                this.covariance = selectedInverseBanded(this.factor);
            }

            const covariance = this.covariance;
            const u = BANDWIDTH;
            const [base, phi] = this.designRow(x, nu);

            let result = 0;

            for (let r = 0; r < 4; ++r) {
                for (let c = 0; c < 4; ++c) {
                    const i = Math.min(r, c);
                    const j = Math.max(r, c);

                    result += phi[r] * phi[c] * covariance[u + i - j][base + j];
                }
            }

            return result;
        }

        /**
         * Returns the effective degrees of freedom, the trace of the hat matrix.
         */
        edf() {
            if (this.cachedEdf === null) {
                let total = 0;

                for (let i = 0; i < this.x.length; ++i) {
                    total += this.w[i] * this.quadraticForm(this.x[i]);
                }

                this.cachedEdf = total;
            }

            return this.cachedEdf;
        }

        /**
         * Returns the variance of an observation with average weight: known if sigma was given,
         * otherwise estimated as n * sum(w r^2) / (n - edf).
         */
        noiseVariance() {
            const n = this.x.length;

            if (this.totalPrecision !== null) {
                return n / this.totalPrecision;
            }

            return (n * this.weightedResidualSquares()) / Math.max(n - this.edf(), 1e-12);
        }

        /**
         * Returns the factor s such that the posterior covariance of theta is s * A^-1.
         */
        posteriorScale() {
            if (this.totalPrecision !== null) {
                return 1 / this.totalPrecision;
            }

            return this.noiseVariance() / this.x.length;
        }

        /**
         * Returns the posterior standard deviation of f (or its derivative nu) at x. With
         * predictive set, the noise of a new observation with average weight is added.
         */
        std(x, nu = 0, predictive = false) {
            let variance = this.posteriorScale() * this.quadraticForm(x, nu);

            if (predictive) {
                variance += this.noiseVariance();
            }

            return Math.sqrt(Math.max(variance, 0));
        }

        /**
         * Draws a parameter vector from the posterior, given a uniform generator on [0, 1).
         */
        sample(random = Math.random) {
            const size = 2 * this.m;
            const noise = new Float64Array(size);

            for (let i = 0; i < size; ++i) {
                noise[i] = gaussian(random);
            }

            solveUpperInPlace(this.factor, noise);

            const scale = Math.sqrt(this.posteriorScale());

            for (let i = 0; i < size; ++i) {
                noise[i] = this.theta[i] + scale * noise[i];
            }

            return noise;
        }

        /**
         * Returns the generalized cross-validation score (lower is better).
         */
        gcv() {
            const n = this.x.length;
            const ratio = 1 - this.edf() / n;

            return (n * this.weightedResidualSquares()) / (ratio * ratio);
        }
    }

    /**
     * Chooses lambda by GCV: a log-spaced grid, then golden-section refinement on log(lambda).
     */
    function selectLambda(x, y, options = {}) {
        const [dataMin, dataMax] = getRange(x);
        const extent = dataMax - dataMin;
        const smootherOptions = options.smoother || {};
        const lambdas =
            options.lambdas ||
            Array.from({ length: 25 }, (_, i) => extent * Math.pow(10, -3.5 + (4 * i) / 24));

        const getScore = (lambda) => {
            try {
                return new CHPSmoother(lambda, smootherOptions).fit(x, y, { w: options.w }).gcv();
            } catch (_error) {
                return Infinity;
            }
        };

        const scores = lambdas.map(getScore);
        let best = 0;

        for (let i = 1; i < scores.length; ++i) {
            if (scores[i] < scores[best]) {
                best = i;
            }
        }

        let bestLambda = lambdas[best];

        if (best > 0 && best < lambdas.length - 1) {
            const ratio = (Math.sqrt(5) - 1) / 2;

            let a = Math.log(lambdas[best - 1]);
            let b = Math.log(lambdas[best + 1]);
            let c = b - ratio * (b - a);
            let d = a + ratio * (b - a);
            let scoreC = getScore(Math.exp(c));
            let scoreD = getScore(Math.exp(d));

            for (let iteration = 0; iteration < 30 && b - a > 1e-4; ++iteration) {
                if (scoreC < scoreD) {
                    b = d;
                    d = c;
                    scoreD = scoreC;
                    c = b - ratio * (b - a);
                    scoreC = getScore(Math.exp(c));
                } else {
                    a = c;
                    c = d;
                    scoreC = scoreD;
                    d = a + ratio * (b - a);
                    scoreD = getScore(Math.exp(d));
                }
            }

            bestLambda = Math.exp((a + b) / 2);
        }

        return { lambda: bestLambda, lambdas, scores };
    }

    /**
     * Returns the CHP length scale equivalent to a classic HP parameter at the given spacing.
     */
    function lambdaFromHp(lambdaHp, spacing) {
        return spacing * Math.pow(lambdaHp, 0.25);
    }

    /**
     * Returns the classic HP parameter equivalent to a CHP length scale at the given spacing.
     */
    function hpFromLambda(lambda, spacing) {
        return Math.pow(lambda / spacing, 4);
    }

    /**
     * Returns the classic Hodrick-Prescott / Whittaker trend of equally spaced observations.
     */
    function hpFilter(y, lambdaHp) {
        const n = y.length;

        if (n < 3) {
            return Float64Array.from(y);
        }

        const u = BANDWIDTH;
        const band = createBand(n, u);
        const stencil = [1, -2, 1];

        for (let i = 0; i < n - 2; ++i) {
            for (let r = 0; r < 3; ++r) {
                for (let c = r; c < 3; ++c) {
                    band[u - (c - r)][i + c] += lambdaHp * stencil[r] * stencil[c];
                }
            }
        }

        for (let i = 0; i < n; ++i) {
            band[u][i] += 1;
        }

        return choSolveBanded(choleskyBanded(band), y);
    }

    return {
        BANDWIDTH,
        ELEMENT_STIFFNESS,
        CHPSmoother,
        hermiteBasis,
        choleskyBanded,
        choSolveBanded,
        selectedInverseBanded,
        selectLambda,
        lambdaFromHp,
        hpFromLambda,
        hpFilter,
        seededRandom,
    };
});
