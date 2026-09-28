# Cubic smoothing splines on a uniform Hermite grid

**A continuous Hodrick–Prescott filter for irregular, weighted data, with linear-time Bayesian
inference.**

This repository contains a manuscript and two implementations (Python and JavaScript) of a smoother
that generalizes the Hodrick–Prescott (HP) filter to a continuous function, for data that may be
irregularly spaced and weighted, with exact pointwise uncertainty and a smoothing parameter measured
in the units of the x-axis.

- **Manuscript:** [`paper/manuscript.pdf`](paper/manuscript.pdf) (LaTeX source in
  [`paper/`](paper/)).
- **Interactive demo:** try it online at <https://gerbenvv.github.io/continuous-hp-filter/>, or
  open [`javascript/index.html`](javascript/index.html) in a browser (no server or dependencies
  needed).

![Interactive demo](https://raw.githubusercontent.com/gerbenvv/continuous-hp-filter/main/docs/demo.png)

## The method

Given observations $`(x_i, y_i)`$ with weights $`w_i \ge 0`$, $`\sum_i w_i = 1`$, find the function $`f`$
minimizing

```math
J(f) = \sum_{i=1}^{n} w_i \bigl(y_i - f(x_i)\bigr)^2 + \frac{h^4}{L} \int_{t_1}^{t_m} f''(x)^2 \, dx
```

over $`C^1`$ piecewise cubic Hermite functions on a **uniform knot grid** $`t_1 < \dots < t_m`$ that is
chosen independently of the data. The unknowns are the value $`f_j`$ and derivative $`f'_j`$ at every
knot; $`h`$ is a bandwidth in the units of $`x`$ and $`L`$ a fixed length (the domain or data extent).

- **Exact, closed-form discretization.** Each observation adds a rank-one $`4 \times 4`$ block; the
  roughness penalty on each element is exactly $`\theta_j^\top K \theta_j / \Delta^3`$ with the
  Euler–Bernoulli beam stiffness matrix $`K`$.
- **Linear time.** The normal equations are symmetric positive definite with 7 diagonals: banded
  Cholesky solves them in $`O(n+m)`$, and evaluation at any $`x`$ is $`O(1)`$ (no knot search on a
  uniform grid).
- **Exact smoothing spline on the grid.** If every $`x_i`$ is a knot the result *is* the classical
  (Reinsch) cubic smoothing spline; otherwise it is its best approximation in the energy norm and
  converges to it as the grid is refined.
- **$`h`$ is a bandwidth.** The fit is exactly invariant under rescaling $`x`$ and $`h`$ together; for
  evenly spread data it is a low-pass filter with gain $`1/(1+(h\omega)^4)`$ and Silverman's
  equivalent kernel of bandwidth $`h`$.
- **HP is the discrete special case.** For equally spaced data the HP filter is the
  finite-difference version of $`J`$ with $`\lambda_{\mathrm{HP}} = (h/\Delta)^4`$, which derives the
  Ravn–Uhlig rule that $`\lambda_{\mathrm{HP}}`$ must scale with the **fourth power** of the
  sampling frequency ($`1600 \to 1600/4^4 = 6.25`$ annual, $`1600 \cdot 3^4 = 129600`$ monthly). The
  customary quarterly $`\lambda_{\mathrm{HP}} = 1600`$ is $`h \approx 1.58`$ years, a cutoff period of
  about 10 years.
- **Bayesian uncertainty in $`O(n+m)`$.** The same matrix is the posterior precision under an
  integrated-Wiener-process prior (exact at the knots). At every $`x`$ the posterior is Gaussian,
  $`f(x) \mid y \sim \mathcal{N}\bigl(\varphi(x)^\top \hat\theta,\ \varphi(x)^\top A^{-1} \varphi(x) / S\bigr)`$,
  which needs only the band of $`A^{-1}`$, computed in $`O(m)`$ by Takahashi selected inversion.
  Posterior samples, effective degrees of freedom, a noise estimate and GCV come at the same cost.

| Frequency response of the HP filter at three sampling rates vs. the continuous filter       | Irregular, heteroscedastic data with a gap: fit, 95% credible band and posterior samples  |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| ![Gain](https://raw.githubusercontent.com/gerbenvv/continuous-hp-filter/main/docs/gain.png) | ![Fit](https://raw.githubusercontent.com/gerbenvv/continuous-hp-filter/main/docs/fit.png) |

The manuscript proves these properties, verifies each numerically, and reviews the related
literature (Whittaker–Henderson graduation, smoothing splines, penalized regression splines,
state-space and Gaussian-Markov-random-field formulations, continuous-time HP filters and
finite-element smoothing) with a candid assessment of what is and is not new.

## Repository layout

| Path                                            | Contents                                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `paper/manuscript.pdf`                          | The manuscript                                                                                    |
| `paper/manuscript.tex`, `paper/references.bib`  | LaTeX source and bibliography                                                                     |
| `paper/data/`                                   | Figure data and numbers used by the manuscript (generated by `python/experiments.py`)             |
| `python/hpspline.py`                            | Python implementation (NumPy; uses SciPy's LAPACK banded routines when installed)                 |
| `python/tests/`                                 | Tests against dense least squares, SciPy's exact smoothing spline, dense inverses and Monte Carlo |
| `python/experiments.py`                         | Reproduces all figures and numbers in the manuscript                                              |
| `python/make_js_fixture.py`                     | Writes the reference results used by the JavaScript tests                                         |
| `javascript/hpspline.js`                        | JavaScript implementation (browser global `hpspline` or CommonJS/Node module, no dependencies)    |
| `javascript/index.html`, `demo.js`, `style.css` | Interactive demo                                                                                  |
| `javascript/test/`                              | JavaScript tests against the Python reference                                                     |
| `docs/`                                         | Images used in this README                                                                        |
| `.github/workflows/pages.yml`                   | Publishes the interactive demo on GitHub Pages                                                    |

## Quick start

### Python

Install with `pip install hpspline` (or `pip install hpspline[fast]` to include SciPy), or
from a clone with `pip install .`.

```python
import numpy as np

from hpspline import HPSpline, select_lambda

x = np.sort(np.random.rand(200))
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(200)

# The bandwidth `lam` is h; optionally pass `w=weights` or `sigma=known_std` to `fit`.
smoother = HPSpline(lam=0.05).fit(x, y)

# Function and derivatives, anywhere.
xs = np.linspace(0, 1, 1000)
f, df, d2f = smoother(xs), smoother(xs, nu=1), smoother(xs, nu=2)

# Posterior standard deviation (Gaussian at every x) and the 95% credible band.
sd = smoother.std(xs)
band = (f - 1.96 * sd, f + 1.96 * sd)

# Posterior samples, effective degrees of freedom, noise estimate and GCV score.
draws = smoother.evaluate(smoother.sample(5), xs)
edf, noise_variance, gcv = smoother.edf, smoother.noise_variance, smoother.gcv()

# Choose the bandwidth by GCV.
best, _, _ = select_lambda(x, y)
```

Options: `m` (number of knots) or `dt` (knot spacing; default `lam / 8`), `bounds`, and
`normalization` (`"domain"`, `"data"` or a number, the length $`L`$). Helpers: `hp_filter(y, lam_hp)`,
`lambda_from_hp(lam_hp, spacing)` and `hp_from_lambda(lam, spacing)`.

### JavaScript

```html
<script src="hpspline.js"></script>
<script>
    const smoother = new hpspline.HPSpline(0.05).fit(x, y, { w }); // Or { sigma }.

    smoother.evaluate(0.3);
    smoother.evaluate(0.3, 1);
    smoother.std(0.3);

    const theta = smoother.sample();
    smoother.evaluateTheta(theta, 0.3);

    smoother.edf();
    smoother.noiseVariance();
    smoother.gcv();

    hpspline.selectLambda(x, y).lambda;
</script>
```

In Node.js: `const hpspline = require('./javascript/hpspline.js');`. A million observations on a million
knots fit in about 0.3 s.

## Development

```bash
# Python tests (NumPy only; the SciPy-based checks run when SciPy is installed).
cd python && python -m unittest discover -s tests -p "*_test.py" -t .

# JavaScript tests and formatting/linting.
cd javascript && npm install && npm test && npm run format

# Formatting of all files (black, isort, flake8, mdformat, eslint, ...).
pre-commit run -a

# Regenerate the manuscript's figures and numbers (needs SciPy and Node.js).
cd python && python experiments.py && python make_js_fixture.py

# Build the manuscript (TeX Live with pgfplots).
cd paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

## Related work

The method combines well-established ideas; see Section 10 of the manuscript. The closest prior work
includes O'Sullivan penalized splines (O'Sullivan 1986; Wand & Ormerod 2008), finite-element
approximations of the integrated Wiener process on equally spaced knots (Zhang, Stringer, Brown &
Stafford 2024), the augmented-state second-order random walk (Rue & Held 2005; Lindgren & Rue 2008),
continuous-time HP filters (Iannaccone & Otranto 2003; McElroy & Trimbur 2007), the Hermite
value–derivative spline parametrization (Costa & Shaw 2009), and the smoothing-spline and Bayesian
foundations of Schoenberg, Reinsch, Wahba and Silverman.

## Citation

If you use this work (the method, the manuscript or the code), please cite it:

> Gerben van Veenendaal. *Cubic Smoothing Splines on a Uniform Hermite Grid: A Continuous
> Hodrick–Prescott Filter for Irregular, Weighted Data with Linear-Time Bayesian Inference.*
> Manuscript, 2026. https://github.com/gerbenvv/continuous-hp-filter

```bibtex
@unpublished{vanVeenendaal2026,
    author = {van Veenendaal, Gerben},
    title  = {Cubic Smoothing Splines on a Uniform {H}ermite Grid: A Continuous {H}odrick--{P}rescott
              Filter for Irregular, Weighted Data with Linear-Time {B}ayesian Inference},
    note   = {Manuscript},
    year   = {2026},
    url    = {https://github.com/gerbenvv/continuous-hp-filter}
}
```

GitHub's "Cite this repository" button (from [`CITATION.cff`](CITATION.cff)) gives the same
reference in other formats.

## License

[MIT](LICENSE)
