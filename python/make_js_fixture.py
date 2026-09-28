"""Writes reference results of `hpspline.py` to `javascript/test/fixture.json` for the JS tests."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from hpspline import HPSpline, hp_filter, select_lambda

LOGGER: logging.Logger = logging.getLogger(__name__)

# The fixture read by the JavaScript tests.
FIXTURE_FILE_PATH: Path = (
    Path(__file__).resolve().parent.parent / "javascript" / "test" / "fixture.json"
)

# Smoother options covered by the fixture, each with the bandwidth under `lam`.
CASES: tuple[dict[str, Any], ...] = (
    dict(lam=0.5),
    dict(lam=0.2, m=300),
    dict(lam=1.0, dt=0.37, bounds=(-1, None)),
    dict(lam=0.5, normalization="data", bounds=(-2, 12)),
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rng = np.random.default_rng(7)

    x = np.sort(rng.uniform(0, 10, 120))
    y = np.sin(x) + 0.2 * x + 0.15 * rng.standard_normal(x.size)
    w = rng.uniform(0.5, 2.0, x.size)
    grid = np.linspace(-0.5, 10.5, 45)

    cases = []

    for case in CASES:
        options = dict(case)
        lam = options.pop("lam")
        smoother = HPSpline(lam, **options).fit(x, y, w)

        cases.append(
            dict(
                lam=lam,
                options=options,
                m=smoother.m_,
                dt=smoother.dt_,
                theta=smoother.theta_.tolist(),
                value=smoother(grid).tolist(),
                slope=smoother(grid, 1).tolist(),
                curvature=smoother(grid, 2).tolist(),
                std=smoother.std(grid).tolist(),
                edf=smoother.edf,
                gcv=smoother.gcv(),
                loss=smoother.loss(),
            )
        )

    sigma_smoother = HPSpline(0.5).fit(x, y, sigma=0.15)
    best_lambda, _, _ = select_lambda(x, y)

    fixture = dict(
        x=x.tolist(),
        y=y.tolist(),
        w=w.tolist(),
        grid=grid.tolist(),
        cases=cases,
        sigma_std=sigma_smoother.std(grid).tolist(),
        best_lambda=best_lambda,
        hp=hp_filter(y, 1600.0).tolist(),
    )

    with FIXTURE_FILE_PATH.open("w") as file:
        json.dump(fixture, file, indent=4)
        file.write("\n")

    LOGGER.info("Wrote %s (best lambda %.6g).", FIXTURE_FILE_PATH, best_lambda)


if __name__ == "__main__":
    main()
