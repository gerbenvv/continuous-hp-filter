// Cross-checks hpspline.js against the Python reference (fixture.json). Run with `npm test`.
'use strict';

const assert = require('assert');
const path = require('path');

const hpspline = require(path.join(__dirname, '..', 'hpspline.js'));
const fixture = require('./fixture.json');

let failures = 0;

function test(name, body) {
    try {
        body();

        console.log(`PASS ${name}`);
    } catch (error) {
        failures += 1;

        console.log(`FAIL ${name}: ${error.message}`);
    }
}

function getMaxRelativeError(actual, expected) {
    let scale = 0;
    let error = 0;

    for (let i = 0; i < expected.length; ++i) {
        scale = Math.max(scale, Math.abs(expected[i]));
        error = Math.max(error, Math.abs(actual[i] - expected[i]));
    }

    return error / Math.max(scale, 1e-300);
}

function assertClose(actual, expected, tolerance, what) {
    const error = getMaxRelativeError(Array.from(actual), expected);

    assert(error < tolerance, `${what}: relative error ${error.toExponential(2)}`);
}

const { x, y, w, grid } = fixture;

fixture.cases.forEach((reference, index) => {
    test(`case ${index} (${JSON.stringify(reference.options)})`, () => {
        const smoother = new hpspline.HPSpline(reference.lam, reference.options).fit(x, y, { w });

        assert.strictEqual(smoother.m, reference.m);
        assert(Math.abs(smoother.dt - reference.dt) < 1e-12);

        const values = grid.map((v) => smoother.evaluate(v));
        const slopes = grid.map((v) => smoother.evaluate(v, 1));
        const curvatures = grid.map((v) => smoother.evaluate(v, 2));
        const deviations = grid.map((v) => smoother.std(v));
        const statistics = [smoother.edf(), smoother.gcv(), smoother.loss()];

        assertClose(smoother.theta, reference.theta, 1e-9, 'theta');
        assertClose(values, reference.value, 1e-9, 'value');
        assertClose(slopes, reference.slope, 1e-8, 'slope');
        assertClose(curvatures, reference.curvature, 1e-7, 'curvature');
        assertClose(deviations, reference.std, 1e-7, 'std');
        assertClose(statistics, [reference.edf, reference.gcv, reference.loss], 1e-9, 'statistics');
    });
});

test('known sigma', () => {
    const smoother = new hpspline.HPSpline(0.5).fit(x, y, { sigma: 0.15 });
    const deviations = grid.map((v) => smoother.std(v));

    assertClose(deviations, fixture.sigma_std, 1e-7, 'std');
});

test('GCV lambda selection', () => {
    const { lambda } = hpspline.selectLambda(x, y);

    assert(
        Math.abs(lambda / fixture.best_lambda - 1) < 1e-3,
        `${lambda} vs ${fixture.best_lambda}`
    );
});

test('HP filter', () => {
    assertClose(hpspline.hpFilter(y, 1600), fixture.hp, 1e-10, 'trend');
});

test('posterior samples have the right spread', () => {
    const smoother = new hpspline.HPSpline(0.5).fit(x, y, { w });
    const random = hpspline.seededRandom(42);
    const probe = 5.0;
    const count = 20000;

    let sum = 0;
    let sumSquares = 0;

    for (let i = 0; i < count; ++i) {
        const value = smoother.evaluateTheta(smoother.sample(random), probe);

        sum += value;
        sumSquares += value * value;
    }

    const mean = sum / count;
    const std = Math.sqrt(sumSquares / count - mean * mean);

    assert(Math.abs(std / smoother.std(probe) - 1) < 0.03, `std ${std} vs ${smoother.std(probe)}`);
    assert(
        Math.abs(mean - smoother.evaluate(probe)) < (5 * smoother.std(probe)) / Math.sqrt(count)
    );
});

test('scale invariance', () => {
    const reference = new hpspline.HPSpline(0.5, { m: 200 }).fit(x, y);
    const scaled = new hpspline.HPSpline(0.5 * 1000, { m: 200 }).fit(
        x.map((v) => v * 1000),
        y
    );

    assertClose(scaled.knots().h, reference.knots().h, 1e-8, 'heights');
});

test('speed: 100k points, 100k knots', () => {
    const n = 100000;
    const random = hpspline.seededRandom(1);
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);

    for (let i = 0; i < n; ++i) {
        xs[i] = random();
        ys[i] = Math.sin(6 * xs[i]) + 0.1 * (random() - 0.5);
    }

    const start = Date.now();
    const smoother = new hpspline.HPSpline(0.001, { m: 100000 }).fit(xs, ys);
    const fitTime = Date.now() - start;

    smoother.std(0.5);
    const totalTime = Date.now() - start;

    console.log(`     fit ${fitTime} ms, fit + uncertainty ${totalTime} ms`);

    assert(totalTime < 5000);
});

process.exit(failures ? 1 : 0);
