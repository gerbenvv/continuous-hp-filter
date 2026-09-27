// Interactive demo for chp.js. Data live in the unit square; the knot domain is [0, 1].
(function () {
    'use strict';

    // Fraction of the canvas left free around the unit square.
    const PADDING = 0.06;

    const canvas = document.getElementById('plot');
    const context = canvas.getContext('2d');

    const controls = {
        lambda: document.getElementById('lambda'),
        lambdaValue: document.getElementById('lambda-value'),
        knots: document.getElementById('knots'),
        knotsValue: document.getElementById('knots-value'),
        band: document.getElementById('show-band'),
        samples: document.getElementById('show-samples'),
        knotMarks: document.getElementById('show-knots'),
        hp: document.getElementById('show-hp'),
        stats: document.getElementById('stats'),
    };

    const points = [];
    const random = CHP.seededRandom(12345);

    let dragging = -1;
    let sampleSeed = 1;

    /**
     * Returns the current theme colors from the CSS custom properties.
     */
    function getColors() {
        const style = getComputedStyle(document.documentElement);
        const get = (name) => style.getPropertyValue(name).trim();

        return {
            accent: get('--accent'),
            soft: get('--accent-soft'),
            sample: get('--sample'),
            point: get('--point'),
            knot: get('--knot'),
            hp: get('--hp'),
            grid: get('--grid'),
            muted: get('--muted'),
        };
    }

    /**
     * Converts data coordinates (unit square) to canvas pixels.
     */
    function toPixel(x, y) {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        const px = (PADDING + (1 - 2 * PADDING) * x) * width;
        const py = (1 - PADDING - (1 - 2 * PADDING) * y) * height;

        return [px, py];
    }

    /**
     * Converts canvas pixels to data coordinates (unit square).
     */
    function toData(px, py) {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        const x = (px / width - PADDING) / (1 - 2 * PADDING);
        const y = (1 - PADDING - py / height) / (1 - 2 * PADDING);

        return [x, y];
    }

    function getLambda() {
        return Math.pow(10, parseFloat(controls.lambda.value));
    }

    function getSmootherOptions() {
        const knots = controls.knots.value;
        const options = { bounds: [0, 1] };

        if (knots !== 'auto') {
            options.m = parseInt(knots, 10);
        } else {
            options.maxKnots = 20000;
        }

        return options;
    }

    function createRandomData() {
        points.length = 0;

        for (let i = 0; i < 40; ++i) {
            let x = random();

            // Leave a gap to show the band widening.
            if (x > 0.55 && x < 0.72) {
                x = random() * 0.5;
            }

            const noise =
                Math.sqrt(-2 * Math.log(random() || 1e-9)) * Math.cos(2 * Math.PI * random());
            const y =
                0.5 +
                0.28 * Math.sin(2 * Math.PI * x * 1.3) +
                0.08 * Math.cos(9 * x) +
                0.05 * noise;

            points.push([x, y]);
        }
    }

    function resize() {
        const ratio = window.devicePixelRatio || 1;

        canvas.width = Math.round(canvas.clientWidth * ratio);
        canvas.height = Math.round(canvas.clientHeight * ratio);
        context.setTransform(ratio, 0, 0, ratio, 0, 0);

        draw();
    }

    function drawPath(xs, ys, color, width) {
        context.strokeStyle = color;
        context.lineWidth = width;
        context.beginPath();

        xs.forEach((x, i) => {
            const [px, py] = toPixel(x, ys[i]);

            if (i === 0) {
                context.moveTo(px, py);
            } else {
                context.lineTo(px, py);
            }
        });

        context.stroke();
    }

    function drawLine(x0, y0, x1, y1) {
        const [px0, py0] = toPixel(x0, y0);
        const [px1, py1] = toPixel(x1, y1);

        context.beginPath();
        context.moveTo(px0, py0);
        context.lineTo(px1, py1);
        context.stroke();
    }

    function setStats(entries) {
        controls.stats.innerHTML = entries
            .map(([name, value]) => `<dt>${name}</dt><dd>${value}</dd>`)
            .join('');
    }

    function drawBand(grid, mean, deviation, colors) {
        context.fillStyle = colors.soft;
        context.beginPath();

        grid.forEach((x, i) => {
            const [px, py] = toPixel(x, mean[i] + deviation[i]);

            if (i === 0) {
                context.moveTo(px, py);
            } else {
                context.lineTo(px, py);
            }
        });

        for (let i = grid.length - 1; i >= 0; --i) {
            const [px, py] = toPixel(grid[i], mean[i] - deviation[i]);
            context.lineTo(px, py);
        }

        context.closePath();
        context.fill();
    }

    function drawHpFilter(xs, ys, colors) {
        // Classic HP on the points in x order, ignoring their spacing, with the equivalent
        // lambda_HP = (lambda / mean spacing)^4.
        const order = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
        const sortedX = order.map((i) => xs[i]);
        const spacing = (sortedX[sortedX.length - 1] - sortedX[0]) / (sortedX.length - 1);

        const trend = CHP.hpFilter(
            order.map((i) => ys[i]),
            CHP.hpFromLambda(getLambda(), spacing)
        );

        drawPath(sortedX, Array.from(trend), colors.hp, 1.5);
    }

    function draw() {
        const colors = getColors();
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        context.clearRect(0, 0, width, height);

        // Grid.
        context.strokeStyle = colors.grid;
        context.lineWidth = 1;

        for (let i = 0; i <= 10; ++i) {
            drawLine(i / 10, 0, i / 10, 1);
            drawLine(0, i / 10, 1, i / 10);
        }

        controls.lambdaValue.textContent = getLambda().toPrecision(3);

        const xs = points.map((p) => p[0]);
        const ys = points.map((p) => p[1]);

        if (new Set(xs).size < 2) {
            controls.knotsValue.textContent = '';
            setStats([
                ['points n', points.length],
                ['', 'add at least two distinct x'],
            ]);

            drawPoints(colors);

            return;
        }

        const start = performance.now();
        let smoother;

        try {
            smoother = new CHP.CHPSmoother(getLambda(), getSmootherOptions()).fit(xs, ys);
        } catch (error) {
            setStats([['error', error.message]]);
            drawPoints(colors);

            return;
        }

        const fitTime = performance.now() - start;

        const count = Math.max(200, Math.round(width));
        const grid = Array.from(
            { length: count },
            (_, i) => -PADDING + ((1 + 2 * PADDING) * i) / (count - 1)
        );
        const mean = grid.map((x) => smoother.evaluate(x));

        if (controls.band.checked) {
            const deviation = grid.map((x) => 1.96 * smoother.std(x));
            drawBand(grid, mean, deviation, colors);
        }

        if (controls.samples.checked) {
            const sampleRandom = CHP.seededRandom(sampleSeed);

            for (let i = 0; i < 6; ++i) {
                const theta = smoother.sample(sampleRandom);
                const values = grid.map((x) => smoother.evaluateTheta(theta, x));

                drawPath(grid, values, colors.sample, 1);
            }
        }

        if (controls.hp.checked && points.length >= 3) {
            drawHpFilter(xs, ys, colors);
        }

        drawPath(grid, mean, colors.accent, 2.5);

        if (controls.knotMarks.checked && smoother.m <= 2000) {
            const { t, h } = smoother.knots();
            context.fillStyle = colors.knot;

            t.forEach((x, i) => {
                const [px, py] = toPixel(x, h[i]);
                context.fillRect(px - 1.5, py - 1.5, 3, 3);
            });
        }

        const hpLambda = CHP.hpFromLambda(getLambda(), 1 / Math.max(points.length - 1, 1));

        controls.knotsValue.textContent = `m = ${smoother.m}`;
        setStats([
            ['points n', points.length],
            ['knots m', smoother.m],
            ['knot spacing', smoother.dt.toPrecision(3)],
            ['effective d.o.f.', smoother.edf().toFixed(2)],
            ['noise σ (est.)', Math.sqrt(smoother.noiseVariance()).toPrecision(3)],
            ['GCV', smoother.gcv().toExponential(3)],
            ['λ_HP at mean spacing', points.length > 1 ? hpLambda.toPrecision(3) : '–'],
            ['fit time', `${fitTime.toFixed(1)} ms`],
        ]);

        drawPoints(colors);
    }

    function drawPoints(colors) {
        context.fillStyle = colors.point;

        for (const [x, y] of points) {
            const [px, py] = toPixel(x, y);

            context.beginPath();
            context.arc(px, py, 3.5, 0, 2 * Math.PI);
            context.fill();
        }
    }

    /**
     * Returns the index of the point within 10 pixels of (px, py), or -1.
     */
    function findNearest(px, py) {
        let best = -1;
        let bestDistance = 10;

        points.forEach(([x, y], i) => {
            const [qx, qy] = toPixel(x, y);
            const distance = Math.hypot(px - qx, py - qy);

            if (distance < bestDistance) {
                best = i;
                bestDistance = distance;
            }
        });

        return best;
    }

    function getEventPosition(event) {
        const box = canvas.getBoundingClientRect();

        return [event.clientX - box.left, event.clientY - box.top];
    }

    canvas.addEventListener('contextmenu', (event) => event.preventDefault());

    canvas.addEventListener('pointerdown', (event) => {
        const [px, py] = getEventPosition(event);
        const hit = findNearest(px, py);

        // Right-click or shift-click removes a point.
        if (event.button === 2 || event.shiftKey) {
            if (hit >= 0) {
                points.splice(hit, 1);
                draw();
            }

            return;
        }

        if (hit >= 0) {
            dragging = hit;
        } else {
            const [x, y] = toData(px, py);

            if (x >= 0 && x <= 1) {
                points.push([x, y]);
                dragging = points.length - 1;
            }
        }

        canvas.setPointerCapture(event.pointerId);
        draw();
    });

    canvas.addEventListener('pointermove', (event) => {
        if (dragging < 0) {
            return;
        }

        const [px, py] = getEventPosition(event);
        const [x, y] = toData(px, py);

        points[dragging] = [Math.min(Math.max(x, 0), 1), y];
        draw();
    });

    canvas.addEventListener('pointerup', () => {
        dragging = -1;
    });

    controls.lambda.addEventListener('input', draw);
    controls.knots.addEventListener('change', draw);

    for (const box of [controls.band, controls.samples, controls.knotMarks, controls.hp]) {
        box.addEventListener('change', () => {
            sampleSeed += 1;
            draw();
        });
    }

    document.getElementById('auto').addEventListener('click', () => {
        if (new Set(points.map((p) => p[0])).size < 4) {
            return;
        }

        const { lambda: best } = CHP.selectLambda(
            points.map((p) => p[0]),
            points.map((p) => p[1]),
            { smoother: getSmootherOptions() }
        );

        controls.lambda.value = Math.min(
            Math.max(Math.log10(best), controls.lambda.min),
            controls.lambda.max
        );

        draw();
    });

    document.getElementById('random').addEventListener('click', () => {
        createRandomData();
        draw();
    });

    document.getElementById('clear').addEventListener('click', () => {
        points.length = 0;
        draw();
    });

    window.addEventListener('resize', resize);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

    createRandomData();
    resize();
})();
