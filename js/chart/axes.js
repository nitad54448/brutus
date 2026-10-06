// js/chart/axes.js
// Plot axis modes (2θ / θ / d / Q, linear / sqrt / log).
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// ===================== AXIS SYSTEM =====================================
// Everything the program computes with stays in its natural units: peak
// positions in 2-theta degrees, intensities in raw counts. The plot is a
// *view* on top of that. xF/yF map an internal quantity onto whatever the
// user selected; xInv maps a plotted abscissa back to 2-theta for
// hit-testing. Chart.js is always handed a plain linear scale, even in log
// mode, so pixel math, floating-bar markers and getValueForPixel behave
// identically in every combination - only the tick formatter converts back
// to real units. Nothing downstream of the plot ever sees a transformed
// value, so indexing, refinement and the report are untouched by the view.
const LOG_FLOOR = 0.1;                 // counts: keeps log10 finite at zero
const DEG_PER_RAD = 180 / Math.PI;
const getLambda = () => getWavelength();
const AXIS_X = {
    tth: {
        label: '2\u03B8 (degrees)', reverse: false, digits: 3,
        f: (t) => t,
        inv: (v) => v
    },
    theta: {
        label: '\u03B8 (degrees)', reverse: false, digits: 3,
        f: (t) => t / 2,
        inv: (v) => v * 2
    },
    d: {
        // Reversed on purpose: large d belongs on the left so the pattern
        // keeps the same left-to-right shape it has in 2-theta.
        label: 'd-spacing (\u00C5)', reverse: true, digits: 4,
        f: (t) => getLambda() / (2 * Math.sin(Math.max(t, 1e-4) * Math.PI / 360)),
        inv: (v) => {
            const s = getLambda() / (2 * Math.max(v, 1e-9));
            return s >= 1 ? 180 : 2 * Math.asin(s) * DEG_PER_RAD;
        }
    },
    q: {
        label: 'Q = 4\u03C0\u00B7sin\u03B8/\u03BB (\u00C5\u207B\u00B9)', reverse: false, digits: 4,
        f: (t) => 4 * Math.PI * Math.sin(Math.max(t, 1e-4) * Math.PI / 360) / getLambda(),
        inv: (v) => {
            const s = Math.max(v, 0) * getLambda() / (4 * Math.PI);
            return s >= 1 ? 180 : 2 * Math.asin(s) * DEG_PER_RAD;
        }
    }
};
const AXIS_Y = {
    linear: { label: 'Intensity (a.u.)',        f: (i) => i,                          inv: (v) => v },
    sqrt:   { label: '\u221AIntensity (a.u.)',  f: (i) => Math.sqrt(Math.max(0, i)),  inv: (v) => v * v },
    log:    { label: 'Intensity (a.u., log)',   f: (i) => Math.log10(Math.max(i, LOG_FLOOR)), inv: (v) => Math.pow(10, v) }
};
let xAxisMode = 'tth';    // 2-theta by default
let yAxisMode = 'sqrt';   // sqrt(I) by default: shows weak lines without hiding strong ones
const xAx = () => AXIS_X[xAxisMode] || AXIS_X.tth;
const yAx = () => AXIS_Y[yAxisMode] || AXIS_Y.sqrt;
const xF   = (t) => xAx().f(t);
const xInv = (v) => xAx().inv(v);
const yF   = (i) => yAx().f(i);
// The 2-theta interval currently on screen, ordered low-to-high whatever
// direction the axis runs in.
const visibleTthRange = () => {
    if (!xrdChart) return [-Infinity, Infinity];
    const a = xInv(xrdChart.scales.x.min);
    const b = xInv(xrdChart.scales.x.max);
    return a <= b ? [a, b] : [b, a];
};
// Points carry their untransformed tth and I so tooltips and hit-testing
// never have to invert anything.
const buildExperimentalPoints = () => workingExperimentalData.tth.map((t, i) => {
    const I = Math.max(0, workingExperimentalData.intensity[i]);
    return { x: xF(t), y: yF(I), tth: t, I };
});
// The ONLY supported way to push the measured pattern into the chart.
// Writing datasets[0].data directly is what let raw counts land on a
// transformed axis: the tick formatter then inverted values that had never
// been transformed, so a 1e4 count was labelled 1e8 on the sqrt scale, and
// the trace was drawn with the wrong shape into the bargain. Everything
// that changes the working data - file load, wavelength, Ka2 stripping,
// deconvolution - goes through here instead of touching the dataset.
const setExperimentalTrace = (rescaleY = false) => {
    if (!xrdChart || !workingExperimentalData || !workingExperimentalData.tth.length) return;
    xrdChart.data.datasets[0].data = buildExperimentalPoints();
    if (rescaleY) {
        const yb = yBoundsFor(workingExperimentalData.intensity);
        xrdChart.options.scales.y.min = yb.min;
        xrdChart.options.scales.y.max = yb.max;
    }
    xrdChart.update('none');
};
// Y bounds expressed in transformed space, from raw intensities.
const yBoundsFor = (intensities) => {
    const raw = maxOfArray(intensities) || 1000;
    if (yAxisMode === 'log') {
        const top = Math.log10(Math.max(raw, LOG_FLOOR));
        const bot = Math.log10(LOG_FLOOR);
        const span = Math.max(top - bot, 1);
        return { min: bot - span * 0.02, max: top + span * 0.05 };
    }
    const top = yF(raw);
    return { min: -top * 0.05, max: top * 1.1 };
};
