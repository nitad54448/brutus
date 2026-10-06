// regression_test.mjs -- crystallography regression tests, no browser needed.
//
// Loads js/crystallography/*.js in manifest order into a Node vm context and
// drives the real code paths with synthetic data whose answer is known:
//   - the CPU index worker (js/workers/index-worker.js) finds a cubic, a
//     tetragonal and a hexagonal cell;
//   - refineAndTestSolution pulls a perturbed orthorhombic cell back;
//   - Niggli reduction undoes a unimodular transformation;
//   - least-squares error propagation matches a Monte-Carlo estimate
//     (orthorhombic and monoclinic).
//
//   node regression_test.mjs          exit code 0 = all passed
import { readFileSync } from 'fs';
import vm from 'vm';

const ROOT = new URL('.', import.meta.url).pathname;
const read = (p) => readFileSync(ROOT + p, 'utf8');

// ---- load the crystallography scripts the way a worker does ----------------
const posted = [];
const ctx = {
    console: { log() {}, warn() {}, error: (...a) => console.error(...a) },
    performance: { now: () => Date.now() }, setTimeout, clearTimeout, URL,
    importScripts() {},                       // files are loaded explicitly below
    postMessage: (m) => posted.push(m),
    location: { search: '', href: 'file:///js/workers/index-worker.js' },
};
ctx.self = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);
const manifest = read('js/crystallography/manifest.js');
vm.runInContext(manifest, ctx, { filename: 'manifest.js' });
const files = [...manifest.matchAll(/^\s*'([^']+\.js)',\s*$/gm)].map(m => 'js/crystallography/' + m[1]);
for (const f of files) vm.runInContext(read(f), ctx, { filename: f });
const expose = (names) => vm.runInContext(`({ ${names.join(', ')} })`, ctx);
const W = expose(['generateHKL_for_analysis', 'getSortedPeaks', 'refineAndTestSolution', 'solveLeastSquares',
                  'getLSDesignRow', 'extractCellFromFit', 'propagateErrors', 'niggliReduceFromCell',
                  'getSolutionKey']);

let failures = 0;
const check = (name, ok, detail) => {
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  -- ' + detail : ''}`);
    if (!ok) failures++;
};

// ---- synthetic data ----------------------------------------------------------
const LAMBDA = 1.54056;
const tthOf = (q) => 2 * Math.asin(LAMBDA * Math.sqrt(q) / 2) * 180 / Math.PI;
// Distinct reflection positions of `cell` up to tthMax, lowest first.
function peaksFor(cell, tthMax = 70, maxPeaks = 24) {
    const d_min = LAMBDA / (2 * Math.sin(tthMax * Math.PI / 360));
    const list = W.generateHKL_for_analysis(cell, LAMBDA, tthMax) || [];
    const tths = [...new Set(list.map(h => (h.tth ?? tthOf(h.q)).toFixed(4)))].map(Number).sort((a, b) => a - b);
    return tths.filter(t => t > 5 && t < tthMax && LAMBDA / (2 * Math.sin(t * Math.PI / 360)) > d_min * 0.999)
               .slice(0, maxPeaks).map(t => ({ tth: t, intensity: 100 }));
}
const baseData = (peaks) => ({ peaks, wavelength: LAMBDA, tth_error: 0.03, max_volume: 1500,
    impurity_peaks: 0, refineZero: false, fom_threshold: 1.5, max_solutions: 1000,
    min_axis: 2.0, max_axis: 50.0, min_volume: 20.0 });

// ---- 1. CPU index worker -----------------------------------------------------
vm.runInContext(read('js/workers/index-worker.js'), ctx, { filename: 'index-worker.js' });
// max_volume is kept small so each search takes seconds, not minutes.
const runWorker = (system, peaks, maxVolume = 250) => {
    posted.length = 0;
    ctx.self.onmessage({ data: { ...baseData(peaks), max_volume: maxVolume, systemToSearch: system, allowedSystems: [system] } });
    return posted.filter(m => m && m.type === 'solution').map(m => m.payload);
};
const best = (sols) => sols.slice().sort((a, b) => (b.m20 || 0) - (a.m20 || 0))[0];
const near = (x, y, tol) => Math.abs(x - y) <= tol;

{
    const truth = { a: 5.4309, b: 5.4309, c: 5.4309, alpha: 90, beta: 90, gamma: 90, system: 'cubic' };
    const s = best(runWorker('cubic', peaksFor(truth, 70, 12)));
    check('CPU worker: cubic a = 5.4309', !!s && near(s.a, truth.a, 0.003), s ? `a=${s.a.toFixed(4)} M20=${s.m20.toFixed(1)}` : 'no solution');
}
{
    const truth = { a: 4.5937, b: 4.5937, c: 2.9587, alpha: 90, beta: 90, gamma: 90, system: 'tetragonal' };
    const s = best(runWorker('tetragonal', peaksFor(truth, 70, 14)));
    check('CPU worker: tetragonal a = 4.5937, c = 2.9587', !!s && near(s.a, truth.a, 0.003) && near(s.c, truth.c, 0.003),
          s ? `a=${s.a.toFixed(4)} c=${s.c.toFixed(4)} M20=${s.m20.toFixed(1)}` : 'no solution');
}
{
    const truth = { a: 3.2498, b: 3.2498, c: 5.2066, alpha: 90, beta: 90, gamma: 120, system: 'hexagonal' };
    const s = best(runWorker('hexagonal', peaksFor(truth, 70, 14), 120));
    check('CPU worker: hexagonal a = 3.2498, c = 5.2066', !!s && near(s.a, truth.a, 0.003) && near(s.c, truth.c, 0.003),
          s ? `a=${s.a.toFixed(4)} c=${s.c.toFixed(4)} M20=${s.m20.toFixed(1)}` : 'no solution');
}

// ---- 2. refinement of a perturbed orthorhombic cell --------------------------
{
    const truth = { a: 8.4782, b: 5.3980, c: 6.9580, alpha: 90, beta: 90, gamma: 90, system: 'orthorhombic' };
    const peaks = peaksFor(truth, 60, 30);
    const data = baseData(peaks);
    const sorted = W.getSortedPeaks(peaks, LAMBDA);
    const d_min = LAMBDA / (2 * Math.sin(Math.max(...peaks.map(p => p.tth)) * Math.PI / 360));
    const state = { ...sorted, N_FOR_M20: Math.min(20, peaks.length), min_m20: 2.0, q_max: 1 / (d_min * d_min),
                    d_min, foundSolutions: [], foundSolutionMap: new Map(), diag: {} };
    const out = [];
    W.refineAndTestSolution({ ...truth, a: 8.4740, b: 5.4010, c: 6.9545 }, data, state, (m) => out.push(m));
    const s = out.find(m => m.type === 'solution');
    const p = s && s.payload;
    check('refineAndTestSolution: orthorhombic cell recovered', !!p && near(p.a, truth.a, 0.002) && near(p.b, truth.b, 0.002) && near(p.c, truth.c, 0.002),
          p ? `a=${p.a.toFixed(4)} b=${p.b.toFixed(4)} c=${p.c.toFixed(4)} M20=${p.m20.toFixed(1)}` : 'rejected');
}

// ---- 3. Niggli reduction -----------------------------------------------------
{
    // A cell built from the PbSO4 basis with b' = a + b, c' = a + b + c.
    const base = { a: 8.4782, b: 5.3980, c: 6.9580, alpha: 90, beta: 90, gamma: 90 };
    const ax = [base.a, 0, 0], bx = [0, base.b, 0], cx = [0, 0, base.c];
    const add = (...v) => v.reduce((s, x) => s.map((y, i) => y + x[i]), [0, 0, 0]);
    const len = (v) => Math.hypot(...v);
    const ang = (u, v) => Math.acos(u.reduce((s, x, i) => s + x * v[i], 0) / (len(u) * len(v))) * 180 / Math.PI;
    const a2 = ax, b2 = add(ax, bx), c2 = add(ax, bx, cx);
    const skew = { a: len(a2), b: len(b2), c: len(c2), alpha: ang(b2, c2), beta: ang(a2, c2), gamma: ang(a2, b2) };
    const r = W.niggliReduceFromCell(skew);
    const cell = r && (r.cell || r);
    const edges = cell ? [cell.a, cell.b, cell.c].sort((x, y) => x - y) : [];
    check('Niggli reduction undoes a unimodular transform', edges.length === 3 &&
          near(edges[0], 5.3980, 1e-3) && near(edges[1], 6.9580, 1e-3) && near(edges[2], 8.4782, 1e-3),
          cell ? `edges ${edges.map(x => x.toFixed(4)).join(', ')}` : 'no result');
}

// ---- 4. error propagation vs Monte Carlo -------------------------------------
function esdCheck(system, cell, coeffs, hkls, names) {
    const M = hkls.map(h => W.getLSDesignRow(h, system));
    const qTrue = M.map(r => r.reduce((s, x, i) => s + x * coeffs[i], 0));
    let seed = 12345;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const gauss = () => { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
    const sigma = 2e-5;
    const fit = () => W.solveLeastSquares(M, qTrue.map(v => v + sigma * gauss()), null);
    // Average the propagated esd over many independent fits and compare it with
    // the scatter of the fitted values themselves. (A single fit's esd is itself
    // noisy by ~1/sqrt(2*dof); the average is not.)
    const samples = names.map(() => []), esds = names.map(() => []);
    for (let t = 0; t < 2000; t++) {
        const f = fit();
        const c = W.extractCellFromFit(f.solution, system);
        const e = W.propagateErrors(system, f, c);
        names.forEach(([field, key], i) => { samples[i].push(c[field]); esds[i].push(e[key]); });
    }
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
    names.forEach(([field], i) => {
        const mc = sd(samples[i]), e = mean(esds[i]);
        check(`${system} esd ${field} matches Monte Carlo`, Math.abs(e / mc - 1) < 0.08,
              `propagated ${e.toExponential(2)}, Monte Carlo ${mc.toExponential(2)}`);
    });
}
{
    const hkls = [];
    for (let h = 0; h <= 4; h++) for (let k = 0; k <= 4; k++) for (let l = 0; l <= 4; l++) if (h || k || l) hkls.push([h, k, l]);
    const cell = { a: 8.4782, b: 5.3980, c: 6.9580 };
    esdCheck('orthorhombic', cell, [1 / cell.a ** 2, 1 / cell.b ** 2, 1 / cell.c ** 2], hkls.slice(0, 40),
             [['a', 's_a'], ['b', 's_b'], ['c', 's_c']]);
}
{
    const cell = { a: 9.1, b: 6.2, c: 11.4, beta: 117 };
    const sb = Math.sin(cell.beta * Math.PI / 180), cb = Math.cos(cell.beta * Math.PI / 180);
    const coeffs = [1 / (cell.a ** 2 * sb ** 2), 1 / cell.b ** 2, 1 / (cell.c ** 2 * sb ** 2), -2 * cb / (cell.a * cell.c * sb ** 2)];
    const hkls = [];
    for (let h = -3; h <= 3; h++) for (let k = 0; k <= 3; k++) for (let l = -3; l <= 3; l++) if (h || k || l) hkls.push([h, k, l]);
    esdCheck('monoclinic', cell, coeffs, hkls.slice(0, 40), [['a', 's_a'], ['c', 's_c'], ['beta', 's_beta']]);
}

console.log(failures ? `\n${failures} test(s) FAILED` : '\nAll regression tests passed.');
process.exit(failures ? 1 : 0);
