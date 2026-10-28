// regression_test.mjs -- crystallography regression tests, no browser needed.
//
// Loads js/crystallography/*.js in manifest order into a Node vm context and
// drives the real code paths with synthetic data whose answer is known:
//   - the CPU index worker (js/workers/index-worker.js) finds a cubic, a
//     tetragonal and a hexagonal cell;
//   - refineAndTestSolution pulls a perturbed orthorhombic cell back;
//   - Niggli reduction undoes a unimodular transformation;
//   - least-squares error propagation matches a Monte-Carlo estimate
//     (orthorhombic and monoclinic);
//   - Ka1 lines that carry their own wavelength (p.lambda) are indexed with
//     it: cell and zero;
//   - the final sieve keeps the most symmetric description of a lattice, or
//     both, marked as a pair, when a less symmetric one fits clearly better
//     (js/indexing/solutions.js).
//
//   node regression_test.mjs          exit code 0 = all passed
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import vm from 'vm';

// The app root: this script's directory, or its parent when the script lives
// in tests/ (it used to assume the former and failed with ENOENT on js/...).
const HERE = fileURLToPath(new URL('.', import.meta.url));
const ROOT = existsSync(join(HERE, 'js/crystallography/manifest.js')) ? HERE : join(HERE, '..');
const read = (p) => readFileSync(join(ROOT, p), 'utf8');

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

// ---- 5. Ka1 lines carry their own wavelength (p.lambda) -----------------------
// With a K-alpha AVERAGE preset and no stripping, a peak whose Ka2 companion
// was identified is a pure Ka1 line and carries p.lambda = lambda(Ka1). The
// fit has to use it: those lines used to be indexed with the average, which
// left the cell 0.08 % too large (Cu) with a normal-looking M(20).
{
    const KA1 = 1.54056, KAVG = 1.54184, Z = 0.02;
    const truth = { a: 5.4309, b: 5.4309, c: 5.4309, alpha: 90, beta: 90, gamma: 90, system: 'cubic' };
    // Below 60 deg the doublet is unresolved and the peak sits at the average
    // position; above, it is resolved and the picked peak is the Ka1 line.
    // Every measured position carries the zero error Z.
    const peaks = W.generateHKL_for_analysis(truth, KA1, 140).map(L => {
        const resolved = L.tth > 60;
        const t = 2 * Math.asin((resolved ? KA1 : KAVG) / (2 * L.d)) * 180 / Math.PI + Z;
        return resolved ? { tth: t, height: 100, lambda: KA1 } : { tth: t, height: 100 };
    }).filter(p => p.tth < 140);
    // refineAndTestSolution polishes a trial cell (pairing happens against the
    // TRIAL cell's lines), so each fit starts from the cell a search would hand it.
    const fitWith = (pk, a0) => {
        const sorted = W.getSortedPeaks(pk, KAVG);
        const d_min = KAVG / (2 * Math.sin(sorted.tth_max_deg * Math.PI / 360));
        const state = { ...sorted, N_FOR_M20: Math.min(20, pk.length), min_m20: 2.0, q_max: 1 / (d_min * d_min),
                        d_min, foundSolutions: [], foundSolutionMap: new Map(), diag: {} };
        const out = [];
        W.refineAndTestSolution({ ...truth, a: a0 }, { ...baseData(pk), wavelength: KAVG, refineZero: true }, state, (m) => out.push(m));
        const s = out.find(m => m.type === 'solution');
        return s && s.payload;
    };
    const nKa1 = peaks.filter(p => p.lambda).length;
    const p = fitWith(peaks, truth.a);
    check(`Ka1 lines indexed with lambda(Ka1): a and zero (${nKa1} of ${peaks.length} peaks are Ka1)`,
          !!p && near(p.a, truth.a, 2e-4) && near(p.zero_correction, Z, 3e-3),
          p ? `a=${p.a.toFixed(5)} Z=${p.zero_correction.toFixed(4)} M20=${p.m20.toFixed(1)}` : 'rejected');
    // What the old code did: the same peaks without p.lambda, refined from the
    // cell the search then found (the Ka1 lines read at the average wavelength).
    // The resolved and unresolved lines then disagree: no cell fits both, so
    // the refit either fails or keeps a cell that is too large.
    const old = fitWith(peaks.map(({ lambda, ...rest }) => rest), truth.a * KAVG / KA1);
    check('  the same peaks without p.lambda: rejected, or a cell too large (the old behaviour)',
          !old || old.a - truth.a > 1e-3, old ? `a=${old.a.toFixed(5)} M20=${old.m20.toFixed(1)}` : 'rejected (M20 = 0)');
}

// ---- 6. Final sieve: the more symmetric description of a lattice is kept ------
// applyFinalSieve lives in js/indexing/solutions.js (main thread, next to DOM
// code), so the sieve block is taken from the real source, as check_pipeline
// does for run.js, and run against the crystallography loaded above.
{
    const src = read('js/indexing/solutions.js');
    const i0 = src.indexOf('const SIEVE_SYMMETRY_RANK'), i1 = src.indexOf('const sortSolutions');
    if (i0 < 0 || i1 < i0) {
        check('final sieve: block found in js/indexing/solutions.js', false, 'markers not found');
    } else {
        vm.runInContext('var showStatus = () => {};\n' + src.slice(i0, i1) + '\nglobalThis.__sieve = applyFinalSieve;', ctx);
        const sieve = ctx.__sieve;
        const vol = (s) => s.system === 'hexagonal' ? Math.sqrt(3) / 2 * s.a * s.a * s.c
                         : (s.a * (s.b ?? s.a) * (s.c ?? s.a)) * Math.sin((s.beta ?? 90) * Math.PI / 180);
        const S = (o) => ({ zero_correction: 0, _fitContext: 'x', ...o, volume: vol(o) });
        const show = (out) => out.map(s => `${s.system}${s.lattice === 'R' ? '(R)' : ''} a=${s.a}${s.c ? ' c=' + s.c : ''}`).join('; ');
        // Si: the cubic cell, a pseudo-cubic tetragonal one (same volume) and
        // the R cell with alpha = 90 (3x the volume), M(20) as measured.
        let out = sieve([S({ system: 'tetragonal', a: 5.4310, c: 5.4320, m20: 1518 }),
                         S({ system: 'hexagonal', lattice: 'R', a: 7.6800, c: 9.4069, m20: 1427 }),
                         S({ system: 'cubic', a: 5.4310, m20: 1054 })]);
        check('final sieve: Si kept as cubic over tetragonal c = a and R (alpha 90)',
              out.length === 1 && out[0].system === 'cubic', show(out));
        // F-centred cubic: R cell with alpha = 60 (3/4 of the volume), tetragonal a = c.
        out = sieve([S({ system: 'hexagonal', lattice: 'R', a: 8.2021, c: 20.0938, m20: 760 }),
                     S({ system: 'tetragonal', a: 11.6001, c: 11.6001, m20: 244.5 }),
                     S({ system: 'cubic', a: 11.6001, m20: 243.9 })]);
        check('final sieve: F-cubic kept as cubic over R (alpha 60) and tetragonal a = c',
              out.length === 1 && out[0].system === 'cubic', show(out));
        // A lower-symmetry cell that fits clearly better (M20 x4, a genuine
        // distortion or not) is kept AND the cubic one is listed beside it,
        // each marked with the other, for the user to decide.
        out = sieve([S({ system: 'tetragonal', a: 4.0000, c: 4.0060, m20: 400 }),
                     S({ system: 'cubic', a: 4.0020, m20: 100 })]);
        const tet = out.find(s => s.system === 'tetragonal'), cub = out.find(s => s.system === 'cubic');
        check('final sieve: a clearly better lower-symmetry cell is kept, with the cubic one beside it',
              out.length === 2 && !!tet && !!cub && tet._symAlt && tet._symAlt[0].system === 'cubic' &&
              cub._symAltOf && cub._symAltOf[0].system === 'tetragonal', show(out));
        // The 50-row cap never keeps one of such a pair without the other.
        const filler = Array.from({ length: 49 }, (_, i) => S({ system: 'cubic', a: 5.05 + 0.1 * i, m20: 1000 - i }));
        out = sieve([...filler, S({ system: 'tetragonal', a: 4.0000, c: 4.0060, m20: 400 }),
                     S({ system: 'cubic', a: 4.0020, m20: 100 })]);
        check('final sieve: the result cap keeps both cells of a pair',
              out.length === 51 && out.some(s => s.system === 'tetragonal') && out.some(s => s.a === 4.0020),
              `${out.length} kept`);
        // Monoclinic fits best; tetragonal is listed beside it, then the cubic
        // cell replaces the tetragonal one (within 2x of it): the monoclinic
        // cell is paired with the cubic one only, no mark names the dropped cell.
        const monTet = sieve([S({ system: 'monoclinic', a: 4.0000, b: 4.0010, c: 4.0060, beta: 90.1, m20: 400 }),
                              S({ system: 'tetragonal', a: 4.0000, c: 4.0060, m20: 150 })]);
        out = sieve([S({ system: 'monoclinic', a: 4.0000, b: 4.0010, c: 4.0060, beta: 90.1, m20: 400 }),
                     S({ system: 'tetragonal', a: 4.0000, c: 4.0060, m20: 150 }),
                     S({ system: 'cubic', a: 4.0020, m20: 100 })]);
        const mon = out.find(s => s.system === 'monoclinic'), cub2 = out.find(s => s.system === 'cubic');
        check('final sieve: a replaced cell leaves no pair mark behind',
              monTet.length === 2 && monTet.every(s => s._symAlt || s._symAltOf) &&   // the pair being replaced
              out.length === 2 && !!mon && !!cub2 && mon._symAlt && mon._symAlt.length === 1 &&
              mon._symAlt[0].system === 'cubic' && cub2._symAltOf && cub2._symAltOf[0].system === 'monoclinic',
              show(out) + (mon && mon._symAlt ? ' | mono marks: ' + mon._symAlt.map(o => o.system).join(',') : ''));
        // Equal volumes, different lattices: both stay (the help's example).
        out = sieve([S({ system: 'orthorhombic', a: 4, b: 5, c: 6, m20: 50 }),
                     S({ system: 'orthorhombic', a: 3, b: 5, c: 8, m20: 40 })]);
        check('final sieve: distinct lattices of equal volume are both kept', out.length === 2, show(out));
        // A real R lattice (corundum) is not taken for a cubic one.
        out = sieve([S({ system: 'hexagonal', lattice: 'R', a: 4.7589, c: 12.991, m20: 300 }),
                     S({ system: 'cubic', a: 4.2, m20: 20 })]);
        check('final sieve: a non-cubic R cell is kept beside an unrelated cubic one', out.length === 2, show(out));
    }
}

console.log(failures ? `\n${failures} test(s) FAILED` : '\nAll regression tests passed.');
process.exit(failures ? 1 : 0);
