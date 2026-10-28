// js/crystallography/monte-carlo.js
// Monte-Carlo cell polishing.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (index.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// ============================================================================
// --- MONTE-CARLO / SIMULATED-ANNEALING CELL POLISH -------------------------
// ============================================================================

// Deterministic PRNG (mulberry32). A fixed seed makes runs reproducible.
function mcMakeRng(seed) {
    let t = (seed >>> 0) || 0x9e3779b9;
    return function () {
        t += 0x6D2B79F5;
        let r = t;
        r = Math.imul(r ^ (r >>> 15), r | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}
// Box-Muller, guarded against log(0).
function mcGauss(rng) {
    let u = 0, v = 0;
    while (u <= 1e-12) u = rng();
    v = rng();
    const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return isFinite(g) ? g : 0;
}
// Number of free reciprocal-space parameters per system.
const MC_NPAR = {
    cubic: 1, tetragonal: 2, hexagonal: 2,
    orthorhombic: 3, monoclinic: 4, triclinic: 6
};
// Cell -> reciprocal parameter vector, matching getLSDesignRow's column order
// exactly. getQcalc(hkl, cell) === dot(getLSDesignRow(hkl, system), vector).
function mcCellToParams(cell) {
    const sys = cell.system;
    const a = cell.a, b = cell.b ?? cell.a, c = cell.c ?? cell.a;
    if (!(a > 0) || !(b > 0) || !(c > 0)) return null;
    try {
        switch (sys) {
            case 'cubic':
                return [1 / (a * a)];
            case 'tetragonal':
                return [1 / (a * a), 1 / (c * c)];
            case 'hexagonal':
                return [1 / (a * a), 1 / (c * c)];
            case 'orthorhombic':
                return [1 / (a * a), 1 / (b * b), 1 / (c * c)];
            case 'monoclinic': {
                const beta = (cell.beta ?? 90) * RAD;
                const sb = Math.sin(beta), cb = Math.cos(beta);
                if (!(Math.abs(sb) > 1e-9)) return null;
                const s2 = sb * sb;
                return [
                    1 / (a * a * s2),
                    1 / (b * b),
                    1 / (c * c * s2),
                    -2 * cb / (a * c * s2)
                ];
            }
            case 'triclinic': {
                const al = (cell.alpha ?? 90) * RAD;
                const be = (cell.beta ?? 90) * RAD;
                const ga = (cell.gamma ?? 90) * RAD;
                const ca = Math.cos(al), cb = Math.cos(be), cg = Math.cos(ga);
                const V_sq = a * a * b * b * c * c *
                             (1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);
                if (!(V_sq > 1e-12)) return null;
                return [
                    (b * b * c * c * (1 - ca * ca)) / V_sq,
                    (a * a * c * c * (1 - cb * cb)) / V_sq,
                    (a * a * b * b * (1 - cg * cg)) / V_sq,
                    2 * b * c * a * a * (cb * cg - ca) / V_sq,
                    2 * a * c * b * b * (ca * cg - cb) / V_sq,
                    2 * a * b * c * c * (ca * cb - cg) / V_sq
                ];
            }
        }
    } catch (e) { return null; }
    return null;
}
// Per-parameter step scale.
function mcStepScales(params, fracFallback) {
    const n = params.length;
    const s = new Array(n);
    for (let i = 0; i < n; i++) {
        const p = params[i];
        let base;
        if (i < 3 && p > 0) base = p;
        else {
            let dmax = 0;
            for (let j = 0; j < Math.min(3, n); j++) dmax = Math.max(dmax, Math.abs(params[j]));
            base = dmax > 0 ? dmax : Math.abs(p);
        }
        let v = Math.abs(base) * fracFallback;
        if (!isFinite(v) || v <= 0) v = 1e-6;
        s[i] = v;
    }
    return s;
}
// Reciprocal parameter vector -> cell, reusing the existing extractor.
function mcParamsToCell(params, system, maxVolume, lattice, data = {}) {
    if (!params || params.some(p => !isFinite(p))) return null;
    const cell = extractCellFromFit(params, system);
    if (!cell) return null;
    cell.system = system;
    if (lattice === 'R' && system === 'hexagonal') cell.lattice = 'R';
    if (!cellWithinPhysicalLimits(cell, data, maxVolume)) return null;
    cell.volume = getVolume(cell);
    return cell;
}
// Smooth guidance score.
function mcSmoothScore(q_calc_sorted, peaks, tolFn, zero_deg, wavelength) {
    if (!q_calc_sorted || q_calc_sorted.length === 0) return -Infinity;
    const n = peaks.length;
    if (n === 0) return -Infinity;
    let s = 0, nIdx = 0;
    for (let i = 0; i < n; i++) {
        const p = peaks[i];
        const tc = p.tth - zero_deg;
        if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
        const st = Math.sin(tc * RAD / 2);
        const q_o = (4 * st * st) / (wavelength * wavelength);
        if (!isFinite(q_o)) continue;
        const tol = tolFn(p.original_index);
        if (!(tol > 0)) continue;
        const bi = binarySearchClosest(q_calc_sorted, q_o);
        const dq = Math.abs(q_o - q_calc_sorted[bi]);
        if (!isFinite(dq)) continue;
        const x = dq / tol;
        const contrib = Math.exp(-x * x);
        const s2t = Math.sin(tc * RAD);
        const w = 1.0 / Math.max(s2t * s2t, 0.0012);
        s += w * contrib;
        if (x < 1) nIdx++;
    }
    if (!isFinite(s)) return -Infinity;
    return s * (1 + 0.05 * nIdx);
}
// Full M20 / F_N evaluation for a trial cell.
function mcEvaluateCell(cell, ctx) {
    const { wavelength, q_max, impurity_peaks,
            peaks_sorted_by_q, n_20, n_all, tolFn } = ctx;
    let qsorted;
    try {
        qsorted = generateQArray_for_worker(cell, q_max, wavelength);
    } catch (e) { return null; }
    if (!qsorted || qsorted.length === 0) return null;

    const z = cell.zero_correction || 0;
    const mk = (n) => {
        const out = [];
        for (let i = 0; i < n; i++) {
            const p = peaks_sorted_by_q[i];
            const tc = p.tth - z;
            if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
            const st = Math.sin(tc * RAD / 2);
            const q = (4 * st * st) / (wavelength * wavelength);
            if (!isFinite(q)) continue;
            out.push({ ...p, q, tth: tc });
        }
        return out;
    };

    const p20 = mk(n_20);
    if (p20.length === 0) return null;
    let f20, fAll;
    try {
        f20 = calculateFiguresOfMerit(qsorted, p20, impurity_peaks, tolFn, wavelength);
        const pAll = mk(n_all);
        fAll = calculateFiguresOfMerit(qsorted, pAll, impurity_peaks, tolFn, wavelength);
        cell.n_20 = p20.length;
        cell.n_all = pAll.length;
    } catch (e) { return null; }

    const m20 = (f20 && isFinite(f20.m20)) ? f20.m20 : 0;
    const mAll = (fAll && isFinite(fAll.m20)) ? fAll.m20 : 0;
    cell.m20 = m20;
    cell.fN_20 = (f20 && isFinite(f20.fN)) ? f20.fN : 0;
    cell.m_all = mAll;
    cell.fN_all = (fAll && isFinite(fAll.fN)) ? fAll.fN : 0;
    return { m20, m_all: mAll, qsorted };
}
// Adaptive-step simulated annealing on the reciprocal parameter vector.
function monteCarloPolish(sol, data, state, opts) {
    const o = opts || {};
    const nIter        = o.iterations   ?? BRUTUS_LIMITS.monteCarlo.iterations;
    const fracRange    = o.range        ?? BRUTUS_LIMITS.monteCarlo.range;
    const T0           = o.T0           ?? BRUTUS_LIMITS.monteCarlo.temperature;
    const cooling      = o.cooling      ?? BRUTUS_LIMITS.monteCarlo.cooling;
    const seed         = o.seed         ?? 12345;
    const zeroStepDeg  = o.zeroStep     ?? BRUTUS_LIMITS.monteCarlo.zeroStep;
    const zeroMaxDeg   = o.zeroMax      ?? BRUTUS_LIMITS.monteCarlo.zeroMax;

    try {
        const system = sol && sol.system;
        if (!system || !MC_NPAR[system]) return null;

        const { wavelength, tth_error, impurity_peaks, refineZero, max_volume } = data;
        const { peaks_sorted_by_q, tth_obs_rad, N_FOR_M20, q_max, d_min } = state;
        if (!peaks_sorted_by_q || peaks_sorted_by_q.length === 0) return null;
        if (!(wavelength > 0) || !isFinite(wavelength)) return null;

        const tolFn = (idx) => get_q_tolerance(idx, tth_obs_rad, wavelength, tth_error);
        const n_all = peaks_sorted_by_q.length;
        const n_20 = Math.min(N_FOR_M20, n_all);

        const ctx = {
            wavelength, q_max, d_min, impurity_peaks,
            peaks_sorted_by_q, n_20, n_all, tolFn
        };

        const p0 = mcCellToParams(sol);
        if (!p0) return null;
        const nPar = p0.length;
        if (nPar !== MC_NPAR[system]) return null;

        const rt = mcParamsToCell(p0, system, max_volume, sol.lattice, data);
        if (!rt) return null;
        const drift = Math.abs(rt.a - sol.a) / Math.max(sol.a, 1e-9);
        if (!(drift < 1e-6)) return null;

        const scales = mcStepScales(p0, fracRange * 0.5);
        const lo = p0.map((p, i) => p - Math.abs(p) * fracRange * 2 - scales[i] * 2);
        const hi = p0.map((p, i) => p + Math.abs(p) * fracRange * 2 + scales[i] * 2);

        const rng = mcMakeRng(seed);
        const z0 = refineZero ? (sol.zero_correction || 0) : 0;

        // --- reference state -------------------------------------------------
        const baseCell = mcParamsToCell(p0, system, max_volume, sol.lattice, data);
        if (!baseCell) return null;
        if (refineZero) baseCell.zero_correction = z0;
        const baseEval = mcEvaluateCell(baseCell, ctx);
        if (!baseEval) return null;
        const baseSmooth = mcSmoothScore(baseEval.qsorted, peaks_sorted_by_q, tolFn, z0, wavelength);
        if (!isFinite(baseSmooth)) return null;

        let curP = p0.slice(), curZ = z0;
        let curSmooth = baseSmooth;
        let bestP = p0.slice(), bestZ = z0;
        let bestM20 = baseEval.m20, bestMAll = baseEval.m_all;
        let bestCell = null;
        const startM20 = baseEval.m20;

        let T = T0 * Math.max(Math.abs(baseSmooth), 1e-6);
        let accepted = 0, evaluated = 0, stepMul = 1.0;

        for (let it = 0; it < nIter; it++) {
            const trialP = new Array(nPar);
            let ok = true;
            for (let i = 0; i < nPar; i++) {
                let v = curP[i] + mcGauss(rng) * scales[i] * stepMul;
                if (!isFinite(v)) { ok = false; break; }
                if (v < lo[i]) v = lo[i];
                if (v > hi[i]) v = hi[i];
                trialP[i] = v;
            }
            if (!ok) continue;

            let trialZ = curZ;
            if (refineZero) {
                trialZ = curZ + mcGauss(rng) * zeroStepDeg * stepMul;
                if (!isFinite(trialZ)) continue;
                if (trialZ < z0 - zeroMaxDeg) trialZ = z0 - zeroMaxDeg;
                if (trialZ > z0 + zeroMaxDeg) trialZ = z0 + zeroMaxDeg;
            }

            const trialCell = mcParamsToCell(trialP, system, max_volume, sol.lattice, data);
            if (!trialCell) continue;
            if (refineZero) trialCell.zero_correction = trialZ;

            let qsorted;
            try { qsorted = generateQArray_for_worker(trialCell, q_max, wavelength); }
            catch (e) { continue; }
            if (!qsorted || qsorted.length === 0) continue;

            const sm = mcSmoothScore(qsorted, peaks_sorted_by_q, tolFn, trialZ, wavelength);
            if (!isFinite(sm)) continue;
            evaluated++;

            const d = sm - curSmooth;
            let accept = d > 0;
            if (!accept && T > 1e-12) {
                const pr = Math.exp(d / T);
                accept = isFinite(pr) && rng() < pr;
            }
            if (accept) {
                curP = trialP; curZ = trialZ; curSmooth = sm; accepted++;

                if (sm >= baseSmooth) {
                    const ev = mcEvaluateCell(trialCell, ctx);
                    if (ev && isFinite(ev.m20)) {
                        const better = ev.m20 > bestM20 + 1e-9 ||
                                      (Math.abs(ev.m20 - bestM20) <= 1e-9 && ev.m_all > bestMAll + 1e-9);
                        if (better) {
                            bestM20 = ev.m20; bestMAll = ev.m_all;
                            bestP = trialP.slice(); bestZ = trialZ;
                        }
                    }
                }
            }

            T *= cooling;
            if ((it + 1) % 50 === 0) {
                const rate = accepted / 50;
                if (rate < 0.15) stepMul = Math.max(0.15, stepMul * 0.75);
                else if (rate > 0.55) stepMul = Math.min(4.0, stepMul * 1.3);
                accepted = 0;
            }
        }

        if (!(bestM20 > startM20 + 1e-6)) return null;

        bestCell = mcParamsToCell(bestP, system, max_volume, sol.lattice, data);
        if (!bestCell) return null;
        if (refineZero) bestCell.zero_correction = bestZ;

        let finalCell = bestCell;
        const polished = mcLeastSquaresPolish(bestCell, data, state, ctx);
        if (polished && polished.errors && Object.keys(polished.errors).length) {
            const evP = mcEvaluateCell(polished, ctx);
            const m20Pol = (evP && isFinite(evP.m20)) ? evP.m20 : -Infinity;
            if (m20Pol > startM20 + 1e-6) {
                finalCell = polished;
            }
        }

        if (!finalCell.errors) {
            const cov = mcErrorsAtFixedCell(finalCell, data, state, ctx);
            if (cov) finalCell.errors = cov;
        }

        const evF = mcEvaluateCell(finalCell, ctx);
        if (!evF || !isFinite(evF.m20)) return null;
        if (!(evF.m20 > startM20 + 1e-6)) return null;

        finalCell.system = system;
        finalCell.volume = getVolume(finalCell);
        if (!isFinite(finalCell.volume) || finalCell.volume <= 0) return null;
        finalCell.mcPolished = true;
        finalCell.mcFrom = { m20: startM20, iterations: nIter, evaluated };
        return finalCell;
    } catch (e) {
        console.warn('monteCarloPolish failed:', e);
        return null;
    }
}
// Multi-restart driver.
function monteCarloRefineCell(sol, data, state, opts) {
    const o = opts || {};
    const restarts = Math.max(1, o.restarts ?? BRUTUS_LIMITS.monteCarlo.restarts);
    const perRun = Math.max(BRUTUS_LIMITS.monteCarlo.minIterations, o.iterations ?? BRUTUS_LIMITS.monteCarlo.refineIterations);
    let best = null, bestM20 = -Infinity, bestMAll = -Infinity;

    for (let r = 0; r < restarts; r++) {
        let cand = null;
        try {
            cand = monteCarloPolish(sol, data, state, {
                ...o,
                iterations: perRun,
                seed: (o.seed ?? 12345) + r * 7919
            });
        } catch (e) { cand = null; }
        if (!cand) continue;
        const m = isFinite(cand.m20) ? cand.m20 : -Infinity;
        const ma = isFinite(cand.m_all) ? cand.m_all : -Infinity;
        if (m > bestM20 + 1e-9 || (Math.abs(m - bestM20) <= 1e-9 && ma > bestMAll + 1e-9)) {
            best = cand; bestM20 = m; bestMAll = ma;
        }
    }
    return best;
}
// Helper to pair peaks to reflections best-first (closest peak wins a contested line).
function _mcPairPeaks(peaks_sorted_by_q, refl, z, wavelength, tolFn, system, refineZero) {
    const cands = [];
    for (let i = 0; i < peaks_sorted_by_q.length; i++) {
        const p = peaks_sorted_by_q[i];
        const tc = p.tth - z;
        if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
        const st = Math.sin(tc * RAD / 2);
        const q_to_match = (4 * st * st) / (wavelength * wavelength);
        if (!isFinite(q_to_match)) continue;
        const tol = tolFn(p.original_index);
        let best = -1, bd = Infinity;
        for (let j = 0; j < refl.length; j++) {
            const dd = Math.abs(refl[j].q - q_to_match);
            if (dd < bd) { bd = dd; best = j; }
        }
        if (best >= 0 && bd < tol) {
            cands.push({ i, p, q_to_match, best, bd });
        }
    }
    cands.sort((a, b) => a.bd - b.bd);
    const used = new Set();
    const claimed = [];
    for (const c of cands) {
        if (used.has(c.best)) continue;
        used.add(c.best);
        claimed.push(c);
    }
    claimed.sort((a, b) => a.i - b.i);

    const rows = [], qv = [], qCorr = [], tthRads = [];
    for (const { p, q_to_match, best } of claimed) {
        const R = refl[best];
        const row = getLSDesignRow([R.h, R.k, R.l], system);
        if (!row || row.some(x => !isFinite(x))) continue;
        const stRaw = Math.sin(p.tth * RAD / 2);
        const qRaw = (4 * stRaw * stRaw) / (wavelength * wavelength);
        if (refineZero) row.push((2 / (wavelength * wavelength)) * Math.sin(p.tth * RAD));
        rows.push(row);
        qv.push(refineZero ? qRaw : q_to_match);
        qCorr.push(q_to_match);
        tthRads.push(p.tth * RAD);
    }
    return { rows, qv, qCorr, tthRads };
}
// Standard deviations for a cell we do NOT want to move.
function mcErrorsAtFixedCell(cell, data, state, ctx) {
    try {
        const { wavelength, refineZero } = data;
        const { peaks_sorted_by_q } = state;
        const system = cell.system;
        const minIndexed = BRUTUS_LIMITS.search.minIndexed;

        const refl = generateHKL_for_worker(cell, state.q_max, state.d_min, wavelength);
        if (!refl || refl.length === 0) return null;
        const z = cell.zero_correction || 0;

        const { rows, qv, qCorr, tthRads } = _mcPairPeaks(peaks_sorted_by_q, refl, z, wavelength, ctx.tolFn, system, refineZero);

        const needCell = (minIndexed[system] || 6);
        if (rows.length < needCell) return null;
        const useZeroColumn = refineZero && rows.length >= needCell + 1;

        let designRows = rows, targets = qv;
        if (refineZero && !useZeroColumn) {
            designRows = rows.map(r => r.slice(0, r.length - 1));
            targets = qCorr;
        }

        const fit = solveLeastSquares(designRows, targets, ls_weights_for_2theta(tthRads));
        if (!fit || !fit.covarianceMatrix) return null;

        const errs = propagateErrors(system, fit, cell);
        if (!errs || !Object.keys(errs).length) return null;
        for (const k of Object.keys(errs)) {
            if (!isFinite(errs[k]) || errs[k] < 0) return null;
        }
        return errs;
    } catch (e) {
        return null;
    }
}
// Constrained least-squares refinement at a fixed assignment, used as the final polish.
function mcLeastSquaresPolish(cell, data, state, ctx) {
    try {
        const { wavelength, refineZero } = data;
        const { peaks_sorted_by_q } = state;
        const system = cell.system;
        const minIndexed = BRUTUS_LIMITS.search.minIndexed;

        const refl = generateHKL_for_worker(cell, state.q_max, state.d_min, wavelength);
        if (!refl || refl.length === 0) return null;
        const z = cell.zero_correction || 0;

        const { rows, qv, qCorr, tthRads } = _mcPairPeaks(peaks_sorted_by_q, refl, z, wavelength, ctx.tolFn, system, refineZero);

        const needCell = (minIndexed[system] || 6);
        const needWithZero = needCell + 1;
        const useZeroColumn = refineZero && rows.length >= needWithZero;
        if (rows.length < needCell) return null;

        let designRows = rows, targets = qv;
        if (refineZero && !useZeroColumn) {
            designRows = rows.map(r => r.slice(0, r.length - 1));
            targets = qCorr;
        }

        const fit = solveLeastSquares(designRows, targets, ls_weights_for_2theta(tthRads));
        if (!fit || !fit.solution) return null;
        if (fit.solution.some(x => !isFinite(x))) return null;

        const nCellPar = MC_NPAR[system];
        const out = mcParamsToCell(fit.solution.slice(0, nCellPar), system, data.max_volume, cell.lattice, data);
        if (!out) return null;
        out.system = system;
        if (cell.lattice === 'R' && system === 'hexagonal') out.lattice = 'R';
        if (refineZero) {
            if (useZeroColumn) {
                const zNew = fit.solution[fit.solution.length - 1] * DEG;
                if (!isFinite(zNew) || Math.abs(zNew) > BRUTUS_LIMITS.monteCarlo.polishZeroMax) return null;
                out.zero_correction = zNew;
            } else {
                out.zero_correction = z;
            }
        }
        out.volume = getVolume(out);
        if (!isFinite(out.volume) || out.volume <= 0) return null;
        try { out.errors = propagateErrors(system, fit, out); } catch (e) { out.errors = null; }
        return out;
    } catch (e) {
        return null;
    }
}
