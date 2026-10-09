// js/crystallography/monte-carlo.js
// Monte-Carlo cell polishing.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// ============================================================================
// --- MONTE-CARLO / SIMULATED-ANNEALING CELL POLISH -------------------------
// ============================================================================
// Least squares converges to the nearest minimum of the *residual*. M20 is not
// the residual: a cell a few tenths of a percent away can index more lines and
// score far higher, and no amount of relabelling from the current cell reaches
// it. This does a stochastic local hunt around a refined solution.
//
// Design notes:
//   * The walk happens in RECIPROCAL-space parameters (A=1/a^2, ...), the same
//     vector getLSDesignRow/extractCellFromFit already use, because q is LINEAR
//     in them. A step is therefore a uniform shift in q, not a distortion that
//     depends on where you are in the cell. Symmetry is enforced by
//     construction: the parameter vector for a tetragonal cell has 2 entries,
//     so no perturbation can ever break a==b.
//   * Zero error is a search dimension whenever refineZero is on. Z and the
//     cell parameters are strongly correlated -- a shift in `a` is nearly
//     compensated by a shift in Z -- so without Z in the vector the walk
//     explores a valley floor and never crosses the ridge.
//   * Scoring is SMOOTH for guidance (a peak moving toward its line improves
//     the score before it crosses the tolerance threshold) and M20 for
//     acceptance/reporting. Raw M20 is a plateau function: most small moves
//     change nothing at all, so an annealer driven by M20 alone gets no
//     gradient and random-walks.
//   * Every trial cell is validated (finite, positive, physical) before use.
//     extractCellFromFit already rejects non-physical parameter vectors; we
//     add explicit NaN/Inf guards at every stage on top of that.
// ----------------------------------------------------------------------------

// --- TUNING ----------------------------------------------------------------
// MC_ENABLED     master switch; set false to fall back to no post-processing.
// MC_MIN_M20     only polish solutions already worth polishing. The MC is a
//                LOCAL search: it sharpens a nearly-right cell, it does not
//                rescue a wrong one, and running it on every candidate would
//                dominate runtime for no gain.
// MC_ITERATIONS  length of EACH annealing run (not a budget to be divided).
// MC_RESTARTS    independent runs; cuts seed-to-seed spread roughly in half.
// MC_RANGE       search envelope as a fraction of each reciprocal parameter.
// MC_SEED        fixed so repeated runs on the same pattern agree.
const MC_ENABLED    = true;
// Solutions only reach this point after passing min_m20 (2.0) in
// refineAndTestSolution, so a gate above ~2 would skip exactly the marginal
// cells that have the most to gain. Keep it at the pipeline threshold.
const MC_MIN_M20    = 2;
const MC_ITERATIONS = 700;
const MC_RESTARTS   = 2;
const MC_RANGE      = 0.01;   // +/- 1%
const MC_SEED       = 20260723;
// Defaults for the "Refine MC" dialog. The user can override all three; these
// are just the values the dialog opens with. Cost is ~0.2-0.5 s per solution
// per restart, so Solutions x Restarts is the number that drives wall time.
const MC_DEFAULT_SOLUTIONS = 10;
// Deterministic PRNG (mulberry32). A fixed seed makes runs reproducible, which
// matters for a stochastic step inside an otherwise deterministic program: a
// user who reruns the same pattern gets the same answer.
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
                // design row is (4/3)(h^2+hk+k^2), so the coefficient is 1/a^2
                return [1 / (a * a), 1 / (c * c)];
            case 'orthorhombic':
                return [1 / (a * a), 1 / (b * b), 1 / (c * c)];
            case 'monoclinic': {
                const beta = (cell.beta ?? 90) * RAD;
                const sb = Math.sin(beta), cb = Math.cos(beta);
                if (!(Math.abs(sb) > 1e-9)) return null;
                const s2 = sb * sb;
                // q = A h^2 + B k^2 + C l^2 + D h l  with
                // A = 1/(a^2 sin^2 b), C = 1/(c^2 sin^2 b), D = -2cos(b)/(a c sin^2 b)
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
                // Same expressions as getQcalc's triclinic branch, in the
                // column order of getLSDesignRow: h^2 k^2 l^2 kl hl hk
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
// Per-parameter step scale. Uses the propagated standard errors when they are
// available and sane, otherwise falls back to a fraction of the parameter
// magnitude. A move of ~1 sigma is meaningful; a flat "1% of the cell edge" is
// arbitrary and can be 50 sigma or 0.1 sigma depending on the data quality.
function mcStepScales(params, cell, fracFallback) {
    const n = params.length;
    const s = new Array(n);
    for (let i = 0; i < n; i++) {
        const p = params[i];
        // Off-diagonal terms (monoclinic D, triclinic 4..6) can legitimately be
        // zero or negative, so scale them off the diagonal magnitude instead.
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
// Reciprocal parameter vector -> cell, reusing the existing extractor so the
// monoclinic/triclinic algebra stays in one place. Returns null on anything
// non-physical (extractCellFromFit already checks positivity, beta range,
// determinant sanity), plus our own finite/volume guards.
// `lattice` ('R' or undefined) is carried over from the cell being refined, so
// an R cell is scored on its R lines throughout the walk (see hkl.js).
function mcParamsToCell(params, system, maxVolume, lattice) {
    if (!params || params.some(p => !isFinite(p))) return null;
    const cell = extractCellFromFit(params, system);
    if (!cell) return null;
    cell.system = system;
    if (lattice === 'R' && system === 'hexagonal') cell.lattice = 'R';
    const axes = [cell.a, cell.b ?? cell.a, cell.c ?? cell.a];
    if (axes.some(x => !isFinite(x) || x < 2.0 || x > 50.0)) return null;
    const angs = [cell.alpha ?? 90, cell.beta ?? 90, cell.gamma ?? 90];
    if (angs.some(x => !isFinite(x) || x < 10 || x > 170)) return null;
    const vol = getVolume(cell);
    if (!(vol >= 20) || !isFinite(vol)) return null;
    if (maxVolume && !(vol <= maxVolume)) return null;
    cell.volume = vol;
    return cell;
}
// ---------------------------------------------------------------------------
// Smooth guidance score.
//
//   score = SUM over peaks of  w_i * exp(-(dq_i/tol_i)^2)   -  unindexed penalty
//
// Every peak contributes continuously: a line drifting toward a peak raises the
// score before it ever crosses the tolerance, which is exactly the gradient
// M20 fails to provide. Low-angle peaks are weighted up (same rationale as
// ls_weights_for_2theta: they carry the most information per unit of 2theta
// error). The result is bounded, so no single peak can dominate.
// ---------------------------------------------------------------------------
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
        // weight: favour low angle, matching the LS weighting philosophy
        const s2t = Math.sin(tc * RAD);
        const w = 1.0 / Math.max(s2t * s2t, 0.0012);
        s += w * contrib;
        if (x < 1) nIdx++;
    }
    if (!isFinite(s)) return -Infinity;
    // Mild bonus for genuinely indexed peaks so the smooth term cannot be
    // gamed by many near-misses beating a few exact hits.
    return s * (1 + 0.05 * nIdx);
}
// Full M20 / F_N evaluation for a trial cell, using exactly the same routine
// and tolerances as the main indexing path so numbers stay comparable.
function mcEvaluateCell(cell, ctx) {
    const { wavelength, q_max, d_min, impurity_peaks,
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
// ---------------------------------------------------------------------------
// The optimiser. Adaptive-step simulated annealing on the reciprocal parameter
// vector (+ zero when refined). Returns the best cell found, or null.
//
// Trial cost is deliberately asymmetric: every trial gets a cheap smooth score,
// but the expensive M20 is only computed for trials that survive that filter.
// LS re-refinement is a POLISH applied once at the end, not a search step --
// refining every trial would collapse most of them back into the same minimum
// and waste a full fit to learn nothing.
// ---------------------------------------------------------------------------
function monteCarloPolish(sol, data, state, opts) {
    const o = opts || {};
    const nIter        = o.iterations   ?? 400;
    const fracRange    = o.range        ?? 0.01;   // +/- 1% search envelope
    const T0           = o.T0           ?? 0.05;
    const cooling      = o.cooling      ?? 0.995;
    const seed         = o.seed         ?? 12345;
    const zeroStepDeg  = o.zeroStep     ?? 0.01;
    const zeroMaxDeg   = o.zeroMax      ?? 0.20;

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

        // Sanity check: the parameter round-trip must reproduce the cell. If it
        // does not, the vector convention is wrong for this system and the walk
        // would silently explore the wrong space -- bail out instead.
        const rt = mcParamsToCell(p0, system, max_volume, sol.lattice);
        if (!rt) return null;
        const drift = Math.abs(rt.a - sol.a) / Math.max(sol.a, 1e-9);
        if (!(drift < 1e-6)) return null;

        const scales = mcStepScales(p0, sol, fracRange * 0.5);
        // Hard envelope: never wander further than +/-fracRange in q-space.
        const lo = p0.map((p, i) => p - Math.abs(p) * fracRange * 2 - scales[i] * 2);
        const hi = p0.map((p, i) => p + Math.abs(p) * fracRange * 2 + scales[i] * 2);

        const rng = mcMakeRng(seed);
        const z0 = refineZero ? (sol.zero_correction || 0) : 0;

        // --- reference state -------------------------------------------------
        const baseCell = mcParamsToCell(p0, system, max_volume, sol.lattice);
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
        let bestCell = null;                 // only set if we beat the parent
        const startM20 = baseEval.m20;

        let T = T0 * Math.max(Math.abs(baseSmooth), 1e-6);
        let accepted = 0, evaluated = 0, stepMul = 1.0;

        for (let it = 0; it < nIter; it++) {
            // --- propose -----------------------------------------------------
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
                if (trialZ < -zeroMaxDeg) trialZ = -zeroMaxDeg;
                if (trialZ > zeroMaxDeg) trialZ = zeroMaxDeg;
            }

            const trialCell = mcParamsToCell(trialP, system, max_volume, sol.lattice);
            if (!trialCell) continue;
            if (refineZero) trialCell.zero_correction = trialZ;

            // --- cheap smooth score -----------------------------------------
            let qsorted;
            try { qsorted = generateQArray_for_worker(trialCell, q_max, wavelength); }
            catch (e) { continue; }
            if (!qsorted || qsorted.length === 0) continue;
            
            
            
            
            const sm = mcSmoothScore(qsorted, peaks_sorted_by_q, tolFn, trialZ, wavelength);
            if (!isFinite(sm)) continue;
            evaluated++;

            // --- Metropolis --------------------------------------------------
            const d = sm - curSmooth;
            let accept = d > 0;
            if (!accept && T > 1e-12) {
                const pr = Math.exp(d / T);
                accept = isFinite(pr) && rng() < pr;
            }
            if (accept) {
                curP = trialP; curZ = trialZ; curSmooth = sm; accepted++;

                // Only now pay for M20, and only when the smooth score says
                // this state is at least as good as where we started.
                if (sm >= baseSmooth) {
                    const ev = mcEvaluateCell(trialCell, ctx);
                    if (ev && isFinite(ev.m20)) {
                        // Primary criterion M20; m_all breaks ties, so the walk
                        // prefers cells that also index the high-angle lines.
                        const better = ev.m20 > bestM20 + 1e-9 ||
                                      (Math.abs(ev.m20 - bestM20) <= 1e-9 && ev.m_all > bestMAll + 1e-9);
                        if (better) {
                            bestM20 = ev.m20; bestMAll = ev.m_all;
                            bestP = trialP.slice(); bestZ = trialZ;
                        }
                    }
                }
            }

            // --- adapt ------------------------------------------------------
            T *= cooling;
            if ((it + 1) % 50 === 0) {
                const rate = accepted / 50;
                if (rate < 0.15) stepMul = Math.max(0.15, stepMul * 0.75);
                else if (rate > 0.55) stepMul = Math.min(4.0, stepMul * 1.3);
                accepted = 0;
            }
        }

        if (!(bestM20 > startM20 + 1e-6)) return null;   // nothing gained

        // --- final polish: one constrained LS refinement at the best point ---
        // This turns the walk's raw sampled point into a properly refined cell
        // with propagated standard deviations. It matters for more than tidiness:
        // a sampled point has no covariance matrix behind it, so a cell taken
        // straight from the walk would be reported without error bars while every
        // other solution in the ledger has them.
        //
        // The polish is preferred whenever it succeeds and does not lose M20. If
        // it wins on M20 it is used outright. If it is slightly worse but still
        // beats the parent, it is STILL used -- a refined cell with standard
        // deviations is worth more than a fraction of a point of M20 on a raw
        // sample, and the two cells are within a hair of each other by
        // construction. Only if the polish fails outright, or drops below the
        // parent, do we fall back to the sampled point.
        bestCell = mcParamsToCell(bestP, system, max_volume, sol.lattice);
        if (!bestCell) return null;
        if (refineZero) bestCell.zero_correction = bestZ;

        let finalCell = bestCell;
        const polished = mcLeastSquaresPolish(bestCell, data, state, ctx);
        if (polished && polished.errors && Object.keys(polished.errors).length) {
            const evP = mcEvaluateCell(polished, ctx);
            const m20Pol = (evP && isFinite(evP.m20)) ? evP.m20 : -Infinity;
            // Accept the refined cell as long as it still beats the PARENT. It
            // does not have to beat the raw sample: the sample is one draw from
            // the walk with no covariance behind it, while the polish is a real
            // fit at essentially the same point. Requiring the polish to win on
            // M20 as well was the bug -- when the sample happened to score a
            // fraction higher, the polished cell was thrown away and the result
            // reached the report with no error bars at all.
            if (m20Pol > startM20 + 1e-6) {
                finalCell = polished;
            }
        }

        // If we are keeping the raw sampled point, it still needs standard
        // deviations. They cannot come from mcLeastSquaresPolish, because that
        // MOVES the cell to the least-squares minimum -- and the whole reason we
        // are on the sample is that the minimum scores worse on M20. The sample
        // often sits on a narrow M20 spike that the residual minimum is not on.
        //
        // So: build the design matrix at the sampled cell, solve, and take only
        // the covariance matrix, discarding the shifted parameters. The result is
        // the uncertainty of the reported cell rather than of some other cell
        // nearby, which is exactly what should be quoted.
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
        // Carry provenance so the UI can flag these, and preserve any manual
        // swaps already attached to the parent.
        finalCell.manualSwaps = sol.manualSwaps || [];
        finalCell.mcPolished = true;
        finalCell.mcFrom = { m20: startM20, iterations: nIter, evaluated };
        return finalCell;
    } catch (e) {
        console.warn('monteCarloPolish failed:', e);
        return null;
    }
}
// ---------------------------------------------------------------------------
// Multi-restart driver. A single annealing run has noticeable seed-to-seed
// spread (~15-20% in M20 on realistic data) because the walk can settle in a
// side basin. Several short independent runs beat one long run of the same
// total cost and, more importantly, make the answer far less dependent on the
// arbitrary seed. Restarts are cheap: each one reuses the same peak list and
// tolerance closure.
//
// This is the function the rest of the program should call.
// ---------------------------------------------------------------------------
// `iterations` is the length of EACH run, not a budget to be divided. An
// annealing run that is cut short has not cooled, so splitting a fixed budget
// into short restarts measurably underperforms one long run (tested: 3x300 is
// worse than 1x900). Restarts buy reproducibility on top of a converged run,
// so total cost is iterations * restarts.
function monteCarloRefineCell(sol, data, state, opts) {
    const o = opts || {};
    const restarts = Math.max(1, o.restarts ?? 2);
    const perRun = Math.max(200, o.iterations ?? 700);
    let best = null, bestM20 = -Infinity, bestMAll = -Infinity;

    for (let r = 0; r < restarts; r++) {
        let cand = null;
        try {
            cand = monteCarloPolish(sol, data, state, {
                ...o,
                iterations: perRun,
                // Distinct but deterministic seed per restart.
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
// Standard deviations for a cell we do NOT want to move.
//
// mcLeastSquaresPolish returns a refined cell -- new parameters plus their
// errors. That is wrong when the cell being reported is a sampled point sitting
// on an M20 spike: refining walks off the spike, so the errors would belong to a
// different cell than the one quoted. This builds the same design matrix and
// solves the same system, but keeps only the covariance, so the uncertainties
// describe the reported cell.
function mcErrorsAtFixedCell(cell, data, state, ctx) {
    try {
        const { wavelength, refineZero } = data;
        const { peaks_sorted_by_q } = state;
        const system = cell.system;
        const minIndexed = { cubic: 4, tetragonal: 5, hexagonal: 5,
                             orthorhombic: 6, monoclinic: 7, triclinic: 7 };

        const refl = generateHKL_for_worker(cell, state.q_max, state.d_min, wavelength);
        if (!refl || refl.length === 0) return null;
        const z = cell.zero_correction || 0;

        const rows = [], qv = [], qCorr = [], tthRads = [];
        const used = new Set();
        for (let i = 0; i < peaks_sorted_by_q.length; i++) {
            const p = peaks_sorted_by_q[i];
            const tc = p.tth - z;
            if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
            const st = Math.sin(tc * RAD / 2);
            const q_to_match = (4 * st * st) / (wavelength * wavelength);
            if (!isFinite(q_to_match)) continue;
            const tol = ctx.tolFn(p.original_index);
            let best = -1, bd = Infinity;
            for (let j = 0; j < refl.length; j++) {
                const dd = Math.abs(refl[j].q - q_to_match);
                if (dd < bd) { bd = dd; best = j; }
            }
            if (best < 0 || !(bd < tol) || used.has(best)) continue;
            used.add(best);
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

        // Errors are propagated against the REPORTED cell, not the fitted one.
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
// Constrained least-squares refinement at a fixed assignment, used as the final
// polish. Mirrors refineAndTestSolution's pairing so the result is consistent
// with the rest of the program.
function mcLeastSquaresPolish(cell, data, state, ctx) {
    try {
        const { wavelength, tth_error, refineZero } = data;
        const { peaks_sorted_by_q, tth_obs_rad, original_indices } = state;
        const system = cell.system;
        const minIndexed = { cubic: 4, tetragonal: 5, hexagonal: 5,
                             orthorhombic: 6, monoclinic: 7, triclinic: 7 };

        const refl = generateHKL_for_worker(cell, state.q_max, state.d_min, wavelength);
        if (!refl || refl.length === 0) return null;
        const z = cell.zero_correction || 0;

        // qv holds the target for the zero-refining fit (uncorrected q, so the
        // zero column has something to absorb); qCorr holds the zero-corrected q
        // used by the fallback fit that holds the zero fixed.
        const rows = [], qv = [], qCorr = [], tthRads = [];
        const used = new Set();
        for (let i = 0; i < peaks_sorted_by_q.length; i++) {
            const p = peaks_sorted_by_q[i];
            const oidx = p.original_index;
            const tc = p.tth - z;
            if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
            const st = Math.sin(tc * RAD / 2);
            const q_to_match = (4 * st * st) / (wavelength * wavelength);
            if (!isFinite(q_to_match)) continue;
            const tol = ctx.tolFn(oidx);
            let best = -1, bd = Infinity;
            for (let j = 0; j < refl.length; j++) {
                const dd = Math.abs(refl[j].q - q_to_match);
                if (dd < bd) { bd = dd; best = j; }
            }
            if (best < 0 || !(bd < tol) || used.has(best)) continue;
            used.add(best);
            const R = refl[best];
            const row = getLSDesignRow([R.h, R.k, R.l], system);
            if (!row || row.some(x => !isFinite(x))) continue;
            // Fit against the UNCORRECTED q when the zero is being refined, so
            // the zero column has something to absorb.
            const stRaw = Math.sin(p.tth * RAD / 2);
            const qRaw = (4 * stRaw * stRaw) / (wavelength * wavelength);
            if (refineZero) row.push((2 / (wavelength * wavelength)) * Math.sin(p.tth * RAD));
            rows.push(row);
            qv.push(refineZero ? qRaw : q_to_match);
            qCorr.push(q_to_match);
            tthRads.push(p.tth * RAD);
        }

        // The zero column costs one degree of freedom. With a short peak list
        // that extra column can be the difference between a fit and no fit at
        // all -- and returning null here means the caller falls back to the raw
        // sampled point, which has no covariance matrix and therefore reaches
        // the report with no error bars. That is exactly the case a 7-peak
        // orthorhombic pattern hits: 6 cell parameters + 1 zero = 7 rows needed,
        // so a single unpaired peak sinks it.
        //
        // So: try with the zero column, and if there is not enough data for it,
        // refit the cell alone holding the zero at its current value. A cell
        // with standard deviations and a fixed zero is far more useful than a
        // cell with neither.
        const needCell = (minIndexed[system] || 6);
        const needWithZero = needCell + 1;
        let useZeroColumn = refineZero && rows.length >= needWithZero;
        if (rows.length < needCell) return null;

        let designRows = rows, targets = qv;
        if (refineZero && !useZeroColumn) {
            // Drop the zero column and fit the zero-corrected q instead, so the
            // current zero is applied but not refined.
            designRows = rows.map(r => r.slice(0, r.length - 1));
            targets = qCorr;
        }

        const fit = solveLeastSquares(designRows, targets, ls_weights_for_2theta(tthRads));
        if (!fit || !fit.solution) return null;
        if (fit.solution.some(x => !isFinite(x))) return null;

        const nCellPar = MC_NPAR[system];
        const out = extractCellFromFit(fit.solution.slice(0, nCellPar), system);
        if (!out) return null;
        out.system = system;
        if (cell.lattice === 'R' && system === 'hexagonal') out.lattice = 'R';
        if (refineZero) {
            if (useZeroColumn) {
                const zNew = fit.solution[fit.solution.length - 1] * DEG;
                if (!isFinite(zNew) || Math.abs(zNew) > 1.0) return null;
                out.zero_correction = zNew;
            } else {
                // Zero was held, not refined; carry the value forward unchanged.
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
