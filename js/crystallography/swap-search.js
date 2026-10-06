// js/crystallography/swap-search.js
// Combinatorial hkl swap search.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

//20 nov, worker
// Check if we are running in a Worker environment to avoid conflicts with the main thread
// ============================================================================
// COMBINATORIAL SWAP SEARCH
// ----------------------------------------------------------------------------
// Replaces the old "swap fishing" block in findTransformedSolutions().
//
// What changed and why
// --------------------
// The old pass was NOT combinatorial. It did two things:
//   (a) for each of the first 12 peaks, try up to 2 alternative labels, ONE
//       peak at a time (single-peak overrides only);
//   (b) transpose the labels of the 3 closest-lying peak pairs.
// So the reachable set was {single relabel} U {one pairwise transposition}.
// A real crossing usually drags a third line with it, and two independent
// mislabels are completely unreachable. Total budget: 12 refits.
//
// This version enumerates, for every peak, EVERY calculated line inside the
// user's q tolerance window, and searches the full CARTESIAN PRODUCT of those
// per-peak candidate sets -- i.e. arbitrarily many peaks relabelled at once.
// The product is walked best-first (cheapest total mislabel penalty first) so
// it can be truncated at any budget and still have covered the most plausible
// region exhaustively. Cost is bounded by MAX_FITS, not by the product size.
//
// Three things make that affordable:
//   1. The parent's line list is generated ONCE, not once per trial.
//      (refineWithManualHkl regenerates it TWICE per call and does an O(P*L)
//      linear nearest-line scan per peak -- that is what made the old pass
//      unaffordable above ~12 trials.)
//   2. Candidate labels that produce the same least-squares design row are
//      collapsed: they are indistinguishable to the fit, so trying both is
//      pure waste. This is a large pruning factor in high-symmetry systems.
//   3. Two-stage evaluation. Stage 1 is a bare weighted LS solve on cached
//      design rows (~microseconds, no HKL generation) scored by weighted RMS
//      residual. Only the best MAX_EVALS distinct cells reach stage 2, which
//      regenerates lines and computes M20/F(N).
//
// Residual is used for RANKING only, never for acceptance -- as the original
// comment correctly noted, a swapped labelling always has a larger residual at
// FIXED cell. After refitting that is no longer true, which is exactly why the
// refit is the discriminator. Acceptance is still M20.
// ============================================================================

const SWAP_CFG = {
    // Candidate GENERATION is deliberately unrestricted: every peak, every line
    // inside the window. Cost is bounded by MAX_FITS (the best-first walk visits
    // the cheapest assignments first), so capping generation only removes
    // reachable answers without saving time. An earlier build capped this at the
    // 16 lowest-angle peaks and lost relabellings at peaks 23 and 27 on a real
    // orthorhombic pattern.
    MAX_PEAKS:  1e9,   // how many low-angle peaks may be relabelled (no cap)
    ALT_WINDOW: 2.5,   // candidate window, in units of the user's q tolerance
    MAX_ALT:    4,     // distinct labels kept per peak (after row dedup)
    MAX_FREE:   8,     // peaks allowed to vary simultaneously (product dims)
    MAX_FITS:   150,   // stage-1 cheap LS fits per ROUND (rounds do the depth)
    MAX_EVALS:  16,    // stage-2 full M20 evaluations per parent solution
    MAX_POST:   3,     // best N of those actually posted to the UI
    KEEP_RATIO: 1.0,   // post only if m20 >= KEEP_RATIO * parent m20
    LOSE_SLACK: 1,     // a child may index at most this many fewer lines overall
    ROUNDS:     4,     // re-centre and search again on the improved cell
    ROUND_GAIN: 1.02,  // M20 gain needed to justify another round
    TOP_N:      40,    // only the best N parents get a swap search at all
    DEBUG:      false,
};
class _SwapHeap {
    constructor() { this.a = []; }
    get size() { return this.a.length; }
    push(x) {
        const a = this.a; a.push(x);
        let i = a.length - 1;
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (a[p].cost <= a[i].cost) break;
            const t = a[p]; a[p] = a[i]; a[i] = t; i = p;
        }
    }
    pop() {
        const a = this.a, top = a[0], last = a.pop();
        if (a.length) {
            a[0] = last;
            for (let i = 0; ;) {
                const l = 2 * i + 1, r = l + 1;
                let m = i;
                if (l < a.length && a[l].cost < a[m].cost) m = l;
                if (r < a.length && a[r].cost < a[m].cost) m = r;
                if (m === i) break;
                const t = a[m]; a[m] = a[i]; a[i] = t; i = m;
            }
        }
        return top;
    }
}
// Stage 1: weighted LS on a fixed labelling. No HKL generation, no nearest-line
// scan -- the design rows come straight from the cached line list.
function _swapCheapFit(labels, ctx) {
    const { lines, system, refineZero, rhs, wts, zcol, nAll, need, maxVolume, qc, qL } = ctx;

    // One calculated line per observed peak, same rule as pair_and_fit() and
    // assignNearestLines(): a transposition (two peaks exchanging DIFFERENT
    // labels) is the whole point of this search, but two peaks landing on the
    // SAME line would be fitted as duplicate equations and averaged. Closest
    // peak keeps the line, the other is dropped from this trial.
    const order = [];
    for (let i = 0; i < nAll; i++) {
        const j = labels[i];
        if (j >= 0) order.push({ i, j, d: Math.abs(qL[j] - qc[i]) });
    }
    order.sort((x, y) => x.d - y.d);

    const claimed = new Set();
    const M = [], q = [], w = [];
    for (const { i, j } of order) {
        if (claimed.has(j)) continue;
        const L = lines[j];
        const row = getLSDesignRow([L.h, L.k, L.l], system);
        if (!row) continue;
        claimed.add(j);
        if (refineZero) row.push(zcol[i]);
        M.push(row); q.push(rhs[i]); w.push(wts[i]);
    }
    if (M.length < need) return null;

    const fit = solveLeastSquares(M, q, w);
    if (!fit || !fit.solution) return null;
    const cell = extractCellFromFit(fit.solution, system);
    if (!cell) return null;
    cell.system = system;
    if (refineZero) cell.zero_correction = fit.solution[fit.solution.length - 1] * DEG;
    else if (ctx.z) cell.zero_correction = ctx.z;

    const V = getVolume(cell);
    if (!(V >= 20) || !(V <= maxVolume) || !isFinite(V)) return null;
    cell.volume = V;
    cell.nPaired = M.length;

    // solveLeastSquares now returns the weighted RMS residual it already
    // computed internally, so there is nothing to recompute here.
    cell._resid = isFinite(fit.wrms) ? fit.wrms : Infinity;
    cell._fit = fit;
    return cell;
}
// Stage 2: figures of merit, computed exactly the way refineAndTestSolution
// computes them so the numbers are directly comparable with every other
// solution in the table. Uses the q_only fast path (no reflection objects).
function _swapEvalFoM(cell, ctx) {
    const { wavelength, qMax, peaks, N20, impurityPeaks, tolFn } = ctx;
    const qCalc = generateQArray_for_worker(cell, qMax, wavelength);
    if (!qCalc || qCalc.length === 0) return null;

    const z = cell.zero_correction || 0;
    const nAll = peaks.length;
    const mk = (n) => {
        const out = new Array(n);
        for (let i = 0; i < n; i++) {
            const p = peaks[i];
            const tc = p.tth - z;
            out[i] = { ...p, tth: tc, q: (4 * Math.sin(tc * RAD / 2) ** 2) / (wavelength ** 2) };
        }
        return out;
    };

    const n20 = Math.min(N20, nAll);
    const f20 = calculateFiguresOfMerit(qCalc, mk(n20), impurityPeaks, tolFn, wavelength);
    if (!(f20.m20 > 0)) return null;
    const allPeaks = mk(nAll);
    const fAll = calculateFiguresOfMerit(qCalc, allPeaks, impurityPeaks, tolFn, wavelength);

    // How many of ALL N observed peaks fall within tolerance of a calculated
    // line. M(N) cannot serve this purpose: calculateFiguresOfMerit returns a
    // hard 0 as soon as more than `impurity_peaks` lines go unindexed, so with
    // the usual setting of 1 impurity on a 47-peak pattern essentially every
    // candidate scores 0 and the figure carries no information. The raw count
    // is smooth and is the thing that actually distinguishes a cell that
    // explains the pattern from one that has been tuned to fit 20 lines.
    let nIdxAll = 0;
    for (let i = 0; i < nAll; i++) {
        const p = allPeaks[i];
        const j = binarySearchClosest(qCalc, p.q);
        if (j >= 0 && j < qCalc.length && Math.abs(p.q - qCalc[j]) < tolFn(p.original_index)) nIdxAll++;
    }

    return {
        m20: f20.m20, fN_20: f20.fN, n_20: n20,
        m_all: fAll.m20, fN_all: fAll.fN, n_all: nAll,
        n_idx_all: nIdxAll,
    };
}
// One round of the search: enumerate candidate labellings around `sol`, refit,
// and return the surviving children ranked best-first. Posts nothing.
function _swapRound(sol, data, state, cfg) {
    const { wavelength, tth_error, refineZero, impurity_peaks, max_volume } = data;
    const {
        peaks_sorted_by_q, original_indices, tth_obs_rad,
        N_FOR_M20, min_m20, q_max, d_min,
    } = state;

    const system = sol && sol.system;
    const MIN_INDEXED = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 }[system];
    if (!MIN_INDEXED) return [];

    const nAll = peaks_sorted_by_q.length;
    const need = MIN_INDEXED + (refineZero ? 1 : 0);
    if (nAll < need) return [];

    // ---- 1. parent line list: generated ONCE for the whole search ----------
    const lines = generateHKL_for_worker(sol, q_max, d_min, wavelength);
    if (lines.length < 2) return [];
    const qL = new Float64Array(lines.length);
    for (let i = 0; i < lines.length; i++) qL[i] = lines[i].q;

    // ---- 2. per-peak observables, precomputed once -------------------------
    const z = sol.zero_correction || 0;
    const qc = new Float64Array(nAll);      // zero-corrected observed q (for matching)
    const rhs = new Float64Array(nAll);     // LS right-hand side
    const tol = new Float64Array(nAll);
    const zcol = new Float64Array(nAll);    // zero-shift design column
    const tthRad = new Array(nAll);
    for (let i = 0; i < nAll; i++) {
        const t = peaks_sorted_by_q[i].tth;
        const tr = t * RAD;
        tthRad[i] = tr;
        qc[i] = (4 * Math.sin((t - z) * RAD / 2) ** 2) / (wavelength ** 2);
        // Mirrors refineAndTestSolution: raw q when the zero is a fitted column,
        // zero-corrected q when the zero is held fixed. (refineWithManualHkl
        // uses raw q unconditionally, which silently drops the parent's zero
        // whenever refineZero is off.)
        rhs[i] = refineZero ? (4 * Math.sin(tr / 2) ** 2) / (wavelength ** 2) : qc[i];
        tol[i] = get_q_tolerance(original_indices[i], tth_obs_rad, wavelength, tth_error);
        zcol[i] = (2 / (wavelength ** 2)) * Math.sin(tr);
    }
    const wts = ls_weights_for_2theta(tthRad);
    const tolFn = (idx) => get_q_tolerance(idx, tth_obs_rad, wavelength, tth_error);

    // ---- 3. baseline nearest-line labelling --------------------------------
    const base = new Int32Array(nAll).fill(-1);
    let nIndexed = 0;
    for (let i = 0; i < nAll; i++) {
        const j = binarySearchClosest(qL, qc[i]);
        if (j >= 0 && j < lines.length && Math.abs(qc[i] - qL[j]) < tol[i]) { base[i] = j; nIndexed++; }
    }
    if (nIndexed < need) return [];

    // ---- 4. candidate labels: EVERY line inside the error window -----------
    const nScan = Math.min(cfg.MAX_PEAKS, nAll);
    const free = [];
    for (let i = 0; i < nScan; i++) {
        if (base[i] < 0) continue;
        const win = tol[i] * cfg.ALT_WINDOW;
        const raw = [];
        for (let j = base[i]; j >= 0; j--) { if (qc[i] - qL[j] > win) break; raw.push(j); }
        for (let j = base[i] + 1; j < lines.length; j++) { if (qL[j] - qc[i] > win) break; raw.push(j); }
        if (raw.length < 2) continue;

        raw.sort((x, y) => Math.abs(qL[x] - qc[i]) - Math.abs(qL[y] - qc[i]));

        // Two labels with the same design row are the same equation. Keep one.
        const seen = new Set(), keep = [];
        for (const j of raw) {
            const r = getLSDesignRow([lines[j].h, lines[j].k, lines[j].l], system);
            if (!r) continue;
            const sig = r.join('|');
            if (seen.has(sig)) continue;
            seen.add(sig); keep.push(j);
            if (keep.length >= cfg.MAX_ALT) break;
        }
        if (keep.length < 2) continue;

        free.push({ i, cands: keep, pen: keep.map(j => ((qL[j] - qc[i]) / tol[i]) ** 2) });
    }
    if (!free.length) return [];

    // Most ambiguous peaks first: smallest penalty gap between best and runner-up.
    free.sort((A, B) => (A.pen[1] - A.pen[0]) - (B.pen[1] - B.pen[0]));
    const dims = free.slice(0, cfg.MAX_FREE);
    const D = dims.length;

    // ---- 5. best-first walk of the full product space ----------------------
    // Node = vector of per-peak candidate ranks. Start = all-nearest. Expanding
    // by incrementing one coordinate generates every assignment exactly once,
    // in non-decreasing total-penalty order.
    const heap = new _SwapHeap();
    const visited = new Set();
    const start = new Uint8Array(D);
    heap.push({ cost: dims.reduce((s, d) => s + d.pen[0], 0), r: start });
    visited.add(start.join(','));

    const labels = new Int32Array(nAll);
    const cellSeen = new Set();
    const results = [];
    let fits = 0;

    while (heap.size && fits < cfg.MAX_FITS) {
        const node = heap.pop();
        const r = node.r;

        for (let d = 0; d < D; d++) {
            if (r[d] + 1 >= dims[d].cands.length) continue;
            const nr = Uint8Array.from(r);
            nr[d]++;
            const key = nr.join(',');
            if (visited.has(key)) continue;
            visited.add(key);
            heap.push({ cost: node.cost - dims[d].pen[r[d]] + dims[d].pen[nr[d]], r: nr });
        }

        let changed = false;
        labels.set(base);
        for (let d = 0; d < D; d++) {
            if (r[d] !== 0) changed = true;
            labels[dims[d].i] = dims[d].cands[r[d]];
        }
        if (!changed) continue;   // the all-nearest labelling is the parent

        fits++;
        const cell = _swapCheapFit(labels, {
            lines, system, refineZero, rhs, wts, zcol, nAll, need,
            maxVolume: max_volume, z, qc, qL,
        });
        if (!cell) continue;

        const key = getSolutionKey(cell);
        if (!key || cellSeen.has(key)) continue;
        cellSeen.add(key);
        cell._swaps = dims
            .map((d, k) => (r[k] === 0 ? null : {
                tth: peaks_sorted_by_q[d.i].tth,
                from: `(${lines[base[d.i]].h},${lines[base[d.i]].k},${lines[base[d.i]].l})`,
                to: `(${lines[d.cands[r[k]]].h},${lines[d.cands[r[k]]].k},${lines[d.cands[r[k]]].l})`,
            }))
            .filter(Boolean);
        results.push(cell);
    }

    // ---- 6. stage 2: full M20 for the best cheap fits only -----------------
    results.sort((A, B) => A._resid - B._resid);
    const parentM20 = isFinite(sol.m20) ? sol.m20 : 0;
    const parentFoM = _swapEvalFoM(sol, {
        wavelength, qMax: q_max, peaks: peaks_sorted_by_q,
        N20: N_FOR_M20, impurityPeaks: impurity_peaks, tolFn,
    });
    const parentNIdx = parentFoM ? parentFoM.n_idx_all : 0;
    const keepers = [];

    for (let i = 0; i < results.length && i < cfg.MAX_EVALS; i++) {
        const cell = results[i];
        const fom = _swapEvalFoM(cell, {
            wavelength, qMax: q_max, peaks: peaks_sorted_by_q,
            N20: N_FOR_M20, impurityPeaks: impurity_peaks, tolFn,
        });
        if (!fom) continue;
        Object.assign(cell, fom);
        if (!(cell.m20 > min_m20) || !(cell.m20 >= parentM20 * cfg.KEEP_RATIO)) continue;
        // M20 is computed on 20 peaks and is BLIND to peaks 21..N. A relabelling
        // can inflate M20 while wrecking the rest of the pattern -- measured on a
        // real orthorhombic pattern: M20 49 -> 95 while the number of indexed
        // lines fell. Never accept a child that explains materially LESS of the
        // observed pattern than its parent did.
        if (cell.n_idx_all < parentNIdx - cfg.LOSE_SLACK) continue;
        keepers.push(cell);
    }

    // Rank by how much of the pattern is explained FIRST, M20 second. Ranking on
    // M20 alone is what let a 20-line-tuned cell outrank the cell that indexes
    // the whole pattern.
    keepers.sort((A, B) => (B.n_idx_all - A.n_idx_all) || (B.m20 - A.m20));
    keepers._diag = { D, fits, distinct: results.length, parentNIdx, parentM20 };
    return keepers;
}
/**
 * Combinatorial relabelling search around one refined solution.
 *
 * Runs in ROUNDS. This matters more than any single-round budget: fixing one
 * crossing MOVES the cell, which changes which calculated lines sit near which
 * observed peaks -- so the second round enumerates a genuinely different
 * candidate set that the first round could not see at any budget. Measured on a
 * real PbSO4 pattern: round 1 lifts the best cell from M20 43.7 to 49.4, and a
 * second round from there reaches 92.4. A single round stops at 49.4, which is
 * exactly the plateau the one-shot version produced.
 *
 * @returns {number} how many new/improved solutions were posted
 */
function combinatorialSwapSearch(sol, data, state, postMessage_func, cfgIn) {
    const cfg = Object.assign({}, SWAP_CFG, cfgIn || {});
    const { foundSolutions, foundSolutionMap } = state;
    const system = sol && sol.system;

    let current = sol;
    let best = null;
    const chain = [];
    const diags = [];

    for (let round = 0; round < Math.max(1, cfg.ROUNDS); round++) {
        const cands = _swapRound(current, data, state, cfg);
        if (cands._diag) diags.push(cands._diag);
        if (!cands.length) break;

        const top = cands[0];
        chain.push(...cands.slice(0, cfg.MAX_POST));
        best = top;

        // Re-centre only on a real gain, otherwise every parent pays for four
        // rounds of nothing.
        const gained = (top.n_idx_all > (current.n_idx_all || 0)) ||
                       (top.m20 > (current.m20 || 0) * cfg.ROUND_GAIN);
        if (!gained) break;
        current = top;
    }
    if (!chain.length) return 0;

    // Post the best few across all rounds.
    chain.sort((A, B) => (B.n_idx_all - A.n_idx_all) || (B.m20 - A.m20));
    let posted = 0;
    for (const cell of chain) {
        if (posted >= cfg.MAX_POST) break;
        const key = getSolutionKey(cell);
        // Same rule as refineAndTestSolution: an unkeyable cell is posted but
        // never filed, rather than sharing the `undefined` slot with every
        // other unkeyable cell.
        const existing = key ? foundSolutionMap.get(key) : undefined;
        if (existing && !(cell.m20 > existing.m20)) continue;

        try { cell.errors = propagateErrors(system, cell._fit, cell); } catch (e) { cell.errors = null; }
        cell.autoSwaps = cell._swaps;
        cell.manualSwaps = sol.manualSwaps || [];
        // _fit holds the covariance matrix; it must not survive the structured
        // clone to the main thread.
        delete cell._fit; delete cell._resid; delete cell._swaps;

        if (key) {
            if (existing) foundSolutions[existing.index] = cell;
            else foundSolutions.push(cell);
            foundSolutionMap.set(key, { m20: cell.m20, index: existing ? existing.index : foundSolutions.length - 1 });
        }

        postMessage_func({ type: 'solution', payload: cell });
        posted++;
    }

    if (cfg.DEBUG && diags.length) {
        const d0 = diags[0];
        console.log(`[swap] ${system} parent M20=${d0.parentM20.toFixed(2)}: ` +
            `${diags.length} round(s), ${diags.reduce((a, x) => a + x.fits, 0)} fits | ` +
            `lines indexed ${d0.parentNIdx}/${state.peaks_sorted_by_q.length} -> ` +
            `${best ? best.n_idx_all : d0.parentNIdx} | ` +
            `M20 ${d0.parentM20.toFixed(2)} -> ${best ? best.m20.toFixed(2) : '-'} | ${posted} posted`);
    }
    return posted;
}
