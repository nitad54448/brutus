// js/crystallography/transforms.js
// Transformed-cell search over found solutions (super/sub/equivalent cells).
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

function findTransformedSolutions(initialSolutions, data, state, postMessage_func) {
    const { allowedSystems } = data;
    const { refineAndTestSolution, q_obs, original_indices, N_FOR_M20, q_max, d_min, tth_obs_rad, peaks_sorted_by_q } = state;
    const { wavelength, tth_error } = data;
    const cellTransforms = [ { P: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] }, { P: [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]] }, { P: [[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]] }, { P: [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]] }, { P: [[1, 0, 0], [0, 0.5, 0], [0, 0, 1]] }, { P: [[1, 0, 0], [0, 1, 0], [0, 0, 0.5]] }, { P: [[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]] } ];
    // refineAndTestSolution and the swap search both write back into
    // state.foundSolutions -- which IS this array. Appends are invisible to
    // forEach (it caches length up front), but a REPLACEMENT at an index the
    // loop has not reached yet silently substitutes a different cell for `sol`
    // mid-pass. Iterate a snapshot so the input set is fixed for the whole run.
    const parents = initialSolutions.slice();
    const totalSolutions = parents.length; if (totalSolutions === 0) return;
    const local_get_q_tolerance = (idx) => get_q_tolerance(idx, tth_obs_rad, wavelength, tth_error);

    // Relabelling costs ~3-30 ms per parent. Running it on every one of several
    // thousand GPU candidates is what made the old pass a serial bottleneck in
    // the post_process step, and a cell scoring M20 = 2.1 is not worth it
    // anyway. Only the best TOP_N parents get a swap search.
    const swapAllowed = new Set(
        parents
            .map((s, i) => ({ i, m: (s && isFinite(s.m20)) ? s.m20 : 0 }))
            .sort((x, y) => y.m - x.m)
            .slice(0, (typeof SWAP_CFG !== 'undefined' ? SWAP_CFG.TOP_N : 40))
            .map(x => x.i)
    );

    const stats = {
        parents: totalSolutions, swapEligible: swapAllowed.size,
        swapRan: 0, swapPosted: 0, swapErrors: 0, solutionErrors: 0,
        bestBefore: parents.reduce((m, x) => (x && isFinite(x.m20) && x.m20 > m) ? x.m20 : m, 0),
        bestAfter: 0,
    };

    parents.forEach((sol, index) => {
      try {
        // === NIGGLI REDUCTION & SYMMETRY SQUEEZE

        try {
            // 1. "Squeeze" the cell into its most basic form
            const niggliResult = reduceToNiggliCell(sol);
            const nCell = niggliResult.cell;
            
            // 2. Use the "Label Maker" to find the "true" symmetry of the squeezed cell.
            // Tolerance 0.25 (A / deg) is deliberately loose to catch pseudo-symmetries
            // in experimental powder data.
            const idealSymmetry = getSymmetry(nCell.a, nCell.b, nCell.c, nCell.alpha, nCell.beta, nCell.gamma, 0.25);

            // 3. Compare the "true" label to the original label
            const symmetryOrder = { 'cubic': 6, 'hexagonal': 5, 'tetragonal': 4, 'orthorhombic': 3, 'monoclinic': 2, 'triclinic': 1 };
            
            // 4. If the "true" symmetry is *higher* (e.g., we found a 'cubic' disguised as 'orthorhombic'),
            //    AND the user *wants* to search for that higher symmetry...
            if (symmetryOrder[idealSymmetry] > symmetryOrder[sol.system] && allowedSystems.includes(idealSymmetry)) {
                
                let newTrialCell = { system: idealSymmetry };
                
                // 5. Create a *new* trial cell based on the "true" symmetry
                switch (idealSymmetry) {
                    case 'cubic':
                        newTrialCell.a = (nCell.a + nCell.b + nCell.c) / 3.0; // Average the axes
                        break;
                    case 'tetragonal':
                    case 'hexagonal':
                        // Robustly find repeated axis ('a') and unique axis ('c') by closest pair
                        const diffAB = Math.abs(nCell.a - nCell.b);
                        const diffAC = Math.abs(nCell.a - nCell.c);
                        const diffBC = Math.abs(nCell.b - nCell.c);
                        if (diffAB <= diffAC && diffAB <= diffBC) { // a == b
                            newTrialCell.a = (nCell.a + nCell.b) / 2.0;
                            newTrialCell.c = nCell.c;
                        } else if (diffAC <= diffAB && diffAC <= diffBC) { // a == c
                            newTrialCell.a = (nCell.a + nCell.c) / 2.0;
                            newTrialCell.c = nCell.b;
                        } else { // b == c
                            newTrialCell.a = (nCell.b + nCell.c) / 2.0;
                            newTrialCell.c = nCell.a;
                        }
                        break;
                    case 'orthorhombic':
                        newTrialCell.a = nCell.a;
                        newTrialCell.b = nCell.b;
                        newTrialCell.c = nCell.c;
                        break;
                    case 'monoclinic':
                        newTrialCell.a = nCell.a;
                        newTrialCell.b = nCell.b;
                        newTrialCell.c = nCell.c;
                        newTrialCell.beta = nCell.beta; // Niggli cell will have alpha=gamma=90
                        break;
                }


                // 6. Send this new, "squeezed" cell to be re-tested

                refineAndTestSolution(newTrialCell);
            }
        } catch (e) {
            console.warn("Niggli-reduction post-processing failed for a solution:", e);
        }
        

        // --- Original Transform Logic 
        cellTransforms.forEach(tf => {
            try {
                const G = metricFromCell(sol); const Pt = transpose(tf.P); const Gprime = matMul(matMul(Pt, G), tf.P);
                const candCell = cellFromMetric(Gprime);
                const newSystem = getSymmetry(candCell.a, candCell.b, candCell.c, candCell.alpha, candCell.beta, candCell.gamma);
                if (allowedSystems.includes(newSystem)) refineAndTestSolution({ ...candCell, system: newSystem });
            } catch {}
        });
     


        // Tsend wave
        const theoretical_hkls = generateHKL_for_worker(sol, q_max, d_min, wavelength);
        const theoretical_q_array = theoretical_hkls.map(h => h.q); //map once, not every time, mod 13 07 2026

        // Zero-correct BEFORE matching. This loop used to pair the raw q_obs
        // against the calculated lines while every other consumer of
        // zero_correction subtracts it first, so on a zero-refined solution the
        // gcd sub-cell test below ran on a partly wrong assignment list.
        const z_gcd_deg = sol.zero_correction || 0;
        const nGcd = Math.min(N_FOR_M20, peaks_sorted_by_q.length);
        const indexedPeaks = [];
        for (let i = 0; i < nGcd; i++) {
             const tc_deg = peaks_sorted_by_q[i].tth - z_gcd_deg;
             const q_o = (4 * Math.sin(tc_deg * RAD / 2) ** 2) / (wavelength ** 2);
             const best_match_idx = binarySearchClosest(theoretical_q_array, q_o); // <--- AND REUSE IT HERE
             if (best_match_idx >= 0 && best_match_idx < theoretical_hkls.length && Math.abs(q_o - theoretical_hkls[best_match_idx].q) < local_get_q_tolerance(original_indices[i])){
                 indexedPeaks.push(theoretical_hkls[best_match_idx]);
             }
        }
        if (indexedPeaks.length > 5) {
            const h_div = gcdOfList(indexedPeaks.map(p => Math.abs(p.h)).filter(h => h > 0));
            const k_div = gcdOfList(indexedPeaks.map(p => Math.abs(p.k)).filter(k => k > 0));
            const l_div = gcdOfList(indexedPeaks.map(p => Math.abs(p.l)).filter(l => l > 0));
            if (h_div > 1 || k_div > 1 || l_div > 1) {
                const candCell = { ...sol, a: sol.a/h_div, b: (sol.b??sol.a)/k_div, c:(sol.c??sol.a)/l_div };
                const newSystem = getSymmetry(candCell.a, candCell.b, candCell.c, candCell.alpha, candCell.beta, candCell.gamma);
                if (allowedSystems.includes(newSystem)) refineAndTestSolution({ ...candCell, system: newSystem });
            }
        }
        if (sol.system === 'orthorhombic' && allowedSystems.includes('hexagonal')) {
            const axes = { a: sol.a, b: sol.b, c: sol.c }; const pairs = [['a','b','c'], ['a','c','b'], ['b','c','a']];
            pairs.forEach(([ax1, ax2, unique_ax]) => {
                if (Math.abs(axes[ax2] / axes[ax1] / Math.sqrt(3) - 1) < 0.03) {
                    refineAndTestSolution({ system: 'hexagonal', a: axes[ax1], c: axes[unique_ax], beta: 90, gamma: 120 });
                }
            });
        }
        


        // --- COMBINATORIAL SWAP SEARCH ------------------------------------
        // Replaces the old "swap fishing" pass, which was not combinatorial: it
        // tried one peak relabelled at a time (max 2 alternatives each) plus a
        // transposition of the 3 closest peak pairs, 12 refits in total. Two
        // independent mislabels, or a crossing that drags a third line with it,
        // were structurally unreachable. combinatorialSwapSearch enumerates
        // EVERY calculated line inside each peak's error window and searches the
        // full product of those per-peak candidate sets, best-first.
        if (swapAllowed.has(index)) {
            try {
                stats.swapRan++;
                // data.swapCfg lets the caller widen the search when it knows it
                // is handing over few, already-deduplicated parents.
                stats.swapPosted += (combinatorialSwapSearch(sol, data, state, postMessage_func, data.swapCfg) || 0);
            } catch (e) {
                stats.swapErrors++;
                console.warn("Swap search failed:", e && e.message, e && e.stack);
            }
        }

        // The Monte-Carlo cell polish is no longer run automatically here.
        // It is invoked on demand from the solutions context menu ("Refine MC"),
        // which lets the user choose how many solutions / iterations / restarts
        // to spend rather than paying ~0.3 s on every candidate.

      } catch (err) {
        // One bad solution must not abort the whole pass. Until now a throw
        // anywhere in the unguarded middle of this body -- the sub-cell gcd
        // test, the orthorhombic->hexagonal test, generateHKL_for_worker on a
        // degenerate cell -- propagated out of forEach, out of the worker's
        // onmessage, and killed the post-process worker outright. main_app's
        // onerror handler simply resolved the promise, so the run finished with
        // NO transformed and NO swapped solutions and printed nothing anywhere.
        stats.solutionErrors++;
        console.warn(`[post-process] solution ${index} (${sol && sol.system}) failed:`,
                     err && err.message, err && err.stack);
      }

        const progress = 80 + ((index + 1) / totalSolutions) * 15;
        postMessage_func({ type: 'progress', payload: progress });
    });

    stats.bestAfter = (state.foundSolutions || [])
        .reduce((m, x) => (x && isFinite(x.m20) && x.m20 > m) ? x.m20 : m, stats.bestBefore);
    return stats;
}
