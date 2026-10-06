// js/crystallography/refine.js
// Candidate refinement (refineAndTestSolution) and the CPU cubic / tetragonal / hexagonal searches.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// --- CORE REFINEMENT AND ANALYSIS FUNCTION ---
function refineAndTestSolution( initialParams, data, state, postMessage_func ) {
    const { wavelength, tth_error, max_volume, impurity_peaks, refineZero } = data;
    const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q, N_FOR_M20, min_m20, q_max, d_min, foundSolutions, foundSolutionMap } = state;

    // --- Single exit function ---
    // Only a real solution is ever posted. This function is called synchronously
    // by every caller (indexCubic/indexTetragonal.../findTransformedSolutions and
    // the refinement worker's runOneCell), none of which await a reply, so the
    // former `postMessage_func({})` "resolve" on rejection did nothing but
    // structure-clone an empty object to the main thread on EVERY rejected trial
    // (~10^5-10^6 per run in the CPU indexing path, where postMessage_func IS
    // self.postMessage). The main thread matched no branch for it. Dropped.
    const exitFunction = (payload = null) => {
        if (payload) {
            postMessage_func({ type: 'solution', payload: payload });
        }
    };

    // --- Rejection accounting -------------------------------------------
    // Optional: present only on the CPU indexing worker's state (see the
    // self.onmessage handler at the bottom of this file). A run that ends with
    // zero solutions previously had nothing to say about WHY -- every trial
    // returned through the same silent exitFunction(). These counters are the
    // difference between "no solutions" and "every cell was bigger than your
    // Max Volume". Cost is one property read plus an increment on paths that
    // are already returning, which is not measurable against the least-squares
    // fit that dominates this function.
    const diag = state.diag;
    const bump = (k) => { if (diag) diag[k] = (diag[k] || 0) + 1; };
    if (diag) diag.trials = (diag.trials || 0) + 1;

    // console.log("REFINE: Starting refinement for", initialParams);

    if (!initialParams || !initialParams.system) {
        // console.log("REFINE: Rejected - no params");
        bump('badParams');
        return exitFunction();
    }
    
    const { system } = initialParams;

    const min_lp_check = 2.0, max_lp_check = 50.0;
    const axes_to_check = [initialParams.a, initialParams.b ?? initialParams.a, initialParams.c ?? initialParams.a];
    const angles_to_check = [initialParams.alpha ?? 90, initialParams.beta ?? 90, initialParams.gamma ?? 90];

    // Explicitly validate both linear dimensions and angles against NaN/Infinity and physical limits
    if (axes_to_check.some(p => !isFinite(p) || p < min_lp_check || p > max_lp_check) ||
        angles_to_check.some(a => !isFinite(a) || a < 10.0 || a > 170.0)) {
        bump('axisRange');
        return exitFunction(); 
    }
    
    const initial_cell_volume = getVolume(initialParams);
    // !(vol >= 20) safely catches NaN, 0, and negative volumes
    if (!(initial_cell_volume >= 20) || !(initial_cell_volume <= max_volume) || !isFinite(initial_cell_volume)) {
        // Split the two directions: "every cell was too big" points at Max
        // Volume, "every cell was too small" points at the peak list. Lumping
        // them together names neither.
        if (isFinite(initial_cell_volume)) {
            if (initial_cell_volume > max_volume) {
                bump('volumeTooLarge');
                if (diag) diag.volOverMin = Math.min(diag.volOverMin ?? Infinity, initial_cell_volume);
            } else {
                bump('volumeTooSmall');
            }
        } else {
            bump('volumeInvalid');
        }
        return exitFunction();
    }
    if (diag) {
        diag.volPassed = (diag.volPassed || 0) + 1;
        diag.volMin = Math.min(diag.volMin ?? Infinity, initial_cell_volume);
        diag.volMax = Math.max(diag.volMax ?? -Infinity, initial_cell_volume);
    }


    const local_get_q_tolerance = (idx) => get_q_tolerance(idx, tth_obs_rad, wavelength, tth_error);
    const min_indexed = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 };
    let final_solution_to_post = null;
    const n_20 = Math.min(N_FOR_M20, peaks_sorted_by_q.length);
    const n_all = peaks_sorted_by_q.length;
    const all_possible_reflections = generateHKL_for_worker(initialParams, q_max, d_min, wavelength);
    
    if (all_possible_reflections.length === 0) {
        // console.log("REFINE: Rejected - could not generate HKLs");
        bump('noReflections');
        return exitFunction();
    }

    // --- REFINEMENT LOGIC ---

    {
        // --- REFINEMENT ---
        // refineZero=true  (PATH A): add a zero-shift column to the LS design
        //   matrix and do TWO rounds of pairing+fit with proper LS weighting:
        //     Round 1: pair with z=0 assumption → fit cell+z → get z_estimate
        //     Round 2: pair using q values corrected by z_estimate → re-fit
        // refineZero=false (PATH B): fit the cell with a FIXED zero (no extra
        //   column, single pairing round, no zero correction applied). This
        //   restores the non-zero-refined path that used to live here — without
        //   it, unchecking "Refine Zero" made this function post nothing at all.

        const pair_and_fit = (zero_corr_deg) => {
            const indexed_pairs = [];
            const peak_indices = [];
            const used = new Set();
            for (let i = 0; i < n_all; i++) {
                const original_idx = original_indices[i];
                let q_to_match;
                if (Math.abs(zero_corr_deg) > 1e-9) {
                    // zero_corr_deg is a shift in 2-theta DEGREES, so it converts to
                    // radians directly. The stray factor of 2 that used to sit here made
                    // round 2 re-pair against positions overshot by a full extra zero
                    // offset, so the second fit was pulled away from the first instead of
                    // converging on it. Every other consumer of zero_correction in this
                    // file subtracts it exactly once; this line now agrees with them.
                    const corrected_tth_rad = tth_obs_rad[original_idx] - zero_corr_deg * RAD;
                    q_to_match = (4 * Math.sin(corrected_tth_rad / 2) ** 2) / (wavelength ** 2);
                } else {
                    q_to_match = q_obs[i];
                }
                const tolerance = local_get_q_tolerance(original_idx);
                let best_match_idx = -1, min_diff = Infinity;
                let low = 0, high = all_possible_reflections.length - 1;
                while (low <= high) { let mid = Math.floor((low + high) / 2); if (all_possible_reflections[mid].q < q_to_match) low = mid + 1; else high = mid - 1; }
                for (let j = Math.max(0, high - 1); j <= Math.min(all_possible_reflections.length - 1, low + 1); j++) {
                    const diff = Math.abs(all_possible_reflections[j].q - q_to_match);
                    if (diff < min_diff) { min_diff = diff; best_match_idx = j; }
                }
                if (best_match_idx !== -1 && min_diff < tolerance && !used.has(best_match_idx)) {
                    const { h, k, l } = all_possible_reflections[best_match_idx];
                    indexed_pairs.push({ q_obs: q_obs[i], hkl: [h, k, l] });
                    peak_indices.push(original_idx);
                    used.add(best_match_idx);
                }
            }

            // How many observed peaks this trial cell could actually account
            // for. The single most useful number when nothing is found: if the
            // best cell in a whole run matched 4 of 25 peaks, the peak list or
            // the 2-theta error is wrong, not the search settings.
            if (diag) diag.bestIndexed = Math.max(diag.bestIndexed || 0, indexed_pairs.length);
            if (indexed_pairs.length < min_indexed[system]) { bump('tooFewIndexed'); return null; }

            const M = indexed_pairs.map(p => getLSDesignRow(p.hkl, system));
            const q_vec = indexed_pairs.map(p => p.q_obs);
            if (refineZero) {
                // Extra design column for the zero-shift parameter. Omitted when
                // refineZero is false so the fit has exactly the cell params.
                M.forEach((row, i) => {
                    const tth_rad = tth_obs_rad[peak_indices[i]];
                    row.push((2 / (wavelength ** 2)) * Math.sin(tth_rad));
                });
            }
            const tth_rads_for_rows = peak_indices.map(idx => tth_obs_rad[idx]);
            const ls_weights = ls_weights_for_2theta(tth_rads_for_rows);

            const fit = solveLeastSquares(M, q_vec, ls_weights);
            if (!fit || !fit.solution) { bump('fitFailed'); return null; }
            return { fit, indexed_pairs, peak_indices };
        };

        // Round 1
        let result = pair_and_fit(0);
        // Round 2 (only when refining zero): re-pair using the round-1 estimate.
        // With refineZero=false there is no zero-shift param to iterate on.
        if (result && refineZero) {
            const z1_deg = result.fit.solution[result.fit.solution.length - 1] * DEG;
            if (Math.abs(z1_deg) > 1e-4) {
                const result2 = pair_and_fit(z1_deg);
                if (result2) result = result2; // accept refined pairing
            }
        }

        if (result) {
            const fitResult_with_zero_final = result.fit;
            const refined_cell = extractCellFromFit(fitResult_with_zero_final.solution, system);

            if (!refined_cell) bump('extractFailed');
            if (refined_cell) {
                // Only attach a zero_correction when we actually refined one.
                // Leaving it undefined lets applyFinalSieve treat this as a
                // 0-DoF (fixed-zero) model, distinct from a zero-refined one.
                if (refineZero) {
                    refined_cell.zero_correction = fitResult_with_zero_final.solution[fitResult_with_zero_final.solution.length - 1] * DEG;
                }
                refined_cell.volume = getVolume(refined_cell);
                
                // q_only fast path: a sorted, deduped Float64Array straight out of
                // the generator. The old line built a full reflection object
                // (tth, d, h, k, l) for every line, then a Set, then sorted again,
                // purely to obtain the q list -- on the hot path of every accepted
                // refinement. The two dedup rules are now provably identical (see
                // generateHKL_for_analysis), so the result is unchanged.
                const q_calc_sorted_refined = generateQArray_for_worker(refined_cell, q_max, wavelength);
                
                const peaks_for_merit_20_refined = [];
                for (let i = 0; i < n_20; i++) {
                    const original_peak = peaks_sorted_by_q[i];
                    const corrected_tth_deg = original_peak.tth - (refined_cell.zero_correction || 0);
                    const corrected_tth_rad = corrected_tth_deg * RAD;
                    const corrected_q = (4 * Math.sin(corrected_tth_rad / 2)**2) / (wavelength**2);
                    peaks_for_merit_20_refined.push({ ...original_peak, q: corrected_q, tth: corrected_tth_deg });
                }
                
                const { m20: final_m20, fN: final_fN_20 } = calculateFiguresOfMerit(q_calc_sorted_refined, peaks_for_merit_20_refined, impurity_peaks, local_get_q_tolerance, wavelength);
                
                // Best M(20) any refined cell reached, kept even when it fails
                // the threshold. A run whose best was 1.9 against a 2.0 floor is
                // a completely different problem from one whose best was 0.2.
                if (diag && isFinite(final_m20)) {
                    diag.bestM20 = Math.max(diag.bestM20 ?? -Infinity, final_m20);
                    diag.refined = (diag.refined || 0) + 1;
                }
                if (final_m20 <= min_m20) bump('lowM20');

                if (final_m20 > min_m20) {
                    const peaks_for_merit_all_refined = [];
                    for (let i = 0; i < n_all; i++) {
                        const original_peak = peaks_sorted_by_q[i]; const corrected_tth_deg = original_peak.tth - (refined_cell.zero_correction || 0);
                        const corrected_tth_rad = corrected_tth_deg * RAD; const corrected_q = (4 * Math.sin(corrected_tth_rad / 2)**2) / (wavelength**2);
                        peaks_for_merit_all_refined.push({ ...original_peak, q: corrected_q, tth: corrected_tth_deg });
                    }
                    const { m20: final_m_all, fN: final_fN_all } = calculateFiguresOfMerit(q_calc_sorted_refined, peaks_for_merit_all_refined, impurity_peaks, local_get_q_tolerance, wavelength);
                    
                    refined_cell.m20 = final_m20; refined_cell.fN_20 = final_fN_20; refined_cell.n_20 = n_20;
                    refined_cell.m_all = final_m_all; refined_cell.fN_all = final_fN_all; refined_cell.n_all = n_all;
                    refined_cell.errors = propagateErrors(system, fitResult_with_zero_final, refined_cell);
                    final_solution_to_post = refined_cell;
                }
            }
        }
    }
    
    // 4. --- POST THE SOLUTION ---
    if (final_solution_to_post) {
        const key = getSolutionKey(final_solution_to_post);
        // An unkeyable cell (unrecognised system) cannot be deduped, and must
        // not be filed under the shared `undefined` slot -- that made unrelated
        // cells shadow each other. Post it and skip the ledger: no dedup is
        // better than wrong dedup.
        if (!key) {
            return exitFunction(final_solution_to_post);
        }
        const existing = foundSolutionMap.get(key);
        
        if (!existing || final_solution_to_post.m20 > existing.m20) {
            
            if (existing) { 
                foundSolutions[existing.index] = final_solution_to_post; 
            } else { 
                foundSolutions.push(final_solution_to_post); 
            }
            foundSolutionMap.set(key, { 
                m20: final_solution_to_post.m20, 
                index: existing ? existing.index : foundSolutions.length - 1 
            });
            
            return exitFunction(final_solution_to_post);
        }
    }

    // 5. --- FINAL EXIT ---
    // If we get here, no solution was posted.
    return exitFunction();
}
// --- cpu index, with status, hmax changed to 40 for cubic
function indexCubic(data, state, postMessage_func) {
    const { peaks } = data; const { q_obs, refineAndTestSolution } = state;
    const h_max = 40;
    const peak_depth = Math.min(peaks.length, 12);
    const hkls = []; for (let h = 1; h <= h_max; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= k; l++) { if (!h && !k && !l) continue; hkls.push([h,k,l]); }
    
    
    const totalTrialsToRun = peak_depth * hkls.length;
    let totalTrialsCompleted = 0;
        
let trialsBatch = 0; let lastReportTime = performance.now();
    for (let i = 0; i < peak_depth; i++) {
        for (const hkl of hkls) {
            refineAndTestSolution({ a: Math.sqrt((hkl[0]*hkl[0] + hkl[1]*hkl[1] + hkl[2]*hkl[2]) / q_obs[i]), system: 'cubic' });
            trialsBatch++; 

if (trialsBatch % 5000 === 0 && performance.now() - lastReportTime >= 50) { // Report at most every 50ms
                postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch }); 
                totalTrialsCompleted += trialsBatch;
                const progress = (totalTrialsCompleted / totalTrialsToRun) * 80; // 80% reserved
                postMessage_func({ type: 'progress', payload: progress });
                trialsBatch = 0; 
                lastReportTime = performance.now();
            }

        }
        
    }
    if (trialsBatch > 0) postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch });
}
function indexTetragonalOrHexagonal(data, state, postMessage_func, system) {
    const { peaks } = data; const { q_obs, refineAndTestSolution } = state;
    const max_hkl = 12, i_depth = Math.min(12, peaks.length), j_depth = Math.min(12, peaks.length);
    const hkls = []; for (let h = 0; h <= max_hkl; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= max_hkl; l++) { if (!h && !k && !l) continue; hkls.push([h,k,l]); }

    
    let totalPeakCombos = 0;
    for (let i = 0; i < i_depth; i++) {
        for (let j = i + 1; j < j_depth; j++) {
            totalPeakCombos++;
        }
    }
    const totalTrialsToRun = totalPeakCombos * hkls.length * hkls.length;
    let totalTrialsCompleted = 0;
    

    let trialsBatch = 0; let lastReportTime = performance.now();

    for (let i = 0; i < i_depth; i++) {
        for (let j = i + 1; j < j_depth; j++) {
            for (const hkl1 of hkls) {
                const l1 = hkl1[2]; const S1 = system === 'tetragonal' ? hkl1[0] * hkl1[0] + hkl1[1] * hkl1[1] : hkl1[0] * hkl1[0] + hkl1[0] * hkl1[1] + hkl1[1] * hkl1[1];
                for (const hkl2 of hkls) {
                    const l2 = hkl2[2]; const S2 = system === 'tetragonal' ? hkl2[0] * hkl2[0] + hkl2[1] * hkl2[1] : hkl2[0] * hkl2[0] + hkl2[0] * hkl2[1] + hkl2[1] * hkl2[1];
                    
                    trialsBatch++; 

                    if (trialsBatch % 5000 === 0 && performance.now() - lastReportTime >= 50) { // Report at most every 50ms
                postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch }); 
                totalTrialsCompleted += trialsBatch;
                const progress = (totalTrialsCompleted / totalTrialsToRun) * 80; // 80% reserved
                postMessage_func({ type: 'progress', payload: progress });
                trialsBatch = 0; 
                lastReportTime = performance.now();
            }

                    const det = S1 * l2 * l2 - S2 * l1 * l1;
                    if (Math.abs(det) < 1e-6) continue;
                    const a_term_inv = (q_obs[i] * l2 * l2 - q_obs[j] * l1 * l1) / det, c_term_inv = (q_obs[j] * S1 - q_obs[i] * S2) / det;
                   
                    if (a_term_inv > 0 && c_term_inv > 0) {
                        const a = system === 'tetragonal' ? 1 / Math.sqrt(a_term_inv) : Math.sqrt(4 / (3 * a_term_inv));
                        const c = 1 / Math.sqrt(c_term_inv);
                        const min_lp = 2.0, max_lp = 50.0;
                        if (a < min_lp || a > max_lp || c < min_lp || c > max_lp || (a != a) || (c != c)) {
                            continue;
                        }
                        refineAndTestSolution({ a: a, c: c, system });
                    }
                }
            }
        }
        
    }
    if (trialsBatch > 0) postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch });
}
