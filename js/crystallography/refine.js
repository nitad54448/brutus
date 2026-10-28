// js/crystallography/refine.js
// Candidate refinement (refineAndTestSolution) and the CPU cubic / tetragonal / hexagonal searches.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (index.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// --- CORE REFINEMENT AND ANALYSIS FUNCTION ---
function refineAndTestSolution( initialParams, data, state, postMessage_func ) {
    const { wavelength, tth_error, max_volume, impurity_peaks, refineZero } = data;
    const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q, N_FOR_M20, min_m20, q_max, d_min, foundSolutions, foundSolutionMap } = state;

    // --- Single exit function ---
    // Only a real solution is ever posted.
    const exitFunction = (payload = null) => {
        if (payload) {
            postMessage_func({ type: 'solution', payload: payload });
        }
    };

    // --- Rejection accounting -------------------------------------------
    const diag = state.diag;
    const bump = (k) => { if (diag) diag[k] = (diag[k] || 0) + 1; };
    if (diag) diag.trials = (diag.trials || 0) + 1;

    if (!initialParams || !initialParams.system) {
        bump('badParams');
        return exitFunction();
    }
    
    const { system } = initialParams;

    const limits = getPhysicalLimits(data);
    const min_lp_check = limits.minAxis, max_lp_check = limits.maxAxis;
    const axes_to_check = [initialParams.a, initialParams.b ?? initialParams.a, initialParams.c ?? initialParams.a];
    const angles_to_check = [initialParams.alpha ?? 90, initialParams.beta ?? 90, initialParams.gamma ?? 90];

    // Explicitly validate both linear dimensions and angles against NaN/Infinity and physical limits
    if (axes_to_check.some(p => !isFinite(p) || p < min_lp_check || p > max_lp_check) ||
        angles_to_check.some(a => !isFinite(a) || a < limits.minAngle || a > limits.maxAngle)) {
        bump('axisRange');
        return exitFunction(); 
    }
    
    const initial_cell_volume = getVolume(initialParams);
    // !(vol >= 20) safely catches NaN, 0, and negative volumes
    if (!(initial_cell_volume >= limits.minVolume) || !(initial_cell_volume <= max_volume) || !isFinite(initial_cell_volume)) {
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
    const min_indexed = BRUTUS_LIMITS.search.minIndexed;
    let final_solution_to_post = null;
    const n_20 = Math.min(N_FOR_M20, peaks_sorted_by_q.length);
    const n_all = peaks_sorted_by_q.length;
    const all_possible_reflections = generateHKL_for_worker(initialParams, q_max, d_min, wavelength);
    
    if (all_possible_reflections.length === 0) {
        bump('noReflections');
        return exitFunction();
    }

    // --- REFINEMENT LOGIC ---
    {
        // `quiet`: a trial of the zero scan below; its failures are expected
        // and must not inflate the diagnostics that explain an empty run.
        const pair_and_fit = (zero_corr_deg, quiet = false) => {
            const cands = [];
            for (let i = 0; i < n_all; i++) {
                const original_idx = original_indices[i];
                let q_to_match;
                if (Math.abs(zero_corr_deg) > 1e-9) {
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
                if (best_match_idx !== -1 && min_diff < tolerance) {
                    cands.push({ i, original_idx, best_match_idx, min_diff });
                }
            }
            // Closest peak wins a contested reflection.
            cands.sort((a, b) => a.min_diff - b.min_diff);
            const used = new Set();
            const claimed = [];
            for (const c of cands) {
                if (used.has(c.best_match_idx)) continue;
                used.add(c.best_match_idx);
                claimed.push(c);
            }
            claimed.sort((a, b) => a.i - b.i);
            const indexed_pairs = [];
            const peak_indices = [];
            for (const c of claimed) {
                const { h, k, l } = all_possible_reflections[c.best_match_idx];
                indexed_pairs.push({ q_obs: q_obs[c.i], hkl: [h, k, l] });
                peak_indices.push(c.original_idx);
            }

            if (diag) diag.bestIndexed = Math.max(diag.bestIndexed || 0, indexed_pairs.length);
            if (indexed_pairs.length < min_indexed[system]) { if (!quiet) bump('tooFewIndexed'); return null; }

            const M = indexed_pairs.map(p => getLSDesignRow(p.hkl, system));
            const q_vec = indexed_pairs.map(p => p.q_obs);
            if (refineZero) {
                M.forEach((row, i) => {
                    const tth_rad = tth_obs_rad[peak_indices[i]];
                    row.push((2 / (wavelength ** 2)) * Math.sin(tth_rad));
                });
            }
            const tth_rads_for_rows = peak_indices.map(idx => tth_obs_rad[idx]);
            const ls_weights = ls_weights_for_2theta(tth_rads_for_rows);

            const fit = solveLeastSquares(M, q_vec, ls_weights);
            if (!fit || !fit.solution) { if (!quiet) bump('fitFailed'); return null; }
            return { fit, indexed_pairs, peak_indices };
        };

        // Round 1. With the zero refined, matching at zero = 0 can index
        // nothing when the true offset exceeds the 2θ error, and the cell was
        // dropped before its zero was ever fitted. Then try trial offsets
        // (steps of the 2θ error, up to zeroScanMaxDeg) and keep the one
        // indexing the most peaks; round 2 re-pairs at its fitted zero.
        let result = pair_and_fit(0, refineZero);
        if (!result && refineZero) {
            const { zeroScanMaxDeg, zeroScanMinStepDeg } = BRUTUS_LIMITS.search;
            const step = Math.max(tth_error, zeroScanMinStepDeg);
            for (let z = step; z <= zeroScanMaxDeg + 1e-9; z += step) {
                for (const trial of [z, -z]) {
                    const r = pair_and_fit(trial, true);
                    if (r && (!result || r.indexed_pairs.length > result.indexed_pairs.length)) result = r;
                }
            }
            if (!result) pair_and_fit(0);   // record the zero-offset failure reason once
        }
        // Round 2 (only when refining zero): re-pair using the round-1 estimate.
        if (result && refineZero) {
            const z1_deg = result.fit.solution[result.fit.solution.length - 1] * DEG;
            if (Math.abs(z1_deg) > 1e-4) {
                const result2 = pair_and_fit(z1_deg);
                if (result2) result = result2;
            }
        }

        if (result) {
            const fitResult_with_zero_final = result.fit;
            const refined_cell = extractCellFromFit(fitResult_with_zero_final.solution, system);

            if (!refined_cell) bump('extractFailed');
            if (refined_cell) {
                if (!cellWithinPhysicalLimits(refined_cell, data)) {
                    bump('refinedOutOfRange');
                    return exitFunction();
                }
                if (refineZero) {
                    refined_cell.zero_correction = fitResult_with_zero_final.solution[fitResult_with_zero_final.solution.length - 1] * DEG;
                }
                if (system === 'hexagonal' && initialParams.lattice === 'R') refined_cell.lattice = 'R';
                refined_cell.volume = getVolume(refined_cell);
                
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
                
                if (diag && isFinite(final_m20)) {
                    diag.bestM20 = Math.max(diag.bestM20 ?? -Infinity, final_m20);
                    diag.refined = (diag.refined || 0) + 1;
                }
                if (final_m20 <= min_m20) bump('lowM20');

                const peaks_for_merit_all_refined = [];
                for (let i = 0; i < n_all; i++) {
                    const original_peak = peaks_sorted_by_q[i]; const corrected_tth_deg = original_peak.tth - (refined_cell.zero_correction || 0);
                    const corrected_tth_rad = corrected_tth_deg * RAD; const corrected_q = (4 * Math.sin(corrected_tth_rad / 2)**2) / (wavelength**2);
                    peaks_for_merit_all_refined.push({ ...original_peak, q: corrected_q, tth: corrected_tth_deg });
                }

                // --- R-lattice probe (hexagonal cells only) -------------------
                if (system === 'hexagonal' && refined_cell.lattice !== 'R' && final_m20 > 0) {
                    const q_R = generateQArray_for_worker({ ...refined_cell, lattice: 'R' }, q_max, wavelength);
                    if (q_R.length > 0 &&
                        countIndexedPeaks(q_R, peaks_for_merit_20_refined, local_get_q_tolerance) ===
                            countIndexedPeaks(q_calc_sorted_refined, peaks_for_merit_20_refined, local_get_q_tolerance) &&
                        countIndexedPeaks(q_R, peaks_for_merit_all_refined, local_get_q_tolerance) ===
                            countIndexedPeaks(q_calc_sorted_refined, peaks_for_merit_all_refined, local_get_q_tolerance)) {
                        if (diag) diag.rLatticeProbes = (diag.rLatticeProbes || 0) + 1;
                        refineAndTestSolution({ a: refined_cell.a, c: refined_cell.c, system: 'hexagonal', lattice: 'R' },
                                              data, state, postMessage_func);
                    }
                }

                if (final_m20 > min_m20) {
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
    return exitFunction();
}
// --- cpu index, with status, hmax changed to 40 for cubic
function indexCubic(data, state, postMessage_func) {
    const { peaks } = data; const { q_obs, refineAndTestSolution } = state;
    const h_max = BRUTUS_LIMITS.hkl.cpuCubicH;
    const peak_depth = Math.min(peaks.length, BRUTUS_LIMITS.search.cpuPeakDepth);
    // Only h²+k²+l² enters the cubic solve. Several triples have the same
    // norm, so evaluating each of those triples repeats the identical fit.
    const norms = new Set();
    for (let h = 1; h <= h_max; h++) {
        for (let k = 0; k <= h; k++) {
            for (let l = 0; l <= k; l++) norms.add(h * h + k * k + l * l);
        }
    }
    const totalTrialsToRun = peak_depth * norms.size;
    let totalTrialsCompleted = 0;
        
    let trialsBatch = 0; let lastReportTime = performance.now();
    for (let i = 0; i < peak_depth; i++) {
        for (const norm of norms) {
            refineAndTestSolution({ a: Math.sqrt(norm / q_obs[i]), system: 'cubic' });
            trialsBatch++; 

            if (trialsBatch % BRUTUS_LIMITS.search.progressTrials === 0 && performance.now() - lastReportTime >= BRUTUS_LIMITS.search.progressIntervalMs) {
                postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch }); 
                totalTrialsCompleted += trialsBatch;
                const progress = (totalTrialsCompleted / totalTrialsToRun) * 80;
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
    const { minAxis: min_lp, maxAxis: max_lp } = getPhysicalLimits(data);
    const max_hkl = BRUTUS_LIMITS.hkl.highSymH;
    const i_depth = Math.min(BRUTUS_LIMITS.search.cpuPeakDepth, peaks.length), j_depth = i_depth;
    const hkls = []; for (let h = 0; h <= max_hkl; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= max_hkl; l++) { if (!h && !k && !l) continue; hkls.push([h,k,l]); }

    const totalPeakCombos = i_depth * (i_depth - 1) / 2;
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

                    if (trialsBatch % BRUTUS_LIMITS.search.progressTrials === 0 && performance.now() - lastReportTime >= BRUTUS_LIMITS.search.progressIntervalMs) {
                        postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch }); 
                        totalTrialsCompleted += trialsBatch;
                        const progress = (totalTrialsCompleted / totalTrialsToRun) * 80;
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
