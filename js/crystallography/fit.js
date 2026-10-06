// js/crystallography/fit.js
// Tolerances, weights, figures of merit, least squares, error propagation and peak sorting.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// Returns a tolerance in A^-2, i.e. on 1/d^2, matching q_obs / q_calc.
// It is NOT a tolerance on the scattering vector or on 2-theta.
const get_q_tolerance = (original_peak_index, tth_obs_rad, wavelength, tth_error) => {
    const theta_rad = tth_obs_rad[original_peak_index] / 2.0;
    const d_theta_rad = tth_error * Math.PI / 360;
    const tolerance = ((8 * Math.sin(theta_rad) * Math.cos(theta_rad)) / (wavelength**2)) * d_theta_rad;
    return tolerance + 1e-9; // Add epsilon to prevent division by zero
};
/**
 * Statistically correct least-squares weights for q-space cell refinement.
 *
 * The least-squares system fits q_obs = (4/λ²) sin²(θ_obs) to a linear
 * combination of cell-parameter columns plus optionally a zero-error column.
 * For a constant 2θ measurement uncertainty σ_2θ:
 *
 *     σ_q = |∂q/∂(2θ)| · σ_2θ = (2 sin(2θ)/λ²) · σ_2θ
 *
 * So σ_q² ∝ sin²(2θ), and the optimal weight per row is:
 *
 *     w_i = 1/σ_q,i² ∝ 1/sin²(2θ_i)
 *
 * With this weighting, low-angle peaks (which have the smallest σ_q) carry
 * the most weight, and high-angle peaks (with the largest σ_q) carry less.
 * This is the OPPOSITE of using w_i = q_obs,i, which is what the code did
 * historically — that scheme heavily over-weights high-angle peaks and
 * biases both the cell parameters and (especially) the zero correction.
 *
 * A small floor of 1.0 / sin²(178°) is built in so weights don't blow up
 * for peaks very close to 0° or 180° (which shouldn't normally exist anyway).
 */
const ls_weights_for_2theta = (tth_rad_array) => {
    const n = tth_rad_array.length;
    const w = new Array(n);
    const minSin2 = Math.sin(178 * Math.PI / 180); // floor at 2θ = 178° → sin = 0.0349
    const minSin2Sq = minSin2 * minSin2;
    for (let i = 0; i < n; i++) {
        const s = Math.sin(tth_rad_array[i]); // sin(2θ)
        const s2 = s * s;
        w[i] = 1.0 / Math.max(s2, minSin2Sq);
    }
    return w;
};
const binarySearchClosest = (arr, target) => {
    const n = arr.length;
    // Guard against empty arrays or non-finite targets to prevent out-of-bounds lookups
    if (n === 0 || !isFinite(target)) return 0; 
    if (target <= arr[0]) return 0;
    if (target >= arr[n - 1]) return n - 1;
    
    let low = 0, high = n - 1;
    while (low <= high) {
        let mid = (low + high) >> 1;
        if (arr[mid] < target) low = mid + 1;
        else high = mid - 1;
    }
    return (low >= n) ? high : ((arr[low] - target) < (target - arr[high]) ? low : high);
};
// Count of entries <= target in an ascending-sorted array. Exactly equal to
// arr.filter(q => q <= target).length, but O(log n) instead of a full scan.
// (Inputs here are finite q values from generateHKL, so no NaN handling needed.)
const countLE = (arr, target) => {
    let lo = 0, hi = arr.length; // hi is exclusive
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
};
const calculateFiguresOfMerit = (q_calc_sorted, peaks_for_merit, impurity_peaks, get_q_tolerance_func, wavelength) => {
    if (!q_calc_sorted || q_calc_sorted.length === 0) return { m20: 0, fN: 0 };
    const N = peaks_for_merit.length; if (N === 0) return { m20: 0, fN: 0 };
    let N_indexed = 0, sum_delta_q = 0, sum_delta_tth = 0;
    const q_n = peaks_for_merit[N - 1].q; const tth_n_deg = peaks_for_merit[N - 1].tth;
    for (let i = 0; i < N; i++) {
        const obs_peak = peaks_for_merit[i]; const q_o = obs_peak.q; const tth_o_deg = obs_peak.tth;
        const tolerance_q = get_q_tolerance_func(obs_peak.original_index);
        const closest_q_calc_idx = binarySearchClosest(q_calc_sorted, q_o);
        const q_c = q_calc_sorted[closest_q_calc_idx];
        const diff_q = Math.abs(q_o - q_c);
        if (diff_q < tolerance_q) {
            N_indexed++; sum_delta_q += diff_q;
            const sinThetaSq_c = (q_c * wavelength**2) / 4;
            if (sinThetaSq_c >= 0 && sinThetaSq_c <= 1) {
                const tth_c_rad = 2 * Math.asin(Math.sqrt(sinThetaSq_c));
                const tth_c_deg = tth_c_rad * DEG;
                sum_delta_tth += Math.abs(tth_o_deg - tth_c_deg);
            }
        }
    }
    if (N - N_indexed > impurity_peaks || N_indexed === 0) return { m20: 0, fN: 0 };
    const avg_delta_q = sum_delta_q / N_indexed;
    const N_calc_M = countLE(q_calc_sorted, q_n);
    const mN = (N_calc_M > 0 && avg_delta_q > 1e-12) ? (q_n / (2 * avg_delta_q * N_calc_M)) : 0;
    const avg_delta_tth = sum_delta_tth / N_indexed;
    const q_limit_fN = (4 * Math.sin(tth_n_deg * RAD / 2)**2) / (wavelength**2);
    const N_calc_FN = countLE(q_calc_sorted, q_limit_fN * 1.0001);
    const fN = (N_calc_FN > 0 && avg_delta_tth > 1e-12) ? ((1 / avg_delta_tth) * (N_indexed / N_calc_FN)) : 0;
    return { m20: mN, fN: fN };
};
const solveLeastSquares = (M, q_vec, weights) => {
    const num_eq = M.length, num_params = M[0].length;
    if (num_eq < num_params) return null;
    
    const w = weights || Array(num_eq).fill(1);
    
    // Build normal equations matrix: M^T * W * M
    const MTWM = Array(num_params).fill(0).map(() => Array(num_params).fill(0));
    for (let i = 0; i < num_params; i++) {
        for (let j = 0; j < num_params; j++) {
            let sum = 0; 
            for (let k = 0; k < num_eq; k++) { 
                sum += M[k][i] * w[k] * M[k][j]; 
            } 
            MTWM[i][j] = sum;
        }
    }
    
    // Build normal equations vector: M^T * W * q
    const MTWq = Array(num_params).fill(0);
    for (let i = 0; i < num_params; i++) {
        let sum = 0; 
        for (let k = 0; k < num_eq; k++) { 
            sum += M[k][i] * w[k] * q_vec[k]; 
        } 
        MTWq[i] = sum;
    }
    
    // Solve using Cholesky Decomposition (much safer for symmetric positive-definite matrices)
    const L = choleskyDecomposition(MTWM); 
    if (!L) return null;
    
    const x = choleskySolve(L, MTWq); 
    if (!x) return null;
    
    // Weighted sum of squared residuals. Computed unconditionally now (it used
    // to be skipped on the df <= 0 early return) and returned to the caller:
    // the combinatorial swap search ranks candidate labellings by it, and
    // recomputing it outside would duplicate this exact loop on every trial.
    const q_calc = M.map(row => row.reduce((s, v, j) => s + v * x[j], 0));
    const SSR = q_vec.reduce((sum, q_o, i) => sum + w[i] * (q_o - q_calc[i]) ** 2, 0);
    let sumW = 0; for (let k = 0; k < num_eq; k++) sumW += w[k];
    const wrms = Math.sqrt(SSR / Math.max(sumW, 1e-30));

    const df = num_eq - num_params;
    if (df <= 0) return { solution: x, covarianceMatrix: null, ssr: SSR, df, wrms };

    // Invert the matrix to get covariance
    const MTWM_inv = choleskyInvert(L);
    if (!MTWM_inv) return { solution: x, covarianceMatrix: null, ssr: SSR, df, wrms };

    // Scale inverted matrix by standard error of the estimate
    const V = MTWM_inv.map(row => row.map(el => el * (SSR / df)));

    return { solution: x, covarianceMatrix: V, ssr: SSR, df, wrms };
};
// First-order error propagation through a numerical Jacobian.
//   params : fitted parameters that define the cell (zero-shift excluded)
//   V      : covariance matrix of the fit; its leading params.length block is used
//   toVec  : params -> [cell quantities] (or null if the params give no cell)
//   names  : output keys, one per cell quantity
// Central differences with a relative step; returns {} if any evaluation fails.
const propagateCellErrorsNumerically = (params, V, toVec, names) => {
    const n = params.length;
    const vals = params.slice();
    const base = toVec(vals);
    if (!base) return {};
    const J = names.map(() => new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
        const original = vals[j];
        const step = (Math.abs(original) * 1e-5) || 1e-7;
        vals[j] = original + step;
        const forward = toVec(vals);
        vals[j] = original - step;
        const backward = toVec(vals);
        vals[j] = original;
        if (!forward || !backward) return {};
        for (let i = 0; i < names.length; i++) J[i][j] = (forward[i] - backward[i]) / (2 * step);
    }
    const out = {};
    for (let i = 0; i < names.length; i++) {
        let variance = 0;
        for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) variance += J[i][j] * V[j][k] * J[i][k];
        out[names[i]] = Math.sqrt(Math.max(0, variance));
    }
    return out;
};
const propagateErrors = (system, fitResult, cell) => {
    if (!fitResult || !fitResult.covarianceMatrix) return {};
    const V = fitResult.covarianceMatrix;
    const errors = {};
    const num_params = V.length; // 6 or 7 (if zero shift included)

    try {
        switch (system) {
            case 'cubic': 
                errors.s_a = 0.5 * cell.a**3 * Math.sqrt(Math.abs(V[0][0])); 
                break;
            case 'tetragonal': 
            case 'hexagonal':
                errors.s_a = 0.5 * cell.a**3 * Math.sqrt(Math.abs(V[0][0]));
                errors.s_c = 0.5 * cell.c**3 * Math.sqrt(Math.abs(V[1][1]));
                break;
            case 'orthorhombic':
                errors.s_a = 0.5 * cell.a**3 * Math.sqrt(Math.abs(V[0][0]));
                errors.s_b = 0.5 * cell.b**3 * Math.sqrt(Math.abs(V[1][1]));
                errors.s_c = 0.5 * cell.c**3 * Math.sqrt(Math.abs(V[2][2]));
                break;
            case 'monoclinic': {
                // Numerical Jacobian of (a, b, c, beta) with respect to the fit
                // parameters (A, B, C, D), propagated through the FULL 4x4
                // covariance block. The former closed form used a/(2A)*sigma_A,
                // which is (1/2)a^3 sin^2(beta) sigma_A -- low by a factor
                // sin^2(beta), 25% at beta = 120 deg -- and dropped the C and D
                // terms that a and c also depend on through sin(beta).
                // extractCellFromFit is the same mapping that produced `cell`,
                // including the beta > 90 convention, so the derivatives are
                // taken on exactly the function being reported.
                const toVec = (p) => {
                    const c = extractCellFromFit(p, 'monoclinic');
                    return c ? [c.a, c.b, c.c, c.beta] : null;
                };
                Object.assign(errors, propagateCellErrorsNumerically(
                    fitResult.solution.slice(0, 4), V, toVec, ['s_a', 's_b', 's_c', 's_beta']));
                break;
            }
            case 'triclinic': 
                // --- Numerical Differentiation for Triclinic
                //maybe need some changes here... à voir, si 0 on nan, ça devrait marcher
                
                // 1. Define helper to go from Params -> [a, b, c, alpha, beta, gamma]
                const calcTriclinic = (p) => {
                    // Reconstruct Reciprocal Metric Tensor (G_star) from 6 LS params
                    const Gs = [[p[0], p[5]/2, p[4]/2], [p[5]/2, p[1], p[3]/2], [p[4]/2, p[3]/2, p[2]]];
                    const G = metricFromReciprocalMetric(Gs); // Invert to Real Metric Tensor
                    if (!G) return null;
                    const c = cellFromMetric_worker(G); // Extract cell constants
                    if (!c) return null;
                    return [c.a, c.b, c.c, c.alpha, c.beta, c.gamma];
                };

                const vals = [...fitResult.solution]; // Copy params
                const base = calcTriclinic(vals);

                if (base) {
                    const J = Array(6).fill(0).map(() => Array(6).fill(0)); // 6 cell params x 6 fit params
                    const delta = 1e-7;

                    // 2. Compute Jacobian Column by Column
                    for (let j = 0; j < 6; j++) { // Loop over fit params p1...p6
                        const original = vals[j];
                        const step = (Math.abs(original) * 1e-5) || delta; // Adaptive step

                        vals[j] = original + step;
                        const forward = calcTriclinic(vals);
                        
                        vals[j] = original - step;
                        const backward = calcTriclinic(vals);
                        
                        vals[j] = original; // Restore

                       if (forward && backward) {
                            for (let i = 0; i < 6; i++) { // Loop over cell params a...gamma
                                // Central difference derivative
                                J[i][j] = (forward[i] - backward[i]) / (2 * step);
                            }
                        } else {
                            return {};
                        }
                    }

                    // 3. Matrix Multiplication: Error[i] = sqrt( sum( J[i][j] * V[j][k] * J[i][k] ) )
                    const indices = ['s_a', 's_b', 's_c', 's_alpha', 's_beta', 's_gamma'];
                    for (let i = 0; i < 6; i++) {
                        let variance = 0;
                        for (let j = 0; j < 6; j++) {
                            for (let k = 0; k < 6; k++) {
                                variance += J[i][j] * V[j][k] * J[i][k];
                            }
                        }
                        errors[indices[i]] = Math.sqrt(Math.max(0, variance));
                    }
                }
                break;
        }
        
        // Zero error calculation (common to all systems if refined)
        const cell_param_count = { cubic: 1, tetragonal: 2, hexagonal: 2, orthorhombic: 3, monoclinic: 4, triclinic: 6 };
        if (num_params > cell_param_count[system]) { 
            errors.s_zero = Math.sqrt(Math.abs(V[num_params - 1][num_params - 1])) * DEG; 
        }

    } catch (e) { console.error("Error during propagation:", e); }
    
    return errors;
};
const hkl_search_list_cache = {};
const get_hkl_search_list = (system) => {
    if (hkl_search_list_cache[system]) return hkl_search_list_cache[system];
    const hkls = []; 
    const max_mono = 6, max_tri = 5;

    if (system === 'monoclinic') {
        const max_h = max_mono; // Use max_mono for clarity
        for (let h = 0; h <= max_h; h++) { // Rule 2: h >= 0
            for (let k = 0; k <= max_h; k++) { // Rule 1: k >= 0
                for (let l = -max_h; l <= max_h; l++) {
                    if (h === 0 && k === 0 && l === 0) continue;
                    if (h === 0 && l < 0) continue; // Apply special h=0 rule
                    hkls.push([h, k, l]);
                }
            }
        }
        // Deep axial reflections (h00)/(0k0)/(00l) beyond the general grid depth.
        // buildHklBasis (splitSpecial=true for monoclinic) front-loads all axial
        // HKLs, so these survive truncation to N_hkl and let the search index a
        // low-angle peak coming from a single long/"strange" axis whose diagnostic
        // axial order exceeds max_mono (=6). Depth 12 matches the orthorhombic grid
        // and triples the previous reach; cost is ~3 reflections per extra order and
        // the combinatorial count C(N_hkl,4) is unchanged (N_hkl is fixed), so these
        // only displace the highest-magnitude tail regulars, never the low-magnitude
        // mixed reflections that carry beta (the D=hl term). (Note: a trial made of
        // four pure axials is rank-deficient in D and is cheaply rejected by the
        // shader's near-zero-determinant filter; this stays a small fraction, ~1.5%
        // of combinations at N_hkl=100.) l>0 only, matching the (h===0 && l<0)
        // exclusion above; no overlap with the grid since n starts at max_h+1.
        const max_axial_mono = 12;
        for (let n = max_h + 1; n <= max_axial_mono; n++) {
            hkls.push([n, 0, 0]);
            hkls.push([0, n, 0]);
            hkls.push([0, 0, n]);
        }
    } else if (system === 'triclinic') {
        for (let h = -max_tri; h <= max_tri; h++) for (let k = -max_tri; k <= max_tri; k++) for (let l = 0; l <= max_tri; l++) {
            if (h === 0 && k === 0 && l === 0) continue; if (l === 0 && k < 0) continue; if (l === 0 && k === 0 && h <= 0) continue; hkls.push([h, k, l]);
        }
    } else if (system === 'orthorhombic') {
        
        const max_h = 12; // Generates up to 12x12x12 = 1728 potential HKLs before sort/filter
        for (let h = 0; h <= max_h; h++)
            for (let k = 0; k <= max_h; k++)
                for (let l = 0; l <= max_h; l++)
                    if (!(h===0 && k===0 && l===0))
                        hkls.push([h,k,l]);
    } else if (system === 'tetragonal' || system === 'hexagonal') {
        const max_h = 8;
        for (let h = 0; h <= max_h; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= max_h; l++) {
            if (h === 0 && k === 0 && l === 0) continue; hkls.push([h, k, l]);
        }
    } else if (system === 'cubic') {
        const max_h = 8;
        for (let h = 0; h <= max_h; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= k; l++) {
            if (h === 0 && k === 0 && l === 0) continue; hkls.push([h, k, l]);
        }
    }
    
    // Q-sort (h^2+k^2+l^2) is applied to all systems
    hkls.sort((a,b) => (a[0]*a[0]+a[1]*a[1]+a[2]*a[2])-(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]));
    
    return hkl_search_list_cache[system] = hkls;
};
function getSortedPeaks(peaks, wavelength) {
    const peaks_sorted_by_q = peaks.map((p, i) => {
        const q = (4 * Math.sin(p.tth * RAD / 2)**2) / (wavelength**2);
        return {...p, original_index: i, q: q};
    }).sort((a,b) => a.q - b.q);
    const q_obs = new Float64Array(peaks_sorted_by_q.map(p => p.q));
    const original_indices = peaks_sorted_by_q.map(p => p.original_index);
    const tth_obs_rad = new Float64Array(peaks.map(p => p.tth * RAD));
    return { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q };
}
