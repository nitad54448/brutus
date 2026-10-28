// js/crystallography/fit.js
// Tolerances, weights, figures of merit, least squares, error propagation and peak sorting.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (index.html) and in both workers, which load them
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
 */
const ls_weights_for_2theta = (tth_rad_array) => {
    const n = tth_rad_array.length;
    const w = new Array(n);
    const minSin2 = Math.sin(178 * Math.PI / 180);
    const minSin2Sq = minSin2 * minSin2;
    for (let i = 0; i < n; i++) {
        const s = Math.sin(tth_rad_array[i]);
        const s2 = s * s;
        w[i] = 1.0 / Math.max(s2, minSin2Sq);
    }
    return w;
};
const binarySearchClosest = (arr, target) => {
    const n = arr.length;
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
const countLE = (arr, target) => {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
};
const countIndexedPeaks = (q_calc_sorted, peaks, get_q_tolerance_func) => {
    if (!q_calc_sorted || q_calc_sorted.length === 0) return 0;
    let n = 0;
    for (const p of peaks) {
        const q_c = q_calc_sorted[binarySearchClosest(q_calc_sorted, p.q)];
        if (Math.abs(p.q - q_c) < get_q_tolerance_func(p.original_index)) n++;
    }
    return n;
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
    
    const MTWq = Array(num_params).fill(0);
    for (let i = 0; i < num_params; i++) {
        let sum = 0; 
        for (let k = 0; k < num_eq; k++) { 
            sum += M[k][i] * w[k] * q_vec[k]; 
        } 
        MTWq[i] = sum;
    }
    
    const L = choleskyDecomposition(MTWM); 
    if (!L) return null;
    
    const x = choleskySolve(L, MTWq); 
    if (!x) return null;
    
    const q_calc = M.map(row => row.reduce((s, v, j) => s + v * x[j], 0));
    const SSR = q_vec.reduce((sum, q_o, i) => sum + w[i] * (q_o - q_calc[i]) ** 2, 0);
    let sumW = 0; for (let k = 0; k < num_eq; k++) sumW += w[k];
    const wrms = Math.sqrt(SSR / Math.max(sumW, 1e-30));

    const df = num_eq - num_params;
    if (df <= 0) return { solution: x, covarianceMatrix: null, ssr: SSR, df, wrms };

    const MTWM_inv = choleskyInvert(L);
    if (!MTWM_inv) return { solution: x, covarianceMatrix: null, ssr: SSR, df, wrms };

    const V = MTWM_inv.map(row => row.map(el => el * (SSR / df)));

    return { solution: x, covarianceMatrix: V, ssr: SSR, df, wrms };
};
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
    const num_params = V.length;

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
                const toVec = (p) => {
                    const c = extractCellFromFit(p, 'monoclinic');
                    return c ? [c.a, c.b, c.c, c.beta] : null;
                };
                Object.assign(errors, propagateCellErrorsNumerically(
                    fitResult.solution.slice(0, 4), V, toVec, ['s_a', 's_b', 's_c', 's_beta']));
                break;
            }
            case 'triclinic': {
                const toVec = (p) => {
                    const c = extractCellFromFit(p, 'triclinic');
                    return c ? [c.a, c.b, c.c, c.alpha, c.beta, c.gamma] : null;
                };
                Object.assign(errors, propagateCellErrorsNumerically(
                    fitResult.solution.slice(0, 6), V, toVec,
                    ['s_a', 's_b', 's_c', 's_alpha', 's_beta', 's_gamma']));
                break;
            }
        }
        
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
        const max_h = max_mono;
        for (let h = 0; h <= max_h; h++) {
            for (let k = 0; k <= max_h; k++) {
                for (let l = -max_h; l <= max_h; l++) {
                    if (h === 0 && k === 0 && l === 0) continue;
                    if (h === 0 && l < 0) continue;
                    hkls.push([h, k, l]);
                }
            }
        }
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
        const max_h = 12;
        for (let h = 0; h <= max_h; h++)
            for (let k = 0; k <= max_h; k++)
                for (let l = 0; l <= max_h; l++)
                    if (!(h===0 && k===0 && l===0))
                        hkls.push([h,k,l]);
    } else if (system === 'tetragonal' || system === 'hexagonal') {
        const hex = system === 'hexagonal';
        const max_h = 12, max_l = 24, max_S = 144;
        const seen = new Set();
        for (let h = 0; h <= max_h; h++) for (let k = 0; k <= h; k++) {
            const S = hex ? h * h + h * k + k * k : h * h + k * k;
            if (S > max_S) continue;
            for (let l = 0; l <= max_l; l++) {
                if (h === 0 && k === 0 && l === 0) continue;
                const key = S * 64 + l;
                if (seen.has(key)) continue;
                seen.add(key);
                hkls.push([h, k, l]);
            }
        }
        const sKey = (t) => (hex ? (4 / 3) * (t[0] * t[0] + t[0] * t[1] + t[1] * t[1])
                                 : t[0] * t[0] + t[1] * t[1]) + t[2] * t[2];
        hkls.sort((a, b) => sKey(a) - sKey(b));
        return hkl_search_list_cache[system] = hkls;
    } else if (system === 'cubic') {
        const max_h = 30, max_N = 900;
        const seen = new Set();
        for (let h = 0; h <= max_h; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= k; l++) {
            const N = h * h + k * k + l * l;
            if (N === 0 || N > max_N || seen.has(N)) continue;
            seen.add(N);
            hkls.push([h, k, l]);
        }
    }

    hkls.sort((a,b) => (a[0]*a[0]+a[1]*a[1]+a[2]*a[2])-(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]));

    return hkl_search_list_cache[system] = hkls;
};
function peakTthAtWavelength(p, wavelength) {
    const lam = p && p.lambda;
    if (!(lam > 0) || !(wavelength > 0) || lam === wavelength || !Number.isFinite(p.tth)) return p ? p.tth : NaN;
    const s = Math.sin(p.tth * RAD / 2) * wavelength / lam;
    return (s > 0 && s < 1) ? 2 * Math.asin(s) * DEG : p.tth;
}
function peaksAtWavelength(peaks, wavelength) {
    return (peaks || []).map(p => {
        const t = peakTthAtWavelength(p, wavelength);
        if (!p || t === p.tth) return p;
        const { lambda, ...rest } = p;
        return { ...rest, tth: t, tthObs: p.tth, lambdaObs: lambda };
    });
}
function getSortedPeaks(peaks, wavelength) {
    const atWl = peaksAtWavelength(peaks, wavelength);
    const peaks_sorted_by_q = atWl.map((p, i) => {
        const q = (4 * Math.sin(p.tth * RAD / 2)**2) / (wavelength**2);
        return {...p, original_index: i, q: q};
    }).sort((a,b) => a.q - b.q);
    const q_obs = new Float64Array(peaks_sorted_by_q.map(p => p.q));
    const original_indices = peaks_sorted_by_q.map(p => p.original_index);
    const tth_obs_rad = new Float64Array(atWl.map(p => p.tth * RAD));
    let tth_max_deg = -Infinity;
    for (const p of atWl) if (p.tth > tth_max_deg) tth_max_deg = p.tth;
    return { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q, tth_max_deg };
}
