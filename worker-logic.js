// "worker-logic.js" 
// most of the combs.html functions are moved here, version 15 nov 2025
//20 nov, moved here the former webworker, indexing-logic
const RAD = Math.PI / 180.0;
const DEG = 180.0 / Math.PI;

const metricFromCell = (cell) => {
    const a = cell.a; const b = cell.b ?? cell.a; const c = cell.c ?? cell.a;
    const alpha = (cell.alpha ?? 90) * RAD; const beta  = (cell.beta  ?? 90) * RAD; const gamma = (cell.gamma ?? 90) * RAD;
    
    // Compute cosines
    let ca = Math.cos(alpha), cb = Math.cos(beta), cg = Math.cos(gamma);
    
    // Snap floating-point dust from 90° (and 270°) angles to exact 0
    if (Math.abs(ca) < 1e-15) ca = 0;
    if (Math.abs(cb) < 1e-15) cb = 0;
    if (Math.abs(cg) < 1e-15) cg = 0;

    const G = [
        [a*a, a*b*cg, a*c*cb], 
        [a*b*cg, b*b, b*c*ca], 
        [a*c*cb, b*c*ca, c*c]
    ];
    
    // Final guard: clean any remaining multiplication dust on off-diagonals
    const clean = (val) => Math.abs(val) < 1e-14 ? 0 : val;
    return G.map(row => row.map(clean));
};

const cellFromMetric = (G) => {
    if (!G) return null;
    try {
        const a = Math.sqrt(Math.max(0, G[0][0])), b = Math.sqrt(Math.max(0, G[1][1])), c = Math.sqrt(Math.max(0, G[2][2]));
        if (a < 1e-6 || b < 1e-6 || c < 1e-6) return null;
        const clamp = v => Math.max(-1, Math.min(1, v));
        const alpha = Math.acos(clamp(G[1][2]/(b*c)))*DEG, beta=Math.acos(clamp(G[0][2]/(a*c)))*DEG, gamma=Math.acos(clamp(G[0][1]/(a*b)))*DEG;
        if (isNaN(alpha) || isNaN(beta) || isNaN(gamma)) return null;
        return { a, b, c, alpha, beta, gamma };
    } catch { return null; }
};

/*
version avant le 12 juillet 2026
const metricFromCell = (cell) => {
    const a = cell.a; const b = cell.b ?? cell.a; const c = cell.c ?? cell.a;
    const alpha = (cell.alpha ?? 90) * RAD; const beta  = (cell.beta  ?? 90) * RAD; const gamma = (cell.gamma ?? 90) * RAD;
    const ca = Math.cos(alpha), cb = Math.cos(beta), cg = Math.cos(gamma);
    return [ [a*a, a*b*cg, a*c*cb], [a*b*cg, b*b, b*c*ca], [a*c*cb, b*c*ca, c*c] ];
};


const cellFromMetric = (G) => {
    const a = Math.sqrt(G[0][0]), b = Math.sqrt(G[1][1]), c = Math.sqrt(G[2][2]);
    const clamp = v => Math.max(-1, Math.min(1, v));
    const alpha = Math.acos(clamp(G[1][2]/(b*c)))*DEG, beta=Math.acos(clamp(G[0][2]/(a*c)))*DEG, gamma=Math.acos(clamp(G[0][1]/(a*b)))*DEG;
    return { a, b, c, alpha, beta, gamma };
};

*/
const transpose = (M) => M[0].map((_,i) => M.map(r => r[i]));
const matMul = (A,B) => { const r=A.length, c=B[0].length, k=A[0].length; const C = Array.from({length:r}, () => Array(c).fill(0)); for(let i=0; i<r; i++) for(let j=0; j<c; j++) for(let t=0; t<k; t++) C[i][j] += A[i][t] * B[t][j]; return C; };

const getSymmetry = (a, b, c, alpha, beta, gamma, tol = 0.25) => {
    const eq = (v1, v2) => Math.abs(v1 - v2) < tol;
    const is90 = (v) => Math.abs(v - 90) < tol; const is120 = (v) => Math.abs(v - 120) < tol;
    const angles90 = is90(alpha) && is90(beta) && is90(gamma);
    if (angles90) {
        if (eq(a, b) && eq(b, c)) return 'cubic';
        if (eq(a, b) || eq(b, c) || eq(a, c)) return 'tetragonal';
        return 'orthorhombic';
    }
    // Hexagonal needs BOTH the angle pattern (two 90s + one 120) AND the two
    // edges spanning the 120 to be equal. Checking angles alone misclassifies a
    // monoclinic cell that merely happens to have a 120 angle. Note the 120 can
    // land on alpha, beta or gamma: the Niggli reduction of a hexagonal lattice
    // orders axes by length (A <= B <= C), so whenever c < a in the conventional
    // hexagonal setting the reduced cell comes out as (c, a, a, 120, 90, 90) and
    // the 120 appears as alpha, not gamma.
    if (is90(alpha) && is90(gamma) && is120(beta)  && eq(a, c)) return 'hexagonal';
    if (is90(beta)  && is90(gamma) && is120(alpha) && eq(b, c)) return 'hexagonal';
    if (is90(alpha) && is90(beta)  && is120(gamma) && eq(a, b)) return 'hexagonal';
    if (is90(alpha) && is90(gamma) && !is90(beta)) return 'monoclinic';
    if (is90(beta) && is90(gamma) && !is90(alpha)) return 'monoclinic'; // b,c unique
    if (is90(alpha) && is90(beta) && !is90(gamma)) return 'monoclinic'; // a,b unique
    return 'triclinic';
};

const standardizeCell = (cell) => {
    const newCell = { ...cell };
    switch (cell.system) {
        case 'tetragonal': { const axes = [cell.a, cell.b, cell.c]; const tol = 0.02; let uniqueAxis, repeatedAxis; if (Math.abs(axes[0] - axes[1]) < tol) { uniqueAxis = axes[2]; repeatedAxis = axes[0]; } else if (Math.abs(axes[0] - axes[2]) < tol) { uniqueAxis = axes[1]; repeatedAxis = axes[0]; } else { uniqueAxis = axes[0]; repeatedAxis = axes[1]; } newCell.a = repeatedAxis; newCell.b = repeatedAxis; newCell.c = uniqueAxis; break; }
        case 'orthorhombic': { const sorted = [cell.a, cell.b, cell.c].sort((x,y)=>x-y); newCell.a=sorted[0]; newCell.b=sorted[1]; newCell.c=sorted[2]; break; }
        case 'cubic': { newCell.b = newCell.a; newCell.c = newCell.a; break; }
    }
    return newCell;
};

const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
const gcdOfList = (arr) => arr.length > 0 ? arr.reduce((acc, val) => gcd(acc, val), arr[0]) : 1;

const determinant3x3 = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

const invert3x3 = (M) => {
    const det = determinant3x3(M); 
    // Check for finite det and prevent division by zero/NaN
    if (!(Math.abs(det) >= 1e-14) || !isFinite(det)) return null;
    const invDet = 1.0 / det;
    return [
        [(M[1][1] * M[2][2] - M[1][2] * M[2][1]) * invDet, (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invDet, (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet],
        [(M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invDet, (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet, (M[0][2] * M[1][0] - M[0][0] * M[1][2]) * invDet],
        [(M[1][0] * M[2][1] - M[1][1] * M[2][0]) * invDet, (M[0][1] * M[2][0] - M[0][0] * M[2][1]) * invDet, (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * invDet]
    ];
};


const choleskyDecomposition = (matrix) => {
    const n = matrix.length;
    // Fast procedural 2D array initialization (avoids .map/.fill closure overhead)
    const L = new Array(n);
    for (let i = 0; i < n; i++) {
        L[i] = new Float64Array(n);
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            if (i === j) {
                const val = matrix[i][i] - sum;
                if (!(val > 1e-12) || !isFinite(val)) return null; 
                L[i][j] = Math.sqrt(val);
            } else {
                if (!isFinite(L[j][j]) || L[j][j] === 0) return null;
                L[i][j] = (matrix[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
};

const choleskyInvert = (L) => {
    const n = L.length;
    const inverse = new Array(n);
    for (let i = 0; i < n; i++) {
        inverse[i] = new Float64Array(n);
    }
    const b = new Float64Array(n);
    for (let j = 0; j < n; j++) {
        b.fill(0); 
        b[j] = 1;
        const invCol = choleskySolve(L, b);
        for (let i = 0; i < n; i++) inverse[i][j] = invCol[i];
    }
    return inverse;
};


const metricFromReciprocalMetric = (G_star) => invert3x3(G_star);

const cellFromMetric_worker = (G) => {
    if (!G) return null;
    try {
        const a = Math.sqrt(Math.max(0, G[0][0])); 
        const b = Math.sqrt(Math.max(0, G[1][1])); 
        const c = Math.sqrt(Math.max(0, G[2][2]));

        if (!(a >= 1e-6) || !(b >= 1e-6) || !(c >= 1e-6) || !isFinite(a) || !isFinite(b) || !isFinite(c)) return null;
        
        // Safe clamp that catches NaN or Infinity before passing to acos
        const clamp = v => isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
        
        const alpha = Math.acos(clamp(G[1][2] / (b * c))) * DEG;
        const beta = Math.acos(clamp(G[0][2] / (a * c))) * DEG;
        const gamma = Math.acos(clamp(G[0][1] / (a * b))) * DEG;
        
        if (!isFinite(alpha) || !isFinite(beta) || !isFinite(gamma)) return null;
        return { a, b, c, alpha, beta, gamma };
    } catch { return null; }
};

const getVolumeTriclinic = (cell) => {
    const { a, b, c, alpha, beta, gamma } = cell;
    const ca = Math.cos(alpha * RAD), cb = Math.cos(beta * RAD), cg = Math.cos(gamma * RAD);
    
    // Formula for volume squared
    const term = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg;
    
    // FIX: Clamp to 0 to prevent NaN cascades from floating point inaccuracies
    const safeTerm = Math.max(0, term);
    
    if (safeTerm === 0) return 0;
    
    return a * b * c * Math.sqrt(safeTerm);
};


// HKL generator... 
function generateHKL_for_analysis(params, lambda, maxTth) {
    const { a, b: b_in, c: c_in, alpha: alpha_in, beta: beta_in, gamma: gamma_in, system } = params;
    const b = b_in ?? a; const c = c_in ?? a;
    const alpha = alpha_in ?? 90; const beta = beta_in ?? 90;
    const gamma = gamma_in ?? (system === 'hexagonal' ? 120 : 90);

    const reflections = [];
    const d_min = lambda / (2 * Math.sin(maxTth * Math.PI / 360));
    const q_max_limit = (1 / (d_min * d_min)) * 1.05;
    const h_max = Math.ceil(a / d_min) + 1;
    const k_max = Math.ceil(b / d_min) + 1;
    const l_max = Math.ceil(c / d_min) + 1;

    const processReflection = (h, k, l, inv_d_sq) => {
        // Inverted logic catches NaN, <= 0, and out-of-bounds strictly
        if (!(inv_d_sq > 0) || !(inv_d_sq <= q_max_limit) || !isFinite(inv_d_sq)) return;
        const sinThetaSq = (lambda * lambda / 4) * inv_d_sq;
        if (sinThetaSq <= 1) {
            const tth = 2 * Math.asin(Math.sqrt(sinThetaSq)) * DEG;
            reflections.push({ tth, h, k, l, d: 1 / Math.sqrt(inv_d_sq), q: inv_d_sq });
        }
    };


    switch (system) {
        case 'cubic':
            for (let h = 0; h <= h_max; h++) { const h_term = (h * h) / (a * a); if (h_term > q_max_limit) break;
                for (let k = 0; k <= h; k++) { const hk_term = h_term + (k * k) / (a * a); if (hk_term > q_max_limit) break;
                    for (let l = 0; l <= k; l++) { if (h === 0 && k === 0 && l === 0) continue; const inv_d_sq = hk_term + (l * l) / (a * a); processReflection(h, k, l, inv_d_sq); }
                }
            } break;
        case 'tetragonal':
        case 'hexagonal':
             for (let l = 0; l <= l_max; l++) { const l_term = (l * l) / (c * c); if (l_term > q_max_limit) break;
                for (let h = 0; h <= h_max; h++) { let h_term_base = (system === 'tetragonal') ? (h * h) / (a * a) : (4 / 3) * (h * h) / (a * a); if (l_term + h_term_base > q_max_limit && h > 0) break;
                    for (let k = 0; k <= h; k++) { if (h === 0 && k === 0 && l === 0) continue; let inv_d_sq = (system === 'tetragonal') ? l_term + (h * h + k * k) / (a * a) : l_term + (4 / 3) * (h * h + h * k + k * k) / (a * a); processReflection(h, k, l, inv_d_sq); }
                }
            } break;
        case 'orthorhombic':
            for (let h = 0; h <= h_max; h++) { const h_term = (h * h) / (a * a); if (h_term > q_max_limit) break;
                for (let k = 0; k <= k_max; k++) { const hk_term = h_term + (k * k) / (b * b); if (hk_term > q_max_limit) break;
                    for (let l = 0; l <= l_max; l++) { if (h === 0 && k === 0 && l === 0) continue; const inv_d_sq = hk_term + (l * l) / (c * c); processReflection(h, k, l, inv_d_sq); }
                }
            } break;
        case 'monoclinic':
            const sinBeta = Math.sin(beta * RAD), cosBeta = Math.cos(beta * RAD), sinBetaSq = sinBeta * sinBeta;
           if (!(sinBetaSq >= 1e-6) || !isFinite(sinBetaSq)) return [];
            const a_star_sq = 1 / (a * a * sinBetaSq), b_star_sq = 1 / (b * b), c_star_sq = 1 / (c * c * sinBetaSq), ac_star_term = 2 * cosBeta / (a * c * sinBetaSq);
            for (let h = -h_max; h <= h_max; h++) {
                const h_term = h * h * a_star_sq, h_l_coeff = h * ac_star_term;
                const l_vertex_h_only = (c_star_sq !== 0) ? h_l_coeff / (2 * c_star_sq) : 0;
                const q_min_for_h = (c_star_sq * l_vertex_h_only * l_vertex_h_only) - (h_l_coeff * l_vertex_h_only) + h_term;
                if (q_min_for_h > q_max_limit) continue;
                for (let k = 0; k <= k_max; k++) {
                    const k_term = (k * k) * b_star_sq, hk_term = h_term + k_term;
                    const l_vertex = l_vertex_h_only;
                    const q_min_for_hk = (c_star_sq * l_vertex * l_vertex) - (h_l_coeff * l_vertex) + hk_term;
                    if (q_min_for_hk > q_max_limit) { if (k === 0) break; else continue; }
                    // q(l) = c*^2 l^2 - h_l_coeff l + hk_term is an upward parabola,
                    // so the l values with q <= q_max_limit form a contiguous interval.
                    // Solve for it instead of sweeping the whole [-l_max, l_max] box.
                    // The range is WIDENED BY 1 on each side and clamped to the old
                    // bounds, and every guard/filter below is left untouched, so the
                    // emitted set is provably identical - we only skip l values that
                    // could not have passed processReflection anyway.
                    const disc_m = (h_l_coeff * h_l_coeff) - 4 * c_star_sq * (hk_term - q_max_limit);
                    if (!(disc_m >= 0)) continue; // no real l satisfies q <= limit
                    const sq_m = Math.sqrt(disc_m);
                    const l_lo = Math.max(-l_max, Math.ceil((h_l_coeff - sq_m) / (2 * c_star_sq)) - 1);
                    const l_hi = Math.min(l_max, Math.floor((h_l_coeff + sq_m) / (2 * c_star_sq)) + 1);
                    for (let l = l_lo; l <= l_hi; l++) {
                        if (h === 0 && k === 0 && l === 0) continue;
                        if (k === 0) { if (h < 0) continue; if (h === 0 && l <= 0) continue; }
                        const inv_d_sq = (c_star_sq * l * l) - (h_l_coeff * l) + hk_term;
                        processReflection(h, k, l, inv_d_sq);
                    }
                }
            } break;
       
            case 'triclinic':
            // Robust calculation of Reciprocal Metric Tensor components
            const ca = Math.cos(alpha * RAD), cb = Math.cos(beta * RAD), cg = Math.cos(gamma * RAD);
            const sa = Math.sin(alpha * RAD), sb = Math.sin(beta * RAD), sg = Math.sin(gamma * RAD);
            
            // Calculate Volume first to ensure validity
            const term = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg;
            if (!(term > 0) || !isFinite(term)) return [];
            const V = a * b * c * Math.sqrt(term);

            // Reciprocal lattice parameters (a*, b*, c*, alpha*, beta*, gamma*)
            // using standard crystallographic formulas
            const a_star = (b * c * sa) / V;
            const b_star = (a * c * sb) / V;
            const c_star = (a * b * sg) / V;
            
            const ca_star = (cb * cg - ca) / (sb * sg);
            const cb_star = (ca * cg - cb) / (sa * sg);
            const cg_star = (ca * cb - cg) / (sa * sb);
            
            // Components for d*^2 calculation: d*^2 = h^2 a*^2 + ... + 2hk a*b* cos(gamma*)
            const S11 = a_star * a_star;
            const S22 = b_star * b_star;
            const S33 = c_star * c_star;
            const S12 = 2 * a_star * b_star * cg_star;
            const S13 = 2 * a_star * c_star * cb_star;
            const S23 = 2 * b_star * c_star * ca_star;

            for (let h = -h_max; h <= h_max; h++) {
                const h_term = h * h * S11;
                for (let k = -k_max; k <= k_max; k++) {
                    const k_term = k * k * S22;
                    const hk_term = h_term + k_term + h * k * S12;
                    
                    // q(l) = S33 l^2 + (k S23 + h S13) l + hk_term, an upward parabola
                    // (S33 = c*^2 > 0), so the l values with q <= q_max_limit form a
                    // contiguous interval. The original swept the entire [-l_max, l_max]
                    // box with no pruning at all (~79-85% wasted iterations, measured).
                    // We solve for the interval, WIDEN IT BY 1 each side, clamp to the
                    // old bounds, and leave every guard below untouched - so the emitted
                    // set is provably identical.
                    const B_l = k * S23 + h * S13;
                    const disc_t = B_l * B_l - 4 * S33 * (hk_term - q_max_limit);
                    if (!(disc_t >= 0)) continue; // no real l can satisfy q <= limit
                    const sq_t = Math.sqrt(disc_t);
                    let l_lo = Math.max(-l_max, Math.ceil((-B_l - sq_t) / (2 * S33)) - 1);
                    const l_hi = Math.min(l_max, Math.floor((-B_l + sq_t) / (2 * S33)) + 1);
                    if (l_lo < 0) l_lo = 0; // Friedel half-space; l<0 is discarded below

                    for (let l = l_lo; l <= l_hi; l++) {
                        // Skip (0,0,0)
                        if (h === 0 && k === 0 && l === 0) continue;

                        // Friedel's Law: We only need half of reciprocal space.
                        // Standard convention: l > 0, or (l=0, k>0), or (l=0, k=0, h>0)
                        if (l < 0) continue;
                        if (l === 0 && k < 0) continue;
                        if (l === 0 && k === 0 && h <= 0) continue;

                        const inv_d_sq = hk_term + (l * l * S33) + (k * l * S23) + (h * l * S13);
                        processReflection(h, k, l, inv_d_sq);
                    }
                }
            } 
            break;


    }
    const uniqueReflections = []; const tolerance = 1e-4;
    if (reflections.length > 0) {
        reflections.sort((a, b) => a.tth - b.tth);
        uniqueReflections.push(reflections[0]);
        for (let i = 1; i < reflections.length; i++) {
            if (Math.abs(reflections[i].tth - uniqueReflections[uniqueReflections.length - 1].tth) > tolerance) {
                uniqueReflections.push(reflections[i]);
            }
        }
    }
    return uniqueReflections;
};

// This is the same as generateHKL_for_analysis.
// fallback to Cu if lambda missing
const generateHKL = (maxTth, params, system, defaultLambda = 1.54056) => {
    const lambda = params.lambda || defaultLambda;
    return generateHKL_for_analysis(params, lambda, maxTth);
};

const generateHKL_for_worker = (cell, q_max, d_min, lambda) => {
    // We need lambda to correctly calculate the max 2-theta from q_max
    const maxTth = Math.asin(Math.sqrt(q_max * lambda * lambda / 4.0)) * 360.0 / Math.PI;
    return generateHKL_for_analysis(cell, lambda, maxTth);
};


const getQcalc = (hkl, cell) => {
    const [h, k, l] = hkl;
    const { a, b, c, beta, system, alpha, gamma } = cell;
    switch (system) {
        case 'cubic': return (h*h + k*k + l*l) / (a*a);
        case 'tetragonal': return (h*h + k*k) / (a*a) + (l*l) / (c*c);
        case 'hexagonal': return (4/3) * (h*h + h*k + k*k) / (a*a) + (l*l) / (c*c);
        case 'orthorhombic': return h*h/(a*a) + k*k/(b*b) + l*l/(c*c);
        case 'monoclinic':
            const sinBeta = Math.sin(beta * RAD), cosBeta = Math.cos(beta * RAD);
            if (Math.abs(sinBeta) < 1e-9) return 0;
            return (1/(sinBeta*sinBeta)) * (h*h/(a*a) + l*l/(c*c) - 2*h*l*cosBeta/(a*c)) + k*k/(b*b);
        case 'triclinic':
            const ca = Math.cos(alpha * RAD), cb = Math.cos(beta * RAD), cg = Math.cos(gamma * RAD);
            const V_sq = a*a*b*b*c*c * (1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
            if (V_sq < 1e-6) return 0;
            const Gs_11 = (b*b*c*c * (1 - ca*ca)) / V_sq, Gs_22 = (a*a*c*c * (1 - cb*cb)) / V_sq, Gs_33 = (a*a*b*b * (1 - cg*cg)) / V_sq;
            const Gs_23 = 2 * b*c*a*a * (cb*cg - ca) / V_sq, Gs_13 = 2 * a*c*b*b * (ca*cg - cb) / V_sq, Gs_12 = 2 * a*b*c*c * (ca*cb - cg) / V_sq;
            return h*h*Gs_11 + k*k*Gs_22 + l*l*Gs_33 + k*l*Gs_23 + h*l*Gs_13 + h*k*Gs_12;
    }
    return 0;
};

const getLSDesignRow = (hkl, system) => {
    const [h, k, l] = hkl;
    switch(system) {
        case 'cubic': return [h*h + k*k + l*l];
        case 'tetragonal': return [h*h + k*k, l*l];
        case 'hexagonal': return [(4/3)*(h*h + h*k + k*k), l*l];
        case 'orthorhombic': return [h*h, k*k, l*l];
        case 'monoclinic': return [h*h, k*k, l*l, h*l];
        case 'triclinic': return [h*h, k*k, l*l, k*l, h*l, h*k];
    }
};

const extractCellFromFit = (params, system) => {
    let cell = { system };
    try {
        if (params.some(p => isNaN(p))) return null;
        switch(system) {
            case 'cubic': if (params[0] <= 0) return null; cell.a = 1/Math.sqrt(params[0]); cell.b = cell.a; cell.c = cell.a; cell.alpha = 90; cell.beta = 90; cell.gamma = 90; break;
            case 'tetragonal': if (params[0] <= 0 || params[1] <= 0) return null; cell.a = 1/Math.sqrt(params[0]); cell.b = cell.a; cell.c = 1/Math.sqrt(params[1]); cell.alpha = 90; cell.beta = 90; cell.gamma = 90; break;
            case 'hexagonal': if (params[0] <= 0 || params[1] <= 0) return null; cell.a = 1/Math.sqrt(params[0]); cell.b = cell.a; cell.c = 1/Math.sqrt(params[1]); cell.alpha = 90; cell.beta = 90; cell.gamma = 120; break;
            case 'orthorhombic': if (params.slice(0, 3).some(p => p <= 0)) return null; cell.a = 1/Math.sqrt(params[0]); cell.b = 1/Math.sqrt(params[1]); cell.c = 1/Math.sqrt(params[2]); cell.alpha = 90; cell.beta = 90; cell.gamma = 90; break;
            case 'monoclinic':
                const [A, B, C, D] = params.slice(0, 4);
                if (A <= 0 || B <= 0 || C <= 0 || D*D >= 4*A*C) return null;
                const cosBeta_calc = -D / (2 * Math.sqrt(A*C));
                if (Math.abs(cosBeta_calc) >= 1) return null;
                let beta_calc = Math.acos(cosBeta_calc) * DEG;
                if (beta_calc < 90.0) beta_calc = 180.0 - beta_calc;
                if (beta_calc < 90.0 || beta_calc > 150.0) return null;
                cell.beta = beta_calc; const sinBetaSq = Math.sin(cell.beta * RAD)**2;
                if (sinBetaSq <= 1e-6) return null;
                cell.a = 1/Math.sqrt(A * sinBetaSq); cell.b = 1/Math.sqrt(B); cell.c = 1/Math.sqrt(C * sinBetaSq);
                cell.alpha = 90; cell.gamma = 90;
                break;
            case 'triclinic':
                const [p1, p2, p3, p4, p5, p6] = params;
                const G_star = [ [p1, p6/2, p5/2], [p6/2, p2, p4/2], [p5/2, p4/2, p3] ];
                const G = metricFromReciprocalMetric(G_star);
                if (!G) return null;
                const triclinicCell = cellFromMetric_worker(G);
                if (!triclinicCell) return null;
                cell = { ...cell, ...triclinicCell };
                break;
        }
    } catch (e) { return null; }
    if (isNaN(cell.a) || isNaN(cell.b) || isNaN(cell.c) || isNaN(cell.alpha) || isNaN(cell.beta) || isNaN(cell.gamma)) return null;
    return cell;
};

const getVolume = (cell) => {
    const { a, b, c, beta, system } = cell;
    switch(system){
        case 'cubic': return a**3;
        case 'tetragonal': return a**2 * c;
        case 'hexagonal': return a**2 * c * Math.sqrt(3)/2;
        case 'orthorhombic': return a * b * c;
        case 'monoclinic': return a * b * c * Math.sin(beta * RAD);
        case 'triclinic': return getVolumeTriclinic(cell);
    }
};

const getSolutionKey = (cell) => {
    const P = 4; // Increased from 2 to 4 digits to prevent aggressive deduplication
    const std = standardizeCell(cell);
    switch(std.system) {
        case 'cubic': 
            return `${std.system}_${std.a.toFixed(P)}`;
        case 'tetragonal': 
        case 'hexagonal': 
            return `${std.system}_${std.a.toFixed(P)}_${std.c.toFixed(P)}`;
        case 'orthorhombic': 
            return `${std.system}_${[std.a,std.b,std.c].sort().map(p => p.toFixed(P)).join('_')}`;
        case 'monoclinic': 
            const ac = [std.a, std.c].sort((x, y) => x - y).map(p => p.toFixed(P)).join('_');
            // Updated beta to use P instead of hardcoded 2
            return `${std.system}_${ac}_${std.b.toFixed(P)}_${std.beta.toFixed(P)}`; 
        case 'triclinic': 
            // Updated volume and angles to use P instead of hardcoded 2 and 1
            const vol = getVolumeTriclinic(std).toFixed(P); 
            const angles = [std.alpha, std.beta, std.gamma].sort().map(a => a.toFixed(P)).join('_'); 
            return `${std.system}_${vol}_${angles}`;
    }
};
    


const choleskySolve = (L, b) => {
    const n = L.length;
    const y = new Array(n);
    // Forward substitution
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
        y[i] = (b[i] - sum) / L[i][i];
    }
    // Backward substitution
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
        let sum = 0;
        for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j];
        x[i] = (y[i] - sum) / L[i][i];
    }
    return x;
};



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
    const N_calc_M = countLE(q_calc_sorted, q_n * 1.05);
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
    
    const df = num_eq - num_params; 
    if (df <= 0) return { solution: x, covarianceMatrix: null };
    
    // Calculate Sum of Squared Residuals (SSR)
    const q_calc = M.map(row => row.reduce((s, v, j) => s + v * x[j], 0));
    const SSR = q_vec.reduce((sum, q_obs, i) => sum + w[i] * (q_obs - q_calc[i])**2, 0);
    
    // Invert the matrix to get covariance
    const MTWM_inv = choleskyInvert(L); 
    if (!MTWM_inv) return { solution: x, covarianceMatrix: null };
    
    // Scale inverted matrix by standard error of the estimate
    const V = MTWM_inv.map(row => row.map(el => el * (SSR / df)));
    
    return { solution: x, covarianceMatrix: V };
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
            case 'monoclinic':
                const [A,B,C,D] = fitResult.solution;
                errors.s_b = 0.5 * cell.b**3 * Math.sqrt(Math.abs(V[1][1]));
                const betaRad = cell.beta*RAD, sinBeta=Math.sin(betaRad), sqrtAC=Math.sqrt(A*C);
                const d_beta_d_A = (-1/sinBeta)*(D/(4*A*sqrtAC)), d_beta_d_C = (-1/sinBeta)*(D/(4*C*sqrtAC)), d_beta_d_D = (-1/sinBeta)*(-1/(2*sqrtAC));
                errors.s_beta = Math.sqrt(Math.abs(d_beta_d_A**2*V[0][0] + d_beta_d_C**2*V[2][2] + d_beta_d_D**2*V[3][3] + 2*(d_beta_d_A*d_beta_d_C*V[0][2] + d_beta_d_A*d_beta_d_D*V[0][3] + d_beta_d_C*d_beta_d_D*V[2][3]))) * DEG;
                errors.s_a = (cell.a / (2*A)) * Math.sqrt(Math.abs(V[0][0]));
                errors.s_c = (cell.c / (2*C)) * Math.sqrt(Math.abs(V[2][2]));
                break;
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
    
    // console.log("REFINE: Starting refinement for", initialParams);

    if (!initialParams || !initialParams.system) {
        // console.log("REFINE: Rejected - no params");
        return exitFunction();
    }
    
    const { system } = initialParams;

    const min_lp_check = 2.0, max_lp_check = 50.0;
    const axes_to_check = [initialParams.a, initialParams.b ?? initialParams.a, initialParams.c ?? initialParams.a];
    const angles_to_check = [initialParams.alpha ?? 90, initialParams.beta ?? 90, initialParams.gamma ?? 90];

    // Explicitly validate both linear dimensions and angles against NaN/Infinity and physical limits
    if (axes_to_check.some(p => !isFinite(p) || p < min_lp_check || p > max_lp_check) ||
        angles_to_check.some(a => !isFinite(a) || a < 10.0 || a > 170.0)) {
        return exitFunction(); 
    }
    
    const initial_cell_volume = getVolume(initialParams);
    // !(vol >= 20) safely catches NaN, 0, and negative volumes
    if (!(initial_cell_volume >= 20) || !(initial_cell_volume <= max_volume) || !isFinite(initial_cell_volume)) {
        return exitFunction();
    }


    const local_get_q_tolerance = (idx) => get_q_tolerance(idx, tth_obs_rad, wavelength, tth_error);
    const min_indexed = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 };
    let final_solution_to_post = null;
    const n_20 = Math.min(N_FOR_M20, peaks_sorted_by_q.length);
    const n_all = peaks_sorted_by_q.length;
    const all_possible_reflections = generateHKL_for_worker(initialParams, q_max, d_min, wavelength);
    
    if (all_possible_reflections.length === 0) {
        // console.log("REFINE: Rejected - could not generate HKLs");
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

            if (indexed_pairs.length < min_indexed[system]) return null;

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
            if (!fit || !fit.solution) return null;
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

            if (refined_cell) {
                // Only attach a zero_correction when we actually refined one.
                // Leaving it undefined lets applyFinalSieve treat this as a
                // 0-DoF (fixed-zero) model, distinct from a zero-refined one.
                if (refineZero) {
                    refined_cell.zero_correction = fitResult_with_zero_final.solution[fitResult_with_zero_final.solution.length - 1] * DEG;
                }
                refined_cell.volume = getVolume(refined_cell);
                
                const q_calc_set_refined = new Set(generateHKL_for_worker(refined_cell, q_max, d_min, wavelength).map(r => r.q));
                const q_calc_sorted_refined = new Float64Array(Array.from(q_calc_set_refined)).sort((a,b)=>a-b);
                
                const peaks_for_merit_20_refined = [];
                for (let i = 0; i < n_20; i++) {
                    const original_peak = peaks_sorted_by_q[i];
                    const corrected_tth_deg = original_peak.tth - (refined_cell.zero_correction || 0);
                    const corrected_tth_rad = corrected_tth_deg * RAD;
                    const corrected_q = (4 * Math.sin(corrected_tth_rad / 2)**2) / (wavelength**2);
                    peaks_for_merit_20_refined.push({ ...original_peak, q: corrected_q, tth: corrected_tth_deg });
                }
                
                const { m20: final_m20, fN: final_fN_20 } = calculateFiguresOfMerit(q_calc_sorted_refined, peaks_for_merit_20_refined, impurity_peaks, local_get_q_tolerance, wavelength);
                
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
};



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

if (performance.now() - lastReportTime >= 50) { // Report at most every 50ms
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

                    if (performance.now() - lastReportTime >= 50) { // Report at most every 50ms
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


function findTransformedSolutions(initialSolutions, data, state, postMessage_func) {
    const { allowedSystems } = data;
    const { refineAndTestSolution, q_obs, original_indices, N_FOR_M20, q_max, d_min, tth_obs_rad } = state;
    const { wavelength, tth_error } = data;
    const cellTransforms = [ { P: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] }, { P: [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]] }, { P: [[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]] }, { P: [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]] }, { P: [[1, 0, 0], [0, 0.5, 0], [0, 0, 1]] }, { P: [[1, 0, 0], [0, 1, 0], [0, 0, 0.5]] }, { P: [[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]] } ];
    const totalSolutions = initialSolutions.length; if (totalSolutions === 0) return;
    const local_get_q_tolerance = (idx) => get_q_tolerance(idx, tth_obs_rad, wavelength, tth_error);
    initialSolutions.forEach((sol, index) => {
        
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

        const indexedPeaks = [];
        for(let i=0; i<N_FOR_M20; i++){
             const q_o = q_obs[i];
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
        


        // --- SWAP FISHING (restored) --------------------------------------
        // Enumerates a few alternative hkl labels at a FIXED cell. Kept as an
        // automatic pass; the Monte-Carlo search is now user-driven via the
        // "Refine MC" context-menu entry instead of running on every solution.
        //
        // BUG FIX: the two early exits below were bare `return`s. Because this
        // whole body runs inside initialSolutions.forEach(...), a `return` left
        // the ENTIRE iteration, skipping the progress postMessage at the end --
        // so the progress bar silently stalled below 95% whenever a solution
        // indexed too few peaks. They now break out of the labelled block only.
        try {
        swapFishing: {
            // --- SWAP FISHING ---
            const system = sol.system;
            const min_peaks_needed = {cubic: 1, tetragonal: 2, hexagonal: 2, orthorhombic: 3, monoclinic: 4, triclinic: 6}[system];
            if (!min_peaks_needed || data.peaks.length < min_peaks_needed) break swapFishing;

            console.log(`\n[SWAP DEBUG] === Starting for ${system}, M20=${sol.m20.toFixed(2)}, Z=${(sol.zero_correction||0).toFixed(4)} ===`);

            const SWAP_FISH_MAX_PEAKS  = 12;
            const SWAP_FISH_MAX_ALTS   = 2;
            const SWAP_FISH_MAX_TRIALS = 12;
            const SWAP_FISH_ALT_WINDOW = 2.5;
            const SWAP_FISH_MAX_PAIRS  = 3;   // transposition pairs to try (see (b))

            const nScan = Math.min(SWAP_FISH_MAX_PEAKS, q_obs.length);
            const indexed = [];
            const wl = wavelength;
            const z_deg = sol.zero_correction || 0;

            for (let i = 0; i < nScan; i++) {
                const orig = state.peaks_sorted_by_q[i];
                
                // CRITICAL FIX: The old code used raw q_obs. We must subtract zero error
                // before searching theoretical lines, or the peaks won't mathematically align!
                const tc_deg = orig.tth - z_deg;
                const corr_q_o = (4 * Math.sin(tc_deg * Math.PI / 360)**2) / (wl**2);
                
                const tol = local_get_q_tolerance(original_indices[i]);
                const bi = binarySearchClosest(theoretical_q_array, corr_q_o);
                
                if (bi < 0 || bi >= theoretical_hkls.length) {
                    console.log(`[SWAP DEBUG] Peak ${orig.tth.toFixed(3)} skipped. Closest index out of bounds.`);
                    continue;
                }
                
                const diff = Math.abs(corr_q_o - theoretical_hkls[bi].q);
                if (diff >= tol) {
                    console.log(`[SWAP DEBUG] Peak ${orig.tth.toFixed(3)} skipped. Diff ${diff.toFixed(5)} >= Tol ${tol.toFixed(5)}`);
                    continue;
                }
                
                const t = theoretical_hkls[bi];
                indexed.push({ q_obs: corr_q_o, tol: tol, bi: bi, hkl: [t.h, t.k, t.l], tth: orig.tth });
                console.log(`[SWAP DEBUG] Peak ${orig.tth.toFixed(3)} -> indexed to (${t.h},${t.k},${t.l})`);
            }
            
            if (indexed.length < min_peaks_needed) {
                console.log(`[SWAP DEBUG] Not enough indexed peaks (${indexed.length}). Aborting.`);
                break swapFishing;
            }

            let trials = 0;
            const maxTth = Math.max(...data.peaks.map(p => p.tth));

            const applyAndTestSwap = (overrides) => {
                const ovStr = overrides.map(o => `${o.tth.toFixed(3)}->(${o.h},${o.k},${o.l})`).join(', ');
                console.log(`[SWAP DEBUG] Trying override: ${ovStr}`);
                
const res = refineWithManualHkl(sol, data.peaks, overrides, wavelength, tth_error, maxTth, data.refineZero, data.impurity_peaks, true);                
                if (!res) { console.log(`[SWAP DEBUG] -> Failed: refineWithManualHkl returned null.`); return; }
                if (res.error) { console.log(`[SWAP DEBUG] -> Failed: ${res.error}`); return; }
                
                if (res.cell) {
                    const newCell = res.cell;
                    console.log(`[SWAP DEBUG] -> Success! New M20=${newCell.m20.toFixed(2)} (Old=${sol.m20.toFixed(2)})`);
                    
                    const key = getSolutionKey(newCell);
                    const existing = state.foundSolutionMap.get(key);
                    
                    if (!existing || newCell.m20 > existing.m20) {
                        console.log(`[SWAP DEBUG] -> Posting new solution to UI! (Key: ${key})`);
                        if (existing) state.foundSolutions[existing.index] = newCell;
                        else state.foundSolutions.push(newCell);
                        state.foundSolutionMap.set(key, { m20: newCell.m20, index: existing ? existing.index : state.foundSolutions.length - 1 });
                        postMessage_func({ type: 'solution', payload: newCell });
                    } else {
                        console.log(`[SWAP DEBUG] -> Rejected: Existing solution with key ${key} has better/equal M20 (${existing.m20.toFixed(2)})`);
                    }
                }
            };

            // (a) Reassignment: alternatives within tolerance of the SAME peak.
            // Capped below SWAP_FISH_MAX_TRIALS so the transposition pass in (b)
            // always gets its slots. Previously (a) could consume the entire
            // budget and (b) would never run at all -- which mattered little
            // when (b) tried a single pair, but does now that transpositions are
            // known to catch a class of error (a) cannot reach.
            const reassignBudget = SWAP_FISH_MAX_TRIALS - SWAP_FISH_MAX_PAIRS;
            for (let n = 0; n < indexed.length && trials < reassignBudget; n++) {
                const pk = indexed[n];
                const searchTol = pk.tol * SWAP_FISH_ALT_WINDOW;
                const alts = [];
                
                for (let j = pk.bi - 1; j >= 0; j--) {
                    if (Math.abs(pk.q_obs - theoretical_hkls[j].q) > searchTol) break;
                    alts.push(theoretical_hkls[j]);
                }
                for (let j = pk.bi + 1; j < theoretical_hkls.length; j++) {
                    if (Math.abs(pk.q_obs - theoretical_hkls[j].q) > searchTol) break;
                    alts.push(theoretical_hkls[j]);
                }
                
                alts.sort((x, y) => Math.abs(x.q - pk.q_obs) - Math.abs(y.q - pk.q_obs));
                if (alts.length > 0) {
                    console.log(`[SWAP DEBUG] Peak ${pk.tth.toFixed(3)} alts found: ` + alts.map(a => `(${a.h},${a.k},${a.l})`).join(', '));
                }

                let used = 0;
                for (const alt of alts) {
                    if (used >= SWAP_FISH_MAX_ALTS || trials >= reassignBudget) break;
                    if (alt.h === pk.hkl[0] && alt.k === pk.hkl[1] && alt.l === pk.hkl[2]) continue;
                    
                    applyAndTestSwap([{ tth: pk.tth, h: alt.h, k: alt.k, l: alt.l }]);
                    trials++;
                    used++;
                }
            }

            // (b) Transposition: exchange the labels of close-lying peaks.
            //
            // Nearest-line assignment can hand each of two neighbouring peaks
            // the other one's line. Nothing flags it: in a permissive space
            // group both labels are allowed, and -- crucially -- the swapped
            // labelling ALWAYS has a larger |q_obs - q_calc| than the nearest-
            // line one, by construction. So no local residual comparison can
            // ever detect a crossing; the only discriminator is to refit the
            // cell with the labels exchanged and see whether M20 improves,
            // which is what applyAndTestSwap already does.
            //
            // The job here is therefore candidate GENERATION, not detection.
            // Measured on planted crossings across orthorhombic, monoclinic and
            // tetragonal patterns (189 cases): ranking pairs by |q_i - q_j| and
            // taking the closest one catches 77%, the closest TWO catch 100%
            // (max rank observed was 1). The original code tried only the single
            // closest pair, so it missed roughly a quarter of real crossings.
            // Three are tried here for margin, budget permitting.
            const pairs = [];
            for (let i = 0; i < indexed.length; i++) {
                for (let j = i + 1; j < indexed.length; j++) {
                    const diff = Math.abs(indexed[i].q_obs - indexed[j].q_obs);
                    if (isFinite(diff)) pairs.push({ i, j, diff });
                }
            }
            pairs.sort((x, y) => x.diff - y.diff);

            let pairsTried = 0;
            for (const pr of pairs) {
                if (pairsTried >= SWAP_FISH_MAX_PAIRS) break;
                if (trials >= SWAP_FISH_MAX_TRIALS) break;
                const pk1 = indexed[pr.i];
                const pk2 = indexed[pr.j];
                // Exchanging identical labels is a no-op; skip so a degenerate
                // pair does not consume one of the three slots.
                if (pk1.hkl[0] === pk2.hkl[0] && pk1.hkl[1] === pk2.hkl[1] &&
                    pk1.hkl[2] === pk2.hkl[2]) continue;
                console.log(`[SWAP DEBUG] Trying transposition between ${pk1.tth.toFixed(3)} and ${pk2.tth.toFixed(3)} (|dq|=${pr.diff.toExponential(2)})`);
                applyAndTestSwap([
                    { tth: pk1.tth, h: pk2.hkl[0], k: pk2.hkl[1], l: pk2.hkl[2] },
                    { tth: pk2.tth, h: pk1.hkl[0], k: pk1.hkl[1], l: pk1.hkl[2] }
                ]);
                trials++;
                pairsTried++;
            }
        }
        } catch (e) { console.warn("Swap-fishing attempt failed:", e); }

        // The Monte-Carlo cell polish is no longer run automatically here.
        // It is invoked on demand from the solutions context menu ("Refine MC"),
        // which lets the user choose how many solutions / iterations / restarts
        // to spend rather than paying ~0.3 s on every candidate.

        const progress = 80 + ((index + 1) / totalSolutions) * 15;
        postMessage_func({ type: 'progress', payload: progress });
    });
};

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
function mcParamsToCell(params, system, maxVolume) {
    if (!params || params.some(p => !isFinite(p))) return null;
    const cell = extractCellFromFit(params, system);
    if (!cell) return null;
    cell.system = system;
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
    let refl;
    try {
        refl = generateHKL_for_worker(cell, q_max, d_min, wavelength);
    } catch (e) { return null; }
    if (!refl || refl.length === 0) return null;

    const qset = new Set();
    for (let i = 0; i < refl.length; i++) {
        const q = refl[i].q;
        if (isFinite(q) && q > 0) qset.add(q);
    }
    if (qset.size === 0) return null;
    const qsorted = new Float64Array(Array.from(qset)).sort((a, b) => a - b);

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
        const rt = mcParamsToCell(p0, system, max_volume);
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
        const baseCell = mcParamsToCell(p0, system, max_volume);
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

            const trialCell = mcParamsToCell(trialP, system, max_volume);
            if (!trialCell) continue;
            if (refineZero) trialCell.zero_correction = trialZ;

            // --- cheap smooth score -----------------------------------------
            let refl;
            try { refl = generateHKL_for_worker(trialCell, q_max, d_min, wavelength); }
            catch (e) { continue; }
            if (!refl || refl.length === 0) continue;
            const qset = new Set();
            for (let i = 0; i < refl.length; i++) {
                const q = refl[i].q;
                if (isFinite(q) && q > 0) qset.add(q);
            }
            if (qset.size === 0) continue;
            const qsorted = new Float64Array(Array.from(qset)).sort((a, b) => a - b);
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
        bestCell = mcParamsToCell(bestP, system, max_volume);
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

// --- SPACE GROUP / NIGGLI FUNCTIONS ---
// These are all now part of worker-logic.js; 
// ---
const getSymmetryForEquivCells = (a, b, c, alpha, beta, gamma, tol = 0.25) => getSymmetry(a,b,c,alpha,beta,gamma,tol);
const getVolumeForEquivCells = (cell) => getVolume(cell);

function cellToBasis(a, b, c, alpha, beta, gamma) {
  const ca = Math.cos(alpha * RAD), cb = Math.cos(beta * RAD), cg = Math.cos(gamma * RAD), sg = Math.sin(gamma * RAD);
  const ax = a, ay = 0, az = 0; const bx = b * cg, by = b * sg, bz = 0;
  const cx = c * cb; const cy = c * (ca - cb * cg) / sg;
  const cz2 = c * c - cx * cx - cy * cy; const cz = cz2 > 0 ? Math.sqrt(cz2) : 0;
  return [[ax, bx, cx], [ay, by, cy], [az, bz, cz]];
}
function basisToCell(B) {
  const a = Math.hypot(B[0][0], B[1][0], B[2][0]); const b = Math.hypot(B[0][1], B[1][1], B[2][1]); const c = Math.hypot(B[0][2], B[1][2], B[2][2]);
  const dot_ab = B[0][0] * B[0][1] + B[1][0] * B[1][1] + B[2][0] * B[2][1];
  const dot_ac = B[0][0] * B[0][2] + B[1][0] * B[1][2] + B[2][0] * B[2][2];
  const dot_bc = B[0][1] * B[0][2] + B[1][1] * B[1][2] + B[2][1] * B[2][2];
  const clamp01 = v => Math.max(-1, Math.min(1, v));
  const alpha = Math.acos(clamp01(dot_bc / (b * c))) * DEG; const beta = Math.acos(clamp01(dot_ac / (a * c))) * DEG; const gamma = Math.acos(clamp01(dot_ab / (a * b))) * DEG;
  return { a, b, c, alpha, beta, gamma };
}
function basisToMetric(B) {
  const v = (i, j) => B[0][i] * B[0][j] + B[1][i] * B[1][j] + B[2][i] * B[2][j];
  const G = [[v(0, 0), v(0, 1), v(0, 2)], [v(1, 0), v(1, 1), v(1, 2)], [v(2, 0), v(2, 1), v(2, 2)]];
  const A = G[0][0], Bm = G[1][1], C = G[2][2]; const zeta = 2 * G[0][1]; const eta = 2 * G[0][2]; const xi = 2 * G[1][2];
  return { G, A, B: Bm, C, xi, eta, zeta };
}
function rightMul(B, C) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 3; r++) { for (let j = 0; j < 3; j++) { out[r][j] = B[r][0] * C[0][j] + B[r][1] * C[1][j] + B[r][2] * C[2][j]; } }
  return out;
}
function matMul3(M, C) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) { out[i][j] = M[i][0] * C[0][j] + M[i][1] * C[1][j] + M[i][2] * C[2][j]; } }
  return out;
}
function I3() { return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; }
// Conventional-to-primitive transforms (columns = primitive vectors in terms of
// the conventional a,b,c). All are right-handed (det > 0). The A/B/C matrices
// preserve their own unique axis: A keeps a, B keeps b, C keeps c.
const primitiveTransformByCentering = { P: [[1,0,0],[0,1,0],[0,0,1]], A: [[1, 0, 0], [0, 0.5, -0.5], [0, 0.5, 0.5]], B: [[0.5, 0, -0.5], [0, 1, 0], [0.5, 0, 0.5]], C: [[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]], I: [[-0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5, -0.5]], F: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], R: [[ 2/3, -1/3, -1/3], [ 1/3,  1/3, -2/3], [ 1/3,  1/3,  1/3]], };






// ==========================================
// 1. THE ADAPTER (Call this one from your main code)
// ==========================================
function reduceToNiggliCell(sol, opts) {
    const a = sol.a, b = sol.b || sol.a, c = sol.c || sol.a;
    const alpha = sol.alpha ?? 90;
    const beta = sol.beta ?? 90;
    const gamma = sol.gamma ?? (sol.system === 'hexagonal' ? 120 : 90);

    // The Niggli cell and the higher-symmetry "squeeze" are questions about the
    // METRIC of the lattice we actually solved. They must not be driven by
    // sol.analysis.centering, which is a space-group *guess* from systematic
    // absences. That guess is unreliable exactly when reduction matters most:
    // a pseudo-symmetric cell (e.g. a hexagonal lattice forced into a monoclinic
    // setting) produces a false centering, and applying the corresponding
    // primitive transform extracts a half-volume SUBLATTICE with no relation to
    // the true symmetry. Reduce the solved cell's own lattice; callers wanting a
    // true-primitive reduction may pass opts.centering explicitly.
    const centering = opts && opts.centering ? opts.centering : 'P';

    // Call the engine
    return niggliReduceFromCell({ a, b, c, alpha, beta, gamma, centering }, opts);
}

// ==========================================
// 2. THE ENGINE (Robust Krivy-Gruber)
// ==========================================
// ==========================================
// 2. THE ENGINE (Robust Krivy-Gruber)
// ==========================================
function niggliReduceFromCell(cell, opts = {}) {
    const { a, b, c, alpha, beta, gamma, centering = 'P' } = cell;
    const maxIter = opts.maxIterations || 1000;
    let eps = opts.eps || 1e-5; 

    // Initialize Basis
    let B = cellToBasis(a, b, c, alpha, beta, gamma);
    let T = I3(); 

    // Robust Centering Extraction (prioritize non-primitive centerings F, I, R, A, B, C over P)
    let centeringKey = 'P';
    if (centering) {
        const str = String(centering).toUpperCase();
        const priority = ['F', 'I', 'R', 'A', 'B', 'C', 'P'];
        for (const cType of priority) {
            if (str.includes(`(${cType})`) || str.startsWith(cType) || str === cType) {
                centeringKey = cType;
                break;
            }
        }
    }

    const Cp = primitiveTransformByCentering[centeringKey];
    if (Cp && centeringKey !== 'P') {
        B = rightMul(B, Cp);
        T = matMul3(T, Cp);
    }

    // Metric Tensor Components
    let A, Bm, C_val, xi, eta, zeta;
    const updateMetric = () => {
        const dot = (i, j) => B[0][i]*B[0][j] + B[1][i]*B[1][j] + B[2][i]*B[2][j];
        A = dot(0,0); Bm = dot(1,1); C_val = dot(2,2);
        xi = 2 * dot(1,2); eta = 2 * dot(0,2); zeta = 2 * dot(0,1);
    };
    updateMetric();

    let iterations = 0;
    let changed = true;
    let converged = false;

    const applyTrans = (M) => {
        B = rightMul(B, M);
        T = matMul3(T, M);
        updateMetric();
        changed = true;
    };

    while (changed && iterations < maxIter) {
        changed = false;
        iterations++;

        // Gradually relax the tolerance if a near-degenerate metric stalls the
        // reduction, so ties resolve instead of cycling. Convergence is normally
        // reached in well under 20 iterations; maxIter is the real ceiling.
        if (iterations > 50) eps = 1e-4;
        if (iterations > 200) eps = 1e-3;

        // Step 1: Sort A <= B <= C (strictly preserving determinant = +1)
        if (A > Bm + eps || (Math.abs(A - Bm) <= eps && Math.abs(xi) > Math.abs(eta) + eps)) {
            applyTrans([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]); 
            continue;
        }
        if (Bm > C_val + eps || (Math.abs(Bm - C_val) <= eps && Math.abs(eta) > Math.abs(zeta) + eps)) {
            applyTrans([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]); 
            continue;
        }

        // Step 2: Sign Adjustment (Force strictly valid Type I or Type II)
        let s_xi = xi > eps ? 1 : (xi < -eps ? -1 : 0);
        let s_eta = eta > eps ? 1 : (eta < -eps ? -1 : 0);
        let s_zeta = zeta > eps ? 1 : (zeta < -eps ? -1 : 0);
        
        const allPositive = (s_xi > 0 && s_eta > 0 && s_zeta > 0);
        const allNonPositive = (s_xi <= 0 && s_eta <= 0 && s_zeta <= 0);
        
        if (allPositive || allNonPositive) {
            // Already valid Type I (all > 0) or Type II (all <= 0), proceed to reduction steps
        } else {
            // Transform mixed signs or zero-with-positive to valid Type I or Type II (det = +1).
            // Case 1: Two strictly negative, one strictly positive -> flip the two negatives to get Type I (+ + +)
            if (s_xi < 0 && s_eta < 0 && s_zeta > 0) { applyTrans([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]); continue; }
            if (s_xi < 0 && s_eta > 0 && s_zeta < 0) { applyTrans([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]); continue; }
            if (s_xi > 0 && s_eta < 0 && s_zeta < 0) { applyTrans([[1, 0, 0], [0, -1, 0], [0, 0, -1]]); continue; }
            
            // Case 2: Two positive, one non-positive -> flip the two positives to get Type II (- - -)
            if (s_xi > 0 && s_eta > 0 && s_zeta <= 0) { applyTrans([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]); continue; }
            if (s_xi > 0 && s_eta <= 0 && s_zeta > 0) { applyTrans([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]); continue; }
            if (s_xi <= 0 && s_eta > 0 && s_zeta > 0) { applyTrans([[1, 0, 0], [0, -1, 0], [0, 0, -1]]); continue; }
            
            // Case 3: One positive, two non-positive (with at least one zero) -> flip the positive to get Type II
            if (s_xi > 0 && s_eta <= 0 && s_zeta === 0) { applyTrans([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]); continue; }
            if (s_xi > 0 && s_eta === 0 && s_zeta < 0)  { applyTrans([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]); continue; }
            if (s_eta > 0 && s_xi <= 0 && s_zeta === 0) { applyTrans([[1, 0, 0], [0, -1, 0], [0, 0, -1]]); continue; }
            if (s_eta > 0 && s_xi === 0 && s_zeta < 0)  { applyTrans([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]); continue; }
            if (s_zeta > 0 && s_xi === 0 && s_eta <= 0) { applyTrans([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]); continue; }
            if (s_zeta > 0 && s_eta === 0 && s_xi < 0)  { applyTrans([[1, 0, 0], [0, -1, 0], [0, 0, -1]]); continue; }
        }

        // Step 3: Reduction
        if (Math.abs(xi) > Bm + eps || (Math.abs(xi - Bm) <= eps && 2*eta < zeta - eps) || (Math.abs(xi + Bm) <= eps && zeta < -eps)) {
            const s = Math.sign(xi) || 1; applyTrans([[1,0,0],[0,1,-s],[0,0,1]]); continue;
        }
        if (Math.abs(eta) > A + eps || (Math.abs(eta - A) <= eps && 2*xi < zeta - eps) || (Math.abs(eta + A) <= eps && zeta < -eps)) {
            const s = Math.sign(eta) || 1; applyTrans([[1,0,-s],[0,1,0],[0,0,1]]); continue;
        }
        if (Math.abs(zeta) > A + eps || (Math.abs(zeta - A) <= eps && 2*xi < eta - eps) || (Math.abs(zeta + A) <= eps && eta < -eps)) {
            const s = Math.sign(zeta) || 1; applyTrans([[1,-s,0],[0,1,0],[0,0,1]]); continue;
        }

        // Step 4: Body Diagonal 
        if ((xi + eta + zeta + A + Bm) < -eps || (Math.abs(xi + eta + zeta + A + Bm) <= eps && 2*(A + eta) + zeta > eps)) {
             applyTrans([[1, 0, 1], [0, 1, 1], [0, 0, 1]]); continue;
        }
    }
    // The loop ends either because no transform fired this pass (changed stayed
    // false -> reduced) or because the iteration ceiling was hit (not reduced).
    converged = !changed || iterations < maxIter;
    if (iterations >= maxIter) converged = false;

    const finalCell = basisToCell(B);
    // Clean numerical dust around 90 degrees
    const cleanAngle = (ang) => Math.abs(ang - 90) < 1e-10 ? 90 : ang;
    finalCell.alpha = cleanAngle(finalCell.alpha);
    finalCell.beta = cleanAngle(finalCell.beta);
    finalCell.gamma = cleanAngle(finalCell.gamma);

    const finalMetric = basisToMetric(B);
    return {
        cell: finalCell,
        transform: T,
        basis: B,
        metric: finalMetric.G,
        iterations: iterations,
        converged: converged
    };
}

function generateEquivalentCells(niggliCell, N_ignored, originalSystem = null) {
    const results = { primitiveCells: [], centeredCells: {} };
    if (!niggliCell || typeof niggliCell !== 'object' || !niggliCell.a) { console.error("Invalid Niggli cell provided."); return results; }
    const minAngle = 60.0, maxAngle = 150.0;
    const niggliSystemGuess = getSymmetryForEquivCells(niggliCell.a, niggliCell.b, niggliCell.c, niggliCell.alpha, niggliCell.beta, niggliCell.gamma);
    const niggliVolume = getVolumeForEquivCells({ ...niggliCell, system: niggliSystemGuess });
    results.primitiveCells.push({ ...niggliCell, description: "Reduced Cell (Niggli, centering not applied)", centering: 'P', volume: niggliVolume });
    if (originalSystem) {
        const niggliBasis = cellToBasis(niggliCell.a, niggliCell.b, niggliCell.c, niggliCell.alpha, niggliCell.beta, niggliCell.gamma);
        const primitiveToCenteredTransforms = { 'I': [[0,1,1],[1,0,1],[1,1,0]], 'F': [[-1,1,1],[1,-1,1],[1,1,-1]], 'A': [[1,0,0],[0,1,-1],[0,1,1]], 'B': [[1,0,1],[0,1,0],[-1,0,1]], 'C': [[1,-1,0],[1,1,0],[0,0,1]], 'R': [[1,0,1],[-1,1,1],[0,-1,1]] };
        const validBravaisCenterings = { 'cubic': ['P', 'I', 'F'], 'tetragonal': ['P', 'I'], 'orthorhombic': ['P', 'I', 'F', 'A', 'B', 'C'], 'hexagonal': ['P', 'R'], 'monoclinic': ['P', 'A', 'B', 'C', 'I'], 'triclinic': ['P'] };
        const allowedCenterings = validBravaisCenterings[originalSystem] || ['P'];
        for (const [centeringType, transform] of Object.entries(primitiveToCenteredTransforms)) {
            if (allowedCenterings.includes(centeringType) && centeringType !== 'P') {
                try {
                    const centeredBasis = rightMul(niggliBasis, transform); const centeredCellParams = basisToCell(centeredBasis);
                    if (Object.values(centeredCellParams).every(v => isFinite(v) && v > -1e-6) && [centeredCellParams.alpha, centeredCellParams.beta, centeredCellParams.gamma].every(a => a >= minAngle && a <= maxAngle)) {
                        const systemGuess = getSymmetryForEquivCells(centeredCellParams.a, centeredCellParams.b, centeredCellParams.c, centeredCellParams.alpha, centeredCellParams.beta, centeredCellParams.gamma);
                        const systemAllowed = (centeringType === 'I' && ['cubic', 'tetragonal', 'orthorhombic', 'monoclinic'].includes(systemGuess)) || (centeringType === 'F' && ['cubic', 'orthorhombic'].includes(systemGuess)) || (['A','B','C'].includes(centeringType) && ['orthorhombic', 'monoclinic'].includes(systemGuess)) || (centeringType === 'R' && ['hexagonal', 'trigonal'].includes(systemGuess));
                        if (systemAllowed) { const centeredVolume = getVolumeForEquivCells({ ...centeredCellParams, system: systemGuess }); results.centeredCells[centeringType] = { ...centeredCellParams, system: systemGuess, centering: centeringType, volume: centeredVolume, description: `Conventional ${centeringType}-centered` }; }
                    }
                } catch (error) { console.error(`Error transforming to ${centeringType}-centered cell:`, error); }
            }
        }
    }
    return results;
}

// --- groups; utilise cctbx
function analyzeSystematicAbsences(solution, obs_peaks, spaceGroupData, wavelength, tthError, tthMax, impurity_peaks) {
    const MAX_VIOLATIONS = 2;
    const fallbackResult = {
        centering: 'Unknown',
        rankedSpaceGroups: [],
        detectedExtinctions: [],
        ambiguousHkls: new Set(),
        hklList:[]
    };
    if (!spaceGroupData?.space_groups) { console.warn("Space group data not loaded"); return fallbackResult; }
    const all_calc_hkls = generateHKL_for_analysis(solution, wavelength, tthMax);

    if (all_calc_hkls.length === 0) {
    fallbackResult.hklList = [];
    return fallbackResult;
}
    
    // Carry the per-peak Ka2-suspect flag through the indexing step. Each
    // observed peak that matches a calculated hkl produces an indexed_hkl
    // record; we tag the record with the parent peak's ka2Suspect flag so the
    // downstream centering / extinction / ranking code can distinguish
    // "hard" violations (driven by genuine reflections) from "soft" ones
    // (driven by suspected Ka2 ghost peaks).
    const indexed_hkls = []; const zero_correction = solution.zero_correction || 0;
    // Two windows around each observed peak:
    //   indexWindow  - the indexing tolerance proper (1.5 * tthError).
    //                  A peak is indexed only if its closest calculated
    //                  hkl is inside this window.
    //   overlapWindow - same width as indexWindow. Used to detect
    //                   *overlapping* reflections within one peak. The
    //                   IUCr International Tables note that for powder
    //                   data, peak overlap is the dominant source of
    //                   ambiguity in systematic-absence detection: a
    //                   "forbidden" reflection that overlaps an allowed
    //                   one cannot be used as hard evidence against a
    //                   space group, because its observed intensity is
    //                   contaminated. The principled solution is the
    //                   Bayesian intensity-based test of Markvardsen et
    //                   al. (Acta Cryst. A57, 47, 2001; ExtSym/DASH),
    //                   but in the absence of integrated intensities the
    //                   overlap check is a useful proxy: if a violating
    //                   hkl overlaps an allowed neighbour within one
    //                   tolerance bar, the violation is downgraded from
    //                   hard to soft. Using the SAME window as indexing
    //                   keeps the rule conservative and well defined.
    const indexWindow   = tthError * 1.5;
    const overlapWindow = tthError * 1.5;

    // --- Intensity-aware demotion ---
    // A "forbidden" reflection in the true space group should have
    // intrinsic intensity ZERO. If the observed peak that drives a
    // violation is weak compared with the strong peaks of the pattern,
    // it is much more likely to be a tail / wing / weak overlap / noise
    // artefact than a genuine reflection contradicting the space group.
    // We therefore tag each indexed peak as "lowIntensity" when its
    // height is below 10% of the strongest observed peak in the pattern;
    // such peaks count as soft (not hard) evidence in countViolations
    // and determineCentering. This is the same intuition that drives
    // the Bayesian ExtSym algorithm (Markvardsen 2001), without the
    // full Wilson-distribution machinery that requires properly
    // background-subtracted, Lorentz-polarisation-corrected integrated
    // intensities. Heights are missing for older callers that did not
    // pass them through; in that case the threshold is never triggered
    // and behaviour is identical to the position-only check.
    // The baseline must be LOCAL in 2theta, not a fraction of the global
    // maximum. Diffracted intensity falls off with angle (Lorentz-polarisation,
    // Debye-Waller, absorption), so a global 5%-of-max cut does not mean "weak"
    // — above roughly 90 deg it means "high angle", and it silently strips the
    // whole back-reflection region of any power to falsify an absence rule.
    // Comparing each peak with peaks at COMPARABLE 2theta keeps the original
    // crystallographic intent (a truly forbidden reflection has zero intensity)
    // without the angular bias.
    const LOW_INTENSITY_FRACTION = 0.05;
    const LOCAL_WINDOW_DEG = 15.0;

    const heightedPeaks = obs_peaks.filter(p => typeof p.height === 'number' && isFinite(p.height));
    let globalMaxHeight = 0;
    for (const p of heightedPeaks) if (p.height > globalMaxHeight) globalMaxHeight = p.height;

    const lowIntensityThresholdAt = (tth) => {
        if (globalMaxHeight <= 0) return -Infinity; // no heights -> disable demotion
        let localMax = 0;
        for (const p of heightedPeaks) {
            if (Math.abs(p.tth - tth) <= LOCAL_WINDOW_DEG && p.height > localMax) localMax = p.height;
        }
        // Fall back to the global scale only if the local window is empty.
        const scale = localMax > 0 ? localMax : globalMaxHeight;
        return scale * LOW_INTENSITY_FRACTION;
    };

    // Assignments the user set explicitly via "Swap hkl". These override the
    // nearest-line rule wherever they apply - otherwise the analysis of a
    // swapped solution would quietly revert to the indexing the user rejected.
    const manualByTth = new Map();
    for (const sw of (solution.manualSwaps || [])) {
        if (sw && Number.isFinite(sw.h) && Number.isFinite(sw.k) && Number.isFinite(sw.l)) {
            manualByTth.set(Number(sw.tth).toFixed(4), sw);
        }
    }

    obs_peaks.forEach(peak => {
        const corrected_tth = peak.tth - zero_correction;
        let bestMatch = all_calc_hkls.reduce((best, hkl) => { const diff = Math.abs(hkl.tth - corrected_tth); return diff < best.minDiff ? { hkl, minDiff: diff } : best; }, { hkl: null, minDiff: Infinity });
        const man = manualByTth.get(Number(peak.tth).toFixed(4));
        if (man) {
            const forced = all_calc_hkls.find(x => x.h === man.h && x.k === man.k && x.l === man.l);
            // Honour it even if that line is not the nearest; only skip when the
            // reflection does not exist for this lattice at all.
            if (forced) bestMatch = { hkl: forced, minDiff: 0 };
        }
        if (bestMatch.hkl && bestMatch.minDiff < indexWindow) {
            // Collect every calculated hkl whose 2theta is within
            // overlapWindow of the BEST-MATCH calculated 2theta (not of
            // the observed 2theta). The overlap is between the candidate
            // reflection and its neighbours in reciprocal space — that
            // is what determines whether intensity from one can leak
            // into the other.
            const altHkls = all_calc_hkls
                .filter(hkl => Math.abs(hkl.tth - bestMatch.hkl.tth) < overlapWindow)
                .map(hkl => ({ h: hkl.h, k: hkl.k, l: hkl.l, tth: hkl.tth }));
            const peakHeight = (typeof peak.height === 'number' && isFinite(peak.height)) ? peak.height : null;
            const isLowIntensity = (peakHeight !== null) && (peakHeight < lowIntensityThresholdAt(peak.tth));
            indexed_hkls.push({
                h: bestMatch.hkl.h, k: bestMatch.hkl.k, l: bestMatch.hkl.l,
                tth: peak.tth, calc_tth: bestMatch.hkl.tth,
                ka2Suspect: !!peak.ka2Suspect,
                altHkls: altHkls,
                tol: tthError,
                height: peakHeight,
                lowIntensity: isLowIntensity
            });
        }
    });

    // Dedup by (h,k,l). When two observed peaks index to the same hkl, prefer
    // the NON-suspect, HIGH-intensity one (strong, real evidence outweighs
    // weak or Ka2-ghost evidence for the same reflection). This avoids
    // accidentally flipping a real reflection to "soft" just because a Ka2
    // ghost or a weak-tail peak from a different parent happened to index
    // to the same hkl by coincidence.
    const uniqMap = new Map();
    const recordPriority = (r) => {
        // Higher = preferred. Non-suspect > suspect; high-intensity > low.
        let p = 0;
        if (!r.ka2Suspect) p += 2;
        if (!r.lowIntensity) p += 1;
        return p;
    };
    for (const r of indexed_hkls) {
        const key = `${r.h},${r.k},${r.l}`;
        const existing = uniqMap.get(key);
        if (!existing) { uniqMap.set(key, r); continue; }
        if (recordPriority(r) > recordPriority(existing)) uniqMap.set(key, r);
        // Otherwise keep first (existing).
    }
    const unique_indexed_hkls = Array.from(uniqMap.values());

    const unambiguous_hkls = unique_indexed_hkls.filter(refl => {
        const nearbyCount = all_calc_hkls.filter(calc => { if (calc.h === refl.h && calc.k === refl.k && calc.l === refl.l) return false; return Math.abs(calc.tth - refl.calc_tth) < tthError; }).length;
        return nearbyCount === 0;
    });

    // Do NOT restrict the analysis to isolated reflections. Overlap is handled
    // per-reflection downstream (the altHkls demotion in countViolations /
    // determineCentering), so pre-filtering here is both redundant and harmful:
    // in a pseudo-symmetric cell nearly every reflection overlaps a neighbour,
    // so this filter used to discard ~3/4 of the indexed peaks and leave a
    // biased remnant. The set it kept was precisely the set least able to
    // falsify a centering. Keep every unique indexed reflection and let the
    // hard/soft accounting weigh them.
    const hkls_for_analysis = unique_indexed_hkls;
    if (hkls_for_analysis.length < 5) { fallbackResult.centering = 'Unknown (too few unambiguous peaks in range)'; return fallbackResult; }
    const unambiguousSet = new Set(unambiguous_hkls.map(r => `${r.h},${r.k},${r.l}`));
    const ambiguousHkls = new Set(unique_indexed_hkls.filter(r => !unambiguousSet.has(`${r.h},${r.k},${r.l}`)).map(r => `${r.h},${r.k},${r.l}`));

    const anyKa2Suspects = hkls_for_analysis.some(r => r.ka2Suspect);

    const centeringResult = determineCentering(hkls_for_analysis, solution.system);
    const detectedExtinctions = detectExtinctions(hkls_for_analysis, solution.system, spaceGroupData);
    // NOTHING is re-assigned here. The analysis reports what it finds and leaves
    // the indexing alone: silently rewriting an hkl behind the user's back is
    // exactly the behaviour this was changed to avoid. Correcting an assignment
    // is a deliberate act, done through the "Swap hkl" command, which produces a
    // separate solution the user can compare against this one.
    const rankedSpaceGroups = rankSpaceGroups(hkls_for_analysis, solution.system, centeringResult.plausibleCenterings, spaceGroupData, MAX_VIOLATIONS, detectedExtinctions);
    return {
        centering: centeringResult.description,
        rankedSpaceGroups: rankedSpaceGroups.slice(0, 20),
        detectedExtinctions: detectedExtinctions,
        centeringViolations: centeringResult.violations,
        centeringViolationsHard: centeringResult.violationsHard,
        centeringViolationsSoft: centeringResult.violationsSoft,
        centeringViolationDetails: centeringResult.violationDetails,
        ambiguousHkls: ambiguousHkls,
        hklList: all_calc_hkls,
        usedKa2SoftScoring: anyKa2Suspects
    };
}
// How much worse than the assigned reflection an allowed alternative may fit
// and still count as a genuine competitor.
const AMBIGUITY_MARGIN = 2.0;

// --- EXTINCTION-AWARE CELL RE-REFINEMENT ---
// Re-assigning hkl labels alone is cosmetic, and worse, it makes the report
// internally inconsistent: the cell was least-squares fitted against the OLD
// pairing, so after relabelling, the diff column measures the new hkl against a
// cell refined to the old one. For PbSO4 the 16.423 deg peak reads 0.023 off as
// (1,0,1), against 0.019 as the forbidden (0,1,0) - the corrected assignment
// looks worse purely because the cell was pulled toward 010 while fitting.
//
// The fix is to redo the fit with an extinction-filtered line list. Restricting
// the candidate reflections to those the detected rules allow makes the pairing,
// the cell, the zero error, the ESDs and the figures of merit all consistent
// with the table in one step, because every one of them derives from that list.
//
// M20 improves for two independent reasons: <|dQ|> falls because the pairing is
// correct, and N20 falls because extinct lines are no longer counted as
// possible. This is reported alongside the original rather than replacing it,
// so the effect of the constraint stays visible and auditable.
// --- MANUAL HKL SWAP ---
// Indexing assigns each peak to its nearest calculated line, and that choice can
// be wrong without producing any violation at all: in a permissive space group
// (P222 and friends) both candidates are allowed, so nothing flags the swap. A
// rule-driven search cannot find those cases by construction - it only ever sees
// assignments that some space group forbids - so the decision is handed to the
// user instead.
//
// getPeakAssignments() reports what the indexer currently believes, and
// refineWithManualHkl() re-fits the cell with chosen assignments overridden.
// Figures of merit are computed the ordinary way against the FULL line list, so
// the resulting solution is directly comparable with every other in the table.

// What is each observed peak currently indexed as?
function getPeakAssignments(solution, obs_peaks, wavelength, tthError, tthMax, limit) {
    const lines = generateHKL_for_analysis(solution, wavelength, tthMax);
    if (!lines.length) return [];
    const zero = solution.zero_correction || 0;
    const window = tthError * 1.5;
    const out = [];
    const peaks = (obs_peaks || [])
        .filter(p => typeof p.tth === 'number' && isFinite(p.tth))
        .slice().sort((x, y) => x.tth - y.tth);
    for (const p of peaks) {
        const tc = p.tth - zero;
        let best = null, bd = Infinity;
        for (const L of lines) { const d = Math.abs(L.tth - tc); if (d < bd) { bd = d; best = L; } }
        const inRange = best && bd <= window;
        const dObs = wavelength / (2 * Math.sin(tc * RAD / 2));
        out.push({
            tth: p.tth, tth_corr: tc,
            h: inRange ? best.h : null, k: inRange ? best.k : null, l: inRange ? best.l : null,
            calc_tth: inRange ? best.tth : null,
            diff: inRange ? (tc - best.tth) : null,
            d_obs: isFinite(dObs) ? dObs : null,
            d_calc: inRange ? best.d : null,
            indexed: !!inRange
        });
        if (limit && out.length >= limit) break;
    }
    return out;
}

// Re-fit with user-supplied assignments. `overrides` is a list of
// { tth, h, k, l }; every other peak keeps its nearest-line assignment.
// Returns { cell, swaps } on success or { error } with a reason, so the caller
// can tell the user exactly why nothing happened.
function refineWithManualHkl(solution, obs_peaks, overrides, wavelength, tthError, tthMax, refineZero, impurity_peaks, isAuto = false) {
const system = solution.system;
    if (!system) return { error: 'solution has no crystal system' };
    const lines = generateHKL_for_analysis(solution, wavelength, tthMax);
    if (!lines.length) return { error: 'no calculated reflections for this cell' };

    const zero = solution.zero_correction || 0;
    const window = tthError * 1.5;
    const ovr = new Map();
    for (const o of (overrides || [])) {
        if (o == null) continue;
        const h = Math.round(Number(o.h)), k = Math.round(Number(o.k)), l = Math.round(Number(o.l));
        if (![h, k, l].every(Number.isFinite)) continue;
        if (h === 0 && k === 0 && l === 0) return { error: '(0,0,0) is not a reflection' };
        ovr.set(Number(o.tth).toFixed(4), { h, k, l });
    }
    if (ovr.size === 0) return { error: 'no changes to apply' };

    const toQ = (t) => (4 * Math.sin(t * RAD / 2) ** 2) / (wavelength ** 2);
    const peaks = (obs_peaks || []).filter(p => typeof p.tth === 'number' && isFinite(p.tth))
                                   .slice().sort((x, y) => x.tth - y.tth);
    const rows = [], qv = [], tthRads = [], swaps = [];
    for (const p of peaks) {
        const key = p.tth.toFixed(4);
        const tc = p.tth - zero;
        let best = null, bd = Infinity;
        for (const L of lines) { const d = Math.abs(L.tth - tc); if (d < bd) { bd = d; best = L; } }
        let hkl = (best && bd <= window) ? { h: best.h, k: best.k, l: best.l } : null;
        if (ovr.has(key)) {
            const man = ovr.get(key);
            // A manual assignment is honoured even when it is not the nearest
            // line, and even when it falls outside the indexing window. That is
            // the entire point: the user is overruling the nearest-line rule.
            swaps.push({
                tth: p.tth,
                h: man.h, k: man.k, l: man.l,      // numeric, so downstream
                from: hkl ? `(${hkl.h},${hkl.k},${hkl.l})` : '(unindexed)',
                to: `(${man.h},${man.k},${man.l})` // consumers need not re-parse
            });
            hkl = man;
        }
        if (!hkl) continue;                      // unindexed and untouched: skip
        const row = getLSDesignRow([hkl.h, hkl.k, hkl.l], system);
        if (!row) continue;
        if (refineZero) row.push((2 / (wavelength ** 2)) * Math.sin(p.tth * RAD));
        rows.push(row); qv.push(toQ(p.tth)); tthRads.push(p.tth * RAD);
    }
    const minIndexed = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 };
    const need = (minIndexed[system] || 6) + (refineZero ? 1 : 0);
    if (rows.length < need) return { error: `only ${rows.length} indexed peaks; ${need} needed for ${system}` };

    const fit = solveLeastSquares(rows, qv, ls_weights_for_2theta(tthRads));
    if (!fit || !fit.solution) return { error: 'least-squares fit failed (singular design matrix?)' };
    const cell = extractCellFromFit(fit.solution, system);
    if (!cell) return { error: 'fit did not yield a valid cell' };
    cell.system = system;
    if (refineZero) cell.zero_correction = fit.solution[fit.solution.length - 1] * DEG;
    cell.volume = getVolume(cell);
    if (!isFinite(cell.volume) || cell.volume <= 0) return { error: 'refined cell has non-physical volume' };
    try { cell.errors = propagateErrors(system, fit, cell); } catch (e) { cell.errors = null; }

    // Figures of merit exactly as for any other solution: full line list, so the
    // number is directly comparable with the parent and with independent hits.
    try {
        const refLines = generateHKL_for_analysis(cell, wavelength, tthMax);
        const qSorted = new Float64Array(Array.from(new Set(refLines.map(r => toQ(r.tth))))).sort((a, b) => a - b);
        const mk = (n) => peaks.slice(0, n).map((p, i) => {
            const tc2 = p.tth - (cell.zero_correction || 0);
            return { ...p, original_index: i, q: toQ(tc2), tth: tc2 };
        });
        const tolFor = (arr) => (i) => {
            const th = (arr[i] ? arr[i].tth : 0) * RAD / 2;
            return ((8 * Math.sin(th) * Math.cos(th)) / (wavelength ** 2)) * (tthError * Math.PI / 360) + 1e-9;
        };
        const p20 = mk(Math.min(20, peaks.length));
        const f20 = calculateFiguresOfMerit(qSorted, p20, impurity_peaks || 0, tolFor(p20), wavelength);
        cell.m20 = f20.m20; cell.fN_20 = f20.fN; cell.n_20 = p20.length;
        const pAll = mk(peaks.length);
        const fAll = calculateFiguresOfMerit(qSorted, pAll, impurity_peaks || 0, tolFor(pAll), wavelength);
        cell.m_all = fAll.m20; cell.fN_all = fAll.fN; cell.n_all = pAll.length;
    } catch (e) { /* FoM optional */ }
    if (!isFinite(cell.m20)) cell.m20 = 0;

// If it's an auto-swap, don't append it to the manual tracking log!
    cell.manualSwaps = isAuto ? (solution.manualSwaps || []) : (solution.manualSwaps || []).concat(swaps);
    cell.nPaired = rows.length;
    return { cell: cell, swaps: swaps };
}

// Indexing is extinction-blind: each observed peak is assigned to the NEAREST
// calculated line, whatever that line's parity, because the absence rules do
// not exist yet at that point. When two lines straddle a peak the nearest one
// can easily be systematically absent. PbSO4 (anglesite, Pnma) is the worked
// example: the peak at 16.423 deg is assigned (0,1,0) at 16.404 (0.019 off)
// rather than (1,0,1) at 16.447 (0.024 off) - a 0.005 deg margin - even though
// 010 breaks 0kl: k+l=2n and cannot exist in that space group.
//
// Once detectExtinctions() has established the absence rules, that decision can
// be revisited: a peak whose assignment is forbidden is re-assigned to the
// nearest ALLOWED line, provided one lies inside the same indexing window.
//
// Three guards keep this from becoming a self-fulfilling loop:
//   1. If no allowed line is in range the original assignment is KEPT. Such a
//      peak is genuine evidence against the rules and must not be hidden.
//   2. The rules are NOT re-derived afterwards. Re-running detectExtinctions on
//      re-assigned data would confirm them trivially, since the data were just
//      edited to satisfy them.
//   3. A single pass, with a ceiling on how much may be re-assigned. Needing to
//      move a large fraction of the pattern means the rules are wrong, not the
//      assignments, so the whole pass is abandoned.


/**
 * Is there an ALLOWED alternative hkl that explains this observed peak about as
 * well as the assigned (rule-violating) one?
 *
 * The old test asked only whether an allowed hkl existed anywhere in the overlap
 * window. That is far too permissive, and it degrades badly with wavelength.
 * Peak separation follows d(2theta) = 2*tan(theta) * (dd/d), so a fixed 2theta
 * window corresponds to a lattice resolution of (dd/d) = d(2theta)/(2 tan theta)
 * — which blows up as theta falls. Short-wavelength anodes push the whole
 * pattern to low 2theta, so the same window swallows far more reflections. For
 * one real cell (the FAP monoclinic solution) the mean number of calculated hkl
 * within +-0.06 deg is ~4 for Cr, ~6 for Cu, ~15 for Mo and ~20 for Ag. Under a
 * mere presence test, essentially every violation on Mo/Ag data is demoted and
 * the extinction analysis stops working.
 *
 * Note the fix is NOT a wavelength factor on the window. Instrumental 2theta
 * uncertainty is set by alignment, sample displacement and detector resolution,
 * and is very nearly independent of the anode — so widening or narrowing the
 * window per wavelength would be inventing physics. What was wrong is the
 * binary test. Requiring the alternative to be COMPETITIVE with the assigned
 * reflection adapts automatically: on a crowded short-wavelength pattern the
 * extra neighbours are mostly far from the observed position and no longer
 * excuse the violation, while a genuine near-coincidence still does.
 *
 * A candidate competes if it lies within the user's stated tolerance of the
 * observed peak, or fits no more than AMBIGUITY_MARGIN times worse than the
 * assigned reflection does.
 */
function hasCompetingAllowedAlt(refl, isAllowed) {
    const alts = refl && refl.altHkls;
    if (!alts || alts.length <= 1) return false;

    const isSame = (a) => a.h === refl.h && a.k === refl.k && a.l === refl.l;
    const obs = refl.tth;

    // Without an observed position (or without per-alt positions) we cannot
    // judge proximity; fall back to the original presence test.
    if (typeof obs !== 'number' || !isFinite(obs)) {
        return alts.some(a => !isSame(a) && isAllowed(a));
    }

    const assignedTth = (typeof refl.calc_tth === 'number' && isFinite(refl.calc_tth))
        ? refl.calc_tth : obs;
    const dAssigned = Math.abs(assignedTth - obs);
    const floor = (typeof refl.tol === 'number' && isFinite(refl.tol)) ? refl.tol : Infinity;
    const limit = Math.max(dAssigned * AMBIGUITY_MARGIN, floor);

    return alts.some(a => {
        if (isSame(a) || !isAllowed(a)) return false;
        if (typeof a.tth !== 'number' || !isFinite(a.tth)) return true; // unknown -> old behaviour
        return Math.abs(a.tth - obs) <= limit;
    });
}

function determineCentering(indexed_hkls, system) {
    const centeringTests = { 'P': { name: 'Primitive (P)', forbidden: (h, k, l) => false }, 'I': { name: 'Body-centered (I)', forbidden: (h, k, l) => (h + k + l) % 2 !== 0 }, 'F': { name: 'Face-centered (F)', forbidden: (h, k, l) => !( (h%2===0 && k%2===0 && l%2===0) || (h%2!==0 && k%2!==0 && l%2!==0) ) }, 'A': { name: 'A-centered (A)', forbidden: (h, k, l) => (k + l) % 2 !== 0 }, 'B': { name: 'B-centered (B)', forbidden: (h, k, l) => (h + l) % 2 !== 0 }, 'C': { name: 'C-centered (C)', forbidden: (h, k, l) => (h + k) % 2 !== 0 } };
    const validBravaisCenterings = { 'cubic': ['P', 'I', 'F'], 'tetragonal': ['P', 'I'], 'orthorhombic': ['P', 'I', 'F', 'A', 'B', 'C'], 'hexagonal': ['P'], 'monoclinic': ['P', 'A', 'B', 'C', 'I'], 'triclinic': ['P'] };
    // Two parallel violation tallies: hard (non-Ka2-suspect peaks) and soft
    // (Ka2-suspect peaks). The centering decision uses HARD only — a centering
    // mode is not ruled out just because a Ka2 ghost happens to violate it.
    const violations = {};       // total (hard + soft), kept for backward compat
    const violationsHard = {};
    const violationsSoft = {};
    const violationDetails = {};
    const MAX_DETAILS_TO_STORE = 2;
    for (const [key, test] of Object.entries(centeringTests)) {
        const allowedForSystem = validBravaisCenterings[system] || ['P'];
        if (allowedForSystem.includes(key)) {
            const violatingPeaks = indexed_hkls.filter(({h, k, l}) => test.forbidden(Math.round(h), Math.round(k), Math.round(l)));
            // A peak whose best-match hkl violates the centering rule
            // is downgraded from hard to soft if any of these apply:
            //   1. it is a Ka2-suspect ghost,
            //   2. it has an allowed alternative hkl within the
            //      overlap window (the rule could equally apply to a
            //      neighbour), or
            //   3. its observed intensity is below the low-intensity
            //      threshold (a near-zero peak is not strong evidence
            //      against the centering rule, since forbidden peaks
            //      should have intensity zero in the true cell).
            // This protects high-symmetry centerings from being killed
            // by a single ambiguous or weak hkl assignment.
            const hardViolatingPeaks = violatingPeaks.filter(p => {
                if (p.ka2Suspect) return false;
                if (p.lowIntensity) return false;
                if (hasCompetingAllowedAlt(p, alt => !test.forbidden(Math.round(alt.h), Math.round(alt.k), Math.round(alt.l)))) {
                    return false;
                }
                return true;
            });
            let softViolatingPeaks = violatingPeaks.filter(p => !hardViolatingPeaks.includes(p));

            // --- CONSENSUS OVERRIDE (see countViolations for rationale) ---
            // A centering mode must not survive on the strength of demotions
            // alone. Ka2 ghosts, weak tails and overlaps do not preferentially
            // land on the forbidden parity class; a large soft pile is evidence
            // that the centering is simply wrong.
            const CONSENSUS_MIN_COUNT = 5;
            const CONSENSUS_MIN_FRACTION = 0.15;
            let effectiveHard = hardViolatingPeaks;
            let effectiveSoft = softViolatingPeaks;
            if (effectiveHard.length === 0 &&
                effectiveSoft.length >= CONSENSUS_MIN_COUNT &&
                effectiveSoft.length >= CONSENSUS_MIN_FRACTION * indexed_hkls.length) {
                effectiveHard = effectiveSoft;
                effectiveSoft = [];
            }

            violations[key] = violatingPeaks.length; // total
            violationsHard[key] = effectiveHard.length;
            violationsSoft[key] = effectiveSoft.length;
            const hardViolatingPeaksFinal = effectiveHard;
            const softViolatingPeaksFinal = effectiveSoft;
            // Details: store hard violators preferentially, fall back to soft.
            if (violations[key] > 0 && violations[key] <= MAX_DETAILS_TO_STORE) {
                const detailsSource = hardViolatingPeaksFinal.length > 0 ? hardViolatingPeaksFinal : softViolatingPeaksFinal;
                violationDetails[key] = detailsSource.slice(0, MAX_DETAILS_TO_STORE).map(p => ({ h: p.h, k: p.k, l: p.l, tth: p.tth, ka2Suspect: !!p.ka2Suspect, lowIntensity: !!p.lowIntensity }));
            }
        }
    }
    // Pick centering(s) with the FEWEST HARD violations (was: fewest total).
    const hardKeys = Object.keys(violationsHard);
    const minHardViolations = hardKeys.length > 0 ? Math.min(...hardKeys.map(k => violationsHard[k])) : 0;
    let plausible = hardKeys.filter(key => violationsHard[key] === minHardViolations && (validBravaisCenterings[system] || ['P']).includes(key));
    if (plausible.length === 0 && violationsHard['P'] === minHardViolations) { plausible = ['P']; } else if (plausible.length === 0) { plausible = ['P']; }
    let finalCenterings;
    if (plausible.includes('F')) finalCenterings = ['F'];
    else if (plausible.includes('I')) finalCenterings = ['I'];
    else { const specialCenterings = plausible.filter(c => ['A', 'B', 'C'].includes(c)); finalCenterings = specialCenterings.length > 0 ? specialCenterings : ['P']; if (plausible.includes('P') && !finalCenterings.includes('P') && specialCenterings.length > 0) { finalCenterings.push('P'); } if (finalCenterings.length === 0) finalCenterings = ['P']; }
    finalCenterings = finalCenterings.filter(c => (validBravaisCenterings[system] || ['P']).includes(c));
    if (finalCenterings.length === 0) finalCenterings = ['P'];

    // --- REPORTED centering vs SEARCHED centering ---
    // P is defined with forbidden() === false, so it can never accumulate a
    // violation and is therefore ALWAYS among the zero-violation candidates.
    // Listing it next to a real centering ("A-centered (A) or Primitive (P)")
    // is structurally guaranteed rather than informative, and it is not done
    // for F or I, which are collapsed to a single symbol above. So the
    // human-readable description reports only the genuine centering.
    //
    // finalCenterings itself keeps P, because it is passed to rankSpaceGroups()
    // as the allowed-centering filter: dropping it there would remove every
    // primitive space group from the ranking. Whether the lattice is primitive
    // is settled by the space-group analysis and the extinction list, not by
    // this line. P remains relevant to cell reduction, where it means
    // "no centering transform applied".
    let reportedCenterings = finalCenterings.filter(c => c !== 'P');
    if (reportedCenterings.length === 0) reportedCenterings = ['P'];
    
    // Remove Primitive (P) from the reported dictionaries since it cannot have violations
    delete violations['P'];
    delete violationsHard['P'];
    delete violationsSoft['P'];
    delete violationDetails['P'];
    
    
    return {
        plausibleCenterings: finalCenterings,
        reportedCenterings: reportedCenterings,
        description: reportedCenterings.map(c => centeringTests[c]?.name || c).join(' or '),
        violations: violations,
        violationsHard: violationsHard,
        violationsSoft: violationsSoft,
        violationDetails: violationDetails,
        minViolations: minHardViolations
    };
}
function detectExtinctions(indexed_hkls, system, spaceGroupData) {
    const confirmedRules = new Set();
    if (!spaceGroupData?.space_groups || indexed_hkls.length === 0) { return ["None detected (no data or rules)"]; }
    const potentialRules = new Set();
    Object.values(spaceGroupData.space_groups).forEach(sg => { if (sg.crystal_system === system) { sg.settings.forEach(setting => { const conditions = setting.reflection_conditions || {}; Object.entries(conditions).forEach(([zone, condList]) => { condList.forEach(condStr => { potentialRules.add(`${zone}: ${condStr}`); }); }); }); } });
    if (potentialRules.size === 0) { return ["None detected (no rules for system)"]; }
    const parseRuleString = (ruleStr) => { const parts = ruleStr.split(': '); if (parts.length === 2) { return { zone: parts[0].trim(), condition: parts[1].trim() }; } return null; };
    potentialRules.forEach(ruleStr => {
        const parsedRule = parseRuleString(ruleStr); if (!parsedRule) return;
        const { zone, condition } = parsedRule;
            const zoneReflections = indexed_hkls.filter(refl => zoneApplies(zone, refl.h, refl.k, refl.l));
        if (zoneReflections.length === 0) { return; }
        // A rule is "confirmed" if every reliable reflection in the
        // zone satisfies it. Two classes of unreliable reflections are
        // excluded from this check:
        //   - Ka2-suspect peaks: their 2θ is shifted, so we can't
        //     reliably re-index them back to the same hkl;
        //   - low-intensity peaks: a near-zero observed peak doesn't
        //     contradict an extinction rule even if its assigned hkl
        //     formally violates the rule (forbidden peaks SHOULD be
        //     near zero).
        // If no reliable reflections remain in the zone we fall back
        // to the full set so the test can still operate.
        const reliableZoneRefls = zoneReflections.filter(r => !r.ka2Suspect && !r.lowIntensity);
        const refSetForRule = reliableZoneRefls.length > 0 ? reliableZoneRefls : zoneReflections;
        // A reflection that formally breaks the rule is forgiven if another
        // calculated hkl within the same peak's ambiguity window is not
        // forbidden by it — the peak could equally well be that reflection.
        // This is the SAME demotion countViolations() applies when ranking; without
        // it the two analyses disagree, and a rule the ranking is happy to treat
        // as satisfied never appears in the detected list. The rule does not
        // constrain an alternative outside its own zone, so such an alternative
        // counts as allowed.
        const ruleAllows = (a) => !zoneApplies(zone, a.h, a.k, a.l) ||
                                  satisfiesCondition(a.h, a.k, a.l, condition);
        let hardFails = 0, forgiven = 0;
        for (const refl of refSetForRule) {
            if (satisfiesCondition(refl.h, refl.k, refl.l, condition)) continue;
            if (hasCompetingAllowedAlt(refl, ruleAllows)) forgiven++;
            else hardFails++;
        }
        // Consensus guard, mirroring countViolations(): forgiving one overlap is
        // reasonable, forgiving a systematic trend is not. Overlap is uncorrelated
        // with index parity, so if many reflections in the zone all break the same
        // rule, the rule is genuinely broken however excusable each case looks.
        const EXT_CONSENSUS_MIN_COUNT = 5;
        const EXT_CONSENSUS_MIN_FRACTION = 0.15;
        const consensusBroken = forgiven >= EXT_CONSENSUS_MIN_COUNT &&
                                forgiven >= EXT_CONSENSUS_MIN_FRACTION * refSetForRule.length;
        const allSatisfy = (hardFails === 0) && !consensusBroken;
        if (allSatisfy) { confirmedRules.add(ruleStr); }
    });
    if (confirmedRules.size === 0) { return ["None detected"]; }

    // --- COLLAPSE SUBSUMED RULES ---
    // A rule set built by "keep everything nothing contradicts" is riddled with
    // redundancy. Two distinct cases:
    //
    //  (a) one rule implies another. If the only observed 00l is 004, both
    //      00l: l=2n and 00l: l=4n survive, because l=4n implies l=2n. l=4n is
    //      the stronger claim (it also forbids 002 and 006), so it is the only
    //      one carrying information.
    //  (b) a rule is implied by the CONJUNCTION of others without being implied
    //      by any single one. Zektzerite reports hk0: h=2n, hk0: k=2n AND
    //      hk0: h+k=2n; the third follows from the first two (even + even is
    //      even) but from neither alone, so a pairwise test cannot see it.
    //
    // A rule is therefore dropped when every reflection allowed by ALL the
    // other surviving rules already satisfies it. Removal is iterative, so a
    // set of mutually-redundant rules collapses to a minimal equivalent subset
    // rather than vanishing entirely. Rules using more indices are offered for
    // removal first, so the conventional form (h=2n, k=2n) is kept over the
    // derived combination (h+k=2n).
    //
    // Implication is tested by enumeration, not algebraically, so compound
    // shorthand and any modulus are handled without special cases. Rules from
    // OTHER zones are honoured too when they apply to the reflection, so e.g. a
    // general hkl condition can subsume a zonal restatement of itself.
    const rulesArr = Array.from(confirmedRules);
    const zoneLatticeCache = {};
    const zonePoints = (zone) => {
        if (zoneLatticeCache[zone]) return zoneLatticeCache[zone];
        const pts = [];
        const R = 8;
        for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
            if (h === 0 && k === 0 && l === 0) continue;
            if (zoneApplies(zone, h, k, l)) pts.push([h, k, l]);
        }
        zoneLatticeCache[zone] = pts;
        return pts;
    };
    // Is `target` implied by the conjunction of `others` over target's zone?
    // Vacuous cases (nothing survives the others) are NOT treated as implied,
    // so an over-constrained set can never silently delete a real rule.
    const impliedByConjunction = (others, target) => {
        const pts = zonePoints(target.zone);
        if (pts.length === 0) return false;
        let allowedAny = false;
        for (const [h, k, l] of pts) {
            let allowed = true;
            for (const o of others) {
                if (!zoneApplies(o.zone, h, k, l)) continue;
                if (!satisfiesCondition(h, k, l, o.condition)) { allowed = false; break; }
            }
            if (!allowed) continue;
            allowedAny = true;
            if (!satisfiesCondition(h, k, l, target.condition)) return false;
        }
        return allowedAny;
    };
    const nIndices = (cond) => ['h', 'k', 'l'].filter(v => new RegExp('(^|[^a-z])' + v).test(String(cond))).length;
    let kept = rulesArr.map(parseRuleString).map((p, i) => p ? { ...p, raw: rulesArr[i] } : null).filter(Boolean);
    const unparsed = rulesArr.filter((r, i) => !parseRuleString(r));
    // Offer the most "derived-looking" rules for removal first.
    kept.sort((a, b) => nIndices(b.condition) - nIndices(a.condition) || a.raw.localeCompare(b.raw));
    let removedSomething = true;
    while (removedSomething && kept.length > 1) {
        removedSomething = false;
        for (let i = 0; i < kept.length; i++) {
            const others = kept.filter((_, j) => j !== i);
            if (others.length === 0) break;
            if (impliedByConjunction(others, kept[i])) {
                kept.splice(i, 1);
                removedSomething = true;
                break;
            }
        }
    }
    const survivors = kept.map(k => k.raw).concat(unparsed);
    return (survivors.length > 0 ? survivors : rulesArr).sort();
}


function rankSpaceGroups(indexed_hkls, system, allowedCenterings, spaceGroupData, maxViolations, detectedExtinctions) {
    const candidateGroups = Object.values(spaceGroupData.space_groups).filter(sg => sg.crystal_system === system);
    const validSettings = [];
    
    // Statistical weights for centering order: higher symmetry constrains more reciprocal space
    const centeringWeights = { 'P': 1.0, 'A': 1.5, 'B': 1.5, 'C': 1.5, 'I': 2.0, 'F': 2.0, 'R': 2.0 };
    // Cache of condition selectivity weights, shared across all candidate settings.
    const selectivityCache = {};

    // --- DETECTED-EXTINCTION AGREEMENT ---
    // countViolations() only ever looks at reflections that ARE present: it can
    // punish a group for predicting an absence that did not happen, but it can
    // never reward one for predicting an absence that did. That asymmetry means
    // a group whose condition set is a strict SUBSET of another's can never
    // score worse on violations, so the least-constrained group wins by default.
    // Zektzerite is the worked example: Abma (#64) is exactly Abmm (#67) plus
    // hk0: h=2n, and the data show h00: h=2n - a condition Abmm cannot explain,
    // because A-centering gives k+l=0 for h00, which is even for every h.
    //
    // detectExtinctions() already distilled the observed absences into rules.
    // Here each detected rule is treated as evidence: a setting that ENTAILS it
    // is rewarded, one that leaves it unexplained is penalised. Weighting is by
    // the condition's selectivity times the number of observed reflections in
    // its zone, so a well-supported, highly restrictive condition counts for
    // more than a weakly-sampled one.
    const EXT_WEIGHT = 0.5;
    const detectedList = (Array.isArray(detectedExtinctions) ? detectedExtinctions : [])
        .map(s => { const p = String(s).split(': '); return p.length === 2 ? { zone: p[0].trim(), cond: p[1].trim() } : null; })
        .filter(Boolean);
    // Reflections observed in each detected zone = how much data backs the rule.
    detectedList.forEach(dc => {
        dc.nObs = indexed_hkls.filter(r => zoneApplies(dc.zone, r.h, r.k, r.l)).length;
    });

    // Does this rule set entail `cond` over `zone`? i.e. is every reflection the
    // setting allows in that zone already required to satisfy the condition?
    const entailmentCache = {};
    const entails = (rules, zone, cond) => {
        const key = JSON.stringify(rules) + '|' + zone + '|' + cond;
        if (entailmentCache[key] !== undefined) return entailmentCache[key];
        const R = 8;
        let allowedAny = false, ok = true;
        for (let h = -R; h <= R && ok; h++) for (let k = -R; k <= R && ok; k++) for (let l = -R; l <= R && ok; l++) {
            if (h === 0 && k === 0 && l === 0) continue;
            if (!zoneApplies(zone, h, k, l)) continue;
            let allowed = true;
            for (const { cond: rc } of applicableRules(rules, h, k, l)) {
                if (!satisfiesCondition(h, k, l, rc)) { allowed = false; break; }
            }
            if (!allowed) continue;
            allowedAny = true;
            if (!satisfiesCondition(h, k, l, cond)) ok = false;
        }
        const res = allowedAny && ok;
        entailmentCache[key] = res;
        return res;
    };

    for (const sg of candidateGroups) {
        const sgNumber = sg.number;
        for (const setting of sg.settings) {
            const centering = setting.symbol.charAt(0);
            if (!allowedCenterings.includes(centering) && !(allowedCenterings.includes('P') && !['I','F','A','B','C','R'].includes(centering))) { continue; }
            
            const rules = setting.reflection_conditions || {};
            const violations = countViolations(indexed_hkls, rules);
            
            // Cutoff remains on HARD violations only
            if (violations.hardCount <= maxViolations) {
                let nConfirmTotal = 0;

                // Selectivity weight for a condition, NORMALISED so that an ordinary
                // "=2n" rule (which forbids half its zone) keeps weight 1.0. A
                // reflection satisfying l=4n is stronger evidence than one
                // satisfying l=2n, because l=4n forbids 3/4 of the zone against
                // 1/2, so it scores 1.5. Without this, 004 confirms both equally
                // and P41/P43 tie with P42/P4222 on identical data, leaving the
                // ranking to fall through to the space-group-number tiebreak.
                // Normalising (rather than using the raw forbidden fraction)
                // keeps every existing 2n-based score unchanged, so this only
                // promotes genuinely stronger rules instead of shifting the
                // whole ranking relative to the rule-free baseline of 1.0.
                const selectivity = (zone, cond) => {
                    const key = zone + '|' + cond;
                    if (selectivityCache[key] !== undefined) return selectivityCache[key];
                    const R = 8;
                    let total = 0, allowed = 0;
                    for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
                        if (h === 0 && k === 0 && l === 0) continue;
                        if (!zoneApplies(zone, h, k, l)) continue;
                        total++;
                        if (satisfiesCondition(h, k, l, cond)) allowed++;
                    }
                    // forbidden fraction / 0.5, clamped: a degenerate or unparsed
                    // condition must not silently zero out a confirmation.
                    let w = total > 0 ? (1 - allowed / total) / 0.5 : 1.0;
                    w = Math.min(3.0, Math.max(0.5, w));
                    selectivityCache[key] = w;
                    return w;
                };

                // Harvest positive confirmations across all space group rules
                Object.entries(rules).forEach(([zone, conditions]) => {
                    conditions.forEach(cond => {
                        const w = selectivity(zone, cond);
                        indexed_hkls.forEach(refl => {
                            // General 'hkl' rules apply to all reflections; specific zones apply only to their zone
                            const applies = zoneApplies(zone, refl.h, refl.k, refl.l);

                            if (applies && satisfiesCondition(refl.h, refl.k, refl.l, cond)) {
                                // Full point for strong/reliable reflections; 0.25 for Ka2-suspect or weak tails,
                                // each scaled by how selective the confirmed condition is.
                                if (!refl.ka2Suspect && !refl.lowIntensity) {
                                    nConfirmTotal += 1.0 * w;
                                } else {
                                    nConfirmTotal += 0.25 * w;
                                }
                            }
                        });
                    });
                });

                const wCenter = centeringWeights[centering] || 1.0;

                // Reward/penalise agreement with the observed systematic absences.
                let extBonus = 0, extExplained = 0, extMissed = [];
                for (const dc of detectedList) {
                    if (dc.nObs === 0) continue;
                    const w = selectivity(dc.zone, dc.cond);
                    if (entails(rules, dc.zone, dc.cond)) {
                        extExplained++;
                        extBonus += EXT_WEIGHT * w * dc.nObs;
                    } else {
                        extMissed.push(`${dc.zone}: ${dc.cond}`);
                        extBonus -= EXT_WEIGHT * w * dc.nObs;
                    }
                }

                // FoM_stat: Weighted confirmations minus penalized soft violations,
                // plus agreement with the detected systematic absences.
                // We store this in 'matchScore' to maintain seamless backward compatibility with your UI
                let fomStat = wCenter * (nConfirmTotal - (1.5 * violations.softCount)) + extBonus;
                if (fomStat === 0 && Object.keys(rules).length === 0) fomStat = 1.0; // Baseline for P with no rules

                validSettings.push({
                    number: sgNumber,
                    symbol: setting.symbol,
                    standardSymbol: sg.standard_symbol,
                    pointGroup: sg.point_group,
                    centrosymmetric: sg.centrosymmetric,
                    violations: violations.hardCount,
                    hardViolations: violations.hardCount,
                    softViolations: violations.softCount,
                    violatedReflections: violations.details,
                    violatedReflectionsHard: violations.detailsHard,
                    violatedReflectionsSoft: violations.detailsSoft,
                    extinctionsExplained: extExplained,
                    extinctionsTotal: detectedList.filter(d => d.nObs > 0).length,
                    extinctionsUnexplained: extMissed,
                    extinctionBonus: extBonus,
                    matchScore: fomStat
                });
            }
        }
    }
    
    // Sort: hard violations ASC -> unexplained detected absences ASC
    //       -> FoM_stat (matchScore) DESC -> soft violations ASC -> number DESC
    //
    // The absence pattern is promoted above matchScore deliberately. matchScore
    // counts reflections that ARE present, so it cannot distinguish a setting
    // from a strictly less-constrained one except by the handful of extra
    // confirmations the extra rule happens to collect - and that difference is
    // swamped by the shared, well-populated zones. Systematic ABSENCES are the
    // primary evidence in space-group determination, so a setting that accounts
    // for every detected absence ranks above one that leaves some unexplained,
    // and matchScore then separates settings that explain the same pattern.
    validSettings.sort((a, b) => {
        if (a.hardViolations !== b.hardViolations) return a.hardViolations - b.hardViolations;
        const au = (a.extinctionsUnexplained || []).length, bu = (b.extinctionsUnexplained || []).length;
        if (au !== bu) return au - bu;
        if (Math.abs(a.matchScore - b.matchScore) > 1e-4) return b.matchScore - a.matchScore;
        if (a.softViolations !== b.softViolations) return a.softViolations - b.softViolations;
        return b.number - a.number;
    });
    
    return validSettings;
}

const satisfiesCondition = (h, k, l, condStr) => {
    if (condStr === "h+k, k+l, h+l=2n") { 
        const h_int = Math.round(h), k_int = Math.round(k), l_int = Math.round(l); 
        return ((h_int + k_int) % 2 === 0 && (k_int + l_int) % 2 === 0 && (h_int + l_int) % 2 === 0); 
    }
    
    // Extract shared equality suffix (e.g., "=2n" or "=4n") to handle shorthand like "h, l=2n"
    const rhsMatch = condStr.match(/=\s*(\d+)n/);
    const defaultRhs = rhsMatch ? rhsMatch[0] : "=2n";
    
    const conditions = condStr.split(',').map(s => s.trim());
    for (const condition of conditions) {
        let cleanCond = condition.replace(/\*/g, '');
        
        // If a shorthand part like "h" is missing its modulus, append the shared suffix
        if (!cleanCond.includes('=')) {
            cleanCond += defaultRhs;
        }
        
        const match = cleanCond.match(/([0-9]*[hkl\+\-]+)\s*=\s*(\d+)n/);
        if (!match) { 
            console.warn(`[satisfiesCondition] Could not parse rule part: "${condition}" in rule string "${condStr}"`); 
            continue; 
        }
        const [, expr, modStr] = match; 
        const mod = parseInt(modStr);
        if (isNaN(mod) || mod <= 0) { 
            console.warn(`[satisfiesCondition] Invalid modulus in rule part: "${condition}"`); 
            continue; 
        }
        let value = 0; 
        const terms = expr.match(/[+-]?[0-9]*[hkl]/g) || [];
        for (const term of terms) {
            let sign = 1, coeff = 1, variable = '';
            const coeffMatch = term.match(/^([+-]?)(\d*)([hkl])$/);
            if (coeffMatch) {
                sign = (coeffMatch[1] === '-') ? -1 : 1; 
                coeff = coeffMatch[2] ? parseInt(coeffMatch[2]) : 1; 
                variable = coeffMatch[3];
                const h_int = Math.round(h), k_int = Math.round(k), l_int = Math.round(l);
                if (variable === 'h') value += sign * coeff * h_int;
                else if (variable === 'k') value += sign * coeff * k_int;
                else if (variable === 'l') value += sign * coeff * l_int;
            } else { 
                console.warn(`[satisfiesCondition] Could not parse term "${term}" in expression "${expr}"`); 
            }
        }
        if (Math.round(value) % mod !== 0) { return false; }
    }
    return true;
};

function countViolations(indexed_hkls, rules) {
    let hardCount = 0;
    let softCount = 0;
    const detailsHard = [];
    const detailsSoft = [];

    // Helper: does this single (h,k,l) violate any applicable rule?
    // Mirrors the per-reflection checks below so we can probe alternative hkls.
    const hklViolatesRules = (h, k, l) => {
        for (const { cond } of applicableRules(rules, h, k, l)) {
            if (!satisfiesCondition(h, k, l, cond)) return true;
        }
        return false;
    };

    for (const reflection of indexed_hkls) {
        const { h, k, l, calc_tth } = reflection;
        const isSuspect = !!reflection.ka2Suspect;
        const isLowIntensity = !!reflection.lowIntensity;
        // A reflection is treated as soft if it is either a Ka2-ghost
        // suspect OR if its observed intensity is below the
        // low-intensity cutoff (10% of the strongest peak). The
        // low-intensity case captures the crystallographic intuition
        // that a "forbidden" reflection in the true space group should
        // have intensity zero — a weak observed peak is consistent with
        // residual background, weak overlap from a neighbour, or noise,
        // and is not strong evidence against the systematic-absence
        // rule.
        const isSoftSource = isSuspect || isLowIntensity;
        let isViolation = false;
        let violationDetail = null;
        const softTagFor = (refl) => {
            const tags = [];
            if (refl.ka2Suspect) tags.push('Ka2-suspect');
            if (refl.lowIntensity) tags.push('weak');
            return tags.length > 0 ? ` [${tags.join(', ')}]` : '';
        };
        for (const { zone, cond } of applicableRules(rules, h, k, l)) {
            if (!satisfiesCondition(h, k, l, cond)) {
                isViolation = true;
                const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
                violationDetail = `(${h},${k},${l})${tth_string} violates ${zone}: ${cond}${softTagFor(reflection)}`;
                break;
            }
        }

        // --- AMBIGUOUS-HKL DEMOTION ---
        // If the best-match hkl violates a rule but a different calculated
        // hkl within the same peak's tolerance window satisfies all the
        // rules, the violation is not real evidence against the space
        // group: the peak could equally well be assigned to the allowed
        // alternative. Treat such cases as soft so a single near-tolerance
        // peak can't kill an otherwise excellent space group.
        if (isViolation && hasCompetingAllowedAlt(reflection, alt => !hklViolatesRules(alt.h, alt.k, alt.l))) {
            const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
            violationDetail = `(${h},${k},${l})${tth_string} ambiguous (allowed alt within tol)`;
            softCount++;
            detailsSoft.push(violationDetail);
            continue; // skip the original hard/soft accounting below
        }

        if (isViolation) {
            if (isSoftSource) {
                softCount++;
                detailsSoft.push(violationDetail);
            } else {
                hardCount++;
                detailsHard.push(violationDetail);
            }
        }
    }
    // --- CONSENSUS OVERRIDE ---
    // Each demotion above (Ka2-ghost, weak, overlapped) is a statement that ONE
    // reflection is poor evidence. None of them licenses ignoring a systematic
    // trend. Noise, tails and Ka2 ghosts are not correlated with h+k+l parity,
    // so if many independent reflections all break the SAME rule, the rule is
    // genuinely broken and the demotions are concealing a real result. Promote
    // the whole soft pile to hard once it passes both an absolute and a
    // proportional floor.
    const CONSENSUS_MIN_COUNT = 5;
    const CONSENSUS_MIN_FRACTION = 0.15;
    const nExamined = indexed_hkls.length;
    if (hardCount === 0 && softCount >= CONSENSUS_MIN_COUNT &&
        softCount >= CONSENSUS_MIN_FRACTION * nExamined) {
        hardCount = softCount;
        softCount = 0;
        detailsHard.push(...detailsSoft.splice(0, detailsSoft.length));
    }

    // 'count' and 'details' kept as combined values for any pre-existing
    // caller that doesn't yet read the split fields. detailsHard/detailsSoft
    // are uncapped (used by the PDF report to list every violating hkl);
    // 'details' stays capped as a short legacy summary.
    const count = hardCount + softCount;
    const details = detailsHard.concat(detailsSoft).slice(0, 5);
    return { count, hardCount, softCount, details, detailsHard, detailsSoft };
}

function getReflectionZone(h, k, l) {
    const ah = Math.abs(h), ak = Math.abs(k), al = Math.abs(l);
    if (ak === 0 && al === 0 && ah !== 0) return 'h00'; 
    if (ah === 0 && al === 0 && ak !== 0) return '0k0'; 
    if (ah === 0 && ak === 0 && al !== 0) return '00l';
    if (ah === 0 && ak !== 0 && al !== 0) return '0kl'; 
    if (ak === 0 && ah !== 0 && al !== 0) return 'h0l'; 
    if (al === 0 && ah !== 0 && ak !== 0) return 'hk0';
    if (ah !== 0 && ah === ak && al !== 0) return 'hhl'; 
    if (ak !== 0 && ak === al && ah !== 0) return 'hkk'; 
    if (ah !== 0 && ah === al && ak !== 0) return 'hll';
    return 'hkl';
}

// --- ZONE MEMBERSHIP (inclusive) ---
// getReflectionZone() above returns ONE label, which is right for reporting but
// wrong for testing reflection conditions: a reflection can belong to several
// zone families at once and must satisfy the rules of EVERY family it belongs
// to. 330 is an hk0 reflection (l=0) and simultaneously an hh0/hhl one (h=k);
// 004 is 00l and also hhl (h=k=0). Testing only the single label silently
// misses real extinctions - e.g. in I-42d (122), which has hhl: 2h+l=4n and no
// hk0 rule, reflection 110 is systematically absent (2h+l=2, not 4n) yet was
// classified 'hk0' and never tested against the hhl rule.
//
// Membership is deliberately INCLUSIVE (hkl contains everything, hk0 contains
// h00, hhl contains hh0 and 00l, ...). This is REQUIRED, not merely tidier:
// cctbx_space_groups_all_settings_v4.json stores only the conditions that are
// not already implied by a more general class. Pna21 (33) is stored as just
// {0kl: k+l=2n, h0l: h=2n}; the axial conditions International Tables prints
// for it (00l: l=2n, 0k0: k=2n, h00: h=2n) are omitted because each follows by
// restriction - k+l=2n at k=0 IS l=2n. Without inheritance those extinctions
// were never tested: verified against structure factors from the group
// operations, the old single-zone test missed 001/003/005/010/030/050/100/300/
// 500 for Pna21, and changed verdicts for 264 of the 527 settings in the file.
// A cross-check over all 527 settings found no case where a general-class
// condition contradicts a listed special-class one, so inheritance can only
// restore a missing absence, never invent a false one.
//
// NOTE: 'hkk', 'hll', 'hkh', 'h-hl', 'hh0' and 'hhh' do not occur in the
// bundled database (only hkl, h0l, 0kl, hk0, 00l, hhl, 0k0, h00 are used).
// They are kept for forward compatibility; 'hll' preserves the original
// code's reading (|h|==|l|) which is ambiguous against a positional reading,
// so re-check it if a database that uses that key is ever loaded.
const ZONE_PREDICATES = {
    'hkl':  () => true,
    'hk0':  (h, k, l) => l === 0,
    'h0l':  (h, k, l) => k === 0,
    '0kl':  (h, k, l) => h === 0,
    'h00':  (h, k, l) => k === 0 && l === 0,
    '0k0':  (h, k, l) => h === 0 && l === 0,
    '00l':  (h, k, l) => h === 0 && k === 0,
    'hhl':  (h, k, l) => Math.abs(h) === Math.abs(k),
    'hh0':  (h, k, l) => Math.abs(h) === Math.abs(k) && l === 0,
    'hhh':  (h, k, l) => Math.abs(h) === Math.abs(k) && Math.abs(k) === Math.abs(l),
    'hkk':  (h, k, l) => Math.abs(k) === Math.abs(l),
    'hll':  (h, k, l) => Math.abs(h) === Math.abs(l),
    'hkh':  (h, k, l) => Math.abs(h) === Math.abs(l),
    'h-hl': (h, k, l) => h === -k,
};

// Does a rule labelled `zoneLabel` apply to this reflection?
// Unknown labels fall back to the old exact-match behaviour so an unrecognised
// key in the space-group database can never start matching everything.
function zoneApplies(zoneLabel, h, k, l) {
    const H = Math.round(h), K = Math.round(k), L = Math.round(l);
    const pred = ZONE_PREDICATES[zoneLabel];
    if (pred) return pred(H, K, L);
    return getReflectionZone(H, K, L) === zoneLabel;
}

// All rule conditions that apply to a reflection, gathered across every zone
// the reflection belongs to. Deduplicated by "zone: condition" so the same
// predicate listed under two zones is not counted twice, while keeping the
// zone label for reporting.
function applicableRules(rules, h, k, l) {
    const seen = new Set();
    const out = [];
    for (const [zone, conds] of Object.entries(rules || {})) {
        if (!Array.isArray(conds)) continue;
        if (!zoneApplies(zone, h, k, l)) continue;
        for (const cond of conds) {
            if (seen.has(cond)) continue;
            seen.add(cond);
            out.push({ zone, cond });
        }
    }
    return out;
}

//20 nov, worker
// Check if we are running in a Worker environment to avoid conflicts with the main thread
// ============================================================================
// CRITICAL IMPORT ORDER WARNING:
// When imported into refinement-worker.js via importScripts(), this top-level
// self.onmessage handler is executed FIRST, and then safely CLOBBERED by 
// refinement-worker.js's own self.onmessage handler. 
// DO NOT move this block below importScripts() or reorder script initialization, 
// otherwise this standalone handler will override the batched refinement engine!
// ============================================================================

if (typeof self !== 'undefined' && typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope && !self.IS_REFINEMENT_WORKER) {
    self.onmessage = function(e) {
        
        // --- 1. Get data from main thread ---
        const data = e.data;
        const { systemToSearch, peaks, wavelength, tth_error, impurity_peaks, fom_threshold, max_solutions } = data;
        // --- 2. Set up the "global" state for the functions ---
        const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q } = getSortedPeaks(peaks, wavelength);
        const N_FOR_M20 = Math.min(20, peaks.length);
        const min_m20 = 2.0;
        const d_min = wavelength / (2 * Math.sin(Math.max(...peaks.map(p => p.tth)) * Math.PI / 360));
        const q_max = 1 / (d_min * d_min);
        
        // Live-updating arrays
        const foundSolutions = [];
        const foundSolutionMap = new Map();
        
        // Wrapper for refineAndTestSolution to match the signature expected by logic functions
        const refineAndTestWrapper = (cell) => {
            refineAndTestSolution(
                cell, 
                data, 
                { 
                    q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q,
                    N_FOR_M20, min_m20, q_max, d_min,
                    foundSolutions, foundSolutionMap
                },
                self.postMessage.bind(self) 
            );
        };

        // State object passed to logic functions
        const workerState = {
            q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q,
            N_FOR_M20, min_m20, q_max, d_min,
            foundSolutions, foundSolutionMap,
            refineAndTestSolution: refineAndTestWrapper 
        };

        // --- 3. Run the requested search ---
        self.postMessage({ type: 'progress', payload: 1 });
        
if (systemToSearch === 'cubic') {
            indexCubic(data, workerState, self.postMessage.bind(self));
        } else if (systemToSearch === 'tetragonal') {
            indexTetragonalOrHexagonal(data, workerState, self.postMessage.bind(self), 'tetragonal');
        } else if (systemToSearch === 'hexagonal') {
            indexTetragonalOrHexagonal(data, workerState, self.postMessage.bind(self), 'hexagonal');
        } else if (systemToSearch === 'post_process') {
            // Inject the GPU solutions into the worker so it runs them through findTransformedSolutions
            foundSolutions.push(...(data.gpuSolutions || []));
        }

        self.postMessage({ type: 'progress', payload: 80 });
        
        // Run transformation/symmetry checks on found solutions
        findTransformedSolutions(foundSolutions, data, workerState, self.postMessage.bind(self));
        
        self.postMessage({ type: 'progress', payload: 100 });
        self.postMessage({ type: 'done' });
    };
}