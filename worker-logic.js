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
    if (is90(alpha) && is90(gamma) && is120(beta)) return 'hexagonal';
    if (is90(beta) && is90(gamma) && is120(alpha)) return 'hexagonal';
    if (is90(alpha) && is90(beta) && is120(gamma)) return 'hexagonal';
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
                    for (let l = -l_max; l <= l_max; l++) {
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
                    
                    for (let l = -l_max; l <= l_max; l++) {
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
    const P = 2; const std = standardizeCell(cell);
    switch(std.system) {
        case 'cubic': return `${std.system}_${std.a.toFixed(P)}`;
        case 'tetragonal': case 'hexagonal': return `${std.system}_${std.a.toFixed(P)}_${std.c.toFixed(P)}`;
        case 'orthorhombic': return `${std.system}_${[std.a,std.b,std.c].sort().map(p => p.toFixed(P)).join('_')}`;
       
        case 'monoclinic': 
    const ac = [std.a, std.c].sort((x, y) => x - y).map(p => p.toFixed(P)).join('_');
    return `${std.system}_${ac}_${std.b.toFixed(P)}_${std.beta.toFixed(2)}`;

        case 'triclinic': const vol = getVolumeTriclinic(std).toFixed(2); const angles = [std.alpha, std.beta, std.gamma].sort().map(a => a.toFixed(1)).join('_'); return `${std.system}_${vol}_${angles}`;
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



const generatePermutations = (n) => {
    if (n <= 0) return [[]]; if (n === 1) return [[0]]; const perms = [];
    const subPerms = generatePermutations(n - 1); const item = n - 1;
    for (const p of subPerms) { for (let i = 0; i < n; i++) { const newPerm = p.slice(0, i).concat(item).concat(p.slice(i)); perms.push(newPerm); } }
    return perms;
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
    const N_calc_M = q_calc_sorted.filter(q => q <= q_n * 1.05).length;
    const mN = (N_calc_M > 0 && avg_delta_q > 1e-12) ? (q_n / (2 * avg_delta_q * N_calc_M)) : 0;
    const avg_delta_tth = sum_delta_tth / N_indexed;
    const q_limit_fN = (4 * Math.sin(tth_n_deg * RAD / 2)**2) / (wavelength**2);
    const N_calc_FN = q_calc_sorted.filter(q => q <= q_limit_fN * 1.0001).length;
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

    // --- Single exit function to resolve the promise ---
    const exitFunction = (payload = null) => {
        if (payload) {
            postMessage_func({ type: 'solution', payload: payload });
        } else {
            postMessage_func({}); // Send empty object to resolve
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

    if (refineZero) {
        // --- PATH A: Full Zero-Correction Refinement ---
        // Unified for both GPU (Monoclinic/Triclinic/Ortho) and CPU (Cubic/Tetra/Hex) candidates.
        // We do TWO rounds of pairing+fit with proper LS weighting:
        //   Round 1: pair with z=0 assumption → fit cell+z → get z_estimate
        //   Round 2: pair using q values corrected by z_estimate → re-fit

        const pair_and_fit = (zero_corr_deg) => {
            const indexed_pairs = [];
            const peak_indices = [];
            const used = new Set();
            for (let i = 0; i < n_all; i++) {
                const original_idx = original_indices[i];
                let q_to_match;
                if (Math.abs(zero_corr_deg) > 1e-9) {
                    const corrected_tth_rad = tth_obs_rad[original_idx] - 2 * zero_corr_deg * RAD;
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
            M.forEach((row, i) => {
                const tth_rad = tth_obs_rad[peak_indices[i]];
                row.push((2 / (wavelength ** 2)) * Math.sin(tth_rad));
            });
            const tth_rads_for_rows = peak_indices.map(idx => tth_obs_rad[idx]);
            const ls_weights = ls_weights_for_2theta(tth_rads_for_rows);

            const fit = solveLeastSquares(M, q_vec, ls_weights);
            if (!fit || !fit.solution) return null;
            return { fit, indexed_pairs, peak_indices };
        };

        // Round 1
        let result = pair_and_fit(0);
        // Round 2: re-pair using the zero estimate from round 1
        if (result) {
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
                refined_cell.zero_correction = fitResult_with_zero_final.solution[fitResult_with_zero_final.solution.length - 1] * DEG;
                refined_cell.volume = getVolume(refined_cell);
                
                const q_calc_set_refined = new Set(generateHKL_for_worker(refined_cell, q_max, d_min, wavelength).map(r => r.q));
                const q_calc_sorted_refined = new Float64Array(Array.from(q_calc_set_refined)).sort((a,b)=>a-b);
                
                const peaks_for_merit_20_refined = [];
                for (let i = 0; i < n_20; i++) {
                    const original_peak = peaks_sorted_by_q[i];
                    const corrected_tth_deg = original_peak.tth - refined_cell.zero_correction;
                    const corrected_tth_rad = corrected_tth_deg * RAD;
                    const corrected_q = (4 * Math.sin(corrected_tth_rad / 2)**2) / (wavelength**2);
                    peaks_for_merit_20_refined.push({ ...original_peak, q: corrected_q, tth: corrected_tth_deg });
                }
                
                const { m20: final_m20, fN: final_fN_20 } = calculateFiguresOfMerit(q_calc_sorted_refined, peaks_for_merit_20_refined, impurity_peaks, local_get_q_tolerance, wavelength);
                
                if (final_m20 > min_m20) {
                    const peaks_for_merit_all_refined = [];
                    for (let i = 0; i < n_all; i++) {
                        const original_peak = peaks_sorted_by_q[i]; const corrected_tth_deg = original_peak.tth - refined_cell.zero_correction;
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
            
            // 2. Use the "Label Maker" to find the "true" symmetry of the squeezed cell
            // We use a slightly loose tolerance (0.05) to catch pseudo-symmetries
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
        try {
            const system = sol.system;
            const min_peaks_needed = {cubic: 1, tetragonal: 2, hexagonal: 2, orthorhombic: 3, monoclinic: 4, triclinic: 6}[system];
            if (!min_peaks_needed || data.peaks.length < min_peaks_needed) return;

            // (Removed the duplicate generateHKL_for_worker call on 13th july)

            const first_four_indexed = [];
            for(let i=0; i<4 && i < q_obs.length; i++){
                const q_o = q_obs[i];
                const best_match_idx = binarySearchClosest(theoretical_q_array, q_o); // <--- AND USE IT HERE TOO!
                if(best_match_idx >= 0 && best_match_idx < theoretical_hkls.length && Math.abs(q_o - theoretical_hkls[best_match_idx].q) < local_get_q_tolerance(original_indices[i])){
                    first_four_indexed.push({q_obs: q_o, hkl: [theoretical_hkls[best_match_idx].h, theoretical_hkls[best_match_idx].k, theoretical_hkls[best_match_idx].l]});
                }
            }




            if (first_four_indexed.length < min_peaks_needed) return;
            let closest_pair = {i: -1, j: -1, diff: Infinity};
            for(let i=0; i < first_four_indexed.length; i++){
                for(let j=i+1; j<first_four_indexed.length; j++){
                    const diff = Math.abs(first_four_indexed[i].q_obs - first_four_indexed[j].q_obs);
                    if(diff < closest_pair.diff){ closest_pair = {i, j, diff}; }
                }
            }
            if(closest_pair.i !== -1){
                const swapped_indexed_peaks = JSON.parse(JSON.stringify(first_four_indexed));
                const temp_hkl = swapped_indexed_peaks[closest_pair.i].hkl;
                swapped_indexed_peaks[closest_pair.i].hkl = swapped_indexed_peaks[closest_pair.j].hkl;
                swapped_indexed_peaks[closest_pair.j].hkl = temp_hkl;
                const peaks_for_solve = swapped_indexed_peaks.slice(0, min_peaks_needed);
                const M = peaks_for_solve.map(p => getLSDesignRow(p.hkl, system));
                const q_vec = peaks_for_solve.map(p => p.q_obs);
                const fit = solveLeastSquares(M, q_vec);
                if(fit && fit.solution){ const new_trial_cell = extractCellFromFit(fit.solution, system); if(new_trial_cell){ refineAndTestSolution(new_trial_cell); } }
            }
        } catch (e) { console.warn("Swap-fishing attempt failed:", e); }
        const progress = 80 + ((index + 1) / totalSolutions) * 15;
        postMessage_func({ type: 'progress', payload: progress });
    });
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
function signStrict(x) { return x >= 0 ? 1 : -1; }
const primitiveTransformByCentering = { P: [[1,0,0],[0,1,0],[0,0,1]], A: [[0.5, 0.5, 0], [0.5, -0.5, 0], [0, 0, 1]], B: [[0.5, 0, 0.5], [0.5, 0, -0.5], [0, 1, 0]], C: [[1, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]], I: [[-0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5, -0.5]], F: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], R: [[ 2/3, -1/3, -1/3], [ 1/3,  1/3, -2/3], [ 1/3,  1/3,  1/3]], };






// ==========================================
// 1. THE ADAPTER (Call this one from your main code)
// ==========================================
function reduceToNiggliCell(sol, opts) {
    const a = sol.a, b = sol.b || sol.a, c = sol.c || sol.a;
    const alpha = sol.alpha ?? 90;
    const beta = sol.beta ?? 90;
    const gamma = sol.gamma ?? (sol.system === 'hexagonal' ? 120 : 90);
    const centering = (sol.analysis?.centering) || 'P';
    
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

    const applyTrans = (M) => {
        B = rightMul(B, M);
        T = matMul3(T, M);
        updateMetric();
        changed = true;
    };

    while (changed && iterations < maxIter) {
        changed = false;
        iterations++;

        if (iterations > 50) eps = 1e-4;
        if (iterations > 100) eps = 1e-3;
        if (iterations > 150) break; // Hard bailout

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
        iterations: iterations
    };
}

function generateEquivalentCells(niggliCell, N_ignored, originalSystem = null) {
    const results = { primitiveCells: [], centeredCells: {} };
    if (!niggliCell || typeof niggliCell !== 'object' || !niggliCell.a) { console.error("Invalid Niggli cell provided."); return results; }
    const minAngle = 60.0, maxAngle = 150.0;
    const niggliSystemGuess = getSymmetryForEquivCells(niggliCell.a, niggliCell.b, niggliCell.c, niggliCell.alpha, niggliCell.beta, niggliCell.gamma);
    const niggliVolume = getVolumeForEquivCells({ ...niggliCell, system: niggliSystemGuess });
    results.primitiveCells.push({ ...niggliCell, description: "Niggli Cell", centering: 'P', volume: niggliVolume });
    if (originalSystem) {
        const niggliBasis = cellToBasis(niggliCell.a, niggliCell.b, niggliCell.c, niggliCell.alpha, niggliCell.beta, niggliCell.gamma);
        const primitiveToCenteredTransforms = { 'I': [[0,1,1],[1,0,1],[1,1,0]], 'F': [[-1,1,1],[1,-1,1],[1,1,-1]], 'A': [[1,1,0],[-1,1,0],[0,0,1]], 'B': [[1,0,1],[0,1,0],[-1,0,1]], 'C': [[1,0,0],[0,1,-1],[0,1,1]], 'R': [[1,0,1],[-1,1,1],[0,-1,1]] };
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
function analyzeSystematicAbsences(solution, obs_peaks, spaceGroupData, wavelength, tthError, tthMax) {
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
    const LOW_INTENSITY_FRACTION = 0.05;
    let maxHeight = 0;
    for (const p of obs_peaks) {
        const h = p.height;
        if (typeof h === 'number' && isFinite(h) && h > maxHeight) maxHeight = h;
    }
    const lowIntensityThreshold = maxHeight > 0
        ? maxHeight * LOW_INTENSITY_FRACTION
        : -Infinity; // disable demotion if no heights available

    obs_peaks.forEach(peak => {
        const corrected_tth = peak.tth - zero_correction;
        const bestMatch = all_calc_hkls.reduce((best, hkl) => { const diff = Math.abs(hkl.tth - corrected_tth); return diff < best.minDiff ? { hkl, minDiff: diff } : best; }, { hkl: null, minDiff: Infinity });
        if (bestMatch.hkl && bestMatch.minDiff < indexWindow) {
            // Collect every calculated hkl whose 2theta is within
            // overlapWindow of the BEST-MATCH calculated 2theta (not of
            // the observed 2theta). The overlap is between the candidate
            // reflection and its neighbours in reciprocal space — that
            // is what determines whether intensity from one can leak
            // into the other.
            const altHkls = all_calc_hkls
                .filter(hkl => Math.abs(hkl.tth - bestMatch.hkl.tth) < overlapWindow)
                .map(hkl => ({ h: hkl.h, k: hkl.k, l: hkl.l }));
            const peakHeight = (typeof peak.height === 'number' && isFinite(peak.height)) ? peak.height : null;
            const isLowIntensity = (peakHeight !== null) && (peakHeight < lowIntensityThreshold);
            indexed_hkls.push({
                h: bestMatch.hkl.h, k: bestMatch.hkl.k, l: bestMatch.hkl.l,
                tth: peak.tth, calc_tth: bestMatch.hkl.tth,
                ka2Suspect: !!peak.ka2Suspect,
                altHkls: altHkls,
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
    const hkls_for_analysis = unambiguous_hkls.length > 0 ? unambiguous_hkls : unique_indexed_hkls;
    if (hkls_for_analysis.length < 5) { fallbackResult.centering = 'Unknown (too few unambiguous peaks in range)'; return fallbackResult; }
    const unambiguousSet = new Set(unambiguous_hkls.map(r => `${r.h},${r.k},${r.l}`));
    const ambiguousHkls = new Set(unique_indexed_hkls.filter(r => !unambiguousSet.has(`${r.h},${r.k},${r.l}`)).map(r => `${r.h},${r.k},${r.l}`));

    const anyKa2Suspects = hkls_for_analysis.some(r => r.ka2Suspect);

    const centeringResult = determineCentering(hkls_for_analysis, solution.system);
    const detectedExtinctions = detectExtinctions(hkls_for_analysis, solution.system, spaceGroupData);
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
                if (p.altHkls && p.altHkls.length > 1) {
                    const hasAllowedAlt = p.altHkls.some(alt => {
                        if (alt.h === p.h && alt.k === p.k && alt.l === p.l) return false;
                        return !test.forbidden(Math.round(alt.h), Math.round(alt.k), Math.round(alt.l));
                    });
                    if (hasAllowedAlt) return false;
                }
                return true;
            });
            const softViolatingPeaks = violatingPeaks.filter(p => !hardViolatingPeaks.includes(p));
            violations[key] = violatingPeaks.length; // total
            violationsHard[key] = hardViolatingPeaks.length;
            violationsSoft[key] = softViolatingPeaks.length;
            // Details: store hard violators preferentially, fall back to soft.
            if (violations[key] > 0 && violations[key] <= MAX_DETAILS_TO_STORE) {
                const detailsSource = hardViolatingPeaks.length > 0 ? hardViolatingPeaks : softViolatingPeaks;
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
    
    // Remove Primitive (P) from the reported dictionaries since it cannot have violations
    delete violations['P'];
    delete violationsHard['P'];
    delete violationsSoft['P'];
    delete violationDetails['P'];
    
    
    return {
        plausibleCenterings: finalCenterings,
        description: finalCenterings.map(c => centeringTests[c]?.name || c).join(' or '),
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
        const zoneReflections = indexed_hkls.filter(refl => getReflectionZone(refl.h, refl.k, refl.l) === zone);
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
        const allSatisfy = refSetForRule.every(refl => satisfiesCondition(refl.h, refl.k, refl.l, condition));
        if (allSatisfy) { confirmedRules.add(ruleStr); }
    });
    if (confirmedRules.size === 0) { return ["None detected"]; } else { return Array.from(confirmedRules).sort(); }
}


function rankSpaceGroups(indexed_hkls, system, allowedCenterings, spaceGroupData, maxViolations, detectedExtinctions) {
    const candidateGroups = Object.values(spaceGroupData.space_groups).filter(sg => sg.crystal_system === system);
    const validSettings = [];
    
    // Statistical weights for centering order: higher symmetry constrains more reciprocal space
    const centeringWeights = { 'P': 1.0, 'A': 1.5, 'B': 1.5, 'C': 1.5, 'I': 2.0, 'F': 2.0, 'R': 2.0 };

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
                
                // Harvest positive confirmations across all space group rules
                Object.entries(rules).forEach(([zone, conditions]) => {
                    conditions.forEach(cond => {
                        indexed_hkls.forEach(refl => {
                            const reflZone = getReflectionZone(refl.h, refl.k, refl.l);
                            // General 'hkl' rules apply to all reflections; specific zones apply only to their zone
                            const applies = (zone === 'hkl') || (reflZone === zone);
                            
                            if (applies && satisfiesCondition(refl.h, refl.k, refl.l, cond)) {
                                // Full point for strong/reliable reflections; 0.25 for Ka2-suspect or weak tails
                                if (!refl.ka2Suspect && !refl.lowIntensity) {
                                    nConfirmTotal += 1.0;
                                } else {
                                    nConfirmTotal += 0.25;
                                }
                            }
                        });
                    });
                });

                const wCenter = centeringWeights[centering] || 1.0;
                
                // FoM_stat: Weighted confirmations minus penalized soft violations
                // We store this in 'matchScore' to maintain seamless backward compatibility with your UI
                let fomStat = wCenter * (nConfirmTotal - (1.5 * violations.softCount));
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
                    matchScore: fomStat
                });
            }
        }
    }
    
    // Sort: hard violations ASC -> FoM_stat (matchScore) DESC -> soft violations ASC -> number DESC
    validSettings.sort((a, b) => {
        if (a.hardViolations !== b.hardViolations) return a.hardViolations - b.hardViolations;
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

    // Helper: does this single (h,k,l) violate any zone or general rule?
    // Mirrors the per-reflection checks below so we can probe alternative hkls.
    const hklViolatesRules = (h, k, l) => {
        const zone = getReflectionZone(h, k, l);
        const zoneRules = rules[zone] || [];
        for (const cond of zoneRules) {
            if (!satisfiesCondition(h, k, l, cond)) return true;
        }
        const genRules = rules.hkl || [];
        for (const cond of genRules) {
            if (zone === 'hkl' || !rules[zone] || !rules[zone].some(zc => zc === cond)) {
                if (!satisfiesCondition(h, k, l, cond)) return true;
            }
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
        const zone = getReflectionZone(h, k, l);
        const applicableZoneRules = rules[zone] || [];
        const generalRules = rules.hkl || [];
        const softTagFor = (refl) => {
            const tags = [];
            if (refl.ka2Suspect) tags.push('Ka2-suspect');
            if (refl.lowIntensity) tags.push('weak');
            return tags.length > 0 ? ` [${tags.join(', ')}]` : '';
        };
        for (const cond of applicableZoneRules) {
            if (!satisfiesCondition(h, k, l, cond)) {
                isViolation = true;
                const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
                violationDetail = `(${h},${k},${l})${tth_string} violates ${zone}: ${cond}${softTagFor(reflection)}`;
                break;
            }
        }
        if (!isViolation) {
            for (const cond of generalRules) {
                if (zone === 'hkl' || !rules[zone] || !rules[zone].some(zoneCond => zoneCond === cond)) {
                    if (!satisfiesCondition(h, k, l, cond)) {
                        isViolation = true;
                        const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
                        violationDetail = `(${h},${k},${l})${tth_string} violates hkl: ${cond}${softTagFor(reflection)}`;
                        break;
                    }
                }
            }
        }

        // --- AMBIGUOUS-HKL DEMOTION ---
        // If the best-match hkl violates a rule but a different calculated
        // hkl within the same peak's tolerance window satisfies all the
        // rules, the violation is not real evidence against the space
        // group: the peak could equally well be assigned to the allowed
        // alternative. Treat such cases as soft so a single near-tolerance
        // peak can't kill an otherwise excellent space group.
        if (isViolation && reflection.altHkls && reflection.altHkls.length > 1) {
            const hasAllowedAlt = reflection.altHkls.some(alt => {
                if (alt.h === h && alt.k === k && alt.l === l) return false;
                return !hklViolatesRules(alt.h, alt.k, alt.l);
            });
            if (hasAllowedAlt) {
                const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
                violationDetail = `(${h},${k},${l})${tth_string} ambiguous (allowed alt within tol)`;
                if (isViolation) {
                    softCount++;
                    detailsSoft.push(violationDetail);
                }
                continue; // skip the original hard/soft accounting below
            }
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
        }

        self.postMessage({ type: 'progress', payload: 80 });
        
        // Run transformation/symmetry checks on found solutions
        findTransformedSolutions(foundSolutions, data, workerState, self.postMessage.bind(self));
        
        self.postMessage({ type: 'progress', payload: 100 });
        self.postMessage({ type: 'done' });
    };
}