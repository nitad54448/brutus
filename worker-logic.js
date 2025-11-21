// "worker-logic.js" 
// most of the combs.html functions are moved here, version 15 nov 2025
//20 nov, moved here the former webworker, indexing-logic
const RAD = Math.PI / 180.0;
const DEG = 180.0 / Math.PI;

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

const transpose = (M) => M[0].map((_,i) => M.map(r => r[i]));
const matMul = (A,B) => { const r=A.length, c=B[0].length, k=A[0].length; const C = Array.from({length:r}, () => Array(c).fill(0)); for(let i=0; i<r; i++) for(let j=0; j<c; j++) for(let t=0; t<k; t++) C[i][j] += A[i][t] * B[t][j]; return C; };

const getSymmetry = (a, b, c, alpha, beta, gamma, tol = 0.02) => {
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
    const det = determinant3x3(M); if (Math.abs(det) < 1e-14) return null;
    const invDet = 1.0 / det;
    return [
        [(M[1][1] * M[2][2] - M[1][2] * M[2][1]) * invDet, (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invDet, (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet],
        [(M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invDet, (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet, (M[0][2] * M[1][0] - M[0][0] * M[1][2]) * invDet],
        [(M[1][0] * M[2][1] - M[1][1] * M[2][0]) * invDet, (M[0][1] * M[2][0] - M[0][0] * M[2][1]) * invDet, (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * invDet]
    ];
};

const metricFromReciprocalMetric = (G_star) => invert3x3(G_star);
const cellFromMetric_worker = (G) => {
    if (!G) return null;
    try {
        const a = Math.sqrt(G[0][0]); const b = Math.sqrt(G[1][1]); const c = Math.sqrt(G[2][2]);
        if (a < 1e-6 || b < 1e-6 || c < 1e-6) return null;
        const clamp = v => Math.max(-1, Math.min(1, v));
        const alpha = Math.acos(clamp(G[1][2] / (b * c))) * DEG;
        const beta = Math.acos(clamp(G[0][2] / (a * c))) * DEG;
        const gamma = Math.acos(clamp(G[0][1] / (a * b))) * DEG;
        if (isNaN(a) || isNaN(b) || isNaN(c) || isNaN(alpha) || isNaN(beta) || isNaN(gamma)) return null;
        return { a, b, c, alpha, beta, gamma };
    } catch { return null; }
};

const getVolumeTriclinic = (cell) => {
    const { a, b, c, alpha, beta, gamma } = cell;
    const ca = Math.cos(alpha * RAD), cb = Math.cos(beta * RAD), cg = Math.cos(gamma * RAD);
    const V_sq = a*a*b*b*c*c * (1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
    return (V_sq > 0) ? Math.sqrt(V_sq) : 0;
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
        if (inv_d_sq <= 0 || inv_d_sq > q_max_limit) return;
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
            if (sinBetaSq < 1e-6) return [];
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
            const ca = Math.cos(alpha * RAD), cb = Math.cos(beta * RAD), cg = Math.cos(gamma * RAD);
            const V_sq = a*a*b*b*c*c * (1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
            if (V_sq < 1e-6) return [];
            const Gs_11 = (b*b*c*c * (1 - ca*ca)) / V_sq, Gs_22 = (a*a*c*c * (1 - cb*cb)) / V_sq, Gs_33 = (a*a*b*b * (1 - cg*cg)) / V_sq;
            const Gs_23 = 2 * b*c*a*a * (cb*cg - ca) / V_sq, Gs_13 = 2 * a*c*b*b * (ca*cg - cb) / V_sq, Gs_12 = 2 * a*b*c*c * (ca*cb - cg) / V_sq;
            for (let h = -h_max; h <= h_max; h++) {
                const h_term = h * h * Gs_11;
                for (let k = -k_max; k <= k_max; k++) {
                    const k_term = k * k * Gs_22, hk_term = h_term + k_term + h * k * Gs_12;
                    for (let l = -l_max; l <= l_max; l++) {
                        if (h === 0 && k === 0 && l === 0) continue;
                        if (l < 0) continue; if (l === 0 && k < 0) continue; if (l === 0 && k === 0 && h <= 0) continue;
                        const inv_d_sq = hk_term + (l * l * Gs_33) + (k * l * Gs_23) + (h * l * Gs_13);
                        processReflection(h, k, l, inv_d_sq);
                    }
                }
            } break;
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
const generateHKL = (maxTth, params, system) => generateHKL_for_analysis(params, params.lambda, maxTth);

const generateHKL_for_worker = (cell, q_max, d_min, lambda) => {
    // We need lambda to correctly calculate the max 2-theta from q_max
    const maxTth = Math.asin(Math.sqrt(q_max * 1.05 * lambda * lambda / 4.0)) * 360.0 / Math.PI;
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
        case 'monoclinic': return `${std.system}_${std.volume.toFixed(2)}_${std.beta.toFixed(2)}`;
        case 'triclinic': const vol = getVolumeTriclinic(std).toFixed(2); const angles = [std.alpha, std.beta, std.gamma].sort().map(a => a.toFixed(1)).join('_'); return `${std.system}_${vol}_${angles}`;
    }
};
    
const luDecomposition = (matrix) => {
    const n = matrix.length; const lu = matrix.map(row => row.slice()); const P = Array(n).fill(0).map((_, i) => i);
    for (let k = 0; k < n; k++) {
        let maxVal = 0, k_prime = k;
        for (let i = k; i < n; i++) { if (Math.abs(lu[i][k]) > maxVal) { maxVal = Math.abs(lu[i][k]); k_prime = i; } }
        if (maxVal < 1e-12) return null;
        [P[k], P[k_prime]] = [P[k_prime], P[k]]; [lu[k], lu[k_prime]] = [lu[k_prime], lu[k]];
        for (let i = k + 1; i < n; i++) {
            lu[i][k] /= lu[k][k];
            for (let j = k + 1; j < n; j++) { lu[i][j] -= lu[i][k] * lu[k][j]; }
        }
    }
    return { lu, P };
};

const generatePermutations = (n) => {
    if (n <= 0) return [[]]; if (n === 1) return [[0]]; const perms = [];
    const subPerms = generatePermutations(n - 1); const item = n - 1;
    for (const p of subPerms) { for (let i = 0; i < n; i++) { const newPerm = p.slice(0, i).concat(item).concat(p.slice(i)); perms.push(newPerm); } }
    return perms;
};

const luSolve = (luDecomp, b) => {
    const { lu, P } = luDecomp; const n = lu.length; const x = new Array(n); const y = new Array(n);
    const Pb = P.map(pi => b[pi]);
    for (let i = 0; i < n; i++) { let sum = 0; for (let j = 0; j < i; j++) { sum += lu[i][j] * y[j]; } y[i] = Pb[i] - sum; }
    for (let i = n - 1; i >= 0; i--) { let sum = 0; for (let j = i + 1; j < n; j++) { sum += lu[i][j] * x[j]; } x[i] = (y[i] - sum) / lu[i][i]; if (isNaN(x[i])) return null; }
    return x;
};

const luInvert = (luDecomp) => {
    const n = luDecomp.lu.length; const identity = Array(n).fill(0).map((_, i) => Array(n).fill(0).map((__, j) => (i === j ? 1 : 0)));
    const inverse = Array(n).fill(0).map(() => Array(n).fill(0));
    for (let j = 0; j < n; j++) {
        const b = Array(n).fill(0); b[j] = 1; const invCol = luSolve(luDecomp, b);
        if (!invCol) return null;
        for (let i = 0; i < n; i++) { inverse[i][j] = invCol[i]; }
    }
    return inverse;
};

const get_q_tolerance = (original_peak_index, tth_obs_rad, wavelength, tth_error) => {
    const theta_rad = tth_obs_rad[original_peak_index] / 2.0;
    const d_theta_rad = tth_error * Math.PI / 360;
    const tolerance = ((8 * Math.sin(theta_rad) * Math.cos(theta_rad)) / (wavelength**2)) * d_theta_rad;
    return tolerance + 1e-9; // Add epsilon to prevent division by zero
};

const binarySearchClosest = (arr, target) => {
    let low = 0, high = arr.length - 1;
    if (arr.length === 0 || target <= arr[low]) return 0;
    if (target >= arr[high]) return high;
    while (low <= high) { let mid = Math.floor((low + high) / 2); if (arr[mid] < target) low = mid + 1; else high = mid - 1; }
    return (low >= arr.length) ? high : ((arr[low] - target) < (target - arr[high]) ? low : high);
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
    const MTWM = Array(num_params).fill(0).map(() => Array(num_params).fill(0));
    for (let i = 0; i < num_params; i++) {
        for (let j = 0; j < num_params; j++) {
            let sum = 0; for (let k = 0; k < num_eq; k++) { sum += M[k][i] * w[k] * M[k][j]; } MTWM[i][j] = sum;
        }
    }
    const MTWq = Array(num_params).fill(0);
    for (let i = 0; i < num_params; i++) {
        let sum = 0; for (let k = 0; k < num_eq; k++) { sum += M[k][i] * w[k] * q_vec[k]; } MTWq[i] = sum;
    }
    const luDecomp = luDecomposition(MTWM); if (!luDecomp) return null;
    const x = luSolve(luDecomp, MTWq); if (!x) return null;
    const df = num_eq - num_params; if (df <= 0) return { solution: x, covarianceMatrix: null };
    const q_calc = M.map(row => row.reduce((s, v, j) => s + v * x[j], 0));
    const SSR = q_vec.reduce((sum, q_obs, i) => sum + w[i] * (q_obs - q_calc[i])**2, 0);
    const MTWM_inv = luInvert(luDecomp); if (!MTWM_inv) return { solution: x, covarianceMatrix: null };
    const V = MTWM_inv.map(row => row.map(el => el * (SSR / df)));
    return { solution: x, covarianceMatrix: V };
};

const propagateErrors = (system, fitResult, cell) => {
    if (!fitResult || !fitResult.covarianceMatrix) return {};
    const V = fitResult.covarianceMatrix, errors = {}, num_params = V.length;
    try {
        switch (system) {
            case 'cubic': errors.s_a = 0.5 * cell.a**3 * Math.sqrt(Math.abs(V[0][0])); break;
            case 'tetragonal': case 'hexagonal':
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
            case 'triclinic': errors.s_a = 0; errors.s_b = 0; errors.s_c = 0; errors.s_alpha = 0; errors.s_beta = 0; errors.s_gamma = 0; break;
        }
        const cell_param_count = { cubic: 1, tetragonal: 2, hexagonal: 2, orthorhombic: 3, monoclinic: 4, triclinic: 6 };
        if (num_params > cell_param_count[system]) { errors.s_zero = Math.sqrt(Math.abs(V[num_params - 1][num_params - 1])) * DEG; }
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
    if (axes_to_check.some(p => (isNaN(p) || p < min_lp_check || p > max_lp_check))) {
        // console.log("REFINE: Rejected - bad GPU params");
        return exitFunction(); 
    }
    
    const initial_cell_volume = getVolume(initialParams);
    if (initial_cell_volume > max_volume || initial_cell_volume < 20) {
        // console.log("REFINE: Rejected - bad GPU volume");
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
        
        const isGpuCandidate = (system === 'monoclinic' || system === 'triclinic');

        if (isGpuCandidate) {
            // Re-solve from scratch for GPU candidates ---
            const final_indexed_pairs = []; 
            const final_peak_indices_for_ls = []; 
            const used_reflections = new Set();
            
            for (let i = 0; i < n_all; i++) {
                const q_o = q_obs[i]; 
                const tolerance = local_get_q_tolerance(original_indices[i]);
                let best_match_idx = -1, min_diff = Infinity;

                let low = 0, high = all_possible_reflections.length - 1;
                while (low <= high) { let mid = Math.floor((low + high) / 2); if (all_possible_reflections[mid].q < q_o) low = mid + 1; else high = mid - 1; }
                for (let j = Math.max(0, high - 1); j <= Math.min(all_possible_reflections.length - 1, low + 1); j++) {
                    const diff = Math.abs(all_possible_reflections[j].q - q_o);
                    if (diff < min_diff) { min_diff = diff; best_match_idx = j; }
                }

                if (best_match_idx !== -1 && min_diff < tolerance && !used_reflections.has(best_match_idx)) {
                    const {h, k, l} = all_possible_reflections[best_match_idx];
                    final_indexed_pairs.push({ q_obs: q_o, hkl: [h, k, l] });
                    final_peak_indices_for_ls.push(original_indices[i]);
                    used_reflections.add(best_match_idx);
                }
            }

            // 2. solve for all parameters (cell + zero) from scratch
            if (final_indexed_pairs.length > min_indexed[system]) {
                let M_with_zero_final = final_indexed_pairs.map(p => getLSDesignRow(p.hkl, system));
                const q_vec_final = final_indexed_pairs.map(p => p.q_obs);
                
                // Add the zero-error column to the matrix
                M_with_zero_final.forEach((row, i) => { 
                    const tth_rad = tth_obs_rad[final_peak_indices_for_ls[i]]; 
                    row.push((-4 / (wavelength**2)) * Math.sin(tth_rad)); 
                });
                        
                const fitResult_with_zero_final = solveLeastSquares(M_with_zero_final, q_vec_final, q_vec_final);
    
                // 3. Score and post the *new* cell
                if (fitResult_with_zero_final && fitResult_with_zero_final.solution) {
                    const refined_cell = extractCellFromFit(fitResult_with_zero_final.solution, system);
                    
                    if (refined_cell) {
                        refined_cell.zero_correction = fitResult_with_zero_final.solution[fitResult_with_zero_final.solution.length - 1] * DEG;
                        refined_cell.volume = getVolume(refined_cell);
                        
                        // Generate HKLs from the NEW, UN-WARPED cell
                        const q_calc_set_refined = new Set(generateHKL_for_worker(refined_cell, q_max, d_min, wavelength).map(r => r.q));
                        const q_calc_sorted_refined = new Float64Array(Array.from(q_calc_set_refined)).sort((a,b)=>a-b);
                        
                        // Create corrected observed peaks for M20
                        const peaks_for_merit_20_refined = [];
                        for (let i = 0; i < n_20; i++) {
                            const original_peak = peaks_sorted_by_q[i];
                            const corrected_tth_deg = original_peak.tth - refined_cell.zero_correction;
                            const corrected_tth_rad = corrected_tth_deg * RAD;
                            const corrected_q = (4 * Math.sin(corrected_tth_rad / 2)**2) / (wavelength**2);
                            peaks_for_merit_20_refined.push({ ...original_peak, q: corrected_q, tth: corrected_tth_deg });
                        }
                        
                        // "honest" M20
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
                        } // else { console.log(`REFINE: Rejected (GPU-ZeroRef) - Low M20 = ${final_m20.toFixed(2)}`); }
                    } // else { console.log("REFINE: Rejected (GPU-ZeroRef) - Bad cell from fit"); }
                } // else { console.log("REFINE: Rejected (GPU-ZeroRef) - LSQ solve failed"); }
            } // else { console.log("REFINE: Rejected (GPU-ZeroRef) - Not enough indexed peaks"); }

        } else {
            // --- ORIGINAL PATH: For CPU candidates (Cubic, Tetra, Ortho) ---
            const final_indexed_pairs = []; 
            const final_peak_indices_for_ls = []; 
            const used_reflections = new Set();
            
            for (let i = 0; i < n_all; i++) {
                const q_o = q_obs[i]; 
                const tolerance = local_get_q_tolerance(original_indices[i]);
                let best_match_idx = -1, min_diff = Infinity;

                let low = 0, high = all_possible_reflections.length - 1;
                while (low <= high) { let mid = Math.floor((low + high) / 2); if (all_possible_reflections[mid].q < q_o) low = mid + 1; else high = mid - 1; }
                for (let j = Math.max(0, high - 1); j <= Math.min(all_possible_reflections.length - 1, low + 1); j++) {
                    const diff = Math.abs(all_possible_reflections[j].q - q_o);
                    if (diff < min_diff) { min_diff = diff; best_match_idx = j; }
                }

                if (best_match_idx !== -1 && min_diff < tolerance && !used_reflections.has(best_match_idx)) {
                    const {h, k, l} = all_possible_reflections[best_match_idx];
                    final_indexed_pairs.push({ q_obs: q_o, hkl: [h, k, l] });
                    final_peak_indices_for_ls.push(original_indices[i]);
                    used_reflections.add(best_match_idx);
                }
            }

            if (final_indexed_pairs.length > min_indexed[system]) {
                let M_with_zero_final = final_indexed_pairs.map(p => getLSDesignRow(p.hkl, system));
                const q_vec_final = final_indexed_pairs.map(p => p.q_obs);
                M_with_zero_final.forEach((row, i) => { 
                    const tth_rad = tth_obs_rad[final_peak_indices_for_ls[i]]; 
                    row.push((-4 / (wavelength**2)) * Math.sin(tth_rad)); 
                });
                        
                const fitResult_with_zero_final = solveLeastSquares(M_with_zero_final, q_vec_final, q_vec_final);
    
                if (fitResult_with_zero_final && fitResult_with_zero_final.solution) {
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
            //                console.log(`REFINE: Cell PASSED (ZeroRef) M20 = ${final_m20.toFixed(2)}`, refined_cell);
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
                        } // else { console.log(`REFINE: Rejected (ZeroRef) - Low M20 = ${final_m20.toFixed(2)}`); }
                    } // else { console.log("REFINE: Rejected (ZeroRef) - Bad cell from fit"); }
                } // else { console.log("REFINE: Rejected (ZeroRef) - LSQ solve failed"); }
            } // else { console.log("REFINE: Rejected (ZeroRef) - Not enough indexed peaks"); }
        }
        
    } else {
        // --- PATH B: No Zero-Correction Refinement ---
        let initial_indexed_pairs = []; 
        let initial_peak_indices = [];
        
        for (let i = 0; i < n_all; i++) {
            const q_o = q_obs[i]; 
            const tolerance = local_get_q_tolerance(original_indices[i]);
            let best_match_idx = -1, min_diff = Infinity;
            
            let low = 0, high = all_possible_reflections.length - 1;
            while (low <= high) { let mid = Math.floor((low + high) / 2); if (all_possible_reflections[mid].q < q_o) low = mid + 1; else high = mid - 1; }
            for (let j = Math.max(0, high - 1); j <= Math.min(all_possible_reflections.length - 1, low + 1); j++) {
                const diff = Math.abs(all_possible_reflections[j].q - q_o);
                if (diff < min_diff) { min_diff = diff; best_match_idx = j; }
            }
            
            if (best_match_idx !== -1 && min_diff < tolerance) {
                initial_indexed_pairs.push({ q_obs: q_o, hkl: [all_possible_reflections[best_match_idx].h, all_possible_reflections[best_match_idx].k, all_possible_reflections[best_match_idx].l] });
                initial_peak_indices.push(original_indices[i]);
            }
        }
        
        if (initial_indexed_pairs.length < min_indexed[system]) {
            // console.log("REFINE: Rejected (NoRef) - Not enough indexed peaks");
            return exitFunction();
        }
        
        let M_no_zero = initial_indexed_pairs.map(p => getLSDesignRow(p.hkl, system));
        const q_vec_initial = initial_indexed_pairs.map(p => p.q_obs);
        const fitResult_no_zero = solveLeastSquares(M_no_zero, q_vec_initial, q_vec_initial);

        if (!fitResult_no_zero || !fitResult_no_zero.solution) {
            // console.log("REFINE: Rejected (NoRef) - LSQ solve failed");
            return exitFunction();
        }

        const initial_cell = extractCellFromFit(fitResult_no_zero.solution, system);
        if (!initial_cell) {
            // console.log("REFINE: Rejected (NoRef) - Bad cell from fit");
            return exitFunction();
        }
        
        const refined_axes = [initial_cell.a, initial_cell.b ?? initial_cell.a, initial_cell.c ?? initial_cell.a];
        if (refined_axes.some(p => (isNaN(p) || p < min_lp_check || p > max_lp_check))) {
            // console.log("REFINE: Rejected (NoRef) - Bad refined params");
            return exitFunction();
        }
        initial_cell.volume = getVolume(initial_cell);
        if (initial_cell.volume > max_volume || initial_cell.volume < 20) {
            // console.log("REFINE: Rejected (NoRef) - Bad refined volume");
            return exitFunction();
        }

        const q_calc_set = new Set(generateHKL_for_worker(initial_cell, q_max, d_min, wavelength).map(r => r.q));
        const q_calc_sorted = new Float64Array(Array.from(q_calc_set)).sort((a,b)=>a-b);
        
        const peaks_for_merit_20 = peaks_sorted_by_q.slice(0, n_20);
        const { m20, fN: fN_20 } = calculateFiguresOfMerit(q_calc_sorted, peaks_for_merit_20, impurity_peaks, local_get_q_tolerance, wavelength);

        if (m20 > min_m20) {
        //    console.log(`REFINE: Cell PASSED (NoRef) M20 = ${m20.toFixed(2)}`, initial_cell);
            const { m20: m_all, fN: fN_all } = calculateFiguresOfMerit(q_calc_sorted, peaks_sorted_by_q, impurity_peaks, local_get_q_tolerance, wavelength);
            initial_cell.m20 = m20; initial_cell.fN_20 = fN_20; initial_cell.n_20 = n_20;
            initial_cell.m_all = m_all; initial_cell.fN_all = fN_all; initial_cell.n_all = n_all;
            initial_cell.errors = propagateErrors(system, fitResult_no_zero, initial_cell);
            final_solution_to_post = initial_cell;
        } // else { console.log(`REFINE: Rejected (NoRef) - Low M20 = ${m20.toFixed(2)}`); }
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



// --- TOP-LEVEL INDEXING FUNCTIONS (now accept state) ---
function indexCubic(data, state, postMessage_func) {
    const { peaks } = data; const { q_obs, refineAndTestSolution } = state;
    const h_max = 8;
    const peak_depth = Math.min(peaks.length, 12);
    const hkls = []; for (let h = 1; h <= h_max; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= k; l++) { if (!h && !k && !l) continue; hkls.push([h,k,l]); }
    
    
    const totalTrialsToRun = peak_depth * hkls.length;
    let totalTrialsCompleted = 0;
        
    let trialsBatch = 0; const batchSize = 1000;
    for (let i = 0; i < peak_depth; i++) {
        for (const hkl of hkls) {
            refineAndTestSolution({ a: Math.sqrt((hkl[0]*hkl[0] + hkl[1]*hkl[1] + hkl[2]*hkl[2]) / q_obs[i]), system: 'cubic' });
            trialsBatch++; 
            if (trialsBatch >= batchSize) { 
                postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch }); 
                totalTrialsCompleted += trialsBatch;
                const progress = (totalTrialsCompleted / totalTrialsToRun) * 80; // 80% reserved
                postMessage_func({ type: 'progress', payload: progress });
                trialsBatch = 0; 
            }
        }
        
    }
    if (trialsBatch > 0) postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch });
}

function indexTetragonalOrHexagonal(data, state, postMessage_func, system) {
    const { peaks } = data; const { q_obs, refineAndTestSolution } = state;
    const max_hkl = 5, i_depth = Math.min(12, peaks.length), j_depth = Math.min(12, peaks.length);
    const hkls = []; for (let h = 0; h <= max_hkl; h++) for (let k = 0; k <= h; k++) for (let l = 0; l <= max_hkl; l++) { if (!h && !k && !l) continue; hkls.push([h,k,l]); }

    
    let totalPeakCombos = 0;
    for (let i = 0; i < i_depth; i++) {
        for (let j = i + 1; j < j_depth; j++) {
            totalPeakCombos++;
        }
    }
    const totalTrialsToRun = totalPeakCombos * hkls.length * hkls.length;
    let totalTrialsCompleted = 0;
    

    let trialsBatch = 0; const batchSize = 5000;
    for (let i = 0; i < i_depth; i++) {
        for (let j = i + 1; j < j_depth; j++) {
            for (const hkl1 of hkls) {
                const l1 = hkl1[2]; const S1 = system === 'tetragonal' ? hkl1[0] * hkl1[0] + hkl1[1] * hkl1[1] : hkl1[0] * hkl1[0] + hkl1[0] * hkl1[1] + hkl1[1] * hkl1[1];
                for (const hkl2 of hkls) {
                    const l2 = hkl2[2]; const S2 = system === 'tetragonal' ? hkl2[0] * hkl2[0] + hkl2[1] * hkl2[1] : hkl2[0] * hkl2[0] + hkl2[0] * hkl2[1] + hkl2[1] * hkl2[1];
                    
                    trialsBatch++; 
                    if (trialsBatch >= batchSize) { 
                        postMessage_func({ type: 'trials_completed_batch', payload: trialsBatch }); 
                        totalTrialsCompleted += trialsBatch;
                        const progress = (totalTrialsCompleted / totalTrialsToRun) * 80;
                        postMessage_func({ type: 'progress', payload: progress });
                        trialsBatch = 0; 
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
            const idealSymmetry = getSymmetry(nCell.a, nCell.b, nCell.c, nCell.alpha, nCell.beta, nCell.gamma, 0.05);

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
                        // Figure out which axes are 'a' and which is 'c'
                        const axes = [nCell.a, nCell.b, nCell.c];
                        const tol = 0.05;
                        if (Math.abs(axes[0] - axes[1]) < tol) { // a == b
                            newTrialCell.a = (axes[0] + axes[1]) / 2.0;
                            newTrialCell.c = axes[2];
                        } else if (Math.abs(axes[0] - axes[2]) < tol) { // a == c
                            newTrialCell.a = (axes[0] + axes[2]) / 2.0;
                            newTrialCell.c = axes[1];
                        } else { // b == c
                            newTrialCell.a = (axes[1] + axes[2]) / 2.0;
                            newTrialCell.c = axes[0];
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
const theoretical_hkls_for_tf = generateHKL_for_worker(sol, q_max, d_min, wavelength);

        const indexedPeaks = [];
        for(let i=0; i<N_FOR_M20; i++){
             const q_o = q_obs[i];
             const best_match_idx = binarySearchClosest(theoretical_hkls_for_tf.map(h => h.q), q_o);
             if (best_match_idx >= 0 && best_match_idx < theoretical_hkls_for_tf.length && Math.abs(q_o - theoretical_hkls_for_tf[best_match_idx].q) < local_get_q_tolerance(original_indices[i])){
                 indexedPeaks.push(theoretical_hkls_for_tf[best_match_idx]);
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
            const min_peaks_needed = {cubic: 1, tetragonal: 2, hexagonal: 2, orthorhombic: 3, monoclinic: 4}[system];
            if (!min_peaks_needed || data.peaks.length < min_peaks_needed) return;

            // add send wave
const theoretical_hkls = generateHKL_for_worker(sol, q_max, d_min, wavelength);

            const first_four_indexed = [];
            for(let i=0; i<4 && i < q_obs.length; i++){
                const q_o = q_obs[i];
                const best_match_idx = binarySearchClosest(theoretical_hkls.map(h => h.q), q_o);
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
const getSymmetryForEquivCells = (a, b, c, alpha, beta, gamma, tol = 0.02) => getSymmetry(a,b,c,alpha,beta,gamma,tol);
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

function niggliReduceFromCell(cell, opts = {}) {
  const { a, b, c, alpha, beta, gamma, centering = 'P' } = cell;
  const epsUser = opts.eps; const maxIter = opts.maxIterations || 1000;
  let B = cellToBasis(a, b, c, alpha, beta, gamma);
  let centeringKey = (centering || 'P').toUpperCase();
  const match = centeringKey.match(/\(([A-Z])\)/);
  if (match) { centeringKey = match[1]; } else if (centeringKey.length > 1) { centeringKey = centeringKey.charAt(0); }
  const Cp = primitiveTransformByCentering[centeringKey];
  if (!Cp) { console.warn(`Unknown centering type: ${centering}. Defaulting to 'P'.`); B = cellToBasis(a, b, c, alpha, beta, gamma); }
  if (Cp && centeringKey !== 'P') { B = rightMul(B, Cp); }
  let T = (Cp && centeringKey !== 'P') ? Cp.map(row => row.slice()) : I3();
  let { A, B: BB, C, xi, eta, zeta } = basisToMetric(B);
  const scale = Math.max(A, BB, C) || 1; const eps = typeof epsUser === 'number' ? epsUser : 1e-8 * scale;
  function updateState() { const s = basisToMetric(B); A = s.A; BB = s.B; C = s.C; xi = s.xi; eta = s.eta; zeta = s.zeta; }
  function angleSigns() { let l = 0, m = 0, n = 0; if (xi < -eps) l = -1; else if (xi > eps) l = 1; if (eta < -eps) m = -1; else if (eta > eps) m = 1; if (zeta < -eps) n = -1; else if (zeta > eps) n = 1; return { l, m, n, lmn: l * m * n }; }
  let iter;
  for (iter = 0; iter < maxIter; iter++) {
    let changed = false;
    if ((A > BB + eps) || (Math.abs(A - BB) <= eps && Math.abs(xi) > Math.abs(eta) + eps)) { const C1 = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]; B = rightMul(B, C1); T = matMul3(T, C1); updateState(); changed = true; }
    if (!changed && ((BB > C + eps) || (Math.abs(BB - C) <= eps && Math.abs(eta) > Math.abs(zeta) + eps))) { const C2 = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]; B = rightMul(B, C2); T = matMul3(T, C2); updateState(); changed = true; }
    if (changed) continue;
    const { l, m, n, lmn } = angleSigns();
    if (lmn === 1) { const i = (l === -1) ? -1 : 1; const j = (m === -1) ? -1 : 1; const k = (n === -1) ? -1 : 1; const C3 = [[i, 0, 0], [0, j, 0], [0, 0, k]]; B = rightMul(B, C3); T = matMul3(T, C3); updateState(); continue; }
    if (lmn !== 1) { let i = 1, j = 1, k = 1; if (l === 1) i = -1; if (m === 1) j = -1; if (n === 1) k = -1; const C4 = [[i, 0, 0], [0, j, 0], [0, 0, k]]; B = rightMul(B, C4); T = matMul3(T, C4); updateState(); }
    if ((Math.abs(xi) > BB + eps) || (Math.abs(BB - xi) <= eps && (2 * eta < zeta - eps)) || (Math.abs(BB + xi) <= eps && (zeta < -eps))) { const C5 = [[1, 0, 0], [0, 1, -signStrict(xi)], [0, 0, 1]]; B = rightMul(B, C5); T = matMul3(T, C5); updateState(); continue; }
    if ((Math.abs(eta) > A + eps) || (Math.abs(A - eta) <= eps && (2 * xi < zeta - eps)) || (Math.abs(A + eta) <= eps && (zeta < -eps))) { const C6 = [[1, 0, -signStrict(eta)], [0, 1, 0], [0, 0, 1]]; B = rightMul(B, C6); T = matMul3(T, C6); updateState(); continue; }
    if ((Math.abs(zeta) > A + eps) || (Math.abs(A - zeta) <= eps && (2 * xi < eta - eps)) || (Math.abs(A + zeta) <= eps && (eta < -eps))) { const C7 = [[1, -signStrict(zeta), 0], [0, 1, 0], [0, 0, 1]]; B = rightMul(B, C7); T = matMul3(T, C7); updateState(); continue; }
    if ((xi + eta + zeta + A + BB < -eps) || (Math.abs(xi + eta + zeta + A + BB) <= eps && (2 * (A + eta) + zeta > eps))) { const C8 = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]; B = rightMul(B, C8); T = matMul3(T, C8); updateState(); continue; }
    break;
  }
  const reduced = basisToCell(B); const finalMetric = basisToMetric(B);
  return {
    cell: { a: reduced.a, b: reduced.b, c: reduced.c, alpha: reduced.alpha, beta: reduced.beta, gamma: reduced.gamma },
    transform: T, basis: B, metric: finalMetric.G,
    g6: { A: finalMetric.A, B: finalMetric.B, C: finalMetric.C, xi: finalMetric.xi, eta: finalMetric.eta, zeta: finalMetric.zeta },
    iterations: iter, converged: iter < maxIter
  };
}
function reduceToNiggliCell(sol, opts) {
  const a = sol.a, b = sol.b || sol.a, c = sol.c || sol.a;
  const alpha = sol.alpha ?? 90; const beta = sol.beta ?? 90; const gamma = sol.gamma ?? (sol.system === 'hexagonal' ? 120 : 90);
  const centering = (sol.analysis?.centering) || 'P';
  return niggliReduceFromCell({ a, b, c, alpha, beta, gamma, centering }, opts);
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

// --- SPACE GROUP ANALYSIS FUNCTIONS ---
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
    
    const indexed_hkls = []; const zero_correction = solution.zero_correction || 0;
    obs_peaks.forEach(peak => {
        const corrected_tth = peak.tth - zero_correction;
        const bestMatch = all_calc_hkls.reduce((best, hkl) => { const diff = Math.abs(hkl.tth - corrected_tth); return diff < best.minDiff ? { hkl, minDiff: diff } : best; }, { hkl: null, minDiff: Infinity });
        if (bestMatch.hkl && bestMatch.minDiff < (tthError * 1.5)) { indexed_hkls.push({ h: bestMatch.hkl.h, k: bestMatch.hkl.k, l: bestMatch.hkl.l, tth: peak.tth, calc_tth: bestMatch.hkl.tth }); }
    });
    const unique_indexed_hkls = Array.from(new Map(indexed_hkls.map(r => [`${r.h},${r.k},${r.l}`, r])).values());
    const unambiguous_hkls = unique_indexed_hkls.filter(refl => {
        const nearbyCount = all_calc_hkls.filter(calc => { if (calc.h === refl.h && calc.k === refl.k && calc.l === refl.l) return false; return Math.abs(calc.tth - refl.calc_tth) < tthError; }).length;
        return nearbyCount === 0;
    });
    const hkls_for_analysis = unambiguous_hkls.length > 0 ? unambiguous_hkls : unique_indexed_hkls;
    if (hkls_for_analysis.length < 5) { fallbackResult.centering = 'Unknown (too few unambiguous peaks in range)'; return fallbackResult; }
    const unambiguousSet = new Set(unambiguous_hkls.map(r => `${r.h},${r.k},${r.l}`));
    const ambiguousHkls = new Set(unique_indexed_hkls.filter(r => !unambiguousSet.has(`${r.h},${r.k},${r.l}`)).map(r => `${r.h},${r.k},${r.l}`));
    const centeringResult = determineCentering(hkls_for_analysis, solution.system);
    const detectedExtinctions = detectExtinctions(hkls_for_analysis, solution.system, spaceGroupData);
    const rankedSpaceGroups = rankSpaceGroups(hkls_for_analysis, solution.system, centeringResult.plausibleCenterings, spaceGroupData, MAX_VIOLATIONS, detectedExtinctions);
    return { centering: centeringResult.description, rankedSpaceGroups: rankedSpaceGroups.slice(0, 20), detectedExtinctions: detectedExtinctions, centeringViolations: centeringResult.violations, centeringViolationDetails: centeringResult.violationDetails, ambiguousHkls: ambiguousHkls, hklList: all_calc_hkls };
}
function determineCentering(indexed_hkls, system) {
    const centeringTests = { 'P': { name: 'Primitive (P)', forbidden: (h, k, l) => false }, 'I': { name: 'Body-centered (I)', forbidden: (h, k, l) => (h + k + l) % 2 !== 0 }, 'F': { name: 'Face-centered (F)', forbidden: (h, k, l) => !( (h%2===0 && k%2===0 && l%2===0) || (h%2!==0 && k%2!==0 && l%2!==0) ) }, 'A': { name: 'A-centered (A)', forbidden: (h, k, l) => (k + l) % 2 !== 0 }, 'B': { name: 'B-centered (B)', forbidden: (h, k, l) => (h + l) % 2 !== 0 }, 'C': { name: 'C-centered (C)', forbidden: (h, k, l) => (h + k) % 2 !== 0 } };
    const validBravaisCenterings = { 'cubic': ['P', 'I', 'F'], 'tetragonal': ['P', 'I'], 'orthorhombic': ['P', 'I', 'F', 'A', 'B', 'C'], 'hexagonal': ['P'], 'monoclinic': ['P', 'C'], 'triclinic': ['P'] };
    const violations = {}; const violationDetails = {}; const MAX_DETAILS_TO_STORE = 2;
    for (const [key, test] of Object.entries(centeringTests)) {
        const allowedForSystem = validBravaisCenterings[system] || ['P'];
        if (allowedForSystem.includes(key)) {
            const violatingPeaks = indexed_hkls.filter(({h, k, l}) => test.forbidden(Math.round(h), Math.round(k), Math.round(l)));
            violations[key] = violatingPeaks.length;
            if (violations[key] > 0 && violations[key] <= MAX_DETAILS_TO_STORE) { violationDetails[key] = violatingPeaks.slice(0, MAX_DETAILS_TO_STORE).map(p => ({ h: p.h, k: p.k, l: p.l, tth: p.tth })); }
        }
    }
    const minViolations = Object.keys(violations).length > 0 ? Math.min(...Object.values(violations)) : 0;
    let plausible = Object.keys(violations).filter(key => violations[key] === minViolations && (validBravaisCenterings[system] || ['P']).includes(key));
    if (plausible.length === 0 && violations['P'] === minViolations) { plausible = ['P']; } else if (plausible.length === 0) { plausible = ['P']; }
    let finalCenterings;
    if (plausible.includes('F')) finalCenterings = ['F'];
    else if (plausible.includes('I')) finalCenterings = ['I'];
    else { const specialCenterings = plausible.filter(c => ['A', 'B', 'C'].includes(c)); finalCenterings = specialCenterings.length > 0 ? specialCenterings : ['P']; if (plausible.includes('P') && !finalCenterings.includes('P') && specialCenterings.length > 0) { finalCenterings.push('P'); } if (finalCenterings.length === 0) finalCenterings = ['P']; }
    finalCenterings = finalCenterings.filter(c => (validBravaisCenterings[system] || ['P']).includes(c));
    if (finalCenterings.length === 0) finalCenterings = ['P'];
    return { plausibleCenterings: finalCenterings, description: finalCenterings.map(c => centeringTests[c]?.name || c).join(' or '), violations: violations, violationDetails: violationDetails, minViolations: minViolations };
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
        const allSatisfy = zoneReflections.every(refl => satisfiesCondition(refl.h, refl.k, refl.l, condition));
        if (allSatisfy) { confirmedRules.add(ruleStr); }
    });
    if (confirmedRules.size === 0) { return ["None detected"]; } else { return Array.from(confirmedRules).sort(); }
}
function rankSpaceGroups(indexed_hkls, system, allowedCenterings, spaceGroupData, maxViolations, detectedExtinctions) {
    const candidateGroups = Object.values(spaceGroupData.space_groups).filter(sg => sg.crystal_system === system);
    const validSettings = [];
    const detectedExtinctionsSet = new Set(detectedExtinctions.filter(e => e !== "None detected"));
    for (const sg of candidateGroups) {
        const sgNumber = sg.number;
        for (const setting of sg.settings) {
            const centering = setting.symbol.charAt(0);
            if (!allowedCenterings.includes(centering) && !(allowedCenterings.includes('P') && !['I','F','A','B','C','R'].includes(centering))) { continue; }
            const rules = setting.reflection_conditions || {};
            const violations = countViolations(indexed_hkls, rules);
            if (violations.count <= maxViolations) {
                let matchScore = 0; const sgRulesSet = new Set();
                Object.entries(rules).forEach(([zone, conditions]) => { conditions.forEach(cond => { sgRulesSet.add(`${zone}: ${cond}`); }); });
                detectedExtinctionsSet.forEach(detectedRule => { if (sgRulesSet.has(detectedRule)) { matchScore += 10; } });
                sgRulesSet.forEach(sgRule => { if (!detectedExtinctionsSet.has(sgRule)) { matchScore -= 1; } });
                if (detectedExtinctionsSet.size === 0 && sgRulesSet.size === 0) { matchScore = 1; }
                validSettings.push({ number: sgNumber, symbol: setting.symbol, standardSymbol: sg.standard_symbol, pointGroup: sg.point_group, centrosymmetric: sg.centrosymmetric, violations: violations.count, violatedReflections: violations.details, matchScore: matchScore });
            }
        }
    }
    validSettings.sort((a, b) => { if (a.violations !== b.violations) { return a.violations - b.violations; } if (a.matchScore !== b.matchScore) { return b.matchScore - a.matchScore; } return b.number - a.number; });
    return validSettings;
}
const satisfiesCondition = (h, k, l, condStr) => {
    if (condStr === "h+k, k+l, h+l=2n") { const h_int = Math.round(h), k_int = Math.round(k), l_int = Math.round(l); return ((h_int + k_int) % 2 === 0 && (k_int + l_int) % 2 === 0 && (h_int + l_int) % 2 === 0); }
    const conditions = condStr.split(',').map(s => s.trim());
    for (const condition of conditions) {
        const match = condition.match(/([0-9]*[hkl\+\-]+)\s*=\s*(\d+)n/);
        if (!match) { console.warn(`[satisfiesCondition] Could not parse rule part: "${condition}" in rule string "${condStr}"`); continue; }
        const [, expr, modStr] = match; const mod = parseInt(modStr);
        if (isNaN(mod) || mod <= 0) { console.warn(`[satisfiesCondition] Invalid modulus in rule part: "${condition}"`); continue; }
        let value = 0; const terms = expr.match(/[+-]?[0-9]*[hkl]/g) || [];
        for (const term of terms) {
            let sign = 1, coeff = 1, variable = '';
            const coeffMatch = term.match(/^([+-]?)(\d*)([hkl])$/);
            if (coeffMatch) {
                sign = (coeffMatch[1] === '-') ? -1 : 1; coeff = coeffMatch[2] ? parseInt(coeffMatch[2]) : 1; variable = coeffMatch[3];
                const h_int = Math.round(h), k_int = Math.round(k), l_int = Math.round(l);
                if (variable === 'h') value += sign * coeff * h_int;
                else if (variable === 'k') value += sign * coeff * k_int;
                else if (variable === 'l') value += sign * coeff * l_int;
            } else { console.warn(`[satisfiesCondition] Could not parse term "${term}" in expression "${expr}"`); }
        }
        if (Math.round(value) % mod !== 0) { return false; }
    }
    return true;
};
function countViolations(indexed_hkls, rules) {
    let count = 0; const details = [];
    for (const reflection of indexed_hkls) {
        const { h, k, l, calc_tth } = reflection; let isViolation = false;
        const zone = getReflectionZone(h, k, l);
        const applicableZoneRules = rules[zone] || [];
        const generalRules = rules.hkl || [];
        for (const cond of applicableZoneRules) {
            if (!satisfiesCondition(h, k, l, cond)) {
                isViolation = true; const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
                details.push(`(${h},${k},${l})${tth_string} violates ${zone}: ${cond}`);
                break;
            }
        }
        if (!isViolation) {
            for (const cond of generalRules) {
                 if (zone === 'hkl' || !rules[zone] || !rules[zone].some(zoneCond => zoneCond === cond)) {
                      if (!satisfiesCondition(h, k, l, cond)) {
                           isViolation = true; const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
                           details.push(`(${h},${k},${l})${tth_string} violates hkl: ${cond}`);
                           break;
                      }
                 }
            }
        }
        if (isViolation) { count++; }
    }
    return { count, details: details.slice(0, 5) };
}
function getReflectionZone(h, k, l) {
    if (k === 0 && l === 0 && h !== 0) return 'h00'; if (h === 0 && l === 0 && k !== 0) return '0k0'; if (h === 0 && k === 0 && l !== 0) return '00l';
    if (h === 0 && k !== 0 && l !== 0) return '0kl'; if (k === 0 && h !== 0 && l !== 0) return 'h0l'; if (l === 0 && h !== 0 && k !== 0) return 'hk0';
    if (h !== 0 && h === k && l !== 0 && h !== l) return 'hhl'; if (k !== 0 && k === l && h !== 0 && h !== k) return 'hkk'; if (h !== 0 && h === l && k !== 0 && k !== h) return 'hll';
    return 'hkl';
}

//20 nov, worker
// Check if we are running in a Worker environment to avoid conflicts with the main thread
if (typeof self !== 'undefined' && typeof  WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {

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