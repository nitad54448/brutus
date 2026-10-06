// js/crystallography/metric.js
// Reciprocal-space conventions, metric tensors, cell symmetry and small linear algebra.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// =====================================================================
//  RECIPROCAL-SPACE CONVENTION  --  READ BEFORE TOUCHING ANYTHING NAMED q
// =====================================================================
//
//  Two different quantities are legitimately called "Q" in the literature,
//  and BOTH are used in this project. They differ by more than a constant:
//  one is the square of the other times 4*pi^2. Confusing them does not
//  throw -- it silently produces plausible-looking wrong cells.
//
//    qsq   = 1/d^2 = 4 sin^2(theta) / lambda^2      [ A^-2 ]
//            The indexing quantity. Linear in the quadratic form, which is
//            the whole reason it is used:
//                qsq = h^2 A + k^2 B + l^2 C + hk D + hl E + kl F
//            This is what the least-squares fit solves for, and it is what
//            EVERY q-named identifier in js/crystallography/ means: q_obs, q_max,
//            q_calc_sorted, q_to_match, peaks_sorted_by_q, get_q_tolerance,
//            and the .q field of a peak object.
//
//    dstar = 1/d = 2 sin(theta) / lambda            [ A^-1 ]
//            The square root of the above. Not currently used by name.
//
//    Qscat = 4*pi*sin(theta)/lambda = 2*pi/d        [ A^-1 ]
//            The scattering vector. Used in the UI code ONLY: the plot's Q
//            axis mode, the Kalpha2 stripping (Zhang dual-wavelength
//            differencing), and the peak-finder's Radius / Smoothing
//            sliders. Never appears in the indexing math.
//
//  Relationship:  Qscat = 2*pi*dstar = 2*pi*sqrt(qsq)
//
//  The names were NOT mass-renamed to qsq_obs etc. because q_obs, q_max,
//  peaks_sorted_by_q and q_calc_sorted cross the postMessage boundary
//  between the UI code, these scripts and the two workers as plain
//  object keys. A rename missed in one of the three fails silently at
//  runtime rather than at parse time, so the risk outweighs the tidiness.
//  Use the named helpers below in new code instead of rewriting the
//  formula inline.
//
const RAD = Math.PI / 180.0;
const DEG = 180.0 / Math.PI;
// 1/d^2 in A^-2 from 2-theta in DEGREES. The indexing quantity.
const dstarSqFromTthDeg = (tth_deg, lambda) => {
    const s = Math.sin(tth_deg * RAD / 2);
    return (4 * s * s) / (lambda * lambda);
};
// 1/d^2 in A^-2 from theta in RADIANS (note: theta, not 2-theta).
const dstarSqFromThetaRad = (theta_rad, lambda) => {
    const s = Math.sin(theta_rad);
    return (4 * s * s) / (lambda * lambda);
};
// Scattering vector 4*pi*sin(theta)/lambda in A^-1 from 2-theta in DEGREES.
// Present here only so both conventions are defined in one place; the
// indexing math must never call this.
const scatteringQFromTthDeg = (tth_deg, lambda) =>
    4 * Math.PI * Math.sin(tth_deg * RAD / 2) / lambda;
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
