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


// --- SPACE-GROUP LINE FILTER -------------------------------------------------
// A predicate (h,k,l) => boolean consulted by generateHKL_for_analysis, and so
// by EVERY consumer of it: generateHKL_for_worker, generateQArray_for_worker,
// mcEvaluateCell, mcLeastSquaresPolish, mcErrorsAtFixedCell and the figures of
// merit. Setting it once makes the whole Monte-Carlo pipeline extinction-aware
// in one place instead of threading an extra argument through eight functions.
//
// It is null by default, so nothing in the program changes unless a caller
// deliberately turns it on. It is a module-level global rather than a parameter
// because JS is single-threaded here: the scan is synchronous per candidate, so
// there is no interleaving. It MUST always be cleared in a finally block --
// leaving it set would silently constrain the ordinary indexing run.
let _SG_FILTER = null;
function setSpaceGroupFilter(fn) { _SG_FILTER = (typeof fn === 'function') ? fn : null; }
function getSpaceGroupFilter() { return _SG_FILTER; }

// HKL generator... 
function generateHKL_for_analysis(params, lambda, maxTth, mode = 'full') {
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

        // Space-group reflection conditions, when a scan has installed them.
        // Placed before the mode split so the 'full' and 'q_only' paths emit the
        // same set -- they are required to agree on N_calc (see the q_only note
        // below), and a filter applied to only one of them would break that.
        if (_SG_FILTER !== null && !_SG_FILTER(h, k, l)) return;

        // The physical diffractability check must gate BOTH modes, otherwise
        // the q_only fast path returns reflections the full path rejects and
        // the M20 figures of merit silently disagree between the two.
        const sinThetaSq = (lambda * lambda / 4) * inv_d_sq;
        if (sinThetaSq > 1) return;

        if (mode === 'q_only') {
            reflections.push(inv_d_sq);
            return;
        }

        const tth = 2 * Math.asin(Math.sqrt(sinThetaSq)) * DEG;
        reflections.push({ tth, h, k, l, d: 1 / Math.sqrt(inv_d_sq), q: inv_d_sq });
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
                    const S33_safe = Math.max(S33, 1e-14); // Prevent division by zero from MC drift
                    let l_lo = Math.max(-l_max, Math.ceil((-B_l - sq_t) / (2 * S33_safe)) - 1);
                    const l_hi = Math.min(l_max, Math.floor((-B_l + sq_t) / (2 * S33_safe)) + 1);
                    
                    
                    
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

    // Fast path: the MC/annealing loop only ever needs a sorted unique q list,
    // so skip the object allocation and the tth dedupe entirely.
    // The comparator is explicit on purpose -- a bare .sort() only happens to
    // work because this is a Float64Array, and would silently become
    // lexicographic if this were ever changed to a plain Array.
    if (mode === 'q_only') {
        // Sort first, then collapse near-duplicates. A bare Set is not enough:
        // symmetry-equivalent reflections reach the same q by different orders
        // of floating-point addition and differ in the last bit or two, so a
        // Set keeps both while the full path's tth tolerance merges them.
        // Deduping on a relative tolerance keeps the two paths in agreement.
        reflections.sort((a, b) => a - b);
        // The relative-1e-9 window this used to apply is far TIGHTER than the
        // full path's 1e-4 deg tth window, so the two paths disagreed on the
        // line count (measured: 307 vs 305 monoclinic, 418 vs 417 triclinic)
        // and therefore on N_calc inside calculateFiguresOfMerit -- i.e. the
        // Monte-Carlo score and the headline M20 were computed against
        // different line lists. Reproduce the tth window exactly instead:
        //   dq/d(2th) = 2 sin(2th)/lambda^2,
        //   sin(2th)  = lambda*sqrt(q)*sqrt(1 - lambda^2 q/4),
        // which needs no asin per reflection.
        const DTTH_RAD = 1e-4 * Math.PI / 180;
        const lam2 = lambda * lambda;
        const uniqueQ = [];
        for (let i = 0; i < reflections.length; i++) {
            const q = reflections[i];
            if (uniqueQ.length === 0) { uniqueQ.push(q); continue; }
            const last = uniqueQ[uniqueQ.length - 1];
            const sin2th = lambda * Math.sqrt(last) * Math.sqrt(Math.max(0, 1 - lam2 * last / 4));
            if (q - last > (2 * sin2th / lam2) * DTTH_RAD) uniqueQ.push(q);
        }
        return new Float64Array(uniqueQ);
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
    const sineThetaSq = Math.min(1.0, Math.max(0.0, q_max * lambda * lambda / 4.0));
    const maxTth = Math.asin(Math.sqrt(sineThetaSq)) * 360.0 / Math.PI;
    return generateHKL_for_analysis(cell, lambda, maxTth);
};

const generateQArray_for_worker = (cell, q_max, lambda) => {
    const sineThetaSq = Math.min(1.0, Math.max(0.0, q_max * lambda * lambda / 4.0));
    const maxTth = Math.asin(Math.sqrt(sineThetaSq)) * 360.0 / Math.PI;
    return generateHKL_for_analysis(cell, lambda, maxTth, 'q_only');
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
        default:
            // The switch fell through and the function returned UNDEFINED for
            // any cell whose system it did not recognise (or that had no system
            // at all). Callers then did foundSolutionMap.get(undefined), which
            // collapses every such cell onto a single shared slot -- so the
            // second one and all after it were discarded as "duplicates" of a
            // completely unrelated cell. The indexer only emits the six systems
            // above, so this is a malformed-input path, but it must be explicit
            // and falsy so the `if (!key)` guards downstream can see it.
            return null;
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
   
   
   let qsorted;
    try {
        qsorted = generateQArray_for_worker(cell, q_max, wavelength);
    } catch (e) { return null; }
    if (!qsorted || qsorted.length === 0) return null;



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
            let qsorted;
            try { qsorted = generateQArray_for_worker(trialCell, q_max, wavelength); }
            catch (e) { continue; }
            if (!qsorted || qsorted.length === 0) continue;
            
            
            
            
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
// `tthMin` is optional: when omitted the measured window is inferred from the
// observed peaks, which is conservative (it can only shrink the range in which
// the extinction test counts verified absences).
function analyzeSystematicAbsences(solution, obs_peaks, spaceGroupData, wavelength, tthError, tthMax, impurity_peaks, tthMin) {
    const MAX_VIOLATIONS = 2;
    const fallbackResult = {
        centering: 'Unknown',
        rankedSpaceGroups: [],
        detectedExtinctions: [],
        ambiguousHkls: new Set(),
        hklList:[]
    };
    if (!spaceGroupData?.space_groups) { console.warn("Space group data not loaded"); return fallbackResult; }
    if (!sgEnsureDatabase(spaceGroupData)) {
        console.warn("Space group database has no operator table (rebuild with sg_pack.py)");
        return fallbackResult;
    }
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
    // Evidence bundle for the extinction test: the extinction-blind line list,
    // the measured window, and the observed peak positions expressed in the
    // SAME zero-corrected frame the calculated lines use. Ka2 ghosts are left
    // out - an artefact must not be allowed to refute an absence rule.
    const obsTthCorrected = obs_peaks
        .filter(p => p && !p.ka2Suspect && Number.isFinite(p.tth))
        .map(p => p.tth - zero_correction);
    const measuredLo = Number.isFinite(tthMin)
        ? tthMin - zero_correction
        : (obsTthCorrected.length ? Math.min(...obsTthCorrected) : -Infinity);
    const measuredHi = Number.isFinite(tthMax) ? tthMax - zero_correction : Infinity;
    const detectedExtinctions = detectExtinctions(
        hkls_for_analysis,
        solution.system,
        spaceGroupData,
        centeringResult.plausibleCenterings,
        {
            calcLines: all_calc_hkls,
            obsTth: obsTthCorrected,
            indexWindow,
            overlapWindow,
            tthMin: measuredLo,
            tthMax: measuredHi
        }
    );
    // NOTHING is re-assigned here. The analysis reports what it finds and leaves
    // the indexing alone: silently rewriting an hkl behind the user's back is
    // exactly the behaviour this was changed to avoid. Correcting an assignment
    // is a deliberate act, done through the "Swap hkl" command, which produces a
    // separate solution the user can compare against this one.
    const rankedSpaceGroups = rankSpaceGroups(hkls_for_analysis, solution.system, centeringResult.plausibleCenterings, spaceGroupData, MAX_VIOLATIONS, detectedExtinctions);

    // --- DEMOTED FROM A RANKING TO A COMPATIBILITY LIST ---
    //
    // rankSpaceGroups() orders by matchScore, which counts reflections that ARE
    // PRESENT. As its own comment concedes, that cannot distinguish a rule set
    // from a strictly less-constrained one: confirmations are nearly free, and
    // systematic ABSENCES are the evidence in space-group determination. The
    // extinction bonus was added to patch the asymmetry, but it is an
    // unnormalised heuristic with no scale, so matchScore differences are not
    // comparable across patterns and cannot be turned into odds.
    //
    // Worse, this whole list is judged against ONE cell that was refined
    // without knowing about any extinction rule -- so it was pulled toward the
    // forbidden reflections it is now being asked to rule on -- and the
    // candidate pool is pre-filtered by centeringResult.plausibleCenterings, so
    // a wrong centering verdict removes the correct setting from the list
    // entirely rather than merely demoting it.
    //
    // The ranking authority is sgScoreClass()/sgRankRows() (the Space Group MC):
    // it refits the cell per hypothesis, merges settings the data cannot
    // separate, and scores a real likelihood ratio in nats. What survives here
    // is the part that is still sound -- WHICH settings the observed absences
    // contradict, and by how many reflections -- presented in a neutral order
    // (fewest hard violations first, then space-group number) so no ordering
    // claim is made beyond that.
    //
    // The cap also had to move off matchScore: slicing the top 20 of a heuristic
    // order is a heuristic selection, so a setting could vanish from the report
    // for scoring reasons while being presented as merely "compatible".
    const compatibleSorted = rankedSpaceGroups.slice().sort((a, b) =>
        ((a.hardViolations || 0) - (b.hardViolations || 0)) ||
        ((b.number || 0) - (a.number || 0)) ||
        String(a.symbol || '').localeCompare(String(b.symbol || '')));
    const SG_LIST_CAP = 40;
    const compatibleSettings = compatibleSorted.slice(0, SG_LIST_CAP);

    return {
        centering: centeringResult.description,
        compatibleSettings: compatibleSettings,
        compatibleSettingsTotal: compatibleSorted.length,
        // Deprecated alias, same array. Nothing should rank by this order.
        rankedSpaceGroups: compatibleSettings,
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

// Nearest-line assignment with a one-peak-per-line constraint, resolved
// best-first. Shared by getPeakAssignments() and refineWithManualHkl() so the
// table the user is shown is exactly the assignment the refit will use -- they
// used to run separate copies of this logic and could disagree.
//
// `lines` must be sorted ascending by tth (generateHKL_for_analysis guarantees
// it), so the nearest line is a binary search rather than a linear scan.
// Returns an array, one entry per peak: { line, d } or null.
function assignNearestLines(peaks, lines, zero, window) {
    const n = peaks.length;
    const out = new Array(n).fill(null);
    if (!lines.length || !n) return out;

    const lineTth = new Float64Array(lines.length);
    for (let i = 0; i < lines.length; i++) lineTth[i] = lines[i].tth;

    const cand = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
        const tc = peaks[i].tth - zero;
        const j = binarySearchClosest(lineTth, tc);
        if (j < 0 || j >= lines.length) continue;
        const d = Math.abs(lineTth[j] - tc);
        if (d <= window) cand[i] = { j, d };
    }

    // A calculated line may only be claimed by ONE observed peak -- the rule
    // pair_and_fit() has always enforced. Closest peak wins; the loser is
    // reported unindexed rather than fitted to the same reflection.
    const claimed = new Set();
    for (const { i } of cand.map((c, i) => ({ i, d: c ? c.d : Infinity }))
                            .sort((x, y) => x.d - y.d)) {
        const c = cand[i];
        if (!c || claimed.has(c.j)) continue;
        claimed.add(c.j);
        out[i] = { line: lines[c.j], d: c.d };
    }
    return out;
}

// What is each observed peak currently indexed as?
function getPeakAssignments(solution, obs_peaks, wavelength, tthError, tthMax, limit) {
    const lines = generateHKL_for_analysis(solution, wavelength, tthMax);
    if (!lines.length) return [];
    const zero = solution.zero_correction || 0;
    const window = tthError * 1.5;
    const peaks = (obs_peaks || [])
        .filter(p => typeof p.tth === 'number' && isFinite(p.tth))
        .slice().sort((x, y) => x.tth - y.tth);

    const assigned = assignNearestLines(peaks, lines, zero, window);
    const out = [];
    for (let i = 0; i < peaks.length; i++) {
        const p = peaks[i];
        const tc = p.tth - zero;
        const a = assigned[i];
        const best = a ? a.line : null;
        const dObs = wavelength / (2 * Math.sin(tc * RAD / 2));
        out.push({
            tth: p.tth, tth_corr: tc,
            h: best ? best.h : null, k: best ? best.k : null, l: best ? best.l : null,
            calc_tth: best ? best.tth : null,
            diff: best ? (tc - best.tth) : null,
            d_obs: isFinite(dObs) ? dObs : null,
            d_calc: best ? best.d : null,
            indexed: !!best
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
        const k4 = Number(o.tth).toFixed(4);
        // Overrides are matched to peaks by 2-theta printed to 4 dp. Two
        // overrides on the same key used to silently overwrite each other.
        if (ovr.has(k4)) return { error: `two overrides target the same peak position (${k4} deg)` };
        ovr.set(k4, { h, k, l });
    }
    if (ovr.size === 0) return { error: 'no changes to apply' };

    const toQ = (t) => (4 * Math.sin(t * RAD / 2) ** 2) / (wavelength ** 2);
    const peaks = (obs_peaks || []).filter(p => typeof p.tth === 'number' && isFinite(p.tth))
                                   .slice().sort((x, y) => x.tth - y.tth);

    // Every override must name exactly one peak. An override whose 2-theta
    // matches no peak (the caller changed the range between opening the dialog
    // and applying it) used to be dropped in silence; one that matches two
    // peaks would have been applied to both.
    const keyCount = new Map();
    for (const p of peaks) {
        const k4 = p.tth.toFixed(4);
        keyCount.set(k4, (keyCount.get(k4) || 0) + 1);
    }
    for (const k4 of ovr.keys()) {
        const c = keyCount.get(k4) || 0;
        if (c === 0) return { error: `no peak at ${k4} deg in the current range` };
        if (c > 1) return { error: `two peaks share the position ${k4} deg; cannot target one unambiguously` };
    }

    // Shared with getPeakAssignments: binary search (this used to be an
    // O(peaks x lines) linear scan per peak, on every call) plus the
    // one-peak-per-line rule pair_and_fit() enforces. Manual overrides below
    // still win unconditionally.
    const assigned = assignNearestLines(peaks, lines, zero, window);
    const autoHkl = assigned.map(a => a ? { h: a.line.h, k: a.line.k, l: a.line.l } : null);

    const rows = [], qv = [], tthRads = [], swaps = [];
    for (let i = 0; i < peaks.length; i++) {
        const p = peaks[i];
        const key = p.tth.toFixed(4);
        const tc = p.tth - zero;
        let hkl = autoHkl[i];
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
        rows.push(row);
        // With a zero column in the design matrix the RAW q is the correct
        // right-hand side and the fit recovers the zero itself. With NO zero
        // column the parent's zero has to be applied here instead -- the old
        // code used raw q unconditionally, so every fixed-zero refit was
        // offset by the whole zero shift while its ASSIGNMENTS used tc.
        qv.push(refineZero ? toQ(p.tth) : toQ(tc));
        tthRads.push(p.tth * RAD);
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
    else if (zero) cell.zero_correction = zero;   // carry the parent's fixed zero
    cell.volume = getVolume(cell);
    if (!isFinite(cell.volume) || cell.volume <= 0) return { error: 'refined cell has non-physical volume' };
    try { cell.errors = propagateErrors(system, fit, cell); } catch (e) { cell.errors = null; }

    // Figures of merit exactly as for any other solution: full line list, so the
    // number is directly comparable with the parent and with independent hits.
    try {
        // r.q is the inv_d_sq the generator already computed; toQ(r.tth) just
        // round-tripped it through asin and back.
        const refLines = generateHKL_for_analysis(cell, wavelength, tthMax);
        const qSorted = new Float64Array(Array.from(new Set(refLines.map(r => r.q)))).sort((a, b) => a - b);
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

// R-centring, obverse and reverse. The indexer derives hkl labels from a
// hexagonal METRIC and never fixes the handedness of the in-plane basis against
// c, so obverse (-h+k+l = 3n) and reverse (h-k+l = 3n) are equally consistent
// with the same pattern -- they are a labelling convention, not physics, and no
// powder measurement can separate them. A reflection therefore only counts
// against R if it violates BOTH.
const _R_OBVERSE = (h, k, l) => (((-h + k + l) % 3) + 3) % 3 === 0;
const _R_REVERSE = (h, k, l) => (((h - k + l) % 3) + 3) % 3 === 0;

function determineCentering(indexed_hkls, system) {
    const centeringTests = { 'P': { name: 'Primitive (P)', forbidden: (h, k, l) => false }, 'I': { name: 'Body-centered (I)', forbidden: (h, k, l) => (h + k + l) % 2 !== 0 }, 'F': { name: 'Face-centered (F)', forbidden: (h, k, l) => !( (h%2===0 && k%2===0 && l%2===0) || (h%2!==0 && k%2!==0 && l%2!==0) ) }, 'A': { name: 'A-centered (A)', forbidden: (h, k, l) => (k + l) % 2 !== 0 }, 'B': { name: 'B-centered (B)', forbidden: (h, k, l) => (h + l) % 2 !== 0 }, 'C': { name: 'C-centered (C)', forbidden: (h, k, l) => (h + k) % 2 !== 0 },
        // R was missing entirely, and 'hexagonal' below listed only ['P']. So
        // even once the trigonal groups became reachable, determineCentering()
        // could never return R, allowedCenterings stayed ['P'], and
        // settingCenteringAllowed() then rejected every R setting downstream.
        // Both halves had to be fixed for a rhombohedral cell to be considered.
        'R': { name: 'Rhombohedral (R)', forbidden: (h, k, l) => !(_R_OBVERSE(h, k, l) || _R_REVERSE(h, k, l)) } };
    const validBravaisCenterings = { 'cubic': ['P', 'I', 'F'], 'tetragonal': ['P', 'I'], 'orthorhombic': ['P', 'I', 'F', 'A', 'B', 'C'], 'hexagonal': ['P', 'R'], 'monoclinic': ['P', 'A', 'B', 'C', 'I'], 'triclinic': ['P'] };
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
    // F and I keep P alongside them, exactly as A/B/C and R already do.
    //
    // They used to collapse to ['F'] / ['I'] alone, and that single letter is a
    // HARD COMMIT: plausibleCenterings is the candidate filter for BOTH
    // detectExtinctions() and the compatibility list, so a wrong F or I verdict
    // did not demote the primitive settings, it deleted them. The correct group
    // was then absent from the report rather than ranked below the wrong one --
    // and "never a candidate" is indistinguishable, on the page, from "ruled
    // out". That is the pseudo-symmetry failure mode described at length in the
    // Space Group MC scoring notes: on PbSO4 an I-centred hypothesis looks clean
    // because no measured reflection happens to land in its forbidden parity
    // class, and the true P2_1/a cell is the thing that disappears.
    //
    // The asymmetry was almost certainly an oversight rather than a decision:
    // the R branch immediately below reasons explicitly about not slamming the
    // door on the primitive groups, and the A/B/C branch does the same. F and I
    // qualify on the same bar those do -- zero HARD violations, which P also
    // always has by construction -- so there is no ground for treating them as
    // more certain.
    //
    // This does NOT change the centering line the user sees: `description` is
    // built from reportedCenterings, which strips P a few lines below. Only the
    // candidate pool widens.
    if (plausible.includes('F')) finalCenterings = plausible.includes('P') ? ['F', 'P'] : ['F'];
    else if (plausible.includes('I')) finalCenterings = plausible.includes('P') ? ['I', 'P'] : ['I'];
    // R was absent from this hierarchy too, so even after it became testable it
    // fell through to the A/B/C branch, matched nothing, and was replaced by
    // ['P'] -- the R verdict was computed and then discarded one line later.
    // P is kept alongside it, following the A/B/C precedent rather than the F/I
    // one: the R test accepts obverse OR reverse and is correspondingly
    // permissive, so it should narrow the candidate pool without slamming the
    // door on every primitive hexagonal group.
    else if (plausible.includes('R')) finalCenterings = ['R', 'P'];
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
function detectExtinctions(indexed_hkls, system, spaceGroupData, allowedCenterings, evidence) {
    const confirmedRules = new Set();
    if (!spaceGroupData?.space_groups || indexed_hkls.length === 0) { return ["None detected (no data or rules)"]; }
    if (!sgEnsureDatabase(spaceGroupData)) { return ["None detected (database has no operators)"]; }
    // --- CANDIDATE POOL: CRYSTAL SYSTEM *AND* CENTERING ---
    // Only conditions that a still-viable space group could actually own are
    // testable. Admitting rules from centerings the lattice test has already
    // eliminated lets an accidental agreement in a thinly-sampled zone
    // masquerade as a detected absence. See settingCenteringAllowed().
    const potentialRules = new Set();
    Object.values(spaceGroupData.space_groups).forEach(sg => {
        if (!sgSystemMatches(sg.crystal_system, system)) return;
        sg.settings.forEach(setting => {
            if (!sgSettingAxesMatch(setting, system)) return;
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) return;
            // Printed conditions only. The pool exists to name which INDIVIDUAL
            // condition the data supports, and that is an International Tables
            // presentation question -- absences themselves come from the
            // operators, in countViolations().
            const conditions = setting.conditions || {};
            Object.entries(conditions).forEach(([zone, condList]) => {
                condList.forEach(condStr => { potentialRules.add(`${zone}: ${condStr}`); });
            });
        });
    });
    if (potentialRules.size === 0) { return ["None detected (no rules for system)"]; }
    const parseRuleString = (ruleStr) => { const parts = ruleStr.split(': '); if (parts.length === 2) { return { zone: parts[0].trim(), condition: parts[1].trim() }; } return null; };

    // ================= EVIDENTIAL FLOOR =================
    // "Nothing contradicts it" is not evidence. The loop below only ever looks
    // at reflections that ARE present, so a rule survives by default in any
    // zone too thinly sampled to break it. A condition is only reported if the
    // measurement could have refuted it and did not.
    //
    // (1) ABSENCE TEST. Every calculated line the rule forbids that lies in the
    //     measured range and is RESOLVABLE - no line the rule permits sits
    //     close enough to lend it intensity - must actually be absent. A
    //     resolvable forbidden line carrying an observed peak refutes the rule
    //     outright, and a minimum number of clean absences must remain to
    //     support it.
    //
    //     This stage deliberately does NOT forgive weak peaks, unlike the
    //     contradiction test below. The lowIntensity demotion exists to stop a
    //     weak peak ELIMINATING a space group; letting it also help ASSERT an
    //     absence rule inverts its purpose. Both stages then err the same way:
    //     they refuse to over-constrain the answer.
    //
    // (2) SAMPLE TEST, for rules stronger than "=2n". If a zone permits a
    //     fraction p of its reflections, n observed reflections agree with the
    //     rule by chance with probability p^n. Two observed 0kl reflections
    //     that both happen to have k+l divisible by 4 (p = 1/4, p^n = 6%) are
    //     a coincidence, not a d glide. Plain =2n rules are exempt: they would
    //     need n >= 5 to clear the same bar, which a sparsely populated powder
    //     zone rarely supplies, and they are carried by the absence test.
    //
    // Zektzerite drove both. Its 0kl zone holds five observed reflections, of
    // which 012, 014 and 002 are demoted as weak, leaving 022 and 004 - both
    // with k+l = 4. Nothing contradicted 0kl: k+l=4n, so it was reported; the
    // subsumption pass then deleted the real 0kl: k=2n and 0kl: l=2n because
    // both follow from it; and every candidate group was marked down for
    // failing to explain a condition none of them can have.
    const MIN_VERIFIED_ABSENCES = 1;        // rules forbidding <= half a zone
    const MIN_VERIFIED_ABSENCES_STRONG = 2; // 4n, 3n, compound "h, l=2n", ...
    const CHANCE_LEVEL = 0.05;

    const obsTth = Array.isArray(evidence?.obsTth) ? evidence.obsTth.filter(Number.isFinite) : [];
    const indexWindow = Number.isFinite(evidence?.indexWindow) ? evidence.indexWindow : 0;
    const overlapWindow = Number.isFinite(evidence?.overlapWindow) ? evidence.overlapWindow : indexWindow;
    // A forbidden line is shadowed if an allowed line lies within one overlap
    // window of it, OR close enough that a peak within indexWindow of the
    // forbidden line could equally be that allowed line. Summing the two
    // windows closes the gap between "overlapping lines" and "the peak was
    // indexed to the neighbour", so a real reflection can never be mistaken
    // for a broken absence.
    const shadowWindow = overlapWindow + indexWindow;
    const tthLo = Number.isFinite(evidence?.tthMin) ? evidence.tthMin : -Infinity;
    const tthHi = Number.isFinite(evidence?.tthMax) ? evidence.tthMax : Infinity;
    // NOTE: generateHKL_for_analysis() collapses lines that coincide to within
    // 1e-4 deg, so in high-symmetry systems one of a set of exactly overlapping
    // reflections represents the rest. Such a line is untestable anyway (its
    // partners shadow it), so the only effect is a slightly conservative
    // absence count - hence the deliberately low minimums above.
    const rangeLines = Array.isArray(evidence?.calcLines)
        ? evidence.calcLines
            .filter(x => x && Number.isFinite(x.tth) && x.tth >= tthLo - 1e-9 && x.tth <= tthHi + 1e-9)
            .slice()
            .sort((a, b) => a.tth - b.tth)
        : null;

    const allowedFractionCache = {};
    const allowedFraction = (zone, cond) => {
        const key = zone + '|' + cond;
        if (allowedFractionCache[key] !== undefined) return allowedFractionCache[key];
        const R = 8;
        let total = 0, allowed = 0;
        for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
            if (h === 0 && k === 0 && l === 0) continue;
            if (!zoneApplies(zone, h, k, l)) continue;
            total++;
            if (satisfiesCondition(h, k, l, cond)) allowed++;
        }
        const f = total > 0 ? allowed / total : 1;
        allowedFractionCache[key] = f;
        return f;
    };

    // Per-line lookups that do not depend on the rule, computed once.
    const nLines = rangeLines ? rangeLines.length : 0;
    const peakOnLine = rangeLines
        ? Uint8Array.from(rangeLines, L => obsTth.some(o => Math.abs(o - L.tth) <= indexWindow) ? 1 : 0)
        : null;
    const zoneMaskCache = {};
    const zoneMask = (zone) => {
        if (zoneMaskCache[zone]) return zoneMaskCache[zone];
        const mask = Uint8Array.from(rangeLines, L => zoneApplies(zone, L.h, L.k, L.l) ? 1 : 0);
        zoneMaskCache[zone] = mask;
        return mask;
    };

    // Absences the data can vouch for, and absences the data breaks.
    const absenceEvidence = (zone, cond) => {
        if (!rangeLines || nLines === 0) {
            return { verified: Infinity, broken: 0, tested: false }; // no line list -> test disabled
        }
        const inZone = zoneMask(zone);
        const permits = new Uint8Array(nLines);
        let anyForbidden = false;
        for (let i = 0; i < nLines; i++) {
            const L = rangeLines[i];
            permits[i] = (!inZone[i] || satisfiesCondition(L.h, L.k, L.l, cond)) ? 1 : 0;
            if (!permits[i]) anyForbidden = true;
        }
        if (!anyForbidden) return { verified: 0, broken: 0, tested: true };
        let verified = 0, broken = 0;
        for (let i = 0; i < nLines; i++) {
            if (permits[i]) continue; // the rule allows it; it says nothing
            const t = rangeLines[i].tth;
            let shadowed = false;
            for (let j = i - 1; j >= 0 && t - rangeLines[j].tth <= shadowWindow; j--) {
                if (permits[j]) { shadowed = true; break; }
            }
            for (let j = i + 1; !shadowed && j < nLines && rangeLines[j].tth - t <= shadowWindow; j++) {
                if (permits[j]) { shadowed = true; break; }
            }
            if (shadowed) continue;
            if (peakOnLine[i]) broken++; else verified++;
        }
        return { verified, broken, tested: true };
    };

    // Could n reflections agree with this rule purely by chance?
    const sampleSufficient = (zone, cond, nObs) => {
        const p = allowedFraction(zone, cond);
        if (!(p > 0) || p >= 0.5 - 1e-9) return true; // =2n class and degenerate cases exempt
        return nObs >= Math.ceil(Math.log(CHANCE_LEVEL) / Math.log(p));
    };
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
        if (!allSatisfy) return;

        // --- EVIDENTIAL FLOOR (see the block above the loop) ---
        const strong = allowedFraction(zone, condition) < 0.5 - 1e-9;
        const minAbsences = strong ? MIN_VERIFIED_ABSENCES_STRONG : MIN_VERIFIED_ABSENCES;
        const { verified, broken, tested } = absenceEvidence(zone, condition);
        if (broken > 0) {
            console.debug(`[detectExtinctions] rejected "${ruleStr}": ${broken} resolvable forbidden line(s) carry an observed peak.`);
            return;
        }
        if (tested && verified < minAbsences) {
            console.debug(`[detectExtinctions] rejected "${ruleStr}": only ${verified} verified absence(s) in range, need ${minAbsences}.`);
            return;
        }
        if (!sampleSufficient(zone, condition, refSetForRule.length)) {
            console.debug(`[detectExtinctions] rejected "${ruleStr}": ${refSetForRule.length} reliable reflection(s) in ${zone} cannot distinguish it from chance.`);
            return;
        }
        confirmedRules.add(ruleStr);
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
    if (!sgEnsureDatabase(spaceGroupData)) return [];
    const candidateGroups = Object.values(spaceGroupData.space_groups)
        .filter(sg => sgSystemMatches(sg.crystal_system, system));
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
            // Same filter detectExtinctions() applies to its candidate pool.
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) { continue; }
            
            const rules = setting.conditions || {};
            const violations = countViolations(indexed_hkls, setting);
            
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

// A violation is an OBSERVED reflection the candidate says is systematically
// absent. `setting` supplies the operators, which answer that exactly and
// completely: no zone lookup, no inheritance, and no dependence on whether the
// tables happened to print the condition that kills a particular reflection.
//
// The rule strings are still consulted, but only to NAME the violated condition
// in the detail line the report shows. If no printed condition matches -- which
// happens when the absence follows from a condition the tables leave implied --
// the detail says so rather than inventing one.
function countViolations(indexed_hkls, setting) {
    let hardCount = 0;
    let softCount = 0;
    const detailsHard = [];
    const detailsSoft = [];

    const C = sgOpsCompile(setting);
    if (!C) return { count: 0, hardCount: 0, softCount: 0,
                     details: [], detailsHard: [], detailsSoft: [] };
    const printed = sgSettingConditions(setting);

    const hklViolatesRules = (h, k, l) => sgOpsAbsent(h, k, l, C);

    // Which printed condition explains this absence? Presentation only.
    const nameFor = (h, k, l) => {
        for (let i = 0; i < printed.length; i++) {
            const { zone, cond } = printed[i];
            if (zoneApplies(zone, h, k, l) && !satisfiesCondition(h, k, l, cond)) {
                return `${zone}: ${cond}`;
            }
        }
        return 'a systematic absence of this group';
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
        if (sgOpsAbsent(h, k, l, C)) {
            isViolation = true;
            const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
            violationDetail = `(${h},${k},${l})${tth_string} violates ${nameFor(h, k, l)}${softTagFor(reflection)}`;
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



// Does a database group belong to the crystal system the indexer reported?
//
// The indexer classifies cells by METRIC, and getSymmetry() has no 'trigonal'
// verdict by design: a trigonal cell in hexagonal axes has a = b, gamma = 120,
// which it correctly calls 'hexagonal'. The database, generated with gemmi,
// labels groups 143-167 'trigonal'. A strict equality test therefore hid 25
// groups and 32 settings from EVERY stage of the space-group analysis --
// detectExtinctions(), rankSpaceGroups() and the Monte-Carlo scan alike.
//
// Among them is every R-centred group: 146, 148, 155, 160, 161, 166 and 167.
// Calcite, corundum, hematite, the carbonates and most of the rhombohedral
// oxides were not ranked poorly, they were never candidates at all, and no
// message anywhere said so. The reverse mapping is deliberately NOT applied: a
// genuine 6-fold group must not be offered for a cell the metric only supports
// as trigonal, and since getSymmetry() never emits 'trigonal' the question does
// not arise from the other side.
function sgSystemMatches(sgSystem, system) {
    if (sgSystem === system) return true;
    return system === 'hexagonal' && sgSystem === 'trigonal';
}

// Is this setting's condition list written for the same index convention the
// indexer produces?
//
// This is NOT a question about which cell a lattice can be described in. Every
// rhombohedral lattice can of course be indexed in hexagonal axes, and that is
// exactly what this program does. The question is which of the SEVERAL condition
// lists the database stores for one group refers to the indices we actually
// have. A reflection has different hkl in different settings, so a condition
// list is only meaningful alongside the axes it was written for.
//
// HEXAGONAL. The seven R groups each carry two settings:
//     R-3c  hexagonal axes  hall "-R 3 2\"c"  {hkl: -h+k+l=3n, 0kl: l=2n,
//                                               h0l: l=2n, 00l: l=6n}
//     R-3c  RHOMBOHEDRAL    hall "-P 3* 2n"   {hhl: l=2n, h00: h=2n, 0k0: k=2n}
// Same group, disjoint condition lists, because the second describes the
// primitive rhombohedral cell where the lattice is no longer centred at all.
// Worse, R3, R-3, R32, R3m and R-3m have an EMPTY condition list in rhombohedral
// axes: admitted, they join the class that forbids nothing, so a primitive row
// ends up listing R-3m among its members. Hall notation marks the rhombohedral
// three-fold with '3*' -- exactly seven settings in the bundled database.
//
// MONOCLINIC. The database stores all three unique-axis conventions: 105
// settings, 35 a-unique, 35 b-unique, 35 c-unique. P2_1/c appears as
//     P121/c1  b-unique  {h0l: l=2n, 0k0: k=2n}   <- matches this program
//     P1121/a  c-unique  {hk0: h=2n, 00l: l=2n}
//     P21/c11  a-unique  {0kl: l=2n, h00: h=2n}
// and these are genuinely different behaviours, not restatements. This program
// is b-unique throughout -- getLSDesignRow() returns [h2, k2, l2, h*l] with the
// beta cross-term, extractCellFromFit() pins alpha = gamma = 90, and
// sgEquivalents() uses the 2/m orbit about b -- so the other 70 settings would
// have their conditions tested against indices they do not describe. In Hall
// notation the axis follows the rotation order and z is the default, so '2y'
// marks the b-unique settings exactly.
//
// A setting with no Hall symbol is kept: better to test a condition list that
// might not apply than to silently drop a group over missing metadata.
function sgSettingAxesMatch(setting, system) {
    const hall = String((setting && setting.hall) || '');
    if (!hall) return true;
    if (system === 'hexagonal')  return !hall.includes('3*');
    if (system === 'monoclinic') return hall.includes('2y');
    return true;
}

// Is a space-group setting compatible with the centering(s) the lattice
// analysis left standing?
//
// Shared by rankSpaceGroups() and detectExtinctions() so both stages consider
// exactly the same settings. They used to disagree: the ranking filtered by
// centering, the extinction detector did not, so the detector could "confirm"
// a condition that no surviving space group is even able to possess, and then
// the ranking penalised every candidate for failing to explain it. Zektzerite
// is the worked example - 0kl: k+l=4n exists only in Fdd2 (43) and Fddd (70),
// both F-centred, on a lattice the centering test had already fixed as B with
// zero violations.
//
// An empty/absent allow-list means "no filtering" so older callers behave as
// before.
function settingCenteringAllowed(symbol, allowedCenterings) {
    if (!Array.isArray(allowedCenterings) || allowedCenterings.length === 0) return true;
    const centering = String(symbol || '').charAt(0);
    if (allowedCenterings.includes(centering)) return true;
    // A leading letter that is not a centering type at all is admitted whenever
    // P survived.
    return allowedCenterings.includes('P') && !['I', 'F', 'A', 'B', 'C', 'R'].includes(centering);
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

// ============================================================================
// CRITICAL IMPORT ORDER WARNING:
// This standalone handler belongs to the CPU indexing worker only. It is not
// installed at all inside a refinement worker: refinement-worker.js sets
// self.IS_REFINEMENT_WORKER = true BEFORE its importScripts() call, and the
// guard below sees the flag and skips the assignment. (It is not "clobbered
// afterwards" as this comment used to claim -- refinement-worker.js assigning
// its own self.onmessage happens to have the same effect, but the guard is
// what actually protects the batched engine.)
// DO NOT move the IS_REFINEMENT_WORKER flag below importScripts(), or this
// handler WILL be installed and will then race the batched refinement engine.
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
        let ftStats = null;
        try {
            ftStats = findTransformedSolutions(foundSolutions, data, workerState, self.postMessage.bind(self));
        } catch (err) {
            ftStats = { fatal: String((err && err.message) || err), stack: String(err && err.stack || '') };
        }
        self.postMessage({ type: 'postProcessSummary', payload: ftStats });
        
        self.postMessage({ type: 'progress', payload: 100 });
        self.postMessage({ type: 'done' });
    };
}

// ============================================================================
// SPACE-GROUP MONTE-CARLO SCAN
// ----------------------------------------------------------------------------
// analyzeSystematicAbsences() ranks space groups by counting rule violations
// against ONE fixed cell and ONE fixed peak-to-line assignment. That is a
// bookkeeping test: it can only ever punish a group for an absence that did not
// happen, and it never lets the cell move. On real data the two weaknesses
// compound -- a cell refined against an extinction-blind line list is pulled
// toward forbidden reflections, which mislabels the peaks near them, which then
// manufactures the violations that condemn the correct group.
//
// This does the opposite. For each candidate rule set the line list is
// REGENERATED with the forbidden reflections removed, the cell is re-refined
// (Monte-Carlo + least squares) against that restricted list, and the figures of
// merit are recomputed. Every quantity in the row -- pairing, cell, zero, M20,
// F(N) -- then belongs to the same hypothesis, and the comparison between rows
// is a comparison between hypotheses rather than between bookkeeping artefacts.
//
// Two things make the ranking meaningful:
//
//   * M20 = Q20 / (2<|dQ|> N20) counts N20 = the number of POSSIBLE lines below
//     the 20th observed one. Removing extinct lines lowers N20, so the correct
//     rule set raises M20 for free. An over-restrictive rule set lowers N20 too,
//     but only by throwing away lines that are actually observed -- which shows
//     up as violations, below.
//   * A violation here is an observed peak that has NO allowed line within
//     tolerance but DOES have a forbidden one. That is the direct, physical
//     falsification of a rule set, and it is counted after the cell has been
//     given every chance to move away from it.
//
// Powder absences cannot distinguish space groups that forbid exactly the same
// reflections, so candidates are grouped into EXTINCTION CLASSES first (by what
// they actually forbid, not by how the condition happens to be written) and one
// refinement is run per class. Every group in the class shares the row. This is
// both honest -- it stops the table implying a discrimination the data cannot
// support -- and roughly 5-10x cheaper than one run per setting.
// ============================================================================

// Symmetry-equivalent reflections, used so a line is only treated as extinct
// when EVERY member of its orbit is forbidden.
//
// generateHKL_for_analysis emits one representative per equivalent set (h>=k>=l
// for cubic, and so on) and a reflection condition is not always invariant under
// the choice of representative. R-centring is the worked example: 101 satisfies
// -h+k+l=3n while its equivalent 011 does not, so testing the representative
// alone would delete a line that is present. Erring toward "allowed" is the safe
// direction: a false absence destroys M20, while a missed absence only leaves
// two classes tied.
function sgEquivalents(h, k, l, system) {
    const out = [];
    const push = (a, b, c) => { out.push([a, b, c]); };
    switch (system) {
        case 'cubic': {
            const perms = [[h,k,l],[h,l,k],[k,h,l],[k,l,h],[l,h,k],[l,k,h]];
            for (const [x, y, z] of perms)
                for (const sx of [1,-1]) for (const sy of [1,-1]) for (const sz of [1,-1])
                    push(sx*x, sy*y, sz*z);
            break;
        }
        case 'tetragonal':
            for (const [x, y] of [[h,k],[k,h]])
                for (const sx of [1,-1]) for (const sy of [1,-1]) for (const sz of [1,-1])
                    push(sx*x, sy*y, sz*l);
            break;
        case 'hexagonal': {
            // 6-fold in-plane orbit (h,k) -> (-k,h+k) -> ... plus the mirror
            // (k,h), each with +-l. Covers 3-fold/R settings as well.
            let a = h, b = k;
            for (let r = 0; r < 6; r++) {
                for (const sz of [1,-1]) { push(a, b, sz*l); push(b, a, sz*l); }
                const na = -b, nb = a + b; a = na; b = nb;
            }
            break;
        }
        case 'orthorhombic':
            for (const sx of [1,-1]) for (const sy of [1,-1]) for (const sz of [1,-1])
                push(sx*h, sy*k, sz*l);
            break;
        case 'monoclinic':   // b unique: 2/m
            push(h, k, l); push(-h, k, -l); push(h, -k, l); push(-h, -k, -l);
            break;
        default:             // triclinic: Friedel pair only
            push(h, k, l); push(-h, -k, -l);
            break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Laue-class orbits
// ---------------------------------------------------------------------------
// sgEquivalents() returns the HOLOHEDRAL orbit of a crystal system, which is the
// right thing for undoing the generator's index folding but is NOT the symmetry
// of most space groups. Using it to decide absences over-constrains every group
// whose Laue class is smaller than the holohedral one.
//
// Pa-3 is the worked example. Its point group is m-3, which contains the cyclic
// permutations of h, k, l but NOT the transpositions. The database lists
// 0kl: k=2n, h0l: l=2n, hk0: h=2n. For 210 the hk0 rule gives h=2, fine. But the
// holohedral orbit also contains 120, and applying hk0: h=2n to THAT demanded
// k=2n as well -- so 210 and 320 were marked absent. Both are strong pyrite
// lines. Every m-3 group (Pa-3, Pn-3, Pm-3, Ia-3, Pn-3n ...) lost reflections
// this way.
//
// The Laue class is derivable from the point_group field the database already
// carries, so the orbit can be built correctly.
//
// -3m is mapped to -3, and that is safe rather than merely convenient. The
// point-group string does not record whether the two-folds run along <100>
// (-3m1) or <210> (-31m) -- only standard_symbol does, via the position of the
// "1" in "P 3 2 1" against "P 3 1 2". It turns out not to matter, and the check
// is cheap to state: -3 is contained in both -3m1 and -31m, which are in turn
// contained in 6/mmm, so if the two EXTREMES agree then everything between them
// agrees. For all 18 primitive trigonal groups the -3 and 6/mmm line lists are
// identical, which settles those by bracketing. The 7 R groups do differ between
// the extremes -- 6/mmm is wrong for them, since the six-fold is not an
// operation of the R lattice -- so those were compared against -3m1 directly,
// and are identical for all seven. Calcite predicts the same nine d-spacings
// either way. sgAuditLaueBracket() in the test suite re-checks this.
// The holohedral Laue class of each crystal system -- i.e. the symmetry the HKL
// generator's index folding assumes.
const SG_HOLOHEDRY = {
    cubic: 'm-3m', tetragonal: '4/mmm', hexagonal: '6/mmm',
    orthorhombic: 'mmm', monoclinic: '2/m', triclinic: '-1',
};



// ---------------------------------------------------------------------------
// Compiled reflection conditions
// ---------------------------------------------------------------------------
// satisfiesCondition() re-parses its condition string on every call: two regex
// matches, a split, a map, and a per-term regex. That is fine for the handful of
// calls the absence analysis makes, and ruinous here -- building the extinction
// classes asks it tens of millions of times. This compiles each distinct string
// ONCE into a closure over integer coefficients.
//
// The compiled form is VERIFIED against satisfiesCondition() on a small grid
// before it is trusted, and anything that disagrees (or fails to parse) falls
// back to the original function. The two can therefore never diverge, whatever
// a future database throws at them.
const _SG_COND_CACHE = new Map();
function sgCompileCondition(condStr) {
    const hit = _SG_COND_CACHE.get(condStr);
    if (hit !== undefined) return hit;

    let fn = null;
    try {
        if (condStr === 'h+k, k+l, h+l=2n') {
            fn = (h, k, l) => ((h + k) % 2 === 0) && ((k + l) % 2 === 0) && ((h + l) % 2 === 0);
        } else {
            const rhsMatch = condStr.match(/=\s*(\d+)n/);
            const defaultRhs = rhsMatch ? rhsMatch[0] : '=2n';
            const parts = [];
            for (let piece of condStr.split(',')) {
                let clean = piece.trim().replace(/\*/g, '');
                if (!clean.includes('=')) clean += defaultRhs;
                const m = clean.match(/([0-9]*[hkl+\-]+)\s*=\s*(\d+)n/);
                if (!m) { parts.length = 0; break; }
                const mod = parseInt(m[2], 10);
                if (!isFinite(mod) || mod <= 0) { parts.length = 0; break; }
                let ch = 0, ck = 0, cl = 0, bad = false;
                for (const term of (m[1].match(/[+-]?[0-9]*[hkl]/g) || [])) {
                    const t = term.match(/^([+-]?)(\d*)([hkl])$/);
                    if (!t) { bad = true; break; }
                    const v = (t[1] === '-' ? -1 : 1) * (t[2] ? parseInt(t[2], 10) : 1);
                    if (t[3] === 'h') ch += v; else if (t[3] === 'k') ck += v; else cl += v;
                }
                if (bad) { parts.length = 0; break; }
                parts.push([ch, ck, cl, mod]);
            }
            if (parts.length) {
                fn = (h, k, l) => {
                    for (let i = 0; i < parts.length; i++) {
                        const p = parts[i];
                        const v = p[0] * h + p[1] * k + p[2] * l;
                        if (((v % p[3]) + p[3]) % p[3] !== 0) return false;
                    }
                    return true;
                };
            }
        }
        // Trust nothing that does not reproduce the reference implementation.
        if (fn) {
            for (let h = -3; h <= 3 && fn; h++)
                for (let k = -3; k <= 3 && fn; k++)
                    for (let l = -3; l <= 3; l++) {
                        if (fn(h, k, l) !== !!satisfiesCondition(h, k, l, condStr)) { fn = null; break; }
                    }
        }
    } catch (e) { fn = null; }

    const out = fn || ((h, k, l) => !!satisfiesCondition(h, k, l, condStr));
    _SG_COND_CACHE.set(condStr, out);
    return out;
}


// ============================================================================
// EXTINCTION-CLASS CONSTRUCTION
// ============================================================================
//
// Two levels of grouping are used, and the difference between them matters.
//
//   ABSTRACT class   - groups settings that forbid the same reflections as a
//                      matter of arithmetic, over a small hkl box. Cheap and
//                      cell-independent; used only to avoid enumerating the
//                      same rule set five hundred times.
//   OBSERVABLE class - groups abstract classes that produce the SAME calculated
//                      pattern for THIS cell, at THIS wavelength, over THIS
//                      2-theta range, at THIS tolerance. Two rule sets that
//                      differ only at reflections beyond q_max, or only at
//                      lines that coincide with an allowed line inside the
//                      matching window, are not distinguishable by the
//                      experiment and must not appear as separate rows with
//                      separate figures of merit. That WAS the old behaviour,
//                      and it manufactured exactly the discrimination this
//                      module exists to refuse.
//
// The observable merge runs in sgObservableMerge(), once the parent cell is
// known.
// ============================================================================

// 32-bit FNV-1a. The abstract signature used to be a ~5000-character string
// used directly as a Map key, once per setting; hashing with bucket
// verification keeps the grouping exact at a fraction of the memory.
function sgHash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h >>> 0;
}

// Abstract fingerprint: what does this rule set forbid, arithmetically?
//
// The box has to be wide enough to separate every modulus that occurs: 4n
// (d-glides) and 6n (6_1 screws) need indices that reach the residues those
// conditions reject, and the negative half is needed because conditions like
// -h+k+l = 3n are not symmetric in sign. Range 6 covers all of them with room to
// spare. It is only a PRE-grouping in any case -- the observable merge below
// decides what actually shares a row -- so the cost of widening it further is
// not worth paying.
const SG_SIG_RANGE = 6;
function sgBehaviourSignatureString(allowed) {
    const bits = [];
    for (let h = 0; h <= SG_SIG_RANGE; h++)
        for (let k = -SG_SIG_RANGE; k <= SG_SIG_RANGE; k++)
            for (let l = -SG_SIG_RANGE; l <= SG_SIG_RANGE; l++) {
                if (h === 0 && k === 0 && l === 0) continue;
                bits.push(allowed(h, k, l) ? 1 : 0);
            }
    return bits.join('');
}


const _sgMod = (x, n) => ((x % n) + n) % n;

// Zone probes.
//
// `gen` enumerates the zone with its DEGENERATE SUB-ZONES REMOVED -- 0kl runs
// over k != 0 and l != 0, not over the whole h = 0 plane. A reflection like 00l
// belongs to the 0kl zone AND to the h0l zone, so it carries both conditions;
// including it in the 0kl probe means no single 0kl candidate can ever reproduce
// the pattern and the probe returns '?' for perfectly ordinary groups. (Pbca did
// exactly that.) The axial zones are probed separately, which is where those
// reflections belong.
//
// `tests` are ordered SIMPLEST-FIRST and the first exact match wins. Two
// candidates can both reproduce the pattern once the centering has thinned the
// probe set, and the convention is to name the simpler operation: on a C-centred
// lattice a c-glide and an n-glide perpendicular to b are indistinguishable
// (h is already even, so h+l even means l even), and International Tables writes
// C-c-, not C-n-. Nesting is not a problem here because a stronger condition
// never matches the weaker candidate exactly -- if the truth is k+l = 4n then
// (0,1,1) is absent while "k+l = 2n" predicts it present, so the 2n candidate is
// rejected outright.
const SG_ZONE_PROBES = {
    '0kl': {
        gen: function* (R) { for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) if (k && l) yield [0, k, l]; },
        tests: [['b', (h, k, l) => _sgMod(k, 2) === 0], ['c', (h, k, l) => _sgMod(l, 2) === 0],
                ['n', (h, k, l) => _sgMod(k + l, 2) === 0], ['d', (h, k, l) => _sgMod(k + l, 4) === 0]],
    },
    'h0l': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let l = -R; l <= R; l++) if (h && l) yield [h, 0, l]; },
        tests: [['a', (h, k, l) => _sgMod(h, 2) === 0], ['c', (h, k, l) => _sgMod(l, 2) === 0],
                ['n', (h, k, l) => _sgMod(h + l, 2) === 0], ['d', (h, k, l) => _sgMod(h + l, 4) === 0]],
    },
    'hk0': {
        // |h| == |k| is excluded as well as the axes: (h,h,0) belongs to the hhl
        // zone too, so in a tetragonal or cubic group it carries the hhl
        // condition and no hk0 candidate can reproduce the mixture. I4_1md came
        // out as "I?-d" for exactly that reason.
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) if (h && k && Math.abs(h) !== Math.abs(k)) yield [h, k, 0]; },
        tests: [['a', (h, k, l) => _sgMod(h, 2) === 0], ['b', (h, k, l) => _sgMod(k, 2) === 0],
                ['n', (h, k, l) => _sgMod(h + k, 2) === 0], ['d', (h, k, l) => _sgMod(h + k, 4) === 0]],
    },
    'hhl': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let l = -R; l <= R; l++) if (h && l) yield [h, h, l]; },
        tests: [['c', (h, k, l) => _sgMod(l, 2) === 0], ['b', (h, k, l) => _sgMod(h, 2) === 0],
                ['n', (h, k, l) => _sgMod(2 * h + l, 2) === 0], ['d', (h, k, l) => _sgMod(2 * h + l, 4) === 0]],
    },
    'h-hl': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let l = -R; l <= R; l++) if (h && l) yield [h, -h, l]; },
        tests: [['c', (h, k, l) => _sgMod(l, 2) === 0]],
    },
    '00l': {
        gen: function* (R) { for (let l = -R; l <= R; l++) if (l) yield [0, 0, l]; },
        // axis zones report the MODULUS; the caller names the screw according to
        // the rotation order of the direction (see SG_SYMBOL_DIRECTIONS.names)
        tests: [[2, (h, k, l) => _sgMod(l, 2) === 0], [3, (h, k, l) => _sgMod(l, 3) === 0],
                [4, (h, k, l) => _sgMod(l, 4) === 0], [6, (h, k, l) => _sgMod(l, 6) === 0]],
    },
    '0k0': {
        gen: function* (R) { for (let k = -R; k <= R; k++) if (k) yield [0, k, 0]; },
        tests: [[2, (h, k, l) => _sgMod(k, 2) === 0], [4, (h, k, l) => _sgMod(k, 4) === 0]],
    },
    'h00': {
        gen: function* (R) { for (let h = -R; h <= R; h++) if (h) yield [h, 0, 0]; },
        tests: [[2, (h, k, l) => _sgMod(h, 2) === 0], [4, (h, k, l) => _sgMod(h, 4) === 0]],
    },
};

// One entry per symmetry direction:
//   glide    - the zone whose condition names a glide plane perpendicular to it
//   axis     - the axial zone whose condition names a screw along it
//   names    - modulus -> screw symbol, because the SAME condition means
//              different operations on different axes. 00l: l = 2n is a 2_1 along
//              b in monoclinic, a 4_2 along c in tetragonal and cubic, and a 6_3
//              in hexagonal. Naming them all "2_1" would be wrong, and naming
//              them by modulus alone loses the rotation order.
//   inZones  - the zones that CONTAIN this axis. A screw is only reported when
//              none of them carries a glide, because International Tables omits
//              an axial extinction that already follows by restriction from a
//              zonal one. Pnma is the standard illustration: 0kl: k+l=2n and
//              hk0: h=2n between them force 0k0: k=2n, so the symbol is Pn-a,
//              not Pn2_1a. Note 00l lies inside BOTH hhl and h-hl (h = k = 0
//              satisfies |h| = |k| and h = -k), which is how a c-glide in a
//              trigonal group accounts for its own 00l: l = 2n.
//
// When a direction carries a glide AND a screw that is NOT implied by a
// neighbouring zone, both are reported as "screw/glide" -- P2_1/c, P4_2/n. That
// case was previously collapsed to the glide alone, which put P2_1/c and P2/c on
// one label ("Pc") and P4_2/n and P4/n on another ("Pn--"): four distinct
// hypotheses shown under two names.
const SG_SYMBOL_DIRECTIONS = {
    orthorhombic: [{ glide: '0kl', axis: 'h00', names: { 2: '2\u2081' }, inZones: ['hk0', 'h0l'] },
                   { glide: 'h0l', axis: '0k0', names: { 2: '2\u2081' }, inZones: ['hk0', '0kl'] },
                   { glide: 'hk0', axis: '00l', names: { 2: '2\u2081' }, inZones: ['h0l', '0kl'] }],
    monoclinic:   [{ glide: 'h0l', axis: '0k0', names: { 2: '2\u2081' }, inZones: [] }],
    tetragonal:   [{ glide: 'hk0', axis: '00l', names: { 2: '4\u2082', 4: '4\u2081' }, inZones: ['h0l', '0kl'] },
                   { glide: 'h0l', axis: 'h00', names: { 2: '2\u2081' }, inZones: ['hk0', 'h0l'] },
                   { glide: 'hhl', axis: null,  names: {}, inZones: [] }],
    cubic:        [{ glide: 'hk0', axis: '00l', names: { 2: '4\u2082', 4: '4\u2081' }, inZones: ['h0l', '0kl'] },
                   { glide: 'hhl', axis: null,  names: {}, inZones: [] }],
    hexagonal:    [{ glide: null,  axis: '00l', names: { 2: '6\u2083', 3: '3\u2081', 6: '6\u2081' },
                     inZones: ['h-hl', 'hhl'] },
                   { glide: 'h-hl', axis: null, names: {}, inZones: [] },
                   { glide: 'hhl', axis: null,  names: {}, inZones: [] }],
    triclinic:    [],
};

// One letter for one zone, given that the centering already accounts for part
// of the absences. '-' when the centering explains everything, a letter when
// exactly one candidate reproduces the residual, '?' when none does.
function sgProbeZone(allowed, centPred, zoneKey, R) {
    const probe = SG_ZONE_PROBES[zoneKey];
    if (!probe) return '?';
    const pts = [];
    for (const p of probe.gen(R)) if (centPred(p[0], p[1], p[2])) pts.push(p);
    if (!pts.length) return '-';

    let anyForbidden = false;
    for (const [h, k, l] of pts) if (!allowed(h, k, l)) { anyForbidden = true; break; }
    if (!anyForbidden) return '-';

    for (const [letter, pred] of probe.tests) {
        let exact = true;
        for (const [h, k, l] of pts) {
            if (allowed(h, k, l) !== pred(h, k, l)) { exact = false; break; }
        }
        if (exact) return letter;         // simplest-first, so the first hit is the name
    }
    return '?';
}


// Falls back to "<centering>?" for a system with no direction table, which keeps
// the label honest instead of inventing a symbol we did not derive.
function sgExtinctionSymbol(allowed, system, centering, centPredIn) {
    // centPredIn comes from the setting's own centring operators (identity
    // rotation, non-zero translation), so R-obverse and every other centring
    // are exact rather than looked up from a per-letter table. The 'P' default
    // keeps the probe honest if a caller has no setting to hand.
    const cent = String(centering || 'P').charAt(0) || 'P';
    const centPred = centPredIn || (() => true);
    const dirs = SG_SYMBOL_DIRECTIONS[system];
    if (!dirs) return cent + '?';
    if (!dirs.length) return cent;

    const R = 8;
    const zoneCache = new Map();
    const zone = (key) => {
        if (!key) return '-';
        if (!zoneCache.has(key)) zoneCache.set(key, sgProbeZone(allowed, centPred, key, R));
        return zoneCache.get(key);
    };

    const parts = [];
    for (const d of dirs) {
        if (!d) { parts.push('-'); continue; }
        const g = zone(d.glide);
        // The screw is reported only when it is not already implied by a glide in
        // one of the zones containing this axis.
        const implied = (d.inZones || []).some(z => zone(z) !== '-');
        let sName = '-';
        if (!implied && d.axis) {
            const mod = zone(d.axis);
            if (mod !== '-' && mod !== '?') sName = (d.names && d.names[mod]) || ('?' + mod);
            else sName = mod;
        }
        if (g !== '-' && g !== '?' && sName !== '-' && sName !== '?') parts.push(sName + '/' + g);
        else if (g !== '-') parts.push(g);
        else parts.push(sName);
    }
    return cent + parts.join('');
}

// Every distinct ABSTRACT extinction class of a crystal system.
// `allowedCenterings` is optional; pass the list from determineCentering() to
// scan only the lattices the absences already allow, or omit it to scan all.
// ============================================================================
// SPACE-GROUP CORE: OPERATORS AND ZONES
// ============================================================================
//
// Everything about systematic absences now comes from the symmetry OPERATORS,
// and everything about zone membership from the zone NORMALS. Both arrive in
// sg_ops.json (see sg_pack.py); neither is inferred from a rule string.
//
// ABSENCE. From
//     F(h) = exp(2*pi*i * h.t) * F(hR)      for every operator (R, t)
// it follows that if hR = h then F(h) = exp(2*pi*i h.t) F(h), so F(h) must
// vanish unless h.t is an integer. That is the definition, and it is complete:
// it needs no zone table, no inheritance rule and no centering proxy, because
// the centring translations are themselves operators.
//
// ZONES. A zone is the set of reflections killed by n.h = 0 for each of its
// normals. The old ZONE_PREDICATES table guessed these from the label and got
// three families wrong: 'hhl' was Math.abs(h) === Math.abs(k), which also
// matches h === -k -- the separate 'h-hl' zone, with different conditions in
// trigonal and hexagonal groups. Same for 'hkk'/'hll'/'hkh'. Refereed against
// the generator's own zone records, the string path was wrong on 180
// reflections of P6_3/mmc and 48 of R-3c inside |h|,|k|,|l| <= 5; the operator
// path was wrong on none.
//
// Normals also make membership INCLUSIVE for free, which the old table needed a
// long argument to justify: h00 satisfies the hk0 normal (l = 0), so h00 is in
// hk0 automatically, and a condition stated only on the general zone still
// reaches the special reflections it governs.
//
// INDEX CONVENTION. cctbx writes x' = R x + t with R row-major, so reflection
// indices transform as the ROW vector h' = h R:
//     h'_c = h*R[0*3+c] + k*R[1*3+c] + l*R[2*3+c]
// matching _zone_fixed_by() in the generator, which contracts b[i] with
// R[3*i + c]. Translations are exact rationals t = t_num / t_den.

// ---------------------------------------------------------------------------
// Database registration
// ---------------------------------------------------------------------------
// The zone table is global to a database, so it is installed once at load
// rather than threaded through every call. Both the main thread and each worker
// call this after fetching, because they each hold their own copy of this file.
let SG_ZONE_NORMALS = null;       // label -> [[n0,n1,n2], ...]
let SG_ROTATIONS = null;          // shared rotation table
let _SG_INSTALLED_DB = null;      // identity of the database currently installed

function sgInstallDatabase(db) {
    SG_ROTATIONS = (db && db.rotations) || null;
    SG_ZONE_NORMALS = (db && db.zone_defs) || null;
    _SG_INSTALLED_DB = db || null;
    _SG_ZONE_PRED_CACHE.clear();
    // _SG_OPS_CACHE is keyed on the setting objects themselves, so a different
    // database brings different objects and cannot collide with stale entries.
    return !!(SG_ROTATIONS && SG_ZONE_NORMALS);
}

// Idempotent, and cheap when nothing changed. Called at the top of every entry
// point that receives the database, so neither the main thread nor any worker
// has to remember to install it -- the database is structured-cloned into each
// worker and every context holds its own copy of this file.
function sgEnsureDatabase(db) {
    if (db && db !== _SG_INSTALLED_DB) sgInstallDatabase(db);
    return !!SG_ROTATIONS;
}

const _SG_ZONE_PRED_CACHE = new Map();

// Does a rule labelled `zoneLabel` apply to this reflection?
//
// Exact when the database supplies normals for the label. An unknown label
// falls back to an exact-match on the reported zone name, which can only ever
// under-apply a rule -- never make one match everything.
function zoneApplies(zoneLabel, h, k, l) {
    const H = Math.round(h), K = Math.round(k), L = Math.round(l);
    let pred = _SG_ZONE_PRED_CACHE.get(zoneLabel);
    if (pred === undefined) {
        const normals = SG_ZONE_NORMALS ? SG_ZONE_NORMALS[zoneLabel] : null;
        if (normals && normals.length) {
            const N = normals.map(v => [v[0] | 0, v[1] | 0, v[2] | 0]);
            pred = (a, b, c) => {
                for (let i = 0; i < N.length; i++) {
                    const n = N[i];
                    if (n[0] * a + n[1] * b + n[2] * c !== 0) return false;
                }
                return true;
            };
        } else if (normals) {
            pred = () => true;                       // no normals == the whole of hkl
        } else {
            pred = null;
        }
        _SG_ZONE_PRED_CACHE.set(zoneLabel, pred);
    }
    if (pred) return pred(H, K, L);
    return getReflectionZone(H, K, L) === zoneLabel;
}

// ---------------------------------------------------------------------------
// Compiling a setting's operators
// ---------------------------------------------------------------------------
// Two lists come out, used for different things:
//   R/Tn     operators with a NON-ZERO translation. Only these can extinguish
//            anything, so the absence test never looks at pure rotations. For a
//            symmorphic group this holds just the centring vectors.
//   pgR      the point group: rotation parts with duplicates removed. Needed
//            for epsilon and centricity, never for absence.
//   cenTn    the centring translations: operators whose rotation is the
//            identity. These give the lattice predicate that SG_CENTERING_PRED
//            used to hard-code per letter.
const _SG_OPS_CACHE = new WeakMap();

function sgOpsCompile(setting) {
    if (!setting) return null;
    const cached = _SG_OPS_CACHE.get(setting);
    if (cached !== undefined) return cached;

    const packed = setting.ops;
    const rotations = SG_ROTATIONS;
    if (!packed || !packed.length || !rotations) { _SG_OPS_CACHE.set(setting, null); return null; }
    const den = setting.t_den || 1;

    const absIdx = [], pgRot = [], cen = [];
    const seenRot = new Set();
    for (let i = 0; i < packed.length; i++) {
        const op = packed[i];
        const r = rotations[op[0]];
        if (!r || r.length !== 9) { _SG_OPS_CACHE.set(setting, null); return null; }
        if (!seenRot.has(op[0])) { seenRot.add(op[0]); pgRot.push(r); }
        const isIdentity = r[0] === 1 && r[4] === 1 && r[8] === 1 &&
                           !r[1] && !r[2] && !r[3] && !r[5] && !r[6] && !r[7];
        if (isIdentity) cen.push([op[1], op[2], op[3]]);
        if (op[1] || op[2] || op[3]) absIdx.push(i);
    }

    const nA = absIdx.length;
    const R = new Int32Array(nA * 9);
    const Tn = new Int32Array(nA * 3);
    for (let a = 0; a < nA; a++) {
        const op = packed[absIdx[a]];
        const r = rotations[op[0]];
        for (let j = 0; j < 9; j++) R[a * 9 + j] = r[j];
        Tn[a * 3] = op[1]; Tn[a * 3 + 1] = op[2]; Tn[a * 3 + 2] = op[3];
    }
    const nP = pgRot.length;
    const P = new Int32Array(nP * 9);
    for (let i = 0; i < nP; i++) for (let j = 0; j < 9; j++) P[i * 9 + j] = pgRot[i][j];

    const nC = cen.length;
    const Cn = new Int32Array(nC * 3);
    for (let i = 0; i < nC; i++) { Cn[i * 3] = cen[i][0]; Cn[i * 3 + 1] = cen[i][1]; Cn[i * 3 + 2] = cen[i][2]; }

    const out = { nAbs: nA, R, Tn, den, nPg: nP, pgR: P, nCen: nC, cenTn: Cn,
                  orderZ: packed.length, orderP: nP };
    _SG_OPS_CACHE.set(setting, out);
    return out;
}

// h is systematically absent iff some operator fixes h and shifts its phase.
function sgOpsAbsent(h, k, l, C) {
    const R = C.R, Tn = C.Tn, den = C.den, n = C.nAbs;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        // hR == h ?  One component at a time; most operators fail on the first.
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== l) continue;
        // h.t integral ?  Exact integer arithmetic: h.t = num / den.
        const t = i * 3;
        const num = h * Tn[t] + k * Tn[t + 1] + l * Tn[t + 2];
        // JS % keeps the sign of the dividend and -0 === 0, so this is a
        // correct divisibility test for negative indices as written.
        if (num % den !== 0) return true;
    }
    return false;
}

// Is this reflection allowed by the LATTICE alone? Derived from the centring
// operators, so R-obverse and every other centring come out right without a
// per-letter table.
function sgOpsCenteringPred(C) {
    if (!C || C.nCen === 0) return () => true;
    const Cn = C.cenTn, den = C.den, n = C.nCen;
    return (h, k, l) => {
        for (let i = 0; i < n; i++) {
            const t = i * 3;
            if ((h * Cn[t] + k * Cn[t + 1] + l * Cn[t + 2]) % den !== 0) return false;
        }
        return true;
    };
}

// The powder question: is there a peak at this d-spacing?
//
// A powder line collects every reflection sharing its d, which the METRIC
// decides, not the space group -- so the line survives if ANY member of the
// metric orbit does. The old sgAllowedFn needed "AND inside a Laue orbit, OR
// across orbits" plus a folding walk to work out which members a condition
// governed. Absence is constant on a point-group orbit -- |F(hR)| = |F(h)| --
// so every member of one answers identically and the AND collapses to a plain
// OR over the metric orbit. The Fd-3m and Pa-3 cases the old comment describes
// both come out right without any of that machinery.
function sgOpsAllowedFn(setting, system) {
    const C = sgOpsCompile(setting);
    if (!C) return null;
    if (C.nAbs === 0) return () => true;
    const cache = new Map();
    return (h, k, l) => {
        const key = h + ',' + k + ',' + l;
        const hit = cache.get(key);
        if (hit !== undefined) return hit;
        const orbit = sgEquivalents(h, k, l, system);
        let ok = false;
        for (let i = 0; i < orbit.length; i++) {
            const m = orbit[i];
            if (!sgOpsAbsent(m[0], m[1], m[2], C)) { ok = true; break; }
        }
        cache.set(key, ok);
        return ok;
    };
}

// The label question: what does this group forbid, per reflection rather than
// per powder shell? With operators it is the absence test itself. The Laue-orbit
// restriction the old sgLabelPredicate needed is unnecessary: asking about a
// reflection already asks about its whole orbit.
function sgOpsLabelPredicate(setting) {
    const C = sgOpsCompile(setting);
    if (!C) return null;
    if (C.nAbs === 0) return () => true;
    return (h, k, l) => !sgOpsAbsent(h, k, l, C);
}

// epsilon(h): the order of the stabiliser of h in the point group.
//     <|F(h)|^2> = epsilon(h) * sum_j f_j^2
// Reflections on a symmetry axis or plane are expected epsilon times STRONGER
// than a general reflection at the same resolution.
function sgOpsEpsilon(h, k, l, C) {
    const R = C.pgR, n = C.nPg;
    let eps = 0;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== l) continue;
        eps++;
    }
    return eps || 1;
}

// h is centric iff some operation sends it to -h, which happens in
// non-centrosymmetric groups too: centricity belongs to the reflection, not the
// group. Deliberately unused in the class score -- see SG_USE_EPSILON_WEIGHT.
function sgOpsIsCentric(h, k, l, C) {
    const R = C.pgR, n = C.nPg;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== -h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== -k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== -l) continue;
        return true;
    }
    return false;
}

// A stable key for deduplicating settings that carry identical symmetry.
function sgOpsKey(setting) {
    const ops = (setting && setting.ops) || [];
    let s = (setting.t_den || 1) + '#';
    for (let i = 0; i < ops.length; i++) s += ops[i].join('.') + ';';
    return s;
}

// Every printed condition of a setting, flattened to {zone, cond} pairs. This
// is the presentation layer: what the tables print, for display and for the
// condition-by-condition evidence hunt in detectExtinctions. Absences never
// come from here.
function sgSettingConditions(setting) {
    const out = [];
    const conds = (setting && setting.conditions) || {};
    for (const zone of Object.keys(conds)) {
        const list = conds[zone] || [];
        for (let i = 0; i < list.length; i++) out.push({ zone, cond: list[i] });
    }
    return out;
}

function sgExtinctionClasses(spaceGroupData, system, allowedCenterings) {
    if (!sgEnsureDatabase(spaceGroupData)) return [];
    const groups = Object.values(spaceGroupData?.space_groups || {})
        .filter(sg => sgSystemMatches(sg.crystal_system, system));
    const buckets = new Map();   // hash -> [cls, ...], verified on the full string

    // Settings that carry literally the same conditions must fingerprint the
    // same, so probe once per DISTINCT (conditions, centering) pair rather than
    // once per setting. The bundled database restates the same rule set across
    // many settings of the same group, and the fingerprint is by far the most
    // expensive thing here.
    const probed = new Map();

    for (const sg of groups) {
        for (const setting of (sg.settings || [])) {
            if (!sgSettingAxesMatch(setting, system)) continue;
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) continue;
            const C = sgOpsCompile(setting);
            if (!C) continue;                       // no operators, nothing to say
            const cent = String(setting.centering || setting.symbol || '').charAt(0);
            const ruleKey = sgOpsKey(setting);

            let probe = probed.get(ruleKey);
            if (!probe) {
                const allowed = sgOpsAllowedFn(setting, system);
                probe = { allowed, sigStr: sgBehaviourSignatureString(allowed),
                          labelFn: sgOpsLabelPredicate(setting),
                          centPred: sgOpsCenteringPred(C) };
                probed.set(ruleKey, probe);
            }
            const allowed = probe.allowed;
            const sigStr = probe.sigStr;
            const labelFn = probe.labelFn;
            const centPred = probe.centPred;
            const rules = setting.conditions || {};
            const key = sgHash(sigStr);
            let bucket = buckets.get(key);
            if (!bucket) { bucket = []; buckets.set(key, bucket); }
            let cls = bucket.find(c => c.sigStr === sigStr);
            if (!cls) {
                cls = {
                    sig: key + ':' + bucket.length, sigStr, rules, allowed, labelFn,
                    centPred,
                    label: setting.symbol,
                    centering: cent,
                    members: [],
                    conditions: Object.entries(rules)
                        .map(([z, c]) => `${z}: ${(c || []).join(', ')}`)
                        .sort()
                };
                bucket.push(cls);
            }
            // hall travels with the member because it is the ONLY field that
            // names one setting unambiguously. Several IT numbers cover more
            // than one setting -- 62 is Pnma, Pmnb, Pbnm, Pcmn, Pmcn and Pnam --
            // and they impose different absences on the same cell. A consumer
            // handed only the number and an H-M string has to re-parse the
            // string to work out which; handed the Hall symbol it does not.
            cls.members.push({ number: sg.number, symbol: setting.symbol,
                               hall: setting.hall || null,
                               centric: !!sg.centrosymmetric });
        }
    }

    const out = [];
    for (const bucket of buckets.values()) for (const c of bucket) out.push(c);
    for (const c of out) {
        c.members.sort((a, b) => (a.number - b.number) || a.symbol.localeCompare(b.symbol));
        c.nRules = c.conditions.length;
        c.centric = c.members.length > 0 && c.members.every(m => m.centric);
        c.repSymbol = c.members.length ? c.members[0].symbol : c.label;
        // Label with the extinction symbol, NOT with a representative group.
        try { c.label = sgExtinctionSymbol(c.labelFn || c.allowed, system, c.centering, c.centPred); }
        catch (e) { c.label = c.centering + '?'; }
        c.sigStr = null;   // release the fingerprint; only the key is needed now
    }
    return out;
}

// ============================================================================
// RESOLUTION GROUPS AND THE OBSERVABLE MERGE
// ============================================================================

// The q-space matching tolerance as a function of q, so the resolution of the
// experiment can be evaluated at CALCULATED line positions and not only at
// observed peaks. Algebraically identical to get_q_tolerance():
//   tol = (2 sin(2th)/lambda^2) * d(2th),
//   sin(2th) = lambda*sqrt(q)*sqrt(1 - lambda^2 q/4)
function qToleranceAtQ(q, wavelength, tth_error) {
    if (!(q > 0)) return 1e-9;
    const lam2 = wavelength * wavelength;
    const arg = Math.max(0, 1 - lam2 * q / 4);
    const sin2th = wavelength * Math.sqrt(q) * Math.sqrt(arg);
    const dtth = tth_error * Math.PI / 360;   // half of tth_error, in radians
    return (2 * sin2th / lam2) * dtth + 1e-9;
}

// COMPLETE-linkage clustering of a sorted q list at the matching tolerance. Two
// calculated lines closer together than the tolerance can never be told apart
// by an observed peak, so they form one resolution group and the pattern only
// records whether the GROUP is populated. Everything downstream -- the merge,
// the line count, the informative-absence count -- is expressed in groups
// rather than raw reflections for that reason.
//
// SINGLE linkage was wrong for this. It only asks whether each new line is
// within tolerance of its immediate predecessor, so in a dense high-angle
// region a chain of lines each just inside the tolerance of the next merges
// into one group many tolerances wide. The two ends of such a group are
// perfectly resolvable from each other, and collapsing them costs evidence
// twice over: nInformative falls because several distinct forbidden positions
// are counted as one, and nAllowedInRange falls the same way, which biases p.
// Worse, a single peak anywhere in the chain marks the whole width as
// populated, so a genuine absence at the far end is silently forgiven.
//
// Requiring the new line to lie within tolerance of the FIRST line of the group
// as well caps every group at one tolerance wide, which is exactly the
// statement "no observed peak could tell these apart". Because the list is
// sorted, that first-to-last test is complete linkage: it bounds the maximum
// pairwise separation, not just the nearest-neighbour one.
function sgResolutionGroups(qSorted, wavelength, tth_error) {
    const n = qSorted.length;
    const groupOf = new Int32Array(n);
    if (n === 0) return { groupOf, centres: new Float64Array(0), count: 0 };
    const centres = [];
    let g = 0, sum = qSorted[0], cnt = 1, q0 = qSorted[0];
    for (let i = 1; i < n; i++) {
        const tol = qToleranceAtQ(qSorted[i], wavelength, tth_error);
        if (qSorted[i] - qSorted[i - 1] <= tol && qSorted[i] - q0 <= tol) {
            groupOf[i] = g; sum += qSorted[i]; cnt++;
        } else {
            centres.push(sum / cnt);
            g++; groupOf[i] = g; sum = qSorted[i]; cnt = 1; q0 = qSorted[i];
        }
    }
    centres.push(sum / cnt);
    return { groupOf, centres: Float64Array.from(centres), count: centres.length };
}

// The frame every class is first compared in: the FULL (unrestricted) line list
// for the parent cell, collapsed into resolution groups. Built once per scan.
function sgLatticeFrame(cell, data, state) {
    const wl = data.wavelength;
    let refl;
    try {
        setSpaceGroupFilter(null);
        refl = generateHKL_for_worker(cell, state.q_max, state.d_min, wl);
    } finally { setSpaceGroupFilter(null); }
    if (!refl || !refl.length) return null;

    refl = refl.slice().sort((a, b) => a.q - b.q);
    const qs = Float64Array.from(refl.map(r => r.q));
    const grp = sgResolutionGroups(qs, wl, data.tth_error);

    const w = sgMeasuredWindow(data, state);
    return { refl, qs, groupOf: grp.groupOf, centres: grp.centres, nGroups: grp.count,
             qLo: w.qLo, qHi: w.qHi };
}

// The window inside which an absence is EVIDENCE.
//
// This is the SCANNED range, not the range spanned by the observed peaks. The
// difference decides cases like Pa-3: for pyrite the first allowed reflection is
// 111 at 28.5 deg, while the conditions forbid 100 at 16.3 and 110 at 23.2. If
// the window starts at the first peak, those two absences fall outside it and
// contribute nothing -- the most diagnostic part of the pattern is discarded,
// and Pa-3 cannot be separated from P. If the window starts where the
// diffractometer started, a gap between the scan start and the first peak is
// exactly what it looks like: two lines that should have been there and were
// not.
//
// data.tth_min / data.tth_max are the user's own 2-theta limits. When a caller
// does not supply them the observed span is used, which is the old, conservative
// behaviour.
function sgMeasuredWindow(data, state) {
    const wl = data.wavelength;
    const qOf = (tth) => {
        const st = Math.sin(tth * RAD / 2);
        return (4 * st * st) / (wl * wl);
    };
    let qLo = Infinity, qHi = -Infinity;
    for (const p of state.peaks_sorted_by_q) {
        if (!isFinite(p.q)) continue;
        if (p.q < qLo) qLo = p.q;
        if (p.q > qHi) qHi = p.q;
    }
    if (!isFinite(qLo)) return { qLo: -Infinity, qHi: Infinity };
    qLo *= 0.999; qHi *= 1.001;

    const tLo = Number(data.tth_min), tHi = Number(data.tth_max);
    if (isFinite(tLo) && tLo > 0) {
        const q = qOf(tLo);
        if (isFinite(q) && q < qLo) qLo = q;
    }
    if (isFinite(tHi) && tHi > 0) {
        const q = qOf(tHi);
        if (isFinite(q) && q > qHi) qHi = q;
    }
    return { qLo, qHi };
}

// Which resolution groups survive a rule set? A group survives if ANY line in
// it is allowed -- that is what a powder pattern shows.
function sgGroupMask(frame, allowed) {
    const mask = new Uint8Array(frame.nGroups);
    for (let i = 0; i < frame.refl.length; i++) {
        const g = frame.groupOf[i];
        if (mask[g]) continue;
        const r = frame.refl[i];
        if (allowed(r.h, r.k, r.l)) mask[g] = 1;
    }
    // Outside the measured window nothing is observable, so nothing there may
    // distinguish two rule sets. Clearing those bits is what makes the merge
    // agree with the score.
    const lo = frame.qLo, hi = frame.qHi;
    if (isFinite(lo) && isFinite(hi)) {
        for (let g = 0; g < frame.nGroups; g++) {
            const q = frame.centres[g];
            if (q < lo || q > hi) mask[g] = 0;
        }
    }
    return mask;
}

// Merge abstract classes whose OBSERVABLE pattern is identical for this cell.
// The surviving representative is the one with the fewest stated rules: the
// merged classes are indistinguishable here, and if the Monte-Carlo walk moves
// the cell far enough for them to diverge, erring toward the permissive member
// keeps a real line from being deleted. (A false absence destroys M20; a missed
// absence only leaves two rows tied, which is what the merge already asserts.)
function sgObservableMerge(classes, frame) {
    const byMask = new Map();
    for (const cls of classes) {
        const mask = sgGroupMask(frame, cls.allowed);
        cls.nAllowedGroups = 0;
        for (let i = 0; i < mask.length; i++) if (mask[i]) cls.nAllowedGroups++;
        const key = mask.join('');        // exact, no hashing: a few dozen keys
        const bucket = byMask.get(key);
        if (bucket) bucket.push(cls); else byMask.set(key, [cls]);
    }

    const out = [];
    for (const bucket of byMask.values()) {
        bucket.sort((a, b) => (a.nRules - b.nRules) ||
                              ((a.members?.[0]?.number || 999) - (b.members?.[0]?.number || 999)));
        const rep = bucket[0];
        rep.mergedFrom = bucket.length;
        if (bucket.length > 1) {
            const seen = new Set(rep.members.map(m => m.number + '|' + m.symbol));
            const labels = new Set([rep.label]);
            const conds = new Set(rep.conditions);
            for (let i = 1; i < bucket.length; i++) {
                for (const m of bucket[i].members) {
                    const k = m.number + '|' + m.symbol;
                    if (!seen.has(k)) { seen.add(k); rep.members.push(m); }
                }
                labels.add(bucket[i].label);
                for (const c of bucket[i].conditions) conds.add(c);
            }
            rep.members.sort((a, b) => (a.number - b.number) || a.symbol.localeCompare(b.symbol));
            rep.centric = rep.members.every(m => m.centric);
            rep.mergedLabels = Array.from(labels).sort();
            rep.allConditions = Array.from(conds).sort();
            if (rep.mergedLabels.length > 1) rep.label = rep.mergedLabels.join(' \u2261 ');
        } else {
            rep.mergedLabels = [rep.label];
            rep.allConditions = rep.conditions;
        }
        out.push(rep);
    }
    return out;
}

// ============================================================================
// INTENSITY CONTEXT
// ============================================================================
//
// A violation on the strongest peak in the pattern and a violation on a 0.4%
// shoulder are not the same evidence; the old code counted them identically.
// Relative intensity is measured against a LOCAL baseline for the same reason
// analyzeSystematicAbsences() does: diffracted intensity falls off with angle,
// so a fixed fraction of the GLOBAL maximum quietly strips the whole
// back-reflection region of any power to falsify a rule. The threshold and the
// window match that function exactly so the two analyses cannot disagree about
// which peaks are weak.
const SG_LOW_INTENSITY_FRACTION = 0.05;
const SG_LOCAL_WINDOW_DEG = 15.0;

function sgIntensityContext(peaks) {
    const heighted = (peaks || []).filter(
        p => typeof p.height === 'number' && isFinite(p.height) && p.height > 0);
    let globalMax = 0;
    for (const p of heighted) if (p.height > globalMax) globalMax = p.height;

    const cache = new Map();
    const localMaxAt = (tth) => {
        let m = cache.get(tth);
        if (m !== undefined) return m;
        m = 0;
        for (const p of heighted) {
            if (Math.abs(p.tth - tth) <= SG_LOCAL_WINDOW_DEG && p.height > m) m = p.height;
        }
        if (!(m > 0)) m = globalMax;
        cache.set(tth, m);
        return m;
    };

    // null means "no height information" -- every consumer must treat that as
    // "cannot judge", never as "weak".
    const relI = (p) => {
        if (globalMax <= 0) return null;
        if (!(typeof p.height === 'number' && isFinite(p.height))) return null;
        const scale = localMaxAt(p.tth);
        return scale > 0 ? (p.height / scale) : null;
    };
    const isWeak = (p) => {
        const r = relI(p);
        return (r !== null) && (r < SG_LOW_INTENSITY_FRACTION);
    };
    return { relI, isWeak, haveHeights: globalMax > 0 };
}

// ============================================================================
// WILSON STATISTICS FROM PEAK HEIGHTS
// ============================================================================
//
// A limited, honest borrowing from the Bayesian extinction-symbol method of
// Markvardsen, David, Johnston & Shankland (Acta Cryst A57 (2001) 47).
//
// WHAT THAT METHOD DOES, AND WHY WE CANNOT DO ALL OF IT. ExtSym works from
// Pawley-extracted integrated intensities together with their full covariance
// matrix, which is what lets it ask the sharp question: given the overlaps, what
// is the intensity AT this forbidden position? Brutus has a peak list, not a
// profile. Where no peak was picked there is no measurement at all -- only
// absence. So the sharp question is out of reach, and anything claiming
// otherwise from this data would be pretending.
//
// WHAT IS REACHABLE, AND IS WORTH HAVING. The weakest part of the scoring is the
// single global number p = P(an allowed line is observed). It is the same for a
// strong low-angle reflection with multiplicity 24 and a weak high-angle one with
// multiplicity 2, which is plainly wrong: the first would have been seen if it
// were there, the second might well not. Wilson statistics turn that one number
// into a per-reflection probability, using exactly the information a peak list
// does carry -- position, height, and the multiplicity the lattice implies.
//
//   1. Correct each observed height to a quantity proportional to |F|^2 by
//      dividing out the Lorentz-polarisation factor and the multiplicity.
//   2. Fit the Wilson plot, ln<I/(m*Lp)> = ln K - 2B s^2 with s = sin(theta)/lambda,
//      giving the overall scale K and temperature factor B.
//   3. Take the weakest peak actually picked as the detection limit, and convert
//      it at each position into the |E|^2 a reflection there would need in order
//      to have been seen: z_min = I_min / (m * Lp * K * exp(-2B s^2)).
//   4. p at that position is then the Wilson tail probability of exceeding it:
//      exp(-z) for an acentric distribution, erfc(sqrt(z/2)) for a centric one.
//      The database's centrosymmetric flag says which applies.
//
// PEAK HEIGHTS ARE NOT INTEGRATED INTENSITIES, and that matters less than it
// first appears. The ratio between them is the peak width, which varies smoothly
// and monotonically with angle; a smooth monotonic error in the intensities is
// absorbed almost entirely into the fitted B. The fitted B is therefore NOT a
// physically meaningful temperature factor and is not reported as one. What
// survives the fit is the normalised |E|^2 scale, which is what the statistics
// actually use. Overlapped peaks remain a genuine error: their height is shared
// between reflections the peak list cannot separate.
//
// The context is built ONCE per scan from the unfiltered lattice, never per
// class. Letting each hypothesis fit its own Wilson scale would let it rescale
// the value of its own evidence -- the same failure that made M20 useless for
// ranking, in a new costume.

// Complementary error function, Numerical Recipes erfcc. |error| < 1.2e-7.
function sgErfc(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
              t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
              t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
}

// THE INTENSITY WEIGHT OF A POWDER LINE
//
// The expected line intensity is
//     <I> = Lp * SUM over the coincident reflections of <|F|^2>
//         = Lp * SUM epsilon(h) * sum_j f_j^2
// and by orbit-stabiliser |orbit| * epsilon = |G|. So summing epsilon over a
// full orbit gives |G| REGARDLESS OF THE REFLECTION: at a given resolution the
// mean line intensity does not depend on multiplicity at all. High-multiplicity
// general lines have many modest contributions, low-multiplicity special lines
// have few, each epsilon times stronger, and the two balance exactly. That is
// Wilson's result, and it is not the intuitive answer.
//
// Weighting by multiplicity alone -- which this file used to do -- therefore
// expects a cubic {h00} line (m = 6) to be an EIGHTH the strength of a {hkl}
// line (m = 48) at the same angle, when their means are equal. z_min at h00 is
// inflated eightfold, p comes out far too low, and a clean absence there earns
// almost no credit. On the axial reflections. Which are the ones that decide
// screw axes. Tetragonal is up to 4x, orthorhombic and monoclinic up to 2x.
//
// Computed under the HOLOHEDRY of the crystal system, so it depends only on the
// cell and never on the candidate. That matters: the calibration note below is
// right that a hypothesis able to set its own intensity scale is the M20
// failure in a new costume, and a hypothesis-free weight cannot be gamed.
//
// It is also why centricity stays out, though sgOpsIsCentric exists. An
// extinction class is not a space group: C2, Cm and C2/m share one class with
// three different point groups and three different centricities, so "is this
// class centric" has no answer. The note at the calibration step reached the
// same conclusion from the Pbca/Pbc- failure; the operator view says why.
//
// Set false to restore the old multiplicity weighting for an A/B comparison.
const SG_USE_EPSILON_WEIGHT = true;

const SG_HOLOHEDRY_ORDER = {
    cubic: 48, tetragonal: 16, hexagonal: 24,
    orthorhombic: 8, monoclinic: 4, triclinic: 2,
};

// m * epsilon over the metric orbit. Written out rather than collapsed to the
// constant so the algebra stays visible, and so a resolution group holding two
// coincident families still weighs twice as much as one holding one.
function sgHolohedryWeight(h, k, l, system) {
    const m = sgMultiplicity(h, k, l, system);
    const H = SG_HOLOHEDRY_ORDER[system] || 2;
    return m * (H / m);
}

// Powder multiplicity: how many distinct hkl share this d-spacing by symmetry.
// The generator emits one representative per folded orbit, so this is the size of
// that orbit after removing duplicates.
function sgMultiplicity(h, k, l, system) {
    const orb = sgEquivalents(h, k, l, system);
    const seen = new Set();
    for (let i = 0; i < orb.length; i++) seen.add(orb[i][0] + ',' + orb[i][1] + ',' + orb[i][2]);
    return seen.size || 1;
}

// Lorentz-polarisation for flat-plate Bragg-Brentano with an unpolarised source.
function sgLorentzPol(tthDeg) {
    const th = tthDeg * RAD / 2;
    const st = Math.sin(th), ct = Math.cos(th);
    const c2 = Math.cos(tthDeg * RAD);
    const d = st * st * ct;
    if (!(Math.abs(d) > 1e-9)) return null;
    return (1 + c2 * c2) / d;
}

const SG_WILSON_MIN_PEAKS  = 12;    // below this there are not enough shells
const SG_WILSON_MIN_SHELLS = 3;     // fewer than three and there is no curve

// Build the Wilson context from the observed peaks and the UNFILTERED line list.
// Returns null whenever the data cannot support it, in which case every caller
// falls back to the single global p and behaves exactly as before.
function sgWilsonContext(frame, data, state, system) {
    const wl = data.wavelength;
    if (!frame || !frame.refl || !frame.refl.length) return null;

    // multiplicity and Lp per resolution group
    // Intensity weight per resolution group. m * epsilon, not m -- see the note
    // on sgHolohedryWeight for why multiplicity alone under-values exactly the
    // axial reflections that carry the screw-axis evidence.
    const grpMult = new Float64Array(frame.nGroups);
    for (let i = 0; i < frame.refl.length; i++) {
        const r = frame.refl[i];
        grpMult[frame.groupOf[i]] += SG_USE_EPSILON_WEIGHT
            ? sgHolohedryWeight(r.h, r.k, r.l, system)
            : sgMultiplicity(r.h, r.k, r.l, system);
    }

    // pair observed peaks with groups, and correct their heights
    const pts = [];
    let minHeight = Infinity;
    for (const p of state.peaks_sorted_by_q) {
        const hgt = p.height;
        if (!(typeof hgt === 'number' && isFinite(hgt) && hgt > 0)) continue;
        const j = binarySearchClosest(frame.qs, p.q);
        const tol = qToleranceAtQ(p.q, wl, data.tth_error);
        if (!(Math.abs(frame.qs[j] - p.q) < tol)) continue;      // unindexed: keep it out of the fit
        const g = frame.groupOf[j];
        const m = grpMult[g] || 1;
        const lp = sgLorentzPol(p.tth);
        if (!lp) continue;
        pts.push({ s2: p.q / 4, h: hgt, mlp: m * lp, q: p.q });
        if (hgt < minHeight) minHeight = hgt;
    }
    if (pts.length < SG_WILSON_MIN_PEAKS || !isFinite(minHeight)) return null;

    // BIN IN SHELLS OF s^2 BEFORE FITTING. This is not a refinement, it is the
    // definition: the Wilson relation holds for the MEAN intensity in a shell,
    // <I> = K exp(-2B s^2). Individual reflections scatter enormously about it --
    // for an acentric structure |F|^2 is exponentially distributed, so its
    // standard deviation equals its mean, and a centric one is worse. Fitting
    // reflection by reflection gives a hopeless correlation and the quality gate
    // below (correctly) throws every fit away. Averaging the shells first is what
    // makes the plot linear.
    //
    // The fit uses ln<I>, not <ln I>: the mean of the logarithm is biased low by
    // roughly Euler's constant for an exponential variate, and the bias enters
    // the scale K directly.
    const NB = Math.max(4, Math.min(12, Math.floor(pts.length / 8)));
    let s2min = Infinity, s2max = -Infinity;
    for (const t of pts) { if (t.s2 < s2min) s2min = t.s2; if (t.s2 > s2max) s2max = t.s2; }
    if (!(s2max > s2min)) return null;
    //
    // TRUNCATION CORRECTION. A peak list contains only what rose above the
    // detection limit, and that limit bites hardest exactly where intensities are
    // weakest -- at high angle. The surviving shell means are therefore biased
    // upward by more and more as s^2 grows, the plot flattens, and B comes out
    // far too small. Measured on synthetic data with a known B this cost roughly
    // 2 A^2 and rejected the fit outright whenever the true falloff was gentle.
    //
    // For an exponentially distributed variate observed only above a threshold t,
    // E[y | y > t] = t + mu. Subtracting each reflection's own threshold before
    // averaging therefore recovers mu without needing to know how many
    // reflections were lost -- which is fortunate, because we cannot know that.
    // The threshold is constant in RAW height (it is a property of the pattern,
    // not of the reflection), so in corrected units it is t_i = T/(m_i Lp_i) and
    // the correction is simply (h_i - T)/(m_i Lp_i).
    //
    // This is exact for the acentric case and approximate for the centric one,
    // whose distribution has a heavier weak tail.
    const sum = new Float64Array(NB), cnt = new Float64Array(NB), mid = new Float64Array(NB);
    for (const t of pts) {
        let b = Math.floor((t.s2 - s2min) / (s2max - s2min) * NB);
        if (b >= NB) b = NB - 1; if (b < 0) b = 0;
        sum[b] += (t.h - minHeight) / t.mlp; cnt[b]++; mid[b] += t.s2;
    }
    const shells = [];
    for (let b = 0; b < NB; b++) {
        if (cnt[b] < 2) continue;                       // a shell of one is noise
        const mu = sum[b] / cnt[b];
        if (!(mu > 0)) continue;                        // whole shell sat at the limit
        shells.push({ s2: mid[b] / cnt[b], y: Math.log(mu) });
    }
    if (shells.length < SG_WILSON_MIN_SHELLS) return null;

    // least squares on ln<I/(m Lp)> = lnK - 2B s^2
    let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (const t of shells) { sx += t.s2; sy += t.y; sxx += t.s2 * t.s2; sxy += t.s2 * t.y; syy += t.y * t.y; }
    const n = shells.length;
    const den = n * sxx - sx * sx;
    if (!(Math.abs(den) > 1e-12)) return null;
    const slope = (n * sxy - sx * sy) / den;
    const inter = (sy - slope * sx) / n;
    const rNum = n * sxy - sx * sy;
    const rDen = Math.sqrt(Math.max(1e-30, (n * sxx - sx * sx) * (n * syy - sy * sy)));
    const r2 = (rNum / rDen) * (rNum / rDen);
    if (!isFinite(slope) || !isFinite(inter)) return null;

    const B = -slope / 2;                 // apparent, not physical -- see the note above
    const K = Math.exp(inter);

    // USE THE SHELLS THEMSELVES, NOT THE STRAIGHT LINE.
    //
    // What the scoring needs is <I>(q), the mean corrected intensity at a given
    // position. The Wilson straight line is one model of that, and requiring the
    // data to obey it was a mistake: a structure with a gentle falloff produces a
    // nearly flat plot, the correlation is poor, and a hard R^2 gate then threw
    // away a perfectly usable intensity scale for no reason (measured: B = 0.5
    // and B = 1.5 were both rejected outright while B = 3 and B = 5 passed).
    //
    // Interpolating ln<I> between the shell means needs no model at all and is
    // exactly as good wherever there are shells. The fitted slope is kept only to
    // extrapolate past the ends, and B and R^2 are reported as diagnostics rather
    // than used as gates.
    shells.sort((a, b) => a.s2 - b.s2);
    const first = shells[0], last = shells[shells.length - 1];
    const meanF2 = (q) => {
        const x = q / 4;
        if (x <= first.s2) return Math.exp(first.y + slope * (x - first.s2));
        if (x >= last.s2)  return Math.exp(last.y  + slope * (x - last.s2));
        for (let i = 1; i < shells.length; i++) {
            if (x <= shells[i].s2) {
                const a = shells[i - 1], b = shells[i];
                const f = (b.s2 > a.s2) ? (x - a.s2) / (b.s2 - a.s2) : 0;
                return Math.exp(a.y + f * (b.y - a.y));
            }
        }
        return Math.exp(last.y);
    };
    const zMin = (q, mult) => {
        const tth = 2 * Math.asin(Math.min(1, wl * Math.sqrt(Math.max(0, q)) / 2)) * DEG;
        const lp = sgLorentzPol(tth);
        const mf = meanF2(q);
        if (!lp || !(mf > 0) || !(mult > 0)) return null;
        return (minHeight / (mult * lp)) / mf;
    };

    // ------------------------------------------------------------------
    // CALIBRATION: Wilson supplies the SHAPE, the data supply the LEVEL.
    //
    // Two things went wrong when the raw Wilson probability was used directly.
    //
    // First, choosing the centric or acentric distribution per class made the
    // ranking swing on a property a powder pattern cannot see. A centrosymmetric
    // class got the centric distribution, which has more weak reflections, so a
    // lower p and less credit per absence; a class that merely happened to
    // contain one acentric member got the acentric distribution and more credit.
    // Pbca lost to Pbc- on real synthetic Pbca data for precisely this reason --
    // fewer clean absences, higher score. One distribution is now used for every
    // row, so the choice cancels out of every comparison.
    //
    // Second, the absolute level was too high. Measured against synthetic data
    // where 62% of lines were detectable, the raw Wilson p averaged about 80%,
    // which inflated each absence from 0.94 nats to 1.59. Multiplied over a
    // hundred forbidden positions that is enough for a class with 28 hard
    // violations to outscore one with none -- the over-restriction failure this
    // whole scoring scheme exists to prevent, returning by another route.
    //
    // The fix keeps what Wilson is genuinely good for -- knowing that a strong
    // low-angle reflection of high multiplicity would have been seen while a weak
    // high-angle one might not -- and takes the overall level from the observed
    // detection rate instead. A single scale factor on the threshold is solved by
    // bisection so that the mean predicted detectability over the lattice matches
    // the fraction of positions that actually carry a peak.
    // ------------------------------------------------------------------
    // THE REFERENCE SET AND THE TARGET RATE MUST MATCH.
    //
    // What follows is only the SEED calibration, and it is deliberately the same
    // hypothesis as pass 1 of sgRescoreAll(): the unfiltered lattice, i.e. "no
    // extinctions at all". That reference is biased low for exactly the reason
    // pass 1 is -- if the true lattice is centred, every systematically absent
    // position sits in the denominator and can never be observed, so the
    // detection rate looks worse than it is.
    //
    // The bias used to be permanent, because lambda was solved once here and
    // never revisited while p-hat was re-estimated from the winner. The two
    // levels then disagreed: absences were weighted by a detectability pinned to
    // the no-extinction lattice and compared against a p measured on the
    // winner's own allowed positions. solveLambda() below is exported so the
    // scoring can redo this calibration against whatever reference set it is
    // using at the time, which is what keeps the two consistent.
    const inWinQ = [], inWinM = [];
    for (let g = 0; g < frame.nGroups; g++) {
        const qc = frame.centres[g];
        if (qc < frame.qLo || qc > frame.qHi) continue;
        inWinQ.push(qc); inWinM.push(grpMult[g] || 1);
    }
    if (!inWinQ.length) return null;
    // centres are ascending, so the in-window slice is too and a peak can be
    // placed by bisection instead of by scanning every group (this was O(pts x
    // groups), with a half-finished binary search left declared beside it).
    const inWinQArr = Float64Array.from(inWinQ);
    const seen = new Uint8Array(inWinQArr.length);
    for (const t of pts) {
        const i = binarySearchClosest(inWinQArr, t.q);
        if (i >= 0 && Math.abs(inWinQArr[i] - t.q) < qToleranceAtQ(t.q, wl, data.tth_error)) seen[i] = 1;
    }
    let nSeen = 0;
    for (let i = 0; i < seen.length; i++) nSeen += seen[i];
    const pEmpirical = Math.min(0.98, Math.max(0.02, nSeen / inWinQArr.length));

    // z is the detection threshold at a position in units of the local mean
    // intensity: it depends on the pattern and the position, never on lambda.
    // Every calibration is therefore just a rescaling of a fixed list of z.
    const zList = [];
    for (let i = 0; i < inWinQArr.length; i++) {
        const z = zMin(inWinQArr[i], inWinM[i]);
        if (z !== null && isFinite(z)) zList.push(Math.max(0, z));
    }

    // Solve mean(exp(-z*lambda)) = target over a supplied list of z. Monotone
    // decreasing in lambda, so plain bisection in log-lambda converges.
    const meanPOf = (zs, lam) => {
        if (!zs || !zs.length) return null;
        let acc = 0;
        for (let i = 0; i < zs.length; i++) acc += Math.exp(-zs[i] * lam);
        return acc / zs.length;
    };
    const solveLambda = (zs, target) => {
        const tgt = Math.min(0.98, Math.max(0.02, target));
        if (!zs || !zs.length || !isFinite(tgt)) return null;
        let lamLo = 1e-3, lamHi = 1e3, lam = 1;
        for (let it = 0; it < 60; it++) {
            lam = Math.sqrt(lamLo * lamHi);
            const mp = meanPOf(zs, lam);
            if (mp === null) return null;
            if (mp > tgt) lamLo = lam; else lamHi = lam;
        }
        return lam;
    };
    const lambda = solveLambda(zList, pEmpirical) ?? 1;
    const clampP = (v) => Math.min(0.999, Math.max(0.001, v));
    const rawP = (q, mult, lam) => {
        const z = zMin(q, mult);
        if (z === null || !isFinite(z)) return null;
        return Math.exp(-Math.max(0, z) * lam);
    };

    return {
        B, K, r2, nUsed: pts.length, nShells: n, minHeight,
        pEmpirical, lambda, calibrated: meanPOf(zList, lambda),
        groupMultiplicity: (g) => grpMult[g] || 1,
        // The lambda-free part of the detectability: exported so callers can
        // store z once and re-weight later under a recalibrated lambda.
        zAt: (q, mult) => {
            const z = zMin(q, mult);
            return (z === null || !isFinite(z)) ? null : Math.max(0, z);
        },
        pFromZ: (z, lam) => {
            if (z === null || z === undefined || !isFinite(z)) return null;
            const l = (lam !== null && lam !== undefined && isFinite(lam)) ? lam : lambda;
            return clampP(Math.exp(-Math.max(0, z) * l));
        },
        solveLambda,
        meanPOf,
        // Probability that a reflection PRESENT at this position would have been
        // detected. The SHAPE across positions is Wilson's; the overall level is
        // pinned to the observed detection rate by `lambda`.
        pDetect: (q, mult) => {
            const v = rawP(q, mult, lambda);
            if (v === null) return null;
            return clampP(v);
        },
        // observed |E|^2 of a peak: how strong is it, in units of the local mean
        eSquared: (q, height, mult) => {
            if (!(height > 0) || !(mult > 0)) return null;
            const tth = 2 * Math.asin(Math.min(1, wl * Math.sqrt(Math.max(0, q)) / 2)) * DEG;
            const lp = sgLorentzPol(tth);
            const mf = meanF2(q);
            if (!lp || !(mf > 0)) return null;
            return (height / (mult * lp)) / mf;
        },
    };
}

// ============================================================================
// PER-CLASS INDEXING STATISTICS
// ============================================================================
//
//   indexed    - the peak is matched to an ALLOWED line
//   violation  - the peak is matched only to a FORBIDDEN line. Direct evidence
//                against the rule set, graded by how believable the peak is.
//   unindexed  - no line at all within tolerance. Historically excluded from the
//                ranking on the grounds that "every class carries it equally";
//                that stopped being true the moment each class got its own
//                refined cell, so it now enters the score.
//   clean      - a forbidden resolution group that is INFORMATIVE (contains no
//                allowed line, so its emptiness is attributable) and where no
//                peak was observed. This is the POSITIVE evidence for the rule
//                set, and the old version never counted it at all.
//
// Peak-to-line matching is INJECTIVE: one calculated line serves at most one
// observed peak, closest pair first. That is the rule pair_and_fit() and
// _swapCheapFit() already use. Without it a class with a dense line list can
// "index" thirty peaks onto ten lines and be flattered for it.
//
// Allowed lines are offered first, so a peak that could sit on either an allowed
// or a forbidden line is credited to the hypothesis rather than counted against
// it. Only what is left over can become a violation.
function sgIndexingStats(cell, data, state, allowed, ictx, preRefl, wilson) {
    const wl = data.wavelength;
    const z = cell.zero_correction || 0;
    const tolFn = (idx) => get_q_tolerance(idx, state.tth_obs_rad, wl, data.tth_error);

    // TWO generation passes, and it has to be two.
    //
    // generateHKL_for_analysis() DEDUPES reflections that share a 2-theta, keeping
    // one arbitrary representative. Generating the full list once and labelling
    // each survivor allowed/forbidden therefore misclassifies every q where an
    // allowed and a forbidden reflection coincide: if the dedupe happened to keep
    // the forbidden one, the line is recorded as forbidden and any peak sitting
    // on it becomes a violation against the correct rule set.
    //
    // Pa-3 is the worked example. 330 is forbidden (hk0: h=2n) and 411 is
    // allowed, and in a cubic cell they sit at exactly the same q (N = 18). The
    // single-pass version kept 330, so pyrite's own pattern produced a hard
    // violation against Pa-3 at 74.2 deg.
    //
    // Generating the allowed list WITH the filter installed makes the dedupe
    // happen among allowed reflections only, which is what the original
    // two-array version did correctly.
    let reflAll = preRefl;
    if (!reflAll) {
        try {
            setSpaceGroupFilter(null);
            reflAll = generateHKL_for_worker(cell, state.q_max, state.d_min, wl);
        } finally { setSpaceGroupFilter(null); }
        if (!reflAll || !reflAll.length) return null;
        reflAll = reflAll.slice().sort((a, b) => a.q - b.q);
    }
    let reflOk;
    try {
        setSpaceGroupFilter(allowed);
        reflOk = generateHKL_for_worker(cell, state.q_max, state.d_min, wl);
    } finally { setSpaceGroupFilter(null); }
    reflOk = (reflOk || []).slice().sort((a, b) => a.q - b.q);

    // Merge into one position list, marking which positions carry an allowed
    // line. A position counts as allowed when an allowed reflection falls within
    // the matching tolerance of it.
    const nL = reflAll.length;
    if (!nL) return null;
    const allowedQ = reflOk.map(r => r.q);
    const qs = new Float64Array(nL);
    const isAllowed = new Uint8Array(nL);
    for (let j = 0; j < nL; j++) {
        qs[j] = reflAll[j].q;
        if (allowedQ.length) {
            const i = binarySearchClosest(allowedQ, qs[j]);
            isAllowed[j] = Math.abs(allowedQ[i] - qs[j]) <= qToleranceAtQ(qs[j], wl, data.tth_error) ? 1 : 0;
        }
    }
    const refl = reflAll;

    const grp = sgResolutionGroups(qs, wl, data.tth_error);
    const nG = grp.count;

    // --- observed peaks in q, zero-corrected ---------------------------------
    const obs = [];
    for (const p of state.peaks_sorted_by_q) {
        const tc = p.tth - z;
        if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
        const st = Math.sin(tc * RAD / 2);
        const q = (4 * st * st) / (wl * wl);
        if (!isFinite(q)) continue;
        obs.push({ p, q, tol: tolFn(p.original_index), assigned: false, line: -1, dq: 0 });
    }
    if (!obs.length) return null;

    // --- injective assignment ------------------------------------------------
    const pairUp = (wantAllowed) => {
        const cand = [];
        for (let i = 0; i < obs.length; i++) {
            const o = obs[i];
            if (o.assigned) continue;
            const start = binarySearchClosest(qs, o.q);
            // walk outward from the nearest line until BOTH sides leave the window
            for (let d = 0; ; d++) {
                const a = start - d, b = start + d;
                const aIn = a >= 0 && Math.abs(qs[a] - o.q) < o.tol;
                const bIn = b < nL && Math.abs(qs[b] - o.q) < o.tol;
                if (aIn && isAllowed[a] === wantAllowed) cand.push({ i, j: a, dq: Math.abs(qs[a] - o.q) });
                if (d > 0 && bIn && isAllowed[b] === wantAllowed) cand.push({ i, j: b, dq: Math.abs(qs[b] - o.q) });
                const aDead = (a < 0) || !aIn;
                const bDead = (b >= nL) || !bIn;
                if (aDead && bDead) break;
            }
        }
        cand.sort((x, y) => x.dq - y.dq);
        const takenLine = new Set();
        for (const c of cand) {
            if (obs[c.i].assigned || takenLine.has(c.j)) continue;
            obs[c.i].assigned = true;
            obs[c.i].line = c.j;
            obs[c.i].dq = c.dq;
            takenLine.add(c.j);
        }
    };

    pairUp(1);   // allowed lines first
    pairUp(0);   // then whatever forbidden lines explain the leftovers

    // --- tallies -------------------------------------------------------------
    const nearAllowed = (q, tol) => {
        if (!allowedQ.length) return false;
        const j = binarySearchClosest(allowedQ, q);
        return Math.abs(allowedQ[j] - q) < tol * 1.5;
    };

    let indexed = 0;
    const violations = [];
    const unindexed = [];
    for (const o of obs) {
        if (!o.assigned) { unindexed.push(o); continue; }
        if (isAllowed[o.line]) { indexed++; continue; }
        const gMult = wilson ? wilson.groupMultiplicity(grp.groupOf[o.line]) : null;
        violations.push({
            tth: o.p.tth,
            rel: ictx.relI(o.p),
            // |E|^2 of the offending peak: its intensity in units of the mean at
            // this angle, corrected for Lp and multiplicity. A value near or
            // above 1 is unmistakably a real reflection; well below 0.1 is the
            // sort of thing a tail or a little noise produces.
            eSq: wilson ? wilson.eSquared(o.q, o.p.height, gMult) : null,
            // probability a reflection present here would have been detected,
            // at the seed calibration; zLocal is the same quantity before
            // lambda is applied, so the scoring can re-weight it once lambda
            // has been re-solved against the winner (see sgRescoreAll).
            pLocal: wilson ? wilson.pDetect(o.q, gMult) : null,
            zLocal: (wilson && wilson.zAt) ? wilson.zAt(o.q, gMult) : null,
            dqOverTol: o.tol > 0 ? o.dq / o.tol : null,
            ka2: !!o.p.ka2Suspect,
            weak: ictx.isWeak(o.p),
            // an allowed line sits close enough that the assignment is a
            // judgement call rather than a fact
            ambiguous: nearAllowed(o.q, o.tol),
        });
    }

    // --- informative absences ------------------------------------------------
    // A forbidden group only carries evidence if NO allowed line shares it
    // (otherwise a peak appears there either way) and it lies inside the
    // measured range. Of those, count the ones that are in fact empty.
    const groupHasAllowed = new Uint8Array(nG);
    const groupExists = new Uint8Array(nG);
    for (let j = 0; j < nL; j++) {
        groupExists[grp.groupOf[j]] = 1;
        if (isAllowed[j]) groupHasAllowed[grp.groupOf[j]] = 1;
    }
    const groupObserved = new Uint8Array(nG);
    for (const o of obs) if (o.assigned) groupObserved[grp.groupOf[o.line]] = 1;

    // A peak within tolerance of a forbidden position means that position is NOT
    // empty, whoever ends up owning the peak.
    //
    // The assignment above is injective and offers allowed lines first, which is
    // right for deciding what counts as a violation but wrong for deciding what
    // counts as an ABSENCE. A peak sitting within tolerance of both an allowed
    // line and a forbidden one is credited to the allowed line -- so it raises no
    // violation -- and the forbidden group, having no peak assigned to it, was
    // then also banked as a clean absence. The hypothesis collected positive
    // evidence from a position where a peak demonstrably sits, and collected it
    // precisely in the ambiguous cases where it has earned the least.
    //
    // Only groups with no allowed line of their own are re-marked here, so
    // nAllowedInRange / nAllowedObserved -- and therefore p -- are untouched.
    for (const o of obs) {
        const start = binarySearchClosest(qs, o.q);
        for (let d = 0; ; d++) {
            const a = start - d, b = start + d;
            const aIn = a >= 0 && Math.abs(qs[a] - o.q) < o.tol;
            const bIn = b < nL && Math.abs(qs[b] - o.q) < o.tol;
            if (aIn && !groupHasAllowed[grp.groupOf[a]]) groupObserved[grp.groupOf[a]] = 1;
            if (d > 0 && bIn && !groupHasAllowed[grp.groupOf[b]]) groupObserved[grp.groupOf[b]] = 1;
            if ((a < 0 || !aIn) && (b >= nL || !bIn)) break;
        }
    }

    // Same window as the merge: the scanned range, not the observed span.
    const win = sgMeasuredWindow(data, state);
    const qLo = win.qLo, qHi = win.qHi;

    let nInformative = 0, nClean = 0, nAllowedInRange = 0, nAllowedObserved = 0, nLines = 0;
    const cleanP = [], cleanZ = [], allowedZ = [];
    for (let g = 0; g < nG; g++) {
        if (!groupExists[g]) continue;
        const qc = grp.centres[g];
        // nLines used to be tallied HERE, before the window test, so the
        // displayed line count included groups the experiment never scanned and
        // disagreed with every other count in the row. It is a display column
        // and nothing reads it back, but it should still mean what the header
        // says: resolvable allowed lines inside the measured range.
        if (qc < qLo || qc > qHi) continue;
        if (groupHasAllowed[g]) {
            nLines++;
            nAllowedInRange++;
            if (groupObserved[g]) nAllowedObserved++;
            // The allowed positions are the reference set p is measured on, so
            // they are also the set lambda has to be calibrated against if the
            // two are to describe the same hypothesis.
            if (wilson && wilson.zAt) {
                const za = wilson.zAt(qc, wilson.groupMultiplicity(g));
                if (za !== null) allowedZ.push(za);
            }
        } else {
            nInformative++;
            if (!groupObserved[g]) {
                nClean++;
                // Per-position weight for this absence. A forbidden line that
                // would have been strong and obvious is powerful evidence when
                // it is missing; one that would have been invisible anyway is
                // almost none. That distinction is exactly what a single global
                // p cannot make.
                if (wilson) {
                    const pd = wilson.pDetect(qc, wilson.groupMultiplicity(g));
                    if (pd !== null) cleanP.push(pd);
                    if (wilson.zAt) {
                        const zc = wilson.zAt(qc, wilson.groupMultiplicity(g));
                        if (zc !== null) cleanZ.push(zc);
                    }
                }
            }
        }
    }

    // strongest violations first: those are the ones worth showing
    violations.sort((a, b) => (b.rel ?? 1) - (a.rel ?? 1));

    return {
        indexed,
        violations,
        nViolations: violations.length,
        nHard: violations.filter(v => !v.ka2 && !v.weak && !v.ambiguous).length,
        unindexed: unindexed.length,
        unindexedRel: unindexed.map(o => ictx.relI(o.p)),
        nClean, nInformative, cleanP, cleanZ, allowedZ,
        nAllowedInRange, nAllowedObserved,
        nLines,
        violatingTth: violations.slice(0, 8).map(v => v.tth),
    };
}

// ============================================================================
// SCORING
// ============================================================================
//
// M20 CANNOT arbitrate between rule sets, and the old ranking used it as if it
// could. M20 = Q20 / (2<|dQ|> N20), where N20 counts the POSSIBLE lines below
// the 20th observed one -- so deleting ANY line raises M20. The previous comment
// argued that an over-restrictive class pays for that in violations, but that
// only holds if every allowed line is observed. Real patterns miss plenty of
// weak ones, so a class could delete unobserved lines, gain M20 for nothing at
// zero violations, and then win the nRules tie-break as well: the same bias
// applied twice.
//
// What replaces it is a likelihood ratio, in the spirit of the Bayesian
// extinction-symbol work (Markvardsen, David, Johnston & Shankland, Acta Cryst
// A57 (2001) 47), reduced to what peak POSITIONS alone can support -- no Wilson
// statistics, because that needs background-subtracted, Lorentz-polarisation
// corrected integrated intensities and we have peak heights.
//
// Let p = P(a symmetry-allowed line inside the measured range yields a
// detectable peak), estimated from the data. For a candidate rule set H:
//
//   * every INFORMATIVE forbidden group that is EMPTY contributes
//         log( (1 - eps_clean) / (1 - p) )
//     Under H the group should be empty bar a spurious-peak rate; under "no
//     extinction" it would have been empty only with probability 1 - p. When p
//     is small -- most allowed lines unobserved anyway -- this is near zero,
//     which is right: absences prove little in a sparse pattern. When p
//     approaches 1, each clean absence is worth several nats. THIS is the term
//     that makes the criterion self-limiting: adding a restriction that deletes
//     an unobserved line buys log(1/(1-p)), not a free M20 increase.
//
//   * every VIOLATION contributes log( eps_i / p ), with eps_i set by how
//     believable that particular peak is. A strong, clean, unambiguous peak on a
//     forbidden line is close to fatal; a Ka2 ghost or a 2%-of-local shoulder is
//     not.
//
//   * every UNINDEXED peak contributes log( eps_unindexed ): the refined cell
//     explains it with nothing at all.
//
//   * a BIC term -k/2 * ln(n_indexed) charges for the refined parameters. k is
//     the same for every row, so this only differentiates through how many lines
//     actually constrain the fit -- which is exactly the asymmetry that let a
//     class with very few allowed lines absorb error into its own zero-point for
//     free.
//
// p is estimated ONCE and shared by every row (see sgRescoreAll). Letting each
// class estimate its own p would let it raise the value of its own evidence by
// discarding lines, which is the M20 failure wearing a different hat.

// The rate at which a peak appears where a rule set says none should be.
//
// The soft categories are CONDITIONAL on an identified alternative explanation
// for that particular peak -- there is a Ka1 parent at the right offset, or an
// allowed line sits inside the window -- so they are genuine probabilities for
// that peak and do not depend on how large the pattern is.
//
// The base rate for a strong, clean, unexplained peak is different in kind: it is
// a rate PER FORBIDDEN POSITION, and it must therefore scale with how many
// positions there are. It used to be pinned at 0.02, which on a pattern with 700
// resolvable positions asserts that fourteen spurious peaks are expected. On a
// real PbSO4 dataset -- 192 peaks, 700 positions, so only 27% of possible lines
// observed -- that let the I-centred class survive NINETEEN hard violations:
// its 219 empty forbidden positions earned +65.8 nats while the violations cost
// only -46.6, so it outscored the correct P2_1/a class, which had 48 clean
// absences and no violations at all. A class that forbids half of reciprocal
// space collects absence credit in proportion to how much it forbids, and at
// eps = 0.02 the violations could not pay it back.
//
// sgBaseEps() derives the rate instead: roughly one unexplained peak across the
// whole pattern. On 700 positions that is 0.0014, making each hard violation
// cost 5.3 nats rather than 2.6, and nineteen of them fatal -- which is the
// textbook rule that a single genuine reflection at a systematically absent
// position rules a space group out. On a small pattern with ~50 positions it
// returns 0.02 and reproduces the old behaviour exactly.
const SG_EPS = {
    weak:       0.25,   // below 5% of the local maximum
    ka2:        0.50,   // Ka2 ghost
    ambiguous:  0.50,   // an allowed line sits inside 1.5x the tolerance
    unindexed:  0.15,   // peak the refined cell explains with nothing at all
    hardFloor:  1e-4,   // never claim more certainty than this
    hardCap:    0.05,   // nor less, however few positions there are
};

// How much total evidence a pile of empty forbidden positions is allowed to be
// worth, as a multiple of the decisive threshold. See the saturation note in
// sgScoreFromStats(): absences are the one term that grows without bound with
// how much a class forbids, and they are also the weakest kind of evidence per
// unit, so they are the term that has to saturate.
//
// TWENTY, not ten. The cap has to sit above the largest separation the absences
// can legitimately produce, or it eats real evidence instead of runaway
// evidence. Measured: two clean classes differing by thirteen informative
// absences at p = 0.9 are 26.9 nats apart, which a ceiling of ten times decisive
// (23 nats) compresses to 0.95 -- an overwhelming result reported as a tie,
// because both rows sat deep in the flat part of the curve. At twenty times
// (46 nats) the same pair reads 8.0 nats and the runaway cases are still bounded:
// the PbSO4 I-centred class collects 346 raw nats of absence credit and still
// cannot buy its way past seventy hard violations.
const SG_CLEAN_CAP_MULT = 20;

// tanh() reaches 1.0 EXACTLY in double precision once its argument passes about
// 19, so a pure saturation makes every heavily-restrictive row score identically
// on absences and the ordering the cap was supposed to preserve is lost after
// all -- silently, and only on the rows most likely to be wrong. A vanishing
// linear term keeps the sequence strictly increasing forever: at 1e-3 nats per
// nat it is a tie-break and nothing more, worth one nat against a thousand.
const SG_CLEAN_LEAK = 1e-3;

function sgBaseEps(nPositions, impurityAllowance) {
    const nPos = Math.max(1, nPositions || 0);
    // The user's impurity setting is their own estimate of how many foreign
    // peaks the pattern carries; one is assumed even when they say none.
    const nStray = Math.max(1, Math.floor(impurityAllowance || 0));
    return Math.min(SG_EPS.hardCap, Math.max(SG_EPS.hardFloor, nStray / nPos));
}

function sgViolationEps(v, epsBase) {
    const base = (epsBase !== undefined && epsBase !== null) ? epsBase : 0.02;
    if (v.ka2)       return SG_EPS.ka2;
    if (v.ambiguous) return SG_EPS.ambiguous;
    if (v.weak)      return SG_EPS.weak;
    // With a Wilson scale the taper runs on |E|^2 rather than on raw local
    // height. That is the better variable: it already accounts for the angle and
    // the multiplicity, so a moderate peak where reflections are weak anyway is
    // correctly read as strong evidence, and a tall one at low angle where
    // everything is tall is not over-credited.
    if (v.eSq !== null && v.eSq !== undefined && isFinite(v.eSq)) {
        const t = Math.min(1, Math.max(0, (v.eSq - 0.05) / (0.5 - 0.05)));
        return SG_EPS.weak + (base - SG_EPS.weak) * t;
    }
    if (v.rel === null || v.rel === undefined) return base;   // no heights: judge on position alone
    // Intensity taper between the weak threshold and "unmistakably strong", so a
    // 6%-of-local peak is not treated as identical to a 100% one.
    const t = Math.min(1, Math.max(0, (v.rel - SG_LOW_INTENSITY_FRACTION) / (0.35 - SG_LOW_INTENSITY_FRACTION)));
    return SG_EPS.weak + (base - SG_EPS.weak) * t;
}

// The impurity allowance is a COUNT, not a list, so it has to be spent
// somewhere. Spend it on the LEAST believable unexplained peaks first (weakest
// relative intensity). The old code subtracted it from the raw violation total,
// which let a class buy forgiveness for the strongest peak in the pattern.
//
// UNINDEXED PEAKS ARE SPENT ON LAST. Sorting purely by relative intensity meant
// the allowance was almost always consumed by weak leftovers before it ever
// reached a violation, because a hard violation is by definition not weak. A
// user who sees three foreign lines and sets the allowance to 3 expects those
// lines forgiven; instead the budget went to three faint unindexed peaks and
// every class stayed falsified. A foreign phase contributes a peak the cell
// cannot index OR a peak on a forbidden line, and only the second kind
// falsifies, so the second kind is where the budget has to be able to go.
// Within each kind the weakest still goes first, so the strongest peak in the
// pattern is still the last thing anyone can buy.
function sgApplyImpurityAllowance(stats, allowance) {
    const n = Math.max(0, Math.floor(allowance || 0));
    if (!n) return { violations: stats.violations, nUnindexed: stats.unindexed };

    const items = [];
    stats.violations.forEach((v, i) => items.push({ kind: 'viol', idx: i, rel: v.rel ?? 1 }));
    stats.unindexedRel.forEach((r, i) => items.push({ kind: 'unidx', idx: i, rel: r ?? 1 }));
    const rank = (it) => (it.kind === 'viol' ? 0 : 1);
    items.sort((a, b) => (rank(a) - rank(b)) || (a.rel - b.rel));

    const dropViol = new Set();
    let dropUnidx = 0;
    for (let i = 0; i < Math.min(n, items.length); i++) {
        if (items[i].kind === 'viol') dropViol.add(items[i].idx); else dropUnidx++;
    }
    return {
        violations: stats.violations.filter((_, i) => !dropViol.has(i)),
        nUnindexed: stats.unindexed - dropUnidx,
    };
}

function sgScoreFromStats(stats, pHat, opts) {
    const o = opts || {};
    const p = Math.min(0.98, Math.max(0.02, pHat));
    const kept = sgApplyImpurityAllowance(stats, o.impurityAllowance);

    // Every resolvable position the lattice offers inside the measured window.
    // Each group in range either carries an allowed line or is an informative
    // absence, so the two counts partition it.
    //
    // MEASURED ON THE PARENT FRAME WHEN THERE IS ONE. This count feeds epsBase,
    // which is the rate of unexplained peaks PER POSITION and is meant to be a
    // property of the pattern, not of the hypothesis. Taking it from each row's
    // own refined cell made it drift slightly from row to row -- second order
    // (about 0.07 nats per violation for a fifty-position difference) but enough
    // that "every row is scored under the same assumptions" was only
    // approximately true. opts.nPositions carries the shared count when the
    // caller has one; the row's own is the fallback.
    const nPosRow = (stats.nAllowedInRange || 0) + (stats.nInformative || 0);
    const nPositions = (isFinite(o.nPositions) && o.nPositions > 0) ? o.nPositions : nPosRow;
    const epsBase = sgBaseEps(nPositions, o.impurityAllowance);

    // Lambda re-solved against the reference class this pass is using, if the
    // caller supplied one; otherwise the seed value baked into pDetect().
    const wil = o.wilson || null;
    const lam = (isFinite(o.lambda) && o.lambda > 0) ? o.lambda : null;
    const pOfZ = (z, fallback) => {
        if (wil && lam !== null && z !== null && z !== undefined && isFinite(z)) {
            const v = wil.pFromZ(z, lam);
            if (v !== null) return v;
        }
        return fallback;
    };

    let score = 0;

    // Clean absences. With a Wilson scale each absence carries its own weight,
    // set by how likely that particular reflection was to be seen at all; without
    // one they all share the single global p.
    //
    // The two paths are mixed PER POSITION, not per row. The old gate was
    // `cleanP.length === nClean`, all-or-nothing: a single position where
    // pDetect() returned null (no Lp, no multiplicity, mean intensity zero)
    // dropped that entire row back onto the global p while its neighbours in
    // the same table stayed on Wilson. Measured on a 300-absence row at
    // p = 0.45 that one missing entry was worth +50 nats -- twenty times the
    // decisive threshold -- purely from being scored under a different model
    // than the row above it. Positions that do have a weight now use it, and
    // only the leftovers fall back.
    const cleanP = (stats.cleanP && stats.cleanP.length) ? stats.cleanP : [];
    const cleanZ = (stats.cleanZ && stats.cleanZ.length === cleanP.length) ? stats.cleanZ : null;
    let clean = 0;
    for (let i = 0; i < cleanP.length; i++) {
        const pc = pOfZ(cleanZ ? cleanZ[i] : null, Math.min(0.999, Math.max(0.001, cleanP[i])));
        clean += Math.log((1 - epsBase) / (1 - pc));
    }

    // THE UNIFORM TERM IS SHRUNK BY THE DETECTABLE FRACTION.
    //
    // log(1/(1-p)) is the credit for an empty position ASSUMING a reflection
    // there would have been detectable with probability p. Applied to every
    // position without a per-position weight, that assumption is wrong in a way
    // that always favours the restrictive class: conditioning on "this position
    // is empty" preferentially selects the positions where nothing would have
    // shown up anyway, whose true credit is near zero, and pays them the
    // population average instead. Only about a fraction p-bar of positions are
    // detectable at all, so the honest per-position expectation is smaller by
    // roughly that factor -- (1 - p_undetectable), which is the mean
    // detectability itself.
    //
    // The factor is shared by every row (it depends only on the shared p and,
    // when Wilson is available, on the shared intensity scale), so it rescales
    // the term without touching the ordering it induces.
    const detFrac = (wil && lam !== null && stats.allowedZ && stats.allowedZ.length)
        ? Math.min(1, Math.max(0.02, wil.meanPOf(stats.allowedZ, lam) ?? p))
        : p;
    const nCleanRest = Math.max(0, (stats.nClean || 0) - cleanP.length);
    clean += nCleanRest * detFrac * Math.log((1 - epsBase) / (1 - p));

    // AND THE TOTAL SATURATES.
    //
    // Absences are the only term that grows with how much a hypothesis forbids
    // rather than with what the pattern actually shows, and the whole history of
    // this module is failures of that shape: I-centring outscoring P2_1/a by
    // burying nineteen hard violations under 219 empty positions. Falsification
    // tiering catches that case, but nothing stopped a merely soft-violating
    // over-restrictive class from doing the same thing more quietly.
    //
    // A hard cap would flatten every row above it into a tie, so this is a
    // smooth saturation instead: c*tanh(x/c) is strictly increasing, so the
    // ordering among rows survives intact; it is within a percent of x while x
    // is small compared with c, and it can never exceed c however many
    // positions a class deletes. Absences can still be decisive many times
    // over -- they simply cannot outvote the peaks that are actually there.
    const cleanCap = SG_CLEAN_CAP_MULT * SG_DECISIVE_NATS;
    const cleanRaw = clean;
    clean = cleanCap * Math.tanh(clean / cleanCap) + SG_CLEAN_LEAK * clean;
    score += clean;

    for (const v of kept.violations) {
        const pv = pOfZ(v.zLocal,
            (v.pLocal !== null && v.pLocal !== undefined && isFinite(v.pLocal))
                ? Math.min(0.999, Math.max(0.001, v.pLocal)) : p);
        // A VIOLATION CAN NEVER BE POSITIVE EVIDENCE FOR THE RULE SET IT BREAKS.
        //
        // The ratio eps/p exceeds one whenever the detectability model says a
        // reflection here would have been invisible: p bottoms out at its 0.001
        // floor, eps for a soft violation is 0.25, and the peak that
        // CONTRADICTS the hypothesis is then worth +5.5 nats IN ITS FAVOUR.
        // Measured on a synthetic I-centred pattern: a deliberately mis-set cell
        // collected seventeen such violations worth +51 nats and beat the cell
        // that indexed the pattern exactly, which had none. The correct reading
        // of p < eps is not "this peak supports H" but "the intensity model is
        // wrong about this position" -- a peak is sitting where the model says
        // nothing could be seen, so the model, not the hypothesis, is what the
        // observation bears on. Clamping at zero says exactly that: at best a
        // violation is uninformative, and it is never support.
        score += Math.min(0, Math.log(sgViolationEps(v, epsBase) / pv));
    }
    score += Math.max(0, kept.nUnindexed) * Math.log(SG_EPS.unindexed);

    const nPar = (MC_NPAR[o.system] || 3) + (o.refineZero ? 1 : 0);

    // MODEL-SELECTION CHARGE.
    //
    // The old term was -0.5*k*ln(n_indexed), and its sign ran the wrong way
    // against its own stated intent. BIC's penalty GROWS with sample size, so
    // charging it on n_indexed -- which differs per row, because a restrictive
    // class matches fewer peaks to allowed lines -- hands the restrictive class
    // a discount. Measured: dropping from 200 indexed peaks to 30 is worth
    // +3.8 nats, and 200 to 120 is worth +1.0, against a decisive threshold of
    // 2.3. The intent was the opposite: charge the class whose line list barely
    // constrains the cell.
    //
    // BIC's n is the number of OBSERVATIONS, which is the peak list and is
    // therefore identical for every row -- so it cancels out of every
    // comparison, as it should. What does not cancel is the small-sample
    // correction: AICc's k(k+1)/(n-k-1) blows up as the number of constraining
    // lines approaches the number of free parameters, which is exactly the
    // "absorb the error into my own zero-point" case the old comment described.
    const nObs = Math.max(2, (stats.indexed || 0) + (stats.nViolations || 0) +
                             (stats.unindexed || 0));
    score -= 0.5 * nPar * Math.log(nObs);
    const nCon = Math.max(nPar + 2, stats.indexed || 0);
    score -= nPar * (nPar + 1) / (nCon - nPar - 1);

    return {
        score,
        pHat: p, epsBase, nPositions,
        lambda: lam,
        // What the absences were worth before and after saturation. When the two
        // differ the row's lead is being carried by how much it forbids, which
        // is worth being able to see.
        cleanNats: clean, cleanNatsRaw: cleanRaw, cleanCap,
        cleanCapped: cleanRaw > cleanCap * 0.9,
        nCleanEff: stats.nClean,
        nViolEff: kept.violations.length,
        nHardEff: kept.violations.filter(v => !v.ka2 && !v.weak && !v.ambiguous).length,
        nUnindexedEff: Math.max(0, kept.nUnindexed),
    };
}

// Two-pass estimate of p. Pass 1 uses the most permissive class present, which
// is biased LOW whenever the true lattice is centred (all the systematically
// absent lines sit in the denominator and are never observed). Pass 2
// re-estimates from the class pass 1 favoured and rescores everything against
// that single shared value, so the rows stay directly comparable. One iteration
// suffices in practice and the estimate is clamped either way.
function sgRescoreAll(rows, opts) {
    const o = opts || {};
    const scored = rows.filter(r => !r.error && r.stats);
    if (!scored.length) return rows;

    const estimate = (r) => (r.stats.nAllowedInRange > 0)
        ? r.stats.nAllowedObserved / r.stats.nAllowedInRange : 0.5;

    // THE SHARED POSITION COUNT. epsBase is a property of the pattern, so it is
    // measured once on the unrestricted parent lattice and handed to every row
    // rather than being recomputed on each row's own refined cell.
    let nPosShared = 0;
    if (o.frame && o.frame.centres) {
        for (let g = 0; g < o.frame.centres.length; g++) {
            const qc = o.frame.centres[g];
            if (qc >= o.frame.qLo && qc <= o.frame.qHi) nPosShared++;
        }
    }
    if (!nPosShared) {
        for (const r of scored) {
            const n = (r.stats.nAllowedInRange || 0) + (r.stats.nInformative || 0);
            if (n > nPosShared) nPosShared = n;
        }
    }

    // LAMBDA FOLLOWS P-HAT THROUGH BOTH PASSES.
    //
    // lambda sets the level of the per-position detectability and p-hat sets the
    // level of the uniform one; they are the same physical quantity measured two
    // ways, so calibrating them against different hypotheses makes the clean and
    // violation terms disagree about how detectable the pattern is. lambda was
    // solved once inside sgWilsonContext() against the UNFILTERED lattice -- the
    // no-extinction hypothesis, biased low for exactly the reason pass 1 is --
    // and then never revisited, while p-hat was re-estimated from the winner.
    // Each pass now re-solves lambda on the same reference class, and against
    // that class's own detection rate, that it takes p-hat from.
    const wilson = o.wilson || null;
    const lamFor = (r, target) => {
        if (!wilson || !wilson.solveLambda) return null;
        const zs = r && r.stats && r.stats.allowedZ;
        if (!zs || !zs.length) return wilson.lambda;
        return wilson.solveLambda(zs, target) ?? wilson.lambda;
    };
    const applyAll = (pHat, lambda) => {
        const so = { ...o, nPositions: nPosShared, wilson, lambda };
        for (const r of scored) Object.assign(r, sgScoreFromStats(r.stats, pHat, so));
    };

    // WHICH CLASS IS "THE MOST PERMISSIVE"?
    //
    // It used to be the one with the fewest STATED rules, and rule count is not
    // permissiveness. I-centring is one condition and deletes half of reciprocal
    // space; Pbca is three and deletes far less. On an orthorhombic pattern the
    // old seed therefore handed pass 1 to the F-centred class -- p estimated
    // over the 330 positions F leaves open instead of the 600 the true class
    // leaves open, 0.515 instead of 0.300 -- and since the clean-absence term is
    // n_clean * log(1/(1-p)), that single misestimate moved the leader's score
    // by tens of nats.
    //
    // nAllowedGroups is what the observable merge already computed for exactly
    // this quantity: how many resolvable positions the rule set leaves open in
    // this window. stats.nAllowedInRange is the same count measured on the row's
    // own refined cell, and serves as the fallback.
    const openness = (r) => (isFinite(r.nAllowedGroups) ? r.nAllowedGroups
                                                        : (r.stats.nAllowedInRange || 0));
    const permissive = scored.slice().sort((a, b) => openness(b) - openness(a))[0];
    let pHat = estimate(permissive);
    let lambda = lamFor(permissive, pHat);
    applyAll(pHat, lambda);

    // Pass 2 re-estimates from the winner, and "the winner" has to mean the same
    // thing here as it does in sgRankRows(): falsification first, then score.
    // Taking the top raw score let a class the ranking was about to throw out --
    // one carrying hard violations -- set the p that every surviving row is then
    // scored against.
    const alive = scored.filter(r => (r.nHardEff || 0) === 0);
    const pool = alive.length ? alive : scored;
    const best = pool.slice().sort((a, b) => b.score - a.score)[0];
    const pHat2 = estimate(best);
    const lambda2 = lamFor(best, isFinite(pHat2) ? pHat2 : pHat);
    // Rescore if EITHER level moved. The reference set changes even when the two
    // rates happen to agree, and lambda is solved on the set, not on the rate.
    const pMoved = isFinite(pHat2) && Math.abs(pHat2 - pHat) > 0.01;
    const lMoved = isFinite(lambda2) && isFinite(lambda) &&
                   Math.abs(Math.log(lambda2 / lambda)) > 0.01;
    if (pMoved || lMoved) {
        if (pMoved) pHat = pHat2;
        if (isFinite(lambda2)) lambda = lambda2;
        applyAll(pHat, lambda);
    }
    for (const r of rows) if (!r.error) { r.pHat = pHat; r.lambda = lambda; }
    return rows;
}

// ============================================================================
// PER-CLASS EVALUATION
// ============================================================================
//
// opts.mode:
//   'fixed' - score the parent cell as it stands. Cheapest, and the only mode
//             that reproduces the old stage-1 behaviour.
//   'ls'    - one constrained least-squares refit against the restricted line
//             list. Cheap enough to run on EVERY class, and unlike 'fixed' it
//             does not judge a hypothesis using a cell that was fitted to the
//             very reflections the hypothesis forbids -- which is the whole
//             reason this module exists. This is the default first pass.
//   'mc'    - full Monte-Carlo plus least squares (the expensive shortlist).
//
// The returned `cell` is a normal solution object -- system, volume, errors,
// m20, analysis-ready -- so the caller can drop it into the solutions ledger.
function sgScoreClass(cls, sol, data, state, opts) {
    const o = opts || {};
    const mode = o.mode || (o.mc === false ? 'ls' : 'mc');
    const ctx = sgMakeCtx(data, state);
    const ictx = o.ictx || sgIntensityContext(state.peaks_sorted_by_q);

    const row = {
        label: cls.label, members: cls.members,
        conditions: cls.allConditions || cls.conditions,
        mergedLabels: cls.mergedLabels || [cls.label],
        repSymbol: cls.repSymbol, centering: cls.centering, nRules: cls.nRules,
        // How many resolvable positions this rule set leaves open in the window,
        // measured on the parent frame by sgObservableMerge(). This is the
        // honest measure of permissiveness; nRules is not (see sgRescoreAll).
        nAllowedGroups: cls.nAllowedGroups,
        sig: cls.sig, mode,
        m20: 0, m_all: 0, n20: 0, cell: null, mcGain: 0, error: null,
        score: -Infinity, stats: null,
    };

    try {
        // --- baseline: the parent cell, restricted line list ------------------
        const base = { ...sol };
        let baseEval = null;
        try {
            setSpaceGroupFilter(cls.allowed);
            baseEval = mcEvaluateCell(base, ctx);
        } finally { setSpaceGroupFilter(null); }
        if (!baseEval) { row.error = 'cell generates no allowed lines'; return row; }

        row.m20 = base.m20 || 0;
        row.m_all = base.m_all || 0;
        row.n20 = base.n_20 || 0;
        row.cell = base;
        let moved = false;

        // WHAT DECIDES WHETHER A REFINED CELL IS KEPT.
        //
        // It used to be M20, in both modes, and M20 is not the quantity this
        // table ranks on. Within one class the line list is fixed, so comparing
        // M20 between two cells of the same class is at least legitimate -- but
        // a cell that raises the SCORE while nudging M20 down was discarded, and
        // the score is what decides the row's fate three lines later. The two
        // criteria are not the same: M20 rewards a tight fit to the twenty
        // lowest lines, the score weighs every absence and every violation
        // across the whole pattern.
        //
        //   'ls' accepts unconditionally. The least-squares cell IS the
        //        hypothesis -- the cell fitted to the restricted line list,
        //        which is the entire reason the mode exists. Refusing it
        //        because M20 fell means judging the hypothesis on a cell fitted
        //        to reflections it forbids, which is the bias this module was
        //        written to remove. Only a degenerate solve is rejected.
        //
        //   'mc'  accepts on the score, evaluated against a FIXED reference p
        //        taken from the baseline. Letting each candidate supply its own
        //        p would let the walk improve its apparent score by changing the
        //        yardstick; the shortlist is short, so the extra stats pass is
        //        affordable here in a way it would not be at stage 1.
        //
        // M20 SURVIVES AS THE TIE-BREAK, and it has to. The score measures how
        // well a RULE SET fits, not how well a cell fits: once every peak is
        // indexed and every forbidden position is empty, two cells of the same
        // class score within noise of each other however differently they are
        // refined. Measured on a synthetic I-centred pattern, the parent cell
        // and a cell fitted to 0.1 mA scored 16.15 against 16.16 -- a hundredth
        // of a nat deciding a factor of 1.7 in M20. Inside one class the line
        // list is fixed, so M20 is a legitimate comparison there, and it is the
        // only one of the two that can see the difference. The score leads; M20
        // speaks only when the score is silent.
        const SG_SCORE_TIE_NATS = 0.5;      // well under SG_DECISIVE_NATS
        const beats = (sA, mA, sB, mB) =>
            (sA > sB + SG_SCORE_TIE_NATS) ||
            (Math.abs(sA - sB) <= SG_SCORE_TIE_NATS && (mA || 0) > (mB || 0));
        const statsFor = (cell, preRefl) =>
            sgIndexingStats(cell, data, state, cls.allowed, ictx, preRefl, o.wilson || null);
        const refP = (st) => (st && st.nAllowedInRange > 0)
            ? st.nAllowedObserved / st.nAllowedInRange : 0.5;
        const scoreOf = (st, pRef, nPosRef) => {
            if (!st) return -Infinity;
            try {
                const s = sgScoreFromStats(st, pRef, {
                    system: sol.system,
                    refineZero: o.refineZero,
                    impurityAllowance: o.impurityAllowance,
                    nPositions: nPosRef,
                    wilson: o.wilson || null,
                    lambda: o.wilson ? o.wilson.lambda : null,
                });
                return isFinite(s.score) ? s.score : -Infinity;
            } catch (e) { return -Infinity; }
        };
        let stats = null;

        // --- refinement --------------------------------------------------------
        if (mode === 'ls') {
            let ls = null;
            try {
                setSpaceGroupFilter(cls.allowed);
                ls = mcLeastSquaresPolish(base, data, state, ctx);
                if (ls) { ls.system = sol.system; mcEvaluateCell(ls, ctx); }
            } finally { setSpaceGroupFilter(null); }
            if (ls && isFinite(ls.m20) && isFinite(ls.a) && ls.a > 0) {
                row.mcGain = ls.m20 - row.m20;      // may be negative now, by design
                row.m20 = ls.m20;
                row.m_all = isFinite(ls.m_all) ? ls.m_all : row.m_all;
                row.n20 = ls.n_20 || row.n20;
                row.cell = ls;
                moved = true;
            }
        } else if (mode === 'mc') {
            // The LS polish is run HERE TOO, not only in stage 1. Stage 2 starts
            // over from the parent cell, so a class whose Monte-Carlo walk
            // returns nothing useful used to fall all the way back to the
            // unrefined parent -- ending up with a WORSE cell than the same
            // class had after stage 1, and being ranked against stage-1 rows on
            // that basis. Measured on a synthetic I-centred pattern: the P class
            // came out of stage 2 at M20 79.9 having left stage 1 at 128.7.
            // Offering all three candidates and taking the best by score makes
            // stage 2 monotone in the only sense that matters.
            let ls = null, mc = null;
            try {
                setSpaceGroupFilter(cls.allowed);
                ls = mcLeastSquaresPolish(base, data, state, ctx);
                if (ls) { ls.system = sol.system; mcEvaluateCell(ls, ctx); }
                // monteCarloRefineCell returns null when it cannot beat its
                // starting point, which here is the constrained baseline -- so
                // null simply means "the parent cell was already the best under
                // these rules".
                mc = monteCarloRefineCell(sol, data, state, {
                    iterations: o.iterations ?? 600,
                    restarts: o.restarts ?? 4
                    // The seed is deliberately left at its default: every class
                    // walks the SAME pseudo-random sequence, so a difference
                    // between two rows is a difference between hypotheses and
                    // not between two draws.
                });
            } finally { setSpaceGroupFilter(null); }
            if (mc) mc.system = sol.system;

            const usable = (c) => c && isFinite(c.m20) && isFinite(c.a) && c.a > 0;
            const baseStats = statsFor(base, o.frame ? o.frame.refl : null);
            const pRef = refP(baseStats);
            const nPosRef = baseStats
                ? (baseStats.nAllowedInRange || 0) + (baseStats.nInformative || 0) : 0;
            let bestCell = null, bestStats = baseStats;
            let bestScore = scoreOf(baseStats, pRef, nPosRef);
            let bestM20 = row.m20;
            for (const cand of [ls, mc]) {
                if (!usable(cand)) continue;
                const st = statsFor(cand, null);
                const s = scoreOf(st, pRef, nPosRef);
                if (beats(s, cand.m20, bestScore, bestM20)) {
                    bestScore = s; bestM20 = cand.m20; bestCell = cand; bestStats = st;
                }
            }
            if (bestCell) {
                row.mcGain = bestCell.m20 - row.m20;
                row.m20 = bestCell.m20;
                row.m_all = isFinite(bestCell.m_all) ? bestCell.m_all : row.m_all;
                row.n20 = bestCell.n_20 || row.n20;
                row.cell = bestCell;
                moved = true;
            }
            stats = bestStats;                  // reuse whichever won
        }

        row.zero = row.cell.zero_correction ?? 0;

        // --- how well does the winning cell obey the rules? --------------------
        // When the cell did not move, the parent's line list is still valid, so
        // reuse the frame instead of regenerating it once per class.
        if (!stats) {
            const preRefl = (!moved && o.frame) ? o.frame.refl : null;
            stats = statsFor(row.cell, preRefl);
        }
        if (!stats) { row.error = 'cell generates no lines'; return row; }

        row.stats = stats;
        row.indexed = stats.indexed;
        row.violations = stats.nViolations;
        row.hardViolations = stats.nHard;
        row.unindexed = stats.unindexed;
        row.violatingTth = stats.violatingTth;
        row.violationDetail = stats.violations.slice(0, 8);
        row.nClean = stats.nClean;
        row.nInformative = stats.nInformative;
        row.nLines = stats.nLines;
    } catch (err) {
        row.error = String((err && err.message) || err);
    }
    return row;
}

// The ctx object mcEvaluateCell expects, built from the same data/state pair the
// rest of the MC machinery uses.
function sgMakeCtx(data, state) {
    const n_all = state.peaks_sorted_by_q.length;
    return {
        wavelength: data.wavelength,
        q_max: state.q_max,
        d_min: state.d_min,
        impurity_peaks: data.impurity_peaks,
        peaks_sorted_by_q: state.peaks_sorted_by_q,
        n_20: Math.min(state.N_FOR_M20 || 20, n_all),
        n_all,
        tolFn: (idx) => get_q_tolerance(idx, state.tth_obs_rad, data.wavelength, data.tth_error)
    };
}

// ============================================================================
// RANKING
// ============================================================================
//
// FALSIFICATION FIRST, THEN THE LOG-ODDS SCORE.
//
// A systematic absence is not a statistical tendency. If a space group has an
// a-glide then |F| is EXACTLY zero for h0l with h odd, and a single genuine
// reflection there rules the group out however many other absences hold. The
// likelihood score cannot express that on its own: it multiplies evidence across
// reflections, so a class forbidding a great deal of reciprocal space accrues
// absence credit in proportion to how much it forbids, and with enough of it any
// number of violations can be outweighed. On a real PbSO4 pattern the I-centred
// class did exactly that -- nineteen hard violations, and it still outscored a
// P2_1/a class that violated nothing.
//
// So rows are ranked in two tiers. A row with hard violations the impurity
// allowance does not cover is FALSIFIED and cannot outrank an unfalsified one,
// whatever its score. Within each tier the score decides, so the ordering among
// survivors is still the full likelihood comparison and the falsified rows are
// still ordered least-bad first -- knowing WHICH group the data exclude, and by
// how much, is half the answer.
//
// Only HARD violations falsify. Ka2 ghosts, peaks below the local weak
// threshold, and peaks with an allowed line inside the matching window are all
// graded soft and do not, because none of them is a reliable reflection.
//
// M20 survives as a DISPLAY column and as the last tie-break. It is a figure of
// merit for a CELL, not a criterion for choosing between line lists, and the
// nRules tie-break that used to sit above it has been removed outright: it
// rewarded restrictiveness for its own sake, which the likelihood already
// accounts for wherever the data support it.
const SG_DECISIVE_NATS = 2.3;   // ~10:1 odds; below this the table is a tie

function sgRankRows(rows, impurityAllowance, opts) {
    sgRescoreAll(rows, { ...(opts || {}), impurityAllowance });
    return rows.slice().sort((a, b) => {
        if (a.error && !b.error) return 1;
        if (b.error && !a.error) return -1;
        if (a.error && b.error) return 0;
        // tier 1: falsified or not
        const fa = ((a.nHardEff || 0) > 0) ? 1 : 0;
        const fb = ((b.nHardEff || 0) > 0) ? 1 : 0;
        if (fa !== fb) return fa - fb;
        // tier 2: the log-odds score
        const sa = isFinite(a.score) ? a.score : -Infinity;
        const sb = isFinite(b.score) ? b.score : -Infinity;
        if (Math.abs(sa - sb) > 1e-9) return sb - sa;
        if ((a.nHardEff || 0) !== (b.nHardEff || 0)) return (a.nHardEff || 0) - (b.nHardEff || 0);
        if ((b.nCleanEff || 0) !== (a.nCleanEff || 0)) return (b.nCleanEff || 0) - (a.nCleanEff || 0);
        if (Math.abs((b.m20 || 0) - (a.m20 || 0)) > 1e-6) return (b.m20 || 0) - (a.m20 || 0);
        return (a.members?.[0]?.number || 999) - (b.members?.[0]?.number || 999);
    });
}

// Is the winner actually separated from the runner-up, or is the table a tie?
// Reported to the user instead of silently presenting row 1 as the answer.
// The margin is measured WITHIN the surviving tier. Comparing an unfalsified row
// against a falsified one would report a lead that means nothing: the two are not
// competing on the same question.
// A SECOND COMPARABILITY CONDITION: THE SAME REFINEMENT DEPTH.
//
// Stage 1 gives every class one least-squares solve; stage 2 gives the
// shortlist a full Monte-Carlo walk. An MC row can beat an LS row on compute
// alone -- a better cell for the same hypothesis -- so a margin measured across
// the two is partly a measure of who got refined, not of what the data say. The
// table already marks stage-1 rows, but the margin and the note treated every
// row as comparable.
//
// The margin is therefore measured among rows sharing the LEADER's mode. Once
// stage 2 has run that is the Monte-Carlo set, which is the honest comparison;
// during stage 1 every row is 'ls' and nothing changes. sgMarginInfo() reports
// what was compared so the caller can say so.
function sgComparableTier(ranked) {
    const ok = (ranked || []).filter(r => !r.error && isFinite(r.score));
    if (!ok.length) return { tier: [], restricted: false, mode: null };
    const alive = ok.filter(r => (r.nHardEff || 0) === 0);
    const surviving = alive.length ? alive : ok;
    const mode = surviving[0].mode || null;
    const same = surviving.filter(r => (r.mode || null) === mode);
    const restricted = same.length > 0 && same.length < surviving.length;
    return { tier: same.length ? same : surviving, restricted, mode };
}

function sgMarginInfo(ranked) {
    const { tier, restricted, mode } = sgComparableTier(ranked);
    const margin = tier.length < 2 ? Infinity : tier[0].score - tier[1].score;
    return { margin, mode, nCompared: tier.length, restricted, tier };
}

function sgMargin(ranked) {
    return sgMarginInfo(ranked).margin;
}

// Did anything survive at all? When every class carries hard violations the
// answer is not "the least bad one wins" but "none of these fits", and the caller
// should say so rather than presenting row 1 as a determination.
function sgAnySurvivor(ranked) {
    return (ranked || []).some(r => !r.error && (r.nHardEff || 0) === 0);
}
