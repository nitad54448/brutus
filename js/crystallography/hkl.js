// js/crystallography/hkl.js
// Reflection lists: hkl generation, the space-group filter hook, Q(hkl) and LS design rows.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

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

// --- LATTICE LINE FILTER: R lattices in hexagonal axes (9 Oct 2026) --------
// A hexagonal cell may carry `lattice: 'R'`: a rhombohedral lattice described
// on its hexagonal triple cell. Two thirds of the hexagonal REFLECTIONS are
// then absent; a powder LINE merges two -3m stars and survives if either is
// allowed, so about half of the calculated lines vanish (51 -> 25 for
// corundum to 75 deg 2-theta). Counting the absent ones made M(20) about two
// times too low, so R cells ranked below lower-symmetry sub-cells.
// With the flag set, generateHKL_for_analysis emits only the R lines, and so
// does every consumer that takes the cell: refinement and M(20)
// (refineAndTestSolution), the swap search, Refine MC, Swap hkl, the chart
// ticks and the report's hkl table.
//
// The obverse condition is -h+k+l = 3n, the reverse one h-k+l = 3n. The
// generator emits one representative per 6/mmm star (h >= k >= 0, l >= 0),
// and a 6/mmm star is two -3m stars, (h,k,l) and (k,h,l): the line exists if
// either is allowed, i.e. -h+k+l = 3n OR h-k+l = 3n. This also makes the
// obverse and reverse settings the same line list, as they must be for powder
// positions.
//
// The space-group analysis must NOT see this filter: it detects R centring
// by finding the R-forbidden lines empty, so it needs them generated. It
// strips the flag (withoutLattice) before generating; Space Group MC strips
// it from its parent cell for the same reason.
const R_LATTICE_LINE = (h, k, l) =>
    ((((-h + k + l) % 3) + 3) % 3 === 0) || ((((h - k + l) % 3) + 3) % 3 === 0);
function latticeLineFilter(params) {
    return (params && params.lattice === 'R' && params.system === 'hexagonal') ? R_LATTICE_LINE : null;
}
// A shallow copy without the lattice flag (or the cell itself if it has none).
function withoutLattice(cell) {
    if (!cell || cell.lattice === undefined) return cell;
    const out = { ...cell };
    delete out.lattice;
    return out;
}
// HKL generator... 
function generateHKL_for_analysis(params, lambda, maxTth, mode = 'full') {
    const { a, b: b_in, c: c_in, alpha: alpha_in, beta: beta_in, gamma: gamma_in, system } = params;
    const b = b_in ?? a; const c = c_in ?? a;
    const alpha = alpha_in ?? 90; const beta = beta_in ?? 90;
    const gamma = gamma_in ?? (system === 'hexagonal' ? 120 : 90);

    const reflections = [];
    const latticeFilter = latticeLineFilter(params);
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
        // Lattice (R) condition, same placement and same reason as above.
        if (latticeFilter !== null && !latticeFilter(h, k, l)) return;

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
            const last = uniqueReflections[uniqueReflections.length - 1];
            if (Math.abs(reflections[i].tth - last.tth) > tolerance) {
                uniqueReflections.push(reflections[i]);
            } else {
                // Same line, different reflection: cubic 221 and 300 (N = 9),
                // 410 and 322 (N = 17), tetragonal 500 and 430 ... The line
                // keeps ONE representative, chosen by generation order, and
                // that one may be the reflection a space group forbids while
                // the other is allowed. Record the merged ones so the absence
                // test can ask about the whole line (countViolations); nothing
                // else reads this, and the line list itself is unchanged.
                const r = reflections[i];
                (last.coincident || (last.coincident = [])).push({ h: r.h, k: r.k, l: r.l });
            }
        }
    }
    return uniqueReflections;
}
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
