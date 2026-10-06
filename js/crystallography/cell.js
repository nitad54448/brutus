// js/crystallography/cell.js
// Cells from fitted parameters, volumes, geometry/solution keys and lattice equivalence.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

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
const getCellGeometryKey = (cell) => {
    const P = 4; // Increased from 2 to 4 digits to prevent aggressive deduplication
    const std = standardizeCell(cell);
    switch(std.system) {
        case 'cubic': 
            return `${std.system}_${std.a.toFixed(P)}`;
        case 'tetragonal': 
        case 'hexagonal': 
            return `${std.system}_${std.a.toFixed(P)}_${std.c.toFixed(P)}`;
        case 'orthorhombic': 
            // Numeric comparator: a bare .sort() orders numbers as STRINGS
            // ("10.5" < "3.1"). Same rule as the monoclinic branch below.
            return `${std.system}_${[std.a,std.b,std.c].sort((x, y) => x - y).map(p => p.toFixed(P)).join('_')}`;
        case 'monoclinic': 
            const ac = [std.a, std.c].sort((x, y) => x - y).map(p => p.toFixed(P)).join('_');
            // Updated beta to use P instead of hardcoded 2
            return `${std.system}_${ac}_${std.b.toFixed(P)}_${std.beta.toFixed(P)}`; 
        case 'triclinic': 
            // Keep lengths paired with their angles. Volume and angles alone
            // do not identify a triclinic lattice. The final sieve handles
            // equivalent bases; this hot-path key must not merge distinct cells.
            return `${std.system}_${[std.a, std.b, std.c, std.alpha, std.beta, std.gamma]
                .map(v => v.toFixed(P)).join('_')}`;
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
    


// Fixed-zero and fitted-zero cells are different models even when their
// rounded dimensions happen to coincide.
const hasRefinedZero = cell => cell.zero_correction !== undefined && cell.zero_correction !== null;
const getSolutionKey = cell => {
    const geometry = getCellGeometryKey(cell);
    return geometry ? `${geometry}_${hasRefinedZero(cell) ? 'zero-fit' : 'zero-fixed'}` : null;
};
// Conservative lattice-equivalence check for the final sieve. Reduce each
// cell once, then explicitly seek a unimodular basis change. No centering is
// inferred from an absence analysis. Near-equal volumes are only a prefilter.
// Tolerances: 0.2% in each edge, 0.15 degrees in each inter-edge angle.
const makeLatticeComparison = cell => {
    try {
        const normalized = { ...cell, b: cell.b ?? cell.a, c: cell.c ?? cell.a,
            alpha: cell.alpha ?? 90, beta: cell.beta ?? 90,
            gamma: cell.gamma ?? (cell.system === 'hexagonal' ? 120 : 90) };
        const raw = metricFromCell(normalized);
        if (!choleskyDecomposition(raw)) return null;
        const reduced = reduceToNiggliCell(normalized);
        const G = reduced && reduced.converged ? reduced.metric : raw;
        const dot = (u, v) => u.reduce((s, x, i) =>
            s + x * v.reduce((t, y, j) => t + G[i][j] * y, 0), 0);
        const vectors = [];
        // Includes signed axis permutations and adjacent reduced-cell boundary
        // settings (a+b, a-b, etc.). Every accepted mapping has determinant +/-1.
        for (let h = -1; h <= 1; h++) for (let k = -1; k <= 1; k++)
            for (let l = -1; l <= 1; l++) {
                if (!h && !k && !l) continue;
                const v = [h, k, l];
                vectors.push({ v, length: Math.sqrt(dot(v, v)) });
            }
        return { G, dot, vectors, volume: Math.sqrt(determinant3x3(G)) };
    } catch (_) { return null; } // Uncertain equivalence retains the candidate.
};
const equivalentLattices = (a, b) => {
    if (!a || !b || Math.abs(a.volume - b.volume) > 0.01 * Math.min(a.volume, b.volume)) return false;
    const match = (source, target) => {
        const lengths = target.G.map((row, i) => Math.sqrt(row[i]));
        const candidates = lengths.map(length => source.vectors.filter(x =>
            Math.abs(x.length - length) <= 0.002 * Math.min(x.length, length)));
        if (candidates.some(list => !list.length)) return false;
        const angle = cosine => Math.acos(Math.max(-1, Math.min(1, cosine))) * DEG;
        const angleMatches = (u, v, i, j) => Math.abs(
            angle(source.dot(u.v, v.v) / (u.length * v.length)) -
            angle(target.G[i][j] / (lengths[i] * lengths[j]))) <= 0.15;
        for (const u of candidates[0]) for (const v of candidates[1]) {
            if (!angleMatches(u, v, 0, 1)) continue;
            for (const w of candidates[2]) {
                if (Math.abs(determinant3x3([u.v, v.v, w.v])) !== 1) continue;
                if (angleMatches(u, w, 0, 2) && angleMatches(v, w, 1, 2)) return true;
            }
        }
        return false;
    };
    return match(a, b) || match(b, a);
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
