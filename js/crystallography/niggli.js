// js/crystallography/niggli.js
// Niggli reduction and equivalent cells.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// --- SPACE GROUP / NIGGLI FUNCTIONS ---
// These are all part of the crystallography scripts; 
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
