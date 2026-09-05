// monoclinic_solver.wgsl
// 4-Peak Direct Solve + Combinadics + Optimized FoM (Abs Diff) + Fail-Fast

// === Structs ===
struct RawMonoSolution {
    a: f32,
    b: f32,
    c: f32,
    beta: f32,
}

// === Type Aliases ===
alias Vec4 = vec4<f32>;
alias Mat4x4 = mat4x4<f32>; 

// === Bindings ===
@group(0) @binding(0) var<storage, read> q_obs: array<f32>;
// PRECOMPUTED hkl products, NOT raw indices. Each element is
//     vec4(h*h, k*k, l*l, h*l)
// packed on the JS side (see HKL_PACKERS in main_app.js) -- which is exactly
// one row of the 4x4 system, and exactly the dot product the FoM needs. Same
// 16 bytes per reflection as the old [h,k,l,pad] layout, so buffer sizing is
// unchanged. The products are small integers (|h|,|k|,|l| <= 12) and exact in
// f32, so this is bit-for-bit what the shader used to compute per candidate.
@group(0) @binding(1) var<storage, read> hkl_basis: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k,l]

// Pascal's Triangle Table for Combinadics
@group(0) @binding(3) var<storage, read> binomial_table: array<u32>; 

// Slot 0 is the solution count, as before. Slots 1-3 are run DIAGNOSTICS, and
// they exist because a run that finds nothing currently says only that: no
// solutions, no reason, no hint which of Max Volume / 2-theta Error / FoM
// threshold was the one that shut the search out.
//
// They are min/max rather than counts ON PURPOSE. Counting rejected trials
// looks like the obvious design and does not survive contact with the numbers:
// total trials are C(n_hkl,3) x C(n_peaks,3) x 6, which is 9.4e8 at a
// 300-reflection basis but 3.5e10 at 1000 and 9.0e11 at the 2954 maximum --
// past u32 by a factor of 210. A wrapped counter reports confident nonsense,
// which is worse than reporting nothing. A min and a max cannot overflow.
//
// A range is also the more useful answer. "4.2M cells rejected on volume" says
// the cap fired; "candidates ran 2100-7800 A^3" says what to raise it to.
//
//   [0] solution count                                     atomicAdd
//   [1] most peaks any candidate kept inside the error
//       budget before the FoM fail-fast gave up (0..32)    atomicMax, init 0
//   [2] smallest cell volume, A^3, among cells that passed
//       the AXIS test -- recorded BEFORE the volume gate   atomicMin, init MAX
//   [3] largest such volume                                atomicMax, init 0
//   [4..7] reserved
//
// The writes sit only on already-filtered paths: [2]/[3] fire after the axis
// test, [1] once per FoM call. Nothing touches an atomic on the hot reject path
// where most trials die.
@group(0) @binding(4) var<storage, read_write> solution_counter: array<atomic<u32>, 8>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawMonoSolution>;


struct Config { 
    u_params1: vec4<u32>, // Indices 0-3
    u_params2: vec4<u32>, // Indices 4-7
    f_params: vec4<f32>,  // Indices 8-11
    // Chunk 4 (F32): min_axis, max_axis, min_volume, pad
    // These were hard-coded as 2.0 / 50.0 / 20.0 in extractCell*, in all three
    // shaders, while main_app.js kept its own copy of the same three numbers.
    // Two sources for one limit is how limits drift apart.
    f_params2: vec4<f32>  // Indices 12-15
};


@group(0) @binding(6) var<uniform> config: Config;

@group(0) @binding(7) var<storage, read> q_tolerances: array<f32>;

// NOTE: bindings 7 (debug_counter) and 8 (debug_log) are GONE, and
// q_tolerances moved from 9 to 7. Nothing ever wrote to debug_log on the hot
// path and nothing ever read it back, but they still counted against
// maxStorageBuffersPerShaderStage -- whose DEFAULT in WebGPU is 8. The old
// layout needed 9 storage buffers, so createComputePipeline failed outright on
// every adapter that reports exactly 8 (common on integrated and mobile GPUs).
// The layout now needs 7.


// === Constants ===
const PI: f32 = 3.1415926535;
const DEG: f32 = 180.0 / PI;
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_Y_WORKGROUPS: u32 = 16383u; 
const MAX_FOM_PEAKS: u32 = 32u; 

// Monoclinic Constants (K=4)
const K_VALUE: u32 = 4u; 
const BINOMIAL_STRIDE: u32 = 5u; // Columns 0..4

// 4! = 24 permutations.
const PERMUTATIONS_4: array<u32, 96> = array<u32, 96>(
    0u, 1u, 2u, 3u, 0u, 1u, 3u, 2u, 0u, 2u, 1u, 3u, 0u, 2u, 3u, 1u, 0u, 3u, 1u, 2u, 0u, 3u, 2u, 1u,
    1u, 0u, 2u, 3u, 1u, 0u, 3u, 2u, 1u, 2u, 0u, 3u, 1u, 2u, 3u, 0u, 1u, 3u, 0u, 2u, 1u, 3u, 2u, 0u,
    2u, 0u, 1u, 3u, 2u, 0u, 3u, 1u, 2u, 1u, 0u, 3u, 2u, 1u, 3u, 0u, 2u, 3u, 0u, 1u, 2u, 3u, 1u, 0u,
    3u, 0u, 1u, 2u, 3u, 0u, 2u, 1u, 3u, 1u, 0u, 2u, 3u, 1u, 2u, 0u, 3u, 2u, 0u, 1u, 3u, 2u, 1u, 0u
);

// === Helper Functions ===

// --- FACTOR ONCE / SUBSTITUTE MANY ----------------------------------------
//
// solve4x4() used to run a complete Gaussian elimination on every call, and
// main_4p calls it 24 times per hkl combination with THE SAME MATRIX -- only
// the right-hand side changes from one peak permutation to the next. The
// elimination is therefore repeated 24 times to produce the same 6 multipliers
// and the same 4 pivot choices, every time.
//
// Split exactly as triclinic_solver.wgsl already does: factor4x4() runs the
// elimination once and records what it did (the upper triangle U, the six
// multipliers in elimination order, and the row swap chosen at each step);
// substitute4x4() replays that on one RHS and back-substitutes.
//
// This is BIT-IDENTICAL to the old code, not merely equivalent. The pivot
// search reads only M, and `fac = M[r][i] / pivot` depends only on M, so the
// numbers and their order of operations are unchanged -- the RHS updates were
// always just along for the ride. (Note the contrast with the triclinic split,
// which deliberately DID change behaviour by adding the pivoting that shader
// was missing. Monoclinic already pivoted; nothing about the search changes
// here, it just stops doing the same work 24 times.)
//
// Second, smaller win: a singular M is now detected once and the whole
// combination is abandoned, instead of 24 eliminations each running to the
// same near-zero pivot and returning a zero vector for extractCell to reject.
//
// On the matrix convention: Mat4x4 is mat4x4<f32>, whose constructor takes
// COLUMN vectors, and main_4p builds it from four hkl ROWS. Both functions
// below index M[i] as row i and M[i][j] as (row i, col j) -- i.e. they walk the
// WGSL columns as if they were rows -- so the two transposes cancel, exactly as
// the old solve4x4 did. Do not "fix" one without the other.

// Returns false only if the matrix is genuinely singular (no usable pivot in
// some column), in which case every permutation would have yielded a zero
// vector anyway.
fn factor4x4(A: Mat4x4, U: ptr<function, Mat4x4>, facs: ptr<function, array<f32, 6>>,
             piv: ptr<function, array<u32, 4>>) -> bool {
    (*U) = A;
    let n: u32 = 4u;
    var fi: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        var max_row = i;
        var max_val = abs((*U)[i][i]);
        for (var k = i + 1u; k < n; k = k + 1u) {
            let val = abs((*U)[k][i]);
            if (val > max_val) { max_val = val; max_row = k; }
        }
        (*piv)[i] = max_row;
        if (max_row != i) {
            let temp_row = (*U)[i]; (*U)[i] = (*U)[max_row]; (*U)[max_row] = temp_row;
        }
        let pivot: f32 = (*U)[i][i];
        if (abs(pivot) < 1e-10) { return false; }
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let fac: f32 = (*U)[r][i] / pivot;
            (*facs)[fi] = fac;
            fi = fi + 1u;
            (*U)[r] = (*U)[r] - (fac * (*U)[i]);
        }
    }
    return true;
}

// Apply the stored row swaps and multipliers to one RHS, then back-substitute.
// The swap for step i must be replayed BEFORE that step's eliminations, in the
// same order factor4x4 performed them.
fn substitute4x4(U: ptr<function, Mat4x4>, facs: ptr<function, array<f32, 6>>,
                 piv: ptr<function, array<u32, 4>>, b_in: Vec4) -> Vec4 {
    var v: Vec4 = b_in;
    let n: u32 = 4u;
    var fi: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let p = (*piv)[i];
        if (p != i) { let t = v[i]; v[i] = v[p]; v[p] = t; }
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            v[r] = v[r] - (*facs)[fi] * v[i];
            fi = fi + 1u;
        }
    }
    var x: Vec4;
    for (var i_s: i32 = 3; i_s >= 0; i_s = i_s - 1) {
        let i: u32 = u32(i_s);
        var s: f32 = v[i];
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            s = s - (*U)[i][j] * x[j];
        }
        x[i] = s / (*U)[i][i];
    }
    return x;
}

fn extractCell(params: Vec4) -> RawMonoSolution {
    let A: f32 = params[0]; let B: f32 = params[1]; 
    let C: f32 = params[2]; let D: f32 = params[3]; 

    if (A <= 1e-12 || B <= 1e-12 || C <= 1e-12) { return RawMonoSolution(0.0, 0.0, 0.0, 0.0); }

    let det_AC = 4.0 * A * C;
    let D_sq = D * D;
    if (D_sq >= det_AC) { return RawMonoSolution(0.0, 0.0, 0.0, 0.0); }

    let cosBeta_calc = -D / (2.0 * sqrt(A*C));
    if (abs(cosBeta_calc) >= 1.0) { return RawMonoSolution(0.0, 0.0, 0.0, 0.0); }

    var beta_calc = acos(cosBeta_calc) * DEG;
    if (beta_calc < 90.0) { beta_calc = 180.0 - beta_calc; }
    if (beta_calc < 89.0 || beta_calc > 150.0) { return RawMonoSolution(0.0, 0.0, 0.0, 0.0); }

    let sinBetaSq = 1.0 - cosBeta_calc * cosBeta_calc; 
    if (sinBetaSq <= 1e-6) { return RawMonoSolution(0.0, 0.0, 0.0, 0.0); }

    let a_val = 1.0 / sqrt(A * sinBetaSq);
    let b_val = 1.0 / sqrt(B);
    let c_val = 1.0 / sqrt(C * sinBetaSq);

    let min_ax = config.f_params2.x;
    let max_ax = config.f_params2.y;
    if (a_val < min_ax || a_val > max_ax || b_val < min_ax || b_val > max_ax || c_val < min_ax || c_val > max_ax) { 
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0); 
    }
    
    let volume = a_val * b_val * c_val * sqrt(sinBetaSq);

    // Diagnostics: volume range of everything that cleared the axis test, taken
    // BEFORE the volume gate below. If a run finds nothing and this range lies
    // entirely above config.f_params.z, the Max Volume setting is the reason,
    // and the range says what to raise it to.
    let vol_diag = u32(clamp(volume, 0.0, 4.0e9));
    atomicMin(&solution_counter[2], vol_diag);
    atomicMax(&solution_counter[3], vol_diag);

    if (volume < config.f_params2.z || volume > config.f_params.z) { 
         return RawMonoSolution(0.0, 0.0, 0.0, 0.0);
    }

    return RawMonoSolution(a_val, b_val, c_val, beta_calc);
}


// === Combinatorial Number System (K=4) ===
fn get_combinadic_indices(linear_index: u32, n_max: u32) -> array<u32, 4> {
    var m = linear_index;
    var out: array<u32, 4>;
    var v = n_max - 1u; 
    
    for (var k_idx: u32 = K_VALUE; k_idx > 0u; k_idx = k_idx - 1u) {
        loop {
            // Look up Pascal Triangle from buffer
            let binom = binomial_table[v * BINOMIAL_STRIDE + k_idx];
            if (binom <= m) {
                out[k_idx - 1u] = v;
                m = m - binom;
                v = v - 1u;
                break;
            }
            if (v == 0u) { break; }
            v = v - 1u;
        }
    }
    return out;
}

// === Optimized FoM with Fail-Fast & Fast Sort (Using Absolute Difference) ===
fn validate_fom_avg_diff(A: f32, B: f32, C: f32, D: f32) -> f32 {
    let n_peaks_to_check = min(config.u_params1.z, MAX_FOM_PEAKS);

    // --- FIX: unsigned underflow guard ---------------------------------
    // count_to_sum was computed as `n_peaks_to_check - config.u_params1.y`
    // with BOTH operands u32. If max_impurities >= n_peaks_for_fom that
    // wraps to ~4.29e9 and the selection-sort loop below runs essentially
    // forever -> GPU hang / device lost. At exactly equal it produced a
    // 0/0 NaN average instead. The UI clamps the impurity field to max=3,
    // but only on 'blur', and the value is read raw with parseInt at run
    // time, so nothing downstream actually enforces it.
    // Clamp here so the shader is safe regardless of what JS sends.
    if (n_peaks_to_check == 0u) { return 999.0; }
    let max_imp = min(config.u_params1.y, n_peaks_to_check - 1u);

    let abcd = Vec4(A, B, C, D);
    let n_basis = config.u_params1.w;

    // --- OPTIMIZATION 1: Skip Sorting if No Impurities ---
    if (max_imp == 0u) {
        var sum_abs_error: f32 = 0.0;
        var peaks_ok: u32 = 0u;
        
        // --- OPTIMIZATION 2: Fail-Fast Threshold ---
        let max_allowed_total = config.f_params.w * f32(n_peaks_to_check);

        for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
            let q_obs_val = q_obs[i];
            let tol = q_tolerances[i]; 
            var min_diff: f32 = 1e10; 
            
            for (var j: u32 = 0u; j < n_basis; j = j + 1u) {
                // q_calc = A*h^2 + B*k^2 + C*l^2 + D*h*l, products precomputed.
                let q_calc = dot(abcd, hkl_basis[j]);
                let diff = abs(q_obs_val - q_calc);
                
                if (diff < min_diff) { min_diff = diff; }
            }
            
            let norm = min_diff / tol;
            // Using absolute difference
            sum_abs_error += norm;

            // Fail-Fast Check
            // Diagnostics: how many peaks the best candidate kept inside the
            // error budget. The FoM VALUE cannot serve here -- the fail-fast
            // returns 999.0 the moment the sum exceeds budget, so it is 999 for
            // everything that misses and says nothing about how badly. The peak
            // count does: 9 of 10 means nudge the tolerance, 2 of 10 means the
            // cell is nowhere near.
            if (sum_abs_error > max_allowed_total) {
                atomicMax(&solution_counter[1], peaks_ok);
                return 999.0;
            }
            peaks_ok = peaks_ok + 1u;
        }
        atomicMax(&solution_counter[1], peaks_ok);
        
        let avg = sum_abs_error / f32(n_peaks_to_check);
        return avg;
    }

    // --- PATH B: With Impurities (Requires Sorting) ---
    let count_to_sum = n_peaks_to_check - max_imp; // guarded above, cannot underflow
    let max_allowed_total = config.f_params.w * f32(count_to_sum);

    // Fail-fast for the impurity path. Path A has had one for a while; this
    // one had to score all 32 peaks before it could reject anything, which is
    // the expensive half of the run whenever impurity_peaks > 0.
    //
    // The bound: the final score drops the max_imp LARGEST errors, so after
    // seeing i+1 peaks,
    //     (sum of all errors so far) - (sum of the max_imp largest so far)
    // is a lower bound on the final sum. It never decreases as peaks are added
    // (a new error either misses the top-k and adds to the sum, or joins the
    // top-k and evicts something no larger), and unseen peaks contribute >= 0.
    // So exceeding the budget here means exceeding it at the end -- this can
    // only reject candidates the full computation would have rejected too.
    //
    // `top` holds the largest max_imp errors, zero-initialised so that the
    // first max_imp insertions fill it naturally. Capped at 4 entries; a larger
    // impurity count simply disables the bound rather than reading OOB.
    let use_fast_bound = (max_imp <= 4u);
    var top: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var top_sum: f32 = 0.0;
    var sum_all: f32 = 0.0;

    var errors: array<f32, 32>; 
    for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
        let q_obs_val = q_obs[i];
        let tol = q_tolerances[i]; 
        var min_diff: f32 = 1e10; 
        
        for (var j: u32 = 0u; j < n_basis; j = j + 1u) {
            let q_calc = dot(abcd, hkl_basis[j]);
            let diff = abs(q_obs_val - q_calc);
            if (diff < min_diff) { min_diff = diff; }
        }
        let norm = min_diff / tol; 
        // Using absolute difference
        errors[i] = norm;

        if (use_fast_bound) {
            // Evict the smallest of the current top-k if this error beats it.
            var min_i: u32 = 0u;
            var min_v: f32 = top[0];
            for (var t: u32 = 1u; t < max_imp; t = t + 1u) {
                if (top[t] < min_v) { min_v = top[t]; min_i = t; }
            }
            if (norm > min_v) { top_sum = top_sum - min_v + norm; top[min_i] = norm; }
            sum_all += norm;
            if ((sum_all - top_sum) > max_allowed_total) {
                atomicMax(&solution_counter[1], i);
                return 999.0;
            }
        }
    }
    atomicMax(&solution_counter[1], n_peaks_to_check);

    // --- OPTIMIZATION 3: Partial Selection Sort ---
    var sum_of_valid_errors: f32 = 0.0;

    for (var i: u32 = 0u; i < count_to_sum; i = i + 1u) {
        var min_val = errors[i];
        var min_idx = i;
        // Find min in remaining array
        for (var j: u32 = i + 1u; j < n_peaks_to_check; j = j + 1u) {
            if (errors[j] < min_val) {
                min_val = errors[j];
                min_idx = j;
            }
        }
        // Swap
        let temp = errors[i];
        errors[i] = min_val;
        errors[min_idx] = temp;
        
        sum_of_valid_errors += min_val;
    }
    
    let avg = sum_of_valid_errors / f32(count_to_sum);
    if (avg > config.f_params.w) { return 999.0; }
    return avg; 
}


// === Main Kernel ===
@compute @workgroup_size(8, WORKGROUP_SIZE_Y, 1)
fn main_4p(
    @builtin(global_invocation_id) global_id: vec3<u32>
) { 
    // Fix: config.u_params2.z (max_solutions)
    if (atomicLoad(&solution_counter[0]) >= config.u_params2.z) { return; }

    // 1. Calculate Indices
    let peak_combo_idx: u32 = global_id.x;
    let hkl_linear_idx: u32 = config.u_params1.x + global_id.y;

    // 2. Bounds Checks
    let num_peak_combos = arrayLength(&peak_combos) / 4u;
    if (peak_combo_idx >= num_peak_combos) { return; }
    if (hkl_linear_idx >= config.u_params2.y) { return; }

    // 3. Generate HKL Indices (Combinadics K=4)
    let hkl_indices = get_combinadic_indices(hkl_linear_idx, config.u_params2.x);

    // 4. Build M Matrix. Each basis entry already holds (h^2, k^2, l^2, h*l),
    //    which IS one row, so this is four vector loads and nothing else.
    //
    //    On the constructor convention: Mat4x4(v0..v3) takes COLUMN vectors, so
    //    M_hkl is the transpose of the matrix these rows describe. That is NOT
    //    a bug here -- factor4x4/substitute4x4 index M[i] as a row and M[i][j]
    //    as (row i, col j), i.e. they walk the WGSL columns as if they were rows,
    //    and the two transposes cancel exactly. (ortho_solver.wgsl used the
    //    same constructor with a cofactor solver that did NOT cancel it, which
    //    is the transpose bug fixed there. Leaving this note so the two are not
    //    "made consistent" in the wrong direction later.)
    let M_hkl = Mat4x4(
        hkl_basis[hkl_indices[0]],
        hkl_basis[hkl_indices[1]],
        hkl_basis[hkl_indices[2]],
        hkl_basis[hkl_indices[3]]
    );

    // 5. Get q_obs
    let p_offset = peak_combo_idx * 4u;
    let q_base = array<f32, 4>(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]],
        q_obs[peak_combos[p_offset + 3u]]
    );

    // 5b. Factor M ONCE. M does not depend on the permutation -- only the RHS
    // does -- so the elimination that used to run 24 times now runs once.
    // If M is singular, every permutation would have produced a zero vector and
    // been rejected by extractCell, so the whole combination can be abandoned.
    var U_lu: Mat4x4;
    var lu_facs: array<f32, 6>;
    var lu_piv: array<u32, 4>;
    if (!factor4x4(M_hkl, &U_lu, &lu_facs, &lu_piv)) { return; }

    // 6. Loop 24 Permutations
    for(var p_idx: u32 = 0u; p_idx < 24u; p_idx = p_idx + 1u) {
        let perm_offset = p_idx * 4u;
        let q_perm = Vec4(
             q_base[PERMUTATIONS_4[perm_offset + 0u]], 
             q_base[PERMUTATIONS_4[perm_offset + 1u]], 
             q_base[PERMUTATIONS_4[perm_offset + 2u]], 
             q_base[PERMUTATIONS_4[perm_offset + 3u]]
        );
         
        let fit_params = substitute4x4(&U_lu, &lu_facs, &lu_piv, q_perm);
        let cell = extractCell(fit_params);
         
        if (cell.a > 0.0) { 
            let A_sol = fit_params[0]; let B_sol = fit_params[1]; 
            let C_sol = fit_params[2]; let D_sol = fit_params[3];
            
            // Call the Optimized Function
            let avg_err = validate_fom_avg_diff(A_sol, B_sol, C_sol, D_sol);
            
            if (avg_err < config.f_params.w) { 
                let idx = atomicAdd(&solution_counter[0], 1u);
                if (idx < config.u_params2.z) {
                    results_list[idx] = cell;
                }
                // Debug log removed (was: writes to debug_log[] on every accepted cell,
                // never read back on the JS side). Same cleanup as ortho had.
                break; // Break permutation loop
            }
        }
    } 
}