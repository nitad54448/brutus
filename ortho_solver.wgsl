// ortho_solver.wgsl
// 3-Peak Direct Solve + Combinadics + Optimized FoM + Fail-Fast + 16-byte Config

// === Structs ===
struct RawOrthoSolution {
    a: f32, b: f32, c: f32,
    pad1: f32 
}

// === Type Aliases ===
alias Vec3 = vec3<f32>;

// === Bindings ===
@group(0) @binding(0) var<storage, read> q_obs: array<f32>;

// PRECOMPUTED hkl products, NOT raw indices. Each element is
//     vec4(h*h, k*k, l*l, 0)
// packed on the JS side (see HKL_PACKERS in main_app.js). The layout is the
// same 16 bytes per reflection the raw [h,k,l,pad] form used, so nothing about
// buffer sizing changes -- but the FoM inner loop no longer recomputes h*h,
// k*k and l*l for every candidate cell, and one 16-byte vector load replaces
// three scalar loads. The products are small integers (max 144 at h<=12) and
// so are exact in f32: this is bit-for-bit what the shader used to compute.
@group(0) @binding(1) var<storage, read> hkl_basis: array<vec4<f32>>;

@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k]

@group(0) @binding(3) var<storage, read> binomial_table: array<u32>; 

@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawOrthoSolution>;

// --- ALIGNED CONFIG STRUCT (16-byte alignment) ---
struct Config { 
    // Chunk 1 (U32): z_offset, max_impurities, n_peaks_for_fom, n_hkl_for_fom
    u_params1: vec4<u32>, 
    // Chunk 2 (U32): n_basis_total, total_hkl_combos, max_solutions, pad
    u_params2: vec4<u32>,
    // Chunk 3 (F32): wavelength, tth_error, max_volume, fom_threshold
    f_params: vec4<f32>,
    // Chunk 4 (F32): min_axis, max_axis, min_volume, pad
    // These were hard-coded as 2.0 / 50.0 / 20.0 in extractCell*, in all three
    // shaders, while main_app.js kept its own copy of the same three numbers.
    // Two sources for one limit is how limits drift apart.
    f_params2: vec4<f32>
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
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_FOM_PEAKS: u32 = 32u; // unified with mono/tri (errors[] is already array<f32,32>)

// Orthorhombic Constants (K=3)
const K_VALUE: u32 = 3u; 
const BINOMIAL_STRIDE: u32 = 4u; // Columns 0..3

// 3! = 6 permutations
const PERMUTATIONS_3: array<u32, 18> = array<u32, 18>(
    0u, 1u, 2u, 0u, 2u, 1u,
    1u, 0u, 2u, 1u, 2u, 0u,
    2u, 0u, 1u, 2u, 1u, 0u
);

// === Helper Functions ===

// Solve A x = b where A is given by its ROWS r0, r1, r2.
//
// --- FIX: this used to return the transpose of the answer ----------------
// The caller built the matrix with Mat3x3(row0, row1, row2). WGSL's matrix
// constructor takes COLUMN vectors, so that expression is already A^T, and
// A[i] indexes a column. The cross products below are therefore
//     cross(r1, r2)/det, cross(r2, r0)/det, cross(r0, r1)/det
// which are the COLUMNS of A^-1 (the old name `A_inv_t` said as much). Dotting
// each of them with b returns A^-T b, not A^-1 b -- the solver was answering a
// different linear system than the one the indexer posed.
//
// It did not blow up because a wrong fit is simply rejected by the FoM, and
// because the orthorhombic basis is a full 0..12 box: the "transposed" hkl
// triple is often itself in the basis, so the right cell still turned up by a
// side door. That is why the search looked like it worked.
//
// Measured by enumerating every basis triple for a known cell and asking which
// observed peak triples each version can reach (scoring an axis-permuted cell
// as a hit, since for orthorhombic it is the same cell):
//     basis 60,  10 peaks   old  92/120    new 120/120
//     basis 120, 12 peaks   old 148/220    new 220/220
//     basis 120, 12 peaks, 35% of lines missing
//                           old  98/120    new 120/120
// So roughly a quarter to a third of the peak triples were unusable, and the
// ones that did work had far fewer independent routes to the answer -- which
// is what makes a search brittle against noise and missing lines.
//
// Taking the rows explicitly instead of a Mat3x3 removes the trap at the
// source: there is no longer a column-vs-row convention to get wrong.
fn solve3x3_rows(r0: Vec3, r1: Vec3, r2: Vec3, b: Vec3) -> Vec3 {
    let x12 = cross(r1, r2);
    let detA = dot(r0, x12);
    if (abs(detA) < 1e-10) { return Vec3(0.0, 0.0, 0.0); }

    let invDet = 1.0 / detA;
    // Columns of A^-1.
    let c0 = x12 * invDet;
    let c1 = cross(r2, r0) * invDet;
    let c2 = cross(r0, r1) * invDet;

    // x = A^-1 b = c0*b.x + c1*b.y + c2*b.z
    return c0 * b.x + c1 * b.y + c2 * b.z;
}

fn extractCellOrtho(params: Vec3) -> RawOrthoSolution {
    let A: f32 = params[0]; let B: f32 = params[1]; let C: f32 = params[2]; 

    if (A <= 1e-12 || B <= 1e-12 || C <= 1e-12) { return RawOrthoSolution(0.0, 0.0, 0.0, 0.0); }

    let a_val = 1.0 / sqrt(A);
    let b_val = 1.0 / sqrt(B);
    let c_val = 1.0 / sqrt(C);

    let min_ax = config.f_params2.x;
    let max_ax = config.f_params2.y;
    if (a_val < min_ax || a_val > max_ax || b_val < min_ax || b_val > max_ax || c_val < min_ax || c_val > max_ax) { 
        return RawOrthoSolution(0.0, 0.0, 0.0, 0.0); 
    }
    
    let volume = a_val * b_val * c_val;
    
    if (volume < config.f_params2.z || volume > config.f_params.z) { 
         return RawOrthoSolution(0.0, 0.0, 0.0, 0.0);
    }
    return RawOrthoSolution(a_val, b_val, c_val, 0.0);
}

// === Combinatorial Number System (K=3) ===
fn get_combinadic_indices(linear_index: u32, n_max: u32) -> array<u32, 3> {
    var m = linear_index;
    var out: array<u32, 3>;
    var v = n_max - 1u; 
    
    for (var k_idx: u32 = K_VALUE; k_idx > 0u; k_idx = k_idx - 1u) {
        loop {
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

// === Optimized FoM (Absolute Difference) ===
fn validate_fom_avg_diff(A: f32, B: f32, C: f32) -> f32 {
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

    let abc = Vec3(A, B, C);
    let n_basis = config.u_params1.w;
    
    // --- OPTIMIZATION 1: Fail-Fast (No Impurities) ---
    if (max_imp == 0u) {
        var sum_abs_error: f32 = 0.0;
        let max_allowed_total = config.f_params.w * f32(n_peaks_to_check);

        for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
            let q_obs_val = q_obs[i];
            let tol = q_tolerances[i]; 
            var min_diff: f32 = 1e10; 
            
            for (var j: u32 = 0u; j < n_basis; j = j + 1u) {
                // q_calc = A*h^2 + B*k^2 + C*l^2, with the squares precomputed.
                let q_calc = dot(abc, hkl_basis[j].xyz);
                let diff = abs(q_obs_val - q_calc);
                if (diff < min_diff) { min_diff = diff; }
            }
            let norm = min_diff / tol;  
            // Changed to Abs Difference (FoM = Mean |diff|/tol)
            sum_abs_error += norm;  
            
            if (sum_abs_error > max_allowed_total) { return 999.0; }
        }
        return sum_abs_error / f32(n_peaks_to_check);
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
            let q_calc = dot(abc, hkl_basis[j].xyz);
            let diff = abs(q_obs_val - q_calc);
            if (diff < min_diff) { min_diff = diff; }
        }
        let norm = min_diff / tol; 
        
        // CRITICAL FIX: Uncommented this line. Use Abs Difference.
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
            if ((sum_all - top_sum) > max_allowed_total) { return 999.0; }
        }
    }

    var sum_of_valid_errors: f32 = 0.0;

    for (var i: u32 = 0u; i < count_to_sum; i = i + 1u) {
        var min_val = errors[i];
        var min_idx = i;
        for (var j: u32 = i + 1u; j < n_peaks_to_check; j = j + 1u) {
            if (errors[j] < min_val) { min_val = errors[j]; min_idx = j; }
        }
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
fn main_3p(
    @builtin(global_invocation_id) global_id: vec3<u32>
) { 
    if (atomicLoad(&solution_counter) >= config.u_params2.z) { return; }

    // 1. Calculate Indices
    let peak_combo_idx: u32 = global_id.x;
    // z_offset is .x of u_params1
    let hkl_linear_idx: u32 = config.u_params1.x + global_id.y;

    // 2. Bounds Checks
    let num_peak_combos = arrayLength(&peak_combos) / 3u;
    if (peak_combo_idx >= num_peak_combos) { return; }
    
    if (hkl_linear_idx >= config.u_params2.y) { return; }

    // 3. Generate HKL Indices
    let hkl_indices = get_combinadic_indices(hkl_linear_idx, config.u_params2.x);

    // 4. Build M rows. Each basis entry already holds (h^2, k^2, l^2, 0), so a
    //    row IS the .xyz of one vector load.
    let row0 = hkl_basis[hkl_indices[0]].xyz;
    let row1 = hkl_basis[hkl_indices[1]].xyz;
    let row2 = hkl_basis[hkl_indices[2]].xyz;

    // 5. Get q_obs
    let p_offset = peak_combo_idx * 3u;
    let q_base = array<f32, 3>(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]]
    );

    // 6. Loop 6 Permutations
    for(var p_idx: u32 = 0u; p_idx < 6u; p_idx = p_idx + 1u) {
        let perm_offset = p_idx * 3u;
        let q_perm = Vec3(
             q_base[PERMUTATIONS_3[perm_offset + 0u]], 
             q_base[PERMUTATIONS_3[perm_offset + 1u]], 
             q_base[PERMUTATIONS_3[perm_offset + 2u]]
        );
         
        let fit_params = solve3x3_rows(row0, row1, row2, q_perm);
        let cell = extractCellOrtho(fit_params);

        if (cell.a > 0.0) { 
            let A_sol = fit_params[0]; let B_sol = fit_params[1]; let C_sol = fit_params[2]; 
            
            let avg_err = validate_fom_avg_diff(A_sol, B_sol, C_sol);
            
            if (avg_err < config.f_params.w) { 
                let idx = atomicAdd(&solution_counter, 1u);
                if (idx < config.u_params2.z) {
                    results_list[idx] = cell;
                }
                break; 
            }
        }
    } 
}
