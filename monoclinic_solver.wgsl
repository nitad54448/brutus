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
@group(0) @binding(1) var<storage, read> hkl_basis: array<f32>; // [h,k,l,pad]
@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k,l]

// Pascal's Triangle Table for Combinadics
@group(0) @binding(3) var<storage, read> binomial_table: array<u32>; 

@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawMonoSolution>;


struct Config { 
    u_params1: vec4<u32>, // Indices 0-3
    u_params2: vec4<u32>, // Indices 4-7
    f_params: vec4<f32>   // Indices 8-11
};


@group(0) @binding(6) var<uniform> config: Config;

@group(0) @binding(7) var<storage, read_write> debug_counter: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> debug_log: array<f32>;
@group(0) @binding(9) var<storage, read> q_tolerances: array<f32>;


// === Constants ===
const PI: f32 = 3.1415926535;
const DEG: f32 = 180.0 / PI;
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_Y_WORKGROUPS: u32 = 16383u; 
const MAX_DEBUG_CELLS: u32 = 10u;
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

fn solve4x4(A_in: Mat4x4, b_in: Vec4) -> Vec4 {
    var M: Mat4x4 = A_in; 
    var v: Vec4 = b_in;
    let n: u32 = 4u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        var max_row = i;
        var max_val = abs(M[i][i]);
        for (var k = i + 1u; k < n; k = k + 1u) {
            let val = abs(M[k][i]);
            if (val > max_val) { max_val = val; max_row = k; }
        }
        if (max_row != i) {
            let temp_row = M[i]; M[i] = M[max_row]; M[max_row] = temp_row;
            let temp_v = v[i]; v[i] = v[max_row]; v[max_row] = temp_v;
        }
        let pivot: f32 = M[i][i]; 
        if (abs(pivot) < 1e-10) { return Vec4(0.0, 0.0, 0.0, 0.0); }
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let fac: f32 = M[r][i] / pivot;
            M[r] = M[r] - (fac * M[i]);
            v[r] = v[r] - (fac * v[i]);
        }
    }
    var x: Vec4;
    for (var i_s: i32 = 3; i_s >= 0; i_s = i_s - 1) {
        let i: u32 = u32(i_s);
        var s: f32 = v[i];
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            s = s - M[i][j] * x[j];
        }
        x[i] = s / M[i][i];
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

    if (a_val < 2.0 || a_val > 50.0 || b_val < 2.0 || b_val > 50.0 || c_val < 2.0 || c_val > 50.0) { 
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0); 
    }
    
    let volume = a_val * b_val * c_val * sqrt(sinBetaSq);
    if (volume < 20.0 || volume > config.f_params.z) { 
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

    // --- OPTIMIZATION 1: Skip Sorting if No Impurities ---
    if (config.u_params1.y == 0u) {
        var sum_abs_error: f32 = 0.0;
        
        // --- OPTIMIZATION 2: Fail-Fast Threshold ---
        let max_allowed_total = config.f_params.w * f32(n_peaks_to_check);

        for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
            let q_obs_val = q_obs[i];
            let tol = q_tolerances[i]; 
            var min_diff: f32 = 1e10; 
            
            for (var j: u32 = 0u; j < config.u_params1.w; j = j + 1u) {
                let h = hkl_basis[j * 4u + 0u];
                let k = hkl_basis[j * 4u + 1u];
                let l = hkl_basis[j * 4u + 2u];
                
                let q_calc = A*h*h + B*k*k + C*l*l + D*h*l;
                let diff = abs(q_obs_val - q_calc);
                
                if (diff < min_diff) { min_diff = diff; }
            }
            
            let norm = min_diff / tol;
            // Using absolute difference
            sum_abs_error += norm;

            // Fail-Fast Check
            if (sum_abs_error > max_allowed_total) { return 999.0; }
        }
        
        let avg = sum_abs_error / f32(n_peaks_to_check);
        return avg;
    }

    // --- PATH B: With Impurities (Requires Sorting) ---
    var errors: array<f32, 32>; 
    for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
        let q_obs_val = q_obs[i];
        let tol = q_tolerances[i]; 
        var min_diff: f32 = 1e10; 
        
        for (var j: u32 = 0u; j < config.u_params1.w; j = j + 1u) {
            let h = hkl_basis[j * 4u + 0u];
            let k = hkl_basis[j * 4u + 1u];
            let l = hkl_basis[j * 4u + 2u];
            let q_calc = A*h*h + B*k*k + C*l*l + D*h*l;
            
            let diff = abs(q_obs_val - q_calc);
            if (diff < min_diff) { min_diff = diff; }
        }
        let norm = min_diff / tol; 
        // Using absolute difference
        errors[i] = norm;
    }

    // --- OPTIMIZATION 3: Partial Selection Sort ---
    let count_to_sum = n_peaks_to_check - config.u_params1.y;
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
    if (atomicLoad(&solution_counter) >= config.u_params2.z) { return; }

    // 1. Calculate Indices
    let peak_combo_idx: u32 = global_id.x;
    let hkl_linear_idx: u32 = config.u_params1.x + global_id.y;

    // 2. Bounds Checks
    let num_peak_combos = arrayLength(&peak_combos) / 4u;
    if (peak_combo_idx >= num_peak_combos) { return; }
    if (hkl_linear_idx >= config.u_params2.y) { return; }

    // 3. Generate HKL Indices (Combinadics K=4)
    let hkl_indices = get_combinadic_indices(hkl_linear_idx, config.u_params2.x);

    // 4. Build M Matrix
    var M_hkl_rows: array<vec4<f32>, 4>; 
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let hkl_idx = hkl_indices[i];
        let h = hkl_basis[hkl_idx * 4u + 0u];
        let k = hkl_basis[hkl_idx * 4u + 1u];
        let l = hkl_basis[hkl_idx * 4u + 2u];
        M_hkl_rows[i] = Vec4(h*h, k*k, l*l, h*l);
    }
    let M_hkl = Mat4x4(M_hkl_rows[0], M_hkl_rows[1], M_hkl_rows[2], M_hkl_rows[3]);

    // 5. Get q_obs
    let p_offset = peak_combo_idx * 4u;
    let q_base = array<f32, 4>(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]],
        q_obs[peak_combos[p_offset + 3u]]
    );
    
    // 6. Loop 24 Permutations
    for(var p_idx: u32 = 0u; p_idx < 24u; p_idx = p_idx + 1u) {
        let perm_offset = p_idx * 4u;
        let q_perm = Vec4(
             q_base[PERMUTATIONS_4[perm_offset + 0u]], 
             q_base[PERMUTATIONS_4[perm_offset + 1u]], 
             q_base[PERMUTATIONS_4[perm_offset + 2u]], 
             q_base[PERMUTATIONS_4[perm_offset + 3u]]
        );
         
        let fit_params = solve4x4(M_hkl, q_perm);
        let cell = extractCell(fit_params);
         
        if (cell.a > 0.0) { 
            let A_sol = fit_params[0]; let B_sol = fit_params[1]; 
            let C_sol = fit_params[2]; let D_sol = fit_params[3];
            
            // Call the Optimized Function
            let avg_err = validate_fom_avg_diff(A_sol, B_sol, C_sol, D_sol);
            
            if (avg_err < config.f_params.w) { 
                let idx = atomicAdd(&solution_counter, 1u);
                if (idx < config.u_params2.z) {
                    results_list[idx] = cell;
                }
                
                // Debug log logic (Unchanged)
                let debug_idx = atomicAdd(&debug_counter, 1u);
                if (debug_idx < MAX_DEBUG_CELLS) {
                    let debug_cell_offset = debug_idx * 25u; 
                    for(var k_log: u32 = 0u; k_log < 4u; k_log = k_log + 1u) {
                        let log_offset = debug_cell_offset + (k_log * 5u);
                        let hkl_idx = hkl_indices[k_log];
                        let q_obs_val = q_perm[k_log];
                        let q_calc = (A_sol * M_hkl_rows[k_log][0] + B_sol * M_hkl_rows[k_log][1] +
                                      C_sol * M_hkl_rows[k_log][2] + D_sol * M_hkl_rows[k_log][3]);
                        debug_log[log_offset + 0u] = hkl_basis[hkl_idx * 4u + 0u]; 
                        debug_log[log_offset + 1u] = hkl_basis[hkl_idx * 4u + 1u]; 
                        debug_log[log_offset + 2u] = hkl_basis[hkl_idx * 4u + 2u]; 
                        debug_log[log_offset + 3u] = 1.0 / sqrt(q_obs_val); 
                        debug_log[log_offset + 4u] = 1.0 / sqrt(q_calc);
                    }
                }
                break; // Break permutation loop
            }
        }
    } 
}