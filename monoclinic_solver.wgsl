// monoclinic_solver.wgsl
// 4-Peak Direct Solve + TRUE Figure of Merit (FoM) Validation
//
// Stage 1: 4-peak direct solve (24 permutations).
// Stage 2: Basic boolean filters (singular, volume, params).
// Stage 3: For passing cells, run a "TRUE FoM" filter.
//          This filter calculates the mean *squared* normalized error
//          for the first N peaks and checks it against a threshold.

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
@group(0) @binding(3) var<storage, read> hkl_combos: array<u32>; // [n1,n2,n3,n4]
@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawMonoSolution>;

struct Config { 
    z_offset: u32,
    wavelength: f32,
    tth_error: f32, // degrees
    max_volume: f32,
    max_impurities: u32,
    n_peaks_for_fom: u32, // e.g., 20
    n_hkl_for_fom: u32,   // e.g., 100
    // 16-byte alignment
    pad1: u32,
    pad2: u32
};
@group(0) @binding(6) var<uniform> config: Config;

// === DEBUG BINDINGS ===
@group(0) @binding(7) var<storage, read_write> debug_counter: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> debug_log: array<f32>;

// === TOLERANCE BUFFER ===
@group(0) @binding(9) var<storage, read> q_tolerances: array<f32>;


// === Constants ===
const PI: f32 = 3.1415926535;
const RAD: f32 = PI / 180.0;
const DEG: f32 = 180.0 / PI;
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_Y_WORKGROUPS: u32 = 16383u;  //attention à cette valeur, probleme possible si conflit avec webgpu définition
const MAX_SOLUTIONS: u32 = 20000u;  //réduit à 20k le 16 nov
const MAX_DEBUG_CELLS: u32 = 10u;

// threshold for the *mean squared* normalized error
const FOM_THRESHOLD: f32 = 3.; // 2.25 est 1.5^2, 1.5 times average of delta_q.. si trop grand ça remplit le buffer de mauvaises cellules

// 4! = 24 permutations.
const PERMUTATIONS_4: array<u32, 96> = array<u32, 96>(
    0u, 1u, 2u, 3u, 0u, 1u, 3u, 2u, 0u, 2u, 1u, 3u, 0u, 2u, 3u, 1u, 0u, 3u, 1u, 2u, 0u, 3u, 2u, 1u,
    1u, 0u, 2u, 3u, 1u, 0u, 3u, 2u, 1u, 2u, 0u, 3u, 1u, 2u, 3u, 0u, 1u, 3u, 0u, 2u, 1u, 3u, 2u, 0u,
    2u, 0u, 1u, 3u, 2u, 0u, 3u, 1u, 2u, 1u, 0u, 3u, 2u, 1u, 3u, 0u, 2u, 3u, 0u, 1u, 2u, 3u, 1u, 0u,
    3u, 0u, 1u, 2u, 3u, 0u, 2u, 1u, 3u, 1u, 0u, 2u, 3u, 1u, 2u, 0u, 3u, 2u, 0u, 1u, 3u, 2u, 1u, 0u
);

// === Helper Functions ===

// 4x4 Matrix Solve (Gaussian Elimination)
fn solve4x4(A_in: Mat4x4, b_in: Vec4) -> Vec4 {
    var M: Mat4x4 = A_in; 
    var v: Vec4 = b_in;
    let n: u32 = 4u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        // Find pivot
        var max_row = i;
        var max_val = abs(M[i][i]);
        for (var k = i + 1u; k < n; k = k + 1u) {
            let val = abs(M[k][i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        if (max_row != i) {
            // Swap rows
            let temp_row = M[i];
            M[i] = M[max_row];
            M[max_row] = temp_row;
            let temp_v = v[i];
            v[i] = v[max_row];
            v[max_row] = temp_v;
        }

        let pivot: f32 = M[i][i]; 
        
        // --- SINGULAR MATRIX CHECK ---
        if (abs(pivot) < 1e-10) { 
            return Vec4(0.0, 0.0, 0.0, 0.0); // Singular
        }

        // Elimination
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let fac: f32 = M[r][i] / pivot;
            M[r] = M[r] - (fac * M[i]);
            v[r] = v[r] - (fac * v[i]);
        }
    }

    // Back substitution
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

// Extract cell from 4 params (A=h^2, B=k^2, C=l^2, D=hl)
// This IS the "basic boolean filter"
fn extractCell(params: Vec4) -> RawMonoSolution {
    let A: f32 = params[0]; 
    let B: f32 = params[1]; 
    let C: f32 = params[2]; 
    let D: f32 = params[3]; 

    // Filter 1: REMOVED explicit A,B,C > 0 check.
    
    // Filter 1: Discriminant for beta (cos^2 beta < 1)
    let det_AC = 4.0 * A * C;
    let D_sq = D * D;
    if (D_sq >= det_AC) { 
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0);
    }

    // Filter 2: Calculate beta
    let cosBeta_calc = -D / (2.0 * sqrt(A*C));
    if (abs(cosBeta_calc) >= 1.0) {
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0);
    }

    // Filter 3: Enforce beta definition (>= 90)
    var beta_calc = acos(cosBeta_calc) * DEG;
    if (beta_calc < 90.0) {
        beta_calc = 180.0 - beta_calc;
    }
    
    // Filter 4: Beta plausibility range
    if (beta_calc < 89.0 || beta_calc > 150.0) {
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0);
    }

    let sinBetaSq = 1.0 - cosBeta_calc * cosBeta_calc; 
    if (sinBetaSq <= 1e-6) {
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0);
    }

    let a_val = 1.0 / sqrt(A * sinBetaSq);
    let b_val = 1.0 / sqrt(B);
    let c_val = 1.0 / sqrt(C * sinBetaSq);

    // Filter 5: Lattice parameter plausibility
    if (a_val < 2.0 || a_val > 50.0 || b_val < 2.0 || b_val > 50.0 || c_val < 2.0 || c_val > 50.0) { 
        return RawMonoSolution(0.0, 0.0, 0.0, 0.0); 
    }
    
    // Filter 6: Volume range
    let volume = a_val * b_val * c_val * sqrt(sinBetaSq);
    if (volume < 20.0 || volume > config.max_volume) { 
         return RawMonoSolution(0.0, 0.0, 0.0, 0.0);
    }

    return RawMonoSolution(a_val, b_val, c_val, beta_calc);
}


// calculates the MEAN SQUARED NORMALIZED ERROR
fn validate_fom_avg_diff(A: f32, B: f32, C: f32, D: f32) -> f32 {

    var sum_of_squared_normalized_diffs: f32 = 0.0;
    
    // Loop over the first N peaks (e.g., N=20)
    for (var i: u32 = 0u; i < config.n_peaks_for_fom; i = i + 1u) {
        
        let q_obs_val = q_obs[i];
        let tol = q_tolerances[i]; 

        var min_diff: f32 = 1e10; // A very large number
        
        // Inner loop: check all HKLs to find the best match
        for (var j: u32 = 0u; j < config.n_hkl_for_fom; j = j + 1u) {
            let h = hkl_basis[j * 4u + 0u];
            let k = hkl_basis[j * 4u + 1u];
            let l = hkl_basis[j * 4u + 2u];
            
            // Monoclinic q_calc formula
            let q_calc = A*h*h + B*k*k + C*l*l + D*h*l;
            
            let diff = abs(q_obs_val - q_calc);
            
            if (diff < min_diff) {
                min_diff = diff;
            }
        }
        
        // Add the *square* of the normalized difference for this peak.
        let normalized_diff = min_diff / tol; 
        sum_of_squared_normalized_diffs += (normalized_diff * normalized_diff);
    }
    
    // Return the mean squared normalized difference
    let avg_squared_norm_error = sum_of_squared_normalized_diffs / f32(config.n_peaks_for_fom);

    // Fail if average squared error is too high
    if (avg_squared_norm_error > FOM_THRESHOLD) {
        return 999.0; // Return a value guaranteed to fail
    }

    return avg_squared_norm_error; // Pass
}


// === Main Kernel (4-Peak) ===
@compute @workgroup_size(8, WORKGROUP_SIZE_Y, 1)
fn main_4p(
    @builtin(global_invocation_id) global_id: vec3<u32>
) { 
    if (atomicLoad(&solution_counter) >= MAX_SOLUTIONS) {
        return;
    }

    // 1. Get indices for this thread
    let peak_combo_idx: u32 = global_id.x;
    let hkl_threads_per_z_slice = MAX_Y_WORKGROUPS * WORKGROUP_SIZE_Y;
    let z_index = global_id.z + config.z_offset;
    let hkl_combo_idx: u32 = z_index * hkl_threads_per_z_slice + global_id.y;

    // Bounds check
    let num_peak_combos = arrayLength(&peak_combos) / 4u;
    let num_hkl_combos = arrayLength(&hkl_combos) / 4u;

    if (peak_combo_idx >= num_peak_combos || hkl_combo_idx >= num_hkl_combos) {
        return;
    }

    // 2. Get q_obs vector for this combo (4 peaks)
    let p_offset = peak_combo_idx * 4u;
    let q_base = array<f32, 4>(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]],
        q_obs[peak_combos[p_offset + 3u]]
    );

    // 3. Get HKLs and build M_hkl matrix (4 rows, 4 cols)
    let h_offset = hkl_combo_idx * 4u;
    var M_hkl_rows: array<vec4<f32>, 4>; 
    var hkl_indices: array<u32, 4>;
    
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let hkl_idx = hkl_combos[h_offset + i];
        hkl_indices[i] = hkl_idx; // Store for debug log
        
        let h = hkl_basis[hkl_idx * 4u + 0u];
        let k = hkl_basis[hkl_idx * 4u + 1u];
        let l = hkl_basis[hkl_idx * 4u + 2u];
        
        M_hkl_rows[i] = Vec4(h*h, k*k, l*l, h*l);
    }
    
    let M_hkl = Mat4x4(M_hkl_rows[0], M_hkl_rows[1], M_hkl_rows[2], M_hkl_rows[3]);
    
    // 4. Loop over all 24 permutations of indices [0,1,2,3]
    for(var p_idx: u32 = 0u; p_idx < 24u; p_idx = p_idx + 1u) {
        
        let perm_offset = p_idx * 4u;
        
        // 5. Build permuted q_vec
        let q_perm = Vec4(
             q_base[PERMUTATIONS_4[perm_offset + 0u]], 
             q_base[PERMUTATIONS_4[perm_offset + 1u]], 
             q_base[PERMUTATIONS_4[perm_offset + 2u]],
             q_base[PERMUTATIONS_4[perm_offset + 3u]]
        );
         
        // 6. STAGE 1: Solve the 4x4 system
        let fit_params = solve4x4(M_hkl, q_perm);
         
        // 7. STAGE 2: Basic Boolean Filters
        let cell = extractCell(fit_params);
         
        // 8. STAGE 3: Strong "True FoM" Filter
        if (cell.a > 0.0) { // Passed basic filters
            
            let A_sol = fit_params[0]; 
            let B_sol = fit_params[1]; 
            let C_sol = fit_params[2]; 
            let D_sol = fit_params[3];
            
            // avg_squared_norm_error
            let avg_squared_norm_error = validate_fom_avg_diff(A_sol, B_sol, C_sol, D_sol);
            
            // Compare against the FOM_THRESHOLD
            if (avg_squared_norm_error < FOM_THRESHOLD) { 
            
                // 9. Store result (passed ALL tests)
                let idx = atomicAdd(&solution_counter, 1u);
                if (idx < MAX_SOLUTIONS) {
                    results_list[idx] = cell;
                }
                
                // 10. Write to debug log
                let debug_idx = atomicAdd(&debug_counter, 1u);
                if (debug_idx < MAX_DEBUG_CELLS) {
                    let debug_cell_offset = debug_idx * 25u; // 5 floats * 5 "peaks"
                    
                    for(var k_log: u32 = 0u; k_log < 4u; k_log = k_log + 1u) {
                        let log_offset = debug_cell_offset + (k_log * 5u);
                        let hkl_idx = hkl_indices[k_log];
                        let q_obs_val = q_perm[k_log];
                        let q_calc = (A_sol * M_hkl_rows[k_log][0] +
                                      B_sol * M_hkl_rows[k_log][1] +
                                      C_sol * M_hkl_rows[k_log][2] +
                                      D_sol * M_hkl_rows[k_log][3]);
                    
                        debug_log[log_offset + 0u] = hkl_basis[hkl_idx * 4u + 0u]; // h
                        debug_log[log_offset + 1u] = hkl_basis[hkl_idx * 4u + 1u]; // k
                        debug_log[log_offset + 2u] = hkl_basis[hkl_idx * 4u + 2u]; // l
                        debug_log[log_offset + 3u] = 1.0 / sqrt(q_obs_val); // d_obs
                        debug_log[log_offset + 4u] = 1.0 / sqrt(q_calc);    // d_calc
                    }
                }
                
                // Found a good permutation, stop checking perms for this combo
                break; 
            }
        }
         
    } // p_idx loop
}