// ortho_solver.wgsl
// 3-Peak Direct Solve + TRUE Figure of Merit (FoM) Validation
//
// Stage 1: 3-peak direct solve (6 permutations).
// Stage 2: Basic boolean filters (singular, volume, params).
// Stage 3: For passing cells, run a "TRUE FoM" filter.For now impurities is not used
//

// === Structs ===
struct RawOrthoSolution {
    a: f32,
    b: f32,
    c: f32,
}

// === Type Aliases ===
alias Vec3 = vec3<f32>;
alias Mat3x3 = mat3x3<f32>; 

// === Bindings ===
@group(0) @binding(0) var<storage, read> q_obs: array<f32>;
@group(0) @binding(1) var<storage, read> hkl_basis: array<f32>; // [h,k,l,pad]
@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k]
@group(0) @binding(3) var<storage, read> hkl_combos: array<u32>; // [n1,n2,n3]
@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawOrthoSolution>;

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
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_Y_WORKGROUPS: u32 = 16383u;  //check webGPU in the engine.js
const MAX_SOLUTIONS: u32 = 20000u; //16 nov 
const MAX_DEBUG_CELLS: u32 = 10u;
const MAX_FOM_PEAKS: u32 = 32u; //impurities, discard in the FOM calculations, 18 nov

// threshold for the *mean squared* normalized error
const FOM_THRESHOLD: f32 = 3.;

// 3! = 6 permutations.
const PERMUTATIONS_3: array<u32, 18> = array<u32, 18>(
    0u, 1u, 2u,
    0u, 2u, 1u,
    1u, 0u, 2u,
    1u, 2u, 0u,
    2u, 0u, 1u,
    2u, 1u, 0u
);

// === Helper Functions ===

// 3x3 Matrix Solve (Cramer's Rule for stability)
fn solve3x3(A: Mat3x3, b: Vec3) -> Vec3 {
    let detA = determinant(A);
    
    // --- SINGULAR MATRIX CHECK ---
    if (abs(detA) < 1e-10) { 
        return Vec3(0.0, 0.0, 0.0); // Singular
    }

    let invDet = 1.0 / detA;

    var A_inv_t: Mat3x3; // Transpose of the inverse
    A_inv_t[0] = (cross(A[1], A[2])) * invDet;
    A_inv_t[1] = (cross(A[2], A[0])) * invDet;
    A_inv_t[2] = (cross(A[0], A[1])) * invDet;
    
    // x = (A_inv_t * b)
    return Vec3(
        dot(A_inv_t[0], b),
        dot(A_inv_t[1], b),
        dot(A_inv_t[2], b)
    );
}

// Extract cell from 3 params (A=h^2, B=k^2, C=l^2)
fn extractCellOrtho(params: Vec3) -> RawOrthoSolution {
    let A: f32 = params[0]; 
    let B: f32 = params[1]; 
    let C: f32 = params[2]; 

    // Filter 1: All params must be positive
    if (A <= 1e-6 || B <= 1e-6 || C <= 1e-6) {
        return RawOrthoSolution(0.0, 0.0, 0.0);
    }

    let a_val = 1.0 / sqrt(A);
    let b_val = 1.0 / sqrt(B);
    let c_val = 1.0 / sqrt(C);

    // Filter 2: Lattice parameter plausibility
    if (a_val < 2.0 || a_val > 50.0 || b_val < 2.0 || b_val > 50.0 || c_val < 2.0 || c_val > 50.0) { 
        return RawOrthoSolution(0.0, 0.0, 0.0); 
    }
    
    // Filter 3: Volume range
    let volume = a_val * b_val * c_val;
    if (volume < 20.0 || volume > config.max_volume) { 
         return RawOrthoSolution(0.0, 0.0, 0.0);
    }

    return RawOrthoSolution(a_val, b_val, c_val);
}


// calculates the MEAN SQUARED NORMALIZED ERROR (Excluding Impurities)
fn validate_fom_avg_diff(A: f32, B: f32, C: f32) -> f32 {

    // 1. Local buffer for sorting
    var errors: array<f32, 32>;
    
    let n_peaks_to_check = min(config.n_peaks_for_fom, MAX_FOM_PEAKS);

    // 2. Calculate Error for every peak
    for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
        
        let q_obs_val = q_obs[i];
        let tol = q_tolerances[i]; 

        var min_diff: f32 = 1e10; 
        
        // Inner loop: check all HKLs
        for (var j: u32 = 0u; j < config.n_hkl_for_fom; j = j + 1u) {
            let h = hkl_basis[j * 4u + 0u];
            let k = hkl_basis[j * 4u + 1u];
            let l = hkl_basis[j * 4u + 2u];
            
            // Orthorhombic q_calc formula
            let q_calc = A*h*h + B*k*k + C*l*l;
            
            let diff = abs(q_obs_val - q_calc);
            
            if (diff < min_diff) {
                min_diff = diff;
            }
        }
        
        let normalized_diff = min_diff / tol; 
        errors[i] = (normalized_diff * normalized_diff);
    }
    
    // 3. Sort errors (Ascending)
    for (var i: u32 = 0u; i < n_peaks_to_check; i++) {
        for (var j: u32 = 0u; j < n_peaks_to_check - 1u - i; j++) {
            if (errors[j] > errors[j+1]) {
                let temp = errors[j];
                errors[j] = errors[j+1];
                errors[j+1] = temp;
            }
        }
    }

    // 4. Determine truncation limit
    var count_to_sum = n_peaks_to_check;
    if (config.max_impurities > 0u && config.max_impurities < n_peaks_to_check) {
        count_to_sum = n_peaks_to_check - config.max_impurities;
    }

    // 5. Sum best peaks
    var sum_of_valid_errors: f32 = 0.0;
    for (var k: u32 = 0u; k < count_to_sum; k++) {
        sum_of_valid_errors += errors[k];
    }
    
    let avg_squared_norm_error = sum_of_valid_errors / f32(count_to_sum);

    if (avg_squared_norm_error > FOM_THRESHOLD) {
        return 999.0; 
    }

    return avg_squared_norm_error; 
}


// === Main Kernel (3-Peak) ===
@compute @workgroup_size(8, WORKGROUP_SIZE_Y, 1)
fn main_3p(
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
    let num_peak_combos = arrayLength(&peak_combos) / 3u;
    let num_hkl_combos = arrayLength(&hkl_combos) / 3u;

    if (peak_combo_idx >= num_peak_combos || hkl_combo_idx >= num_hkl_combos) {
        return;
    }

    // 2. Get q_obs vector for this combo (3 peaks)
    let p_offset = peak_combo_idx * 3u;
    let q_base = array<f32, 3>(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]]
    );

    // 3. Get HKLs and build M_hkl matrix (3 rows, 3 cols)
    let h_offset = hkl_combo_idx * 3u;
    var M_hkl_rows: array<vec3<f32>, 3>; 
    var hkl_indices: array<u32, 3>;
    
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let hkl_idx = hkl_combos[h_offset + i];
        hkl_indices[i] = hkl_idx; // Store for debug log
        
        let h = hkl_basis[hkl_idx * 4u + 0u];
        let k = hkl_basis[hkl_idx * 4u + 1u];
        let l = hkl_basis[hkl_idx * 4u + 2u];
        
        M_hkl_rows[i] = Vec3(h*h, k*k, l*l);
    }
    
    let M_hkl = Mat3x3(M_hkl_rows[0], M_hkl_rows[1], M_hkl_rows[2]);
    
    // 4. Loop over all 6 permutations of indices [0,1,2]
    for(var p_idx: u32 = 0u; p_idx < 6u; p_idx = p_idx + 1u) {
        
        let perm_offset = p_idx * 3u;
        
        // 5. Build permuted q_vec
        let q_perm = Vec3(
             q_base[PERMUTATIONS_3[perm_offset + 0u]], 
             q_base[PERMUTATIONS_3[perm_offset + 1u]], 
             q_base[PERMUTATIONS_3[perm_offset + 2u]]
        );
         
        // 6. STAGE 1: Solve the 3x3 system
        let fit_params = solve3x3(M_hkl, q_perm);
         
        // 7. STAGE 2: Basic Boolean Filters
        let cell = extractCellOrtho(fit_params);
         
        // 8. STAGE 3: Strong "True FoM" Filter
        if (cell.a > 0.0) { // Passed basic filters
            
            let A_sol = fit_params[0]; 
            let B_sol = fit_params[1]; 
            let C_sol = fit_params[2]; 
            
            let avg_squared_norm_error = validate_fom_avg_diff(A_sol, B_sol, C_sol);
            
            if (avg_squared_norm_error < FOM_THRESHOLD) { 
            
                // 9. Store result
                let idx = atomicAdd(&solution_counter, 1u);
                if (idx < MAX_SOLUTIONS) {
                    results_list[idx].a = cell.a;
                    results_list[idx].b = cell.b;
                    results_list[idx].c = cell.c;
                }
                
                // 10. Write to debug log
                let debug_idx = atomicAdd(&debug_counter, 1u);
                if (debug_idx < MAX_DEBUG_CELLS) {
                    let debug_cell_offset = debug_idx * 20u; // 4 "peaks" * 5 floats
                    
                    for(var k_log: u32 = 0u; k_log < 3u; k_log = k_log + 1u) {
                        let log_offset = debug_cell_offset + (k_log * 5u);
                        let hkl_idx = hkl_indices[k_log];
                        let q_obs_val = q_perm[k_log];
                        let q_calc = (A_sol * M_hkl_rows[k_log][0] +
                                      B_sol * M_hkl_rows[k_log][1] +
                                      C_sol * M_hkl_rows[k_log][2]);
                    
                        debug_log[log_offset + 0u] = hkl_basis[hkl_idx * 4u + 0u]; // h
                        debug_log[log_offset + 1u] = hkl_basis[hkl_idx * 4u + 1u]; // k
                        debug_log[log_offset + 2u] = hkl_basis[hkl_idx * 4u + 2u]; // l
                        debug_log[log_offset + 3u] = 1.0 / sqrt(q_obs_val); // d_obs
                        debug_log[log_offset + 4u] = 1.0 / sqrt(q_calc);    // d_calc
                    }
                }
                
                break; 
            }
        }
         
    } // p_idx loop
}