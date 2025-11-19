// triclinic_solver.wgsl
// 6-Peak Direct Solve + Combinadics + Optimized FoM

// === Structs ===
struct RawSolution {
    a: f32, b: f32, c: f32,
    alpha: f32, beta: f32, gamma: f32,
    pad1: f32, // <--- Padding
    pad2: f32  // <--- Padding (Total 32 bytes)
}

// === Type Aliases ===
alias Vec6 = array<f32, 6>;
alias Mat6x6 = array<f32, 36>; // Flat 6x6 matrix, row-major

// === Bindings ===
@group(0) @binding(0) var<storage, read> q_obs: array<f32>;
@group(0) @binding(1) var<storage, read> hkl_basis: array<f32>; // [h,k,l,pad]
@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k,l,m,n]

// CHANGED: Replaced massive hkl_combos with Pascal's Triangle Lookup
@group(0) @binding(3) var<storage, read> binomial_table: array<u32>; 

@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawSolution>;

struct Config { 
    z_offset: u32,
    wavelength: f32,
    tth_error: f32,
    max_volume: f32,
    max_impurities: u32,
    n_peaks_for_fom: u32, 
    n_hkl_for_fom: u32,   
    n_basis_total: u32, 
    total_hkl_combos: u32 //  Was pad2, now stores the strict limit
};

@group(0) @binding(6) var<uniform> config: Config;

@group(0) @binding(7) var<storage, read_write> debug_counter: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> debug_log: array<f32>;
@group(0) @binding(9) var<storage, read> q_tolerances: array<f32>;

// === Constants ===
const PI: f32 = 3.1415926535;
const DEG: f32 = 180.0 / PI;
const WORKGROUP_SIZE_Y: u32 = 4u;
const MAX_Y_WORKGROUPS: u32 = 16383u; 
const MAX_SOLUTIONS: u32 = 20000u;
const MAX_DEBUG_CELLS: u32 = 10u;
const MAX_FOM_PEAKS: u32 = 32u; 
const FOM_THRESHOLD: f32 = 3.0;

// Triclinic Constants for Combinadics
const K_VALUE: u32 = 6u; 
const BINOMIAL_STRIDE: u32 = 7u; // Table stores columns 0..6

// === 6! = 720 Permutations ===
const PERMUTATIONS_6: array<u32, 4320> = array<u32, 4320>(
    0u, 1u, 2u, 3u, 4u, 5u, 0u, 1u, 2u, 3u, 5u, 4u, 0u, 1u, 2u, 4u, 3u, 5u, 0u, 1u, 2u, 4u, 5u, 3u, 0u, 1u, 2u, 5u, 3u, 4u, 0u, 1u, 2u, 5u, 4u, 3u, 
    0u, 1u, 3u, 2u, 4u, 5u, 0u, 1u, 3u, 2u, 5u, 4u, 0u, 1u, 3u, 4u, 2u, 5u, 0u, 1u, 3u, 4u, 5u, 2u, 0u, 1u, 3u, 5u, 2u, 4u, 0u, 1u, 3u, 5u, 4u, 2u, 
    0u, 1u, 4u, 2u, 3u, 5u, 0u, 1u, 4u, 2u, 5u, 3u, 0u, 1u, 4u, 3u, 2u, 5u, 0u, 1u, 4u, 3u, 5u, 2u, 0u, 1u, 4u, 5u, 2u, 3u, 0u, 1u, 4u, 5u, 3u, 2u, 
    0u, 1u, 5u, 2u, 3u, 4u, 0u, 1u, 5u, 2u, 4u, 3u, 0u, 1u, 5u, 3u, 2u, 4u, 0u, 1u, 5u, 3u, 4u, 2u, 0u, 1u, 5u, 4u, 2u, 3u, 0u, 1u, 5u, 4u, 3u, 2u, 
    0u, 2u, 1u, 3u, 4u, 5u, 0u, 2u, 1u, 3u, 5u, 4u, 0u, 2u, 1u, 4u, 3u, 5u, 0u, 2u, 1u, 4u, 5u, 3u, 0u, 2u, 1u, 5u, 3u, 4u, 0u, 2u, 1u, 5u, 4u, 3u, 
    0u, 2u, 3u, 1u, 4u, 5u, 0u, 2u, 3u, 1u, 5u, 4u, 0u, 2u, 3u, 4u, 1u, 5u, 0u, 2u, 3u, 4u, 5u, 1u, 0u, 2u, 3u, 5u, 1u, 4u, 0u, 2u, 3u, 5u, 4u, 1u, 
    0u, 2u, 4u, 1u, 3u, 5u, 0u, 2u, 4u, 1u, 5u, 3u, 0u, 2u, 4u, 3u, 1u, 5u, 0u, 2u, 4u, 3u, 5u, 1u, 0u, 2u, 4u, 5u, 1u, 3u, 0u, 2u, 4u, 5u, 3u, 1u, 
    0u, 2u, 5u, 1u, 3u, 4u, 0u, 2u, 5u, 1u, 4u, 3u, 0u, 2u, 5u, 3u, 1u, 4u, 0u, 2u, 5u, 3u, 4u, 1u, 0u, 2u, 5u, 4u, 1u, 3u, 0u, 2u, 5u, 4u, 3u, 1u, 
    0u, 3u, 1u, 2u, 4u, 5u, 0u, 3u, 1u, 2u, 5u, 4u, 0u, 3u, 1u, 4u, 2u, 5u, 0u, 3u, 1u, 4u, 5u, 2u, 0u, 3u, 1u, 5u, 2u, 4u, 0u, 3u, 1u, 5u, 4u, 2u, 
    0u, 3u, 2u, 1u, 4u, 5u, 0u, 3u, 2u, 1u, 5u, 4u, 0u, 3u, 2u, 4u, 1u, 5u, 0u, 3u, 2u, 4u, 5u, 1u, 0u, 3u, 2u, 5u, 1u, 4u, 0u, 3u, 2u, 5u, 4u, 1u, 
    0u, 3u, 4u, 1u, 2u, 5u, 0u, 3u, 4u, 1u, 5u, 2u, 0u, 3u, 4u, 2u, 1u, 5u, 0u, 3u, 4u, 2u, 5u, 1u, 0u, 3u, 4u, 5u, 1u, 2u, 0u, 3u, 4u, 5u, 2u, 1u, 
    0u, 3u, 5u, 1u, 2u, 4u, 0u, 3u, 5u, 1u, 4u, 2u, 0u, 3u, 5u, 2u, 1u, 4u, 0u, 3u, 5u, 2u, 4u, 1u, 0u, 3u, 5u, 4u, 1u, 2u, 0u, 3u, 5u, 4u, 2u, 1u, 
    0u, 4u, 1u, 2u, 3u, 5u, 0u, 4u, 1u, 2u, 5u, 3u, 0u, 4u, 1u, 3u, 2u, 5u, 0u, 4u, 1u, 3u, 5u, 2u, 0u, 4u, 1u, 5u, 2u, 3u, 0u, 4u, 1u, 5u, 3u, 2u, 
    0u, 4u, 2u, 1u, 3u, 5u, 0u, 4u, 2u, 1u, 5u, 3u, 0u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 2u, 3u, 5u, 1u, 0u, 4u, 2u, 5u, 1u, 3u, 0u, 4u, 2u, 5u, 3u, 1u, 
    0u, 4u, 3u, 1u, 2u, 5u, 0u, 4u, 3u, 1u, 5u, 2u, 0u, 4u, 3u, 2u, 1u, 5u, 0u, 4u, 3u, 2u, 5u, 1u, 0u, 4u, 3u, 5u, 1u, 2u, 0u, 4u, 3u, 5u, 2u, 1u, 
    0u, 4u, 5u, 1u, 2u, 3u, 0u, 4u, 5u, 1u, 3u, 2u, 0u, 4u, 5u, 2u, 1u, 3u, 0u, 4u, 5u, 2u, 3u, 1u, 0u, 4u, 5u, 3u, 1u, 2u, 0u, 4u, 5u, 3u, 2u, 1u, 
    0u, 5u, 1u, 2u, 3u, 4u, 0u, 5u, 1u, 2u, 4u, 3u, 0u, 5u, 1u, 3u, 2u, 4u, 0u, 5u, 1u, 3u, 4u, 2u, 0u, 5u, 1u, 4u, 2u, 3u, 0u, 5u, 1u, 4u, 3u, 2u, 
    0u, 5u, 2u, 1u, 3u, 4u, 0u, 5u, 2u, 1u, 4u, 3u, 0u, 5u, 2u, 3u, 1u, 4u, 0u, 5u, 2u, 3u, 4u, 1u, 0u, 5u, 2u, 4u, 1u, 3u, 0u, 5u, 2u, 4u, 3u, 1u, 
    0u, 5u, 3u, 1u, 2u, 4u, 0u, 5u, 3u, 1u, 4u, 2u, 0u, 5u, 3u, 2u, 1u, 4u, 0u, 5u, 3u, 2u, 4u, 1u, 0u, 5u, 3u, 4u, 1u, 2u, 0u, 5u, 3u, 4u, 2u, 1u, 
    0u, 5u, 4u, 1u, 2u, 3u, 0u, 5u, 4u, 1u, 3u, 2u, 0u, 5u, 4u, 2u, 1u, 3u, 0u, 5u, 4u, 2u, 3u, 1u, 0u, 5u, 4u, 3u, 1u, 2u, 0u, 5u, 4u, 3u, 2u, 1u, 
    1u, 0u, 2u, 3u, 4u, 5u, 1u, 0u, 2u, 3u, 5u, 4u, 1u, 0u, 2u, 4u, 3u, 5u, 1u, 0u, 2u, 4u, 5u, 3u, 1u, 0u, 2u, 5u, 3u, 4u, 1u, 0u, 2u, 5u, 4u, 3u, 
    1u, 0u, 3u, 2u, 4u, 5u, 1u, 0u, 3u, 2u, 5u, 4u, 1u, 0u, 3u, 4u, 2u, 5u, 1u, 0u, 3u, 4u, 5u, 2u, 1u, 0u, 3u, 5u, 2u, 4u, 1u, 0u, 3u, 5u, 4u, 2u, 
    1u, 0u, 4u, 2u, 3u, 5u, 1u, 0u, 4u, 2u, 5u, 3u, 1u, 0u, 4u, 3u, 2u, 5u, 1u, 0u, 4u, 3u, 5u, 2u, 1u, 0u, 4u, 5u, 2u, 3u, 1u, 0u, 4u, 5u, 3u, 2u, 
    1u, 0u, 5u, 2u, 3u, 4u, 1u, 0u, 5u, 2u, 4u, 3u, 1u, 0u, 5u, 3u, 2u, 4u, 1u, 0u, 5u, 3u, 4u, 2u, 1u, 0u, 5u, 4u, 2u, 3u, 1u, 0u, 5u, 4u, 3u, 2u, 
    1u, 2u, 0u, 3u, 4u, 5u, 1u, 2u, 0u, 3u, 5u, 4u, 1u, 2u, 0u, 4u, 3u, 5u, 1u, 2u, 0u, 4u, 5u, 3u, 1u, 2u, 0u, 5u, 3u, 4u, 1u, 2u, 0u, 5u, 4u, 3u, 
    1u, 2u, 3u, 0u, 4u, 5u, 1u, 2u, 3u, 0u, 5u, 4u, 1u, 2u, 3u, 4u, 0u, 5u, 1u, 2u, 3u, 4u, 5u, 0u, 1u, 2u, 3u, 5u, 0u, 4u, 1u, 2u, 3u, 5u, 4u, 0u, 
    1u, 2u, 4u, 0u, 3u, 5u, 1u, 2u, 4u, 0u, 5u, 3u, 1u, 2u, 4u, 3u, 0u, 5u, 1u, 2u, 4u, 3u, 5u, 0u, 1u, 2u, 4u, 5u, 0u, 3u, 1u, 2u, 4u, 5u, 3u, 0u, 
    1u, 2u, 5u, 0u, 3u, 4u, 1u, 2u, 5u, 0u, 4u, 3u, 1u, 2u, 5u, 3u, 0u, 4u, 1u, 2u, 5u, 3u, 4u, 0u, 1u, 2u, 5u, 4u, 0u, 3u, 1u, 2u, 5u, 4u, 3u, 0u, 
    1u, 3u, 0u, 2u, 4u, 5u, 1u, 3u, 0u, 2u, 5u, 4u, 1u, 3u, 0u, 4u, 2u, 5u, 1u, 3u, 0u, 4u, 5u, 2u, 1u, 3u, 0u, 5u, 2u, 4u, 1u, 3u, 0u, 5u, 4u, 2u, 
    1u, 3u, 2u, 0u, 4u, 5u, 1u, 3u, 2u, 0u, 5u, 4u, 1u, 3u, 2u, 4u, 0u, 5u, 1u, 3u, 2u, 4u, 5u, 0u, 1u, 3u, 2u, 5u, 0u, 4u, 1u, 3u, 2u, 5u, 4u, 0u, 
    1u, 3u, 4u, 0u, 2u, 5u, 1u, 3u, 4u, 0u, 5u, 2u, 1u, 3u, 4u, 2u, 0u, 5u, 1u, 3u, 4u, 2u, 5u, 0u, 1u, 3u, 4u, 5u, 0u, 2u, 1u, 3u, 4u, 5u, 2u, 0u, 
    1u, 3u, 5u, 0u, 2u, 4u, 1u, 3u, 5u, 0u, 4u, 2u, 1u, 3u, 5u, 2u, 0u, 4u, 1u, 3u, 5u, 2u, 4u, 0u, 1u, 3u, 5u, 4u, 0u, 2u, 1u, 3u, 5u, 4u, 2u, 0u, 
    1u, 4u, 0u, 2u, 3u, 5u, 1u, 4u, 0u, 2u, 5u, 3u, 1u, 4u, 0u, 3u, 2u, 5u, 1u, 4u, 0u, 3u, 5u, 2u, 1u, 4u, 0u, 5u, 2u, 3u, 1u, 4u, 0u, 5u, 3u, 2u, 
    1u, 4u, 2u, 0u, 3u, 5u, 1u, 4u, 2u, 0u, 5u, 3u, 1u, 4u, 2u, 3u, 0u, 5u, 1u, 4u, 2u, 3u, 5u, 0u, 1u, 4u, 2u, 5u, 0u, 3u, 1u, 4u, 2u, 5u, 3u, 0u, 
    1u, 4u, 3u, 0u, 2u, 5u, 1u, 4u, 3u, 0u, 5u, 2u, 1u, 4u, 3u, 2u, 0u, 5u, 1u, 4u, 3u, 2u, 5u, 0u, 1u, 4u, 3u, 5u, 0u, 2u, 1u, 4u, 3u, 5u, 2u, 0u, 
    1u, 4u, 5u, 0u, 2u, 3u, 1u, 4u, 5u, 0u, 3u, 2u, 1u, 4u, 5u, 2u, 0u, 3u, 1u, 4u, 5u, 2u, 3u, 0u, 1u, 4u, 5u, 3u, 0u, 2u, 1u, 4u, 5u, 3u, 2u, 0u, 
    1u, 5u, 0u, 2u, 3u, 4u, 1u, 5u, 0u, 2u, 4u, 3u, 1u, 5u, 0u, 3u, 2u, 4u, 1u, 5u, 0u, 3u, 4u, 2u, 1u, 5u, 0u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 3u, 2u, 
    1u, 5u, 2u, 0u, 3u, 4u, 1u, 5u, 2u, 0u, 4u, 3u, 1u, 5u, 2u, 3u, 0u, 4u, 1u, 5u, 2u, 3u, 4u, 0u, 1u, 5u, 2u, 4u, 0u, 3u, 1u, 5u, 2u, 4u, 3u, 0u, 
    1u, 5u, 3u, 0u, 2u, 4u, 1u, 5u, 3u, 0u, 4u, 2u, 1u, 5u, 3u, 2u, 0u, 4u, 1u, 5u, 3u, 2u, 4u, 0u, 1u, 5u, 3u, 4u, 0u, 2u, 1u, 5u, 3u, 4u, 2u, 0u, 
    1u, 5u, 4u, 0u, 2u, 3u, 1u, 5u, 4u, 0u, 3u, 2u, 1u, 5u, 4u, 2u, 0u, 3u, 1u, 5u, 4u, 2u, 3u, 0u, 1u, 5u, 4u, 3u, 0u, 2u, 1u, 5u, 4u, 3u, 2u, 0u, 
    2u, 0u, 1u, 3u, 4u, 5u, 2u, 0u, 1u, 3u, 5u, 4u, 2u, 0u, 1u, 4u, 3u, 5u, 2u, 0u, 1u, 4u, 5u, 3u, 2u, 0u, 1u, 5u, 3u, 4u, 2u, 0u, 1u, 5u, 4u, 3u, 
    2u, 0u, 3u, 1u, 4u, 5u, 2u, 0u, 3u, 1u, 5u, 4u, 2u, 0u, 3u, 4u, 1u, 5u, 2u, 0u, 3u, 4u, 5u, 1u, 2u, 0u, 3u, 5u, 1u, 4u, 2u, 0u, 3u, 5u, 4u, 1u, 
    2u, 0u, 4u, 1u, 3u, 5u, 2u, 0u, 4u, 1u, 5u, 3u, 2u, 0u, 4u, 3u, 1u, 5u, 2u, 0u, 4u, 3u, 5u, 1u, 2u, 0u, 4u, 5u, 1u, 3u, 2u, 0u, 4u, 5u, 3u, 1u, 
    2u, 0u, 5u, 1u, 3u, 4u, 2u, 0u, 5u, 1u, 4u, 3u, 2u, 0u, 5u, 3u, 1u, 4u, 2u, 0u, 5u, 3u, 4u, 1u, 2u, 0u, 5u, 4u, 1u, 3u, 2u, 0u, 5u, 4u, 3u, 1u, 
    2u, 1u, 0u, 3u, 4u, 5u, 2u, 1u, 0u, 3u, 5u, 4u, 2u, 1u, 0u, 4u, 3u, 5u, 2u, 1u, 0u, 4u, 5u, 3u, 2u, 1u, 0u, 5u, 3u, 4u, 2u, 1u, 0u, 5u, 4u, 3u, 
    2u, 1u, 3u, 0u, 4u, 5u, 2u, 1u, 3u, 0u, 5u, 4u, 2u, 1u, 3u, 4u, 0u, 5u, 2u, 1u, 3u, 4u, 5u, 0u, 2u, 1u, 3u, 5u, 0u, 4u, 2u, 1u, 3u, 5u, 4u, 0u, 
    2u, 1u, 4u, 0u, 3u, 5u, 2u, 1u, 4u, 0u, 5u, 3u, 2u, 1u, 4u, 3u, 0u, 5u, 2u, 1u, 4u, 3u, 5u, 0u, 2u, 1u, 4u, 5u, 0u, 3u, 2u, 1u, 4u, 5u, 3u, 0u, 
    2u, 1u, 5u, 0u, 3u, 4u, 2u, 1u, 5u, 0u, 4u, 3u, 2u, 1u, 5u, 3u, 0u, 4u, 2u, 1u, 5u, 3u, 4u, 0u, 2u, 1u, 5u, 4u, 0u, 3u, 2u, 1u, 5u, 4u, 3u, 0u, 
    2u, 3u, 0u, 1u, 4u, 5u, 2u, 3u, 0u, 1u, 5u, 4u, 2u, 3u, 0u, 4u, 1u, 5u, 2u, 3u, 0u, 4u, 5u, 1u, 2u, 3u, 0u, 5u, 1u, 4u, 2u, 3u, 0u, 5u, 4u, 1u, 
    2u, 3u, 1u, 0u, 4u, 5u, 2u, 3u, 1u, 0u, 5u, 4u, 2u, 3u, 1u, 4u, 0u, 5u, 2u, 3u, 1u, 4u, 5u, 0u, 2u, 3u, 1u, 5u, 0u, 4u, 2u, 3u, 1u, 5u, 4u, 0u, 
    2u, 3u, 4u, 0u, 1u, 5u, 2u, 3u, 4u, 0u, 5u, 1u, 2u, 3u, 4u, 1u, 0u, 5u, 2u, 3u, 4u, 1u, 5u, 0u, 2u, 3u, 4u, 5u, 0u, 1u, 2u, 3u, 4u, 5u, 1u, 0u, 
    2u, 3u, 5u, 0u, 1u, 4u, 2u, 3u, 5u, 0u, 4u, 1u, 2u, 3u, 5u, 1u, 0u, 4u, 2u, 3u, 5u, 1u, 4u, 0u, 2u, 3u, 5u, 4u, 0u, 1u, 2u, 3u, 5u, 4u, 1u, 0u, 
    2u, 4u, 0u, 1u, 3u, 5u, 2u, 4u, 0u, 1u, 5u, 3u, 2u, 4u, 0u, 3u, 1u, 5u, 2u, 4u, 0u, 3u, 5u, 1u, 2u, 4u, 0u, 5u, 1u, 3u, 2u, 4u, 0u, 5u, 3u, 1u, 
    2u, 4u, 1u, 0u, 3u, 5u, 2u, 4u, 1u, 0u, 5u, 3u, 2u, 4u, 1u, 3u, 0u, 5u, 2u, 4u, 1u, 3u, 5u, 0u, 2u, 4u, 1u, 5u, 0u, 3u, 2u, 4u, 1u, 5u, 3u, 0u, 
    2u, 4u, 3u, 0u, 1u, 5u, 2u, 4u, 3u, 0u, 5u, 1u, 2u, 4u, 3u, 1u, 0u, 5u, 2u, 4u, 3u, 1u, 5u, 0u, 2u, 4u, 3u, 5u, 0u, 1u, 2u, 4u, 3u, 5u, 1u, 0u, 
    2u, 4u, 5u, 0u, 1u, 3u, 2u, 4u, 5u, 0u, 3u, 1u, 2u, 4u, 5u, 1u, 0u, 3u, 2u, 4u, 5u, 1u, 3u, 0u, 2u, 4u, 5u, 3u, 0u, 1u, 2u, 4u, 5u, 3u, 1u, 0u, 
    2u, 5u, 0u, 1u, 3u, 4u, 2u, 5u, 0u, 1u, 4u, 3u, 2u, 5u, 0u, 3u, 1u, 4u, 2u, 5u, 0u, 3u, 4u, 1u, 2u, 5u, 0u, 4u, 1u, 3u, 2u, 5u, 0u, 4u, 3u, 1u, 
    2u, 5u, 1u, 0u, 3u, 4u, 2u, 5u, 1u, 0u, 4u, 3u, 2u, 5u, 1u, 3u, 0u, 4u, 2u, 5u, 1u, 3u, 4u, 0u, 2u, 5u, 1u, 4u, 0u, 3u, 2u, 5u, 1u, 4u, 3u, 0u, 
    2u, 5u, 3u, 0u, 1u, 4u, 2u, 5u, 3u, 0u, 4u, 1u, 2u, 5u, 3u, 1u, 0u, 4u, 2u, 5u, 3u, 1u, 4u, 0u, 2u, 5u, 3u, 4u, 0u, 1u, 2u, 5u, 3u, 4u, 1u, 0u, 
    2u, 5u, 4u, 0u, 1u, 3u, 2u, 5u, 4u, 0u, 3u, 1u, 2u, 5u, 4u, 1u, 0u, 3u, 2u, 5u, 4u, 1u, 3u, 0u, 2u, 5u, 4u, 3u, 0u, 1u, 2u, 5u, 4u, 3u, 1u, 0u, 
    3u, 0u, 1u, 2u, 4u, 5u, 3u, 0u, 1u, 2u, 5u, 4u, 3u, 0u, 1u, 4u, 2u, 5u, 3u, 0u, 1u, 4u, 5u, 2u, 3u, 0u, 1u, 5u, 2u, 4u, 3u, 0u, 1u, 5u, 4u, 2u, 
    3u, 0u, 2u, 1u, 4u, 5u, 3u, 0u, 2u, 1u, 5u, 4u, 3u, 0u, 2u, 4u, 1u, 5u, 3u, 0u, 2u, 4u, 5u, 1u, 3u, 0u, 2u, 5u, 1u, 4u, 3u, 0u, 2u, 5u, 4u, 1u, 
    3u, 0u, 4u, 1u, 2u, 5u, 3u, 0u, 4u, 1u, 5u, 2u, 3u, 0u, 4u, 2u, 1u, 5u, 3u, 0u, 4u, 2u, 5u, 1u, 3u, 0u, 4u, 5u, 1u, 2u, 3u, 0u, 4u, 5u, 2u, 1u, 
    3u, 0u, 5u, 1u, 2u, 4u, 3u, 0u, 5u, 1u, 4u, 2u, 3u, 0u, 5u, 2u, 1u, 4u, 3u, 0u, 5u, 2u, 4u, 1u, 3u, 0u, 5u, 4u, 1u, 2u, 3u, 0u, 5u, 4u, 2u, 1u, 
    3u, 1u, 0u, 2u, 4u, 5u, 3u, 1u, 0u, 2u, 5u, 4u, 3u, 1u, 0u, 4u, 2u, 5u, 3u, 1u, 0u, 4u, 5u, 2u, 3u, 1u, 0u, 5u, 2u, 4u, 3u, 1u, 0u, 5u, 4u, 2u, 
    3u, 1u, 2u, 0u, 4u, 5u, 3u, 1u, 2u, 0u, 5u, 4u, 3u, 1u, 2u, 4u, 0u, 5u, 3u, 1u, 2u, 4u, 5u, 0u, 3u, 1u, 2u, 5u, 0u, 4u, 3u, 1u, 2u, 5u, 4u, 0u, 
    3u, 1u, 4u, 0u, 2u, 5u, 3u, 1u, 4u, 0u, 5u, 2u, 3u, 1u, 4u, 2u, 0u, 5u, 3u, 1u, 4u, 2u, 5u, 0u, 3u, 1u, 4u, 5u, 0u, 2u, 3u, 1u, 4u, 5u, 2u, 0u, 
    3u, 1u, 5u, 0u, 2u, 4u, 3u, 1u, 5u, 0u, 4u, 2u, 3u, 1u, 5u, 2u, 0u, 4u, 3u, 1u, 5u, 2u, 4u, 0u, 3u, 1u, 5u, 4u, 0u, 2u, 3u, 1u, 5u, 4u, 2u, 0u, 
    3u, 2u, 0u, 1u, 4u, 5u, 3u, 2u, 0u, 1u, 5u, 4u, 3u, 2u, 0u, 4u, 1u, 5u, 3u, 2u, 0u, 4u, 5u, 1u, 3u, 2u, 0u, 5u, 1u, 4u, 3u, 2u, 0u, 5u, 4u, 1u, 
    3u, 2u, 1u, 0u, 4u, 5u, 3u, 2u, 1u, 0u, 5u, 4u, 3u, 2u, 1u, 4u, 0u, 5u, 3u, 2u, 1u, 4u, 5u, 0u, 3u, 2u, 1u, 5u, 0u, 4u, 3u, 2u, 1u, 5u, 4u, 0u, 
    3u, 2u, 4u, 0u, 1u, 5u, 3u, 2u, 4u, 0u, 5u, 1u, 3u, 2u, 4u, 1u, 0u, 5u, 3u, 2u, 4u, 1u, 5u, 0u, 3u, 2u, 4u, 5u, 0u, 1u, 3u, 2u, 4u, 5u, 1u, 0u, 
    3u, 2u, 5u, 0u, 1u, 4u, 3u, 2u, 5u, 0u, 4u, 1u, 3u, 2u, 5u, 1u, 0u, 4u, 3u, 2u, 5u, 1u, 4u, 0u, 3u, 2u, 5u, 4u, 0u, 1u, 3u, 2u, 5u, 4u, 1u, 0u, 
    3u, 4u, 0u, 1u, 2u, 5u, 3u, 4u, 0u, 1u, 5u, 2u, 3u, 4u, 0u, 2u, 1u, 5u, 3u, 4u, 0u, 2u, 5u, 1u, 3u, 4u, 0u, 5u, 1u, 2u, 3u, 4u, 0u, 5u, 2u, 1u, 
    3u, 4u, 1u, 0u, 2u, 5u, 3u, 4u, 1u, 0u, 5u, 2u, 3u, 4u, 1u, 2u, 0u, 5u, 3u, 4u, 1u, 2u, 5u, 0u, 3u, 4u, 1u, 5u, 0u, 2u, 3u, 4u, 1u, 5u, 2u, 0u, 
    3u, 4u, 2u, 0u, 1u, 5u, 3u, 4u, 2u, 0u, 5u, 1u, 3u, 4u, 2u, 1u, 0u, 5u, 3u, 4u, 2u, 1u, 5u, 0u, 3u, 4u, 2u, 5u, 0u, 1u, 3u, 4u, 2u, 5u, 1u, 0u, 
    3u, 4u, 5u, 0u, 1u, 2u, 3u, 4u, 5u, 0u, 2u, 1u, 3u, 4u, 5u, 1u, 0u, 2u, 3u, 4u, 5u, 1u, 2u, 0u, 3u, 4u, 5u, 2u, 0u, 1u, 3u, 4u, 5u, 2u, 1u, 0u, 
    3u, 5u, 0u, 1u, 2u, 4u, 3u, 5u, 0u, 1u, 4u, 2u, 3u, 5u, 0u, 2u, 1u, 4u, 3u, 5u, 0u, 2u, 4u, 1u, 3u, 5u, 0u, 4u, 1u, 2u, 3u, 5u, 0u, 4u, 2u, 1u, 
    3u, 5u, 1u, 0u, 2u, 4u, 3u, 5u, 1u, 0u, 4u, 2u, 3u, 5u, 1u, 2u, 0u, 4u, 3u, 5u, 1u, 2u, 4u, 0u, 3u, 5u, 1u, 4u, 0u, 2u, 3u, 5u, 1u, 4u, 2u, 0u, 
    3u, 5u, 2u, 0u, 1u, 4u, 3u, 5u, 2u, 0u, 4u, 1u, 3u, 5u, 2u, 1u, 0u, 4u, 3u, 5u, 2u, 1u, 4u, 0u, 3u, 5u, 2u, 4u, 0u, 1u, 3u, 5u, 2u, 4u, 1u, 0u, 
    3u, 5u, 4u, 0u, 1u, 2u, 3u, 5u, 4u, 0u, 2u, 1u, 3u, 5u, 4u, 1u, 0u, 2u, 3u, 5u, 4u, 1u, 2u, 0u, 3u, 5u, 4u, 2u, 0u, 1u, 3u, 5u, 4u, 2u, 1u, 0u, 
    4u, 0u, 1u, 2u, 3u, 5u, 4u, 0u, 1u, 2u, 5u, 3u, 4u, 0u, 1u, 3u, 2u, 5u, 4u, 0u, 1u, 3u, 5u, 2u, 4u, 0u, 1u, 5u, 2u, 3u, 4u, 0u, 1u, 5u, 3u, 2u, 
    4u, 0u, 2u, 1u, 3u, 5u, 4u, 0u, 2u, 1u, 5u, 3u, 4u, 0u, 2u, 3u, 1u, 5u, 4u, 0u, 2u, 3u, 5u, 1u, 4u, 0u, 2u, 5u, 1u, 3u, 4u, 0u, 2u, 5u, 3u, 1u, 
    4u, 0u, 3u, 1u, 2u, 5u, 4u, 0u, 3u, 1u, 5u, 2u, 4u, 0u, 3u, 2u, 1u, 5u, 4u, 0u, 3u, 2u, 5u, 1u, 4u, 0u, 3u, 5u, 1u, 2u, 4u, 0u, 3u, 5u, 2u, 1u, 
    4u, 0u, 5u, 1u, 2u, 3u, 4u, 0u, 5u, 1u, 3u, 2u, 4u, 0u, 5u, 2u, 1u, 3u, 4u, 0u, 5u, 2u, 3u, 1u, 4u, 0u, 5u, 3u, 1u, 2u, 4u, 0u, 5u, 3u, 2u, 1u, 
    4u, 1u, 0u, 2u, 3u, 5u, 4u, 1u, 0u, 2u, 5u, 3u, 4u, 1u, 0u, 3u, 2u, 5u, 4u, 1u, 0u, 3u, 5u, 2u, 4u, 1u, 0u, 5u, 2u, 3u, 4u, 1u, 0u, 5u, 3u, 2u, 
    4u, 1u, 2u, 0u, 3u, 5u, 4u, 1u, 2u, 0u, 5u, 3u, 4u, 1u, 2u, 3u, 0u, 5u, 4u, 1u, 2u, 3u, 5u, 0u, 4u, 1u, 2u, 5u, 0u, 3u, 4u, 1u, 2u, 5u, 3u, 0u, 
    4u, 1u, 3u, 0u, 2u, 5u, 4u, 1u, 3u, 0u, 5u, 2u, 4u, 1u, 3u, 2u, 0u, 5u, 4u, 1u, 3u, 2u, 5u, 0u, 4u, 1u, 3u, 5u, 0u, 2u, 4u, 1u, 3u, 5u, 2u, 0u, 
    4u, 1u, 5u, 0u, 2u, 3u, 4u, 1u, 5u, 0u, 3u, 2u, 4u, 1u, 5u, 2u, 0u, 3u, 4u, 1u, 5u, 2u, 3u, 0u, 4u, 1u, 5u, 3u, 0u, 2u, 4u, 1u, 5u, 3u, 2u, 0u, 
    4u, 2u, 0u, 1u, 3u, 5u, 4u, 2u, 0u, 1u, 5u, 3u, 4u, 2u, 0u, 3u, 1u, 5u, 4u, 2u, 0u, 3u, 5u, 1u, 4u, 2u, 0u, 5u, 1u, 3u, 4u, 2u, 0u, 5u, 3u, 1u, 
    4u, 2u, 1u, 0u, 3u, 5u, 4u, 2u, 1u, 0u, 5u, 3u, 4u, 2u, 1u, 3u, 0u, 5u, 4u, 2u, 1u, 3u, 5u, 0u, 4u, 2u, 1u, 5u, 0u, 3u, 4u, 2u, 1u, 5u, 3u, 0u, 
    4u, 2u, 3u, 0u, 1u, 5u, 4u, 2u, 3u, 0u, 5u, 1u, 4u, 2u, 3u, 1u, 0u, 5u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 2u, 3u, 5u, 0u, 1u, 4u, 2u, 3u, 5u, 1u, 0u, 
    4u, 2u, 5u, 0u, 1u, 3u, 4u, 2u, 5u, 0u, 3u, 1u, 4u, 2u, 5u, 1u, 0u, 3u, 4u, 2u, 5u, 1u, 3u, 0u, 4u, 2u, 5u, 3u, 0u, 1u, 4u, 2u, 5u, 3u, 1u, 0u, 
    4u, 3u, 0u, 1u, 2u, 5u, 4u, 3u, 0u, 1u, 5u, 2u, 4u, 3u, 0u, 2u, 1u, 5u, 4u, 3u, 0u, 2u, 5u, 1u, 4u, 3u, 0u, 5u, 1u, 2u, 4u, 3u, 0u, 5u, 2u, 1u, 
    4u, 3u, 1u, 0u, 2u, 5u, 4u, 3u, 1u, 0u, 5u, 2u, 4u, 3u, 1u, 2u, 0u, 5u, 4u, 3u, 1u, 2u, 5u, 0u, 4u, 3u, 1u, 5u, 0u, 2u, 4u, 3u, 1u, 5u, 2u, 0u, 
    4u, 3u, 2u, 0u, 1u, 5u, 4u, 3u, 2u, 0u, 5u, 1u, 4u, 3u, 2u, 1u, 0u, 5u, 4u, 3u, 2u, 1u, 5u, 0u, 4u, 3u, 2u, 5u, 0u, 1u, 4u, 3u, 2u, 5u, 1u, 0u, 
    4u, 3u, 5u, 0u, 1u, 2u, 4u, 3u, 5u, 0u, 2u, 1u, 4u, 3u, 5u, 1u, 0u, 2u, 4u, 3u, 5u, 1u, 2u, 0u, 4u, 3u, 5u, 2u, 0u, 1u, 4u, 3u, 5u, 2u, 1u, 0u, 
    4u, 5u, 0u, 1u, 2u, 3u, 4u, 5u, 0u, 1u, 3u, 2u, 4u, 5u, 0u, 2u, 1u, 3u, 4u, 5u, 0u, 2u, 3u, 1u, 4u, 5u, 0u, 3u, 1u, 2u, 4u, 5u, 0u, 3u, 2u, 1u, 
    4u, 5u, 1u, 0u, 2u, 3u, 4u, 5u, 1u, 0u, 3u, 2u, 4u, 5u, 1u, 2u, 0u, 3u, 4u, 5u, 1u, 2u, 3u, 0u, 4u, 5u, 1u, 3u, 0u, 2u, 4u, 5u, 1u, 3u, 2u, 0u, 
    4u, 5u, 2u, 0u, 1u, 3u, 4u, 5u, 2u, 0u, 3u, 1u, 4u, 5u, 2u, 1u, 0u, 3u, 4u, 5u, 2u, 1u, 3u, 0u, 4u, 5u, 2u, 3u, 0u, 1u, 4u, 5u, 2u, 3u, 1u, 0u, 
    4u, 5u, 3u, 0u, 1u, 2u, 4u, 5u, 3u, 0u, 2u, 1u, 4u, 5u, 3u, 1u, 0u, 2u, 4u, 5u, 3u, 1u, 2u, 0u, 4u, 5u, 3u, 2u, 0u, 1u, 4u, 5u, 3u, 2u, 1u, 0u, 
    5u, 0u, 1u, 2u, 3u, 4u, 5u, 0u, 1u, 2u, 4u, 3u, 5u, 0u, 1u, 3u, 2u, 4u, 5u, 0u, 1u, 3u, 4u, 2u, 5u, 0u, 1u, 4u, 2u, 3u, 5u, 0u, 1u, 4u, 3u, 2u, 
    5u, 0u, 2u, 1u, 3u, 4u, 5u, 0u, 2u, 1u, 4u, 3u, 5u, 0u, 2u, 3u, 1u, 4u, 5u, 0u, 2u, 3u, 4u, 1u, 5u, 0u, 2u, 4u, 1u, 3u, 5u, 0u, 2u, 4u, 3u, 1u, 
    5u, 0u, 3u, 1u, 2u, 4u, 5u, 0u, 3u, 1u, 4u, 2u, 5u, 0u, 3u, 2u, 1u, 4u, 5u, 0u, 3u, 2u, 4u, 1u, 5u, 0u, 3u, 4u, 1u, 2u, 5u, 0u, 3u, 4u, 2u, 1u, 
    5u, 0u, 4u, 1u, 2u, 3u, 5u, 0u, 4u, 1u, 3u, 2u, 5u, 0u, 4u, 2u, 1u, 3u, 5u, 0u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 3u, 1u, 2u, 5u, 0u, 4u, 3u, 2u, 1u, 
    5u, 1u, 0u, 2u, 3u, 4u, 5u, 1u, 0u, 2u, 4u, 3u, 5u, 1u, 0u, 3u, 2u, 4u, 5u, 1u, 0u, 3u, 4u, 2u, 5u, 1u, 0u, 4u, 2u, 3u, 5u, 1u, 0u, 4u, 3u, 2u, 
    5u, 1u, 2u, 0u, 3u, 4u, 5u, 1u, 2u, 0u, 4u, 3u, 5u, 1u, 2u, 3u, 0u, 4u, 5u, 1u, 2u, 3u, 4u, 0u, 5u, 1u, 2u, 4u, 0u, 3u, 5u, 1u, 2u, 4u, 3u, 0u, 
    5u, 1u, 3u, 0u, 2u, 4u, 5u, 1u, 3u, 0u, 4u, 2u, 5u, 1u, 3u, 2u, 0u, 4u, 5u, 1u, 3u, 2u, 4u, 0u, 5u, 1u, 3u, 4u, 0u, 2u, 5u, 1u, 3u, 4u, 2u, 0u, 
    5u, 1u, 4u, 0u, 2u, 3u, 5u, 1u, 4u, 0u, 3u, 2u, 5u, 1u, 4u, 2u, 0u, 3u, 5u, 1u, 4u, 2u, 3u, 0u, 5u, 1u, 4u, 3u, 0u, 2u, 5u, 1u, 4u, 3u, 2u, 0u, 
    5u, 2u, 0u, 1u, 3u, 4u, 5u, 2u, 0u, 1u, 4u, 3u, 5u, 2u, 0u, 3u, 1u, 4u, 5u, 2u, 0u, 3u, 4u, 1u, 5u, 2u, 0u, 4u, 1u, 3u, 5u, 2u, 0u, 4u, 3u, 1u, 
    5u, 2u, 1u, 0u, 3u, 4u, 5u, 2u, 1u, 0u, 4u, 3u, 5u, 2u, 1u, 3u, 0u, 4u, 5u, 2u, 1u, 3u, 4u, 0u, 5u, 2u, 1u, 4u, 0u, 3u, 5u, 2u, 1u, 4u, 3u, 0u, 
    5u, 2u, 3u, 0u, 1u, 4u, 5u, 2u, 3u, 0u, 4u, 1u, 5u, 2u, 3u, 1u, 0u, 4u, 5u, 2u, 3u, 1u, 4u, 0u, 5u, 2u, 3u, 4u, 0u, 1u, 5u, 2u, 3u, 4u, 1u, 0u, 
    5u, 2u, 4u, 0u, 1u, 3u, 5u, 2u, 4u, 0u, 3u, 1u, 5u, 2u, 4u, 1u, 0u, 3u, 5u, 2u, 4u, 1u, 3u, 0u, 5u, 2u, 4u, 3u, 0u, 1u, 5u, 2u, 4u, 3u, 1u, 0u, 
    5u, 3u, 0u, 1u, 2u, 4u, 5u, 3u, 0u, 1u, 4u, 2u, 5u, 3u, 0u, 2u, 1u, 4u, 5u, 3u, 0u, 2u, 4u, 1u, 5u, 3u, 0u, 4u, 1u, 2u, 5u, 3u, 0u, 4u, 2u, 1u, 
    5u, 3u, 1u, 0u, 2u, 4u, 5u, 3u, 1u, 0u, 4u, 2u, 5u, 3u, 1u, 2u, 0u, 4u, 5u, 3u, 1u, 2u, 4u, 0u, 5u, 3u, 1u, 4u, 0u, 2u, 5u, 3u, 1u, 4u, 2u, 0u, 
    5u, 3u, 2u, 0u, 1u, 4u, 5u, 3u, 2u, 0u, 4u, 1u, 5u, 3u, 2u, 1u, 0u, 4u, 5u, 3u, 2u, 1u, 4u, 0u, 5u, 3u, 2u, 4u, 0u, 1u, 5u, 3u, 2u, 4u, 1u, 0u, 
    5u, 3u, 4u, 0u, 1u, 2u, 5u, 3u, 4u, 0u, 2u, 1u, 5u, 3u, 4u, 1u, 0u, 2u, 5u, 3u, 4u, 1u, 2u, 0u, 5u, 3u, 4u, 2u, 0u, 1u, 5u, 3u, 4u, 2u, 1u, 0u, 
    5u, 4u, 0u, 1u, 2u, 3u, 5u, 4u, 0u, 1u, 3u, 2u, 5u, 4u, 0u, 2u, 1u, 3u, 5u, 4u, 0u, 2u, 3u, 1u, 5u, 4u, 0u, 3u, 1u, 2u, 5u, 4u, 0u, 3u, 2u, 1u, 
    5u, 4u, 1u, 0u, 2u, 3u, 5u, 4u, 1u, 0u, 3u, 2u, 5u, 4u, 1u, 2u, 0u, 3u, 5u, 4u, 1u, 2u, 3u, 0u, 5u, 4u, 1u, 3u, 0u, 2u, 5u, 4u, 1u, 3u, 2u, 0u, 
    5u, 4u, 2u, 0u, 1u, 3u, 5u, 4u, 2u, 0u, 3u, 1u, 5u, 4u, 2u, 1u, 0u, 3u, 5u, 4u, 2u, 1u, 3u, 0u, 5u, 4u, 2u, 3u, 0u, 1u, 5u, 4u, 2u, 3u, 1u, 0u, 
    5u, 4u, 3u, 0u, 1u, 2u, 5u, 4u, 3u, 0u, 2u, 1u, 5u, 4u, 3u, 1u, 0u, 2u, 5u, 4u, 3u, 1u, 2u, 0u, 5u, 4u, 3u, 2u, 0u, 1u, 5u, 4u, 3u, 2u, 1u, 0u
);

// === Helper Functions ===

fn solve6x6(A: Mat6x6, b: Vec6) -> Vec6 {
    var M: Mat6x6 = A;
    var v: Vec6 = b;
    let n: u32 = 6u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let pivot: f32 = M[i * n + i];
        if (abs(pivot) < 1e-10) { return Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let fac: f32 = M[r * n + i] / pivot;
            for (var c: u32 = i; c < n; c = c + 1u) {
                M[r * n + c] = M[r * n + c] - fac * M[i * n + c];
            }
            v[r] = v[r] - fac * v[i];
        }
    }
    var x: Vec6;
    for (var i_s: i32 = 5; i_s >= 0; i_s = i_s - 1) {
        let i: u32 = u32(i_s);
        var s: f32 = v[i];
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            s = s - M[i * n + j] * x[j];
        }
        x[i] = s / M[i * n + i];
    }
    return x;
}

fn invert3x3(M: mat3x3<f32>) -> mat3x3<f32> {
    let det = determinant(M);
    if (abs(det) < 1e-14) { return mat3x3<f32>(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0); }
    let invDet: f32 = 1.0 / det;
    var inv: mat3x3<f32>;
    inv[0][0] = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * invDet;
    inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * invDet;
    inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * invDet;
    inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * invDet;
    inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * invDet;
    inv[1][2] = (M[0][2] * M[1][0] - M[0][0] * M[1][2]) * invDet;
    inv[2][0] = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * invDet;
    inv[2][1] = (M[0][1] * M[2][0] - M[0][0] * M[2][1]) * invDet;
    inv[2][2] = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * invDet;
    return inv;
}

fn extractCell(params: Vec6) -> RawSolution {
    let p1: f32 = params[0]; let p2: f32 = params[1]; let p3: f32 = params[2];
    let p4: f32 = params[3]; let p5: f32 = params[4]; let p6: f32 = params[5];

    if (p1 <= 1e-12 || p2 <= 1e-12 || p3 <= 1e-12) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.,0.); }

    let G_star = mat3x3<f32>(
        vec3<f32>(p1, p6/2.0, p5/2.0),
        vec3<f32>(p6/2.0, p2, p4/2.0),
        vec3<f32>(p5/2.0, p4/2.0, p3)
    );
    
    let G = invert3x3(G_star);
    if (G[0][0] <= 1e-6 || G[1][1] <= 1e-6 || G[2][2] <= 1e-6) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0); }

    let a: f32 = sqrt(G[0][0]);
    let b: f32 = sqrt(G[1][1]);
    let c: f32 = sqrt(G[2][2]);
    if (a < 2.0 || a > 50.0 || b < 2.0 || b > 50.0 || c < 2.0 || c > 50.0) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0); }

    let alpha_cos = clamp(G[1][2] / (b*c), -1.0, 1.0);
    let beta_cos = clamp(G[0][2] / (a*c), -1.0, 1.0);
    let gamma_cos = clamp(G[0][1] / (a*b), -1.0, 1.0);
    let alpha = acos(alpha_cos) * DEG;
    let beta = acos(beta_cos) * DEG;
    let gamma = acos(gamma_cos) * DEG;

    if (alpha < 60.0 || alpha > 150.0 || beta < 60.0 || beta > 150.0 || gamma < 60.0 || gamma > 150.0) { 
        return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0); 
    }

    let V_star_sq = determinant(G_star);
    if (V_star_sq <= 1e-12) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0); }
    let volume = 1.0 / sqrt(V_star_sq);
    if (volume < 20.0 || volume > config.max_volume) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.,0.); }
    
    return RawSolution(a, b, c, alpha, beta, gamma,0.,0.);
}

// === Combinatorial Number System Helper ===
// Calculates the k-th combination [c6...c1] for linear index 'm'
fn get_combinadic_indices(linear_index: u32, n_max: u32) -> array<u32, 6> {
    var m = linear_index;
    var out: array<u32, 6>;
    // Start searching from the largest possible index (n_max - 1)
    var v = n_max - 1u; 
    
    for (var k_idx: u32 = K_VALUE; k_idx > 0u; k_idx = k_idx - 1u) {
        loop {
            // Lookup Binomial(v, k) from precomputed table
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

// === Optimized FoM Validator ===
fn validate_fom_avg_diff(p: Vec6) -> f32 {
    let n_peaks_to_check = min(config.n_peaks_for_fom, MAX_FOM_PEAKS);
    
    // --- OPTIMIZATION: If no impurities, skip sorting entirely ---
    if (config.max_impurities == 0u) {
        var sum_sq_error: f32 = 0.0;
        for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
            let q_obs_val = q_obs[i];
            let tol = q_tolerances[i]; 
            var min_diff: f32 = 1e10; 
            
            for (var j: u32 = 0u; j < config.n_hkl_for_fom; j = j + 1u) {
                let h = hkl_basis[j * 4u + 0u];
                let k = hkl_basis[j * 4u + 1u];
                let l = hkl_basis[j * 4u + 2u];
                let q_calc = p[0]*h*h + p[1]*k*k + p[2]*l*l + p[3]*k*l + p[4]*h*l + p[5]*h*k;
                let diff = abs(q_obs_val - q_calc);
                if (diff < min_diff) { min_diff = diff; }
            }
            let norm = min_diff / tol;
            sum_sq_error += (norm * norm);
        }
        let avg = sum_sq_error / f32(n_peaks_to_check);
        if (avg > FOM_THRESHOLD) { return 999.0; }
        return avg;
    } 
    // --- END OPTIMIZATION ---

    // --- IMPURITY PATH (Partial Sort) ---
    var errors: array<f32, 32>;
    for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
        let q_obs_val = q_obs[i];
        let tol = q_tolerances[i]; 
        var min_diff: f32 = 1e10; 
        for (var j: u32 = 0u; j < config.n_hkl_for_fom; j = j + 1u) {
            let h = hkl_basis[j * 4u + 0u];
            let k = hkl_basis[j * 4u + 1u];
            let l = hkl_basis[j * 4u + 2u];
            let q_calc = p[0]*h*h + p[1]*k*k + p[2]*l*l + p[3]*k*l + p[4]*h*l + p[5]*h*k;
            let diff = abs(q_obs_val - q_calc);
            if (diff < min_diff) { min_diff = diff; }
        }
        let norm = min_diff / tol;
        errors[i] = (norm * norm);
    }

    // Simple selection logic: we only need to sum the best (N - impurities)
    let count_to_sum = n_peaks_to_check - config.max_impurities;
    var sum_of_valid_errors: f32 = 0.0;

    // Insertion sort logic for top K items
    // Since count_to_sum is usually large (e.g. 18 out of 20), it's often faster 
    // to sort the WHOLE array of 20 than to maintain a complex heap. 
    // Bubble sort for small N is very fast in registers.
    
    for (var i: u32 = 0u; i < count_to_sum; i = i + 1u) {
        var min_val = errors[i];
        var min_idx = i;
        for (var j: u32 = i + 1u; j < n_peaks_to_check; j = j + 1u) {
            if (errors[j] < min_val) {
                min_val = errors[j];
                min_idx = j;
            }
        }
        // Swap found min to position i
        let temp = errors[i];
        errors[i] = min_val;
        errors[min_idx] = temp;
        
        // Accumulate immediately
        sum_of_valid_errors += min_val;
    }

    let avg = sum_of_valid_errors / f32(count_to_sum);
    if (avg > FOM_THRESHOLD) { return 999.0; }
    return avg;
}

// === Main Kernel ===
@compute @workgroup_size(4, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Optimization: Early exit if buffer full
    if (atomicLoad(&solution_counter) >= MAX_SOLUTIONS) { return; }
    
    // 1. Calculate Global Indices
    let peak_combo_idx: u32 = global_id.x;
   // let hkl_threads_per_z = MAX_Y_WORKGROUPS * WORKGROUP_SIZE_Y;
    let hkl_linear_idx: u32 = config.z_offset + global_id.y;

    if (hkl_linear_idx >= config.total_hkl_combos) { return; }

    // 2. Bounds Check
    let num_peak_combos = arrayLength(&peak_combos) / 6u;
    if (peak_combo_idx >= num_peak_combos) { return; }

    // 3. Combinadics: Generate HKL Indices on the fly
    // n_basis_total (N) is passed in config.n_basis_total (mapped to pad1 effectively in struct)
    let hkl_indices = get_combinadic_indices(hkl_linear_idx, config.n_basis_total);

    // 4. Build M Matrix
    var M: Mat6x6;
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let hkl_idx = hkl_indices[i];
        let h = hkl_basis[hkl_idx * 4u + 0u];
        let k = hkl_basis[hkl_idx * 4u + 1u];
        let l = hkl_basis[hkl_idx * 4u + 2u];
        let row_offset = i * 6u;
        M[row_offset + 0u] = h*h;
        M[row_offset + 1u] = k*k;
        M[row_offset + 2u] = l*l;
        M[row_offset + 3u] = k*l;
        M[row_offset + 4u] = h*l;
        M[row_offset + 5u] = h*k;
    }

    // 5. Get q_obs base vector
    let p_offset = peak_combo_idx * 6u;
    let q_base = Vec6(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]],
        q_obs[peak_combos[p_offset + 3u]],
        q_obs[peak_combos[p_offset + 4u]],
        q_obs[peak_combos[p_offset + 5u]]
    );

    // 6. Loop over all 720 permutations
    for(var p_idx: u32 = 0u; p_idx < 720u; p_idx = p_idx + 1u) {
        let perm_offset = p_idx * 6u;
        let q_perm = Vec6(
             q_base[PERMUTATIONS_6[perm_offset + 0u]], 
             q_base[PERMUTATIONS_6[perm_offset + 1u]], 
             q_base[PERMUTATIONS_6[perm_offset + 2u]],
             q_base[PERMUTATIONS_6[perm_offset + 3u]],
             q_base[PERMUTATIONS_6[perm_offset + 4u]],
             q_base[PERMUTATIONS_6[perm_offset + 5u]]
        );
         
        let fit_params = solve6x6(M, q_perm);
        let cell = extractCell(fit_params);
         
        if (cell.a > 0.0) { 
            let avg_err = validate_fom_avg_diff(fit_params);
            if (avg_err < FOM_THRESHOLD) {
                // Atomic Add to Global Counter
                let idx = atomicAdd(&solution_counter, 1u);
                if (idx < MAX_SOLUTIONS) {
                    results_list[idx] = cell;
                }
                // Debug Log Logic
                let debug_idx = atomicAdd(&debug_counter, 1u);
                if (debug_idx < MAX_DEBUG_CELLS) {
                    let debug_cell_offset = debug_idx * 30u; 
                    for(var k_log: u32 = 0u; k_log < 6u; k_log = k_log + 1u) {
                        let log_offset = debug_cell_offset + (k_log * 5u);
                        let hkl_idx = hkl_indices[k_log]; // Use local indices
                        let q_obs_val = q_perm[k_log];
                        let q_calc = (fit_params[0]*M[k_log*6u+0u] + fit_params[1]*M[k_log*6u+1u] +
                                      fit_params[2]*M[k_log*6u+2u] + fit_params[3]*M[k_log*6u+3u] +
                                      fit_params[4]*M[k_log*6u+4u] + fit_params[5]*M[k_log*6u+5u]);
                        debug_log[log_offset + 0u] = hkl_basis[hkl_idx * 4u + 0u]; 
                        debug_log[log_offset + 1u] = hkl_basis[hkl_idx * 4u + 1u]; 
                        debug_log[log_offset + 2u] = hkl_basis[hkl_idx * 4u + 2u]; 
                        debug_log[log_offset + 3u] = 1.0 / sqrt(q_obs_val); 
                        debug_log[log_offset + 4u] = 1.0 / sqrt(q_calc);
                    }
                }
                break; // Stop checking permutations for this combo
            }
        }
    } 
}