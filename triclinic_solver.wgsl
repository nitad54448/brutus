// triclinic_solver.wgsl
// 6-Peak Direct Solve + TRUE Figure of Merit (FoM) Validation
//
// Stage 1: 6-peak direct solve (720 permutations).
// Stage 2: Basic boolean filters (singular, volume, params).
// Stage 3: For passing cells, run a "TRUE FoM" filter.
//          This filter calculates the mean *squared* normalized error
//          for the first N peaks and checks it against a threshold.

// === Structs ===
struct RawSolution {
    a: f32,
    b: f32,
    c: f32,
    alpha: f32,
    beta: f32,
    gamma: f32,
}

// === Type Aliases ===
alias Vec6 = array<f32, 6>;
alias Mat6x6 = array<f32, 36>; // A flat 6x6 matrix, stored row-major

// === Bindings ===
@group(0) @binding(0) var<storage, read> q_obs: array<f32>;
@group(0) @binding(1) var<storage, read> hkl_basis: array<f32>; // [h,k,l,pad]
@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k,l,m,n]
@group(0) @binding(3) var<storage, read> hkl_combos: array<u32>; // [n1,n2,n3,n4,n5,n6]
@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawSolution>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<storage, read_write> debug_counter: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> debug_log: array<f32>;
// <-- CHANGED: Add new tolerance buffer -->
@group(0) @binding(9) var<storage, read> q_tolerances: array<f32>;


// === Config Struct ===
// <-- CHANGED: Matched to 36-byte monoclinic struct for JS consistency -->
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

// === Constants ===
const PI: f32 = 3.1415926535;
const RAD: f32 = PI / 180.0;
const DEG: f32 = 180.0 / PI;
const WORKGROUP_SIZE_Y: u32 = 4u;
const MAX_Y_WORKGROUPS: u32 = 256u; // la meme que dans le webgpu
const MAX_SOLUTIONS: u32 = 50000u;
const MAX_DEBUG_CELLS: u32 = 10u;

// threshold for the *mean squared* normalized error
const FOM_THRESHOLD: f32 = 3.; // c'est le carré de la moyenne de dq

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

// 6x6 Matrix Solve (Gaussian Elimination)
fn solve6x6(A: Mat6x6, b: Vec6) -> Vec6 {
    var M: Mat6x6 = A;
    var v: Vec6 = b;
    let n: u32 = 6u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        // Find pivot (simple, no swapping)
        let pivot: f32 = M[i * n + i]; // M[i][i]
        if (abs(pivot) < 1e-10) { 
            return Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); // Singular
        }

        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let fac: f32 = M[r * n + i] / pivot; // M[r][i] / pivot
            for (var c: u32 = i; c < n; c = c + 1u) {
                M[r * n + c] = M[r * n + c] - fac * M[i * n + c]; // M[r][c] -= fac * M[i][c]
            }
            v[r] = v[r] - fac * v[i];
        }
    }

    // Back substitution
    var x: Vec6;
    for (var i_s: i32 = 5; i_s >= 0; i_s = i_s - 1) {
        let i: u32 = u32(i_s);
        var s: f32 = v[i];
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            s = s - M[i * n + j] * x[j]; // s -= M[i][j] * x[j]
        }
        x[i] = s / M[i * n + i]; // x[i] = s / M[i][i]
    }
    return x;
}

// Invert 3x3
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

// Extract cell from 6 params (G* -> G -> cell) + Basic Filters
fn extractCell(params: Vec6) -> RawSolution {
    let p1: f32 = params[0];
    let p2: f32 = params[1];
    let p3: f32 = params[2];
    let p4: f32 = params[3];
    let p5: f32 = params[4];
    let p6: f32 = params[5];

    // Filter 0: Ensure reciprocal params are positive
    if (p1 <= 1e-6 || p2 <= 1e-6 || p3 <= 1e-6) {
        return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0);
    }

    let G_star = mat3x3<f32>(
        vec3<f32>(p1, p6/2.0, p5/2.0),
        vec3<f32>(p6/2.0, p2, p4/2.0),
        vec3<f32>(p5/2.0, p4/2.0, p3)
    );
    
    let G = invert3x3(G_star);
    // Filter 1: Invert failed (singular) or non-positive definite
    if (G[0][0] <= 1e-6 || G[1][1] <= 1e-6 || G[2][2] <= 1e-6) { 
        return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0);
    }

    let a: f32 = sqrt(G[0][0]);
    let b: f32 = sqrt(G[1][1]);
    let c: f32 = sqrt(G[2][2]);

    // Filter 2: Plausible params
    if (a < 2.0 || a > 50.0 || (a != a)) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }
    if (b < 2.0 || b > 50.0 || (b != b)) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }
    if (c < 2.0 || c > 50.0 || (c != c)) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }

    let alpha_cos = clamp(G[1][2] / (b*c), -1.0, 1.0);
    let beta_cos = clamp(G[0][2] / (a*c), -1.0, 1.0);
    let gamma_cos = clamp(G[0][1] / (a*b), -1.0, 1.0);

    let alpha = acos(alpha_cos) * DEG;
    let beta = acos(beta_cos) * DEG;
    let gamma = acos(gamma_cos) * DEG;

    // Filter 3: Plausible angles
    if (alpha < 60.0 || alpha > 150.0 || (alpha != alpha)) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }
    if (beta < 60.0 || beta > 150.0 || (beta != beta)) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }
    if (gamma < 60.0 || gamma > 150.0 || (gamma != gamma)) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }

    // Filter 4: Volume (using config)
    let V_star_sq = determinant(G_star);
    if (V_star_sq <= 1e-10) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0); }
    let volume = 1.0 / sqrt(V_star_sq);
    if (volume < 20.0 || volume > config.max_volume) {
        return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0);
    }
    
    return RawSolution(a, b, c, alpha, beta, gamma);
}


// <-- CHANGED: This function now calculates the MEAN SQUARED NORMALIZED ERROR -->
fn validate_fom_avg_diff(p: Vec6) -> f32 {

    var sum_of_squared_normalized_diffs: f32 = 0.0;
    
    // Loop over the first N peaks (e.g., N=20)
    for (var i: u32 = 0u; i < config.n_peaks_for_fom; i = i + 1u) {
        
        let q_obs_val = q_obs[i];
        
        // Read pre-calculated tolerance
        let tol = q_tolerances[i]; 

        var min_diff: f32 = 1e10; // A very large number
        
        // Inner loop: check all HKLs to find the best match
        for (var j: u32 = 0u; j < config.n_hkl_for_fom; j = j + 1u) {
            let h = hkl_basis[j * 4u + 0u];
            let k = hkl_basis[j * 4u + 1u];
            let l = hkl_basis[j * 4u + 2u];
            
            // Triclinic q_calc formula
            let q_calc = p[0]*h*h + p[1]*k*k + p[2]*l*l + p[3]*k*l + p[4]*h*l + p[5]*h*k;
            
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


// === Main Kernel ===
@compute @workgroup_size(4, WORKGROUP_SIZE_Y, 1)
fn main(
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
    let num_peak_combos = arrayLength(&peak_combos) / 6u;
    let num_hkl_combos = arrayLength(&hkl_combos) / 6u;

    if (peak_combo_idx >= num_peak_combos || 
        hkl_combo_idx >= num_hkl_combos) {
        return;
    }

    // 2. Get q_obs base vector
    let p_offset = peak_combo_idx * 6u;
    let q_base = Vec6(
        q_obs[peak_combos[p_offset + 0u]],
        q_obs[peak_combos[p_offset + 1u]],
        q_obs[peak_combos[p_offset + 2u]],
        q_obs[peak_combos[p_offset + 3u]],
        q_obs[peak_combos[p_offset + 4u]],
        q_obs[peak_combos[p_offset + 5u]]
    );

    // 3. Get HKLs and build M matrix
    let h_offset = hkl_combo_idx * 6u;
    var M: Mat6x6;
    var hkl_indices: array<u32, 6>; // For debug
    
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let hkl_idx = hkl_combos[h_offset + i];
        hkl_indices[i] = hkl_idx; // Store for debug
        
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

    // 4. Loop over all 720 permutations
    for(var p_idx: u32 = 0u; p_idx < 720u; p_idx = p_idx + 1u) {
        
        let perm_offset = p_idx * 6u;
        
        // 5. Build permuted q_vec
        let q_perm = Vec6(
             q_base[PERMUTATIONS_6[perm_offset + 0u]], 
             q_base[PERMUTATIONS_6[perm_offset + 1u]], 
             q_base[PERMUTATIONS_6[perm_offset + 2u]],
             q_base[PERMUTATIONS_6[perm_offset + 3u]],
             q_base[PERMUTATIONS_6[perm_offset + 4u]],
             q_base[PERMUTATIONS_6[perm_offset + 5u]]
        );
         
        // 6. STAGE 1: Solve the 6x6 system
        let fit_params = solve6x6(M, q_perm);
         
        // 7. STAGE 2: Basic Boolean Filters
        let cell = extractCell(fit_params);
         
        // 8. STAGE 3: Strong "True FoM" Filter
        if (cell.a > 0.0) { // Passed basic filters
            
            // <-- CHANGED: This variable is now avg_squared_norm_error
            let avg_squared_norm_error = validate_fom_avg_diff(fit_params);
            
            // <-- CHANGED: Compare against the (new) FOM_THRESHOLD
            if (avg_squared_norm_error < FOM_THRESHOLD) {
            
                // 9. Store result (passed ALL tests)
                let idx = atomicAdd(&solution_counter, 1u);
                if (idx < MAX_SOLUTIONS) {
                    results_list[idx] = cell;
                }
                
                // 10. Write to debug log
                let debug_idx = atomicAdd(&debug_counter, 1u);
                if (debug_idx < MAX_DEBUG_CELLS) {
                    let debug_cell_offset = debug_idx * 30u; // 6 peaks * 5 floats
                    
                    for(var k_log: u32 = 0u; k_log < 6u; k_log = k_log + 1u) {
                        let log_offset = debug_cell_offset + (k_log * 5u);
                        let hkl_idx = hkl_indices[k_log];
                        let q_obs_val = q_perm[k_log];
                        let q_calc = (fit_params[0]*M[k_log*6u+0u] +
                                      fit_params[1]*M[k_log*6u+1u] +
                                      fit_params[2]*M[k_log*6u+2u] +
                                      fit_params[3]*M[k_log*6u+3u] +
                                      fit_params[4]*M[k_log*6u+4u] +
                                      fit_params[5]*M[k_log*6u+5u]);
                    
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