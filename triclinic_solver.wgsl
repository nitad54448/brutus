// triclinic_solver.wgsl
// 6-Peak Direct Solve + Combinadics + Optimized FoM (Abs Diff)

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

// Replaced massive hkl_combos with Pascal's Lookup
@group(0) @binding(3) var<storage, read> binomial_table: array<u32>; 

@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawSolution>;



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
const WORKGROUP_SIZE_Y: u32 = 4u;
const MAX_Y_WORKGROUPS: u32 = 16383u; 
const MAX_DEBUG_CELLS: u32 = 10u;
const MAX_FOM_PEAKS: u32 = 32u; 

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

// solve6x6 is GONE. It used to be called 720x per invocation with an
// IDENTICAL M and only the right-hand side changing, and it took M *by value*
// (`var M: Mat6x6 = A`) - so each call copied 36 floats into private memory and
// redid the whole O(n^3) elimination from scratch. That is ~25,900 redundant
// float copies plus 719 redundant factorisations per invocation.
//
// It is replaced by a factor-once / substitute-many pair below. This is
// bitwise identical, not merely equivalent: the elimination multiplier
//     fac = M[r*n+i] / pivot
// depends only on M, never on the RHS, and so does the pivot-singularity test.
// Storing the 15 multipliers and replaying `v[r] -= fac * v[i]` in the same
// nested order performs the exact same float ops on v, in the exact same
// sequence, as the original did. Back-substitution then runs against the same
// reduced U. Same bits out.

// Reduce A to upper-triangular U, recording the 15 elimination multipliers.
// Returns false if a pivot is singular - in which case the ORIGINAL code would
// have returned a zero vector for every one of the 720 permutations, so the
// caller can bail out of the whole permutation loop at once.
fn factor6x6(A: Mat6x6, U: ptr<function, Mat6x6>, facs: ptr<function, array<f32, 15>>) -> bool {
    (*U) = A;
    let n: u32 = 6u;
    var fi: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let pivot: f32 = (*U)[i * n + i];
        if (abs(pivot) < 1e-10) { return false; }
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let fac: f32 = (*U)[r * n + i] / pivot;
            (*facs)[fi] = fac;
            fi = fi + 1u;
            for (var c: u32 = i; c < n; c = c + 1u) {
                (*U)[r * n + c] = (*U)[r * n + c] - fac * (*U)[i * n + c];
            }
        }
    }
    return true;
}

// Apply the stored multipliers to one RHS, then back-substitute.
fn substitute6x6(U: ptr<function, Mat6x6>, facs: ptr<function, array<f32, 15>>, b: Vec6) -> Vec6 {
    var v: Vec6 = b;
    let n: u32 = 6u;
    var fi: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            v[r] = v[r] - (*facs)[fi] * v[i];
            fi = fi + 1u;
        }
    }
    var x: Vec6;
    for (var i_s: i32 = 5; i_s >= 0; i_s = i_s - 1) {
        let i: u32 = u32(i_s);
        var s: f32 = v[i];
        for (var j: u32 = i + 1u; j < n; j = j + 1u) {
            s = s - (*U)[i * n + j] * x[j];
        }
        x[i] = s / (*U)[i * n + i];
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
    
    if (volume < 20.0 || volume > config.f_params.z) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.,0.); }
    
    return RawSolution(a, b, c, alpha, beta, gamma,0.,0.);
}

// === Combinatorial Number System Helper ===
// Calculates the k-th combination [c6...c1] for linear index 'm'
fn get_combinadic_indices(linear_index: u32, n_max: u32) -> array<u32, 6> {
    var m = linear_index;
    var out: array<u32, 6>;
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

// === Optimized FoM Validator (Abs Diff) ===
fn validate_fom_avg_diff(p: Vec6) -> f32 {
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
    
    // --- OPTIMIZATION: If no impurities, skip sorting entirely ---
    if (max_imp == 0u) {
        var sum_abs_error: f32 = 0.0;
        
        // Fail-Fast Threshold
        let max_allowed_total = config.f_params.w * f32(n_peaks_to_check);

        for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
            let q_obs_val = q_obs[i];
            let tol = q_tolerances[i]; 
            var min_diff: f32 = 1e10; 
            
            for (var j: u32 = 0u; j < config.u_params1.w; j = j + 1u) {
                let h = hkl_basis[j * 4u + 0u];
                let k = hkl_basis[j * 4u + 1u];
                let l = hkl_basis[j * 4u + 2u];
                let q_calc = p[0]*h*h + p[1]*k*k + p[2]*l*l + p[3]*k*l + p[4]*h*l + p[5]*h*k;
                let diff = abs(q_obs_val - q_calc);
                if (diff < min_diff) { min_diff = diff; }
            }
            let norm = min_diff / tol;
            sum_abs_error += norm;
            
            // Fail-Fast Check
            if (sum_abs_error > max_allowed_total) { return 999.0; }
        }
        let avg = sum_abs_error / f32(n_peaks_to_check);
        return avg;
    } 

    // --- IMPURITY PATH (Partial Sort) ---
    var errors: array<f32, 32>;
    for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
        let q_obs_val = q_obs[i];
        let tol = q_tolerances[i]; 
        var min_diff: f32 = 1e10; 
        
        for (var j: u32 = 0u; j < config.u_params1.w; j = j + 1u) {
            let h = hkl_basis[j * 4u + 0u];
            let k = hkl_basis[j * 4u + 1u];
            let l = hkl_basis[j * 4u + 2u];
            let q_calc = p[0]*h*h + p[1]*k*k + p[2]*l*l + p[3]*k*l + p[4]*h*l + p[5]*h*k;
            let diff = abs(q_obs_val - q_calc);
            if (diff < min_diff) { min_diff = diff; }
        }
        let norm = min_diff / tol;
        // Using absolute difference
        errors[i] = norm;
    }

    let count_to_sum = n_peaks_to_check - max_imp; // guarded above, cannot underflow
    var sum_of_valid_errors: f32 = 0.0;

    for (var i: u32 = 0u; i < count_to_sum; i = i + 1u) {
        var min_val = errors[i];
        var min_idx = i;
        for (var j: u32 = i + 1u; j < n_peaks_to_check; j = j + 1u) {
            if (errors[j] < min_val) {
                min_val = errors[j];
                min_idx = j;
            }
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
@compute @workgroup_size(4, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Fix: config.u_params2.z (max_solutions)
    if (atomicLoad(&solution_counter) >= config.u_params2.z) { return; }
    
    // 1. Calculate Global Indices
    let peak_combo_idx: u32 = global_id.x;
    
    // Fix: config.u_params1.x (z_offset)
    let hkl_linear_idx: u32 = config.u_params1.x + global_id.y;

    // Fix: config.u_params2.y (total_hkl_combos)
    if (hkl_linear_idx >= config.u_params2.y) { return; }

    // 2. Bounds Check
    let num_peak_combos = arrayLength(&peak_combos) / 6u;
    if (peak_combo_idx >= num_peak_combos) { return; }

    // 3. Combinadics: Generate HKL Indices on the fly
    let hkl_indices = get_combinadic_indices(hkl_linear_idx, config.u_params2.x);

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

    // 5b. Factor M ONCE. M does not depend on the permutation - only the RHS
    // does - so the elimination that used to run 720 times now runs once.
    // If M is singular, every permutation would have yielded a zero vector and
    // been rejected by extractCell, so we can abandon this combo entirely.
    var U_lu: Mat6x6;
    var lu_facs: array<f32, 15>;
    if (!factor6x6(M, &U_lu, &lu_facs)) { return; }

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
         
        let fit_params = substitute6x6(&U_lu, &lu_facs, q_perm);
        let cell = extractCell(fit_params);
         
        if (cell.a > 0.0) { 
            let avg_err = validate_fom_avg_diff(fit_params);
            
            if (avg_err < config.f_params.w) {
                // Atomic Add to Global Counter
                let idx = atomicAdd(&solution_counter, 1u);
                
                if (idx < config.u_params2.z) {
                    results_list[idx] = cell;
                }
                // Debug log removed (was: writes to debug_log[] on every accepted cell,
                // never read back on the JS side). Same cleanup as ortho had.
                break; // Stop checking permutations for this combo
            }
        }
    } 
}