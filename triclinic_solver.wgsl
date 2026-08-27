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
// PRECOMPUTED hkl products, NOT raw indices. TWO vec4 per reflection:
//     [2*i + 0] = vec4(h*h, k*k, l*l, k*l)
//     [2*i + 1] = vec4(h*l, h*k, 0, 0)
// packed on the JS side (see HKL_PACKERS in main_app.js). This is the only
// system whose stride changes (4 -> 8 floats per reflection); the engine reads
// it from cfg.hklFloats. The FoM inner loop no longer recomputes six products
// per candidate cell, which is where this shader spends nearly all its time.
// The products are small integers (|h|,|k|,|l| <= 5) and exact in f32.
@group(0) @binding(1) var<storage, read> hkl_basis: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i,j,k,l,m,n]

// Replaced massive hkl_combos with Pascal's Lookup
@group(0) @binding(3) var<storage, read> binomial_table: array<u32>; 

@group(0) @binding(4) var<storage, read_write> solution_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawSolution>;



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
const WORKGROUP_SIZE_Y: u32 = 4u;
const MAX_Y_WORKGROUPS: u32 = 16383u; 
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
// It is replaced by a factor-once / substitute-many pair below. The split is
// sound for the same reason it always was: everything the factorisation
// decides -- the elimination multipliers
//     fac = M[r*n+i] / pivot
// and now also the pivot row -- depends only on M, never on the RHS. So it is
// still computed once and replayed 720 times.
//
// It is NO LONGER bitwise identical to the original solve6x6, and deliberately
// so: factor6x6 now does partial pivoting, which the original did not. See the
// FIX note on factor6x6 -- the unpivoted version was silently discarding about
// half of all non-singular hkl combinations.

// Reduce A to upper-triangular U, recording the 15 elimination multipliers AND
// the row chosen at each pivot step.
//
// --- FIX: this had no partial pivoting, and that was throwing away half the
// --- triclinic search space -----------------------------------------------
// solve4x4 in monoclinic_solver.wgsl pivots. This did not: it took U[i*n+i] as
// the pivot and gave up the moment that entry was near zero. A row of this
// matrix is (h^2, k^2, l^2, k*l, h*l, h*k), which is FULL of structural zeros
// -- the very first basis reflection is (0,0,1), whose row is
// [0,0,1,0,0,0] -- so a zero on the diagonal is the normal case here, not a
// degenerate one. Rejecting on it is not a singularity test, it is an artefact
// of the row ORDER the combinadic unranker happened to produce.
//
// Measured over 200k random 6-combinations of the default 123-reflection
// triclinic basis: 9.2% of the systems are genuinely singular, but the
// unpivoted factorisation rejected 61.5% of them. That is 52.3% of ALL
// combinations -- perfectly well-conditioned systems -- discarded silently,
// with no error and no counter anywhere. Every one of them was a cell the
// search could have found and did not. With partial pivoting, 0 out of 50k
// non-singular combinations are rejected.
//
// This deliberately breaks the "bitwise identical to the old solve6x6" property
// the comment above claims, because the behaviour it was preserving was wrong.
// The factor-once/substitute-many split itself is untouched and still correct:
// the pivot choice, like the multipliers, depends only on M and never on the
// RHS, so it is still computed once for all 720 permutations.
//
// Returns false only when the matrix is ACTUALLY singular (no usable pivot in
// the whole column), in which case every permutation would have produced a zero
// vector anyway and the caller can abandon the combination.
fn factor6x6(A: Mat6x6, U: ptr<function, Mat6x6>, facs: ptr<function, array<f32, 15>>,
             piv: ptr<function, array<u32, 6>>) -> bool {
    (*U) = A;
    let n: u32 = 6u;
    var fi: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        // Partial pivoting: take the largest-magnitude candidate in column i.
        // Picking the largest rather than merely the first non-zero also keeps
        // the multipliers <= 1, which matters at f32 precision.
        var best_row: u32 = i;
        var best_val: f32 = abs((*U)[i * n + i]);
        for (var r: u32 = i + 1u; r < n; r = r + 1u) {
            let v = abs((*U)[r * n + i]);
            if (v > best_val) { best_val = v; best_row = r; }
        }
        if (best_val < 1e-10) { return false; }   // genuinely singular

        (*piv)[i] = best_row;
        if (best_row != i) {
            for (var c: u32 = 0u; c < n; c = c + 1u) {
                let t = (*U)[i * n + c];
                (*U)[i * n + c] = (*U)[best_row * n + c];
                (*U)[best_row * n + c] = t;
            }
        }

        let pivot: f32 = (*U)[i * n + i];
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

// Apply the stored row swaps and multipliers to one RHS, then back-substitute.
// The swap for step i must be replayed BEFORE that step's eliminations, in the
// same order factor6x6 performed them.
fn substitute6x6(U: ptr<function, Mat6x6>, facs: ptr<function, array<f32, 15>>,
                 piv: ptr<function, array<u32, 6>>, b: Vec6) -> Vec6 {
    var v: Vec6 = b;
    let n: u32 = 6u;
    var fi: u32 = 0u;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let p = (*piv)[i];
        if (p != i) { let t = v[i]; v[i] = v[p]; v[p] = t; }
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
    let min_ax = config.f_params2.x;
    let max_ax = config.f_params2.y;
    if (a < min_ax || a > max_ax || b < min_ax || b > max_ax || c < min_ax || c > max_ax) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0); }

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
    
    if (volume < config.f_params2.z || volume > config.f_params.z) { return RawSolution(0.0,0.0,0.0,0.0,0.0,0.0,0.,0.); }
    
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
    // Split the 6 fit parameters to match the packed hkl layout:
    //   pA . (h^2, k^2, l^2, k*l)  +  pB . (h*l, h*k)
    let pA = vec4<f32>(p[0], p[1], p[2], p[3]);
    let pB = vec2<f32>(p[4], p[5]);
    let n_basis = config.u_params1.w;

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
            
            for (var j: u32 = 0u; j < n_basis; j = j + 1u) {
                let q_calc = dot(pA, hkl_basis[j * 2u]) + dot(pB, hkl_basis[j * 2u + 1u].xy);
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
    let count_to_sum = n_peaks_to_check - max_imp; // guarded above, cannot underflow
    let max_allowed_total = config.f_params.w * f32(count_to_sum);

    // Fail-fast for the impurity path. The no-impurity path has had one for a
    // while; this one had to score all 32 peaks before it could reject
    // anything, which is the expensive half of the run whenever
    // impurity_peaks > 0 -- and in this shader the FoM is called up to 720
    // times per hkl combination.
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
            let q_calc = dot(pA, hkl_basis[j * 2u]) + dot(pB, hkl_basis[j * 2u + 1u].xy);
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
            if ((sum_all - top_sum) > max_allowed_total) { return 999.0; }
        }
    }

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

    // 4. Build M Matrix. Each basis entry already holds the six products across
    //    two vec4s, so a row is two vector loads and no arithmetic.
    var M: Mat6x6;
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let v0 = hkl_basis[hkl_indices[i] * 2u];
        let v1 = hkl_basis[hkl_indices[i] * 2u + 1u];
        let row_offset = i * 6u;
        M[row_offset + 0u] = v0.x;   // h*h
        M[row_offset + 1u] = v0.y;   // k*k
        M[row_offset + 2u] = v0.z;   // l*l
        M[row_offset + 3u] = v0.w;   // k*l
        M[row_offset + 4u] = v1.x;   // h*l
        M[row_offset + 5u] = v1.y;   // h*k
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
    var lu_piv: array<u32, 6>;
    if (!factor6x6(M, &U_lu, &lu_facs, &lu_piv)) { return; }

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
         
        let fit_params = substitute6x6(&U_lu, &lu_facs, &lu_piv, q_perm);
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