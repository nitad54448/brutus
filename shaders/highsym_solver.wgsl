// highsym_solver.wgsl
// Cubic (1 unknown) and Tetragonal / Hexagonal (2 unknowns)
// Direct Solve + Combinadics + Optimized FoM (Abs Diff) + Fail-Fast.
//
// One module, three entry points (9 Oct 2026). These three systems used to run
// on the CPU index worker, which refined EVERY trial cell with a full
// least-squares fit. Here, as for ortho/mono/tri, each trial is solved
// exactly from K peaks and K basis reflections and only the cells that pass
// the GPU figure of merit are sent to the CPU refinement pool.
//
//   entry point          K   q(hkl)                     basis vec4 (.xy used)
//   main_cubic           1   A N                        (N, 0, 0, 0)
//   main_tetragonal      2   A S + C l^2                (h^2+k^2, l^2, 0, 0)
//   main_hexagonal       2   A S + C l^2                (h^2+hk+k^2, l^2, 0, 0)
//
// with N = h^2+k^2+l^2, A = 1/a^2 (cubic, tetragonal), A = 4/(3a^2)
// (hexagonal), C = 1/c^2. All products are small integers, exact in f32, so
// the 2x2 determinant is an exact integer too.
//
// Rhombohedral (R) lattices are found by the hexagonal search: every R
// lattice is indexed by the hexagonal triple cell, which is also how the rest
// of the program -- refinement, space groups (R centring), reports -- handles
// them. (A separate primitive-rhombohedral search existed briefly and was
// removed as redundant.)

// === Structs ===
// p0 = a, p1 = c (cubic: p1 = a), p2/p3 unused (keeps the 16-byte stride)
struct RawHighSymSolution {
    p0: f32,
    p1: f32,
    p2: f32,
    p3: f32,
}

// === Type Aliases ===
alias Vec2 = vec2<f32>;

// === Bindings ===
// Identical to the other three solvers, so the engine's single explicit
// bind-group layout serves all of them.
@group(0) @binding(0) var<storage, read> q_obs: array<f32>;

// PRECOMPUTED hkl products, NOT raw indices -- see the table above and
// HKL_PACKERS in js/indexing/run.js. 16 bytes per reflection.
//
// Unlike ortho/mono/tri, the buffer may hold MORE reflections than the
// combinadic search uses: the first config.u_params2.x entries are the search
// basis, and the FoM scores against the first config.u_params1.w entries (the
// whole list). With one or two unknowns the trial count is tiny, so scoring
// against the full list costs little and stops a correct cell with a long axis
// failing the FoM merely because its deep lines were truncated away.
@group(0) @binding(1) var<storage, read> hkl_basis: array<vec4<f32>>;

@group(0) @binding(2) var<storage, read> peak_combos: array<u32>; // [i] or [i,j]

// Pascal's triangle for the K = 2 unranking (stride 3). Bound but unused by
// main_cubic, whose combination index IS the basis index.
@group(0) @binding(3) var<storage, read> binomial_table: array<u32>;

//   [0] solution count                                     atomicAdd
//   [1] most peaks any candidate kept inside the error
//       budget before the FoM fail-fast gave up (0..32)    atomicMax, init 0
//   [2] smallest cell volume, A^3, among cells that passed
//       the AXIS test -- recorded BEFORE the volume gate   atomicMin, init MAX
//   [3] largest such volume                                atomicMax, init 0
//   [4] first HKL-combination index left incomplete
//       because the candidate buffer was full          atomicMin, init MAX
//   [5..7] reserved
// (Same contract as the other solvers; see ortho_solver.wgsl for the reasons
// these are ranges and not counters.)
@group(0) @binding(4) var<storage, read_write> solution_counter: array<atomic<u32>, 8>;
@group(0) @binding(5) var<storage, read_write> results_list: array<RawHighSymSolution>;

// Slot [4]: the smallest HKL-combination index whose work was NOT completed
// because the candidate buffer was already full (a thread that early-outed, or
// an accepted cell that found no free slot). Every index below it was searched
// in full, so it is an exact lower bound on how far a truncated search got;
// the engine turns it into the "Trials: done / total" figure. Only touched
// once the buffer is full, never on the normal hot path.
fn mark_incomplete(hkl_linear_idx: u32) {
    if (hkl_linear_idx < atomicLoad(&solution_counter[4])) {
        atomicMin(&solution_counter[4], hkl_linear_idx);
    }
}

struct Config {
    // z_offset, max_impurities, n_peaks_for_fom, n_hkl_for_fom
    u_params1: vec4<u32>,
    // n_basis_total (search), total_hkl_combos, max_solutions, pad
    u_params2: vec4<u32>,
    // wavelength, tth_error, max_volume, fom_threshold
    f_params: vec4<f32>,
    // min_axis, max_axis, min_volume, pad
    f_params2: vec4<f32>
};
@group(0) @binding(6) var<uniform> config: Config;

@group(0) @binding(7) var<storage, read> q_tolerances: array<f32>;

// === Constants ===
const PI: f32 = 3.1415926535;
const DEG: f32 = 180.0 / PI;
const SQRT3_2: f32 = 0.8660254038;      // sin(120 deg): hexagonal cell area factor
const WORKGROUP_SIZE_Y: u32 = 8u;
const MAX_FOM_PEAKS: u32 = 32u;
const BINOMIAL_STRIDE_2: u32 = 3u;      // K = 2 -> columns 0..2

const SYS_TETRAGONAL: u32 = 1u;
const SYS_HEXAGONAL: u32 = 2u;

fn no_cell() -> RawHighSymSolution { return RawHighSymSolution(0.0, 0.0, 0.0, 0.0); }

// Axis gate, volume diagnostics, volume gate -- in that order, as in the
// other solvers. Returns true when the cell survives.
fn passes_limits(a: f32, c: f32, volume: f32) -> bool {
    let min_ax = config.f_params2.x;
    let max_ax = config.f_params2.y;
    if (!(a >= min_ax && a <= max_ax && c >= min_ax && c <= max_ax)) { return false; }

    let vol_diag = u32(clamp(volume, 0.0, 4.0e9));
    atomicMin(&solution_counter[2], vol_diag);
    atomicMax(&solution_counter[3], vol_diag);

    // Written as a positive test so a NaN volume is rejected too.
    return (volume >= config.f_params2.z && volume <= config.f_params.z);
}

fn extract_cubic(A: f32) -> RawHighSymSolution {
    if (!(A > 1e-12)) { return no_cell(); }
    let a = 1.0 / sqrt(A);
    if (!passes_limits(a, a, a * a * a)) { return no_cell(); }
    return RawHighSymSolution(a, a, 0.0, 0.0);
}

fn extract_two(x: Vec2, sys: u32) -> RawHighSymSolution {
    let A = x.x;
    let B = x.y;
    var a: f32;
    var volume: f32;

    if (!(A > 1e-12) || !(B > 1e-12)) { return no_cell(); }
    let c = 1.0 / sqrt(B);
    if (sys == SYS_HEXAGONAL) {
        a = sqrt(4.0 / (3.0 * A));
        volume = SQRT3_2 * a * a * c;
    } else {
        a = 1.0 / sqrt(A);
        volume = a * a * c;
    }

    if (!passes_limits(a, c, volume)) { return no_cell(); }
    return RawHighSymSolution(a, c, 0.0, 0.0);
}

// === Combinatorial Number System (K = 2) ===
// Same unranking as the other solvers, specialised to K = 2.
fn get_combinadic_indices_2(linear_index: u32, n_max: u32) -> vec2<u32> {
    var m = linear_index;
    var out: vec2<u32> = vec2<u32>(0u, 0u);
    var v = n_max - 1u;
    for (var k_idx: u32 = 2u; k_idx > 0u; k_idx = k_idx - 1u) {
        loop {
            let binom = binomial_table[v * BINOMIAL_STRIDE_2 + k_idx];
            if (binom <= m) {
                out[k_idx - 1u] = v;
                m = m - binom;
                if (v > 0u) { v = v - 1u; }
                break;
            }
            if (v == 0u) { break; }
            v = v - 1u;
        }
    }
    return out;
}

// === Optimized FoM (Absolute Difference) ===
// Identical contract to the other solvers: mean of |q_obs - q_calc| / tol over
// the first n_peaks_for_fom peaks, the max_impurities worst dropped, 999.0 for
// a rejection. q_calc = dot(x, basis.xy) for every system in this file
// (cubic passes x = (A, 0)).
fn validate_fom_avg_diff(x: Vec2) -> f32 {
    let n_peaks_to_check = min(config.u_params1.z, MAX_FOM_PEAKS);
    // Unsigned underflow guard (see ortho_solver.wgsl).
    if (n_peaks_to_check == 0u) { return 999.0; }
    let max_imp = min(config.u_params1.y, n_peaks_to_check - 1u);
    let n_basis = config.u_params1.w;

    // --- PATH A: no impurities, plain fail-fast sum ---
    if (max_imp == 0u) {
        var sum_abs_error: f32 = 0.0;
        let max_allowed_total = config.f_params.w * f32(n_peaks_to_check);
        var peaks_ok: u32 = 0u;

        for (var i: u32 = 0u; i < n_peaks_to_check; i = i + 1u) {
            let q_obs_val = q_obs[i];
            let tol = q_tolerances[i];
            var min_diff: f32 = 1e10;
            for (var j: u32 = 0u; j < n_basis; j = j + 1u) {
                let diff = abs(q_obs_val - dot(x, hkl_basis[j].xy));
                if (diff < min_diff) { min_diff = diff; }
            }
            sum_abs_error += min_diff / tol;
            if (sum_abs_error > max_allowed_total) {
                if (peaks_ok > atomicLoad(&solution_counter[1])) {
                    atomicMax(&solution_counter[1], peaks_ok);
                }
                return 999.0;
            }
            peaks_ok = peaks_ok + 1u;
        }
        atomicMax(&solution_counter[1], peaks_ok);
        return sum_abs_error / f32(n_peaks_to_check);
    }

    // --- PATH B: with impurities (drop the max_imp largest errors) ---
    let count_to_sum = n_peaks_to_check - max_imp; // guarded above
    let max_allowed_total = config.f_params.w * f32(count_to_sum);

    // Lower-bound fail-fast, as in the other solvers: (sum so far) - (sum of
    // the max_imp largest so far) never decreases, so exceeding the budget
    // early means exceeding it at the end.
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
            let diff = abs(q_obs_val - dot(x, hkl_basis[j].xy));
            if (diff < min_diff) { min_diff = diff; }
        }
        let norm = min_diff / tol;
        errors[i] = norm;

        if (use_fast_bound) {
            var min_i: u32 = 0u;
            var min_v: f32 = top[0];
            for (var t: u32 = 1u; t < max_imp; t = t + 1u) {
                if (top[t] < min_v) { min_v = top[t]; min_i = t; }
            }
            if (norm > min_v) { top_sum = top_sum - min_v + norm; top[min_i] = norm; }
            sum_all += norm;
            if ((sum_all - top_sum) > max_allowed_total) {
                if (i > atomicLoad(&solution_counter[1])) {
                    atomicMax(&solution_counter[1], i);
                }
                return 999.0;
            }
        }
    }
    atomicMax(&solution_counter[1], n_peaks_to_check);

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

fn emit(cell: RawHighSymSolution, hkl_linear_idx: u32) {
    let idx = atomicAdd(&solution_counter[0], 1u);
    if (idx < config.u_params2.z) {
        results_list[idx] = cell;
    } else {
        mark_incomplete(hkl_linear_idx);
    }
}

// === K = 2 kernel body, shared by the tetragonal and hexagonal entries ===
// `sys` is a literal at every call site, so the compiler can fold the branch.
fn solve_two(global_id: vec3<u32>, sys: u32) {
    if (atomicLoad(&solution_counter[0]) >= config.u_params2.z) {
        mark_incomplete(config.u_params1.x + global_id.y);
        return;
    }

    let peak_combo_idx: u32 = global_id.x;
    let hkl_linear_idx: u32 = config.u_params1.x + global_id.y;

    let num_peak_combos = arrayLength(&peak_combos) / 2u;
    if (peak_combo_idx >= num_peak_combos) { return; }
    if (hkl_linear_idx >= config.u_params2.y) { return; }

    let hkl_indices = get_combinadic_indices_2(hkl_linear_idx, config.u_params2.x);
    let r0 = hkl_basis[hkl_indices.x].xy;
    let r1 = hkl_basis[hkl_indices.y].xy;

    // Integer rows, so the determinant is an exact integer: < 0.5 is zero.
    // (Two hk0 lines, two 00l lines, or proportional rows: no unique cell.)
    let det = r0.x * r1.y - r0.y * r1.x;
    if (abs(det) < 0.5) { return; }
    let inv_det = 1.0 / det;

    let p_offset = peak_combo_idx * 2u;
    let q0 = q_obs[peak_combos[p_offset + 0u]];
    let q1 = q_obs[peak_combos[p_offset + 1u]];

    // Two assignments of the two peaks to the two reflections.
    for (var p_idx: u32 = 0u; p_idx < 2u; p_idx = p_idx + 1u) {
        var b = Vec2(q0, q1);
        if (p_idx == 1u) { b = Vec2(q1, q0); }
        // [r0; r1] x = b
        let x = Vec2(r1.y * b.x - r0.y * b.y, r0.x * b.y - r1.x * b.x) * inv_det;
        let cell = extract_two(x, sys);
        if (cell.p0 > 0.0) {
            let avg_err = validate_fom_avg_diff(x);
            if (avg_err < config.f_params.w) {
                emit(cell, hkl_linear_idx);
                break;
            }
        }
    }
}

// === Entry points ===

@compute @workgroup_size(8, WORKGROUP_SIZE_Y, 1)
fn main_cubic(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (atomicLoad(&solution_counter[0]) >= config.u_params2.z) {
        mark_incomplete(config.u_params1.x + global_id.y);
        return;
    }

    let peak_idx: u32 = global_id.x;
    // K = 1: the combination index is the basis index itself.
    let hkl_idx: u32 = config.u_params1.x + global_id.y;

    if (peak_idx >= arrayLength(&peak_combos)) { return; }
    if (hkl_idx >= config.u_params2.y) { return; }

    let N = hkl_basis[hkl_idx].x;
    if (N < 0.5) { return; }
    let A = q_obs[peak_combos[peak_idx]] / N;
    let cell = extract_cubic(A);
    if (cell.p0 > 0.0) {
        let avg_err = validate_fom_avg_diff(Vec2(A, 0.0));
        if (avg_err < config.f_params.w) {
            emit(cell, hkl_idx);
        }
    }
}

@compute @workgroup_size(8, WORKGROUP_SIZE_Y, 1)
fn main_tetragonal(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    solve_two(global_id, SYS_TETRAGONAL);
}

@compute @workgroup_size(8, WORKGROUP_SIZE_Y, 1)
fn main_hexagonal(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    solve_two(global_id, SYS_HEXAGONAL);
}
