# Changes — 6 October 2026 (v=20261006)

## Before you deploy

- Copy back the files you removed: `scripts/` (Chart.js, Hammer, the zoom
  plugin, jsPDF), the fonts, and `sg_ops.json`. Nothing that references them
  has changed.
- Delete the old top-level `main_app.js`, `worker-logic.js`, `webgpu-engine.js`,
  `refinement-worker.js` and `*_solver.wgsl`; they now live under `js/` and
  `shaders/`.
- `test_sg_ops.mjs` and any `tests/*.cjs` (not in the package I received)
  probably load `worker-logic.js` by path. Load the crystallography files in
  manifest order instead, as `check_sg_ops.mjs` now does:

  ```js
  const manifest = readFileSync('./js/crystallography/manifest.js', 'utf8');
  for (const m of manifest.matchAll(/^\s*'([^']+\.js)',\s*$/gm))
      vm.runInContext(readFileSync('./js/crystallography/' + m[1], 'utf8'), ctx);
  ```

## Layout

| Was | Now |
|---|---|
| `main_app.js` (7,800 lines, one closure) | 42 files under `js/core`, `js/data`, `js/parsers`, `js/peaks`, `js/chart`, `js/gpu`, `js/indexing`, `js/dialogs`, `js/report`, plus `js/main.js` |
| `worker-logic.js` | 15 files under `js/crystallography/`, listed in load order in `manifest.js` |
| CPU-worker handler at the bottom of `worker-logic.js` | `js/workers/index-worker.js` (the `IS_REFINEMENT_WORKER` guard is gone) |
| `refinement-worker.js`, `webgpu-engine.js` | `js/workers/`, `js/gpu/` |
| `*_solver.wgsl` | `shaders/` |

The files are classic scripts sharing one global scope, loaded with `defer` in
the order `brutus.html` lists them, so the code itself was moved, not
rewritten. A parser-based check confirmed every original top-level
statement appears exactly once in the new files (335 from `main_app.js`,
172 from `worker-logic.js`), with multi-line template strings byte-identical.

## Bugs fixed

- **Startup without WebGPU** (Firefox, older Safari, plain `http://` on a LAN
  address): `checkWebGPUCapabilities()` ran before `const showStatus` existed,
  threw `ReferenceError`, never showed the warning and left the GPU systems
  enabled. Startup calls now run last (`js/main.js`). Reproduced and confirmed
  fixed in Chromium with `navigator.gpu` removed.
- **Kα1 files read as Kα-average**: the preset match used a 0.005 Å tolerance,
  wider than the 0.0013 Å Kα1/average gap, and tested the average first. A
  pdCIF declaring 1.5406 Å was loaded at 1.54184 Å (0.08 % on every d). Now the
  nearest line within 3·10⁻⁴ Å wins (`classifyFileWavelength`,
  `js/data/wavelength.js`).
- **Lab doublet data loaded as monochromatic**: XRDML, Bruker XML, UXD, RAS
  and UDF read only Kα1, so Kα₂ stripping and Kα₂ ghost flags were disabled for
  ordinary Cu-tube scans. All of them now read Kα2, the ratio and (XRDML) the
  `intended` attribute, and select the Kα-average or Kα1 preset accordingly.
- **Readers**:
  - XRDML: `listPositions` (non-equidistant) scans, and intensities and
    positions taken from the same scan.
  - UXD: `KEY = value` with spaces, `_CPS`, `_2THETACOUNTS`, and several ranges.
    Range 2 used to be appended with range 1's start and step.
  - RAS: quoted header values.
  - pdCIF: `_pd_meas_counts_total`, scans given as `_pd_meas_2theta_range_*`,
    s.u.'s in parentheses, and loop rows that wrap across lines.
  - Two-column files: a `Wavelength …` header line is read.
    `test_files/C61Br2_079764.XY` (λ = 0.79764 Å) used to load as Cu and found
    nothing; it now indexes as tetragonal, a = 9.438 Å, c = 13.347 Å, M(20) = 1778.
- **Descending scans** had trailing zeros trimmed at the wrong end; the scan is
  now sorted first.
- **GPU buffer leak**: `_runSolver` allocated every buffer before checks that
  could throw outside its `try/finally`. All checks now run before allocation
  and every buffer is tracked and destroyed.
- **Monoclinic uncertainties**: σ(a), σ(c) used `a/(2A)·σ_A`, low by sin²β (25 %
  at β = 120°) and missing the C and D covariance. Now a numerical Jacobian over
  the full covariance. Against Monte Carlo: σ(a) 2.53e-4 (MC 2.53e-4, old 2.15e-4),
  σ(c) 6.47e-4 (MC 6.42e-4, old 5.23e-4).
- **Raw numeric reads**: wavelength, 2θ error, max volume, FoM, candidates and
  peak count went straight from the input box into the search; min/max only
  applied on blur, and the FoM fallback was 0.8 while the field says 1.5. One
  reader (`readNumberInput`, `js/core/inputs.js`) now parses, defaults from the
  markup and clamps for every consumer, and the blur handler uses it too.
- Ag Kα2 / Kα-average corrected (0.56380 / 0.56087 Å); broken `K&` in the
  data-formats help text; leftover `[cite: 2]` markers and the out-of-date
  description of the Kα₂ stripping removed; unused `getOrthogonalityScore`
  removed.

## Improvements

- **GPU errors are reported instead of producing "no solutions"**: shader
  compile errors (`getCompilationInfo`), pipeline errors
  (`createComputePipelineAsync`), allocation/binding errors (error scopes, plus a
  clear message when the candidate buffer exceeds the device's binding limit)
  and errors during the run (`uncapturederror`). `createPipeline` is now async.
- **High-performance GPU** requested first on dual-GPU machines.
- **Triclinic shader**: the 4,320-entry permutation table, indexed at runtime,
  is replaced by generating the same lexicographic sequence in place. Output is
  bit-identical (same 12,172 candidate cells in the A/B test); the gain depends
  on how a backend lowered the table, so time it on your GPUs.
- **Refinement workers are reused** across runs (re-initialised in place, only
  busy workers replaced) instead of respawned, and the pool is capped at 12
  (`REFINE_POOL_MAX` in `js/indexing/worker-pool.js`; raise it if refinement is
  your bottleneck on a many-core machine).
- **Exports record the doublet**: XRDML / UXD / Bruker XML saved from an
  unstripped Kα-average session carry Kα1, Kα2 and the ratio, so they reload
  with the same preset.
- **Version stamping covers everything**: shaders and both workers inherit the
  `?v=` of `js/core/version.js`; `bump_version.py` updated.

## New tools

- `check_load_order.mjs` — static check that nothing executed while the
  scripts load uses a name a later script defines (the class of the startup
  bug), that `brutus.html` and the worker manifest agree, and that no file is
  orphaned. Needs `npm install --no-save typescript`. It flags the original
  startup bug and catches planted ones.
- `regression_test.mjs` — Node tests on synthetic data: cubic, tetragonal and
  hexagonal searches through the real CPU worker, orthorhombic refinement,
  Niggli reduction, and error propagation vs Monte Carlo (the monoclinic check
  fails on the old code). About three minutes.
- `check_pipeline.mjs`, `check_sg_ops.mjs`, `bump_version.py` updated for the
  new paths.

## How this was verified

Headless Chromium with software WebGPU (SwiftShader), original versus new, on
the same inputs:

- GPU kernels: identical candidate cells for orthorhombic (4,402) and triclinic
  (12,172); monoclinic identical diagnostics.
- PbSO4 (GPU orthorhombic plus CPU systems): the same three solutions (best
  a = 6.9690, b = 5.4056, c = 8.4903 Å, M(20) = 58.74), identical pdCIF export,
  same 9-page report. Monoclinic and triclinic runs: same outcome.
- Synthetic tetragonal pattern: same solutions, and the same results from
  Refine MC, Swap hkl, pdCIF export, PDF report, unload and every Save-as format
  (except the intended doublet records in XRDML/UXD/Bruker).
- No page errors or console errors in any run.

Not verified: real GPUs (SwiftShader timings say nothing about speed), the
space-group analysis paths (no `sg_ops.json` here; that code moved verbatim),
and Bruker `.brml`-derived XML beyond the structure the reader already
expected.
