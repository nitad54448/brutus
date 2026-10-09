// js/gpu/gpu-setup.js
// WebGPU availability, the shared engine, the per-system search table and the
// GPU search parameter controls.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// ------------------------------------------------------------------
// The crystal systems the indexer searches (9 Oct 2026).
//
// Every system now runs on the GPU, one after another in SEARCH_ORDER
// (highest symmetry first). The per-system "HKL Basis Size" and "Peaks to
// Combine" boxes are gone; both are derived here from two global settings:
//
//   HKL basis (% / unknown)  -- the search basis is K x this many percent of
//                               the system's maximum usable basis
//                               (hklBasisMax). K is the number of unknown
//                               cell parameters. At the default 5 this gives
//                               ortho 15% (329), mono 20% (114), tri 30% (37),
//                               i.e. the old hand-tuned defaults.
//   Depth                    -- peaks combined = peakFactor x (K + Depth):
//                               orthorhombic/monoclinic/triclinic K + Depth
//                               (Depth 3: 6, 7, 9 peaks); cubic, hexagonal,
//                               tetragonal 3 x (K + Depth)
//                               (Depth 3: 12 and 15 peaks).
//
// Rhombohedral (R) lattices are searched by the hexagonal system: every R
// lattice is indexed by its hexagonal triple cell, which is how refinement and
// the space-group analysis (R centring) handle them.
//
//   K             number of unknowns = peaks AND basis reflections per solve
//   permutations  K!, assignments of K peaks to K reflections
//   cpuFallback   searchable by the CPU index worker when WebGPU is missing
//                 (capabilities kept from the CPU era)
//   fomFullList   the GPU FoM scores against the WHOLE hkl list, not just
//                 the search basis (cheap at K <= 2; see highsym_solver.wgsl)
//   splitSpecialHkls  axial reflections are moved to the front of the basis
//   peakFactor    multiplier on K + Depth for the peaks combined. With one or
//                 two unknowns a search over 12-15 peaks costs almost nothing
//                 (at most C(15,2) = 105 peak pairs), and it is needed: when
//                 the first 4-5 lines all come from one zone (00l of a long c
//                 axis, hk0 of a long a axis) the second parameter cannot be
//                 solved from them at all. The old CPU search used 12 peaks.
// ------------------------------------------------------------------
const SEARCH_SYSTEMS = {
    cubic:        { K: 1, label: 'Cubic',        shortLabel: 'Cubic',  permutations: 1,   peakFactor: 3, cpuFallback: true,  fomFullList: true,  splitSpecialHkls: false },
    hexagonal:    { K: 2, label: 'Hexagonal',    shortLabel: 'Hex',    permutations: 2,   peakFactor: 3, cpuFallback: true,  fomFullList: true,  splitSpecialHkls: false },
    tetragonal:   { K: 2, label: 'Tetragonal',   shortLabel: 'Tetra',  permutations: 2,   peakFactor: 3, cpuFallback: true,  fomFullList: true,  splitSpecialHkls: false },
    orthorhombic: { K: 3, label: 'Orthorhombic', shortLabel: 'Ortho',  permutations: 6,   peakFactor: 1, cpuFallback: false, fomFullList: false, splitSpecialHkls: true },
    monoclinic:   { K: 4, label: 'Monoclinic',   shortLabel: 'Mono',   permutations: 24,  peakFactor: 1, cpuFallback: false, fomFullList: false, splitSpecialHkls: true },
    triclinic:    { K: 6, label: 'Triclinic',    shortLabel: 'Tri',    permutations: 720, peakFactor: 1, cpuFallback: false, fomFullList: false, splitSpecialHkls: false },
};
// Highest point-group order first: m-3m 48, 6/mmm 24, 4/mmm 16, mmm 8,
// 2/m 4, -1 2.
const SEARCH_ORDER = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'];
const isGpuOnlySystem = (system) => !!SEARCH_SYSTEMS[system] && !SEARCH_SYSTEMS[system].cpuFallback;
// Minimum peaks a refined cell must index (refineAndTestSolution's
// min_indexed); a search with fewer peaks than this cannot post anything.
const MIN_PEAKS_FOR_SYSTEM = { cubic: 4, hexagonal: 5, tetragonal: 5,
                               orthorhombic: 6, monoclinic: 7, triclinic: 7 };

/**
 * Asynchronously checks WebGPU compute capabilities on page load.
 * Disables and greys out the GPU-only systems if support is absent; cubic,
 * tetragonal and hexagonal stay available through the CPU worker.
 */
async function checkWebGPUCapabilities() {
    try {
        if ('gpu' in navigator) {
            // Reuse the shared engine. This probe used to construct its own
            // and abandon it, so the page held a spare GPUDevice from load
            // onwards that no run ever touched.
            await getWebGPUEngine(); //  critical test
            // If this line is reached, GPU is fine.
            webGPUSupportsCompute = true;
            console.log("WebGPU compute capabilities verified.");
        } else {
            throw new Error("WebGPU not found in navigator.");
        }
    } catch (err) {
        console.warn("WebGPU initialization failed:", err.message);
        // error message, permanent toast red warning
        showStatus("⚠ WebGPU is not initialized. Cubic, tetragonal and hexagonal run on the CPU; " +
                   "the other systems are disabled. See the Help file for details.", "error", 86400000);
        disableGpuOnlySystems();
    }
    toggleGpuParamsVisibility();
    updateStartIndexingButtonState();
}
// Mark WebGPU unusable and grey out every system that has no CPU path.
function disableGpuOnlySystems() {
    webGPUSupportsCompute = false;
    ui.systemCheckboxes.forEach(cb => {
        if (!isGpuOnlySystem(cb.value)) return;
        cb.disabled = true;
        cb.checked = false;
        const label = cb.parentElement;
        if (label) {
            label.style.opacity = '0.5';
            label.style.cursor = 'not-allowed';
        }
    });
}
let webGPUSupportsCompute = true;
// ------------------------------------------------------------------
// Shared WebGPU engine.
//
// Previously checkWebGPUCapabilities() built one engine (and one GPUDevice)
// at startup and startIndexing() built a fresh one on EVERY run, and
// device.destroy() was never called anywhere. Devices, their pipelines and
// their buffers therefore accumulated for the lifetime of the page; a long
// session would eventually exhaust the driver.
//
// One engine is created lazily, reused by every run, and rebuilt only if the
// device is genuinely lost. getWebGPUEngine() is the single entry point.
// ------------------------------------------------------------------
let sharedWebGPUEngine = null;
const releaseWebGPUEngine = () => {
    if (sharedWebGPUEngine) {
        try { sharedWebGPUEngine.destroy(); } catch (_) {}
        sharedWebGPUEngine = null;
    }
};
let webgpuInitPromise = null;
async function getWebGPUEngine() {
    // A lost device cannot be revived: its pipelines and buffers are gone.
    // Drop it and build a replacement.
    if (sharedWebGPUEngine && !sharedWebGPUEngine.isUsable()) {
        releaseWebGPUEngine();
    }
    if (sharedWebGPUEngine) return sharedWebGPUEngine;

    if (webgpuInitPromise) return webgpuInitPromise;
    webgpuInitPromise = (async () => {
    const engine = new WebGPUEngine();
    await engine.init();
    engine.onDeviceLost = (info) => {
        // Device loss used to be console-only, so a driver reset mid-run
        // just looked like an indexing run that quietly produced nothing.
        webGPUSupportsCompute = false;
        showStatus(
            `GPU device lost (${info && info.reason ? info.reason : 'unknown'}). ` +
            `Reload the page to re-enable GPU searches.`, 'error', 12000);
        gpuStopSignal.stop = true;
    };
    sharedWebGPUEngine = engine;
    return engine;
    })();
    try { return await webgpuInitPromise; }
    finally { webgpuInitPromise = null; }
}
// Release the device on navigation rather than relying on GC.
window.addEventListener('pagehide', releaseWebGPUEngine);


// --- HKL basis size limits, per system -----------------------------------
//
// Two real limits exist, and they differ per system because they depend on
// K (the number of basis reflections combined per trial):
//
//   1. SUPPLY. get_hkl_search_list(system) only generates so many
//      reflections. buildHklBasis does ordered.slice(0, n), so asking for
//      more than exists silently returns fewer.
//   2. ADDRESSING. The WGSL solvers unrank HKL K-combinations with a u32
//      index, so C(n, K) must stay under 2^32.
//
//   cubic         K=1   supply  752   u32 4294967295 ->  752  (supply binds)
//   hexagonal     K=2   supply 1249   u32 92682      -> 1249  (supply binds)
//   tetragonal    K=2   supply 1474   u32 92682      -> 1474  (supply binds)
//   orthorhombic  K=3   supply 2196   u32 2954       -> 2196  (supply binds)
//   monoclinic    K=4   supply  612   u32  568       ->  568  (u32 binds)
//   triclinic     K=6   supply  665   u32  123       ->  123  (u32 binds)
//
// The "% / unknown" setting is a percentage OF THIS MAXIMUM, so it can never
// ask for a basis the solver would have to clamp.
const U32_MAX_BIG = 4294967295n;
const u32CapForK = (K) => {
    // Largest n with C(n, K) <= 2^32 - 1. C(n, K) is increasing in n >= K,
    // so a binary search on exact BigInt binomials is enough.
    const C = (n) => {
        let r = 1n;
        for (let i = 1n; i <= BigInt(K); i++) r = r * (BigInt(n) - i + 1n) / i;
        return r;
    };
    let lo = K, hi = 4294967295;
    while (lo < hi) {
        const mid = Math.floor((lo + hi + 1) / 2);
        if (C(mid) <= U32_MAX_BIG) lo = mid; else hi = mid - 1;
    }
    return lo;
};
const _hklBasisMaxCache = {};
const hklBasisMax = (system) => {
    const cfg = SEARCH_SYSTEMS[system];
    if (!cfg) return null;
    if (_hklBasisMaxCache[system]) return _hklBasisMaxCache[system];
    const u32Cap = u32CapForK(cfg.K);
    let supply = Infinity;
    try {
        // The crystallography scripts load on the main thread ahead of this file.
        if (typeof get_hkl_search_list === 'function') {
            supply = get_hkl_search_list(system).length;
        }
    } catch (e) {
        console.warn('hklBasisMax: could not size the basis list for', system, e);
    }
    const max = Math.min(u32Cap, supply);
    if (Number.isFinite(max)) _hklBasisMaxCache[system] = max;
    return max;
};
/* * Calculates combinations C(n, k) = "n choose k"
 * Uses the stable multiplicative formula: (n/1) * ((n-1)/2) * ... * ((n-k+1)/k)
 */
function combinations(n, k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k === 0 || k === n) {
        return 1;
    }
    // Use the identity C(n, k) == C(n, n-k) for efficiency
    if (k > n / 2) {
        k = n - k;
    }

    let res = 1;
    for (let i = 1; i <= k; i++) {
        // (n - i + 1) is equivalent to (n, n-1, n-2, ...)
        res = res * (n - i + 1) / i;
    }

    return Math.round(res);
}
// Minimum number of non-axial reflections kept after the front-loaded
// axials (orthorhombic, monoclinic); see the floor in planGpuSearch.
const MIN_MIXED_HKL = 40;
const _axialCountCache = {};
// Axial reflections (two of h,k,l zero) in a system's search list: the ones
// buildHklBasis moves to the front when splitSpecialHkls is set.
const axialHklCount = (system) => {
    if (_axialCountCache[system] !== undefined) return _axialCountCache[system];
    let n = 0;
    try {
        for (const [h, k, l] of get_hkl_search_list(system)) {
            if ((k === 0 && l === 0) || (h === 0 && l === 0) || (h === 0 && k === 0)) n++;
        }
    } catch (_) { n = 0; }
    return (_axialCountCache[system] = n);
};
// The two global settings, read once. A run takes a snapshot of them at its
// start and plans every system from that snapshot (the boxes are also locked
// while it runs), so the systems searched last cannot pick up an edit made
// in the middle of the run.
const readGpuSearchSettings = () => ({ perUnknown: getHklPercentPerUnknown(), depth: getSearchDepth() });
// Peaks usable for indexing right now: inside the 2-theta window and not
// flagged as Ka2. The same filter startIndexing applies.
const countIndexablePeaks = () => {
    const tthMin = parseFloat(ui.tthMinSlider.value);
    const tthMax = parseFloat(ui.tthMaxSlider.value);
    const lo = Number.isFinite(tthMin) ? tthMin : -Infinity;
    const hi = Number.isFinite(tthMax) ? tthMax : Infinity;
    return pickedPeaks.filter(p => p.tth >= lo && p.tth <= hi && !p.ka2Suspect).length;
};
// The search one system will actually run, from the two global settings.
//   nPeaksAvailable  peaks in the indexing window; the planned peak count is
//                    capped by it. Infinity (no peaks yet) plans the nominal.
//   settings         { perUnknown, depth }; defaults to the boxes' values.
// Returns null for an unknown system.
//   nPeaks     peaks combined (peakFactor x (K + Depth), capped)
//   fomPeaks   K + Depth: the peak count the GPU figure of merit is based on
//              (max(10, fomPeaks) peaks are scored), so extra combination
//              peaks for the high-symmetry systems do not tighten the FoM.
//   percent    the basis actually used, as % of max (differs from the
//              requested K x perUnknown when the floor or the cap applies)
const planGpuSearch = (system, nPeaksAvailable = Infinity, settings = null) => {
    const cfg = SEARCH_SYSTEMS[system];
    if (!cfg) return null;
    const K = cfg.K;
    const max = hklBasisMax(system);
    const { perUnknown, depth } = settings || readGpuSearchSettings();
    const requestedPercent = Math.min(100, K * perUnknown);
    // Floor: enough reflections for a handful of combinations even at the
    // lowest setting (the old input's own min was 10).
    //
    // Orthorhombic and monoclinic put every axial reflection (h00/0k0/00l,
    // 36 of them) at the front of the basis. A small basis is then mostly
    // axials, and monoclinic gets beta only from MIXED reflections (the h*l
    // term): at 1 %/unknown its 23-reflection basis was all axial and the
    // search could not produce a single cell; at 2 % only 9 were mixed. So
    // for those two systems the floor is all axials + MIN_MIXED_HKL mixed.
    let floor = Math.max(10, 2 * K);
    if (cfg.splitSpecialHkls) floor = Math.max(floor, axialHklCount(system) + MIN_MIXED_HKL);
    floor = Math.min(max, floor);
    const nHkl = Math.min(max, Math.max(floor, Math.round(max * requestedPercent / 100)));
    const percent = 100 * nHkl / max;
    const fomPeaks = K + depth;
    const nPeaks = Math.min((cfg.peakFactor || 1) * fomPeaks, nPeaksAvailable);
    const trials = combinations(nHkl, K) * combinations(Math.max(nPeaks, K), K) * cfg.permutations;
    return { system, K, max, percent, requestedPercent, nHkl, nPeaks, fomPeaks, depth, trials, cfg };
};
const checkedSystems = () =>
    SEARCH_ORDER.filter(s => {
        const cb = document.querySelector(`.system-checkbox[value="${s}"]`);
        return cb && cb.checked;
    });
/**
 * Writes the per-system plan (basis size, peaks, trials) under the GPU
 * parameters, so the effect of the two global settings is visible before a
 * run. It used to go to the one-line status text, which a seven-system
 * summary does not fit.
 */
function updateGpuStatusText() {
    const box = ui.gpuPlan;
    if (!box) return;
    if (!webGPUSupportsCompute) { box.textContent = ''; return; }
    // Capped by the peaks actually picked, so the panel shows the search that
    // will run; with no peaks yet it shows the nominal plan.
    const available = countIndexablePeaks() || Infinity;
    const lines = checkedSystems().map(s => {
        const p = planGpuSearch(s, available);
        // The percentage actually used (the minimum basis or the cap can
        // override the requested one), rounded like the input itself.
        const pct = p.percent >= 10 ? p.percent.toFixed(0) : p.percent.toFixed(1);
        return `${p.cfg.shortLabel}: ${p.nHkl} hkl (${pct}%) · ` +
               `${p.nPeaks} peaks · ${p.trials.toLocaleString('en-US', { maximumFractionDigits: 0 })} trials`;
    });
    box.textContent = lines.join('\n');
}
/**
 * Shows or hides the GPU-specific parameter inputs based on checkbox state.
 */
function toggleGpuParamsVisibility() {
    const anyGpu = webGPUSupportsCompute && checkedSystems().length > 0;
    if (anyGpu) {
        ui.gpuParamsContainer.classList.remove('hidden');
        updateGpuStatusText();
    } else {
        ui.gpuParamsContainer.classList.add('hidden');
    }
}
ui.systemCheckboxes.forEach(checkbox => {
    checkbox.addEventListener('change', () => {
        // avant le 16 janv 2026, there was a filter here.
        // The systems are no longer mutually exclusive (9 Oct 2026): every
        // checked one is searched, in SEARCH_ORDER.
        toggleGpuParamsVisibility();
        updateStartIndexingButtonState();
    });
});
ui.gpuHklPercent.addEventListener('input', updateGpuStatusText);
ui.gpuDepth.addEventListener('input', updateGpuStatusText);
