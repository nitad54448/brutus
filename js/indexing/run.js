// js/indexing/run.js
// Indexing runs: startIndexing (GPU searches, CPU fallback, refinement),
// finalizeIndexing and stopping a run.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// What the last run searched with, for the final status line and the report.
// null when the run did not use the GPU (CPU fallback only).
let lastGpuRunSettings = null;
// Systems whose GPU search stopped because the candidate buffer filled, as
// { label, fraction } with the fraction of the search space actually visited.
// Reported in the final status line and the PDF report.
let lastTruncatedSystems = [];
// One row per checked system, in search order, for the PDF report and the
// console: basis size, peaks combined, trials searched / planned, candidates
// sent to refinement, time, and a note (truncated, skipped, stopped...).
// `done` is filled in by finalizeIndexing from the same taskProgress the
// overall "Trials:" line uses, so the rows always add up to it.
let lastSystemSearchStats = [];
// Totals of the last run for the PDF report, printed under the per-system
// table: trials searched / planned (planned is null for a CPU-only run),
// wall-clock time, and any failure messages.
let lastRunTotals = null;
// "12%", "3.4%", "0.0089%": enough digits that a search cut off near its start
// does not read as 0.
// Fixed-width text rows (for the Courier font of the PDF report and the
// console): header first, then one line per system.
const formatSystemSearchStats = (rows) => {
    const n = (v) => (v === null || v === undefined) ? '-' : Math.round(v).toLocaleString('en-US');
    const out = ['System         HKL  Peaks          Searched / Planned trials      Cand.    Time  Note'];
    for (const r of rows) {
        const pct = (r.total && r.total > 0) ? ` (${fmtSearchedPercent(Math.min(1, r.done / r.total))})` : '';
        const trials = (r.total === null && r.mode === 'CPU')
            ? `${n(r.done)} (CPU)`
            : `${n(r.done)} / ${n(r.total)}${pct}`;
        const time = (r.timeMs === null) ? '-' : (r.timeMs >= 60000
            ? `${Math.floor(r.timeMs / 60000)}m${String(Math.round(r.timeMs % 60000 / 1000)).padStart(2, '0')}s`
            : `${(r.timeMs / 1000).toFixed(1)}s`);
        out.push(r.label.padEnd(13) + n(r.nHkl).padStart(5) + n(r.nPeaks).padStart(7) + '  ' +
                 trials.padStart(37) + n(r.candidates).padStart(10) + time.padStart(8) +
                 (r.note ? '  ' + r.note : ''));
    }
    return out;
};
const fmtSearchedPercent = (fraction) => {
    const pct = fraction * 100;
    if (pct >= 10 || pct === 0) return pct.toFixed(0) + '%';
    if (pct >= 1) return pct.toFixed(1) + '%';
    return pct.toPrecision(2) + '%';
};

// How each system's hkl basis is packed for the GPU.
//
// The shaders do not receive raw [h,k,l,pad]; they receive the PRODUCTS they
// actually consume. Both the matrix rows and the FoM inner loop want them, and
// the FoM used to recompute them for every candidate cell against every basis
// reflection -- which is where nearly all GPU time went. Computing them once
// here removes those multiplies from the innermost loop and turns scalar loads
// into one 16-byte vector load.
//
// This is not an approximation: the indices are small, so every product is a
// small integer, exactly representable in f32. The shader gets bit-for-bit
// what it used to compute.
//
//   cubic          vec4(h^2+k^2+l^2, 0, 0, 0)          highsym_solver.wgsl
//   tetragonal     vec4(h^2+k^2, l^2, 0, 0)            highsym_solver.wgsl
//   hexagonal      vec4(h^2+hk+k^2, l^2, 0, 0)         highsym_solver.wgsl
//   orthorhombic   vec4(h^2, k^2, l^2, 0)              ortho_solver.wgsl
//   monoclinic     vec4(h^2, k^2, l^2, hl)             monoclinic_solver.wgsl
//   triclinic      2 x vec4: (h^2,k^2,l^2,kl), (hl,hk,0,0)   triclinic_solver.wgsl
//
// The engine reads the stride from cfg.hklFloats. Keep this table in sync
// with the @binding(1) comments in the .wgsl files.
//
// HKL_PACKING is stamped onto the returned array and re-checked by the
// engine. Raw indices and packed products have the SAME stride for most
// systems, so a mismatch is otherwise invisible: the run completes and
// silently finds nothing. That is exactly the bug this tag exists to catch.
const HKL_PACKING = 'products/v1';

const HKL_PACKERS = {
    cubic: { floats: 4, pack: (o, i, h, k, l) => {
        const b = i * 4;
        o[b] = h * h + k * k + l * l; o[b + 1] = 0; o[b + 2] = 0; o[b + 3] = 0;
    } },
    tetragonal: { floats: 4, pack: (o, i, h, k, l) => {
        const b = i * 4;
        o[b] = h * h + k * k; o[b + 1] = l * l; o[b + 2] = 0; o[b + 3] = 0;
    } },
    hexagonal: { floats: 4, pack: (o, i, h, k, l) => {
        const b = i * 4;
        o[b] = h * h + h * k + k * k; o[b + 1] = l * l; o[b + 2] = 0; o[b + 3] = 0;
    } },
    orthorhombic: { floats: 4, pack: (o, i, h, k, l) => {
        const b = i * 4;
        o[b] = h * h; o[b + 1] = k * k; o[b + 2] = l * l; o[b + 3] = 0;
    } },
    monoclinic: { floats: 4, pack: (o, i, h, k, l) => {
        const b = i * 4;
        o[b] = h * h; o[b + 1] = k * k; o[b + 2] = l * l; o[b + 3] = h * l;
    } },
    triclinic: { floats: 8, pack: (o, i, h, k, l) => {
        const b = i * 8;
        o[b] = h * h; o[b + 1] = k * k; o[b + 2] = l * l; o[b + 3] = k * l;
        o[b + 4] = h * l; o[b + 5] = h * k; o[b + 6] = 0; o[b + 7] = 0;
    } },
};

// Build the HKL basis array.
//   n_hkl_for_basis  reflections whose K-combinations are searched
//   splitSpecial     axial HKLs (two of h,k,l are 0) are moved to the front of
//                    the list before truncation (ortho, mono)
//   fomFullList      upload the WHOLE list, so the GPU FoM scores against all
//                    of it while the search still uses only the first
//                    n_hkl_for_basis (high-symmetry systems; see
//                    highsym_solver.wgsl). Otherwise only the search basis
//                    is uploaded and both counts are the same, as before.
const buildHklBasis = (system, n_hkl_for_basis, splitSpecial, fomFullList = false) => {
    const hkl_full = get_hkl_search_list(system);
    let ordered;
    if (splitSpecial) {
        const special = [];
        const regular = [];
        for (const hkl of hkl_full) {
            const [h, k, l] = hkl;
            if ((k === 0 && l === 0) || (h === 0 && l === 0) || (h === 0 && k === 0)) {
                special.push(hkl);
            } else {
                regular.push(hkl);
            }
        }
        ordered = [...special, ...regular];
    } else {
        ordered = hkl_full;
    }
    const nSearch = Math.max(0, Math.min(n_hkl_for_basis, ordered.length));
    const hkl_basis_raw = fomFullList ? ordered : ordered.slice(0, nSearch);
    const packer = HKL_PACKERS[system];
    if (!packer) throw new Error(`No HKL packer registered for system '${system}'.`);
    const hklBasisArray = new Float32Array(hkl_basis_raw.length * packer.floats);
    hkl_basis_raw.forEach((hkl, i) => { packer.pack(hklBasisArray, i, hkl[0], hkl[1], hkl[2]); });
    return { nSearch, hkl_basis_raw, hklBasisArray, hklFloats: packer.floats, hklPacking: HKL_PACKING };
};

const startIndexing = async () => {
if (isIndexing) return;
// Canonical order (SEARCH_ORDER, highest symmetry first), whatever the order
// of the checkboxes in the markup: it is the order the GPU searches run in.
const checkedValues = new Set(Array.from(ui.systemCheckboxes).filter(cb => cb.checked).map(cb => cb.value));
const systemsToSearch = SEARCH_ORDER.filter(s => checkedValues.has(s));
if (systemsToSearch.length === 0) {
    showStatus("Please select at least one crystal system to search.", "error");
    return;
}
// The systems the post-processing transforms may produce.
const allowedSystems = systemsToSearch;

const mySessionToken = ++indexingRunToken;
const runStopSignal = { stop: false };
gpuStopSignal = runStopSignal;
indexingFailures = [];
let fitContext = null;
const isCurrentRun = () => mySessionToken === indexingRunToken && !runStopSignal.stop;
const failRun = message => recordIndexingFailure(message, mySessionToken);
// The page-lifetime pool, pointed at this run's sinks (see init/release).
const runPool = refinementPool;
runPool.setHandlers(
    sol => { if (isCurrentRun()) handleNewSolution(sol, mySessionToken, fitContext); }, failRun);
indexingStartTime = performance.now();
setUIState(true); // Lock startup before the first await.
try {
let webgpuEngine = null;

// Every system can run on the GPU now. Try the engine whenever WebGPU is
// believed usable; if it is not, fall back to the CPU worker for the systems
// that have a CPU search (cubic, tetragonal, hexagonal) instead of stopping.
if (webGPUSupportsCompute) {
    try {
        // Shared, page-lifetime engine -- not a new GPUDevice per run.
        webgpuEngine = await getWebGPUEngine();
        if (!isCurrentRun()) return;
    } catch (err) {
        if (!isCurrentRun()) return;
        console.warn("WebGPU initialization failed:", err.message);
        showStatus("WebGPU failed unexpectedly. GPU-only systems are disabled; " +
                   "cubic, tetragonal and hexagonal run on the CPU.", "error", 8000);
        releaseWebGPUEngine();
        webgpuEngine = null;
        disableGpuOnlySystems();
    }
}

const tthMinVal = parseFloat(ui.tthMinSlider.value);
const tthMaxVal = parseFloat(ui.tthMaxSlider.value);
// Exclude Ka2-suspect peaks from indexing entirely. They are likely Ka2
// ghosts of stronger Ka1 lines, so feeding them to the cell search would
// either waste time (they don't satisfy the cell) or actively bias the
// fit. The user can right-click a yellow row to mark it real and override
// the exclusion if they suspect mis-tagging.
const filteredPeaks = pickedPeaks.filter(
    p => p.tth >= tthMinVal && p.tth <= tthMaxVal && !p.ka2Suspect
);

if (filteredPeaks.length < 3) {
    showStatus("Please find at least 3 peaks in the selected range.", 'error');
    return;
}

gpuDiagnostics = [];
cpuDiagnostics = [];

const webgpuSystems = [];
const workerSystems = [];
const skippedSystems = [];
for (const system of systemsToSearch) {
    if (webgpuEngine && webGPUSupportsCompute) {
        webgpuSystems.push(system);
    } else if (SEARCH_SYSTEMS[system] && SEARCH_SYSTEMS[system].cpuFallback) {
        workerSystems.push(system);
    } else {
        skippedSystems.push(system);
    }
}
if (skippedSystems.length) {
    console.warn(`Skipping ${skippedSystems.join(', ')} - GPU unavailable and no CPU search for them.`);
    showStatus(`Skipping ${skippedSystems.map(s => SEARCH_SYSTEMS[s].label).join(', ')} (GPU required)`, "error", 4000);
}

if (workerSystems.length === 0 && webgpuSystems.length === 0) {
    showStatus("No tasks to run. Check selections and GPU support.", "info");
    return;
}

setUIState(true); // Disable UI

// modif le 16 janvier 2025, garder les mailles trouvées avant
ui.tabButtons.forEach(btn => btn.classList.remove('active'));
ui.tabPanels.forEach(panel => panel.classList.remove('active'));
document.querySelector('.tab-btn[data-tab="solutions"]').classList.add('active');
document.getElementById('solutions-tab-content').classList.add('active');

   // solutions = [];
   // displayedSolutions = [];
selectedSolution = null;
currentHklList = [];
activeWorkers = [];
//foundSolutionMap.clear();
updateSolutionsTable();
updateAllMarkers();
showStatus(`Indexing started...`, 'info');

cumulativeTrials = 0;
gpuTotalTrials = 0;
indexingStartTime = performance.now();
lastTruncatedSystems = [];
lastSystemSearchStats = systemsToSearch.map(s => ({
    system: s, label: SEARCH_SYSTEMS[s].label,
    mode: webgpuSystems.includes(s) ? 'GPU' : (workerSystems.includes(s) ? 'CPU' : null),
    taskIndex: -1, nHkl: null, nPeaks: null, total: null, done: 0,
    candidates: null, timeMs: null, truncated: null,
    note: skippedSystems.includes(s) ? 'skipped (GPU required)' : '',
}));
const systemStat = (s) => lastSystemSearchStats.find(e => e.system === s) || {};
// One snapshot of the GPU settings for the whole run: every system is planned
// from it (runGpuSystem), and the final status line reports it. The boxes are
// locked while the run lasts as well (setUIState).
const runGpuSettings = readGpuSearchSettings();
lastGpuRunSettings = webgpuSystems.length
    ? { ...runGpuSettings, fom: getFomThreshold(), candidates: getCandidateCells() }
    : null;

const baseParams = {
    peaks: filteredPeaks,
    wavelength: getWavelength(),
    tth_error: getTthError(),
    max_volume: getMaxVolume(),
    impurity_peaks: getImpurityPeaks(),
    refineZero: !!ui.refineZeroCheckbox.checked,
    fom_threshold: getFomThreshold(),
    max_solutions: getCandidateCells(),
    // Physical plausibility limits. extractCell* in all the shaders used
    // to hard-code 2.0 / 50.0 / 20.0 while refineAndTestSolution kept its
    // own copies of the same three numbers; the shader side now reads them
    // from here (config.f_params2) so there is one place to change them.
    // These values reproduce the previous behaviour exactly.
    min_axis: 2.0,
    max_axis: 50.0,
    min_volume: 20.0
};

// Stable data provenance: repeated searches on identical inputs can share
// results, but changed peaks/wavelength/tolerance cannot suppress each other.
fitContext = JSON.stringify({ peaks: baseParams.peaks, wavelength: baseParams.wavelength,
    tth_error: baseParams.tth_error, impurity_peaks: baseParams.impurity_peaks });

let currentWorkerTaskIndex = 0;
const totalTasks = workerSystems.length + webgpuSystems.length;
taskProgress = new Array(totalTasks).fill(0);
taskTotals = new Array(totalTasks).fill(0);

const updateProgressBar = () => {
    if (!isCurrentRun() || totalTasks === 0) return;
    const totalSum = taskProgress.reduce((a, b) => a + b, 0);
    const totalPercentage = totalSum / totalTasks;
    ui.progressBar.style.width = `${Math.min(100, totalPercentage)}%`;
};

if (statusTextElement) statusTextElement.textContent = '[0%] Starting...';

const getGpuProgressCallback = (systemName, absoluteTaskIndex) => {
    const cb = (chunkProgress, numFound) => {
        if (!isCurrentRun()) return;
        taskProgress[absoluteTaskIndex] = chunkProgress * 100;
        updateProgressBar();
        const totalPercentage = taskProgress.reduce((a, b) => a + b, 0) / totalTasks;
        const message = `[${totalPercentage.toFixed(0)}%] GPU (${systemName}): ${numFound} candidates`;
        throttledSetStatusText(message);
    };
    // The engine calls this once, before the chunk loop, with the dispatch
    // geometry it actually resolved. checkGpuLimits() estimates the same
    // numbers ahead of time, but this is the ground truth.
    cb.reportPlan = (plan) => {
        console.log(`[perf] GPU plan (${plan.system}): ${plan.totalChunks.toLocaleString()} dispatches, ` +
                    `${plan.hklsPerChunk} hkl/chunk, ${plan.numPeakCombos} peak combos`);
        if (plan.totalChunks > CHUNK_COUNT_WARN_LIMIT) {
            showStatus(
                `${systemName}: ~${plan.totalChunks.toLocaleString()} GPU dispatches queued — this run will be slow. ` +
                `Press Stop and lower "Depth" to speed it up.`, 'error', 10000);
        }
    };
    return cb;
};

const taskPromises = [];

// GPU
let qTolerancesArray, qObsArray;
if (webgpuSystems.length > 0) {
    const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q } = getSortedPeaks(filteredPeaks, baseParams.wavelength);
    // 32, not 20: the WGSL solvers cap the FoM loop at MAX_FOM_PEAKS and
    // index q_tolerances[i] for i < min(finalFomCount, MAX_FOM_PEAKS). With
    // MAX_FOM_PEAKS unified to 32 across all the shaders, a 20-length
    // buffer was read out of bounds whenever finalFomCount (= max(10,
    // peaks combined)) exceeded 20. Sizing to 32 covers every system; the
    // shader's own min() still caps the actual number of peaks read.
    const n_peaks_for_fom = Math.min(q_obs.length, 32);
    qTolerancesArray = new Float32Array(n_peaks_for_fom);
    for (let i = 0; i < n_peaks_for_fom; i++) {
        qTolerancesArray[i] = get_q_tolerance(
            peaks_sorted_by_q[i].original_index,
            tth_obs_rad,
            baseParams.wavelength,
            baseParams.tth_error
        ) + 1e-9;
    }
    qObsArray = new Float32Array(q_obs);

    // Initialize (or re-initialize) the refinement worker pool with the constants
    // that apply to this whole run. Each worker keeps its own foundSolutionMap;
    // cross-worker duplicates are resolved by applyFinalSieve at the end of the run.
    const N_FOR_M20 = Math.min(20, filteredPeaks.length);
    const maxTthDeg = maxOfArray(filteredPeaks.map(p => p.tth));
    const d_min = baseParams.wavelength / (2 * Math.sin(maxTthDeg * Math.PI / 360));
    const q_max = 1 / (d_min * d_min);
    runPool.init({
        baseParams,
        q_obs,
        original_indices,
        tth_obs_rad,
        peaks_sorted_by_q,
        N_FOR_M20,
        min_m20: 2.0,
        q_max,
        d_min,
    });
    runPool.reset();
}

// CPU (fallback only: cubic / tetragonal / hexagonal without WebGPU)
if (workerSystems.length > 0) {
    const workerTask = new Promise((resolve) => {
        resolveWorkerTask = resolve;
        if (!workerURL) {
            showStatus("Error: Indexing engine is not available.", "error");
            resolve();
            return;
        }
        let workersRemaining = workerSystems.length;
        workerSystems.forEach((system) => {
            const absoluteTaskIndex = currentWorkerTaskIndex;
            currentWorkerTaskIndex++;

            taskTotals[absoluteTaskIndex] = 0;
            const stat = systemStat(system);
            stat.taskIndex = absoluteTaskIndex;

            const worker = new Worker(workerURL);
            activeWorkers.push(worker);
            const workerT0 = performance.now();

            // Both 'done' and onerror used to decrement workersRemaining and
            // test it against 0 independently. terminate() makes the race
            // unlikely but not impossible, and if the count ever skips past 0
            // the CPU-worker promise never resolves -- startIndexing() then
            // waits forever at its Promise.all and the run hangs with the UI
            // stuck mid-progress. One settle path, guarded, instead.
            let workerSettled = false;
            const settleWorker = () => {
                if (workerSettled) return;
                workerSettled = true;
                try { worker.terminate(); } catch (_) {}
                activeWorkers = activeWorkers.filter(w => w !== worker);
                workersRemaining--;
                if (workersRemaining === 0) {
                    resolveWorkerTask = null;
                    resolve();
                }
            };

            worker.onmessage = (e) => {
                if (!isCurrentRun()) return;
                const { type, payload } = e.data;
                if (type === 'trials_completed_batch') {
                    cumulativeTrials += payload;
                    stat.done += payload;
                    const elapsedTimeSeconds = (performance.now() - indexingStartTime) / 1000;
                    const trialsPerSecond = (elapsedTimeSeconds > 0.1) ? cumulativeTrials / elapsedTimeSeconds : 0;
                    const totalPercentage = taskProgress.reduce((a, b) => a + b, 0) / totalTasks;
                    const message = `[${totalPercentage.toFixed(0)}%] Trials: ${cumulativeTrials.toLocaleString()} (${trialsPerSecond.toLocaleString('en-US', { maximumFractionDigits: 0 })}/s)`;
                    throttledSetStatusText(message);

                } else if (type === 'solution') {
                 handleNewSolution(payload, mySessionToken, fitContext); //new helper, 20 nov
                } else if (type === 'postProcessSummary') {
                    if (payload && (payload.fatal || payload.solutionErrors || payload.swapErrors))
                        failRun(`Post-processing for ${system} was incomplete.`);
                } else if (type === 'searchDiagnostics') {
                    cpuDiagnostics.push(payload);
                } else if (type === 'progress') {

                    taskProgress[absoluteTaskIndex] = payload;
                    updateProgressBar();
                } else if (type === 'done') {

                    if (!runStopSignal.stop) {
                        taskProgress[absoluteTaskIndex] = 100;
                        updateProgressBar();
                    }

                    stat.timeMs = performance.now() - workerT0;
                    console.log(`[perf] CPU worker '${system}': ${(performance.now() - workerT0).toFixed(0)} ms`);
                    settleWorker();
                }
            };

            worker.onerror = (err) => {
        if (!isCurrentRun()) return;
        failRun(`CPU search for ${system} failed.`);
        console.error(`Worker error in indexing task ${absoluteTaskIndex}:`, err);
        taskProgress[absoluteTaskIndex] = 100; // Allow queue to continue
        showStatus(`Warning: An indexing task encountered an error and was skipped. Check console for details.`, 'error');
        updateProgressBar();


                console.error(`Worker for ${system} crashed:`, err.message);
                settleWorker();
            };
            worker.postMessage({ ...baseParams, systemToSearch: system, allowedSystems });
        });
    });
    taskPromises.push(workerTask);
}


// Per-system GPU configuration: which kernel runs the search. Everything
// else about a system (K, permutations, labels, basis options) lives in
// SEARCH_SYSTEMS (js/gpu/gpu-setup.js), shared with the pre-flight checks.
//
// The three high-symmetry systems share shaders/highsym_solver.wgsl through
// three entry points; ortho, mono and tri keep their own solvers.
const HIGHSYM_SHADER = 'shaders/highsym_solver.wgsl' + APP_VERSION_QS;
const GPU_SYSTEM_CONFIG = {
    cubic:        { shader: HIGHSYM_SHADER, entryPoint: 'main_cubic',        engineMethod: 'runCubicSolver' },
    hexagonal:    { shader: HIGHSYM_SHADER, entryPoint: 'main_hexagonal',    engineMethod: 'runHexagonalSolver' },
    tetragonal:   { shader: HIGHSYM_SHADER, entryPoint: 'main_tetragonal',   engineMethod: 'runTetragonalSolver' },
    orthorhombic: { shader: 'shaders/ortho_solver.wgsl' + APP_VERSION_QS,      entryPoint: 'main_3p', engineMethod: 'runOrthoSolver' },
    monoclinic:   { shader: 'shaders/monoclinic_solver.wgsl' + APP_VERSION_QS, entryPoint: 'main_4p', engineMethod: 'runMonoclinicSolver' },
    triclinic:    { shader: 'shaders/triclinic_solver.wgsl' + APP_VERSION_QS,  entryPoint: 'main',    engineMethod: 'runTriclinicSolver' },
};

// Build the peak-combo flat Uint32Array using the already-present createCombinationGenerator.
// Previously these were hand-written nested for-loops (3, 4, and 6 levels deep).
const buildPeakCombos = (max_p, K) => {
    const numCombos = combinations(max_p, K);
    const peakCombos = new Uint32Array(numCombos * K);
    let offset = 0;
    for (const combo of createCombinationGenerator(max_p, K)) {
        peakCombos.set(combo, offset);
        offset += K;
    }
    return peakCombos;
};

// One GPU search. Run strictly one system at a time (see the loop below): the
// engine holds a single current pipeline, and loadShader/createPipeline for
// the next system would otherwise replace it under a search still running --
// which the old Promise.all launch of all three GPU tasks did.
const runGpuSystem = async (system, absoluteTaskIndex) => {
    const sys = SEARCH_SYSTEMS[system];
    const cfg = GPU_SYSTEM_CONFIG[system];
    if (!sys || !cfg) return;
    const K_VALUE = sys.K;

    // Basis size and peak count come from the two global settings (HKL
    // basis % per unknown, Depth); planGpuSearch is the same function the
    // parameter panel and the pre-flight check use, so what was shown is
    // what runs. The basis is already capped at hklBasisMax (supply and
    // u32 combinadic limit), so no run-time clamp is needed.
    const plan = planGpuSearch(system, qObsArray.length, runGpuSettings);
    const minPeaks = Math.max(K_VALUE, MIN_PEAKS_FOR_SYSTEM[system] || K_VALUE);
    const stat = systemStat(system);
    if (filteredPeaks.length < minPeaks || plan.nPeaks < K_VALUE) {
        stat.note = `skipped (needs ${minPeaks} peaks)`;
        showStatus(`${sys.label} search requires at least ${minPeaks} peaks. Skipping.`, "error");
        taskProgress[absoluteTaskIndex] = 100;
        updateProgressBar();
        return;
    }

    const tGpuStart = performance.now();
    let cellsDispatchedToRefine = 0;
    try {
        showStatus(`GPU search: ${sys.label.toLowerCase()}...`, 'info');
        const engine = webgpuEngine;
        await engine.loadShader(cfg.shader);
        if (!isCurrentRun()) return;
        await engine.createPipeline(cfg.entryPoint);
        if (!isCurrentRun()) return;

        // 1. HKL basis (+ the full FoM list for the high-symmetry systems)
        const { nSearch, hklBasisArray, hklPacking } =
            buildHklBasis(system, plan.nHkl, sys.splitSpecialHkls, sys.fomFullList);

        // 2. Peak combinations, over the lowest-angle plan.nPeaks peaks
        // (peakFactor x (K + Depth): 3x for the systems with 1-2 unknowns)
        const max_p = plan.nPeaks;
        const peakCombos = buildPeakCombos(max_p, K_VALUE);
        if (peakCombos.length === 0) throw new Error(`Not enough peaks to generate ${K_VALUE}-peak combinations.`);

        // 3. Stats
        const totalHklCombos = combinations(nSearch, K_VALUE);
        const taskTotalTrials = totalHklCombos * combinations(max_p, K_VALUE) * sys.permutations;
        gpuTotalTrials += taskTotalTrials;
        taskTotals[absoluteTaskIndex] = taskTotalTrials;
        Object.assign(stat, { taskIndex: absoluteTaskIndex, nHkl: nSearch, nPeaks: max_p, total: taskTotalTrials });
        console.log(`[perf] GPU task '${sys.label}': ${nSearch} hkl (${plan.percent.toFixed(1)}% of ${plan.max}), ` +
                    `${max_p} peaks, ${taskTotalTrials.toLocaleString()} trials`);

        // 4. Execution
        const progressCallback = getGpuProgressCallback(sys.shortLabel, absoluteTaskIndex);
        let dispatchMs = 0;
        const handleIntermediateResults = (newCells) => {
            if (!isCurrentRun()) return;
            cellsDispatchedToRefine += newCells.length;
            const t0 = performance.now();
            // Send the whole GPU-chunk's worth of cells as a single batched
            // call to the pool. The pool splits across workers round-robin,
            // sending at most N messages per batch (N = pool size) regardless
            // of how many cells are in the chunk.
            runPool.refineBatch(newCells);
            dispatchMs += performance.now() - t0;
        };

        const engineFn = engine[cfg.engineMethod];
        if (typeof engineFn !== 'function') {
            throw new Error(`Engine missing method ${cfg.engineMethod}`);
        }
        const tEngineStart = performance.now();
        const engineResult = await engineFn.call(
            engine,
            qObsArray,
            hklBasisArray,
            peakCombos,
            null,
            qTolerancesArray,
            progressCallback,   // pass directly: an arrow wrapper would drop .reportPlan
            runStopSignal,
            // hklPacking, n_hkl_search and gpu_peaks_count ride in baseParams
            // rather than as more positional arguments -- this list is long
            // enough that inserting one shifts everything after it. Spread
            // rather than stored on baseParams, so the values report what was
            // ACTUALLY built for this system. gpu_peaks_count sets how many
            // peaks the GPU FoM scores (max(10, it)): K + Depth, NOT the 3x
            // combination count of the high-symmetry systems, so that their
            // extra seed peaks do not also make the FoM stricter.
            { ...baseParams, hklPacking, n_hkl_search: nSearch, gpu_peaks_count: plan.fomPeaks },
            handleIntermediateResults
        );
        if (!isCurrentRun()) return;
        const engineMs = performance.now() - tEngineStart;
        console.log(`[perf]   engineFn('${sys.label}') wall time: ${engineMs.toFixed(0)} ms  |  ` +
                    `dispatch: ${dispatchMs.toFixed(0)} ms  |  cells: ${cellsDispatchedToRefine}`);
        // taskProgress drives BOTH the progress bar and the "Trials: done /
        // total" line in finalizeIndexing. A run cut short by a full
        // candidate buffer must therefore record the fraction it actually
        // searched -- setting 100 here made a truncated search report every
        // trial as done. searchedFraction comes from the shaders' first
        // incomplete HKL-combination index: an exact lower bound.
        if (engineResult?.stoppedEarly && !runStopSignal.stop) {
            const fraction = Number.isFinite(engineResult.searchedFraction) ? engineResult.searchedFraction : 0;
            taskProgress[absoluteTaskIndex] = fraction * 100;
            lastTruncatedSystems.push({ label: sys.label, fraction });
            stat.truncated = fraction;
            console.warn(`[perf]   ${sys.label}: candidate buffer full (${baseParams.max_solutions}) after ` +
                         `${fmtSearchedPercent(fraction)} of the search -- the rest was not visited.`);
            showStatus(`${sys.label}: candidate limit reached after ${fmtSearchedPercent(fraction)} of the search. ` +
                       `Lower FoM Tolerance or raise Candidates.`, 'error', 8000);
        } else {
            taskProgress[absoluteTaskIndex] = 100;
        }
        updateProgressBar();
        if (engineResult?.diagnostics) {
            gpuDiagnostics.push(engineResult.diagnostics);
            console.log('[perf]   ' + describeGpuDiagnostics(engineResult.diagnostics));
        }
    } catch (err) {
        if (!isCurrentRun()) return;
        stat.note = 'failed';
        failRun(`${sys.label} GPU search failed: ${err.message}`);
        console.error(`WebGPU Error (${sys.label}):`, err);
        showStatus(`${sys.shortLabel} GPU Error: ${err.message}`, 'error');
    } finally {
        const tGpuEnd = performance.now();
        stat.candidates = cellsDispatchedToRefine;
        stat.timeMs = tGpuEnd - tGpuStart;
        console.log(`[perf] GPU task '${sys.label}': ${(tGpuEnd - tGpuStart).toFixed(0)} ms (${cellsDispatchedToRefine} candidate cells sent to refinement)`);
        updateProgressBar();
    }
};

// GPU systems run SEQUENTIALLY in SEARCH_ORDER (highest symmetry first).
// Refinement of one system's candidates on the CPU pool overlaps the GPU
// search of the next; the pool is drained once, after the last system.
if (webgpuSystems.length > 0) {
    const gpuSequence = (async () => {
        const poolMark = runPool.mark();
        for (let t = 0; t < webgpuSystems.length; t++) {
            if (!isCurrentRun()) return;
            await runGpuSystem(webgpuSystems[t], currentWorkerTaskIndex + t);
        }
        if (!isCurrentRun()) return;
        // Wait for the refinement pool to finish processing the backlog of
        // cells dispatched during the GPU runs. If the GPU produced cells
        // faster than the workers could refine them, drainMs > 0.
        if (statusTextElement) statusTextElement.textContent = 'Refining candidate cells...';
        const tDrainStart = performance.now();
        await runPool.drain(30000, poolMark);
        if (!isCurrentRun()) return;
        console.log(`[perf] pool-drain after the last GPU system: ${(performance.now() - tDrainStart).toFixed(0)} ms  |  workers: ${REFINE_POOL_SIZE}`);
    })();
    taskPromises.push(gpuSequence);
}

// final
await Promise.all(taskPromises);
if (!isCurrentRun()) return;

// --- Send GPU solutions through the post-processing worker ---
if (webgpuSystems.length > 0 && solutions.length > 0 && !runStopSignal.stop) {
    if (statusTextElement) statusTextElement.textContent = 'Post-processing GPU cells...';
    const bestOfSolutionsBefore = solutions.reduce((m, s) => (s && isFinite(s.m20) && s.m20 > m) ? s.m20 : m, 0);

    // De-duplicate BEFORE post-processing, not only after it. The candidate
    // list arriving here is overwhelmingly redundant -- a real PbSO4 run
    // produced 49 solutions that the sieve collapsed to 2, i.e. ~24 copies
    // of each distinct lattice -- and relabelling all 49 does the same work
    // two dozen times over to reach the same two answers. Measured on that
    // run: 2291 ms over 40 parents versus 132 ms over the sieved set, same
    // final cell. `solutions` itself is left intact; this only decides which
    // cells are worth handing to the search.
    const postParents = applyFinalSieve(solutions.filter(s => s._fitContext === fitContext &&
        hasRefinedZero(s) === baseParams.refineZero));

    // Spend the saving on a deeper search instead of pocketing it. With a
    // handful of genuinely distinct parents the per-parent budget can be
    // several times the default and still cost a fraction of the old pass.
    const nP = postParents.length;
    const swapCfg = nP <= 8
        ? { MAX_FITS: 600, MAX_FREE: 16, MAX_EVALS: 32, MAX_POST: 5, ROUNDS: 6 }
        : nP <= 20
            ? { MAX_FITS: 300, MAX_FREE: 12, MAX_EVALS: 24, ROUNDS: 5 }
            : {};   // many distinct lattices: fall back to the defaults
    console.log(`[post-process] ${solutions.length} solutions -> ${nP} distinct parents ` +
                `(budget: ${swapCfg.MAX_FITS || 'default'} fits/round)`);
    if (nP > 0) await new Promise(resolve => {
        const worker = new Worker(workerURL);
        // Register with activeWorkers, like every other worker. Without this
        // abortActiveIndexing() cannot reach it, so pressing Stop during
        // post-processing left it running to completion -- holding a core and
        // its own copy of the crystallography scripts -- and, worse, still posting
        // 'solution' messages. abortActiveIndexing() has already bumped
        // indexingRunToken by then, so handleNewSolution() stamped those late
        // cells with the NEXT run's token: results from an aborted run filed
        // as belonging to the following one.
        activeWorkers.push(worker);
        let ppSettled = false;
        const settlePostProcess = () => {
            if (ppSettled) return;
            ppSettled = true;
            try { worker.terminate(); } catch (_) {}
            activeWorkers = activeWorkers.filter(w => w !== worker);
            resolvePostProcessTask = null;
            resolve();
        };
        resolvePostProcessTask = settlePostProcess;
        worker.onmessage = (e) => {
            if (!isCurrentRun()) return;
            if (e.data.type === 'solution') handleNewSolution(e.data.payload, mySessionToken, fitContext);
            else if (e.data.type === 'postProcessSummary') {
                const st = e.data.payload || {};
                if (st.solutionErrors || st.swapErrors) failRun('Some post-processing candidates failed.');
                if (st.fatal) { failRun('Post-processing failed.'); console.error('[post-process] ABORTED:', st.fatal, st.stack); }
                else console.log(`[post-process] ${st.parents} parents | swap search ran on ` +
                    `${st.swapRan}/${st.swapEligible} | ${st.swapPosted} swap solutions posted | ` +
                    `${st.solutionErrors} solution errors, ${st.swapErrors} swap errors | ` +
                    `best M20 ${(+st.bestBefore).toFixed(2)} -> ${(+st.bestAfter).toFixed(2)}`);
            }
            else if (e.data.type === 'done') { settlePostProcess(); }
        };
        // Never swallow this. A throw inside the post-process worker used to
        // resolve the promise in complete silence, so a run that lost every
        // transformed and swapped solution looked exactly like a run that
        // simply found nothing better.
        worker.onerror = (err) => {
            if (!isCurrentRun()) return;
            failRun('Post-processing failed.');
            console.error('[post-process] worker crashed:',
                          err && err.message, err && err.filename, err && err.lineno);
            showStatus('Post-processing failed: ' + ((err && err.message) || 'worker error') +
                       ' - results are pre-post-processing only.', 'error', 10000);
            settlePostProcess();
        };
        worker.postMessage({
            ...baseParams,
            systemToSearch: 'post_process',
            gpuSolutions: postParents,
            allowedSystems,
            swapCfg
        });
    });
    if (!isCurrentRun()) return;
    console.log(`[post-process] main thread: best M20 ${bestOfSolutionsBefore.toFixed(2)} -> ` +
                `${solutions.reduce((m, s) => (s && isFinite(s.m20) && s.m20 > m) ? s.m20 : m, 0).toFixed(2)}`);
}
// ------------------------------------------------------------------

await new Promise(resolve => setTimeout(resolve, 250));
if (isCurrentRun()) finalizeIndexing(false, mySessionToken);
} catch (err) {
    if (isCurrentRun()) {
        failRun(err.message || String(err));
        runStopSignal.stop = true;
        activeWorkers.forEach(w => w.terminate());
        activeWorkers = [];
        if (resolveWorkerTask) { resolveWorkerTask(); resolveWorkerTask = null; }
        if (resolvePostProcessTask) { resolvePostProcessTask(); resolvePostProcessTask = null; }
        finalizeIndexing(false, mySessionToken);
    }
} finally {
    runPool.release();
    if (mySessionToken === indexingRunToken && isIndexing) setUIState(false);
}
};
// `runToken` identifies which solutions were produced BY this run (see
// handleNewSolution). It is normally the same as sessionToken, but the abort
// path bumps indexingRunToken before finalizing, so it passes the pre-bump
// value explicitly. Solutions carrying any other token came from an earlier
// run, possibly under a different wavelength or peak selection.
const finalizeIndexing = (stoppedByUser = false, sessionToken = null, runToken = undefined) => {
if (runToken === undefined) runToken = sessionToken;
// A manual Stop or a new file load bumps indexingRunToken via
// abortActiveIndexing(), which already re-enabled the UI. If that
// happened after this run started, this is a stale tail from an
// abandoned run — bail out instead of overwriting fresher state
// (or a fresher run's in-progress state) with old results.
if (sessionToken !== null && sessionToken !== indexingRunToken) {
    console.log('[perf] Discarding stale finalizeIndexing() call (superseded by a newer run or a reset).');
    return;
}
const statusTextElement = document.getElementById('status-text');

// 1. Calculate Duration with seconds
const durationMs = performance.now() - indexingStartTime;
console.log(`[perf] === Total indexing time: ${durationMs.toFixed(0)} ms ===`);
const durationSec = (durationMs / 1000).toFixed(1);
const durationStr = durationMs > 60000 
    ? `${Math.floor(durationMs/60000)}m ${(durationMs % 60000 / 1000).toFixed(0)}s` 
    : `${durationSec}s`;
lastDurationStr = durationStr; 

// 2. Calculate Real GPU Trials based on actual progress %
let gpuActualTrials = 0;
if (taskTotals && taskProgress) {
    taskTotals.forEach((total, i) => {
        const progress = taskProgress[i] || 0;
        gpuActualTrials += total * (progress / 100.0);
    });
}

// 3. Compile Totals
const totalCpuTrials = cumulativeTrials; 
const totalActualTrials = totalCpuTrials + gpuActualTrials;
const totalMaxTrials = totalCpuTrials + gpuTotalTrials; 

const fmtActual = totalActualTrials.toLocaleString('en-US', {maximumFractionDigits: 0});
const fmtMax = totalMaxTrials.toLocaleString('en-US', {maximumFractionDigits: 0});

// What the run ACTUALLY used (captured when it started), not what the boxes
// say now: the user may have edited them while the search ran.
const gpuSettings = lastGpuRunSettings;

// 4. Construct Report String
let finalStatus = "";
if (gpuSettings) {
    // Format: Trials: Actual / Max
    finalStatus = `Trials: ${fmtActual} / ${fmtMax}    Time: ${durationStr}    ` +
                  `HKL: ${gpuSettings.perUnknown}%/unknown    Depth: ${gpuSettings.depth}    ` +
                  `FoM: ${gpuSettings.fom}    Cand.: ${Math.round(gpuSettings.candidates / 1000)}k`;
} else {
    finalStatus = `CPU Trials: ${fmtActual}    Time: ${durationStr}`;
}

// A full candidate buffer ends a system's search early. The trial count above
// already reflects it; name the systems too, since "done < total" alone does
// not say why.
if (gpuSettings && lastTruncatedSystems.length) {
    finalStatus += ' | TRUNCATED (candidate buffer full): ' +
        lastTruncatedSystems.map(t => `${t.label} at ${fmtSearchedPercent(t.fraction)}`).join(', ');
}
if (indexingFailures.length) finalStatus += ' | INCOMPLETE: ' + indexingFailures.join(' ');

// Per-system rows. GPU rows take `done` from taskTotals x taskProgress, the
// very numbers summed into the line above, so the rows add up to it.
for (const st of lastSystemSearchStats) {
    if (st.mode === 'GPU' && st.taskIndex >= 0 && taskTotals[st.taskIndex]) {
        st.done = taskTotals[st.taskIndex] * ((taskProgress[st.taskIndex] || 0) / 100);
    }
    if (!st.note && st.truncated !== null) st.note = `truncated at ${fmtSearchedPercent(st.truncated)} (buffer full)`;
    if (!st.note && st.mode === 'GPU' && st.total === null) st.note = stoppedByUser ? 'not reached (stopped)' : 'not run';
    if (!st.note && stoppedByUser && st.total !== null && st.done < st.total) st.note = 'stopped';
}
if (lastSystemSearchStats.length) {
    console.log('[indexing] per system:\n' + formatSystemSearchStats(lastSystemSearchStats).join('\n'));
}
lastIndexingStats = finalStatus; 
lastRunTotals = { done: totalActualTrials, total: gpuSettings ? totalMaxTrials : null,
                  time: durationStr, failures: indexingFailures.slice() };


// Update screen status temporarily
if (statusTextElement) statusTextElement.textContent = 'Applying final sieve...';

// Apply Sieve
const _perfSieveEnd = perfStart('applyFinalSieve');
const _nBeforeSieve = solutions.length;
solutions = applyFinalSieve(solutions); 
_perfSieveEnd(`(${_nBeforeSieve} -> ${solutions.length})`);

// Space Group Analysis
if (statusTextElement) statusTextElement.textContent = 'Analyzing space groups...';

const tthMinVal = parseFloat(ui.tthMinSlider.value);
const tthMaxVal = parseFloat(ui.tthMaxSlider.value);
const filteredPeaks = pickedPeaks.filter(p => p.tth >= tthMinVal && p.tth <= tthMaxVal);

// analyzeSystematicAbsences (js/crystallography/absences.js) is now Ka2-aware:
// each peak's `ka2Suspect` flag is propagated into the indexed-hkl
// records, and downstream centering/extinction/ranking logic counts
// hard (real) and soft (Ka2-suspect) violations separately. A space
// group disqualified only by Ka2-suspect peaks ends up with
// hardViolations === 0 and is shown alongside the truly viable groups.
// Only (re)analyse cells belonging to this run. `solutions` intentionally
// survives across runs, and re-running the absence analysis on a solution
// found under a previous wavelength / 2-theta window / peak list would
// silently re-label it against data it was never fitted to -- and pay for
// the analysis again every subsequent run. A solution keeps whatever
// analysis it was given when it was found.
const needsAnalysis = (sol) =>
    !sol.analysis || runToken === null || sol._runToken === runToken;
const _nStale = solutions.filter(s => !needsAnalysis(s)).length;
if (_nStale > 0) {
    console.log(`[perf] spaceGroupAnalysis: keeping ${_nStale} analysis result(s) from earlier run(s).`);
}

const _perfSgEnd = perfStart('spaceGroupAnalysis');
if (spaceGroupData) {
    solutions.forEach(sol => {
        if (!needsAnalysis(sol)) return;
        sol.analysis = analyzeSystematicAbsences(
            sol,
            filteredPeaks,
            spaceGroupData,
            getWavelength(),
            getTthError(),
            tthMaxVal,
            getImpurityPeaks(),
            tthMinVal
        );
    });
} else {
    console.warn('Space group data not available');
    const lambda = getWavelength();
    
    solutions.forEach(sol => {
        if (!needsAnalysis(sol)) return;
        const basicHklList = generateHKL_for_analysis(sol, lambda, tthMaxVal);
        sol.analysis = {
            centering: 'Unknown (data not loaded)',
            rankedSpaceGroups: [],
            detectedExtinctions: [],
            hklList: basicHklList 
        };
    });
}
_perfSgEnd(`(${solutions.length} solutions)`);
                            
// Clear status text on screen
if (statusTextElement) statusTextElement.textContent = '';

setUIState(false);    

// Sort and Update Table
sortState = { column: 'm20', direction: 'desc' };
sortSolutions(); 

displayedSolutions = [...solutions]; 
updateSolutionsTable(); 

// Final Toast Message / LED
if (solutions.length > 0) {
    const message = stoppedByUser ? 
        'Indexing stopped by user.' : 
        'Indexing complete.';
    showStatus(`${message} Found ${solutions.length} potential solution(s).`, 'success');
    ui.solutionsLed.className = 'led-indicator green';
} else {
    const message = stoppedByUser ? 'Indexing stopped by user.' : 'Indexing finished.';
    // Say WHY. A bare "no solutions" leaves the user guessing between Max
    // Volume, 2θ Error, the peak list and the FoM threshold, and those are
    // not guessable. A stopped run is the one case with a known reason, so
    // it does not need the diagnostics.
    if (stoppedByUser) {
        showStatus(`${message} No valid solutions were found in the part of the search that ran.`, 'info', 8000);
    } else {
        const why = explainNoSolutions(gpuDiagnostics, cpuDiagnostics, {
            nPeaks: filteredPeaks.length,
            tthMin: tthMinVal,
            tthMax: tthMaxVal,
            lambda: getWavelength(),
            tthError: getTthError(),
            maxVolume: getMaxVolume(),
            systems: Array.from(ui.systemCheckboxes).filter(cb => cb.checked).map(cb => cb.value),
        });
        // explainNoSolutions never returns an empty list, so this branch
        // always has something to show.
        showStatus(`${message} No valid solutions were found. ${why.join(' | ')}`, 'error', 20000);
        for (const r of why) console.log(`[indexing] no solutions — ${r}`);
    }
    ui.solutionsLed.className = 'led-indicator red';
}
if (indexingFailures.length) {
    const warning = 'Search incomplete: ' + indexingFailures.join(' ');
    if (statusTextElement) statusTextElement.textContent = warning;
    showStatus(warning, 'error', 20000);
}
};
ui.startIndexingButton.addEventListener('click', startIndexing);
 const abortActiveIndexing = (shouldFinalize = false) => {
    activeWorkers.forEach(w => w.terminate());
    activeWorkers = [];
    gpuStopSignal.stop = true;

    // Kill the refinement pool too. Any in-flight cells are dropped.
    refinementPool.terminate();

    if (typeof resolveWorkerTask === 'function') {
        resolveWorkerTask(); // Unblock the CPU-worker await, if any
        resolveWorkerTask = null;
    }

    // Same for the post-processing stage. The forEach above already
    // terminated its worker (it is in activeWorkers now), and a terminated
    // worker fires neither 'done' nor onerror, so nothing else would ever
    // settle that promise.
    if (typeof resolvePostProcessTask === 'function') {
        const settle = resolvePostProcessTask;
        resolvePostProcessTask = null;
        settle();
    }

    // Token of the run we are stopping. Its solutions carry this value, so
    // finalizeIndexing needs it to recognise them as belonging to the
    // current run -- comparing against the post-bump token would classify
    // every cell this run just found as stale and skip its analysis.
    const abortedRunToken = indexingRunToken;

    indexingRunToken++; // invalidate any background startIndexing() loop

    if (shouldFinalize) {
        // Immediately run final sieve and space group check on solutions found so far
        finalizeIndexing(true, indexingRunToken, abortedRunToken);
    } else {
        setUIState(false);
    }
};
ui.reportButton.addEventListener('click', () => {
    if (isIndexing) {
        abortActiveIndexing(true); // Pass true to trigger space group check & sieve
    } else { 
        generatePDFReport(); 
    }
});
