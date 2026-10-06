// js/indexing/run.js
// Indexing runs: startIndexing (CPU and GPU searches, refinement), finalizeIndexing
// and stopping a run.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

const startIndexing = async () => {
if (isIndexing) return;
const systemsToSearch = Array.from(ui.systemCheckboxes).filter(cb => cb.checked).map(cb => cb.value);
if (systemsToSearch.length === 0) {
    showStatus("Please select at least one crystal system to search.", "error");
    return;
}

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
const needsWebGPU = systemsToSearch.includes('monoclinic') || systemsToSearch.includes('triclinic') || systemsToSearch.includes('orthorhombic');
let webgpuEngine = null;

if (needsWebGPU) {
    try {
        // Shared, page-lifetime engine -- not a new GPUDevice per run.
        webgpuEngine = await getWebGPUEngine();
        if (!isCurrentRun()) return;
    } catch (err) {
        if (!isCurrentRun()) return;
        console.warn("WebGPU initialization failed:", err.message);
        showStatus("WebGPU failed unexpectedly. GPU searches are disabled.", "error", 8000);
        webGPUSupportsCompute = false; 
        releaseWebGPUEngine();
        // Disable all GPU checkboxes
        [orthoCheckbox, monoCheckbox, triCheckbox].forEach(cb => {
            if (cb) {
                cb.checked = false;
                cb.disabled = true;
                if (cb.parentElement) cb.parentElement.style.opacity = '0.5';
            }
        });
        return; // Stop indexing
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


systemsToSearch.forEach(system => {
if (['orthorhombic', 'monoclinic', 'triclinic'].includes(system)) {
    if (webgpuEngine && webGPUSupportsCompute) {
        webgpuSystems.push(system);
    } else {
        // GPU unavailable: Do NOT push to workerSystems. 
        // Just skip it or warn.
        console.warn(`Skipping ${system} - GPU unavailable and CPU fallback disabled.`);
        showStatus(`Skipping ${system} (GPU required)`, "error", 4000);
    }
} else {
    // Cubic, Tetragonal, Hexagonal go to CPU
    workerSystems.push(system);
}
});


if (needsWebGPU && webgpuSystems.length === 0) {
    ui.systemCheckboxes.forEach(cb => {
        if (['monoclinic', 'triclinic', 'orthorhombic'].includes(cb.value)) cb.checked = false;
    });
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


const baseParams = {
    peaks: filteredPeaks,
    wavelength: getWavelength(),
    tth_error: getTthError(),
    max_volume: getMaxVolume(),
    impurity_peaks: getImpurityPeaks(),
    refineZero: !!ui.refineZeroCheckbox.checked,
    fom_threshold: getFomThreshold(),
    max_solutions: getCandidateCells(),
    gpu_peaks_count: getGpuPeaksCount(),
    // Physical plausibility limits. extractCell* in all three shaders used
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

let currentGpuTaskIndex = 0;
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
    // numbers ahead of time, but this is the ground truth and covers the
    // case where the basis was capped by the u32 guard after the pre-flight
    // check ran.
    cb.reportPlan = (plan) => {
        console.log(`[perf] GPU plan (${plan.system}): ${plan.totalChunks.toLocaleString()} dispatches, ` +
                    `${plan.hklsPerChunk} hkl/chunk, ${plan.numPeakCombos} peak combos`);
        if (plan.totalChunks > CHUNK_COUNT_WARN_LIMIT) {
            showStatus(
                `${systemName}: ~${plan.totalChunks.toLocaleString()} GPU dispatches queued — this run will be slow. ` +
                `Press Stop and lower "Peaks to Combine" to speed it up.`, 'error', 10000);
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
    // MAX_FOM_PEAKS unified to 32 across all three shaders, a 20-length
    // buffer was read out of bounds whenever finalFomCount (= max(10,
    // gpuPeaksCount)) exceeded 20. Sizing to 32 covers every system; the
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

// CPU
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
            worker.postMessage({ ...baseParams, systemToSearch: system, allowedSystems: systemsToSearch });
        });
    });
    taskPromises.push(workerTask);
}


// Unified GPU task factory (replaces the previous three near-identical task blocks
// for orthorhombic, monoclinic, and triclinic searches).
//
// The three systems differ only in:
//  - K (number of peaks per trial)
//  - the shader file and entry point
//  - the engine method that runs the solver
//  - the permutation multiplier used for the "total trials" estimate
//  - a default for n_peaks_for_combo and n_hkl_for_basis
//  - whether HKL basis is split into special (axial) and regular HKLs (ortho, mono do, tri doesn't)
//  - the display label for status messages
const GPU_SYSTEM_CONFIG = {
    orthorhombic: {
        K: 3,
        label: 'Orthorhombic',
        shortLabel: 'Ortho',
        shader: 'shaders/ortho_solver.wgsl' + APP_VERSION_QS,
        entryPoint: 'main_3p',
        engineMethod: 'runOrthoSolver',
        permutations: 6,
        defaultPeaks: 7,
        defaultHkl: 300,
        // min(basis supply, u32 combinadic cap). See HKL_U32_CAPS above --
        // this is a single source of truth shared with the UI input's max,
        // so the box can never offer a value the solver will silently clamp.
        maxHkl: hklBasisMax('orthorhombic'),   // 2196: the generated list runs out before the u32 cap (2954)
        splitSpecialHkls: true,
    },
    monoclinic: {
        K: 4,
        label: 'Monoclinic',
        shortLabel: 'Monoclinic',
        shader: 'shaders/monoclinic_solver.wgsl' + APP_VERSION_QS,
        entryPoint: 'main_4p',
        engineMethod: 'runMonoclinicSolver',
        permutations: 24,
        defaultPeaks: 7,
        defaultHkl: 100,
        maxHkl: hklBasisMax('monoclinic'),     // 568: C(568,4) < 2^32 < C(569,4), u32 binds before the 612-strong list
        splitSpecialHkls: true,
    },
    triclinic: {
        K: 6,
        label: 'Triclinic',
        shortLabel: 'Triclinic',
        shader: 'shaders/triclinic_solver.wgsl' + APP_VERSION_QS,
        entryPoint: 'main',
        engineMethod: 'runTriclinicSolver',
        permutations: 720,
        defaultPeaks: 8,
        defaultHkl: 40,
        maxHkl: hklBasisMax('triclinic'),      // 123: C(123,6) < 2^32 < C(124,6), u32 binds far below the 665-strong list
        splitSpecialHkls: false,
    },
};

// How each system's hkl basis is packed for the GPU.
//
// The shaders no longer receive raw [h,k,l,pad]; they receive the PRODUCTS
// they actually consume. Both the matrix rows and the FoM inner loop want
// h^2/k^2/l^2 (and h*l, or k*l/h*l/h*k), and the FoM recomputed them for
// every candidate cell against every basis reflection -- which is where
// nearly all GPU time went. Computing them once here removes 3 (ortho),
// 4 (mono) or 6 (triclinic) multiplies from the innermost loop and turns
// three scalar loads into one 16-byte vector load.
//
// This is not an approximation: |h|,|k|,|l| <= 12 so every product is a
// small integer, exactly representable in f32. The shader gets bit-for-bit
// what it used to compute.
//
// Ortho and mono keep the old 16-byte stride, so their buffers are the same
// size as before. Triclinic needs six products and so uses two vec4s
// (32 bytes); the engine reads the stride from cfg.hklFloats and the shader
// indexes hkl_basis[i*2] / hkl_basis[i*2+1]. Keep these three in sync with
// the @binding(1) comments in the .wgsl files.
//
// HKL_PACKING is stamped onto the returned array and re-checked by the
// engine. Raw indices and packed products have the SAME stride for ortho
// and mono, so a mismatch is otherwise invisible: the run completes and
// silently finds nothing. That is exactly the bug this tag exists to catch.
const HKL_PACKING = 'products/v1';

const HKL_PACKERS = {
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

// Build the HKL basis array. If splitSpecial is true, axial HKLs (two of h,k,l are 0)
// are placed at the front of the list before truncation to n_hkl_for_basis.
const buildHklBasis = (system, n_hkl_for_basis, splitSpecial) => {
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
    const hkl_basis_raw = ordered.slice(0, n_hkl_for_basis);
    const packer = HKL_PACKERS[system];
    if (!packer) throw new Error(`No HKL packer registered for system '${system}'.`);
    const hklBasisArray = new Float32Array(hkl_basis_raw.length * packer.floats);
    hkl_basis_raw.forEach((hkl, i) => { packer.pack(hklBasisArray, i, hkl[0], hkl[1], hkl[2]); });
    return { hkl_basis_raw, hklBasisArray, hklFloats: packer.floats, hklPacking: HKL_PACKING };
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

// Create a GPU task for a given system. Returns an async function that can be
// invoked to run the task. Returns null if the system isn't configured.
const makeGpuTask = (system, absoluteTaskIndex) => {
    const cfg = GPU_SYSTEM_CONFIG[system];
    if (!cfg) return null;

    return async () => {
        const K_VALUE = cfg.K;
        // Clamp to the input's own max attribute, not just its floor. A
        // number input's `value` is whatever was typed -- `max` is a
        // validation hint, not an enforced bound -- and this figure drives
        // C(n_peaks, K) peak combinations, hence the X dispatch dimension
        // and the size of the peakCombos buffer. At 30 peaks triclinic asks
        // for 148k workgroups in X (limit 65,535) and a 14 MB buffer; at 50
        // it asks for 381 MB. Reading the max from the element keeps one
        // source of truth with the markup.
        const peaksMax = parseInt(ui.gpuPeaksCount.max, 10) || 20;
        const n_peaks_for_combo = Math.min(peaksMax,
            Math.max(K_VALUE, parseInt(ui.gpuPeaksCount.value, 10) || cfg.defaultPeaks));
        let n_hkl_for_basis = Math.max(K_VALUE * 2, parseInt(ui.gpuHklTriplets.value, 10) || cfg.defaultHkl);

        // --- u32 combinadic guard ---------------------------------------
        // The WGSL solvers unrank HKL K-combinations with a u32 linear index
        // and a u32 binomial_table (get_combinadic_indices). Once C(n_hkl, K)
        // reaches 2^32, both the linear index AND the unranking binomials
        // overflow, silently corrupting the search (truncated bound + wrong
        // HKL triplets). cfg.maxHkl is the largest basis size that keeps the
        // whole combination space u32-addressable (ortho 2954 / mono 568 /
        // tri 123). Widening the JS binomial table would NOT help: the shader
        // is u32 end-to-end.
        const maxHklForU32 = cfg.maxHkl;
        if (n_hkl_for_basis > maxHklForU32) {
            showStatus(`${cfg.label}: HKL basis capped at ${maxHklForU32} (GPU ${K_VALUE}-peak u32 combination limit).`, 'error', 6000);
            n_hkl_for_basis = maxHklForU32;
        }
        // ----------------------------------------------------------------

        if (filteredPeaks.length < K_VALUE) {
            showStatus(`${cfg.label} search requires at least ${K_VALUE} peaks. Skipping.`, "error");
            taskProgress[absoluteTaskIndex] = 100;
            updateProgressBar();
            return;
        }

        const tGpuStart = performance.now();
        let cellsDispatchedToRefine = 0;
        try {
            showStatus(`Initializing WebGPU for ${cfg.label.toLowerCase()}...`, 'info');
            const engine = webgpuEngine;
            await engine.loadShader(cfg.shader);
            if (!isCurrentRun()) return;
            await engine.createPipeline(cfg.entryPoint);

            // 1. HKL basis
            const { hkl_basis_raw, hklBasisArray, hklPacking } = buildHklBasis(system, n_hkl_for_basis, cfg.splitSpecialHkls);

            // 2. Peak combinations
            const max_p = Math.min(n_peaks_for_combo, qObsArray.length);
            if (max_p < K_VALUE) throw new Error(`Not enough peaks (${max_p}) for ${K_VALUE}-peak solve.`);
            const peakCombos = buildPeakCombos(max_p, K_VALUE);
            if (peakCombos.length === 0) throw new Error(`Not enough peaks to generate ${K_VALUE}-peak combinations.`);

            // 3. Stats
            const num_hkls = hkl_basis_raw.length;
            const totalHklCombos = combinations(num_hkls, K_VALUE);
            const taskTotalTrials = totalHklCombos * combinations(max_p, K_VALUE) * cfg.permutations;
            gpuTotalTrials += taskTotalTrials;
            taskTotals[absoluteTaskIndex] = taskTotalTrials;

            // 4. Execution
            const progressCallback = getGpuProgressCallback(cfg.shortLabel, absoluteTaskIndex);
            // Main-thread dispatch is near-instant now (just postMessage), but we
            // still need to wait for the worker pool to drain before moving on.
            // The main metric is `drainMs` below (how long after GPU finishes
            // that workers are still chewing through the backlog).
            let dispatchMs = 0;
            // Everything this task dispatches gets a batch id >= poolMark,
            // so the drain below waits on this task's work only rather than
            // on whatever else happens to be in flight pool-wide.
            const poolMark = runPool.mark();
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
                // hklPacking rides in baseParams rather than as another
                // positional argument -- this list is long enough that
                // inserting one shifts everything after it, which is exactly
                // the mistake the tag exists to catch. Spread rather than
                // stored on baseParams at construction, so the tag reports
                // what buildHklBasis ACTUALLY produced for this run.
                { ...baseParams, hklPacking },
                handleIntermediateResults
            );
            if (!isCurrentRun()) return;
            const engineMs = performance.now() - tEngineStart;
            // Wait for the refinement pool to finish processing the backlog of
            // cells dispatched during the GPU run. If GPU produced cells faster
            // than workers could refine, drainMs > 0. If workers kept up, it's ~0.
            const tDrainStart = performance.now();
            await runPool.drain(30000, poolMark);
            if (!isCurrentRun()) return;
            const drainMs = performance.now() - tDrainStart;
            console.log(`[perf]   engineFn('${cfg.label}') wall time: ${engineMs.toFixed(0)} ms`);
            console.log(`[perf]   dispatch: ${dispatchMs.toFixed(0)} ms  |  pool-drain: ${drainMs.toFixed(0)} ms  |  cells: ${cellsDispatchedToRefine}  |  workers: ${REFINE_POOL_SIZE}`);
            if (engineResult?.diagnostics) {
                gpuDiagnostics.push(engineResult.diagnostics);
                console.log('[perf]   ' + describeGpuDiagnostics(engineResult.diagnostics));
            }
        } catch (err) {
            if (!isCurrentRun()) return;
            failRun(`${cfg.label} GPU search failed: ${err.message}`);
            console.error(`WebGPU Error (${cfg.label}):`, err);
            showStatus(`${cfg.shortLabel} GPU Error: ${err.message}`, 'error');
        } finally {
            const tGpuEnd = performance.now();
            console.log(`[perf] GPU task '${cfg.label}': ${(tGpuEnd - tGpuStart).toFixed(0)} ms (${cellsDispatchedToRefine} candidate cells refined on CPU)`);
            updateProgressBar();
        }
    };
};

// Launch one GPU task per selected low-symmetry system.
// Iterate in a canonical order (ortho, mono, tri) regardless of the checkbox order,
// so task indices remain deterministic and match the prior behavior.
const GPU_ORDER = ['orthorhombic', 'monoclinic', 'triclinic'];
for (const system of GPU_ORDER) {
    if (!webgpuSystems.includes(system)) continue;
    if (!GPU_SYSTEM_CONFIG[system]) continue;
    const absoluteTaskIndex = currentWorkerTaskIndex + currentGpuTaskIndex;
    currentGpuTaskIndex++;
    const taskFn = makeGpuTask(system, absoluteTaskIndex);
    if (taskFn) taskPromises.push(taskFn());
}

// final
await Promise.all(taskPromises);
if (!isCurrentRun()) return;

// --- NEW: Send GPU solutions through the post-processing worker ---
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
    await new Promise(resolve => {
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
        const bestOf = (arr) => arr.reduce((m, s) => (s && isFinite(s.m20) && s.m20 > m) ? s.m20 : m, 0);
        const m20Before = bestOf(solutions);
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
            allowedSystems: systemsToSearch,
            swapCfg
        });
    });
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

const isGpuRun = (orthoCheckbox && orthoCheckbox.checked) || 
                 (monoCheckbox && monoCheckbox.checked) || 
                 (triCheckbox && triCheckbox.checked);

// 4. Construct Report String
let finalStatus = "";
if (isGpuRun) {
    const hklSize = ui.gpuHklTriplets.value;
    const peaksComb = ui.gpuPeaksCount.value;
    // Format: Trials: Actual / Max
    finalStatus = `Trials: ${fmtActual} / ${fmtMax}    Time: ${durationStr}    HKL: ${hklSize}    Peaks: ${peaksComb}`;
} else {
    finalStatus = `CPU Trials: ${fmtActual}    Time: ${durationStr}`;
}

if (indexingFailures.length) finalStatus += ' | INCOMPLETE: ' + indexingFailures.join(' ');
lastIndexingStats = finalStatus; 


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
