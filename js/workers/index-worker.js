// js/workers/index-worker.js
// CPU indexing worker. Runs the cubic, tetragonal and hexagonal searches, and
// the 'post_process' pass that looks for transformed (super/sub/equivalent)
// cells among GPU solutions. Started by setupWorker() (js/indexing/worker-pool.js)
// as js/workers/index-worker.js?v=..., one worker per search.
//
// This handler used to live at the bottom of worker-logic.js behind an
// IS_REFINEMENT_WORKER guard, because the same file was also imported by the
// refinement workers. Giving it its own entry script removes that guard and
// the import-order rule it depended on.

importScripts('../crystallography/manifest.js' + ((self.location && self.location.search) || ''));
importBrutusCrystallography();

self.onmessage = function(e) {
    
    // --- 1. Get data from main thread ---
    const data = e.data;
    const { systemToSearch, peaks, wavelength, tth_error, impurity_peaks, fom_threshold, max_solutions } = data;
    // --- 2. Set up the "global" state for the functions ---
    const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q } = getSortedPeaks(peaks, wavelength);
    const N_FOR_M20 = Math.min(20, peaks.length);
    const min_m20 = 2.0;
    const d_min = wavelength / (2 * Math.sin(Math.max(...peaks.map(p => p.tth)) * Math.PI / 360));
    // A^-2. This is 1/d_min^2, NOT a scattering vector -- see the
    // convention block at the top of js/crystallography/metric.js.
    const q_max = 1 / (d_min * d_min);
    
    // Live-updating arrays
    const foundSolutions = [];
    const foundSolutionMap = new Map();

    // Per-run rejection tally, filled in by refineAndTestSolution and
    // posted with 'done' so the main thread can explain an empty result.
    const diag = { system: systemToSearch, trials: 0, min_m20, max_volume: data.max_volume };
    
    // Wrapper for refineAndTestSolution to match the signature expected by logic functions
    const refineAndTestWrapper = (cell) => {
        refineAndTestSolution(
            cell, 
            data, 
            { 
                q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q,
                N_FOR_M20, min_m20, q_max, d_min,
                foundSolutions, foundSolutionMap, diag
            },
            self.postMessage.bind(self) 
        );
    };

    // State object passed to logic functions
    const workerState = {
        q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q,
        N_FOR_M20, min_m20, q_max, d_min,
        foundSolutions, foundSolutionMap, diag,
        refineAndTestSolution: refineAndTestWrapper 
    };

    // --- 3. Run the requested search ---
    self.postMessage({ type: 'progress', payload: 1 });
    
if (systemToSearch === 'cubic') {
        indexCubic(data, workerState, self.postMessage.bind(self));
    } else if (systemToSearch === 'tetragonal') {
        indexTetragonalOrHexagonal(data, workerState, self.postMessage.bind(self), 'tetragonal');
    } else if (systemToSearch === 'hexagonal') {
        indexTetragonalOrHexagonal(data, workerState, self.postMessage.bind(self), 'hexagonal');
    } else if (systemToSearch === 'post_process') {
        // Inject the GPU solutions into the worker so it runs them through findTransformedSolutions
        foundSolutions.push(...(data.gpuSolutions || []));
    }

    self.postMessage({ type: 'progress', payload: 80 });
    
    // Run transformation/symmetry checks on found solutions
    let ftStats = null;
    try {
        ftStats = findTransformedSolutions(foundSolutions, data, workerState, self.postMessage.bind(self));
    } catch (err) {
        ftStats = { fatal: String((err && err.message) || err), stack: String(err && err.stack || '') };
    }
    self.postMessage({ type: 'postProcessSummary', payload: ftStats });
    
    self.postMessage({ type: 'progress', payload: 100 });
    // Normalise the sentinels so the main thread does not have to reason
    // about Infinity coming out of a structured clone.
    if (!isFinite(diag.volMin)) diag.volMin = null;
    if (!isFinite(diag.volMax)) diag.volMax = null;
    if (!isFinite(diag.volOverMin)) diag.volOverMin = null;
    if (!isFinite(diag.bestM20)) diag.bestM20 = null;
    diag.solutionsFound = foundSolutions.length;
    self.postMessage({ type: 'searchDiagnostics', payload: diag });
    self.postMessage({ type: 'done' });
};
