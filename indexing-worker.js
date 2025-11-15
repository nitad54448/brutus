// This is the entire content of your new "indexing-worker.js" file.

// 1. Import all the functions (like indexCubic, refineAndTestSolution, etc.)
importScripts('worker-logic.js'); 

// 2. Set up the worker's "onmessage" handler
self.onmessage = function(e) {
    
    // --- 1. Get data from main thread ---
    const data = e.data;
    const { systemToSearch, peaks, wavelength, tth_error, impurity_peaks } = data;

    // --- 2. Set up the "global" state for the functions ---
    // These are the variables all your functions need
    const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q } = getSortedPeaks(peaks, wavelength);
    const N_FOR_M20 = Math.min(20, peaks.length);
    const min_m20 = 2.0;
    const d_min = wavelength / (2 * Math.sin(Math.max(...peaks.map(p => p.tth)) * Math.PI / 360));
    const q_max = 1 / (d_min * d_min);
    
    // These are the live-updating arrays
    const foundSolutions = [];
    const foundSolutionMap = new Map();
    
    // This is the "state" object we pass to all functions
    // We create a simple wrapper for refineAndTestSolution
    const refineAndTestWrapper = (cell) => {
        // This calls the "real" function from worker-logic.js
        refineAndTestSolution(
            cell, 
            data, // The full `e.data` object
            { // The "state" object
                q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q,
                N_FOR_M20, min_m20, q_max, d_min,
                foundSolutions, foundSolutionMap
            },
            self.postMessage.bind(self) // The callback to send solutions
        );
    };

    // This is the "state" object we pass to all functions
    const workerState = {
        q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q,
        N_FOR_M20, min_m20, q_max, d_min,
        foundSolutions, foundSolutionMap,
        refineAndTestSolution: refineAndTestWrapper // Pass the wrapper
    };

    // --- 3. Call the correct indexing function ---
    self.postMessage({ type: 'progress', payload: 1 });
    
    if (systemToSearch === 'cubic') indexCubic(data, workerState, self.postMessage.bind(self));
    if (systemToSearch === 'tetragonal') indexTetragonalOrHexagonal(data, workerState, self.postMessage.bind(self), 'tetragonal');
    if (systemToSearch === 'hexagonal') indexTetragonalOrHexagonal(data, workerState, self.postMessage.bind(self), 'hexagonal');
    if (systemToSearch === 'orthorhombic') indexOrthorhombic(data, workerState, self.postMessage.bind(self));
    if (systemToSearch === 'monoclinic') indexMonoclinic(data, workerState, self.postMessage.bind(self));
    // We are no longer calling triclinic from the worker
    
    self.postMessage({ type: 'progress', payload: 80 });
    
    findTransformedSolutions(foundSolutions, data, workerState, self.postMessage.bind(self));
    
    self.postMessage({ type: 'progress', payload: 95 });
    // findTransformedSolutions already posted its solutions
    self.postMessage({ type: 'progress', payload: 100 });
    self.postMessage({ type: 'done' });
};