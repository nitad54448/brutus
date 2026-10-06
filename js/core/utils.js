// js/core/utils.js
// Small generic helpers: debounce/throttle, timing, number formatting,
// combinations, and the scattering-vector conversions used by the plot.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// debounce
const debounce = (func, delay) => {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), delay);
    };
};
// Safe max for arrays of any length. Math.max(...arr) throws on large arrays
// (stack limit ~65k in V8) and is slow even when it works.
const maxOfArray = (arr) => {
    const n = arr.length;
    if (n === 0) return -Infinity;
    let m = arr[0];
    for (let i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
    return m;
};
// Lightweight perf helper. Usage:
//   const end = perfStart('findPeaks');
//   ... do work ...
//   end();
// Logs one line per call to the console. Prefix 'perf' makes them easy to filter.
const perfStart = (label) => {
    const t0 = performance.now();
    return (extra = '') => {
        const dt = performance.now() - t0;
        console.log(`[perf] ${label}: ${dt.toFixed(1)} ms${extra ? ' ' + extra : ''}`);
        return dt;
    };
};
const formatWithError = (value, error) => {
    if (error === undefined || error === null || !isFinite(error) || error <= 0) {
        const places = Math.abs(value) > 10 ? 3 : 4;
        return value.toFixed(places);
    }
    const errorMagnitude = Math.floor(Math.log10(error));
    const firstSigDigit = Math.floor(error / Math.pow(10, errorMagnitude));

    let decimalPlaces;
    if (firstSigDigit >= 3) {
        // Use 1 significant figure for error 
        decimalPlaces = -errorMagnitude;
    } else {
        decimalPlaces = -errorMagnitude + 1;
    }
    
    // Ensure decimalPlaces is reasonable and non-negative
    decimalPlaces = Math.max(0, Math.min(8, decimalPlaces));
    const multiplier = Math.pow(10, decimalPlaces);
    const roundedValue = (Math.round(value * multiplier) / multiplier).toFixed(decimalPlaces);
    const errorInLastDigits = Math.round(error * multiplier);
    return `${roundedValue}(${errorInLastDigits})`;
};
//8 nov, major modif, chunks, needed if parameters changes in TASKS 2 and 3 in StartIndexing
/**
 * Creates an efficient generator for C(n, k) combinations.
 * This function is memory-efficient and yields combinations one by one
 * without storing them all in memory.
 *
 * @param {number} n - The number of items to choose from (e.g., 80 for C(80, 6)).
 * @param {number} k - The number of items to choose (e.g., 6 for C(80, 6)).
 * @returns {Generator<Uint32Array, void, void>} A generator that yields a Uint32Array.
 */
function* createCombinationGenerator(n, k) {
// 1. Initialize the first combination: [0, 1, 2, ..., k-1]
// We use Uint32Array because that's what the GPU buffer expects.
const combo = new Uint32Array(k);
for (let i = 0; i < k; i++) {
    combo[i] = i;
}

while (true) {
    // 2. Yield the current combination array.
    // The calling loop will copy this array's values into the GPU buffer.
    yield combo;

    // 3. Find the rightmost element (i) that can be incremented.
    let i = k - 1;
    
    // We check i >= 0.
    // The max value for combo[i] is (n - k + i).
    // We look for the first element from the right that is *not*
    // at its maximum value.
    while (i >= 0 && combo[i] === (n - k + i)) {
        i--;
    }

    // 4. If i < 0, all elements are at their max.
    // e.g., for C(80, 6), this would be [74, 75, 76, 77, 78, 79].
    // We are done.
    if (i < 0) {
        return;
    }

    // 5. Increment the element we found.
    combo[i]++;

    for (let j = i + 1; j < k; j++) {
        combo[j] = combo[j - 1] + 1;
    }
}
}
let lastThrottleTime = 0;
/**
 * Throttles a function call to only execute once every `delay` milliseconds.
 * @param {function} func The function to throttle.
 * @param {number} delay The delay in milliseconds.
 */
const throttle = (func, delay) => {
    return (...args) => {
        const now = new Date().getTime();
        if (now - lastThrottleTime < delay) {
            return;
        }
        lastThrottleTime = now;
        func(...args);
    };
};
// This creates a throttled version of the status text update.
// It will update at most 4 times per second (every 250ms).
// It can now see the global 'statusTextElement'
const throttledSetStatusText = throttle((message) => {
    if (statusTextElement) {
        statusTextElement.textContent = message;
    }
}, 250); // 250ms delay
// =================================================================
//  RECIPROCAL-SPACE CONVENTION -- see the full block at the top of
//  js/crystallography/metric.js. Short version, because the UI uses both:
//
//    Qscat = 4*pi*sin(theta)/lambda = 2*pi/d   [A^-1]
//        The scattering vector. Used by: the plot's Q axis mode, the
//        Kalpha2 stripping (q_a1 / q_a2), and the peak-finder's Radius
//        and Smoothing sliders. Use scatteringQ() below.
//
//    qsq   = 1/d^2 = 4 sin^2(theta)/lambda^2   [A^-2]
//        The indexing quantity. Anything named q_obs, q_max,
//        peaks_sorted_by_q or a peak's .q field is THIS, not the
//        above -- those values are produced by and passed to
//        the crystallography scripts, which are entirely in A^-2.
//
//  They differ by a square, not a scale factor. A value moved from one
//  to the other without conversion will not throw; it will just index
//  to a wrong cell.
// =================================================================

// Scattering vector in A^-1 from 2-theta in DEGREES. Single definition
// so new code cannot pick the wrong formula by accident.
const scatteringQ = (tth_deg, lambda) =>
    4 * Math.PI * Math.sin(tth_deg * Math.PI / 360) / lambda;
// Inverse: 2-theta in DEGREES from a scattering vector in A^-1.
const tthFromScatteringQ = (q, lambda) =>
    2 * Math.asin(Math.min(1, Math.max(0, q * lambda / (4 * Math.PI)))) * 180 / Math.PI;
