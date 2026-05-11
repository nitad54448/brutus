// refinement-worker.js
//
// Each refinement worker is a stateless-ish CPU refinement engine.
//
// Lifecycle:
//   1. Main thread sends { type: 'init', ... } once at the start of indexing.
//      This passes the constants that don't change per-cell: wavelength, peak data,
//      tolerances, baseParams. We store them on the worker's closure state.
//   2. For each batch of GPU-found candidate cells, main thread sends
//      { type: 'refineBatch', cells, batchId }. Worker runs refineAndTestSolution
//      on each cell in the batch, posts back { type: 'solution', payload, batchId }
//      for each cell that yields a solution, then posts exactly one
//      { type: 'done', batchId } at the end of the batch.
//   3. (Legacy) { type: 'refine', cell, taskId } is still supported as a
//      single-cell batch, for any code that hasn't been migrated.
//   4. Main thread may call { type: 'reset' } between runs to clear per-worker
//      state (mainly the worker's own foundSolutionMap).
//
// The worker imports worker-logic.js which defines refineAndTestSolution and friends.

importScripts('worker-logic.js');

// Per-worker state, initialised by the 'init' message
let state = null;
let baseParams = null;
let foundSolutions = [];
let foundSolutionMap = new Map();

//possible issue here; changed on 11th may 2026... kept the old version
function runOneCell(cell, idField, idValue) {
    try {
        refineAndTestSolution(cell, baseParams, state, (innerMsg) => {
            if (innerMsg && innerMsg.type === 'solution') {
                const out = { type: 'solution', payload: innerMsg.payload };
                out[idField] = idValue;
                self.postMessage(out);
            }
        });
    } catch (err) {
        // Don't let one bad cell take down the whole batch.
        self.postMessage({ type: 'cellError', message: String(err && err.message || err) });
    }
}

self.onmessage = (e) => {
    const msg = e.data;
    if (!msg || !msg.type) return;

    switch (msg.type) {
        case 'init': {
            baseParams = msg.baseParams;
            state = {
                q_obs: msg.q_obs,
                original_indices: msg.original_indices,
                tth_obs_rad: msg.tth_obs_rad,
                peaks_sorted_by_q: msg.peaks_sorted_by_q,
                N_FOR_M20: msg.N_FOR_M20,
                min_m20: msg.min_m20,
                q_max: msg.q_max,
                d_min: msg.d_min,
                foundSolutions,
                foundSolutionMap,
            };
            break;
        }

        case 'reset': {
            foundSolutions = [];
            foundSolutionMap = new Map();
            if (state) {
                state.foundSolutions = foundSolutions;
                state.foundSolutionMap = foundSolutionMap;
            }
            break;
        }

        case 'refineBatch': {
            const { cells, batchId } = msg;
            if (!state || !baseParams) {
                self.postMessage({ type: 'done', batchId });
                return;
            }
            for (let i = 0; i < cells.length; i++) {
                runOneCell(cells[i], 'batchId', batchId);
            }
            self.postMessage({ type: 'done', batchId });
            break;
        }

        case 'refine': {
            // Legacy single-cell entry point, kept for backward compatibility.
            const { cell, taskId } = msg;
            if (!state || !baseParams) {
                self.postMessage({ type: 'done', taskId });
                return;
            }
            runOneCell(cell, 'taskId', taskId);
            self.postMessage({ type: 'done', taskId });
            break;
        }

        default:
            // Unknown message type; ignore.
            break;
    }
};
