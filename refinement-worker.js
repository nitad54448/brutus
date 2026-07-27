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
self.IS_REFINEMENT_WORKER = true;
importScripts('worker-logic.js');

// Per-worker state, initialised by the 'init' message
let state = null;
let baseParams = null;
let foundSolutions = [];
let foundSolutionMap = new Map();

// Bound on the per-worker ledgers.
//
// foundSolutions and foundSolutionMap only exist so refineAndTestSolution can
// recognise a cell it has already accepted; nothing reads them back at the end
// of the run (every accepted cell is posted to the main thread immediately, and
// the main thread keeps the authoritative list). On a long GPU run a worker can
// accept tens of thousands of cells, so both structures grew without limit for
// the whole run and were only ever released by 'init' or 'reset'.
//
// Trimming is safe: losing an old entry can at worst let one duplicate through
// to the main thread, which dedups by _solKey anyway.
const MAX_WORKER_LEDGER = 20000;
const TRIM_LEDGER_TO   = 15000;

function trimLedgerIfNeeded() {
    if (foundSolutions.length <= MAX_WORKER_LEDGER) return;

    // CAREFUL: foundSolutionMap values are { m20, index } where `index` is a
    // POSITION in foundSolutions (see refineAndTestSolution, which does
    // `foundSolutions[existing.index] = ...`). Dropping elements shifts every
    // later position, so the map has to be rebuilt from the survivors rather
    // than patched entry by entry -- otherwise a subsequent better-M20 hit
    // overwrites an unrelated cell.
    //
    // Both structures are mutated IN PLACE. refineAndTestSolution destructures
    // them out of `state` on entry, so replacing them with fresh objects here
    // would leave that code writing into an orphaned array.
    const keep = foundSolutions.slice(foundSolutions.length - TRIM_LEDGER_TO);
    foundSolutions.length = 0;
    for (let i = 0; i < keep.length; i++) foundSolutions.push(keep[i]);

    foundSolutionMap.clear();
    for (let i = 0; i < foundSolutions.length; i++) {
        const sol = foundSolutions[i];
        let key;
        try { key = getSolutionKey(sol); } catch (_) { key = undefined; }
        if (key === undefined || key === null) continue;
        const prev = foundSolutionMap.get(key);
        // slice() can only have kept one entry per key, but rebuild defensively.
        if (!prev || sol.m20 > prev.m20) {
            foundSolutionMap.set(key, { m20: sol.m20, index: i });
        }
    }
}

// 13 jul 2026: restored the legacy 'refine' case below (see point 3 in the
// header) — it had been dropped from the switch statement so it fell
// through to default and was silently ignored, contradicting this file's
// own docs and leaving any legacy caller waiting forever on a reply.
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
        // Attach the batchId/taskId so the main thread knows which batch errored
        const out = { type: 'cellError', message: String(err && err.message || err) };
        out[idField] = idValue;
        self.postMessage(out);
    }
}

self.onmessage = (e) => {
    const msg = e.data;
    if (!msg || !msg.type) return;

    try {
        switch (msg.type) {
            case 'init': {
                baseParams = msg.baseParams;
                // Clear per-worker state on init. Without this, a second 'init'
                // (a new run that doesn't send 'reset' first) would keep the old
                // foundSolutions/foundSolutionMap, leaking stale solutions into the
                // new run and growing the dedup map unbounded across runs.
                foundSolutions = [];
                foundSolutionMap = new Map();
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
                    self.postMessage({ type: 'cellError', message: 'Worker not initialized (missing state/baseParams)', batchId });
                    self.postMessage({ type: 'done', batchId });
                    return;
                }
                if (!Array.isArray(cells)) {
                    self.postMessage({ type: 'cellError', message: 'cells payload is not an array', batchId });
                    self.postMessage({ type: 'done', batchId });
                    return;
                }
                for (let i = 0; i < cells.length; i++) {
                    runOneCell(cells[i], 'batchId', batchId);
                }
                // Once per batch, not per cell: the check is O(1) but the
                // occasional rebuild is O(n).
                trimLedgerIfNeeded();
                self.postMessage({ type: 'done', batchId });
                break;
            }

            case 'refine': {
                const { cell, taskId } = msg;
                if (!state || !baseParams) {
                    self.postMessage({ type: 'cellError', message: 'Worker not initialized (missing state/baseParams)', taskId });
                    self.postMessage({ type: 'done', taskId });
                    return;
                }
                runOneCell(cell, 'taskId', taskId);
                self.postMessage({ type: 'done', taskId });
                break;
            }

            default:
                break;
        }
    } catch (err) {
        // Catch-all to guarantee the main thread is NEVER left hanging
        const idField = (msg.batchId !== undefined) ? 'batchId' : 'taskId';
        const idValue = msg[idField];
        if (idValue !== undefined) {
            self.postMessage({ type: 'cellError', message: `Fatal batch error: ${err.message || err}`, [idField]: idValue });
            self.postMessage({ type: 'done', [idField]: idValue });
        } else {
            console.error("Fatal worker error without batch ID:", err);
        }
    }
};