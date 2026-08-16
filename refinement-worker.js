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
//      on each cell in the batch, posts back { type: 'solutions', payloads, batchId }
//      (one message per SOLUTION_FLUSH_SIZE accepted cells, plus a final flush),
//      then posts exactly one { type: 'done', batchId } at the end of the batch.
//   3. (Legacy) { type: 'refine', cell, taskId } is still supported as a
//      single-cell batch, for any code that hasn't been migrated.
//   4. Main thread may call { type: 'reset' } between runs to clear per-worker
//      state (mainly the worker's own foundSolutionMap).
//
// The worker imports worker-logic.js which defines refineAndTestSolution and friends.
self.IS_REFINEMENT_WORKER = true;

// The pool spawns us as 'refinement-worker.js?v=YYYYMMDD'. Importing
// 'worker-logic.js' bare meant the 350 KB logic file was fetched on its own
// cache key: a deploy that bumped the version string got a fresh shim paired
// with whatever worker-logic.js the HTTP cache still held. Inherit the query
// string so both halves are always versioned together. Falls back to no query
// (e.g. a blob: worker URL), which is the old behaviour.
(function importWorkerLogic() {
    let q = '';
    try { q = (self.location && self.location.search) || ''; } catch (_) { q = ''; }
    importScripts('worker-logic.js' + q);
})();

// Per-worker state, initialised by the 'init' message
let state = null;
let baseParams = null;
let foundSolutions = null;
let foundSolutionMap = new Map();

// --- The write-only cell ledger -------------------------------------------
//
// refineAndTestSolution keeps two structures in `state`: foundSolutionMap
// (key -> { m20, index }) and foundSolutions (array of accepted cells). Only
// the MAP is ever read -- the dedup test is `existing.m20`. The ARRAY is
// write-only on this code path: nothing in the refinement worker reads a cell
// back out of it, and the only function that does (findTransformedSolutions,
// via combinatorialSwapSearch) is never called here. Every accepted cell is
// posted to the main thread immediately and the main thread keeps the
// authoritative list.
//
// So the array was pinning up to MAX_WORKER_LEDGER fully-refined cell objects
// per worker, times the pool size, for nothing. This stand-in satisfies the
// three things refineAndTestSolution actually does to it --
//
//     foundSolutions.push(cell)
//     foundSolutions[existing.index] = cell
//     foundSolutions.length - 1
//
// -- while retaining no references at all. Index writes are swallowed by the
// set trap; `length` still advances so the indices handed to the map stay
// unique and monotonic.
//
// Reading a cell back would now silently yield undefined, so the get trap
// throws on the array methods that would imply someone is trying. That turns a
// future misuse into a loud, caught cellError instead of silent bad data.
const GHOST_INDEX_RE = /^(0|[1-9]\d*)$/;
const GHOST_FORBIDDEN = new Set([
    'slice', 'map', 'forEach', 'filter', 'reduce', 'reduceRight', 'sort',
    'indexOf', 'lastIndexOf', 'find', 'findIndex', 'concat', 'join', 'pop',
    'shift', 'unshift', 'splice', 'some', 'every', 'includes', 'flat',
    'flatMap', 'at', 'entries', 'keys', 'values', 'reverse', 'fill', 'copyWithin'
]);
// Iteration is the hole the string list above can't cover: `[...foundSolutions]`,
// `Array.from(...)` and `for (const c of ...)` all go through Symbol.iterator,
// which used to be absent on the plain-object target and so failed with a bare
// "is not iterable" instead of the diagnostic below.
const GHOST_FORBIDDEN_SYMBOLS = new Set([Symbol.iterator]);

function createGhostLedger() {
    // The target is a REAL array, not a plain object, purely so that
    // Array.isArray(state.foundSolutions) stays true. With an object target it
    // was silently false, which is exactly the sort of thing a future
    // `if (Array.isArray(...))` branch gets wrong without saying anything.
    // It never holds an element -- the set trap drops index writes -- so this
    // stays a length-only sparse array and costs nothing.
    const target = [];
    Object.defineProperty(target, 'push', {
        value: function push() { return ++target.length; },
        writable: true, configurable: true, enumerable: false
    });
    return new Proxy(target, {
        set(t, prop, value) {
            // Drop element writes on the floor; keep `length` (and anything
            // non-numeric) working normally.
            if (typeof prop === 'string' && GHOST_INDEX_RE.test(prop)) return true;
            t[prop] = value;
            return true;
        },
        get(t, prop, recv) {
            if ((typeof prop === 'string' && GHOST_FORBIDDEN.has(prop)) ||
                GHOST_FORBIDDEN_SYMBOLS.has(prop)) {
                throw new TypeError(
                    `refinement-worker ghost ledger: '${String(prop)}' is unavailable — ` +
                    `this worker's foundSolutions retains no cells (see createGhostLedger)`);
            }
            return Reflect.get(t, prop, recv);
        }
    });
}

// Bound on the per-worker dedup map.
//
// The map only exists so refineAndTestSolution can recognise a cell it has
// already accepted. On a long GPU run a worker can accept tens of thousands of
// cells, so it grew without limit for the whole run and was only ever released
// by 'init' or 'reset'.
//
// Trimming is safe: losing an old entry can at worst let one duplicate through
// to the main thread, which dedups by _solKey anyway.
const MAX_WORKER_LEDGER = 20000;
const TRIM_LEDGER_TO   = 15000;

function trimLedgerIfNeeded() {
    if (foundSolutionMap.size <= MAX_WORKER_LEDGER) return;

    // Map iteration is insertion-ordered, and re-set()ing an existing key keeps
    // its original position, so this drops the oldest first-seen cells.
    //
    // Note what this no longer has to do: the previous version rebuilt the map
    // from a trimmed array because the stored `index` was a POSITION in
    // foundSolutions and dropping elements shifted every later one. The ghost
    // ledger never shrinks, so indices stay valid and stale-index corruption
    // (a later better-M20 hit overwriting an unrelated cell) is structurally
    // impossible. It is also no longer O(n) in getSolutionKey/standardizeCell.
    const drop = foundSolutionMap.size - TRIM_LEDGER_TO;
    let i = 0;
    for (const key of foundSolutionMap.keys()) {
        if (i++ >= drop) break;
        // Deleting the entry the iterator is currently on is well-defined:
        // a Map iterator skips deleted entries and advances to the next live
        // one, so the previous collect-into-an-array-then-delete pass was
        // allocating a k-element array for nothing.
        foundSolutionMap.delete(key);
    }
}

// --- Outbound solution batching -------------------------------------------
//
// The inbound direction was batched (one 'refineBatch' per worker instead of
// one message per cell) but the return path still paid a structured clone per
// accepted cell. Accumulate and post in chunks; flushing every
// SOLUTION_FLUSH_SIZE keeps the UI updating during a long batch and bounds the
// size of any single clone.
const SOLUTION_FLUSH_SIZE = 64;
let solutionBuffer = [];

function flushSolutions(idField, idValue) {
    if (solutionBuffer.length === 0) return;
    const out = { type: 'solutions', payloads: solutionBuffer };
    solutionBuffer = [];              // new array: the old one is now owned by the clone
    if (idField) out[idField] = idValue;
    self.postMessage(out);
}

// --- Per-batch accounting --------------------------------------------------
//
// Two things depend on this. First, an error CAP: runOneCell used to post one
// 'cellError' per failing cell, and the main thread console.warn'd each one. A
// systematic failure -- a stale worker-logic.js, bad baseParams, a degenerate
// peak list -- makes every cell in the slice throw, so a 5000-cell batch became
// 5000 structured clones and 5000 console writes. That costs far more than the
// refinement it replaced and is enough to wedge the tab. Post the first few,
// count the rest.
//
// Second, OBSERVABILITY: 'done' now carries the counts, so a batch that the
// main thread force-resolved after the 30 s stall watchdog is distinguishable
// from one that genuinely finished, and suppressed errors are still visible as
// a number.
const MAX_CELL_ERRORS_PER_BATCH = 5;
let batchProcessed = 0;
let batchErrors = 0;
let batchErrorsPosted = 0;

// Which batch (if any) this worker is inside right now. Only non-null while a
// 'refineBatch'/'refine' handler is on the stack, so the last-resort handlers
// at the bottom of this file know what to ack. See the self.onerror block.
let currentBatch = null;

function resetBatchStats() {
    batchProcessed = 0;
    batchErrors = 0;
    batchErrorsPosted = 0;
}

function doneMessage(idField, idValue) {
    const out = { type: 'done', processed: batchProcessed, errors: batchErrors };
    out[idField] = idValue;
    return out;
}

// 13 jul 2026: restored the legacy 'refine' case below (see point 3 in the
// header) — it had been dropped from the switch statement so it fell
// through to default and was silently ignored, contradicting this file's
// own docs and leaving any legacy caller waiting forever on a reply.
function runOneCell(cell, idField, idValue, cellIndex) {
    batchProcessed++;
    try {
        refineAndTestSolution(cell, baseParams, state, (innerMsg) => {
            if (innerMsg && innerMsg.type === 'solution') {
                solutionBuffer.push(innerMsg.payload);
                if (solutionBuffer.length >= SOLUTION_FLUSH_SIZE) flushSolutions(idField, idValue);
            }
        });
    } catch (err) {
        batchErrors++;
        if (batchErrorsPosted >= MAX_CELL_ERRORS_PER_BATCH) return;
        batchErrorsPosted++;
        // Attach the batchId/taskId so the main thread knows which batch errored,
        // plus enough about the cell to actually reproduce it -- a bare message
        // with no cell identity is an unreproducible bug report.
        const out = {
            type: 'cellError',
            message: String(err && err.message || err),
            cellIndex: cellIndex,
            system: (cell && cell.system) || null,
            cell: describeCell(cell)
        };
        if (batchErrorsPosted === MAX_CELL_ERRORS_PER_BATCH) {
            out.suppressing = true;   // further errors in this batch are counted only
        }
        out[idField] = idValue;
        self.postMessage(out);
    }
}

// Compact, always-cloneable description of a cell for error reports.
function describeCell(cell) {
    if (!cell || typeof cell !== 'object') return null;
    const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? Number(v.toFixed(4)) : null;
    return {
        a: fmt(cell.a), b: fmt(cell.b), c: fmt(cell.c),
        alpha: fmt(cell.alpha), beta: fmt(cell.beta), gamma: fmt(cell.gamma)
    };
}

// Which id field a given inbound message type acks with. Sniffing
// `msg.batchId !== undefined ? 'batchId' : 'taskId'` silently mislabelled the
// ack when a refineBatch arrived with an undefined batchId, so the main thread
// resolved nothing and the run stalled until the 30 s drain watchdog.
const ID_FIELD_BY_TYPE = { refineBatch: 'batchId', refine: 'taskId' };

function makeLedger() {
    foundSolutions = createGhostLedger();
    foundSolutionMap = new Map();
}

self.onmessage = (e) => {
    const msg = e.data;
    if (!msg || !msg.type) return;

    try {
        switch (msg.type) {
            case 'init': {
                baseParams = msg.baseParams;
                // importScripts throwing kills the whole script before this file
                // ever defines onmessage, so that failure surfaces via the main
                // thread's w.onerror. This catches the quieter case: the fetch
                // succeeded but the file we got doesn't define what we need
                // (version skew, a truncated response, a cached 200-with-HTML
                // error page). Without it, every cell in every batch throws the
                // same ReferenceError and the real cause is buried.
                if (typeof refineAndTestSolution !== 'function') {
                    self.postMessage({
                        type: 'cellError',
                        message: 'worker-logic.js loaded but refineAndTestSolution is not defined ' +
                                 '(version mismatch or truncated response) — this worker cannot refine'
                    });
                }
                // Clear per-worker state on init. Without this, a second 'init'
                // (a new run that doesn't send 'reset' first) would keep the old
                // foundSolutionMap, leaking stale solutions into the new run and
                // growing the dedup map unbounded across runs.
                makeLedger();
                solutionBuffer = [];
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
                makeLedger();
                solutionBuffer = [];
                if (state) {
                    state.foundSolutions = foundSolutions;
                    state.foundSolutionMap = foundSolutionMap;
                }
                break;
            }

            case 'refineBatch': {
                const { cells, batchId } = msg;
                resetBatchStats();
                if (!state || !baseParams) {
                    self.postMessage({ type: 'cellError', message: 'Worker not initialized (missing state/baseParams)', batchId });
                    batchErrors++;
                    self.postMessage(doneMessage('batchId', batchId));
                    return;
                }
                if (!Array.isArray(cells)) {
                    self.postMessage({ type: 'cellError', message: 'cells payload is not an array', batchId });
                    batchErrors++;
                    self.postMessage(doneMessage('batchId', batchId));
                    return;
                }
                currentBatch = { idField: 'batchId', idValue: batchId };
                for (let i = 0; i < cells.length; i++) {
                    runOneCell(cells[i], 'batchId', batchId, i);
                }
                // Solutions must reach the main thread BEFORE the ack that
                // resolves this batch's promise.
                flushSolutions('batchId', batchId);
                // Once per batch, not per cell: the check is O(1) but the
                // occasional trim is O(k).
                trimLedgerIfNeeded();
                self.postMessage(doneMessage('batchId', batchId));
                currentBatch = null;
                break;
            }

            case 'refine': {
                const { cell, taskId } = msg;
                resetBatchStats();
                if (!state || !baseParams) {
                    self.postMessage({ type: 'cellError', message: 'Worker not initialized (missing state/baseParams)', taskId });
                    batchErrors++;
                    self.postMessage(doneMessage('taskId', taskId));
                    return;
                }
                currentBatch = { idField: 'taskId', idValue: taskId };
                runOneCell(cell, 'taskId', taskId, 0);
                flushSolutions('taskId', taskId);
                // The legacy path grew the ledger without ever trimming it,
                // which defeated the bound entirely for any caller still on it.
                trimLedgerIfNeeded();
                self.postMessage(doneMessage('taskId', taskId));
                currentBatch = null;
                break;
            }

            default:
                break;
        }
    } catch (err) {
        // Catch-all to guarantee the main thread is NEVER left hanging
        const idField = ID_FIELD_BY_TYPE[msg.type] ||
                        ((msg.batchId !== undefined) ? 'batchId' : 'taskId');
        const idValue = msg[idField];
        currentBatch = null;
        if (idValue !== undefined) {
            // Don't discard solutions already accepted in this batch.
            try { flushSolutions(idField, idValue); } catch (_) { solutionBuffer = []; }
            batchErrors++;
            self.postMessage({ type: 'cellError', message: `Fatal batch error: ${err.message || err}`, [idField]: idValue });
            self.postMessage(doneMessage(idField, idValue));
        } else {
            solutionBuffer = [];
            console.error("Fatal worker error without batch ID:", err);
        }
    }
};

// --- Last-resort acks ------------------------------------------------------
//
// Everything above is wrapped in try/catch, so the only way to throw past it is
// to throw outside the onmessage frame: a compile/eval error in worker-logic.js
// during importScripts, a rejected promise from anything that ever goes async,
// or a stack overflow unwinding oddly. The main thread's pool does force-resolve
// those batches, but only after the 30 s stall watchdog, and only by declaring
// the whole worker dead. Acking here turns a 30 s pause into an immediate,
// attributable failure of one batch.
//
// Note the ordering contract is the same as the normal path: flush buffered
// solutions, then post 'done'. Nothing accepted before the crash is lost.
function bailOut(what, err) {
    const message = `${what}: ${(err && (err.message || err.reason || err)) || 'unknown'}`;
    if (!currentBatch) {
        console.error('[refinement-worker]', message, err);
        return false;
    }
    const { idField, idValue } = currentBatch;
    currentBatch = null;
    try { flushSolutions(idField, idValue); } catch (_) { solutionBuffer = []; }
    batchErrors++;
    try {
        self.postMessage({ type: 'cellError', message, [idField]: idValue });
        self.postMessage(doneMessage(idField, idValue));
    } catch (_) { /* the port is gone; nothing left to do */ }
    return true;
}

self.onerror = (evt) => {
    // evt is an ErrorEvent for real script errors; some engines pass a string.
    const err = (evt && evt.error) || (evt && evt.message) || evt;
    bailOut('Uncaught worker error', err);
    // Deliberately NOT preventDefault(): the main thread's w.onerror is what
    // splices a genuinely broken worker out of the pool, and suppressing the
    // event would keep handing batches to a worker whose logic file failed to
    // load. The ack above just stops that batch waiting 30 s first.
};

self.onunhandledrejection = (evt) => {
    // refineAndTestSolution is synchronous today, so this should never fire.
    // If it ever does, it means something in the refinement path went async
    // and the batch would otherwise hang with no error surfaced anywhere.
    if (bailOut('Unhandled rejection in worker', evt && evt.reason)) {
        if (evt && typeof evt.preventDefault === 'function') evt.preventDefault();
    }
};
