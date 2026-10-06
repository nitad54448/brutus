// js/indexing/worker-pool.js
// The CPU index worker URL and the refinement worker pool.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

const setupWorker = () => {
try {
    workerURL = 'js/workers/index-worker.js' + APP_VERSION_QS;
} catch (error) {
    console.error("Failed to set up worker URL:", error);
    showStatus("Critical error: Could not initialize indexing engine.", "error", 10000);
}
};
// ========================================================================
// Refinement Worker Pool (batched)
// ========================================================================
// Parallelises the CPU refinement of GPU-produced candidate cells across N
// Web Workers. Each worker loads the crystallography scripts plus the refinement-worker.js
// shim. The pool exposes:
//   - init(sharedState): sends one-time per-run constants to each worker
//   - refineBatch(cells): splits an array of cells across workers in round-robin
//     slices. Each slice is sent as a single 'refineBatch' postMessage, amortising
//     structured-clone overhead across many cells. Returns a Promise that resolves
//     once every worker has acked its slice.
//   - refine(cell): convenience wrapper around refineBatch([cell]).
//   - drain(): resolves once all outstanding batches have completed
//   - reset(): clears each worker's dedup map between runs
//   - terminate(): kills workers (used on global stop)
//
// Design notes:
//   - Workers dedup internally (own foundSolutionMap). Cross-worker duplicates
//     are resolved by applyFinalSieve at end-of-run.
//   - Batching: when GPU hands us K cells, we split them evenly across N workers,
//     sending N messages instead of K. For K=50 and N=18 that's 18 messages
//     instead of 50. For K=50000 total over the run, we go from 50000 messages
//     to a few thousand at most.
//   - Resolvers are keyed by batch id. Each postMessage creates one pending entry.
// ========================================================================
class RefinementWorkerPool {
    constructor(size, onSolution = handleNewSolution, onFailure = console.warn) {
        this.size = Math.max(1, size | 0);
        this.onSolution = onSolution;
        this.onFailure = onFailure;
        this.workers = [];
        this.nextBatchId = 1;
        this.pendingResolvers = new Map();
        this.rrIndex = 0;
        this._initialised = false;
        this._initPayload = null;
        this._crashCount = 0;
        this._drainWake = null;
    }
    _now() { return performance.now(); }
    _wake() {
        if (this._drainWake) { const wake = this._drainWake; this._drainWake = null; wake(); }
    }
    _removeWorker(w) {
        // Invalidate membership before termination, so already-queued events
        // cannot deliver solutions from abandoned work.
        this.workers = this.workers.filter(worker => worker !== w);
        try { w.terminate(); } catch (_) {}
        for (const id of w.activeBatches) {
            const resolve = this.pendingResolvers.get(id);
            this.pendingResolvers.delete(id);
            if (resolve) resolve();
        }
        w.activeBatches.clear();
        this._wake();
    }
    _spawn() {
        if (this._crashCount > this.size * 3) return;
        while (this.workers.length < this.size) {
            const w = new Worker('js/workers/refinement-worker.js' + APP_VERSION_QS);
            w.activeBatches = new Set();
            w.lastActivity = this._now();
            this.workers.push(w);
            w.onmessage = e => this._onMessage(e, w);
            w.onerror = err => {
                if (!this.workers.includes(w)) return;
                this._crashCount++;
                this.onFailure('Refinement worker failed; some candidates were not refined.');
                console.error('Refinement worker failed:', err.message);
                this._removeWorker(w);
            };
            if (this._initPayload) w.postMessage(this._initPayload);
        }
    }
    _onMessage(e, w) {
        const msg = e.data;
        if (!this._initialised || !this.workers.includes(w) || !msg || !msg.type) return;
        const id = msg.batchId !== undefined ? msg.batchId : msg.taskId;
        if (!w.activeBatches.has(id) || !this.pendingResolvers.has(id)) return;
        w.lastActivity = this._now();
        if (msg.type === 'solutions') {
            if (Array.isArray(msg.payloads)) msg.payloads.forEach(sol => this.onSolution(sol));
        } else if (msg.type === 'solution') {
            this.onSolution(msg.payload);
        } else if (msg.type === 'heartbeat') {
            // Sent every ~2 s while a worker grinds through a long batch.
            // Its only job is the lastActivity refresh above, which keeps
            // drain()'s stall watchdog from abandoning a busy worker.
        } else if (msg.type === 'cellError' || msg.type === 'batchError') {
            this.onFailure('Some candidate cells could not be refined.');
            console.warn(`Refinement batch ${id}:`, msg.message);
        } else if (msg.type === 'done') {
            if (msg.errors) this.onFailure('Some candidate cells could not be refined.');
            w.activeBatches.delete(id);
            const resolve = this.pendingResolvers.get(id);
            this.pendingResolvers.delete(id);
            if (resolve) resolve();
            this._wake();
        }
    }
    // Per-run callbacks. The pool lives for the whole page; each run
    // points it at its own solution sink and failure reporter.
    setHandlers(onSolution, onFailure) {
        this.onSolution = onSolution || handleNewSolution;
        this.onFailure = onFailure || console.warn;
    }

    // Start a run. Idle workers are re-initialised IN PLACE -- their
    // 'init' handler rebuilds every piece of per-run state -- instead of
    // being terminated and respawned, which made every worker re-download
    // and re-compile the crystallography scripts on every run. A worker
    // still busy with an earlier run's batch cannot be interrupted, so
    // only those are replaced.
    init(shared) {
        for (const w of [...this.workers]) if (w.activeBatches.size) this._removeWorker(w);
        this._crashCount = 0;
        this._initPayload = { type: 'init', ...shared };
        this._initialised = true;
        for (const w of this.workers) w.postMessage(this._initPayload);
        this._spawn();
    }

    // End of a run. Late messages are dropped (not initialised), busy
    // workers are discarded, idle ones are kept for the next run.
    release() {
        this._initialised = false;
        for (const w of [...this.workers]) if (w.activeBatches.size) this._removeWorker(w);
        for (const resolve of this.pendingResolvers.values()) resolve();
        this.pendingResolvers.clear();
        this._wake();
    }
    reset() {
        for (const w of this.workers) w.postMessage({ type: 'reset' });
    }
    refineBatch(cells) {
        if (!this._initialised || !cells || !cells.length) return Promise.resolve();
        this._spawn();
        if (!this.workers.length) {
            this.onFailure('Refinement is unavailable; candidate cells were skipped.');
            return Promise.resolve();
        }
        const chunks = Math.min(this.workers.length, cells.length);
        const size = Math.floor(cells.length / chunks), extra = cells.length % chunks;
        const promises = [];
        let offset = 0;
        for (let c = 0; c < chunks; c++) {
            const n = size + (c < extra ? 1 : 0);
            const slice = cells.slice(offset, offset + n); offset += n;
            this.rrIndex %= this.workers.length;
            const w = this.workers[this.rrIndex++];
            const batchId = this.nextBatchId++;
            promises.push(new Promise(resolve => this.pendingResolvers.set(batchId, resolve)));
            // New queued work must not reset a busy worker's watchdog clock.
            if (!w.activeBatches.size) w.lastActivity = this._now();
            w.activeBatches.add(batchId);
            try { w.postMessage({ type: 'refineBatch', cells: slice, batchId }); }
            catch (err) {
                this.onFailure('A refinement batch could not be sent.');
                this._removeWorker(w);
                break;
            }
        }
        return Promise.all(promises);
    }
    mark() { return this.nextBatchId; }
    refine(cell) { return this.refineBatch([cell]); }
    async drain(stallMs = 30000, sinceId = 0) {
        const outstanding = () => [...this.pendingResolvers.keys()].some(id => id >= sinceId);
        while (outstanding()) {
            await new Promise(resolve => {
                let tick;
                const wake = () => { clearTimeout(tick); resolve(); };
                this._drainWake = wake;
                tick = setTimeout(wake, 250);
            });
            this._drainWake = null;
            for (const w of [...this.workers]) {
                if (![...w.activeBatches].some(id => id >= sinceId)) continue;
                if (this._now() - w.lastActivity <= stallMs) continue;
                this.onFailure('Refinement timed out; remaining candidates on that worker were abandoned.');
                this._removeWorker(w);
            }
        }
    }
    terminate() {
        this._initialised = false;
        for (const w of [...this.workers]) this._removeWorker(w);
        for (const resolve of this.pendingResolvers.values()) resolve();
        this.pendingResolvers.clear();
        this._initPayload = null;
        this._wake();
    }
}
// Spawn size: one less than hardware concurrency, to leave a core for UI + GPU
// driver. Minimum 1.
// Leave two hardware threads for the UI and the GPU driver, and cap the
// pool: past ~12 workers the GPU, not refinement, is the bottleneck,
// while each worker still costs a full copy of the crystallography code.
const REFINE_POOL_MAX = 12;
const REFINE_POOL_SIZE = Math.max(1, Math.min(REFINE_POOL_MAX, (navigator.hardwareConcurrency || 4) - 2));
const refinementPool = new RefinementWorkerPool(REFINE_POOL_SIZE);
console.log(`[perf] Refinement worker pool: ${REFINE_POOL_SIZE} workers (hardwareConcurrency=${navigator.hardwareConcurrency || 'unknown'})`);
