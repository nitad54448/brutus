// webgpu-engine.js
class WebGPUEngine {
    // Fetched WGSL source is plain text and does not depend on the GPUDevice, so
    // it is cached at class level and survives across engine instances (i.e.
    // across indexing runs). The compiled GPUShaderModule and GPUComputePipeline
    // ARE bound to a specific device, so those live on the instance below.
    static _shaderTextCache = new Map();   // url -> Promise<string>

    constructor() {
        this.device = null;
        this.adapter = null; 
        this.shaderModule = null;
        this.pipeline = null;
        this.bindGroupLayout = null; // Store explicit layout

        this._moduleCache = new Map();   // url -> GPUShaderModule
        this._pipelineCache = new Map(); // `${url}::${entryPoint}` -> GPUComputePipeline
        this._currentShaderUrl = null;   // set by loadShader, read by createPipeline

        // Initialised HERE, not in init(). _runSolver gates on this flag, and an
        // engine whose init() threw (or was never called) used to leave it
        // `undefined` -- falsy -- so the guard passed and the run failed later
        // with an opaque validation error instead of the clear message below.
        this.deviceLost = false;
        this.destroyed = false;

        // Optional callback, assigned by the host app, invoked once if the GPU
        // device is lost. Without it a driver reset / TDR / tab suspension is
        // only visible in the console: the UI just watches the run quietly fail.
        this.onDeviceLost = null;
    }

    // 1. Initialize WebGPU
    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported on this browser.");
        }
        this.adapter = await navigator.gpu.requestAdapter(); 
        if (!this.adapter) { 
            throw new Error("No compatible GPUAdapter found.");
        }
        this.device = await this.adapter.requestDevice({
            requiredLimits: {
                maxBufferSize: this.adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
                maxStorageBuffersPerShaderStage: this.adapter.limits.maxStorageBuffersPerShaderStage
            }
        });

        // Without this a lost device (driver reset, TDR, tab suspended) surfaces
        // as a wall of GPU validation errors from stale cached pipelines rather
        // than one clear message. The cached modules/pipelines belong to THIS
        // device, so they have to go with it.
        this.deviceLost = false;
        this.device.lost.then((info) => {
            this.deviceLost = true;
            this._moduleCache.clear();
            this._pipelineCache.clear();
            this.bindGroupLayout = null;
            this.shaderModule = null;
            this.pipeline = null;
            console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
            // 'destroyed' is the reason reported when WE called device.destroy(),
            // which is a normal teardown, not a fault. Only escalate real losses.
            if (info.reason !== 'destroyed' && typeof this.onDeviceLost === 'function') {
                try { this.onDeviceLost(info); } catch (_) {}
            }
        });

        return true;
    }

    // Explicit teardown. The host app previously constructed a fresh
    // WebGPUEngine (and therefore a fresh GPUDevice) on every indexing run and
    // never released the old one, so devices and their buffers accumulated for
    // the lifetime of the page. Callers should either reuse one engine or call
    // this before dropping the reference.
    destroy() {
        if (this.destroyed) return;
        this.destroyed = true;
        this._moduleCache.clear();
        this._pipelineCache.clear();
        this.bindGroupLayout = null;
        this.shaderModule = null;
        this.pipeline = null;
        this.onDeviceLost = null;
        if (this.device) {
            try { this.device.destroy(); } catch (err) {
                console.warn('GPUDevice.destroy() failed:', err && err.message);
            }
        }
        this.device = null;
        this.adapter = null;
    }

    // True when this engine can still be used for a run.
    isUsable() {
        return !!this.device && !this.deviceLost && !this.destroyed;
    }

    // 2. Load the WGSL Shader
    //
    // The WGSL never changes at runtime, so both the network fetch and the
    // compile are cached. Previously every indexing run re-fetched and
    // re-compiled the shader before dispatching any work.
    async loadShader(url) {
        this._currentShaderUrl = url;

        // Already compiled on THIS device? Nothing to do.
        const cachedModule = this._moduleCache.get(url);
        if (cachedModule) {
            this.shaderModule = cachedModule;
            return;
        }

        // Cache the in-flight promise, not just the result, so two concurrent
        // callers for the same url share one request instead of racing.
        let textPromise = WebGPUEngine._shaderTextCache.get(url);
        if (!textPromise) {
            textPromise = fetch(url).then(response => {
                // A 404 from a dev server returns an HTML error page with status
                // 200-ish handling downstream; without this check that HTML was
                // passed to createShaderModule and surfaced as a baffling WGSL
                // syntax error instead of "file not found".
                if (!response.ok) {
                    throw new Error(`Failed to load shader ${url}: HTTP ${response.status}`);
                }
                return response.text();
            });
            WebGPUEngine._shaderTextCache.set(url, textPromise);
        }

        let shaderCode;
        try {
            shaderCode = await textPromise;
        } catch (err) {
            // Don't cache a failure permanently; let the next attempt retry.
            WebGPUEngine._shaderTextCache.delete(url);
            throw err;
        }

        const module = this.device.createShaderModule({ code: shaderCode });
        this._moduleCache.set(url, module);
        this.shaderModule = module;
    }

    // 3. Create Explicit Bind Group Layout (Fixes "Binding not present" error)
    createBindGroupLayout() {
        // The layout is identical for every system/shader, so build it once per
        // device rather than on every createPipeline call.
        if (this.bindGroupLayout) return this.bindGroupLayout;
        // Define the layout manually rather than relying on layout: 'auto'
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // q_obs
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // hkl_basis
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // peak_combos
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // binomial_table
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // solution_counter (RW)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // results_list (RW)
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // config
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }  // q_tolerances
            ]
            // debug_counter and debug_log (formerly 7 and 8) are gone; q_tolerances
            // moved 9 -> 7. Nothing ever wrote debug_log on the hot path and nothing
            // read it back, but they still counted toward maxStorageBuffersPerShaderStage,
            // whose WebGPU DEFAULT is 8. This layout used to need 9 storage buffers, so
            // createComputePipeline failed outright on any adapter reporting exactly 8 --
            // common on integrated and mobile GPUs. It now needs 7.
        });
        return this.bindGroupLayout;
    }

    // 4. Create the compute pipeline with Explicit Layout
    createPipeline(entryPoint = "main") {
        // Reuse a previously-built pipeline for this (shader, entryPoint) pair.
        // Pipeline compilation is one of the most expensive WebGPU calls, and
        // both the shader url and entry point are fixed per crystal system.
        const cacheKey = `${this._currentShaderUrl}::${entryPoint}`;
        const cachedPipeline = this._pipelineCache.get(cacheKey);
        if (cachedPipeline) {
            this.pipeline = cachedPipeline;
            return this.pipeline;
        }

        this.createBindGroupLayout(); // Ensure layout exists

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });

        this.pipeline = this.device.createComputePipeline({
            layout: pipelineLayout, // Use explicit layout instead of 'auto'
            compute: {
                module: this.shaderModule,
                entryPoint: entryPoint,
            },
        });
        this._pipelineCache.set(cacheKey, this.pipeline);
        return this.pipeline;
    }

    // 5. Helper to create a buffer and write data to it
    createBuffer(data, usage) {
        // 4-byte alignment padding if needed (though TypedArrays usually handle this)
        const buffer = this.device.createBuffer({
            size: Math.ceil(data.byteLength / 4) * 4, 
            usage: usage | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        
        // Copy data into mapped range
        if(data instanceof Float32Array) new Float32Array(buffer.getMappedRange()).set(data);
        else if(data instanceof Uint32Array) new Uint32Array(buffer.getMappedRange()).set(data);
        else new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer));
        
        buffer.unmap();
        return buffer;
    }

    // 6. Helper for Read/Storage buffers
    createReadBuffer(size) {
        return this.device.createBuffer({
            size: Math.ceil(size / 4) * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    createStorageBuffer(size) {
        return this.device.createBuffer({
            size: Math.ceil(size / 4) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    // --- Binomial Generator ---
    generateBinomialTable(n, k) {
        const stride = k + 1;
        const table = new Uint32Array((n + 1) * stride);
        const bigTable = new BigUint64Array((n + 1) * stride);

        for (let i = 0; i <= n; i++) {
            bigTable[i * stride + 0] = 1n; 
            if (i <= k) bigTable[i * stride + i] = 1n; 
            
            for (let j = 1; j < i && j <= k; j++) {
                 const val = bigTable[(i - 1) * stride + (j - 1)] + bigTable[(i - 1) * stride + j];
                 bigTable[i * stride + j] = val;
            }
        }

        let overflow = false;
        for (let i = 0; i < table.length; i++) {
            if (bigTable[i] > 4294967295n) { overflow = true; table[i] = 4294967295; }
            else table[i] = Number(bigTable[i]);
        }
        if (overflow) {
            // The shader is u32 end to end and UNRANKS combinations from this
            // table. A single clamped entry silently corrupts every hkl triplet
            // it decodes and truncates totalHklCombos, so the run returns wrong
            // cells with no error anywhere. Fail loudly instead.
            throw new Error(
                `Binomial table overflows u32 for n=${n}, k=${k} (C(n,k) > 2^32-1). ` +
                `Reduce the HKL basis size for this system.`);
        }
        return table;
    }

    // --- Cooperative yield -------------------------------------------------
    //
    // The chunk loop has to hand control back to the event loop so the progress
    // bar repaints and the Stop button stays responsive. It used to do that with
    // `await new Promise(r => setTimeout(r, 0))`.
    //
    // That is a trap: browsers clamp setTimeout to ~4 ms once the nesting level
    // exceeds 5, and an async loop that re-arms a timer from inside a timer
    // callback hits that immediately. A 2000-chunk run therefore spent ~8
    // seconds asleep doing nothing, and a run with many chunks spent most of its
    // wall time in the clamp rather than on the GPU.
    //
    // A MessageChannel post is a macrotask like setTimeout but is NOT clamped,
    // so it yields to rendering and input at a fraction of the cost.
    static _yieldChannel = null;

    static yieldToEventLoop() {
        if (typeof MessageChannel === 'undefined') {
            return new Promise(r => setTimeout(r, 0));
        }
        if (!WebGPUEngine._yieldChannel) {
            WebGPUEngine._yieldChannel = new MessageChannel();
            WebGPUEngine._yieldChannel.port1.start();
        }
        const ch = WebGPUEngine._yieldChannel;
        return new Promise(resolve => {
            const onMsg = () => { ch.port1.removeEventListener('message', onMsg); resolve(); };
            ch.port1.addEventListener('message', onMsg);
            ch.port2.postMessage(0);
        });
    }

    // How often to yield. Yielding on every chunk is wasteful when chunks are
    // short; ~16 ms is one display frame, which is as often as a repaint can
    // possibly matter.
    static YIELD_INTERVAL_MS = 16;

    // How many chunk dispatches to queue before stalling for a counter readback.
    // Each sync point costs a full CPU<->GPU round trip (onSubmittedWorkDone +
    // mapAsync), which used to be paid once per chunk. Larger values amortise
    // that further but delay the max_solutions early-out and coarsen the
    // candidate count in the status line; 8 keeps both within a frame or two.
    static CHUNKS_PER_SYNC = 8;

    // === Unified Solver (replaces runOrthoSolver / runMonoclinicSolver / runTriclinicSolver) ===
    //
    // Per-system configuration table. Each entry describes the differences between the three
    // previously-near-identical solver methods. `parseCell` reads one cell from the flat
    // f32 array returned by the GPU and produces the JS object the refinement path expects.
    //
    // Previously these three methods were ~100 lines each with ~5 small differences. Now
    // they are thin wrappers around `_runSolver(cfg, ...)`.
    static SYSTEM_CONFIGS = {
        orthorhombic: {
            K: 3,
            structFloats: 4,                // 4 f32 per cell (a, b, c, pad) = 16 bytes
            hklFloats: 4,                   // hkl_basis: vec4(h^2, k^2, l^2, 0)
            peakComboStride: 3,
            workgroupX: 8,
            workgroupY: 8,
            maxThreadsPerDispatch: 500_000,
            systemName: 'orthorhombic',
            parseCell: (r, off) => ({ a: r[off+0], b: r[off+1], c: r[off+2], system: 'orthorhombic' }),
        },
        monoclinic: {
            K: 4,
            structFloats: 4,                // 4 f32 per cell (a, b, c, beta) = 16 bytes
            hklFloats: 4,                   // hkl_basis: vec4(h^2, k^2, l^2, h*l)
            peakComboStride: 4,
            workgroupX: 8,
            workgroupY: 8,
            maxThreadsPerDispatch: 500_000,
            systemName: 'monoclinic',
            parseCell: (r, off) => ({ a: r[off+0], b: r[off+1], c: r[off+2], beta: r[off+3], system: 'monoclinic' }),
        },
        triclinic: {
            K: 6,
            structFloats: 8,                // 8 f32 per cell (a,b,c,alpha,beta,gamma,pad,pad) = 32 bytes
            hklFloats: 8,                   // hkl_basis: vec4(h^2,k^2,l^2,k*l) + vec4(h*l,h*k,0,0)
            peakComboStride: 6,
            workgroupX: 4,                  // Triclinic: more work per thread -> smaller groups
            workgroupY: 4,
            maxThreadsPerDispatch: 50_000,  // Triclinic: TDR protection
            systemName: 'triclinic',
            parseCell: (r, off) => ({
                a: r[off+0], b: r[off+1], c: r[off+2],
                alpha: r[off+3], beta: r[off+4], gamma: r[off+5],
                system: 'triclinic',
            }),
        },
    };

    async _runSolver(cfg, qObsArray, hklBasisArray, peakCombos, hklCombos, qTolerancesArray,
                     progressCallback, stopSignal, baseParams, onIntermediateResults = null) {
        if (this.deviceLost) throw new Error("WebGPU device was lost; reload the page to restart the GPU engine.");
        if (!this.pipeline) throw new Error("Pipeline not created.");

        const K_VALUE = cfg.K;

        // Packing invariants. hkl_basis now holds PRECOMPUTED hkl products, not
        // raw indices: `cfg.hklFloats` f32 per reflection (4 for ortho/mono,
        // 8 for triclinic — see HKL_PACKERS in main_app.js and the binding
        // comments in the shaders). peak_combos is `peakComboStride` u32 per
        // combo. If either array is ever built with a length that isn't a whole
        // number of records, n_hkls / numPeakCombos silently go fractional and
        // mis-size the combinadic space and dispatch — no throw, just wrong
        // results. Fail loud and early instead.
        const hklFloats = cfg.hklFloats;
        if (hklBasisArray.length % hklFloats !== 0) {
            throw new Error(`hklBasisArray length ${hklBasisArray.length} is not a multiple of ${hklFloats} for system ${cfg.systemName}.`);
        }
        // The shaders read PRECOMPUTED PRODUCTS, not raw indices. For ortho and
        // mono both forms have the same 4-float stride, so sending the raw
        // [h,k,l,pad] form is invisible to every check above: the dispatch
        // succeeds, every candidate cell is nonsense, the FoM rejects all of
        // them, and the run reports no solutions with no error anywhere. That
        // is not a hypothetical -- it is what happens when main_app.js and this
        // file drift apart. buildHklBasis stamps what it produced; refuse
        // anything else.
        if (baseParams.hklPacking !== 'products/v1') {
            throw new Error(
                `hkl basis packing mismatch: expected 'products/v1', got ` +
                `${baseParams.hklPacking === undefined ? 'nothing' : `'${baseParams.hklPacking}'`}. ` +
                `main_app.js buildHklBasis() and webgpu-engine.js are out of sync — ` +
                `the shaders need precomputed hkl products, not raw indices.`);
        }
        if (peakCombos.length % cfg.peakComboStride !== 0) {
            throw new Error(`peakCombos length ${peakCombos.length} is not a multiple of stride ${cfg.peakComboStride} for system ${cfg.systemName}.`);
        }

        const n_hkls = hklBasisArray.length / hklFloats;
        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        const maxSolutions = baseParams.max_solutions || 20000;
        const solutionStructSize = cfg.structFloats * 4; // bytes per cell

        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        // 32 bytes, not 4: slot 0 is the solution count as before, slots 1-3 are
        // run diagnostics (see the binding comment in the shaders). Slot 2 is an
        // atomicMin, so it must start at u32 MAX -- WebGPU zero-fills new
        // buffers, and a min seeded at 0 would stay 0 forever and report every
        // run as having produced a zero-volume cell.
        const counterBuffer = this.createStorageBuffer(32);
        const counterInit = new Uint32Array(8);
        counterInit[2] = 0xFFFFFFFF;
        this.device.queue.writeBuffer(counterBuffer, 0, counterInit);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(32);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);

        // (debugCounterBuffer / debugLogBuffer removed along with bindings 7 and 8.)

        // 64, not 48: Config gained a fourth vec4<f32> holding the physical
        // limits (min_axis, max_axis, min_volume) that extractCell* used to
        // hard-code while main_app.js kept its own copy of the same numbers.
        const configBufferSize = 64;
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        const userPeaksSetting = baseParams.gpu_peaks_count || 7;
        const targetFomCount = Math.max(10, userPeaksSetting);
        const finalFomCount = Math.min(qObsArray.length, targetFomCount);

        configViewU32[0] = 0;                                   // z_offset (overwritten per chunk)
        configViewU32[1] = baseParams.impurity_peaks;
        configViewU32[2] = finalFomCount;
        // n_hkls, NOT hklBasisArray.length / 4: the triclinic basis is packed at
        // 8 f32 per reflection, so the old literal 4 would have told the shader
        // there were twice as many reflections as there are — a FoM loop running
        // off the end of the basis and a combinadic space sized against the
        // wrong n. Both fields are reflection COUNTS.
        configViewU32[3] = n_hkls;   // n_hkl_for_fom
        configViewU32[4] = n_hkls;   // n_basis_total
        configViewU32[5] = totalHklCombos;
        configViewU32[6] = maxSolutions;                         // FIXED: use resolved maxSolutions, not raw baseParams.max_solutions
        configViewU32[7] = 0;
        configViewF32[8] = baseParams.wavelength;
        configViewF32[9] = baseParams.tth_error;
        configViewF32[10] = baseParams.max_volume;
        configViewF32[11] = baseParams.fom_threshold;
        // f_params2: physical limits, previously hard-coded inside every shader.
        // Defaults reproduce the old literals exactly, so behaviour is unchanged
        // unless the caller deliberately overrides them.
        const minAxis   = Number.isFinite(baseParams.min_axis)   ? baseParams.min_axis   : 2.0;
        const maxAxis   = Number.isFinite(baseParams.max_axis)   ? baseParams.max_axis   : 50.0;
        const minVolume = Number.isFinite(baseParams.min_volume) ? baseParams.min_volume : 20.0;
        configViewF32[12] = minAxis;
        configViewF32[13] = maxAxis;
        configViewF32[14] = minVolume;
        configViewF32[15] = 0;

        this.device.queue.writeBuffer(configBuffer, 0, configData);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: qObsBuffer } },
                { binding: 1, resource: { buffer: hklBasisBuffer } },
                { binding: 2, resource: { buffer: peakCombosBuffer } },
                { binding: 3, resource: { buffer: binomialBuffer } },
                { binding: 4, resource: { buffer: counterBuffer } },
                { binding: 5, resource: { buffer: resultsBuffer } },
                { binding: 6, resource: { buffer: configBuffer } },
                { binding: 7, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        const numPeakCombos = peakCombos.length / cfg.peakComboStride;
        const maxHklPerDispatch = Math.floor(cfg.maxThreadsPerDispatch / Math.max(1, numPeakCombos));
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / cfg.workgroupY);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383));
        const hklsPerChunk = safeWorkgroupsY * cfg.workgroupY;
        const workgroupsX = Math.ceil(numPeakCombos / cfg.workgroupX);
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        // Chunk-count blow-up guard. hklsPerChunk shrinks as numPeakCombos grows
        // (maxThreadsPerDispatch is a fixed budget), so raising "Peaks to
        // Combine" quietly multiplies the number of dispatches. Triclinic at 20
        // peaks gives C(20,6)=38760 combos -> maxHklPerDispatch=1 -> 4 hkls per
        // chunk -> ~1e9 chunks for C(123,6): a run that never ends, with no
        // error and a progress bar that looks merely slow. Report the geometry
        // so the caller can warn before committing.
        if (typeof progressCallback === 'function' && progressCallback.reportPlan) {
            try {
                progressCallback.reportPlan({
                    totalChunks, hklsPerChunk, numPeakCombos, totalHklCombos,
                    system: cfg.systemName,
                });
            } catch (_) {}
        }
        if (totalChunks > 200000) {
            console.warn(
                `[WebGPUEngine] ${cfg.systemName}: ${totalChunks.toLocaleString()} dispatches ` +
                `(${hklsPerChunk} hkl/chunk x ${numPeakCombos} peak combos). This will be very slow. ` +
                `Reduce "Peaks to Combine" or "HKL Basis Size".`);
        }

        let solutionsReadCount = 0;
        let stoppedEarly = false;
        let lastYield = performance.now();
        // Last counter value actually read back; used for progress on the chunks
        // that no longer stall for a readback (see CHUNKS_PER_SYNC below).
        let lastKnownSolutions = 0;
        // Run diagnostics, refreshed at every sync point. When a run finds
        // nothing these are the only evidence of WHY, so they are reported
        // whether or not the run succeeded.
        const diag = { peaksInBudget: 0, volMin: 0xFFFFFFFF, volMax: 0 };

        // Byte alignment for copyBufferToBuffer: WebGPU requires size to be a multiple of 4.
        // One cell is already a multiple of 4 bytes (4 or 8 f32s), so any count*structSize is safe.

try {
            for (let i = 0; i < totalChunks; i++) {
                if (stopSignal.stop) break;
                // Yield at most once per frame instead of once per chunk, via an
                // unclamped MessageChannel task (see yieldToEventLoop above).
                const nowMs = performance.now();
                if (nowMs - lastYield >= WebGPUEngine.YIELD_INTERVAL_MS) {
                    lastYield = nowMs;
                    await WebGPUEngine.yieldToEventLoop();
                    if (stopSignal.stop) break;
                }

                configViewU32[0] = i * hklsPerChunk;
                this.device.queue.writeBuffer(configBuffer, 0, configData);

                const commandEncoder = this.device.createCommandEncoder();
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(this.pipeline);
                passEncoder.setBindGroup(0, bindGroup);
                passEncoder.dispatchWorkgroups(workgroupsX, safeWorkgroupsY, 1);
                passEncoder.end();

                // Only the LAST chunk of a sync group copies the counter back.
                const isSyncChunk = ((i + 1) % WebGPUEngine.CHUNKS_PER_SYNC === 0) || (i === totalChunks - 1);
                if (isSyncChunk) {
                    // Copy only the counter. We will decide how many result bytes to
                    // copy once we know the counter value.
                    commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 32);
                }
                this.device.queue.submit([commandEncoder.finish()]);

                // --- Round-trip reduction -----------------------------------
                // This used to await onSubmittedWorkDone() and mapAsync() on EVERY
                // chunk, so each chunk paid a full CPU<->GPU stall (a couple of ms)
                // no matter how little work it contained. Triclinic at defaults is
                // ~2,150 chunks, i.e. several seconds of pure latency; a run that
                // trips the chunk-count warning is far worse.
                //
                // The per-chunk writeBuffer + submit pair stays exactly as it was --
                // batching several dispatches into ONE encoder would be wrong,
                // because queue.writeBuffer is ordered against SUBMITS, not against
                // encoded passes, so every pass in the batch would see the last
                // z_offset written. Only the readback is batched.
                //
                // Bounded to CHUNKS_PER_SYNC dispatches in flight, so nothing queues
                // without limit. The cost is that the max_solutions early-out and the
                // candidate count can lag by up to that many chunks, which is
                // harmless: the shader itself early-outs once the counter reaches
                // max_solutions, so the extra chunks do almost no work.
                if (!isSyncChunk) {
                    if (progressCallback) progressCallback((i + 1) / totalChunks, lastKnownSolutions);
                    continue;
                }

                await this.device.queue.onSubmittedWorkDone();

                // Safely map counter buffer (catching aborts if device is lost or stopped)
                try {
                    await counterReadBuffer.mapAsync(GPUMapMode.READ);
                } catch (err) {
                    console.warn("GPU mapAsync aborted (counter):", err.message);
                    stoppedEarly = true;
                    break;
                }

                const counters = new Uint32Array(counterReadBuffer.getMappedRange());
                const numSolutions = counters[0];
                // Snapshot the diagnostics: the mapped range dies at unmap().
                diag.peaksInBudget = counters[1];
                diag.volMin = counters[2];
                diag.volMax = counters[3];
                counterReadBuffer.unmap();
                lastKnownSolutions = numSolutions;

                // SMART BUFFER COPY: only copy back the cells that were actually written.
                // Previously the engine copied resultsBuffer.size bytes every chunk, which
                // means copying the entire (possibly 50k-cell) staging buffer even when only
                // a handful of new cells landed. Now we copy ceil to (countToRead * structSize).
                if (numSolutions > solutionsReadCount && solutionsReadCount < maxSolutions) {
                    const countToRead = Math.min(numSolutions, maxSolutions);

                    // DELTA COPY. The previous version copied and mapped
                    // [0, countToRead) every single chunk, so cell #1 was
                    // re-copied and re-mapped once per chunk for the rest of the
                    // run -- O(n^2) PCIe traffic, up to 1.6 MB per chunk at the
                    // default 50k-cell cap. Only the cells written since the last
                    // read are new, so copy only those.
                    //
                    // Alignment: copyBufferToBuffer needs 4-byte-aligned offsets
                    // and size; mapAsync needs an 8-byte-aligned offset and
                    // 4-byte-aligned size. solutionStructSize is 16 or 32 bytes,
                    // so every offset here is a multiple of 16 and both hold.
                    const byteOffset = solutionsReadCount * solutionStructSize;
                    const bytesToCopy = (countToRead - solutionsReadCount) * solutionStructSize;

                    const copyEncoder = this.device.createCommandEncoder();
                    copyEncoder.copyBufferToBuffer(
                        resultsBuffer, byteOffset,
                        resultsReadBuffer, byteOffset,
                        bytesToCopy);
                    this.device.queue.submit([copyEncoder.finish()]);
                    await this.device.queue.onSubmittedWorkDone();

                    try {
                        await resultsReadBuffer.mapAsync(GPUMapMode.READ, byteOffset, bytesToCopy);
                    } catch (err) {
                        console.warn("GPU mapAsync aborted (results):", err.message);
                        stoppedEarly = true;
                        break;
                    }

                    // getMappedRange is relative to the buffer, and the returned
                    // ArrayBuffer starts at byteOffset -- so index from 0 here,
                    // not from solutionsReadCount.
                    const rawResults = new Float32Array(resultsReadBuffer.getMappedRange(byteOffset, bytesToCopy));
                    const newBatch = [];
                    const newCount = countToRead - solutionsReadCount;

                    for (let k = 0; k < newCount; k++) {
                        const cellObj = cfg.parseCell(rawResults, k * cfg.structFloats);
                        // Fast pre-filter on physically reasonable axis lengths.
                        // Reads the same min_axis/max_axis the shaders now get via
                        // config.f_params2, rather than a third hard-coded copy of
                        // 2.0 / 50.0.
                        if (cellObj && cellObj.a >= minAxis && cellObj.a <= maxAxis) {
                            newBatch.push(cellObj);
                        }
                    }

                    resultsReadBuffer.unmap();
                    solutionsReadCount = countToRead;
                    if (onIntermediateResults && newBatch.length > 0) onIntermediateResults(newBatch);
                }

                if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
                // Report progress for this chunk BEFORE breaking, otherwise the
                // bar freezes short of 100% on exactly the runs that hit the cap.
                if (numSolutions >= maxSolutions) { stoppedEarly = true; break; }
            }
        } finally {
            // Guaranteed cleanup: prevents VRAM leaks even if an error or abort occurs above
            qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
            counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
            configBuffer.destroy(); qTolerancesBuffer.destroy();
        }

        // Cells are delivered incrementally through onIntermediateResults; there
        // is no accumulated list to hand back. The old `potentialCells: []` was
        // permanently empty and every caller ignored it, which is exactly the
        // kind of return value someone eventually trusts. Report counts instead.
        return {
            cellsEmitted: solutionsReadCount,
            stoppedEarly,
            diagnostics: {
                system: cfg.systemName,
                // Most peaks any candidate kept inside the FoM error budget,
                // out of how many were scored. Short of nPeaksScored means no
                // cell ever fitted the whole set.
                peaksInBudget: diag.peaksInBudget,
                nPeaksScored: Math.min(configViewU32[2], 32),
                // Volume range of cells that cleared the axis test, BEFORE the
                // volume gate. null when nothing cleared the axis test at all.
                volMin: diag.volMin === 0xFFFFFFFF ? null : diag.volMin,
                volMax: diag.volMax === 0 && diag.volMin === 0xFFFFFFFF ? null : diag.volMax,
                maxVolume: configViewF32[10],
                minVolume: configViewF32[14],
                fomThreshold: configViewF32[11],
            },
        };
    }

    // Backward-compatible entry points. brutus.html's makeGpuTask references these by name
    // (cfg.engineMethod in GPU_SYSTEM_CONFIG), so we preserve them.
    runOrthoSolver(...args) {
        return this._runSolver(WebGPUEngine.SYSTEM_CONFIGS.orthorhombic, ...args);
    }
    runMonoclinicSolver(...args) {
        return this._runSolver(WebGPUEngine.SYSTEM_CONFIGS.monoclinic, ...args);
    }
    runTriclinicSolver(...args) {
        return this._runSolver(WebGPUEngine.SYSTEM_CONFIGS.triclinic, ...args);
    }
}
