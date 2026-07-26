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
        });

        return true;
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
        // Define the layout manually to prevent compiler from stripping unused debug bindings
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // q_obs
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // hkl_basis
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // peak_combos
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // binomial_table
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // solution_counter (RW)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // results_list (RW)
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // config
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // debug_counter (RW)
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // debug_log (RW)
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }  // q_tolerances
            ]
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

        // Packing invariants. hkl_basis is stored as [h,k,l,pad] (4 f32 per
        // reflection) and peak_combos as `peakComboStride` u32 per combo. If
        // either array is ever built with a length that isn't a whole number of
        // records, n_hkls / numPeakCombos silently go fractional and mis-size
        // the combinadic space and dispatch — no throw, just wrong results.
        // Fail loud and early instead.
        if (hklBasisArray.length % 4 !== 0) {
            throw new Error(`hklBasisArray length ${hklBasisArray.length} is not a multiple of 4 (expected [h,k,l,pad] records).`);
        }
        if (peakCombos.length % cfg.peakComboStride !== 0) {
            throw new Error(`peakCombos length ${peakCombos.length} is not a multiple of stride ${cfg.peakComboStride} for system ${cfg.systemName}.`);
        }

        const n_hkls = hklBasisArray.length / 4;
        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        const maxSolutions = baseParams.max_solutions || 20000;
        const solutionStructSize = cfg.structFloats * 4; // bytes per cell

        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);

        // Debug buffers: still created and bound because the shader declares the bindings,
        // but nothing is ever written to debug_log on the hot path (see shaders). These are
        // effectively no-ops and we never read them back.
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(32 * 32 * 4); // Generous for any system

        const configBufferSize = 48;
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
        configViewU32[3] = hklBasisArray.length / 4;
        configViewU32[4] = hklBasisArray.length / 4;
        configViewU32[5] = totalHklCombos;
        configViewU32[6] = maxSolutions;                         // FIXED: use resolved maxSolutions, not raw baseParams.max_solutions
        configViewU32[7] = 0;
        configViewF32[8] = baseParams.wavelength;
        configViewF32[9] = baseParams.tth_error;
        configViewF32[10] = baseParams.max_volume;
        configViewF32[11] = baseParams.fom_threshold;

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
                { binding: 7, resource: { buffer: debugCounterBuffer } },
                { binding: 8, resource: { buffer: debugLogBuffer } },
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        const numPeakCombos = peakCombos.length / cfg.peakComboStride;
        const maxHklPerDispatch = Math.floor(cfg.maxThreadsPerDispatch / Math.max(1, numPeakCombos));
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / cfg.workgroupY);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383));
        const hklsPerChunk = safeWorkgroupsY * cfg.workgroupY;
        const workgroupsX = Math.ceil(numPeakCombos / cfg.workgroupX);
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

        // Byte alignment for copyBufferToBuffer: WebGPU requires size to be a multiple of 4.
        // One cell is already a multiple of 4 bytes (4 or 8 f32s), so any count*structSize is safe.

try {
            for (let i = 0; i < totalChunks; i++) {
                if (stopSignal.stop) break;
                await new Promise(r => setTimeout(r, 0));

                configViewU32[0] = i * hklsPerChunk;
                this.device.queue.writeBuffer(configBuffer, 0, configData);

                const commandEncoder = this.device.createCommandEncoder();
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(this.pipeline);
                passEncoder.setBindGroup(0, bindGroup);
                passEncoder.dispatchWorkgroups(workgroupsX, safeWorkgroupsY, 1);
                passEncoder.end();

                // Copy only the counter this chunk. We will decide how many result bytes to
                // copy once we know the counter value.
                commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
                this.device.queue.submit([commandEncoder.finish()]);
                await this.device.queue.onSubmittedWorkDone();

                // Safely map counter buffer (catching aborts if device is lost or stopped)
                try {
                    await counterReadBuffer.mapAsync(GPUMapMode.READ);
                } catch (err) {
                    console.warn("GPU mapAsync aborted (counter):", err.message);
                    stoppedEarly = true;
                    break;
                }

                const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
                counterReadBuffer.unmap();

                // SMART BUFFER COPY: only copy back the cells that were actually written.
                // Previously the engine copied resultsBuffer.size bytes every chunk, which
                // means copying the entire (possibly 50k-cell) staging buffer even when only
                // a handful of new cells landed. Now we copy ceil to (countToRead * structSize).
                if (numSolutions > solutionsReadCount) {
                    const countToRead = Math.min(numSolutions, maxSolutions);
                    const bytesToCopy = countToRead * solutionStructSize;

                    const copyEncoder = this.device.createCommandEncoder();
                    copyEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, bytesToCopy);
                    this.device.queue.submit([copyEncoder.finish()]);
                    await this.device.queue.onSubmittedWorkDone();

                    try {
                        await resultsReadBuffer.mapAsync(GPUMapMode.READ, 0, bytesToCopy);
                    } catch (err) {
                        console.warn("GPU mapAsync aborted (results):", err.message);
                        stoppedEarly = true;
                        break;
                    }

                    const rawResults = new Float32Array(resultsReadBuffer.getMappedRange(0, bytesToCopy));
                    const newBatch = [];

                    for (let k = solutionsReadCount; k < countToRead; k++) {
                        const cellObj = cfg.parseCell(rawResults, k * cfg.structFloats);
                        // Fast pre-filter: only keep cells with physically reasonable unit cell dimensions (2.0 Å to 50.0 Å)
                        if (cellObj && cellObj.a >= 2.0 && cellObj.a <= 50.0) {
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
            configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();
        }

        return { potentialCells: [], stoppedEarly };
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