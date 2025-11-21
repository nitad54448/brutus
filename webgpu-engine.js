// webgpu-engine.js
class WebGPUEngine {
    constructor() {
        this.device = null;
        this.adapter = null; 
        this.shaderModule = null;
        this.pipeline = null;
        this.bindGroupLayout = null; // Store explicit layout
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
        return true;
    }

    // 2. Load the WGSL Shader
    async loadShader(url) {
        const response = await fetch(url);
        const shaderCode = await response.text();
        this.shaderModule = this.device.createShaderModule({ code: shaderCode });
    }

    // 3. Create Explicit Bind Group Layout (Fixes "Binding not present" error)
    createBindGroupLayout() {
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
    }

    // 4. Create the compute pipeline with Explicit Layout
    createPipeline(entryPoint = "main") {
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

        for(let i=0; i<table.length; i++) {
            if (bigTable[i] > 4294967295n) table[i] = 4294967295; 
            else table[i] = Number(bigTable[i]);
        }
        return table;
    }

    // 8. runMonoclinicSolver
    async runMonoclinicSolver(qObsArray, hklBasisArray, peakCombos, hklCombos, qTolerancesArray, progressCallback, stopSignal, baseParams, onIntermediateResults = null) {
        if (!this.pipeline) throw new Error("Pipeline not created.");

        const K_VALUE = 4;
        const n_hkls = hklBasisArray.length / 4;
        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        const maxSolutions = baseParams.max_solutions || 20000; 
        const solutionStructSize = 4 * 4; // 16 bytes
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);

        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(10 * 25 * 4); // Mono debug size

        // Config Buffer
        const configBufferSize = 48; 
        const configBuffer = this.device.createBuffer({ size: configBufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        const userPeaksSetting = baseParams.gpu_peaks_count || 7;
        const targetFomCount = Math.max(10, userPeaksSetting);
        const finalFomCount = Math.min(qObsArray.length, targetFomCount);

        configViewU32[0] = 0; configViewU32[1] = baseParams.impurity_peaks; configViewU32[2] = finalFomCount; configViewU32[3] = hklBasisArray.length / 4;
        configViewU32[4] = hklBasisArray.length / 4; configViewU32[5] = totalHklCombos; configViewU32[6] = baseParams.max_solutions; configViewU32[7] = 0;
        configViewF32[8] = baseParams.wavelength; configViewF32[9] = baseParams.tth_error; configViewF32[10] = baseParams.max_volume; configViewF32[11] = baseParams.fom_threshold;

        this.device.queue.writeBuffer(configBuffer, 0, configData);

        // Explicit Bind Group using this.bindGroupLayout
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

        // Execution Logic (TDR Safe)
        const numPeakCombos = peakCombos.length / 4;
        const MAX_THREADS_PER_DISPATCH = 500_000; 
        const maxHklPerDispatch = Math.floor(MAX_THREADS_PER_DISPATCH / Math.max(1, numPeakCombos));
        const WORKGROUP_SIZE_Y = 8; 
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / WORKGROUP_SIZE_Y);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383)); 
        const hklsPerChunk = safeWorkgroupsY * WORKGROUP_SIZE_Y;
        const workgroupsX = Math.ceil(numPeakCombos / 8); 
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

        for (let i = 0; i < totalChunks; i++) {
            if (stopSignal.stop) break;
            await new Promise(r => setTimeout(r, 0));

            configViewU32[0] = i * hklsPerChunk; // Update z_offset
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(workgroupsX, safeWorkgroupsY, 1);
            passEncoder.end();
            
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            if (numSolutions > solutionsReadCount) {
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                const newBatch = [];
                const countToRead = Math.min(numSolutions, maxSolutions);

                for (let k = solutionsReadCount; k < countToRead; k++) {
                    const offset = k * 4;
                    newBatch.push({
                        a: rawResults[offset + 0],
                        b: rawResults[offset + 1],
                        c: rawResults[offset + 2],
                        beta: rawResults[offset + 3],
                        system: 'monoclinic',
                    });
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = countToRead;
                if (onIntermediateResults && newBatch.length > 0) onIntermediateResults(newBatch);
            }

            if (numSolutions >= maxSolutions) { stoppedEarly = true; break; }
            if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
        }

        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }

    // 7. runTriclinicSolver
    async runTriclinicSolver(qObsArray, hklBasisArray, peakCombos, hklCombos, qTolerancesArray, progressCallback, stopSignal, baseParams, onIntermediateResults = null) {
        if (!this.pipeline) throw new Error("Pipeline not created.");

        const K_VALUE = 6;
        const n_hkls = hklBasisArray.length / 4; 
        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        const maxSolutions = baseParams.max_solutions || 20000;
        const solutionStructSize = 8 * 4; // 32 bytes
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(10 * 30 * 4); // Tri debug size

        const configBufferSize = 48; 
        const configBuffer = this.device.createBuffer({ size: configBufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        const userPeaksSetting = baseParams.gpu_peaks_count || 6;
        const targetFomCount = Math.max(10, userPeaksSetting);
        const finalFomCount = Math.min(qObsArray.length, targetFomCount);

        configViewU32[0] = 0; configViewU32[1] = baseParams.impurity_peaks; configViewU32[2] = finalFomCount; configViewU32[3] = hklBasisArray.length / 4;
        configViewU32[4] = hklBasisArray.length / 4; configViewU32[5] = totalHklCombos; configViewU32[6] = baseParams.max_solutions; configViewU32[7] = 0;
        configViewF32[8] = baseParams.wavelength; configViewF32[9] = baseParams.tth_error; configViewF32[10] = baseParams.max_volume; configViewF32[11] = baseParams.fom_threshold;

        this.device.queue.writeBuffer(configBuffer, 0, configData);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout, // Explicit
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

        const numPeakCombos = peakCombos.length / 6;
        const MAX_THREADS_PER_DISPATCH = 50_000; 
        const maxHklPerDispatch = Math.floor(MAX_THREADS_PER_DISPATCH / Math.max(1, numPeakCombos));
        const WORKGROUP_SIZE_Y = 4;
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / WORKGROUP_SIZE_Y);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383)); 
        const hklsPerChunk = safeWorkgroupsY * WORKGROUP_SIZE_Y;
        const workgroupsX = Math.ceil(numPeakCombos / 4); 
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

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

            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

             await counterReadBuffer.mapAsync(GPUMapMode.READ);
             const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
             counterReadBuffer.unmap();

             if (numSolutions > solutionsReadCount) {
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                const newBatch = [];
                const countToRead = Math.min(numSolutions, maxSolutions);
                
                for(let k=solutionsReadCount; k<countToRead; k++) {
                     const offset = k * 8;
                     newBatch.push({
                        a: rawResults[offset+0],
                        b: rawResults[offset+1],
                        c: rawResults[offset+2],
                        alpha: rawResults[offset+3],
                        beta: rawResults[offset+4],
                        gamma: rawResults[offset+5],
                         system: 'triclinic'
                     });
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = countToRead;
                if (onIntermediateResults && newBatch.length > 0) onIntermediateResults(newBatch);
             }
            
            if (numSolutions >= maxSolutions) { stoppedEarly = true; break; }
            if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
        }

        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }

    // 9. runOrthoSolver
    async runOrthoSolver(qObsArray, hklBasisArray, peakCombos, hklCombos, qTolerancesArray, progressCallback, stopSignal, baseParams, onIntermediateResults = null) {
        if (!this.pipeline) throw new Error("Pipeline not created.");

        const K_VALUE = 3;
        const n_hkls = hklBasisArray.length / 4;
        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        const maxSolutions = baseParams.max_solutions || 20000;
        const solutionStructSize = 4 * 4; // 16 bytes
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);

        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(10 * 20 * 4); // Ortho debug size

        const configBufferSize = 48; 
        const configBuffer = this.device.createBuffer({ size: configBufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        const userPeaksSetting = baseParams.gpu_peaks_count || 7;
        const targetFomCount = Math.max(10, userPeaksSetting);
        const finalFomCount = Math.min(qObsArray.length, targetFomCount);

        configViewU32[0] = 0; configViewU32[1] = baseParams.impurity_peaks; configViewU32[2] = finalFomCount; configViewU32[3] = hklBasisArray.length / 4;
        configViewU32[4] = hklBasisArray.length / 4; configViewU32[5] = totalHklCombos; configViewU32[6] = baseParams.max_solutions; configViewU32[7] = 0;
        configViewF32[8] = baseParams.wavelength; configViewF32[9] = baseParams.tth_error; configViewF32[10] = baseParams.max_volume; configViewF32[11] = baseParams.fom_threshold;

        this.device.queue.writeBuffer(configBuffer, 0, configData);

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout, // Explicit
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

        const numPeakCombos = peakCombos.length / 3;
        const MAX_THREADS_PER_DISPATCH = 500_000; 
        const maxHklPerDispatch = Math.floor(MAX_THREADS_PER_DISPATCH / Math.max(1, numPeakCombos));
        const WORKGROUP_SIZE_Y = 8; 
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / WORKGROUP_SIZE_Y);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383));
        const hklsPerChunk = safeWorkgroupsY * WORKGROUP_SIZE_Y;
        const workgroupsX = Math.ceil(numPeakCombos / 8); 
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

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
            
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            if (numSolutions > solutionsReadCount) {
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                const newBatch = [];
                const countToRead = Math.min(numSolutions, maxSolutions);

                for (let k = solutionsReadCount; k < countToRead; k++) {
                    const offset = k * 4;
                    newBatch.push({
                        a: rawResults[offset + 0],
                        b: rawResults[offset + 1],
                        c: rawResults[offset + 2],
                        system: 'orthorhombic',
                    });
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = countToRead;
                if (onIntermediateResults && newBatch.length > 0) onIntermediateResults(newBatch);
            }

            if (numSolutions >= maxSolutions) { stoppedEarly = true; break; }
            if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
        }
        
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }
}