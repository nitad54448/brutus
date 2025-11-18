// webgpu-engine.js
class WebGPUEngine {
    constructor() {
        this.device = null;
        this.adapter = null; 
        this.shaderModule = null;
        this.pipeline = null;
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
        // This requests the memory limits 
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

    // 3. Create the compute pipeline
    createPipeline(entryPoint = "main") {
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: entryPoint,
            },
        });
    }

    // 4. Helper to create a buffer and write data to it (do not start mono and tri at the same time)
    createBuffer(data, usage) {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new data.constructor(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    // 5. Helper to create a buffer for reading results back
    createReadBuffer(size) {
        return this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    // 6. Helper to create a buffer for GPU-only storage
    createStorageBuffer(size) {
        return this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }



    // 8. runMonoclinicSolver (Updated for Live Processing)
    async runMonoclinicSolver(
        qObsArray,        
        hklBasisArray,    
        peakCombos,       
        hklCombos,        
        qTolerancesArray, 
        progressCallback, 
        stopSignal,       
        baseParams,
        // NEW PARAMETER: Callback for live results
        onIntermediateResults = null 
    ) {
        if (!this.pipeline) throw new Error("Pipeline not created.");
        if (!this.adapter) throw new Error("Engine not initialized.");

        let stoppedEarly = false;
        let solutionsReadCount = 0; // Track how many we've processed

        // --- Create Buffers (Same as before) ---
        const maxSolutions = 20000;
        const solutionStructSize = 4 * 4; // 4 floats (a,b,c,beta)
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const hklCombosBuffer = this.createBuffer(hklCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        // ... (Debug buffers setup remains the same) ...
        const debugLogSize = 10 * 25 * 4; 
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(debugLogSize);
        const debugCounterReadBuffer = this.createReadBuffer(4);
        const debugLogReadBuffer = this.createReadBuffer(debugLogSize);

        // ... (Config buffer setup remains the same) ...
        const configBufferSize = 36; 
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);
        const n_hkls = hklBasisArray.length / 4;

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks;
        configViewU32[5] = qTolerancesArray.length;
        configViewU32[6] = Math.min(n_hkls, 100);

        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: qObsBuffer } },
                { binding: 1, resource: { buffer: hklBasisBuffer } },
                { binding: 2, resource: { buffer: peakCombosBuffer } },
                { binding: 3, resource: { buffer: hklCombosBuffer } },
                { binding: 4, resource: { buffer: counterBuffer } },
                { binding: 5, resource: { buffer: resultsBuffer } },
                { binding: 6, resource: { buffer: configBuffer } },
                { binding: 7, resource: { buffer: debugCounterBuffer } },
                { binding: 8, resource: { buffer: debugLogBuffer } },
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        // --- Chunk Logic ---
        const numPeakCombos = peakCombos.length / 4;
        const numHklCombos = hklCombos.length / 4;
        const workgroupSizeX = 8;
        const workgroupSizeY = 8;
        const workgroupsX = Math.ceil(numPeakCombos / workgroupSizeX);
        const totalHklWorkgroups = Math.ceil(numHklCombos / workgroupSizeY);
        const maxDimY = this.adapter.limits.maxComputeWorkgroupsPerDimension || 65535;
        const safeChunkY = 16383;
        const workgroupsY = Math.min(totalHklWorkgroups, safeChunkY, maxDimY);
        const totalWorkgroupsZ = Math.ceil(totalHklWorkgroups / workgroupsY);

        for (let z_chunk = 0; z_chunk < totalWorkgroupsZ; z_chunk++) {
            if (stopSignal.stop) break;

            configViewU32[0] = z_chunk;
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            const hklWorkgroupsInThisChunk = (z_chunk === totalWorkgroupsZ - 1) 
                ? (totalHklWorkgroups % workgroupsY || workgroupsY) 
                : workgroupsY;

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(workgroupsX, hklWorkgroupsInThisChunk, 1);
            passEncoder.end();
            
            // --- LIVE READ: Copy current results to ReadBuffer NOW ---
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // --- LIVE READ: Check Counter ---
            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            // --- LIVE READ: Extract NEW solutions ---
            if (numSolutions > solutionsReadCount) {
                // Map the results buffer
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                
                const newBatch = [];
                const maxToRead = Math.min(numSolutions, maxSolutions);
                const startIdx = solutionsReadCount;

                for (let i = startIdx; i < maxToRead; i++) {
                    const offset = i * 4; // 4 floats per solution (Monoclinic)
                    // NOTE: For Ortho use i*3, For Triclinic use i*6
                    const cell = {
                        a: rawResults[offset + 0],
                        b: rawResults[offset + 1],
                        c: rawResults[offset + 2],
                        beta: rawResults[offset + 3],
                        system: 'monoclinic',
                    };
                    // Optional CPU-side sanity check
                    if (cell.a > 0) newBatch.push(cell);
                }
                resultsReadBuffer.unmap();

                // Update our tracker
                solutionsReadCount = maxToRead;

                // Send to CPU immediately via callback
                if (newBatch.length > 0 && onIntermediateResults) {
                    onIntermediateResults(newBatch);
                }
            }

            // Buffer limit check
            if (numSolutions >= maxSolutions) {
                if (progressCallback) progressCallback(1.0, numSolutions);
                stoppedEarly = true;
                break;
            }

            if (progressCallback) progressCallback((z_chunk + 1) / totalWorkgroupsZ, numSolutions);
        }

        // Cleanup
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); hklCombosBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy();
        debugCounterReadBuffer.destroy(); debugLogReadBuffer.destroy(); qTolerancesBuffer.destroy();

        // Return empty array because we sent everything via callback
        return { potentialCells: [], stoppedEarly };
    }

// webgpu-engine.js (Partial Update)

    // 7. runTriclinicSolver (Updated for Live Processing)
    async runTriclinicSolver(
        qObsArray,        
        hklBasisArray,    
        peakCombos,       
        hklCombos, 
        qTolerancesArray,
        progressCallback, 
        stopSignal,       
        baseParams,
        // NEW PARAMETER
        onIntermediateResults = null
    ) {
        if (!this.pipeline) throw new Error("Pipeline not created.");
        if (!this.adapter) throw new Error("Engine not initialized.");

        let stoppedEarly = false;
        let solutionsReadCount = 0;

        // --- Create Buffers ---
        const maxSolutions = 20000;
        const solutionStructSize = 6 * 4; // 6 floats (a,b,c,al,be,ga)
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const hklCombosBuffer = this.createBuffer(hklCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        // Debug Buffers
        const debugLogSize = 10 * 30 * 4; 
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(debugLogSize);
        const debugCounterReadBuffer = this.createReadBuffer(4);
        const debugLogReadBuffer = this.createReadBuffer(debugLogSize);

        // Config Buffer
        const configBufferSize = 36; 
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);
        const n_hkls = hklBasisArray.length / 4;

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks; 
        configViewU32[5] = qTolerancesArray.length;
        configViewU32[6] = Math.min(n_hkls, 100); 

        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: qObsBuffer } },
                { binding: 1, resource: { buffer: hklBasisBuffer } },
                { binding: 2, resource: { buffer: peakCombosBuffer } },
                { binding: 3, resource: { buffer: hklCombosBuffer } },
                { binding: 4, resource: { buffer: counterBuffer } }, 
                { binding: 5, resource: { buffer: resultsBuffer } },
                { binding: 6, resource: { buffer: configBuffer } },
                { binding: 7, resource: { buffer: debugCounterBuffer } },
                { binding: 8, resource: { buffer: debugLogBuffer } },
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        // --- Chunk Logic ---
        const numPeakCombos = peakCombos.length / 6;
        const numHklCombos = hklCombos.length / 6;
        const workgroupSizeX = 4; 
        const workgroupSizeY = 4; 
        const workgroupsX = Math.ceil(numPeakCombos / workgroupSizeX);
        const totalHklWorkgroups = Math.ceil(numHklCombos / workgroupSizeY);
        const maxDimY = this.adapter.limits.maxComputeWorkgroupsPerDimension || 65535;
        const safeChunkY = 16383;
        const workgroupsY = Math.min(totalHklWorkgroups, safeChunkY, maxDimY);
        const totalWorkgroupsZ = Math.ceil(totalHklWorkgroups / workgroupsY);

        for (let z_chunk = 0; z_chunk < totalWorkgroupsZ; z_chunk++) {
            if (stopSignal.stop) break;

            configViewU32[0] = z_chunk; 
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            const hklWorkgroupsInThisChunk = (z_chunk === totalWorkgroupsZ - 1) 
                ? (totalHklWorkgroups % workgroupsY || workgroupsY) 
                : workgroupsY;

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(workgroupsX, hklWorkgroupsInThisChunk, 1); 
            passEncoder.end();
            
            // --- LIVE READ: Copy buffers immediately ---
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // --- LIVE READ: Check Counter ---
            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            // --- LIVE READ: Process new solutions ---
            if (numSolutions > solutionsReadCount) {
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                
                const newBatch = [];
                const maxToRead = Math.min(numSolutions, maxSolutions);
                const startIdx = solutionsReadCount;

                for (let i = startIdx; i < maxToRead; i++) {
                    const offset = i * 6; // 6 floats per solution
                    const cell = {
                        a: rawResults[offset + 0],
                        b: rawResults[offset + 1],
                        c: rawResults[offset + 2],
                        alpha: rawResults[offset + 3],
                        beta: rawResults[offset + 4],
                        gamma: rawResults[offset + 5],
                        system: 'triclinic',
                    };
                    if (cell.a > 0) newBatch.push(cell);
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = maxToRead;

                if (newBatch.length > 0 && onIntermediateResults) {
                    onIntermediateResults(newBatch);
                }
            }

            if (numSolutions >= maxSolutions) {
                if (progressCallback) progressCallback(1.0, numSolutions);
                stoppedEarly = true;
                break;
            }

            if (progressCallback) progressCallback((z_chunk + 1) / totalWorkgroupsZ, numSolutions);
        }

        // Cleanup
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); hklCombosBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy();
        debugCounterReadBuffer.destroy(); debugLogReadBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }

    // 9. runOrthoSolver (Updated for Live Processing)
    async runOrthoSolver(
        qObsArray,        
        hklBasisArray,    
        peakCombos,       
        hklCombos,        
        qTolerancesArray, 
        progressCallback, 
        stopSignal,       
        baseParams,
        // NEW PARAMETER
        onIntermediateResults = null
    ) {
        if (!this.pipeline) throw new Error("Pipeline not created.");
        if (!this.adapter) throw new Error("Engine not initialized.");

        let stoppedEarly = false;
        let solutionsReadCount = 0;

        // --- Create Buffers ---
        const maxSolutions = 20000;
        const solutionStructSize = 3 * 4; // 3 floats (a,b,c)
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const hklCombosBuffer = this.createBuffer(hklCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        // Debug Buffers
        const debugLogSize = 10 * 20 * 4; 
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(debugLogSize);
        const debugCounterReadBuffer = this.createReadBuffer(4);
        const debugLogReadBuffer = this.createReadBuffer(debugLogSize);

        // Config Buffer
        const configBufferSize = 36; 
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);
        const n_hkls = hklBasisArray.length / 4;

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks;
        configViewU32[5] = qTolerancesArray.length; 
        configViewU32[6] = Math.min(n_hkls, 100);   

        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: qObsBuffer } },
                { binding: 1, resource: { buffer: hklBasisBuffer } },
                { binding: 2, resource: { buffer: peakCombosBuffer } },
                { binding: 3, resource: { buffer: hklCombosBuffer } },
                { binding: 4, resource: { buffer: counterBuffer } },
                { binding: 5, resource: { buffer: resultsBuffer } },
                { binding: 6, resource: { buffer: configBuffer } },
                { binding: 7, resource: { buffer: debugCounterBuffer } },
                { binding: 8, resource: { buffer: debugLogBuffer } },
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        // --- Chunk Logic ---
        const numPeakCombos = peakCombos.length / 3;
        const numHklCombos = hklCombos.length / 3;
        const workgroupSizeX = 8;
        const workgroupSizeY = 8;
        const workgroupsX = Math.ceil(numPeakCombos / workgroupSizeX);
        const totalHklWorkgroups = Math.ceil(numHklCombos / workgroupSizeY);
        const maxDimY = this.adapter.limits.maxComputeWorkgroupsPerDimension || 65535;
        const safeChunkY = 16383;
        const workgroupsY = Math.min(totalHklWorkgroups, safeChunkY, maxDimY);
        const totalWorkgroupsZ = Math.ceil(totalHklWorkgroups / workgroupsY);

        for (let z_chunk = 0; z_chunk < totalWorkgroupsZ; z_chunk++) {
            if (stopSignal.stop) break;

            configViewU32[0] = z_chunk; 
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            const hklWorkgroupsInThisChunk = (z_chunk === totalWorkgroupsZ - 1) 
                ? (totalHklWorkgroups % workgroupsY || workgroupsY) 
                : workgroupsY;

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(workgroupsX, hklWorkgroupsInThisChunk, 1);
            passEncoder.end();
            
            // --- LIVE READ: Copy buffers immediately ---
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // --- LIVE READ: Check Counter ---
            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            // --- LIVE READ: Process new solutions ---
            if (numSolutions > solutionsReadCount) {
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                
                const newBatch = [];
                const maxToRead = Math.min(numSolutions, maxSolutions);
                const startIdx = solutionsReadCount;

                for (let i = startIdx; i < maxToRead; i++) {
                    const offset = i * 3; // 3 floats per solution
                    const cell = {
                        a: rawResults[offset + 0],
                        b: rawResults[offset + 1],
                        c: rawResults[offset + 2],
                        system: 'orthorhombic',
                    };
                    if (cell.a > 0) newBatch.push(cell);
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = maxToRead;

                if (newBatch.length > 0 && onIntermediateResults) {
                    onIntermediateResults(newBatch);
                }
            }

            if (numSolutions >= maxSolutions) {
                if (progressCallback) progressCallback(1.0, numSolutions);
                stoppedEarly = true;
                break;
            }

            if (progressCallback) progressCallback((z_chunk + 1) / totalWorkgroupsZ, numSolutions);
        }

        // Cleanup
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); hklCombosBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy();
        debugCounterReadBuffer.destroy(); debugLogReadBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }

}