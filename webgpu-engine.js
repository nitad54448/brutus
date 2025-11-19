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



    // 8. runMonoclinicSolver (TDR Safe + Combinadics)
    async runMonoclinicSolver(
        qObsArray,        
        hklBasisArray,    
        peakCombos,       
        hklCombos, // Unused
        qTolerancesArray, 
        progressCallback, 
        stopSignal,       
        baseParams,
        onIntermediateResults = null 
    ) {
        if (!this.pipeline) throw new Error("Pipeline not created.");

        // --- 1. Precompute Binomial Table (K=4) ---
        const K_VALUE = 4;
        const n_hkls = hklBasisArray.length / 4;

        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);

        // C(n, 4) is at [n * stride + K]
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        // --- 2. Buffers ---
        const maxSolutions = 20000;
        const solutionStructSize = 4 * 4; 
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(10 * 25 * 4); 

        // --- 3. Config ---
        const configBufferSize = 36; 
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks;
        configViewU32[5] = qTolerancesArray.length;
        configViewU32[6] = Math.min(n_hkls, 100);
        
        // Params for Combinadics & Safety
        configViewU32[7] = n_hkls; // n_basis_total
        configViewU32[8] = totalHklCombos; // total limit

        // --- 4. Bind Group ---
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: qObsBuffer } },
                { binding: 1, resource: { buffer: hklBasisBuffer } },
                { binding: 2, resource: { buffer: peakCombosBuffer } },
                { binding: 3, resource: { buffer: binomialBuffer } }, // New Table
                { binding: 4, resource: { buffer: counterBuffer } },
                { binding: 5, resource: { buffer: resultsBuffer } },
                { binding: 6, resource: { buffer: configBuffer } },
                { binding: 7, resource: { buffer: debugCounterBuffer } },
                { binding: 8, resource: { buffer: debugLogBuffer } },
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        // --- 5. TDR-Safe Chunking ---
        const numPeakCombos = peakCombos.length / 4;
        
        // Safety Target: 10M threads max
        const MAX_THREADS_PER_DISPATCH = 10_000_000; 
        const maxHklPerDispatch = Math.floor(MAX_THREADS_PER_DISPATCH / Math.max(1, numPeakCombos));
        
        const WORKGROUP_SIZE_Y = 8; // Matches shader
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / WORKGROUP_SIZE_Y);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383)); 

        const hklsPerChunk = safeWorkgroupsY * WORKGROUP_SIZE_Y;
        const workgroupsX = Math.ceil(numPeakCombos / 8); // X dim workgroup size is 8
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

        console.log(`[GPU Mono] Safe Mode. Peaks: ${numPeakCombos}, HKLs: ${totalHklCombos}, Passes: ${totalChunks}`);

        for (let i = 0; i < totalChunks; i++) {
            if (stopSignal.stop) break;

            const startIndex = i * hklsPerChunk;
            configViewU32[0] = startIndex;
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            
            passEncoder.dispatchWorkgroups(workgroupsX, safeWorkgroupsY, 1);
            passEncoder.end();
            
            // Readback Logic
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

                if (onIntermediateResults && newBatch.length > 0) {
                    onIntermediateResults(newBatch);
                }
            }

            if (numSolutions >= maxSolutions) {
                stoppedEarly = true;
                break;
            }

            if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
        }

        // Cleanup
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }


    // Helper to generate Pascal's Triangle (Binomial Coeffs) flattened
    generateBinomialTable(n, k) {
        // Table size: (n+1) rows, (k+1) columns
        // We only need up to column k.
        const stride = k + 1;
        const table = new Uint32Array((n + 1) * stride);
        
        for (let i = 0; i <= n; i++) {
            table[i * stride + 0] = 1; // C(i, 0) = 1
            if (i <= k) table[i * stride + i] = 1; // C(i, i) = 1
            
            for (let j = 1; j < i && j <= k; j++) {
                 // Pascal's Identity: C(i, j) = C(i-1, j-1) + C(i-1, j)
                 const val = table[(i - 1) * stride + (j - 1)] + table[(i - 1) * stride + j];
                 table[i * stride + j] = val;
            }
        }
        return table;
    }

// 7. runTriclinicSolver (TDR Safe Version)
    async runTriclinicSolver(
        qObsArray,        
        hklBasisArray,    
        peakCombos,       
        hklCombos, // Unused
        qTolerancesArray,
        progressCallback, 
        stopSignal,       
        baseParams,
        onIntermediateResults = null
    ) {
        if (!this.pipeline) throw new Error("Pipeline not created.");

        // --- 1. Precompute Binomial Table ---
        const K_VALUE = 6;
        const n_hkls = hklBasisArray.length / 4; 
        
        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);

        // Total HKL combinations C(n, 6)
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        // --- 2. Create Buffers ---
        const maxSolutions = 20000;
        const solutionStructSize = 6 * 4; 
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);
        
        // Debug buffers
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(10 * 30 * 4); 

        // --- 3. Config ---
        const configBufferSize = 36; 
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks;
        configViewU32[5] = qTolerancesArray.length; 
        configViewU32[6] = Math.min(n_hkls, 100); 
        configViewU32[7] = n_hkls; // n_basis_total

        configViewU32[8] = totalHklCombos;


        // --- 4. Bind Group ---
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
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

        // --- 5. TDR-Safe Chunking Logic ---
        const numPeakCombos = peakCombos.length / 6;
        
        // Safety Target: Max 10 million threads per dispatch to avoid TDR (2-second limit)
        // Threads = (Peak Combos) * (HKL Combos per Dispatch)
        const MAX_THREADS_PER_DISPATCH = 5_000_000; 
        
        const maxHklPerDispatch = Math.floor(MAX_THREADS_PER_DISPATCH / Math.max(1, numPeakCombos));
        
        // Convert to Workgroups (Y dim, size 4)
        // Ensure at least 1 workgroup, and clamp to hardware max
        const WORKGROUP_SIZE_Y = 4;
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / WORKGROUP_SIZE_Y);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383)); // Hardware limit is usually 65535, but we keep 16k safe

        const hklsPerChunk = safeWorkgroupsY * WORKGROUP_SIZE_Y;
        const workgroupsX = Math.ceil(numPeakCombos / 4); // X dim workgroup size is 4
        
        // Calculate total loop iterations
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

        console.log(`[GPU Triclinic] Safe Mode Active.`);
        console.log(`- Peak Combos: ${numPeakCombos}`);
        console.log(`- HKL Combos: ${totalHklCombos}`);
        console.log(`- Safe HKLs/Pass: ${hklsPerChunk} (Y-Workgroups: ${safeWorkgroupsY})`);
        console.log(`- Total Passes: ${totalChunks}`);

        for (let i = 0; i < totalChunks; i++) {
            if (stopSignal.stop) break;

            // Calculate Start Index
            const startIndex = i * hklsPerChunk;
            
            configViewU32[0] = startIndex; // Update z_offset
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            
            // Dispatch Safe Dimensions
            passEncoder.dispatchWorkgroups(workgroupsX, safeWorkgroupsY, 1); 
            passEncoder.end();

            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // Read results
             await counterReadBuffer.mapAsync(GPUMapMode.READ);
             const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
             counterReadBuffer.unmap();

             if (numSolutions > solutionsReadCount) {
                await resultsReadBuffer.mapAsync(GPUMapMode.READ);
                const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
                const newBatch = [];
                const countToRead = Math.min(numSolutions, maxSolutions);
                
                for(let k=solutionsReadCount; k<countToRead; k++) {
                     const offset = k * 6;
                     newBatch.push({
                         a: rawResults[offset], b: rawResults[offset+1], c: rawResults[offset+2],
                         alpha: rawResults[offset+3], beta: rawResults[offset+4], gamma: rawResults[offset+5],
                         system: 'triclinic'
                     });
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = countToRead;
                
                if (onIntermediateResults && newBatch.length > 0) onIntermediateResults(newBatch);
             }
            
            // Buffer full check
            if (numSolutions >= maxSolutions) {
                stoppedEarly = true;
                break;
            }

            if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
        }

        // Cleanup
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }



    // 9. runOrthoSolver (TDR Safe + Combinadics)
    async runOrthoSolver(
        qObsArray,        
        hklBasisArray,    
        peakCombos,       
        hklCombos, // Unused
        qTolerancesArray, 
        progressCallback, 
        stopSignal,       
        baseParams,
        onIntermediateResults = null
    ) {
        if (!this.pipeline) throw new Error("Pipeline not created.");

        // --- 1. Precompute Binomial Table (K=3) ---
        const K_VALUE = 3;
        const n_hkls = hklBasisArray.length / 4;

        const binomialData = this.generateBinomialTable(n_hkls, K_VALUE);
        const binomialBuffer = this.createBuffer(binomialData, GPUBufferUsage.STORAGE);

        // C(n, 3)
        const totalHklCombos = binomialData[n_hkls * (K_VALUE + 1) + K_VALUE];

        // --- 2. Buffers ---
        const maxSolutions = 20000;
        const solutionStructSize = 3 * 4; 
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);

        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(10 * 20 * 4); 

        // --- 3. Config ---
        const configBufferSize = 36; 
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks;
        configViewU32[5] = qTolerancesArray.length; 
        configViewU32[6] = Math.min(n_hkls, 100);   
        
        // Combinadics & Safety
        configViewU32[7] = n_hkls; 
        configViewU32[8] = totalHklCombos; 

        // --- 4. Bind Group ---
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: qObsBuffer } },
                { binding: 1, resource: { buffer: hklBasisBuffer } },
                { binding: 2, resource: { buffer: peakCombosBuffer } },
                { binding: 3, resource: { buffer: binomialBuffer } }, // Table
                { binding: 4, resource: { buffer: counterBuffer } },
                { binding: 5, resource: { buffer: resultsBuffer } },
                { binding: 6, resource: { buffer: configBuffer } },
                { binding: 7, resource: { buffer: debugCounterBuffer } },
                { binding: 8, resource: { buffer: debugLogBuffer } },
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        // --- 5. TDR-Safe Chunking ---
        const numPeakCombos = peakCombos.length / 3;
        
        // Safety Target
        const MAX_THREADS_PER_DISPATCH = 15_000_000; // Slightly higher for ortho is fine
        const maxHklPerDispatch = Math.floor(MAX_THREADS_PER_DISPATCH / Math.max(1, numPeakCombos));
        
        const WORKGROUP_SIZE_Y = 8; 
        let safeWorkgroupsY = Math.ceil(maxHklPerDispatch / WORKGROUP_SIZE_Y);
        safeWorkgroupsY = Math.max(1, Math.min(safeWorkgroupsY, 16383));

        const hklsPerChunk = safeWorkgroupsY * WORKGROUP_SIZE_Y;
        const workgroupsX = Math.ceil(numPeakCombos / 8); 
        const totalChunks = Math.ceil(totalHklCombos / hklsPerChunk);

        let solutionsReadCount = 0;
        let stoppedEarly = false;

        console.log(`[GPU Ortho] Safe Mode. Peaks: ${numPeakCombos}, HKLs: ${totalHklCombos}, Passes: ${totalChunks}`);

        for (let i = 0; i < totalChunks; i++) {
            if (stopSignal.stop) break;

            const startIndex = i * hklsPerChunk;
            configViewU32[0] = startIndex; 
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
                    const offset = k * 3;
                    newBatch.push({
                        a: rawResults[offset + 0],
                        b: rawResults[offset + 1],
                        c: rawResults[offset + 2],
                        system: 'orthorhombic',
                    });
                }
                resultsReadBuffer.unmap();
                solutionsReadCount = countToRead;

                if (onIntermediateResults && newBatch.length > 0) {
                    onIntermediateResults(newBatch);
                }
            }

            if (numSolutions >= maxSolutions) {
                stoppedEarly = true;
                break;
            }

            if (progressCallback) progressCallback((i + 1) / totalChunks, numSolutions);
        }

        // Cleanup
        qObsBuffer.destroy(); hklBasisBuffer.destroy(); peakCombosBuffer.destroy(); binomialBuffer.destroy();
        counterBuffer.destroy(); resultsBuffer.destroy(); counterReadBuffer.destroy(); resultsReadBuffer.destroy();
        configBuffer.destroy(); debugCounterBuffer.destroy(); debugLogBuffer.destroy(); qTolerancesBuffer.destroy();

        return { potentialCells: [], stoppedEarly };
    }




}