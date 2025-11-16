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


    // 7. This is the main function for triclinique
    async runTriclinicSolver(
        qObsArray,        // Float32Array
        hklBasisArray,    // Float32Array
        peakCombos,       // Uint32Array (N x 6)
        hklCombos, 
        qTolerancesArray,
        progressCallback, // function
        stopSignal,       // { stop: false }
        baseParams        // { wavelength, tth_error, max_volume, impurity_peaks }
    ) {
        if (!this.pipeline) {
            throw new Error("Pipeline not created. Call createPipeline() first.");
        }
        if (!this.adapter) { 
            throw new Error("Engine not initialized. Call init() first.");
        }

        let stoppedEarly=false;

        // --- Create Buffers ---
        const maxSolutions = 50000;
        const solutionStructSize = 6 * 4; // RawSolution: a,b,c,al,be,ga
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const hklCombosBuffer = this.createBuffer(hklCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
    
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);
        // --- Debug Buffers, on les utilise si besoin
        const debugLogSize = 10 * 30 * 4; // 10 cells * 6 peaks * 5 floats
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(debugLogSize);
        const debugCounterReadBuffer = this.createReadBuffer(4);
        const debugLogReadBuffer = this.createReadBuffer(debugLogSize);

        // --- Config Buffer (Uniforms) ---
        
        const configBufferSize = 36; // 9 * 4 bytes
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        const n_peaks = qObsArray.length;
        const n_hkls = hklBasisArray.length / 4;

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks; // Not used by FoM, but good to pass
        configViewU32[5] = qTolerancesArray.length;
        configViewU32[6] = Math.min(n_hkls, 100); // N_HKL_FOR_FOM
        // 7, 8 are padding

        // --- Create Bind Group (with all 9 bindings) ---
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

        // --- Chunked Dispatch Logic ---
        const numPeakCombos = peakCombos.length / 6;
        const numHklCombos = hklCombos.length / 6;
        const workgroupSizeX = 4; // From WGSL
        const workgroupSizeY = 4; // From WGSL

        const workgroupsX = Math.ceil(numPeakCombos / workgroupSizeX);
        const totalHklWorkgroups = Math.ceil(numHklCombos / workgroupSizeY);

        //some limits, see harcoded wgsl limit
        const maxDimY = this.adapter.limits.maxComputeWorkgroupsPerDimension || 4096;
        const safeChunkY = 256;
        const workgroupsY = Math.min(totalHklWorkgroups, safeChunkY, maxDimY);

        const totalWorkgroupsZ = Math.ceil(totalHklWorkgroups / workgroupsY);

        for (let z_chunk = 0; z_chunk < totalWorkgroupsZ; z_chunk++) {
            
            if (stopSignal.stop) {
                console.log("WebGPU engine stopping (triclinic)...");
                break;
            }

            // 1. Update uniform buffer
            configViewU32[0] = z_chunk; // Set z_offset
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            // 2. Calculate workgroups for this chunk
            const hklWorkgroupsInThisChunk = (z_chunk === totalWorkgroupsZ - 1) 
                ? (totalHklWorkgroups % workgroupsY || workgroupsY) 
                : workgroupsY;

            // 3. Create and submit commands
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(workgroupsX, hklWorkgroupsInThisChunk, 1); 
            passEncoder.end();
            
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // 4. Check Counter
            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            if (numSolutions >= maxSolutions) {
                console.log(`GPU Buffer Full (Triclinic: ${numSolutions}). Stopping early.`);
                if (progressCallback) progressCallback(1.0, numSolutions);
                stoppedEarly = true;
                break;
            }

            // 5. Report Progress
            if (progressCallback) {
                progressCallback((z_chunk + 1) / totalWorkgroupsZ, numSolutions);
            }
        }

        // After all chunks are done (or stopped), copy final results
        const finalEncoder = this.device.createCommandEncoder();
        finalEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
        finalEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
        finalEncoder.copyBufferToBuffer(debugCounterBuffer, 0, debugCounterReadBuffer, 0, 4);
        finalEncoder.copyBufferToBuffer(debugLogBuffer, 0, debugLogReadBuffer, 0, debugLogBuffer.size);

        this.device.queue.submit([finalEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();


        // --- Read Results ---
        await counterReadBuffer.mapAsync(GPUMapMode.READ);
        const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
        counterReadBuffer.unmap();

        const potentialCells = [];
        if (numSolutions > 0) {
            await resultsReadBuffer.mapAsync(GPUMapMode.READ);
            const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
            
            const countToRead = Math.min(numSolutions, maxSolutions);
            for (let i = 0; i < countToRead; i++) {
                const offset = i * 6;
                potentialCells.push({
                    a: rawResults[offset + 0],
                    b: rawResults[offset + 1],
                    c: rawResults[offset + 2],
                    alpha: rawResults[offset + 3],
                    beta: rawResults[offset + 4],
                    gamma: rawResults[offset + 5],
                    system: 'triclinic',
                });
            }
            resultsReadBuffer.unmap();
        }

        // --- Read Debug Log ---
        await debugCounterReadBuffer.mapAsync(GPUMapMode.READ);
        const numDebugLogs = new Uint32Array(debugCounterReadBuffer.getMappedRange())[0];
        debugCounterReadBuffer.unmap();
         debugLogReadBuffer.unmap(); //moved from the IF below, keep the IF block for debug later, if needed
        
         /*
        if (numDebugLogs > 0) {
            console.log(`--- GPU DEBUG LOG (Triclinic, First ${Math.min(numDebugLogs, 10)} passing cells) ---`);
            await debugLogReadBuffer.mapAsync(GPUMapMode.READ);
            const debugData = new Float32Array(debugLogReadBuffer.getMappedRange());
            
            for (let i = 0; i < Math.min(numDebugLogs, 10); i++) {
                console.log(`--- CELL ${i+1} ---`);
                let logStr = "HKL\t\td_obs\t\td_calc\n";
                logStr += "--------------------------------------\n";
                const offset = i * 30; // 30 floats per cell
                for (let p = 0; p < 6; p++) { 
                    const p_offset = offset + p * 5;
                    const h = debugData[p_offset + 0];
                    const k = debugData[p_offset + 1];
                    const l = debugData[p_offset + 2];
                    const d_obs = debugData[p_offset + 3];
                    const d_calc = debugData[p_offset + 4];
                    logStr += `(${h},${k},${l})\t\t${d_obs.toFixed(5)}\t${d_calc.toFixed(5)}\n`;
                }
                console.log(logStr);
            }
            debugLogReadBuffer.unmap();
            console.log(`--- END GPU DEBUG LOG (Triclinic) ---`);
        }
*/

        // --- Cleanup ---
        qObsBuffer.destroy();
        hklBasisBuffer.destroy();
        peakCombosBuffer.destroy();
        hklCombosBuffer.destroy();
        counterBuffer.destroy();
        resultsBuffer.destroy();
        counterReadBuffer.destroy();
        resultsReadBuffer.destroy();
        configBuffer.destroy();
        debugCounterBuffer.destroy();
        debugLogBuffer.destroy();
        debugCounterReadBuffer.destroy();
        debugLogReadBuffer.destroy();
        qTolerancesBuffer.destroy();
        return { potentialCells, stoppedEarly };
    }


    // 8. runMonoclinicSolver (MODIFIED for 4-Peak + FoM, 15 nov 2025)
    async runMonoclinicSolver(
        qObsArray,        // Float32Array
        hklBasisArray,    // Float32Array
        peakCombos,       // Uint32Array (N x 4)
        hklCombos,        // Uint32Array (N x 4)
        qTolerancesArray, // tol
        progressCallback, // function
        stopSignal,       // { stop: false }
        baseParams        // { wavelength, tth_error, max_volume, impurity_peaks }
    ) {
        if (!this.pipeline) {
            throw new Error("Pipeline not created. Call createPipeline() first.");
        }
        if (!this.adapter) { 
            throw new Error("Engine not initialized. Call init() first.");
        }


        let stoppedEarly=false;

        // --- Create Buffers ---
        const maxSolutions = 50000;
        const solutionStructSize = 4 * 4; // RawMonoSolution: a,b,c,beta
        
        const qObsBuffer = this.createBuffer(qObsArray, GPUBufferUsage.STORAGE);
        const hklBasisBuffer = this.createBuffer(hklBasisArray, GPUBufferUsage.STORAGE);
        const peakCombosBuffer = this.createBuffer(peakCombos, GPUBufferUsage.STORAGE);
        const hklCombosBuffer = this.createBuffer(hklCombos, GPUBufferUsage.STORAGE);

        const counterBuffer = this.createStorageBuffer(4);
        const resultsBuffer = this.createStorageBuffer(maxSolutions * solutionStructSize);
        const counterReadBuffer = this.createReadBuffer(4);
        const resultsReadBuffer = this.createReadBuffer(maxSolutions * solutionStructSize);
        const qTolerancesBuffer = this.createBuffer(qTolerancesArray, GPUBufferUsage.STORAGE);


        const debugLogSize = 10 * 25 * 4; 
        const debugCounterBuffer = this.createStorageBuffer(4);
        const debugLogBuffer = this.createStorageBuffer(debugLogSize);
        const debugCounterReadBuffer = this.createReadBuffer(4);
        const debugLogReadBuffer = this.createReadBuffer(debugLogSize);

        // --- Config Buffer (Uniforms) ---
        // Layout (9 values, 16-byte aligned)
        const configBufferSize = 36; // 9 * 4 bytes
        const configBuffer = this.device.createBuffer({
            size: configBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        const configData = new ArrayBuffer(configBufferSize);
        const configViewU32 = new Uint32Array(configData);
        const configViewF32 = new Float32Array(configData);

        const n_peaks = qObsArray.length;
        const n_hkls = hklBasisArray.length / 4;

        configViewF32[1] = baseParams.wavelength;
        configViewF32[2] = baseParams.tth_error;
        configViewF32[3] = baseParams.max_volume;
        configViewU32[4] = baseParams.impurity_peaks;
        configViewU32[5] = qTolerancesArray.length; // N_PEAKS_FOR_FOM (now read from array length)
        configViewU32[6] = Math.min(n_hkls, 100);   // N_HKL_FOR_FOM (e.g., 100)
        // 7, 8 are padding

        // --- Create Bind Group
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
                // --- 15 nov
                { binding: 9, resource: { buffer: qTolerancesBuffer } },
            ],
        });

        // --- Chunked Dispatch Logic ---
        const numPeakCombos = peakCombos.length / 4;
        const numHklCombos = hklCombos.length / 4;
        const workgroupSizeX = 8;
        const workgroupSizeY = 8;

        const workgroupsX = Math.ceil(numPeakCombos / workgroupSizeX);
        const totalHklWorkgroups = Math.ceil(numHklCombos / workgroupSizeY);

        const maxDimY = this.adapter.limits.maxComputeWorkgroupsPerDimension || 4096;
        const safeChunkY = 256; //voir aussi wgsl
        const workgroupsY = Math.min(totalHklWorkgroups, safeChunkY, maxDimY);
        
        const totalWorkgroupsZ = Math.ceil(totalHklWorkgroups / workgroupsY);

        for (let z_chunk = 0; z_chunk < totalWorkgroupsZ; z_chunk++) {

            if (stopSignal.stop) {
                console.log("WebGPU engine stopping...");
                break;
            }

            // 1. Update Uniforms (Z Offset is at index 0)
            configViewU32[0] = z_chunk; // Set z_offset
            this.device.queue.writeBuffer(configBuffer, 0, configData);

            // 2. Calculate workgroups
            const hklWorkgroupsInThisChunk = (z_chunk === totalWorkgroupsZ - 1) 
                ? (totalHklWorkgroups % workgroupsY || workgroupsY) 
                : workgroupsY;

            // 3. Submit Command
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(workgroupsX, hklWorkgroupsInThisChunk, 1);
            passEncoder.end();
            
            commandEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
            
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // 4. Check Counter
            await counterReadBuffer.mapAsync(GPUMapMode.READ);
            const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
            counterReadBuffer.unmap();

            if (numSolutions >= maxSolutions) {
                console.log(`GPU Buffer Full (${numSolutions}). Stopping early.`);
                if (progressCallback) progressCallback(1.0, numSolutions);
                stoppedEarly = true;
                break;
            }

            // 5. Report Progress
            if (progressCallback) {
                progressCallback((z_chunk + 1) / totalWorkgroupsZ, numSolutions);
            }
        }

        // --- Retrieve Results ---
        const finalEncoder = this.device.createCommandEncoder();
        finalEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadBuffer, 0, 4);
        finalEncoder.copyBufferToBuffer(resultsBuffer, 0, resultsReadBuffer, 0, resultsBuffer.size);
       
        finalEncoder.copyBufferToBuffer(debugCounterBuffer, 0, debugCounterReadBuffer, 0, 4);
        finalEncoder.copyBufferToBuffer(debugLogBuffer, 0, debugLogReadBuffer, 0, debugLogBuffer.size);
       
        this.device.queue.submit([finalEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await counterReadBuffer.mapAsync(GPUMapMode.READ);
        const numSolutions = new Uint32Array(counterReadBuffer.getMappedRange())[0];
        counterReadBuffer.unmap();

        const potentialCells = [];
        if (numSolutions > 0) {
            await resultsReadBuffer.mapAsync(GPUMapMode.READ);
            const rawResults = new Float32Array(resultsReadBuffer.getMappedRange());
            
            const maxVol = baseParams.max_volume;
            const countToRead = Math.min(numSolutions, maxSolutions);

            for (let i = 0; i < countToRead; i++) {
                const offset = i * 4;
                const cell = {
                    a: rawResults[offset + 0],
                    b: rawResults[offset + 1],
                    c: rawResults[offset + 2],
                    beta: rawResults[offset + 3],
                    system: 'monoclinic',
                };
                
                // CPU-side Volume Safety Check
                const vol = cell.a * cell.b * cell.c * Math.sin(cell.beta * Math.PI / 180.0);
                if (vol > 0 && vol <= maxVol) {
                     potentialCells.push(cell);
                }
            }
            resultsReadBuffer.unmap();
        }

        // --- Debug Log Readback ---
        await debugCounterReadBuffer.mapAsync(GPUMapMode.READ);
        const numDebugLogs = new Uint32Array(debugCounterReadBuffer.getMappedRange())[0];
        debugCounterReadBuffer.unmap();
        debugLogReadBuffer.unmap(); //moved from the disabled "if" below
        /*
        if (numDebugLogs > 0) {
            console.log(`--- GPU DEBUG LOG (First ${Math.min(numDebugLogs, 10)} passing cells) ---`);
            await debugLogReadBuffer.mapAsync(GPUMapMode.READ);
            const debugData = new Float32Array(debugLogReadBuffer.getMappedRange());
            
            
            for (let i = 0; i < Math.min(numDebugLogs, 10); i++) {  //debug print, à eenlever plus tard
                console.log(`--- CELL ${i+1} ---`);
                let logStr = "HKL\t\td_obs\t\td_calc\n";
                logStr += "--------------------------------------\n";
                const offset = i * 25; // 25 floats per cell
                for (let p = 0; p < 5; p++) { // Still prints up to 5, but 5th line will be 0s
                    const p_offset = offset + p * 5;
                    const h = debugData[p_offset + 0];
                    const k = debugData[p_offset + 1];
                    const l = debugData[p_offset + 2];
                    const d_obs = debugData[p_offset + 3];
                    const d_calc = debugData[p_offset + 4];
                    if (h === 0 && k === 0 && l === 0 && d_obs === 0) continue; 
                    logStr += `(${h},${k},${l})\t\t${d_obs.toFixed(5)}\t${d_calc.toFixed(5)}\n`;
                }
                console.log(logStr);
            }
            debugLogReadBuffer.unmap();
            console.log(`--- END GPU DEBUG LOG ---`);
        }

        */

        // --- Cleanup ---
        qObsBuffer.destroy();
        hklBasisBuffer.destroy();
        peakCombosBuffer.destroy();
        hklCombosBuffer.destroy();
        counterBuffer.destroy();
        resultsBuffer.destroy();
        counterReadBuffer.destroy();
        resultsReadBuffer.destroy();
        configBuffer.destroy();
        debugCounterBuffer.destroy();
        debugLogBuffer.destroy();
        debugCounterReadBuffer.destroy();
        debugLogReadBuffer.destroy();
        qTolerancesBuffer.destroy();  //15 nov, passe dq dans wgsl, ne pas la calculer 1 milliard de fois

        return { potentialCells, stoppedEarly };
    }




}