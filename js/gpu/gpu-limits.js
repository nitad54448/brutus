// js/gpu/gpu-limits.js
// GPU search-size limits and the Start button state.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

const UINT32_MAX = 4294967295n; // 2^32 - 1 (BigInt)
// Helper: BigInt combinations to prevent JS precision loss ? 
const bigCombinations = (n, k) => {
    if (k < 0 || k > n) return 0n;
    if (k === 0 || k === n) return 1n;
    if (k > n / 2) k = n - k;
    let res = 1n;
    for (let i = 1n; i <= BigInt(k); i++) {
        res = res * (BigInt(n) - i + 1n) / i;
    }
    return res;
};
// Dispatch-count thresholds for checkGpuLimits(). A dispatch costs roughly a
// GPU submit plus a fence wait, so ~1e5 of them is already a slow run and
// ~2e6 is not going to finish in any session a person will sit through.
const CHUNK_COUNT_WARN_LIMIT = 100000;
const CHUNK_COUNT_HARD_LIMIT = 2000000;
const checkGpuLimits = () => {
    // If WebGPU isn't even available, don't block (CPU fallback logic handles it)
    if (!webGPUSupportsCompute) return true;

    let n_hkl = 0;
    let k_val = 0;
    let activeSystem = "";

    // 1. Identify Active System & Parameters
    if (triCheckbox && triCheckbox.checked) {
        n_hkl = parseInt(ui.gpuHklTriplets.value, 10) || 40;
        k_val = 6;
        activeSystem = "Triclinic";
    } else if (monoCheckbox && monoCheckbox.checked) {
        n_hkl = parseInt(ui.gpuHklTriplets.value, 10) || 80;
        k_val = 4;
        activeSystem = "Monoclinic";
    } else if (orthoCheckbox && orthoCheckbox.checked) {
        n_hkl = parseInt(ui.gpuHklTriplets.value, 10) || 300;
        k_val = 3;
        activeSystem = "Orthorhombic";
    } else {
        return true; // No GPU system selected
    }

    // 2. Calculate Shader Loop Size (The dangerous number)
    const totalHklCombos = bigCombinations(n_hkl, k_val);

    // 2b. Dispatch-count estimate.
    //
    // hklsPerChunk in the engine is derived from a FIXED thread budget
    // divided by the number of peak combinations, so raising "Peaks to
    // Combine" shrinks the chunk and multiplies the number of dispatches.
    // Triclinic at the UI-allowed maximum of 20 peaks gives C(20,6)=38760
    // combos, which collapses the chunk to 4 hkls and demands ~1e9
    // dispatches for C(123,6): a run that never finishes, with no error and
    // a progress bar that merely looks slow. Warn while it is still cheap.
    const n_peaks_ui = parseInt(ui.gpuPeaksCount.value, 10) || 0;
    const numPeakCombos = combinations(Math.max(n_peaks_ui, k_val), k_val);
    // Must mirror WebGPUEngine.SYSTEM_CONFIGS.
    const threadBudget = (k_val === 6) ? 50000 : 500000;
    const wgY = (k_val === 6) ? 4 : 8;
    const maxHklPerDispatch = Math.floor(threadBudget / Math.max(1, numPeakCombos));
    const wgCount = Math.max(1, Math.min(Math.ceil(maxHklPerDispatch / wgY), 16383));
    const hklsPerChunk = wgCount * wgY;
    const estChunks = Number(totalHklCombos) / hklsPerChunk;

    if (estChunks > CHUNK_COUNT_HARD_LIMIT) {
        ui.startIndexingButton.disabled = true;
        ui.startIndexingButton.textContent = "Start (**** Too Slow)";
        ui.startIndexingButton.style.backgroundColor = "var(--error-red)";
        ui.startIndexingButton.style.borderColor = "var(--error-red)";
        if (statusTextElement) {
            statusTextElement.textContent =
                `Error: ${activeSystem} needs ~${Math.round(estChunks).toLocaleString('en-US')} GPU dispatches ` +
                `(only ${hklsPerChunk} HKL per dispatch at ${numPeakCombos} peak combinations). ` +
                `Reduce "Peaks to Combine" or "HKL Basis Size".`;
        }
        return false;
    }
    if (estChunks > CHUNK_COUNT_WARN_LIMIT && statusTextElement) {
        statusTextElement.textContent =
            `Warning: ${activeSystem} will issue ~${Math.round(estChunks).toLocaleString('en-US')} GPU dispatches. ` +
            `Lowering "Peaks to Combine" would speed this up substantially.`;
    }

    // 3. Check against Hardware Limit (u32)
    if (totalHklCombos > UINT32_MAX) {
        //  LIMIT EXCEEDED: BLOCK UI ?
        ui.startIndexingButton.disabled = true;
        ui.startIndexingButton.textContent = "Start (**** Too Large)";
        ui.startIndexingButton.style.backgroundColor = "var(--error-red)";
        ui.startIndexingButton.style.borderColor = "var(--error-red)";
        
        if (document.getElementById('status-text')) {
            const fmt = totalHklCombos.toLocaleString('en-US');
            document.getElementById('status-text').textContent = 
                `Error: ${activeSystem} HKL combos (${fmt}) exceeds GPU limit (4.29 Billion). Reduce HKL Basis.`;
        }
        return false; // Invalid
    } 
    
    // 4. Valid State
    return true;
};
// update
const updateStartIndexingButtonState = () => {
    // Step 1: Check Peaks
    const tthMin = parseFloat(ui.tthMinSlider.value) || -Infinity;
    const tthMax = parseFloat(ui.tthMaxSlider.value) || Infinity;
    const validPeaks = pickedPeaks.filter(p => p.tth >= tthMin && p.tth <= tthMax && !p.ka2Suspect);
    
    let minRequired = 4;
    if (ui.gpuParamsContainer && !ui.gpuParamsContainer.classList.contains('hidden')) {
        minRequired = parseInt(ui.gpuPeaksCount.value, 10) || 7;
    }
    const needed = minRequired - validPeaks.length;
    
    if (needed > 0) { 
        // Not enough peaks
        ui.startIndexingButton.disabled = true; 
        ui.startIndexingButton.textContent = `Need ${needed} more peak${needed > 1 ? 's' : ''}`; 
        ui.startIndexingButton.style.backgroundColor = ""; 
        ui.startIndexingButton.style.borderColor = "";
        if (document.getElementById('status-text')) document.getElementById('status-text').textContent = "";
    } else { 
        // Step 2: Peaks are good, now Check GPU Safety
        const gpuSafe = checkGpuLimits();
        
        if (gpuSafe) {
            // Safe to run
            ui.startIndexingButton.disabled = false; 
            ui.startIndexingButton.textContent = 'Start Indexing'; 
            ui.startIndexingButton.style.backgroundColor = ""; 
            ui.startIndexingButton.style.borderColor = "";
            
            // Clear any previously-set pre-flight message. checkGpuLimits
            // can now emit a dispatch-count warning as well as the u32 one,
            // and a warning left on screen after the parameter was fixed
            // reads as a live error.
            const statusEl = document.getElementById('status-text');
            if (statusEl && (statusEl.textContent.includes("exceeds GPU limit") ||
                             statusEl.textContent.includes("GPU dispatches"))) {
                statusEl.textContent = "";
            }
        }
        // If gpuSafe is false, checkGpuLimits already set the button to Red/****
    }
};
// Attach listeners
ui.gpuHklTriplets.addEventListener('input', updateStartIndexingButtonState);
ui.gpuPeaksCount.addEventListener('input', updateStartIndexingButtonState);
ui.systemCheckboxes.forEach(cb => cb.addEventListener('change', updateStartIndexingButtonState));
// Also run on init
// Size the HKL input to whichever GPU system is already checked, so the
// limit is correct before the user touches anything.
for (const s of ['orthorhombic', 'monoclinic', 'triclinic']) {
    const cb = document.querySelector(`.system-checkbox[value="${s}"]`);
    if (cb && cb.checked) { applyHklBasisLimits(s); break; }
}
updateStartIndexingButtonState();
