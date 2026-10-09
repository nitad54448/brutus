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
// Dispatch geometry for one planned search. Must mirror
// WebGPUEngine.SYSTEM_CONFIGS / _runSolver: every system now dispatches
// 8 x 8 workgroups; triclinic keeps its smaller thread budget (TDR guard).
const estimateGpuChunks = (plan) => {
    const totalHklCombos = bigCombinations(plan.nHkl, plan.K);
    const numPeakCombos = combinations(Math.max(plan.nPeaks, plan.K), plan.K);
    const threadBudget = (plan.K === 6) ? 50000 : 500000;
    const wgY = 8;
    const maxHklPerDispatch = Math.floor(threadBudget / Math.max(1, numPeakCombos));
    const wgCount = Math.max(1, Math.min(Math.ceil(maxHklPerDispatch / wgY), 16383));
    const hklsPerChunk = wgCount * wgY;
    return { totalHklCombos, numPeakCombos, hklsPerChunk,
             estChunks: Number(totalHklCombos) / hklsPerChunk };
};
// Pre-flight check over EVERY checked system (they are no longer mutually
// exclusive). The first system that would never finish blocks the Start
// button and is named; slow-but-feasible ones only warn.
const checkGpuLimits = () => {
    // If WebGPU isn't even available, don't block (CPU fallback logic handles it)
    if (!webGPUSupportsCompute) return true;

    const warnings = [];
    // Plan with the peaks actually available: a search can never combine
    // more peaks than there are, and planning with the nominal count blocked
    // or warned about searches that would in fact be small.
    const available = countIndexablePeaks() || Infinity;
    for (const system of checkedSystems()) {
        const plan = planGpuSearch(system, available);
        if (!plan) continue;
        const label = plan.cfg.label;
        const { totalHklCombos, numPeakCombos, hklsPerChunk, estChunks } = estimateGpuChunks(plan);

        // The X dimension is one workgroup per 8 peak combinations and the
        // engine refuses anything over the device limit (65,535 by default).
        // Block here, where it can still be fixed, rather than at run time.
        if (Math.ceil(numPeakCombos / 8) > 65535) {
            blockStartButton("Start (**** Too Many Peaks)",
                `Error: ${label} combines ${plan.nPeaks} peaks (${numPeakCombos.toLocaleString('en-US')} ` +
                `combinations), over the GPU dispatch limit. Lower "Depth".`);
            return false;
        }
        // Hklsperchunk shrinks as the peak-combination count grows (fixed
        // thread budget), so raising Depth multiplies the number of dispatches.
        if (estChunks > CHUNK_COUNT_HARD_LIMIT) {
            blockStartButton("Start (**** Too Slow)",
                `Error: ${label} needs ~${Math.round(estChunks).toLocaleString('en-US')} GPU dispatches ` +
                `(only ${hklsPerChunk} HKL per dispatch at ${numPeakCombos} peak combinations). ` +
                `Reduce "Depth" or "HKL Basis".`);
            return false;
        }
        // Cannot trigger with the current lists (hklBasisMax already applies
        // the u32 cap), kept as a guard should a list ever grow.
        if (totalHklCombos > UINT32_MAX) {
            blockStartButton("Start (**** Too Large)",
                `Error: ${label} HKL combos (${totalHklCombos.toLocaleString('en-US')}) exceeds GPU limit ` +
                `(4.29 Billion). Reduce HKL Basis.`);
            return false;
        }
        if (estChunks > CHUNK_COUNT_WARN_LIMIT) {
            warnings.push(`${label} ~${Math.round(estChunks).toLocaleString('en-US')} dispatches`);
        }
    }
    if (warnings.length && statusTextElement) {
        statusTextElement.textContent =
            `Warning: slow GPU search (${warnings.join(', ')}). Lower "Depth" to speed up.`;
    }
    return true;
};
const blockStartButton = (label, message) => {
    ui.startIndexingButton.disabled = true;
    ui.startIndexingButton.textContent = label;
    ui.startIndexingButton.style.backgroundColor = "var(--error-red)";
    ui.startIndexingButton.style.borderColor = "var(--error-red)";
    if (statusTextElement) statusTextElement.textContent = message;
};
// update
const updateStartIndexingButtonState = () => {
    // A run in progress owns the button (setUIState); checkbox or parameter
    // edits made meanwhile must not re-enable Start under it.
    if (isIndexing) return;
    // The per-system plan depends on how many peaks there are; refresh it
    // whenever the peak list (or anything else feeding this) changes.
    updateGpuStatusText();

    // Step 1: Check Peaks
    const tthMin = parseFloat(ui.tthMinSlider.value) || -Infinity;
    const tthMax = parseFloat(ui.tthMaxSlider.value) || Infinity;
    const validPeaks = pickedPeaks.filter(p => p.tth >= tthMin && p.tth <= tthMax && !p.ka2Suspect);

    // The smallest search that can still post a cell decides the floor: a
    // system short of peaks is skipped on its own at run time, it does not
    // block the others. With no system checked the old floor of 4 stands.
    const systems = checkedSystems();
    const minRequired = systems.length
        ? Math.min(...systems.map(s => MIN_PEAKS_FOR_SYSTEM[s] || 4))
        : 4;
    const needed = minRequired - validPeaks.length;

    if (needed > 0) {
        // Not enough peaks
        ui.startIndexingButton.disabled = true;
        ui.startIndexingButton.textContent = `Need ${needed} more peak${needed > 1 ? 's' : ''}`;
        ui.startIndexingButton.style.backgroundColor = "";
        ui.startIndexingButton.style.borderColor = "";
        if (statusTextElement) statusTextElement.textContent = "";
    } else {
        // Step 2: Peaks are good, now Check GPU Safety
        // Clear any previously-set pre-flight message first: a warning left
        // on screen after the parameter was fixed reads as a live error.
        if (statusTextElement && (statusTextElement.textContent.includes("exceeds GPU limit") ||
                                  statusTextElement.textContent.includes("GPU dispatch") ||
                                  statusTextElement.textContent.includes("slow GPU search"))) {
            statusTextElement.textContent = "";
        }
        const gpuSafe = checkGpuLimits();

        if (gpuSafe) {
            // Safe to run
            ui.startIndexingButton.disabled = false;
            ui.startIndexingButton.textContent = 'Start Indexing';
            ui.startIndexingButton.style.backgroundColor = "";
            ui.startIndexingButton.style.borderColor = "";
        }
        // If gpuSafe is false, checkGpuLimits already set the button to Red/****
    }
};
// Attach listeners
ui.gpuHklPercent.addEventListener('input', updateStartIndexingButtonState);
ui.gpuDepth.addEventListener('input', updateStartIndexingButtonState);
// (System checkboxes: gpu-setup.js registers one listener that calls both
// toggleGpuParamsVisibility and updateStartIndexingButtonState.)
// Also run on init
toggleGpuParamsVisibility();
updateStartIndexingButtonState();
