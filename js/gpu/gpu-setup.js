// js/gpu/gpu-setup.js
// WebGPU availability, the shared engine, and the GPU search parameter controls.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

/**
 * Asynchronously checks WebGPU compute capabilities on page load.
 * Disables and grays out Monoclinic/Triclinic checkboxes if support is absent.
 */
async function checkWebGPUCapabilities() {
    // Find the checkboxes and their parent <label> elements
    const monoCheckbox = document.querySelector('.system-checkbox[value="monoclinic"]');
    const triCheckbox = document.querySelector('.system-checkbox[value="triclinic"]');
    const orthoCheckbox = document.querySelector('.system-checkbox[value="orthorhombic"]'); // depuis le 16 nov
    const monoLabel = monoCheckbox ? monoCheckbox.parentElement : null;
    const triLabel = triCheckbox ? triCheckbox.parentElement : null;
    const orthoLabel = orthoCheckbox ? orthoCheckbox.parentElement : null;

    try {
        if ('gpu' in navigator) {
            // Reuse the shared engine. This probe used to construct its own
            // and abandon it, so the page held a spare GPUDevice from load
            // onwards that no run ever touched.
            await getWebGPUEngine(); //  critical test
            // If this line is reached, GPU is fine.
            webGPUSupportsCompute = true;
            console.log("WebGPU compute capabilities verified.");
        } else {
            throw new Error("WebGPU not found in navigator.");
        }
    } catch (err) {
        console.warn("WebGPU initialization failed:", err.message);
        
        webGPUSupportsCompute = false; 
        
        // error message, permanent toast red warning
        showStatus("⚠ WebGPU is not initialized. GPU searches are disabled. See the Help file for details.", "error", 86400000);
        
        // Disable and gray out the monoclinic checkbox
        if (monoCheckbox) {
            monoCheckbox.disabled = true;
            monoCheckbox.checked = false;
        }
        if (monoLabel) {
            monoLabel.style.opacity = '0.5';
            monoLabel.style.cursor = 'not-allowed';
        }
        
        // Disable and gray out the triclinic checkbox
        if (triCheckbox) {
            triCheckbox.disabled = true;
            triCheckbox.checked = false;
        }
        if (triLabel) {
            triLabel.style.opacity = '0.5';
            triLabel.style.cursor = 'not-allowed';
        }
        
        // Disable and gray out the orthorhombic checkbox, 16 nov version
        if (orthoCheckbox) {
            orthoCheckbox.disabled = true;
            orthoCheckbox.checked = false;
        }
        if (orthoLabel) {
            orthoLabel.style.opacity = '0.5';
            orthoLabel.style.cursor = 'not-allowed';
        }
        
    }
}
let webGPUSupportsCompute = true;
// ------------------------------------------------------------------
// Shared WebGPU engine.
//
// Previously checkWebGPUCapabilities() built one engine (and one GPUDevice)
// at startup and startIndexing() built a fresh one on EVERY run, and
// device.destroy() was never called anywhere. Devices, their pipelines and
// their buffers therefore accumulated for the lifetime of the page; a long
// session would eventually exhaust the driver.
//
// One engine is created lazily, reused by every run, and rebuilt only if the
// device is genuinely lost. getWebGPUEngine() is the single entry point.
// ------------------------------------------------------------------
let sharedWebGPUEngine = null;
const releaseWebGPUEngine = () => {
    if (sharedWebGPUEngine) {
        try { sharedWebGPUEngine.destroy(); } catch (_) {}
        sharedWebGPUEngine = null;
    }
};
let webgpuInitPromise = null;
async function getWebGPUEngine() {
    // A lost device cannot be revived: its pipelines and buffers are gone.
    // Drop it and build a replacement.
    if (sharedWebGPUEngine && !sharedWebGPUEngine.isUsable()) {
        releaseWebGPUEngine();
    }
    if (sharedWebGPUEngine) return sharedWebGPUEngine;

    if (webgpuInitPromise) return webgpuInitPromise;
    webgpuInitPromise = (async () => {
    const engine = new WebGPUEngine();
    await engine.init();
    engine.onDeviceLost = (info) => {
        // Device loss used to be console-only, so a driver reset mid-run
        // just looked like an indexing run that quietly produced nothing.
        webGPUSupportsCompute = false;
        showStatus(
            `GPU device lost (${info && info.reason ? info.reason : 'unknown'}). ` +
            `Reload the page to re-enable GPU searches.`, 'error', 12000);
        gpuStopSignal.stop = true;
    };
    sharedWebGPUEngine = engine;
    return engine;
    })();
    try { return await webgpuInitPromise; }
    finally { webgpuInitPromise = null; }
}
// Release the device on navigation rather than relying on GC.
window.addEventListener('pagehide', releaseWebGPUEngine);
// Load data on startup, add message to console, see file event 


// --- HKL basis size limits, per system -----------------------------------
//
// The "HKL Basis Size" input used to carry a single hardcoded max="600" in
// the HTML, shared by all three GPU systems. That number is not a limit of
// anything: it is simultaneously too small and too large.
//
// Two real limits exist, and they differ per system because they depend on
// K (the number of basis reflections combined per trial):
//
//   1. SUPPLY. get_hkl_search_list(system) only generates so many
//      reflections (ortho h,k,l <= 12 -> 2196; mono 612; tri 665).
//      buildHklBasis does ordered.slice(0, n), so asking for more than
//      exists silently returns fewer -- no error, just a smaller search
//      than the number in the box claims.
//   2. ADDRESSING. The WGSL solvers unrank HKL K-combinations with a u32
//      index, so C(n, K) must stay under 2^32.
//
// The usable max is the smaller of the two, and the binding one is a
// different limit for each system:
//
//   orthorhombic  K=3   supply 2196   u32 2954   -> 2196  (supply binds)
//   monoclinic    K=4   supply  612   u32  568   ->  568  (u32 binds)
//   triclinic     K=6   supply  665   u32  123   ->  123  (u32 binds)
//
// So the old 600 cost orthorhombic ~73% of its available basis -- which is
// exactly where the deep reflections that pin down a large c axis live --
// while letting triclinic accept values nearly 5x above what the shader can
// address. Anything over the cap was then silently clamped at run time by
// the guard in makeGpuTask, so the search that ran was not the one the user
// configured.
const HKL_U32_CAPS = { orthorhombic: 2954, monoclinic: 568, triclinic: 123 };
const hklBasisMax = (system) => {
    const u32Cap = HKL_U32_CAPS[system];
    if (!u32Cap) return null;
    let supply = Infinity;
    try {
        // The crystallography scripts load on the main thread ahead of this file.
        if (typeof get_hkl_search_list === 'function') {
            supply = get_hkl_search_list(system).length;
        }
    } catch (e) {
        console.warn('hklBasisMax: could not size the basis list for', system, e);
    }
    return Math.min(u32Cap, supply);
};
// Point the input's own min/max at the active system, so the browser's
// number-input clamping and the blur validator both enforce the real limit
// instead of a made-up one.
const applyHklBasisLimits = (system, defaultValue) => {
    const el = ui.gpuHklTriplets;
    if (!el) return;
    const max = hklBasisMax(system);
    if (max === null) return;
    el.max = String(max);
    el.min = '10';
    if (defaultValue !== undefined) el.value = String(Math.min(defaultValue, max));
    const cur = parseInt(el.value, 10);
    if (!isFinite(cur) || cur > max) el.value = String(max);
    const lbl = document.querySelector('label[for="gpu-hkl-triplets"]');
    if (lbl) lbl.textContent = `HKL Basis Size (max ${max})`;
};
// Make Monoclinic, Triclinic, and Orthorhombic mutually exclusive (only one GPU task, 16 nov 2025)
const monoCheckbox = document.querySelector('.system-checkbox[value="monoclinic"]');
const triCheckbox = document.querySelector('.system-checkbox[value="triclinic"]');
const orthoCheckbox = document.querySelector('.system-checkbox[value="orthorhombic"]');
if (monoCheckbox && triCheckbox && orthoCheckbox) {
    
    monoCheckbox.addEventListener('change', () => {
        if (monoCheckbox.checked) {
            triCheckbox.checked = false;
            orthoCheckbox.checked = false;
            applyHklBasisLimits('monoclinic', 100);
            ui.gpuPeaksCount.value = 7;
            ui.gpuPeaksCount.min = 4;
        }
        toggleGpuParamsVisibility();
        updateStartIndexingButtonState();
    });

    triCheckbox.addEventListener('change', () => {
        if (triCheckbox.checked) {
            monoCheckbox.checked = false;
            orthoCheckbox.checked = false;
            applyHklBasisLimits('triclinic', 40);
            ui.gpuPeaksCount.value = 9;
            ui.gpuPeaksCount.min = 6;
        }
        toggleGpuParamsVisibility();
        updateStartIndexingButtonState();
    });
    
    orthoCheckbox.addEventListener('change', () => {
        if (orthoCheckbox.checked) {
            monoCheckbox.checked = false;
            triCheckbox.checked = false;
            applyHklBasisLimits('orthorhombic', 300); // Default 300
            ui.gpuPeaksCount.value = 7;    
            ui.gpuPeaksCount.min = 3;      // Min 3
        }
        toggleGpuParamsVisibility();
        updateStartIndexingButtonState();
    });

    // Add listeners to new inputs to update status text
    ui.gpuHklTriplets.addEventListener('input', updateGpuStatusText);
    ui.gpuPeaksCount.addEventListener('input', updateGpuStatusText);
}
/* * Calculates combinations C(n, k) = "n choose k"
 * Uses the stable multiplicative formula: (n/1) * ((n-1)/2) * ... * ((n-k+1)/k)
 */
function combinations(n, k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k === 0 || k === n) {
        return 1;
    }
    // Use the identity C(n, k) == C(n, n-k) for efficiency
    if (k > n / 2) {
        k = n - k;
    }
    
    let res = 1;
    for (let i = 1; i <= k; i++) {
        // (n - i + 1) is equivalent to (n, n-1, n-2, ...)
        res = res * (n - i + 1) / i;
    }
    
    return Math.round(res);
}
/**
 * Updates the status text with GPU calculation estimates. 16 nov
 */
function updateGpuStatusText() {
    if (!monoCheckbox || !triCheckbox || !orthoCheckbox || !statusTextElement) return;

    const n_hkl = parseInt(ui.gpuHklTriplets.value, 10);
    const n_peaks = parseInt(ui.gpuPeaksCount.value, 10);

    if (orthoCheckbox.checked) {
        const k_hkl = 3;
        const k_peaks = 3;
        const min_peaks = parseInt(ui.gpuPeaksCount.min, 10) || k_peaks;
        
        if (isNaN(n_hkl) || isNaN(n_peaks) || n_hkl < k_hkl || n_peaks < min_peaks) {
            statusTextElement.textContent = `Ortho: Requires min ${min_peaks} peaks and ${k_hkl} HKLs.`;
            return;
        }
        const peakCombos = combinations(n_peaks, k_peaks);
        const hklCombos = combinations(n_hkl, k_hkl);
        const totalTests = peakCombos * hklCombos * 6; // 6 permutations (3!)
        statusTextElement.textContent = `Ortho (GPU): ${totalTests.toLocaleString()} cells to test.`;

    } else if (monoCheckbox.checked) {
        const k_hkl = 4;
        const k_peaks = 4;
        const min_peaks = parseInt(ui.gpuPeaksCount.min, 10) || k_peaks;
        
        if (isNaN(n_hkl) || isNaN(n_peaks) || n_hkl < k_hkl || n_peaks < min_peaks) {
            statusTextElement.textContent = `Monoclinic: Requires min ${min_peaks} peaks and ${k_hkl} HKLs.`;
            return;
        }
        const peakCombos = combinations(n_peaks, k_peaks);
        const hklCombos = combinations(n_hkl, k_hkl);
        const totalTests = peakCombos * hklCombos * 24; // 24 permutations
        statusTextElement.textContent = `Monoclinic: ${totalTests.toLocaleString()} cells to test.`;

    } else if (triCheckbox.checked) {
        const k_hkl = 6;
        const k_peaks = 6;
        const min_peaks = parseInt(ui.gpuPeaksCount.min, 10) || k_peaks;

        if (isNaN(n_hkl) || isNaN(n_peaks) || n_hkl < k_hkl || n_peaks < min_peaks) {
            statusTextElement.textContent = `Triclinic: Requires min ${min_peaks} peaks and ${k_hkl} HKLs.`;
            return;
        }
        const peakCombos = combinations(n_peaks, k_peaks);
        const hklCombos = combinations(n_hkl, k_hkl);
        const totalTests = peakCombos * hklCombos * 720; // 720 permutations
        statusTextElement.textContent = `Triclinic: ${totalTests.toLocaleString()} cells to test.`;
    } else {
        statusTextElement.textContent = '';
    }
}
/**
 * Shows or hides the GPU-specific parameter inputs based on checkbox state.
 */
function toggleGpuParamsVisibility() {
    if (monoCheckbox.checked || triCheckbox.checked || orthoCheckbox.checked) {
        ui.gpuParamsContainer.classList.remove('hidden');
        updateGpuStatusText();
    } else {
        ui.gpuParamsContainer.classList.add('hidden');
        if (statusTextElement) {
            statusTextElement.textContent = ''; // Restore last indexing status or clear
        }
    }
}
ui.systemCheckboxes.forEach(checkbox => {
    checkbox.addEventListener('change', () => {
        // avant le 16 janv 2026, there was a filter here
        updateStartIndexingButtonState();
    });
});
