
document.addEventListener('DOMContentLoaded', () => {



    // Updated Presets with precise Bearden (1967) values
    const WAVELENGTH_PRESETS = {
        'Cu': { ka1: 1.54056, ka2: 1.54439, ka_avg: 1.54184, ratio: 0.497 },
        'Co': { ka1: 1.78896, ka2: 1.79285, ka_avg: 1.79026, ratio: 0.497 },
        'Fe': { ka1: 1.93604, ka2: 1.93998, ka_avg: 1.93735, ratio: 0.497 },
        'Mo': { ka1: 0.70930, ka2: 0.71359, ka_avg: 0.71073, ratio: 0.497 },
        // Cr kept just in case, though not in dropdown
        'Cr': { ka1: 2.28970, ka2: 2.29361, ka_avg: 2.29100, ratio: 0.497 },
        // Ag: short-wavelength anode, increasingly common for total-scattering /
        // high-Q work. ka_avg follows the same (2*ka1 + ka2)/3 convention as the
        // entries above. VERIFY the 5th decimal against your Bearden (1967) table
        // before relying on it for precise work.
        'Ag': { ka1: 0.55941, ka2: 0.56381, ka_avg: 0.56088, ratio: 0.497 },
        'custom': { ka1: null, ka2: null, ka_avg: null, ratio: 0.5 }
    };

    /**
     * Returns the currently-active preset object IF the radiation has a
     * meaningful Ka2 component (i.e. preset is _avg AND we have ka1+ka2).
     * Returns null for Ka1-only setups, custom monochromatic, or stripped data.
     * Used by the Ka2-suspect tagger.
     */
    const getActiveKa2Preset = () => {
        const sel = ui.wavelengthPreset?.value;
        if (!sel || sel === 'custom') return null;
        const [element, type] = sel.split('_');
        if (type === 'ka1') return null; // pure Ka1, no doublet to expect
        const data = WAVELENGTH_PRESETS[element];
        if (!data || !data.ka1 || !data.ka2) return null;
        return data;
    };

    /**
     * Returns the expected 2θ position (deg) of the Ka2 ghost of a parent
     * Ka1 line at parentTthDeg, given the active doublet preset.
     * Returns NaN if the geometry is invalid (arg >= 1 means high-angle clip).
     */
    const expectedKa2TthDeg = (parentTthDeg, preset) => {
        const theta1 = parentTthDeg * Math.PI / 360; // theta in rad
        const arg = (preset.ka2 / preset.ka1) * Math.sin(theta1);
        if (!(arg < 1)) return NaN;
        return 2 * Math.asin(arg) * 180 / Math.PI;
    };

    /**
     * Angle-adaptive tolerance for matching a peak to the expected Ka2 position.
     * The Ka1/Ka2 separation grows with 2θ; we let the tolerance grow with it
     * but never shrink below the user's configured 2θ error.
     * Returned tolerance is in degrees.
     */
    const ka2MatchTolerance = (parentTthDeg, preset) => {
        const userTol = parseFloat(ui.tthError?.value) || 0.04;
        // Expected doublet split at this angle
        const ka2Pred = expectedKa2TthDeg(parentTthDeg, preset);
        if (!isFinite(ka2Pred)) return userTol;
        const split = Math.abs(ka2Pred - parentTthDeg);
        // Tolerance = max(userTol, 25% of the expected split)
        // 25% is empirical: tight enough to avoid flagging unrelated peaks,
        // loose enough to catch real Ka2 lines slightly displaced by overlap.
        return Math.max(userTol, 0.25 * split);
    };

    /**
     * Intensity sanity check for a candidate Kα₂ companion.
     *
     * A Kα₂ line carries a fixed fraction of its Kα₁ parent — preset.ratio,
     * ≈0.497 for all the characteristic doublets. We accept a candidate only if
     * the measured height ratio is consistent with that, allowing generous
     * slack for peak-fitting error, background and partial overlap.
     *
     * The upper bound is the one that matters: a peak as tall as, or taller
     * than, its supposed parent CANNOT be that parent's ghost. Flagging it
     * would discard a real reflection from the space-group evidence. The lower
     * bound is looser, since a Kα₂ sitting on a falling background or partly
     * merged into a neighbour can measure low.
     *
     * If heights are unavailable we fall back to the old position-only
     * behaviour rather than silently rejecting everything.
     */
    const KA2_RATIO_MAX_FACTOR = 1.7;  // vs expected ratio -> ~0.85 of parent
    const KA2_RATIO_MIN_FACTOR = 0.35; // vs expected ratio -> ~0.17 of parent
    const ka2RatioIsPlausible = (parent, child, preset) => {
        const hp = parent?.intensity ?? parent?.height;
        const hc = child?.intensity ?? child?.height;
        if (typeof hp !== 'number' || typeof hc !== 'number' ||
            !isFinite(hp) || !isFinite(hc) || hp <= 0) {
            return true; // no usable heights -> position-only, as before
        }
        const expected = (preset && preset.ratio) ? preset.ratio : 0.497;
        const observed = hc / hp;
        return observed <= expected * KA2_RATIO_MAX_FACTOR &&
               observed >= expected * KA2_RATIO_MIN_FACTOR;
    };

    /**
     * Walk pickedPeaks, set p.ka2Suspect = true on any peak whose 2θ matches
     * the predicted Ka2 ghost of an earlier (lower-2θ) peak. Also stores
     * p.ka2ParentIdx for traceability.
     *
     * Skipped entirely when:
     *   - preset has no Ka2 (custom or _ka1)
     *   - data has already been Ka2-stripped (the doublet is gone)
     *   - user has explicitly confirmed a peak as real (p.userConfirmedReal)
     *
     * Re-runs cleanly: clears flags before re-tagging.
     */
    const flagKa2SuspectPeaks = () => {
        // Reset ALL flags first, unconditionally. This must clear
        // userConfirmedReal peaks too: a peak flagged under (say) Cu Ka and
        // then right-clicked as "real" would otherwise keep ka2Suspect = true
        // forever, since this loop skipped it and the tagging loop below also
        // skips it. That silently defeats the manual override — the peak stays
        // excluded from indexing and keeps counting as a soft violation.
        // Clearing here is safe because the tagging loop re-applies the flag
        // only to peaks that still match, and never to userConfirmedReal ones.
        //
        // It also guarantees that switching to a radiation WITHOUT a Ka2
        // component (Custom, or any *_ka1 preset) leaves no stale flags behind:
        // the reset runs before the `!preset` early return below, so every peak
        // is cleared and no Ka2-based demotion can survive the switch.
        pickedPeaks.forEach(p => {
            p.ka2Suspect = false;
            p.ka2ParentIdx = null;
            p.hasKa2Child = false; // becomes true if a Ka2-suspect child is found below
        });

        const preset = getActiveKa2Preset();
        // No doublet to expect: Custom radiation, any Ka1-only preset, or a
        // preset lacking ka1/ka2. Nothing is flagged, so no peak is demoted to
        // "soft" on Ka2 grounds and none is withheld from the indexing search.
        if (!preset) return;
        if (ui.stripKa2Checkbox?.checked) return; // already stripped

        // pickedPeaks is kept sorted by 2θ; rely on that.
        for (let i = 0; i < pickedPeaks.length; i++) {
            const parent = pickedPeaks[i];
            // A peak that is itself a Ka2-suspect cannot be a parent
            if (parent.ka2Suspect) continue;
            const tth2Pred = expectedKa2TthDeg(parent.tth, preset);
            if (!isFinite(tth2Pred)) continue;
            const tol = ka2MatchTolerance(parent.tth, preset);

            for (let j = i + 1; j < pickedPeaks.length; j++) {
                const q = pickedPeaks[j];
                // Stop early: 2θ values past tth2Pred + tol can't match
                if (q.tth - tth2Pred > tol) break;
                if (q.userConfirmedReal) continue; // user said this is real
                if (Math.abs(q.tth - tth2Pred) <= tol) {
                    // Position alone is not enough. The Ka2 line is a fixed
                    // fraction of its Ka1 parent (preset.ratio, ~0.497 for the
                    // Cu/Co/Fe/Mo/Cr doublets), so a candidate sitting at the
                    // right angle but carrying the wrong intensity is a genuine
                    // reflection that happens to fall near the predicted ghost
                    // position — not a ghost. Without this check any real peak
                    // at the doublet spacing is silently demoted to soft
                    // evidence and stops constraining the space group.
                    if (!ka2RatioIsPlausible(parent, q, preset)) continue;
                    q.ka2Suspect = true;
                    q.ka2ParentIdx = i;
                    parent.hasKa2Child = true; // parent is provably Kα1 → use λ_Kα1 for it
                    break; // only one Ka2 per parent
                }
            }
        }
    };

    /**
     * Updates the yellow warning banner above the peak table, reflecting how
     * many peaks are currently flagged and reminding the user what it means.
     */
    const updateKa2Banner = () => {
        const banner = document.getElementById('ka2-warning-banner');
        if (!banner) return;
        const n = pickedPeaks.filter(p => p.ka2Suspect).length;
        const preset = getActiveKa2Preset();
        if (!preset || ui.stripKa2Checkbox?.checked || n === 0) {
            banner.style.display = 'none';
            return;
        }
        const nParents = pickedPeaks.filter(p => p.hasKa2Child).length;
        banner.style.display = 'block';
        banner.innerHTML =
            `<b>⚠ ${n} peak${n > 1 ? 's' : ''} flagged as possible Kα₂ companion${n > 1 ? 's' : ''}.</b> ` +
            `Yellow rows are <b>excluded from indexing and FOM</b>, ` +
            `and count as <i>soft</i> violations in space-group analysis. ` +
            `Their ${nParents} Kα₁ parent${nParents > 1 ? 's' : ''} ` +
            `(marked with <b>*</b>) use λ_Kα₁ for the d-spacing. ` +
            `Right-click a yellow row to confirm it as real instead.`;
    };


    //turn off stripping if no average, v 123
    // Updated: strip is now ENABLED BY DEFAULT for K-alpha average presets,
    // since untreated Ka2 lines are the main source of false positives in
    // space-group analysis. User can still uncheck if their data has
    // already been deconvoluted.
    /**
     * Apply a wavelength preset to the UI.
     *
     * @param {object} opts
     * @param {boolean} opts.onLoad - True when called during initial file load.
     *   File-load path: strip is enabled by default for Kα-doublet presets
     *   (the user hasn't expressed a preference yet, and stripping is the
     *   safer default for indexing).
     *   Manual path (default): strip is turned OFF whenever the user changes
     *   the preset, reverting the chart to the original (un-stripped) data.
     *   The user can re-tick the strip checkbox to apply Rachinger correction
     *   with the newly selected radiation's constants — fresh from the
     *   original data, never re-stripping already-stripped data.
     */
    const handleWavelengthPresetChange = (opts = {}) => {
        // Note: opts.onLoad is still accepted for call-site compatibility, but
        // preset behaviour is now identical on load and on manual change (strip
        // OFF, λ = Kα-avg for _avg presets), so it is no longer read here.
        const selection = ui.wavelengthPreset.value;

        if (selection === 'custom') {
            ui.stripKa2Checkbox.checked = false;
            ui.stripKa2Checkbox.disabled = true;
        } else {
            const [element, type] = selection.split('_'); 
            const data = WAVELENGTH_PRESETS[element];

            if (data) {
                if (type === 'ka1') {
                    ui.wavelength.value = data.ka1.toFixed(5);
                    // Disable and uncheck for Ka1 (no Ka2 present)
                    ui.stripKa2Checkbox.checked = false;
                    ui.stripKa2Checkbox.disabled = true;
                } else {
                    // Kα-average preset.
                    //   Strip is OFF by default (both on file load and on manual
                    //   preset change): the user opts in to stripping explicitly.
                    //   λ defaults to Kα-avg, the correct value for raw doublet
                    //   data. When the user later ticks Strip, the handler swaps
                    //   λ to Kα1 (stripping leaves peaks at the Kα1 position).
                    ui.stripKa2Checkbox.disabled = false;
                    ui.stripKa2Checkbox.checked = false;
                    ui.wavelength.value = data.ka_avg.toFixed(5);
                }
            }
        }
        
        debouncedUpdateAndRedraw();
        recalculatePeakValues();
        updatePeakTable();
    };


    /**
     * Handles manual edits to the wavelength input field.
     * If the typed value differs from the currently-selected preset's expected
     * value, switch the preset dropdown to "custom" and disable Kα2 stripping.
     * This runs on every keystroke but is cheap.
     */
    const handleWavelengthValueChange = () => {
        const selection = ui.wavelengthPreset.value;
        if (selection === 'custom') return; // already custom; nothing to do

        const typed = parseFloat(ui.wavelength.value);
        if (!isFinite(typed)) return; // mid-edit empty field or invalid; wait

        const [element, type] = selection.split('_');
        const data = WAVELENGTH_PRESETS[element];
        if (!data) return;
        // When strip is ON, peaks are at Kα1 positions, so the wavelength
        // input is expected to hold ka1 (not ka_avg). When strip is OFF on
        // an _avg preset, the input holds ka_avg.
        const stripOn = !!ui.stripKa2Checkbox?.checked;
        let expected;
        if (type === 'ka1') expected = data.ka1;
        else if (stripOn)   expected = data.ka1;     // _avg + strip → ka1
        else                expected = data.ka_avg;  // _avg + no strip → avg

        // Compare rounded to 5 dp, matching the precision at which we set preset values.
        if (Math.abs(typed - expected) > 5e-6) {
            ui.wavelengthPreset.value = 'custom';
            // Switching to custom disables Kα2 stripping because the preset no longer
            // describes a Kα1/Kα_avg pair.
            ui.stripKa2Checkbox.checked = false;
            ui.stripKa2Checkbox.disabled = true;
            debouncedUpdateAndRedraw();
            recalculatePeakValues();
            updatePeakTable();
        }
    };


    const ui = {
        fileInput: document.getElementById('file-input'),
        // Was document.querySelector('.file-input-label') on a <label>. The
        // label carried its own background, which overrode .btn-secondary and
        // made `Load File` a different colour from `Save as`. It is now a
        // <button> with exactly the same classes, so they cannot drift.
        fileInputLabel: document.getElementById('load-file-button'),
        fileName: document.getElementById('file-name-box'),
        fileChipName: document.getElementById('file-chip-name'),
        fileChipSize: document.getElementById('file-chip-size'),
        fileChipClear: document.getElementById('file-chip-clear'),
        unloadOverlay: document.getElementById('unload-overlay'),
        unloadFile: document.getElementById('unload-file'),
        unloadLosses: document.getElementById('unload-losses'),
        unloadCancel: document.getElementById('unload-cancel'),
        unloadConfirm: document.getElementById('unload-confirm'),
        saveAsButton: document.getElementById('save-as-button'),
        saveMenuOverlay: document.getElementById('save-menu-overlay'),
        saveFormatSelect: document.getElementById('save-format-select'),
        saveMenuCancel: document.getElementById('save-menu-cancel'),
        saveMenuConfirm: document.getElementById('save-menu-confirm'),
        saveMenuMsg: document.getElementById('save-menu-msg'),
        peakControls: document.getElementById('peak-controls'),
        peakThresholdSlider: document.getElementById('peak-threshold-slider'),
        peakThresholdValue: document.getElementById('peak-threshold-value'),
        peakTableContainer: document.getElementById('peak-table-container'),
        peakListBody: document.getElementById('peak-list-body'),
        indexingControls: document.getElementById('indexing-controls'),
        
        // Wavelength Controls 
        wavelengthPreset: document.getElementById('wavelength-preset'),
        stripKa2Checkbox: document.getElementById('strip-ka2-checkbox'),
        wavelength: document.getElementById('wavelength'), // This is the K-alpha1 input
        
        tthError: document.getElementById('tth-error'),
        maxVolume: document.getElementById('max-volume'),
        impurityPeaksInput: document.getElementById('impurity-peaks'),
        refineZeroCheckbox: document.getElementById('refine-zero-checkbox'),
        systemCheckboxes: document.querySelectorAll('.system-checkbox'),
        startIndexingButton: document.getElementById('start-indexing-button'),
        reportButton: document.getElementById('report-button'),

                tabButtonsContainer: document.querySelector('.tab-buttons'),
        tabButtons: document.querySelectorAll('.tab-btn'),
        tabPanels: document.querySelectorAll('.tab-content-panel'),
        // New GPU Param UI Elements 
        gpuParamsContainer: document.getElementById('gpu-params-container'),
        gpuHklTriplets: document.getElementById('gpu-hkl-triplets'),
        gpuPeaksCount: document.getElementById('gpu-peaks-count'),
        gpuFomThreshold: document.getElementById('gpu-fom-threshold'),
        gpuBufferSize: document.getElementById('gpu-buffer-size'),
        progressBar: document.getElementById('progress-bar'),
        progressBarContainer: document.getElementById('progress-bar-container'),
        solutionsTableBody: document.getElementById('solutions-table-body'),
        solutionsTableHeaders: document.querySelectorAll('#solutions-table-container th'),
        solutionsLed: document.getElementById('solutions-led'),
        chartCanvas: document.getElementById('xrd-chart'),
        xAxisMode: document.getElementById('x-axis-mode'),
        yAxisMode: document.getElementById('y-axis-mode'),
        snapMode: document.getElementById('snap-mode'),
        snapReadout: document.getElementById('snap-readout'),
        placeholder: document.getElementById('placeholder'),
        resultsContainer: document.getElementById('results-container'),
        tthMinSlider: document.getElementById('tth-min-slider'),
        tthMaxSlider: document.getElementById('tth-max-slider'),
        tthMinValue: document.getElementById('tth-min-value'),
        tthMaxValue: document.getElementById('tth-max-value'),
        ballRadiusSlider: document.getElementById('ball-radius-slider'),
        ballRadiusValue: document.getElementById('ball-radius-value'),
        smoothingWidthSlider: document.getElementById('smoothing-width-slider'),
        smoothingWidthValue: document.getElementById('smoothing-width-value'),
        statusBar: document.getElementById('status-box')
    };
    
    const statusTextElement = document.getElementById('status-text');


/**
     * Enforces min/max constraints on a number input element when the user clicks away.
     * @param {HTMLInputElement} inputEl - The input element to validate.
     * @param {number} defaultVal - A default value to use if parsing fails.
     */
    function validateNumberInput(inputEl, defaultVal = 0) {
        const min = parseFloat(inputEl.min) || 0;
        const max = parseFloat(inputEl.max) || Infinity;
        let value = parseFloat(inputEl.value);

        if (isNaN(value)) {
            value = defaultVal;
        }
        
        if (value < min) {
            value = min;
        } else if (value > max) {
            value = max;
        }
        
        // Update the element's value to the constrained value
        inputEl.value = value;
    }

    // Single source of truth for the "Impurity Peaks" value used at RUN TIME.
    // The input has min/max attributes, but those are only enforced on 'blur'
    // (via validateNumberInput). Every place that actually starts a computation
    // used to read the raw field with parseInt, so a value typed but not blurred
    // — or any value outside [min,max] — reached the GPU shaders and the CPU
    // refinement UNCLAMPED, and the two stages then disagreed about how many
    // unindexed peaks a cell is allowed (the shader clamps to n-1 internally,
    // the CPU FoM does not). Reading through this helper makes all consumers
    // see the same clamped integer. Bounds come from the input's own min/max
    // attributes so the runtime clamp can never drift from the markup.
    function getImpurityPeaks() {
        const el = ui.impurityPeaksInput;
        const lo = Number.isFinite(parseInt(el.min, 10)) ? parseInt(el.min, 10) : 0;
        const hiParsed = parseInt(el.max, 10);
        const hi = Number.isFinite(hiParsed) ? hiParsed : Infinity;
        const raw = parseInt(el.value, 10) || 0;
        return Math.max(lo, Math.min(raw, hi));
    }

    //  Add validation to all number inputs
    //c'est fait plus tard ?
    //ui.maxVolume.addEventListener('blur', () => validateNumberInput(ui.maxVolume, 4000));
    ui.impurityPeaksInput.addEventListener('blur', () => validateNumberInput(ui.impurityPeaksInput, 1));
    ui.tthError.addEventListener('blur', () => validateNumberInput(ui.tthError, 0.04));
    
    // Add validation to the new GPU inputs
    ui.gpuHklTriplets.addEventListener('blur', () => {
        validateNumberInput(ui.gpuHklTriplets, 100);
        updateGpuStatusText(); // Also update status text on blur
    });
    ui.gpuPeaksCount.addEventListener('blur', () => {
        // Special min-value logic for gpuPeaksCount
        const currentMin = parseFloat(ui.gpuPeaksCount.min) || 4;
        validateNumberInput(ui.gpuPeaksCount, currentMin);
        updateGpuStatusText(); // Also update status text on blur
    });

    ui.gpuFomThreshold.addEventListener('blur', () => validateNumberInput(ui.gpuFomThreshold, 3.0));
    ui.gpuBufferSize.addEventListener('blur', () => validateNumberInput(ui.gpuBufferSize, 50));


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

    // space group, fichier crée avec Gemmi, je mets le script aussi sur git
    // nouvelle version, règles de CCTBX, depuis le 17 janvier 2026
    //v2 depuis le 14 juillet 2026
    let spaceGroupData = null;
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

    async function getWebGPUEngine() {
        // A lost device cannot be revived: its pipelines and buffers are gone.
        // Drop it and build a replacement.
        if (sharedWebGPUEngine && !sharedWebGPUEngine.isUsable()) {
            releaseWebGPUEngine();
        }
        if (sharedWebGPUEngine) return sharedWebGPUEngine;

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
    }

    // Release the device on navigation rather than relying on GC.
    window.addEventListener('pagehide', releaseWebGPUEngine);
  
   // Function to load space group JSON data, error if not found, ..
    async function loadSpaceGroupData() {
        try {
            const response = await fetch('cctbx_space_groups_all_settings_v4.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            spaceGroupData = await response.json();
            console.log("Space group data (all settings) loaded successfully.");
        } catch (error) {
            console.error("Could not load space group data:", error);
            showStatus("Warning: Could not load cctbx_space_groups_all_settings_v4.json. Space group analysis will be disabled.", "error", 8000);
        }
    }
    
    // Load data on startup, add message to console, see file event 
    loadSpaceGroupData();
    checkWebGPUCapabilities(); //check webGPU
    


// Make Monoclinic, Triclinic, and Orthorhombic mutually exclusive (only one GPU task, 16 nov 2025)
    const monoCheckbox = document.querySelector('.system-checkbox[value="monoclinic"]');
    const triCheckbox = document.querySelector('.system-checkbox[value="triclinic"]');
    const orthoCheckbox = document.querySelector('.system-checkbox[value="orthorhombic"]');

    if (monoCheckbox && triCheckbox && orthoCheckbox) {
        
        monoCheckbox.addEventListener('change', () => {
            if (monoCheckbox.checked) {
                triCheckbox.checked = false;
                orthoCheckbox.checked = false;
                ui.gpuHklTriplets.value = 100;
                ui.gpuPeaksCount.value = 7;
                ui.gpuPeaksCount.min = 4;
            }
            toggleGpuParamsVisibility();
        });

        triCheckbox.addEventListener('change', () => {
            if (triCheckbox.checked) {
                monoCheckbox.checked = false;
                orthoCheckbox.checked = false;
                ui.gpuHklTriplets.value = 40;
                ui.gpuPeaksCount.value = 9;
                ui.gpuPeaksCount.min = 6;
            }
            toggleGpuParamsVisibility();
        });
        
        orthoCheckbox.addEventListener('change', () => {
            if (orthoCheckbox.checked) {
                monoCheckbox.checked = false;
                triCheckbox.checked = false;
                ui.gpuHklTriplets.value = 300; // Default 300
                ui.gpuPeaksCount.value = 7;    
                ui.gpuPeaksCount.min = 3;      // Min 3
            }
            toggleGpuParamsVisibility();
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




    ui.solutionsTableHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const column = header.dataset.sort;
            if (!column) return;

            if (sortState.column === column) {
                sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
            } else {
                sortState.column = column;
                sortState.direction = (column === 'm20' || column === 'volume') ? 'desc' : 'asc';
            }

            sortSolutions();
  //          const selectedSystems = Array.from(ui.systemCheckboxes)
  //                                       .filter(cb => cb.checked)
  //                                       .map(cb => cb.value);
            // A COPY, not an alias. handleNewSolution does
            // `solutions = solutions.slice(...)` when pruning, which rebinds
            // `solutions` to a new array and leaves an aliased
            // displayedSolutions pointing at the stale one -- after which the
            // rendered table and the array the context menu indexes into are
            // two different lists.
            displayedSolutions = [...solutions];
            updateSolutionsTable();
        });
    });


    ui.systemCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            // avant le 16 janv 2026, there was a filter here
            updateStartIndexingButtonState();
        });
    });


    // echelle log, 0.1% c'est suffisant ?
    const minPeak = 0.1;
    const maxPeak = 20;
    const minLog = Math.log(minPeak);
    const maxLog = Math.log(maxPeak);
    const scale = (maxLog - minLog) / 100;

    function valueToLogSlider(value) {
        if (!isFinite(value) || value <= 0) return 0;
        return (Math.log(value) - minLog) / scale;
    }

    function logSliderToValue(position) {
        if (!isFinite(position)) return minPeak;
        return Math.exp(minLog + scale * position);
    }

    // debounce
    const debounce = (func, delay) => {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), delay);
        };
    };
    const debouncedFindPeaks = debounce(findPeaks, 250);
    
    const debouncedUpdateAndRedraw = debounce(() => {
        updateWorkingData();
        if (xrdChart) {
            // Stripping and deconvolution both change the intensities, so the
            // ordinate is rescaled as well as redrawn.
            setExperimentalTrace(true); // pas d'animation, sinon c'est trop lent
        }
        findPeaks();
    }, 250);


    // log slider, 
    const initialPeakThreshold = 2.0;
    ui.peakThresholdSlider.value = valueToLogSlider(initialPeakThreshold);
    ui.peakThresholdValue.textContent = initialPeakThreshold.toFixed(1);

    // tabs
    ui.tabButtonsContainer.addEventListener('click', (e) => {
        const clickedTab = e.target.closest('.tab-btn');
        if (!clickedTab || clickedTab.disabled) return;
        const tabTarget = clickedTab.dataset.tab;
        ui.tabButtons.forEach(btn => btn.classList.remove('active'));
        ui.tabPanels.forEach(panel => panel.classList.remove('active'));
        clickedTab.classList.add('active');
        document.getElementById(`${tabTarget}-tab-content`).classList.add('active');
    });

    let statusTimeout;


    const showStatus = (message, type = 'info', duration = 4000) => {
        if (!ui.statusBar) {
            console.warn(`Status bar element (#status-box) not found. Message: "${message}"`);
            return;
        }
        if (statusTimeout) clearTimeout(statusTimeout);
        ui.statusBar.textContent = message;
        ui.statusBar.className = `show ${type}`;
        statusTimeout = setTimeout(() => {
            if (ui.statusBar) {
                ui.statusBar.classList.remove('show');
            }
        }, duration);
    };

const getOrthogonalityScore = (cell) => {
        const alpha = cell.alpha ?? 90;
        const beta = cell.beta ?? 90;
        const gamma = cell.gamma ?? 90;
        
        return Math.abs(alpha - 90) + Math.abs(beta - 90) + Math.abs(gamma - 90);
    };

    // Safe max for arrays of any length. Math.max(...arr) throws on large arrays
    // (stack limit ~65k in V8) and is slow even when it works.
    const maxOfArray = (arr) => {
        const n = arr.length;
        if (n === 0) return -Infinity;
        let m = arr[0];
        for (let i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
        return m;
    };

    // Lightweight perf helper. Usage:
    //   const end = perfStart('findPeaks');
    //   ... do work ...
    //   end();
    // Logs one line per call to the console. Prefix 'perf' makes them easy to filter.
    const perfStart = (label) => {
        const t0 = performance.now();
        return (extra = '') => {
            const dt = performance.now() - t0;
            console.log(`[perf] ${label}: ${dt.toFixed(1)} ms${extra ? ' ' + extra : ''}`);
            return dt;
        };
    };


    // data, si Ka stripped ou pas, on copie les données
    let fullExperimentalData = { tth: [], intensity: [] }; // The original, unmodified data
    let loadedFileName = ''; // basename of the currently loaded file, for Save-as naming
    let workingExperimentalData = { tth: [], intensity: [] }; // The data to be plotted and analyzed (raw or stripped)

    let pickedPeaks = [];
    // Index of the peak whose row in the side table is currently being
    // edited (focused or clicked). When non-null, updateAllMarkers draws a
    // tall translucent vertical line across the whole plot at that peak's
    // 2θ so the user can immediately see which peak in the diffractogram
    // they are editing. Cleared on blur or click outside.
    let selectedPeakIndex = null;
    let taskProgress = [];
    let taskTotals = [];
    let lastDurationStr = "";
    let solutions = [];
    let displayedSolutions = [];
    let selectedSolution = null;
    let currentHklList = [];
    let foundSolutionMap = new Map();
    let xrdChart;
    let isIndexing = false;
    
    let gpuStopSignal = { stop: false };

    let lastIndexingStats = ''; // Stores the final trial count and speed
    let cumulativeTrials = 0;
    let gpuTotalTrials = 0;
    let indexingStartTime = 0;
    
    let activeWorkers = [];
    let resolveWorkerTask = null;
    // Bumped by abortActiveIndexing() (Stop button, or loading a new file
    // while indexing). startIndexing() captures the value at its own start;
    // finalizeIndexing() compares against the current value to detect that
    // its run was aborted/superseded and should no-op instead of overwriting
    // fresher state. See abortActiveIndexing() and finalizeIndexing().
    let indexingRunToken = 0;
    let sortState = { column: 'm20', direction: 'desc' };
    let workerURL = null;    

    // --- ONE SOURCE OF TRUTH FOR THE CACHE-BUSTING VERSION -----------------
    //
    // worker-logic.js used to be fetched under THREE different URLs, and only
    // one of them was in brutus.html:
    //
    //   main thread        worker-logic.js?v=NNN   (the <script> tag)
    //   CPU index worker   worker-logic.js         (bare -- never busted)
    //   refinement workers worker-logic.js?v=NNN   (from a hardcoded literal here)
    //
    // Three cache entries for one file. Bumping the version in the HTML alone
    // left the workers on the old key, so the main thread and the workers could
    // run DIFFERENT builds of the same file -- and a change to anything they
    // share (getSolutionKey, the scoring, the FoM) then disagrees across the
    // boundary. That presents as "the table and the report don't match", which
    // is about the least obvious symptom a stale cache could have chosen.
    //
    // Take the query string off whatever tag loaded main_app.js and reuse it
    // for every worker. refinement-worker.js already inherits it in turn via
    // self.location.search, so the whole chain now follows from the single
    // ?v= edit in brutus.html.
    //
    // Falls back to no query when the tag cannot be found (inlined script,
    // renamed bundle, a test harness with no DOM) -- which is exactly the old
    // behaviour, so nothing breaks, it merely stops busting.
    const APP_VERSION_QS = (() => {
        try {
            const el = (typeof document !== 'undefined' && document.currentScript &&
                        /main_app\.js/.test(document.currentScript.src || ''))
                ? document.currentScript
                : (typeof document !== 'undefined'
                    ? document.querySelector('script[src*="main_app.js"]')
                    : null);
            const src = el ? (el.getAttribute('src') || el.src || '') : '';
            const i = src.indexOf('?');
            return i >= 0 ? src.slice(i) : '';
        } catch (_) {
            return '';
        }
    })();
    // In-run reservoir size. This used to be 50/40, which meant any cell outside
    // the top 40 by M20 *at the moment it arrived* was destroyed before
    // applyFinalSieve, the post-process worker, or space-group analysis ever saw
    // it. On a GPU run producing tens of thousands of candidates that is a very
    // early, very lossy cut: a strong cell arriving late could be dropped simply
    // because the list was momentarily full of near-duplicates of each other.
    // The list is deduped by _solKey on insert and sieved at the end anyway, so
    // a larger reservoir costs a few hundred small objects and nothing else.
    const MAX_SOLUTIONS_BEFORE_PRUNING = 500;
    const PRUNE_TO_COUNT = 400; // Prune down to this many

    // The table is rebuilt from scratch on every animation frame during a run,
    // so the number of RENDERED rows is what actually costs time -- not the
    // number retained. Keep the display at the old order of magnitude.
    const MAX_DISPLAYED_SOLUTIONS = 50;



    // 1. Declare the throttle flag right above the function
let isTableUpdateScheduled = false;

// New function to centralize updates (Throttled with requestAnimationFrame)
const handleNewSolution = (newSolution) => {
    if (!newSolution || !newSolution.system) return;

    // Stamp the producing run. `solutions` is deliberately NOT cleared between
    // runs, so the array mixes cells found under the current peak set /
    // wavelength with cells found under whatever was configured previously.
    // finalizeIndexing() uses this to avoid re-running space-group analysis on
    // an old solution against a peak list it was never derived from.
    newSolution._runToken = indexingRunToken;

    // --- CROSS-WORKER DEDUP ---
    // Each refinement worker dedups only against its OWN foundSolutionMap, so
    // two workers can independently post the same physical cell. Undeduped
    // duplicates eat slots in the M20-capped list (50 -> 40) and can evict
    // genuinely distinct solutions BEFORE applyFinalSieve ever runs. The list
    // is capped at ~50, so a linear scan per insert is negligible, and keying
    // on getSolutionKey (loaded on the main thread via worker-logic.js) is
    // robust against the sort/slice/sieve reassignments elsewhere.
    let solKey = null;
    try { solKey = getSolutionKey(newSolution); } catch (e) { solKey = null; }

    let isDuplicate = false;
    // getSolutionKey now returns null (not undefined) for an unkeyable cell;
    // test truthiness so neither form can be filed as a real key -- an
    // undefined _solKey used to match every other undefined _solKey here.
    if (solKey) {
        newSolution._solKey = solKey;
        const dupIdx = solutions.findIndex(s => s._solKey === solKey);
        if (dupIdx !== -1) {
            isDuplicate = true;
            if (newSolution.m20 > solutions[dupIdx].m20) {
                solutions[dupIdx] = newSolution;   // keep the better copy
            } else {
                return;                            // weaker duplicate: drop, no re-render
            }
        }
    }

    if (!isDuplicate) {
        // --- FAST SYNCHRONOUS DATA OPERATIONS ---
        solutions.push(newSolution);

        // Max Limit Pruning
        if (solutions.length > MAX_SOLUTIONS_BEFORE_PRUNING) {
            // Always sort by quality (M20) before cutting
            // A missing or non-finite m20 makes this comparator return NaN,
            // which leaves the array in an arbitrary order -- and this sort
            // decides which solutions survive the prune.
            const rank = (x) => (x && isFinite(x.m20)) ? x.m20 : -Infinity;
            solutions.sort((a, b) => rank(b) - rank(a));
            solutions = solutions.slice(0, PRUNE_TO_COUNT);
        }
    }

    // LED indicator update (lightweight DOM manipulation is fine here)
    if (solutions.length === 1) ui.solutionsLed.className = 'led-indicator green';

    // --- HEAVY ASYNCHRONOUS DOM RENDERING ---
    // Gate the heavy sorting and DOM rebuilding behind requestAnimationFrame
    if (!isTableUpdateScheduled) {
        isTableUpdateScheduled = true;

        requestAnimationFrame(() => {
            // Sort solutions based on current UI state
            sortSolutions();

            // Re-Sync the ledger. Only the leading slice is rendered; row
            // click-handlers index into displayedSolutions, and the slice
            // preserves order, so indices stay aligned.
            displayedSolutions = solutions.slice(0, MAX_DISPLAYED_SOLUTIONS);

            // Rebuild the DOM table (now capped at browser refresh rate, ~60Hz)
            updateSolutionsTable();

            // Release the lock for the next frame
            isTableUpdateScheduled = false;
        });
    }
};


// Tabula rasa. A new file makes every previous session-scoped value (peaks,
// solutions, indexing run stats, sort/context state) meaningless. Critically,
// abort first: if an indexing run for the OLD file is still in flight, its
// GPU/worker results would keep trickling in via handleNewSolution() and land
// straight in the arrays reset below, silently mixing solutions from two
// different files together.
//
// Factored out of the file-load handler so the Unload button clears exactly the
// same things. Two copies of a reset list is how one of them ends up forgetting
// a field.
const resetSessionState = () => {
    abortActiveIndexing();

    pickedPeaks = [];
    solutions = [];
    displayedSolutions = [];
    selectedSolution = null;
    selectedPeakIndex = null;
    currentHklList = [];
    foundSolutionMap.clear();
    sortState = { column: 'm20', direction: 'desc' };
    ctxMenuTargetIndex = -1;
    taskProgress = [];
    taskTotals = [];
    cumulativeTrials = 0;
    gpuTotalTrials = 0;
    indexingStartTime = 0;
    lastDurationStr = '';
    lastIndexingStats = '';

    updatePeakTable();
    updateSolutionsTable();
    updateStartIndexingButtonState();
    ui.solutionsLed.className = 'led-indicator gray';
    ui.reportButton.disabled = true; // solutions is now empty (abortActiveIndexing() set this from the pre-reset count)
};

// Middle truncation. CSS text-overflow cuts the END, which removes the
// extension -- the one part of a filename you always want to keep. Keep the
// head and the whole suffix, elide the middle.
const truncateMiddle = (name, max = 30) => {
    const str = String(name || '');
    if (str.length <= max) return str;
    const dot = str.lastIndexOf('.');
    // Only treat a trailing dot-group as an extension if it plausibly is one.
    const ext = (dot > 0 && str.length - dot <= 8) ? str.slice(dot) : '';
    const keep = max - ext.length - 1;
    if (keep < 4) return str.slice(0, Math.max(1, max - 1)) + '\u2026';
    return str.slice(0, keep) + '\u2026' + ext;
};

const formatFileSize = (bytes) => {
    if (!Number.isFinite(bytes) || bytes < 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

// Single place that paints the chip, so the empty and loaded states cannot
// disagree about which classes are set.
const renderFileChip = (name, sizeBytes) => {
    if (!ui.fileName || !ui.fileChipName) return;
    if (!name) {
        ui.fileName.classList.add('is-empty');
        ui.fileChipName.textContent = 'No file loaded';
        ui.fileName.title = '';
        if (ui.fileChipSize) ui.fileChipSize.textContent = '';
        return;
    }
    ui.fileName.classList.remove('is-empty');
    ui.fileChipName.textContent = truncateMiddle(name);
    ui.fileName.title = name;   // full name on hover
    if (ui.fileChipSize) ui.fileChipSize.textContent = formatFileSize(sizeBytes);
};

// Full unload: session state AND the data itself, back to the pristine
// first-load view. Everything the load path switches ON is switched off here,
// which is why the two live next to each other.
const clearLoadedFile = () => {
    resetSessionState();

    fullExperimentalData = { tth: [], intensity: [] };
    workingExperimentalData = { tth: [], intensity: [] };
    loadedFileName = '';

    // Re-arm the input: without this, re-selecting the SAME file fires no
    // change event and Load File appears dead.
    if (ui.fileInput) ui.fileInput.value = '';
    if (ui.fileInputLabel) ui.fileInputLabel.classList.remove('error');
    if (ui.saveAsButton) ui.saveAsButton.disabled = true;

    // The load path disables Ka2 stripping when a file carries its own
    // wavelength. Leaving it disabled would strand the control with no file to
    // justify it.
    if (ui.stripKa2Checkbox) ui.stripKa2Checkbox.disabled = false;

    if (xrdChart) {
        try { xrdChart.resetZoom('none'); } catch (_) { /* no zoom state yet */ }
    }
    if (typeof updateAllMarkers === 'function') updateAllMarkers();

    if (ui.peakControls) ui.peakControls.classList.add('hidden');
    if (ui.indexingControls) ui.indexingControls.classList.add('hidden');
    if (ui.resultsContainer) ui.resultsContainer.style.display = 'none';
    if (ui.placeholder) ui.placeholder.style.display = '';

    renderFileChip(null);
    showStatus('File unloaded.', 'info', 2000);
};

const setupWorker = () => {
    try {
        workerURL = 'worker-logic.js' + APP_VERSION_QS;
    } catch (error) {
        console.error("Failed to set up worker URL:", error);
        showStatus("Critical error: Could not initialize indexing engine.", "error", 10000);
    }
};


    setupWorker(); 

    // ========================================================================
    // Refinement Worker Pool (batched)
    // ========================================================================
    // Parallelises the CPU refinement of GPU-produced candidate cells across N
    // Web Workers. Each worker loads worker-logic.js plus the refinement-worker.js
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
        constructor(size) {
            this.size = Math.max(1, size | 0);
            this.workers = [];
            this.nextBatchId = 1;
            this.pendingResolvers = new Map(); // batchId -> resolver
            this.rrIndex = 0;
            this._initialised = false;
            this._lastActivity = 0;   // timestamp of the last worker message
            this._drainWake = null;   // resolver woken when the last batch acks
            // Retained so a worker respawned mid-run can be brought up to the
            // same state as its peers. Without it a replacement worker would sit
            // in the pool uninitialised and reject every batch it was handed.
            this._initPayload = null;
            // Hard-crash tally, used to stop _spawn() rebuilding a worker that
            // dies deterministically (a bad worker-logic.js deploy crashes every
            // replacement the instant it is handed a batch, and refineBatch calls
            // _spawn() on every batch -- an unbounded spawn/crash loop).
            this._crashCount = 0;
        }

        _now() {
            return (typeof performance !== 'undefined' && performance.now)
                ? performance.now() : Date.now();
        }


        _spawn() {
            // Top the pool back up rather than bailing out whenever it is
            // non-empty. onerror removes crashed workers from this.workers, and
            // the old `length > 0` guard meant a pool that lost a worker never
            // got it back for the rest of the session.
            if (this.workers.length >= this.size) return; // idempotent
            // Give up rebuilding after enough crashes that the fault is clearly
            // the worker script, not one unlucky cell. Logged loudly because
            // refineBatch()'s empty-pool guard silently resolves after this.
            if (this._crashCount > this.size * 3) {
                if (!this._crashGiveUpLogged) {
                    this._crashGiveUpLogged = true;
                    console.error('[RefinementWorkerPool] Refusing to respawn: ' +
                                  `${this._crashCount} worker crashes. Refinement is disabled ` +
                                  'for this run (check refinement-worker.js / worker-logic.js).');
                }
                return;
            }
            for (let i = this.workers.length; i < this.size; i++) {
                const w = new Worker('refinement-worker.js' + APP_VERSION_QS);
                w.activeBatches = new Set(); // Track pending batches on this worker
                w.onmessage = (e) => this._onMessage(e, w);
                w.onerror = (err) => {
                    console.error(`Refinement worker hard crash:`, err.message);
                    this._crashCount++;
                    // Prevent infinite hangs by resolving all batches assigned to this crashed worker
                    for (const id of w.activeBatches) {
                        const resolve = this.pendingResolvers.get(id);
                        if (resolve) {
                            this.pendingResolvers.delete(id);
                            resolve();
                        }
                    }
                    w.activeBatches.clear();

                    // Remove the dead worker from the pool...
                    const deadIndex = this.workers.indexOf(w);
                    if (deadIndex !== -1) {
                        this.workers.splice(deadIndex, 1);
                    }
                    // ...and actually kill it. Dropping the last reference does
                    // NOT stop a Worker: the thread keeps running, holding its
                    // ~350 KB copy of worker-logic.js and its dedup map, and it
                    // can still post messages for batches we just force-resolved.
                    // Over a long session with repeated crashes that is a real
                    // leak of both memory and cores.
                    try { w.terminate(); } catch (_) {}
                };
                // A worker created after init() has already run starts blank and
                // would reject every batch with 'Worker not initialized'. Replay
                // the init payload so a replacement is immediately usable.
                if (this._initPayload) w.postMessage(this._initPayload);
                this.workers.push(w);
            }
        }


        _onMessage(e, w) {
            const msg = e.data;
            if (!msg || !msg.type) return;
            this._lastActivity = this._now();
            if (msg.type === 'solutions') {
                // Batched return path: the worker accumulates accepted cells and
                // posts them in chunks instead of paying a structured clone per
                // cell. Kept separate from 'solution' so the single-payload form
                // below still works for any unbatched producer.
                const list = msg.payloads;
                if (Array.isArray(list)) {
                    for (let i = 0; i < list.length; i++) handleNewSolution(list[i]);
                }
            } else if (msg.type === 'solution') {
                handleNewSolution(msg.payload);
            } else if (msg.type === 'cellError' || msg.type === 'batchError') {
                // Point 2 Fix: Stop dropping error messages
                const where = (msg.cellIndex !== undefined && msg.cellIndex !== null)
                    ? ` [cell #${msg.cellIndex}${msg.system ? ' ' + msg.system : ''}]` : '';
                console.warn(`[RefinementWorkerPool] Error in batch/task ${msg.batchId ?? msg.taskId}${where}:`,
                             msg.message, msg.cell || '');
            } else if (msg.type === 'done') {
                const id = (msg.batchId !== undefined) ? msg.batchId : msg.taskId;
                // The worker caps how many cellError messages it posts per batch,
                // so this count is the only place the true total shows up.
                if (msg.errors) {
                    console.warn(`[RefinementWorkerPool] batch ${id}: ${msg.errors} of ` +
                                 `${msg.processed} cell(s) failed (only the first few were reported).`);
                }
                if (w && w.activeBatches) w.activeBatches.delete(id);
                const resolve = this.pendingResolvers.get(id);
                if (resolve) {
                    this.pendingResolvers.delete(id);
                    resolve();
                }
                // Wake drain() on every ack, not just when the map empties. A
                // scoped drain (drain(stallMs, sinceId)) can be satisfied while
                // other callers' batches are still outstanding, and the old
                // size===0 condition would have made it sit out the 250 ms tick.
                // drain() re-checks its own predicate on wake, so an early
                // wake-up is free.
                if (this._drainWake) {
                    const wake = this._drainWake;
                    this._drainWake = null;
                    wake();
                }
            }
        }

        // Send per-run constants to every worker. `shared` is built from the
        // same data the old synchronous refineGpuCell used to rebuild per-cell.
        init(shared) {
            this._spawn();
            const payload = {
                type: 'init',
                baseParams: shared.baseParams,
                q_obs: shared.q_obs,
                original_indices: shared.original_indices,
                tth_obs_rad: shared.tth_obs_rad,
                peaks_sorted_by_q: shared.peaks_sorted_by_q,
                N_FOR_M20: shared.N_FOR_M20,
                min_m20: shared.min_m20,
                q_max: shared.q_max,
                d_min: shared.d_min,
            };
            this._initPayload = payload;
            for (const w of this.workers) {
                // postMessage structured-clones the payload. For the modest sizes
                // here (<100 KB total) this is fine.
                w.postMessage(payload);
            }
            this._initialised = true;
        }

        reset() {
            for (const w of this.workers) {
                w.postMessage({ type: 'reset' });
            }
        }

        // Split `cells` across workers round-robin and send one 'refineBatch'
        // message per worker. Returns a Promise that resolves when every worker
        // has acknowledged its slice.
        refineBatch(cells) {
            if (!this._initialised || !cells || cells.length === 0) {
                return Promise.resolve();
            }
            // Self-heal. _spawn() used to be reachable only from init(), so a
            // worker lost to a crash stayed lost for the rest of the run and the
            // pool silently shrank towards zero -- at which point every batch
            // was dropped on the floor by the guard above. Topping up here costs
            // nothing when the pool is full (_spawn is idempotent).
            this._spawn();
            if (this.workers.length === 0) return Promise.resolve();

            const n = cells.length;
            const numWorkers = this.workers.length;
            // Target chunk size: one chunk per worker if there are enough cells,
            // otherwise as many chunks as there are cells (one per worker, leftover
            // workers sit idle for this batch — fine, they'll get the next batch).
            const chunksToSend = Math.min(numWorkers, n);
            // How many cells per chunk? We distribute the remainder across the first
            // few chunks so sizes are within 1 of each other.
            const base = Math.floor(n / chunksToSend);
            const extra = n % chunksToSend;
            const promises = [];
            let offset = 0;
            for (let c = 0; c < chunksToSend; c++) {
                const size = base + (c < extra ? 1 : 0);
                if (size === 0) continue;
                const slice = cells.slice(offset, offset + size);
                offset += size;
                // Re-read the live count every iteration. `numWorkers` was
                // sampled before the loop, but onerror can splice a dead worker
                // out while we are still dispatching, which left rrIndex past the
                // end of the array -- `worker` came back undefined and
                // .activeBatches threw, aborting the whole batch.
                const live = this.workers.length;
                if (live === 0) break;
                this.rrIndex %= live;
                const worker = this.workers[this.rrIndex];
                this.rrIndex = (this.rrIndex + 1) % live;
                if (!worker) break;

                const batchId = this.nextBatchId++;
                const p = new Promise((resolve) => {
                    this.pendingResolvers.set(batchId, resolve);
                });
                worker.activeBatches.add(batchId);
                this._lastActivity = this._now();
                worker.postMessage({ type: 'refineBatch', cells: slice, batchId });
                promises.push(p);
            }
            return Promise.all(promises);
        }

        // Id of the next batch that will be created. Callers snapshot this
        // before dispatching work so they can drain only their own batches
        // (see drain(stallMs, sinceId)).
        mark() {
            return this.nextBatchId;
        }

        // Convenience wrapper for single-cell callers.
        refine(cell) {
            return this.refineBatch([cell]);
        }

        // Resolve once all currently-outstanding batches have settled.
        //
        // Event-driven (woken by the last 'done') with a slow watchdog tick, and
        // BOUNDED: a worker can die without ever firing onerror -- an OOM kill, a
        // frozen tab, a browser reclaiming a background process -- and the old
        // unbounded poll then spun forever, hanging the whole run with the UI
        // stuck mid-progress. If nothing is heard from any worker for stallMs the
        // outstanding batches are force-resolved and the run continues with
        // whatever was found.
        // `sinceId` (from mark()) scopes the wait to batches this caller
        // dispatched. Draining the whole pool made one task block on unrelated
        // work and made the stall watchdog measure pool-wide rather than
        // caller-relevant activity. Omit it to drain everything, as before.
        async drain(stallMs = 30000, sinceId = 0) {
            const outstanding = () => {
                if (!sinceId) return this.pendingResolvers.size;
                let n = 0;
                for (const id of this.pendingResolvers.keys()) if (id >= sinceId) n++;
                return n;
            };

            while (outstanding() > 0) {
                await new Promise(resolve => {
                    this._drainWake = resolve;
                    setTimeout(resolve, 250);   // watchdog tick
                });
                this._drainWake = null;
                const stillOut = outstanding();
                if (stillOut > 0 && (this._now() - this._lastActivity) > stallMs) {
                    console.warn(`[RefinementWorkerPool] ${stillOut} batch(es) ` +
                                 `silent for >${stallMs} ms; force-resolving so the run can finish.`);
                    for (const [id, resolve] of Array.from(this.pendingResolvers.entries())) {
                        if (!sinceId || id >= sinceId) {
                            resolve();
                            this.pendingResolvers.delete(id);
                        }
                    }
                    for (const w of this.workers) {
                        if (!w.activeBatches) continue;
                        for (const id of Array.from(w.activeBatches)) {
                            if (!sinceId || id >= sinceId) w.activeBatches.delete(id);
                        }
                    }
                    break;
                }
            }
        }

        terminate() {
            for (const w of this.workers) {
                try { w.terminate(); } catch (_) {}
            }
            this.workers = [];
            for (const resolve of this.pendingResolvers.values()) resolve();
            this.pendingResolvers.clear();
            if (this._drainWake) { const wake = this._drainWake; this._drainWake = null; wake(); }
            this._initialised = false;
            // Drop the retained payload too: it describes the run we just
            // killed, and replaying it into a worker spawned for the next run
            // would seed that run with the previous run's peaks and tolerances.
            this._initPayload = null;
        }
    }

    // Spawn size: one less than hardware concurrency, to leave a core for UI + GPU
    // driver. Minimum 1.
    const REFINE_POOL_SIZE = Math.max(1, (navigator.hardwareConcurrency || 4) - 2);
    const refinementPool = new RefinementWorkerPool(REFINE_POOL_SIZE);
    console.log(`[perf] Refinement worker pool: ${REFINE_POOL_SIZE} workers (hardwareConcurrency=${navigator.hardwareConcurrency || 'unknown'})`);

    //  systematic absences
    const max_hkl_analysis = 10;



    /**
     * Zhang Kα2 elimination algorithm (2026 method).
     *
     * Dual-wavelength q-space differencing method without empirical parameters
     * or iterative deconvolution[cite: 2]. Grounded in the physical separation of the Kα
     * doublet in reciprocal space[cite: 2]:
     *   1. Map observed 2θ profile to q-space independently using λ_Kα1 and λ_Kα2[cite: 2].
     *   2. Resample both profiles onto a common, uniformly spaced q-grid[cite: 2].
     *   3. Compute scaled difference Δy_α(q) = y_α2(q) - R * y_α1(q)[cite: 2].
     *   4. Isolate positive anomalous component corresponding to pure Kα1[cite: 2].
     *   5. Project back to 2θ domain via inverse mapping T_α2^(-1)[cite: 2].
     *
     * Reference: Zhang et al., Measurement 276 (2026) 121448[cite: 2].
     */
    const stripZhang = (tth, intensity, ka1, ka2, ratio) => {
        const n = tth.length;
        if (n === 0) return [];
        if (n === 1) return [intensity[0]];

        // Step 1: Coordinate mapping to q-space for both wavelengths
        const q_a1 = new Float64Array(n);
        const q_a2 = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const sin_th = Math.sin(tth[i] * Math.PI / 360);
            q_a1[i] = (4 * Math.PI * sin_th) / ka1;
            q_a2[i] = (4 * Math.PI * sin_th) / ka2;
        }

        // Define common uniform q-grid spanning the full mapped range
        const q_min = Math.min(q_a2[0], q_a1[0]);
        const q_max = Math.max(q_a2[n - 1], q_a1[n - 1]);
        const M = Math.max(2000, n * 3); // Dense sampling to prevent interpolation error
        const dq = (q_max - q_min) / (M - 1);

        // Helper: fast binary-search linear interpolation
        const interp = (target, x_src, y_src) => {
            const len = x_src.length;
            if (target <= x_src[0]) return y_src[0];
            if (target >= x_src[len - 1]) return y_src[len - 1];
            let lo = 0, hi = len - 1;
            while (hi - lo > 1) {
                const mid = (lo + hi) >> 1;
                if (x_src[mid] <= target) lo = mid;
                else hi = mid;
            }
            const t = (target - x_src[lo]) / (x_src[hi] - x_src[lo]);
            return y_src[lo] * (1 - t) + y_src[hi] * t;
        };

        // Step 2, 3 & 4: Resample to uniform grid, compute difference, and clamp positive
        const q_grid = new Float64Array(M);
        const delta_y_pos = new Float64Array(M);
        for (let m = 0; m < M; m++) {
            const q = q_min + m * dq;
            q_grid[m] = q;
            const y1 = interp(q, q_a1, intensity);
            const y2 = interp(q, q_a2, intensity);
            const diff = y2 - ratio * y1;
            delta_y_pos[m] = Math.max(diff, (1 - ratio) * Math.min(y1, y2));
        }

        // Step 5: Inverse mapping T_α2^(-1) back to experimental 2θ grid
        const corrected = new Array(n);
        for (let i = 0; i < n; i++) {
            // Under T_α2^(-1), angle tth[i] corresponds to reciprocal space coordinate q_a2[i]
            corrected[i] = interp(q_a2[i], q_grid, delta_y_pos);
        }

        return corrected;
    };

    const updateWorkingData = () => {
        if (fullExperimentalData.tth.length === 0) {
            workingExperimentalData = { tth: [], intensity: [] };
            return;
        }

        // Check if stripping is requested and we aren't in custom mode
        if (ui.stripKa2Checkbox.checked && ui.wavelengthPreset.value !== 'custom') {
            
            const selection = ui.wavelengthPreset.value;
            const element = selection.split('_')[0]; // "Cu", "Co", etc.
            
            const preset = WAVELENGTH_PRESETS[element];
            
            if (preset) {
                const { tth, intensity } = fullExperimentalData;
                // Apply Zhang Kα2 elimination (non-iterative q-space differencing)
                const strippedIntensity = stripZhang(tth, intensity, preset.ka1, preset.ka2, preset.ratio);
                
                // Save this as the working data for plotting AND peak search
                workingExperimentalData = { tth: [...tth], intensity: strippedIntensity };
            } else {
                // Fallback if preset parsing fails
                workingExperimentalData = { ...fullExperimentalData };
            }

        } else {
            // No stripping requested, use raw data
            workingExperimentalData = { 
                tth: [...fullExperimentalData.tth], 
                intensity: [...fullExperimentalData.intensity] 
            };
        }
    };

    /* ------------------------------------------------------------------
       Save-as / file-converter export
       ------------------------------------------------------------------
       The exported pattern is exactly what is currently plotted:
       workingExperimentalData already holds the raw-or-Kα2-stripped trace
       (whichever the strip checkbox selects), and we clip it to the 2θ
       interval currently visible on the chart, so zooming acts as a range
       selection for the export. */

    const getExportPattern = () => {
        const src = workingExperimentalData;
        if (!src || !src.tth || src.tth.length === 0) return null;

        // visibleTthRange() returns the on-screen 2θ window (full span when
        // the chart is not zoomed). It is defined later in the file but only
        // called here at click-time, so hoisting is not a concern.
        let lo = -Infinity, hi = Infinity;
        try {
            const r = visibleTthRange();
            if (r && isFinite(r[0]) && isFinite(r[1])) { lo = r[0]; hi = r[1]; }
        } catch (_) { /* chart not ready → export full pattern */ }

        const tth = [], intensity = [];
        for (let i = 0; i < src.tth.length; i++) {
            const t = src.tth[i];
            if (t >= lo && t <= hi) { tth.push(t); intensity.push(src.intensity[i]); }
        }
        if (tth.length === 0) return null;
        return { tth, intensity };
    };

    // Wavelength currently in effect (for formats that store it).
    const getExportWavelength = () => {
        const w = parseFloat(ui.wavelength.value);
        return (isFinite(w) && w > 0) ? w : 1.54056;
    };

    const num = (v, dp = 6) => Number(v).toFixed(dp);

    const buildExportContent = (fmt, pattern) => {
        const { tth, intensity } = pattern;
        const n = tth.length;
        const lambda = getExportWavelength();
        const step = n > 1 ? (tth[n - 1] - tth[0]) / (n - 1) : 0;
        let out = [];

        switch (fmt) {
            case 'xy':
            case 'dat': {
                for (let i = 0; i < n; i++) out.push(`${num(tth[i], 5)} ${num(intensity[i], 4)}`);
                return { text: out.join('\n') + '\n', ext: fmt === 'xy' ? 'xy' : 'dat', mime: 'text/plain' };
            }
            case 'csv': {
                out.push('2theta,intensity');
                for (let i = 0; i < n; i++) out.push(`${num(tth[i], 5)},${num(intensity[i], 4)}`);
                return { text: out.join('\n') + '\n', ext: 'csv', mime: 'text/csv' };
            }
            case 'uxd': {
                out.push('_FILEVERSION=1');
                out.push('; Exported by Brutus');
                out.push(`_WL1=${num(lambda, 6)}`);
                out.push(`_START=${num(tth[0], 6)}`);
                out.push(`_STEPSIZE=${num(step, 6)}`);
                out.push('_STEPCOUNT=' + n);
                out.push('_COUNTS');
                // 5 values per line, matching common UXD style
                for (let i = 0; i < n; i += 5) {
                    out.push(intensity.slice(i, i + 5).map(v => num(v, 3)).join(' '));
                }
                return { text: out.join('\n') + '\n', ext: 'uxd', mime: 'text/plain' };
            }
            case 'gsas': {
                // Minimal GSAS ESD (constant-step, ESD format). Header + BANK line.
                // GSAS uses centidegrees for CONST start/step.
                const start100 = tth[0] * 100;
                const step100 = step * 100;
                const lines = [];
                const npts = n;
                const recPerLine = 5; // ESD packs (pos?, no) — here we use "ESD": Y and sigma
                // ESD format: each value field width 12: intensity then sqrt(I) as esd
                const dataLines = [];
                let buf = '';
                let count = 0;
                // This app's ESD reader takes tokens at index 1,3,5,... as the
                // intensity, so we write each record as "<esd> <intensity>".
                for (let i = 0; i < n; i++) {
                    const yi = Math.max(0, intensity[i]);
                    const esd = Math.sqrt(yi > 0 ? yi : 1);
                    buf += esd.toFixed(2).padStart(12) + yi.toFixed(2).padStart(12);
                    count++;
                    if (count === recPerLine) { dataLines.push(buf); buf = ''; count = 0; }
                }
                if (buf.length) dataLines.push(buf);
                lines.push('Exported by Brutus'.padEnd(80));
                lines.push(
                    `BANK 1 ${npts} ${Math.ceil(npts / recPerLine)} CONST ` +
                    `${start100.toFixed(2)} ${step100.toFixed(2)} 0 0 ESD`
                );
                for (const dl of dataLines) lines.push(dl);
                return { text: lines.join('\n') + '\n', ext: 'esd', mime: 'text/plain' };
            }
            case 'xrdml': {
                const positions = `        <positions axis="2Theta" unit="deg">\n` +
                    `          <startPosition>${num(tth[0], 6)}</startPosition>\n` +
                    `          <endPosition>${num(tth[n - 1], 6)}</endPosition>\n` +
                    `        </positions>`;
                const counts = intensity.map(v => Math.round(v)).join(' ');
                const xml =
`<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5">
  <xrdMeasurement measurementType="Scan" status="Completed">
    <usedWavelength intended="K-Alpha 1">
      <kAlpha1 unit="Angstrom">${num(lambda, 6)}</kAlpha1>
    </usedWavelength>
    <scan appendNumber="0" mode="Continuous" scanAxis="2Theta">
      <dataPoints>
${positions}
        <intensities unit="counts">${counts}</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
`;
                return { text: xml, ext: 'xrdml', mime: 'application/xml' };
            }
            case 'brukerxml': {
                const counts = intensity.map(v => Math.round(v)).join(' ');
                const xml =
`<?xml version="1.0" encoding="utf-8"?>
<RawDataFile>
  <DataRoutes>
    <DataRoute>
      <ScanInformation>
        <ScanAxes>
          <ScanAxisInfo AxisName="TwoTheta">
            <Start axis="TwoTheta">${num(tth[0], 6)}</Start>
            <startPosition axis="TwoTheta">${num(tth[0], 6)}</startPosition>
            <increment axis="TwoTheta">${num(step, 6)}</increment>
          </ScanAxisInfo>
        </ScanAxes>
        <usedWavelength kAlpha1="${num(lambda, 6)}" />
      </ScanInformation>
      <Datum>
        <dataPoints>
          <counts>${counts}</counts>
        </dataPoints>
      </Datum>
    </DataRoute>
  </DataRoutes>
</RawDataFile>
`;
                return { text: xml, ext: 'xml', mime: 'application/xml' };
            }
            default:
                return { text: '', ext: 'txt', mime: 'text/plain' };
        }
    };

    const baseNameNoExt = (name) => {
        if (!name) return 'pattern';
        const dot = name.lastIndexOf('.');
        return dot > 0 ? name.slice(0, dot) : name;
    };

    const triggerDownload = (text, filename, mime) => {
        const blob = new Blob([text], { type: mime + ';charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };

    // Formats that do not store an explicit per-point 2θ column, so they must
    // assume a constant step and/or embed a single wavelength. We warn the user
    // what metadata is being written into these on their behalf.
    const SAVE_FORMAT_META = {
        xrdml:     { wavelength: true,  constStep: true,  label: 'XRDML' },
        brukerxml: { wavelength: true,  constStep: true,  label: 'Bruker XML' },
        uxd:       { wavelength: true,  constStep: true,  label: 'UXD' },
        gsas:      { wavelength: false, constStep: true,  label: 'GSAS ESD' },
        xy:        { wavelength: false, constStep: false, label: 'XY' },
        csv:       { wavelength: false, constStep: false, label: 'CSV' },
        dat:       { wavelength: false, constStep: false, label: 'DAT' }
    };

    // True when the exported (currently visible) 2θ grid is not evenly spaced.
    // Constant-step formats would silently resample such data onto a regular grid.
    const patternStepIsIrregular = () => {
        const p = getExportPattern();
        const t = p ? p.tth : null;
        if (!t || t.length < 3) return false;
        const step = (t[t.length - 1] - t[0]) / (t.length - 1);
        if (!(Math.abs(step) > 0)) return false;
        const tol = Math.abs(step) * 0.02; // 2% of the nominal step
        for (let i = 1; i < t.length; i++) {
            if (Math.abs((t[i] - t[i - 1]) - step) > tol) return true;
        }
        return false;
    };

    const updateSaveInfo = () => {
        const fmt = ui.saveFormatSelect.value;
        const meta = SAVE_FORMAT_META[fmt] || {};
        const notes = [];

        // Kα2 state of the data being written (workingExperimentalData is what
        // is plotted, already stripped when the main strip control is on).
        const stripped = !!(ui.stripKa2Checkbox && ui.stripKa2Checkbox.checked
                            && ui.wavelengthPreset && ui.wavelengthPreset.value !== 'custom');
        if (stripped) {
            notes.push(`Kα2-stripped data will be saved (the plotted pattern has Kα2 removed).`);
        } else {
            notes.push(`Raw data will be saved (no Kα2 stripping applied to the plotted pattern).`);
        }

        // The export is clipped to what's on screen when zoomed.
        const p = getExportPattern();
        if (p && p.tth.length) {
            const full = workingExperimentalData.tth.length;
            if (p.tth.length < full) {
                notes.push(`Only the visible range is exported: ${p.tth[0].toFixed(3)}–${p.tth[p.tth.length - 1].toFixed(3)}° 2θ (${p.tth.length} of ${full} points). Reset the zoom to export the full pattern.`);
            }
        }
        if (meta.wavelength) {
            const w = getExportWavelength();
            notes.push(`${meta.label} stores a wavelength — the value from the wavelength box (${w.toFixed(5)} Å) will be written.`);
        }
        if (meta.constStep && patternStepIsIrregular()) {
            notes.push(`${meta.label} assumes a constant 2θ step; this scan is not evenly spaced, so intensities will be written against a uniform start/step and positions may shift slightly.`);
        }
        ui.saveMenuMsg.innerHTML = notes.length ? 'Info: ' + notes.join('<br>Info: ') : '';
    };

    const openSaveMenu = () => {
        const p = getExportPattern();
        if (!p) { showStatus('No data to save.', 'error'); return; }
        updateSaveInfo();
        ui.saveMenuOverlay.classList.add('open');
    };
    const closeSaveMenu = () => ui.saveMenuOverlay.classList.remove('open');

    // MIME + human description per extension, for the native save dialog's type list.
    const EXT_DESC = {
        xy: 'XY data', csv: 'CSV', dat: 'Data', xrdml: 'XRDML',
        xml: 'Bruker XML', uxd: 'Bruker UXD', esd: 'GSAS ESD'
    };

    // Save via the File System Access API when available (real OS save dialog
    // where the user chooses folder + name), otherwise fall back to a normal
    // browser download into the default downloads folder.
    const saveTextToFile = async (text, suggestedName, ext, mime) => {
        if (window.showSaveFilePicker) {
            try {
                const handle = await window.showSaveFilePicker({
                    suggestedName,
                    types: [{
                        description: EXT_DESC[ext] || 'Data file',
                        accept: { [mime || 'text/plain']: ['.' + ext] }
                    }]
                });
                const writable = await handle.createWritable();
                await writable.write(text);
                await writable.close();
                return handle.name || suggestedName;
            } catch (err) {
                // AbortError = user cancelled the dialog; do nothing.
                if (err && err.name === 'AbortError') return null;
                // Any other failure (e.g. permissions) → fall through to download.
                console.warn('showSaveFilePicker failed, falling back to download:', err);
            }
        }
        triggerDownload(text, suggestedName, mime);
        return suggestedName;
    };

    const doSave = async () => {
        const pattern = getExportPattern();
        if (!pattern) { showStatus('No data to save.', 'error'); return; }

        const fmt = ui.saveFormatSelect.value;
        const { text, ext, mime } = buildExportContent(fmt, pattern);
        const suggestedName = `${baseNameNoExt(loadedFileName) || 'pattern'}.${ext}`;

        ui.saveMenuConfirm.disabled = true;
        const savedName = await saveTextToFile(text, suggestedName, ext, mime);
        ui.saveMenuConfirm.disabled = false;

        if (savedName === null) return; // user cancelled the native dialog
        closeSaveMenu();
        showStatus(`Saved ${savedName} (${pattern.tth.length} points).`, 'info', 4000);
    };

    // `Load File` is a <button> now rather than a <label for>, so the click has
    // to be forwarded to the hidden input by hand. The trade is worth it: as a
    // label it carried .file-input-label, whose background overrode
    // .btn-secondary and made it a visibly different colour from `Save as`.
    if (ui.fileInputLabel) ui.fileInputLabel.addEventListener('click', () => {
        if (ui.fileInput) ui.fileInput.click();
    });

    // Unloading throws away peaks and solutions, so confirm when there is
    // something to lose. A stray click on a small x should not silently bin an
    // indexing run.
    //
    // This used window.confirm(), which draws the browser's own panel: wrong
    // font, wrong colours, wrong button order, and it announces the origin
    // ("127.0.0.1:5500 indique") above the question. Every other destructive or
    // modal step in Brutus -- Save as, Refine MC, Swap hkl, Space Group MC --
    // uses the same in-page dialog, so this one does too.
    const openUnloadDialog = () => {
        if (!ui.unloadOverlay) { clearLoadedFile(); return; }   // markup missing: do not trap the user
        if (ui.unloadFile) ui.unloadFile.textContent = loadedFileName || 'the current file';
        if (ui.unloadLosses) {
            const losses = [];
            if (pickedPeaks.length) losses.push(`${pickedPeaks.length} picked peak${pickedPeaks.length === 1 ? '' : 's'}`);
            if (solutions.length)   losses.push(`${solutions.length} solution${solutions.length === 1 ? '' : 's'}`);
            if (isIndexing)  losses.push('the indexing run now in progress');
            ui.unloadLosses.innerHTML = losses.map(t => `<li>${t}</li>`).join('');
            ui.unloadLosses.style.display = losses.length ? '' : 'none';
        }
        ui.unloadOverlay.classList.add('open');
        if (ui.unloadConfirm) ui.unloadConfirm.focus();   // Enter confirms, Esc cancels
    };
    const closeUnloadDialog = () => {
        if (ui.unloadOverlay) ui.unloadOverlay.classList.remove('open');
    };

    if (ui.fileChipClear) ui.fileChipClear.addEventListener('click', () => {
        // Nothing to lose: unload straight away rather than making the user
        // dismiss a dialog to discard nothing.
        const atRisk = (solutions.length > 0) || (pickedPeaks.length > 0) || isIndexing;
        if (!atRisk) { clearLoadedFile(); return; }
        openUnloadDialog();
    });
    if (ui.unloadCancel)  ui.unloadCancel.addEventListener('click', closeUnloadDialog);
    if (ui.unloadConfirm) ui.unloadConfirm.addEventListener('click', () => {
        closeUnloadDialog();
        clearLoadedFile();
    });
    if (ui.unloadOverlay) ui.unloadOverlay.addEventListener('click', (e) => {
        if (e.target === ui.unloadOverlay) closeUnloadDialog();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && ui.unloadOverlay && ui.unloadOverlay.classList.contains('open')) {
            closeUnloadDialog();
        }
    });

    if (ui.saveAsButton)      ui.saveAsButton.addEventListener('click', openSaveMenu);
    if (ui.saveFormatSelect)  ui.saveFormatSelect.addEventListener('change', updateSaveInfo);
    if (ui.saveMenuCancel)    ui.saveMenuCancel.addEventListener('click', closeSaveMenu);
    if (ui.saveMenuConfirm)   ui.saveMenuConfirm.addEventListener('click', doSave);
    if (ui.saveMenuOverlay)   ui.saveMenuOverlay.addEventListener('click', (e) => {
        if (e.target === ui.saveMenuOverlay) closeSaveMenu();
    });

    // Chart help: click the "?" button to toggle the panel (no hover trigger).
    const helpBtn = document.getElementById('help-icon-btn');
    const helpPanel = document.getElementById('help-tooltip');
    if (helpBtn && helpPanel) {
        const closeHelp = () => { helpPanel.classList.remove('open'); helpBtn.setAttribute('aria-expanded', 'false'); };
        helpBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const open = helpPanel.classList.toggle('open');
            helpBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        // Clicking anywhere outside the panel or the button closes it.
        document.addEventListener('click', (e) => {
            if (!helpPanel.contains(e.target) && e.target !== helpBtn) closeHelp();
        });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeHelp(); });
    }

    // event listeners 
    ui.wavelengthPreset.addEventListener('change', handleWavelengthPresetChange);
    // Edits to the wavelength input auto-switch the preset to "custom" when the
    // typed value no longer matches the selected preset.
    const debouncedWavelengthChange = debounce(() => {
    const typed = parseFloat(ui.wavelength.value);
    // FIX: Only trigger heavy recalculations if the user typed a valid, realistic wavelength
    if (isFinite(typed) && typed >= 0.1 && typed <= 10.0) {
        handleWavelengthValueChange();
    }
}, 400); // Wait 400ms after they finish typing

ui.wavelength.addEventListener('input', debouncedWavelengthChange);


    if (ui.stripKa2Checkbox) {
        ui.stripKa2Checkbox.addEventListener('change', () => {
            // Sync the wavelength input to match the active radiation:
            //   strip ON  → peaks are at Kα1, so λ = ka1
            //   strip OFF → peaks are at the doublet centroid, so λ = ka_avg
            const sel = ui.wavelengthPreset.value;
            if (sel !== 'custom') {
                const [element, type] = sel.split('_');
                const data = WAVELENGTH_PRESETS[element];
                if (data && type !== 'ka1') {
                    ui.wavelength.value = ui.stripKa2Checkbox.checked
                        ? data.ka1.toFixed(5)
                        : data.ka_avg.toFixed(5);
                }
            }

            updateWorkingData(); // Calculates new intensity
            
            // Redraw Chart
            if (xrdChart) {
                // d and Q are functions of lambda, so a new wavelength moves every
                // abscissa on those axes - the whole plot has to be rebuilt, not
                // just re-fed with intensities. On theta / 2-theta nothing moves.
                if (xAxisMode === 'd' || xAxisMode === 'q') {
                    rebuildPlot(false);
                } else {
                    setExperimentalTrace(false);
                }
            }

            // Re-find peaks on the (possibly newly stripped) data using the
            // updated wavelength. findPeaks rebuilds pickedPeaks from scratch
            // so no separate recalculatePeakValues call is needed.
            findPeaks(); 
        });
    }

    // Validation , added wave, april 20
    const inputsToValidate = [
        { id: 'wavelength', el: ui.wavelength, default: 1.54184 },
        { id: 'max-volume', el: ui.maxVolume, default: 2000 },
        { id: 'tth-error', el: ui.tthError, default: 0.04 },
        { id: 'impurity-peaks', el: ui.impurityPeaksInput, default: 1 },
        // Add new inputs to validation if they exist
        { id: 'gpu-hkl-triplets', el: ui.gpuHklTriplets, default: 100 },
        { id: 'gpu-peaks-count', el: ui.gpuPeaksCount, default: 7 },
        { id: 'gpu-fom-threshold', el: ui.gpuFomThreshold, default: 3.0 },
        { id: 'gpu-buffer-size', el: ui.gpuBufferSize, default: 50 },
    ];

    inputsToValidate.forEach(({ id, el, default: defaultValue }) => {
        if (el) {
            el.addEventListener('blur', () => {
                const min = parseFloat(el.min);
                const max = parseFloat(el.max);
                let value = parseFloat(el.value);
                if (isNaN(value)) { el.value = defaultValue; return; }
                if (!isNaN(min) && value < min) el.value = min;
                if (!isNaN(max) && value > max) el.value = max;
                
                // Optional: Update status text if GPU params changed
                if (id.startsWith('gpu')) updateGpuStatusText(); 
            });
        } else {
            // Suppress error for elements that might not be in the DOM yet/anymore
            // console.warn(`Initialization Warning: Element id="${id}" not checked.`);
        }
    });

    if (ui.wavelength) {
        // Recalculate d-spacings if the user manually changes the wavelength (Custom mode)
        ui.wavelength.addEventListener('change', () => {
            if (pickedPeaks.length > 0) {
                recalculatePeakValues();
                updatePeakTable();
            }
        });
    }

/**
         * Smart file detector. It checks for known headers and extensions
         * and falls back to a generic 2-column parser.
         */
        // --- FullProf free-format .dat -------------------------------------
        //
        //     !COMM  Y2O3 D8 0.3 0.3 0.1      (zero or more comment lines)
        //     15.00 0.01 145.90               (start, step, end in 2-theta)
        //        25   22   19   21  ...       (intensities, N per line)
        //
        // Detection is by CONTENT, never by extension. `.dat` is already the
        // generic whitespace 2-column fallback and is also one of Brutus's own
        // export formats, so keying on the extension would break every existing
        // 2-column .dat. The discriminator is the header triple together with
        // the value count: a file whose second line is three numbers AND that
        // then carries (end-start)/step + 1 values is this format and is very
        // unlikely to be anything else. A 2-column file fails on the triple; a
        // 3-column x/y/sigma file fails on the count.
        //
        // Left unguarded, such a file parses SILENTLY AND WRONGLY through the
        // generic parser: it reads "15.00 0.01" as a point, then "25 22" as the
        // next, producing a plausible-looking pattern made entirely of counts
        // plotted against counts.
        const scanFullProfDat = (lines) => {
            // Find the header: the first line that is exactly three numbers.
            // Written as a search rather than "skip comments, then read line 1"
            // because the preamble is not standardised -- some files open with
            // '!COMM ...', some with '#', some with a bare title line carrying
            // no marker at all, and some with nothing.
            const HEADER_SEARCH_LIMIT = 25;
            let i = -1, start = 0, step = 0, end = 0;
            for (let j = 0; j < lines.length && j < HEADER_SEARCH_LIMIT; j++) {
                const t = lines[j].trim();
                if (t === '') continue;
                const nums = t.split(/[\s,]+/).map(Number);
                if (nums.length !== 3 || !nums.every(Number.isFinite)) continue;
                if (!(nums[1] > 0) || !(nums[2] > nums[0])) continue;
                [start, step, end] = nums;
                i = j;
                break;
            }
            if (i === -1) return null;

            const nExpected = Math.round((end - start) / step) + 1;
            if (!(nExpected >= 2) || nExpected > 5e6) return null;

            // Every remaining token must be a number, or this is not the format.
            const values = [];
            const perLine = [];
            for (let j = i + 1; j < lines.length; j++) {
                const t = lines[j].trim();
                if (t === '') continue;
                if (t.startsWith('!') || t.startsWith('#')) continue;   // trailing comments
                const parts = t.split(/[\s,]+/);
                perLine.push(parts.length);
                for (const part of parts) {
                    const v = Number(part);
                    if (!Number.isFinite(v)) return null;
                    values.push(v);
                }
            }
            if (values.length === 0) return null;

            // TWO independent signatures, either of which is decisive. Relying on
            // the count alone was too brittle: a real file whose header declares
            // a longer scan than it carries -- a truncated collection, a scan
            // stopped early -- was rejected and fell through to the generic
            // two-column reader, which then parsed it silently and wrongly.
            //
            //   (a) the count matches the header arithmetic; or
            //   (b) the data are PACKED, i.e. most lines carry four or more
            //       values. That alone separates this format from everything
            //       else that reaches here: two-column files have two per line
            //       and x/y/sigma files have three.
            const slack = Math.max(2, ...perLine);
            const countMatches = values.length >= nExpected - 1 && values.length <= nExpected + slack;

            const packedLines = perLine.filter(n => n >= 4).length;
            const isPacked = packedLines >= 2 && packedLines >= 0.8 * perLine.length;

            if (!countMatches && !isPacked) return null;

            return { start, step, end, nExpected, values, countMatches };
        };

        const parseFullProfDatFile = (text) => {
            const scan = scanFullProfDat(text.trim().split(/\r?\n/));
            if (!scan) throw new Error('Not a FullProf free-format .dat file.');

            // Trust the header for the count when the file over-runs it (padded
            // final line); trust the file when it carries fewer.
            const n = Math.min(scan.nExpected, scan.values.length);
            const intensity = scan.values.slice(0, n);
            const tth = Array.from({ length: n }, (_, k) => scan.start + k * scan.step);

            if (n < scan.nExpected) {
                console.warn(`FullProf .dat: header declares ${scan.nExpected} points ` +
                             `(${scan.start} to ${scan.end} step ${scan.step}), file carries ` +
                             `${scan.values.length}. Reading ${n}; the pattern ends at ` +
                             `${(scan.start + (n - 1) * scan.step).toFixed(4)} deg.`);
            }
            // No wavelength is recorded in this format, so none is returned and
            // the caller falls back to the Cu preset as it does for .xy.
            return { tth, intensity };
        };

        const detectAndParseFile = (fileName, fileContent) => {
            const name = fileName.toLowerCase();
            const lines = fileContent.trim().split(/\r?\n/);
            const firstLine = lines.length > 0 ? lines[0].trim() : '';
            const upperContent = fileContent.substring(0, 500).toUpperCase(); // Check first 500 chars

            // Parser Registry
            const PARSER_REGISTRY = [
                { // XRDML
                    test: (name, content) => name.endsWith('.xrdml') || (content.includes('<?xml') && content.includes('<xrdMeasurement')),
                    parser: parseXrdmlFile
                },
                { // Bruker XML (uncompressed RawDataFile; e.g. RawData0.xml from an unzipped .brml)
                    test: (name, content) => (name.endsWith('.xml') && content.includes('<RawDataFile')) || (content.includes('<?xml') && content.includes('<RawDataFile')),
                    parser: parseBrukerXmlFile
                },
                { // UXD
                    test: (name, content, firstLine) => name.endsWith('.uxd') || firstLine.startsWith('_FILEVERSION'),
                    parser: parseUxdFile
                },
                { // Rigaku RAS
                    test: (name, content, firstLine, upper) => name.endsWith('.ras') || upper.includes('*RAS_HEADER_START'),
                    parser: parseRigakuRasFile
                },
                { // Philips UDF/RD/SD
                    test: (name) => name.endsWith('.udf') || name.endsWith('.rd') || name.endsWith('.sd'),
                    parser: parsePhilipsUdfFile
                },
                { // GSAS ESD/XRA
                    test: (name, content, firstLine, upper, allLines) => allLines.some(line => line.trim().toUpperCase().startsWith('BANK')),
                    parser: (content, allLines) => {
                        const bankLine = allLines.find(line => line.trim().toUpperCase().startsWith('BANK'));
                        if (bankLine.toUpperCase().includes('STD')) {
                            return parseGsasXraFile(content);
                        }
                        return parseGsasEsdFile(content);
                    }
                },
                { // FullProf free format (start/step/end + N values per line)
                    test: (name, content, firstLine, upper, allLines) => scanFullProfDat(allLines) !== null,
                    parser: parseFullProfDatFile
                },
                { // Jade MDI (treat as 2-column)
                     test: (name, content, firstLine, upper) => name.endsWith('.mdi') && (upper.includes('2-THETA, INTENSITY') || upper.startsWith('(SAMPLE')),
                     parser: parseDataFile
                },
                { // pdCIF
                    test: (name, content) => name.endsWith('.cif') || content.includes('_pd_meas_2theta_scan'),
                    parser: parsePdCifFile
                }
            ];
            
            // registres
            for (const rule of PARSER_REGISTRY) {
                try {
                    if (rule.test(name, fileContent, firstLine, upperContent, lines)) {
                        // Pass 'content' to parser, but 'lines' to the special GSAS one
                        if (rule.parser.length > 1) {
                             return rule.parser(fileContent, lines); // For GSAS parser
                        }
                        return rule.parser(fileContent);
                    }
                } catch (e) {
                    console.warn(`Parser ${rule.parser.name} failed, trying next...`, e.message);
                }
            }

            // Fallback for all other 2-column-like formats
            // This will attempt to parse: .xy, .csv, .txt, .dat, .asc, etc.... à revoir les fichiers type dans Convert 2 ?
            return parseDataFile(fileContent, fileName);
        };
    
        /**
         * Generic 2-column parser. This is the fallback for most text files.
         * Includes validation logic for 2-theta (X) and step size (dX).
         */
        const parseDataFile = (text, fileName = "") => {
            const lines = text.trim().split(/\r?\n/);
            const tth = [], intensity = [];
            let last_x = -Infinity;
            let suspicious_steps = 0;
            let positive_x_values = 0;
            let negative_steps = 0;
            let headerLines = 0;
            let dataStarted = false;

            lines.forEach(line => {
                // Skip commented or empty lines
                if (line.startsWith('#') || line.startsWith('//') || line.startsWith('!') || line.startsWith(';') || line.trim() === '') {
                    if (!dataStarted) headerLines++;
                    return;
                }
                
                // Skip non-commented header lines (that contain letters)
                if (!dataStarted) {
                    if (/[a-zA-Z]/.test(line)) { 
                        headerLines++;
                        return;
                    }
                }

                // Decimal comma vs comma separator.
                // "10,5 200" (European decimal) and "10.5,200.7" (CSV) are both
                // valid and cannot be told apart by a blanket substitution, so
                // decide per line. A comma is a DECIMAL MARK when either the
                // line already has another delimiter doing the separating
                // (whitespace or semicolon), or there is a single comma and no
                // dot anywhere. Otherwise the comma is the field separator.
                // Dots always win: if the line contains a dot, that is the
                // decimal mark and any comma must be a separator.
                const rawLine = line.trim();
                const commaCount = (rawLine.match(/,/g) || []).length;
                const hasOtherDelim = /[\s;]/.test(rawLine);
                const hasDot = rawLine.includes('.');
                const commaIsDecimal = commaCount > 0 && !hasDot &&
                    (hasOtherDelim || commaCount === 1);
                const sanitizedLine = commaIsDecimal
                    ? rawLine.replace(/,(\d)/g, '.$1')
                    : rawLine.replace(/,/g, ' ');
                const parts = sanitizedLine.split(/[\s;]+/);
                if (parts.length < 2) return;

                const x = parseFloat(parts[0]);
                const y = parseFloat(parts[1]);

                // If we get non-numeric data, it's either a header or a bad line
                if (isNaN(x) || isNaN(y)) {
                    if (!dataStarted) headerLines++; // Still in the header
                    return;
                }
                
                dataStarted = true; // First valid numeric pair found

                // vérif
                if (x > 0) positive_x_values++;

                if (last_x !== -Infinity) {
                    const dX = x - last_x;
                    if (dX < 0) {
                        negative_steps++; // Data is descending
                    } else if (dX > 0 && (dX < 0.0001 || dX > 0.2)) { 
                        suspicious_steps++; // Step size is weird
                    }
                }
                last_x = x;
                

                tth.push(x);
                intensity.push(y);
            });

            // Final checks (log warnings to console) 
            if (tth.length > 10) { 
                if (positive_x_values / tth.length < 0.5) {
                    console.warn(`Data File (${fileName}) Warning: Most 2-theta (X) values are zero or negative. This is unusual for XRD data.`);
                }
                if (negative_steps / tth.length > 0.8) {
                     console.warn(`Data File (${fileName}) Warning: Data appears to be sorted in descending 2-theta order.`);
                }
                if (suspicious_steps / tth.length > 0.2) {
                    console.warn(`Data File (${fileName}) Warning: Many data points have a step size outside the typical range (0.0001° - 0.2°). Check file format.`);
                }
            } else if (tth.length === 0) {
                 throw new Error(`Could not parse any 2-column data from ${fileName}. File may be binary or have an unknown header.`);
            }

            return { tth, intensity };
        };

        const parseBrukerXmlFile = (xmlString) => { const parser = new DOMParser(); const xmlDoc = parser.parseFromString(xmlString, "application/xml"); if (xmlDoc.querySelector("parsererror")) { throw new Error("Error parsing Bruker XML file."); } let wavelength = null; const wlNode = xmlDoc.querySelector('usedWavelength'); if (wlNode) { const kAlpha1 = wlNode.getAttribute('kAlpha1'); if (kAlpha1) wavelength = parseFloat(kAlpha1); } const intensityNode = xmlDoc.querySelector("dataPoints > counts"); if (!intensityNode) throw new Error("No <counts> data found in Bruker XML file."); const intensity = intensityNode.textContent.trim().split(/\s+/).map(Number); const startPosNode = xmlDoc.querySelector('startPosition[axis="TwoTheta"]'); const stepSizeNode = xmlDoc.querySelector('increment[axis="TwoTheta"]'); if (!startPosNode || !stepSizeNode) throw new Error("Could not find scan parameters in Bruker XML file."); const startPos = parseFloat(startPosNode.textContent); const stepSize = parseFloat(stepSizeNode.textContent); const tth = Array.from({ length: intensity.length }, (_, i) => startPos + i * stepSize); return { tth, intensity, wavelength }; };
        const parseXrdmlFile = (xmlString) => { const parser = new DOMParser(); const xmlDoc = parser.parseFromString(xmlString, "application/xml"); if (xmlDoc.querySelector("parsererror")) { throw new Error("Error parsing XRDML file."); } let wavelength = null; const kAlpha1Node = xmlDoc.querySelector("kAlpha1"); if (kAlpha1Node?.textContent) wavelength = parseFloat(kAlpha1Node.textContent); const intensityNode = xmlDoc.querySelector("intensities") || xmlDoc.querySelector("counts"); if (!intensityNode) throw new Error("Could not find <intensities> or <counts> in XRDML file."); const intensity = intensityNode.textContent.trim().split(/\s+/).map(Number); const positionsNode = xmlDoc.querySelector('positions[axis="2Theta"]'); if (!positionsNode) throw new Error("Could not find <positions> in XRDML file."); const startPosNode = positionsNode.querySelector("startPosition"); const endPosNode = positionsNode.querySelector("endPosition"); if (!startPosNode || !endPosNode) throw new Error("Could not find start/end positions in XRDML."); const startPos = parseFloat(startPosNode.textContent); const endPos = parseFloat(endPosNode.textContent); if (!isFinite(startPos) || !isFinite(endPos)) throw new Error("XRDML start/end positions are not numeric."); if (intensity.length === 0) throw new Error("XRDML file contains no intensity points."); /* With a single point the old expression divided by zero and produced NaN/Infinity for every 2-theta; emit the single start position instead. */ const step = intensity.length > 1 ? (endPos - startPos) / (intensity.length - 1) : 0; const tth = Array.from({ length: intensity.length }, (_, i) => startPos + i * step); return { tth, intensity, wavelength }; };
        const parseRigakuRasFile = (text) => { const lines = text.trim().split(/\r?\n/); const tth = [], intensity = []; let inDataSection = false; let wavelength = null; for (const line of lines) { const upperLine = line.toUpperCase(); if (upperLine.startsWith('*WAVE_LENGTH') || upperLine.startsWith('*MEAS_COND_XG_WAVE_LENGTH')) { const parts = line.trim().split(/\s+/); if (parts.length > 1) { const wl = parseFloat(parts[1]); if (!isNaN(wl)) wavelength = wl; } } if (upperLine.startsWith('*RAS_INT_START')) { inDataSection = true; continue; } if (upperLine.startsWith('*RAS_INT_END')) break; if (inDataSection) { const parts = line.trim().split(/[\s,]+/); if (parts.length >= 2) { const x = parseFloat(parts[0]); const y = parseFloat(parts[1]); if (!isNaN(x) && !isNaN(y)) { tth.push(x); intensity.push(y); } } } } if (tth.length === 0) throw new Error("No data found in RAS file data section."); return { tth, intensity, wavelength }; };
        const parseGsasEsdFile = (text) => { const lines = text.trim().split(/\r?\n/); let wavelength = null; let startTth, stepSize; let dataStartIndex = -1; lines.forEach((line, index) => { const upperLine = line.toUpperCase(); if (upperLine.includes('WAVELENGTH')) { const match = line.match(/wavelength\s+([0-9.]+)/i); if (match && match[1]) wavelength = parseFloat(match[1]); } if (upperLine.startsWith('BANK')) { const parts = line.trim().split(/\s+/); /* parts[6] is the step size, so a CONST line needs 7 tokens, not 6. With >= 6 a 6-token line made stepSize = parseFloat(undefined) = NaN, and the "=== undefined" guard below let NaN through, turning every 2-theta into start + i*NaN. */ if (parts.length >= 7 && parts[4].toUpperCase() === 'CONST') { const s0 = parseFloat(parts[5]) / 100.0; const ds = parseFloat(parts[6]) / 100.0; /* Reject non-numeric or zero/negative steps here rather than emitting a NaN/constant 2-theta axis downstream. */ if (isFinite(s0) && isFinite(ds) && ds > 0) { startTth = s0; stepSize = ds; dataStartIndex = index + 1; } } } }); if (!isFinite(startTth) || !isFinite(stepSize)) throw new Error("GSAS Parse Error: Could not find a valid 'BANK' line with CONST scan parameters."); if (dataStartIndex !== -1 && lines[dataStartIndex]?.toUpperCase().includes('STD')) dataStartIndex++; if (dataStartIndex === -1 || dataStartIndex >= lines.length) throw new Error("GSAS Parse Error: Found scan parameters but no subsequent data lines."); const intensity = []; for (let i = dataStartIndex; i < lines.length; i++) { const parts = lines[i].trim().split(/\s+/); for (let j = 1; j < parts.length; j += 2) { const val = parseFloat(parts[j]); if (!isNaN(val)) intensity.push(val); } } if (intensity.length === 0) throw new Error("GSAS Parse Error: No intensity data could be parsed."); const tth = Array.from({ length: intensity.length }, (_, i) => startTth + i * stepSize); return { tth, intensity, wavelength }; };
        
        const parseGsasXraFile = (text) => {
    const lines = text.trim().split(/\r?\n/);
    let wavelength = null;
    let startTth, stepSize;
    let dataStartIndex = -1;
    lines.forEach((line, index) => {
        const upperLine = line.toUpperCase();
        if (upperLine.includes('WAVELENGTH')) {
            const match = line.match(/wavelength\s+([0-9.]+)/i);
            if (match && match[1]) wavelength = parseFloat(match[1]);
        }
        if (upperLine.startsWith('BANK')) {
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 7 && parts[4].toUpperCase() === 'CONST') {
                const s0 = parseFloat(parts[5]) / 100.0;
                const ds = parseFloat(parts[6]) / 100.0;
                // A non-numeric token yields NaN, which the old "=== undefined"
                // guard below did not catch. Validate here instead.
                if (isFinite(s0) && isFinite(ds) && ds > 0) {
                    startTth = s0;
                    stepSize = ds;
                    dataStartIndex = index + 1;
                }
            }
        }
    });

    if (!isFinite(startTth) || !isFinite(stepSize)) throw new Error("GSAS XRA Parse Error: Could not find a valid 'BANK' line with CONST scan parameters.");
    if (dataStartIndex === -1 || dataStartIndex >= lines.length) throw new Error("GSAS XRA Parse Error: Found scan parameters but no subsequent data lines.");

    const intensity = [];
    for (let i = dataStartIndex; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        const parts = lines[i].trim().split(/\s+/);
        for (let j = 0; j < parts.length; j++) {
            const val = parseFloat(parts[j]);
            if (!isNaN(val)) intensity.push(val);
        }
    }
    if (intensity.length === 0) throw new Error("GSAS XRA Parse Error: No intensity data could be parsed.");
    const tth = Array.from({ length: intensity.length }, (_, i) => startTth + i * stepSize);
    return { tth, intensity, wavelength };
};
        
        const parseUxdFile = (text) => { const lines = text.trim().split(/\r?\n/); const intensity = []; let startTth, stepSize, wavelength; let inDataSection = false; for (const line of lines) { const trimmedLine = line.trim(); if (inDataSection) { const parts = trimmedLine.split(/\s+/); parts.forEach(part => { const val = parseFloat(part); if (!isNaN(val)) intensity.push(val); }); } else { if (trimmedLine.toUpperCase().startsWith('_START=')) startTth = parseFloat(trimmedLine.substring(7)); else if (trimmedLine.toUpperCase().startsWith('_STEPSIZE=')) stepSize = parseFloat(trimmedLine.substring(10)); else if (trimmedLine.toUpperCase().startsWith('_WL1=')) wavelength = parseFloat(trimmedLine.substring(5)); else if (trimmedLine.toUpperCase() === '_COUNTS') inDataSection = true; } } if (startTth === undefined || stepSize === undefined) throw new Error("Could not find _START and _STEPSIZE in UXD file."); if (intensity.length === 0) throw new Error("No intensity data found after _COUNTS in UXD file."); const tth = Array.from({ length: intensity.length }, (_, i) => startTth + i * stepSize); return { tth, intensity, wavelength }; };
        
        const parsePhilipsUdfFile = (text) => {
    const lines = text.trim().split(/\r?\n/);
    const isRawScan = lines.some(l => l.trim().toUpperCase() === 'RAWSCAN');

    if (isRawScan) {
        let startTth, endTth, stepSize, wavelength = null;
        let inDataSection = false;
        const intensity = [];

        for (const line of lines) {
            const trimmedLine = line.trim();
            if (!inDataSection) {
                const upper = trimmedLine.toUpperCase();
                if (upper === 'RAWSCAN') { inDataSection = true; continue; }
                const parts = trimmedLine.split(',').map(p => p.trim());
                const key = parts[0].toUpperCase();
                if (key === 'DATAANGLERANGE') {
                    startTth = parseFloat(parts[1]);
                    endTth = parseFloat(parts[2]);
                } else if (key === 'SCANSTEPSIZE') {
                    stepSize = parseFloat(parts[1]);
                } else if (key === 'LABDAALPHA1') {
                    wavelength = parseFloat(parts[1]);
                }
            } else {
                trimmedLine.split(',').forEach(part => {
                    const val = parseFloat(part.trim());
                    if (!isNaN(val)) intensity.push(val);
                });
            }
        }

        if (intensity.length === 0) throw new Error("No intensity data found after RawScan in UDF file.");
        if (startTth === undefined || stepSize === undefined) throw new Error("Could not find DataAngleRange/ScanStepSize in UDF file.");

        const tth = Array.from({ length: intensity.length }, (_, i) => startTth + i * stepSize);
        return { tth, intensity, wavelength };
    }

    // Legacy [DATA]-section UDF format
    const tth = [], intensity = [];
    let inDataSection = false;
    let wavelength = null;
    for (const line of lines) {
        const trimmedLine = line.trim();
        if (trimmedLine.toUpperCase().startsWith('LAMBDA')) {
            const parts = trimmedLine.split('=');
            if (parts.length > 1) wavelength = parseFloat(parts[1]);
        }
        if (trimmedLine.toUpperCase() === '[DATA]') { inDataSection = true; continue; }
        if (trimmedLine.startsWith('[') && trimmedLine.toUpperCase() !== '[DATA]') inDataSection = false;
        if (inDataSection) {
            const parts = trimmedLine.split(/,/).map(p => p.trim());
            if (parts.length >= 2) {
                const x = parseFloat(parts[0]);
                const y = parseFloat(parts[1]);
                if (!isNaN(x) && !isNaN(y)) { tth.push(x); intensity.push(y); }
            }
        }
    }
    if (tth.length === 0) throw new Error("No [Data] section found in UDF file.");
    return { tth, intensity, wavelength };
};

        const parsePdCifFile = (text) => {
            const lines = text.trim().split(/\r?\n/);
            const tth = [], intensity = [];
            let inLoop = false;
            let loopHeaders = [];
            let wavelengths = [];

            for (let i = 0; i < lines.length; i++) {
                let line = lines[i].trim();
                if (line === '' || line.startsWith('#')) continue;

                // Single line wavelength (ensure we don't accidentally match a loop id/wt tag)
                if (line.includes('_diffrn_radiation_wavelength') && !line.includes('_id') && !line.includes('_wt')) {
                    const match = line.match(/_diffrn_radiation_wavelength\s+([0-9.]+)/);
                    if (match && match[1]) {
                        wavelengths.push(parseFloat(match[1]));
                    }
                }

                if (line === 'loop_') {
                    inLoop = true;
                    loopHeaders = [];
                    continue;
                }

                if (inLoop) {
                    if (line.startsWith('_')) {
                        loopHeaders.push(line);
                    } else {
                        const tthIndex = loopHeaders.findIndex(h => h === '_pd_meas_2theta_scan' || h === '_pd_proc_2theta_corrected');
                        const intIndex = loopHeaders.findIndex(h => h.includes('_pd_meas_intensity') || h.includes('_pd_proc_intensity'));
                        const wlIndex = loopHeaders.findIndex(h => h === '_diffrn_radiation_wavelength');

                        if (tthIndex !== -1 && intIndex !== -1) {
                            const parts = line.split(/[\s,]+/);
                            if (parts.length >= Math.max(tthIndex, intIndex) + 1) {
                                const x = parseFloat(parts[tthIndex]);
                                const y = parseFloat(parts[intIndex]);
                                if (!isNaN(x) && !isNaN(y)) {
                                    tth.push(x);
                                    intensity.push(y);
                                }
                            }
                        } else if (wlIndex !== -1) {
                            // Extract wavelength if it was declared in a loop
                            const parts = line.split(/[\s,]+/);
                            if (parts.length > wlIndex) {
                                const val = parseFloat(parts[wlIndex]);
                                if (!isNaN(val)) wavelengths.push(val);
                            }
                        } else {
                            inLoop = false;
                        }
                    }
                }
            }
            if (tth.length === 0) throw new Error("Could not find _pd_meas_2theta_scan and intensity data in pdCIF file.");
            
            let presetMatch = null;
            let wavelength = null;

            if (wavelengths.length >= 2) {
                // Doublet found. Match the shortest wavelength (ka1) to our presets.
                const ka1 = Math.min(...wavelengths);
                for (const element of Object.keys(WAVELENGTH_PRESETS)) {
                    if (element === 'custom') continue;
                    if (Math.abs(WAVELENGTH_PRESETS[element].ka1 - ka1) < 0.005) {
                        presetMatch = element + '_avg';
                        break;
                    }
                }
            } else if (wavelengths.length === 1) {
                // Monochromatic found. Check if it perfectly matches an average or a specific Ka1.
                const wl = wavelengths[0];
                for (const element of Object.keys(WAVELENGTH_PRESETS)) {
                    if (element === 'custom') continue;
                    if (Math.abs(WAVELENGTH_PRESETS[element].ka_avg - wl) < 0.005) {
                        presetMatch = element + '_avg';
                        break;
                    }
                    if (Math.abs(WAVELENGTH_PRESETS[element].ka1 - wl) < 0.005) {
                        presetMatch = element + '_ka1';
                        break;
                    }
                }
                wavelength = wl;
            }

            return { tth, intensity, wavelength, presetMatch };
        };

    ui.fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const MAX_FILE_SIZE_MB = 50;
        if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            showStatus(`Error: File is too large (>${MAX_FILE_SIZE_MB} MB).`, "error");
            ui.fileInputLabel.classList.add('error');
            e.target.value = null; // Clear the input
            return;
        }

        renderFileChip(file.name, file.size);
        loadedFileName = file.name;

        resetSessionState();

        const text = await file.text();
        let parsed;
        const _perfParseEnd = perfStart('parseFile');
        try {
            parsed = detectAndParseFile(file.name, text);
        } catch (error) {
            showStatus(`Error parsing file: ${error.message}`, "error");
            console.error(error);
            ui.fileInputLabel.classList.add('error');
            return;
        }
        _perfParseEnd(`(${parsed && parsed.tth ? parsed.tth.length : 0} points, ${file.name})`);
        
        ui.fileInputLabel.classList.remove('error');

        if (!parsed || !parsed.tth || parsed.tth.length === 0) {
            showStatus("Could not read data from file.", "error");
            return;
        }


        // Clean some bad points
        let tth_in = parsed.tth;
        let int_in = parsed.intensity;
        
        if (tth_in.length !== int_in.length) {
            showStatus(`Error: Data file is corrupt. Mismatched column lengths.`, "error");
            return;
        }

        let tth_out = [];
        let int_out = [];
        let stoppedAtIndex = -1;

        for (let i = 0; i < tth_in.length; i++) {
            const tth = tth_in[i];
            const intensity = int_in[i];

            // Check for non-numeric or infinite values
            if (!isFinite(tth) || !isFinite(intensity)) {
                stoppedAtIndex = i;
                break; // Stop at the very first bad point
            }
            
            // Only add if it's a valid, finite point
            tth_out.push(tth);
            int_out.push(Math.max(0, intensity)); // Clamp negative intensities to 0
        }
        
        const originalCount = tth_in.length;
        const removedCount = originalCount - tth_out.length;

        // Write a warning if we trimmed any points
        if (removedCount > 0) {
            const message = `Info: Data read stopped at first invalid (NaN/Inf) point. ${removedCount} points trimmed.`;
            console.warn(message, `Stopped at index ${stoppedAtIndex}`);
            showStatus(message, 'info', 4000);
        }

        if (tth_out.length === 0) {
             showStatus("Error: No valid data could be read from file.", "error");
            return;
        }

        // --some hack... trim trailing zeros, need to rewrite this part... it's a bit shady
        let lastNonZeroIndex = tth_out.length - 1;
        
        // Search backwards from the end
        while (lastNonZeroIndex >= 0) {
            // Use a small epsilon to treat very small numbers as zero
            if (int_out[lastNonZeroIndex] > 1e-9) { 
                break; // Found the last real data point
            }
            lastNonZeroIndex--;
        }

        const trimmedCount = tth_out.length - (lastNonZeroIndex + 1);

        if (trimmedCount > 0) {
            // Trim the arrays by slicing
            tth_out = tth_out.slice(0, lastNonZeroIndex + 1);
            int_out = int_out.slice(0, lastNonZeroIndex + 1);
            
            const message = `Info: Auto-trimmed ${trimmedCount} trailing zero-intensity points.`;
            console.warn(message);
            showStatus(message, 'info', 4000);
        }
        
        if (tth_out.length === 0) {
             showStatus("Error: No non-zero intensity data found.", "error");
             return;
        }
    

        // Handle wavelength from file, if any.. problem here for some xrdml ?
        if (parsed.presetMatch) {
            // A recognized doublet or K-alpha average was detected
            ui.wavelengthPreset.value = parsed.presetMatch;
            handleWavelengthPresetChange({ onLoad: true });
            showStatus(`Loaded preset wavelength from file: ${parsed.presetMatch.replace('_', ' ')}`, 'info');
        } else if (parsed.wavelength) {
            // This is a monochromatic, known wavelength from the file that does not match a preset
            ui.wavelengthPreset.value = 'custom';
            ui.wavelength.value = parsed.wavelength.toFixed(5);
            ui.stripKa2Checkbox.checked = false;
            ui.stripKa2Checkbox.disabled = true;
            showStatus(`Loaded custom wavelength from file: ${parsed.wavelength.toFixed(5)} Å`, 'info');
        } else {
            // File has no wavelength, default to Cu and ENABLE Ka2 stripping by default
            ui.wavelengthPreset.value = 'Cu_avg'; 
            handleWavelengthPresetChange({ onLoad: true });
        }
        
        if (tth_out.length > 1 && tth_out[0] > tth_out[tth_out.length - 1]) {
            console.warn("Descending 2θ scan detected. Sorting ascending to prevent binary search failure...");
            const combined = tth_out.map((t, i) => ({ t, int: int_out[i] })).sort((a, b) => a.t - b.t);
            tth_out = combined.map(item => item.t);
            int_out = combined.map(item => item.int);
        }


        // Store full dataset
        fullExperimentalData = { tth: tth_out, intensity: int_out };
        updateWorkingData(); // This will apply stripping if needed

        // Data is now loaded: the file can be re-exported.
        if (ui.saveAsButton) ui.saveAsButton.disabled = false;


        // Hide placeholder, show chart, custom cursor (depuis sept 2025)
        ui.placeholder.style.display = 'none';
        ui.resultsContainer.style.display = 'flex';
        
        // Initialize chart if not already created
        if (!xrdChart) {
            initializeChart(); // Use the dedicated function
        } else {
            xrdChart.resetZoom('none'); // Reset zoom on new file
        }
        setExperimentalTrace(true);

        // Enable peak controls
        ui.peakControls.classList.remove('hidden');
        ui.indexingControls.classList.remove('hidden');

              
        setupTthSliders();
        findPeaks();
        
    });

    // sliders
    const setupTthSliders = () => {
        // Now uses workingExperimentalData, depuis 22 oct 2025, Ka2 stripping, le 15 nov vanCittert
        if (workingExperimentalData.tth.length === 0) return;
        const min = workingExperimentalData.tth[0];
        const max = workingExperimentalData.tth[workingExperimentalData.tth.length - 1];
        const step = (max - min) / 2000;
        [ui.tthMinSlider, ui.tthMaxSlider].forEach(el => { el.disabled = false; Object.assign(el, { min, max, step }); });
        const initialMin = Math.floor(min);
        const initialMax = Math.ceil(max);
        ui.tthMinSlider.value = initialMin;
        ui.tthMaxSlider.value = initialMax;
        ui.tthMinValue.textContent = initialMin.toFixed(2);
        ui.tthMaxValue.textContent = initialMax.toFixed(2);
        updatePlotRange(true);
    };

    const updatePlotRange = (updateYScale = false) => {
        if(!xrdChart) return;
        const min = parseFloat(ui.tthMinSlider.value);
        const max = parseFloat(ui.tthMaxSlider.value);
        // The sliders are always in 2-theta, the axis may not be. Map both ends
        // and take the numeric extremes, because d runs the other way round.
        const xa = xF(min), xb = xF(max);
        xrdChart.options.scales.x.min = Math.min(xa, xb);
        xrdChart.options.scales.x.max = Math.max(xa, xb);
        if (updateYScale) {
            const visibleIntensities = workingExperimentalData.intensity.filter((_, index) => {
                const tth = workingExperimentalData.tth[index];
                return tth >= min && tth <= max;
            });
            const yb = yBoundsFor(visibleIntensities.length ? visibleIntensities : workingExperimentalData.intensity);
            xrdChart.options.scales.y.min = yb.min;
            xrdChart.options.scales.y.max = yb.max;
        }
        xrdChart.update('none');
        updateAllMarkers();
    };


ui.tthMinSlider.addEventListener('input', () => {
        let minVal = parseFloat(ui.tthMinSlider.value);
        let maxVal = parseFloat(ui.tthMaxSlider.value);
        if (minVal >= maxVal) { minVal = maxVal - parseFloat(ui.tthMinSlider.step); ui.tthMinSlider.value = minVal; }
        ui.tthMinValue.textContent = minVal.toFixed(2);
        updatePlotRange();
        debouncedFindPeaks(); // 
    });
     ui.tthMaxSlider.addEventListener('input', () => {
        let minVal = parseFloat(ui.tthMinSlider.value);
        let maxVal = parseFloat(ui.tthMaxSlider.value);
        if (maxVal <= minVal) { maxVal = minVal + parseFloat(ui.tthMaxSlider.step); ui.tthMaxSlider.value = maxVal; }
        ui.tthMaxValue.textContent = maxVal.toFixed(2);
        updatePlotRange();
        debouncedFindPeaks(); // v114
    });

    ui.ballRadiusSlider.addEventListener('input', () => { ui.ballRadiusValue.textContent = ui.ballRadiusSlider.value; debouncedFindPeaks(); });
    ui.smoothingWidthSlider.addEventListener('input', () => { ui.smoothingWidthValue.textContent = ui.smoothingWidthSlider.value; debouncedFindPeaks(); });
    ui.peakThresholdSlider.addEventListener('input', () => { const value = logSliderToValue(parseFloat(ui.peakThresholdSlider.value)); ui.peakThresholdValue.textContent = value.toFixed(1); debouncedFindPeaks(); });

    const rollingBallBackground = (y, radius, smoothingWidth) => {
        const n = y.length;
        if (n === 0 || radius <= 0) return new Array(n).fill(0);
        let smoothed_y = y;
        if (smoothingWidth > 1) {
            smoothed_y = new Array(n);
            const halfWidth = Math.floor(smoothingWidth / 2);
            for (let i = 0; i < n; i++) {
                const start = Math.max(0, i - halfWidth);
                const end = Math.min(n, i + halfWidth + 1);
                let sum = 0;
                for (let j = start; j < end; j++) sum += y[j];
                smoothed_y[i] = sum / (end - start);
            }
        }
        const eroded = new Array(n);
        for (let i = 0; i < n; i++) {
            const start = Math.max(0, i - radius);
            const end = Math.min(n, i + radius + 1);
            let min = Infinity;
            for (let j = start; j < end; j++) if (smoothed_y[j] < min) min = smoothed_y[j];
            eroded[i] = min;
        }
        const background = new Array(n);
        for (let i = 0; i < n; i++) {
            const start = Math.max(0, i - radius);
            const end = Math.min(n, i + radius + 1);
            let max = -Infinity;
            for (let j = start; j < end; j++) if (eroded[j] > max) max = eroded[j];
            background[i] = max;
        }
        return background;
    };

    const savitzkyGolay = (data, windowSize = 9, polyOrder = 2) => {
        const n = data.length; if (n === 0) return [];
        windowSize = Math.max(3, windowSize); if (windowSize % 2 === 0) windowSize += 1; windowSize = Math.min(windowSize, n);
        const halfWindow = Math.floor(windowSize / 2);
        const result = new Array(n);
        const coefficients = (windowSize === 9 && polyOrder === 2) ? [-0.0909, 0.0606, 0.1687, 0.2333, 0.2545, 0.2333, 0.1687, 0.0606, -0.0909] : (() => { const weights = []; for (let i = -halfWindow; i <= halfWindow; i++) weights.push(1 - Math.abs(i) / (halfWindow + 1)); const sum = weights.reduce((a, b) => a + b, 0); return weights.map(w => w / sum); })();
        for (let i = 0; i < n; i++) {
            let smoothedValue = 0;
            for (let j = -halfWindow; j <= halfWindow; j++) {
                let idx = i + j;
                if (idx < 0) idx = Math.abs(idx);
                else if (idx >= n) idx = n - 1 - (idx - (n - 1));
                smoothedValue += data[idx] * coefficients[j + halfWindow];
            }
            result[i] = smoothedValue;
        }
        return result;
    };



    // In-place quickselect: returns the k-th smallest element of `arr` (0-based),
    // reordering `arr` as a side effect. Average O(n) vs the O(n log n) of a full
    // sort, which matters because findPeaks (and its median-of-absolute-deviations
    // noise estimate) runs on every slider drag over the full-resolution scan.
    // Result for a given k is identical to arr.slice().sort()[k].
    const quickselect = (arr, k) => {
        let lo = 0, hi = arr.length - 1;
        while (lo < hi) {
            // Median-of-three pivot to avoid O(n^2) on sorted/near-sorted input
            // (background-corrected intensities are far from random).
            const mid = (lo + hi) >> 1;
            const a = arr[lo], b = arr[mid], c = arr[hi];
            const pivot = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
            let i = lo, j = hi;
            while (i <= j) {
                while (arr[i] < pivot) i++;
                while (arr[j] > pivot) j--;
                if (i <= j) { const t = arr[i]; arr[i] = arr[j]; arr[j] = t; i++; j--; }
            }
            if (k <= j) hi = j;
            else if (k >= i) lo = i;
            else break;
        }
        return arr[k];
    };

    function findPeaks() {
        // Now uses workingExperimentalData
        if (!workingExperimentalData || !workingExperimentalData.intensity || workingExperimentalData.intensity.length < 5) return;
        const _perfEnd = perfStart('findPeaks');
        
        const { intensity, tth } = workingExperimentalData; const n = tth.length;
        const minTth = parseFloat(ui.tthMinSlider.value) || tth[0];
        const maxTth = parseFloat(ui.tthMaxSlider.value) || tth[n - 1];
        const minHeightPercent = logSliderToValue(parseFloat(ui.peakThresholdSlider.value)) || 2;
        const ballRadius = parseInt(ui.ballRadiusSlider.value, 10);
        const smoothingWidth = parseInt(ui.smoothingWidthSlider.value, 10);
        const background = rollingBallBackground(intensity, ballRadius, smoothingWidth);
        const backgroundCorrected = intensity.map((y, i) => Math.max(0, y - background[i]));
        const windowSize = Math.max(5, Math.min(11, Math.floor(n / 100)));
        const smoothed = savitzkyGolay(backgroundCorrected, windowSize, 2);

        // --- Range-restricted noise/threshold statistics ---
        // changed to search all points on 13th july 2026
        let rangeStart = 0;
        while (rangeStart < n && tth[rangeStart] < minTth) rangeStart++;
        let rangeEnd = n; // exclusive
        while (rangeEnd > rangeStart && tth[rangeEnd - 1] > maxTth) rangeEnd--;
        const rangeView = backgroundCorrected.slice(rangeStart, rangeEnd);

        const maxCorrectedIntensity = (rangeView.length > 0 ? maxOfArray(rangeView) : maxOfArray(backgroundCorrected)) || 1;
        const minAbsoluteHeight = (minHeightPercent / 100) * maxCorrectedIntensity;
        
        const calculateNoiseLevel = (data) => {
    const n_s = data.length;
    if (n_s < 10) return 0;
    // MAD-based robust noise estimate. Two O(n) quickselects on scratch copies
    // replace the two full O(n log n) sorts the original did; the median index
    // (floor(len/2)) is unchanged, so the result is bit-identical.
    const work = Float64Array.from(data);
    const midIdx = Math.floor(n_s / 2);
    const median = quickselect(work, midIdx);
    for (let i = 0; i < n_s; i++) work[i] = Math.abs(work[i] - median);
    const mad = quickselect(work, midIdx);
    return mad * 1.4826;
};

        const noiseSrc = rangeView.length >= 10 ? rangeView : backgroundCorrected;
        const adaptiveThreshold = Math.max(minAbsoluteHeight, calculateNoiseLevel(noiseSrc) * 3);
        const localMaxIndices = [];
        // Restrict the local-max scan to the slider range so out-of-range
        // structure cannot produce candidates that get filtered out later
        // (the only effect of those is to perturb plateau detection at
        // the boundaries).
        const scanStart = Math.max(1, rangeStart);
        const scanEnd   = Math.min(n - 1, rangeEnd);
        for (let i = scanStart; i < scanEnd; i++) {
            const current = smoothed[i]; if (current < adaptiveThreshold) continue;
            if (current > smoothed[i - 1] && current > smoothed[i + 1]) localMaxIndices.push(i);
            else if (current === smoothed[i + 1] && current > smoothed[i - 1]) { let plateauEnd = i + 1; while (plateauEnd < n - 1 && Math.abs(smoothed[plateauEnd] - current) < maxCorrectedIntensity * 0.001) plateauEnd++; if (plateauEnd < n && smoothed[plateauEnd] < current) localMaxIndices.push(Math.round((i + plateauEnd - 1) / 2)); i = plateauEnd - 1; }
        }
        const candidates = localMaxIndices.filter(idx => tth[idx] >= minTth && tth[idx] <= maxTth && backgroundCorrected[idx] >= adaptiveThreshold)
            .map(idx => ({ idx, tth: tth[idx], height: smoothed[idx], backgroundCorrectedHeight: backgroundCorrected[idx] }));
        
        // ref 5 ou 3 savitzky
        const refinedPeaks = [];
        for (const peak of candidates) {
            const { idx } = peak; let refinedTth = peak.tth;

            // Calculate a robust average step size around the peak
            const avgStep = (idx > 0 && idx < n - 1) 
                ? (tth[idx+1] - tth[idx-1]) / 2.0 
                : (idx > 0 ? tth[idx] - tth[idx-1] : (idx < n-1 ? tth[idx+1] - tth[idx] : 0.01));

            // Try 5-point parabola first (more accurate)
            if (idx > 1 && idx < n - 2) { 
                const y1 = smoothed[idx - 2];
                const y2 = smoothed[idx - 1];
                const y3 = smoothed[idx];
                const y4 = smoothed[idx + 1];
                const y5 = smoothed[idx + 2];

                // 5-point least-squares quadratic fit (Savitzky-Golay coefficients)
                // Parabola y = ax^2 + bx + c, centered at x=0 (idx)
                // Least-squares quadratic through 5 equally spaced points, x = -2..2.
                // Sum(x^2) = 10, Sum(x^4) = 34, N = 5 give a = (Sum(x^2 y) - 2 Sum(y)) / 14.
                // The divisor was 7, i.e. a came out twice too large, which halved every
                // vertex offset delta = -b / (2a). Sub-step peak positions were therefore
                // systematically pulled only half way to the true maximum.
                const a = (2*y1 - y2 - 2*y3 - y4 + 2*y5) / 14.0;
                const b = (-2*y1 - y2 + 0*y3 + y4 + 2*y5) / 10.0;
                
                // Check for valid maximum (downward parabola, a < 0)
                if (a < -1e-10) { 
                    const delta = -b / (2 * a); // Vertex x = -b / (2a)
                    
                    // delta should be within the 5-point window
                    if (Math.abs(delta) < 2.0) { 
                        refinedTth = tth[idx] + delta * avgStep;
                    }
                }

                
            //  Use 3-point fit if 5-point fails or is near edge
            } else if (idx > 0 && idx < n - 1) { 
                const y1 = smoothed[idx - 1], y2 = smoothed[idx], y3 = smoothed[idx + 1]; 
                const denominator = (y1 - 2 * y2 + y3); 
                
if (denominator < -1e-10) { 
                    // Three-point parabolic interpolation: delta = (y1 - y3) / (2 (y1 - 2 y2 + y3)).
                    // The factor 1/2 was missing, so this branch overshot by exactly 2x -
                    // the opposite sign of error to the 5-point branch above, which is why
                    // edge peaks and interior peaks disagreed about where a maximum was.
                    const delta = 0.5 * (y1 - y3) / denominator; 
                    if (Math.abs(delta) < 1.0) { 
                        refinedTth = tth[idx] + delta * avgStep; 
                    } 
                } 
            }
            
            if (refinedTth <= 1e-4) {
                refinedTth = 1e-4; // Prevent tth=0 or negative tth
            }


            refinedPeaks.push({ ...peak, tth: refinedTth });
        }
        

        // Continue 
        refinedPeaks.sort((a, b) => a.tth - b.tth);
        const finalPeaks = [];

        // Angle-adaptive merge threshold.
        // Two refined candidates closer than this are treated as one peak
        // (keeping the higher one).
        //
        // Goals:
        //   - Always at least 0.02° (instrumental resolution at low 2θ)
        //   - Grow gently with 2θ to swallow small numerical duplicates from
        //     the parabolic refinement at high angle
        //   - When the data is an unstripped Kα-doublet pattern, never let
        //     the threshold reach the Kα1/Kα2 split — otherwise a real
        //     Kα2 ghost would be merged into its Kα1 parent and we'd lose
        //     the soft-violation signal in space-group analysis.
        const lambda = parseFloat(ui.wavelength.value) || 1.54184;
        const stripOn = !!ui.stripKa2Checkbox?.checked;
        const presetForMerge = stripOn ? null : getActiveKa2Preset();
        const FLOOR = 0.02; // never tighter than this
        const CEIL  = 0.10; // never looser than this
        const mergeThresholdAt = (tthDeg) => {
            const baseScale = 0.02 + 0.001 * tthDeg; // gentle linear ramp
            let mt = Math.min(CEIL, Math.max(FLOOR, baseScale));
            if (presetForMerge) {
                const tth2 = expectedKa2TthDeg(tthDeg, presetForMerge);
                if (isFinite(tth2)) {
                    const split = Math.abs(tth2 - tthDeg);
                    // Cap by 45% of the doublet split so a Kα1+Kα2 pair
                    // is preserved as TWO peaks. But never drop below the
                    // floor — at low 2θ where the doublet is unresolved
                    // the cap would go below instrumental resolution.
                    mt = Math.max(FLOOR, Math.min(mt, 0.45 * split));
                }
            }
            return mt;
        };

        for (const peak of refinedPeaks) {
            if (finalPeaks.length === 0) { finalPeaks.push(peak); continue; }
            const last = finalPeaks[finalPeaks.length - 1];
            const mt = mergeThresholdAt(last.tth);
            if (Math.abs(peak.tth - last.tth) >= mt) {
                finalPeaks.push(peak);
            } else if (peak.height > last.height) {
                finalPeaks[finalPeaks.length - 1] = peak;
            }
        }

        // Build initial pickedPeaks with the user's (default) wavelength.
        // recalculatePeakValues() below will flag Ka2-suspects AND re-derive
        // d/q for parents of tagged ghosts using λ_Ka1 (since we know those
        // peaks are physically Ka1 lines).
       // Include height so the space group analyzer can filter noise
pickedPeaks = finalPeaks.map(p => ({ tth: p.tth, d: 0, q: 0, height: p.height }));

        recalculatePeakValues();

        updatePeakTable(); updateStartIndexingButtonState();
        _perfEnd(`(${pickedPeaks.length} peaks, n=${n})`);
    };


    const recalculatePeakValues = () => {
        const lambda = parseFloat(ui.wavelength.value);
        // First flag the Ka2-children (i.e., who is parent of a tagged ghost),
        // because the d/q computation for parents needs to know that.
        flagKa2SuspectPeaks();
        const preset = getActiveKa2Preset();
        const lambdaKa1 = preset ? preset.ka1 : null;
        pickedPeaks.forEach(peak => {
            // Parents of tagged Ka2 ghosts are provably Ka1 lines, so their
            // d-spacing should be derived from λ_Ka1, not the user's main
            // (Ka-avg) wavelength. All other peaks use the main λ.
            const lam = (peak.hasKa2Child && lambdaKa1) ? lambdaKa1 : lambda;
            peak.d = lam / (2 * Math.sin(peak.tth * Math.PI / 360));
            peak.q = 1 / (peak.d * peak.d);
            peak.lambdaUsed = lam; // for debugging / display traceability
        });
    };

    const updatePeakTable = () => {
        // Clamp the selected index in case the peak list shrank.
        if (selectedPeakIndex !== null && selectedPeakIndex >= pickedPeaks.length) {
            selectedPeakIndex = null;
        }
        ui.peakListBody.innerHTML = '';
        pickedPeaks.forEach((peak, index) => {
            const row = document.createElement('tr');
            const isSuspect = !!peak.ka2Suspect;
            const isParent = !!peak.hasKa2Child;
            if (isSuspect) {
                row.classList.add('ka2-suspect');
                const parentTth = peak.ka2ParentIdx != null && pickedPeaks[peak.ka2ParentIdx]
                    ? pickedPeaks[peak.ka2ParentIdx].tth.toFixed(3)
                    : '?';
                row.title =
                    `Possible Kα₂ companion of the peak at 2θ = ${parentTth}°.\n` +
                    `Excluded from indexing and figure of merit.\n` +
                    `Marked SOFT in space-group analysis.\n` +
                    `Right-click to mark as a real (independent) peak.`;
            } else if (isParent) {
                // Parent of a tagged Ka2 ghost: provably a Ka1 line, d uses λ_Ka1
                row.title =
                    `Kα₁ line — its Kα₂ ghost was identified.\n` +
                    `d-spacing computed with λ_Kα₁.`;
            }
            if (index === selectedPeakIndex) {
                row.classList.add('peak-row-selected');
            }
            const indexLabel = `${index + 1}${isParent ? '*' : ''}`;
            row.innerHTML = `<td>${indexLabel}</td><td><input type="number" class="peak-tth-input" value="${peak.tth.toFixed(4)}" data-index="${index}" step="0.0001"></td><td><input type="number" class="peak-d-input" value="${peak.d.toFixed(5)}" data-index="${index}" disabled></td><td><button class="delete-peak-btn" data-index="${index}">X</button></td>`;
            ui.peakListBody.appendChild(row);
        });
        ui.peakTableContainer.classList.toggle('hidden', pickedPeaks.length === 0);
        updateKa2Banner();
        updateAllMarkers();
    };
    


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
        const needed = 4 - pickedPeaks.length;
        
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
    ui.systemCheckboxes.forEach(cb => cb.addEventListener('change', updateStartIndexingButtonState));
    
    // Also run on init
    updateStartIndexingButtonState();



    ui.peakListBody.addEventListener('change', (e) => {
        if (e.target.classList.contains('peak-tth-input')) {
            const index = parseInt(e.target.dataset.index, 10); 
            let tth = parseFloat(e.target.value); 
            
            // Explicitly check for NaN or non-finite numbers before clamping
            if (!isFinite(tth) || tth <= 1e-4) {
                // Fallback to the existing value if invalid, or a safe default
                tth = pickedPeaks[index]?.tth || 1.0; 
            }
            
            e.target.value = tth.toFixed(4); 
            const wasUserConfirmed = !!pickedPeaks[index]?.userConfirmedReal;
            const existingHeight = pickedPeaks[index]?.height; 
            
            pickedPeaks[index] = { tth, d: 0, q: 0, height: existingHeight, userConfirmedReal: wasUserConfirmed };
            pickedPeaks.sort((a, b) => a.tth - b.tth);
            recalculatePeakValues();
            updatePeakTable();
            updateStartIndexingButtonState();
        }
    });


     ui.peakListBody.addEventListener('click', (e) => {
        if (e.target.classList.contains('delete-peak-btn')) {
            const index = parseInt(e.target.dataset.index);
            pickedPeaks.splice(index, 1);
            recalculatePeakValues(); // re-flag + recompute after deletion
            updatePeakTable();
            updateStartIndexingButtonState();
        }
    });

    // Right-click on a Ka2-suspect row → toggle "this is actually a real peak"
    ui.peakListBody.addEventListener('contextmenu', (e) => {
        const row = e.target.closest('tr.ka2-suspect');
        if (!row) return;
        e.preventDefault();
        const idx = Array.from(ui.peakListBody.children).indexOf(row);
        if (idx < 0 || !pickedPeaks[idx]) return;
        // User explicitly says: not a Ka2 ghost — keep as real.
        pickedPeaks[idx].userConfirmedReal = true;
        // Recompute flags + d/q. The previous parent of this peak may lose
        // its hasKa2Child status (and revert from λ_Ka1 to user λ for d).
        recalculatePeakValues();
        updatePeakTable();
    });

    // --- Selected-peak highlight ---
    // While the user is editing a peak's row (focus inside an input) or
    // has clicked the row, a tall translucent vertical line is drawn at
    // that peak's 2θ across the full plot height. This makes it easy to
    // identify which peak in the diffractogram corresponds to the row
    // being edited. The visual is rendered in updateAllMarkers via a
    // dedicated "Selected Peak" dataset.
    const applySelectedPeakRowClass = () => {
        const rows = ui.peakListBody.children;
        for (let i = 0; i < rows.length; i++) {
            rows[i].classList.toggle('peak-row-selected', i === selectedPeakIndex);
        }
    };
    // 'focusin' bubbles, unlike 'focus', so a single delegated listener
    // catches focus on any per-row input.
    ui.peakListBody.addEventListener('focusin', (e) => {
        const row = e.target.closest('tr');
        if (!row) return;
        const idx = Array.from(ui.peakListBody.children).indexOf(row);
        if (idx < 0 || idx === selectedPeakIndex) return;
        selectedPeakIndex = idx;
        applySelectedPeakRowClass();
        updateAllMarkers();
    });
    ui.peakListBody.addEventListener('focusout', (e) => {
        // When tabbing between inputs of the same row, focusout fires
        // before focusin on the new target. Defer the clear so we can
        // see whether focus actually left the table.
        const leavingRow = e.target.closest('tr');
        const leavingIdx = leavingRow ? Array.from(ui.peakListBody.children).indexOf(leavingRow) : -1;
        setTimeout(() => {
            const active = document.activeElement;
            if (active && ui.peakListBody.contains(active)) return; // focus still inside table
            if (selectedPeakIndex === leavingIdx) {
                selectedPeakIndex = null;
                applySelectedPeakRowClass();
                updateAllMarkers();
            }
        }, 0);
    });
    // A click on the row outside the inputs (e.g. on the index cell) also
    // toggles selection. Clicks on inputs/buttons keep their original
    // semantics — focusin handles those.
    ui.peakListBody.addEventListener('click', (e) => {
        if (e.target.closest('input, button')) return;
        const row = e.target.closest('tr');
        if (!row) return;
        const idx = Array.from(ui.peakListBody.children).indexOf(row);
        if (idx < 0) return;
        selectedPeakIndex = (selectedPeakIndex === idx) ? null : idx;
        applySelectedPeakRowClass();
        updateAllMarkers();
    });
    
    const setUIState = (indexing) => {
        isIndexing = indexing; 
        document.body.style.cursor = indexing ? 'wait' : 'default';
        
        const controlsToDisable = [ 
            ui.fileInput, ui.peakThresholdSlider, ui.tthMinSlider, ui.tthMaxSlider, 
            ui.ballRadiusSlider, ui.smoothingWidthSlider, ui.wavelength, ui.tthError, 
            ui.maxVolume, ui.impurityPeaksInput, ui.refineZeroCheckbox, 
            // ...ui.systemCheckboxes, modif nov 25
            ...ui.tabButtons, ui.wavelengthPreset, ui.stripKa2Checkbox 
        ];
        
        controlsToDisable.forEach(el => { if (el) el.disabled = indexing; });
        

        //  Manually handle checkboxes based on GPU support, if WebGPU error disable mono and tric
        ui.systemCheckboxes.forEach(cb => {
            if (indexing) {
                // When indexing starts, disable all
                cb.disabled = true; 
            } else {
                // When indexing stops, re-enable based on GPU support
                if (webGPUSupportsCompute) {
                    cb.disabled = false; // Re-enable all
                } else {
                    // Only re-enable non-GPU ones
                    
                    if (cb.value === 'monoclinic' || cb.value === 'triclinic' || cb.value === 'orthorhombic') {
                    
                        cb.disabled = true; 
                        cb.checked = false; 
                    } else {
                        cb.disabled = false;
                    }
                }
            }
        });
        

        ui.peakListBody.querySelectorAll('input, button').forEach(el => { el.disabled = indexing; });
        ui.fileInputLabel.style.pointerEvents = indexing ? 'none' : 'auto'; 
        ui.fileInputLabel.style.opacity = indexing ? '0.7' : '1';
        
        if (indexing) {
            ui.startIndexingButton.disabled = true;
            ui.reportButton.textContent = 'Stop'; 
            ui.reportButton.disabled = false;
            ui.progressBarContainer.classList.remove('hidden'); 
            ui.progressBar.style.width = '0%';
        } else {
            updateStartIndexingButtonState(); 
            ui.reportButton.textContent = 'Generate PDF Report'; 
            ui.reportButton.disabled = (solutions.length === 0);
            ui.progressBarContainer.classList.add('hidden'); 
            ui.progressBar.style.width = '0%';
            
            // Re-enable controls based on state
            if (fullExperimentalData.tth.length > 0) { 
                ui.tthMinSlider.disabled = false; 
                ui.tthMaxSlider.disabled = false; 
                ui.wavelengthPreset.disabled = false;
                if (ui.wavelengthPreset.value !== 'custom') {
                    ui.stripKa2Checkbox.disabled = false;
                }
                // Wavelength input is always editable once data is loaded.
                ui.wavelength.disabled = false;
            }
        }
    };


    const applyFinalSieve = (solutions) => {

        // Ensure solutions have strictly positive, finite volumes and figures of merit
    const validSolutions = solutions.filter(s => 
        s && 
        isFinite(s.volume) && s.volume > 0 && 
        isFinite(s.m20) && s.m20 > 0
    );

    if (validSolutions.length <= 1) return validSolutions;
    
    showStatus('Applying final sieve to results...', 'info', 2000);
    const symmetryOrder = { 'cubic': 5, 'hexagonal': 4, 'tetragonal': 4, 'orthorhombic': 3, 'monoclinic': 2, 'triclinic': 1 };
    
    // 1. Sort by Volume for the duplicate detection algorithm
    validSolutions.sort((a, b) => a.volume - b.volume);
    
    const toKeep = new Array(validSolutions.length).fill(true);
    
    for (let i = 0; i < validSolutions.length; i++) {
        if (!toKeep[i]) continue;
        for (let j = i + 1; j < validSolutions.length; j++) {
            if (!toKeep[j]) continue;
            
            //  Check Zero Shift Mode, version 16 jan 2026
            // If one solution has a refined zero shift (property exists) and the other doesn't (undefined),
            // we consider them DISTINCT models (e.g. 0-DoF shift vs 1-DoF shift).
            // We skip the comparison, keeping both.
            const hasZero_i = (validSolutions[i].zero_correction !== undefined && validSolutions[i].zero_correction !== null);
            const hasZero_j = (validSolutions[j].zero_correction !== undefined && validSolutions[j].zero_correction !== null);
            
            if (hasZero_i !== hasZero_j) {
                continue; 
            }

            // Use the filtered array for all comparisons
            const vol_i = validSolutions[i].volume; const vol_j = validSolutions[j].volume;
            
            // Break if volumes are different by more than 1% (sorted list)
            if (vol_j > vol_i * 1.01) break;
            
            const sym_i = symmetryOrder[validSolutions[i].system] || 0; 
            const sym_j = symmetryOrder[validSolutions[j].system] || 0;
            
            if (sym_i > sym_j) {
                toKeep[j] = false; // i has higher symmetry
            } else if (sym_j > sym_i) {
                toKeep[i] = false; // j has higher symmetry
                break;
            } else { 
                // Symmetries are equal
                const m20_i = validSolutions[i].m20;
                const m20_j = validSolutions[j].m20;
                
                const m20_percent_tolerance = 0.02;

                // Check if i is significantly better than j
                if (m20_i > (m20_j * (1.0 + m20_percent_tolerance))) {
                    toKeep[j] = false; // i is clearly better
                
                // Check if j is significantly better than i
                } else if (m20_j > (m20_i * (1.0 + m20_percent_tolerance))) {
                    toKeep[i] = false; // j is clearly better
                    break;
                
                } else {
                    // M(20) are "equal"
                    if (validSolutions[i].system === 'monoclinic') {
                        // Keep conventional setting for Monoclinic
                        const beta_i = validSolutions[i].beta || 90;
                        const beta_j = validSolutions[j].beta || 90;
                        const conventional_i = Math.abs(beta_i - 90);
                        const conventional_j = Math.abs(beta_j - 90);

                        if (conventional_i <= conventional_j) {
                             toKeep[j] = false; 
                        } else {
                             toKeep[i] = false; 
                             break;
                        }
                    } else {
                        // For other systems, just prune the one with lower M20 (even if slight)
                        if (m20_j > m20_i) {
                            toKeep[i] = false;
                            break;
                        } else {
                            toKeep[j] = false;
                        }
                    }
                }
            }
        }
    }
    
    let filteredSolutions = validSolutions.filter((_, index) => toKeep[index]);
    const numDiscarded = validSolutions.length - filteredSolutions.length;
    
    // Sort final list by quality (M20) instead of volume.
    // Guarded: a non-finite m20 would make the comparator return NaN and leave
    // the order arbitrary right before the slice(0, 50) below.
    const rankM20 = (x) => (x && isFinite(x.m20)) ? x.m20 : -Infinity;
    filteredSolutions.sort((a, b) => rankM20(b) - rankM20(a));

    if (numDiscarded > 0) showStatus(`Sieve discarded ${numDiscarded} redundant solution(s).`, 'success');
    return filteredSolutions.slice(0, 50);
};


const formatWithError = (value, error) => {
        if (error === undefined || error === null || !isFinite(error) || error <= 0) {
            const places = Math.abs(value) > 10 ? 3 : 4;
            return value.toFixed(places);
        }
        const errorMagnitude = Math.floor(Math.log10(error));
        const firstSigDigit = Math.floor(error / Math.pow(10, errorMagnitude));

        let decimalPlaces;
        if (firstSigDigit >= 3) {
            // Use 1 significant figure for error 
            decimalPlaces = -errorMagnitude;
        } else {
            decimalPlaces = -errorMagnitude + 1;
        }
        
        // Ensure decimalPlaces is reasonable and non-negative
        decimalPlaces = Math.max(0, Math.min(8, decimalPlaces));
        const multiplier = Math.pow(10, decimalPlaces);
        const roundedValue = (Math.round(value * multiplier) / multiplier).toFixed(decimalPlaces);
        const errorInLastDigits = Math.round(error * multiplier);
        return `${roundedValue}(${errorInLastDigits})`;
    };


//8 nov, major modif, chunks, needed if parameters changes in TASKS 2 and 3 in StartIndexing
/**
 * Creates an efficient generator for C(n, k) combinations.
 * This function is memory-efficient and yields combinations one by one
 * without storing them all in memory.
 *
 * @param {number} n - The number of items to choose from (e.g., 80 for C(80, 6)).
 * @param {number} k - The number of items to choose (e.g., 6 for C(80, 6)).
 * @returns {Generator<Uint32Array, void, void>} A generator that yields a Uint32Array.
 */
function* createCombinationGenerator(n, k) {
    // 1. Initialize the first combination: [0, 1, 2, ..., k-1]
    // We use Uint32Array because that's what the GPU buffer expects.
    const combo = new Uint32Array(k);
    for (let i = 0; i < k; i++) {
        combo[i] = i;
    }

    while (true) {
        // 2. Yield the current combination array.
        // The calling loop will copy this array's values into the GPU buffer.
        yield combo;

        // 3. Find the rightmost element (i) that can be incremented.
        let i = k - 1;
        
        // We check i >= 0.
        // The max value for combo[i] is (n - k + i).
        // We look for the first element from the right that is *not*
        // at its maximum value.
        while (i >= 0 && combo[i] === (n - k + i)) {
            i--;
        }

        // 4. If i < 0, all elements are at their max.
        // e.g., for C(80, 6), this would be [74, 75, 76, 77, 78, 79].
        // We are done.
        if (i < 0) {
            return;
        }

        // 5. Increment the element we found.
        combo[i]++;

        for (let j = i + 1; j < k; j++) {
            combo[j] = combo[j - 1] + 1;
        }
    }
}



let lastThrottleTime = 0;
    /**
     * Throttles a function call to only execute once every `delay` milliseconds.
     * @param {function} func The function to throttle.
     * @param {number} delay The delay in milliseconds.
     */
    const throttle = (func, delay) => {
        return (...args) => {
            const now = new Date().getTime();
            if (now - lastThrottleTime < delay) {
                return;
            }
            lastThrottleTime = now;
            func(...args);
        };
    };

    // This creates a throttled version of the status text update.
    // It will update at most 4 times per second (every 250ms).
    // It can now see the global 'statusTextElement'
    const throttledSetStatusText = throttle((message) => {
        if (statusTextElement) {
            statusTextElement.textContent = message;
        }
    }, 250); // 250ms delay


// In brutus.html
const startIndexing = async () => { 
    // Snapshot the current token; if abortActiveIndexing() runs before this
    // run's tail reaches finalizeIndexing(), the token will have moved on
    // and that call will recognize itself as stale (see finalizeIndexing()).
    const mySessionToken = indexingRunToken;
    const systemsToSearch = Array.from(ui.systemCheckboxes).filter(cb => cb.checked).map(cb => cb.value);
    if (systemsToSearch.length === 0) {
        showStatus("Please select at least one crystal system to search.", "error");
        return;
    }

    const needsWebGPU = systemsToSearch.includes('monoclinic') || systemsToSearch.includes('triclinic') || systemsToSearch.includes('orthorhombic');
    let webgpuEngine = null;

    if (needsWebGPU) {
        try {
            // Shared, page-lifetime engine -- not a new GPUDevice per run.
            webgpuEngine = await getWebGPUEngine();
        } catch (err) {
            console.warn("WebGPU initialization failed:", err.message);
            showStatus("WebGPU failed unexpectedly. GPU searches are disabled.", "error", 8000);
            webGPUSupportsCompute = false; 
            releaseWebGPUEngine();
            // Disable all GPU checkboxes
            [orthoCheckbox, monoCheckbox, triCheckbox].forEach(cb => {
                if (cb) {
                    cb.checked = false;
                    cb.disabled = true;
                    if (cb.parentElement) cb.parentElement.style.opacity = '0.5';
                }
            });
            // NOT setUIState(false): setUIState(true) has not run yet at this
            // point, so the old call was undoing state that was never applied --
            // it hid the progress bar, relabelled the Stop button back to
            // "Generate PDF Report" and re-enabled controls for a run that never
            // began. Just refresh the start button and bail.
            updateStartIndexingButtonState();
            return; // Stop indexing
        }
    }

    const tthMinVal = parseFloat(ui.tthMinSlider.value);
    const tthMaxVal = parseFloat(ui.tthMaxSlider.value);
    // Exclude Ka2-suspect peaks from indexing entirely. They are likely Ka2
    // ghosts of stronger Ka1 lines, so feeding them to the cell search would
    // either waste time (they don't satisfy the cell) or actively bias the
    // fit. The user can right-click a yellow row to mark it real and override
    // the exclusion if they suspect mis-tagging.
    const filteredPeaks = pickedPeaks.filter(
        p => p.tth >= tthMinVal && p.tth <= tthMaxVal && !p.ka2Suspect
    );

    if (filteredPeaks.length < 3) { 
        showStatus("Please find at least 3 peaks in the selected range.", 'error');
        return;
    }

    const webgpuSystems = [];
    const workerSystems = [];
    gpuStopSignal.stop = false;

systemsToSearch.forEach(system => {
    if (['orthorhombic', 'monoclinic', 'triclinic'].includes(system)) {
        if (webgpuEngine && webGPUSupportsCompute) {
            webgpuSystems.push(system);
        } else {
            // GPU unavailable: Do NOT push to workerSystems. 
            // Just skip it or warn.
            console.warn(`Skipping ${system} - GPU unavailable and CPU fallback disabled.`);
            showStatus(`Skipping ${system} (GPU required)`, "error", 4000);
        }
    } else {
        // Cubic, Tetragonal, Hexagonal go to CPU
        workerSystems.push(system);
    }
});


    if (needsWebGPU && webgpuSystems.length === 0) {
        ui.systemCheckboxes.forEach(cb => {
            if (['monoclinic', 'triclinic', 'orthorhombic'].includes(cb.value)) cb.checked = false;
        });
    }
    
    if (workerSystems.length === 0 && webgpuSystems.length === 0) {
        showStatus("No tasks to run. Check selections and GPU support.", "info");
        return;
    }
    
    setUIState(true); // Disable UI

    // modif le 16 janvier 2025, garder les mailles trouvées avant
    ui.tabButtons.forEach(btn => btn.classList.remove('active'));
    ui.tabPanels.forEach(panel => panel.classList.remove('active'));
    document.querySelector('.tab-btn[data-tab="solutions"]').classList.add('active');
    document.getElementById('solutions-tab-content').classList.add('active');
    
   // solutions = []; 
   // displayedSolutions = [];
    selectedSolution = null; 
    currentHklList = []; 
    activeWorkers = [];
    //foundSolutionMap.clear(); 
    updateSolutionsTable(); 
    updateAllMarkers();
    showStatus(`Indexing started...`, 'info');

    cumulativeTrials = 0;
    gpuTotalTrials = 0;
    indexingStartTime = performance.now();

  
    const baseParams = {
        peaks: filteredPeaks,
        wavelength: parseFloat(ui.wavelength.value) || 1.54184,
        tth_error: parseFloat(ui.tthError.value) || 0.04,
        max_volume: parseFloat(ui.maxVolume.value) || 2000,
        impurity_peaks: getImpurityPeaks(),
        refineZero: !!ui.refineZeroCheckbox.checked,
        fom_threshold: parseFloat(ui.gpuFomThreshold.value) || 0.8,
        max_solutions: (parseInt(ui.gpuBufferSize.value, 10) || 50) * 1000,
        gpu_peaks_count: parseInt(ui.gpuPeaksCount.value, 10) || 7
    };
    
    let currentGpuTaskIndex = 0;
    let currentWorkerTaskIndex = 0;
    const totalTasks = workerSystems.length + webgpuSystems.length;
    taskProgress = new Array(totalTasks).fill(0); 
    taskTotals = new Array(totalTasks).fill(0);

    const updateProgressBar = () => {
        if (totalTasks === 0) return;
        const totalSum = taskProgress.reduce((a, b) => a + b, 0);
        const totalPercentage = totalSum / totalTasks;
        ui.progressBar.style.width = `${Math.min(100, totalPercentage)}%`;
    };
    
    if (statusTextElement) statusTextElement.textContent = '[0%] Starting...';

    const getGpuProgressCallback = (systemName, absoluteTaskIndex) => {
        const cb = (chunkProgress, numFound) => { 
            taskProgress[absoluteTaskIndex] = chunkProgress * 100;
            updateProgressBar();
            const totalPercentage = taskProgress.reduce((a, b) => a + b, 0) / totalTasks;
            const message = `[${totalPercentage.toFixed(0)}%] GPU (${systemName}): ${numFound} candidates`;
            throttledSetStatusText(message);
        };
        // The engine calls this once, before the chunk loop, with the dispatch
        // geometry it actually resolved. checkGpuLimits() estimates the same
        // numbers ahead of time, but this is the ground truth and covers the
        // case where the basis was capped by the u32 guard after the pre-flight
        // check ran.
        cb.reportPlan = (plan) => {
            console.log(`[perf] GPU plan (${plan.system}): ${plan.totalChunks.toLocaleString()} dispatches, ` +
                        `${plan.hklsPerChunk} hkl/chunk, ${plan.numPeakCombos} peak combos`);
            if (plan.totalChunks > CHUNK_COUNT_WARN_LIMIT) {
                showStatus(
                    `${systemName}: ~${plan.totalChunks.toLocaleString()} GPU dispatches queued — this run will be slow. ` +
                    `Press Stop and lower "Peaks to Combine" to speed it up.`, 'error', 10000);
            }
        };
        return cb;
    };

    const taskPromises = [];

    // GPU 
    let qTolerancesArray, qObsArray;
    if (webgpuSystems.length > 0) {
        const { q_obs, original_indices, tth_obs_rad, peaks_sorted_by_q } = getSortedPeaks(filteredPeaks, baseParams.wavelength);
        // 32, not 20: the WGSL solvers cap the FoM loop at MAX_FOM_PEAKS and
        // index q_tolerances[i] for i < min(finalFomCount, MAX_FOM_PEAKS). With
        // MAX_FOM_PEAKS unified to 32 across all three shaders, a 20-length
        // buffer was read out of bounds whenever finalFomCount (= max(10,
        // gpuPeaksCount)) exceeded 20. Sizing to 32 covers every system; the
        // shader's own min() still caps the actual number of peaks read.
        const n_peaks_for_fom = Math.min(q_obs.length, 32);
        qTolerancesArray = new Float32Array(n_peaks_for_fom);
        for (let i = 0; i < n_peaks_for_fom; i++) {
            qTolerancesArray[i] = get_q_tolerance(
                peaks_sorted_by_q[i].original_index,
                tth_obs_rad,
                baseParams.wavelength,
                baseParams.tth_error
            ) + 1e-9;
        }
        qObsArray = new Float32Array(q_obs);

        // Initialize (or re-initialize) the refinement worker pool with the constants
        // that apply to this whole run. Each worker keeps its own foundSolutionMap;
        // cross-worker duplicates are resolved by applyFinalSieve at the end of the run.
        const N_FOR_M20 = Math.min(20, filteredPeaks.length);
        const maxTthDeg = maxOfArray(filteredPeaks.map(p => p.tth));
        const d_min = baseParams.wavelength / (2 * Math.sin(maxTthDeg * Math.PI / 360));
        const q_max = 1 / (d_min * d_min);
        refinementPool.init({
            baseParams,
            q_obs,
            original_indices,
            tth_obs_rad,
            peaks_sorted_by_q,
            N_FOR_M20,
            min_m20: 2.0,
            q_max,
            d_min,
        });
        refinementPool.reset();
    }

    // CPU
    if (workerSystems.length > 0) {
        const workerTask = new Promise((resolve) => {
            resolveWorkerTask = resolve;
            if (!workerURL) {
                showStatus("Error: Indexing engine is not available.", "error");
                resolve();
                return;
            }
            let workersRemaining = workerSystems.length;
            workerSystems.forEach((system) => {
                const absoluteTaskIndex = currentWorkerTaskIndex;
                currentWorkerTaskIndex++;

                taskTotals[absoluteTaskIndex] = 0;

                const worker = new Worker(workerURL);
                activeWorkers.push(worker);
                const workerT0 = performance.now();

                worker.onmessage = (e) => {
                    const { type, payload } = e.data;
                    if (type === 'trials_completed_batch') {
                        cumulativeTrials += payload;
                        const elapsedTimeSeconds = (performance.now() - indexingStartTime) / 1000;
                        const trialsPerSecond = (elapsedTimeSeconds > 0.1) ? cumulativeTrials / elapsedTimeSeconds : 0;
                        const totalPercentage = taskProgress.reduce((a, b) => a + b, 0) / totalTasks;
                        const message = `[${totalPercentage.toFixed(0)}%] Trials: ${cumulativeTrials.toLocaleString()} (${trialsPerSecond.toLocaleString('en-US', { maximumFractionDigits: 0 })}/s)`;
                        throttledSetStatusText(message);           

                    } else if (type === 'solution') {
                     handleNewSolution(payload); //new helper, 20 nov
                    } else if (type === 'progress') {
                                 
                        taskProgress[absoluteTaskIndex] = payload; 
                        updateProgressBar();
                    } else if (type === 'done') {

                        if (!gpuStopSignal.stop) {
                            taskProgress[absoluteTaskIndex] = 100;
                            updateProgressBar();
                        }

                        console.log(`[perf] CPU worker '${system}': ${(performance.now() - workerT0).toFixed(0)} ms`);
                        worker.terminate();
                        activeWorkers = activeWorkers.filter(w => w !== worker);
                        workersRemaining--;
                        if (workersRemaining === 0) {
                            resolveWorkerTask = null;
                            resolve();
                        }
                    }
                };

                worker.onerror = (err) => { 
            console.error(`Worker error in indexing task ${absoluteTaskIndex}:`, err);
            taskProgress[absoluteTaskIndex] = 100; // Allow queue to continue
            showStatus(`Warning: An indexing task encountered an error and was skipped. Check console for details.`, 'error');
            updateProgressBar();


                    console.error(`Worker for ${system} crashed:`, err.message);
                    worker.terminate();
                    activeWorkers = activeWorkers.filter(w => w !== worker);
                    workersRemaining--;
                    if (workersRemaining === 0) {
                        resolveWorkerTask = null;
                        resolve(); 
                    }
                };
                worker.postMessage({ ...baseParams, systemToSearch: system, allowedSystems: systemsToSearch });
            });
        });
        taskPromises.push(workerTask);
    }
    

    // Unified GPU task factory (replaces the previous three near-identical task blocks
    // for orthorhombic, monoclinic, and triclinic searches).
    //
    // The three systems differ only in:
    //  - K (number of peaks per trial)
    //  - the shader file and entry point
    //  - the engine method that runs the solver
    //  - the permutation multiplier used for the "total trials" estimate
    //  - a default for n_peaks_for_combo and n_hkl_for_basis
    //  - whether HKL basis is split into special (axial) and regular HKLs (ortho, mono do, tri doesn't)
    //  - the display label for status messages
    const GPU_SYSTEM_CONFIG = {
        orthorhombic: {
            K: 3,
            label: 'Orthorhombic',
            shortLabel: 'Ortho',
            shader: 'ortho_solver.wgsl',
            entryPoint: 'main_3p',
            engineMethod: 'runOrthoSolver',
            permutations: 6,
            defaultPeaks: 7,
            defaultHkl: 300,
            maxHkl: 2954,   // C(2954,3) < 2^32 < C(2955,3): u32 combinadic limit (K=3)
            splitSpecialHkls: true,
        },
        monoclinic: {
            K: 4,
            label: 'Monoclinic',
            shortLabel: 'Monoclinic',
            shader: 'monoclinic_solver.wgsl',
            entryPoint: 'main_4p',
            engineMethod: 'runMonoclinicSolver',
            permutations: 24,
            defaultPeaks: 7,
            defaultHkl: 100,
            maxHkl: 568,    // C(568,4) < 2^32 < C(569,4): u32 combinadic limit (K=4)
            splitSpecialHkls: true,
        },
        triclinic: {
            K: 6,
            label: 'Triclinic',
            shortLabel: 'Triclinic',
            shader: 'triclinic_solver.wgsl',
            entryPoint: 'main',
            engineMethod: 'runTriclinicSolver',
            permutations: 720,
            defaultPeaks: 8,
            defaultHkl: 40,
            maxHkl: 123,    // C(123,6) < 2^32 < C(124,6): u32 combinadic limit (K=6)
            splitSpecialHkls: false,
        },
    };

    // Build the HKL basis array. If splitSpecial is true, axial HKLs (two of h,k,l are 0)
    // are placed at the front of the list before truncation to n_hkl_for_basis.
    const buildHklBasis = (system, n_hkl_for_basis, splitSpecial) => {
        const hkl_full = get_hkl_search_list(system);
        let ordered;
        if (splitSpecial) {
            const special = [];
            const regular = [];
            for (const hkl of hkl_full) {
                const [h, k, l] = hkl;
                if ((k === 0 && l === 0) || (h === 0 && l === 0) || (h === 0 && k === 0)) {
                    special.push(hkl);
                } else {
                    regular.push(hkl);
                }
            }
            ordered = [...special, ...regular];
        } else {
            ordered = hkl_full;
        }
        const hkl_basis_raw = ordered.slice(0, n_hkl_for_basis);
        const hklBasisArray = new Float32Array(hkl_basis_raw.length * 4);
        hkl_basis_raw.forEach((hkl, i) => { hklBasisArray.set(hkl, i * 4); });
        return { hkl_basis_raw, hklBasisArray };
    };

    // Build the peak-combo flat Uint32Array using the already-present createCombinationGenerator.
    // Previously these were hand-written nested for-loops (3, 4, and 6 levels deep).
    const buildPeakCombos = (max_p, K) => {
        const numCombos = combinations(max_p, K);
        const peakCombos = new Uint32Array(numCombos * K);
        let offset = 0;
        for (const combo of createCombinationGenerator(max_p, K)) {
            peakCombos.set(combo, offset);
            offset += K;
        }
        return peakCombos;
    };

    // Create a GPU task for a given system. Returns an async function that can be
    // invoked to run the task. Returns null if the system isn't configured.
    const makeGpuTask = (system, absoluteTaskIndex) => {
        const cfg = GPU_SYSTEM_CONFIG[system];
        if (!cfg) return null;

        return async () => {
            const K_VALUE = cfg.K;
            const n_peaks_for_combo = Math.max(K_VALUE, parseInt(ui.gpuPeaksCount.value, 10) || cfg.defaultPeaks);
            let n_hkl_for_basis = Math.max(K_VALUE * 2, parseInt(ui.gpuHklTriplets.value, 10) || cfg.defaultHkl);

            // --- u32 combinadic guard ---------------------------------------
            // The WGSL solvers unrank HKL K-combinations with a u32 linear index
            // and a u32 binomial_table (get_combinadic_indices). Once C(n_hkl, K)
            // reaches 2^32, both the linear index AND the unranking binomials
            // overflow, silently corrupting the search (truncated bound + wrong
            // HKL triplets). cfg.maxHkl is the largest basis size that keeps the
            // whole combination space u32-addressable (ortho 2954 / mono 568 /
            // tri 123). Widening the JS binomial table would NOT help: the shader
            // is u32 end-to-end.
            const maxHklForU32 = cfg.maxHkl;
            if (n_hkl_for_basis > maxHklForU32) {
                showStatus(`${cfg.label}: HKL basis capped at ${maxHklForU32} (GPU ${K_VALUE}-peak u32 combination limit).`, 'error', 6000);
                n_hkl_for_basis = maxHklForU32;
            }
            // ----------------------------------------------------------------

            if (filteredPeaks.length < K_VALUE) {
                showStatus(`${cfg.label} search requires at least ${K_VALUE} peaks. Skipping.`, "error");
                taskProgress[absoluteTaskIndex] = 100;
                updateProgressBar();
                return;
            }

            const tGpuStart = performance.now();
            let cellsDispatchedToRefine = 0;
            try {
                showStatus(`Initializing WebGPU for ${cfg.label.toLowerCase()}...`, 'info');
                const engine = webgpuEngine;
                await engine.loadShader(cfg.shader);
                engine.createPipeline(cfg.entryPoint);

                // 1. HKL basis
                const { hkl_basis_raw, hklBasisArray } = buildHklBasis(system, n_hkl_for_basis, cfg.splitSpecialHkls);

                // 2. Peak combinations
                const max_p = Math.min(n_peaks_for_combo, qObsArray.length);
                if (max_p < K_VALUE) throw new Error(`Not enough peaks (${max_p}) for ${K_VALUE}-peak solve.`);
                const peakCombos = buildPeakCombos(max_p, K_VALUE);
                if (peakCombos.length === 0) throw new Error(`Not enough peaks to generate ${K_VALUE}-peak combinations.`);

                // 3. Stats
                const num_hkls = hkl_basis_raw.length;
                const totalHklCombos = combinations(num_hkls, K_VALUE);
                const taskTotalTrials = totalHklCombos * combinations(max_p, K_VALUE) * cfg.permutations;
                gpuTotalTrials += taskTotalTrials;
                taskTotals[absoluteTaskIndex] = taskTotalTrials;

                // 4. Execution
                const progressCallback = getGpuProgressCallback(cfg.shortLabel, absoluteTaskIndex);
                // Main-thread dispatch is near-instant now (just postMessage), but we
                // still need to wait for the worker pool to drain before moving on.
                // The main metric is `drainMs` below (how long after GPU finishes
                // that workers are still chewing through the backlog).
                let dispatchMs = 0;
                // Everything this task dispatches gets a batch id >= poolMark,
                // so the drain below waits on this task's work only rather than
                // on whatever else happens to be in flight pool-wide.
                const poolMark = refinementPool.mark();
                const handleIntermediateResults = (newCells) => {
                    cellsDispatchedToRefine += newCells.length;
                    const t0 = performance.now();
                    // Send the whole GPU-chunk's worth of cells as a single batched
                    // call to the pool. The pool splits across workers round-robin,
                    // sending at most N messages per batch (N = pool size) regardless
                    // of how many cells are in the chunk.
                    refinementPool.refineBatch(newCells);
                    dispatchMs += performance.now() - t0;
                };

                const engineFn = engine[cfg.engineMethod];
                if (typeof engineFn !== 'function') {
                    throw new Error(`Engine missing method ${cfg.engineMethod}`);
                }
                const tEngineStart = performance.now();
                await engineFn.call(
                    engine,
                    qObsArray,
                    hklBasisArray,
                    peakCombos,
                    null,
                    qTolerancesArray,
                    progressCallback,   // pass directly: an arrow wrapper would drop .reportPlan
                    gpuStopSignal,
                    baseParams,
                    handleIntermediateResults
                );
                const engineMs = performance.now() - tEngineStart;
                // Wait for the refinement pool to finish processing the backlog of
                // cells dispatched during the GPU run. If GPU produced cells faster
                // than workers could refine, drainMs > 0. If workers kept up, it's ~0.
                const tDrainStart = performance.now();
                await refinementPool.drain(30000, poolMark);
                const drainMs = performance.now() - tDrainStart;
                console.log(`[perf]   engineFn('${cfg.label}') wall time: ${engineMs.toFixed(0)} ms`);
                console.log(`[perf]   dispatch: ${dispatchMs.toFixed(0)} ms  |  pool-drain: ${drainMs.toFixed(0)} ms  |  cells: ${cellsDispatchedToRefine}  |  workers: ${REFINE_POOL_SIZE}`);
            } catch (err) {
                console.error(`WebGPU Error (${cfg.label}):`, err);
                showStatus(`${cfg.shortLabel} GPU Error: ${err.message}`, 'error');
            } finally {
                const tGpuEnd = performance.now();
                console.log(`[perf] GPU task '${cfg.label}': ${(tGpuEnd - tGpuStart).toFixed(0)} ms (${cellsDispatchedToRefine} candidate cells refined on CPU)`);
                updateProgressBar();
            }
        };
    };

    // Launch one GPU task per selected low-symmetry system.
    // Iterate in a canonical order (ortho, mono, tri) regardless of the checkbox order,
    // so task indices remain deterministic and match the prior behavior.
    const GPU_ORDER = ['orthorhombic', 'monoclinic', 'triclinic'];
    for (const system of GPU_ORDER) {
        if (!webgpuSystems.includes(system)) continue;
        if (!GPU_SYSTEM_CONFIG[system]) continue;
        const absoluteTaskIndex = currentWorkerTaskIndex + currentGpuTaskIndex;
        currentGpuTaskIndex++;
        const taskFn = makeGpuTask(system, absoluteTaskIndex);
        if (taskFn) taskPromises.push(taskFn());
    }

    // final
    await Promise.all(taskPromises);

    // --- NEW: Send GPU solutions through the post-processing worker ---
    if (webgpuSystems.length > 0 && solutions.length > 0 && !gpuStopSignal.stop) {
        if (statusTextElement) statusTextElement.textContent = 'Post-processing GPU cells...';
        const bestOfSolutionsBefore = solutions.reduce((m, s) => (s && isFinite(s.m20) && s.m20 > m) ? s.m20 : m, 0);

        // De-duplicate BEFORE post-processing, not only after it. The candidate
        // list arriving here is overwhelmingly redundant -- a real PbSO4 run
        // produced 49 solutions that the sieve collapsed to 2, i.e. ~24 copies
        // of each distinct lattice -- and relabelling all 49 does the same work
        // two dozen times over to reach the same two answers. Measured on that
        // run: 2291 ms over 40 parents versus 132 ms over the sieved set, same
        // final cell. `solutions` itself is left intact; this only decides which
        // cells are worth handing to the search.
        const postParents = applyFinalSieve(solutions);

        // Spend the saving on a deeper search instead of pocketing it. With a
        // handful of genuinely distinct parents the per-parent budget can be
        // several times the default and still cost a fraction of the old pass.
        const nP = postParents.length;
        const swapCfg = nP <= 8
            ? { MAX_FITS: 600, MAX_FREE: 16, MAX_EVALS: 32, MAX_POST: 5, ROUNDS: 6 }
            : nP <= 20
                ? { MAX_FITS: 300, MAX_FREE: 12, MAX_EVALS: 24, ROUNDS: 5 }
                : {};   // many distinct lattices: fall back to the defaults
        console.log(`[post-process] ${solutions.length} solutions -> ${nP} distinct parents ` +
                    `(budget: ${swapCfg.MAX_FITS || 'default'} fits/round)`);
        await new Promise(resolve => {
            const worker = new Worker(workerURL);
            const bestOf = (arr) => arr.reduce((m, s) => (s && isFinite(s.m20) && s.m20 > m) ? s.m20 : m, 0);
            const m20Before = bestOf(solutions);
            worker.onmessage = (e) => {
                if (e.data.type === 'solution') handleNewSolution(e.data.payload);
                else if (e.data.type === 'postProcessSummary') {
                    const st = e.data.payload || {};
                    if (st.fatal) console.error('[post-process] ABORTED:', st.fatal, st.stack);
                    else console.log(`[post-process] ${st.parents} parents | swap search ran on ` +
                        `${st.swapRan}/${st.swapEligible} | ${st.swapPosted} swap solutions posted | ` +
                        `${st.solutionErrors} solution errors, ${st.swapErrors} swap errors | ` +
                        `best M20 ${(+st.bestBefore).toFixed(2)} -> ${(+st.bestAfter).toFixed(2)}`);
                }
                else if (e.data.type === 'done') { worker.terminate(); resolve(); }
            };
            // Never swallow this. A throw inside the post-process worker used to
            // resolve the promise in complete silence, so a run that lost every
            // transformed and swapped solution looked exactly like a run that
            // simply found nothing better.
            worker.onerror = (err) => {
                console.error('[post-process] worker crashed:',
                              err && err.message, err && err.filename, err && err.lineno);
                showStatus('Post-processing failed: ' + ((err && err.message) || 'worker error') +
                           ' - results are pre-post-processing only.', 'error', 10000);
                resolve();
            };
            worker.postMessage({ 
                ...baseParams, 
                systemToSearch: 'post_process', 
                gpuSolutions: postParents, 
                allowedSystems: systemsToSearch,
                swapCfg
            });
        });
        console.log(`[post-process] main thread: best M20 ${bestOfSolutionsBefore.toFixed(2)} -> ` +
                    `${solutions.reduce((m, s) => (s && isFinite(s.m20) && s.m20 > m) ? s.m20 : m, 0).toFixed(2)}`);
    }
    // ------------------------------------------------------------------

    await new Promise(resolve => setTimeout(resolve, 250));
    finalizeIndexing(gpuStopSignal.stop, mySessionToken);
};



// `runToken` identifies which solutions were produced BY this run (see
// handleNewSolution). It is normally the same as sessionToken, but the abort
// path bumps indexingRunToken before finalizing, so it passes the pre-bump
// value explicitly. Solutions carrying any other token came from an earlier
// run, possibly under a different wavelength or peak selection.
const finalizeIndexing = (stoppedByUser = false, sessionToken = null, runToken = undefined) => {
    if (runToken === undefined) runToken = sessionToken;
    // A manual Stop or a new file load bumps indexingRunToken via
    // abortActiveIndexing(), which already re-enabled the UI. If that
    // happened after this run started, this is a stale tail from an
    // abandoned run — bail out instead of overwriting fresher state
    // (or a fresher run's in-progress state) with old results.
    if (sessionToken !== null && sessionToken !== indexingRunToken) {
        console.log('[perf] Discarding stale finalizeIndexing() call (superseded by a newer run or a reset).');
        return;
    }
    const statusTextElement = document.getElementById('status-text');

    // 1. Calculate Duration with seconds
    const durationMs = performance.now() - indexingStartTime;
    console.log(`[perf] === Total indexing time: ${durationMs.toFixed(0)} ms ===`);
    const durationSec = (durationMs / 1000).toFixed(1);
    const durationStr = durationMs > 60000 
        ? `${Math.floor(durationMs/60000)}m ${(durationMs % 60000 / 1000).toFixed(0)}s` 
        : `${durationSec}s`;
    lastDurationStr = durationStr; 

    // 2. Calculate Real GPU Trials based on actual progress %
    let gpuActualTrials = 0;
    if (taskTotals && taskProgress) {
        taskTotals.forEach((total, i) => {
            const progress = taskProgress[i] || 0;
            gpuActualTrials += total * (progress / 100.0);
        });
    }

    // 3. Compile Totals
    const totalCpuTrials = cumulativeTrials; 
    const totalActualTrials = totalCpuTrials + gpuActualTrials;
    const totalMaxTrials = totalCpuTrials + gpuTotalTrials; 
    
    const fmtActual = totalActualTrials.toLocaleString('en-US', {maximumFractionDigits: 0});
    const fmtMax = totalMaxTrials.toLocaleString('en-US', {maximumFractionDigits: 0});
    
    const isGpuRun = (orthoCheckbox && orthoCheckbox.checked) || 
                     (monoCheckbox && monoCheckbox.checked) || 
                     (triCheckbox && triCheckbox.checked);

    // 4. Construct Report String
    let finalStatus = "";
    if (isGpuRun) {
        const hklSize = ui.gpuHklTriplets.value;
        const peaksComb = ui.gpuPeaksCount.value;
        // Format: Trials: Actual / Max
        finalStatus = `Trials: ${fmtActual} / ${fmtMax}    Time: ${durationStr}    HKL: ${hklSize}    Peaks: ${peaksComb}`;
    } else {
        finalStatus = `CPU Trials: ${fmtActual}    Time: ${durationStr}`;
    }

    lastIndexingStats = finalStatus; 
    
    
    // Update screen status temporarily
    if (statusTextElement) statusTextElement.textContent = 'Applying final sieve...';
    
    // Apply Sieve
    const _perfSieveEnd = perfStart('applyFinalSieve');
    const _nBeforeSieve = solutions.length;
    solutions = applyFinalSieve(solutions); 
    _perfSieveEnd(`(${_nBeforeSieve} -> ${solutions.length})`);
    
    // Space Group Analysis
    if (statusTextElement) statusTextElement.textContent = 'Analyzing space groups...';

    const tthMinVal = parseFloat(ui.tthMinSlider.value);
    const tthMaxVal = parseFloat(ui.tthMaxSlider.value);
    const filteredPeaks = pickedPeaks.filter(p => p.tth >= tthMinVal && p.tth <= tthMaxVal);

    // analyzeSystematicAbsences (in worker-logic.js) is now Ka2-aware:
    // each peak's `ka2Suspect` flag is propagated into the indexed-hkl
    // records, and downstream centering/extinction/ranking logic counts
    // hard (real) and soft (Ka2-suspect) violations separately. A space
    // group disqualified only by Ka2-suspect peaks ends up with
    // hardViolations === 0 and is shown alongside the truly viable groups.
    // Only (re)analyse cells belonging to this run. `solutions` intentionally
    // survives across runs, and re-running the absence analysis on a solution
    // found under a previous wavelength / 2-theta window / peak list would
    // silently re-label it against data it was never fitted to -- and pay for
    // the analysis again every subsequent run. A solution keeps whatever
    // analysis it was given when it was found.
    const needsAnalysis = (sol) =>
        !sol.analysis || runToken === null || sol._runToken === runToken;
    const _nStale = solutions.filter(s => !needsAnalysis(s)).length;
    if (_nStale > 0) {
        console.log(`[perf] spaceGroupAnalysis: keeping ${_nStale} analysis result(s) from earlier run(s).`);
    }

    const _perfSgEnd = perfStart('spaceGroupAnalysis');
    if (spaceGroupData) {
        solutions.forEach(sol => {
            if (!needsAnalysis(sol)) return;
            sol.analysis = analyzeSystematicAbsences(
                sol,
                filteredPeaks,
                spaceGroupData,
                parseFloat(ui.wavelength.value),
                parseFloat(ui.tthError.value),
                tthMaxVal,
                getImpurityPeaks(),
                tthMinVal
            );
        });
    } else {
        console.warn('Space group data not available');
        const lambda = parseFloat(ui.wavelength.value);
        
        solutions.forEach(sol => {
            if (!needsAnalysis(sol)) return;
            const basicHklList = generateHKL_for_analysis(sol, lambda, tthMaxVal);
            sol.analysis = {
                centering: 'Unknown (data not loaded)',
                rankedSpaceGroups: [],
                detectedExtinctions: [],
                hklList: basicHklList 
            };
        });
    }
    _perfSgEnd(`(${solutions.length} solutions)`);
                                
    // Clear status text on screen
    if (statusTextElement) statusTextElement.textContent = '';
    
    setUIState(false);    
    
    // Sort and Update Table
    sortState = { column: 'm20', direction: 'desc' };
    sortSolutions(); 
    
    displayedSolutions = [...solutions]; 
    updateSolutionsTable(); 

    // Final Toast Message / LED
    if (solutions.length > 0) {
        const message = stoppedByUser ? 
            'Indexing stopped by user.' : 
            'Indexing complete.';
        showStatus(`${message} Found ${solutions.length} potential solution(s).`, 'success');
        ui.solutionsLed.className = 'led-indicator green';
    } else {
        const message = stoppedByUser ? 'Indexing stopped by user.' : 'Indexing finished.';
        showStatus(`${message} No valid solutions were found.`, 'info');
        if (solutions.length === 0) {
            ui.solutionsLed.className = 'led-indicator red';
        }
    }
};
 
    ui.startIndexingButton.addEventListener('click', startIndexing);

 const abortActiveIndexing = (shouldFinalize = false) => {
        activeWorkers.forEach(w => w.terminate());
        activeWorkers = [];
        gpuStopSignal.stop = true;

        // Kill the refinement pool too. Any in-flight cells are dropped.
        refinementPool.terminate();

        if (typeof resolveWorkerTask === 'function') {
            resolveWorkerTask(); // Unblock the CPU-worker await, if any
            resolveWorkerTask = null;
        }

        // Token of the run we are stopping. Its solutions carry this value, so
        // finalizeIndexing needs it to recognise them as belonging to the
        // current run -- comparing against the post-bump token would classify
        // every cell this run just found as stale and skip its analysis.
        const abortedRunToken = indexingRunToken;

        indexingRunToken++; // invalidate any background startIndexing() loop

        if (shouldFinalize) {
            // Immediately run final sieve and space group check on solutions found so far
            finalizeIndexing(true, indexingRunToken, abortedRunToken);
        } else {
            setUIState(false);
        }
    };

    ui.reportButton.addEventListener('click', () => {
        if (isIndexing) {
            abortActiveIndexing(true); // Pass true to trigger space group check & sieve
        } else { 
            generatePDFReport(); 
        }
    });



    const sortSolutions = () => {
        const { column, direction } = sortState;
        const dir = direction === 'asc' ? 1 : -1;
        solutions.sort((a, b) => {
            if (column === 'system') {
                return (a.system || '').localeCompare(b.system || '') * dir;
            } else {
                let valA = a[column]; let valB = b[column];
                if (isNaN(valA) || valA == null) valA = -Infinity;
                if (isNaN(valB) || valB == null) valB = -Infinity;
                return (valA - valB) * dir;
            }
        });
    };
    

    const updateSolutionsTable = () => {
        // Build the entire table markup in memory first
        const rowsHtml = displayedSolutions.map((sol, index) => {
            if (!sol || !sol.system) return '';
            let paramsCell = '', anglesCell = '';
            switch(sol.system) {
                case 'cubic': paramsCell = `a = ${sol.a.toFixed(4)}`; anglesCell = `90, 90, 90`; break;
                case 'tetragonal': paramsCell = `a = ${sol.a.toFixed(4)}, c = ${sol.c.toFixed(4)}`; anglesCell = `90, 90, 90`; break;
                case 'hexagonal': paramsCell = `a = ${sol.a.toFixed(4)}, c = ${sol.c.toFixed(4)}`; anglesCell = `90, 90, 120`; break;
                case 'orthorhombic': paramsCell = `a = ${sol.a.toFixed(4)}<br>b = ${sol.b.toFixed(4)}<br>c = ${sol.c.toFixed(4)}`; anglesCell = `90, 90, 90`; break;
                case 'monoclinic': paramsCell = `a = ${sol.a.toFixed(4)}<br>b = ${sol.b.toFixed(4)}<br>c = ${sol.c.toFixed(4)}`; anglesCell = `90, ${sol.beta.toFixed(3)}, 90`; break;
                case 'triclinic': 
                    paramsCell = `a = ${sol.a.toFixed(4)}<br>b = ${sol.b.toFixed(4)}<br>c = ${sol.c.toFixed(4)}`; 
                    anglesCell = `&alpha; = ${sol.alpha.toFixed(3)}<br>&beta; = ${sol.beta.toFixed(3)}<br>&gamma; = ${sol.gamma.toFixed(3)}`; 
                    break;
                default: paramsCell = `${sol.a.toFixed(4)}`; anglesCell = `-`;
            }
            if (sol.zero_correction) {
                anglesCell += `<br><span style="font-size:0.9em; color: var(--text-dark);">(Z=${sol.zero_correction.toFixed(4)}°)</span>`;
            }
            const isSelected = (selectedSolution === sol) ? ' class="selected"' : '';
            // Mark solutions produced by Explore Group, with the number of swaps
            // applied, so a derived cell is never mistaken for an independent hit.
            const nSwaps = (sol.manualSwaps || []).length;
            let sysCell = sol.system.substring(0,4);
            if (nSwaps > 0) {
                sysCell += `<br><span class="sol-badge swap" title="${(sol.manualSwaps||[]).map(x=>x.from+'->'+x.to+' @ '+x.tth.toFixed(3)).join('; ')}">swap&times;${nSwaps}</span>`;
            }
            // Cells improved by the Monte-Carlo polish are flagged so a derived
            // solution is never mistaken for an independent hit.
            // Suppressed when an SG badge is also due: a cell adopted from the
            // space-group scan is nearly always MC-polished as well, and two
            // badges saying almost the same thing crowd the column. The M20
            // provenance is folded into the SG tooltip instead, so nothing is lost.
            if (sol.mcPolished && !sol.sgClass) {
                const from = (sol.mcFrom && isFinite(sol.mcFrom.m20)) ? sol.mcFrom.m20.toFixed(2) : '?';
                sysCell += `<br><span class="sol-badge mc" title="Monte-Carlo polished: M20 ${from} -> ${sol.m20.toFixed(2)}">MC</span>`;
            }
            // Refined under a space-group hypothesis: the forbidden reflections
            // were removed from the line list before fitting, so the cell, the
            // pairing and M20 only mean anything alongside the class they assume.
            if (sol.sgClass) {
                const bits = [`Refined under ${sol.sgClass}`];
                if (sol.sgMembers && sol.sgMembers.length) bits.push(sol.sgMembers.join(', '));
                if (sol.sgConditions && sol.sgConditions.length) bits.push(sol.sgConditions.join(' ; '));
                // The margin decides whether this badge is a result or a guess.
                // A class that led its table by half a nat was not established by
                // the data, and the badge must not let the cell quietly acquire
                // the authority of one that was.
                const ev = sol.sgEvidence;
                if (ev) {
                    bits.push(`${ev.clean}/${ev.informative} forbidden lines clean, ` +
                              `${ev.hardViolations} hard violation(s)` +
                              (ev.softViolations ? ` (+${ev.softViolations} soft)` : '') +
                              (ev.unindexed ? `, ${ev.unindexed} unindexed` : ''));
                    bits.push(ev.wilson ? 'absences weighted per reflection (Wilson)'
                                        : (isFinite(ev.pHat) ? `p(line observed) = ${(ev.pHat * 100).toFixed(0)}%` : ''));
                    if (ev.mode !== 'mc') bits.push('stage-1 (least-squares) result only');
                }
                const decisive = isFinite(sol.sgMargin) && sol.sgMargin >= SG_DECISIVE_NATS;
                if (isFinite(sol.sgMargin)) {
                    bits.push(decisive
                        ? `${sol.sgMargin.toFixed(1)} nats ahead of the runner-up`
                        : `NOT decisive: only ${sol.sgMargin.toFixed(1)} nats ahead of the runner-up, ` +
                          `so the absences do not choose between this class and the next`);
                }
                if (sol.mcPolished && sol.mcFrom && isFinite(sol.mcFrom.m20)) {
                    bits.push(`MC: M20 ${sol.mcFrom.m20.toFixed(2)} -> ${sol.m20.toFixed(2)}`);
                }
                sysCell += `<br><span class="sol-badge sg${decisive ? '' : ' sg-tied'}" ` +
                           `title="${bits.join(' | ')}">SG${decisive ? '' : '?'}</span>`;
            }
            return `<tr data-index="${index}"${isSelected}><td>${sysCell}</td><td>${paramsCell}</td><td>${anglesCell}</td><td>${sol.volume.toFixed(2)}</td><td>${sol.m20.toFixed(2)}</td></tr>`;
        }).join('');
        
        // Single DOM write
        ui.solutionsTableBody.innerHTML = rowsHtml;
        
        ui.solutionsTableHeaders.forEach(h => {
            h.classList.remove('sort-asc', 'sort-desc');
            if (h.dataset.sort === sortState.column) {
               h.classList.add(sortState.direction === 'asc' ? 'sort-asc' : 'sort-desc');
            }
        });
    };

    ui.solutionsTableBody.addEventListener('click', (e) => {
        const row = e.target.closest('tr'); if (!row) return;
        document.querySelectorAll('#solutions-table-body tr').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        const index = parseInt(row.dataset.index);
        // applySolutionSelection rebuilds the line list with the refined zero
        // folded in. The old inline version also took its 2-theta ceiling from
        // the current x-axis maximum, so zooming in and then picking a solution
        // silently truncated the calculated pattern at the edge of the view.
        applySolutionSelection(displayedSolutions[index]);
    });


    // Context Menu Logic
    const contextMenu = document.getElementById('context-menu');
    let ctxMenuTargetIndex = -1;
    // Row indices index into displayedSolutions (what is rendered). Resolving
    // the object here and then looking it up by identity in `solutions` keeps
    // erase / swap / MC / report acting on the row the user actually clicked.
    const ctxTargetSolution = () =>
        (ctxMenuTargetIndex > -1 ? (displayedSolutions[ctxMenuTargetIndex] || null) : null);

    // 1. Show Menu on Right Click
    ui.solutionsTableBody.addEventListener('contextmenu', (e) => {
        const row = e.target.closest('tr');
        if (!row) return;
        
        e.preventDefault(); // Stop default browser menu
        
        // Select the row visually (optional, but good UX)
        document.querySelectorAll('#solutions-table-body tr').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        
        // Store the index. Row indices address displayedSolutions; every action
        // below resolves through ctxTargetSolution() so it can never act on a
        // different element of `solutions` if the two lists drift apart.
        ctxMenuTargetIndex = parseInt(row.dataset.index);
        // Right-clicking a row selects it too, so it must refresh the calculated
        // lines exactly like a left click. It used to assign selectedSolution on
        // its own and leave currentHklList pointing at the previously selected
        // cell, drawing one solution's ticks over another's.
        applySolutionSelection(displayedSolutions[ctxMenuTargetIndex]);
        
        // Position and show menu
        contextMenu.style.top = `${e.pageY}px`;
        contextMenu.style.left = `${e.pageX}px`;
        contextMenu.style.display = 'block';
    });

    // 2. Hide Menu on any click elsewhere
    document.addEventListener('click', () => {
        contextMenu.style.display = 'none';
    });

    // 3. Action: Erase Solution
    document.getElementById('ctx-erase').addEventListener('click', () => {
        const target = ctxTargetSolution();
        if (target) {
            // Remove from main list by identity, not by rendered row index.
            const k = solutions.indexOf(target);
            if (k > -1) solutions.splice(k, 1);

            // Re-sync displayed list
            displayedSolutions = [...solutions]; 
            
            foundSolutionMap.clear(); //= vérifier.. une fois une solution effacée...?

            updateSolutionsTable();
            updateAllMarkers(); // Clear blue lines if we deleted the selected one
            ctxMenuTargetIndex = -1;
        }
    });

    // 3b. Action: Swap hkl
    // The indexer assigns each peak to its nearest calculated line. That can be
    // wrong with nothing to flag it - in a permissive space group both candidates
    // are allowed, so no violation appears - which is why this is manual rather
    // than rule-driven. The user edits assignments; Apply re-refines and adds the
    // result as an ordinary solution so M20 can be compared against the parent.
    // Every peak in the 2-theta range is listed, not just the low-angle ones.
    // This used to stop at the first twelve on the reasoning that low-angle
    // assignments move the cell most -- true, but it left no way to reach a
    // misassignment above the cut, and the refit touches every peak regardless
    // of what the dialog chose to show. The table scrolls instead: the modal is
    // a flex column capped at 86vh with a sticky header, so a long list costs
    // height only until it hits the cap.
    let swapParent = null, swapRows = [];
    const swapOverlay = document.getElementById('swap-overlay');
    const swapMsg = document.getElementById('swap-msg');

    // One delegated listener rather than one per input. The table can now run to
    // hundreds of rows, and re-binding on each open would stack duplicate
    // listeners on the same static tbody.
    document.getElementById('swap-tbody').addEventListener('input', (e) => {
        const inp = e.target;
        if (!inp || inp.tagName !== 'INPUT') return;
        const r = swapRows[parseInt(inp.dataset.row, 10)];
        if (!r) return;
        const orig = r[inp.dataset.f];
        const cur = inp.value.trim();
        const same = (cur === '' && orig == null) || (cur !== '' && Number(cur) === orig);
        inp.classList.toggle('changed', !same);
    });

    const closeSwapModal = () => {
        swapOverlay.classList.remove('open');
        swapParent = null; swapRows = [];
    };

    document.getElementById('ctx-swap').addEventListener('click', () => {
        const parent = ctxTargetSolution();
        if (!parent) return;
        const wl = parseFloat(ui.wavelength.value);
        const te = parseFloat(ui.tthError.value);
        const tMin = parseFloat(ui.tthMinSlider.value);
        const tMax = parseFloat(ui.tthMaxSlider.value);
        const pk = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);
        if (pk.length < 6) { showStatus('Not enough peaks in the 2-theta range.', 'error', 4000); return; }

        swapParent = parent;
        swapRows = getPeakAssignments(parent, pk, wl, te, tMax);
        if (!swapRows.length) { showStatus('No peaks to show for this solution.', 'error', 4000); return; }

        const tbody = document.getElementById('swap-tbody');
        tbody.innerHTML = swapRows.map((r, i) => {
            const cls = r.indexed ? '' : ' class="unindexed"';
            const v = (x) => (x == null ? '' : x);
            return `<tr${cls} data-row="${i}">` +
                   `<td>${r.tth.toFixed(3)}</td>` +
                   `<td>${r.d_obs != null ? r.d_obs.toFixed(4) : '-'}</td>` +
                   `<td><input type="number" step="1" data-f="h" data-row="${i}" value="${v(r.h)}"></td>` +
                   `<td><input type="number" step="1" data-f="k" data-row="${i}" value="${v(r.k)}"></td>` +
                   `<td><input type="number" step="1" data-f="l" data-row="${i}" value="${v(r.l)}"></td>` +
                   `<td>${r.calc_tth != null ? r.calc_tth.toFixed(3) : '-'}</td>` +
                   `<td>${r.diff != null ? r.diff.toFixed(3) : '-'}</td></tr>`;
        }).join('');

        // Say how many rows there are, so a short list does not look truncated
        // and a long one is visibly a scroll rather than a cut.
        const nUn = swapRows.filter(r => !r.indexed).length;
        document.getElementById('swap-sub').textContent =
            `${swapRows.length} peak${swapRows.length === 1 ? '' : 's'} in the 2-theta range` +
            (nUn ? `, ${nUn} unindexed` : '') + '. ' +
            'Edit any assignment, then Apply to create a new refined solution. ' +
            'Blank rows are left unchanged.';

        document.getElementById('swap-title').textContent =
            `Swap hkl - ${parent.system}, a=${parent.a.toFixed(4)}` +
            (parent.b ? `, b=${parent.b.toFixed(4)}` : '') + (parent.c ? `, c=${parent.c.toFixed(4)}` : '');
        swapMsg.textContent = '';
        swapOverlay.classList.add('open');
        contextMenu.style.display = 'none';
    });

    document.getElementById('swap-cancel').addEventListener('click', closeSwapModal);
    swapOverlay.addEventListener('click', (e) => { if (e.target === swapOverlay) closeSwapModal(); });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && swapOverlay.classList.contains('open')) closeSwapModal();
    });

    document.getElementById('swap-apply').addEventListener('click', () => {
        if (!swapParent) return;
        const wl = parseFloat(ui.wavelength.value);
        const te = parseFloat(ui.tthError.value);
        const tMin = parseFloat(ui.tthMinSlider.value);
        const tMax = parseFloat(ui.tthMaxSlider.value);
        const imp = getImpurityPeaks();
        const rz = !!(ui.refineZeroCheckbox && ui.refineZeroCheckbox.checked);
        const pk = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);

        // Collect only rows the user actually changed.
        const overrides = [];
        let bad = null;
        document.querySelectorAll('#swap-tbody tr').forEach(tr => {
            const i = parseInt(tr.dataset.row, 10);
            const r = swapRows[i];
            const get = (f) => tr.querySelector(`input[data-f="${f}"]`).value.trim();
            const hs = get('h'), ks = get('k'), ls = get('l');
            if (hs === '' && ks === '' && ls === '') return;              // untouched blank row
            if (hs === '' || ks === '' || ls === '') { bad = bad || `row at ${r.tth.toFixed(3)} deg: h, k and l must all be given`; return; }
            const h = Number(hs), k = Number(ks), l = Number(ls);
            if (![h, k, l].every(Number.isInteger)) { bad = bad || `row at ${r.tth.toFixed(3)} deg: indices must be integers`; return; }
            if (h === r.h && k === r.k && l === r.l) return;              // unchanged
            overrides.push({ tth: r.tth, h, k, l });
        });
        if (bad) { swapMsg.textContent = bad; return; }
        if (!overrides.length) { swapMsg.textContent = 'No changes to apply.'; return; }

        // Two peaks cannot be the same reflection. Catch it here rather than let
        // the least-squares fit quietly average them into a distorted cell.
        //
        // The assignment list has to cover EVERY peak the refit will touch.
        // That is now the same set the dialog shows, but the check is still made
        // against a freshly computed list rather than against swapRows: the two
        // agree only as long as the dialog is modal, and this guarantee should
        // not depend on that.
        const allRows = getPeakAssignments(swapParent, pk, wl, te, tMax);
        const beingReassigned = new Set(overrides.map(o => o.tth.toFixed(4)));
        const seen = new Map();
        allRows.forEach(r => {
            if (!r.indexed) return;
            if (beingReassigned.has(r.tth.toFixed(4))) return;  // this peak is moving
            seen.set(`${r.h},${r.k},${r.l}`, r.tth);
        });
        for (const o of overrides) {
            const key = `${o.h},${o.k},${o.l}`;
            if (seen.has(key)) {
                swapMsg.textContent = `(${key}) is already assigned to the peak at ${seen.get(key).toFixed(3)} deg.`;
                return;
            }
            seen.set(key, o.tth);
        }

        const res = refineWithManualHkl(swapParent, pk, overrides, wl, te, tMax, rz, imp);
        if (!res || res.error) { swapMsg.textContent = 'Refinement failed: ' + ((res && res.error) || 'unknown'); return; }

        const child = res.cell;
        try {
            child.analysis = analyzeSystematicAbsences(child, pk, spaceGroupData, wl, te, tMax, imp, tMin);
        } catch (err) { child.analysis = null; }
        // Same placement rule as Refine MC: a derived solution sits directly
        // above the cell it came from, so the pair can be read together.
        const swapParentIdx = solutions.indexOf(swapParent);
        if (swapParentIdx >= 0) solutions.splice(swapParentIdx, 0, child);
        else solutions.push(child);
        displayedSolutions = [...solutions];
        updateSolutionsTable();
        updateAllMarkers();
        closeSwapModal();
        const list = res.swaps.map(x => `${x.from}->${x.to}`).join(', ');
        showStatus(`Applied ${res.swaps.length} swap(s): ${list}. ` +
                   `M20 ${(swapParent.m20 || 0).toFixed(1)} -> ${(child.m20 || 0).toFixed(1)}`, 'success', 8000);
    });

    // 3c. Action: Refine MC
    // Least squares converges to the nearest minimum of the RESIDUAL, but M20 is
    // not the residual: a cell a few tenths of a percent away can index more
    // lines and score considerably higher. This runs a symmetry-constrained
    // stochastic search around the selected cell (and around the next best ones,
    // if the user asks for more than one) and keeps only cells that beat their
    // parent's M20. Zero is a search dimension whenever "Refine Zero" is on.
    let mcRunning = false;
    const mcOverlay = document.getElementById('mc-overlay');
    const mcMsg = document.getElementById('mc-msg');
    const mcApplyBtn = document.getElementById('mc-apply');

    const closeMcModal = () => {
        if (mcRunning) return;               // don't close mid-run
        mcOverlay.classList.remove('open');
    };

    document.getElementById('ctx-mc').addEventListener('click', () => {
        const parent = ctxTargetSolution();
        if (!parent) return;
        if (typeof monteCarloRefineCell !== 'function') {
            showStatus('Monte-Carlo refinement is unavailable (worker-logic.js not loaded).', 'error', 5000);
            return;
        }
        if (!parent.system || typeof MC_NPAR === 'undefined' || !MC_NPAR[parent.system]) {
            showStatus('This solution has no crystal system the MC can constrain.', 'error', 4000);
            return;
        }

        document.getElementById('mc-title').textContent =
            `Refine MC - ${parent.system}, a=${parent.a.toFixed(4)}` +
            (parent.b ? `, b=${parent.b.toFixed(4)}` : '') +
            (parent.c ? `, c=${parent.c.toFixed(4)}` : '');
        mcMsg.textContent = '';
        mcMsg.classList.remove('info');
        ['mc-iterations', 'mc-restarts']
            .forEach(id => document.getElementById(id).classList.remove('invalid'));
        mcApplyBtn.disabled = false;
        mcApplyBtn.textContent = 'Apply';
        mcOverlay.classList.add('open');
        contextMenu.style.display = 'none';
    });

    // Clear the error highlight as soon as the user starts fixing a field.
    ['mc-iterations', 'mc-restarts'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', () => el.classList.remove('invalid'));
    });

    document.getElementById('mc-cancel').addEventListener('click', closeMcModal);
    mcOverlay.addEventListener('click', (e) => { if (e.target === mcOverlay) closeMcModal(); });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && mcOverlay.classList.contains('open')) closeMcModal();
    });

    mcApplyBtn.addEventListener('click', () => {
        if (mcRunning) return;
        const parent = ctxTargetSolution();
        if (!parent) { closeMcModal(); return; }

        // --- read and validate the three parameters -------------------------
        const mcInputIds = ['mc-iterations', 'mc-restarts'];
        mcInputIds.forEach(id => document.getElementById(id).classList.remove('invalid'));
        const readInt = (id, lo, hi, label) => {
            const el = document.getElementById(id);
            const raw = el.value.trim();
            const v = parseInt(raw, 10);
            let error = null;
            if (raw === '' || !isFinite(v)) error = `${label} must be a number.`;
            else if (v < lo || v > hi) error = `${label} must be between ${lo} and ${hi}.`;
            if (error) { el.classList.add('invalid'); return { error }; }
            return { value: v };
        };
        const rIt  = readInt('mc-iterations', 50, 20000, 'Iterations');
        const rRes = readInt('mc-restarts', 1, 20, 'Restarts');
        for (const r of [rIt, rRes]) {
            if (r.error) { mcMsg.classList.remove('info'); mcMsg.textContent = r.error; return; }
        }
        const nIterations = rIt.value, nRestarts = rRes.value;

        const wl = parseFloat(ui.wavelength.value);
        const te = parseFloat(ui.tthError.value);
        const tMin = parseFloat(ui.tthMinSlider.value);
        const tMax = parseFloat(ui.tthMaxSlider.value);
        const imp = getImpurityPeaks();
        const rz = !!(ui.refineZeroCheckbox && ui.refineZeroCheckbox.checked);
        const pk = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);
        if (pk.length < 6) {
            mcMsg.classList.remove('info');
            mcMsg.textContent = 'Not enough peaks in the 2-theta range (need at least 6).';
            return;
        }
        if (!isFinite(wl) || wl <= 0) {
            mcMsg.classList.remove('info');
            mcMsg.textContent = 'Invalid wavelength.';
            return;
        }

        // --- build the state object the MC expects --------------------------
        // Same shape refineAndTestSolution uses in the worker, so the tolerances
        // and figures of merit come out identical to the rest of the program.
        let mcState;
        try {
            const sorted = getSortedPeaks(pk, wl);
            const maxTth = Math.max(...pk.map(p => p.tth));
            const d_min = wl / (2 * Math.sin(maxTth * Math.PI / 360));
            if (!isFinite(d_min) || d_min <= 0) throw new Error('bad d_min');
            mcState = {
                q_obs: sorted.q_obs,
                original_indices: sorted.original_indices,
                tth_obs_rad: sorted.tth_obs_rad,
                peaks_sorted_by_q: sorted.peaks_sorted_by_q,
                N_FOR_M20: 20,
                q_max: 1 / (d_min * d_min),
                d_min: d_min,
                foundSolutions: [],
                foundSolutionMap: new Map()
            };
        } catch (err) {
            mcMsg.classList.remove('info');
            mcMsg.textContent = 'Could not prepare peak list: ' + (err.message || 'unknown');
            return;
        }

        const mcData = {
            peaks: pk, wavelength: wl, tth_error: te,
            max_volume: parseFloat(ui.maxVolume.value) || 1e9,
            impurity_peaks: imp, refineZero: rz
        };

        // --- run --------------------------------------------------------------
        // Operates on the clicked solution only. The call is synchronous and can
        // take a couple of seconds, so the button is disabled and the dialog is
        // held open with a message; one setTimeout yield lets the browser paint
        // that message before the walk blocks the thread.
        mcRunning = true;
        mcApplyBtn.disabled = true;
        mcApplyBtn.textContent = 'Running...';
        mcMsg.classList.add('info');
        mcMsg.textContent = `Refining ${nRestarts} run(s) of ${nIterations} iterations...`;

        setTimeout(() => {
            let res = null, failure = null;
            try {
                res = monteCarloRefineCell(parent, mcData, mcState, {
                    iterations: nIterations,
                    restarts: nRestarts
                });
            } catch (err) {
                failure = err && err.message ? err.message : 'unknown error';
                console.warn('MC refinement failed:', err);
            }

            mcRunning = false;
            mcApplyBtn.disabled = false;
            mcApplyBtn.textContent = 'Apply';

            if (failure) {
                mcMsg.classList.remove('info');
                mcMsg.textContent = 'Refinement failed: ' + failure;
                return;
            }
            if (!res || !isFinite(res.m20) || res.m20 <= (parent.m20 || 0)) {
                mcMsg.classList.remove('info');
                mcMsg.textContent =
                    `No improvement on M20 ${(parent.m20 || 0).toFixed(2)}. ` +
                    `Try more iterations or restarts.`;
                return;
            }

            try {
                res.analysis = analyzeSystematicAbsences(res, pk, spaceGroupData, wl, te, tMax, imp, tMin);
            } catch (err) { res.analysis = null; }

            // Place the refined cell immediately above its parent rather than at
            // the end of the list. The two are meant to be read together -- the
            // whole point is comparing the child against the cell it came from --
            // and appending would strand it wherever the ledger happens to end.
            // A higher M20 also means the default ranking would put it above the
            // parent anyway, so this matches where a re-sort would land it.
            const parentIdx = solutions.indexOf(parent);
            if (parentIdx >= 0) solutions.splice(parentIdx, 0, res);
            else solutions.push(res);
            displayedSolutions = [...solutions];
            updateSolutionsTable();
            updateAllMarkers();
            mcOverlay.classList.remove('open');
            showStatus(
                `MC refined: M20 ${(parent.m20 || 0).toFixed(2)} -> ${res.m20.toFixed(2)}`,
                'success', 8000);
        }, 0);
    });

    // 3d. Action: Space Group MC
    // The absence analysis in analyzeSystematicAbsences() judges every space
    // group against ONE cell that was refined without knowing about any of them.
    // This asks a different question: if this space group were true, how well
    // would the pattern index? Each extinction class gets its own line list, its
    // own refinement and its own statistics, so the classes are compared as
    // hypotheses rather than as violation tallies against a cell that was fitted
    // to forbidden reflections.
    //
    // Rows are hypotheses the DATA can separate, not space groups. Candidates
    // are collapsed twice: first by what they forbid arithmetically, then by
    // what they actually do to the calculated pattern for this cell, at this
    // wavelength, over this range, at this tolerance. Rule sets that differ only
    // beyond q_max or only at coincident lines end up on one row, because
    // presenting them separately - with separate figures of merit, no less - is
    // the main way a space-group table misleads.
    //
    // Ranking is a log-odds score, not M20. M20 rises whenever lines are removed
    // from the list, so it structurally rewards restriction and cannot be used
    // to choose between line lists; see the SCORING section in worker-logic.js.
    // M20 stays in the table because it describes the refined CELL, which is
    // still worth seeing.
    let sgRunning = false;
    let sgAbort = false;
    let sgRows = [];
    let sgSelectedSig = null;
    let sgSelectedMemberIdx = 0; // Tracks which specific space group chip is selected
    // Display order. The log-odds score is both the ANALYTIC authority -- it picks
    // the stage-2 shortlist, sets the tie report and estimates p -- and the
    // default ordering, so what the table shows first is what the evidence
    // actually favours. Any column can still be clicked to re-sort.
    let sgSortKey = 'score';
    let sgSortDir = -1;
    let sgParent = null;
    const sgOverlay = document.getElementById('sg-overlay');
    const sgMsg = document.getElementById('sg-msg');
    const sgRunBtn = document.getElementById('sg-run');
    const sgAddBtn = document.getElementById('sg-add');
    const sgTbody = document.getElementById('sg-tbody');

    const closeSgModal = () => {
        if (sgRunning) { sgAbort = true; return; }   // first Esc cancels the run
        sgOverlay.classList.remove('open');
    };

    // Everything the MC needs, built exactly as the Refine MC dialog builds it so
    // the figures of merit are directly comparable with the rest of the table.
    //
    // Ka2-suspect peaks are EXCLUDED here, the same way indexing excludes them.
    // They used to be left in, and a Ka2 ghost sitting a few hundredths of a
    // degree off its parent is exactly the kind of peak that lands on a
    // forbidden line and manufactures a violation against the correct rule set.
    const sgBuildInputs = () => {
        const wl = parseFloat(ui.wavelength.value);
        const te = parseFloat(ui.tthError.value);
        const tMin = parseFloat(ui.tthMinSlider.value);
        const tMax = parseFloat(ui.tthMaxSlider.value);
        const imp = getImpurityPeaks();
        const rz = !!(ui.refineZeroCheckbox && ui.refineZeroCheckbox.checked);
        const inRange = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);
        const pk = inRange.filter(p => !p.ka2Suspect);
        const nGhosts = inRange.length - pk.length;
        if (pk.length < 6) return { error: 'Not enough peaks in the 2-theta range (need at least 6).' };
        if (!isFinite(wl) || wl <= 0) return { error: 'Invalid wavelength.' };

        const sorted = getSortedPeaks(pk, wl);
        const maxTth = Math.max(...pk.map(p => p.tth));
        const d_min = wl / (2 * Math.sin(maxTth * Math.PI / 360));
        if (!isFinite(d_min) || d_min <= 0) return { error: 'Could not derive d_min from the peak list.' };

        return {
            pk, wl, te, tMin, tMax, imp, rz, nGhosts,
            state: {
                q_obs: sorted.q_obs,
                original_indices: sorted.original_indices,
                tth_obs_rad: sorted.tth_obs_rad,
                peaks_sorted_by_q: sorted.peaks_sorted_by_q,
                N_FOR_M20: 20,
                q_max: 1 / (d_min * d_min),
                d_min: d_min,
                foundSolutions: [],
                foundSolutionMap: new Map()
            },
            data: {
                peaks: pk, wavelength: wl, tth_error: te,
                max_volume: parseFloat(ui.maxVolume.value) || 1e9,
                impurity_peaks: imp, refineZero: rz,
                // The scanned limits, so a forbidden line that falls between the
                // start of the scan and the first observed peak still counts as
                // an absence. Without these the most diagnostic low-angle part
                // of the pattern carries no evidence at all.
                tth_min: tMin, tth_max: tMax
            }
        };
    };

    const sgFmt = (x, n) => (isFinite(x) ? Number(x).toFixed(n) : '-');
    const sgSigned = (x, n) => (isFinite(x) ? (x > 0 ? '+' : '') + Number(x).toFixed(n) : '-');

    // #sg-msg is plain by default and red with .error, mirroring the .info
    // convention #mc-msg uses in styles.css.
    const sgSay = (text, isError) => {
        sgMsg.textContent = text || '';
        sgMsg.classList.toggle('error', !!isError);
    };

    // Scoring options shared by every call into the ranking, so a row is never
    // scored under different assumptions than the row above it.
    let sgScoreOpts = { system: null, refineZero: false };

    const sgViolTitle = (r) => {
        if (!r.violationDetail || !r.violationDetail.length) return 'no violations';
        return r.violationDetail.map(v => {
            const bits = [`${v.tth.toFixed(3)}\u00B0`];
            if (v.eSq !== null && v.eSq !== undefined && isFinite(v.eSq)) bits.push(`|E|\u00B2 ${v.eSq.toFixed(2)}`);
            if (v.rel !== null && v.rel !== undefined) bits.push(`I/I_local ${(v.rel * 100).toFixed(0)}%`);
            if (v.dqOverTol !== null && v.dqOverTol !== undefined) bits.push(`\u0394q ${(v.dqOverTol * 100).toFixed(0)}% of tol`);
            const tags = [v.ka2 && 'Ka2', v.weak && 'weak', v.ambiguous && 'ambiguous'].filter(Boolean);
            if (tags.length) bits.push(tags.join('/'));
            return bits.join(', ');
        }).join('\n');
    };

    // Display sort. Errored rows always sink to the bottom; within the rest the
    // chosen key decides. Score is the default: it is what ranks the hypotheses,
    // so the table leads with the row the evidence favours. M20 is available as a
    // sort but describes the refined CELL rather than the hypothesis, and rises
    // whenever lines are deleted -- ordering by it puts the most restrictive
    // surviving class on top, which is not the same question. Whatever the sort,
    // the leading row on score stays marked in bold.
    const SG_SORT = {
        m20:    (r) => r.m20 || 0,
        score:  (r) => isFinite(r.score) ? r.score : -Infinity,
        clean:  (r) => r.nClean || 0,
        viol:   (r) => -(r.hardViolations || 0),
        unidx:  (r) => -(r.unindexed || 0),
        lines:  (r) => r.nLines || 0,
        zero:   (r) => Math.abs(r.zero || 0),
        label:  (r) => r.label || '',
        groups: (r) => (r.members && r.members.length) || 0,
    };

    const sgSortForDisplay = (ranked) => {
        const f = SG_SORT[sgSortKey] || SG_SORT.m20;
        const idx = new Map(ranked.map((r, i) => [r, i]));
        return ranked.slice().sort((a, b) => {
            if (a.error && !b.error) return 1;
            if (b.error && !a.error) return -1;
            const va = f(a), vb = f(b);
            if (typeof va === 'string' || typeof vb === 'string') {
                const c = String(va).localeCompare(String(vb));
                if (c) return sgSortDir < 0 ? -c : c;
            } else if (va !== vb) {
                return sgSortDir < 0 ? (vb - va) : (va - vb);
            }
            // Ties fall back to the ranked order, so sorting by score reproduces
            // sgRankRows() exactly -- including its tie-breaks on hard
            // violations, clean absences, M20 and group number.
            return idx.get(a) - idx.get(b);
        });
    };

    // One row of the table. Pure: everything it needs arrives in `ctx`, nothing is
    // read from the enclosing scope. That is deliberate -- it is the only part of
    // this dialog that can be exercised outside a browser, and it is where a
    // stale loop variable hid until it reached the console as
    // "ReferenceError: i is not defined".
    const sgRowHtml = (r, ctx) => {
        const groupsHtml = r.members.map((m, idx) => {
            const isSelected = (r.sig === ctx.selectedSig && idx === ctx.selectedMemberIdx);
            const chipCls = isSelected ? 'sg-chip selected-chip' : 'sg-chip';
            return `<span class="${chipCls}" data-idx="${idx}">${m.symbol} (${m.number})</span>`;
        }).join(' ');

        const conds = (r.conditions && r.conditions.length) ? r.conditions.join(' ; ') : 'no conditions';
        const isBest = !r.error && r === ctx.best;
        // Score is shown relative to the leader OF THE SAME TIER. Measuring a
        // falsified row against the surviving winner produced a large positive
        // delta on a row sitting at the bottom of the table -- literally
        // "+182.50" under a heading the user reads as a ranking. Falsified rows
        // are not competing on the same question, so they are ranked among
        // themselves, least-bad first, and read 0.00 at the top of their block.
        const falsified = !r.error && (r.nHardEff || 0) > 0;
        const ref = falsified ? ctx.deadScore : ctx.bestScore;
        const dScore = (ref !== null && ref !== undefined && isFinite(r.score))
            ? (r.score - ref) : NaN;

        const cls = [];
        if (r.error) cls.push('sg-bad');
        else if ((r.nHardEff || 0) > 0) cls.push('sg-ruled-out');
        if (r.sig === ctx.selectedSig) cls.push('sg-sel');
        // "best" means best on SCORE, wherever the display sort has put it.
        if (isBest) cls.push('sg-best');
        if (!r.error && r.mode !== 'mc') cls.push('sg-stage1');

        // Hard violations are the ones that count against the row; soft ones
        // (Ka2 / weak / ambiguous) are shown in parentheses so the user can see
        // them without them silently deciding the ranking.
        const soft = (r.violations || 0) - (r.hardViolations || 0);
        const violTxt = r.error ? '-'
            : `${r.hardViolations || 0}` + (soft > 0 ? ` (+${soft})` : '');

        return `<tr class="${cls.join(' ')}" data-sig="${r.sig}" title="${conds}">` +
               `<td title="${(r.mergedLabels || []).join(' \u2261 ')}">${r.label}</td>` +
               `<td class="sg-groups">${groupsHtml}</td>` +
               `<td>${r.error ? '-' : (r.nLines ?? '-')}</td>` +
               `<td>${r.error ? '-' : sgFmt(r.m20, 2)}</td>` +
               `<td class="sg-score">${r.error ? '-' : (isBest ? '0.00' : sgSigned(dScore, 2))}</td>` +
               `<td title="of ${r.nInformative ?? '-'} resolvable forbidden lines">` +
                   `${r.error ? '-' : `${r.nClean ?? '-'}/${r.nInformative ?? '-'}`}</td>` +
               `<td title="${r.error ? '' : sgViolTitle(r)}">${violTxt}</td>` +
               `<td>${r.error ? '-' : r.unindexed}</td>` +
               `<td>${r.error ? '-' : sgSigned(r.zero || 0, 3)}</td></tr>`;
    };

    const sgRenderRows = (impAllowance) => {
        const ranked = sgRankRows(sgRows, impAllowance, sgScoreOpts);
        sgRows = ranked;
        const best = ranked.find(r => !r.error && isFinite(r.score));
        // Two references, one per tier: survivors are measured against the best
        // survivor, falsified rows against the least-bad falsified row.
        const dead = ranked.find(r => !r.error && isFinite(r.score) && (r.nHardEff || 0) > 0);
        const ctx = {
            best,
            bestScore: best ? best.score : null,
            deadScore: dead ? dead.score : null,
            selectedSig: sgSelectedSig,
            selectedMemberIdx: sgSelectedMemberIdx
        };
        const displayed = sgSortForDisplay(ranked);
        sgTbody.innerHTML = displayed.map(r => sgRowHtml(r, ctx)).join('');
        const sel = sgRows.find(r => r.sig === sgSelectedSig);
        sgAddBtn.disabled = !(sel && sel.cell && !sel.error);
        // mark the sorted column in the header
        const ths = document.querySelectorAll('#sg-table thead th');
        ths.forEach(th => {
            const k = th.dataset.sort;
            th.classList.toggle('sg-sorted', k === sgSortKey);
            th.classList.toggle('sg-asc', k === sgSortKey && sgSortDir > 0);
        });
    };

    sgTbody.addEventListener('click', (e) => {
        const tr = e.target.closest('tr');
        if (!tr) return;
        
        const chip = e.target.closest('.sg-chip');
        if (chip) {
            sgSelectedSig = tr.dataset.sig;
            sgSelectedMemberIdx = parseInt(chip.dataset.idx, 10);
        } else {
            // Clicking the row but outside a chip selects the row and defaults to the first chip
            sgSelectedSig = tr.dataset.sig;
            sgSelectedMemberIdx = 0;
        }
        sgRenderRows(getImpurityPeaks());
    });

    // Click a header to re-sort; click the same one again to reverse it.
    document.querySelector('#sg-table thead').addEventListener('click', (e) => {
        const th = e.target.closest('th');
        if (!th || !th.dataset.sort) return;
        if (sgSortKey === th.dataset.sort) sgSortDir = -sgSortDir;
        else { sgSortKey = th.dataset.sort; sgSortDir = -1; }
        sgRenderRows(getImpurityPeaks());
    });

    document.getElementById('ctx-sg').addEventListener('click', () => {
        const parent = ctxTargetSolution();
        if (!parent) return;
        if (typeof sgExtinctionClasses !== 'function' || typeof monteCarloRefineCell !== 'function') {
            showStatus('Space-group MC is unavailable (worker-logic.js not loaded).', 'error', 5000);
            return;
        }
        if (!spaceGroupData || !spaceGroupData.space_groups) {
            showStatus('Space group data not loaded - cannot scan.', 'error', 5000);
            return;
        }
        if (!parent.system || typeof MC_NPAR === 'undefined' || !MC_NPAR[parent.system]) {
            showStatus('This solution has no crystal system the MC can constrain.', 'error', 4000);
            return;
        }

        sgParent = parent;
        sgRows = [];
        sgSelectedSig = null;
        sgSelectedMemberIdx = 0;
        sgTbody.innerHTML = '';
        sgAddBtn.disabled = true;
        sgRunBtn.disabled = false;
        sgRunBtn.textContent = 'Run scan';
        sgSay('', false);
        document.getElementById('sg-title').textContent =
            `Space Group MC - ${parent.system}, a=${parent.a.toFixed(4)}` +
            (parent.b ? `, b=${parent.b.toFixed(4)}` : '') +
            (parent.c ? `, c=${parent.c.toFixed(4)}` : '');
        sgOverlay.classList.add('open');
        contextMenu.style.display = 'none';
    });

    document.getElementById('sg-cancel').addEventListener('click', closeSgModal);
    sgOverlay.addEventListener('click', (e) => { if (e.target === sgOverlay) closeSgModal(); });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && sgOverlay.classList.contains('open')) closeSgModal();
    });

    sgRunBtn.addEventListener('click', () => {
        if (sgRunning) { sgAbort = true; return; }
        const parent = sgParent;
        if (!parent) { closeSgModal(); return; }

        const readInt = (id, lo, hi, dflt) => {
            const el = document.getElementById(id);
            const v = parseInt(el.value, 10);
            return (isFinite(v) && v >= lo && v <= hi) ? v : dflt;
        };
        const nIter = readInt('sg-iterations', 50, 20000, 600);
        const nRes  = readInt('sg-restarts', 1, 20, 4);
        const topN  = readInt('sg-top', 1, 50, 8);

        const inp = sgBuildInputs();
        if (inp.error) { sgSay(inp.error, true); return; }
        sgScoreOpts = { system: parent.system, refineZero: inp.rz };

        let classes;
        try {
            // No centering pre-filter on purpose. The point of this scan is to be
            // an independent second opinion, and pre-filtering with the same
            // centering test that feeds the ranking would just import its verdict.
            classes = sgExtinctionClasses(spaceGroupData, parent.system, null);
        } catch (err) {
            sgSay('Could not build the candidate list: ' + (err.message || err), true);
            return;
        }
        if (!classes.length) { sgSay('No space groups found for this crystal system.', true); return; }

        // Collapse candidates that produce the SAME calculated pattern for this
        // cell, wavelength, range and tolerance. Rule sets that differ only
        // beyond q_max, or only at lines that coincide with an allowed one
        // inside the matching window, are not distinguishable here and must not
        // be presented as separate hypotheses with separate figures of merit.
        const frame = sgLatticeFrame(parent, inp.data, inp.state);
        if (!frame) { sgSay('The parent cell generates no lines in this range.', true); return; }
        const nAbstract = classes.length;
        try { classes = sgObservableMerge(classes, frame); }
        catch (err) { sgSay('Could not merge indistinguishable classes: ' + (err.message || err), true); return; }

        const ictx = sgIntensityContext(inp.state.peaks_sorted_by_q);
        // Wilson scale from the peak heights, built once from the unfiltered
        // lattice so no hypothesis can rescale its own evidence. Null whenever
        // the heights cannot support it, in which case the scoring falls back to
        // the single global p exactly as before.
        let wilson = null;
        try { wilson = sgWilsonContext(frame, inp.data, inp.state, parent.system); }
        catch (err) { wilson = null; }
        const shared = { frame, ictx, wilson };
        // sgRescoreAll() needs the frame (for the shared position count that
        // fixes epsBase for every row alike) and the Wilson context (to re-solve
        // lambda against the same reference class it takes p from). Both are
        // properties of the pattern, not of a row, so they travel with the
        // scoring options.
        sgScoreOpts = { ...sgScoreOpts, frame, wilson };

        sgRunning = true; sgAbort = false;
        sgRows = []; sgSelectedSig = null; sgSelectedMemberIdx = 0;
        sgRunBtn.textContent = 'Stop';
        sgAddBtn.disabled = true;

        // Stage 1 gives every class ONE constrained least-squares refit against
        // its own restricted line list. The old stage 1 scored every class on
        // the parent cell as it stood, which re-imported the exact bias this
        // module exists to remove: a cell refined without knowing about any
        // extinctions is pulled toward forbidden reflections, so the correct
        // class can be demoted at stage 1 and never reach the Monte-Carlo pass
        // at all. One LS solve per class is cheap and breaks that loop.
        const STAGE1_CHUNK = 4;
        let i = 0;

        const noteFor = (ranked) => {
            const best = ranked.find(r => !r.error);
            if (!best) return 'Nothing to report.';
            // Every class contradicted by a real reflection: say so plainly rather
            // than dressing the least-bad row up as an answer.
            if (!sgAnySurvivor(ranked)) {
                return `No class survives: every one is contradicted by at least one strong, ` +
                       `unambiguous peak on a line it forbids. The rows are ordered least-bad first ` +
                       `(${best.label} has ${best.hardViolations}). Check the cell, the 2-theta range, ` +
                       `or raise the impurity-peak allowance if the pattern really does contain ` +
                       `foreign lines.`;
            }
            const info = sgMarginInfo(ranked);
            const margin = info.margin;
            // The tie set has to be measured inside the SURVIVING tier, the same
            // way sgMargin() measures the margin. Filtering the whole table meant
            // a falsified class could be reported as tied with the winner -- and
            // the test was one-sided, so a falsified row scoring 180 nats ABOVE
            // the winner satisfied "(best.score - r.score) < 2.3" and was listed
            // as indistinguishable from it. Ranked order is already tier-then-
            // score, so the leader of the tier is simply its first element.
            //
            // The tier is additionally restricted to rows refined to the same
            // DEPTH as the leader (see sgComparableTier): a stage-1 row and a
            // Monte-Carlo row differ by how much compute they were given as
            // well as by hypothesis, so a lead over one of them is not the same
            // statement as a lead over the other.
            const tier = info.tier.length ? info.tier : [best];
            const lead = tier[0] || best;
            const tied = tier.filter(r => Math.abs(lead.score - r.score) < SG_DECISIVE_NATS);
            const cmpTxt = info.restricted
                ? ` Compared among the ${info.nCompared} fully refined class(es);` +
                  ` stage-1 rows are least-squares only and are not directly comparable.`
                : '';
            const capTxt = (lead.cleanCapped)
                ? ` The leader's absence evidence is at the saturation ceiling` +
                  ` (${(SG_CLEAN_CAP_MULT * SG_DECISIVE_NATS).toFixed(0)} nats), so its lead rests on` +
                  ` violations and unindexed peaks rather than on how much it forbids.`
                : '';
            const pTxt = (wilson
                ? ` Absences weighted per reflection (Wilson, ${wilson.nUsed} peaks, R2 ${wilson.r2.toFixed(2)}).`
                : (isFinite(best.pHat) ? ` p(line observed) = ${(best.pHat * 100).toFixed(0)}%.` : ''))
                + cmpTxt + capTxt;
            if (tied.length > 1) {
                return `No decisive winner: ${tied.length} classes lie within ` +
                       `${SG_DECISIVE_NATS.toFixed(1)} nats of each other ` +
                       `(${tied.map(r => r.label).join(', ')}).` + pTxt +
                       ` The absences in this pattern cannot separate them.`;
            }
            // sgMargin() returns Infinity when the tier has a single member, which
            // used to print as "ahead of the runner-up by Infinity nats (1.1e+13:1)".
            const marginTxt = isFinite(margin)
                ? `ahead of the runner-up by ${margin.toFixed(1)} nats ` +
                  `(${Math.exp(Math.min(30, margin)).toPrecision(2)}:1)`
                : `the only class the data do not contradict`;
            return `Best: ${lead.label} - ${marginTxt}, ` +
                   `${lead.nClean}/${lead.nInformative} forbidden lines clean, ` +
                   `${lead.hardViolations} hard violation(s).` + pTxt +
                   ` Select a row and press "Add as solution" to keep its refined cell.`;
        };

        const finish = (note) => {
            sgRunning = false;
            sgRunBtn.textContent = 'Run scan';
            sgRunBtn.disabled = false;
            sgRenderRows(inp.imp);
            sgSay(note || noteFor(sgRows), false);
        };

        // Which classes deserve the expensive refinement?
        //
        // Taking the top N by score alone repeats the stage-1 problem in a
        // subtler form: the shortlist is chosen from cells that have only had a
        // single LS step, so a class whose true cell needs the annealing walk can
        // still be cut. Three guarantees go on top of the top-N, in priority
        // order:
        //
        //   1. the most permissive class -- the "no extinction" null hypothesis
        //      the whole score is measured against;
        //   2. the class with the most clean absences, which is precisely the one
        //      a contaminated parent cell is most likely to have hidden;
        //   3. one representative per centering type.
        //
        // The extras are capped, because each one costs a full Monte-Carlo run
        // and the user set "Refine top" expecting a bounded wait. If the cap
        // binds, the lower-priority centerings are the ones dropped.
        const SHORTLIST_EXTRA_MAX = 4;
        const pickShortlist = (ranked, n) => {
            const ok = ranked.filter(r => !r.error);
            const out = [];
            const seen = new Set();
            const push = (r, isExtra) => {
                if (!r || seen.has(r.sig)) return false;
                if (isExtra && out.length >= n + SHORTLIST_EXTRA_MAX) return false;
                seen.add(r.sig); out.push(r); return true;
            };
            for (const r of ok.slice(0, n)) push(r, false);
            push(ok.slice().sort((a, b) => (a.nRules || 0) - (b.nRules || 0))[0], true);
            push(ok.slice().sort((a, b) => (b.nClean || 0) - (a.nClean || 0))[0], true);
            const byCent = new Map();
            for (const r of ok) if (!byCent.has(r.centering)) byCent.set(r.centering, r);
            for (const r of byCent.values()) push(r, true);
            return out;
        };

        const stage2 = () => {
            const ranked = sgRankRows(sgRows, inp.imp, sgScoreOpts);
            const shortlist = pickShortlist(ranked, topN);
            let j = 0;
            const step2 = () => {
                if (sgAbort) return finish('Stopped. The classes not yet refined are stage-1 (least-squares) results only.');
                if (j >= shortlist.length) return finish(null);
                const cls = classes.find(c => c.sig === shortlist[j].sig);
                sgSay(`Refining ${j + 1}/${shortlist.length}: ${shortlist[j].label} ...`, false);
                if (cls) {
                    const scored = sgScoreClass(cls, parent, inp.data, inp.state, {
                        ...shared, mode: 'mc', iterations: nIter, restarts: nRes
                    });
                    const at = sgRows.findIndex(r => r.sig === cls.sig);
                    if (at >= 0) sgRows[at] = scored; else sgRows.push(scored);
                }
                j++;
                sgRenderRows(inp.imp);
                setTimeout(step2, 0);
            };
            setTimeout(step2, 0);
        };

        const step1 = () => {
            if (sgAbort) return finish('Stopped.');
            if (i >= classes.length) {
                sgSay(`Stage 1 done: ${nAbstract} settings collapse to ${classes.length} ` +
                      `distinguishable classes here. Refining the best ${topN} (plus coverage) ...`, false);
                return stage2();
            }
            const end = Math.min(i + STAGE1_CHUNK, classes.length);
            for (; i < end; i++) {
                sgRows.push(sgScoreClass(classes[i], parent, inp.data, inp.state, { ...shared, mode: 'ls' }));
            }
            sgSay(`Scoring extinction classes: ${i}/${classes.length} ...`, false);
            sgRenderRows(inp.imp);
            setTimeout(step1, 0);
        };

        const ghostNote = inp.nGhosts ? ` (${inp.nGhosts} Ka2-suspect peak(s) excluded)` : '';
        const wilsonNote = wilson
            ? ` Intensity weighting active (Wilson fit on ${wilson.nUsed} peaks, R2 ${wilson.r2.toFixed(2)}).`
            : ' Intensity weighting unavailable - absences weighted uniformly.';
        sgSay(`Scoring ${classes.length} distinguishable classes from ${nAbstract} settings${ghostNote}.` +
              wilsonNote, false);
        setTimeout(step1, 0);
    });

    sgAddBtn.addEventListener('click', () => {
        const row = sgRows.find(r => r.sig === sgSelectedSig);
        const parent = sgParent;
        if (!row || !row.cell || !parent) return;

        const inp = sgBuildInputs();
        if (inp.error) { sgSay(inp.error, true); return; }

        const child = { ...row.cell };
        child.system = parent.system;
        child.manualSwaps = parent.manualSwaps || [];
        // Provenance: which hypothesis produced this cell. The cell was refined
        // against the restricted line list, so it is only meaningful alongside
        // the label -- and alongside the margin, since a cell taken from a row
        // that was in a three-way tie is not a determination.
        child.sgClass = row.label;
        
        // Grab only the specific chip the user selected
        const selectedMember = row.members[sgSelectedMemberIdx] || row.members[0];
        child.sgMembers = [`${selectedMember.symbol} (${selectedMember.number})`];
        
        child.sgConditions = row.conditions;
        child.sgScore = row.score;
        child.sgMargin = sgMargin(sgRows);
        child.sgEvidence = {
            wilson: !!(row.stats && row.stats.cleanP && row.stats.cleanP.length),
            clean: row.nClean, informative: row.nInformative,
            hardViolations: row.hardViolations, softViolations: (row.violations || 0) - (row.hardViolations || 0),
            unindexed: row.unindexed, pHat: row.pHat, mode: row.mode
        };

        try {
            child.analysis = analyzeSystematicAbsences(child, inp.pk, spaceGroupData,
                                                       inp.wl, inp.te, inp.tMax, inp.imp, inp.tMin);
        } catch (err) { child.analysis = null; }

        const parentIdx = solutions.indexOf(parent);
        if (parentIdx >= 0) solutions.splice(parentIdx, 0, child);
        else solutions.push(child);
        displayedSolutions = [...solutions];
        updateSolutionsTable();
        updateAllMarkers();
        sgOverlay.classList.remove('open');
        const marg = child.sgMargin;
        const margTxt = isFinite(marg)
            ? (marg < SG_DECISIVE_NATS ? ` - NOT decisive (${marg.toFixed(1)} nats over the runner-up)` : ` - ${marg.toFixed(1)} nats ahead`)
            : '';
        showStatus(`Added cell refined under ${row.label} (${selectedMember.symbol}): ` +
                   `M20 ${sgFmt(row.m20, 2)}, ${row.hardViolations} hard violation(s)${margTxt}.`, 'success', 8000);
    });

    // 4. Action: Export pdCIF
    document.getElementById('ctx-pdcif').addEventListener('click', () => {
        const target = ctxTargetSolution();
        if (target) {
            generatePdCIF(target);
        }
    });

    const generatePdCIF = (sol) => {
        // Fetch original wavelength based on preset to match the raw data export
        let wl = parseFloat(ui.wavelength.value) || 1.54184;
        let isDoublet = false;
        let wlKa1, wlKa2, wlRatio;
        
        const presetVal = ui.wavelengthPreset.value;
        if (presetVal !== 'custom') {
            const [element, type] = presetVal.split('_');
            const data = WAVELENGTH_PRESETS[element];
            if (data) {
                if (type === 'avg') {
                    isDoublet = true;
                    wlKa1 = data.ka1;
                    wlKa2 = data.ka2;
                    wlRatio = data.ratio;
                } else {
                    wl = data.ka1;
                }
            }
        }
        
        // Brutus: 2th_corr = 2th_obs - Z. pdCIF: 2th_calc = 2th_meas + offset.
        const pdCifOffset = -(sol.zero_correction || 0);

        let cif = `data_brutus_solution\n\n`;
        cif += `_audit_creation_method 'Brutus'\n`;
        cif += `_cell_length_a       ${sol.a.toFixed(5)}\n`;
        cif += `_cell_length_b       ${(sol.b || sol.a).toFixed(5)}\n`;
        cif += `_cell_length_c       ${(sol.c || sol.a).toFixed(5)}\n`;
        cif += `_cell_angle_alpha    ${(sol.alpha || 90).toFixed(3)}\n`;
        cif += `_cell_angle_beta     ${(sol.beta || 90).toFixed(3)}\n`;
        cif += `_cell_angle_gamma    ${(sol.gamma || 90).toFixed(3)}\n`;
        cif += `_cell_volume         ${sol.volume.toFixed(3)}\n\n`;
        
        // Extract space group and IT number
        let sgName = '?';
        let sgNumber = '?';
        if (sol.sgMembers && sol.sgMembers.length > 0) {
            const parts = sol.sgMembers[0].split(' (');
            sgName = parts[0];
            if (parts.length > 1) {
                sgNumber = parts[1].replace(')', '');
            }
        } else if (sol.sgClass) {
            sgName = sol.sgClass;
        }

        cif += `_space_group_name_H-M_alt       ${sgName}\n`;
        if (sgNumber !== '?') {
            cif += `_space_group_IT_number          ${sgNumber}\n`;
        }
        cif += `\n`;
        
        cif += `_pd_calib_2theta_offset         ${pdCifOffset.toFixed(5)}\n\n`;

        if (isDoublet) {
            cif += `loop_\n`;
            cif += `    _diffrn_radiation_wavelength_id\n`;
            cif += `    _diffrn_radiation_wavelength\n`;
            cif += `    _diffrn_radiation_wavelength_wt\n`;
            cif += `  1  ${wlKa1.toFixed(6)}  1.0\n`;
            cif += `  2  ${wlKa2.toFixed(6)}  ${wlRatio.toFixed(5)}\n\n`;
        } else {
            cif += `_diffrn_radiation_wavelength    ${wl.toFixed(6)}\n\n`;
        }
        
        cif += `_diffrn_radiation_probe         x-ray\n\n`;

        cif += `loop_\n`;
        cif += `_pd_meas_2theta_scan\n`;
        cif += `_pd_meas_intensity_total\n`;
        
        // Write raw data (limited to the indexing range)
        const tMin = parseFloat(ui.tthMinSlider.value);
        const tMax = parseFloat(ui.tthMaxSlider.value);
        const tth = fullExperimentalData.tth;
        const int = fullExperimentalData.intensity;
        for (let i = 0; i < tth.length; i++) {
            if (tth[i] >= tMin && tth[i] <= tMax) {
                cif += `${tth[i].toFixed(5)} ${Math.round(int[i])}\n`;
            }
        }

        // Trigger download
        const blob = new Blob([cif], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${baseNameNoExt(loadedFileName) || 'pattern'}_sol.CIF`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };

    // 5. Action: Single Report
    document.getElementById('ctx-report').addEventListener('click', () => {
        const target = ctxTargetSolution();
        if (target) {
            // Call report function with the specific solution
            generatePDFReport(target);
        }
    });


    // ===================== AXIS SYSTEM =====================================
    // Everything the program computes with stays in its natural units: peak
    // positions in 2-theta degrees, intensities in raw counts. The plot is a
    // *view* on top of that. xF/yF map an internal quantity onto whatever the
    // user selected; xInv maps a plotted abscissa back to 2-theta for
    // hit-testing. Chart.js is always handed a plain linear scale, even in log
    // mode, so pixel math, floating-bar markers and getValueForPixel behave
    // identically in every combination - only the tick formatter converts back
    // to real units. Nothing downstream of the plot ever sees a transformed
    // value, so indexing, refinement and the report are untouched by the view.
    const LOG_FLOOR = 0.1;                 // counts: keeps log10 finite at zero
    const DEG_PER_RAD = 180 / Math.PI;
    const getLambda = () => parseFloat(ui.wavelength.value) || 1.54184;

    const AXIS_X = {
        tth: {
            label: '2\u03B8 (degrees)', reverse: false, digits: 3,
            f: (t) => t,
            inv: (v) => v
        },
        theta: {
            label: '\u03B8 (degrees)', reverse: false, digits: 3,
            f: (t) => t / 2,
            inv: (v) => v * 2
        },
        d: {
            // Reversed on purpose: large d belongs on the left so the pattern
            // keeps the same left-to-right shape it has in 2-theta.
            label: 'd-spacing (\u00C5)', reverse: true, digits: 4,
            f: (t) => getLambda() / (2 * Math.sin(Math.max(t, 1e-4) * Math.PI / 360)),
            inv: (v) => {
                const s = getLambda() / (2 * Math.max(v, 1e-9));
                return s >= 1 ? 180 : 2 * Math.asin(s) * DEG_PER_RAD;
            }
        },
        q: {
            label: 'Q = 4\u03C0\u00B7sin\u03B8/\u03BB (\u00C5\u207B\u00B9)', reverse: false, digits: 4,
            f: (t) => 4 * Math.PI * Math.sin(Math.max(t, 1e-4) * Math.PI / 360) / getLambda(),
            inv: (v) => {
                const s = Math.max(v, 0) * getLambda() / (4 * Math.PI);
                return s >= 1 ? 180 : 2 * Math.asin(s) * DEG_PER_RAD;
            }
        }
    };

    const AXIS_Y = {
        linear: { label: 'Intensity (a.u.)',        f: (i) => i,                          inv: (v) => v },
        sqrt:   { label: '\u221AIntensity (a.u.)',  f: (i) => Math.sqrt(Math.max(0, i)),  inv: (v) => v * v },
        log:    { label: 'Intensity (a.u., log)',   f: (i) => Math.log10(Math.max(i, LOG_FLOOR)), inv: (v) => Math.pow(10, v) }
    };

    let xAxisMode = 'tth';    // 2-theta by default
    let yAxisMode = 'sqrt';   // sqrt(I) by default: shows weak lines without hiding strong ones

    const xAx = () => AXIS_X[xAxisMode] || AXIS_X.tth;
    const yAx = () => AXIS_Y[yAxisMode] || AXIS_Y.sqrt;
    const xF   = (t) => xAx().f(t);
    const xInv = (v) => xAx().inv(v);
    const yF   = (i) => yAx().f(i);

    // The 2-theta interval currently on screen, ordered low-to-high whatever
    // direction the axis runs in.
    const visibleTthRange = () => {
        if (!xrdChart) return [-Infinity, Infinity];
        const a = xInv(xrdChart.scales.x.min);
        const b = xInv(xrdChart.scales.x.max);
        return a <= b ? [a, b] : [b, a];
    };

    // Points carry their untransformed tth and I so tooltips and hit-testing
    // never have to invert anything.
    const buildExperimentalPoints = () => workingExperimentalData.tth.map((t, i) => {
        const I = Math.max(0, workingExperimentalData.intensity[i]);
        return { x: xF(t), y: yF(I), tth: t, I };
    });

    // The ONLY supported way to push the measured pattern into the chart.
    // Writing datasets[0].data directly is what let raw counts land on a
    // transformed axis: the tick formatter then inverted values that had never
    // been transformed, so a 1e4 count was labelled 1e8 on the sqrt scale, and
    // the trace was drawn with the wrong shape into the bargain. Everything
    // that changes the working data - file load, wavelength, Ka2 stripping,
    // deconvolution - goes through here instead of touching the dataset.
    const setExperimentalTrace = (rescaleY = false) => {
        if (!xrdChart || !workingExperimentalData || !workingExperimentalData.tth.length) return;
        xrdChart.data.datasets[0].data = buildExperimentalPoints();
        if (rescaleY) {
            const yb = yBoundsFor(workingExperimentalData.intensity);
            xrdChart.options.scales.y.min = yb.min;
            xrdChart.options.scales.y.max = yb.max;
        }
        xrdChart.update('none');
    };

    // Y bounds expressed in transformed space, from raw intensities.
    const yBoundsFor = (intensities) => {
        const raw = maxOfArray(intensities) || 1000;
        if (yAxisMode === 'log') {
            const top = Math.log10(Math.max(raw, LOG_FLOOR));
            const bot = Math.log10(LOG_FLOOR);
            const span = Math.max(top - bot, 1);
            return { min: bot - span * 0.02, max: top + span * 0.05 };
        }
        const top = yF(raw);
        return { min: -top * 0.05, max: top * 1.1 };
    };

    // Snap indicator: a dashed guide at the position a Ctrl+click would use.
    Chart.register({
        id: 'snapIndicator',
        afterDraw: chart => {
            const tth = chart.$snapTth;
            if (typeof tth !== 'number' || !isFinite(tth)) return;
            const xs = chart.scales.x, ys = chart.scales.y;
            const xv = xF(tth);
            if (xv < xs.min || xv > xs.max) return;
            const x = xs.getPixelForValue(xv);
            const ctx = chart.ctx;
            ctx.save();
            ctx.strokeStyle = chart.$snapKind === 'hkl' ? 'rgba(37, 99, 235, 0.9)' : 'rgba(16, 185, 129, 0.9)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
            ctx.setLineDash([]);
            ctx.beginPath(); ctx.arc(x, ys.top + 7, 4, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
        }
    });

    Chart.register({ id: 'verticalCursorLine', afterDraw: chart => { if (chart.tooltip?._active?.length) { let x = chart.tooltip._active[0].element.x; let yAxis = chart.scales.y; let ctx = chart.ctx; ctx.save(); ctx.beginPath(); ctx.moveTo(x, yAxis.top); ctx.lineTo(x, yAxis.bottom); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(156, 163, 175, 0.7)'; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.restore(); } } });
    // Plugin that draws the "currently-edited peak" indicator as a thin
    // amber band across the full plot height. Implemented as a plugin
    // (not a bar dataset) because Chart.js bar geometry adds a small
    // centering offset that shifts the bar visibly to the right of the
    // peak's true 2θ; getPixelForValue is exact.
    // The chart instance carries the 2θ to draw via `chart.$selectedPeakTth`,
    // which updateAllMarkers sets each time.
    Chart.register({
        id: 'selectedPeakLine',
        afterDraw: chart => {
            const tth = chart.$selectedPeakTth;
            if (typeof tth !== 'number' || !isFinite(tth)) return;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            const xv = xF(tth);
            if (xv < xScale.min || xv > xScale.max) return;
            const x = xScale.getPixelForValue(xv);
            const ctx = chart.ctx;
            ctx.save();
            ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
            ctx.fillRect(x - 2, yScale.top, 4, yScale.bottom - yScale.top);
            ctx.restore();
        }
    });
    Chart.register({ id: 'legendMargin', beforeInit(chart) { const originalFit = chart.legend.fit; chart.legend.fit = function() { originalFit.bind(chart.legend)(); this.height += 15; }; } });

    //  initializeChart
    //  Rebuilt from scratch on every axis-mode change: switching the abscissa
    //  rewrites every x value in every dataset, and switching the ordinate
    //  rewrites the y bounds, so mutating in place would leave stale geometry.
    const initializeChart = () => {
        if (xrdChart) xrdChart.destroy();
        if (!workingExperimentalData || !workingExperimentalData.tth.length) return;

        const experimentalPoints = buildExperimentalPoints();
        const yb = yBoundsFor(workingExperimentalData.intensity);
        const ax = xAx(), ay = yAx();

        xrdChart = new Chart(ui.chartCanvas, {
            type: 'line',
            data: {
                datasets: [
                    { label: 'Intensity', data: experimentalPoints, borderColor: 'rgba(107, 114, 128, 0.7)', showLine: true, borderWidth: 0.75, pointRadius: 1.5, pointHoverRadius: 4, pointBackgroundColor: 'rgba(107, 114, 128, 0.7)' },
                    { type: 'bar', label: 'Observed Peaks', data: [], backgroundColor: 'rgba(239, 68, 68, 0.7)', barThickness: 1 },
                    { type: 'bar', label: 'Calculated Peaks', data: [], backgroundColor: 'rgba(59, 130, 246, 0.9)', barThickness: 1 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: {
                    x: {
                        type: 'linear',
                        reverse: ax.reverse,
                        title: { display: true, text: ax.label },
                        offset: false,
                        ticks: {
                            includeBounds: false,
                            callback: (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: ax.digits })
                        },
                        grid: { drawTicks: true, drawBorder: true }
                    },
                    y: {
                        title: { display: true, text: ay.label },
                        min: yb.min,
                        max: yb.max,
                        offset: false,
                        ticks: {
                            includeBounds: false,
                            // Ticks are spaced in transformed space but labelled with the
                            // intensity they actually stand for, so a sqrt or log plot is
                            // still read in counts rather than in sqrt-counts or decades.
                            callback: (value) => {
                                const real = ay.inv(value);
                                if (!isFinite(real)) return '';
                                if (Math.abs(real) >= 100000) return real.toExponential(1);
                                return real.toLocaleString(undefined, { maximumFractionDigits: Math.abs(real) < 10 ? 2 : 0 });
                            }
                        },
                        grid: { drawTicks: true, drawBorder: true }
                    }
                },
                plugins: {
                    zoom: {
                        pan: { 
                            enabled: true, 
                            mode: 'xy', 
                            modifierKey: 'alt',
                            onPanComplete: () => { updateAllMarkers(); } 
                        },
                        zoom: { 
                            wheel: { enabled: true }, 
                            pinch: { enabled: true }, 
                            drag: { 
                                enabled: true,
                                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                                borderColor: 'rgba(59, 130, 246, 0.5)',
                                borderWidth: 1
                            }, 
                            mode: 'xy',
                            onZoomComplete: () => { updateAllMarkers(); } 
                        }
                    },

                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            // Every abscissa flavour is shown at once, so switching the
                            // X axis is a change of layout, never a loss of information.
                            title: function(tooltipItems) {
                                if (!tooltipItems.length) return '';
                                const item = tooltipItems[0];
                                const raw = item.raw || {};
                                const tth = (typeof raw.tth === 'number') ? raw.tth : xInv(item.parsed.x);
                                const lam = getLambda();
                                const st = Math.sin(tth * Math.PI / 360);
                                const dsp = st > 0 ? lam / (2 * st) : NaN;
                                const Q = 4 * Math.PI * st / lam;
                                const lines = [
                                    `2\u03B8 ${tth.toFixed(4)}\u00B0   \u03B8 ${(tth / 2).toFixed(4)}\u00B0`,
                                    `d ${isFinite(dsp) ? dsp.toFixed(5) : '-'} \u00C5   Q ${Q.toFixed(4)} \u00C5\u207B\u00B9`
                                ];
                                const di = item.datasetIndex;
                                if ((di === 1 || di === 2) && currentHklList && currentHklList.length) {
                                    let best = null, bd = Infinity;
                                    for (const hkl of currentHklList) {
                                        const diff = Math.abs(tth - hkl.tth);
                                        if (diff < bd) { bd = diff; best = hkl; }
                                    }
                                    // A calculated tick is its own line, so it must match
                                    // essentially exactly; an observed peak only has to be
                                    // inside the user's stated 2-theta error.
                                    const tol = (di === 2) ? 1e-4 : (parseFloat(ui.tthError.value) || 0.04);
                                    if (best && bd < tol) {
                                        lines.push(`hkl (${best.h},${best.k},${best.l})`);
                                        if (di === 1) lines.push(`\u0394 from calc: ${(tth - best.tth).toFixed(4)}\u00B0`);
                                    }
                                }
                                return lines;
                            },
                            label: function(context) {
                                const datasetLabel = context.dataset.label || '';
                                if (datasetLabel === 'Observed Peaks' || datasetLabel === 'Calculated Peaks') return null;
                                const raw = context.raw || {};
                                const I = (typeof raw.I === 'number') ? raw.I : yAx().inv(context.parsed.y);
                                return `Intensity: ${Math.round(I)}`;
                            }
                        }
                    }
                }
            }
        });
    };


    const updateAllMarkers = () => {
        if (!xrdChart) return;
        // Filter in 2-theta, plot in whatever the axis currently is.
        const [tthLo, tthHi] = visibleTthRange();
        const yMin = xrdChart.scales.y.min; const yMax = xrdChart.scales.y.max;
        const yRange = yMax - yMin;
        const markerHeight = yRange * 0.02;

        const visibleObsPeaks = pickedPeaks.filter(p => p.tth >= tthLo && p.tth <= tthHi);
        xrdChart.data.datasets[1].data = visibleObsPeaks.map(p => ({ x: xF(p.tth), y: [yMin, yMin + markerHeight], tth: p.tth }));

        if (selectedSolution && currentHklList.length) {
            const calculatedBottom = yMin + markerHeight * 1.2;
            const calculatedTop = calculatedBottom + markerHeight;
            const visibleCalcPeaks = currentHklList.filter(hkl => hkl.tth >= tthLo && hkl.tth <= tthHi);
            xrdChart.data.datasets[2].data = visibleCalcPeaks.map(hkl => ({ x: xF(hkl.tth), y: [calculatedBottom, calculatedTop], tth: hkl.tth }));
        } else {
            xrdChart.data.datasets[2].data = [];
        }

        // Selected-peak indicator: stash the 2-theta on the chart instance; the
        // selectedPeakLine plugin converts it through the current axis on each
        // draw, which is exact (no bar-centering offset). undefined hides it.
        if (selectedPeakIndex !== null && pickedPeaks[selectedPeakIndex]) {
            xrdChart.$selectedPeakTth = pickedPeaks[selectedPeakIndex].tth;
        } else {
            xrdChart.$selectedPeakTth = undefined;
        }
        xrdChart.update('none');
    };

    // ---- Calculated line list ------------------------------------------------
    // The measured pattern is plotted exactly as recorded, zero error and all.
    // The refined cell, on the other hand, describes the sample AFTER the zero
    // error has been taken out: 2theta_corrected = 2theta_observed - Z. So to
    // put a calculated line next to the observed peak it explains, it has to be
    // pushed back into the observed frame, 2theta_plot = 2theta_calc + Z.
    // Skipping that step leaves every blue tick displaced from its red partner
    // by exactly Z - small, uniform, and easily mistaken for a bad cell. The
    // untouched value is kept as tth_calc for anything that wants the corrected
    // frame instead.
    const rebuildHklList = () => {
        if (!selectedSolution || !workingExperimentalData.tth.length) { currentHklList = []; return; }
        const lambda = getLambda();
        const zero = selectedSolution.zero_correction || 0;
        const dataMax = workingExperimentalData.tth[workingExperimentalData.tth.length - 1];
        // Generate past the end of the data by |Z| + 1 so that shifting the list
        // cannot leave the top of the pattern bare.
        const maxTth = Math.min(179.9, dataMax + Math.abs(zero) + 1);
        const raw = generateHKL(maxTth, { ...selectedSolution, lambda }, selectedSolution.system) || [];
        currentHklList = raw.map(r => ({ ...r, tth_calc: r.tth, tth: r.tth + zero }));
    };

    // Single entry point for "this solution is now the displayed one", so the
    // line list can never drift out of step with the highlighted table row.
    const applySolutionSelection = (sol) => {
        selectedSolution = sol || null;
        rebuildHklList();
        updateAllMarkers();
    };

    const rebuildPlot = (resetY = true) => {
        if (!workingExperimentalData || !workingExperimentalData.tth.length) return;
        initializeChart();
        rebuildHklList();
        updatePlotRange(resetY);
    };

    ui.xAxisMode.addEventListener('change', () => { xAxisMode = ui.xAxisMode.value; rebuildPlot(true); });
    ui.yAxisMode.addEventListener('change', () => { yAxisMode = ui.yAxisMode.value; rebuildPlot(true); });
    ui.snapMode.addEventListener('change', () => {
        if (xrdChart) { xrdChart.$snapTth = undefined; ui.snapReadout.textContent = ''; xrdChart.render(); }
    });

    // ---- Snapping ------------------------------------------------------------
    // A pixel is a poor estimate of a peak position: at typical zoom one pixel
    // is a few hundredths of a degree, which is the same size as the tolerance
    // the indexer works to. So a Ctrl+click snaps to something real - the
    // nearest local maximum of the measured trace, or the nearest calculated
    // hkl line of the selected solution, whichever is closer. Hold Shift to
    // place a peak exactly where you clicked instead.
    const SNAP_WINDOW_DEG = 0.35;

    const nearestDataIndex = (tth) => {
        const arr = workingExperimentalData.tth;
        if (!arr || !arr.length) return -1;
        let lo = 0, hi = arr.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < tth) lo = mid + 1; else hi = mid; }
        if (lo > 0 && Math.abs(arr[lo - 1] - tth) <= Math.abs(arr[lo] - tth)) lo--;
        return lo;
    };

    // Highest measured point within the snap window, found by scanning outward
    // from the click rather than hill-climbing, so a click landing in a local
    // dip between two shoulders still finds the real maximum.
    const snapToData = (tth) => {
        const X = workingExperimentalData.tth, Y = workingExperimentalData.intensity;
        if (!X || !X.length) return null;
        const i0 = nearestDataIndex(tth);
        if (i0 < 0) return null;
        let best = i0;
        for (let i = i0; i < X.length && X[i] - tth <= SNAP_WINDOW_DEG; i++) if (Y[i] > Y[best]) best = i;
        for (let i = i0; i >= 0 && tth - X[i] <= SNAP_WINDOW_DEG; i--) if (Y[i] > Y[best]) best = i;
        return { tth: X[best], height: Math.max(0, Y[best]), kind: 'data' };
    };

    const snapToHkl = (tth) => {
        if (!currentHklList || !currentHklList.length) return null;
        let best = null, bd = Infinity;
        for (const h of currentHklList) { const d = Math.abs(h.tth - tth); if (d < bd) { bd = d; best = h; } }
        if (!best || bd > SNAP_WINDOW_DEG) return null;
        return { tth: best.tth, kind: 'hkl', hkl: best };
    };

    const resolveSnap = (tth, mode) => {
        if (mode === 'data') return snapToData(tth);
        if (mode === 'hkl')  return snapToHkl(tth);
        return null; // 'off'
    };

    const tthFromEvent = (e) => {
        if (!xrdChart || !xrdChart.chartArea) return null;
        const rect = xrdChart.canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        if (px < xrdChart.chartArea.left || px > xrdChart.chartArea.right) return null;
        const v = xrdChart.scales.x.getValueForPixel(px);
        if (v === undefined || v === null || !isFinite(v)) return null;
        const tth = xInv(v);
        return isFinite(tth) ? Math.max(1e-4, tth) : null;
    };

    ui.chartCanvas.addEventListener('mousemove', (e) => {
        if (!xrdChart) return;
        const tth = tthFromEvent(e);
        const hit = (tth === null) ? null : resolveSnap(tth, ui.snapMode.value);
        const newTth = hit ? hit.tth : undefined;
        const newKind = hit ? hit.kind : undefined;
        if (tth === null) {
            if (xrdChart.$snapTth !== undefined) { xrdChart.$snapTth = undefined; xrdChart.render(); }
            ui.snapReadout.textContent = '';
            return;
        }
        const shown = hit ? hit.tth : tth;
        const lam = getLambda();
        const st = Math.sin(shown * Math.PI / 360);
        const dsp = st > 0 ? lam / (2 * st) : NaN;
        const Q = 4 * Math.PI * st / lam;
        let txt = `2\u03B8 ${shown.toFixed(3)}\u00B0   d ${isFinite(dsp) ? dsp.toFixed(4) : '-'} \u00C5   Q ${Q.toFixed(3)} \u00C5\u207B\u00B9`;
        if (hit && hit.kind === 'hkl') txt += `   \u2192 hkl (${hit.hkl.h},${hit.hkl.k},${hit.hkl.l})`;
        ui.snapReadout.textContent = txt;
        // Only repaint when the snap target actually moved: mousemove fires far
        // faster than a full-pattern redraw can keep up with.
        if (newTth !== xrdChart.$snapTth || newKind !== xrdChart.$snapKind) {
            xrdChart.$snapTth = newTth;
            xrdChart.$snapKind = newKind;
            xrdChart.render();
        }
    });

    ui.chartCanvas.addEventListener('mouseleave', () => {
        if (!xrdChart) return;
        ui.snapReadout.textContent = '';
        if (xrdChart.$snapTth !== undefined) { xrdChart.$snapTth = undefined; xrdChart.render(); }
    });

    const resizer = document.getElementById('drag-handle'); const leftPanel = document.getElementById('controls-panel');
    // Bounds are read from the panel's own CSS (min-width / max-width) so the
    // JS clamp can never drift from the stylesheet. Falls back to the declared
    // 300/700 if the computed values are unavailable for any reason.
    const panelCS = getComputedStyle(leftPanel);
    const PANEL_MIN = parseFloat(panelCS.minWidth) || 300;
    const PANEL_MAX = parseFloat(panelCS.maxWidth) || 700;
    resizer.addEventListener('mousedown', (e) => { e.preventDefault(); document.body.style.cursor = 'col-resize';
        const moveHandler = (moveEvent) => {
            // Clamp to [PANEL_MIN, PANEL_MAX] and also keep the results area from
            // being squeezed below a usable width. The old code enforced neither
            // the CSS max nor a hard min consistent with the stylesheet.
            const maxByViewport = window.innerWidth - 350;
            const upper = Math.min(PANEL_MAX, maxByViewport);
            const width = Math.max(PANEL_MIN, Math.min(moveEvent.clientX, upper));
            leftPanel.style.width = `${width}px`;
        };
        const upHandler = () => { document.body.style.cursor = 'default'; window.removeEventListener('mousemove', moveHandler); window.removeEventListener('mouseup', upHandler); };
        window.addEventListener('mousemove', moveHandler); window.addEventListener('mouseup', upHandler);
    });
    
    ui.chartCanvas.addEventListener('contextmenu', e => { e.preventDefault(); if (xrdChart) { xrdChart.resetZoom('none'); updateAllMarkers(); } });
    
    
    
ui.chartCanvas.addEventListener('click', (e) => {
        if (!e.ctrlKey || !xrdChart) return;
        const tthRaw = tthFromEvent(e);
        if (tthRaw === null) return;
        // Shift is the deliberate escape hatch from snapping.
        const hit = e.shiftKey ? null : resolveSnap(tthRaw, ui.snapMode.value);
        const tth = Math.max(1e-4, hit ? hit.tth : tthRaw);

        // A hand-added peak needs a height like any other: the space-group
        // analysis uses it to separate a weak reflection from a strong one, and
        // a peak inserted without one reads back as undefined and is scored as
        // though it had no intensity at all.
        let height = (hit && typeof hit.height === 'number') ? hit.height : 0;
        if (!height) {
            const di = nearestDataIndex(tth);
            if (di >= 0) height = Math.max(0, workingExperimentalData.intensity[di]);
        }

        pickedPeaks.push({ tth, d: 0, q: 0, height });
        pickedPeaks.sort((a, b) => a.tth - b.tth);
        recalculatePeakValues(); // tag + d/q in one shot
        updatePeakTable(); updateStartIndexingButtonState();
        const how = hit
            ? (hit.kind === 'hkl' ? ` (snapped to hkl ${hit.hkl.h}${hit.hkl.k}${hit.hkl.l})` : ' (snapped to observed peak)')
            : '';
        showStatus(`Peak added at ${tth.toFixed(3)}\u00B0${how}`, 'success', 2000);
    });



//main report function, on peut optimiser...
const generatePDFReport = async (singleSolution = null) => {
    const solutionsToReport = singleSolution ? [singleSolution] : displayedSolutions;    
    
    if (displayedSolutions.length === 0) {
        showStatus("No solutions found to generate a report.", 'info');
        return;
    }

    const tthMinVal = parseFloat(ui.tthMinSlider.value);
    const tthMaxVal = parseFloat(ui.tthMaxSlider.value);
    const reportPeaks = pickedPeaks.filter(p => p.tth >= tthMinVal && p.tth <= tthMaxVal);

    if (reportPeaks.length === 0) {
         showStatus("No peaks selected in the current 2-theta range for the report.", 'info');
         return;
    }

    // Safeguard: Ensure every solution being reported has undergone space group analysis
    if (spaceGroupData) {
        solutionsToReport.forEach(sol => {
            if (!sol.analysis) {
                sol.analysis = analyzeSystematicAbsences(
                    sol,
                    reportPeaks,
                    spaceGroupData,
                    parseFloat(ui.wavelength.value),
                    parseFloat(ui.tthError.value),
                    tthMaxVal,
                    getImpurityPeaks(),
                    tthMinVal
                );
            }
        });
    }

    ui.reportButton.textContent = 'Generating...';
    ui.reportButton.disabled = true;
    document.body.style.cursor = 'wait';
    
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF({
            orientation: 'p',
            unit: 'mm',
            format: 'a4'
        });
    
        const margin = 15;
        let yPos = 20;
        const pdfWidth = doc.internal.pageSize.getWidth();
        const lambda = parseFloat(ui.wavelength.value);
        const _activeKa2Preset = (typeof getActiveKa2Preset === 'function') ? getActiveKa2Preset() : null;
        const lambdaKa1 = _activeKa2Preset ? _activeKa2Preset.ka1 : null;
        const lambdaKa2 = _activeKa2Preset ? _activeKa2Preset.ka2 : null;
        const tthError = parseFloat(ui.tthError.value);
            
        const FONT = {
            TITLE: 'helvetica',
            LABEL: 'helvetica',
            DATA: 'courier'
        };
        const SIZE = {
            TITLE: 18,
            H1: 14,
            H2: 12,
            BODY: 9,
            TABLE_HEADER: 8,
            TABLE_BODY: 8,
            SMALL: 7
        };
    
        // header
        const now = new Date();
        const timestamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
        const versionInfo = document.getElementById('app-footer')?.textContent || 'Brutus, 22 april 2026';
        const programURL = window.location.href;
            
        doc.setFont(FONT.TITLE, 'bold').setFontSize(SIZE.TITLE).text('Brutus - Powder Indexing Report', pdfWidth / 2, yPos, { align: 'center' });
        yPos += 10;
            
        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY);
        doc.text(`Generated:`, margin, yPos);
        doc.setFont(FONT.DATA, 'normal').text(timestamp, margin + 25, yPos);
        yPos += 5;
    
        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY);
        doc.text(`URL:`, margin, yPos);
        doc.setFont(FONT.DATA, 'normal').text(programURL, margin + 25, yPos);
        yPos += 5;
    
        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY);
        doc.text(`Version:`, margin, yPos);
        doc.setFont(FONT.DATA, 'normal').text(versionInfo, margin + 25, yPos);
        yPos += 5;
            
        doc.setFont(FONT.LABEL, 'normal').text(`Data File:`, margin, yPos);
        // Read the variable, not the DOM. The filename now lives in a chip with
        // sibling nodes (size, clear button), so textContent would drag "248 KB"
        // into the report's Data file line.
        doc.setFont(FONT.DATA, 'normal').text(loadedFileName || '', margin + 25, yPos);
        yPos += 10;
          
        const imgData = xrdChart.toBase64Image('image/png', 1.0);      
        const imgProps = doc.getImageProperties(imgData);
        const availableWidth = pdfWidth - 2 * margin;
        const pdfHeight = doc.internal.pageSize.getHeight();
        const availableHeight = pdfHeight - yPos - margin - 5; 
        const scale = Math.min(availableWidth / imgProps.width, availableHeight / imgProps.height);
        const drawWidth = imgProps.width * scale;
        const drawHeight = imgProps.height * scale;
        const drawX = margin + (availableWidth - drawWidth) / 2; 
        doc.addImage(imgData, 'PNG', drawX, yPos, drawWidth, drawHeight);
    
        // Parameters
        doc.addPage();
        yPos = 20;
    
        doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.H1).text('Indexing Parameters', margin, yPos);
        yPos += 8;
    
        const presetText = ui.wavelengthPreset.options[ui.wavelengthPreset.selectedIndex].text;
        const paramData = [
            { label: 'Radiation:', value: presetText },
            { label: 'Max Volume (A^3):', value: ui.maxVolume.value },
            { label: 'Wavelength (A):', value: parseFloat(ui.wavelength.value).toFixed(5) },
            { label: 'Tolerance (2theta):', value: ui.tthError.value },
            { label: 'Ka2 Identified:',
              value: getActiveKa2Preset()
                         ? (ui.stripKa2Checkbox.checked ? 'True' : 'False')
                         : 'N/A' },
            { label: 'Impurity Peaks:', value: ui.impurityPeaksInput.value },
            { label: 'Min Peak (%):', value: ui.peakThresholdValue.textContent },
            { label: 'Refine Zero:', value: ui.refineZeroCheckbox.checked ? 'True' : 'False' },
            { label: '2theta Min (deg):', value: tthMinVal.toFixed(2) },
            { label: '2theta Max (deg):', value: tthMaxVal.toFixed(2) },
        ];
    
        const col1X = margin;
        const col2X = margin + 85;
        const labelWidth = 35;
    
        paramData.forEach((item, index) => {
            const isCol1 = index % 2 === 0;
            const x = isCol1 ? col1X : col2X;
            if (isCol1) yPos += 5;
            
            doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY).text(item.label, x, yPos);
            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(String(item.value), x + labelWidth, yPos);
        });
        yPos += 7;
              
        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.BODY).text('Systems Searched:', margin, yPos);
        const systems = Array.from(ui.systemCheckboxes).filter(cb => cb.checked).map(cb => cb.value.charAt(0).toUpperCase() + cb.value.slice(1));
        const systemsText = systems.join(', ');
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(systemsText, margin + labelWidth, yPos);
        yPos += 8;
    
        doc.setDrawColor(200); doc.line(margin, yPos, pdfWidth - margin, yPos); yPos += 8;
    
        doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.H1).text('Indexing Solutions Summary', margin, yPos); yPos += 8;
              
        if (lastIndexingStats) {
            const cleanedStats = lastIndexingStats
                .replace(/\u00A0/g, ' ')
                .replace(/\s+/g, ' ');

            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
            doc.text(cleanedStats, margin, yPos);
            yPos += 7; 
        }
              
        doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.TABLE_HEADER);
        doc.text('Sys', margin, yPos);
        const first_sol_n_20 = (displayedSolutions.length > 0 && displayedSolutions[0].n_20) ? displayedSolutions[0].n_20 : Math.min(20, reportPeaks.length);
        doc.text(`M(${first_sol_n_20})`, margin + 15, yPos);
        doc.text(`F(${first_sol_n_20})`, margin + 30, yPos);
        doc.text('Volume(A^3)', margin + 45, yPos);
        doc.text('Parameters', margin + 75, yPos);
        yPos += 5;
    
        doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
        solutionsToReport.slice(0, 30).forEach(sol => {
            if (yPos > 280) { doc.addPage(); yPos = 20; }
            let paramStr = '';
            const p = sol.errors || {};
            switch (sol.system) {
                 case 'cubic': paramStr = `a=${formatWithError(sol.a, p.s_a)}`; break;
                 case 'tetragonal': paramStr = `a=${formatWithError(sol.a, p.s_a)}, c=${formatWithError(sol.c, p.s_c)}`; break;
                 case 'hexagonal': paramStr = `a=${formatWithError(sol.a, p.s_a)}, c=${formatWithError(sol.c, p.s_c)}`; break;
                 case 'orthorhombic': paramStr = `a=${formatWithError(sol.a, p.s_a)}, b=${formatWithError(sol.b, p.s_b)}, c=${formatWithError(sol.c, p.s_c)}`; break;
                 case 'monoclinic': paramStr = `a=${formatWithError(sol.a, p.s_a)}, b=${formatWithError(sol.b, p.s_b)}, c=${formatWithError(sol.c, p.s_c)}, beta=${formatWithError(sol.beta, p.s_beta)}`; break;
                 case 'triclinic': 
                    paramStr = `a=${formatWithError(sol.a, p.s_a)}, b=${formatWithError(sol.b, p.s_b)}, c=${formatWithError(sol.c, p.s_c)}`;
                    doc.text(sol.system.substring(0,4), margin, yPos);
                    doc.text(sol.m20.toFixed(2), margin + 15, yPos);
                    doc.text((sol.fN_20 || 0).toFixed(2), margin + 30, yPos);
                    doc.text(sol.volume.toFixed(2), margin + 45, yPos);
                    doc.text(paramStr, margin + 75, yPos);
                    yPos += 5; 
                    paramStr = `al=${formatWithError(sol.alpha, p.s_alpha)}, be=${formatWithError(sol.beta, p.s_beta)}, ga=${formatWithError(sol.gamma, p.s_gamma)}`;
                    doc.text(paramStr, margin + 75, yPos); 
                    yPos += 5;
                    return; 
            }
            doc.text(sol.system.substring(0,4), margin, yPos);
            doc.text(sol.m20.toFixed(2), margin + 15, yPos);
            doc.text((sol.fN_20 || 0).toFixed(2), margin + 30, yPos);
            doc.text(sol.volume.toFixed(2), margin + 45, yPos);
            doc.text(paramStr, margin + 75, yPos);
            yPos += 5;
        });
    
        // ------------------------------------------------------------------
        // SPACE-GROUP DETERMINATION BLOCK
        //
        // Reproduces the Space Group MC's verdict from the fields it stamped on
        // the solution. The report deliberately performs NO ranking of its own:
        // the two used to disagree because they answered different questions
        // (likelihood ratio over merged extinction CLASSES, each with its own
        // refit, versus a violation tally over individual SETTINGS judged
        // against one extinction-blind cell) on different data (the MC excludes
        // Ka2-suspect peaks; the report's analysis does not) from different
        // candidate pools (the report's is pre-filtered by the centering test,
        // the MC's is not, on purpose).
        //
        // Nothing here recomputes: if a value was not produced by an MC run it
        // is not shown, and the absence of a determination is stated plainly.
        const writeSgVerdict = (sol) => {
            const wrap = (text, x, size, style) => {
                doc.setFont(FONT.DATA, style || 'normal').setFontSize(size);
                doc.splitTextToSize(text, pdfWidth - x - margin).forEach(l => {
                    if (yPos > 280) { doc.addPage(); yPos = 20; }
                    doc.text(l, x, yPos);
                    yPos += (size <= SIZE.SMALL) ? 3.5 : 4;
                });
            };

            if (yPos > 258) { doc.addPage(); yPos = 20; }
            doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2)
               .text('Space Group Determination:', margin, yPos);
            yPos += 6;

            if (!sol.sgClass) {
                wrap('No space-group determination was performed for this cell. The list below states ' +
                     'which settings the observed absences are compatible with; it does not choose ' +
                     'between them. Run "Space Group MC" on this solution to compare the extinction ' +
                     'classes as hypotheses, then add the result as a solution to have it appear here.',
                     margin + 5, SIZE.BODY, 'italic');
                yPos += 3;
                return;
            }

            wrap(`Class: ${sol.sgClass}`, margin + 5, SIZE.BODY, 'bold');
            if (sol.sgMembers && sol.sgMembers.length) {
                wrap(`Space groups in this class: ${sol.sgMembers.join(', ')}`, margin + 5, SIZE.BODY);
            }
            if (sol.sgConditions && sol.sgConditions.length) {
                wrap(`Reflection conditions: ${sol.sgConditions.join(' ; ')}`, margin + 5, SIZE.BODY);
            }

            // The margin, in the MC's own units, with the decisiveness verdict
            // spelled out. A cell taken from a row that was in a tie is not a
            // determination, and the report must not let the bold class name
            // above imply that it was.
            const marg = sol.sgMargin;
            const decisiveNats = (typeof SG_DECISIVE_NATS === 'number') ? SG_DECISIVE_NATS : 2.3;
            if (isFinite(marg)) {
                const odds = Math.exp(Math.min(30, marg));
                if (marg >= decisiveNats) {
                    wrap(`Margin: ${marg.toFixed(1)} nats ahead of the runner-up ` +
                         `(about ${odds.toPrecision(2)}:1).`, margin + 5, SIZE.BODY);
                } else {
                    doc.setTextColor(150, 60, 0);
                    wrap(`NOT DECISIVE: only ${marg.toFixed(1)} nats ahead of the runner-up ` +
                         `(about ${odds.toPrecision(2)}:1), below the ${decisiveNats.toFixed(1)}-nat ` +
                         `threshold. The absences in this pattern do not separate this class from the ` +
                         `next one; the cell below is refined under this hypothesis, not proof of it.`,
                         margin + 5, SIZE.BODY);
                    doc.setTextColor(0, 0, 0);
                }
            } else if (marg === Infinity) {
                wrap('Margin: the only class the data do not contradict.', margin + 5, SIZE.BODY);
            }

            const ev = sol.sgEvidence;
            if (ev) {
                const bits = [];
                if (isFinite(ev.clean) && isFinite(ev.informative)) {
                    bits.push(`${ev.clean}/${ev.informative} forbidden lines clean`);
                }
                if (isFinite(ev.hardViolations)) {
                    bits.push(`${ev.hardViolations} hard violation(s)` +
                              (ev.softViolations ? ` (+${ev.softViolations} soft)` : ''));
                }
                if (isFinite(ev.unindexed)) bits.push(`${ev.unindexed} unindexed peak(s)`);
                if (bits.length) wrap(`Evidence: ${bits.join(', ')}.`, margin + 5, SIZE.BODY);

                // How the number was arrived at matters as much as the number.
                const how = [];
                how.push(ev.wilson
                    ? 'absences weighted per reflection (Wilson |E|^2)'
                    : 'absences weighted uniformly (no intensity weighting)');
                if (isFinite(ev.pHat)) how.push(`p(line observed) = ${(ev.pHat * 100).toFixed(0)}%`);
                how.push(ev.mode === 'mc'
                    ? 'cell refined by Monte-Carlo under this hypothesis'
                    : 'cell refined by least squares only (stage 1; not directly comparable with fully refined classes)');
                wrap(`Method: ${how.join('; ')}.`, margin + 5, SIZE.SMALL, 'italic');
            }
            if (isFinite(sol.sgScore)) {
                wrap(`Log-odds score: ${sol.sgScore.toFixed(2)} nats (relative; only differences are meaningful).`,
                     margin + 5, SIZE.SMALL, 'italic');
            }
            yPos += 3;
            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
        };

        // Detailed Solution 
        solutionsToReport.forEach((sol, solIndex) => {
            doc.addPage(); yPos = 20;
            
            doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.H1); 
            doc.text(`Details for Solution #${solIndex + 1}: ${sol.system}`, margin, yPos); 
            yPos += 8; 
            
            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY); 
            const p = sol.errors || {};
            const paramLines = [];
            
            switch (sol.system) {
                case 'cubic':
                    paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                    break;
                case 'tetragonal':
                    paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                    paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                    break;
                case 'hexagonal':
                    paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                    paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                    break;
                case 'orthorhombic':
                    paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                    paramLines.push({ label: 'b', value: `= ${formatWithError(sol.b, p.s_b)} A` });
                    paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                    break;
                case 'monoclinic':
                    paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                    paramLines.push({ label: 'b', value: `= ${formatWithError(sol.b, p.s_b)} A` });
                    paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                    paramLines.push({ label: 'beta', value: `= ${formatWithError(sol.beta, p.s_beta)} deg` });
                    break;
                case 'triclinic':
                    paramLines.push({ label: 'a', value: `= ${formatWithError(sol.a, p.s_a)} A` });
                    paramLines.push({ label: 'b', value: `= ${formatWithError(sol.b, p.s_b)} A` });
                    paramLines.push({ label: 'c', value: `= ${formatWithError(sol.c, p.s_c)} A` });
                    paramLines.push({ label: 'alpha', value: `= ${formatWithError(sol.alpha, p.s_alpha)} deg` });
                    paramLines.push({ label: 'beta', value: `= ${formatWithError(sol.beta, p.s_beta)} deg` });
                    paramLines.push({ label: 'gamma', value: `= ${formatWithError(sol.gamma, p.s_gamma)} deg` });
                    break;
            }
            
            paramLines.push({ label: 'Volume', value: `= ${sol.volume.toFixed(2)} A^3` });
        
            if (sol.zero_correction !== undefined) { 
                paramLines.push({ label: 'Zero Error (2theta)', value: `= ${formatWithError(sol.zero_correction, p.s_zero)} deg` });
            }
            
            const n_20_pdf = sol.n_20 || Math.min(20, reportPeaks.length);
            paramLines.push({ label: `M(${n_20_pdf})`, value: `= ${sol.m20.toFixed(2)}` });
            paramLines.push({ label: `F(${n_20_pdf})`, value: `= ${(sol.fN_20 || 0).toFixed(2)}` });

            const n_all_pdf = sol.n_all || reportPeaks.length;
            paramLines.push({ label: `M(${n_all_pdf})`, value: `= ${(sol.m_all || 0).toFixed(2)}` });
            paramLines.push({ label: `F(${n_all_pdf})`, value: `= ${(sol.fN_all || 0).toFixed(2)}` });
                     
            const longestLabelWidth = Math.max(...paramLines.map(line => doc.getTextWidth(line.label)));
            const labelEndX = margin + longestLabelWidth;
            const dataStartX = labelEndX + 2; 
        
            paramLines.forEach(line => {
                doc.text(line.label, labelEndX, yPos, { align: 'right' });
                doc.text(line.value, dataStartX, yPos);
                yPos += 4;
            });
        
            yPos += 3; 

            if (sol.analysis && sol.analysis.centering) {
                doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text(`Lattice Centering:`, margin, yPos);
                doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(sol.analysis.centering, margin + 42, yPos);
                yPos += 7;
            }

            if (yPos > 225) { doc.addPage(); yPos = 20; }

            try {
                const r = Math.PI / 180.0;
                const cellVolume = (cl) => {
                    const ca = Math.cos(cl.alpha * r), cb = Math.cos(cl.beta * r), cg = Math.cos(cl.gamma * r);
                    const term = Math.max(0, 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
                    return (term > 0) ? (cl.a * cl.b * cl.c * Math.sqrt(term)) : 0;
                };

                const col1LabelEndX = margin + 15;
                const col1DataStartX = col1LabelEndX + 2;
                const col2LabelEndX = margin + 78;
                const col2DataStartX = col2LabelEndX + 2;

                const drawCellBlock = (title, cl, subtitle) => {
                    doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text(title, margin, yPos);
                    yPos += 5;
                    if (subtitle) {
                        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.SMALL);
                        doc.setTextColor(90, 90, 90);
                        doc.text(subtitle, margin, yPos);
                        doc.setTextColor(0, 0, 0);
                        yPos += 5;
                    }

                    const d = [
                        { label: 'a',     value: `= ${cl.a.toFixed(4)} A` },
                        { label: 'b',     value: `= ${cl.b.toFixed(4)} A` },
                        { label: 'c',     value: `= ${cl.c.toFixed(4)} A` },
                        { label: 'alpha', value: `= ${cl.alpha.toFixed(3)} deg` },
                        { label: 'beta',  value: `= ${cl.beta.toFixed(3)} deg` },
                        { label: 'gamma', value: `= ${cl.gamma.toFixed(3)} deg` }
                    ];

                    doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);

                    for (let i = 0; i < 3; i++) {
                        doc.text(d[i].label, col1LabelEndX, yPos, { align: 'right' });
                        doc.text(d[i].value, col1DataStartX, yPos);
                        doc.text(d[i + 3].label, col2LabelEndX, yPos, { align: 'right' });
                        doc.text(d[i + 3].value, col2DataStartX, yPos);
                        yPos += 4;
                    }

                    doc.text("Volume", col1LabelEndX, yPos, { align: 'right' });
                    doc.text(`= ${cellVolume(cl).toFixed(2)} A^3`, col1DataStartX, yPos);
                    yPos += 6;
                };

                const niggliResult = reduceToNiggliCell(sol);
                const nCell = niggliResult.cell;
                const nVol = cellVolume(nCell);

                const centeringStr = String((sol.analysis && sol.analysis.centering) || '').toUpperCase();
                let centeringKey = null;   
                for (const cType of ['F', 'I', 'R', 'A', 'B', 'C', 'P']) {
                    if (centeringStr.includes(`(${cType})`)) { centeringKey = cType; break; }
                }

                let subtitle1, title1;
                if (centeringKey === 'P') {
                    title1 = 'Reduced (Niggli) Cell:';
                    subtitle1 = 'Krivy-Gruber reduction of the solved lattice metric. Lattice is primitive, so this IS the reduced (Niggli) cell.';
                } else if (centeringKey === null) {
                    title1 = 'Standardised Conventional Cell:';
                    subtitle1 = 'Krivy-Gruber reduction of the solved lattice metric. Centering undetermined, so no primitive cell is derived.';
                } else {
                    title1 = 'Standardised Conventional Cell (centering not applied):';
                    subtitle1 = `Krivy-Gruber reduction of the conventional basis only; the detected ${centeringKey}-centering is NOT applied, so this is not the reduced cell of the lattice.`;
                }

                drawCellBlock(title1, nCell, subtitle1);

                if (centeringKey && centeringKey !== 'P') {
                    try {
                        const primResult = reduceToNiggliCell(sol, { centering: centeringKey });
                        const pCell = primResult.cell;
                        const pVol = cellVolume(pCell);
                        const ratio = (pVol > 0) ? (nVol / pVol) : 0;

                        drawCellBlock(
                            'Reduced (Niggli) Cell - primitive:',
                            pCell,
                            `Reduced cell of the ${centeringKey}-centered lattice, centering applied`
                                + (ratio > 0 ? `; volume is 1/${ratio.toFixed(0)} of the cell above.` : '.')
                        );

                        if (primResult.converged === false) {
                            doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.SMALL);
                            doc.setTextColor(180, 0, 0);
                            doc.text('Warning: reduction did not converge; cell above may not be fully reduced.', margin, yPos);
                            doc.setTextColor(0, 0, 0);
                            yPos += 5;
                        }
                    } catch (ePrim) {
                        doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.SMALL);
                        doc.setTextColor(180, 0, 0);
                        doc.text(`Reduced (Niggli) cell could not be computed for ${centeringKey}-centering.`, margin, yPos);
                        doc.setTextColor(0, 0, 0);
                        yPos += 6;
                    }
                }

                const symmetryOrder = { 'cubic': 6, 'hexagonal': 5, 'tetragonal': 4, 'orthorhombic': 3, 'monoclinic': 2, 'triclinic': 1 };
                const currentOrder = symmetryOrder[sol.system] || 1;
                const niggliSym = getSymmetry(nCell.a, nCell.b, nCell.c, nCell.alpha, nCell.beta, nCell.gamma, 0.25);
                const detectedHigherSyms = [];
                
                if (symmetryOrder[niggliSym] > currentOrder) {
                    detectedHigherSyms.push({
                        system: niggliSym,
                        note: `Metric: ${niggliSym}`,
                        cell: nCell
                    });
                }
                
                const checkSystems = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'monoclinic'];
                checkSystems.forEach(sys => {
                    if (symmetryOrder[sys] > currentOrder) {
                        const equiv = generateEquivalentCells(nCell, 0, sys);
                        if (equiv && equiv.centeredCells) {
                            Object.values(equiv.centeredCells).forEach(cCell => {
                                if (symmetryOrder[cCell.system] > currentOrder) {
                                    detectedHigherSyms.push({
                                        system: cCell.system,
                                        note: `${cCell.centering}-centered ${cCell.system}`,
                                        cell: cCell
                                    });
                                }
                            });
                        }
                    }
                });

                if (detectedHigherSyms.length > 0) {
                    const uniqSyms = Array.from(new Map(detectedHigherSyms.map(item => [item.note, item])).values());
                    
                    if (yPos > 255) { doc.addPage(); yPos = 20; }
                    
                    doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.BODY);
                    doc.setTextColor(220, 38, 38); 
                    doc.text("HIGHER SYMMETRY DETECTED IN CONVENTIONAL CELL METRIC", margin, yPos);
                    doc.setTextColor(0, 0, 0); 
                    yPos += 4.5;
                    
                    doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.SMALL);
                    doc.text(`This ${sol.system} cell reduces to a metric tensor consistent with higher symmetry:`, margin + 2, yPos);
                    yPos += 4;
                    
                    uniqSyms.forEach(sym => {
                        if (yPos > 275) { doc.addPage(); yPos = 20; }
                        const c = sym.cell;
                        let paramStr = `a=${c.a.toFixed(3)}, b=${c.b.toFixed(3)}, c=${c.c.toFixed(3)}`;
                        if (sym.system !== 'cubic' && sym.system !== 'orthorhombic' && sym.system !== 'tetragonal') {
                            paramStr += `, al=${c.alpha.toFixed(2)}, be=${c.beta.toFixed(2)}, ga=${c.gamma.toFixed(2)}`;
                        }
                        const labelStr = `* ${sym.note.toUpperCase()}:`;
                        doc.setFont(FONT.DATA, 'bold');
                        doc.text(labelStr, margin + 4, yPos);
                        doc.setFont(FONT.DATA, 'normal');
                        const labelW = doc.getTextWidth(labelStr);
                        const paramX = Math.max(margin + 50, margin + 4 + labelW + 3);
                        if (paramX + doc.getTextWidth(paramStr) > pdfWidth - margin) {
                            yPos += 3.5;
                            doc.text(paramStr, margin + 8, yPos);
                        } else {
                            doc.text(paramStr, paramX, yPos);
                        }
                        yPos += 4;
                    });
                    
                    doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                    doc.text("Note: Consider refining in the higher symmetry system to reduce degrees of freedom.", margin + 2, yPos);
                    doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
                    yPos += 4;
                }

            } catch (niggliError) {
                console.error("Niggli reduction failed:", niggliError);
                doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.BODY);
                doc.text('Reduction failed. See console for error.', margin + 5, yPos);
                yPos += 5;
            }

            yPos += 4;

            if (sol.analysis) {
                 doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text('Possible Extinctions:', margin, yPos); 
                 let extinctionList = sol.analysis.detectedExtinctions || [];
                 let extinctionText = "";

                 if (extinctionList.length > 0 && extinctionList[0] !== "None detected") {
                      extinctionText = extinctionList.join(', '); 
                 } else {
                      extinctionText = "None clearly detected";
                      doc.setFont(FONT.DATA, 'italic'); 
                 }

                 const extinctionX = margin + 42;
                 const extinctionMaxWidth = pdfWidth - extinctionX - margin; 
                 const extinctionLines = doc.splitTextToSize(extinctionText, extinctionMaxWidth);

                 doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY); 
                 extinctionLines.forEach(line => {
                      if (yPos > 280) { 
                           doc.addPage();
                           yPos = 20;
                      }
                      doc.text(line, extinctionX, yPos);
                      yPos += 4; 
                 });
                 yPos += 3; 

                 // --- SPACE-GROUP DETERMINATION -------------------------------
                 // The determination is the Space Group MC's, not this report's.
                 // It is reproduced verbatim from the fields the MC stamped onto
                 // the solution when the user pressed "Add as solution"; the
                 // report does not re-derive, re-order or second-guess it. When
                 // no MC was run the report says so, rather than substituting
                 // the weaker violation-tally ordering that used to sit here and
                 // silently disagree with what was on screen.
                 writeSgVerdict(sol);

                 const sgCompatible = sol.analysis.compatibleSettings ||
                                      sol.analysis.rankedSpaceGroups || [];
                 if (sgCompatible.length > 0) {
                     if (yPos > 260) { doc.addPage(); yPos = 20; }
                     doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2)
                        .text('Space groups compatible with the observed absences:', margin, yPos);
                     yPos += 6;

                     doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                     const listNote = doc.splitTextToSize(
                         'Not a ranking. Settings are grouped by how many observed reflections contradict them ' +
                         'and listed by space-group number within each group. Ordering candidates by agreement ' +
                         'counts is unreliable, because a less-constrained setting can never score worse than a ' +
                         'more-constrained one it contains; use the Space Group MC to compare hypotheses.',
                         pdfWidth - 2 * margin - 5);
                     listNote.forEach(l => {
                         if (yPos > 280) { doc.addPage(); yPos = 20; }
                         doc.text(l, margin + 5, yPos);
                         yPos += 3.5;
                     });
                     yPos += 1.5;
                     doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);

                     if (sol.analysis.usedKa2SoftScoring) {
                         doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                         const note = '(Hard = violations by real reflections. Soft = from Ka2-suspect/weak peaks only, shown for reference.)';
                         doc.text(note, margin + 5, yPos);
                         yPos += 4;
                         doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
                     }

                     const sgList = sgCompatible;
                     const groupsByViolation = {};
                     sgList.forEach(sg => {
                         const v = sg.hardViolations || 0;
                         if (!groupsByViolation[v]) groupsByViolation[v] = [];
                         groupsByViolation[v].push(sg);
                     });

                     const violationBuckets = Object.keys(groupsByViolation).map(Number).sort((a, b) => a - b);

                     const printViolationList = (list, header, italic) => {
                         if (!list || list.length === 0) return;
                         if (yPos > 280) { doc.addPage(); yPos = 20; }
                         doc.setFont(FONT.DATA, italic ? 'italic' : 'normal').setFontSize(SIZE.SMALL);
                         doc.text(header, margin + 10, yPos);
                         yPos += 3.5;
                         list.forEach(viol => {
                             if (yPos > 280) { doc.addPage(); yPos = 20; }
                             const violationLines = doc.splitTextToSize(`- ${viol}`, pdfWidth - margin - margin - 15);
                             violationLines.forEach(vl => {
                                 if (yPos > 280) { doc.addPage(); yPos = 20; }
                                 doc.text(vl, margin + 12, yPos);
                                 yPos += 3.5;
                             });
                         });
                     };

                     violationBuckets.forEach(v => {
                         if (yPos > 270) { doc.addPage(); yPos = 20; }
                         doc.setFont(FONT.LABEL, 'bold').setFontSize(SIZE.BODY).text(`[${v} hard violation${v !== 1 ? 's' : ''}]:`, margin, yPos);
                         yPos += 5;

                         groupsByViolation[v].forEach(sg => {
                             if (yPos > 280) { doc.addPage(); yPos = 20; }

                             const softTag = sg.softViolations > 0 ? `  [+${sg.softViolations} soft]` : '';
                             const extTag = (sg.extinctionsTotal > 0)
                                 ? `  [explains ${sg.extinctionsExplained}/${sg.extinctionsTotal} absences]` : '';
                             const acentricTag = (sg.centrosymmetric === false) ? '  [acentric]' : '';
                             doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.TABLE_BODY);
                             doc.text(`${sg.number}: ${sg.symbol}${softTag}${extTag}${acentricTag}`, margin + 5, yPos);
                             yPos += 4;

                             const unexplained = sg.extinctionsUnexplained || [];
                             if (unexplained.length > 0) {
                                 if (yPos > 280) { doc.addPage(); yPos = 20; }
                                 doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                                 doc.setTextColor(150, 60, 0);
                                 doc.text(`Does not explain: ${unexplained.join('; ')}`, margin + 10, yPos);
                                 doc.setTextColor(0, 0, 0);
                                 yPos += 4;
                             }

                             const hardList = sg.violatedReflectionsHard || [];
                             const softList = sg.violatedReflectionsSoft || [];

                             printViolationList(hardList, 'Hard violations (reflections observed on lines this setting forbids):', false);
                             printViolationList(softList, 'Soft violations (Ka2-suspect/weak; shown for reference only):', true);

                             doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
                             yPos += 2;
                         });
                         yPos += 2;
                     });

                     // The list is capped. Say so explicitly: a truncated list
                     // read as complete is how "the correct group is not even a
                     // candidate" gets mistaken for "the correct group was ruled
                     // out".
                     const nTotal = sol.analysis.compatibleSettingsTotal;
                     if (isFinite(nTotal) && nTotal > sgList.length) {
                         if (yPos > 280) { doc.addPage(); yPos = 20; }
                         doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                         doc.text(`Showing ${sgList.length} of ${nTotal} compatible settings ` +
                                  `(lowest space-group numbers first within each violation group).`,
                                  margin + 5, yPos);
                         doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY);
                         yPos += 4;
                     }
                 }
                 yPos += 3; 

                 if (sol.analysis.centeringViolations && Object.keys(sol.analysis.centeringViolations).length > 0) {
                      if (yPos > 268) { doc.addPage(); yPos = 20; }
                      doc.setFont(FONT.LABEL, 'normal').setFontSize(SIZE.H2).text('Centering test violations:', margin, yPos); 
                      yPos += 5;
                      const ch = sol.analysis.centeringViolationsHard || {};
                      const cs = sol.analysis.centeringViolationsSoft || {};
                      const violText = Object.entries(sol.analysis.centeringViolations)
                          .sort(([,a], [,b]) => a - b)
                          .map(([key, val]) => {
                              const hardV = ch[key] != null ? ch[key] : val;
                              const softV = cs[key] != null ? cs[key] : 0;
                              return softV > 0 ? `${key}:${hardV}(+${softV} soft)` : `${key}:${hardV}`;
                          })
                          .join(', ');
                      doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY).text(violText, margin, yPos);
                      yPos += 5;
                      if (sol.analysis.centeringViolations && sol.analysis.centeringViolationDetails) {
                           doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL); 
                           let detailsYOffset = 0; 

                           for (const type of ['I', 'F', 'A', 'B', 'C']) {
                               const count = sol.analysis.centeringViolations[type];
                               const details = sol.analysis.centeringViolationDetails[type];

                               if ((count === 1 || count === 2) && details && details.length > 0) {
                                   let detailText = `${type} violation${count > 1 ? 's' : ''}: `;
                                   detailText += details.map(d =>
                                       `(${d.h},${d.k},${d.l}) at ${d.tth.toFixed(3)}°`
                                   ).join('; ');

                                   if (yPos + detailsYOffset > 285) { 
                                       doc.addPage();
                                       yPos = 20;
                                       detailsYOffset = 0; 
                                   }
                                   doc.text(detailText, margin + 5, yPos + detailsYOffset); 
                                   detailsYOffset += 3.5; 
                               }
                           }
                           yPos += detailsYOffset; 
                           doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.BODY); 
                      }
                      yPos += 5;
                 }
            } else {
                // No absence analysis at all (it can throw, and the MC's
                // "Add as solution" path tolerates that). The MC verdict is
                // stored on the solution itself and does not depend on the
                // analysis, so it must still be reported here.
                writeSgVerdict(sol);
            }
            yPos += 4; 
            if (yPos > 255) { doc.addPage(); yPos = 20; }
    
            doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.TABLE_HEADER);
            const tableHeader = ' h  k  l  | 2th_exp 2th_cor 2th_cal diff(2t)|   d_corr   d_calc  diff(d)';
            doc.text(tableHeader, margin, yPos); yPos += 4;
            
            const hklList = sol.analysis?.hklList || generateHKL_for_analysis(sol, lambda, tthMaxVal);
            if (hklList.length === 0) {
                 doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.BODY).text('Could not generate theoretical reflections for this cell.', margin, yPos);
                 yPos += 5;
                 return; 
            }
        
            const ambiguousHkls = (sol.analysis ? sol.analysis.ambiguousHkls : new Set()) || new Set();
            const corrected_tth_obs = reportPeaks.map(p => ({ ...p, tth_corr: p.tth - (sol.zero_correction || 0) }));
            const reportLines = []; 
            const assignedHkls = new Set();
        
            const manualByTth = new Map();
            (sol.manualSwaps || []).forEach(sw => {
                 if (sw && Number.isFinite(sw.h) && Number.isFinite(sw.k) && Number.isFinite(sw.l)) {
                     manualByTth.set(Number(sw.tth).toFixed(4), sw);
                 }
            });

            corrected_tth_obs.forEach((corr_peak) => {
                 let bestMatchHkl = null; let minDiff = Infinity;
                 let isManual = false;

                 if (corr_peak.ka2Suspect && corr_peak.ka2ParentIdx != null && pickedPeaks[corr_peak.ka2ParentIdx]) {
                     const parentPeak = pickedPeaks[corr_peak.ka2ParentIdx];
                     const parentTthCorr = parentPeak.tth - (sol.zero_correction || 0);
                     
                     let parentHkl = null; let minParentDiff = Infinity;
                     hklList.forEach(hkl => {
                         const diff = Math.abs(hkl.tth - parentTthCorr);
                         if (diff < minParentDiff) { minParentDiff = diff; parentHkl = hkl; }
                     });

                     const manParent = manualByTth.get(Number(parentPeak.tth).toFixed(4));
                     if (manParent) {
                         const forced = hklList.find(x => x.h === manParent.h && x.k === manParent.k && x.l === manParent.l);
                         if (forced) parentHkl = forced;
                     }

                     if (parentHkl && lambdaKa2 && parentHkl.d > 0) {
                         const sinTheta2 = lambdaKa2 / (2 * parentHkl.d);
                         if (sinTheta2 < 1.0) {
                             const tthCalcKa2 = 2 * Math.asin(sinTheta2) * (180 / Math.PI);
                             bestMatchHkl = {
                                 h: parentHkl.h, k: parentHkl.k, l: parentHkl.l,
                                 tth: tthCalcKa2,
                                 d: parentHkl.d
                             };
                             minDiff = Math.abs(bestMatchHkl.tth - corr_peak.tth_corr);
                         }
                     }
                 } else {
                     hklList.forEach(hkl => { const diff = Math.abs(hkl.tth - corr_peak.tth_corr); if (diff < minDiff) { minDiff = diff; bestMatchHkl = hkl; } });

                     const _man = manualByTth.get(Number(corr_peak.tth).toFixed(4));
                     if (_man) {
                         const forced = hklList.find(x => x.h === _man.h && x.k === _man.k && x.l === _man.l);
                         if (forced) { bestMatchHkl = forced; minDiff = Math.abs(forced.tth - corr_peak.tth_corr); isManual = true; }
                     }
                 }

                 const indexWindow = tthError * 1.5;

                 if (bestMatchHkl && (isManual || minDiff < indexWindow) && corr_peak.tth_corr >= tthMinVal && corr_peak.tth_corr <= tthMaxVal) {
                      reportLines.push({
                         h: bestMatchHkl.h, k: bestMatchHkl.k, l: bestMatchHkl.l,
                         tth_meas: corr_peak.tth, tth_corr: corr_peak.tth_corr,
                         tth_calc: bestMatchHkl.tth, d_calc: bestMatchHkl.d,
                         ka2Suspect: !!corr_peak.ka2Suspect,
                         hasKa2Child: !!corr_peak.hasKa2Child,
                         manual: isManual
                      });
                      assignedHkls.add(`${bestMatchHkl.h},${bestMatchHkl.k},${bestMatchHkl.l}`);
                 }
            });
        
            hklList.forEach(hkl => {
                 if (!assignedHkls.has(`${hkl.h},${hkl.k},${hkl.l}`) && hkl.tth >= tthMinVal && hkl.tth <= tthMaxVal) {
                      reportLines.push({ h: hkl.h, k: hkl.k, l: hkl.l, tth_meas: null, tth_corr: null, tth_calc: hkl.tth, d_calc: hkl.d });
                 }
            });
            
            reportLines.sort((a, b) => a.tth_calc - b.tth_calc);
            
            doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
            
            reportLines.forEach(line => {
                 if (yPos > 285) { doc.addPage(); yPos = 20; doc.setFont(FONT.DATA, 'bold').setFontSize(SIZE.TABLE_HEADER); doc.text(tableHeader, margin, yPos); yPos += 4; doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY); }
                
                 const hkl_key = `${line.h},${line.k},${line.l}`;
                 const isAmbiguous = ambiguousHkls.has(hkl_key);
                 if (isAmbiguous) {
                     doc.setFont(FONT.DATA, 'italic');
                 }

                 let lambdaForD = lambda;
                 if (line.hasKa2Child && lambdaKa1) lambdaForD = lambdaKa1;
                 else if (line.ka2Suspect && lambdaKa2) lambdaForD = lambdaKa2;

                 const d_corr = line.tth_corr ? lambdaForD / (2 * Math.sin(line.tth_corr * Math.PI / 360)) : null;

                 const tth_m = line.tth_meas ? line.tth_meas.toFixed(3) : '-'; 
                 const tth_c = line.tth_corr ? line.tth_corr.toFixed(3) : '-'; 
                 const diff_2t = line.tth_corr ? (line.tth_corr - line.tth_calc).toFixed(3) : '-'; 

                 const d_c_str = d_corr ? d_corr.toFixed(5) : '-'; 
                 const diff_d = d_corr ? (d_corr - line.d_calc).toFixed(5) : '-';

                 const _marker = line.hasKa2Child ? '*1' : (line.ka2Suspect ? '*2' : ''); 
                 const hkl_nums = `${String(line.h).padStart(2)} ${String(line.k).padStart(2)} ${String(line.l).padStart(2)}`;
                 const hkl_str = `${hkl_nums}${_marker}`.padEnd(10);
                
                 let pdfLine = `${hkl_str}| ${tth_m.padStart(7)} ${tth_c.padStart(7)} ${line.tth_calc.toFixed(3).padStart(7)} ${diff_2t.padStart(8)}| ${d_c_str.padStart(8)} ${line.d_calc.toFixed(5).padStart(8)} ${diff_d.padStart(8)}`;
                
                 if (isAmbiguous) {
                     pdfLine += ' *';
                 }
                 if (line.manual) {
                     pdfLine += '  (manual)';
                 }

                 doc.text(pdfLine, margin, yPos);
                
                 doc.setFont(FONT.DATA, 'normal'); 
                 yPos += 3.5;
            });

            if (reportLines.some(l => l.hasKa2Child || l.ka2Suspect)) {
                if (yPos > 276) { doc.addPage(); yPos = 20; }
                doc.setFont(FONT.DATA, 'italic').setFontSize(SIZE.SMALL);
                const note1 = `(*1) Ka1 parent line: d_corr computed with Ka1 = ${lambdaKa1 ? lambdaKa1.toFixed(5) : '-'} A.`;
                const note2 = `(*2) Ka2 companion line: d_corr computed with Ka2 = ${lambdaKa2 ? lambdaKa2.toFixed(5) : '-'} A (same d-spacing as parent).`;
                if (reportLines.some(l => l.hasKa2Child)) { doc.text(note1, margin, yPos); yPos += 3.5; }
                if (reportLines.some(l => l.ka2Suspect)) { doc.text(note2, margin, yPos); yPos += 3.5; }
                doc.setFont(FONT.DATA, 'normal').setFontSize(SIZE.TABLE_BODY);
            }
        });
    
        const filename = `Indexing-Report-${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}.pdf`;
        doc.save(filename);
        showStatus('PDF report generated and saved.', 'success');
    
    } catch (error) {
        console.error("Failed to generate PDF:", error);
        showStatus("An error occurred during PDF generation.", 'error');
    } finally {
        ui.reportButton.textContent = 'Generate PDF Report';
        ui.reportButton.disabled = (solutions.length === 0);
        document.body.style.cursor = 'default';
    }
};

handleWavelengthPresetChange({ onLoad: true });
window.addEventListener('beforeunload', () => { if (workerURL) URL.revokeObjectURL(workerURL); });
});
   
    
