// js/core/ui.js
// DOM references, the status toast, tab switching and the run-state lock.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

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
    peakProminenceSlider: document.getElementById('peak-prominence-slider'),
    peakProminenceValue: document.getElementById('peak-prominence-value'),
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
const setUIState = (indexing) => {
    isIndexing = indexing; 
    document.body.style.cursor = indexing ? 'wait' : 'default';
    
    const controlsToDisable = [ 
        ui.fileInput, ui.peakThresholdSlider, ui.peakProminenceSlider, ui.tthMinSlider, ui.tthMaxSlider, 
        ui.ballRadiusSlider, ui.smoothingWidthSlider, ui.wavelength, ui.tthError, 
        ui.maxVolume, ui.impurityPeaksInput, ui.refineZeroCheckbox, 
        // ...ui.systemCheckboxes, modif nov 25
        ...ui.tabButtons, ui.wavelengthPreset
    ];
    
    controlsToDisable.forEach(el => { if (el) el.disabled = indexing; });
    // The strip control follows the PRESET, not just the run state. It used
    // to be in the list above, so every finished run re-enabled it -- even
    // under a custom or Ka1 preset, where ticking it strips a Ka2 that is
    // not there (updateWorkingData only excluded 'custom').
    if (ui.stripKa2Checkbox) ui.stripKa2Checkbox.disabled = indexing || !ka2StripAllowed();
    

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
            // (The strip control is set once, above, from ka2StripAllowed().)
            // Wavelength input is always editable once data is loaded.
            ui.wavelength.disabled = false;
        }
    }
};
