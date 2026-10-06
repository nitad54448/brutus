// js/data/file-load.js
// Load File: reading, normalising and installing a pattern.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// `Load File` is a <button> now rather than a <label for>, so the click has
// to be forwarded to the hidden input by hand. The trade is worth it: as a
// label it carried .file-input-label, whose background overrode
// .btn-secondary and made it a visibly different colour from `Save as`.
if (ui.fileInputLabel) ui.fileInputLabel.addEventListener('click', () => {
    if (ui.fileInput) ui.fileInput.click();
});
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

    if (tth_out.length > 1 && tth_out[0] > tth_out[tth_out.length - 1]) {
        console.warn("Descending 2θ scan detected. Sorting ascending to prevent binary search failure...");
        const combined = tth_out.map((t, i) => ({ t, int: int_out[i] })).sort((a, b) => a.t - b.t);
        tth_out = combined.map(item => item.t);
        int_out = combined.map(item => item.int);
    }

    // Trim trailing zero-intensity points (detector parked past the end of
    // the scan). Done after the ascending sort so it always trims the
    // high-angle end, whatever the scan direction was.
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


    // Radiation from the file, if it records one. Parsers return the raw
    // description (doublet lines, ratio, monochromator hint or a single
    // value); classifyFileWavelength maps it onto a preset or 'custom'.
    const fileRadiation = parsed.radiation ||
        (parsed.wavelength ? { single: parsed.wavelength } : null);
    const fileWl = fileRadiation ? classifyFileWavelength(fileRadiation) : { wavelength: null, presetMatch: null };
    if (fileWl.presetMatch) {
        ui.wavelengthPreset.value = fileWl.presetMatch;
        handleWavelengthPresetChange({ onLoad: true });
        showStatus(`Loaded preset wavelength from file: ${fileWl.presetMatch.replace('_', ' ')}`, 'info');
    } else if (fileWl.wavelength) {
        // A known wavelength that does not match a preset line.
        ui.wavelengthPreset.value = 'custom';
        ui.wavelength.value = fileWl.wavelength.toFixed(5);
        ui.stripKa2Checkbox.checked = false;
        ui.stripKa2Checkbox.disabled = true;
        showStatus(`Loaded custom wavelength from file: ${fileWl.wavelength.toFixed(5)} Å`, 'info');
    } else {
        // File has no wavelength: default to the Cu Kα-average preset.
        ui.wavelengthPreset.value = 'Cu_avg'; 
        handleWavelengthPresetChange({ onLoad: true });
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
