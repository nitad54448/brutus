// js/peaks/peak-controls.js
// Peak-search sliders and the 2θ range controls.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

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
// findPeaks is defined in js/peaks/peak-finder.js, which loads later;
// the arrow defers the lookup to call time.
const debouncedFindPeaks = debounce((...args) => findPeaks(...args), 250);
const debouncedUpdateAndRedraw = debounce(() => {
    updateWorkingData();
    if (xrdChart) {
        if (xAxisMode === 'd' || xAxisMode === 'q') {
            rebuildPlot(true);
        } else {
            // Stripping and deconvolution both change the intensities, so the
            // ordinate is rescaled as well as redrawn.
            setExperimentalTrace(true); // pas d'animation, sinon c'est trop lent
        }
    }
    findPeaks();
}, 250);
// log slider, 
const initialPeakThreshold = 2.0;
ui.peakThresholdSlider.value = valueToLogSlider(initialPeakThreshold);
ui.peakThresholdValue.textContent = initialPeakThreshold.toFixed(1);
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
ui.ballRadiusSlider.addEventListener('input', () => { ui.ballRadiusValue.textContent = parseFloat(ui.ballRadiusSlider.value).toFixed(3); debouncedFindPeaks(); });
// The Q radius is converted to channels using lambda, so the background
// window is no longer wavelength-independent the way a raw point count was:
// r scales linearly with lambda, so switching Cu -> Mo roughly halves it.
//
// But findPeaks() REPLACES pickedPeaks, discarding manual additions,
// deletions and userConfirmedReal flags. Silently destroying that work
// because the user corrected a wavelength typo would be far worse than a
// slightly stale background, so only re-detect when the list is still
// purely automatic. Otherwise say the background is stale and let the user
// decide -- any peak-finding slider re-runs the detection anyway.
ui.wavelength.addEventListener('change', () => {
    if (!workingExperimentalData || !workingExperimentalData.intensity) return;
    if (!peaksManuallyEdited) {
        debouncedFindPeaks();
    } else {
        showStatus('Wavelength changed. The background window is set in Q, so it is now stale — ' +
                   'move a peak-finding slider to re-detect (this will discard manual peak edits).',
                   'info', 9000);
    }
});
ui.smoothingWidthSlider.addEventListener('input', () => { ui.smoothingWidthValue.textContent = parseFloat(ui.smoothingWidthSlider.value).toFixed(3); debouncedFindPeaks(); });
ui.peakThresholdSlider.addEventListener('input', () => { const value = logSliderToValue(parseFloat(ui.peakThresholdSlider.value)); ui.peakThresholdValue.textContent = value.toFixed(1); debouncedFindPeaks(); });
if (ui.peakProminenceSlider) {
    ui.peakProminenceSlider.addEventListener('input', () => {
        ui.peakProminenceValue.textContent = ui.peakProminenceSlider.value;
        debouncedFindPeaks();
    });
}
