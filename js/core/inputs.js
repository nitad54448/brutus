// js/core/inputs.js
// Numeric input reading and validation. Everything that consumes a
// parameter goes through readNumberInput, so a typed-but-not-blurred value
// is clamped exactly like a validated one.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

// Markup identifies controls; the shared definition supplies their bounds and
// defaults. Workers and programmatic consumers use the same definition.
for (const [id, spec] of Object.entries(BRUTUS_LIMITS.inputs)) {
    const el = document.getElementById(id);
    if (!el) continue;
    for (const key of ['min', 'max', 'step']) if (spec[key] !== undefined) el[key] = String(spec[key]);
    el.defaultValue = el.value = String(spec.value);
}

// Single source of truth for the "Impurity Peaks" value used at RUN TIME.
// The input has min/max attributes, which are enforced on 'blur' and by
// readNumberInput. Every place that actually starts a computation reads
// through this helper so all consumers see the same clamped integer.
// Bounds come from the input's own min/max attributes so the runtime clamp
// can never drift from the markup.
function getImpurityPeaks() {
    return readNumberInput(ui.impurityPeaksInput, { integer: true, fallback: BRUTUS_LIMITS.inputs['impurity-peaks'].value });
}
// Read a numeric <input> the way its markup says it may be read: parse it,
// fall back to the markup default (the value="" attribute) when the field
// is empty or not a number, and clamp to min/max. Browsers only apply
// min/max on blur, so a value typed and run without leaving the field used
// to reach the search unchecked -- and the fallbacks scattered through the
// code disagreed with the markup (FoM fell back to 0.8, the field says 1.5).
function readNumberInput(el, { integer = false, fallback = NaN } = {}) {
    if (!el) return fallback;
    const parse = value => value == null || String(value).trim() === '' ? NaN : Number(value);
    let v = parse(el.value);
    if (!Number.isFinite(v)) {
        const d = parse(el.defaultValue);
        v = Number.isFinite(d) ? d : fallback;
    }
    if (integer) v = Math.trunc(v);
    const min = parseFloat(el.min), max = parseFloat(el.max);
    if (Number.isFinite(min) && v < min) v = min;
    if (Number.isFinite(max) && v > max) v = max;
    return v;
}
function getWavelength()     { return readNumberInput(ui.wavelength, { fallback: BRUTUS_LIMITS.inputs.wavelength.value }); }
function getTthError()       { return readNumberInput(ui.tthError, { fallback: BRUTUS_LIMITS.inputs['tth-error'].value }); }
function getMaxVolume()      { return readNumberInput(ui.maxVolume, { fallback: BRUTUS_LIMITS.inputs['max-volume'].value }); }
function getFomThreshold()   { return readNumberInput(ui.gpuFomThreshold, { fallback: BRUTUS_LIMITS.inputs['gpu-fom-threshold'].value }); }
// The two global GPU search settings (9 Oct 2026); planGpuSearch() in
// js/gpu/gpu-setup.js turns them into each system's basis size and peak count.
//   HKL basis: percent of the system's maximum basis PER UNKNOWN cell
//              parameter (cubic x1 ... triclinic x6).
//   Depth:     peaks combined = number of unknowns + Depth.
function getHklPercentPerUnknown() { return readNumberInput(ui.gpuHklPercent, { fallback: BRUTUS_LIMITS.inputs['gpu-hkl-percent'].value }); }
function getSearchDepth()    { return readNumberInput(ui.gpuDepth, { integer: true, fallback: BRUTUS_LIMITS.inputs['gpu-depth'].value }); }
function getCandidateCells() { return readNumberInput(ui.gpuBufferSize, { integer: true, fallback: BRUTUS_LIMITS.inputs['gpu-buffer-size'].value }) * 1000; }
// Blur-time validation: rewrite the field with exactly the value the run
// will use (readNumberInput), so what the user sees is what is searched.
const inputsToValidate = [
    { id: 'wavelength', el: ui.wavelength },
    { id: 'max-volume', el: ui.maxVolume },
    { id: 'tth-error', el: ui.tthError },
    { id: 'impurity-peaks', el: ui.impurityPeaksInput, integer: true },
    { id: 'gpu-hkl-percent', el: ui.gpuHklPercent },
    { id: 'gpu-depth', el: ui.gpuDepth, integer: true },
    { id: 'gpu-fom-threshold', el: ui.gpuFomThreshold },
    { id: 'gpu-buffer-size', el: ui.gpuBufferSize, integer: true },
];
inputsToValidate.forEach(({ id, el, integer }) => {
    if (!el) return;
    el.addEventListener('blur', () => {
        const v = readNumberInput(el, { integer });
        if (Number.isFinite(v) && String(v) !== el.value.trim()) el.value = v;
        if (id.startsWith('gpu')) updateGpuStatusText();
    });
});
if (ui.tthError) {
    ui.tthError.addEventListener('change', () => {
        if (isIndexing) return;
        recalculatePeakValues();
        rebuildHklList();
        updatePeakTable();
        updateStartIndexingButtonState();
    });
}
if (ui.wavelength) {
    // Recalculate d-spacings if the user manually changes the wavelength (Custom mode).
    // The Q radius is converted to channels using lambda, so the background
    // window is also wavelength-dependent: re-detect peaks only when the list
    // has not been manually edited, otherwise keep hand edits and warn the user.
    ui.wavelength.addEventListener('change', () => {
        if (isIndexing) return;
        if (typeof handleWavelengthValueChange === 'function') handleWavelengthValueChange();
        if (pickedPeaks.length > 0) {
            recalculatePeakValues();
            updatePeakTable();
        }
        if (xrdChart) {
            if (xAxisMode === 'd' || xAxisMode === 'q') {
                rebuildPlot(false);
            } else {
                rebuildHklList();
                updateAllMarkers();
            }
        }
        if (workingExperimentalData && workingExperimentalData.intensity.length) {
            if (!peaksManuallyEdited) {
                debouncedFindPeaks();
            } else {
                showStatus('Wavelength changed. The background window is set in Q, so it is now stale — ' +
                           'move a peak-finding slider to re-detect (this will discard manual peak edits).',
                           'info', 9000);
            }
        }
    });
}
