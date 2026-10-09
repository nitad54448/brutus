// js/data/session.js
// Session reset, the file chip and the unload dialog.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

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
// The per-run GPU figures (js/indexing/run.js) belong to the old file too.
lastGpuRunSettings = null;
lastTruncatedSystems = [];
lastSystemSearchStats = [];

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

// Re-derive the strip control from the preset instead of force-enabling it:
// under a custom or Ka1 preset it stays off (there is no Ka2 to strip), and
// choosing an average preset re-enables it as usual.
if (ui.stripKa2Checkbox) ui.stripKa2Checkbox.disabled = !ka2StripAllowed();

if (xrdChart) {
        try { xrdChart.resetZoom('none'); } catch (_) { /* no zoom state yet */ }
        xrdChart.data.datasets.forEach(ds => ds.data = []);
        xrdChart.update('none');
    }
if (typeof updateAllMarkers === 'function') updateAllMarkers();


if (ui.peakControls) ui.peakControls.classList.add('hidden');
if (ui.indexingControls) ui.indexingControls.classList.add('hidden');
if (ui.resultsContainer) ui.resultsContainer.style.display = 'none';
if (ui.placeholder) ui.placeholder.style.display = '';

renderFileChip(null);
showStatus('File unloaded.', 'info', 2000);
};
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
