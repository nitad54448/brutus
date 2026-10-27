// js/data/session.js
// Session reset, the file chip and the unload dialog.
//
// Classic script, loaded in order by index.html (see the list there); its
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
if (typeof workspaceRunStarted !== 'undefined') workspaceRunStarted = null;
abortActiveIndexing();

pickedPeaks = [];
if (typeof calculatedBackground !== 'undefined') calculatedBackground = [];
peaksManuallyEdited = false;
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
lastRunTotals = null;

updatePeakTable();
updateSolutionsTable();
updateStartIndexingButtonState();
ui.solutionsLed.className = 'led-indicator gray';
if (typeof refreshWorkspaceStatus === 'function') refreshWorkspaceStatus();
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
// Single place that paints the file accordion's title, so the empty and loaded states cannot
// disagree about which classes are set.
const renderFileChip = (name, sizeBytes) => {
if (!ui.fileName || !ui.fileChipName) return;
if (!name) {
    if (typeof setWorkspaceState === 'function') setWorkspaceState('');
    ui.fileName.classList.add('is-empty');
    ui.fileChipName.textContent = 'No file loaded';
    ui.fileChipName.title = '';
    ui.fileName.open = false;
    return;
}
ui.fileName.classList.remove('is-empty');
ui.fileChipName.textContent = truncateMiddle(name, 34);
ui.fileChipName.title = name;   // full name on hover
};
// Full unload: session state AND the data itself, back to the pristine
// first-load view. Everything the load path switches ON is switched off here,
// which is why the two live next to each other.
const clearLoadedFile = () => {
fileLoadToken++; // an outstanding read must not restore the unloaded pattern
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
if (typeof refreshWorkspaceStatus === 'function') refreshWorkspaceStatus();
if (typeof showWorkspaceView === 'function') showWorkspaceView('plot');
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

// An in-page notice or question, drawn like the other dialogs (never the browser's own alert / confirm, which
// ignore the theme). Resolves true on OK, false on Cancel, Escape or a click outside. With confirmLabel
// only (cancelLabel null) it is a plain notice.
const showNotice = (title, message, { confirmLabel = 'OK', cancelLabel = 'Cancel' } = {}) => new Promise(resolve => {
    const overlay = document.getElementById('notice-overlay');
    if (!overlay) { resolve(true); return; }   // markup missing: do not block the user
    const ok = document.getElementById('notice-confirm'), cancel = document.getElementById('notice-cancel');
    document.getElementById('notice-title').textContent = title;
    document.getElementById('notice-text').textContent = message;
    ok.textContent = confirmLabel;
    cancel.textContent = cancelLabel || '';
    cancel.hidden = !cancelLabel;
    const finish = result => {
        overlay.classList.remove('open');
        ok.removeEventListener('click', onOk); cancel.removeEventListener('click', onCancel);
        overlay.removeEventListener('click', onOutside); document.removeEventListener('keydown', onKey);
        resolve(result);
    };
    const onOk = () => finish(true), onCancel = () => finish(false);
    const onOutside = e => { if (e.target === overlay) finish(false); };
    const onKey = e => { if (e.key === 'Escape') finish(false); };
    ok.addEventListener('click', onOk); cancel.addEventListener('click', onCancel);
    overlay.addEventListener('click', onOutside); document.addEventListener('keydown', onKey);
    overlay.classList.add('open');
    ok.focus();
});
