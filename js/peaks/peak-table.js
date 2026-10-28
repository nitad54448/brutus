// js/peaks/peak-table.js
// The picked-peak table and its editing.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

const setPeakTableLocked = (locked) => {
    ui.peakListBody.querySelectorAll('input, button').forEach(el => {
        el.disabled = locked || el.classList.contains('peak-d-input');
    });
};
const updatePeakTable = () => {
    // Clamp the selected index in case the peak list shrank.
    if (selectedPeakIndex !== null && selectedPeakIndex >= pickedPeaks.length) {
        selectedPeakIndex = null;
    }
    ui.peakListBody.innerHTML = '';
    const fragment = document.createDocumentFragment();
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
                `d-spacing, indexing and refinement use λ_Kα₁.`;
        }
        if (index === selectedPeakIndex) {
            row.classList.add('peak-row-selected');
        }
        const indexLabel = `${index + 1}${isParent ? '*' : ''}`;
        // Keyboard equivalent of the right-click: confirm a Kα2 suspect as a
        // real peak, or send a confirmed one back to the automatic flagging.
        const ka2Btn = isSuspect
            ? `<button type="button" class="ka2-real-btn" data-index="${index}" title="Mark as a real (independent) peak" aria-label="Mark peak ${index + 1} as real">real</button> `
            : peak.userConfirmedReal
                ? `<button type="button" class="ka2-real-btn" data-index="${index}" title="Undo: let Kα2 detection flag this peak again" aria-label="Undo real mark on peak ${index + 1}">Kα₂?</button> `
                : '';
        row.innerHTML = `<td>${indexLabel}</td><td><input type="number" class="peak-tth-input" value="${peak.tth.toFixed(4)}" data-index="${index}" step="0.0001" aria-label="2θ of peak ${index + 1}"></td><td><input type="number" class="peak-d-input" value="${peak.d.toFixed(5)}" data-index="${index}" disabled aria-label="d of peak ${index + 1}"></td><td>${ka2Btn}<button class="delete-peak-btn" data-index="${index}" aria-label="Delete peak ${index + 1}">X</button></td>`;
        fragment.appendChild(row);
    });
    ui.peakListBody.appendChild(fragment);
    setPeakTableLocked(isIndexing);
    ui.peakTableContainer.classList.toggle('hidden', pickedPeaks.length === 0);
    updateKa2Banner();
    updateAllMarkers();
};
// The table is rebuilt after every edit, which destroys the focused control
// and drops keyboard focus to <body>. If that happened, put it back on the
// 2θ input of the given row (clamped to the table), after the browser has
// finished moving focus.
const restorePeakTableFocus = (rowIndex) => setTimeout(() => {
    const active = document.activeElement;
    if (active && active !== document.body && active.isConnected) return;
    const rows = ui.peakListBody.children;
    if (!rows.length) return;
    const row = rows[Math.max(0, Math.min(rowIndex, rows.length - 1))];
    const target = row && row.querySelector('.peak-tth-input');
    if (target && !target.disabled) target.focus();
}, 0);
ui.peakListBody.addEventListener('change', (e) => {
    if (isIndexing) return;
    if (e.target.classList.contains('peak-tth-input')) {
        const index = parseInt(e.target.dataset.index, 10); 
        let tth = parseFloat(e.target.value); 
        
        // Explicitly check for NaN or non-finite numbers before clamping
        // 2θ must lie in (0, 180): beyond that d is non-physical, and a saved
        // project holding such a peak is refused by validateWorkspace.
        if (!isFinite(tth) || tth <= 1e-4 || tth >= 180) {
            // Fallback to the existing value if invalid, or a safe default
            tth = pickedPeaks[index]?.tth || 1.0; 
        }
        
        e.target.value = tth.toFixed(4); 
        const wasUserConfirmed = !!pickedPeaks[index]?.userConfirmedReal;
        const existingHeight = pickedPeaks[index]?.height; 
        
        const editedPeak = { tth, d: 0, q: 0, height: existingHeight, userConfirmedReal: wasUserConfirmed };
        pickedPeaks[index] = editedPeak;
        peaksManuallyEdited = true;
        pickedPeaks.sort((a, b) => a.tth - b.tth);
        if (selectedPeakIndex === index) {
            selectedPeakIndex = pickedPeaks.indexOf(editedPeak);
        }
        recalculatePeakValues();
        updatePeakTable();
        updateStartIndexingButtonState();
        restorePeakTableFocus(pickedPeaks.indexOf(editedPeak));
    }
});
 ui.peakListBody.addEventListener('click', (e) => {
    if (isIndexing) return;
    if (e.target.classList.contains('ka2-real-btn')) {
        const index = parseInt(e.target.dataset.index, 10);
        const peak = pickedPeaks[index];
        if (!peak) return;
        peak.userConfirmedReal = !peak.userConfirmedReal;
        peaksManuallyEdited = true;
        recalculatePeakValues();
        updatePeakTable();
        updateStartIndexingButtonState();
        restorePeakTableFocus(index);
        return;
    }
    if (e.target.classList.contains('delete-peak-btn')) {
        const index = parseInt(e.target.dataset.index);
        pickedPeaks.splice(index, 1);
        peaksManuallyEdited = true;
        
        if (selectedPeakIndex !== null) {
            if (index === selectedPeakIndex) {
                selectedPeakIndex = null;
            } else if (index < selectedPeakIndex) {
                selectedPeakIndex--;
            }
        }
        
        recalculatePeakValues(); // re-flag + recompute after deletion
        updatePeakTable();
        updateStartIndexingButtonState();
        restorePeakTableFocus(index);   // the row that took the deleted one's place
    }
});
// Right-click on a Ka2-suspect row → toggle "this is actually a real peak"
ui.peakListBody.addEventListener('contextmenu', (e) => {
    if (isIndexing) return;
    const row = e.target.closest('tr.ka2-suspect');
    if (!row) return;
    e.preventDefault();
    const idx = Array.from(ui.peakListBody.children).indexOf(row);
    if (idx < 0 || !pickedPeaks[idx]) return;
    // User explicitly says: not a Ka2 ghost — keep as real.
    pickedPeaks[idx].userConfirmedReal = true;
    peaksManuallyEdited = true;
    // Recompute flags + d/q. The previous parent of this peak may lose
    // its hasKa2Child status (and revert from λ_Ka1 to user λ for d).
    recalculatePeakValues();
    updatePeakTable();
    updateStartIndexingButtonState();
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
