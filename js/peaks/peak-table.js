// js/peaks/peak-table.js
// The picked-peak table and its editing.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

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
                `d-spacing computed with λ_Kα₁.`;
        }
        if (index === selectedPeakIndex) {
            row.classList.add('peak-row-selected');
        }
        const indexLabel = `${index + 1}${isParent ? '*' : ''}`;
        row.innerHTML = `<td>${indexLabel}</td><td><input type="number" class="peak-tth-input" value="${peak.tth.toFixed(4)}" data-index="${index}" step="0.0001"></td><td><input type="number" class="peak-d-input" value="${peak.d.toFixed(5)}" data-index="${index}" disabled></td><td><button class="delete-peak-btn" data-index="${index}">X</button></td>`;
        fragment.appendChild(row);
    });
    ui.peakListBody.appendChild(fragment);
    ui.peakTableContainer.classList.toggle('hidden', pickedPeaks.length === 0);
    updateKa2Banner();
    updateAllMarkers();
};
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
        peaksManuallyEdited = true;
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
    peaksManuallyEdited = true;
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
