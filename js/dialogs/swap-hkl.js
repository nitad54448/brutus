// js/dialogs/swap-hkl.js
// Swap hkl dialog: manual re-assignment of reflections.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

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
    const wl = getWavelength();
    const te = getTthError();
    const tMin = parseFloat(ui.tthMinSlider.value);
    const tMax = parseFloat(ui.tthMaxSlider.value);
    const pk = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);
    const minReq = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 }[parent.system] || 6;
    if (pk.length < minReq) { showStatus(`Not enough peaks in the 2-theta range (need ${minReq} for ${parent.system}).`, 'error', 4000); return; }

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
    const wl = getWavelength();
    const te = getTthError();
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
