// js/chart/interaction.js
// Cursor snapping, click-to-add peaks and the panel resizer.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

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
const resizer = document.getElementById('drag-handle');
 const leftPanel = document.getElementById('controls-panel');
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
    peaksManuallyEdited = true;
    pickedPeaks.sort((a, b) => a.tth - b.tth);
    recalculatePeakValues(); // tag + d/q in one shot
    updatePeakTable(); updateStartIndexingButtonState();
    const how = hit
        ? (hit.kind === 'hkl' ? ` (snapped to hkl ${hit.hkl.h}${hit.hkl.k}${hit.hkl.l})` : ' (snapped to observed peak)')
        : '';
    showStatus(`Peak added at ${tth.toFixed(3)}\u00B0${how}`, 'success', 2000);
});
