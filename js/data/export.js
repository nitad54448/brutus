// js/data/export.js
// Save-as: exports the plotted pattern in several formats.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

/* ------------------------------------------------------------------
   Save-as / file-converter export
   ------------------------------------------------------------------
   The exported pattern is exactly what is currently plotted:
   workingExperimentalData already holds the raw-or-Kα2-stripped trace
   (whichever the strip checkbox selects), and we clip it to the 2θ
   interval currently visible on the chart, so zooming acts as a range
   selection for the export. */

const getExportPattern = () => {
    const src = workingExperimentalData;
    if (!src || !src.tth || src.tth.length === 0) return null;

    // visibleTthRange() returns the on-screen 2θ window (full span when
    // the chart is not zoomed). It is defined later in the file but only
    // called here at click-time, so hoisting is not a concern.
    let lo = -Infinity, hi = Infinity;
    try {
        const r = visibleTthRange();
        if (r && isFinite(r[0]) && isFinite(r[1])) { lo = r[0]; hi = r[1]; }
    } catch (_) { /* chart not ready → export full pattern */ }

    const tth = [], intensity = [];
    for (let i = 0; i < src.tth.length; i++) {
        const t = src.tth[i];
        if (t >= lo && t <= hi) { tth.push(t); intensity.push(src.intensity[i]); }
    }
    if (tth.length === 0) return null;
    return { tth, intensity };
};
// Wavelength currently in effect (for formats that store it).
const getExportWavelength = () => {
    return getWavelength();
};
// Radiation to record in exported files. An average preset with no
// stripping means the exported pattern still holds the Kα doublet, so the
// file says so (both lines and their ratio); otherwise it is monochromatic
// at the current wavelength. Re-importing then restores the same preset.
const getExportRadiation = () => {
    const lambda = getExportWavelength();
    const [element, type] = String(ui.wavelengthPreset ? ui.wavelengthPreset.value : 'custom').split('_');
    const p = WAVELENGTH_PRESETS[element];
    const stripped = !!(ui.stripKa2Checkbox && ui.stripKa2Checkbox.checked);
    if (p && p.ka1 && type === 'avg' && !stripped) {
        return { doublet: true, lambda, ka1: p.ka1, ka2: p.ka2, ratio: p.ratio };
    }
    return { doublet: false, lambda };
};
const num = (v, dp = 6) => Number(v).toFixed(dp);
const buildExportContent = (fmt, pattern) => {
    const { tth, intensity } = pattern;
    const n = tth.length;
    const lambda = getExportWavelength();
    const step = n > 1 ? (tth[n - 1] - tth[0]) / (n - 1) : 0;
    let out = [];

    switch (fmt) {
        case 'xy':
        case 'dat': {
            for (let i = 0; i < n; i++) out.push(`${num(tth[i], 5)} ${num(intensity[i], 4)}`);
            return { text: out.join('\n') + '\n', ext: fmt === 'xy' ? 'xy' : 'dat', mime: 'text/plain' };
        }
        case 'csv': {
            out.push('2theta,intensity');
            for (let i = 0; i < n; i++) out.push(`${num(tth[i], 5)},${num(intensity[i], 4)}`);
            return { text: out.join('\n') + '\n', ext: 'csv', mime: 'text/csv' };
        }
        case 'uxd': {
            out.push('_FILEVERSION=1');
            out.push('; Exported by Brutus');
            const rad = getExportRadiation();
            if (rad.doublet) {
                out.push(`_WL1=${num(rad.ka1, 6)}`);
                out.push(`_WL2=${num(rad.ka2, 6)}`);
                out.push(`_WLRATIO=${num(rad.ratio, 6)}`);
            } else {
                out.push(`_WL1=${num(lambda, 6)}`);
            }
            out.push(`_START=${num(tth[0], 6)}`);
            out.push(`_STEPSIZE=${num(step, 6)}`);
            out.push('_STEPCOUNT=' + n);
            out.push('_COUNTS');
            // 5 values per line, matching common UXD style
            for (let i = 0; i < n; i += 5) {
                out.push(intensity.slice(i, i + 5).map(v => num(v, 3)).join(' '));
            }
            return { text: out.join('\n') + '\n', ext: 'uxd', mime: 'text/plain' };
        }
        case 'gsas': {
            // GSAS ESD: title, BANK line, then five (Y, sigma) pairs per
            // record in 10F8.x fields -- intensity FIRST, which is how GSAS,
            // GSAS-II and FullProf read it. This used to write (sigma, Y) in
            // 12-character fields to suit this app's old reader, so every
            // other program took the sigma column for the pattern.
            //
            // CONST start/step are centidegrees, now with 6 decimals: with
            // toFixed(2) a 0.0131303 deg step became 0.0131 deg, a 2-theta
            // drift of ~0.1 deg over a few thousand points.
            const f8 = (v) => {                   // right-justified 8-char field
                if (!Number.isFinite(v)) v = 0;
                for (const dp of [2, 1]) {
                    const s = v.toFixed(dp);
                    if (s.length <= 8) return s.padStart(8);
                }
                const s0 = Math.round(v) + '.';   // Fortran style, e.g. "1234567."
                return (s0.length <= 8 ? s0 : v.toExponential(2).toUpperCase()).padStart(8);
            };
            const recPerLine = 5;
            const dataLines = [];
            let buf = '';
            for (let i = 0; i < n; i++) {
                const yi = Math.max(0, intensity[i]);
                const esd = Math.sqrt(yi > 0 ? yi : 1);
                buf += f8(yi) + f8(esd);
                if ((i + 1) % recPerLine === 0) { dataLines.push(buf); buf = ''; }
            }
            if (buf.length) dataLines.push(buf);
            const lines = [
                'Exported by Brutus'.padEnd(80),
                `BANK 1 ${n} ${dataLines.length} CONST ${(tth[0] * 100).toFixed(6)} ` +
                    `${(step * 100).toFixed(6)} 0 0 ESD`,
                ...dataLines,
            ];
            return { text: lines.join('\n') + '\n', ext: 'esd', mime: 'text/plain' };
        }
        case 'xrdml': {
            const positions = `        <positions axis="2Theta" unit="deg">\n` +
                `          <startPosition>${num(tth[0], 6)}</startPosition>\n` +
                `          <endPosition>${num(tth[n - 1], 6)}</endPosition>\n` +
                `        </positions>`;
            const counts = intensity.map(v => Math.round(v)).join(' ');
            const rad = getExportRadiation();
            const usedWavelengthXml = rad.doublet
                ? `    <usedWavelength intended="K-Alpha">\n` +
                  `      <kAlpha1 unit="Angstrom">${num(rad.ka1, 6)}</kAlpha1>\n` +
                  `      <kAlpha2 unit="Angstrom">${num(rad.ka2, 6)}</kAlpha2>\n` +
                  `      <ratioKAlpha2KAlpha1>${num(rad.ratio, 4)}</ratioKAlpha2KAlpha1>\n` +
                  `    </usedWavelength>`
                : `    <usedWavelength intended="K-Alpha 1">\n` +
                  `      <kAlpha1 unit="Angstrom">${num(lambda, 6)}</kAlpha1>\n` +
                  `    </usedWavelength>`;
            const xml =
`<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5">
  <xrdMeasurement measurementType="Scan" status="Completed">
${usedWavelengthXml}
    <scan appendNumber="0" mode="Continuous" scanAxis="2Theta">
      <dataPoints>
${positions}
        <intensities unit="counts">${counts}</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
`;
            return { text: xml, ext: 'xrdml', mime: 'application/xml' };
        }
        case 'brukerxml': {
            const counts = intensity.map(v => Math.round(v)).join(' ');
            const xml =
`<?xml version="1.0" encoding="utf-8"?>
<RawDataFile>
  <DataRoutes>
    <DataRoute>
      <ScanInformation>
        <ScanAxes>
          <ScanAxisInfo AxisName="TwoTheta">
            <Start axis="TwoTheta">${num(tth[0], 6)}</Start>
            <startPosition axis="TwoTheta">${num(tth[0], 6)}</startPosition>
            <increment axis="TwoTheta">${num(step, 6)}</increment>
          </ScanAxisInfo>
        </ScanAxes>
        <usedWavelength ${(() => { const rad = getExportRadiation(); return rad.doublet
            ? `kAlpha1="${num(rad.ka1, 6)}" kAlpha2="${num(rad.ka2, 6)}" ratioKAlpha2KAlpha1="${num(rad.ratio, 4)}"`
            : `kAlpha1="${num(lambda, 6)}"`; })()} />
      </ScanInformation>
      <Datum>
        <dataPoints>
          <counts>${counts}</counts>
        </dataPoints>
      </Datum>
    </DataRoute>
  </DataRoutes>
</RawDataFile>
`;
            return { text: xml, ext: 'xml', mime: 'application/xml' };
        }
        default:
            return { text: '', ext: 'txt', mime: 'text/plain' };
    }
};
const baseNameNoExt = (name) => {
    if (!name) return 'pattern';
    const dot = name.lastIndexOf('.');
    return dot > 0 ? name.slice(0, dot) : name;
};
const triggerDownload = (text, filename, mime) => {
    const blob = new Blob([text], { type: mime + ';charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
};
// Formats that do not store an explicit per-point 2θ column, so they must
// assume a constant step and/or embed a single wavelength. We warn the user
// what metadata is being written into these on their behalf.
const SAVE_FORMAT_META = {
    xrdml:     { wavelength: true,  constStep: true,  label: 'XRDML' },
    brukerxml: { wavelength: true,  constStep: true,  label: 'Bruker XML' },
    uxd:       { wavelength: true,  constStep: true,  label: 'UXD' },
    gsas:      { wavelength: false, constStep: true,  label: 'GSAS ESD' },
    xy:        { wavelength: false, constStep: false, label: 'XY' },
    csv:       { wavelength: false, constStep: false, label: 'CSV' },
    dat:       { wavelength: false, constStep: false, label: 'DAT' }
};
// True when the exported (currently visible) 2θ grid is not evenly spaced.
// Constant-step formats would silently resample such data onto a regular grid.
const patternStepIsIrregular = () => {
    const p = getExportPattern();
    const t = p ? p.tth : null;
    if (!t || t.length < 3) return false;
    const step = (t[t.length - 1] - t[0]) / (t.length - 1);
    if (!(Math.abs(step) > 0)) return false;
    const tol = Math.abs(step) * 0.02; // 2% of the nominal step
    for (let i = 1; i < t.length; i++) {
        if (Math.abs((t[i] - t[i - 1]) - step) > tol) return true;
    }
    return false;
};
const updateSaveInfo = () => {
    const fmt = ui.saveFormatSelect.value;
    const meta = SAVE_FORMAT_META[fmt] || {};
    const notes = [];

    // Kα2 state of the data being written (workingExperimentalData is what
    // is plotted, already stripped when the main strip control is on).
    const stripped = !!(ui.stripKa2Checkbox && ui.stripKa2Checkbox.checked
                        && ui.wavelengthPreset && ui.wavelengthPreset.value !== 'custom');
    if (stripped) {
        notes.push(`Kα2-stripped data will be saved (the plotted pattern has Kα2 removed).`);
    } else {
        notes.push(`Raw data will be saved (no Kα2 stripping applied to the plotted pattern).`);
    }

    // The export is clipped to what's on screen when zoomed.
    const p = getExportPattern();
    if (p && p.tth.length) {
        const full = workingExperimentalData.tth.length;
        if (p.tth.length < full) {
            notes.push(`Only the visible range is exported: ${p.tth[0].toFixed(3)}–${p.tth[p.tth.length - 1].toFixed(3)}° 2θ (${p.tth.length} of ${full} points). Reset the zoom to export the full pattern.`);
        }
    }
    if (meta.wavelength) {
        const w = getExportWavelength();
        notes.push(`${meta.label} stores a wavelength — the value from the wavelength box (${w.toFixed(5)} Å) will be written.`);
    }
    if (meta.constStep && patternStepIsIrregular()) {
        notes.push(`${meta.label} assumes a constant 2θ step; this scan is not evenly spaced, so intensities will be written against a uniform start/step and positions may shift slightly.`);
    }
    ui.saveMenuMsg.innerHTML = notes.length ? 'Info: ' + notes.join('<br>Info: ') : '';
};
const openSaveMenu = () => {
    const p = getExportPattern();
    if (!p) { showStatus('No data to save.', 'error'); return; }
    updateSaveInfo();
    ui.saveMenuOverlay.classList.add('open');
};
const closeSaveMenu = () => ui.saveMenuOverlay.classList.remove('open');
// MIME + human description per extension, for the native save dialog's type list.
const EXT_DESC = {
    xy: 'XY data', csv: 'CSV', dat: 'Data', xrdml: 'XRDML',
    xml: 'Bruker XML', uxd: 'Bruker UXD', esd: 'GSAS ESD'
};
// Save via the File System Access API when available (real OS save dialog
// where the user chooses folder + name), otherwise fall back to a normal
// browser download into the default downloads folder.
const saveTextToFile = async (text, suggestedName, ext, mime) => {
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName,
                types: [{
                    description: EXT_DESC[ext] || 'Data file',
                    accept: { [mime || 'text/plain']: ['.' + ext] }
                }]
            });
            const writable = await handle.createWritable();
            await writable.write(text);
            await writable.close();
            return handle.name || suggestedName;
        } catch (err) {
            // AbortError = user cancelled the dialog; do nothing.
            if (err && err.name === 'AbortError') return null;
            // Any other failure (e.g. permissions) → fall through to download.
            console.warn('showSaveFilePicker failed, falling back to download:', err);
        }
    }
    triggerDownload(text, suggestedName, mime);
    return suggestedName;
};
const doSave = async () => {
    const pattern = getExportPattern();
    if (!pattern) { showStatus('No data to save.', 'error'); return; }

    const fmt = ui.saveFormatSelect.value;
    const { text, ext, mime } = buildExportContent(fmt, pattern);
    const suggestedName = `${baseNameNoExt(loadedFileName) || 'pattern'}.${ext}`;

    ui.saveMenuConfirm.disabled = true;
    const savedName = await saveTextToFile(text, suggestedName, ext, mime);
    ui.saveMenuConfirm.disabled = false;

    if (savedName === null) return; // user cancelled the native dialog
    closeSaveMenu();
    showStatus(`Saved ${savedName} (${pattern.tth.length} points).`, 'info', 4000);
};
if (ui.saveAsButton)      ui.saveAsButton.addEventListener('click', openSaveMenu);
if (ui.saveFormatSelect)  ui.saveFormatSelect.addEventListener('change', updateSaveInfo);
if (ui.saveMenuCancel)    ui.saveMenuCancel.addEventListener('click', closeSaveMenu);
if (ui.saveMenuConfirm)   ui.saveMenuConfirm.addEventListener('click', doSave);
if (ui.saveMenuOverlay)   ui.saveMenuOverlay.addEventListener('click', (e) => {
    if (e.target === ui.saveMenuOverlay) closeSaveMenu();
});
