// js/data/export.js
// Save-as: exports the plotted pattern in several formats.
//
// Classic script, loaded in order by index.html (see the list there); its
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
// Radiation to record in exported files. An average preset with no
// stripping means the exported pattern still holds the Kα doublet, so the
// file says so (both lines and their ratio); otherwise it is monochromatic
// at the current wavelength. Re-importing then restores the same preset.
const getExportRadiation = () => {
    const lambda = getWavelength();
    const [element, type] = String(ui.wavelengthPreset ? ui.wavelengthPreset.value : 'custom').split('_');
    const p = WAVELENGTH_PRESETS[element];
    const stripped = !!(ui.stripKa2Checkbox && ui.stripKa2Checkbox.checked);
    if (p && p.ka1 && type === 'avg' && !stripped) {
        return { doublet: true, lambda, ka1: p.ka1, ka2: p.ka2, ratio: p.ratio };
    }
    return { doublet: false, lambda };
};
const num = (v, dp = 6) => Number(v).toFixed(dp);
const patternStepIsIrregular = (pattern = getExportPattern()) => {
    const t = pattern && pattern.tth;
    if (!t || t.length < 3) return false;
    const step = (t[t.length - 1] - t[0]) / (t.length - 1);
    return t.some((value, i) => Math.abs(value - (t[0] + i * step)) > BRUTUS_LIMITS.exportUniformGridToleranceDeg);
};
const buildExportContent = (fmt, pattern) => {
    const { tth, intensity } = pattern;
    const n = tth.length;
    const lambda = getWavelength();
    const step = n > 1 ? (tth[n - 1] - tth[0]) / (n - 1) : 0;
    const irregular = patternStepIsIrregular(pattern);
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
            out.push('_STEPCOUNT=' + n);
            out.push('_2THETACOUNTS');
            for (let i = 0; i < n; i++) out.push(`${tth[i]} ${intensity[i]}`);
            return { text: out.join('\n') + '\n', ext: 'uxd', mime: 'text/plain' };
        }
        case 'gsas': {
            if (irregular) {
                out.push('Exported by Brutus');
                out.push(`BANK 1 ${n} ${n} CONST ${tth[0] * 100} ${step * 100} 0 0 FXYE`);
                for (let i = 0; i < n; i++)
                    out.push(`${tth[i] * 100} ${intensity[i]} ${Math.sqrt(Math.max(1, intensity[i]))}`);
                return { text: out.join('\n') + '\n', ext: 'esd', mime: 'text/plain' };
            }
            // GSAS ESD: title, BANK line, then five (Y, sigma) pairs per
            // record in 10F8.x fields -- intensity FIRST, which is how GSAS,
            // GSAS-II and FullProf read it. This used to write (sigma, Y) in
            // 12-character fields to suit this app's old reader, so every
            // other program took the sigma column for the pattern.
            //
            // CONST start/step are centidegrees. Retain full precision:
            // rounding a step accumulates an angle error across the scan.
            // As many decimals as fit, so normalised (0-1) or stripped data keep
            // their values: a fixed 2 decimals wrote 0.001-scale data as 0.00.
            // At most 7 characters, so a blank always separates the fields.
            const f8 = (v) => {                   // right-justified 8-char field
                if (!Number.isFinite(v)) v = 0;
                for (let dp = 6; dp >= 1; dp--) {
                    const s = v.toFixed(dp);
                    if (s.length <= 7) return s.padStart(8);
                }
                // Large counts: the full 8 characters, as before, sooner than
                // dropping the decimal altogether.
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
                `BANK 1 ${n} ${dataLines.length} CONST ${tth[0] * 100} ` +
                    `${step * 100} 0 0 ESD`,
                ...dataLines,
            ];
            return { text: lines.join('\n') + '\n', ext: 'esd', mime: 'text/plain' };
        }
        case 'xrdml': {
            const positions = `        <positions axis="2Theta" unit="deg">\n` +
                `          <listPositions>${tth.join(' ')}</listPositions>\n` +
                `        </positions>`;
            // Not rounded to integers: normalised or Kα2-stripped data would
            // collapse to 0/1. 6 decimals is the precision held since loading.
            const counts = intensity.map(v => Number(v.toFixed(6))).join(' ');
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
            // DIFFRAC RawData layout (the RawData0.xml inside a .brml), the
            // subset its readers use: one <Datum> per step (time, absorber
            // factor, 2θ, θ, counts) with the columns declared in <DataViews>,
            // and the tube lines. Each row carries its own 2θ, so an uneven
            // scan is written as it is.
            const rad = getExportRadiation();
            const tubeLines = rad.doublet
                ? `<WaveLengthAlpha1 Unit="Å" Value="${num(rad.ka1, 6)}" />\n` +
                  `                <WaveLengthAlpha2 Unit="Å" Value="${num(rad.ka2, 6)}" />\n` +
                  `                <WaveLengthAverage Unit="Å" Value="${num(rad.lambda, 6)}" />\n` +
                  `                <WaveLengthRatio Unit="" Value="${num(rad.ratio, 4)}" />`
                : `<WaveLengthAlpha1 Unit="Å" Value="${num(lambda, 6)}" />`;
            const rows = tth.map((t, i) => `      <Datum>1,1,${t},${Number((t / 2).toFixed(8))},${Number(intensity[i].toFixed(6))}</Datum>`).join('\n');
            const xml =
`<?xml version="1.0" encoding="utf-8"?>
<RawData xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <MeasurementStatus>Measured</MeasurementStatus>
  <DataRoutes>
    <DataRoute RouteFlag="Measured" Description="Exported by Brutus.">
      <ScanInformation ScanName="LockedCoupled" VisibleName="Coupled TwoTheta/Theta">
        <MeasurementPoints>${n}</MeasurementPoints>
        <TimePerStep>1</TimePerStep>
        <TimePerStepEffective>1</TimePerStepEffective>
        <ScanAxes>
          <ScanAxisInfo AxisId="TwoTheta" AxisName="TwoTheta" VisibleName="2Theta" Unit="°">
            <Start>${tth[0]}</Start>
            <Stop>${tth[n - 1]}</Stop>
            <Increment>${step}</Increment>
          </ScanAxisInfo>
        </ScanAxes>
      </ScanInformation>
${rows}
      <DataViews>
        <RawDataView xsi:type="FixedRawDataView" Start="0" Length="1" LogicName="MeasuredTime" Unit="s" />
        <RawDataView xsi:type="FixedRawDataView" Start="1" Length="1" LogicName="AbsorptionFactor" Unit="" />
        <RawDataView xsi:type="VaryingRawDataView" Start="2" Length="2">
          <Varying xsi:type="VaryingStepData2D" LogicName="ScanAxes" VectorLength="2">
            <FieldDefinitions FieldName="TwoTheta" AxisId="TwoTheta" />
            <FieldDefinitions FieldName="Theta" AxisId="Theta" />
          </Varying>
        </RawDataView>
        <RawDataView xsi:type="RecordedRawDataView" Start="4" Length="1" MeasuredDataIndex="0" />
      </DataViews>
    </DataRoute>
  </DataRoutes>
  <FixedInformation>
    <Instrument>
      <PrimaryTracks>
        <TrackInfoData>
          <MountedOptics>
            <InfoData xsi:type="TubeMountInfoData" LogicName="TubeMount">
              <Tube LogicName="Tube">
                ${tubeLines}
              </Tube>
            </InfoData>
          </MountedOptics>
        </TrackInfoData>
      </PrimaryTracks>
    </Instrument>
  </FixedInformation>
</RawData>
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
// Export capabilities shown in the save dialog. Every format keeps the
// explicit angle of each point, so none needs an evenly spaced scan.
const SAVE_FORMAT_META = {
    xrdml:     { wavelength: true,  constStep: false, label: 'XRDML' },
    brukerxml: { wavelength: true,  constStep: false, label: 'Bruker RawData XML' },
    uxd:       { wavelength: true,  constStep: false, label: 'UXD' },
    gsas:      { wavelength: false, constStep: false, label: 'GSAS' },
    xy:        { wavelength: false, constStep: false, label: 'XY' },
    csv:       { wavelength: false, constStep: false, label: 'CSV' },
    dat:       { wavelength: false, constStep: false, label: 'DAT' }
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
        const w = getWavelength();
        notes.push(`${meta.label} stores a wavelength — the value from the wavelength box (${w.toFixed(5)} Å) will be written.`);
    }
    const incompatible = meta.constStep && patternStepIsIrregular(p);
    if (incompatible) notes.push(`${meta.label} requires evenly spaced angles. Choose XY, CSV, UXD, XRDML or GSAS to preserve this scan.`);
    ui.saveMenuConfirm.disabled = !!incompatible;
    ui.saveMenuMsg.innerHTML = notes.length ? 'Info: ' + notes.join('<br>Info: ') : '';
};
let saveMenuReturnFocus = null;
const saveMenuFocusables = () =>
    [ui.saveFormatSelect, ui.saveMenuCancel, ui.saveMenuConfirm].filter(el => el && !el.disabled);
const focusSaveMenu = () => {
    const target = saveMenuFocusables()[0] || document.getElementById('save-menu-modal');
    if (target) target.focus();
};
const openSaveMenu = () => {
    const p = getExportPattern();
    if (!p) { showStatus('No data to save.', 'error'); return; }
    updateSaveInfo();
    saveMenuReturnFocus = document.activeElement;
    ui.saveMenuOverlay.classList.add('open');
    focusSaveMenu();
};
const closeSaveMenu = () => {
    if (!ui.saveMenuOverlay.classList.contains('open')) return;
    ui.saveMenuOverlay.classList.remove('open');
    const target = saveMenuReturnFocus;
    saveMenuReturnFocus = null;
    if (target && target.isConnected && !target.disabled) target.focus();
};
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

    ui.saveMenuConfirm.disabled = true;
    try {
        const { text, ext, mime } = buildExportContent(ui.saveFormatSelect.value, pattern);
        const suggestedName = `${baseNameNoExt(loadedFileName) || 'pattern'}.${ext}`;
        const savedName = await saveTextToFile(text, suggestedName, ext, mime);
        if (savedName === null) return; // user cancelled the native dialog
        closeSaveMenu();
        showStatus(`Saved ${savedName} (${pattern.tth.length} points).`, 'info', 4000);
    } catch (error) {
        showStatus(`Could not export pattern: ${error.message}`, 'error');
    } finally { updateSaveInfo(); }
};
if (ui.saveAsButton)      ui.saveAsButton.addEventListener('click', openSaveMenu);
if (ui.saveFormatSelect)  ui.saveFormatSelect.addEventListener('change', updateSaveInfo);
if (ui.saveMenuCancel)    ui.saveMenuCancel.addEventListener('click', closeSaveMenu);
if (ui.saveMenuConfirm)   ui.saveMenuConfirm.addEventListener('click', doSave);
if (ui.saveMenuOverlay)   ui.saveMenuOverlay.addEventListener('click', (e) => {
    if (e.target === ui.saveMenuOverlay) closeSaveMenu();
});
document.addEventListener('keydown', (e) => {
    if (!ui.saveMenuOverlay || !ui.saveMenuOverlay.classList.contains('open')) return;
    if (e.key === 'Escape') {
        e.preventDefault();
        closeSaveMenu();
    } else if (e.key === 'Tab') {
        const controls = saveMenuFocusables();
        const index = controls.indexOf(document.activeElement);
        if (index < 0 || (!e.shiftKey && index === controls.length - 1) || (e.shiftKey && index === 0)) {
            e.preventDefault();
            const target = e.shiftKey ? controls[controls.length - 1] : controls[0];
            if (target) target.focus();
            else focusSaveMenu();
        }
    }
});
document.addEventListener('focusin', (e) => {
    if (!ui.saveMenuOverlay || !ui.saveMenuOverlay.classList.contains('open')) return;
    const modal = document.getElementById('save-menu-modal');
    if (modal && !modal.contains(e.target)) focusSaveMenu();
});
