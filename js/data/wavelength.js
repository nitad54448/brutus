// js/data/wavelength.js
// Radiation: presets, file-wavelength classification, Kα2 helpers
// and the preset / wavelength controls.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// Kα1 / Kα2 from Bearden (1967), as tabulated in International Tables
// Vol. C, Table 4.2.2.1. ka_avg is the (2*ka1 + ka2)/3 centroid.
const WAVELENGTH_PRESETS = {
    'Cu': { ka1: 1.54056, ka2: 1.54439, ka_avg: 1.54184, ratio: 0.497 },
    'Co': { ka1: 1.78896, ka2: 1.79285, ka_avg: 1.79026, ratio: 0.497 },
    'Fe': { ka1: 1.93604, ka2: 1.93998, ka_avg: 1.93735, ratio: 0.497 },
    'Mo': { ka1: 0.70930, ka2: 0.71359, ka_avg: 0.71073, ratio: 0.497 },
    'Cr': { ka1: 2.28970, ka2: 2.29361, ka_avg: 2.29100, ratio: 0.497 },
    // Ag: Kα1 0.5594075, Kα2 0.563798.
    'Ag': { ka1: 0.55941, ka2: 0.56380, ka_avg: 0.56087, ratio: 0.497 },
    'custom': { ka1: null, ka2: null, ka_avg: null, ratio: 0.5 }
};
// Tolerance for recognising a preset line in a data file. Kα1 and the
// Kα average are only ~0.0013 Å apart for every anode, so the old 0.005
// matched BOTH and, testing the average first, turned every Kα1 file into
// a Kα-average one -- a 0.08% error on every d-spacing.
const PRESET_MATCH_TOL = 3e-4;
// Closest preset line to `wl` within PRESET_MATCH_TOL, or null.
//   -> { element: 'Cu', line: 'ka1' | 'avg' | 'ka2', diff }
const nearestPresetLine = (wl) => {
    if (!(wl > 0)) return null;
    let best = null;
    for (const [element, d] of Object.entries(WAVELENGTH_PRESETS)) {
        if (element === 'custom') continue;
        for (const [line, value] of [['ka1', d.ka1], ['avg', d.ka_avg], ['ka2', d.ka2]]) {
            const diff = Math.abs(value - wl);
            if (diff <= PRESET_MATCH_TOL && (!best || diff < best.diff)) best = { element, line, diff };
        }
    }
    return best;
};
/**
 * Turns the radiation description read from a data file into UI state.
 *   input : { ka1, ka2, ratio, intended, single }   (any may be missing)
 *           ka1/ka2/ratio - the doublet as recorded (ratio = I(Kα2)/I(Kα1))
 *           intended      - free-text hint, e.g. XRDML usedWavelength@intended
 *           single        - one wavelength with no doublet information
 *   output: { wavelength, presetMatch }   presetMatch e.g. 'Cu_avg', 'Mo_ka1', or null
 * A recorded doublet maps to the Kα-average preset (unresolved peaks sit at
 * the centroid, and Kα2 stripping becomes available); a monochromated or
 * single Kα1 line maps to the Kα1 preset; anything else is custom.
 */
const classifyFileWavelength = (radiation) => {
    const r = radiation || {};
    const ok = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0.05 && v < 5;
    const hint = String(r.intended || '').toLowerCase().replace(/[\s_-]+/g, '');
    const ka1Only = /alpha1$|ka1$|kα1$/.test(hint) || r.ratio === 0;

    if (ok(r.ka1) && ok(r.ka2) && r.ka2 > r.ka1 && !ka1Only) {
        const m = nearestPresetLine(r.ka1);
        if (m && m.line === 'ka1') {
            return { wavelength: WAVELENGTH_PRESETS[m.element].ka_avg, presetMatch: `${m.element}_avg` };
        }
        // Unknown anode: intensity-weighted centroid of the doublet.
        const w = (ok(r.ratio) && r.ratio < 1.5) ? r.ratio : 0.5;
        return { wavelength: (r.ka1 + w * r.ka2) / (1 + w), presetMatch: null };
    }
    const mono = ok(r.ka1) ? r.ka1 : (ok(r.single) ? r.single : null);
    if (mono === null) return { wavelength: null, presetMatch: null };
    const m = nearestPresetLine(mono);
    if (m && m.line === 'ka1') {
        return { wavelength: WAVELENGTH_PRESETS[m.element].ka1, presetMatch: `${m.element}_ka1` };
    }
    if (m && m.line === 'avg' && !ka1Only) {
        return { wavelength: WAVELENGTH_PRESETS[m.element].ka_avg, presetMatch: `${m.element}_avg` };
    }
    return { wavelength: mono, presetMatch: null };
};
/**
 * Returns the currently-active preset object IF the radiation has a
 * meaningful Ka2 component (i.e. preset is _avg AND we have ka1+ka2).
 * Returns null for Ka1-only setups, custom monochromatic, or stripped data.
 * Used by the Ka2-suspect tagger.
 */
const getActiveKa2Preset = () => {
    const sel = ui.wavelengthPreset?.value;
    if (!sel || sel === 'custom') return null;
    const [element, type] = sel.split('_');
    if (type === 'ka1') return null; // pure Ka1, no doublet to expect
    const data = WAVELENGTH_PRESETS[element];
    if (!data || !data.ka1 || !data.ka2) return null;
    return data;
};
// The one rule for the "Strip K-alpha2" control: it is meaningful only while
// a K-alpha AVERAGE preset is selected. Custom radiation has no known doublet
// and a Ka1 preset has no Ka2 to remove, so stripping either subtracts a line
// that is not there. The preset and wavelength handlers already applied this;
// setUIState, updateWorkingData and clearLoadedFile now use it too.
const ka2StripAllowed = () => getActiveKa2Preset() !== null;
/**
 * Returns the expected 2θ position (deg) of the Ka2 ghost of a parent
 * Ka1 line at parentTthDeg, given the active doublet preset.
 * Returns NaN if the geometry is invalid (arg >= 1 means high-angle clip).
 */
const expectedKa2TthDeg = (parentTthDeg, preset) => {
    const theta1 = parentTthDeg * Math.PI / 360; // theta in rad
    const arg = (preset.ka2 / preset.ka1) * Math.sin(theta1);
    if (!(arg < 1)) return NaN;
    return 2 * Math.asin(arg) * 180 / Math.PI;
};
/**
 * Angle-adaptive tolerance for matching a peak to the expected Ka2 position.
 * The Ka1/Ka2 separation grows with 2θ; we let the tolerance grow with it
 * but never shrink below the user's configured 2θ error.
 * Returned tolerance is in degrees.
 */
const ka2MatchTolerance = (parentTthDeg, preset) => {
    const userTol = getTthError();
    // Expected doublet split at this angle
    const ka2Pred = expectedKa2TthDeg(parentTthDeg, preset);
    if (!isFinite(ka2Pred)) return userTol;
    const split = Math.abs(ka2Pred - parentTthDeg);
    // Tolerance = max(userTol, 25% of the expected split)
    // 25% is empirical: tight enough to avoid flagging unrelated peaks,
    // loose enough to catch real Ka2 lines slightly displaced by overlap.
    return Math.max(userTol, 0.25 * split);
};
/**
 * Intensity sanity check for a candidate Kα₂ companion.
 *
 * A Kα₂ line carries a fixed fraction of its Kα₁ parent — preset.ratio,
 * ≈0.497 for all the characteristic doublets. We accept a candidate only if
 * the measured height ratio is consistent with that, allowing generous
 * slack for peak-fitting error, background and partial overlap.
 *
 * The upper bound is the one that matters: a peak as tall as, or taller
 * than, its supposed parent CANNOT be that parent's ghost. Flagging it
 * would discard a real reflection from the space-group evidence. The lower
 * bound is looser, since a Kα₂ sitting on a falling background or partly
 * merged into a neighbour can measure low.
 *
 * If heights are unavailable we fall back to the old position-only
 * behaviour rather than silently rejecting everything.
 */
const KA2_RATIO_MAX_FACTOR = 1.7;  // vs expected ratio -> ~0.85 of parent
const KA2_RATIO_MIN_FACTOR = 0.35; // vs expected ratio -> ~0.17 of parent
const ka2RatioIsPlausible = (parent, child, preset) => {
    const hp = parent?.intensity ?? parent?.height;
    const hc = child?.intensity ?? child?.height;
    if (typeof hp !== 'number' || typeof hc !== 'number' ||
        !isFinite(hp) || !isFinite(hc) || hp <= 0) {
        return true; // no usable heights -> position-only, as before
    }
    const expected = (preset && preset.ratio) ? preset.ratio : 0.497;
    const observed = hc / hp;
    return observed <= expected * KA2_RATIO_MAX_FACTOR &&
           observed >= expected * KA2_RATIO_MIN_FACTOR;
};
/**
 * Walk pickedPeaks, set p.ka2Suspect = true on any peak whose 2θ matches
 * the predicted Ka2 ghost of an earlier (lower-2θ) peak. Also stores
 * p.ka2ParentIdx for traceability.
 *
 * Skipped entirely when:
 *   - preset has no Ka2 (custom or _ka1)
 *   - data has already been Ka2-stripped (the doublet is gone)
 *   - user has explicitly confirmed a peak as real (p.userConfirmedReal)
 *
 * Re-runs cleanly: clears flags before re-tagging.
 */
const flagKa2SuspectPeaks = () => {
    // Reset ALL flags first, unconditionally. This must clear
    // userConfirmedReal peaks too: a peak flagged under (say) Cu Ka and
    // then right-clicked as "real" would otherwise keep ka2Suspect = true
    // forever, since this loop skipped it and the tagging loop below also
    // skips it. That silently defeats the manual override — the peak stays
    // excluded from indexing and keeps counting as a soft violation.
    // Clearing here is safe because the tagging loop re-applies the flag
    // only to peaks that still match, and never to userConfirmedReal ones.
    //
    // It also guarantees that switching to a radiation WITHOUT a Ka2
    // component (Custom, or any *_ka1 preset) leaves no stale flags behind:
    // the reset runs before the `!preset` early return below, so every peak
    // is cleared and no Ka2-based demotion can survive the switch.
    pickedPeaks.forEach(p => {
        p.ka2Suspect = false;
        p.ka2ParentIdx = null;
        p.hasKa2Child = false; // becomes true if a Ka2-suspect child is found below
    });

    const preset = getActiveKa2Preset();
    // No doublet to expect: Custom radiation, any Ka1-only preset, or a
    // preset lacking ka1/ka2. Nothing is flagged, so no peak is demoted to
    // "soft" on Ka2 grounds and none is withheld from the indexing search.
    if (!preset) return;
    if (ui.stripKa2Checkbox?.checked) return; // already stripped

    // pickedPeaks is kept sorted by 2θ; rely on that.
    for (let i = 0; i < pickedPeaks.length; i++) {
        const parent = pickedPeaks[i];
        // A peak that is itself a Ka2-suspect cannot be a parent
        if (parent.ka2Suspect) continue;
        const tth2Pred = expectedKa2TthDeg(parent.tth, preset);
        if (!isFinite(tth2Pred)) continue;
        const tol = ka2MatchTolerance(parent.tth, preset);

        for (let j = i + 1; j < pickedPeaks.length; j++) {
            const q = pickedPeaks[j];
            // Stop early: 2θ values past tth2Pred + tol can't match
            if (q.tth - tth2Pred > tol) break;
            if (q.userConfirmedReal) continue; // user said this is real
            if (Math.abs(q.tth - tth2Pred) <= tol) {
                // Position alone is not enough. The Ka2 line is a fixed
                // fraction of its Ka1 parent (preset.ratio, ~0.497 for the
                // Cu/Co/Fe/Mo/Cr doublets), so a candidate sitting at the
                // right angle but carrying the wrong intensity is a genuine
                // reflection that happens to fall near the predicted ghost
                // position — not a ghost. Without this check any real peak
                // at the doublet spacing is silently demoted to soft
                // evidence and stops constraining the space group.
                if (!ka2RatioIsPlausible(parent, q, preset)) continue;
                q.ka2Suspect = true;
                q.ka2ParentIdx = i;
                parent.hasKa2Child = true; // parent is provably Kα1 → use λ_Kα1 for it
                break; // only one Ka2 per parent
            }
        }
    }
};
/**
 * Updates the yellow warning banner above the peak table, reflecting how
 * many peaks are currently flagged and reminding the user what it means.
 */
const updateKa2Banner = () => {
    const banner = document.getElementById('ka2-warning-banner');
    if (!banner) return;
    const n = pickedPeaks.filter(p => p.ka2Suspect).length;
    const preset = getActiveKa2Preset();
    if (!preset || ui.stripKa2Checkbox?.checked || n === 0) {
        banner.style.display = 'none';
        return;
    }
    const nParents = pickedPeaks.filter(p => p.hasKa2Child).length;
    banner.style.display = 'block';
    banner.innerHTML =
        `<b>⚠ ${n} peak${n > 1 ? 's' : ''} flagged as possible Kα₂ companion${n > 1 ? 's' : ''}.</b> ` +
        `Yellow rows are <b>excluded from indexing and FOM</b>, ` +
        `and count as <i>soft</i> violations in space-group analysis. ` +
        `Their ${nParents} Kα₁ parent${nParents > 1 ? 's' : ''} ` +
        `(marked with <b>*</b>) use λ_Kα₁ for the d-spacing. ` +
        `Right-click a yellow row to confirm it as real instead.`;
};
//turn off stripping if no average, v 123
// Updated: strip is now ENABLED BY DEFAULT for K-alpha average presets,
// since untreated Ka2 lines are the main source of false positives in
// space-group analysis. User can still uncheck if their data has
// already been deconvoluted.
/**
 * Apply a wavelength preset to the UI.
 *
 * @param {object} opts
 * @param {boolean} opts.onLoad - True when called during initial file load.
 *   File-load path: strip is enabled by default for Kα-doublet presets
 *   (the user hasn't expressed a preference yet, and stripping is the
 *   safer default for indexing).
 *   Manual path (default): strip is turned OFF whenever the user changes
 *   the preset, reverting the chart to the original (un-stripped) data.
 *   The user can re-tick the strip checkbox to apply Rachinger correction
 *   with the newly selected radiation's constants — fresh from the
 *   original data, never re-stripping already-stripped data.
 */
const handleWavelengthPresetChange = (opts = {}) => {
    // Note: opts.onLoad is still accepted for call-site compatibility, but
    // preset behaviour is now identical on load and on manual change (strip
    // OFF, λ = Kα-avg for _avg presets), so it is no longer read here.
    const selection = ui.wavelengthPreset.value;

    if (selection === 'custom') {
        ui.stripKa2Checkbox.checked = false;
        ui.stripKa2Checkbox.disabled = true;
    } else {
        const [element, type] = selection.split('_'); 
        const data = WAVELENGTH_PRESETS[element];

        if (data) {
            if (type === 'ka1') {
                ui.wavelength.value = data.ka1.toFixed(5);
                // Disable and uncheck for Ka1 (no Ka2 present)
                ui.stripKa2Checkbox.checked = false;
                ui.stripKa2Checkbox.disabled = true;
            } else {
                // Kα-average preset.
                //   Strip is OFF by default (both on file load and on manual
                //   preset change): the user opts in to stripping explicitly.
                //   λ defaults to Kα-avg, the correct value for raw doublet
                //   data. When the user later ticks Strip, the handler swaps
                //   λ to Kα1 (stripping leaves peaks at the Kα1 position).
                ui.stripKa2Checkbox.disabled = false;
                ui.stripKa2Checkbox.checked = false;
                ui.wavelength.value = data.ka_avg.toFixed(5);
            }
        }
    }
    
    debouncedUpdateAndRedraw();
    recalculatePeakValues();
    updatePeakTable();
};
/**
 * Handles manual edits to the wavelength input field.
 * If the typed value differs from the currently-selected preset's expected
 * value, switch the preset dropdown to "custom" and disable Kα2 stripping.
 * This runs on every keystroke but is cheap.
 */
const handleWavelengthValueChange = () => {
    const selection = ui.wavelengthPreset.value;
    if (selection === 'custom') return; // already custom; nothing to do

    const typed = parseFloat(ui.wavelength.value);
    if (!isFinite(typed)) return; // mid-edit empty field or invalid; wait

    const [element, type] = selection.split('_');
    const data = WAVELENGTH_PRESETS[element];
    if (!data) return;
    // When strip is ON, peaks are at Kα1 positions, so the wavelength
    // input is expected to hold ka1 (not ka_avg). When strip is OFF on
    // an _avg preset, the input holds ka_avg.
    const stripOn = !!ui.stripKa2Checkbox?.checked;
    let expected;
    if (type === 'ka1') expected = data.ka1;
    else if (stripOn)   expected = data.ka1;     // _avg + strip → ka1
    else                expected = data.ka_avg;  // _avg + no strip → avg

    // Compare rounded to 5 dp, matching the precision at which we set preset values.
    if (Math.abs(typed - expected) > 5e-6) {
        ui.wavelengthPreset.value = 'custom';
        // Switching to custom disables Kα2 stripping because the preset no longer
        // describes a Kα1/Kα_avg pair.
        ui.stripKa2Checkbox.checked = false;
        ui.stripKa2Checkbox.disabled = true;
        debouncedUpdateAndRedraw();
        recalculatePeakValues();
        updatePeakTable();
    }
};
// event listeners 
ui.wavelengthPreset.addEventListener('change', handleWavelengthPresetChange);
// Edits to the wavelength input auto-switch the preset to "custom" when the
// typed value no longer matches the selected preset.
const debouncedWavelengthChange = debounce(() => {
const typed = parseFloat(ui.wavelength.value);
// FIX: Only trigger heavy recalculations if the user typed a valid, realistic wavelength
if (isFinite(typed) && typed >= 0.1 && typed <= 10.0) {
    handleWavelengthValueChange();
}
}, 400); // Wait 400ms after they finish typing
ui.wavelength.addEventListener('input', debouncedWavelengthChange);
