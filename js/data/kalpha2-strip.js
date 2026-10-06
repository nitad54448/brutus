// js/data/kalpha2-strip.js
// Kα2 stripping and the working (plotted) data.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

/**
 * Kα2 removal by exact inversion of the doublet in q-space.
 *
 * Under the Kα1 coordinate u, the measured profile is y(u) = F(u) + r F(u s)
 * with s = λKα1/λKα2 and r = I(Kα2)/I(Kα1); F, the pure Kα1 pattern, is
 * recovered as the alternating series F(u) = Σ_k (-r)^k y(u s^k), evaluated
 * directly on the data with cubic interpolation (details below).
 *
 * The first version followed the dual-wavelength q-space differencing of
 * Zhang et al., Measurement 276 (2026) 121448; the comments below explain
 * why the series replaced the single difference.
 */
const stripZhang = (tth, intensity, ka1, ka2, ratio) => {
    const n = tth.length;
    if (n === 0) return [];
    if (n === 1) return [intensity[0]];
    // Fixed at 8. r^8 ~ 0.004, so further terms change nothing measurable;
    // below 8 the series has not converged and broad peaks show a residual.
    // This was briefly exposed as a slider and removed: the value is flat
    // for sharp peaks, where the limit is interpolation order, not K.
    const K_TERMS = 8;

    // Step 1: Coordinate mapping to q-space for both wavelengths
    const q_a1 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        q_a1[i] = (4 * Math.PI * Math.sin(tth[i] * Math.PI / 360)) / ka1;
    }
    // The lambda2 mapping and the uniform resampling grid (q_a2, q_min,
    // q_max, M, dq, q_grid) are gone: the series below evaluates directly
    // on q_a1, so the intermediate grid and its second interpolation pass
    // are no longer needed.

    // Helper: fast binary-search linear interpolation
    const interpLinear = (target, x_src, y_src) => {
        const len = x_src.length;
        if (target <= x_src[0]) return y_src[0];
        if (target >= x_src[len - 1]) return y_src[len - 1];
        let lo = 0, hi = len - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (x_src[mid] <= target) lo = mid;
            else hi = mid;
        }
        const t = (target - x_src[lo]) / (x_src[hi] - x_src[lo]);
        return y_src[lo] * (1 - t) + y_src[hi] * t;
    };

    // Catmull-Rom cubic. The series below evaluates the profile at
    // u * s^k, which almost never lands on a channel centre, so every term
    // pays an interpolation error. On a sharp peak that error dominates
    // everything else: at 2 channels per FWHM, linear interpolation leaves
    // a max residual of ~28 counts on a 600-count peak and MORE terms do
    // not help (K=4 gives 27.4, K=20 gives 28.2). Cubic more than halves it
    // (12.0), and at 3-4.5 channels per FWHM it is a factor of 3-4 better.
    // Cost is a handful of extra multiplies per evaluation.
    const interp = (target, x_src, y_src) => {
        const len = x_src.length;
        if (len < 4) return interpLinear(target, x_src, y_src);
        if (target <= x_src[0]) return y_src[0];
        if (target >= x_src[len - 1]) return y_src[len - 1];
        let lo = 0, hi = len - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (x_src[mid] <= target) lo = mid;
            else hi = mid;
        }
        const t = (target - x_src[lo]) / (x_src[hi] - x_src[lo]);
        const p0 = y_src[lo > 0 ? lo - 1 : 0];
        const p1 = y_src[lo];
        const p2 = y_src[hi];
        const p3 = y_src[hi < len - 1 ? hi + 1 : len - 1];
        const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        const b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        const c = -0.5 * p0 + 0.5 * p2;
        return ((a * t + b) * t + c) * t + p1;
    };

    // Step 2, 3, 4 & 5: exact inversion of the doublet convolution.
    //
    // Mapped into q under the lambda1 coordinate, the observed profile is
    //     y1(u) = F(u) + r * F(u*s),     s = ka1/ka2 < 1
    // where F is the pure Kalpha1 pattern we want. That recursion inverts
    // in closed form as an alternating geometric series:
    //     F(u) = SUM_k (-r)^k * y1(u * s^k)
    //
    // The previous implementation computed a single difference,
    // y2(q) - r*y1(q), and mapped it back through q_a2. Expanding that:
    //     y2(q) - r*y1(q) = F(q/s) - r^2 * F(q*s)
    // The first term is correct (q_a2[i]/s == q_a1[i]), but the second is a
    // leftover NEGATIVE ghost at r^2 ~ 24.7% of the parent peak, sitting at
    // twice the Kalpha1-Kalpha2 separation above it. Clamped at zero it
    // showed up as flat zero-valued holes on the high-angle flank of every
    // strong peak, with ringing around them; before the clamp was corrected
    // a bogus floor of (1-ratio)*min(y1,y2) had been hiding the ghost while
    // leaving ~50% of every Kalpha2 satellite in place.
    //
    // The series has no such residual, needs no uniform resampling grid,
    // and avoids the double interpolation (data -> q_grid -> output) that
    // fed the ringing. Verified against synthetic doublets with known
    // ground truth: RMS error 1.23 vs 1.35 for classic Rachinger, zero
    // clamped points, and no noise amplification relative to Rachinger.
    //
    // Convergence: r^k must become negligible. For Cu (r = 0.497), K = 6
    // still leaves a visible hole, K = 8 leaves none, K = 12 gains nothing.
    // Raising K helps BROAD peaks (15 channels/FWHM: max error 37.3 at K=4
    // vs 2.4 at K=8) and does nothing for sharp ones, where interpolation
    // order is the limit instead -- hence the cubic interp above.
    // A useful side effect is that a flat background sums to bkg/(1+r),
    // which is the correct Kalpha1-only background level.
    const s = ka1 / ka2;
    const corrected = new Array(n);
    for (let i = 0; i < n; i++) {
        let acc = 0, coef = 1, u = q_a1[i];
        for (let k = 0; k < K_TERMS; k++) {
            acc += coef * interp(u, q_a1, intensity);
            coef *= -ratio;
            u *= s;
        }
        corrected[i] = Math.max(acc, 0);
    }

    return corrected;
};
const updateWorkingData = () => {
    if (fullExperimentalData.tth.length === 0) {
        workingExperimentalData = { tth: [], intensity: [] };
        return;
    }

    // Strip only when requested AND the preset has a Ka2 to remove (an
    // average preset -- never custom or Ka1; see ka2StripAllowed).
    if (ui.stripKa2Checkbox.checked && ka2StripAllowed()) {
        
        const selection = ui.wavelengthPreset.value;
        const element = selection.split('_')[0]; // "Cu", "Co", etc.
        
        const preset = WAVELENGTH_PRESETS[element];
        
        if (preset) {
            const { tth, intensity } = fullExperimentalData;
            // Apply Kα2 elimination (series inversion of the doublet)
            const strippedIntensity = stripZhang(tth, intensity, preset.ka1, preset.ka2, preset.ratio);
            
            // Save this as the working data for plotting AND peak search
            workingExperimentalData = { tth: tth, intensity: strippedIntensity };
        } else {
            // Fallback if preset parsing fails
            workingExperimentalData = fullExperimentalData;
        }

    } else {
        // No stripping requested, use raw data
        workingExperimentalData = fullExperimentalData;
    }
};
// Recompute the stripped profile and refresh everything downstream of it.
// Shared by the strip checkbox and the K slider so the two can never drift.
const refreshAfterStripChange = () => {
    updateWorkingData(); // Calculates new intensity

    // Redraw Chart
    if (xrdChart) {
        // d and Q are functions of lambda, so a new wavelength moves every
        // abscissa on those axes - the whole plot has to be rebuilt, not
        // just re-fed with intensities. On theta / 2-theta nothing moves.
        if (xAxisMode === 'd' || xAxisMode === 'q') {
            rebuildPlot(false);
        } else {
            setExperimentalTrace(false);
        }
    }

    // Re-find peaks on the (possibly newly stripped) data using the
    // updated wavelength. findPeaks rebuilds pickedPeaks from scratch
    // so no separate recalculatePeakValues call is needed.
    findPeaks();
};
if (ui.stripKa2Checkbox) {
    ui.stripKa2Checkbox.addEventListener('change', () => {
        // Sync the wavelength input to match the active radiation:
        //   strip ON  → peaks are at Kα1, so λ = ka1
        //   strip OFF → peaks are at the doublet centroid, so λ = ka_avg
        const sel = ui.wavelengthPreset.value;
        if (sel !== 'custom') {
            const [element, type] = sel.split('_');
            const data = WAVELENGTH_PRESETS[element];
            if (data && type !== 'ka1') {
                ui.wavelength.value = ui.stripKa2Checkbox.checked
                    ? data.ka1.toFixed(5)
                    : data.ka_avg.toFixed(5);
            }
        }

        refreshAfterStripChange();
    });
}
