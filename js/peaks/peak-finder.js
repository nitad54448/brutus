// js/peaks/peak-finder.js
// Background estimation, smoothing and peak picking.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// Convert a rolling-ball radius expressed in Q (A^-1) into a per-point
// half-width in channels.
//
// The radius used to be a raw point count, which made it a property of the
// FILE rather than of the sample: the same specimen measured at 0.01 deg
// and 0.02 deg per step needed two different slider settings to get the
// same background, and a value tuned on one diffractometer was meaningless
// on another. Q = 4*pi*sin(theta)/lambda is the physical axis the
// background actually lives on, so a Q radius transfers between scans.
//
// Q is not linear in 2-theta: dQ/d(2theta) = (2*pi*cos(theta)/lambda), which
// SHRINKS as the angle rises. A fixed Q window therefore spans progressively
// MORE channels toward the back of the scan (typically 2-4x over a lab
// range), which is why this returns an array and not a scalar.
const qRadiusToPoints = (tth, deltaQ, lambda, minR = 1) => {
    const n = tth.length;
    const radii = new Int32Array(n);
    if (n === 0) return radii;
    // A window wider than a quarter of the scan is already flattening real
    // structure; past that it only costs time.
    const maxR = Math.max(1, Math.floor(n / 4));
    if (!isFinite(deltaQ) || deltaQ <= 0 || !isFinite(lambda) || lambda <= 0) {
        radii.fill(Math.min(minR, maxR));
        return radii;
    }
    const qAt = (t) => scatteringQ(t, lambda);   // A^-1 scattering vector, NOT 1/d^2
    let prev = -1;
    for (let i = 0; i < n; i++) {
        const a = (i > 0) ? i - 1 : 0;
        const b = (i < n - 1) ? i + 1 : n - 1;
        const span = b - a;
        const dQ = span > 0 ? (qAt(tth[b]) - qAt(tth[a])) / span : 0;
        let r = (dQ > 1e-12) ? Math.round(deltaQ / dQ) : maxR;
        if (!isFinite(r) || r < minR) r = minR;
        if (r > maxR) r = maxR;
        // The running min/max below uses a monotonic deque, which requires
        // both window edges to advance monotonically. On a normal ascending
        // 2-theta scan dQ/di decreases, so r is naturally non-decreasing and
        // grows by well under one channel per step; these two clamps only
        // bite on malformed or non-monotonic 2-theta axes, where they cost a
        // slightly asymmetric window instead of a wrong answer.
        if (prev >= 0) {
            if (r < prev) r = prev;
            else if (r > prev + 1) r = prev + 1;
        }
        radii[i] = r;
        prev = r;
    }
    return radii;
};
// Running min (isMin) or max over a window whose half-width varies per
// point. Monotonic-deque, O(n) total regardless of radius -- the previous
// fixed-radius version was O(n*r), and a Q radius can reach several hundred
// channels at high angle on a fine scan, where O(n*r) would stall the
// slider drag.
const runningExtreme = (src, radii, isMin) => {
    const n = src.length;
    const out = new Float64Array(n);
    const dq = new Int32Array(n);
    let head = 0, tail = 0, nextIn = 0, prevStart = 0;
    for (let i = 0; i < n; i++) {
        const r = radii[i];
        const end = Math.min(n - 1, i + r);
        let start = i - r;
        if (start < prevStart) start = prevStart;
        if (start < 0) start = 0;
        prevStart = start;
        while (nextIn <= end) {
            const v = src[nextIn];
            while (tail > head && (isMin ? src[dq[tail - 1]] >= v : src[dq[tail - 1]] <= v)) tail--;
            dq[tail++] = nextIn++;
        }
        while (head < tail && dq[head] < start) head++;
        out[i] = src[dq[head]];
    }
    return out;
};
const rollingBallBackground = (y, radii, smoothRadii) => {
    const n = y.length;
    if (n === 0 || !radii || radii.length !== n) return new Array(n).fill(0);
    let smoothed_y = y;
    if (smoothRadii && smoothRadii.length === n) {
        // Same moving average as before -- start = max(0, i-hw),
        // end = min(n, i+hw+1), sum/(end-start) -- but the half-width is now
        // per point, because it comes from a Q window rather than a raw
        // channel count. Computed from a prefix sum so the cost is O(n)
        // instead of O(n*hw): a Q smoothing width maps to a growing number
        // of channels toward high angle, exactly like the ball radius.
        //
        // Prefix-sum cancellation is not a concern at this scale: for 2e4
        // points of ~1e6 counts the running total reaches ~1e10, where f64
        // still resolves ~1e-6 absolute, i.e. ~1e-11 relative on a window
        // sum.
        let any = false;
        for (let i = 0; i < n; i++) { if (smoothRadii[i] > 0) { any = true; break; } }
        if (any) {
            const prefix = new Float64Array(n + 1);
            for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + y[i];
            const sm = new Float64Array(n);
            for (let i = 0; i < n; i++) {
                const hw = smoothRadii[i];
                const a = Math.max(0, i - hw);
                const b = Math.min(n, i + hw + 1);
                sm[i] = (prefix[b] - prefix[a]) / (b - a);
            }
            smoothed_y = sm;
        }
    }
    // Erosion then dilation, exactly as before.
    const eroded = runningExtreme(smoothed_y, radii, true);
    return runningExtreme(eroded, radii, false);
};
const savitzkyGolay = (data, windowSize = 9, polyOrder = 2) => {
    const n = data.length; if (n === 0) return [];
    windowSize = Math.max(3, windowSize); if (windowSize % 2 === 0) windowSize += 1; windowSize = Math.min(windowSize, n);
    const halfWindow = Math.floor(windowSize / 2);
    const result = new Array(n);
    const SG_COEFFS = {
        5: [-3/35, 12/35, 17/35, 12/35, -3/35],
        7: [-2/21, 3/21, 6/21, 7/21, 6/21, 3/21, -2/21],
        9: [-21/231, 14/231, 39/231, 54/231, 59/231, 54/231, 39/231, 14/231, -21/231],
        11: [-36/429, 9/429, 44/429, 69/429, 84/429, 89/429, 84/429, 69/429, 44/429, 9/429, -36/429]
    };
    const coefficients = (polyOrder === 2 && SG_COEFFS[windowSize]) ? SG_COEFFS[windowSize] : (() => { const weights = []; for (let i = -halfWindow; i <= halfWindow; i++) weights.push(1 - Math.abs(i) / (halfWindow + 1)); const sum = weights.reduce((a, b) => a + b, 0); return weights.map(w => w / sum); })();
    for (let i = 0; i < n; i++) {
        let smoothedValue = 0;
        for (let j = -halfWindow; j <= halfWindow; j++) {
            let idx = i + j;
            if (idx < 0) idx = Math.abs(idx);
            else if (idx >= n) idx = n - 1 - (idx - (n - 1));
            smoothedValue += data[idx] * coefficients[j + halfWindow];
        }
        result[i] = smoothedValue;
    }
    return result;
};
// In-place quickselect: returns the k-th smallest element of `arr` (0-based),
// reordering `arr` as a side effect. Average O(n) vs the O(n log n) of a full
// sort, which matters because findPeaks (and its median-of-absolute-deviations
// noise estimate) runs on every slider drag over the full-resolution scan.
// Result for a given k is identical to arr.slice().sort()[k].
const quickselect = (arr, k) => {
    let lo = 0, hi = arr.length - 1;
    while (lo < hi) {
        // Median-of-three pivot to avoid O(n^2) on sorted/near-sorted input
        // (background-corrected intensities are far from random).
        const mid = (lo + hi) >> 1;
        const a = arr[lo], b = arr[mid], c = arr[hi];
        const pivot = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
        let i = lo, j = hi;
        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) { const t = arr[i]; arr[i] = arr[j]; arr[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return arr[k];
};
function findPeaks() {
    // Now uses workingExperimentalData
    if (!workingExperimentalData || !workingExperimentalData.intensity || workingExperimentalData.intensity.length < 5) return;
    const _perfEnd = perfStart('findPeaks');
    
    const { intensity, tth } = workingExperimentalData; const n = tth.length;
    const minTth = parseFloat(ui.tthMinSlider.value) || tth[0];
    const maxTth = parseFloat(ui.tthMaxSlider.value) || tth[n - 1];
    const minHeightPercent = logSliderToValue(parseFloat(ui.peakThresholdSlider.value)) || 2;
    // Radius and Smoothing are both Q windows (A^-1), converted here to
    // per-channel half-widths using this scan's own 2-theta axis and
    // wavelength. Sharing one unit makes the constraint visible: Smoothing
    // must stay well under Radius or the moving average erodes the peaks
    // the rolling ball is supposed to sit under.
    const ballRadiusQ = parseFloat(ui.ballRadiusSlider.value);
    const smoothQ = parseFloat(ui.smoothingWidthSlider.value);
    const lambdaForBg = getWavelength();
    const ballRadii = qRadiusToPoints(tth, ballRadiusQ, lambdaForBg);
    // minR = 0 so the slider's zero position genuinely disables smoothing.
    const smoothRadii = (smoothQ > 0) ? qRadiusToPoints(tth, smoothQ, lambdaForBg, 0) : null;
    const background = rollingBallBackground(intensity, ballRadii, smoothRadii);
    const backgroundCorrected = intensity.map((y, i) => Math.max(0, y - background[i]));
    const windowSize = Math.max(5, Math.min(11, Math.floor(n / 100)));
    const smoothed = savitzkyGolay(backgroundCorrected, windowSize, 2);

    // --- Range-restricted noise/threshold statistics ---
    // changed to search all points on 13th july 2026
    let rangeStart = 0;
    while (rangeStart < n && tth[rangeStart] < minTth) rangeStart++;
    let rangeEnd = n; // exclusive
    while (rangeEnd > rangeStart && tth[rangeEnd - 1] > maxTth) rangeEnd--;
    const rangeView = backgroundCorrected.slice(rangeStart, rangeEnd);

    const maxCorrectedIntensity = (rangeView.length > 0 ? maxOfArray(rangeView) : maxOfArray(backgroundCorrected)) || 1;
    const minAbsoluteHeight = (minHeightPercent / 100) * maxCorrectedIntensity;
    
    const calculateNoiseLevel = (data) => {
const n_s = data.length;
if (n_s < 10) return 0;
// MAD-based robust noise estimate. Two O(n) quickselects on scratch copies
// replace the two full O(n log n) sorts the original did; the median index
// (floor(len/2)) is unchanged, so the result is bit-identical.
const work = Float64Array.from(data);
const midIdx = Math.floor(n_s / 2);
const median = quickselect(work, midIdx);
for (let i = 0; i < n_s; i++) work[i] = Math.abs(work[i] - median);
const mad = quickselect(work, midIdx);
return mad * 1.4826;
};

    const noiseSrc = rangeView.length >= 10 ? rangeView : backgroundCorrected;
    const adaptiveThreshold = Math.max(minAbsoluteHeight, calculateNoiseLevel(noiseSrc) * 3);
    const localMaxIndices = [];
    // Restrict the local-max scan to the slider range so out-of-range
    // structure cannot produce candidates that get filtered out later
    // (the only effect of those is to perturb plateau detection at
    // the boundaries).
    const scanStart = Math.max(1, rangeStart);
    const scanEnd   = Math.min(n - 1, rangeEnd);
    for (let i = scanStart; i < scanEnd; i++) {
        const current = smoothed[i]; if (current < adaptiveThreshold) continue;
        if (current > smoothed[i - 1] && current > smoothed[i + 1]) localMaxIndices.push(i);
        else if (current === smoothed[i + 1] && current > smoothed[i - 1]) { let plateauEnd = i + 1; while (plateauEnd < n - 1 && Math.abs(smoothed[plateauEnd] - current) < maxCorrectedIntensity * 0.001) plateauEnd++; if (plateauEnd < n && smoothed[plateauEnd] < current) localMaxIndices.push(Math.round((i + plateauEnd - 1) / 2)); i = plateauEnd - 1; }
    }
    // --- Prominence filter -------------------------------------------
    //
    // Prominence is how far you must descend from a maximum before you can
    // reach any higher maximum: walk left until the profile rises above
    // this peak (or the scan ends), keep the lowest value seen; walk right
    // the same way; the prominence is the peak height minus the HIGHER of
    // those two valley floors. The higher one, because that is the
    // shallowest escape route to higher ground.
    //
    // This is the right discriminator for shoulders and ringing, where a
    // minimum-separation rule is not: separation cannot distinguish a
    // genuine close doublet from a bump on a flank, prominence can. On
    // synthetic profiles the ratio prominence/height separates cleanly --
    // ~1.00 for a fully resolved peak, ~0.50 for an unresolved shoulder,
    // ~0.23 for a noise spike riding on a flank -- and the ratio is
    // scale-free, so one threshold works for weak and strong peaks alike.
    //
    // Computed on `smoothed`, the same background-corrected array used for
    // detection above. On the raw profile, noise in the valley sets the
    // floor and the numbers stop meaning anything.
    const promFrac = ui.peakProminenceSlider
        ? (parseFloat(ui.peakProminenceSlider.value) || 0) / 100
        : 0;
    let survivingMaxima = localMaxIndices;
    if (promFrac > 0 && localMaxIndices.length > 1) {
        // Bound the walk: beyond a few hundred channels the search is only
        // finding the far side of the pattern, and an unbounded walk makes
        // this O(n) per peak on a scan with few maxima.
        const WALK_LIMIT = Math.max(200, Math.ceil(n / 20));
        const noiseFloor = calculateNoiseLevel(noiseSrc) * 3;
        survivingMaxima = localMaxIndices.filter(idx => {
            const h = smoothed[idx];
            if (!(h > 0)) return false;
            let left = h;
            for (let j = idx - 1, steps = 0; j >= scanStart && steps < WALK_LIMIT; j--, steps++) {
                if (smoothed[j] > h) break;
                if (smoothed[j] < left) left = smoothed[j];
            }
            let right = h;
            for (let j = idx + 1, steps = 0; j <= scanEnd && steps < WALK_LIMIT; j++, steps++) {
                if (smoothed[j] > h) break;
                if (smoothed[j] < right) right = smoothed[j];
            }
            const prominence = h - Math.max(left, right);
            // Two conditions: statistically real (above the noise), and
            // actually resolved (a real valley, not a change of slope).
            return prominence >= noiseFloor && prominence >= promFrac * h;
        });
        // Never let the filter empty the list -- an over-tight setting
        // should degrade the peak list, not silently destroy it and leave
        // the user with an unexplained "find at least 3 peaks" error.
        if (survivingMaxima.length < 3) survivingMaxima = localMaxIndices;
    }

    const candidates = survivingMaxima.filter(idx => tth[idx] >= minTth && tth[idx] <= maxTth && backgroundCorrected[idx] >= adaptiveThreshold)
        .map(idx => ({ idx, tth: tth[idx], height: smoothed[idx], backgroundCorrectedHeight: backgroundCorrected[idx] }));
    
    // ref 5 ou 3 savitzky
    const refinedPeaks = [];
    for (const peak of candidates) {
        const { idx } = peak; let refinedTth = peak.tth;

        // Calculate a robust average step size around the peak
        const avgStep = (idx > 0 && idx < n - 1) 
            ? (tth[idx+1] - tth[idx-1]) / 2.0 
            : (idx > 0 ? tth[idx] - tth[idx-1] : (idx < n-1 ? tth[idx+1] - tth[idx] : 0.01));

        // Try 5-point parabola first (more accurate)
        if (idx > 1 && idx < n - 2) { 
            const y1 = smoothed[idx - 2];
            const y2 = smoothed[idx - 1];
            const y3 = smoothed[idx];
            const y4 = smoothed[idx + 1];
            const y5 = smoothed[idx + 2];

            // 5-point least-squares quadratic fit (Savitzky-Golay coefficients)
            // Parabola y = ax^2 + bx + c, centered at x=0 (idx)
            // Least-squares quadratic through 5 equally spaced points, x = -2..2.
            // Sum(x^2) = 10, Sum(x^4) = 34, N = 5 give a = (Sum(x^2 y) - 2 Sum(y)) / 14.
            // The divisor was 7, i.e. a came out twice too large, which halved every
            // vertex offset delta = -b / (2a). Sub-step peak positions were therefore
            // systematically pulled only half way to the true maximum.
            const a = (2*y1 - y2 - 2*y3 - y4 + 2*y5) / 14.0;
            const b = (-2*y1 - y2 + 0*y3 + y4 + 2*y5) / 10.0;
            
            // Check for valid maximum (downward parabola, a < 0)
            if (a < -1e-10) { 
                const delta = -b / (2 * a); // Vertex x = -b / (2a)
                
                // delta should be within the 5-point window
                if (Math.abs(delta) < 2.0) { 
                    refinedTth = tth[idx] + delta * avgStep;
                }
            }

            
        //  Use 3-point fit if 5-point fails or is near edge
        } else if (idx > 0 && idx < n - 1) { 
            const y1 = smoothed[idx - 1], y2 = smoothed[idx], y3 = smoothed[idx + 1]; 
            const denominator = (y1 - 2 * y2 + y3); 
            
if (denominator < -1e-10) { 
                // Three-point parabolic interpolation: delta = (y1 - y3) / (2 (y1 - 2 y2 + y3)).
                // The factor 1/2 was missing, so this branch overshot by exactly 2x -
                // the opposite sign of error to the 5-point branch above, which is why
                // edge peaks and interior peaks disagreed about where a maximum was.
                const delta = 0.5 * (y1 - y3) / denominator; 
                if (Math.abs(delta) < 1.0) { 
                    refinedTth = tth[idx] + delta * avgStep; 
                } 
            } 
        }
        
        if (refinedTth <= 1e-4) {
            refinedTth = 1e-4; // Prevent tth=0 or negative tth
        }


        refinedPeaks.push({ ...peak, tth: refinedTth });
    }
    

    // Continue 
    refinedPeaks.sort((a, b) => a.tth - b.tth);
    const finalPeaks = [];

    // Angle-adaptive merge threshold.
    // Two refined candidates closer than this are treated as one peak
    // (keeping the higher one).
    //
    // Goals:
    //   - Always at least 0.02° (instrumental resolution at low 2θ)
    //   - Grow gently with 2θ to swallow small numerical duplicates from
    //     the parabolic refinement at high angle
    //   - When the data is an unstripped Kα-doublet pattern, never let
    //     the threshold reach the Kα1/Kα2 split — otherwise a real
    //     Kα2 ghost would be merged into its Kα1 parent and we'd lose
    //     the soft-violation signal in space-group analysis.
    const lambda = getWavelength();
    const stripOn = !!ui.stripKa2Checkbox?.checked;
    const presetForMerge = stripOn ? null : getActiveKa2Preset();
    const FLOOR = 0.02; // never tighter than this
    const CEIL  = 0.10; // never looser than this
    const mergeThresholdAt = (tthDeg) => {
        const baseScale = 0.02 + 0.001 * tthDeg; // gentle linear ramp
        let mt = Math.min(CEIL, Math.max(FLOOR, baseScale));
        if (presetForMerge) {
            const tth2 = expectedKa2TthDeg(tthDeg, presetForMerge);
            if (isFinite(tth2)) {
                const split = Math.abs(tth2 - tthDeg);
                // Cap by 45% of the doublet split so a Kα1+Kα2 pair
                // is preserved as TWO peaks. But never drop below the
                // floor — at low 2θ where the doublet is unresolved
                // the cap would go below instrumental resolution.
                mt = Math.max(FLOOR, Math.min(mt, 0.45 * split));
            }
        }
        return mt;
    };

    for (const peak of refinedPeaks) {
        if (finalPeaks.length === 0) { finalPeaks.push(peak); continue; }
        const last = finalPeaks[finalPeaks.length - 1];
        const mt = mergeThresholdAt(last.tth);
        if (Math.abs(peak.tth - last.tth) >= mt) {
            finalPeaks.push(peak);
        } else if (peak.height > last.height) {
            finalPeaks[finalPeaks.length - 1] = peak;
        }
    }

    // Build initial pickedPeaks with the user's (default) wavelength.
    // recalculatePeakValues() below will flag Ka2-suspects AND re-derive
    // d/q for parents of tagged ghosts using λ_Ka1 (since we know those
    // peaks are physically Ka1 lines).
   // Include height so the space group analyzer can filter noise
pickedPeaks = finalPeaks.map(p => ({ tth: p.tth, d: 0, q: 0, height: p.height }));
    // The list is now purely auto-detected again.
    peaksManuallyEdited = false;

    recalculatePeakValues();

    updatePeakTable(); updateStartIndexingButtonState();
    _perfEnd(`(${pickedPeaks.length} peaks, n=${n})`);
}
const recalculatePeakValues = () => {
    const lambda = getWavelength();
    // First flag the Ka2-children (i.e., who is parent of a tagged ghost),
    // because the d/q computation for parents needs to know that.
    flagKa2SuspectPeaks();
    const preset = getActiveKa2Preset();
    const lambdaKa1 = preset ? preset.ka1 : null;
    pickedPeaks.forEach(peak => {
        // Parents of tagged Ka2 ghosts are provably Ka1 lines, so their
        // d-spacing should be derived from λ_Ka1, not the user's main
        // (Ka-avg) wavelength. All other peaks use the main λ.
        const lam = (peak.hasKa2Child && lambdaKa1) ? lambdaKa1 : lambda;
        peak.d = lam / (2 * Math.sin(peak.tth * Math.PI / 360));
        peak.q = 1 / (peak.d * peak.d);
        peak.lambdaUsed = lam; // for debugging / display traceability
    });
};
