// js/crystallography/absences.js
// Systematic-absence analysis and line assignment.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (index.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// `tthMin` is optional: when omitted the measured window is inferred from the
// observed peaks, which is conservative (it can only shrink the range in which
// the extinction test counts verified absences).
function analyzeSystematicAbsences(solution, obs_peaks, spaceGroupData, wavelength, tthError, tthMax, impurity_peaks, tthMin) {
    const MAX_VIOLATIONS = 2;
    const fallbackResult = {
        centering: 'Unknown',
        rankedSpaceGroups: [],
        detectedExtinctions: [],
        ambiguousHkls: new Set(),
        hklList: []
    };
    if (!spaceGroupData?.space_groups) { console.warn("Space group data not loaded"); return fallbackResult; }
    if (!sgEnsureDatabase(spaceGroupData)) {
        console.warn("Space group database has no operator table (rebuild with build_sg_db.py)");
        return fallbackResult;
    }
    // The FULL lattice, never the R-filtered one: the analysis detects R
    // centring precisely by finding the R-forbidden lines empty (see
    // latticeLineFilter in hkl.js). The R list is only what is displayed.
    const all_calc_hkls = generateHKL_for_analysis(withoutLattice(solution), wavelength, tthMax);

    // Ka1 lines (p.lambda) are matched where the cell fit put them: at their
    // 2-theta for this wavelength (peaksAtWavelength, js/crystallography/fit.js).
    // Matched at the measured value instead, a Ka1 line at high angle sat up
    // to ~0.2 deg (Cu) from its own calculated line once the cell was right.
    obs_peaks = peaksAtWavelength(obs_peaks, wavelength);

    if (all_calc_hkls.length === 0) {
        fallbackResult.hklList = [];
        return fallbackResult;
    }

    // Carry the per-peak Ka2-suspect flag through the indexing step. Each
    // observed peak that matches a calculated hkl produces an indexed_hkl
    // record; we tag the record with the parent peak's ka2Suspect flag so the
    // downstream centering / extinction / ranking code can distinguish
    // "hard" violations (driven by genuine reflections) from "soft" ones
    // (driven by suspected Ka2 ghost peaks).
    const indexed_hkls = []; const zero_correction = solution.zero_correction || 0;
    const indexWindow   = tthError * 1.5;
    const overlapWindow = tthError * 1.5;

    // --- Intensity-aware demotion ---
    // A "forbidden" reflection in the true space group should have
    // intrinsic intensity ZERO. If the observed peak that drives a
    // violation is weak compared with the strong peaks of the pattern,
    // it is much more likely to be a tail / wing / weak overlap / noise
    // artefact than a genuine reflection contradicting the space group.
    // The baseline must be LOCAL in 2theta, not a fraction of the global
    // maximum.
    const LOW_INTENSITY_FRACTION = 0.05;
    const LOCAL_WINDOW_DEG = 15.0;

    const heightedPeaks = obs_peaks.filter(p => typeof p.height === 'number' && isFinite(p.height));
    let globalMaxHeight = 0;
    for (const p of heightedPeaks) if (p.height > globalMaxHeight) globalMaxHeight = p.height;

    const lowIntensityThresholdAt = (tth) => {
        if (globalMaxHeight <= 0) return -Infinity; // no heights -> disable demotion
        let localMax = 0;
        for (const p of heightedPeaks) {
            if (Math.abs(p.tth - tth) <= LOCAL_WINDOW_DEG && p.height > localMax) localMax = p.height;
        }
        // Fall back to the global scale only if the local window is empty.
        const scale = localMax > 0 ? localMax : globalMaxHeight;
        return scale * LOW_INTENSITY_FRACTION;
    };

    obs_peaks.forEach(peak => {
        const corrected_tth = peak.tth - zero_correction;
        const bestMatch = all_calc_hkls.reduce((best, hkl) => { const diff = Math.abs(hkl.tth - corrected_tth); return diff < best.minDiff ? { hkl, minDiff: diff } : best; }, { hkl: null, minDiff: Infinity });
        if (bestMatch.hkl && bestMatch.minDiff < indexWindow) {
            const altHkls = all_calc_hkls
                .filter(hkl => Math.abs(hkl.tth - bestMatch.hkl.tth) < overlapWindow)
                .map(hkl => ({ h: hkl.h, k: hkl.k, l: hkl.l, tth: hkl.tth, coincident: hkl.coincident || [] }));
            const peakHeight = (typeof peak.height === 'number' && isFinite(peak.height)) ? peak.height : null;
            const isLowIntensity = (peakHeight !== null) && (peakHeight < lowIntensityThresholdAt(peak.tth));
            indexed_hkls.push({
                h: bestMatch.hkl.h, k: bestMatch.hkl.k, l: bestMatch.hkl.l,
                tth: peak.tth, calc_tth: bestMatch.hkl.tth,
                tth_obs: peak.tthObs ?? peak.tth,
                coincident: bestMatch.hkl.coincident || [],
                ka2Suspect: !!peak.ka2Suspect,
                altHkls: altHkls,
                tol: tthError,
                height: peakHeight,
                lowIntensity: isLowIntensity
            });
        }
    });

    // Dedup by (h,k,l). When two observed peaks index to the same hkl, prefer
    // the NON-suspect, HIGH-intensity one.
    const uniqMap = new Map();
    const recordPriority = (r) => {
        let p = 0;
        if (!r.ka2Suspect) p += 2;
        if (!r.lowIntensity) p += 1;
        return p;
    };
    for (const r of indexed_hkls) {
        const key = `${r.h},${r.k},${r.l}`;
        const existing = uniqMap.get(key);
        if (!existing) { uniqMap.set(key, r); continue; }
        if (recordPriority(r) > recordPriority(existing)) uniqMap.set(key, r);
    }
    const unique_indexed_hkls = Array.from(uniqMap.values());

    const unambiguous_hkls = unique_indexed_hkls.filter(refl => {
        const nearbyCount = all_calc_hkls.filter(calc => { if (calc.h === refl.h && calc.k === refl.k && calc.l === refl.l) return false; return Math.abs(calc.tth - refl.calc_tth) < tthError; }).length;
        return nearbyCount === 0;
    });

    const hkls_for_analysis = unique_indexed_hkls;
    if (hkls_for_analysis.length < 5) { fallbackResult.centering = 'Unknown (too few unambiguous peaks in range)'; return fallbackResult; }
    const unambiguousSet = new Set(unambiguous_hkls.map(r => `${r.h},${r.k},${r.l}`));
    const ambiguousHkls = new Set(unique_indexed_hkls.filter(r => !unambiguousSet.has(`${r.h},${r.k},${r.l}`)).map(r => `${r.h},${r.k},${r.l}`));

    const anyKa2Suspects = hkls_for_analysis.some(r => r.ka2Suspect);

    const centeringResult = determineCentering(hkls_for_analysis, solution.system);
    const obsTthCorrected = obs_peaks
        .filter(p => p && !p.ka2Suspect && Number.isFinite(p.tth))
        .map(p => p.tth - zero_correction);
    const measuredLo = Number.isFinite(tthMin)
        ? tthMin - zero_correction
        : (obsTthCorrected.length ? Math.min(...obsTthCorrected) : -Infinity);
    const measuredHi = Number.isFinite(tthMax) ? tthMax - zero_correction : Infinity;
    const detectedExtinctions = detectExtinctions(
        hkls_for_analysis,
        solution.system,
        spaceGroupData,
        centeringResult.plausibleCenterings,
        {
            calcLines: all_calc_hkls,
            obsTth: obsTthCorrected,
            indexWindow,
            overlapWindow,
            tthMin: measuredLo,
            tthMax: measuredHi
        }
    );
    const rankedSpaceGroups = rankSpaceGroups(hkls_for_analysis, solution.system, centeringResult.plausibleCenterings, spaceGroupData, MAX_VIOLATIONS, detectedExtinctions);

    const compatibleSorted = rankedSpaceGroups.slice().sort((a, b) =>
        ((a.hardViolations || 0) - (b.hardViolations || 0)) ||
        ((b.number || 0) - (a.number || 0)) ||
        String(a.symbol || '').localeCompare(String(b.symbol || '')));
    const SG_LIST_CAP = 40, SG_LIST_CEILING = 150;
    const nClean = compatibleSorted.filter(s => !(s.hardViolations > 0)).length;
    const compatibleSettings = compatibleSorted.slice(0, Math.min(SG_LIST_CEILING, Math.max(SG_LIST_CAP, nClean)));

    return {
        centering: centeringResult.description,
        compatibleSettings: compatibleSettings,
        compatibleSettingsTotal: compatibleSorted.length,
        rankedSpaceGroups: compatibleSettings,
        detectedExtinctions: detectedExtinctions,
        centeringViolations: centeringResult.violations,
        centeringViolationsHard: centeringResult.violationsHard,
        centeringViolationsSoft: centeringResult.violationsSoft,
        centeringViolationDetails: centeringResult.violationDetails,
        ambiguousHkls: ambiguousHkls,
        hklList: latticeLineFilter(solution) ? generateHKL_for_analysis(solution, wavelength, tthMax) : all_calc_hkls,
        usedKa2SoftScoring: anyKa2Suspects
    };
}
// How much worse than the assigned reflection an allowed alternative may fit
// and still count as a genuine competitor.
const AMBIGUITY_MARGIN = 2.0;
