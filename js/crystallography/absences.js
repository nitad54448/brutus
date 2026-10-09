// js/crystallography/absences.js
// Systematic-absence analysis, line assignment and manual-hkl refinement.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// --- groups; utilise cctbx
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
        hklList:[]
    };
    if (!spaceGroupData?.space_groups) { console.warn("Space group data not loaded"); return fallbackResult; }
    if (!sgEnsureDatabase(spaceGroupData)) {
        console.warn("Space group database has no operator table (rebuild with sg_pack.py)");
        return fallbackResult;
    }
    // The FULL lattice, never the R-filtered one: the analysis detects R
    // centring precisely by finding the R-forbidden lines empty (see
    // latticeLineFilter in hkl.js). The R list is only what is displayed.
    const all_calc_hkls = generateHKL_for_analysis(withoutLattice(solution), wavelength, tthMax);

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
    // Two windows around each observed peak:
    //   indexWindow  - the indexing tolerance proper (1.5 * tthError).
    //                  A peak is indexed only if its closest calculated
    //                  hkl is inside this window.
    //   overlapWindow - same width as indexWindow. Used to detect
    //                   *overlapping* reflections within one peak. The
    //                   IUCr International Tables note that for powder
    //                   data, peak overlap is the dominant source of
    //                   ambiguity in systematic-absence detection: a
    //                   "forbidden" reflection that overlaps an allowed
    //                   one cannot be used as hard evidence against a
    //                   space group, because its observed intensity is
    //                   contaminated. The principled solution is the
    //                   Bayesian intensity-based test of Markvardsen et
    //                   al. (Acta Cryst. A57, 47, 2001; ExtSym/DASH),
    //                   but in the absence of integrated intensities the
    //                   overlap check is a useful proxy: if a violating
    //                   hkl overlaps an allowed neighbour within one
    //                   tolerance bar, the violation is downgraded from
    //                   hard to soft. Using the SAME window as indexing
    //                   keeps the rule conservative and well defined.
    const indexWindow   = tthError * 1.5;
    const overlapWindow = tthError * 1.5;

    // --- Intensity-aware demotion ---
    // A "forbidden" reflection in the true space group should have
    // intrinsic intensity ZERO. If the observed peak that drives a
    // violation is weak compared with the strong peaks of the pattern,
    // it is much more likely to be a tail / wing / weak overlap / noise
    // artefact than a genuine reflection contradicting the space group.
    // We therefore tag each indexed peak as "lowIntensity" when its
    // height is below 10% of the strongest observed peak in the pattern;
    // such peaks count as soft (not hard) evidence in countViolations
    // and determineCentering. This is the same intuition that drives
    // the Bayesian ExtSym algorithm (Markvardsen 2001), without the
    // full Wilson-distribution machinery that requires properly
    // background-subtracted, Lorentz-polarisation-corrected integrated
    // intensities. Heights are missing for older callers that did not
    // pass them through; in that case the threshold is never triggered
    // and behaviour is identical to the position-only check.
    // The baseline must be LOCAL in 2theta, not a fraction of the global
    // maximum. Diffracted intensity falls off with angle (Lorentz-polarisation,
    // Debye-Waller, absorption), so a global 5%-of-max cut does not mean "weak"
    // — above roughly 90 deg it means "high angle", and it silently strips the
    // whole back-reflection region of any power to falsify an absence rule.
    // Comparing each peak with peaks at COMPARABLE 2theta keeps the original
    // crystallographic intent (a truly forbidden reflection has zero intensity)
    // without the angular bias.
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

    // Assignments the user set explicitly via "Swap hkl". These override the
    // nearest-line rule wherever they apply - otherwise the analysis of a
    // swapped solution would quietly revert to the indexing the user rejected.
    const manualByTth = new Map();
    for (const sw of (solution.manualSwaps || [])) {
        if (sw && Number.isFinite(sw.h) && Number.isFinite(sw.k) && Number.isFinite(sw.l)) {
            manualByTth.set(Number(sw.tth).toFixed(4), sw);
        }
    }

    obs_peaks.forEach(peak => {
        const corrected_tth = peak.tth - zero_correction;
        let bestMatch = all_calc_hkls.reduce((best, hkl) => { const diff = Math.abs(hkl.tth - corrected_tth); return diff < best.minDiff ? { hkl, minDiff: diff } : best; }, { hkl: null, minDiff: Infinity });
        const man = manualByTth.get(Number(peak.tth).toFixed(4));
        if (man) {
            const forced = all_calc_hkls.find(x => x.h === man.h && x.k === man.k && x.l === man.l);
            // Honour it even if that line is not the nearest; only skip when the
            // reflection does not exist for this lattice at all.
            if (forced) bestMatch = { hkl: forced, minDiff: 0 };
        }
        if (bestMatch.hkl && bestMatch.minDiff < indexWindow) {
            // Collect every calculated hkl whose 2theta is within
            // overlapWindow of the BEST-MATCH calculated 2theta (not of
            // the observed 2theta). The overlap is between the candidate
            // reflection and its neighbours in reciprocal space — that
            // is what determines whether intensity from one can leak
            // into the other.
            const altHkls = all_calc_hkls
                .filter(hkl => Math.abs(hkl.tth - bestMatch.hkl.tth) < overlapWindow)
                .map(hkl => ({ h: hkl.h, k: hkl.k, l: hkl.l, tth: hkl.tth, coincident: hkl.coincident || [] }));
            const peakHeight = (typeof peak.height === 'number' && isFinite(peak.height)) ? peak.height : null;
            const isLowIntensity = (peakHeight !== null) && (peakHeight < lowIntensityThresholdAt(peak.tth));
            indexed_hkls.push({
                h: bestMatch.hkl.h, k: bestMatch.hkl.k, l: bestMatch.hkl.l,
                tth: peak.tth, calc_tth: bestMatch.hkl.tth,
                // Other reflections on exactly this line (see the generator):
                // a group forbids the LINE only if it forbids all of them.
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
    // the NON-suspect, HIGH-intensity one (strong, real evidence outweighs
    // weak or Ka2-ghost evidence for the same reflection). This avoids
    // accidentally flipping a real reflection to "soft" just because a Ka2
    // ghost or a weak-tail peak from a different parent happened to index
    // to the same hkl by coincidence.
    const uniqMap = new Map();
    const recordPriority = (r) => {
        // Higher = preferred. Non-suspect > suspect; high-intensity > low.
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
        // Otherwise keep first (existing).
    }
    const unique_indexed_hkls = Array.from(uniqMap.values());

    const unambiguous_hkls = unique_indexed_hkls.filter(refl => {
        const nearbyCount = all_calc_hkls.filter(calc => { if (calc.h === refl.h && calc.k === refl.k && calc.l === refl.l) return false; return Math.abs(calc.tth - refl.calc_tth) < tthError; }).length;
        return nearbyCount === 0;
    });

    // Do NOT restrict the analysis to isolated reflections. Overlap is handled
    // per-reflection downstream (the altHkls demotion in countViolations /
    // determineCentering), so pre-filtering here is both redundant and harmful:
    // in a pseudo-symmetric cell nearly every reflection overlaps a neighbour,
    // so this filter used to discard ~3/4 of the indexed peaks and leave a
    // biased remnant. The set it kept was precisely the set least able to
    // falsify a centering. Keep every unique indexed reflection and let the
    // hard/soft accounting weigh them.
    const hkls_for_analysis = unique_indexed_hkls;
    if (hkls_for_analysis.length < 5) { fallbackResult.centering = 'Unknown (too few unambiguous peaks in range)'; return fallbackResult; }
    const unambiguousSet = new Set(unambiguous_hkls.map(r => `${r.h},${r.k},${r.l}`));
    const ambiguousHkls = new Set(unique_indexed_hkls.filter(r => !unambiguousSet.has(`${r.h},${r.k},${r.l}`)).map(r => `${r.h},${r.k},${r.l}`));

    const anyKa2Suspects = hkls_for_analysis.some(r => r.ka2Suspect);

    const centeringResult = determineCentering(hkls_for_analysis, solution.system);
    // Evidence bundle for the extinction test: the extinction-blind line list,
    // the measured window, and the observed peak positions expressed in the
    // SAME zero-corrected frame the calculated lines use. Ka2 ghosts are left
    // out - an artefact must not be allowed to refute an absence rule.
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
    // NOTHING is re-assigned here. The analysis reports what it finds and leaves
    // the indexing alone: silently rewriting an hkl behind the user's back is
    // exactly the behaviour this was changed to avoid. Correcting an assignment
    // is a deliberate act, done through the "Swap hkl" command, which produces a
    // separate solution the user can compare against this one.
    const rankedSpaceGroups = rankSpaceGroups(hkls_for_analysis, solution.system, centeringResult.plausibleCenterings, spaceGroupData, MAX_VIOLATIONS, detectedExtinctions);

    // --- DEMOTED FROM A RANKING TO A COMPATIBILITY LIST ---
    //
    // rankSpaceGroups() orders by matchScore, which counts reflections that ARE
    // PRESENT. As its own comment concedes, that cannot distinguish a rule set
    // from a strictly less-constrained one: confirmations are nearly free, and
    // systematic ABSENCES are the evidence in space-group determination. The
    // extinction bonus was added to patch the asymmetry, but it is an
    // unnormalised heuristic with no scale, so matchScore differences are not
    // comparable across patterns and cannot be turned into odds.
    //
    // Worse, this whole list is judged against ONE cell that was refined
    // without knowing about any extinction rule -- so it was pulled toward the
    // forbidden reflections it is now being asked to rule on -- and the
    // candidate pool is pre-filtered by centeringResult.plausibleCenterings, so
    // a wrong centering verdict removes the correct setting from the list
    // entirely rather than merely demoting it.
    //
    // The ranking authority is sgScoreClass()/sgRankRows() (the Space Group MC):
    // it refits the cell per hypothesis, merges settings the data cannot
    // separate, and scores a real likelihood ratio in nats. What survives here
    // is the part that is still sound -- WHICH settings the observed absences
    // contradict, and by how many reflections -- presented in a neutral order
    // (fewest hard violations first, then space-group number) so no ordering
    // claim is made beyond that.
    //
    // The cap also had to move off matchScore: slicing the top 20 of a heuristic
    // order is a heuristic selection, so a setting could vanish from the report
    // for scoring reasons while being presented as merely "compatible".
    const compatibleSorted = rankedSpaceGroups.slice().sort((a, b) =>
        ((a.hardViolations || 0) - (b.hardViolations || 0)) ||
        ((b.number || 0) - (a.number || 0)) ||
        String(a.symbol || '').localeCompare(String(b.symbol || '')));
    // The cap must never cut a setting with no hard violation: those are the
    // compatible ones, and with the list sorted by DESCENDING space-group
    // number a fixed cap of 40 always dropped the low-numbered ones -- F222,
    // Fmm2, Ccc2, Aba2, I4, I41 ... were absent from the list on patterns
    // that obey them exactly. Every zero-violation setting is kept (up to a
    // generous ceiling against a pathological, evidence-free pattern); the
    // settings WITH violations fill the rest up to the old 40.
    const SG_LIST_CAP = 40, SG_LIST_CEILING = 150;
    const nClean = compatibleSorted.filter(s => !(s.hardViolations > 0)).length;
    const compatibleSettings = compatibleSorted.slice(0, Math.min(SG_LIST_CEILING, Math.max(SG_LIST_CAP, nClean)));

    return {
        centering: centeringResult.description,
        compatibleSettings: compatibleSettings,
        compatibleSettingsTotal: compatibleSorted.length,
        // Deprecated alias, same array. Nothing should rank by this order.
        rankedSpaceGroups: compatibleSettings,
        detectedExtinctions: detectedExtinctions,
        centeringViolations: centeringResult.violations,
        centeringViolationsHard: centeringResult.violationsHard,
        centeringViolationsSoft: centeringResult.violationsSoft,
        centeringViolationDetails: centeringResult.violationDetails,
        ambiguousHkls: ambiguousHkls,
        // Displayed list (report hkl table): an R cell shows its R lines.
        hklList: latticeLineFilter(solution) ? generateHKL_for_analysis(solution, wavelength, tthMax) : all_calc_hkls,
        usedKa2SoftScoring: anyKa2Suspects
    };
}
// How much worse than the assigned reflection an allowed alternative may fit
// and still count as a genuine competitor.
const AMBIGUITY_MARGIN = 2.0;
// --- EXTINCTION-AWARE CELL RE-REFINEMENT ---
// Re-assigning hkl labels alone is cosmetic, and worse, it makes the report
// internally inconsistent: the cell was least-squares fitted against the OLD
// pairing, so after relabelling, the diff column measures the new hkl against a
// cell refined to the old one. For PbSO4 the 16.423 deg peak reads 0.023 off as
// (1,0,1), against 0.019 as the forbidden (0,1,0) - the corrected assignment
// looks worse purely because the cell was pulled toward 010 while fitting.
//
// The fix is to redo the fit with an extinction-filtered line list. Restricting
// the candidate reflections to those the detected rules allow makes the pairing,
// the cell, the zero error, the ESDs and the figures of merit all consistent
// with the table in one step, because every one of them derives from that list.
//
// M20 improves for two independent reasons: <|dQ|> falls because the pairing is
// correct, and N20 falls because extinct lines are no longer counted as
// possible. This is reported alongside the original rather than replacing it,
// so the effect of the constraint stays visible and auditable.
// --- MANUAL HKL SWAP ---
// Indexing assigns each peak to its nearest calculated line, and that choice can
// be wrong without producing any violation at all: in a permissive space group
// (P222 and friends) both candidates are allowed, so nothing flags the swap. A
// rule-driven search cannot find those cases by construction - it only ever sees
// assignments that some space group forbids - so the decision is handed to the
// user instead.
//
// getPeakAssignments() reports what the indexer currently believes, and
// refineWithManualHkl() re-fits the cell with chosen assignments overridden.
// Figures of merit are computed the ordinary way against the FULL line list, so
// the resulting solution is directly comparable with every other in the table.

// Nearest-line assignment with a one-peak-per-line constraint, resolved
// best-first. Shared by getPeakAssignments() and refineWithManualHkl() so the
// table the user is shown is exactly the assignment the refit will use -- they
// used to run separate copies of this logic and could disagree.
//
// `lines` must be sorted ascending by tth (generateHKL_for_analysis guarantees
// it), so the nearest line is a binary search rather than a linear scan.
// Returns an array, one entry per peak: { line, d } or null.
function assignNearestLines(peaks, lines, zero, window) {
    const n = peaks.length;
    const out = new Array(n).fill(null);
    if (!lines.length || !n) return out;

    const lineTth = new Float64Array(lines.length);
    for (let i = 0; i < lines.length; i++) lineTth[i] = lines[i].tth;

    const cand = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
        const tc = peaks[i].tth - zero;
        const j = binarySearchClosest(lineTth, tc);
        if (j < 0 || j >= lines.length) continue;
        const d = Math.abs(lineTth[j] - tc);
        if (d <= window) cand[i] = { j, d };
    }

    // A calculated line may only be claimed by ONE observed peak -- the rule
    // pair_and_fit() has always enforced. Closest peak wins; the loser is
    // reported unindexed rather than fitted to the same reflection.
    const claimed = new Set();
    for (const { i } of cand.map((c, i) => ({ i, d: c ? c.d : Infinity }))
                            .sort((x, y) => x.d - y.d)) {
        const c = cand[i];
        if (!c || claimed.has(c.j)) continue;
        claimed.add(c.j);
        out[i] = { line: lines[c.j], d: c.d };
    }
    return out;
}
// What is each observed peak currently indexed as?
function getPeakAssignments(solution, obs_peaks, wavelength, tthError, tthMax, limit) {
    const lines = generateHKL_for_analysis(solution, wavelength, tthMax);
    if (!lines.length) return [];
    const zero = solution.zero_correction || 0;
    const window = tthError * 1.5;
    const peaks = (obs_peaks || [])
        .filter(p => typeof p.tth === 'number' && isFinite(p.tth))
        .slice().sort((x, y) => x.tth - y.tth);

    const assigned = assignNearestLines(peaks, lines, zero, window);
    const out = [];
    for (let i = 0; i < peaks.length; i++) {
        const p = peaks[i];
        const tc = p.tth - zero;
        const a = assigned[i];
        const best = a ? a.line : null;
        const dObs = wavelength / (2 * Math.sin(tc * RAD / 2));
        out.push({
            tth: p.tth, tth_corr: tc,
            h: best ? best.h : null, k: best ? best.k : null, l: best ? best.l : null,
            calc_tth: best ? best.tth : null,
            diff: best ? (tc - best.tth) : null,
            d_obs: isFinite(dObs) ? dObs : null,
            d_calc: best ? best.d : null,
            indexed: !!best
        });
        if (limit && out.length >= limit) break;
    }
    return out;
}
// Re-fit with user-supplied assignments. `overrides` is a list of
// { tth, h, k, l }; every other peak keeps its nearest-line assignment.
// Returns { cell, swaps } on success or { error } with a reason, so the caller
// can tell the user exactly why nothing happened.
function refineWithManualHkl(solution, obs_peaks, overrides, wavelength, tthError, tthMax, refineZero, impurity_peaks, isAuto = false) {
const system = solution.system;
    if (!system) return { error: 'solution has no crystal system' };
    const lines = generateHKL_for_analysis(solution, wavelength, tthMax);
    if (!lines.length) return { error: 'no calculated reflections for this cell' };

    const zero = solution.zero_correction || 0;
    const window = tthError * 1.5;
    const ovr = new Map();
    for (const o of (overrides || [])) {
        if (o == null) continue;
        const h = Math.round(Number(o.h)), k = Math.round(Number(o.k)), l = Math.round(Number(o.l));
        if (![h, k, l].every(Number.isFinite)) continue;
        if (h === 0 && k === 0 && l === 0) return { error: '(0,0,0) is not a reflection' };
        const k4 = Number(o.tth).toFixed(4);
        // Overrides are matched to peaks by 2-theta printed to 4 dp. Two
        // overrides on the same key used to silently overwrite each other.
        if (ovr.has(k4)) return { error: `two overrides target the same peak position (${k4} deg)` };
        ovr.set(k4, { h, k, l });
    }
    if (ovr.size === 0) return { error: 'no changes to apply' };

    const toQ = (t) => (4 * Math.sin(t * RAD / 2) ** 2) / (wavelength ** 2);
    const peaks = (obs_peaks || []).filter(p => typeof p.tth === 'number' && isFinite(p.tth))
                                   .slice().sort((x, y) => x.tth - y.tth);

    // Every override must name exactly one peak. An override whose 2-theta
    // matches no peak (the caller changed the range between opening the dialog
    // and applying it) used to be dropped in silence; one that matches two
    // peaks would have been applied to both.
    const keyCount = new Map();
    for (const p of peaks) {
        const k4 = p.tth.toFixed(4);
        keyCount.set(k4, (keyCount.get(k4) || 0) + 1);
    }
    for (const k4 of ovr.keys()) {
        const c = keyCount.get(k4) || 0;
        if (c === 0) return { error: `no peak at ${k4} deg in the current range` };
        if (c > 1) return { error: `two peaks share the position ${k4} deg; cannot target one unambiguously` };
    }

    // Shared with getPeakAssignments: binary search (this used to be an
    // O(peaks x lines) linear scan per peak, on every call) plus the
    // one-peak-per-line rule pair_and_fit() enforces. Manual overrides below
    // still win unconditionally.
    const assigned = assignNearestLines(peaks, lines, zero, window);
    const autoHkl = assigned.map(a => a ? { h: a.line.h, k: a.line.k, l: a.line.l } : null);

    const rows = [], qv = [], tthRads = [], swaps = [];
    for (let i = 0; i < peaks.length; i++) {
        const p = peaks[i];
        const key = p.tth.toFixed(4);
        const tc = p.tth - zero;
        let hkl = autoHkl[i];
        if (ovr.has(key)) {
            const man = ovr.get(key);
            // A manual assignment is honoured even when it is not the nearest
            // line, and even when it falls outside the indexing window. That is
            // the entire point: the user is overruling the nearest-line rule.
            swaps.push({
                tth: p.tth,
                h: man.h, k: man.k, l: man.l,      // numeric, so downstream
                from: hkl ? `(${hkl.h},${hkl.k},${hkl.l})` : '(unindexed)',
                to: `(${man.h},${man.k},${man.l})` // consumers need not re-parse
            });
            hkl = man;
        }
        if (!hkl) continue;                      // unindexed and untouched: skip
        const row = getLSDesignRow([hkl.h, hkl.k, hkl.l], system);
        if (!row) continue;
        if (refineZero) row.push((2 / (wavelength ** 2)) * Math.sin(p.tth * RAD));
        rows.push(row);
        // With a zero column in the design matrix the RAW q is the correct
        // right-hand side and the fit recovers the zero itself. With NO zero
        // column the parent's zero has to be applied here instead -- the old
        // code used raw q unconditionally, so every fixed-zero refit was
        // offset by the whole zero shift while its ASSIGNMENTS used tc.
        qv.push(refineZero ? toQ(p.tth) : toQ(tc));
        tthRads.push(p.tth * RAD);
    }
    const minIndexed = { cubic: 4, tetragonal: 5, hexagonal: 5, orthorhombic: 6, monoclinic: 7, triclinic: 7 };
    const need = (minIndexed[system] || 6) + (refineZero ? 1 : 0);
    if (rows.length < need) return { error: `only ${rows.length} indexed peaks; ${need} needed for ${system}` };

    const fit = solveLeastSquares(rows, qv, ls_weights_for_2theta(tthRads));
    if (!fit || !fit.solution) return { error: 'least-squares fit failed (singular design matrix?)' };
    const cell = extractCellFromFit(fit.solution, system);
    if (!cell) return { error: 'fit did not yield a valid cell' };
    cell.system = system;
    if (latticeLineFilter(solution)) cell.lattice = solution.lattice;   // an R parent gives an R child
    if (refineZero) cell.zero_correction = fit.solution[fit.solution.length - 1] * DEG;
    else if (zero) cell.zero_correction = zero;   // carry the parent's fixed zero
    cell.volume = getVolume(cell);
    if (!isFinite(cell.volume) || cell.volume <= 0) return { error: 'refined cell has non-physical volume' };
    try { cell.errors = propagateErrors(system, fit, cell); } catch (e) { cell.errors = null; }

    // Figures of merit exactly as for any other solution: full line list, so the
    // number is directly comparable with the parent and with independent hits.
    try {
        // r.q is the inv_d_sq the generator already computed; toQ(r.tth) just
        // round-tripped it through asin and back.
        const refLines = generateHKL_for_analysis(cell, wavelength, tthMax);
        const qSorted = new Float64Array(Array.from(new Set(refLines.map(r => r.q)))).sort((a, b) => a - b);
        const mk = (n) => peaks.slice(0, n).map((p, i) => {
            const tc2 = p.tth - (cell.zero_correction || 0);
            return { ...p, original_index: i, q: toQ(tc2), tth: tc2 };
        });
        const tolFor = (arr) => (i) => {
            const th = (arr[i] ? arr[i].tth : 0) * RAD / 2;
            return ((8 * Math.sin(th) * Math.cos(th)) / (wavelength ** 2)) * (tthError * Math.PI / 360) + 1e-9;
        };
        const p20 = mk(Math.min(20, peaks.length));
        const f20 = calculateFiguresOfMerit(qSorted, p20, impurity_peaks || 0, tolFor(p20), wavelength);
        cell.m20 = f20.m20; cell.fN_20 = f20.fN; cell.n_20 = p20.length;
        const pAll = mk(peaks.length);
        const fAll = calculateFiguresOfMerit(qSorted, pAll, impurity_peaks || 0, tolFor(pAll), wavelength);
        cell.m_all = fAll.m20; cell.fN_all = fAll.fN; cell.n_all = pAll.length;
    } catch (e) { /* FoM optional */ }
    if (!isFinite(cell.m20)) cell.m20 = 0;

// If it's an auto-swap, don't append it to the manual tracking log!
    cell.manualSwaps = isAuto ? (solution.manualSwaps || []) : (solution.manualSwaps || []).concat(swaps);
    cell.nPaired = rows.length;
    return { cell: cell, swaps: swaps };
}
