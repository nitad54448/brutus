// js/crystallography/centering.js
// Lattice centering and extinction detection.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// Indexing is extinction-blind: each observed peak is assigned to the NEAREST
// calculated line, whatever that line's parity, because the absence rules do
// not exist yet at that point. When two lines straddle a peak the nearest one
// can easily be systematically absent. PbSO4 (anglesite, Pnma) is the worked
// example: the peak at 16.423 deg is assigned (0,1,0) at 16.404 (0.019 off)
// rather than (1,0,1) at 16.447 (0.024 off) - a 0.005 deg margin - even though
// 010 breaks 0kl: k+l=2n and cannot exist in that space group.
//
// Once detectExtinctions() has established the absence rules, that decision can
// be revisited: a peak whose assignment is forbidden is re-assigned to the
// nearest ALLOWED line, provided one lies inside the same indexing window.
//
// Three guards keep this from becoming a self-fulfilling loop:
//   1. If no allowed line is in range the original assignment is KEPT. Such a
//      peak is genuine evidence against the rules and must not be hidden.
//   2. The rules are NOT re-derived afterwards. Re-running detectExtinctions on
//      re-assigned data would confirm them trivially, since the data were just
//      edited to satisfy them.
//   3. A single pass, with a ceiling on how much may be re-assigned. Needing to
//      move a large fraction of the pattern means the rules are wrong, not the
//      assignments, so the whole pass is abandoned.


/**
 * Is there an ALLOWED alternative hkl that explains this observed peak about as
 * well as the assigned (rule-violating) one?
 *
 * The old test asked only whether an allowed hkl existed anywhere in the overlap
 * window. That is far too permissive, and it degrades badly with wavelength.
 * Peak separation follows d(2theta) = 2*tan(theta) * (dd/d), so a fixed 2theta
 * window corresponds to a lattice resolution of (dd/d) = d(2theta)/(2 tan theta)
 * — which blows up as theta falls. Short-wavelength anodes push the whole
 * pattern to low 2theta, so the same window swallows far more reflections. For
 * one real cell (the FAP monoclinic solution) the mean number of calculated hkl
 * within +-0.06 deg is ~4 for Cr, ~6 for Cu, ~15 for Mo and ~20 for Ag. Under a
 * mere presence test, essentially every violation on Mo/Ag data is demoted and
 * the extinction analysis stops working.
 *
 * Note the fix is NOT a wavelength factor on the window. Instrumental 2theta
 * uncertainty is set by alignment, sample displacement and detector resolution,
 * and is very nearly independent of the anode — so widening or narrowing the
 * window per wavelength would be inventing physics. What was wrong is the
 * binary test. Requiring the alternative to be COMPETITIVE with the assigned
 * reflection adapts automatically: on a crowded short-wavelength pattern the
 * extra neighbours are mostly far from the observed position and no longer
 * excuse the violation, while a genuine near-coincidence still does.
 *
 * A candidate competes if it lies within the user's stated tolerance of the
 * observed peak, or fits no more than AMBIGUITY_MARGIN times worse than the
 * assigned reflection does.
 */
function hasCompetingAllowedAlt(refl, isAllowed) {
    const alts = refl && refl.altHkls;
    if (!alts || alts.length <= 1) return false;

    const isSame = (a) => a.h === refl.h && a.k === refl.k && a.l === refl.l;
    const obs = refl.tth;

    // Without an observed position (or without per-alt positions) we cannot
    // judge proximity; fall back to the original presence test.
    if (typeof obs !== 'number' || !isFinite(obs)) {
        return alts.some(a => !isSame(a) && isAllowed(a));
    }

    const assignedTth = (typeof refl.calc_tth === 'number' && isFinite(refl.calc_tth))
        ? refl.calc_tth : obs;
    const dAssigned = Math.abs(assignedTth - obs);
    const floor = (typeof refl.tol === 'number' && isFinite(refl.tol)) ? refl.tol : Infinity;
    const limit = Math.max(dAssigned * AMBIGUITY_MARGIN, floor);

    return alts.some(a => {
        if (isSame(a) || !isAllowed(a)) return false;
        if (typeof a.tth !== 'number' || !isFinite(a.tth)) return true; // unknown -> old behaviour
        return Math.abs(a.tth - obs) <= limit;
    });
}
// R-centring, obverse and reverse. The indexer derives hkl labels from a
// hexagonal METRIC and never fixes the handedness of the in-plane basis against
// c, so obverse (-h+k+l = 3n) and reverse (h-k+l = 3n) are equally consistent
// with the same pattern -- they are a labelling convention, not physics, and no
// powder measurement can separate them. A reflection therefore only counts
// against R if it violates BOTH.
const _R_OBVERSE = (h, k, l) => (((-h + k + l) % 3) + 3) % 3 === 0;
const _R_REVERSE = (h, k, l) => (((h - k + l) % 3) + 3) % 3 === 0;
function determineCentering(indexed_hkls, system) {
    const centeringTests = { 'P': { name: 'Primitive (P)', forbidden: (h, k, l) => false }, 'I': { name: 'Body-centered (I)', forbidden: (h, k, l) => (h + k + l) % 2 !== 0 }, 'F': { name: 'Face-centered (F)', forbidden: (h, k, l) => !( (h%2===0 && k%2===0 && l%2===0) || (h%2!==0 && k%2!==0 && l%2!==0) ) }, 'A': { name: 'A-centered (A)', forbidden: (h, k, l) => (k + l) % 2 !== 0 }, 'B': { name: 'B-centered (B)', forbidden: (h, k, l) => (h + l) % 2 !== 0 }, 'C': { name: 'C-centered (C)', forbidden: (h, k, l) => (h + k) % 2 !== 0 },
        // R was missing entirely, and 'hexagonal' below listed only ['P']. So
        // even once the trigonal groups became reachable, determineCentering()
        // could never return R, allowedCenterings stayed ['P'], and
        // settingCenteringAllowed() then rejected every R setting downstream.
        // Both halves had to be fixed for a rhombohedral cell to be considered.
        'R': { name: 'Rhombohedral (R)', forbidden: (h, k, l) => !(_R_OBVERSE(h, k, l) || _R_REVERSE(h, k, l)) } };
    const validBravaisCenterings = { 'cubic': ['P', 'I', 'F'], 'tetragonal': ['P', 'I'], 'orthorhombic': ['P', 'I', 'F', 'A', 'B', 'C'], 'hexagonal': ['P', 'R'], 'monoclinic': ['P', 'A', 'B', 'C', 'I'], 'triclinic': ['P'] };
    // Two parallel violation tallies: hard (non-Ka2-suspect peaks) and soft
    // (Ka2-suspect peaks). The centering decision uses HARD only — a centering
    // mode is not ruled out just because a Ka2 ghost happens to violate it.
    const violations = {};       // total (hard + soft), kept for backward compat
    const violationsHard = {};
    const violationsSoft = {};
    const violationDetails = {};
    const MAX_DETAILS_TO_STORE = 2;
    for (const [key, test] of Object.entries(centeringTests)) {
        const allowedForSystem = validBravaisCenterings[system] || ['P'];
        if (allowedForSystem.includes(key)) {
            const violatingPeaks = indexed_hkls.filter(({h, k, l}) => test.forbidden(Math.round(h), Math.round(k), Math.round(l)));
            // A peak whose best-match hkl violates the centering rule
            // is downgraded from hard to soft if any of these apply:
            //   1. it is a Ka2-suspect ghost,
            //   2. it has an allowed alternative hkl within the
            //      overlap window (the rule could equally apply to a
            //      neighbour), or
            //   3. its observed intensity is below the low-intensity
            //      threshold (a near-zero peak is not strong evidence
            //      against the centering rule, since forbidden peaks
            //      should have intensity zero in the true cell).
            // This protects high-symmetry centerings from being killed
            // by a single ambiguous or weak hkl assignment.
            const hardViolatingPeaks = violatingPeaks.filter(p => {
                if (p.ka2Suspect) return false;
                if (p.lowIntensity) return false;
                if (hasCompetingAllowedAlt(p, alt => !test.forbidden(Math.round(alt.h), Math.round(alt.k), Math.round(alt.l)))) {
                    return false;
                }
                return true;
            });
            let softViolatingPeaks = violatingPeaks.filter(p => !hardViolatingPeaks.includes(p));

            // --- CONSENSUS OVERRIDE (see countViolations for rationale) ---
            // A centering mode must not survive on the strength of demotions
            // alone. Ka2 ghosts, weak tails and overlaps do not preferentially
            // land on the forbidden parity class; a large soft pile is evidence
            // that the centering is simply wrong.
            const CONSENSUS_MIN_COUNT = 5;
            const CONSENSUS_MIN_FRACTION = 0.15;
            let effectiveHard = hardViolatingPeaks;
            let effectiveSoft = softViolatingPeaks;
            if (effectiveHard.length === 0 &&
                effectiveSoft.length >= CONSENSUS_MIN_COUNT &&
                effectiveSoft.length >= CONSENSUS_MIN_FRACTION * indexed_hkls.length) {
                effectiveHard = effectiveSoft;
                effectiveSoft = [];
            }

            violations[key] = violatingPeaks.length; // total
            violationsHard[key] = effectiveHard.length;
            violationsSoft[key] = effectiveSoft.length;
            const hardViolatingPeaksFinal = effectiveHard;
            const softViolatingPeaksFinal = effectiveSoft;
            // Details: store hard violators preferentially, fall back to soft.
            if (violations[key] > 0 && violations[key] <= MAX_DETAILS_TO_STORE) {
                const detailsSource = hardViolatingPeaksFinal.length > 0 ? hardViolatingPeaksFinal : softViolatingPeaksFinal;
                violationDetails[key] = detailsSource.slice(0, MAX_DETAILS_TO_STORE).map(p => ({ h: p.h, k: p.k, l: p.l, tth: p.tth, ka2Suspect: !!p.ka2Suspect, lowIntensity: !!p.lowIntensity }));
            }
        }
    }
    // Pick centering(s) with the FEWEST HARD violations (was: fewest total).
    const hardKeys = Object.keys(violationsHard);
    const minHardViolations = hardKeys.length > 0 ? Math.min(...hardKeys.map(k => violationsHard[k])) : 0;
    let plausible = hardKeys.filter(key => violationsHard[key] === minHardViolations && (validBravaisCenterings[system] || ['P']).includes(key));
    if (plausible.length === 0 && violationsHard['P'] === minHardViolations) { plausible = ['P']; } else if (plausible.length === 0) { plausible = ['P']; }
    let finalCenterings;
    // F and I keep P alongside them, exactly as A/B/C and R already do.
    //
    // They used to collapse to ['F'] / ['I'] alone, and that single letter is a
    // HARD COMMIT: plausibleCenterings is the candidate filter for BOTH
    // detectExtinctions() and the compatibility list, so a wrong F or I verdict
    // did not demote the primitive settings, it deleted them. The correct group
    // was then absent from the report rather than ranked below the wrong one --
    // and "never a candidate" is indistinguishable, on the page, from "ruled
    // out". That is the pseudo-symmetry failure mode described at length in the
    // Space Group MC scoring notes: on PbSO4 an I-centred hypothesis looks clean
    // because no measured reflection happens to land in its forbidden parity
    // class, and the true P2_1/a cell is the thing that disappears.
    //
    // The asymmetry was almost certainly an oversight rather than a decision:
    // the R branch immediately below reasons explicitly about not slamming the
    // door on the primitive groups, and the A/B/C branch does the same. F and I
    // qualify on the same bar those do -- zero HARD violations, which P also
    // always has by construction -- so there is no ground for treating them as
    // more certain.
    //
    // This does NOT change the centering line the user sees: `description` is
    // built from reportedCenterings, which strips P a few lines below. Only the
    // candidate pool widens.
    if (plausible.includes('F')) finalCenterings = plausible.includes('P') ? ['F', 'P'] : ['F'];
    else if (plausible.includes('I')) finalCenterings = plausible.includes('P') ? ['I', 'P'] : ['I'];
    // R was absent from this hierarchy too, so even after it became testable it
    // fell through to the A/B/C branch, matched nothing, and was replaced by
    // ['P'] -- the R verdict was computed and then discarded one line later.
    // P is kept alongside it, following the A/B/C precedent rather than the F/I
    // one: the R test accepts obverse OR reverse and is correspondingly
    // permissive, so it should narrow the candidate pool without slamming the
    // door on every primitive hexagonal group.
    else if (plausible.includes('R')) finalCenterings = ['R', 'P'];
    else { const specialCenterings = plausible.filter(c => ['A', 'B', 'C'].includes(c)); finalCenterings = specialCenterings.length > 0 ? specialCenterings : ['P']; if (plausible.includes('P') && !finalCenterings.includes('P') && specialCenterings.length > 0) { finalCenterings.push('P'); } if (finalCenterings.length === 0) finalCenterings = ['P']; }
    finalCenterings = finalCenterings.filter(c => (validBravaisCenterings[system] || ['P']).includes(c));
    if (finalCenterings.length === 0) finalCenterings = ['P'];

    // --- REPORTED centering vs SEARCHED centering ---
    // P is defined with forbidden() === false, so it can never accumulate a
    // violation and is therefore ALWAYS among the zero-violation candidates.
    // Listing it next to a real centering ("A-centered (A) or Primitive (P)")
    // is structurally guaranteed rather than informative, and it is not done
    // for F or I, which are collapsed to a single symbol above. So the
    // human-readable description reports only the genuine centering.
    //
    // finalCenterings itself keeps P, because it is passed to rankSpaceGroups()
    // as the allowed-centering filter: dropping it there would remove every
    // primitive space group from the ranking. Whether the lattice is primitive
    // is settled by the space-group analysis and the extinction list, not by
    // this line. P remains relevant to cell reduction, where it means
    // "no centering transform applied".
    let reportedCenterings = finalCenterings.filter(c => c !== 'P');
    if (reportedCenterings.length === 0) reportedCenterings = ['P'];
    
    // Remove Primitive (P) from the reported dictionaries since it cannot have violations
    delete violations['P'];
    delete violationsHard['P'];
    delete violationsSoft['P'];
    delete violationDetails['P'];
    
    
    return {
        plausibleCenterings: finalCenterings,
        reportedCenterings: reportedCenterings,
        description: reportedCenterings.map(c => centeringTests[c]?.name || c).join(' or '),
        violations: violations,
        violationsHard: violationsHard,
        violationsSoft: violationsSoft,
        violationDetails: violationDetails,
        minViolations: minHardViolations
    };
}
function detectExtinctions(indexed_hkls, system, spaceGroupData, allowedCenterings, evidence) {
    const confirmedRules = new Set();
    if (!spaceGroupData?.space_groups || indexed_hkls.length === 0) { return ["None detected (no data or rules)"]; }
    if (!sgEnsureDatabase(spaceGroupData)) { return ["None detected (database has no operators)"]; }
    // --- CANDIDATE POOL: CRYSTAL SYSTEM *AND* CENTERING ---
    // Only conditions that a still-viable space group could actually own are
    // testable. Admitting rules from centerings the lattice test has already
    // eliminated lets an accidental agreement in a thinly-sampled zone
    // masquerade as a detected absence. See settingCenteringAllowed().
    const potentialRules = new Set();
    Object.values(spaceGroupData.space_groups).forEach(sg => {
        if (!sgSystemMatches(sg.crystal_system, system)) return;
        sg.settings.forEach(setting => {
            if (!sgSettingAxesMatch(setting, system)) return;
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) return;
            // Printed conditions only. The pool exists to name which INDIVIDUAL
            // condition the data supports, and that is an International Tables
            // presentation question -- absences themselves come from the
            // operators, in countViolations().
            const conditions = setting.conditions || {};
            Object.entries(conditions).forEach(([zone, condList]) => {
                condList.forEach(condStr => { potentialRules.add(`${zone}: ${condStr}`); });
            });
        });
    });
    if (potentialRules.size === 0) { return ["None detected (no rules for system)"]; }
    const parseRuleString = (ruleStr) => { const parts = ruleStr.split(': '); if (parts.length === 2) { return { zone: parts[0].trim(), condition: parts[1].trim() }; } return null; };

    // ================= EVIDENTIAL FLOOR =================
    // "Nothing contradicts it" is not evidence. The loop below only ever looks
    // at reflections that ARE present, so a rule survives by default in any
    // zone too thinly sampled to break it. A condition is only reported if the
    // measurement could have refuted it and did not.
    //
    // (1) ABSENCE TEST. Every calculated line the rule forbids that lies in the
    //     measured range and is RESOLVABLE - no line the rule permits sits
    //     close enough to lend it intensity - must actually be absent. A
    //     resolvable forbidden line carrying an observed peak refutes the rule
    //     outright, and a minimum number of clean absences must remain to
    //     support it.
    //
    //     This stage deliberately does NOT forgive weak peaks, unlike the
    //     contradiction test below. The lowIntensity demotion exists to stop a
    //     weak peak ELIMINATING a space group; letting it also help ASSERT an
    //     absence rule inverts its purpose. Both stages then err the same way:
    //     they refuse to over-constrain the answer.
    //
    // (2) SAMPLE TEST, for rules stronger than "=2n". If a zone permits a
    //     fraction p of its reflections, n observed reflections agree with the
    //     rule by chance with probability p^n. Two observed 0kl reflections
    //     that both happen to have k+l divisible by 4 (p = 1/4, p^n = 6%) are
    //     a coincidence, not a d glide. Plain =2n rules are exempt: they would
    //     need n >= 5 to clear the same bar, which a sparsely populated powder
    //     zone rarely supplies, and they are carried by the absence test.
    //
    // Zektzerite drove both. Its 0kl zone holds five observed reflections, of
    // which 012, 014 and 002 are demoted as weak, leaving 022 and 004 - both
    // with k+l = 4. Nothing contradicted 0kl: k+l=4n, so it was reported; the
    // subsumption pass then deleted the real 0kl: k=2n and 0kl: l=2n because
    // both follow from it; and every candidate group was marked down for
    // failing to explain a condition none of them can have.
    const MIN_VERIFIED_ABSENCES = 1;        // rules forbidding <= half a zone
    const MIN_VERIFIED_ABSENCES_STRONG = 2; // 4n, 3n, compound "h, l=2n", ...
    const CHANCE_LEVEL = 0.05;

    const obsTth = Array.isArray(evidence?.obsTth) ? evidence.obsTth.filter(Number.isFinite) : [];
    const indexWindow = Number.isFinite(evidence?.indexWindow) ? evidence.indexWindow : 0;
    const overlapWindow = Number.isFinite(evidence?.overlapWindow) ? evidence.overlapWindow : indexWindow;
    // A forbidden line is shadowed if an allowed line lies within one overlap
    // window of it, OR close enough that a peak within indexWindow of the
    // forbidden line could equally be that allowed line. Summing the two
    // windows closes the gap between "overlapping lines" and "the peak was
    // indexed to the neighbour", so a real reflection can never be mistaken
    // for a broken absence.
    const shadowWindow = overlapWindow + indexWindow;
    const tthLo = Number.isFinite(evidence?.tthMin) ? evidence.tthMin : -Infinity;
    const tthHi = Number.isFinite(evidence?.tthMax) ? evidence.tthMax : Infinity;
    // NOTE: generateHKL_for_analysis() collapses lines that coincide to within
    // 1e-4 deg, so in high-symmetry systems one of a set of exactly overlapping
    // reflections represents the rest. Such a line is untestable anyway (its
    // partners shadow it), so the only effect is a slightly conservative
    // absence count - hence the deliberately low minimums above.
    const rangeLines = Array.isArray(evidence?.calcLines)
        ? evidence.calcLines
            .filter(x => x && Number.isFinite(x.tth) && x.tth >= tthLo - 1e-9 && x.tth <= tthHi + 1e-9)
            .slice()
            .sort((a, b) => a.tth - b.tth)
        : null;

    const allowedFractionCache = {};
    const allowedFraction = (zone, cond) => {
        const key = zone + '|' + cond;
        if (allowedFractionCache[key] !== undefined) return allowedFractionCache[key];
        const R = 8;
        let total = 0, allowed = 0;
        for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
            if (h === 0 && k === 0 && l === 0) continue;
            if (!zoneApplies(zone, h, k, l)) continue;
            total++;
            if (satisfiesCondition(h, k, l, cond)) allowed++;
        }
        const f = total > 0 ? allowed / total : 1;
        allowedFractionCache[key] = f;
        return f;
    };

    // Per-line lookups that do not depend on the rule, computed once.
    const nLines = rangeLines ? rangeLines.length : 0;
    const peakOnLine = rangeLines
        ? Uint8Array.from(rangeLines, L => obsTth.some(o => Math.abs(o - L.tth) <= indexWindow) ? 1 : 0)
        : null;
    const zoneMaskCache = {};
    const zoneMask = (zone) => {
        if (zoneMaskCache[zone]) return zoneMaskCache[zone];
        const mask = Uint8Array.from(rangeLines, L => zoneApplies(zone, L.h, L.k, L.l) ? 1 : 0);
        zoneMaskCache[zone] = mask;
        return mask;
    };

    // Absences the data can vouch for, and absences the data breaks.
    const absenceEvidence = (zone, cond) => {
        if (!rangeLines || nLines === 0) {
            return { verified: Infinity, broken: 0, tested: false }; // no line list -> test disabled
        }
        const inZone = zoneMask(zone);
        const permits = new Uint8Array(nLines);
        let anyForbidden = false;
        for (let i = 0; i < nLines; i++) {
            const L = rangeLines[i];
            permits[i] = (!inZone[i] || satisfiesCondition(L.h, L.k, L.l, cond)) ? 1 : 0;
            if (!permits[i]) anyForbidden = true;
        }
        if (!anyForbidden) return { verified: 0, broken: 0, tested: true };
        let verified = 0, broken = 0;
        for (let i = 0; i < nLines; i++) {
            if (permits[i]) continue; // the rule allows it; it says nothing
            const t = rangeLines[i].tth;
            let shadowed = false;
            for (let j = i - 1; j >= 0 && t - rangeLines[j].tth <= shadowWindow; j--) {
                if (permits[j]) { shadowed = true; break; }
            }
            for (let j = i + 1; !shadowed && j < nLines && rangeLines[j].tth - t <= shadowWindow; j++) {
                if (permits[j]) { shadowed = true; break; }
            }
            if (shadowed) continue;
            if (peakOnLine[i]) broken++; else verified++;
        }
        return { verified, broken, tested: true };
    };

    // Could n reflections agree with this rule purely by chance?
    const sampleSufficient = (zone, cond, nObs) => {
        const p = allowedFraction(zone, cond);
        if (!(p > 0) || p >= 0.5 - 1e-9) return true; // =2n class and degenerate cases exempt
        return nObs >= Math.ceil(Math.log(CHANCE_LEVEL) / Math.log(p));
    };
    potentialRules.forEach(ruleStr => {
        const parsedRule = parseRuleString(ruleStr); if (!parsedRule) return;
        const { zone, condition } = parsedRule;
            const zoneReflections = indexed_hkls.filter(refl => zoneApplies(zone, refl.h, refl.k, refl.l));
        if (zoneReflections.length === 0) { return; }
        // A rule is "confirmed" if every reliable reflection in the
        // zone satisfies it. Two classes of unreliable reflections are
        // excluded from this check:
        //   - Ka2-suspect peaks: their 2θ is shifted, so we can't
        //     reliably re-index them back to the same hkl;
        //   - low-intensity peaks: a near-zero observed peak doesn't
        //     contradict an extinction rule even if its assigned hkl
        //     formally violates the rule (forbidden peaks SHOULD be
        //     near zero).
        // If no reliable reflections remain in the zone we fall back
        // to the full set so the test can still operate.
        const reliableZoneRefls = zoneReflections.filter(r => !r.ka2Suspect && !r.lowIntensity);
        const refSetForRule = reliableZoneRefls.length > 0 ? reliableZoneRefls : zoneReflections;
        // A reflection that formally breaks the rule is forgiven if another
        // calculated hkl within the same peak's ambiguity window is not
        // forbidden by it — the peak could equally well be that reflection.
        // This is the SAME demotion countViolations() applies when ranking; without
        // it the two analyses disagree, and a rule the ranking is happy to treat
        // as satisfied never appears in the detected list. The rule does not
        // constrain an alternative outside its own zone, so such an alternative
        // counts as allowed.
        const ruleAllows = (a) => !zoneApplies(zone, a.h, a.k, a.l) ||
                                  satisfiesCondition(a.h, a.k, a.l, condition);
        let hardFails = 0, forgiven = 0;
        for (const refl of refSetForRule) {
            if (satisfiesCondition(refl.h, refl.k, refl.l, condition)) continue;
            if (hasCompetingAllowedAlt(refl, ruleAllows)) forgiven++;
            else hardFails++;
        }
        // Consensus guard, mirroring countViolations(): forgiving one overlap is
        // reasonable, forgiving a systematic trend is not. Overlap is uncorrelated
        // with index parity, so if many reflections in the zone all break the same
        // rule, the rule is genuinely broken however excusable each case looks.
        const EXT_CONSENSUS_MIN_COUNT = 5;
        const EXT_CONSENSUS_MIN_FRACTION = 0.15;
        const consensusBroken = forgiven >= EXT_CONSENSUS_MIN_COUNT &&
                                forgiven >= EXT_CONSENSUS_MIN_FRACTION * refSetForRule.length;
        const allSatisfy = (hardFails === 0) && !consensusBroken;
        if (!allSatisfy) return;

        // --- EVIDENTIAL FLOOR (see the block above the loop) ---
        const strong = allowedFraction(zone, condition) < 0.5 - 1e-9;
        const minAbsences = strong ? MIN_VERIFIED_ABSENCES_STRONG : MIN_VERIFIED_ABSENCES;
        const { verified, broken, tested } = absenceEvidence(zone, condition);
        if (broken > 0) {
            console.debug(`[detectExtinctions] rejected "${ruleStr}": ${broken} resolvable forbidden line(s) carry an observed peak.`);
            return;
        }
        if (tested && verified < minAbsences) {
            console.debug(`[detectExtinctions] rejected "${ruleStr}": only ${verified} verified absence(s) in range, need ${minAbsences}.`);
            return;
        }
        if (!sampleSufficient(zone, condition, refSetForRule.length)) {
            console.debug(`[detectExtinctions] rejected "${ruleStr}": ${refSetForRule.length} reliable reflection(s) in ${zone} cannot distinguish it from chance.`);
            return;
        }
        confirmedRules.add(ruleStr);
    });
    if (confirmedRules.size === 0) { return ["None detected"]; }

    // --- COLLAPSE SUBSUMED RULES ---
    // A rule set built by "keep everything nothing contradicts" is riddled with
    // redundancy. Two distinct cases:
    //
    //  (a) one rule implies another. If the only observed 00l is 004, both
    //      00l: l=2n and 00l: l=4n survive, because l=4n implies l=2n. l=4n is
    //      the stronger claim (it also forbids 002 and 006), so it is the only
    //      one carrying information.
    //  (b) a rule is implied by the CONJUNCTION of others without being implied
    //      by any single one. Zektzerite reports hk0: h=2n, hk0: k=2n AND
    //      hk0: h+k=2n; the third follows from the first two (even + even is
    //      even) but from neither alone, so a pairwise test cannot see it.
    //
    // A rule is therefore dropped when every reflection allowed by ALL the
    // other surviving rules already satisfies it. Removal is iterative, so a
    // set of mutually-redundant rules collapses to a minimal equivalent subset
    // rather than vanishing entirely. Rules using more indices are offered for
    // removal first, so the conventional form (h=2n, k=2n) is kept over the
    // derived combination (h+k=2n).
    //
    // Implication is tested by enumeration, not algebraically, so compound
    // shorthand and any modulus are handled without special cases. Rules from
    // OTHER zones are honoured too when they apply to the reflection, so e.g. a
    // general hkl condition can subsume a zonal restatement of itself.
    const rulesArr = Array.from(confirmedRules);
    const zoneLatticeCache = {};
    const zonePoints = (zone) => {
        if (zoneLatticeCache[zone]) return zoneLatticeCache[zone];
        const pts = [];
        const R = 8;
        for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
            if (h === 0 && k === 0 && l === 0) continue;
            if (zoneApplies(zone, h, k, l)) pts.push([h, k, l]);
        }
        zoneLatticeCache[zone] = pts;
        return pts;
    };
    // Is `target` implied by the conjunction of `others` over target's zone?
    // Vacuous cases (nothing survives the others) are NOT treated as implied,
    // so an over-constrained set can never silently delete a real rule.
    const impliedByConjunction = (others, target) => {
        const pts = zonePoints(target.zone);
        if (pts.length === 0) return false;
        let allowedAny = false;
        for (const [h, k, l] of pts) {
            let allowed = true;
            for (const o of others) {
                if (!zoneApplies(o.zone, h, k, l)) continue;
                if (!satisfiesCondition(h, k, l, o.condition)) { allowed = false; break; }
            }
            if (!allowed) continue;
            allowedAny = true;
            if (!satisfiesCondition(h, k, l, target.condition)) return false;
        }
        return allowedAny;
    };
    const nIndices = (cond) => ['h', 'k', 'l'].filter(v => new RegExp('(^|[^a-z])' + v).test(String(cond))).length;
    let kept = rulesArr.map(parseRuleString).map((p, i) => p ? { ...p, raw: rulesArr[i] } : null).filter(Boolean);
    const unparsed = rulesArr.filter((r, i) => !parseRuleString(r));
    // Offer the most "derived-looking" rules for removal first.
    kept.sort((a, b) => nIndices(b.condition) - nIndices(a.condition) || a.raw.localeCompare(b.raw));
    let removedSomething = true;
    while (removedSomething && kept.length > 1) {
        removedSomething = false;
        for (let i = 0; i < kept.length; i++) {
            const others = kept.filter((_, j) => j !== i);
            if (others.length === 0) break;
            if (impliedByConjunction(others, kept[i])) {
                kept.splice(i, 1);
                removedSomething = true;
                break;
            }
        }
    }
    const survivors = kept.map(k => k.raw).concat(unparsed);
    return (survivors.length > 0 ? survivors : rulesArr).sort();
}
