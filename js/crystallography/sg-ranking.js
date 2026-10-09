// js/crystallography/sg-ranking.js
// Space-group ranking from extinction conditions.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

function rankSpaceGroups(indexed_hkls, system, allowedCenterings, spaceGroupData, maxViolations, detectedExtinctions) {
    if (!sgEnsureDatabase(spaceGroupData)) return [];
    const candidateGroups = Object.values(spaceGroupData.space_groups)
        .filter(sg => sgSystemMatches(sg.crystal_system, system));
    const validSettings = [];
    
    // Statistical weights for centering order: higher symmetry constrains more reciprocal space
    const centeringWeights = { 'P': 1.0, 'A': 1.5, 'B': 1.5, 'C': 1.5, 'I': 2.0, 'F': 2.0, 'R': 2.0 };
    // Cache of condition selectivity weights, shared across all candidate settings.
    const selectivityCache = {};

    // --- DETECTED-EXTINCTION AGREEMENT ---
    // countViolations() only ever looks at reflections that ARE present: it can
    // punish a group for predicting an absence that did not happen, but it can
    // never reward one for predicting an absence that did. That asymmetry means
    // a group whose condition set is a strict SUBSET of another's can never
    // score worse on violations, so the least-constrained group wins by default.
    // Zektzerite is the worked example: Abma (#64) is exactly Abmm (#67) plus
    // hk0: h=2n, and the data show h00: h=2n - a condition Abmm cannot explain,
    // because A-centering gives k+l=0 for h00, which is even for every h.
    //
    // detectExtinctions() already distilled the observed absences into rules.
    // Here each detected rule is treated as evidence: a setting that ENTAILS it
    // is rewarded, one that leaves it unexplained is penalised. Weighting is by
    // the condition's selectivity times the number of observed reflections in
    // its zone, so a well-supported, highly restrictive condition counts for
    // more than a weakly-sampled one.
    const EXT_WEIGHT = 0.5;
    const detectedList = (Array.isArray(detectedExtinctions) ? detectedExtinctions : [])
        .map(s => { const p = String(s).split(': '); return p.length === 2 ? { zone: p[0].trim(), cond: p[1].trim() } : null; })
        .filter(Boolean);
    // Reflections observed in each detected zone = how much data backs the rule.
    detectedList.forEach(dc => {
        dc.nObs = indexed_hkls.filter(r => zoneApplies(dc.zone, r.h, r.k, r.l)).length;
    });

    // Does this rule set entail `cond` over `zone`? i.e. is every reflection the
    // setting allows in that zone already required to satisfy the condition?
    const entailmentCache = {};
    const entails = (rules, zone, cond) => {
        const key = JSON.stringify(rules) + '|' + zone + '|' + cond;
        if (entailmentCache[key] !== undefined) return entailmentCache[key];
        const R = 8;
        let allowedAny = false, ok = true;
        for (let h = -R; h <= R && ok; h++) for (let k = -R; k <= R && ok; k++) for (let l = -R; l <= R && ok; l++) {
            if (h === 0 && k === 0 && l === 0) continue;
            if (!zoneApplies(zone, h, k, l)) continue;
            let allowed = true;
            for (const { cond: rc } of applicableRules(rules, h, k, l)) {
                if (!satisfiesCondition(h, k, l, rc)) { allowed = false; break; }
            }
            if (!allowed) continue;
            allowedAny = true;
            if (!satisfiesCondition(h, k, l, cond)) ok = false;
        }
        const res = allowedAny && ok;
        entailmentCache[key] = res;
        return res;
    };

    for (const sg of candidateGroups) {
        const sgNumber = sg.number;
        for (const setting of sg.settings) {
            const centering = setting.symbol.charAt(0);
            // Same filters detectExtinctions() and sgExtinctionClasses() apply.
            // The axes test was missing here: the rhombohedral-axes settings of
            // the seven R groups (hall "P 3*", conditions written for indices
            // this program never produces, mostly none at all) were ranked as
            // hexagonal candidates, so every R group was listed through a
            // duplicate that forbids nothing.
            if (!sgSettingAxesMatch(setting, system)) { continue; }
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) { continue; }
            
            const rules = setting.conditions || {};
            const violations = countViolations(indexed_hkls, setting, system);
            
            // Cutoff remains on HARD violations only
            if (violations.hardCount <= maxViolations) {
                let nConfirmTotal = 0;

                // Selectivity weight for a condition, NORMALISED so that an ordinary
                // "=2n" rule (which forbids half its zone) keeps weight 1.0. A
                // reflection satisfying l=4n is stronger evidence than one
                // satisfying l=2n, because l=4n forbids 3/4 of the zone against
                // 1/2, so it scores 1.5. Without this, 004 confirms both equally
                // and P41/P43 tie with P42/P4222 on identical data, leaving the
                // ranking to fall through to the space-group-number tiebreak.
                // Normalising (rather than using the raw forbidden fraction)
                // keeps every existing 2n-based score unchanged, so this only
                // promotes genuinely stronger rules instead of shifting the
                // whole ranking relative to the rule-free baseline of 1.0.
                const selectivity = (zone, cond) => {
                    const key = zone + '|' + cond;
                    if (selectivityCache[key] !== undefined) return selectivityCache[key];
                    const R = 8;
                    let total = 0, allowed = 0;
                    for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
                        if (h === 0 && k === 0 && l === 0) continue;
                        if (!zoneApplies(zone, h, k, l)) continue;
                        total++;
                        if (satisfiesCondition(h, k, l, cond)) allowed++;
                    }
                    // forbidden fraction / 0.5, clamped: a degenerate or unparsed
                    // condition must not silently zero out a confirmation.
                    let w = total > 0 ? (1 - allowed / total) / 0.5 : 1.0;
                    w = Math.min(3.0, Math.max(0.5, w));
                    selectivityCache[key] = w;
                    return w;
                };

                // Harvest positive confirmations across all space group rules
                Object.entries(rules).forEach(([zone, conditions]) => {
                    conditions.forEach(cond => {
                        const w = selectivity(zone, cond);
                        indexed_hkls.forEach(refl => {
                            // General 'hkl' rules apply to all reflections; specific zones apply only to their zone
                            const applies = zoneApplies(zone, refl.h, refl.k, refl.l);

                            if (applies && satisfiesCondition(refl.h, refl.k, refl.l, cond)) {
                                // Full point for strong/reliable reflections; 0.25 for Ka2-suspect or weak tails,
                                // each scaled by how selective the confirmed condition is.
                                if (!refl.ka2Suspect && !refl.lowIntensity) {
                                    nConfirmTotal += 1.0 * w;
                                } else {
                                    nConfirmTotal += 0.25 * w;
                                }
                            }
                        });
                    });
                });

                const wCenter = centeringWeights[centering] || 1.0;

                // Reward/penalise agreement with the observed systematic absences.
                let extBonus = 0, extExplained = 0, extMissed = [];
                for (const dc of detectedList) {
                    if (dc.nObs === 0) continue;
                    const w = selectivity(dc.zone, dc.cond);
                    if (entails(rules, dc.zone, dc.cond)) {
                        extExplained++;
                        extBonus += EXT_WEIGHT * w * dc.nObs;
                    } else {
                        extMissed.push(`${dc.zone}: ${dc.cond}`);
                        extBonus -= EXT_WEIGHT * w * dc.nObs;
                    }
                }

                // FoM_stat: Weighted confirmations minus penalized soft violations,
                // plus agreement with the detected systematic absences.
                // We store this in 'matchScore' to maintain seamless backward compatibility with your UI
                let fomStat = wCenter * (nConfirmTotal - (1.5 * violations.softCount)) + extBonus;
                if (fomStat === 0 && Object.keys(rules).length === 0) fomStat = 1.0; // Baseline for P with no rules

                validSettings.push({
                    number: sgNumber,
                    symbol: setting.symbol,
                    settingId: setting.setting_id,
                    standardSymbol: sg.standard_symbol,
                    pointGroup: sg.point_group,
                    centrosymmetric: sg.centrosymmetric,
                    violations: violations.hardCount,
                    hardViolations: violations.hardCount,
                    softViolations: violations.softCount,
                    violatedReflections: violations.details,
                    violatedReflectionsHard: violations.detailsHard,
                    violatedReflectionsSoft: violations.detailsSoft,
                    extinctionsExplained: extExplained,
                    extinctionsTotal: detectedList.filter(d => d.nObs > 0).length,
                    extinctionsUnexplained: extMissed,
                    extinctionBonus: extBonus,
                    matchScore: fomStat
                });
            }
        }
    }
    
    // Sort: hard violations ASC -> unexplained detected absences ASC
    //       -> FoM_stat (matchScore) DESC -> soft violations ASC -> number DESC
    //
    // The absence pattern is promoted above matchScore deliberately. matchScore
    // counts reflections that ARE present, so it cannot distinguish a setting
    // from a strictly less-constrained one except by the handful of extra
    // confirmations the extra rule happens to collect - and that difference is
    // swamped by the shared, well-populated zones. Systematic ABSENCES are the
    // primary evidence in space-group determination, so a setting that accounts
    // for every detected absence ranks above one that leaves some unexplained,
    // and matchScore then separates settings that explain the same pattern.
    validSettings.sort((a, b) => {
        if (a.hardViolations !== b.hardViolations) return a.hardViolations - b.hardViolations;
        const au = (a.extinctionsUnexplained || []).length, bu = (b.extinctionsUnexplained || []).length;
        if (au !== bu) return au - bu;
        if (Math.abs(a.matchScore - b.matchScore) > 1e-4) return b.matchScore - a.matchScore;
        if (a.softViolations !== b.softViolations) return a.softViolations - b.softViolations;
        return b.number - a.number;
    });
    
    return validSettings;
}
const satisfiesCondition = (h, k, l, condStr) => {
    if (condStr === "h+k, k+l, h+l=2n") { 
        const h_int = Math.round(h), k_int = Math.round(k), l_int = Math.round(l); 
        return ((h_int + k_int) % 2 === 0 && (k_int + l_int) % 2 === 0 && (h_int + l_int) % 2 === 0); 
    }
    
    // Extract shared equality suffix (e.g., "=2n" or "=4n") to handle shorthand like "h, l=2n"
    const rhsMatch = condStr.match(/=\s*(\d+)n/);
    const defaultRhs = rhsMatch ? rhsMatch[0] : "=2n";
    
    const conditions = condStr.split(',').map(s => s.trim());
    for (const condition of conditions) {
        let cleanCond = condition.replace(/\*/g, '');
        
        // If a shorthand part like "h" is missing its modulus, append the shared suffix
        if (!cleanCond.includes('=')) {
            cleanCond += defaultRhs;
        }
        
        const match = cleanCond.match(/([0-9]*[hkl\+\-]+)\s*=\s*(\d+)n/);
        if (!match) { 
            console.warn(`[satisfiesCondition] Could not parse rule part: "${condition}" in rule string "${condStr}"`); 
            continue; 
        }
        const [, expr, modStr] = match; 
        const mod = parseInt(modStr);
        if (isNaN(mod) || mod <= 0) { 
            console.warn(`[satisfiesCondition] Invalid modulus in rule part: "${condition}"`); 
            continue; 
        }
        let value = 0; 
        const terms = expr.match(/[+-]?[0-9]*[hkl]/g) || [];
        for (const term of terms) {
            let sign = 1, coeff = 1, variable = '';
            const coeffMatch = term.match(/^([+-]?)(\d*)([hkl])$/);
            if (coeffMatch) {
                sign = (coeffMatch[1] === '-') ? -1 : 1; 
                coeff = coeffMatch[2] ? parseInt(coeffMatch[2]) : 1; 
                variable = coeffMatch[3];
                const h_int = Math.round(h), k_int = Math.round(k), l_int = Math.round(l);
                if (variable === 'h') value += sign * coeff * h_int;
                else if (variable === 'k') value += sign * coeff * k_int;
                else if (variable === 'l') value += sign * coeff * l_int;
            } else { 
                console.warn(`[satisfiesCondition] Could not parse term "${term}" in expression "${expr}"`); 
            }
        }
        if (Math.round(value) % mod !== 0) { return false; }
    }
    return true;
};
// A violation is an OBSERVED reflection the candidate says is systematically
// absent. `setting` supplies the operators, which answer that exactly and
// completely: no zone lookup, no inheritance, and no dependence on whether the
// tables happened to print the condition that kills a particular reflection.
//
// The rule strings are still consulted, but only to NAME the violated condition
// in the detail line the report shows. If no printed condition matches -- which
// happens when the absence follows from a condition the tables leave implied --
// the detail says so rather than inventing one.
// `system` makes the test a question about the LINE, as Space Group MC asks it
// (sgOpsAllowedFn): a powder line is forbidden only if every reflection on it
// is -- its whole metric orbit, plus any other reflection the generator merged
// into it (reflection.coincident). Testing the one representative hkl the
// generator kept was wrong whenever that representative is forbidden while a
// partner on the same line is allowed:
//   - R groups in hexagonal axes: (h,k,l) and (k,h,l) share a line but fall
//     under different obverse conditions (-h+k+l vs h-k+l); every true R
//     group was rejected on its own pattern;
//   - Laue classes below the holohedry (m-3: Pa-3, Ia-3): 320 is forbidden by
//     hk0: h=2n, 230 is not;
//   - exact metric coincidences (cubic 221/300 at N = 9): P-43n, Pm-3n,
//     Pn-3n, F-43c, Fm-3c, Fd-3c, I-43d, Ia-3d all collected hard violations
//     on patterns that obey them exactly.
// Without `system` (an older caller) the representative test is kept.
function countViolations(indexed_hkls, setting, system) {
    let hardCount = 0;
    let softCount = 0;
    const detailsHard = [];
    const detailsSoft = [];

    const C = sgOpsCompile(setting);
    if (!C) return { count: 0, hardCount: 0, softCount: 0,
                     details: [], detailsHard: [], detailsSoft: [] };
    const printed = sgSettingConditions(setting);

    const allowedFn = system ? sgOpsAllowedFn(setting, system) : null;
    const hklViolatesRules = allowedFn
        ? (h, k, l, coincident) => {
            if (allowedFn(h, k, l)) return false;
            if (coincident) for (const c of coincident) if (allowedFn(c.h, c.k, c.l)) return false;
            return true;
        }
        : (h, k, l) => sgOpsAbsent(h, k, l, C);

    // Which printed condition explains this absence? Presentation only.
    const nameFor = (h, k, l) => {
        for (let i = 0; i < printed.length; i++) {
            const { zone, cond } = printed[i];
            if (zoneApplies(zone, h, k, l) && !satisfiesCondition(h, k, l, cond)) {
                return `${zone}: ${cond}`;
            }
        }
        return 'a systematic absence of this group';
    };

    for (const reflection of indexed_hkls) {
        const { h, k, l, calc_tth } = reflection;
        const isSuspect = !!reflection.ka2Suspect;
        const isLowIntensity = !!reflection.lowIntensity;
        // A reflection is treated as soft if it is either a Ka2-ghost
        // suspect OR if its observed intensity is below the
        // low-intensity cutoff (10% of the strongest peak). The
        // low-intensity case captures the crystallographic intuition
        // that a "forbidden" reflection in the true space group should
        // have intensity zero — a weak observed peak is consistent with
        // residual background, weak overlap from a neighbour, or noise,
        // and is not strong evidence against the systematic-absence
        // rule.
        const isSoftSource = isSuspect || isLowIntensity;
        let isViolation = false;
        let violationDetail = null;
        const softTagFor = (refl) => {
            const tags = [];
            if (refl.ka2Suspect) tags.push('Ka2-suspect');
            if (refl.lowIntensity) tags.push('weak');
            return tags.length > 0 ? ` [${tags.join(', ')}]` : '';
        };
        if (hklViolatesRules(h, k, l, reflection.coincident)) {
            isViolation = true;
            const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
            violationDetail = `(${h},${k},${l})${tth_string} violates ${nameFor(h, k, l)}${softTagFor(reflection)}`;
        }

        // --- AMBIGUOUS-HKL DEMOTION ---
        // If the best-match hkl violates a rule but a different calculated
        // hkl within the same peak's tolerance window satisfies all the
        // rules, the violation is not real evidence against the space
        // group: the peak could equally well be assigned to the allowed
        // alternative. Treat such cases as soft so a single near-tolerance
        // peak can't kill an otherwise excellent space group.
        if (isViolation && hasCompetingAllowedAlt(reflection, alt => !hklViolatesRules(alt.h, alt.k, alt.l, alt.coincident))) {
            const tth_string = calc_tth ? ` at ${calc_tth.toFixed(3)}°` : '';
            violationDetail = `(${h},${k},${l})${tth_string} ambiguous (allowed alt within tol)`;
            softCount++;
            detailsSoft.push(violationDetail);
            continue; // skip the original hard/soft accounting below
        }

        if (isViolation) {
            if (isSoftSource) {
                softCount++;
                detailsSoft.push(violationDetail);
            } else {
                hardCount++;
                detailsHard.push(violationDetail);
            }
        }
    }
    // --- CONSENSUS OVERRIDE ---
    // Each demotion above (Ka2-ghost, weak, overlapped) is a statement that ONE
    // reflection is poor evidence. None of them licenses ignoring a systematic
    // trend. Noise, tails and Ka2 ghosts are not correlated with h+k+l parity,
    // so if many independent reflections all break the SAME rule, the rule is
    // genuinely broken and the demotions are concealing a real result. Promote
    // the whole soft pile to hard once it passes both an absolute and a
    // proportional floor.
    const CONSENSUS_MIN_COUNT = 5;
    const CONSENSUS_MIN_FRACTION = 0.15;
    const nExamined = indexed_hkls.length;
    if (hardCount === 0 && softCount >= CONSENSUS_MIN_COUNT &&
        softCount >= CONSENSUS_MIN_FRACTION * nExamined) {
        hardCount = softCount;
        softCount = 0;
        detailsHard.push(...detailsSoft.splice(0, detailsSoft.length));
    }

    // 'count' and 'details' kept as combined values for any pre-existing
    // caller that doesn't yet read the split fields. detailsHard/detailsSoft
    // are uncapped (used by the PDF report to list every violating hkl);
    // 'details' stays capped as a short legacy summary.
    const count = hardCount + softCount;
    const details = detailsHard.concat(detailsSoft).slice(0, 5);
    return { count, hardCount, softCount, details, detailsHard, detailsSoft };
}
function getReflectionZone(h, k, l) {
    const ah = Math.abs(h), ak = Math.abs(k), al = Math.abs(l);
    if (ak === 0 && al === 0 && ah !== 0) return 'h00'; 
    if (ah === 0 && al === 0 && ak !== 0) return '0k0'; 
    if (ah === 0 && ak === 0 && al !== 0) return '00l';
    if (ah === 0 && ak !== 0 && al !== 0) return '0kl'; 
    if (ak === 0 && ah !== 0 && al !== 0) return 'h0l'; 
    if (al === 0 && ah !== 0 && ak !== 0) return 'hk0';
    if (ah !== 0 && ah === ak && al !== 0) return 'hhl'; 
    if (ak !== 0 && ak === al && ah !== 0) return 'hkk'; 
    if (ah !== 0 && ah === al && ak !== 0) return 'hll';
    return 'hkl';
}
// Does a database group belong to the crystal system the indexer reported?
//
// The indexer classifies cells by METRIC, and getSymmetry() has no 'trigonal'
// verdict by design: a trigonal cell in hexagonal axes has a = b, gamma = 120,
// which it correctly calls 'hexagonal'. The database, generated with gemmi,
// labels groups 143-167 'trigonal'. A strict equality test therefore hid 25
// groups and 32 settings from EVERY stage of the space-group analysis --
// detectExtinctions(), rankSpaceGroups() and the Monte-Carlo scan alike.
//
// Among them is every R-centred group: 146, 148, 155, 160, 161, 166 and 167.
// Calcite, corundum, hematite, the carbonates and most of the rhombohedral
// oxides were not ranked poorly, they were never candidates at all, and no
// message anywhere said so. The reverse mapping is deliberately NOT applied: a
// genuine 6-fold group must not be offered for a cell the metric only supports
// as trigonal, and since getSymmetry() never emits 'trigonal' the question does
// not arise from the other side.
function sgSystemMatches(sgSystem, system) {
    if (sgSystem === system) return true;
    return system === 'hexagonal' && sgSystem === 'trigonal';
}
// Is this setting's condition list written for the same index convention the
// indexer produces?
//
// This is NOT a question about which cell a lattice can be described in. Every
// rhombohedral lattice can of course be indexed in hexagonal axes, and that is
// exactly what this program does. The question is which of the SEVERAL condition
// lists the database stores for one group refers to the indices we actually
// have. A reflection has different hkl in different settings, so a condition
// list is only meaningful alongside the axes it was written for.
//
// HEXAGONAL. The seven R groups each carry two settings:
//     R-3c  hexagonal axes  hall "-R 3 2\"c"  {hkl: -h+k+l=3n, 0kl: l=2n,
//                                               h0l: l=2n, 00l: l=6n}
//     R-3c  RHOMBOHEDRAL    hall "-P 3* 2n"   {hhl: l=2n, h00: h=2n, 0k0: k=2n}
// Same group, disjoint condition lists, because the second describes the
// primitive rhombohedral cell where the lattice is no longer centred at all.
// Worse, R3, R-3, R32, R3m and R-3m have an EMPTY condition list in rhombohedral
// axes: admitted, they join the class that forbids nothing, so a primitive row
// ends up listing R-3m among its members. Hall notation marks the rhombohedral
// three-fold with '3*' -- exactly seven settings in the bundled database.
//
// MONOCLINIC. The database stores all three unique-axis conventions: 105
// settings, 35 a-unique, 35 b-unique, 35 c-unique. P2_1/c appears as
//     P121/c1  b-unique  {h0l: l=2n, 0k0: k=2n}   <- matches this program
//     P1121/a  c-unique  {hk0: h=2n, 00l: l=2n}
//     P21/c11  a-unique  {0kl: l=2n, h00: h=2n}
// and these are genuinely different behaviours, not restatements. This program
// is b-unique throughout -- getLSDesignRow() returns [h2, k2, l2, h*l] with the
// beta cross-term, extractCellFromFit() pins alpha = gamma = 90, and
// sgEquivalents() uses the 2/m orbit about b -- so the other 70 settings would
// have their conditions tested against indices they do not describe. In Hall
// notation the axis follows the rotation order and z is the default, so '2y'
// marks the b-unique settings exactly.
//
// A setting with no Hall symbol is kept: better to test a condition list that
// might not apply than to silently drop a group over missing metadata.
function sgSettingAxesMatch(setting, system) {
    const hall = String((setting && setting.hall) || '');
    if (!hall) return true;
    if (system === 'hexagonal')  return !hall.includes('3*');
    if (system === 'monoclinic') return hall.includes('2y');
    return true;
}
// Is a space-group setting compatible with the centering(s) the lattice
// analysis left standing?
//
// Shared by rankSpaceGroups() and detectExtinctions() so both stages consider
// exactly the same settings. They used to disagree: the ranking filtered by
// centering, the extinction detector did not, so the detector could "confirm"
// a condition that no surviving space group is even able to possess, and then
// the ranking penalised every candidate for failing to explain it. Zektzerite
// is the worked example - 0kl: k+l=4n exists only in Fdd2 (43) and Fddd (70),
// both F-centred, on a lattice the centering test had already fixed as B with
// zero violations.
//
// An empty/absent allow-list means "no filtering" so older callers behave as
// before.
function settingCenteringAllowed(symbol, allowedCenterings) {
    if (!Array.isArray(allowedCenterings) || allowedCenterings.length === 0) return true;
    const centering = String(symbol || '').charAt(0);
    if (allowedCenterings.includes(centering)) return true;
    // A leading letter that is not a centering type at all is admitted whenever
    // P survived.
    return allowedCenterings.includes('P') && !['I', 'F', 'A', 'B', 'C', 'R'].includes(centering);
}
// All rule conditions that apply to a reflection, gathered across every zone
// the reflection belongs to. Deduplicated by "zone: condition" so the same
// predicate listed under two zones is not counted twice, while keeping the
// zone label for reporting.
function applicableRules(rules, h, k, l) {
    const seen = new Set();
    const out = [];
    for (const [zone, conds] of Object.entries(rules || {})) {
        if (!Array.isArray(conds)) continue;
        if (!zoneApplies(zone, h, k, l)) continue;
        for (const cond of conds) {
            if (seen.has(cond)) continue;
            seen.add(cond);
            out.push({ zone, cond });
        }
    }
    return out;
}
