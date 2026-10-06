// js/crystallography/sg-context.js
// Space-group MC: resolution, intensity and Wilson context, indexing statistics.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// ============================================================================
// RESOLUTION GROUPS AND THE OBSERVABLE MERGE
// ============================================================================

// The q-space matching tolerance as a function of q, so the resolution of the
// experiment can be evaluated at CALCULATED line positions and not only at
// observed peaks. Algebraically identical to get_q_tolerance():
//   tol = (2 sin(2th)/lambda^2) * d(2th),
//   sin(2th) = lambda*sqrt(q)*sqrt(1 - lambda^2 q/4)
function qToleranceAtQ(q, wavelength, tth_error) {
    if (!(q > 0)) return 1e-9;
    const lam2 = wavelength * wavelength;
    const arg = Math.max(0, 1 - lam2 * q / 4);
    const sin2th = wavelength * Math.sqrt(q) * Math.sqrt(arg);
    const dtth = tth_error * Math.PI / 360;   // half of tth_error, in radians
    return (2 * sin2th / lam2) * dtth + 1e-9;
}
// COMPLETE-linkage clustering of a sorted q list at the matching tolerance. Two
// calculated lines closer together than the tolerance can never be told apart
// by an observed peak, so they form one resolution group and the pattern only
// records whether the GROUP is populated. Everything downstream -- the merge,
// the line count, the informative-absence count -- is expressed in groups
// rather than raw reflections for that reason.
//
// SINGLE linkage was wrong for this. It only asks whether each new line is
// within tolerance of its immediate predecessor, so in a dense high-angle
// region a chain of lines each just inside the tolerance of the next merges
// into one group many tolerances wide. The two ends of such a group are
// perfectly resolvable from each other, and collapsing them costs evidence
// twice over: nInformative falls because several distinct forbidden positions
// are counted as one, and nAllowedInRange falls the same way, which biases p.
// Worse, a single peak anywhere in the chain marks the whole width as
// populated, so a genuine absence at the far end is silently forgiven.
//
// Requiring the new line to lie within tolerance of the FIRST line of the group
// as well caps every group at one tolerance wide, which is exactly the
// statement "no observed peak could tell these apart". Because the list is
// sorted, that first-to-last test is complete linkage: it bounds the maximum
// pairwise separation, not just the nearest-neighbour one.
function sgResolutionGroups(qSorted, wavelength, tth_error) {
    const n = qSorted.length;
    const groupOf = new Int32Array(n);
    if (n === 0) return { groupOf, centres: new Float64Array(0), count: 0 };
    const centres = [];
    let g = 0, sum = qSorted[0], cnt = 1, q0 = qSorted[0];
    for (let i = 1; i < n; i++) {
        const tol = qToleranceAtQ(qSorted[i], wavelength, tth_error);
        if (qSorted[i] - qSorted[i - 1] <= tol && qSorted[i] - q0 <= tol) {
            groupOf[i] = g; sum += qSorted[i]; cnt++;
        } else {
            centres.push(sum / cnt);
            g++; groupOf[i] = g; sum = qSorted[i]; cnt = 1; q0 = qSorted[i];
        }
    }
    centres.push(sum / cnt);
    return { groupOf, centres: Float64Array.from(centres), count: centres.length };
}
// The frame every class is first compared in: the FULL (unrestricted) line list
// for the parent cell, collapsed into resolution groups. Built once per scan.
function sgLatticeFrame(cell, data, state) {
    const wl = data.wavelength;
    let refl;
    try {
        setSpaceGroupFilter(null);
        refl = generateHKL_for_worker(cell, state.q_max, state.d_min, wl);
    } finally { setSpaceGroupFilter(null); }
    if (!refl || !refl.length) return null;

    refl = refl.slice().sort((a, b) => a.q - b.q);
    const qs = Float64Array.from(refl.map(r => r.q));
    const grp = sgResolutionGroups(qs, wl, data.tth_error);

    const w = sgMeasuredWindow(data, state);
    return { refl, qs, groupOf: grp.groupOf, centres: grp.centres, nGroups: grp.count,
             qLo: w.qLo, qHi: w.qHi };
}
// The window inside which an absence is EVIDENCE.
//
// This is the SCANNED range, not the range spanned by the observed peaks. The
// difference decides cases like Pa-3: for pyrite the first allowed reflection is
// 111 at 28.5 deg, while the conditions forbid 100 at 16.3 and 110 at 23.2. If
// the window starts at the first peak, those two absences fall outside it and
// contribute nothing -- the most diagnostic part of the pattern is discarded,
// and Pa-3 cannot be separated from P. If the window starts where the
// diffractometer started, a gap between the scan start and the first peak is
// exactly what it looks like: two lines that should have been there and were
// not.
//
// data.tth_min / data.tth_max are the user's own 2-theta limits. When a caller
// does not supply them the observed span is used, which is the old, conservative
// behaviour.
function sgMeasuredWindow(data, state) {
    const wl = data.wavelength;
    const qOf = (tth) => {
        const st = Math.sin(tth * RAD / 2);
        return (4 * st * st) / (wl * wl);
    };
    let qLo = Infinity, qHi = -Infinity;
    for (const p of state.peaks_sorted_by_q) {
        if (!isFinite(p.q)) continue;
        if (p.q < qLo) qLo = p.q;
        if (p.q > qHi) qHi = p.q;
    }
    if (!isFinite(qLo)) return { qLo: -Infinity, qHi: Infinity };
    qLo *= 0.999; qHi *= 1.001;

    const tLo = Number(data.tth_min), tHi = Number(data.tth_max);
    if (isFinite(tLo) && tLo > 0) {
        const q = qOf(tLo);
        if (isFinite(q) && q < qLo) qLo = q;
    }
    if (isFinite(tHi) && tHi > 0) {
        const q = qOf(tHi);
        if (isFinite(q) && q > qHi) qHi = q;
    }
    return { qLo, qHi };
}
// Which resolution groups survive a rule set? A group survives if ANY line in
// it is allowed -- that is what a powder pattern shows.
function sgGroupMask(frame, allowed) {
    const mask = new Uint8Array(frame.nGroups);
    for (let i = 0; i < frame.refl.length; i++) {
        const g = frame.groupOf[i];
        if (mask[g]) continue;
        const r = frame.refl[i];
        if (allowed(r.h, r.k, r.l)) mask[g] = 1;
    }
    // Outside the measured window nothing is observable, so nothing there may
    // distinguish two rule sets. Clearing those bits is what makes the merge
    // agree with the score.
    const lo = frame.qLo, hi = frame.qHi;
    if (isFinite(lo) && isFinite(hi)) {
        for (let g = 0; g < frame.nGroups; g++) {
            const q = frame.centres[g];
            if (q < lo || q > hi) mask[g] = 0;
        }
    }
    return mask;
}
// Merge abstract classes whose OBSERVABLE pattern is identical for this cell.
// The surviving representative is the one with the fewest stated rules: the
// merged classes are indistinguishable here, and if the Monte-Carlo walk moves
// the cell far enough for them to diverge, erring toward the permissive member
// keeps a real line from being deleted. (A false absence destroys M20; a missed
// absence only leaves two rows tied, which is what the merge already asserts.)
function sgObservableMerge(classes, frame) {
    const byMask = new Map();
    for (const cls of classes) {
        const mask = sgGroupMask(frame, cls.allowed);
        cls.nAllowedGroups = 0;
        for (let i = 0; i < mask.length; i++) if (mask[i]) cls.nAllowedGroups++;
        const key = mask.join('');        // exact, no hashing: a few dozen keys
        const bucket = byMask.get(key);
        if (bucket) bucket.push(cls); else byMask.set(key, [cls]);
    }

    const out = [];
    for (const bucket of byMask.values()) {
        bucket.sort((a, b) => (a.nRules - b.nRules) ||
                              ((a.members?.[0]?.number || 999) - (b.members?.[0]?.number || 999)));
        const rep = bucket[0];
        rep.mergedFrom = bucket.length;
        if (bucket.length > 1) {
            const seen = new Set(rep.members.map(m => m.number + '|' + m.symbol));
            const labels = new Set([rep.label]);
            const conds = new Set(rep.conditions);
            for (let i = 1; i < bucket.length; i++) {
                for (const m of bucket[i].members) {
                    const k = m.number + '|' + m.symbol;
                    if (!seen.has(k)) { seen.add(k); rep.members.push(m); }
                }
                labels.add(bucket[i].label);
                for (const c of bucket[i].conditions) conds.add(c);
            }
            rep.members.sort((a, b) => (a.number - b.number) || a.symbol.localeCompare(b.symbol));
            rep.centric = rep.members.every(m => m.centric);
            rep.mergedLabels = Array.from(labels).sort();
            rep.allConditions = Array.from(conds).sort();
            if (rep.mergedLabels.length > 1) rep.label = rep.mergedLabels.join(' \u2261 ');
        } else {
            rep.mergedLabels = [rep.label];
            rep.allConditions = rep.conditions;
        }
        out.push(rep);
    }
    return out;
}
// ============================================================================
// INTENSITY CONTEXT
// ============================================================================
//
// A violation on the strongest peak in the pattern and a violation on a 0.4%
// shoulder are not the same evidence; the old code counted them identically.
// Relative intensity is measured against a LOCAL baseline for the same reason
// analyzeSystematicAbsences() does: diffracted intensity falls off with angle,
// so a fixed fraction of the GLOBAL maximum quietly strips the whole
// back-reflection region of any power to falsify a rule. The threshold and the
// window match that function exactly so the two analyses cannot disagree about
// which peaks are weak.
const SG_LOW_INTENSITY_FRACTION = 0.05;
const SG_LOCAL_WINDOW_DEG = 15.0;
function sgIntensityContext(peaks) {
    const heighted = (peaks || []).filter(
        p => typeof p.height === 'number' && isFinite(p.height) && p.height > 0);
    let globalMax = 0;
    for (const p of heighted) if (p.height > globalMax) globalMax = p.height;

    const cache = new Map();
    const localMaxAt = (tth) => {
        let m = cache.get(tth);
        if (m !== undefined) return m;
        m = 0;
        for (const p of heighted) {
            if (Math.abs(p.tth - tth) <= SG_LOCAL_WINDOW_DEG && p.height > m) m = p.height;
        }
        if (!(m > 0)) m = globalMax;
        cache.set(tth, m);
        return m;
    };

    // null means "no height information" -- every consumer must treat that as
    // "cannot judge", never as "weak".
    const relI = (p) => {
        if (globalMax <= 0) return null;
        if (!(typeof p.height === 'number' && isFinite(p.height))) return null;
        const scale = localMaxAt(p.tth);
        return scale > 0 ? (p.height / scale) : null;
    };
    const isWeak = (p) => {
        const r = relI(p);
        return (r !== null) && (r < SG_LOW_INTENSITY_FRACTION);
    };
    return { relI, isWeak, haveHeights: globalMax > 0 };
}
// ============================================================================
// WILSON STATISTICS FROM PEAK HEIGHTS
// ============================================================================
//
// A limited, honest borrowing from the Bayesian extinction-symbol method of
// Markvardsen, David, Johnston & Shankland (Acta Cryst A57 (2001) 47).
//
// WHAT THAT METHOD DOES, AND WHY WE CANNOT DO ALL OF IT. ExtSym works from
// Pawley-extracted integrated intensities together with their full covariance
// matrix, which is what lets it ask the sharp question: given the overlaps, what
// is the intensity AT this forbidden position? Brutus has a peak list, not a
// profile. Where no peak was picked there is no measurement at all -- only
// absence. So the sharp question is out of reach, and anything claiming
// otherwise from this data would be pretending.
//
// WHAT IS REACHABLE, AND IS WORTH HAVING. The weakest part of the scoring is the
// single global number p = P(an allowed line is observed). It is the same for a
// strong low-angle reflection with multiplicity 24 and a weak high-angle one with
// multiplicity 2, which is plainly wrong: the first would have been seen if it
// were there, the second might well not. Wilson statistics turn that one number
// into a per-reflection probability, using exactly the information a peak list
// does carry -- position, height, and the multiplicity the lattice implies.
//
//   1. Correct each observed height to a quantity proportional to |F|^2 by
//      dividing out the Lorentz-polarisation factor and the multiplicity.
//   2. Fit the Wilson plot, ln<I/(m*Lp)> = ln K - 2B s^2 with s = sin(theta)/lambda,
//      giving the overall scale K and temperature factor B.
//   3. Take the weakest peak actually picked as the detection limit, and convert
//      it at each position into the |E|^2 a reflection there would need in order
//      to have been seen: z_min = I_min / (m * Lp * K * exp(-2B s^2)).
//   4. p at that position is then the Wilson tail probability of exceeding it:
//      exp(-z) for an acentric distribution, erfc(sqrt(z/2)) for a centric one.
//      The database's centrosymmetric flag says which applies.
//
// PEAK HEIGHTS ARE NOT INTEGRATED INTENSITIES, and that matters less than it
// first appears. The ratio between them is the peak width, which varies smoothly
// and monotonically with angle; a smooth monotonic error in the intensities is
// absorbed almost entirely into the fitted B. The fitted B is therefore NOT a
// physically meaningful temperature factor and is not reported as one. What
// survives the fit is the normalised |E|^2 scale, which is what the statistics
// actually use. Overlapped peaks remain a genuine error: their height is shared
// between reflections the peak list cannot separate.
//
// The context is built ONCE per scan from the unfiltered lattice, never per
// class. Letting each hypothesis fit its own Wilson scale would let it rescale
// the value of its own evidence -- the same failure that made M20 useless for
// ranking, in a new costume.

// Complementary error function, Numerical Recipes erfcc. |error| < 1.2e-7.
function sgErfc(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
              t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
              t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
}
// THE INTENSITY WEIGHT OF A POWDER LINE
//
// The expected line intensity is
//     <I> = Lp * SUM over the coincident reflections of <|F|^2>
//         = Lp * SUM epsilon(h) * sum_j f_j^2
// and by orbit-stabiliser |orbit| * epsilon = |G|. So summing epsilon over a
// full orbit gives |G| REGARDLESS OF THE REFLECTION: at a given resolution the
// mean line intensity does not depend on multiplicity at all. High-multiplicity
// general lines have many modest contributions, low-multiplicity special lines
// have few, each epsilon times stronger, and the two balance exactly. That is
// Wilson's result, and it is not the intuitive answer.
//
// Weighting by multiplicity alone -- which this file used to do -- therefore
// expects a cubic {h00} line (m = 6) to be an EIGHTH the strength of a {hkl}
// line (m = 48) at the same angle, when their means are equal. z_min at h00 is
// inflated eightfold, p comes out far too low, and a clean absence there earns
// almost no credit. On the axial reflections. Which are the ones that decide
// screw axes. Tetragonal is up to 4x, orthorhombic and monoclinic up to 2x.
//
// Computed under the HOLOHEDRY of the crystal system, so it depends only on the
// cell and never on the candidate. That matters: the calibration note below is
// right that a hypothesis able to set its own intensity scale is the M20
// failure in a new costume, and a hypothesis-free weight cannot be gamed.
//
// It is also why centricity stays out, though sgOpsIsCentric exists. An
// extinction class is not a space group: C2, Cm and C2/m share one class with
// three different point groups and three different centricities, so "is this
// class centric" has no answer. The note at the calibration step reached the
// same conclusion from the Pbca/Pbc- failure; the operator view says why.
//
// Set false to restore the old multiplicity weighting for an A/B comparison.
const SG_USE_EPSILON_WEIGHT = true;
const SG_HOLOHEDRY_ORDER = {
    cubic: 48, tetragonal: 16, hexagonal: 24,
    orthorhombic: 8, monoclinic: 4, triclinic: 2,
};
// m * epsilon over the metric orbit. Written out rather than collapsed to the
// constant so the algebra stays visible, and so a resolution group holding two
// coincident families still weighs twice as much as one holding one.
function sgHolohedryWeight(h, k, l, system) {
    const m = sgMultiplicity(h, k, l, system);
    const H = SG_HOLOHEDRY_ORDER[system] || 2;
    return m * (H / m);
}
// Powder multiplicity: how many distinct hkl share this d-spacing by symmetry.
// The generator emits one representative per folded orbit, so this is the size of
// that orbit after removing duplicates.
function sgMultiplicity(h, k, l, system) {
    const orb = sgEquivalents(h, k, l, system);
    const seen = new Set();
    for (let i = 0; i < orb.length; i++) seen.add(orb[i][0] + ',' + orb[i][1] + ',' + orb[i][2]);
    return seen.size || 1;
}
// Lorentz-polarisation for flat-plate Bragg-Brentano with an unpolarised source.
function sgLorentzPol(tthDeg) {
    const th = tthDeg * RAD / 2;
    const st = Math.sin(th), ct = Math.cos(th);
    const c2 = Math.cos(tthDeg * RAD);
    const d = st * st * ct;
    if (!(Math.abs(d) > 1e-9)) return null;
    return (1 + c2 * c2) / d;
}
const SG_WILSON_MIN_PEAKS  = 12;    // below this there are not enough shells
const SG_WILSON_MIN_SHELLS = 3;     // fewer than three and there is no curve
// Build the Wilson context from the observed peaks and the UNFILTERED line list.
// Returns null whenever the data cannot support it, in which case every caller
// falls back to the single global p and behaves exactly as before.
function sgWilsonContext(frame, data, state, system) {
    const wl = data.wavelength;
    if (!frame || !frame.refl || !frame.refl.length) return null;

    // multiplicity and Lp per resolution group
    // Intensity weight per resolution group. m * epsilon, not m -- see the note
    // on sgHolohedryWeight for why multiplicity alone under-values exactly the
    // axial reflections that carry the screw-axis evidence.
    const grpMult = new Float64Array(frame.nGroups);
    for (let i = 0; i < frame.refl.length; i++) {
        const r = frame.refl[i];
        grpMult[frame.groupOf[i]] += SG_USE_EPSILON_WEIGHT
            ? sgHolohedryWeight(r.h, r.k, r.l, system)
            : sgMultiplicity(r.h, r.k, r.l, system);
    }

    // pair observed peaks with groups, and correct their heights
    const pts = [];
    let minHeight = Infinity;
    for (const p of state.peaks_sorted_by_q) {
        const hgt = p.height;
        if (!(typeof hgt === 'number' && isFinite(hgt) && hgt > 0)) continue;
        const j = binarySearchClosest(frame.qs, p.q);
        const tol = qToleranceAtQ(p.q, wl, data.tth_error);
        if (!(Math.abs(frame.qs[j] - p.q) < tol)) continue;      // unindexed: keep it out of the fit
        const g = frame.groupOf[j];
        const m = grpMult[g] || 1;
        const lp = sgLorentzPol(p.tth);
        if (!lp) continue;
        pts.push({ s2: p.q / 4, h: hgt, mlp: m * lp, q: p.q });
        if (hgt < minHeight) minHeight = hgt;
    }
    if (pts.length < SG_WILSON_MIN_PEAKS || !isFinite(minHeight)) return null;

    // BIN IN SHELLS OF s^2 BEFORE FITTING. This is not a refinement, it is the
    // definition: the Wilson relation holds for the MEAN intensity in a shell,
    // <I> = K exp(-2B s^2). Individual reflections scatter enormously about it --
    // for an acentric structure |F|^2 is exponentially distributed, so its
    // standard deviation equals its mean, and a centric one is worse. Fitting
    // reflection by reflection gives a hopeless correlation and the quality gate
    // below (correctly) throws every fit away. Averaging the shells first is what
    // makes the plot linear.
    //
    // The fit uses ln<I>, not <ln I>: the mean of the logarithm is biased low by
    // roughly Euler's constant for an exponential variate, and the bias enters
    // the scale K directly.
    const NB = Math.max(4, Math.min(12, Math.floor(pts.length / 8)));
    let s2min = Infinity, s2max = -Infinity;
    for (const t of pts) { if (t.s2 < s2min) s2min = t.s2; if (t.s2 > s2max) s2max = t.s2; }
    if (!(s2max > s2min)) return null;
    //
    // TRUNCATION CORRECTION. A peak list contains only what rose above the
    // detection limit, and that limit bites hardest exactly where intensities are
    // weakest -- at high angle. The surviving shell means are therefore biased
    // upward by more and more as s^2 grows, the plot flattens, and B comes out
    // far too small. Measured on synthetic data with a known B this cost roughly
    // 2 A^2 and rejected the fit outright whenever the true falloff was gentle.
    //
    // For an exponentially distributed variate observed only above a threshold t,
    // E[y | y > t] = t + mu. Subtracting each reflection's own threshold before
    // averaging therefore recovers mu without needing to know how many
    // reflections were lost -- which is fortunate, because we cannot know that.
    // The threshold is constant in RAW height (it is a property of the pattern,
    // not of the reflection), so in corrected units it is t_i = T/(m_i Lp_i) and
    // the correction is simply (h_i - T)/(m_i Lp_i).
    //
    // This is exact for the acentric case and approximate for the centric one,
    // whose distribution has a heavier weak tail.
    const sum = new Float64Array(NB), cnt = new Float64Array(NB), mid = new Float64Array(NB);
    for (const t of pts) {
        let b = Math.floor((t.s2 - s2min) / (s2max - s2min) * NB);
        if (b >= NB) b = NB - 1; if (b < 0) b = 0;
        sum[b] += (t.h - minHeight) / t.mlp; cnt[b]++; mid[b] += t.s2;
    }
    const shells = [];
    for (let b = 0; b < NB; b++) {
        if (cnt[b] < 2) continue;                       // a shell of one is noise
        const mu = sum[b] / cnt[b];
        if (!(mu > 0)) continue;                        // whole shell sat at the limit
        shells.push({ s2: mid[b] / cnt[b], y: Math.log(mu) });
    }
    if (shells.length < SG_WILSON_MIN_SHELLS) return null;

    // least squares on ln<I/(m Lp)> = lnK - 2B s^2
    let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (const t of shells) { sx += t.s2; sy += t.y; sxx += t.s2 * t.s2; sxy += t.s2 * t.y; syy += t.y * t.y; }
    const n = shells.length;
    const den = n * sxx - sx * sx;
    if (!(Math.abs(den) > 1e-12)) return null;
    const slope = (n * sxy - sx * sy) / den;
    const inter = (sy - slope * sx) / n;
    const rNum = n * sxy - sx * sy;
    const rDen = Math.sqrt(Math.max(1e-30, (n * sxx - sx * sx) * (n * syy - sy * sy)));
    const r2 = (rNum / rDen) * (rNum / rDen);
    if (!isFinite(slope) || !isFinite(inter)) return null;

    const B = -slope / 2;                 // apparent, not physical -- see the note above
    const K = Math.exp(inter);

    // USE THE SHELLS THEMSELVES, NOT THE STRAIGHT LINE.
    //
    // What the scoring needs is <I>(q), the mean corrected intensity at a given
    // position. The Wilson straight line is one model of that, and requiring the
    // data to obey it was a mistake: a structure with a gentle falloff produces a
    // nearly flat plot, the correlation is poor, and a hard R^2 gate then threw
    // away a perfectly usable intensity scale for no reason (measured: B = 0.5
    // and B = 1.5 were both rejected outright while B = 3 and B = 5 passed).
    //
    // Interpolating ln<I> between the shell means needs no model at all and is
    // exactly as good wherever there are shells. The fitted slope is kept only to
    // extrapolate past the ends, and B and R^2 are reported as diagnostics rather
    // than used as gates.
    shells.sort((a, b) => a.s2 - b.s2);
    const first = shells[0], last = shells[shells.length - 1];
    const meanF2 = (q) => {
        const x = q / 4;
        if (x <= first.s2) return Math.exp(first.y + slope * (x - first.s2));
        if (x >= last.s2)  return Math.exp(last.y  + slope * (x - last.s2));
        for (let i = 1; i < shells.length; i++) {
            if (x <= shells[i].s2) {
                const a = shells[i - 1], b = shells[i];
                const f = (b.s2 > a.s2) ? (x - a.s2) / (b.s2 - a.s2) : 0;
                return Math.exp(a.y + f * (b.y - a.y));
            }
        }
        return Math.exp(last.y);
    };
    const zMin = (q, mult) => {
        const tth = 2 * Math.asin(Math.min(1, wl * Math.sqrt(Math.max(0, q)) / 2)) * DEG;
        const lp = sgLorentzPol(tth);
        const mf = meanF2(q);
        if (!lp || !(mf > 0) || !(mult > 0)) return null;
        return (minHeight / (mult * lp)) / mf;
    };

    // ------------------------------------------------------------------
    // CALIBRATION: Wilson supplies the SHAPE, the data supply the LEVEL.
    //
    // Two things went wrong when the raw Wilson probability was used directly.
    //
    // First, choosing the centric or acentric distribution per class made the
    // ranking swing on a property a powder pattern cannot see. A centrosymmetric
    // class got the centric distribution, which has more weak reflections, so a
    // lower p and less credit per absence; a class that merely happened to
    // contain one acentric member got the acentric distribution and more credit.
    // Pbca lost to Pbc- on real synthetic Pbca data for precisely this reason --
    // fewer clean absences, higher score. One distribution is now used for every
    // row, so the choice cancels out of every comparison.
    //
    // Second, the absolute level was too high. Measured against synthetic data
    // where 62% of lines were detectable, the raw Wilson p averaged about 80%,
    // which inflated each absence from 0.94 nats to 1.59. Multiplied over a
    // hundred forbidden positions that is enough for a class with 28 hard
    // violations to outscore one with none -- the over-restriction failure this
    // whole scoring scheme exists to prevent, returning by another route.
    //
    // The fix keeps what Wilson is genuinely good for -- knowing that a strong
    // low-angle reflection of high multiplicity would have been seen while a weak
    // high-angle one might not -- and takes the overall level from the observed
    // detection rate instead. A single scale factor on the threshold is solved by
    // bisection so that the mean predicted detectability over the lattice matches
    // the fraction of positions that actually carry a peak.
    // ------------------------------------------------------------------
    // THE REFERENCE SET AND THE TARGET RATE MUST MATCH.
    //
    // What follows is only the SEED calibration, and it is deliberately the same
    // hypothesis as pass 1 of sgRescoreAll(): the unfiltered lattice, i.e. "no
    // extinctions at all". That reference is biased low for exactly the reason
    // pass 1 is -- if the true lattice is centred, every systematically absent
    // position sits in the denominator and can never be observed, so the
    // detection rate looks worse than it is.
    //
    // The bias used to be permanent, because lambda was solved once here and
    // never revisited while p-hat was re-estimated from the winner. The two
    // levels then disagreed: absences were weighted by a detectability pinned to
    // the no-extinction lattice and compared against a p measured on the
    // winner's own allowed positions. solveLambda() below is exported so the
    // scoring can redo this calibration against whatever reference set it is
    // using at the time, which is what keeps the two consistent.
    const inWinQ = [], inWinM = [];
    for (let g = 0; g < frame.nGroups; g++) {
        const qc = frame.centres[g];
        if (qc < frame.qLo || qc > frame.qHi) continue;
        inWinQ.push(qc); inWinM.push(grpMult[g] || 1);
    }
    if (!inWinQ.length) return null;
    // centres are ascending, so the in-window slice is too and a peak can be
    // placed by bisection instead of by scanning every group (this was O(pts x
    // groups), with a half-finished binary search left declared beside it).
    const inWinQArr = Float64Array.from(inWinQ);
    const seen = new Uint8Array(inWinQArr.length);
    for (const t of pts) {
        const i = binarySearchClosest(inWinQArr, t.q);
        if (i >= 0 && Math.abs(inWinQArr[i] - t.q) < qToleranceAtQ(t.q, wl, data.tth_error)) seen[i] = 1;
    }
    let nSeen = 0;
    for (let i = 0; i < seen.length; i++) nSeen += seen[i];
    const pEmpirical = Math.min(0.98, Math.max(0.02, nSeen / inWinQArr.length));

    // z is the detection threshold at a position in units of the local mean
    // intensity: it depends on the pattern and the position, never on lambda.
    // Every calibration is therefore just a rescaling of a fixed list of z.
    const zList = [];
    for (let i = 0; i < inWinQArr.length; i++) {
        const z = zMin(inWinQArr[i], inWinM[i]);
        if (z !== null && isFinite(z)) zList.push(Math.max(0, z));
    }

    // Solve mean(exp(-z*lambda)) = target over a supplied list of z. Monotone
    // decreasing in lambda, so plain bisection in log-lambda converges.
    const meanPOf = (zs, lam) => {
        if (!zs || !zs.length) return null;
        let acc = 0;
        for (let i = 0; i < zs.length; i++) acc += Math.exp(-zs[i] * lam);
        return acc / zs.length;
    };
    const solveLambda = (zs, target) => {
        const tgt = Math.min(0.98, Math.max(0.02, target));
        if (!zs || !zs.length || !isFinite(tgt)) return null;
        let lamLo = 1e-3, lamHi = 1e3, lam = 1;
        for (let it = 0; it < 60; it++) {
            lam = Math.sqrt(lamLo * lamHi);
            const mp = meanPOf(zs, lam);
            if (mp === null) return null;
            if (mp > tgt) lamLo = lam; else lamHi = lam;
        }
        return lam;
    };
    const lambda = solveLambda(zList, pEmpirical) ?? 1;
    const clampP = (v) => Math.min(0.999, Math.max(0.001, v));
    const rawP = (q, mult, lam) => {
        const z = zMin(q, mult);
        if (z === null || !isFinite(z)) return null;
        return Math.exp(-Math.max(0, z) * lam);
    };

    return {
        B, K, r2, nUsed: pts.length, nShells: n, minHeight,
        pEmpirical, lambda, calibrated: meanPOf(zList, lambda),
        groupMultiplicity: (g) => grpMult[g] || 1,
        // The lambda-free part of the detectability: exported so callers can
        // store z once and re-weight later under a recalibrated lambda.
        zAt: (q, mult) => {
            const z = zMin(q, mult);
            return (z === null || !isFinite(z)) ? null : Math.max(0, z);
        },
        pFromZ: (z, lam) => {
            if (z === null || z === undefined || !isFinite(z)) return null;
            const l = (lam !== null && lam !== undefined && isFinite(lam)) ? lam : lambda;
            return clampP(Math.exp(-Math.max(0, z) * l));
        },
        solveLambda,
        meanPOf,
        // Probability that a reflection PRESENT at this position would have been
        // detected. The SHAPE across positions is Wilson's; the overall level is
        // pinned to the observed detection rate by `lambda`.
        pDetect: (q, mult) => {
            const v = rawP(q, mult, lambda);
            if (v === null) return null;
            return clampP(v);
        },
        // observed |E|^2 of a peak: how strong is it, in units of the local mean
        eSquared: (q, height, mult) => {
            if (!(height > 0) || !(mult > 0)) return null;
            const tth = 2 * Math.asin(Math.min(1, wl * Math.sqrt(Math.max(0, q)) / 2)) * DEG;
            const lp = sgLorentzPol(tth);
            const mf = meanF2(q);
            if (!lp || !(mf > 0)) return null;
            return (height / (mult * lp)) / mf;
        },
    };
}
// ============================================================================
// PER-CLASS INDEXING STATISTICS
// ============================================================================
//
//   indexed    - the peak is matched to an ALLOWED line
//   violation  - the peak is matched only to a FORBIDDEN line. Direct evidence
//                against the rule set, graded by how believable the peak is.
//   unindexed  - no line at all within tolerance. Historically excluded from the
//                ranking on the grounds that "every class carries it equally";
//                that stopped being true the moment each class got its own
//                refined cell, so it now enters the score.
//   clean      - a forbidden resolution group that is INFORMATIVE (contains no
//                allowed line, so its emptiness is attributable) and where no
//                peak was observed. This is the POSITIVE evidence for the rule
//                set, and the old version never counted it at all.
//
// Peak-to-line matching is INJECTIVE: one calculated line serves at most one
// observed peak, closest pair first. That is the rule pair_and_fit() and
// _swapCheapFit() already use. Without it a class with a dense line list can
// "index" thirty peaks onto ten lines and be flattered for it.
//
// Allowed lines are offered first, so a peak that could sit on either an allowed
// or a forbidden line is credited to the hypothesis rather than counted against
// it. Only what is left over can become a violation.
function sgIndexingStats(cell, data, state, allowed, ictx, preRefl, wilson) {
    const wl = data.wavelength;
    const z = cell.zero_correction || 0;
    const tolFn = (idx) => get_q_tolerance(idx, state.tth_obs_rad, wl, data.tth_error);

    // TWO generation passes, and it has to be two.
    //
    // generateHKL_for_analysis() DEDUPES reflections that share a 2-theta, keeping
    // one arbitrary representative. Generating the full list once and labelling
    // each survivor allowed/forbidden therefore misclassifies every q where an
    // allowed and a forbidden reflection coincide: if the dedupe happened to keep
    // the forbidden one, the line is recorded as forbidden and any peak sitting
    // on it becomes a violation against the correct rule set.
    //
    // Pa-3 is the worked example. 330 is forbidden (hk0: h=2n) and 411 is
    // allowed, and in a cubic cell they sit at exactly the same q (N = 18). The
    // single-pass version kept 330, so pyrite's own pattern produced a hard
    // violation against Pa-3 at 74.2 deg.
    //
    // Generating the allowed list WITH the filter installed makes the dedupe
    // happen among allowed reflections only, which is what the original
    // two-array version did correctly.
    let reflAll = preRefl;
    if (!reflAll) {
        try {
            setSpaceGroupFilter(null);
            reflAll = generateHKL_for_worker(cell, state.q_max, state.d_min, wl);
        } finally { setSpaceGroupFilter(null); }
        if (!reflAll || !reflAll.length) return null;
        reflAll = reflAll.slice().sort((a, b) => a.q - b.q);
    }
    let reflOk;
    try {
        setSpaceGroupFilter(allowed);
        reflOk = generateHKL_for_worker(cell, state.q_max, state.d_min, wl);
    } finally { setSpaceGroupFilter(null); }
    reflOk = (reflOk || []).slice().sort((a, b) => a.q - b.q);

    // Merge into one position list, marking which positions carry an allowed
    // line. A position counts as allowed when an allowed reflection falls within
    // the matching tolerance of it.
    const nL = reflAll.length;
    if (!nL) return null;
    const allowedQ = reflOk.map(r => r.q);
    const qs = new Float64Array(nL);
    const isAllowed = new Uint8Array(nL);
    for (let j = 0; j < nL; j++) {
        qs[j] = reflAll[j].q;
        if (allowedQ.length) {
            const i = binarySearchClosest(allowedQ, qs[j]);
            isAllowed[j] = Math.abs(allowedQ[i] - qs[j]) <= qToleranceAtQ(qs[j], wl, data.tth_error) ? 1 : 0;
        }
    }
    const refl = reflAll;

    const grp = sgResolutionGroups(qs, wl, data.tth_error);
    const nG = grp.count;

    // --- observed peaks in q, zero-corrected ---------------------------------
    const obs = [];
    for (const p of state.peaks_sorted_by_q) {
        const tc = p.tth - z;
        if (!isFinite(tc) || tc <= 0 || tc >= 180) continue;
        const st = Math.sin(tc * RAD / 2);
        const q = (4 * st * st) / (wl * wl);
        if (!isFinite(q)) continue;
        obs.push({ p, q, tol: tolFn(p.original_index), assigned: false, line: -1, dq: 0 });
    }
    if (!obs.length) return null;

    // --- injective assignment ------------------------------------------------
    const pairUp = (wantAllowed) => {
        const cand = [];
        for (let i = 0; i < obs.length; i++) {
            const o = obs[i];
            if (o.assigned) continue;
            const start = binarySearchClosest(qs, o.q);
            // walk outward from the nearest line until BOTH sides leave the window
            for (let d = 0; ; d++) {
                const a = start - d, b = start + d;
                const aIn = a >= 0 && Math.abs(qs[a] - o.q) < o.tol;
                const bIn = b < nL && Math.abs(qs[b] - o.q) < o.tol;
                if (aIn && isAllowed[a] === wantAllowed) cand.push({ i, j: a, dq: Math.abs(qs[a] - o.q) });
                if (d > 0 && bIn && isAllowed[b] === wantAllowed) cand.push({ i, j: b, dq: Math.abs(qs[b] - o.q) });
                const aDead = (a < 0) || !aIn;
                const bDead = (b >= nL) || !bIn;
                if (aDead && bDead) break;
            }
        }
        cand.sort((x, y) => x.dq - y.dq);
        const takenLine = new Set();
        for (const c of cand) {
            if (obs[c.i].assigned || takenLine.has(c.j)) continue;
            obs[c.i].assigned = true;
            obs[c.i].line = c.j;
            obs[c.i].dq = c.dq;
            takenLine.add(c.j);
        }
    };

    pairUp(1);   // allowed lines first
    pairUp(0);   // then whatever forbidden lines explain the leftovers

    // --- tallies -------------------------------------------------------------
    const nearAllowed = (q, tol) => {
        if (!allowedQ.length) return false;
        const j = binarySearchClosest(allowedQ, q);
        return Math.abs(allowedQ[j] - q) < tol * 1.5;
    };

    let indexed = 0;
    const violations = [];
    const unindexed = [];
    for (const o of obs) {
        if (!o.assigned) { unindexed.push(o); continue; }
        if (isAllowed[o.line]) { indexed++; continue; }
        const gMult = wilson ? wilson.groupMultiplicity(grp.groupOf[o.line]) : null;
        violations.push({
            tth: o.p.tth,
            rel: ictx.relI(o.p),
            // |E|^2 of the offending peak: its intensity in units of the mean at
            // this angle, corrected for Lp and multiplicity. A value near or
            // above 1 is unmistakably a real reflection; well below 0.1 is the
            // sort of thing a tail or a little noise produces.
            eSq: wilson ? wilson.eSquared(o.q, o.p.height, gMult) : null,
            // probability a reflection present here would have been detected,
            // at the seed calibration; zLocal is the same quantity before
            // lambda is applied, so the scoring can re-weight it once lambda
            // has been re-solved against the winner (see sgRescoreAll).
            pLocal: wilson ? wilson.pDetect(o.q, gMult) : null,
            zLocal: (wilson && wilson.zAt) ? wilson.zAt(o.q, gMult) : null,
            dqOverTol: o.tol > 0 ? o.dq / o.tol : null,
            ka2: !!o.p.ka2Suspect,
            weak: ictx.isWeak(o.p),
            // an allowed line sits close enough that the assignment is a
            // judgement call rather than a fact
            ambiguous: nearAllowed(o.q, o.tol),
        });
    }

    // --- informative absences ------------------------------------------------
    // A forbidden group only carries evidence if NO allowed line shares it
    // (otherwise a peak appears there either way) and it lies inside the
    // measured range. Of those, count the ones that are in fact empty.
    const groupHasAllowed = new Uint8Array(nG);
    const groupExists = new Uint8Array(nG);
    for (let j = 0; j < nL; j++) {
        groupExists[grp.groupOf[j]] = 1;
        if (isAllowed[j]) groupHasAllowed[grp.groupOf[j]] = 1;
    }
    const groupObserved = new Uint8Array(nG);
    for (const o of obs) if (o.assigned) groupObserved[grp.groupOf[o.line]] = 1;

    // A peak within tolerance of a forbidden position means that position is NOT
    // empty, whoever ends up owning the peak.
    //
    // The assignment above is injective and offers allowed lines first, which is
    // right for deciding what counts as a violation but wrong for deciding what
    // counts as an ABSENCE. A peak sitting within tolerance of both an allowed
    // line and a forbidden one is credited to the allowed line -- so it raises no
    // violation -- and the forbidden group, having no peak assigned to it, was
    // then also banked as a clean absence. The hypothesis collected positive
    // evidence from a position where a peak demonstrably sits, and collected it
    // precisely in the ambiguous cases where it has earned the least.
    //
    // Only groups with no allowed line of their own are re-marked here, so
    // nAllowedInRange / nAllowedObserved -- and therefore p -- are untouched.
    for (const o of obs) {
        const start = binarySearchClosest(qs, o.q);
        for (let d = 0; ; d++) {
            const a = start - d, b = start + d;
            const aIn = a >= 0 && Math.abs(qs[a] - o.q) < o.tol;
            const bIn = b < nL && Math.abs(qs[b] - o.q) < o.tol;
            if (aIn && !groupHasAllowed[grp.groupOf[a]]) groupObserved[grp.groupOf[a]] = 1;
            if (d > 0 && bIn && !groupHasAllowed[grp.groupOf[b]]) groupObserved[grp.groupOf[b]] = 1;
            if ((a < 0 || !aIn) && (b >= nL || !bIn)) break;
        }
    }

    // Same window as the merge: the scanned range, not the observed span.
    const win = sgMeasuredWindow(data, state);
    const qLo = win.qLo, qHi = win.qHi;

    let nInformative = 0, nClean = 0, nAllowedInRange = 0, nAllowedObserved = 0, nLines = 0;
    const cleanP = [], cleanZ = [], allowedZ = [];
    for (let g = 0; g < nG; g++) {
        if (!groupExists[g]) continue;
        const qc = grp.centres[g];
        // nLines used to be tallied HERE, before the window test, so the
        // displayed line count included groups the experiment never scanned and
        // disagreed with every other count in the row. It is a display column
        // and nothing reads it back, but it should still mean what the header
        // says: resolvable allowed lines inside the measured range.
        if (qc < qLo || qc > qHi) continue;
        if (groupHasAllowed[g]) {
            nLines++;
            nAllowedInRange++;
            if (groupObserved[g]) nAllowedObserved++;
            // The allowed positions are the reference set p is measured on, so
            // they are also the set lambda has to be calibrated against if the
            // two are to describe the same hypothesis.
            if (wilson && wilson.zAt) {
                const za = wilson.zAt(qc, wilson.groupMultiplicity(g));
                if (za !== null) allowedZ.push(za);
            }
        } else {
            nInformative++;
            if (!groupObserved[g]) {
                nClean++;
                // Per-position weight for this absence. A forbidden line that
                // would have been strong and obvious is powerful evidence when
                // it is missing; one that would have been invisible anyway is
                // almost none. That distinction is exactly what a single global
                // p cannot make.
                if (wilson) {
                    const pd = wilson.pDetect(qc, wilson.groupMultiplicity(g));
                    if (pd !== null) cleanP.push(pd);
                    if (wilson.zAt) {
                        const zc = wilson.zAt(qc, wilson.groupMultiplicity(g));
                        if (zc !== null) cleanZ.push(zc);
                    }
                }
            }
        }
    }

    // strongest violations first: those are the ones worth showing
    violations.sort((a, b) => (b.rel ?? 1) - (a.rel ?? 1));

    return {
        indexed,
        violations,
        nViolations: violations.length,
        nHard: violations.filter(v => !v.ka2 && !v.weak && !v.ambiguous).length,
        unindexed: unindexed.length,
        unindexedRel: unindexed.map(o => ictx.relI(o.p)),
        nClean, nInformative, cleanP, cleanZ, allowedZ,
        nAllowedInRange, nAllowedObserved,
        nLines,
        violatingTth: violations.slice(0, 8).map(v => v.tth),
    };
}
