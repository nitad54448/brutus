// js/crystallography/sg-score.js
// Space-group MC: scoring and ranking.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// ============================================================================
// SCORING
// ============================================================================
//
// M20 CANNOT arbitrate between rule sets, and the old ranking used it as if it
// could. M20 = Q20 / (2<|dQ|> N20), where N20 counts the POSSIBLE lines below
// the 20th observed one -- so deleting ANY line raises M20. The previous comment
// argued that an over-restrictive class pays for that in violations, but that
// only holds if every allowed line is observed. Real patterns miss plenty of
// weak ones, so a class could delete unobserved lines, gain M20 for nothing at
// zero violations, and then win the nRules tie-break as well: the same bias
// applied twice.
//
// What replaces it is a likelihood ratio, in the spirit of the Bayesian
// extinction-symbol work (Markvardsen, David, Johnston & Shankland, Acta Cryst
// A57 (2001) 47), reduced to what peak POSITIONS alone can support -- no Wilson
// statistics, because that needs background-subtracted, Lorentz-polarisation
// corrected integrated intensities and we have peak heights.
//
// Let p = P(a symmetry-allowed line inside the measured range yields a
// detectable peak), estimated from the data. For a candidate rule set H:
//
//   * every INFORMATIVE forbidden group that is EMPTY contributes
//         log( (1 - eps_clean) / (1 - p) )
//     Under H the group should be empty bar a spurious-peak rate; under "no
//     extinction" it would have been empty only with probability 1 - p. When p
//     is small -- most allowed lines unobserved anyway -- this is near zero,
//     which is right: absences prove little in a sparse pattern. When p
//     approaches 1, each clean absence is worth several nats. THIS is the term
//     that makes the criterion self-limiting: adding a restriction that deletes
//     an unobserved line buys log(1/(1-p)), not a free M20 increase.
//
//   * every VIOLATION contributes log( eps_i / p ), with eps_i set by how
//     believable that particular peak is. A strong, clean, unambiguous peak on a
//     forbidden line is close to fatal; a Ka2 ghost or a 2%-of-local shoulder is
//     not.
//
//   * every UNINDEXED peak contributes log( eps_unindexed ): the refined cell
//     explains it with nothing at all.
//
//   * a BIC term -k/2 * ln(n_indexed) charges for the refined parameters. k is
//     the same for every row, so this only differentiates through how many lines
//     actually constrain the fit -- which is exactly the asymmetry that let a
//     class with very few allowed lines absorb error into its own zero-point for
//     free.
//
// p is estimated ONCE and shared by every row (see sgRescoreAll). Letting each
// class estimate its own p would let it raise the value of its own evidence by
// discarding lines, which is the M20 failure wearing a different hat.

// The rate at which a peak appears where a rule set says none should be.
//
// The soft categories are CONDITIONAL on an identified alternative explanation
// for that particular peak -- there is a Ka1 parent at the right offset, or an
// allowed line sits inside the window -- so they are genuine probabilities for
// that peak and do not depend on how large the pattern is.
//
// The base rate for a strong, clean, unexplained peak is different in kind: it is
// a rate PER FORBIDDEN POSITION, and it must therefore scale with how many
// positions there are. It used to be pinned at 0.02, which on a pattern with 700
// resolvable positions asserts that fourteen spurious peaks are expected. On a
// real PbSO4 dataset -- 192 peaks, 700 positions, so only 27% of possible lines
// observed -- that let the I-centred class survive NINETEEN hard violations:
// its 219 empty forbidden positions earned +65.8 nats while the violations cost
// only -46.6, so it outscored the correct P2_1/a class, which had 48 clean
// absences and no violations at all. A class that forbids half of reciprocal
// space collects absence credit in proportion to how much it forbids, and at
// eps = 0.02 the violations could not pay it back.
//
// sgBaseEps() derives the rate instead: roughly one unexplained peak across the
// whole pattern. On 700 positions that is 0.0014, making each hard violation
// cost 5.3 nats rather than 2.6, and nineteen of them fatal -- which is the
// textbook rule that a single genuine reflection at a systematically absent
// position rules a space group out. On a small pattern with ~50 positions it
// returns 0.02 and reproduces the old behaviour exactly.
const SG_EPS = {
    weak:       0.25,   // below 5% of the local maximum
    ka2:        0.50,   // Ka2 ghost
    ambiguous:  0.50,   // an allowed line sits inside 1.5x the tolerance
    unindexed:  0.15,   // peak the refined cell explains with nothing at all
    hardFloor:  1e-4,   // never claim more certainty than this
    hardCap:    0.05,   // nor less, however few positions there are
};
// How much total evidence a pile of empty forbidden positions is allowed to be
// worth, as a multiple of the decisive threshold. See the saturation note in
// sgScoreFromStats(): absences are the one term that grows without bound with
// how much a class forbids, and they are also the weakest kind of evidence per
// unit, so they are the term that has to saturate.
//
// TWENTY, not ten. The cap has to sit above the largest separation the absences
// can legitimately produce, or it eats real evidence instead of runaway
// evidence. Measured: two clean classes differing by thirteen informative
// absences at p = 0.9 are 26.9 nats apart, which a ceiling of ten times decisive
// (23 nats) compresses to 0.95 -- an overwhelming result reported as a tie,
// because both rows sat deep in the flat part of the curve. At twenty times
// (46 nats) the same pair reads 8.0 nats and the runaway cases are still bounded:
// the PbSO4 I-centred class collects 346 raw nats of absence credit and still
// cannot buy its way past seventy hard violations.
const SG_CLEAN_CAP_MULT = 20;
// tanh() reaches 1.0 EXACTLY in double precision once its argument passes about
// 19, so a pure saturation makes every heavily-restrictive row score identically
// on absences and the ordering the cap was supposed to preserve is lost after
// all -- silently, and only on the rows most likely to be wrong. A vanishing
// linear term keeps the sequence strictly increasing forever: at 1e-3 nats per
// nat it is a tie-break and nothing more, worth one nat against a thousand.
const SG_CLEAN_LEAK = 1e-3;
function sgBaseEps(nPositions, impurityAllowance) {
    const nPos = Math.max(1, nPositions || 0);
    // The user's impurity setting is their own estimate of how many foreign
    // peaks the pattern carries; one is assumed even when they say none.
    const nStray = Math.max(1, Math.floor(impurityAllowance || 0));
    return Math.min(SG_EPS.hardCap, Math.max(SG_EPS.hardFloor, nStray / nPos));
}
function sgViolationEps(v, epsBase) {
    const base = (epsBase !== undefined && epsBase !== null) ? epsBase : 0.02;
    if (v.ka2)       return SG_EPS.ka2;
    if (v.ambiguous) return SG_EPS.ambiguous;
    if (v.weak)      return SG_EPS.weak;
    // With a Wilson scale the taper runs on |E|^2 rather than on raw local
    // height. That is the better variable: it already accounts for the angle and
    // the multiplicity, so a moderate peak where reflections are weak anyway is
    // correctly read as strong evidence, and a tall one at low angle where
    // everything is tall is not over-credited.
    if (v.eSq !== null && v.eSq !== undefined && isFinite(v.eSq)) {
        const t = Math.min(1, Math.max(0, (v.eSq - 0.05) / (0.5 - 0.05)));
        return SG_EPS.weak + (base - SG_EPS.weak) * t;
    }
    if (v.rel === null || v.rel === undefined) return base;   // no heights: judge on position alone
    // Intensity taper between the weak threshold and "unmistakably strong", so a
    // 6%-of-local peak is not treated as identical to a 100% one.
    const t = Math.min(1, Math.max(0, (v.rel - SG_LOW_INTENSITY_FRACTION) / (0.35 - SG_LOW_INTENSITY_FRACTION)));
    return SG_EPS.weak + (base - SG_EPS.weak) * t;
}
// The impurity allowance is a COUNT, not a list, so it has to be spent
// somewhere. Spend it on the LEAST believable unexplained peaks first (weakest
// relative intensity). The old code subtracted it from the raw violation total,
// which let a class buy forgiveness for the strongest peak in the pattern.
//
// UNINDEXED PEAKS ARE SPENT ON LAST. Sorting purely by relative intensity meant
// the allowance was almost always consumed by weak leftovers before it ever
// reached a violation, because a hard violation is by definition not weak. A
// user who sees three foreign lines and sets the allowance to 3 expects those
// lines forgiven; instead the budget went to three faint unindexed peaks and
// every class stayed falsified. A foreign phase contributes a peak the cell
// cannot index OR a peak on a forbidden line, and only the second kind
// falsifies, so the second kind is where the budget has to be able to go.
// Within each kind the weakest still goes first, so the strongest peak in the
// pattern is still the last thing anyone can buy.
function sgApplyImpurityAllowance(stats, allowance) {
    const n = Math.max(0, Math.floor(allowance || 0));
    if (!n) return { violations: stats.violations, nUnindexed: stats.unindexed };

    const items = [];
    stats.violations.forEach((v, i) => items.push({ kind: 'viol', idx: i, rel: v.rel ?? 1 }));
    stats.unindexedRel.forEach((r, i) => items.push({ kind: 'unidx', idx: i, rel: r ?? 1 }));
    const rank = (it) => (it.kind === 'viol' ? 0 : 1);
    items.sort((a, b) => (rank(a) - rank(b)) || (a.rel - b.rel));

    const dropViol = new Set();
    let dropUnidx = 0;
    for (let i = 0; i < Math.min(n, items.length); i++) {
        if (items[i].kind === 'viol') dropViol.add(items[i].idx); else dropUnidx++;
    }
    return {
        violations: stats.violations.filter((_, i) => !dropViol.has(i)),
        nUnindexed: stats.unindexed - dropUnidx,
    };
}
function sgScoreFromStats(stats, pHat, opts) {
    const o = opts || {};
    const p = Math.min(0.98, Math.max(0.02, pHat));
    const kept = sgApplyImpurityAllowance(stats, o.impurityAllowance);

    // Every resolvable position the lattice offers inside the measured window.
    // Each group in range either carries an allowed line or is an informative
    // absence, so the two counts partition it.
    //
    // MEASURED ON THE PARENT FRAME WHEN THERE IS ONE. This count feeds epsBase,
    // which is the rate of unexplained peaks PER POSITION and is meant to be a
    // property of the pattern, not of the hypothesis. Taking it from each row's
    // own refined cell made it drift slightly from row to row -- second order
    // (about 0.07 nats per violation for a fifty-position difference) but enough
    // that "every row is scored under the same assumptions" was only
    // approximately true. opts.nPositions carries the shared count when the
    // caller has one; the row's own is the fallback.
    const nPosRow = (stats.nAllowedInRange || 0) + (stats.nInformative || 0);
    const nPositions = (isFinite(o.nPositions) && o.nPositions > 0) ? o.nPositions : nPosRow;
    const epsBase = sgBaseEps(nPositions, o.impurityAllowance);

    // Lambda re-solved against the reference class this pass is using, if the
    // caller supplied one; otherwise the seed value baked into pDetect().
    const wil = o.wilson || null;
    const lam = (isFinite(o.lambda) && o.lambda > 0) ? o.lambda : null;
    const pOfZ = (z, fallback) => {
        if (wil && lam !== null && z !== null && z !== undefined && isFinite(z)) {
            const v = wil.pFromZ(z, lam);
            if (v !== null) return v;
        }
        return fallback;
    };

    let score = 0;

    // Clean absences. With a Wilson scale each absence carries its own weight,
    // set by how likely that particular reflection was to be seen at all; without
    // one they all share the single global p.
    //
    // The two paths are mixed PER POSITION, not per row. The old gate was
    // `cleanP.length === nClean`, all-or-nothing: a single position where
    // pDetect() returned null (no Lp, no multiplicity, mean intensity zero)
    // dropped that entire row back onto the global p while its neighbours in
    // the same table stayed on Wilson. Measured on a 300-absence row at
    // p = 0.45 that one missing entry was worth +50 nats -- twenty times the
    // decisive threshold -- purely from being scored under a different model
    // than the row above it. Positions that do have a weight now use it, and
    // only the leftovers fall back.
    const cleanP = (stats.cleanP && stats.cleanP.length) ? stats.cleanP : [];
    const cleanZ = (stats.cleanZ && stats.cleanZ.length === cleanP.length) ? stats.cleanZ : null;
    let clean = 0;
    for (let i = 0; i < cleanP.length; i++) {
        const pc = pOfZ(cleanZ ? cleanZ[i] : null, Math.min(0.999, Math.max(0.001, cleanP[i])));
        clean += Math.log((1 - epsBase) / (1 - pc));
    }

    // THE UNIFORM TERM IS SHRUNK BY THE DETECTABLE FRACTION.
    //
    // log(1/(1-p)) is the credit for an empty position ASSUMING a reflection
    // there would have been detectable with probability p. Applied to every
    // position without a per-position weight, that assumption is wrong in a way
    // that always favours the restrictive class: conditioning on "this position
    // is empty" preferentially selects the positions where nothing would have
    // shown up anyway, whose true credit is near zero, and pays them the
    // population average instead. Only about a fraction p-bar of positions are
    // detectable at all, so the honest per-position expectation is smaller by
    // roughly that factor -- (1 - p_undetectable), which is the mean
    // detectability itself.
    //
    // The factor is shared by every row (it depends only on the shared p and,
    // when Wilson is available, on the shared intensity scale), so it rescales
    // the term without touching the ordering it induces.
    const detFrac = (wil && lam !== null && stats.allowedZ && stats.allowedZ.length)
        ? Math.min(1, Math.max(0.02, wil.meanPOf(stats.allowedZ, lam) ?? p))
        : p;
    const nCleanRest = Math.max(0, (stats.nClean || 0) - cleanP.length);
    clean += nCleanRest * detFrac * Math.log((1 - epsBase) / (1 - p));

    // AND THE TOTAL SATURATES.
    //
    // Absences are the only term that grows with how much a hypothesis forbids
    // rather than with what the pattern actually shows, and the whole history of
    // this module is failures of that shape: I-centring outscoring P2_1/a by
    // burying nineteen hard violations under 219 empty positions. Falsification
    // tiering catches that case, but nothing stopped a merely soft-violating
    // over-restrictive class from doing the same thing more quietly.
    //
    // A hard cap would flatten every row above it into a tie, so this is a
    // smooth saturation instead: c*tanh(x/c) is strictly increasing, so the
    // ordering among rows survives intact; it is within a percent of x while x
    // is small compared with c, and it can never exceed c however many
    // positions a class deletes. Absences can still be decisive many times
    // over -- they simply cannot outvote the peaks that are actually there.
    const cleanCap = SG_CLEAN_CAP_MULT * SG_DECISIVE_NATS;
    const cleanRaw = clean;
    clean = cleanCap * Math.tanh(clean / cleanCap) + SG_CLEAN_LEAK * clean;
    score += clean;

    for (const v of kept.violations) {
        const pv = pOfZ(v.zLocal,
            (v.pLocal !== null && v.pLocal !== undefined && isFinite(v.pLocal))
                ? Math.min(0.999, Math.max(0.001, v.pLocal)) : p);
        // A VIOLATION CAN NEVER BE POSITIVE EVIDENCE FOR THE RULE SET IT BREAKS.
        //
        // The ratio eps/p exceeds one whenever the detectability model says a
        // reflection here would have been invisible: p bottoms out at its 0.001
        // floor, eps for a soft violation is 0.25, and the peak that
        // CONTRADICTS the hypothesis is then worth +5.5 nats IN ITS FAVOUR.
        // Measured on a synthetic I-centred pattern: a deliberately mis-set cell
        // collected seventeen such violations worth +51 nats and beat the cell
        // that indexed the pattern exactly, which had none. The correct reading
        // of p < eps is not "this peak supports H" but "the intensity model is
        // wrong about this position" -- a peak is sitting where the model says
        // nothing could be seen, so the model, not the hypothesis, is what the
        // observation bears on. Clamping at zero says exactly that: at best a
        // violation is uninformative, and it is never support.
        score += Math.min(0, Math.log(sgViolationEps(v, epsBase) / pv));
    }
    score += Math.max(0, kept.nUnindexed) * Math.log(SG_EPS.unindexed);

    const nPar = (MC_NPAR[o.system] || 3) + (o.refineZero ? 1 : 0);

    // MODEL-SELECTION CHARGE.
    //
    // The old term was -0.5*k*ln(n_indexed), and its sign ran the wrong way
    // against its own stated intent. BIC's penalty GROWS with sample size, so
    // charging it on n_indexed -- which differs per row, because a restrictive
    // class matches fewer peaks to allowed lines -- hands the restrictive class
    // a discount. Measured: dropping from 200 indexed peaks to 30 is worth
    // +3.8 nats, and 200 to 120 is worth +1.0, against a decisive threshold of
    // 2.3. The intent was the opposite: charge the class whose line list barely
    // constrains the cell.
    //
    // BIC's n is the number of OBSERVATIONS, which is the peak list and is
    // therefore identical for every row -- so it cancels out of every
    // comparison, as it should. What does not cancel is the small-sample
    // correction: AICc's k(k+1)/(n-k-1) blows up as the number of constraining
    // lines approaches the number of free parameters, which is exactly the
    // "absorb the error into my own zero-point" case the old comment described.
    const nObs = Math.max(2, (stats.indexed || 0) + (stats.nViolations || 0) +
                             (stats.unindexed || 0));
    score -= 0.5 * nPar * Math.log(nObs);
    const nCon = Math.max(nPar + 2, stats.indexed || 0);
    score -= nPar * (nPar + 1) / (nCon - nPar - 1);

    return {
        score,
        pHat: p, epsBase, nPositions,
        lambda: lam,
        // What the absences were worth before and after saturation. When the two
        // differ the row's lead is being carried by how much it forbids, which
        // is worth being able to see.
        cleanNats: clean, cleanNatsRaw: cleanRaw, cleanCap,
        cleanCapped: cleanRaw > cleanCap * 0.9,
        nCleanEff: stats.nClean,
        nViolEff: kept.violations.length,
        nHardEff: kept.violations.filter(v => !v.ka2 && !v.weak && !v.ambiguous).length,
        nUnindexedEff: Math.max(0, kept.nUnindexed),
    };
}
// Two-pass estimate of p. Pass 1 uses the most permissive class present, which
// is biased LOW whenever the true lattice is centred (all the systematically
// absent lines sit in the denominator and are never observed). Pass 2
// re-estimates from the class pass 1 favoured and rescores everything against
// that single shared value, so the rows stay directly comparable. One iteration
// suffices in practice and the estimate is clamped either way.
function sgRescoreAll(rows, opts) {
    const o = opts || {};
    const scored = rows.filter(r => !r.error && r.stats);
    if (!scored.length) return rows;

    const estimate = (r) => (r.stats.nAllowedInRange > 0)
        ? r.stats.nAllowedObserved / r.stats.nAllowedInRange : 0.5;

    // THE SHARED POSITION COUNT. epsBase is a property of the pattern, so it is
    // measured once on the unrestricted parent lattice and handed to every row
    // rather than being recomputed on each row's own refined cell.
    let nPosShared = 0;
    if (o.frame && o.frame.centres) {
        for (let g = 0; g < o.frame.centres.length; g++) {
            const qc = o.frame.centres[g];
            if (qc >= o.frame.qLo && qc <= o.frame.qHi) nPosShared++;
        }
    }
    if (!nPosShared) {
        for (const r of scored) {
            const n = (r.stats.nAllowedInRange || 0) + (r.stats.nInformative || 0);
            if (n > nPosShared) nPosShared = n;
        }
    }

    // LAMBDA FOLLOWS P-HAT THROUGH BOTH PASSES.
    //
    // lambda sets the level of the per-position detectability and p-hat sets the
    // level of the uniform one; they are the same physical quantity measured two
    // ways, so calibrating them against different hypotheses makes the clean and
    // violation terms disagree about how detectable the pattern is. lambda was
    // solved once inside sgWilsonContext() against the UNFILTERED lattice -- the
    // no-extinction hypothesis, biased low for exactly the reason pass 1 is --
    // and then never revisited, while p-hat was re-estimated from the winner.
    // Each pass now re-solves lambda on the same reference class, and against
    // that class's own detection rate, that it takes p-hat from.
    const wilson = o.wilson || null;
    const lamFor = (r, target) => {
        if (!wilson || !wilson.solveLambda) return null;
        const zs = r && r.stats && r.stats.allowedZ;
        if (!zs || !zs.length) return wilson.lambda;
        return wilson.solveLambda(zs, target) ?? wilson.lambda;
    };
    const applyAll = (pHat, lambda) => {
        const so = { ...o, nPositions: nPosShared, wilson, lambda };
        for (const r of scored) Object.assign(r, sgScoreFromStats(r.stats, pHat, so));
    };

    // WHICH CLASS IS "THE MOST PERMISSIVE"?
    //
    // It used to be the one with the fewest STATED rules, and rule count is not
    // permissiveness. I-centring is one condition and deletes half of reciprocal
    // space; Pbca is three and deletes far less. On an orthorhombic pattern the
    // old seed therefore handed pass 1 to the F-centred class -- p estimated
    // over the 330 positions F leaves open instead of the 600 the true class
    // leaves open, 0.515 instead of 0.300 -- and since the clean-absence term is
    // n_clean * log(1/(1-p)), that single misestimate moved the leader's score
    // by tens of nats.
    //
    // nAllowedGroups is what the observable merge already computed for exactly
    // this quantity: how many resolvable positions the rule set leaves open in
    // this window. stats.nAllowedInRange is the same count measured on the row's
    // own refined cell, and serves as the fallback.
    const openness = (r) => (isFinite(r.nAllowedGroups) ? r.nAllowedGroups
                                                        : (r.stats.nAllowedInRange || 0));
    const permissive = scored.slice().sort((a, b) => openness(b) - openness(a))[0];
    let pHat = estimate(permissive);
    let lambda = lamFor(permissive, pHat);
    applyAll(pHat, lambda);

    // Pass 2 re-estimates from the winner, and "the winner" has to mean the same
    // thing here as it does in sgRankRows(): falsification first, then score.
    // Taking the top raw score let a class the ranking was about to throw out --
    // one carrying hard violations -- set the p that every surviving row is then
    // scored against.
    const alive = scored.filter(r => (r.nHardEff || 0) === 0);
    const pool = alive.length ? alive : scored;
    const best = pool.slice().sort((a, b) => b.score - a.score)[0];
    const pHat2 = estimate(best);
    const lambda2 = lamFor(best, isFinite(pHat2) ? pHat2 : pHat);
    // Rescore if EITHER level moved. The reference set changes even when the two
    // rates happen to agree, and lambda is solved on the set, not on the rate.
    const pMoved = isFinite(pHat2) && Math.abs(pHat2 - pHat) > 0.01;
    const lMoved = isFinite(lambda2) && isFinite(lambda) &&
                   Math.abs(Math.log(lambda2 / lambda)) > 0.01;
    if (pMoved || lMoved) {
        if (pMoved) pHat = pHat2;
        if (isFinite(lambda2)) lambda = lambda2;
        applyAll(pHat, lambda);
    }
    for (const r of rows) if (!r.error) { r.pHat = pHat; r.lambda = lambda; }
    return rows;
}
// ============================================================================
// PER-CLASS EVALUATION
// ============================================================================
//
// opts.mode:
//   'fixed' - score the parent cell as it stands. Cheapest, and the only mode
//             that reproduces the old stage-1 behaviour.
//   'ls'    - one constrained least-squares refit against the restricted line
//             list. Cheap enough to run on EVERY class, and unlike 'fixed' it
//             does not judge a hypothesis using a cell that was fitted to the
//             very reflections the hypothesis forbids -- which is the whole
//             reason this module exists. This is the default first pass.
//   'mc'    - full Monte-Carlo plus least squares (the expensive shortlist).
//
// The returned `cell` is a normal solution object -- system, volume, errors,
// m20, analysis-ready -- so the caller can drop it into the solutions ledger.
function sgScoreClass(cls, sol, data, state, opts) {
    const o = opts || {};
    const mode = o.mode || (o.mc === false ? 'ls' : 'mc');
    const ctx = sgMakeCtx(data, state);
    const ictx = o.ictx || sgIntensityContext(state.peaks_sorted_by_q);

    const row = {
        label: cls.label, members: cls.members,
        conditions: cls.allConditions || cls.conditions,
        mergedLabels: cls.mergedLabels || [cls.label],
        repSymbol: cls.repSymbol, centering: cls.centering, nRules: cls.nRules,
        // How many resolvable positions this rule set leaves open in the window,
        // measured on the parent frame by sgObservableMerge(). This is the
        // honest measure of permissiveness; nRules is not (see sgRescoreAll).
        nAllowedGroups: cls.nAllowedGroups,
        sig: cls.sig, mode,
        m20: 0, m_all: 0, n20: 0, cell: null, mcGain: 0, error: null,
        score: -Infinity, stats: null,
    };

    try {
        // --- baseline: the parent cell, restricted line list ------------------
        const base = { ...sol };
        let baseEval = null;
        try {
            setSpaceGroupFilter(cls.allowed);
            baseEval = mcEvaluateCell(base, ctx);
        } finally { setSpaceGroupFilter(null); }
        if (!baseEval) { row.error = 'cell generates no allowed lines'; return row; }

        row.m20 = base.m20 || 0;
        row.m_all = base.m_all || 0;
        row.n20 = base.n_20 || 0;
        row.cell = base;
        let moved = false;

        // WHAT DECIDES WHETHER A REFINED CELL IS KEPT.
        //
        // It used to be M20, in both modes, and M20 is not the quantity this
        // table ranks on. Within one class the line list is fixed, so comparing
        // M20 between two cells of the same class is at least legitimate -- but
        // a cell that raises the SCORE while nudging M20 down was discarded, and
        // the score is what decides the row's fate three lines later. The two
        // criteria are not the same: M20 rewards a tight fit to the twenty
        // lowest lines, the score weighs every absence and every violation
        // across the whole pattern.
        //
        //   'ls' accepts unconditionally. The least-squares cell IS the
        //        hypothesis -- the cell fitted to the restricted line list,
        //        which is the entire reason the mode exists. Refusing it
        //        because M20 fell means judging the hypothesis on a cell fitted
        //        to reflections it forbids, which is the bias this module was
        //        written to remove. Only a degenerate solve is rejected.
        //
        //   'mc'  accepts on the score, evaluated against a FIXED reference p
        //        taken from the baseline. Letting each candidate supply its own
        //        p would let the walk improve its apparent score by changing the
        //        yardstick; the shortlist is short, so the extra stats pass is
        //        affordable here in a way it would not be at stage 1.
        //
        // M20 SURVIVES AS THE TIE-BREAK, and it has to. The score measures how
        // well a RULE SET fits, not how well a cell fits: once every peak is
        // indexed and every forbidden position is empty, two cells of the same
        // class score within noise of each other however differently they are
        // refined. Measured on a synthetic I-centred pattern, the parent cell
        // and a cell fitted to 0.1 mA scored 16.15 against 16.16 -- a hundredth
        // of a nat deciding a factor of 1.7 in M20. Inside one class the line
        // list is fixed, so M20 is a legitimate comparison there, and it is the
        // only one of the two that can see the difference. The score leads; M20
        // speaks only when the score is silent.
        const SG_SCORE_TIE_NATS = 0.5;      // well under SG_DECISIVE_NATS
        const beats = (sA, mA, sB, mB) =>
            (sA > sB + SG_SCORE_TIE_NATS) ||
            (Math.abs(sA - sB) <= SG_SCORE_TIE_NATS && (mA || 0) > (mB || 0));
        const statsFor = (cell, preRefl) =>
            sgIndexingStats(cell, data, state, cls.allowed, ictx, preRefl, o.wilson || null);
        const refP = (st) => (st && st.nAllowedInRange > 0)
            ? st.nAllowedObserved / st.nAllowedInRange : 0.5;
        const scoreOf = (st, pRef, nPosRef) => {
            if (!st) return -Infinity;
            try {
                const s = sgScoreFromStats(st, pRef, {
                    system: sol.system,
                    refineZero: o.refineZero,
                    impurityAllowance: o.impurityAllowance,
                    nPositions: nPosRef,
                    wilson: o.wilson || null,
                    lambda: o.wilson ? o.wilson.lambda : null,
                });
                return isFinite(s.score) ? s.score : -Infinity;
            } catch (e) { return -Infinity; }
        };
        let stats = null;

        // --- refinement --------------------------------------------------------
        if (mode === 'ls') {
            let ls = null;
            try {
                setSpaceGroupFilter(cls.allowed);
                ls = mcLeastSquaresPolish(base, data, state, ctx);
                if (ls) { ls.system = sol.system; mcEvaluateCell(ls, ctx); }
            } finally { setSpaceGroupFilter(null); }
            if (ls && isFinite(ls.m20) && isFinite(ls.a) && ls.a > 0) {
                row.mcGain = ls.m20 - row.m20;      // may be negative now, by design
                row.m20 = ls.m20;
                row.m_all = isFinite(ls.m_all) ? ls.m_all : row.m_all;
                row.n20 = ls.n_20 || row.n20;
                row.cell = ls;
                moved = true;
            }
        } else if (mode === 'mc') {
            // The LS polish is run HERE TOO, not only in stage 1. Stage 2 starts
            // over from the parent cell, so a class whose Monte-Carlo walk
            // returns nothing useful used to fall all the way back to the
            // unrefined parent -- ending up with a WORSE cell than the same
            // class had after stage 1, and being ranked against stage-1 rows on
            // that basis. Measured on a synthetic I-centred pattern: the P class
            // came out of stage 2 at M20 79.9 having left stage 1 at 128.7.
            // Offering all three candidates and taking the best by score makes
            // stage 2 monotone in the only sense that matters.
            let ls = null, mc = null;
            try {
                setSpaceGroupFilter(cls.allowed);
                ls = mcLeastSquaresPolish(base, data, state, ctx);
                if (ls) { ls.system = sol.system; mcEvaluateCell(ls, ctx); }
                // monteCarloRefineCell returns null when it cannot beat its
                // starting point, which here is the constrained baseline -- so
                // null simply means "the parent cell was already the best under
                // these rules".
                mc = monteCarloRefineCell(sol, data, state, {
                    iterations: o.iterations ?? 600,
                    restarts: o.restarts ?? 4
                    // The seed is deliberately left at its default: every class
                    // walks the SAME pseudo-random sequence, so a difference
                    // between two rows is a difference between hypotheses and
                    // not between two draws.
                });
            } finally { setSpaceGroupFilter(null); }
            if (mc) mc.system = sol.system;

            const usable = (c) => c && isFinite(c.m20) && isFinite(c.a) && c.a > 0;
            const baseStats = statsFor(base, o.frame ? o.frame.refl : null);
            const pRef = refP(baseStats);
            const nPosRef = baseStats
                ? (baseStats.nAllowedInRange || 0) + (baseStats.nInformative || 0) : 0;
            let bestCell = null, bestStats = baseStats;
            let bestScore = scoreOf(baseStats, pRef, nPosRef);
            let bestM20 = row.m20;
            for (const cand of [ls, mc]) {
                if (!usable(cand)) continue;
                const st = statsFor(cand, null);
                const s = scoreOf(st, pRef, nPosRef);
                if (beats(s, cand.m20, bestScore, bestM20)) {
                    bestScore = s; bestM20 = cand.m20; bestCell = cand; bestStats = st;
                }
            }
            if (bestCell) {
                row.mcGain = bestCell.m20 - row.m20;
                row.m20 = bestCell.m20;
                row.m_all = isFinite(bestCell.m_all) ? bestCell.m_all : row.m_all;
                row.n20 = bestCell.n_20 || row.n20;
                row.cell = bestCell;
                moved = true;
            }
            stats = bestStats;                  // reuse whichever won
        }

        row.zero = row.cell.zero_correction ?? 0;

        // --- how well does the winning cell obey the rules? --------------------
        // When the cell did not move, the parent's line list is still valid, so
        // reuse the frame instead of regenerating it once per class.
        if (!stats) {
            const preRefl = (!moved && o.frame) ? o.frame.refl : null;
            stats = statsFor(row.cell, preRefl);
        }
        if (!stats) { row.error = 'cell generates no lines'; return row; }

        row.stats = stats;
        row.indexed = stats.indexed;
        row.violations = stats.nViolations;
        row.hardViolations = stats.nHard;
        row.unindexed = stats.unindexed;
        row.violatingTth = stats.violatingTth;
        row.violationDetail = stats.violations.slice(0, 8);
        row.nClean = stats.nClean;
        row.nInformative = stats.nInformative;
        row.nLines = stats.nLines;
    } catch (err) {
        row.error = String((err && err.message) || err);
    }
    return row;
}
// The ctx object mcEvaluateCell expects, built from the same data/state pair the
// rest of the MC machinery uses.
function sgMakeCtx(data, state) {
    const n_all = state.peaks_sorted_by_q.length;
    return {
        wavelength: data.wavelength,
        q_max: state.q_max,
        d_min: state.d_min,
        impurity_peaks: data.impurity_peaks,
        peaks_sorted_by_q: state.peaks_sorted_by_q,
        n_20: Math.min(state.N_FOR_M20 || 20, n_all),
        n_all,
        tolFn: (idx) => get_q_tolerance(idx, state.tth_obs_rad, data.wavelength, data.tth_error)
    };
}
// ============================================================================
// RANKING
// ============================================================================
//
// FALSIFICATION FIRST, THEN THE LOG-ODDS SCORE.
//
// A systematic absence is not a statistical tendency. If a space group has an
// a-glide then |F| is EXACTLY zero for h0l with h odd, and a single genuine
// reflection there rules the group out however many other absences hold. The
// likelihood score cannot express that on its own: it multiplies evidence across
// reflections, so a class forbidding a great deal of reciprocal space accrues
// absence credit in proportion to how much it forbids, and with enough of it any
// number of violations can be outweighed. On a real PbSO4 pattern the I-centred
// class did exactly that -- nineteen hard violations, and it still outscored a
// P2_1/a class that violated nothing.
//
// So rows are ranked in two tiers. A row with hard violations the impurity
// allowance does not cover is FALSIFIED and cannot outrank an unfalsified one,
// whatever its score. Within each tier the score decides, so the ordering among
// survivors is still the full likelihood comparison and the falsified rows are
// still ordered least-bad first -- knowing WHICH group the data exclude, and by
// how much, is half the answer.
//
// Only HARD violations falsify. Ka2 ghosts, peaks below the local weak
// threshold, and peaks with an allowed line inside the matching window are all
// graded soft and do not, because none of them is a reliable reflection.
//
// M20 survives as a DISPLAY column and as the last tie-break. It is a figure of
// merit for a CELL, not a criterion for choosing between line lists, and the
// nRules tie-break that used to sit above it has been removed outright: it
// rewarded restrictiveness for its own sake, which the likelihood already
// accounts for wherever the data support it.
const SG_DECISIVE_NATS = 2.3;   // ~10:1 odds; below this the table is a tie
function sgRankRows(rows, impurityAllowance, opts) {
    sgRescoreAll(rows, { ...(opts || {}), impurityAllowance });
    return rows.slice().sort((a, b) => {
        if (a.error && !b.error) return 1;
        if (b.error && !a.error) return -1;
        if (a.error && b.error) return 0;
        // tier 1: falsified or not
        const fa = ((a.nHardEff || 0) > 0) ? 1 : 0;
        const fb = ((b.nHardEff || 0) > 0) ? 1 : 0;
        if (fa !== fb) return fa - fb;
        // tier 2: the log-odds score
        const sa = isFinite(a.score) ? a.score : -Infinity;
        const sb = isFinite(b.score) ? b.score : -Infinity;
        if (Math.abs(sa - sb) > 1e-9) return sb - sa;
        if ((a.nHardEff || 0) !== (b.nHardEff || 0)) return (a.nHardEff || 0) - (b.nHardEff || 0);
        if ((b.nCleanEff || 0) !== (a.nCleanEff || 0)) return (b.nCleanEff || 0) - (a.nCleanEff || 0);
        if (Math.abs((b.m20 || 0) - (a.m20 || 0)) > 1e-6) return (b.m20 || 0) - (a.m20 || 0);
        return (a.members?.[0]?.number || 999) - (b.members?.[0]?.number || 999);
    });
}
// Is the winner actually separated from the runner-up, or is the table a tie?
// Reported to the user instead of silently presenting row 1 as the answer.
// The margin is measured WITHIN the surviving tier. Comparing an unfalsified row
// against a falsified one would report a lead that means nothing: the two are not
// competing on the same question.
// A SECOND COMPARABILITY CONDITION: THE SAME REFINEMENT DEPTH.
//
// Stage 1 gives every class one least-squares solve; stage 2 gives the
// shortlist a full Monte-Carlo walk. An MC row can beat an LS row on compute
// alone -- a better cell for the same hypothesis -- so a margin measured across
// the two is partly a measure of who got refined, not of what the data say. The
// table already marks stage-1 rows, but the margin and the note treated every
// row as comparable.
//
// The margin is therefore measured among rows sharing the LEADER's mode. Once
// stage 2 has run that is the Monte-Carlo set, which is the honest comparison;
// during stage 1 every row is 'ls' and nothing changes. sgMarginInfo() reports
// what was compared so the caller can say so.
function sgComparableTier(ranked) {
    const ok = (ranked || []).filter(r => !r.error && isFinite(r.score));
    if (!ok.length) return { tier: [], restricted: false, mode: null };
    const alive = ok.filter(r => (r.nHardEff || 0) === 0);
    const surviving = alive.length ? alive : ok;
    const mode = surviving[0].mode || null;
    const same = surviving.filter(r => (r.mode || null) === mode);
    const restricted = same.length > 0 && same.length < surviving.length;
    return { tier: same.length ? same : surviving, restricted, mode };
}
function sgMarginInfo(ranked) {
    const { tier, restricted, mode } = sgComparableTier(ranked);
    const margin = tier.length < 2 ? Infinity : tier[0].score - tier[1].score;
    return { margin, mode, nCompared: tier.length, restricted, tier };
}
function sgMargin(ranked) {
    return sgMarginInfo(ranked).margin;
}
// Did anything survive at all? When every class carries hard violations the
// answer is not "the least bad one wins" but "none of these fits", and the caller
// should say so rather than presenting row 1 as a determination.
function sgAnySurvivor(ranked) {
    return (ranked || []).some(r => !r.error && (r.nHardEff || 0) === 0);
}
