// js/dialogs/space-group-mc.js
// Space Group MC dialog.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// 3d. Action: Space Group MC
// The absence analysis in analyzeSystematicAbsences() judges every space
// group against ONE cell that was refined without knowing about any of them.
// This asks a different question: if this space group were true, how well
// would the pattern index? Each extinction class gets its own line list, its
// own refinement and its own statistics, so the classes are compared as
// hypotheses rather than as violation tallies against a cell that was fitted
// to forbidden reflections.
//
// Rows are hypotheses the DATA can separate, not space groups. Candidates
// are collapsed twice: first by what they forbid arithmetically, then by
// what they actually do to the calculated pattern for this cell, at this
// wavelength, over this range, at this tolerance. Rule sets that differ only
// beyond q_max or only at coincident lines end up on one row, because
// presenting them separately - with separate figures of merit, no less - is
// the main way a space-group table misleads.
//
// Ranking is a log-odds score, not M20. M20 rises whenever lines are removed
// from the list, so it structurally rewards restriction and cannot be used
// to choose between line lists; see js/crystallography/sg-score.js.
// M20 stays in the table because it describes the refined CELL, which is
// still worth seeing.
let sgRunning = false;
let sgAbort = false;
let sgRows = [];
let sgSelectedSig = null;
let sgSelectedMemberIdx = 0; // Tracks which specific space group chip is selected
// Display order. The log-odds score is both the ANALYTIC authority -- it picks
// the stage-2 shortlist, sets the tie report and estimates p -- and the
// default ordering, so what the table shows first is what the evidence
// actually favours. Any column can still be clicked to re-sort.
let sgSortKey = 'score';
let sgSortDir = -1;
let sgParent = null;
const sgOverlay = document.getElementById('sg-overlay');
const sgMsg = document.getElementById('sg-msg');
const sgRunBtn = document.getElementById('sg-run');
const sgAddBtn = document.getElementById('sg-add');
const sgTbody = document.getElementById('sg-tbody');
const closeSgModal = () => {
    if (sgRunning) { sgAbort = true; return; }   // first Esc cancels the run
    sgOverlay.classList.remove('open');
};
// Everything the MC needs, built exactly as the Refine MC dialog builds it so
// the figures of merit are directly comparable with the rest of the table.
//
// Ka2-suspect peaks are EXCLUDED here, the same way indexing excludes them.
// They used to be left in, and a Ka2 ghost sitting a few hundredths of a
// degree off its parent is exactly the kind of peak that lands on a
// forbidden line and manufactures a violation against the correct rule set.
const sgBuildInputs = () => {
    const wl = getWavelength();
    const te = getTthError();
    const tMin = parseFloat(ui.tthMinSlider.value);
    const tMax = parseFloat(ui.tthMaxSlider.value);
    const imp = getImpurityPeaks();
    const rz = !!(ui.refineZeroCheckbox && ui.refineZeroCheckbox.checked);
    const inRange = pickedPeaks.filter(p => p.tth >= tMin && p.tth <= tMax);
    const pk = inRange.filter(p => !p.ka2Suspect);
    const nGhosts = inRange.length - pk.length;
    if (pk.length < 6) return { error: 'Not enough peaks in the 2-theta range (need at least 6).' };
    if (!isFinite(wl) || wl <= 0) return { error: 'Invalid wavelength.' };

    const sorted = getSortedPeaks(pk, wl);
    const maxTth = Math.max(...pk.map(p => p.tth));
    const d_min = wl / (2 * Math.sin(maxTth * Math.PI / 360));
    if (!isFinite(d_min) || d_min <= 0) return { error: 'Could not derive d_min from the peak list.' };

    return {
        pk, wl, te, tMin, tMax, imp, rz, nGhosts,
        state: {
            q_obs: sorted.q_obs,
            original_indices: sorted.original_indices,
            tth_obs_rad: sorted.tth_obs_rad,
            peaks_sorted_by_q: sorted.peaks_sorted_by_q,
            N_FOR_M20: 20,
            q_max: 1 / (d_min * d_min),
            d_min: d_min,
            foundSolutions: [],
            foundSolutionMap: new Map()
        },
        data: {
            peaks: pk, wavelength: wl, tth_error: te,
            max_volume: getMaxVolume(),
            impurity_peaks: imp, refineZero: rz,
            // The scanned limits, so a forbidden line that falls between the
            // start of the scan and the first observed peak still counts as
            // an absence. Without these the most diagnostic low-angle part
            // of the pattern carries no evidence at all.
            tth_min: tMin, tth_max: tMax
        }
    };
};
const sgFmt = (x, n) => (isFinite(x) ? Number(x).toFixed(n) : '-');
const sgSigned = (x, n) => (isFinite(x) ? (x > 0 ? '+' : '') + Number(x).toFixed(n) : '-');
// #sg-msg is plain by default and red with .error, mirroring the .info
// convention #mc-msg uses in styles.css.
const sgSay = (text, isError) => {
    sgMsg.textContent = text || '';
    sgMsg.classList.toggle('error', !!isError);
};
// Scoring options shared by every call into the ranking, so a row is never
// scored under different assumptions than the row above it.
let sgScoreOpts = { system: null, refineZero: false };
const sgViolTitle = (r) => {
    if (!r.violationDetail || !r.violationDetail.length) return 'no violations';
    return r.violationDetail.map(v => {
        const bits = [`${v.tth.toFixed(3)}\u00B0`];
        if (v.eSq !== null && v.eSq !== undefined && isFinite(v.eSq)) bits.push(`|E|\u00B2 ${v.eSq.toFixed(2)}`);
        if (v.rel !== null && v.rel !== undefined) bits.push(`I/I_local ${(v.rel * 100).toFixed(0)}%`);
        if (v.dqOverTol !== null && v.dqOverTol !== undefined) bits.push(`\u0394q ${(v.dqOverTol * 100).toFixed(0)}% of tol`);
        const tags = [v.ka2 && 'Ka2', v.weak && 'weak', v.ambiguous && 'ambiguous'].filter(Boolean);
        if (tags.length) bits.push(tags.join('/'));
        return bits.join(', ');
    }).join('\n');
};
// Display sort. Errored rows always sink to the bottom; within the rest the
// chosen key decides. Score is the default: it is what ranks the hypotheses,
// so the table leads with the row the evidence favours. M20 is available as a
// sort but describes the refined CELL rather than the hypothesis, and rises
// whenever lines are deleted -- ordering by it puts the most restrictive
// surviving class on top, which is not the same question. Whatever the sort,
// the leading row on score stays marked in bold.
const SG_SORT = {
    m20:    (r) => r.m20 || 0,
    score:  (r) => isFinite(r.score) ? r.score : -Infinity,
    clean:  (r) => r.nClean || 0,
    viol:   (r) => -(r.hardViolations || 0),
    unidx:  (r) => -(r.unindexed || 0),
    lines:  (r) => r.nLines || 0,
    zero:   (r) => Math.abs(r.zero || 0),
    label:  (r) => r.label || '',
    groups: (r) => (r.members && r.members.length) || 0,
};
const sgSortForDisplay = (ranked) => {
    const f = SG_SORT[sgSortKey] || SG_SORT.m20;
    const idx = new Map(ranked.map((r, i) => [r, i]));
    return ranked.slice().sort((a, b) => {
        if (a.error && !b.error) return 1;
        if (b.error && !a.error) return -1;
        const va = f(a), vb = f(b);
        if (typeof va === 'string' || typeof vb === 'string') {
            const c = String(va).localeCompare(String(vb));
            if (c) return sgSortDir < 0 ? -c : c;
        } else if (va !== vb) {
            return sgSortDir < 0 ? (vb - va) : (va - vb);
        }
        // Ties fall back to the ranked order, so sorting by score reproduces
        // sgRankRows() exactly -- including its tie-breaks on hard
        // violations, clean absences, M20 and group number.
        return idx.get(a) - idx.get(b);
    });
};
// One row of the table. Pure: everything it needs arrives in `ctx`, nothing is
// read from the enclosing scope. That is deliberate -- it is the only part of
// this dialog that can be exercised outside a browser, and it is where a
// stale loop variable hid until it reached the console as
// "ReferenceError: i is not defined".
const sgRowHtml = (r, ctx) => {
    const groupsHtml = r.members.map((m, idx) => {
        const isSelected = (r.sig === ctx.selectedSig && idx === ctx.selectedMemberIdx);
        const chipCls = isSelected ? 'sg-chip selected-chip' : 'sg-chip';
        return `<span class="${chipCls}" data-idx="${idx}">${m.symbol} (${m.number})</span>`;
    }).join(' ');

    const conds = (r.conditions && r.conditions.length) ? r.conditions.join(' ; ') : 'no conditions';
    const isBest = !r.error && r === ctx.best;
    // Score is shown relative to the leader OF THE SAME TIER. Measuring a
    // falsified row against the surviving winner produced a large positive
    // delta on a row sitting at the bottom of the table -- literally
    // "+182.50" under a heading the user reads as a ranking. Falsified rows
    // are not competing on the same question, so they are ranked among
    // themselves, least-bad first, and read 0.00 at the top of their block.
    const falsified = !r.error && (r.nHardEff || 0) > 0;
    const ref = falsified ? ctx.deadScore : ctx.bestScore;
    const dScore = (ref !== null && ref !== undefined && isFinite(r.score))
        ? (r.score - ref) : NaN;

    const cls = [];
    if (r.error) cls.push('sg-bad');
    else if ((r.nHardEff || 0) > 0) cls.push('sg-ruled-out');
    if (r.sig === ctx.selectedSig) cls.push('sg-sel');
    // "best" means best on SCORE, wherever the display sort has put it.
    if (isBest) cls.push('sg-best');
    if (!r.error && r.mode !== 'mc') cls.push('sg-stage1');

    // Hard violations are the ones that count against the row; soft ones
    // (Ka2 / weak / ambiguous) are shown in parentheses so the user can see
    // them without them silently deciding the ranking.
    const soft = (r.violations || 0) - (r.hardViolations || 0);
    const violTxt = r.error ? '-'
        : `${r.hardViolations || 0}` + (soft > 0 ? ` (+${soft})` : '');

    return `<tr class="${cls.join(' ')}" data-sig="${r.sig}" title="${conds}">` +
           `<td title="${(r.mergedLabels || []).join(' \u2261 ')}">${r.label}</td>` +
           `<td class="sg-groups">${groupsHtml}</td>` +
           `<td>${r.error ? '-' : (r.nLines ?? '-')}</td>` +
           `<td>${r.error ? '-' : sgFmt(r.m20, 2)}</td>` +
           `<td class="sg-score">${r.error ? '-' : (isBest ? '0.00' : sgSigned(dScore, 2))}</td>` +
           `<td title="of ${r.nInformative ?? '-'} resolvable forbidden lines">` +
               `${r.error ? '-' : `${r.nClean ?? '-'}/${r.nInformative ?? '-'}`}</td>` +
           `<td title="${r.error ? '' : sgViolTitle(r)}">${violTxt}</td>` +
           `<td>${r.error ? '-' : r.unindexed}</td>` +
           `<td>${r.error ? '-' : sgSigned(r.zero || 0, 3)}</td></tr>`;
};
const sgRenderRows = (impAllowance) => {
    const ranked = sgRankRows(sgRows, impAllowance, sgScoreOpts);
    sgRows = ranked;
    const best = ranked.find(r => !r.error && isFinite(r.score));
    // Two references, one per tier: survivors are measured against the best
    // survivor, falsified rows against the least-bad falsified row.
    const dead = ranked.find(r => !r.error && isFinite(r.score) && (r.nHardEff || 0) > 0);
    const ctx = {
        best,
        bestScore: best ? best.score : null,
        deadScore: dead ? dead.score : null,
        selectedSig: sgSelectedSig,
        selectedMemberIdx: sgSelectedMemberIdx
    };
    const displayed = sgSortForDisplay(ranked);
    sgTbody.innerHTML = displayed.map(r => sgRowHtml(r, ctx)).join('');
    const sel = sgRows.find(r => r.sig === sgSelectedSig);
    sgAddBtn.disabled = !(sel && sel.cell && !sel.error);
    // mark the sorted column in the header
    const ths = document.querySelectorAll('#sg-table thead th');
    ths.forEach(th => {
        const k = th.dataset.sort;
        th.classList.toggle('sg-sorted', k === sgSortKey);
        th.classList.toggle('sg-asc', k === sgSortKey && sgSortDir > 0);
    });
};
sgTbody.addEventListener('click', (e) => {
    const tr = e.target.closest('tr');
    if (!tr) return;
    
    const chip = e.target.closest('.sg-chip');
    if (chip) {
        sgSelectedSig = tr.dataset.sig;
        sgSelectedMemberIdx = parseInt(chip.dataset.idx, 10);
    } else {
        // Clicking the row but outside a chip selects the row and defaults to the first chip
        sgSelectedSig = tr.dataset.sig;
        sgSelectedMemberIdx = 0;
    }
    sgRenderRows(getImpurityPeaks());
});
// Click a header to re-sort; click the same one again to reverse it.
document.querySelector('#sg-table thead').addEventListener('click', (e) => {
    const th = e.target.closest('th');
    if (!th || !th.dataset.sort) return;
    if (sgSortKey === th.dataset.sort) sgSortDir = -sgSortDir;
    else { sgSortKey = th.dataset.sort; sgSortDir = -1; }
    sgRenderRows(getImpurityPeaks());
});
document.getElementById('ctx-sg').addEventListener('click', () => {
    const parent = ctxTargetSolution();
    if (!parent) return;
    if (typeof sgExtinctionClasses !== 'function' || typeof monteCarloRefineCell !== 'function') {
        showStatus('Space-group MC is unavailable (crystallography scripts not loaded).', 'error', 5000);
        return;
    }
    if (!spaceGroupData || !spaceGroupData.space_groups) {
        showStatus('Space group data not loaded - cannot scan.', 'error', 5000);
        return;
    }
    if (!parent.system || typeof MC_NPAR === 'undefined' || !MC_NPAR[parent.system]) {
        showStatus('This solution has no crystal system the MC can constrain.', 'error', 4000);
        return;
    }

    sgParent = parent;
    sgRows = [];
    sgSelectedSig = null;
    sgSelectedMemberIdx = 0;
    sgTbody.innerHTML = '';
    sgAddBtn.disabled = true;
    sgRunBtn.disabled = false;
    sgRunBtn.textContent = 'Run scan';
    sgSay('', false);
    document.getElementById('sg-title').textContent =
        `Space Group MC - ${parent.system}, a=${parent.a.toFixed(4)}` +
        (parent.b ? `, b=${parent.b.toFixed(4)}` : '') +
        (parent.c ? `, c=${parent.c.toFixed(4)}` : '');
    sgOverlay.classList.add('open');
    contextMenu.style.display = 'none';
});
document.getElementById('sg-cancel').addEventListener('click', closeSgModal);
sgOverlay.addEventListener('click', (e) => { if (e.target === sgOverlay) closeSgModal(); });
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sgOverlay.classList.contains('open')) closeSgModal();
});
sgRunBtn.addEventListener('click', () => {
    if (sgRunning) { sgAbort = true; return; }
    const parent = sgParent;
    if (!parent) { closeSgModal(); return; }

    const readInt = (id, lo, hi, dflt) => {
        const el = document.getElementById(id);
        const v = parseInt(el.value, 10);
        return (isFinite(v) && v >= lo && v <= hi) ? v : dflt;
    };
    const nIter = readInt('sg-iterations', 50, 20000, 600);
    const nRes  = readInt('sg-restarts', 1, 20, 4);
    const topN  = readInt('sg-top', 1, 50, 8);

    const inp = sgBuildInputs();
    if (inp.error) { sgSay(inp.error, true); return; }
    sgScoreOpts = { system: parent.system, refineZero: inp.rz };

    let classes;
    try {
        // No centering pre-filter on purpose. The point of this scan is to be
        // an independent second opinion, and pre-filtering with the same
        // centering test that feeds the ranking would just import its verdict.
        classes = sgExtinctionClasses(spaceGroupData, parent.system, null);
    } catch (err) {
        sgSay('Could not build the candidate list: ' + (err.message || err), true);
        return;
    }
    if (!classes.length) { sgSay('No space groups found for this crystal system.', true); return; }

    // Collapse candidates that produce the SAME calculated pattern for this
    // cell, wavelength, range and tolerance. Rule sets that differ only
    // beyond q_max, or only at lines that coincide with an allowed one
    // inside the matching window, are not distinguishable here and must not
    // be presented as separate hypotheses with separate figures of merit.
    const frame = sgLatticeFrame(parent, inp.data, inp.state);
    if (!frame) { sgSay('The parent cell generates no lines in this range.', true); return; }
    const nAbstract = classes.length;
    try { classes = sgObservableMerge(classes, frame); }
    catch (err) { sgSay('Could not merge indistinguishable classes: ' + (err.message || err), true); return; }

    const ictx = sgIntensityContext(inp.state.peaks_sorted_by_q);
    // Wilson scale from the peak heights, built once from the unfiltered
    // lattice so no hypothesis can rescale its own evidence. Null whenever
    // the heights cannot support it, in which case the scoring falls back to
    // the single global p exactly as before.
    let wilson = null;
    try { wilson = sgWilsonContext(frame, inp.data, inp.state, parent.system); }
    catch (err) { wilson = null; }
    const shared = { frame, ictx, wilson };
    // sgRescoreAll() needs the frame (for the shared position count that
    // fixes epsBase for every row alike) and the Wilson context (to re-solve
    // lambda against the same reference class it takes p from). Both are
    // properties of the pattern, not of a row, so they travel with the
    // scoring options.
    sgScoreOpts = { ...sgScoreOpts, frame, wilson };

    sgRunning = true; sgAbort = false;
    sgRows = []; sgSelectedSig = null; sgSelectedMemberIdx = 0;
    sgRunBtn.textContent = 'Stop';
    sgAddBtn.disabled = true;

    // Stage 1 gives every class ONE constrained least-squares refit against
    // its own restricted line list. The old stage 1 scored every class on
    // the parent cell as it stood, which re-imported the exact bias this
    // module exists to remove: a cell refined without knowing about any
    // extinctions is pulled toward forbidden reflections, so the correct
    // class can be demoted at stage 1 and never reach the Monte-Carlo pass
    // at all. One LS solve per class is cheap and breaks that loop.
    const STAGE1_CHUNK = 4;
    let i = 0;

    const noteFor = (ranked) => {
        const best = ranked.find(r => !r.error);
        if (!best) return 'Nothing to report.';
        // Every class contradicted by a real reflection: say so plainly rather
        // than dressing the least-bad row up as an answer.
        if (!sgAnySurvivor(ranked)) {
            return `No class survives: every one is contradicted by at least one strong, ` +
                   `unambiguous peak on a line it forbids. The rows are ordered least-bad first ` +
                   `(${best.label} has ${best.hardViolations}). Check the cell, the 2-theta range, ` +
                   `or raise the impurity-peak allowance if the pattern really does contain ` +
                   `foreign lines.`;
        }
        const info = sgMarginInfo(ranked);
        const margin = info.margin;
        // The tie set has to be measured inside the SURVIVING tier, the same
        // way sgMargin() measures the margin. Filtering the whole table meant
        // a falsified class could be reported as tied with the winner -- and
        // the test was one-sided, so a falsified row scoring 180 nats ABOVE
        // the winner satisfied "(best.score - r.score) < 2.3" and was listed
        // as indistinguishable from it. Ranked order is already tier-then-
        // score, so the leader of the tier is simply its first element.
        //
        // The tier is additionally restricted to rows refined to the same
        // DEPTH as the leader (see sgComparableTier): a stage-1 row and a
        // Monte-Carlo row differ by how much compute they were given as
        // well as by hypothesis, so a lead over one of them is not the same
        // statement as a lead over the other.
        const tier = info.tier.length ? info.tier : [best];
        const lead = tier[0] || best;
        const tied = tier.filter(r => Math.abs(lead.score - r.score) < SG_DECISIVE_NATS);
        const cmpTxt = info.restricted
            ? ` Compared among the ${info.nCompared} fully refined class(es);` +
              ` stage-1 rows are least-squares only and are not directly comparable.`
            : '';
        const capTxt = (lead.cleanCapped)
            ? ` The leader's absence evidence is at the saturation ceiling` +
              ` (${(SG_CLEAN_CAP_MULT * SG_DECISIVE_NATS).toFixed(0)} nats), so its lead rests on` +
              ` violations and unindexed peaks rather than on how much it forbids.`
            : '';
        const pTxt = (wilson
            ? ` Absences weighted per reflection (Wilson, ${wilson.nUsed} peaks, R2 ${wilson.r2.toFixed(2)}).`
            : (isFinite(best.pHat) ? ` p(line observed) = ${(best.pHat * 100).toFixed(0)}%.` : ''))
            + cmpTxt + capTxt;
        if (tied.length > 1) {
            return `No decisive winner: ${tied.length} classes lie within ` +
                   `${SG_DECISIVE_NATS.toFixed(1)} nats of each other ` +
                   `(${tied.map(r => r.label).join(', ')}).` + pTxt +
                   ` The absences in this pattern cannot separate them.`;
        }
        // sgMargin() returns Infinity when the tier has a single member, which
        // used to print as "ahead of the runner-up by Infinity nats (1.1e+13:1)".
        const marginTxt = isFinite(margin)
            ? `ahead of the runner-up by ${margin.toFixed(1)} nats ` +
              `(${Math.exp(Math.min(30, margin)).toPrecision(2)}:1)`
            : `the only class the data do not contradict`;
        return `Best: ${lead.label} - ${marginTxt}, ` +
               `${lead.nClean}/${lead.nInformative} forbidden lines clean, ` +
               `${lead.hardViolations} hard violation(s).` + pTxt +
               ` Select a row and press "Add as solution" to keep its refined cell.`;
    };

    const finish = (note) => {
        sgRunning = false;
        sgRunBtn.textContent = 'Run scan';
        sgRunBtn.disabled = false;
        sgRenderRows(inp.imp);
        sgSay(note || noteFor(sgRows), false);
    };

    // Which classes deserve the expensive refinement?
    //
    // Taking the top N by score alone repeats the stage-1 problem in a
    // subtler form: the shortlist is chosen from cells that have only had a
    // single LS step, so a class whose true cell needs the annealing walk can
    // still be cut. Three guarantees go on top of the top-N, in priority
    // order:
    //
    //   1. the most permissive class -- the "no extinction" null hypothesis
    //      the whole score is measured against;
    //   2. the class with the most clean absences, which is precisely the one
    //      a contaminated parent cell is most likely to have hidden;
    //   3. one representative per centering type.
    //
    // The extras are capped, because each one costs a full Monte-Carlo run
    // and the user set "Refine top" expecting a bounded wait. If the cap
    // binds, the lower-priority centerings are the ones dropped.
    const SHORTLIST_EXTRA_MAX = 4;
    const pickShortlist = (ranked, n) => {
        const ok = ranked.filter(r => !r.error);
        const out = [];
        const seen = new Set();
        const push = (r, isExtra) => {
            if (!r || seen.has(r.sig)) return false;
            if (isExtra && out.length >= n + SHORTLIST_EXTRA_MAX) return false;
            seen.add(r.sig); out.push(r); return true;
        };
        for (const r of ok.slice(0, n)) push(r, false);
        push(ok.slice().sort((a, b) => (a.nRules || 0) - (b.nRules || 0))[0], true);
        push(ok.slice().sort((a, b) => (b.nClean || 0) - (a.nClean || 0))[0], true);
        const byCent = new Map();
        for (const r of ok) if (!byCent.has(r.centering)) byCent.set(r.centering, r);
        for (const r of byCent.values()) push(r, true);
        return out;
    };

    const stage2 = () => {
        const ranked = sgRankRows(sgRows, inp.imp, sgScoreOpts);
        const shortlist = pickShortlist(ranked, topN);
        let j = 0;
        const step2 = () => {
            if (sgAbort) return finish('Stopped. The classes not yet refined are stage-1 (least-squares) results only.');
            if (j >= shortlist.length) return finish(null);
            const cls = classes.find(c => c.sig === shortlist[j].sig);
            sgSay(`Refining ${j + 1}/${shortlist.length}: ${shortlist[j].label} ...`, false);
            if (cls) {
                const scored = sgScoreClass(cls, parent, inp.data, inp.state, {
                    ...shared, mode: 'mc', iterations: nIter, restarts: nRes
                });
                const at = sgRows.findIndex(r => r.sig === cls.sig);
                if (at >= 0) sgRows[at] = scored; else sgRows.push(scored);
            }
            j++;
            sgRenderRows(inp.imp);
            setTimeout(step2, 0);
        };
        setTimeout(step2, 0);
    };

    const step1 = () => {
        if (sgAbort) return finish('Stopped.');
        if (i >= classes.length) {
            sgSay(`Stage 1 done: ${nAbstract} settings collapse to ${classes.length} ` +
                  `distinguishable classes here. Refining the best ${topN} (plus coverage) ...`, false);
            return stage2();
        }
        const end = Math.min(i + STAGE1_CHUNK, classes.length);
        for (; i < end; i++) {
            sgRows.push(sgScoreClass(classes[i], parent, inp.data, inp.state, { ...shared, mode: 'ls' }));
        }
        sgSay(`Scoring extinction classes: ${i}/${classes.length} ...`, false);
        sgRenderRows(inp.imp);
        setTimeout(step1, 0);
    };

    const ghostNote = inp.nGhosts ? ` (${inp.nGhosts} Ka2-suspect peak(s) excluded)` : '';
    const wilsonNote = wilson
        ? ` Intensity weighting active (Wilson fit on ${wilson.nUsed} peaks, R2 ${wilson.r2.toFixed(2)}).`
        : ' Intensity weighting unavailable - absences weighted uniformly.';
    sgSay(`Scoring ${classes.length} distinguishable classes from ${nAbstract} settings${ghostNote}.` +
          wilsonNote, false);
    setTimeout(step1, 0);
});
sgAddBtn.addEventListener('click', () => {
    const row = sgRows.find(r => r.sig === sgSelectedSig);
    const parent = sgParent;
    if (!row || !row.cell || !parent) return;

    const inp = sgBuildInputs();
    if (inp.error) { sgSay(inp.error, true); return; }

    const child = { ...row.cell };
    child.system = parent.system;
    child.manualSwaps = parent.manualSwaps || [];
    // Provenance: which hypothesis produced this cell. The cell was refined
    // against the restricted line list, so it is only meaningful alongside
    // the label -- and alongside the margin, since a cell taken from a row
    // that was in a three-way tie is not a determination.
    child.sgClass = row.label;
    
    // Grab only the specific chip the user selected
    const selectedMember = row.members[sgSelectedMemberIdx] || row.members[0];
    child.sgMembers = [`${selectedMember.symbol} (${selectedMember.number})`];
    // Kept beside the symbol, not folded into that string: the Hall symbol
    // is what identifies the SETTING, and the pdCIF export writes it so a
    // reader does not have to guess which of an IT number's settings was
    // meant. Pnma and Pbnm are both No. 62 and forbid different reflections.
    child.sgHall = selectedMember.hall || null;
    
    child.sgConditions = row.conditions;
    child.sgScore = row.score;
    child.sgMargin = sgMargin(sgRows);
    child.sgEvidence = {
        wilson: !!(row.stats && row.stats.cleanP && row.stats.cleanP.length),
        clean: row.nClean, informative: row.nInformative,
        hardViolations: row.hardViolations, softViolations: (row.violations || 0) - (row.hardViolations || 0),
        unindexed: row.unindexed, pHat: row.pHat, mode: row.mode
    };

    try {
        child.analysis = analyzeSystematicAbsences(child, inp.pk, spaceGroupData,
                                                   inp.wl, inp.te, inp.tMax, inp.imp, inp.tMin);
    } catch (err) { child.analysis = null; }

    const parentIdx = solutions.indexOf(parent);
    if (parentIdx >= 0) solutions.splice(parentIdx, 0, child);
    else solutions.push(child);
    displayedSolutions = [...solutions];
    updateSolutionsTable();
    updateAllMarkers();
    sgOverlay.classList.remove('open');
    const marg = child.sgMargin;
    const margTxt = isFinite(marg)
        ? (marg < SG_DECISIVE_NATS ? ` - NOT decisive (${marg.toFixed(1)} nats over the runner-up)` : ` - ${marg.toFixed(1)} nats ahead`)
        : '';
    showStatus(`Added cell refined under ${row.label} (${selectedMember.symbol}): ` +
               `M20 ${sgFmt(row.m20, 2)}, ${row.hardViolations} hard violation(s)${margTxt}.`, 'success', 8000);
});
