// js/indexing/solutions.js
// The solution ledger: intake from workers, pruning, sorting and the results table.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

// Sorting: the Solutions tab has one select, "<column>:<direction>".
const solutionsSortSelect = document.getElementById('solutions-sort');
if (solutionsSortSelect) solutionsSortSelect.addEventListener('change', () => {
    const [column, direction] = solutionsSortSelect.value.split(':');
    sortState.column = column; sortState.direction = direction;
    sortSolutions();
    // A COPY, not an alias: handleNewSolution rebinds `solutions` when pruning,
    // which would leave an aliased displayedSolutions pointing at a stale array.
    displayedSolutions = [...solutions];
    updateSolutionsTable();
});
// New function to centralize updates (Throttled with requestAnimationFrame)
const handleNewSolution = (newSolution, runToken = indexingRunToken, fitContext = null) => {
if (runToken !== indexingRunToken || !newSolution || !newSolution.system) return;

// Stamp the producing run. `solutions` is deliberately NOT cleared between
// runs, so the array mixes cells found under the current peak set /
// wavelength with cells found under whatever was configured previously.
// finalizeIndexing() uses this to avoid re-running space-group analysis on
// an old solution against a peak list it was never derived from.
newSolution._runToken = runToken;
newSolution._fitContext = fitContext;

// --- CROSS-WORKER DEDUP ---
// Each refinement worker dedups only against its OWN foundSolutionMap, so
// two workers can independently post the same physical cell. Undeduped
// duplicates eat slots in the M20-capped list (50 -> 40) and can evict
// genuinely distinct solutions BEFORE applyFinalSieve ever runs. The list
// is capped at ~50, so a linear scan per insert is negligible, and keying
// on getSolutionKey (js/crystallography/cell.js, loaded on the main thread too) is
// robust against the sort/slice/sieve reassignments elsewhere.
let solKey = null;
try { solKey = getSolutionKey(newSolution); } catch (e) { solKey = null; }

let isDuplicate = false;
// getSolutionKey now returns null (not undefined) for an unkeyable cell;
// test truthiness so neither form can be filed as a real key -- an
// undefined _solKey used to match every other undefined _solKey here.
if (solKey) {
    newSolution._solKey = solKey;
    const dupIdx = solutions.findIndex(s => s._solKey === solKey && s._fitContext === fitContext);
    if (dupIdx !== -1) {
        isDuplicate = true;
        if (newSolution.m20 > solutions[dupIdx].m20) {
            solutions[dupIdx] = newSolution;   // keep the better copy
        } else {
            return;                            // weaker duplicate: drop, no re-render
        }
    }
}

if (!isDuplicate) {
    // --- FAST SYNCHRONOUS DATA OPERATIONS ---
    solutions.push(newSolution);

    // Max Limit Pruning
    if (solutions.length > MAX_SOLUTIONS_BEFORE_PRUNING) {
        // Always sort by quality (M20) before cutting
        // A missing or non-finite m20 makes this comparator return NaN,
        // which leaves the array in an arbitrary order -- and this sort
        // decides which solutions survive the prune.
        const rank = (x) => (x && isFinite(x.m20)) ? x.m20 : -Infinity;
        solutions.sort((a, b) => rank(b) - rank(a));
        solutions = selectWithSystemReserve(solutions, PRUNE_TO_COUNT);
    }
}

// LED indicator update (lightweight DOM manipulation is fine here)
if (solutions.length === 1) ui.solutionsLed.className = 'led-indicator green';

// --- HEAVY ASYNCHRONOUS DOM RENDERING ---
// Gate the heavy sorting and DOM rebuilding behind requestAnimationFrame
if (!isTableUpdateScheduled) {
    isTableUpdateScheduled = true;

    requestAnimationFrame(() => {
        // try/finally: if the sort or the rebuild ever throws, the lock must
        // still be released -- otherwise isTableUpdateScheduled stays true and
        // no later solution redraws the table for the rest of the session.
        try {
            // Sort solutions based on current UI state
            sortSolutions();

            // Re-Sync the ledger. Only the leading slice is rendered; row
            // click-handlers index into displayedSolutions, and the slice
            // preserves order, so indices stay aligned.
            displayedSolutions = solutions.slice(0, MAX_DISPLAYED_SOLUTIONS);

            // Rebuild the DOM table (now capped at browser refresh rate, ~60Hz)
            updateSolutionsTable();
        } finally {
            // Release the lock for the next frame
            isTableUpdateScheduled = false;
        }
    });
}
};
// Two descriptions of the same lattice: the sieve keeps the more symmetric one
// (14 Oct 2026). It used to keep the higher M(20) and let symmetry break exact
// ties only, and a description with more free parameters can score a little
// higher by fitting the noise: on Si a tetragonal cell with c = a + 0.001 A (M20 1518)
// replaced the cubic one (M20 1054). When the lower-symmetry cell's M(20) is
// more than SIEVE_SYMMETRY_M20_RATIO times higher, BOTH are kept and marked
// as a pair, and the user decides: a genuine distortion inside the 0.2 %
// equivalence tolerance does that (c/a = 1.0005 with 0.002 deg errors: 3-4x;
// anything larger and the symmetric cell no longer fits at all), while the
// extra parameter fitting noise gave at most 1.4x in tests.
const SIEVE_SYMMETRY_RANK = { cubic: 5, hexagonal: 4, tetragonal: 4,
    orthorhombic: 3, monoclinic: 2, triclinic: 1 };
const SIEVE_SYMMETRY_M20_RATIO = 2;
// A pair kept for the user to decide is marked on both cells with plain
// copies of the other's parameters (no object references: rows are copied,
// sorted and erased independently). _symAltOf: on the more symmetric cell,
// the better-fitting lower-symmetry cell(s); _symAlt: the reverse.
const symPairSummary = (s) => ({ system: s.system, lattice: s.lattice || null, a: s.a, b: s.b, c: s.c,
    alpha: s.alpha, beta: s.beta, gamma: s.gamma, volume: s.volume, m20: s.m20 });
const fmtCellBrief = (s) => {
    const f = (v) => Number(v).toFixed(4);
    switch (s.system) {
        case 'cubic': return `a = ${f(s.a)}`;
        case 'tetragonal': case 'hexagonal': return `a = ${f(s.a)}, c = ${f(s.c)}`;
        case 'orthorhombic': return `a = ${f(s.a)}, b = ${f(s.b)}, c = ${f(s.c)}`;
        case 'monoclinic': return `a = ${f(s.a)}, b = ${f(s.b)}, c = ${f(s.c)}, beta = ${Number(s.beta).toFixed(3)}`;
        default: return `a = ${f(s.a)}, b = ${f(s.b)}, c = ${f(s.c)}, ` +
            `angles ${Number(s.alpha).toFixed(3)}, ${Number(s.beta).toFixed(3)}, ${Number(s.gamma).toFixed(3)}`;
    }
};
// Results-table badges naming the other cell(s) of a pair; the tooltip lists them all.
const symPairBadges = (sol) => {
    const cells = (list) => list.map(o => `the ${o.system} cell ${fmtCellBrief(o)} (M20 ${Number(o.m20).toFixed(2)})`)
        .join(' and ');
    const label = (list) => `≡ ${list[0].system.substring(0, 4)}${list.length > 1 ? ` +${list.length - 1}` : ''}`;
    let html = '';
    if (sol._symAltOf && sol._symAltOf.length) {
        html += `<br><span class="sol-badge alt" title="Same lattice as ${cells(sol._symAltOf)}, listed too: ` +
                `${sol._symAltOf.length > 1 ? 'those lower-symmetry descriptions fit' : 'that lower-symmetry description fits'} ` +
                `clearly better. Choose between them.">${label(sol._symAltOf)}</span>`;
    }
    if (sol._symAlt && sol._symAlt.length) {
        html += `<br><span class="sol-badge alt" title="Same lattice as ${cells(sol._symAlt)}, ` +
                `${sol._symAlt.length > 1 ? 'which are' : 'which is'} more symmetric and listed too. This cell fits ` +
                `clearly better: a real distortion, or noise taken up by its extra parameters? ` +
                `Choose between them.">${label(sol._symAlt)}</span>`;
    }
    return html;
};
// An R lattice found by the hexagonal search can be a cubic lattice: P, I or F
// cubic is rhombohedral with alpha = 90, 109.47 or 60 deg (c/a = 1.225,
// 0.612, 2.449 in hexagonal axes). Its triple cell has 3/1, 3/2 or 3/4 of the
// cubic volume, so the same-volume test above never pairs the two; this
// compares their primitive cells instead. Returns true when `r` (hexagonal,
// lattice R) and `cub` (cubic) describe the same lattice.
const RHOMBOHEDRAL_FROM_HEX = [[2 / 3, -1 / 3, -1 / 3], [1 / 3, 1 / 3, -2 / 3], [1 / 3, 1 / 3, 1 / 3]];
const CUBIC_PRIMITIVE = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],                    // P
    [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]], // I
    [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],        // F
];
const _primitiveCache = new WeakMap();   // cell -> Map(transform -> comparison)
const primitiveComparison = (cell, P) => {
    let byP = _primitiveCache.get(cell);
    if (!byP) _primitiveCache.set(cell, (byP = new Map()));
    if (byP.has(P)) return byP.get(P);
    let out = null;
    try {
        const G = metricFromCell({ a: cell.a, b: cell.b ?? cell.a, c: cell.c ?? cell.a,
            alpha: cell.alpha ?? 90, beta: cell.beta ?? 90,
            gamma: cell.gamma ?? (cell.system === 'hexagonal' ? 120 : 90) });
        const prim = cellFromMetric(matMul(matMul(transpose(P), G), P));
        out = prim ? makeLatticeComparison({ ...prim, system: 'triclinic' }) : null;
    } catch (_) { out = null; }       // uncertain -> no match, the candidate is kept
    byP.set(P, out);
    return out;
};
const rhombohedralIsCubic = (r, cub) => {
    if (r.system !== 'hexagonal' || r.lattice !== 'R' || cub.system !== 'cubic') return false;
    const rp = primitiveComparison(r, RHOMBOHEDRAL_FROM_HEX);
    return !!rp && CUBIC_PRIMITIVE.some(P => equivalentLattices(rp, primitiveComparison(cub, P)));
};
// Every system that has cells keeps a reserved share of the list, so that the cells of one system, which can be
// many and score well, never push another system out altogether. The share is the cap divided by the number of
// systems present in `ranked` (best first), so it adapts: with one system that system gets the whole cap, with
// six systems each is guaranteed cap/6. The slots a system does not use go to the best of the rest, by rank.
// The order of `ranked` is kept.
const FINAL_SOLUTION_CAP = 50;
const selectWithSystemReserve = (ranked, cap) => {
    if (ranked.length <= cap) return ranked.slice();
    const groups = new Map();
    for (const s of ranked) {
        const key = s.system || '?';
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(s);
    }
    const reserve = Math.max(1, Math.floor(cap / groups.size));
    const chosen = new Set();
    for (const g of groups.values()) g.slice(0, reserve).forEach(s => chosen.add(s));
    for (const s of ranked) { if (chosen.size >= cap) break; chosen.add(s); }
    return ranked.filter(s => chosen.has(s));
};
const applyFinalSieve = (candidates) => {
    const valid = candidates.filter(s => s && Number.isFinite(s.volume) &&
        s.volume > 0 && Number.isFinite(s.m20) && s.m20 > 0);
    // Compare best-first: an equivalent cell is never kept twice, and a
    // distinct cell merely sharing its volume is never discarded.
    const symmetryOrder = SIEVE_SYMMETRY_RANK;
    const rank = (s) => symmetryOrder[s.system] || 0;
    valid.sort((a, b) => b.m20 - a.m20 || rank(b) - rank(a));
    const kept = [];
    const comparisons = new Map();
    const comparison = cell => {
        if (!comparisons.has(cell)) comparisons.set(cell, makeLatticeComparison(cell));
        return comparisons.get(cell);
    };
    // Pair marks from an earlier sieve are stale: rows come and go between runs.
    for (const s of valid) { delete s._symAlt; delete s._symAltOf; }
    const pairs = [];                 // [more symmetric cell, better-fitting lower-symmetry cell]
    for (const candidate of valid) {
        // Every kept cell describing the same lattice, not just the first: a
        // cubic cell can match both a pseudo-cubic tetragonal one (same
        // volume) and an R cell (different volume), and must replace both.
        //   replace  the candidate is the more symmetric description: it takes over
        //   alt      it is more symmetric, but the kept cell fits more than
        //            SIEVE_SYMMETRY_M20_RATIO times better: both are listed
        //   dup      a kept cell already describes the lattice as symmetrically
        const matches = [];
        for (let i = 0; i < kept.length; i++) {
            const best = kept[i];
            // Merit values from different input data are not comparable.
            if (best._fitContext !== candidate._fitContext ||
                hasRefinedZero(best) !== hasRefinedZero(candidate)) continue;
            const sameVolume = Math.abs(best.volume - candidate.volume) <=
                0.01 * Math.min(best.volume, candidate.volume);
            if (sameVolume && equivalentLattices(comparison(best), comparison(candidate))) {
                // `best` has the higher M(20) (sorted). Both count the same
                // calculated lines, so the figures compare like with like.
                matches.push({ i, kind: rank(candidate) <= rank(best) ? 'dup'
                    : (best.m20 <= SIEVE_SYMMETRY_M20_RATIO * candidate.m20 ? 'replace' : 'alt') });
            } else if (rhombohedralIsCubic(best, candidate) || rhombohedralIsCubic(candidate, best)) {
                // Same lattice, cubic or rhombohedral description: keep the
                // cubic one. Their M(20) do not compare: the R cell counts only
                // R lines, the cubic one every P line, absent F or I ones included.
                matches.push({ i, kind: candidate.system === 'cubic' ? 'replace' : 'dup' });
            }
        }
        if (!matches.length) { kept.push(candidate); continue; }
        if (matches.some(m => m.kind === 'dup')) continue;
        const replaced = new Set(matches.filter(m => m.kind === 'replace').map(m => kept[m.i]));
        const partners = matches.filter(m => m.kind === 'alt').map(m => kept[m.i]);
        // A lower-symmetry description fits clearly better. It stays, and
        // the more symmetric one is listed with it: the user decides.
        for (const p of partners) pairs.push([candidate, p]);
        if (replaced.size) {
            // Pairs of a replaced cell lapse (filtered below). The candidate
            // was compared with each kept cell itself and has its own pairs:
            // inheriting would be wrong for an R cell, whose sieve partner has
            // the hexagonal cell's volume, not the cubic one's.
            const at = kept.findIndex(k => replaced.has(k));
            kept[at] = candidate;
            for (let i = kept.length - 1; i >= 0; i--) if (i !== at && replaced.has(kept[i])) kept.splice(i, 1);
        } else {
            kept.splice(kept.indexOf(partners[0]) + 1, 0, candidate);   // next to its partner
        }
    }
    // Mark the pairs that survived on both cells.
    const inKept = new Set(kept);
    const keptPairs = pairs.filter(([hi, lo]) => inKept.has(hi) && inKept.has(lo));
    for (const [hi, lo] of keptPairs) {
        (hi._symAltOf = hi._symAltOf || []).push(symPairSummary(lo));
        (lo._symAlt = lo._symAlt || []).push(symPairSummary(hi));
    }
    const nPairs = new Set(keptPairs.map(([hi]) => hi)).size;
    const discarded = valid.length - kept.length;
    if (discarded || nPairs) {
        showStatus((discarded ? `Sieve discarded ${discarded} equivalent solution(s).` : '') +
                   (nPairs ? `${discarded ? ' ' : ''}${nPairs} cell(s) kept beside a lower-symmetry ` +
                             `description that fits better (marked ≡): choose between them.` : ''), 'success');
    }
    // Existing final result cap, which never keeps one of a pair without the other.
    const out = selectWithSystemReserve(kept, FINAL_SOLUTION_CAP);
    const inOut = new Set(out);
    for (let grew = true; grew; ) {
        grew = false;
        for (const [hi, lo] of keptPairs) if (inOut.has(hi) !== inOut.has(lo)) {
            const add = inOut.has(hi) ? lo : hi;
            out.push(add); inOut.add(add); grew = true;
        }
    }
    return out;
};
const sortSolutions = () => {
    const { column, direction } = sortState;
    const dir = direction === 'asc' ? 1 : -1;
    solutions.sort((a, b) => {
        if (column === 'system') {
            return (a.system || '').localeCompare(b.system || '') * dir;
        } else {
            let valA = a[column]; let valB = b[column];
            if (isNaN(valA) || valA == null) valA = -Infinity;
            if (isNaN(valB) || valB == null) valB = -Infinity;
            if (valA === valB) return 0;
            return (valA > valB ? 1 : -1) * dir;
        }
    });
};
// One accordion per solution, one open at a time. The open one is the selected
// solution: its calculated lines are drawn on the plot, and the Solution menu acts on it.
// A solution can hold several space groups (chosen in Explore, or reached by Convert); one of them
// is the one simulated on the plot (sol.simSG, a setting_id). Older projects held a single sol.spaceGroup.
const solutionSGs = (sol) => {
    if (sol.spaceGroup && !sol.spaceGroups) { sol.spaceGroups = [sol.spaceGroup]; }
    delete sol.spaceGroup;
    return sol.spaceGroups || (sol.spaceGroups = []);
};
const sgKey = (sg) => sg.setting_id || sg.label;
const addSolutionSG = (sol, sg) => {
    const list = solutionSGs(sol), at = list.findIndex(x => sgKey(x) === sgKey(sg));
    if (at >= 0) list[at] = sg; else list.push(sg);
    sol.simSG = sg.setting_id || '';
};
const removeSolutionSG = (sol, index) => {
    const list = solutionSGs(sol), [gone] = list.splice(index, 1);
    if (gone && sol.simSG === gone.setting_id) sol.simSG = list.length ? (list[list.length - 1].setting_id || '') : '';
};
// The setting simulated for a solution: the chosen one, else the first group the space-group scan proposed.
const defaultSimSetting = (sol) => {
    if (!sol) return '';
    const list = solutionSGs(sol);
    if (list.length) {
        const hit = list.find(x => x.setting_id && x.setting_id === sol.simSG) || list[list.length - 1];
        return hit.setting_id || '';
    }
    if (sol.sgMembers && sol.sgMembers.length && typeof manualSG !== 'undefined') {
        const m = /^(.*?)(?: \((\d+)\))?$/.exec(String(sol.sgMembers[0]));
        const name = m[1].replace(/\s/g, ''), num = m[2];
        const pool = manualSG.entries.filter(e => !num || String(e.group.number) === num);
        const hit = pool.find(e => String(e.setting.hm || e.setting.symbol).replace(/\s/g, '') === name) || pool[0];
        return hit ? hit.setting.setting_id : '';
    }
    return '';
};
const solutionBadges = (sol) => {
    let sysBadges = '';
    if (sol.lattice === 'R') sysBadges += ` <span class="sol-badge" title="Rhombohedral (R) lattice in hexagonal axes: lines with -h+k+l and h-k+l both non-multiples of 3 are absent, and M20 counts only the R lines">R</span>`;
    sysBadges += symPairBadges(sol).replace(/<br>/g, ' ');
    const nSwaps = (sol.manualSwaps || []).length;
    if (nSwaps > 0) sysBadges += ` <span class="sol-badge swap" title="${(sol.manualSwaps||[]).map(x=>x.from+'->'+x.to+' @ '+x.tth.toFixed(3)).join('; ')}">swap&times;${nSwaps}</span>`;
    if (sol.mcPolished && !sol.sgClass) {
        const from = (sol.mcFrom && isFinite(sol.mcFrom.m20)) ? sol.mcFrom.m20.toFixed(2) : '?';
        sysBadges += ` <span class="sol-badge mc" title="Monte-Carlo polished: M20 ${from} -> ${sol.m20.toFixed(2)}">MC</span>`;
    }
    if (sol.sgClass) {
        const decisive = isFinite(sol.sgMargin) && sol.sgMargin >= SG_DECISIVE_NATS;
        sysBadges += ` <span class="sol-badge sg${decisive ? '' : ' sg-tied'}" title="Refined under ${sol.sgClass}">SG${decisive ? '' : '?'}</span>`;
    }
    const sgs = solutionSGs(sol);
    if (sgs.length) sysBadges += ` <span class="sol-badge sg" title="${sgs.map(x => x.label).join('; ')}">${sgs[0].symbol || 'SG'}${sgs.length > 1 ? ` +${sgs.length - 1}` : ''}</span>`;
    return sysBadges;
};
// The space group attached to a solution, as text: the one the user chose
// (Explore, Convert) first, then what Test Space Group established.
const solutionSpaceGroupText = (sol) => {
    const sgs = solutionSGs(sol);
    if (sgs.length) return sgs.map(x => x.label + (x.source ? ` (${x.source})` : '')).join('; ');
    if (sol.sgClass) {
        const decisive = isFinite(sol.sgMargin) && sol.sgMargin >= SG_DECISIVE_NATS;
        const members = sol.sgMembers && sol.sgMembers.length ? ` — ${sol.sgMembers.join(', ')}` : '';
        return `class ${sol.sgClass}${members}${decisive ? '' : ' (not decisive)'}`;
    }
    return '';
};
const niggliText = (sol) => {
    try {
        if (!sol._niggli) sol._niggli = reduceToNiggliCell(sol).cell;
        const n = sol._niggli;
        return `a ${n.a.toFixed(4)}, b ${n.b.toFixed(4)}, c ${n.c.toFixed(4)} Å; α ${n.alpha.toFixed(3)}, β ${n.beta.toFixed(3)}, γ ${n.gamma.toFixed(3)}°; V ${getVolumeTriclinic(n).toFixed(2)} Å³`;
    } catch (e) { return 'not available'; }
};
// The space groups of a solution: one radio per group (the one drawn as Sim markers), a remove button each,
// then what the space-group scan proposed.
const solutionSGHtml = (sol) => {
    const sgs = solutionSGs(sol), shown = defaultSimSetting(sol);
    const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;');
    let html = sgs.map((g, i) => `<div class="sg-item"><label title="Show this group's simulated reflections (Sim)">` +
        `<input type="radio" name="sim-${solutions.indexOf(sol)}" data-sg-show="${i}"${g.setting_id && g.setting_id === shown ? ' checked' : ''}> ${esc(g.label)}</label>` +
        `<button type="button" class="sg-remove" data-sg-remove="${i}" title="Remove from this solution" aria-label="Remove">&times;</button>` +
        (g.source ? `<div class="sg-src">${esc(g.source)}</div>` : '') + `</div>`).join('');
    if (sol.sgClass) {
        const decisive = isFinite(sol.sgMargin) && sol.sgMargin >= SG_DECISIVE_NATS;
        html += `<div class="sg-src">Proposed by Test Space Group: class ${esc(sol.sgClass)}${sol.sgMembers && sol.sgMembers.length ? ' &mdash; ' + esc(sol.sgMembers.join(', ')) : ''}${decisive ? '' : ' (not decisive)'}</div>`;
    }
    return html || '<span class="sol-none">none &mdash; Solution &rarr; Test Space Group or Explore Space Group</span>';
};
const solutionBodyHtml = (sol) => {
    const b = sol.b ?? sol.a, c = sol.c ?? sol.a;
    const al = sol.alpha ?? 90, be = sol.beta ?? 90, ga = sol.gamma ?? (sol.system === 'hexagonal' ? 120 : 90);
    const row = (k, v, cls = '') => `<dt>${k}</dt><dd${cls ? ` class="${cls}"` : ''}>${v}</dd>`;
    return '<dl class="sol-dl">' +
        row('Symmetry', sol.system + (sol.lattice === 'R' ? ', R lattice (hexagonal axes)' : '')) +
        row('Cell', `a ${sol.a.toFixed(4)}, b ${b.toFixed(4)}, c ${c.toFixed(4)} Å<br>α ${al.toFixed(3)}, β ${be.toFixed(3)}, γ ${ga.toFixed(3)}°`) +
        row('Volume', `${sol.volume.toFixed(2)} Å³`) +
        row('M(20)', sol.m20.toFixed(2)) +
        (sol.zero_correction ? row('Zero', `${sol.zero_correction.toFixed(4)}°`) : '') +
        row('Niggli', niggliText(sol)) +
        row('Space groups', solutionSGHtml(sol)) +
        '</dl>';
};
const solutionSummaryHtml = (sol) => {
    const cellText = (({
        cubic: s => `a ${s.a.toFixed(3)}`,
        tetragonal: s => `a ${s.a.toFixed(3)}, c ${s.c.toFixed(3)}`,
        hexagonal: s => `a ${s.a.toFixed(3)}, c ${s.c.toFixed(3)}`,
    })[sol.system] || (s => `${s.a.toFixed(2)}, ${(s.b ?? s.a).toFixed(2)}, ${(s.c ?? s.a).toFixed(2)}`))(sol);
    return `<span class="sol-sys">${sol.system}</span><span class="sol-cell">${cellText}</span>` +
        `<span class="sol-v">V ${sol.volume.toFixed(0)}</span><span class="sol-m">M20 ${sol.m20.toFixed(1)}</span>${solutionBadges(sol)}`;
};
const updateSolutionsTable = () => {
    const html = displayedSolutions.map((sol, index) => {
        if (!sol || !sol.system) return '';
        const open = selectedSolution === sol;
        return `<details class="sol-acc" data-index="${index}"${open ? ' open' : ''}><summary>${solutionSummaryHtml(sol)}</summary>` +
            `<div class="sol-body">${open ? solutionBodyHtml(sol) : ''}</div></details>`;
    }).join('');
    ui.solutionsTableBody.innerHTML = html || '<p class="hint sol-empty">No solutions yet. Index a pattern to list candidate cells here.</p>';
};
// Opening an accordion selects its solution and closes the others. Driven by the click on the
// summary (not by the details toggle event, which also fires when a table is rendered with one open).
ui.solutionsTableBody.addEventListener('click', (e) => {
    const summary = e.target.closest('details.sol-acc > summary');
    if (!summary) return;
    e.preventDefault();
    const det = summary.parentElement;
    if (det.open) {
        // Closing the open solution deselects it: no markers, and the Solution menu is off.
        const open = displayedSolutions[parseInt(det.dataset.index)];
        det.open = false;
        if (open && open === selectedSolution) applySolutionSelection(null);
        return;
    }
    const sol = displayedSolutions[parseInt(det.dataset.index)];
    if (sol) applySolutionSelection(sol);   // syncSolutionAccordion opens this one and closes the others
});
// Space groups of the open solution: show one as Sim markers, or remove one.
const afterSGChange = (sol) => {
    manualSGSelect.value = defaultSimSetting(sol);
    rebuildManualSG(); updateAllMarkers(); updateSolutionsTable();
};
ui.solutionsTableBody.addEventListener('click', (e) => {
    const rm = e.target.closest('[data-sg-remove]');
    if (!rm || isIndexing) return;
    const sol = displayedSolutions[parseInt(rm.closest('details.sol-acc').dataset.index)];
    if (!sol) return;
    removeSolutionSG(sol, Number(rm.dataset.sgRemove));
    afterSGChange(sol);
});
ui.solutionsTableBody.addEventListener('change', (e) => {
    const pick = e.target.closest('[data-sg-show]');
    if (!pick || isIndexing) return;
    const sol = displayedSolutions[parseInt(pick.closest('details.sol-acc').dataset.index)];
    if (!sol) return;
    sol.simSG = solutionSGs(sol)[Number(pick.dataset.sgShow)].setting_id || '';
    afterSGChange(sol);
});
// Keeps the accordions in step when a solution is selected from elsewhere.
const syncSolutionAccordion = () => {
    ui.solutionsTableBody.querySelectorAll('details.sol-acc').forEach(det => {
        const sol = displayedSolutions[parseInt(det.dataset.index)];
        const want = !!sol && sol === selectedSolution;
        if (want && !det.open) det.open = true;
        else if (!want && det.open) det.open = false;
        const body = det.querySelector('.sol-body');
        if (want && body && !body.firstChild) body.innerHTML = solutionBodyHtml(sol);
    });
};
