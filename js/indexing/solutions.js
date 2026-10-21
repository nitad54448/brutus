// js/indexing/solutions.js
// The solution ledger: intake from workers, pruning, sorting and the results table.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

ui.solutionsTableHeaders.forEach(header => {
    header.addEventListener('click', () => {
        const column = header.dataset.sort;
        if (!column) return;

        if (sortState.column === column) {
            sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
        } else {
            sortState.column = column;
            sortState.direction = (column === 'm20' || column === 'volume') ? 'desc' : 'asc';
        }

        sortSolutions();
  //          const selectedSystems = Array.from(ui.systemCheckboxes)
  //                                       .filter(cb => cb.checked)
  //                                       .map(cb => cb.value);
        // A COPY, not an alias. handleNewSolution does
        // `solutions = solutions.slice(...)` when pruning, which rebinds
        // `solutions` to a new array and leaves an aliased
        // displayedSolutions pointing at the stale one -- after which the
        // rendered table and the array the context menu indexes into are
        // two different lists.
        displayedSolutions = [...solutions];
        updateSolutionsTable();
    });
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
        solutions = solutions.slice(0, PRUNE_TO_COUNT);
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
    const out = kept.slice(0, 50);
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
const updateSolutionsTable = () => {
    // Build the entire table markup in memory first
    const rowsHtml = displayedSolutions.map((sol, index) => {
        if (!sol || !sol.system) return '';
        let paramsCell = '', anglesCell = '';
        switch(sol.system) {
            case 'cubic': paramsCell = `a = ${sol.a.toFixed(4)}`; anglesCell = `90, 90, 90`; break;
            case 'tetragonal': paramsCell = `a = ${sol.a.toFixed(4)}, c = ${sol.c.toFixed(4)}`; anglesCell = `90, 90, 90`; break;
            case 'hexagonal': paramsCell = `a = ${sol.a.toFixed(4)}, c = ${sol.c.toFixed(4)}`; anglesCell = `90, 90, 120`; break;
            case 'orthorhombic': paramsCell = `a = ${sol.a.toFixed(4)}<br>b = ${sol.b.toFixed(4)}<br>c = ${sol.c.toFixed(4)}`; anglesCell = `90, 90, 90`; break;
            case 'monoclinic': paramsCell = `a = ${sol.a.toFixed(4)}<br>b = ${sol.b.toFixed(4)}<br>c = ${sol.c.toFixed(4)}`; anglesCell = `90, ${sol.beta.toFixed(3)}, 90`; break;
            case 'triclinic': 
                paramsCell = `a = ${sol.a.toFixed(4)}<br>b = ${sol.b.toFixed(4)}<br>c = ${sol.c.toFixed(4)}`; 
                anglesCell = `&alpha; = ${sol.alpha.toFixed(3)}<br>&beta; = ${sol.beta.toFixed(3)}<br>&gamma; = ${sol.gamma.toFixed(3)}`; 
                break;
            default: paramsCell = `${sol.a.toFixed(4)}`; anglesCell = `-`;
        }
        if (sol.zero_correction) {
            anglesCell += `<br><span style="font-size:0.9em; color: var(--text-dark);">(Z=${sol.zero_correction.toFixed(4)}°)</span>`;
        }
        const isSelected = (selectedSolution === sol) ? ' class="selected"' : '';
        // Mark solutions produced by Explore Group, with the number of swaps
        // applied, so a derived cell is never mistaken for an independent hit.
        const nSwaps = (sol.manualSwaps || []).length;
        let sysCell = sol.system.substring(0,4);
        // A hexagonal cell that is an R lattice (M20 counted on R lines only).
        if (sol.lattice === 'R') sysCell += ` <span class="sol-badge" title="Rhombohedral (R) lattice in hexagonal axes: lines with -h+k+l and h-k+l both non-multiples of 3 are absent, and M20 counts only the R lines">R</span>`;
        // Two descriptions of one lattice, both kept by the final sieve because
        // the lower-symmetry one fits more than SIEVE_SYMMETRY_M20_RATIO times
        // better: each row names the other, and the user decides.
        sysCell += symPairBadges(sol);
        if (nSwaps > 0) {
            sysCell += `<br><span class="sol-badge swap" title="${(sol.manualSwaps||[]).map(x=>x.from+'->'+x.to+' @ '+x.tth.toFixed(3)).join('; ')}">swap&times;${nSwaps}</span>`;
        }
        // Cells improved by the Monte-Carlo polish are flagged so a derived
        // solution is never mistaken for an independent hit.
        // Suppressed when an SG badge is also due: a cell adopted from the
        // space-group scan is nearly always MC-polished as well, and two
        // badges saying almost the same thing crowd the column. The M20
        // provenance is folded into the SG tooltip instead, so nothing is lost.
        if (sol.mcPolished && !sol.sgClass) {
            const from = (sol.mcFrom && isFinite(sol.mcFrom.m20)) ? sol.mcFrom.m20.toFixed(2) : '?';
            sysCell += `<br><span class="sol-badge mc" title="Monte-Carlo polished: M20 ${from} -> ${sol.m20.toFixed(2)}">MC</span>`;
        }
        // Refined under a space-group hypothesis: the forbidden reflections
        // were removed from the line list before fitting, so the cell, the
        // pairing and M20 only mean anything alongside the class they assume.
        if (sol.sgClass) {
            const bits = [`Refined under ${sol.sgClass}`];
            if (sol.sgMembers && sol.sgMembers.length) bits.push(sol.sgMembers.join(', '));
            if (sol.sgConditions && sol.sgConditions.length) bits.push(sol.sgConditions.join(' ; '));
            // The margin decides whether this badge is a result or a guess.
            // A class that led its table by half a nat was not established by
            // the data, and the badge must not let the cell quietly acquire
            // the authority of one that was.
            const ev = sol.sgEvidence;
            if (ev) {
                bits.push(`${ev.clean}/${ev.informative} forbidden lines clean, ` +
                          `${ev.hardViolations} hard violation(s)` +
                          (ev.softViolations ? ` (+${ev.softViolations} soft)` : '') +
                          (ev.unindexed ? `, ${ev.unindexed} unindexed` : ''));
                bits.push(ev.wilson ? 'absences weighted per reflection (Wilson)'
                                    : (isFinite(ev.pHat) ? `p(line observed) = ${(ev.pHat * 100).toFixed(0)}%` : ''));
                if (ev.mode !== 'mc') bits.push('stage-1 (least-squares) result only');
            }
            const decisive = isFinite(sol.sgMargin) && sol.sgMargin >= SG_DECISIVE_NATS;
            if (isFinite(sol.sgMargin)) {
                bits.push(decisive
                    ? `${sol.sgMargin.toFixed(1)} nats ahead of the runner-up`
                    : `NOT decisive: only ${sol.sgMargin.toFixed(1)} nats ahead of the runner-up, ` +
                      `so the absences do not choose between this class and the next`);
            }
            if (sol.mcPolished && sol.mcFrom && isFinite(sol.mcFrom.m20)) {
                bits.push(`MC: M20 ${sol.mcFrom.m20.toFixed(2)} -> ${sol.m20.toFixed(2)}`);
            }
            sysCell += `<br><span class="sol-badge sg${decisive ? '' : ' sg-tied'}" ` +
                       `title="${bits.join(' | ')}">SG${decisive ? '' : '?'}</span>`;
        }
        return `<tr data-index="${index}"${isSelected}><td>${sysCell}</td><td>${paramsCell}</td><td>${anglesCell}</td><td>${sol.volume.toFixed(2)}</td><td>${sol.m20.toFixed(2)}</td></tr>`;
    }).join('');
    
    // Single DOM write
    ui.solutionsTableBody.innerHTML = rowsHtml;
    
    ui.solutionsTableHeaders.forEach(h => {
        h.classList.remove('sort-asc', 'sort-desc');
        if (h.dataset.sort === sortState.column) {
           h.classList.add(sortState.direction === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
};
ui.solutionsTableBody.addEventListener('click', (e) => {
    const row = e.target.closest('tr'); if (!row) return;
    document.querySelectorAll('#solutions-table-body tr').forEach(r => r.classList.remove('selected'));
    row.classList.add('selected');
    const index = parseInt(row.dataset.index);
    // applySolutionSelection rebuilds the line list with the refined zero
    // folded in. The old inline version also took its 2-theta ceiling from
    // the current x-axis maximum, so zooming in and then picking a solution
    // silently truncated the calculated pattern at the edge of the view.
    applySolutionSelection(displayedSolutions[index]);
});
