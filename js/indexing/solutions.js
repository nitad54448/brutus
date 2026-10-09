// js/indexing/solutions.js
// The solution ledger: intake from workers, pruning, sorting and the results table.
//
// Classic script, loaded in order by brutus.html (see the list there); its
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
const applyFinalSieve = (candidates) => {
    const valid = candidates.filter(s => s && Number.isFinite(s.volume) &&
        s.volume > 0 && Number.isFinite(s.m20) && s.m20 > 0);
    // Compare best-first: an equivalent lower-quality cell is discarded,
    // never a distinct cell merely sharing its volume. Symmetry breaks ties.
    const symmetryOrder = { cubic: 5, hexagonal: 4, tetragonal: 4,
        orthorhombic: 3, monoclinic: 2, triclinic: 1 };
    valid.sort((a, b) => b.m20 - a.m20 ||
        (symmetryOrder[b.system] || 0) - (symmetryOrder[a.system] || 0));
    const kept = [];
    const comparisons = new Map();
    const comparison = cell => {
        if (!comparisons.has(cell)) comparisons.set(cell, makeLatticeComparison(cell));
        return comparisons.get(cell);
    };
    for (const candidate of valid) {
        const duplicate = kept.some(best => {
            // Merit values from different input data are not comparable.
            if (best._fitContext !== candidate._fitContext ||
                hasRefinedZero(best) !== hasRefinedZero(candidate)) return false;
            if (Math.abs(best.volume - candidate.volume) >
                0.01 * Math.min(best.volume, candidate.volume)) return false;
            return equivalentLattices(comparison(best), comparison(candidate));
        });
        if (!duplicate) kept.push(candidate);
    }
    const discarded = valid.length - kept.length;
    if (discarded) showStatus(`Sieve discarded ${discarded} equivalent solution(s).`, 'success');
    return kept.slice(0, 50); // Existing final result cap.
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
