// js/dialogs/context-menu.js
// Right-click menu on the solutions table.
//
// Classic script, loaded in order by index.html (see the list there); its
// top-level names are shared with the other app scripts.

// Context Menu Logic
const contextMenu = document.getElementById('context-menu');
let ctxMenuTargetIndex = -1;
// Row indices index into displayedSolutions (what is rendered). Resolving
// the object here and then looking it up by identity in `solutions` keeps
// erase / swap / MC / report acting on the row the user actually clicked.
const ctxTargetSolution = () =>
    (ctxMenuTargetIndex > -1 ? (displayedSolutions[ctxMenuTargetIndex] || null) : null);
// 1. The actions are reached from the Solution menu in the header (js/core/shell.js),
// which points ctxMenuTargetIndex at the selected solution and clicks the matching
// ctx-* element below. The element stays hidden: it is only the hook.
const solutionMenuTarget = (id) => {
    if (typeof isIndexing !== 'undefined' && isIndexing) return;
    if (!selectedSolution) return;
    ctxMenuTargetIndex = displayedSolutions.indexOf(selectedSolution);
    document.getElementById(id).click();
};
// 2. Hide Menu on any click elsewhere
document.addEventListener('click', () => {
    contextMenu.style.display = 'none';
});
// 3. Action: Erase Solution
document.getElementById('ctx-erase').addEventListener('click', () => {
    const target = ctxTargetSolution();
    if (target) {
        // Remove from main list by identity, not by rendered row index.
        const k = solutions.indexOf(target);
        if (k > -1) solutions.splice(k, 1);
        // Its partner in a sieve pair (js/indexing/solutions.js) no longer has
        // a second description listed: drop the mark that names it.
        if (target._symAlt || target._symAltOf) {
            const same = (x) => x.system === target.system && x.a === target.a && x.m20 === target.m20;
            for (const s of solutions) for (const f of ['_symAlt', '_symAltOf']) {
                if (!s[f]) continue;
                s[f] = s[f].filter(x => !same(x));
                if (!s[f].length) delete s[f];
            }
        }

        // Re-sync displayed list
        displayedSolutions = [...solutions]; 
        
        foundSolutionMap.clear(); //= vérifier.. une fois une solution effacée...?

        if (selectedSolution === target) {
            selectedSolution = null;
            currentHklList = [];
        }

        updateSolutionsTable();
        updateAllMarkers(); // Clear blue lines if we deleted the selected one
        ctxMenuTargetIndex = -1;
    }
});
