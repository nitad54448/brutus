// js/dialogs/context-menu.js
// Right-click menu on the solutions table.
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// Context Menu Logic
const contextMenu = document.getElementById('context-menu');
let ctxMenuTargetIndex = -1;
// Row indices index into displayedSolutions (what is rendered). Resolving
// the object here and then looking it up by identity in `solutions` keeps
// erase / swap / MC / report acting on the row the user actually clicked.
const ctxTargetSolution = () =>
    (ctxMenuTargetIndex > -1 ? (displayedSolutions[ctxMenuTargetIndex] || null) : null);
// 1. Show Menu on Right Click
ui.solutionsTableBody.addEventListener('contextmenu', (e) => {
    const row = e.target.closest('tr');
    if (!row) return;
    
    e.preventDefault(); // Stop default browser menu
    
    // Select the row visually (optional, but good UX)
    document.querySelectorAll('#solutions-table-body tr').forEach(r => r.classList.remove('selected'));
    row.classList.add('selected');
    
    // Store the index. Row indices address displayedSolutions; every action
    // below resolves through ctxTargetSolution() so it can never act on a
    // different element of `solutions` if the two lists drift apart.
    ctxMenuTargetIndex = parseInt(row.dataset.index);
    // Right-clicking a row selects it too, so it must refresh the calculated
    // lines exactly like a left click. It used to assign selectedSolution on
    // its own and leave currentHklList pointing at the previously selected
    // cell, drawing one solution's ticks over another's.
    applySolutionSelection(displayedSolutions[ctxMenuTargetIndex]);
    
    // Position and show menu
    contextMenu.style.top = `${e.pageY}px`;
    contextMenu.style.left = `${e.pageX}px`;
    contextMenu.style.display = 'block';
});
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
