// js/chart/help-panel.js
// The chart help panel ("?" button).
//
// Classic script, loaded in order by brutus.html (see the list there); its
// top-level names are shared with the other app scripts.

// Chart help: click the "?" button to toggle the panel (no hover trigger).
const helpBtn = document.getElementById('help-icon-btn');
const helpPanel = document.getElementById('help-tooltip');
if (helpBtn && helpPanel) {
    const closeHelp = () => { helpPanel.classList.remove('open'); helpBtn.setAttribute('aria-expanded', 'false'); };
    helpBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const open = helpPanel.classList.toggle('open');
        helpBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // Clicking anywhere outside the panel or the button closes it.
    document.addEventListener('click', (e) => {
        if (!helpPanel.contains(e.target) && e.target !== helpBtn) closeHelp();
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeHelp(); });
}
