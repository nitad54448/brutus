// js/core/shell.js
// The window around the plot: header menus, the Index / Stop button, the
// indexing lock, the plot tool strip, and the pattern metadata.
//
// Everything here forwards to controls that already exist (the hidden Start and
// Report buttons, the ctx-* hooks, the sg-browser dialog), so each action has
// one implementation.
//
// Classic script, loaded after js/core/workspace.js; shares the global scope.

const shellNode = id => document.getElementById(id);
let loadedFileMeta = null;   // set by js/data/file-load.js: { size, radiation }

// ---- header buttons ------------------------------------------------------
const SOLUTION_ACTIONS = {
    refine: () => solutionMenuTarget('ctx-mc'),
    'test-sg': () => solutionMenuTarget('ctx-sg'),
    convert: () => { if (typeof openConvertDialog === 'function') openConvertDialog(); },
    explore: () => shellNode('browse-sg').click(),
    report: () => solutionMenuTarget('ctx-report'),
    pdcif: () => solutionMenuTarget('ctx-pdcif'),
    erase: () => solutionMenuTarget('ctx-erase'),
};
function syncHeader() {
    const run = ui.startIndexingButton, rep = ui.reportButton;
    const hbRun = shellNode('hb-run'), hbRep = shellNode('hb-report');
    document.body.classList.toggle('is-indexing', isIndexing);
    if (isIndexing) {
        hbRun.textContent = '■ Stop'; hbRun.disabled = false; hbRun.classList.add('is-stop');
        hbRun.title = 'Stop the search and keep what was found (Esc)';
    } else {
        const label = run.textContent.trim();
        hbRun.classList.remove('is-stop');
        hbRun.textContent = '▶ Index';
        hbRun.disabled = run.disabled;
        // A disabled button says why: "Need 4 more peaks", a GPU limit, or no data yet.
        hbRun.title = run.disabled ? label : 'Start the indexing search';
    }
    hbRep.disabled = isIndexing || rep.disabled;
    hbRep.title = selectedSolution ? 'PDF report of the open solution' : `PDF report of all ${displayedSolutions.length} listed solutions`;
    // The Solution menu exists only for an open solution.
    const enabled = !!selectedSolution && !isIndexing;
    document.querySelectorAll('[data-sol-action]').forEach(b => { b.disabled = !enabled; });
    const menu = shellNode('solution-menu');
    menu.classList.toggle('is-disabled', !enabled);
    menu.querySelector('summary').setAttribute('aria-disabled', String(!enabled));
    if (!enabled) menu.open = false;
}
shellNode('hb-run').addEventListener('click', () => {
    if (isIndexing) ui.reportButton.click();      // the report button is the Stop button during a run
    else ui.startIndexingButton.click();
});
// Report: the open solution only; with none open, every listed solution.
shellNode('hb-report').addEventListener('click', () => {
    if (isIndexing) return;
    if (selectedSolution) solutionMenuTarget('ctx-report'); else ui.reportButton.click();
});
shellNode('help-guide').addEventListener('click', () => window.open('brutus_help.html', '_blank', 'noopener'));
document.querySelectorAll('[data-sol-action]').forEach(b => b.addEventListener('click', () => {
    if (isIndexing || b.disabled) return;
    SOLUTION_ACTIONS[b.dataset.solAction]();
}));
// A run owns the window: Esc stops it, and nothing else is accepted meanwhile.
document.addEventListener('keydown', e => {
    if (!isIndexing) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.menubar details[open]').forEach(d => { d.open = false; });
        }
        return;
    }
    if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); ui.reportButton.click(); return; }
    if (e.ctrlKey || e.metaKey || e.key === 'Enter' || e.key === 'Delete') { e.preventDefault(); e.stopImmediatePropagation(); }
}, true);
// A disabled menu does not open, by mouse or keyboard.
document.addEventListener('click', e => {
    const summary = e.target.closest('.workspace-menu.is-disabled > summary');
    if (summary) e.preventDefault();
}, true);
// Menus: one open at a time; picking an item closes it.
document.addEventListener('click', e => {
    document.querySelectorAll('.menubar details[open]').forEach(d => {
        if (!d.contains(e.target) || e.target.closest('.menu-items button')) d.open = false;
    });
}, true);
document.querySelectorAll('.menubar details').forEach(d => d.addEventListener('toggle', () => {
    if (d.open) document.querySelectorAll('.menubar details[open]').forEach(o => { if (o !== d) o.open = false; });
}));
shellNode('btn-about').addEventListener('click', () => {
    if (isIndexing) return;
    showWorkspaceView(workspaceView === 'about' ? 'plot' : 'about');
});
const headerObserver = new MutationObserver(syncHeader);
// During a run the left of the status bar follows the progress line.
new MutationObserver(renderWorkspaceState).observe(statusTextElement, { childList: true, characterData: true, subtree: true });
for (const el of [ui.startIndexingButton, ui.reportButton])
    headerObserver.observe(el, { attributes: true, childList: true, characterData: true, subtree: true });
const baseRefreshStatus = refreshWorkspaceStatus;
refreshWorkspaceStatus = function () { baseRefreshStatus(); syncHeader(); };

// ---- plot tool strip -----------------------------------------------------
function syncPlotTools() {
    document.querySelectorAll('[data-yscale]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.yscale === ui.yAxisMode.value)));
}
document.querySelectorAll('[data-yscale]').forEach(b => b.addEventListener('click', () => {
    if (isIndexing) return;
    ui.yAxisMode.value = b.dataset.yscale;
    ui.yAxisMode.dispatchEvent(new Event('change'));
    syncPlotTools();
}));
shellNode('save-png').addEventListener('click', () => {
    if (!xrdChart) return;
    const src = xrdChart.canvas, out = document.createElement('canvas');
    out.width = src.width; out.height = src.height;
    const ctx = out.getContext('2d');
    ctx.fillStyle = getComputedStyle(shellNode('results-area')).backgroundColor || '#fff';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(src, 0, 0);
    const link = document.createElement('a');
    link.href = out.toDataURL('image/png');
    link.download = (loadedFileName || 'pattern').replace(/\.[^.]+$/, '') + '.png';
    link.click();
});

// ---- pattern metadata (Data tab) ----------------------------------------
const FORMAT_BY_EXT = { xy: 'XY, two columns', csv: 'CSV, two columns', dat: 'Generic / FullProf .dat', txt: 'Text, two columns',
    asc: 'ASCII', prn: 'ASCII', xrdml: 'Panalytical XRDML', xml: 'Bruker RawData XML', brml: 'Bruker BRML', ras: 'Rigaku RAS', esd: 'GSAS ESD',
    std: 'GSAS STD', gsa: 'GSAS', xra: 'GSAS', udf: 'Philips UDF', uxd: 'Bruker UXD', cif: 'pdCIF' };
function renderFileMeta() {
    const dl = shellNode('file-meta');
    if (!dl) return;
    const acc = shellNode('file-meta-acc');
    if (!loadedFileName || !fullExperimentalData.tth.length) { dl.replaceChildren(); acc.open = false; return; }
    const t = fullExperimentalData.tth, y = fullExperimentalData.intensity;
    const steps = t.slice(1).map((v, i) => v - t[i]).sort((a, b) => a - b);
    const step = steps.length ? steps[Math.floor(steps.length / 2)] : NaN;
    const ext = (loadedFileName.match(/\.([^.]+)$/) || [])[1];
    let imin = Infinity, imax = -Infinity, isum = 0;
    for (const v of y) { if (v < imin) imin = v; if (v > imax) imax = v; isum += v; }
    const rad = loadedFileMeta && loadedFileMeta.radiation;
    // A file can record both a single λ and the doublet (RAS does): show all of it.
    const radText = !rad ? 'none in the file (current setting kept)'
        : [rad.single && `${Number(rad.single).toFixed(5)} Å`,
           rad.ka1 && `Kα1 ${Number(rad.ka1).toFixed(5)} Å`, rad.ka2 && `Kα2 ${Number(rad.ka2).toFixed(5)} Å`,
           rad.ratio != null && `ratio ${rad.ratio}`].filter(Boolean).join(', ');
    const preset = ui.wavelengthPreset.options[ui.wavelengthPreset.selectedIndex];
    const rows = [
        ['File', loadedFileName],
        ['Format', (ext && FORMAT_BY_EXT[ext.toLowerCase()]) || (ext ? ext.toUpperCase() : 'unknown')],
        ['Size', loadedFileMeta ? formatFileSize(loadedFileMeta.size) : ''],
        ['Points', t.length.toLocaleString('en-US')],
        ['2θ range', `${t[0].toFixed(3)} – ${t[t.length - 1].toFixed(3)}°`],
        ['Step', isFinite(step) ? `${step.toFixed(4)}° (median)` : ''],
        ['Intensity', `${imin.toLocaleString('en-US')} – ${imax.toLocaleString('en-US')} (total ${Math.round(isum).toLocaleString('en-US')})`],
        ['Radiation in file', radText],
        ['Radiation used', `${preset ? preset.text : ''}, ${Number(ui.wavelength.value).toFixed(5)} Å`],
    ];
    dl.replaceChildren(...rows.filter(r => r[1] !== '').flatMap(([k, v]) => {
        const dt = document.createElement('dt'), dd = document.createElement('dd');
        dt.textContent = k; dd.textContent = v; return [dt, dd];
    }));
}
[ui.wavelengthPreset, ui.wavelength].forEach(el => el.addEventListener('change', renderFileMeta));

syncPlotTools(); syncHeader(); renderFileMeta();

// The status bar follows the split between the panels (see #workspace-state in workspace.css): its width
// is the controls panel plus the divider, wherever the user drags it.
(function trackSplit() {
    const panel = shellNode('controls-panel'), handle = shellNode('drag-handle');
    const sync = () => document.documentElement.style.setProperty('--split-x', (panel.offsetWidth + handle.offsetWidth) + 'px');
    new ResizeObserver(sync).observe(panel);
    sync();
})();
