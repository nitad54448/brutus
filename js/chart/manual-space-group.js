// Manual hypotheses are display-only. Enumerate the full metric in the displayed
// basis so alternative monoclinic axes and lower-symmetry settings are not folded.
const manualSG = { entries: [], lines: [], label: '', owner: null };
const manualSGSearch = document.getElementById('manual-sg-search');
const manualSGSelect = document.getElementById('manual-sg-select');
const manualSGStatus = document.getElementById('manual-sg-status');
function populateManualSG() {
    if (!spaceGroupData) return;
    manualSG.entries = Object.values(spaceGroupData.space_groups).flatMap(group =>
        group.settings.map(setting => ({ group, setting, label:
            `#${group.number} ${setting.hm || setting.symbol} — ${setting.description.replace(/\0/g, '')} [${setting.setting_id}]` })));
    filterManualSG();
}
function filterManualSG() {
    const previous = manualSGSelect.value;
    const query = manualSGSearch.value.trim().toLowerCase().replace(/\s/g, '');
    manualSGSelect.replaceChildren(new Option('No manual comparison', ''));
    for (const entry of manualSG.entries) {
        const haystack = (entry.label + entry.setting.hall).toLowerCase().replace(/\s/g, '');
        if (haystack.includes(query) || entry.setting.setting_id === previous)
            manualSGSelect.add(new Option(entry.label, entry.setting.setting_id));
    }
    manualSGSelect.value = previous;
}
function manualSGReflections(cell, setting, wavelength, maxTth) {
    const previousFilter = getSpaceGroupFilter();
    sgEnsureDatabase(spaceGroupData);
    try {
        setSpaceGroupFilter(sgOpsLabelPredicate(setting));
        return generateHKL_for_analysis({ ...cell, b: cell.b ?? cell.a, c: cell.c ?? cell.a,
            alpha: cell.alpha ?? 90, beta: cell.beta ?? 90,
            gamma: cell.gamma ?? (cell.system === 'hexagonal' ? 120 : 90),
            system: 'triclinic', lattice: 'P' }, wavelength, maxTth);
    } finally { setSpaceGroupFilter(previousFilter); }
}
function rebuildManualSG() {
    manualSG.lines = [];
    manualSGSearch.disabled = manualSGSelect.disabled = !selectedSolution || !spaceGroupData;
    if (manualSG.owner !== selectedSolution) {
        manualSG.owner = selectedSolution;
        manualSGSelect.value = defaultSimSetting(selectedSolution);
    }
    const entry = manualSG.entries.find(e => e.setting.setting_id === manualSGSelect.value);
    if (!selectedSolution || !entry || !workingExperimentalData.tth.length) {
        manualSGStatus.textContent = !spaceGroupData ? 'Space-group database unavailable.' :
            !selectedSolution ? 'Select a solution to compare reflection positions.' :
            `${manualSG.entries.length} settings available. Choose one to show green comparison ticks.`;
        return;
    }
    try {
        const zero = selectedSolution.zero_correction || 0;
        const lambda = getLambda();
        const maxTth = Math.min(179.9, workingExperimentalData.tth.at(-1) + Math.abs(zero) + 1);
        manualSG.lines = manualSGReflections(selectedSolution, entry.setting, lambda, maxTth)
            .map(r => ({ ...r, tth_calc: r.tth, tth: r.tth + zero }));
        placeKa1Lines(manualSG.lines, lambda, zero);
        manualSG.label = entry.label;
        manualSGStatus.textContent = `${entry.label}: ${manualSG.lines.length} allowed reflections. Green ticks on the plot; blue ticks retain the selected solution. Check that the cell matches this setting's axes and metric.`;
    } catch (error) {
        manualSGStatus.textContent = `Could not calculate comparison: ${error.message}`;
    }
}
manualSGSearch.addEventListener('input', filterManualSG);
manualSGSelect.addEventListener('change', () => { rebuildManualSG(); updateAllMarkers(); });
Chart.register({ id: 'manualSpaceGroup', afterDraw(chart) {
    if (!selectedSolution || manualSG.owner !== selectedSolution || !manualSG.lines.length) return;
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = document.documentElement.dataset.theme === 'dark' ? '#80c9bc' : '#15716a'; ctx.lineWidth = 1;
    ctx.font = '12px Segoe UI, sans-serif';
    ctx.fillText(`Sim: ${manualSG.label}`, chartArea.left + 6, chartArea.top + 15,
        Math.max(0, chartArea.right - chartArea.left - 12));
    ctx.restore();
} });
