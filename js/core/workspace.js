// Workspace chrome and portable, versioned local projects.
const workspaceLog = [];
let workspaceHistory = [];
let workspaceRunStarted = null;
let workspaceView = 'plot';
const workspaceNode = id => document.getElementById(id);
// The status bar: what the program is doing or last did (left), the open solution (right).
let workspaceStateMessage = '';
function setWorkspaceState(message) { workspaceStateMessage = message || ''; renderWorkspaceState(); }
function renderWorkspaceState() {
    const el = workspaceNode('workspace-state');
    if (!el) return;
    let text = workspaceStateMessage;
    if (isIndexing) text = (statusTextElement && statusTextElement.textContent) || 'Indexing…';
    else if (!text) text = loadedFileName
        ? `Pattern loaded · ${pickedPeaks.length} peaks · ${solutions.length} solution${solutions.length === 1 ? '' : 's'}`
        : 'Ready. Load a pattern with File → Load pattern.';
    el.textContent = text; el.title = text;
}
function refreshWorkspaceStatus() {
    renderWorkspaceState();
    const sol = selectedSolution;
    if (typeof manualSG !== 'undefined' && manualSG.owner !== sol) { manualSG.lines = []; manualSG.owner = sol; manualSGSelect.value = defaultSimSetting(sol); }
    workspaceNode('workspace-solution').textContent = sol ?
        `${sol.system} · a ${sol.a.toFixed(4)} b ${(sol.b ?? sol.a).toFixed(4)} c ${(sol.c ?? sol.a).toFixed(4)} Å · α ${(sol.alpha ?? 90).toFixed(2)} β ${(sol.beta ?? 90).toFixed(2)} γ ${(sol.gamma ?? (sol.system === 'hexagonal' ? 120 : 90)).toFixed(2)}° · M(20) ${Number(sol.m20 || 0).toFixed(1)}` : 'No solution selected';
}
function appendWorkspaceLog(message, type = 'info') {
    workspaceLog.push({ time: new Date().toISOString(), message: String(message), type });
    if (workspaceLog.length > 1000) workspaceLog.shift();
    renderWorkspaceLog();
    refreshWorkspaceStatus();
}
function renderWorkspaceLog() {
    workspaceNode('workspace-log').replaceChildren(...workspaceLog.map(entry => {
        const li = document.createElement('li');
        li.textContent = `${new Date(entry.time).toLocaleTimeString()} · ${entry.message}`;
        li.dataset.type = entry.type;
        return li;
    }));
}
function workspaceControls() {
    return [...document.querySelectorAll('#controls-panel input[id], #controls-panel select[id], #chart-toolbar select[id], #chart-toolbar input[id]')]
        .filter(el => !['file', 'search'].includes(el.type) && !el.id.startsWith('manual-sg') && el.id !== 'solutions-sort');
}
function captureWorkspace() {
    return JSON.parse(JSON.stringify({ filename: loadedFileName, full: fullExperimentalData,
        working: workingExperimentalData, peaks: pickedPeaks, edited: peaksManuallyEdited,
        solutions, selected: solutions.indexOf(selectedSolution), background: calculatedBackground,
        controls: Object.fromEntries(workspaceControls().map(el => [el.id, { value: el.value, checked: el.checked }])),
        systems: [...ui.systemCheckboxes].map(el => ({ value: el.value, checked: el.checked })),
        manualSetting: manualSGSelect.value, xAxisMode, yAxisMode,
        stats: { lastDurationStr, lastIndexingStats, lastGpuRunSettings, lastTruncatedSystems, lastSystemSearchStats, lastRunTotals },
        view: xrdChart ? { xMin: xrdChart.scales.x.min, xMax: xrdChart.scales.x.max,
            yMin: xrdChart.scales.y.min, yMax: xrdChart.scales.y.max } : null }));
}
function validateWorkspace(data) {
    const patternValid = pattern => pattern && Array.isArray(pattern.tth) && Array.isArray(pattern.intensity) &&
        pattern.tth.length === pattern.intensity.length && pattern.tth.every((t, i) =>
            Number.isFinite(t) && (i === 0 || t >= pattern.tth[i - 1])) && pattern.intensity.every(Number.isFinite);
    if (!data || !patternValid(data.full) || !patternValid(data.working) || typeof data.filename !== 'string' ||
        !Array.isArray(data.peaks) || !data.peaks.every(p => p && Number.isFinite(p.tth) && Number.isFinite(p.height)) ||
        !Array.isArray(data.solutions) || !data.solutions.every(s => s &&
            ['triclinic','monoclinic','orthorhombic','tetragonal','hexagonal','cubic'].includes(s.system) &&
            Number.isFinite(s.a) && s.a > 0 && Number.isFinite(s.volume) && s.volume > 0 && Number.isFinite(s.m20) &&
            (s.system === 'cubic' || (Number.isFinite(s.c) && s.c > 0)) &&
            (!['orthorhombic','monoclinic','triclinic'].includes(s.system) || (Number.isFinite(s.b) && s.b > 0)) &&
            (!['monoclinic','triclinic'].includes(s.system) || (Number.isFinite(s.beta) && s.beta > 0 && s.beta < 180)) &&
            (s.system !== 'triclinic' || ['alpha','gamma'].every(key => Number.isFinite(s[key]) && s[key] > 0 && s[key] < 180))) ||
        !Array.isArray(data.background) || !data.background.every(Number.isFinite) ||
        (data.background.length && data.background.length !== data.working.tth.length)) throw Error('Invalid project data.');
    if (!Number.isInteger(data.selected) || data.selected < -1 || data.selected >= data.solutions.length) throw Error('Invalid selected solution.');
}
function restoreWorkspace(data) {
    validateWorkspace(data); // Validate before replacing anything.
    fileLoadToken++;
    resetSessionState();
    loadedFileName = data.filename;
    fullExperimentalData = data.full;
    workingExperimentalData = data.working;
    pickedPeaks = data.peaks;
    peaksManuallyEdited = !!data.edited;
    calculatedBackground = data.background;
    for (const el of workspaceControls()) {
        const saved = data.controls?.[el.id];
        if (!saved) continue;
        el.value = saved.value;
        if (el.type === 'checkbox') el.checked = !!saved.checked;
    }
    for (const el of ui.systemCheckboxes) {
        const saved = data.systems?.find(s => s.value === el.value);
        if (saved) el.checked = !!saved.checked;
    }
    xAxisMode = AXIS_X[data.xAxisMode] ? data.xAxisMode : 'tth';
    yAxisMode = AXIS_Y[data.yAxisMode] ? data.yAxisMode : 'sqrt';
    ui.xAxisMode.value = xAxisMode; ui.yAxisMode.value = yAxisMode;
    if (data.stats) {
        lastDurationStr = data.stats.lastDurationStr || '';
        lastIndexingStats = data.stats.lastIndexingStats || '';
        lastGpuRunSettings = data.stats.lastGpuRunSettings || null;
        lastTruncatedSystems = data.stats.lastTruncatedSystems || [];
        lastSystemSearchStats = data.stats.lastSystemSearchStats || [];
        lastRunTotals = data.stats.lastRunTotals || null;
    }
    solutions = data.solutions;
    displayedSolutions = [...solutions];
    foundSolutionMap.clear();
    solutions.forEach(sol => foundSolutionMap.set(getSolutionKey(sol), sol));
    ui.peakControls.classList.toggle('hidden', !data.full.tth.length);
    ui.indexingControls.classList.toggle('hidden', !data.full.tth.length);
    ui.saveAsButton.disabled = !data.full.tth.length;
    ui.reportButton.disabled = !solutions.length;
    ui.stripKa2Checkbox.disabled = !ka2StripAllowed();
    ui.wavelength.readOnly = ui.wavelengthPreset.value !== 'custom';
    for (const [slider, label, digits] of [[ui.peakProminenceSlider, ui.peakProminenceValue, 1],
        [ui.ballRadiusSlider, ui.ballRadiusValue, 2], [ui.smoothingWidthSlider, ui.smoothingWidthValue, 2]]) {
        label.textContent = Number(slider.value).toFixed(digits);
    }
    ui.peakThresholdValue.textContent = logSliderToValue(Number(ui.peakThresholdSlider.value)).toFixed(1);
    renderFileChip(loadedFileName);
    if (data.working.tth.length) {
        initializeChart(); setupTthSliders();
        // setupTthSliders resets range controls: restore their saved values.
        for (const el of [ui.tthMinSlider, ui.tthMaxSlider]) {
            if (data.controls?.[el.id]) el.value = data.controls[el.id].value;
        }
        ui.tthMinValue.textContent = Number(ui.tthMinSlider.value).toFixed(2); ui.tthMaxValue.textContent = Number(ui.tthMaxSlider.value).toFixed(2);
    } else if (xrdChart) { xrdChart.destroy(); xrdChart = null; }
    updatePeakTable(); updateSolutionsTable();
    applySolutionSelection(solutions[data.selected] || null);
    manualSGSelect.value = data.manualSetting || '';
    rebuildManualSG();
    if (xrdChart && data.view && Object.values(data.view).every(Number.isFinite)) {
        Object.assign(xrdChart.options.scales.x, { min: data.view.xMin, max: data.view.xMax });
        Object.assign(xrdChart.options.scales.y, { min: data.view.yMin, max: data.view.yMax });
        xrdChart.update('none');
    }
    updateAllMarkers(); updateStartIndexingButtonState();
    refreshWorkspaceStatus(); showWorkspaceView('plot');
    if (typeof syncPlotTools === 'function') syncPlotTools();
    if (typeof syncSolutionAccordion === 'function') syncSolutionAccordion();
}
// The run history is kept in memory. Once, when it reaches this size, the user is offered to prune the oldest runs
// (a history this large can exhaust the browser's memory). A saved project larger than the second limit is confirmed.
const HISTORY_WARN_BYTES = 100 * 1024 * 1024;
const PROJECT_WARN_BYTES = 10 * 1024 * 1024;
let historyWarned = false;
const formatMB = bytes => (bytes / (1024 * 1024)).toFixed(1) + ' MB';
async function checkHistorySize() {
    if (historyWarned) return;
    let bytes = JSON.stringify(workspaceHistory).length;
    if (bytes <= HISTORY_WARN_BYTES) return;
    historyWarned = true;
    const prune = await showNotice('Large history', `The run history now takes about ${formatMB(bytes)}. A history this large can exhaust the browser's memory. Remove the oldest runs now (the most recent ones are kept)?`, { confirmLabel: 'Remove oldest', cancelLabel: 'Keep all' });
    if (!prune) return;
    // Drop the oldest runs until half the limit is left, always keeping the latest.
    while (workspaceHistory.length > 1 && bytes > HISTORY_WARN_BYTES / 2) {
        bytes -= JSON.stringify(workspaceHistory.shift()).length;
    }
    renderWorkspaceHistory();
    appendWorkspaceLog(`History pruned to ${workspaceHistory.length} run(s), about ${formatMB(bytes)}.`);
}
function recordWorkspaceRun(indexing) {
    if (indexing && !workspaceRunStarted) {
        workspaceRunStarted = new Date().toISOString();
        appendWorkspaceLog(`Indexing started: ${loadedFileName}`);
    } else if (!indexing && workspaceRunStarted) {
        const started = workspaceRunStarted; workspaceRunStarted = null;
        // Capture after finalization and the last solutions-table update.
        queueMicrotask(() => {
            workspaceHistory.push({ started, state: captureWorkspace() });
            renderWorkspaceHistory();
            checkHistorySize();
            appendWorkspaceLog(`Indexing finished: ${solutions.length} solutions.`);
        });
    }
}
function renderWorkspaceHistory() {
    const area = workspaceNode('workspace-history'); area.replaceChildren();
    if (!workspaceHistory.length) { area.textContent = 'No indexing runs recorded yet.'; return; }
    workspaceHistory.forEach((entry, index) => {
        const row = document.createElement('article');
        const text = document.createElement('p');
        text.textContent = `${new Date(entry.started).toLocaleString()} · ${entry.state.filename} · ${entry.state.solutions.length} solutions`;
        const button = document.createElement('button'); button.textContent = 'Restore run';
        button.disabled = isIndexing;
        button.addEventListener('click', () => {
            if (isIndexing) return;
            restoreWorkspace(structuredClone(workspaceHistory[index].state));
            appendWorkspaceLog(`Restored run from ${entry.started}.`);
        });
        row.append(text, button); area.append(row);
    });
}
function showWorkspaceView(view) {
    workspaceView = view;
    workspaceNode('log-view').hidden = view !== 'log';
    workspaceNode('history-view').hidden = view !== 'history';
    workspaceNode('about-view').hidden = view !== 'about';
    workspaceNode('btn-about').setAttribute('aria-pressed', String(view === 'about'));
    ui.resultsContainer.style.display = view === 'plot' && workingExperimentalData.tth.length ? 'flex' : 'none';
    ui.placeholder.style.display = view === 'plot' && !workingExperimentalData.tth.length ? '' : 'none';
    workspaceNode('workspace-views').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    if (view === 'history') renderWorkspaceHistory();
    if (view === 'plot' && xrdChart) xrdChart.resize();
}
function updateWorkspaceDatasets() {
    refreshWorkspaceStatus();
    if (!xrdChart) return;
    const min = xrdChart.scales.y.min, span = xrdChart.scales.y.max - min;
    const active = selectedSolution && manualSG.owner === selectedSolution;
    xrdChart.data.datasets[3].data = active ? manualSG.lines.map(line => ({ x: xF(line.tth),
        y: [min + span * .055, min + span * .075], tth: line.tth })) : [];
    const dark = document.documentElement.dataset.theme === 'dark';
    xrdChart.data.datasets[0].borderColor = dark ? '#c9ced3' : '#626b73';
    xrdChart.data.datasets[0].pointBackgroundColor = dark ? '#c9ced3' : '#626b73';
    xrdChart.data.datasets[3].backgroundColor = dark ? '#80c9bc' : '#15716a';
    xrdChart.data.datasets[4].borderColor = dark ? '#d9b56b' : '#885d18';
    xrdChart.data.datasets[4].order = -1;
    xrdChart.data.datasets[4].borderWidth = 1.5;
    xrdChart.data.datasets[0].hidden = !workspaceNode('show-data').checked;
    xrdChart.data.datasets[1].hidden = !workspaceNode('show-exp').checked;
    xrdChart.data.datasets[2].hidden = !workspaceNode('show-sol').checked;
    xrdChart.data.datasets[3].hidden = !workspaceNode('show-sim').checked;
    const visible = workspaceNode('show-background').checked;
    xrdChart.data.datasets[4].hidden = !visible;
    xrdChart.data.datasets[4].data = visible && calculatedBackground.length === workingExperimentalData.tth.length ?
        calculatedBackground.map((intensity, i) => ({ x: xF(workingExperimentalData.tth[i]), y: yF(intensity), I: intensity, tth: workingExperimentalData.tth[i] })) : [];
}
function setWorkspaceTheme(dark) {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    workspaceNode('theme-toggle').checked = dark;
    try { localStorage.setItem('brutus-theme', dark ? 'dark' : 'light'); } catch (_) {}
    Chart.defaults.color = dark ? '#c9ced3' : '#3f464d';
    Chart.defaults.borderColor = dark ? '#3c4247' : '#dde1e4';
    Chart.defaults.plugins.tooltip.backgroundColor = dark ? '#e9ecef' : '#202327';
    Chart.defaults.plugins.tooltip.titleColor = dark ? '#1b1f23' : '#fff';
    Chart.defaults.plugins.tooltip.bodyColor = dark ? '#1b1f23' : '#fff';
    if (xrdChart) {
        const view = { xMin: xrdChart.scales.x.min, xMax: xrdChart.scales.x.max, yMin: xrdChart.scales.y.min, yMax: xrdChart.scales.y.max };
        rebuildPlot(false);
        Object.assign(xrdChart.options.scales.x, { min: view.xMin, max: view.xMax });
        Object.assign(xrdChart.options.scales.y, { min: view.yMin, max: view.yMax });
        xrdChart.update('none'); updateAllMarkers();
    }
}
workspaceNode('theme-toggle').addEventListener('change', e => setWorkspaceTheme(e.target.checked));
let initialDark = false;
try { initialDark = localStorage.getItem('brutus-theme') === 'dark'; } catch (_) {}
setWorkspaceTheme(initialDark);
workspaceNode('swap-panels').addEventListener('click', e => {
    const swapped = document.body.classList.toggle('panels-swapped');
    e.currentTarget.setAttribute('aria-pressed', String(swapped));
    if (xrdChart) xrdChart.resize();
});
workspaceNode('workspace-views').addEventListener('click', e => { if (e.target.dataset.view) showWorkspaceView(e.target.dataset.view); });
['show-background', 'show-data', 'show-exp', 'show-sol', 'show-sim'].forEach(id => workspaceNode(id).addEventListener('change', updateAllMarkers));
workspaceNode('menu-unload').addEventListener('click', openUnloadDialog);
workspaceNode('file-menu-items').addEventListener('click', e => { if (e.target.closest('button')) workspaceNode('file-menu').open = false; });
document.addEventListener('click', e => { if (!e.target.closest('#file-menu')) workspaceNode('file-menu').open = false; });
document.addEventListener('keydown', e => { if (e.key === 'Escape') workspaceNode('file-menu').open = false; });
workspaceNode('save-project').addEventListener('click', async () => {
    if (isIndexing) { showStatus('Wait for indexing to finish before saving a project.'); return; }
    const project = { format: 'brutus-project', version: 1, state: captureWorkspace(), history: workspaceHistory, log: workspaceLog };
    const blob = new Blob([JSON.stringify(project)], { type: 'application/json' });
    if (blob.size > PROJECT_WARN_BYTES && !await showNotice('Large project', `This project file will be ${formatMB(blob.size)} (more than ${formatMB(PROJECT_WARN_BYTES)}), mostly run history. Save it anyway?`, { confirmLabel: 'Save', cancelLabel: 'Cancel' })) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url;
    link.download = (loadedFileName || 'untitled').replace(/\.[^.]+$/, '') + '.brutus';
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    appendWorkspaceLog('Project saved.');
});
workspaceNode('open-project').addEventListener('click', () => { if (!isIndexing) workspaceNode('project-input').click(); });
workspaceNode('project-input').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    const token = ++fileLoadToken;
    try {
        const project = JSON.parse(await file.text());
        if (token !== fileLoadToken || isIndexing) return;
        if (project.format !== 'brutus-project' || project.version !== 1) throw Error('Unsupported project format or version.');
        validateWorkspace(project.state);
        if (!Array.isArray(project.history) || !Array.isArray(project.log)) throw Error('Invalid project history.');
        project.history.forEach(entry => { if (!entry || !Number.isFinite(Date.parse(entry.started))) throw Error('Invalid history entry.'); validateWorkspace(entry.state); });
        if (!project.log.every(entry => entry && typeof entry.message === 'string' && Number.isFinite(Date.parse(entry.time)))) throw Error('Invalid project log.');
        restoreWorkspace(project.state);
        workspaceHistory = project.history;
        workspaceLog.splice(0, workspaceLog.length, ...project.log.slice(-1000));
        renderWorkspaceHistory(); renderWorkspaceLog();
        appendWorkspaceLog(`Opened project ${file.name}.`);
    } catch (error) { showStatus(`Could not open project: ${error.message}`, 'error'); }
    finally { e.target.value = ''; }
});

// Powder6-style group browser: filters, a list of settings, and explanation.
let browserSetting = null;
function renderSGBrowser() {
    const query = workspaceNode('sg-browser-query').value.toLowerCase().replace(/\s/g, '');
    const system = workspaceNode('sg-filter-system').value;
    const centering = workspaceNode('sg-filter-centering').value;
    const entries = manualSG.entries.filter(entry => (!system || entry.group.crystal_system === system) &&
        (!centering || entry.setting.centering === centering) &&
        (entry.label + entry.setting.hall).toLowerCase().replace(/\s/g, '').includes(query));
    workspaceNode('sg-browser-count').textContent = `${entries.length} settings`;
    workspaceNode('sg-browser-list').replaceChildren(...entries.map(entry => {
        const button = document.createElement('button'); button.type = 'button';
        button.textContent = entry.label; button.setAttribute('aria-pressed', String(browserSetting === entry));
        button.addEventListener('click', () => {
            browserSetting = entry;
            const conditions = Object.entries(entry.setting.conditions || {}).map(([zone, rules]) => `${zone}: ${rules.join(', ')}`).join('\n');
            workspaceNode('sg-browser-detail').textContent = `${entry.label}\n\n${entry.group.crystal_system} · ${entry.setting.centering} centered\nPoint group: ${entry.group.point_group}\nHall: ${entry.setting.hall}\n\nReflection conditions\n${conditions || 'No systematic absences.'}\n\nUse a cell expressed in this setting’s axes. No basis conversion is applied.`;
            workspaceNode('sg-browser-apply').disabled = false;
            workspaceNode('sg-browser-list').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
        }); return button;
    }));
}
workspaceNode('browse-sg').addEventListener('click', () => {
    if (!selectedSolution || !spaceGroupData) { showStatus('Select a solution first.'); return; }
    browserSetting = null; workspaceNode('sg-browser-apply').disabled = true;
    // The crystal system opens on that of the proposed cell (a value the list does not have leaves All systems).
    const systemFilter = workspaceNode('sg-filter-system');
    systemFilter.value = selectedSolution.system || '';
    if (systemFilter.value !== (selectedSolution.system || '')) systemFilter.value = '';
    workspaceNode('sg-browser-detail').textContent = 'Select a setting to inspect its symmetry and reflection conditions.';
    renderSGBrowser(); workspaceNode('sg-browser').showModal(); workspaceNode('sg-browser-query').focus();
});
['sg-browser-query', 'sg-filter-system', 'sg-filter-centering'].forEach(id => workspaceNode(id).addEventListener('input', renderSGBrowser));
workspaceNode('sg-browser-reset').addEventListener('click', () => {
    ['sg-browser-query','sg-filter-system','sg-filter-centering'].forEach(id => workspaceNode(id).value = ''); renderSGBrowser();
});
workspaceNode('sg-browser-cancel').addEventListener('click', () => workspaceNode('sg-browser').close());
workspaceNode('sg-browser-apply').addEventListener('click', () => {
    if (!browserSetting) return;
    manualSGSearch.value = ''; filterManualSG();
    manualSGSelect.value = browserSetting.setting.setting_id;
    // The chosen group becomes part of the solution: it is listed under the
    // solution, saved with the project, and written to the report and pdCIF.
    const chosen = browserSetting;
    addSolutionSG(selectedSolution, { label: chosen.label, symbol: chosen.setting.hm || chosen.setting.symbol,
        number: Number(chosen.group.number), hall: chosen.setting.hall, setting_id: chosen.setting.setting_id,
        system: chosen.group.crystal_system, source: 'explored' });
    rebuildManualSG(); updateAllMarkers(); updateSolutionsTable();
    appendWorkspaceLog(`Space group ${chosen.label} added to the selected solution.`);
    workspaceNode('sg-browser').close();
});
workspaceNode('sg-browser-clear').addEventListener('click', () => {
    if (selectedSolution) {
        const list = solutionSGs(selectedSolution), at = list.findIndex(x => x.setting_id === manualSGSelect.value);
        if (at >= 0) removeSolutionSG(selectedSolution, at);
        manualSGSelect.value = defaultSimSetting(selectedSolution);
        updateSolutionsTable();
    }
    rebuildManualSG(); updateAllMarkers(); workspaceNode('sg-browser').close();
});
refreshWorkspaceStatus(); renderWorkspaceHistory();

