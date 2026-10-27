// Local Chrome smoke test: no npm dependencies. Runs the real static app.
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
const profile = mkdtempSync(join(tmpdir(), 'brutus-browser-'));
const server = spawn('python', ['-m', 'http.server', '8769', '--bind', '127.0.0.1'], { windowsHide: true, stdio: 'ignore' });
const browser = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', ['--headless=new','--no-first-run', '--disable-gpu', '--no-sandbox', '--disable-software-rasterizer', '--remote-debugging-port=9337', `--user-data-dir=${profile}`, 'http://127.0.0.1:8769/index.html'], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
let browserErrors = '';
browser.stderr.on('data', chunk => { browserErrors += chunk; });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
    let tabs;
    for (let i = 0; i < 60; i++) {
        try { tabs = await (await fetch('http://127.0.0.1:9337/json')).json(); if (tabs.some(t => t.type === 'page')) break; } catch {}
        await pause(200);
    }
    if (!tabs) throw Error('Chrome did not start: ' + browserErrors);
    const tab = tabs.find(t => t.type === 'page');
    socket = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
    const pending = new Map(); let id = 0; const errors = [];
    socket.addEventListener('message', e => {
        const message = JSON.parse(e.data);
        if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
        if (message.id) { pending.get(message.id)?.(message); pending.delete(message.id); }
    });
    const send = (method, params = {}) => new Promise(resolve => { pending.set(++id, resolve); socket.send(JSON.stringify({id, method, params})); });
    const evaluate = async expression => {
        const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
        if (result.result.exceptionDetails) throw Error(JSON.stringify(result.result.exceptionDetails));
        return result.result.result.value;
    };
    await send('Runtime.enable');
    for (let i = 0; i < 60; i++) {
        if (await evaluate("typeof spaceGroupData !== 'undefined' && !!spaceGroupData && typeof captureWorkspace === 'function'")) break;
        await pause(200);
    }
    assert.equal(await evaluate('manualSG.entries.length'), 530);
    const pattern = readFileSync('test_files/monoclinic_test_1.xy', 'utf8');
    await evaluate(`(async () => {
        const file = new File([${JSON.stringify(pattern)}], 'monoclinic_test_1.xy');
        const transfer = new DataTransfer(); transfer.items.add(file); ui.fileInput.files = transfer.files;
        ui.fileInput.dispatchEvent(new Event('change')); await new Promise(r => setTimeout(r, 500));
    })()`);
    assert.ok(await evaluate('workingExperimentalData.tth.length > 0'));
    await evaluate(`solutions = [{system:'cubic', a:5, b:5, c:5, alpha:90, beta:90, gamma:90, m20:50, volume:125}]; displayedSolutions = [...solutions]; updateSolutionsTable(); applySolutionSelection(solutions[0]);`);
    await evaluate(`document.getElementById('browse-sg').click()`);
    assert.equal(await evaluate("document.getElementById('sg-browser').open"), true);
    await evaluate(`document.getElementById('sg-browser-query').value = '225'; renderSGBrowser(); document.querySelector('#sg-browser-list button').click(); document.getElementById('sg-browser-apply').click();`);
    assert.ok(await evaluate('manualSG.lines.length > 0 && xrdChart.data.datasets[3].data.length > 0'));
    await evaluate(`document.getElementById('show-background').checked = true; updateAllMarkers()`);
    assert.ok(await evaluate('xrdChart.data.datasets[4].data.length > 0 && !xrdChart.data.datasets[4].hidden'));
    assert.ok(await evaluate(`(() => { const line = manualSG.lines[0]; const snap = snapToHkl(line.tth); return Math.abs(snap.tth-line.tth)<1e-8; })()`));
    assert.ok(await evaluate(`(() => { const data = captureWorkspace(); restoreWorkspace(JSON.parse(JSON.stringify(data))); return pickedPeaks.length === data.peaks.length && displayedSolutions.length === 1 && !!selectedSolution && manualSG.lines.length > 0; })()`));
    assert.ok(await evaluate(`(() => { const data = captureWorkspace(); data.full.tth[0] = null; try { validateWorkspace(data); return false; } catch { return true; } })()`));
    assert.ok(await evaluate(`(async () => {
        recordWorkspaceRun(true); recordWorkspaceRun(false); await Promise.resolve();
        return workspaceHistory.length === 1 && workspaceHistory[0].state.solutions.length === 1;
    })()`));
    const downloadPath = mkdtempSync(join(tmpdir(), 'brutus-project-test-'));
    await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath });
    await evaluate(`document.getElementById('save-project').click()`);
    let savedProject;
    for (let i = 0; i < 30; i++) {
        const filename = readdirSync(downloadPath).find(name => name.endsWith('.brutus'));
        if (filename) { savedProject = readFileSync(join(downloadPath, filename), 'utf8'); break; }
        await pause(100);
    }
    assert.equal(JSON.parse(savedProject).history.length, 1);
    await evaluate(`clearLoadedFile();`);
    await evaluate(`(async () => {
        const file = new File([${JSON.stringify(savedProject)}], 'roundtrip.brutus');
        const transfer = new DataTransfer(); transfer.items.add(file);
        document.getElementById('project-input').files = transfer.files;
        document.getElementById('project-input').dispatchEvent(new Event('change'));
        await new Promise(resolve => setTimeout(resolve, 150));
    })()`);
    assert.ok(await evaluate(`selectedSolution && displayedSolutions.length === 1 && manualSG.lines.length > 0 && workspaceHistory.length === 1 && workspaceLog.some(entry => entry.message.includes('Opened project'))`));
    await evaluate(`showWorkspaceView('history'); document.querySelector('#workspace-history button').click()`);
    assert.ok(await evaluate(`workspaceView === 'plot' && !!selectedSolution && manualSG.lines.length > 0`));
    assert.ok(await evaluate(`(() => {
        const line = manualSG.lines[0];
        const title = xrdChart.options.plugins.tooltip.callbacks.title([{ datasetIndex: 3, raw: { tth: line.tth }, parsed: { x: xF(line.tth) } }]);
        return title.some(text => text.startsWith('hkl'));
    })()`));
    // Explore opens on the crystal system of the proposed cell.
    await evaluate(`document.getElementById('browse-sg').click()`);
    assert.equal(await evaluate(`document.getElementById('sg-filter-system').value === selectedSolution.system`), true);
    await evaluate(`document.getElementById('sg-browser-cancel').click()`);
    // The space group chosen in Explore is part of the solution and survived the project round trip.
    assert.ok(await evaluate(`solutionSGs(selectedSolution).length === 1 && solutionSGs(selectedSolution)[0].number === 225 && solutionSpaceGroupText(selectedSolution).includes('225')`));
    // The Solution menu works on the selected solution only.
    assert.equal(await evaluate(`[...document.querySelectorAll('[data-sol-action]')].every(b => !b.disabled)`), true);
    await evaluate(`applySolutionSelection(null)`);
    assert.equal(await evaluate(`[...document.querySelectorAll('[data-sol-action]')].every(b => b.disabled)`), true);
    await evaluate(`applySolutionSelection(displayedSolutions[0])`);
    // Convert lists the maximal subgroups of the chosen group (naming runs on in the background).
    await evaluate(`document.querySelector('[data-sol-action=convert]').click()`);
    await pause(1500);
    assert.ok(await evaluate(`document.getElementById('convert-dialog').open && convertState.rows.length > 0 && convertState.rows.every(r => r.cell.a > 0)`));
    // Adding the converted cell opens it: it becomes the selected solution and its accordion is the open one.
    await evaluate(`document.querySelector('#convert-results tr[data-row="0"]').click(); document.getElementById('convert-apply').click()`);
    await pause(150);
    assert.ok(await evaluate(`(() => { const open = document.querySelectorAll('.sol-acc[open]'); return open.length === 1 && selectedSolution && selectedSolution.convertedFrom && displayedSolutions[Number(open[0].dataset.index)] === selectedSolution && !document.getElementById('convert-dialog').open && document.getElementById('solutions-tab-content').classList.contains('active'); })()`));
    // Accordions: one open at a time, and the open one is the selected solution.
    await evaluate(`solutions.push({ ...solutions[0], a: 6, volume: 216, spaceGroups: undefined }); displayedSolutions = [...solutions]; updateSolutionsTable(); document.querySelectorAll('.sol-acc > summary')[1].click();`);
    await pause(150);
    assert.ok(await evaluate(`document.querySelectorAll('.sol-acc[open]').length === 1 && selectedSolution === displayedSolutions[1]`));
    // The final list reserves a share for every system present: 60 strong orthorhombic cells cannot push out 3 monoclinic ones.
    assert.ok(await evaluate(`(() => {
        const mk = (system, i, m20) => ({ system, a: 5 + i * 0.7, b: 6 + i * 0.5, c: 7 + i * 0.9, alpha: 90, beta: system === 'monoclinic' ? 100 : 90, gamma: 90,
            volume: (5 + i * 0.7) * (6 + i * 0.5) * (7 + i * 0.9), m20, _fitContext: 'x' });
        const cells = [...Array.from({ length: 60 }, (_, i) => mk('orthorhombic', i, 200 - i)), ...Array.from({ length: 3 }, (_, i) => mk('monoclinic', 70 + i, 5 + i))];
        const out = applyFinalSieve(cells);
        return out.length <= 53 && out.filter(x => x.system === 'monoclinic').length === 3 && out.filter(x => x.system === 'orthorhombic').length >= 47;
    })()`));
    // One version for everything: version.json is what the page, its workers and the footer report.
    assert.equal(await evaluate('APP_VERSION'), JSON.parse(readFileSync('version.json', 'utf8')).version);
    assert.ok(await evaluate(`document.getElementById('app-footer').textContent.includes(APP_VERSION)`));
    assert.equal(await evaluate(`document.getElementById('solutions-sort').options.length`), 6);
    // A solution holds several space groups; one of them is simulated.
    await evaluate(`addSolutionSG(displayedSolutions[0], { label: 'extra', symbol: 'P 1', number: 1, setting_id: '001_01', source: 'test' }); updateSolutionsTable();`);
    assert.equal(await evaluate(`solutionSGs(displayedSolutions[0]).length`), 2);
    // Closing the open solution deselects it, and the Solution menu goes off.
    await evaluate(`document.querySelector('.sol-acc[open] > summary').click()`);
    assert.ok(await evaluate(`selectedSolution === null && document.getElementById('solution-menu').classList.contains('is-disabled')`));
    await evaluate(`document.querySelectorAll('.sol-acc > summary')[1].click()`);
    assert.ok(await evaluate(`selectedSolution === displayedSolutions[1] && !document.getElementById('solution-menu').classList.contains('is-disabled')`));
    // A run owns the window: Index becomes a red Stop, the controls are locked, Esc stops it.
    await evaluate(`setUIState(true)`);
    assert.ok(await evaluate(`document.body.classList.contains('is-indexing') && document.getElementById('hb-run').classList.contains('is-stop') && document.getElementById('hb-run').textContent.includes('Stop') && !document.getElementById('hb-run').disabled && [...document.querySelectorAll('[data-sol-action]')].every(b => b.disabled)`));
    await evaluate(`document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))`);
    for (let i = 0; i < 40 && await evaluate('isIndexing'); i++) await pause(100);
    assert.equal(await evaluate('isIndexing'), false);
    assert.equal(await evaluate(`document.body.classList.contains('is-indexing')`), false);
    await evaluate(`setWorkspaceTheme(true); document.getElementById('swap-panels').click(); showWorkspaceView('log');`);
    assert.equal(await evaluate("document.documentElement.dataset.theme"), 'dark');
    assert.equal(await evaluate("document.body.classList.contains('panels-swapped')"), true);
    assert.equal(await evaluate("document.getElementById('log-view').hidden"), false);
    await evaluate(`showWorkspaceView('plot'); setWorkspaceTheme(false)`);
    await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
    await pause(150);
    const screenshot = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(tmpdir(), 'brutus-workspace-light.png'), Buffer.from(screenshot.result.data, 'base64'));
    await evaluate(`setWorkspaceTheme(true)`);
    const darkScreenshot = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(tmpdir(), 'brutus-workspace-dark.png'), Buffer.from(darkScreenshot.result.data, 'base64'));
    assert.equal(errors.length, 0, JSON.stringify(errors));
    console.log('PASS: real browser loads pattern, shows background, tests group browser, ticks, snapping, project round trip, validation, theme, panel swap and log.');
    await send('Browser.close');
} finally {
    socket?.close(); browser.kill(); server.kill();
}
