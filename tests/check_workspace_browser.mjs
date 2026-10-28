// Local Chrome smoke test: no npm dependencies. Runs the real static app.
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
const profile = mkdtempSync(join(tmpdir(), 'brutus-browser-'));
const server = spawn('python', ['-m', 'http.server', '8769', '--bind', '127.0.0.1'], { windowsHide: true, stdio: 'ignore' });
const gpuMode = process.env.BRUTUS_TEST_GPU === '1';
const gpuFlags = gpuMode ? ['--enable-unsafe-webgpu'] : ['--disable-gpu', '--disable-software-rasterizer'];
const browser = spawn(process.env.BRUTUS_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe', ['--headless=new','--no-first-run', ...gpuFlags, '--no-sandbox', '--remote-debugging-port=9337', `--user-data-dir=${profile}`, 'http://127.0.0.1:8769/index.html'], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
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
    assert.ok(await evaluate(`Object.entries(BRUTUS_LIMITS.inputs).every(([id, spec]) => {
        const el = document.getElementById(id);
        return el && el.defaultValue === String(spec.value) &&
            ['min','max','step'].every(key => spec[key] === undefined || el[key] === String(spec[key]));
    })`));
    if (gpuMode) {
        const checked = await evaluate(`(async () => {
            await checkWebGPUCapabilities();
            if (!webGPUSupportsCompute) {
                try { await getWebGPUEngine(); } catch (error) { throw Error('WebGPU test adapter unavailable: ' + error.message); }
            }
            const engine = await getWebGPUEngine();
            const kernels = [
                ['shaders/highsym_solver.wgsl', 'main_cubic'],
                ['shaders/highsym_solver.wgsl', 'main_tetragonal'],
                ['shaders/highsym_solver.wgsl', 'main_hexagonal'],
                ['shaders/ortho_solver.wgsl', 'main_3p'],
                ['shaders/monoclinic_solver.wgsl', 'main_4p'],
                ['shaders/triclinic_solver.wgsl', 'main'],
            ];
            for (const [url, entry] of kernels) {
                await engine.loadShader(url + APP_VERSION_QS);
                await engine.createPipeline(entry);
            }
            const truths = {
                cubic: {a:5,b:5,c:5,alpha:90,beta:90,gamma:90},
                tetragonal: {a:4,b:4,c:6,alpha:90,beta:90,gamma:90},
                hexagonal: {a:4,b:4,c:6,alpha:90,beta:90,gamma:120},
                orthorhombic: {a:4,b:5,c:6,alpha:90,beta:90,gamma:90},
                monoclinic: {a:4,b:5,c:6,alpha:90,beta:105,gamma:90},
                triclinic: {a:4,b:5,c:6,alpha:80,beta:100,gamma:110},
            };
            const probes = [
                ['cubic',0,[[1,0,0]]],
                ['tetragonal',1,[[1,0,0],[0,0,1]]],
                ['hexagonal',2,[[1,0,0],[0,0,1]]],
                ['orthorhombic',3,[[1,0,0],[0,1,0],[0,0,1]]],
                ['monoclinic',4,[[1,0,0],[0,1,0],[0,0,1],[1,0,1]]],
                ['triclinic',5,[[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]]],
            ];
            const recovered = [];
            for (const [system, kernelIndex, hkls] of probes) {
                const [url, entry] = kernels[kernelIndex];
                await engine.loadShader(url + APP_VERSION_QS); await engine.createPipeline(entry);
                const cfg = WebGPUEngine.SYSTEM_CONFIGS[system], truth = truths[system];
                const G = invert3x3(metricFromCell(truth));
                // These deliberately minimal probes keep the known assignment
                // order: K peaks alone can admit several different valid cells.
                const qs = hkls.map(h => h.reduce((sum,x,i) => sum + h.reduce((s,y,j) => s + x*G[i][j]*y,0),0));
                const packed = new Float32Array(hkls.length * cfg.hklFloats);
                hkls.forEach((h,i) => HKL_PACKERS[system].pack(packed,i,...h));
                const cells = [];
                await engine._runSolver(cfg, new Float32Array(qs), packed,
                    new Uint32Array(hkls.map((_,i)=>i)), null, new Float32Array(qs.map(()=>0.002)),
                    () => {}, {stop:false}, {hklPacking:HKL_PACKING,wavelength:1.54,tth_error:0.04,
                        max_volume:400,fom_threshold:1.25,impurity_peaks:0,max_solutions:1000,gpu_peaks_count:qs.length},
                    batch => cells.push(...batch));
                if (!cells.some(c => Math.abs(c.a-truth.a)<1e-3 && Math.abs((c.b??c.a)-truth.b)<1e-3 &&
                    Math.abs((c.c??c.a)-truth.c)<1e-3 && Math.abs((c.beta??90)-truth.beta)<0.01 &&
                    Math.abs((c.alpha??90)-truth.alpha)<0.01 && Math.abs((c.gamma??(system==='hexagonal'?120:90))-truth.gamma)<0.01)) {
                    throw Error(system + ' GPU dispatch did not recover the known cell');
                }
                recovered.push(system);
            }
            return {kernels:kernels.length, recovered};
        })()`);
        assert.equal(checked.kernels, 6);
        assert.equal(checked.recovered.length, 6);
        console.log('PASS: six real WebGPU pipelines compile and recover known cells in every crystal system.');
    }
    const pattern = readFileSync('test_files/monoclinic_test_1.xy', 'utf8');
    await evaluate(`(async () => {
        const file = new File([${JSON.stringify(pattern)}], 'monoclinic_test_1.xy');
        const transfer = new DataTransfer(); transfer.items.add(file); ui.fileInput.files = transfer.files;
        ui.fileInput.dispatchEvent(new Event('change')); await new Promise(r => setTimeout(r, 500));
    })()`);
    assert.ok(await evaluate('workingExperimentalData.tth.length > 0'));
    // GSAS-II uses stored y * measuredTime / effectiveTime, without multiplying
    // the separate AbsorptionFactor column.
    assert.deepEqual(await evaluate(`(() => {
        let xml = buildExportContent('brukerxml', {tth:[10,11],intensity:[100,609419.2]}).text;
        xml = xml.replace('<TimePerStepEffective>1</TimePerStepEffective>', '<TimePerStepEffective>2</TimePerStepEffective>')
            .replace('<Datum>1,1,10,5,100</Datum>', '<Datum>2,1,10,5,100</Datum>')
            .replace('<Datum>1,1,11,5.5,609419.2</Datum>', '<Datum>4,8.3,11,5.5,609419.2</Datum>');
        const scaled = parseBrukerXmlFile(xml);
        const noTime = parseBrukerXmlFile(xml.replace('<TimePerStepEffective>2</TimePerStepEffective>', ''));
        const irregular = parseBrukerXmlFile(buildExportContent('brukerxml', {tth:[10,11,30],intensity:[0.1,2.5,300]}).text);
        return {scaled:scaled.intensity, preserved:noTime.intensity, angles:irregular.tth, values:irregular.intensity};
    })()`), {scaled:[100,1218838.4],preserved:[100,609419.2],angles:[10,11,30],values:[0.1,2.5,300]});
    // Projects retain file metadata; older projects must not inherit another file's metadata.
    assert.ok(await evaluate(`(() => {
        const data = captureWorkspace();
        const expected = JSON.stringify(loadedFileMeta);
        loadedFileMeta = { size: 999, radiation: { single: 9 } };
        restoreWorkspace(data);
        if (JSON.stringify(loadedFileMeta) !== expected) return false;
        delete data.fileMeta; restoreWorkspace(data);
        if (loadedFileMeta !== null) return false;
        data.fileMeta = JSON.parse(expected); restoreWorkspace(data);
        return JSON.stringify(loadedFileMeta) === expected;
    })()`));
    await evaluate(`solutions = [{system:'cubic', a:5, b:5, c:5, alpha:90, beta:90, gamma:90, m20:50, volume:125}]; displayedSolutions = [...solutions]; updateSolutionsTable(); applySolutionSelection(solutions[0]);`);
    assert.deepEqual(await evaluate(`parseXrdmlFile(buildExportContent('xrdml', {tth:[10,11,30],intensity:[100,200,300]}).text).tth`), [10,11,30]);
    await evaluate(`document.getElementById('browse-sg').click()`);
    assert.equal(await evaluate("document.getElementById('sg-browser').open"), true);
    await evaluate(`document.querySelector('#sg-browser-list button').click(); document.getElementById('sg-browser-query').value = 'no matching group'; renderSGBrowser();`);
    assert.ok(await evaluate(`browserSetting === null && document.getElementById('sg-browser-apply').disabled && document.getElementById('sg-browser-list').children.length === 0`));
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
    // Refinement takes keyboard focus, traps Tab and returns focus on closing.
    await evaluate(`document.getElementById('solution-menu').open = true; document.querySelector('[data-sol-action=refine]').focus(); document.querySelector('[data-sol-action=refine]').click()`);
    assert.equal(await evaluate(`document.activeElement.id`), 'mc-iterations');
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', modifiers: 8 });
    assert.equal(await evaluate(`document.activeElement.id`), 'mc-apply');
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab' });
    assert.equal(await evaluate(`document.activeElement.id`), 'mc-iterations');
    await evaluate(`document.getElementById('help-guide').focus()`);
    assert.equal(await evaluate(`document.activeElement.id`), 'mc-iterations');
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape' });
    assert.ok(await evaluate(`!document.getElementById('mc-overlay').classList.contains('open') && document.activeElement.matches('#solution-menu > summary, [data-sol-action=refine]') && document.activeElement.getClientRects().length > 0`));
    // Result tabs expose selection and support arrow, Home and End navigation.
    await evaluate(`showWorkspaceView('plot'); document.getElementById('view-tab-plot').focus()`);
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight' });
    assert.ok(await evaluate(`workspaceView === 'log' && document.activeElement.id === 'view-tab-log' && document.activeElement.getAttribute('aria-selected') === 'true' && !document.getElementById('log-view').hidden`));
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'End', code: 'End' });
    assert.equal(await evaluate(`workspaceView`), 'history');
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Home', code: 'Home' });
    assert.equal(await evaluate(`workspaceView`), 'plot');
    // Render the GPU plan in this CPU-only browser and check a narrow controls panel.
    assert.ok(await evaluate(`(() => {
        const supported = webGPUSupportsCompute;
        webGPUSupportsCompute = true;
        updateGpuStatusText();
        webGPUSupportsCompute = supported;
        const rows = [...document.querySelectorAll('#gpu-plan tbody tr')];
        const panel = document.getElementById('controls-panel');
        const width = panel.style.width;
        const container = document.getElementById('gpu-params-container');
        const hidden = container.classList.contains('hidden');
        container.classList.remove('hidden');
        panel.style.width = '320px';
        document.querySelector('[data-tab=parameters]').click();
        const box = document.getElementById('gpu-plan');
        const fits = box.scrollWidth <= box.clientWidth;
        panel.style.width = width; container.classList.toggle('hidden', hidden);
        return fits && rows.length > 0 && rows.every(row => row.cells.length === 4 && row.cells[3].title.endsWith(' trials'));
    })()`));
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
    // One version for everything: js/core/version.js is what the page, its workers and the footer report.
    const expectedVersion = /APP_VERSION\s*=\s*'([^']+)'/.exec(readFileSync('js/core/version.js', 'utf8'))[1];
    assert.equal(await evaluate('APP_VERSION'), expectedVersion);
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
    assert.ok(await evaluate(`(async () => {
        const good = captureWorkspace();
        try {
            const sol = solutions[0];
            sol.analysis = analyzeSystematicAbsences(sol, pickedPeaks, spaceGroupData,
                getWavelength(), getTthError(), Number(ui.tthMaxSlider.value), getImpurityPeaks(), Number(ui.tthMinSlider.value));
            sol.sgClass = 'test class'; sol.sgMargin = Infinity; sol.sgMembers = ['P 1 (1)'];
            sol.errors = {s_a:1.2345e-7};
            const data = JSON.parse(JSON.stringify(captureWorkspace(), workspaceJsonReplacer), workspaceJsonReviver);
            restoreWorkspace(data);
            if (solutions[0].errors.s_a !== 1.2345e-7 || solutions[0].sgMargin !== Infinity ||
                !(solutions[0].analysis.ambiguousHkls instanceof Set)) return false;
            // Render a report from an actual restored absence analysis without a download.
            const originalPdf = window.jspdf.jsPDF;
            let saved = false;
            window.jspdf.jsPDF = function(...args) { const doc = new originalPdf(...args); doc.save = () => { saved = true; }; return doc; };
            try { await generatePDFReport(solutions[0]); }
            finally { window.jspdf.jsPDF = originalPdf; }
            if (!saved) return false;
            const bad = captureWorkspace(); bad.peaks[0].d = 'broken';
            try { restoreWorkspace(bad); return false; } catch {}
            if (pickedPeaks[0].d !== data.peaks[0].d) return false;
            const attack = captureWorkspace();
            attack.solutions[0].sgClass = '\"><img src=invalid-review-probe onerror=window.__reviewXss=1>';
            window.__reviewXss = 0; restoreWorkspace(attack);
            await new Promise(resolve => setTimeout(resolve, 100));
            return window.__reviewXss === 0 && !document.querySelector('img[src="invalid-review-probe"]');
        } finally { restoreWorkspace(good); }
    })()`));
    assert.equal(errors.length, 0, JSON.stringify(errors));
    console.log('PASS: real browser covers UI, shared defaults, irregular XRDML round trip, project precision, restored PDF analysis and import XSS protection.');
    await send('Browser.close');
} finally {
    socket?.close(); browser.kill(); server.kill();
}
