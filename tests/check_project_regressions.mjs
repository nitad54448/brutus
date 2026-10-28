// Run with node --test tests/check_project_regressions.mjs.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const read = file => readFileSync(new URL('../' + file, import.meta.url), 'utf8');
function harness() {
    const element = { value: '', addEventListener() {}, querySelectorAll: () => [] };
    const context = vm.createContext({ console, structuredClone,
        document: { getElementById: () => element, querySelectorAll: () => [] },
        ui: { systemCheckboxes: [], solutionsTableBody: element },
        SG_DECISIVE_NATS: 2.3,
        loadedFileName: 'original.xy', loadedFileMeta: null,
        fullExperimentalData: { tth: [10, 11], intensity: [1.234567891, 1e-9] },
        workingExperimentalData: { tth: [10, 11], intensity: [1.234567891, 1e-9] },
        pickedPeaks: [{ tth: 10, height: 1, d: 8.8, q: 0.0129 }], peaksManuallyEdited: false,
        solutions: [], selectedSolution: null, calculatedBackground: [],
        manualSGSelect: element, xAxisMode: 'tth', yAxisMode: 'sqrt', xrdChart: null,
        lastDurationStr: '', lastIndexingStats: '', lastGpuRunSettings: null,
        lastTruncatedSystems: [], lastSystemSearchStats: [], lastRunTotals: null
    });
    vm.runInContext(read('js/crystallography/limits.js'), context);
    // Only the pure persistence functions: UI registration is covered in the
    // real-browser suite rather than duplicated by a large DOM imitation.
    vm.runInContext(read('js/core/workspace.js').split('// The run history is kept in memory.')[0], context);
    vm.runInContext(read('js/indexing/solutions.js'), context);
    vm.runInContext(`solutions = [{system:'cubic',a:5,b:5,c:5,alpha:90,beta:90,gamma:90,volume:125,m20:50}];
        selectedSolution = solutions[0];`, context);
    return { context, run: source => vm.runInContext(source, context) };
}

test('projects preserve tiny uncertainties, measured intensities and absence sets', () => {
    const { run } = harness();
    const result = run(`(() => {
        solutions[0].errors = { s_a: 1.2345e-7 };
        solutions[0].sgMargin = Infinity;
        solutions[0].analysis = { ambiguousHkls: new Set(['1,2,3']) };
        const saved = captureWorkspace();
        validateWorkspace(saved);
        const roundtrip = JSON.parse(JSON.stringify(saved, workspaceJsonReplacer), workspaceJsonReviver);
        if (roundtrip.solutions[0].sgMargin !== Infinity) throw Error('Infinite class margin was lost');
        return JSON.stringify(saved);
    })()`);
    const saved = JSON.parse(result);
    assert.equal(saved.solutions[0].errors.s_a, 1.2345e-7);
    assert.deepEqual(saved.full.intensity, [1.234567891, 1e-9]);
    assert.deepEqual(saved.solutions[0].analysis.ambiguousHkls, ['1,2,3']);
});

test('malformed optional project fields fail before any session replacement', () => {
    const { run } = harness();
    run('let replacements = 0; applyWorkspace = () => { replacements++; };');
    for (const mutation of [
        `data.peaks[0].d = 'broken'`,
        `data.systems = { value: 'cubic' }`,
        `data.controls = { wavelength: { value: {}, checked: false } }`,
        `data.solutions[0].b = 'broken'`,
        `data.solutions[0].zero_correction = 'broken'`,
        `data.solutions[0].spaceGroups = { label: 'broken' }`,
        `data.solutions[0]._symAlt = [{system: '<img>', a: 5}]`,
        `data.solutions[0].sgMembers = 'broken'`,
        `data.solutions[0].analysis = { hklList: [{tth:'broken'}] }`,
        `data.stats.lastRunTotals = { failures: 'broken' }`,
        `data.stats.lastSystemSearchStats = [{label:{}}]`,
        `data.view = { xMin: 10, xMax: 5, yMin: 0, yMax: 100 }`
    ]) {
        assert.throws(() => run(`{ const data = captureWorkspace(); ${mutation}; restoreWorkspace(data); }`), /Invalid project/);
    }
    assert.equal(run('replacements'), 0);
    assert.equal(run('loadedFileName'), 'original.xy');
});

test('restore rolls back the previous session if rendering fails after validation', () => {
    const { run } = harness();
    run(`const applied = [];
        applyWorkspace = data => {
            applied.push(data.filename); loadedFileName = data.filename;
            if (data.filename === 'broken.xy') throw Error('Chart could not be initialized');
        };`);
    assert.throws(() => run(`{ const data = captureWorkspace(); data.filename = 'broken.xy'; restoreWorkspace(data); }`), /Chart could not/);
    assert.equal(run('loadedFileName'), 'original.xy');
    assert.deepEqual(JSON.parse(run('JSON.stringify(applied)')), ['broken.xy', 'original.xy']);
});

test('every solution badge escapes imported text and quoted attributes', () => {
    const { run } = harness();
    const rendered = run(`(() => {
        const attack = '\"><img src=x onerror=window.injected=1>';
        const sol = solutions[0];
        sol.sgClass = attack;
        sol.spaceGroups = [{ label: attack, symbol: attack }];
        const badges = solutionBadges(sol);
        sol._symAlt = [{ system: attack, a:5,b:5,c:5,alpha:90,beta:90,gamma:90,m20:10 }];
        return badges + symPairBadges(sol);
    })()`);
    assert.ok(!rendered.includes('<img'));
    assert.ok(rendered.includes('&quot;&gt;&lt;img'));
    assert.ok(!rendered.includes('title=""><'));
});

test('infinite margins remain decisive and missing MC source merit stays printable', () => {
    const { run } = harness();
    const output = run(`(() => {
        const sol = solutions[0];
        sol.sgClass = 'P--'; sol.sgMargin = Infinity;
        const verdict = solutionBadges(sol) + solutionSpaceGroupText(sol) + solutionSGHtml(sol);
        delete sol.sgClass; sol.mcPolished = true; sol.mcFrom = {m20:null};
        return {verdict, mc:solutionBadges(sol)};
    })()`);
    assert.ok(!output.verdict.includes('not decisive'));
    assert.ok(!output.verdict.includes('SG?'));
    assert.ok(output.mc.includes('M20 ?'));
});
