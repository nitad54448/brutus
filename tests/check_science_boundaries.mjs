// Numerical boundaries that rounded synthetic fixtures cannot exercise.
// Run: node --test tests/check_science_boundaries.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const read = path => readFileSync(new URL('../' + path, import.meta.url), 'utf8');
function science() {
    const context = vm.createContext({ console, URL });
    const manifest = read('js/crystallography/manifest.js');
    for (const match of manifest.matchAll(/^\s*'([^']+\.js)',\s*$/gm)) {
        const file = 'js/crystallography/' + match[1];
        vm.runInContext(read(file), context, { filename: file });
    }
    return context;
}

test('exact and near-exact cubic fits retain large finite merit and survive refinement', () => {
    const context = science();
    const result = vm.runInContext(`(() => {
        const wavelength = 1.54056;
        const cell = {system:'cubic', a:4, b:4, c:4, alpha:90, beta:90, gamma:90};
        const q = [1,2,3,4,5,6,8,9,10,11,12,13].map(n => n / 16);
        const peaks = q.map(q => ({tth: 2 * Math.asin(wavelength * Math.sqrt(q) / 2) * DEG, intensity:100}));
        const sorted = getSortedPeaks(peaks, wavelength);
        const state = {...sorted, N_FOR_M20:20, min_m20:2, q_max:q.at(-1),
            d_min:1 / Math.sqrt(q.at(-1)), foundSolutions:[], foundSolutionMap:new Map(), diag:{}};
        const data = {peaks,wavelength,tth_error:0.03,max_volume:250,impurity_peaks:0,refineZero:false};
        const exact = calculateFiguresOfMerit(q, sorted.peaks_sorted_by_q, 0, () => 0.01, wavelength);
        const near = calculateFiguresOfMerit(q.map(v => v + 1e-13), sorted.peaks_sorted_by_q, 0, () => 0.01, wavelength);
        const posted = [];
        refineAndTestSolution(cell, data, state, m => posted.push(m));
        return {exact,near,solutions:posted.filter(m => m.type === 'solution'),diag:state.diag};
    })()`, context);
    for (const merit of [result.exact, result.near]) {
        assert.ok(Number.isFinite(merit.m20) && merit.m20 > 1e6);
        assert.ok(Number.isFinite(merit.fN) && merit.fN > 1e6);
    }
    assert.ok(result.solutions.length > 0, JSON.stringify(result.diag));
    assert.ok(Math.abs(result.solutions[0].payload.a - 4) < 1e-10);
});

test('monoclinic beta uncertainty is continuous across the 90 degree canonical fold', () => {
    const context = science();
    const result = vm.runInContext(`[-1e-7,0,1e-7].map(D => {
        const solution = [0.04,0.04,0.04,D];
        const covarianceMatrix = solution.map((_,i) => solution.map((_,j) => i === j ? 1e-8 : 0));
        const cell = extractCellFromFit(solution, 'monoclinic');
        return propagateErrors('monoclinic', {solution,covarianceMatrix}, cell).s_beta;
    })`, context);
    const expected = 180 / Math.PI * Math.sqrt(1e-8) / 0.08;
    for (const uncertainty of result) assert.ok(Math.abs(uncertainty - expected) < 1e-6);
});

test('equivalent-cell volumes preserve the full metric despite approximate symmetry', () => {
    const context = science();
    const result = vm.runInContext(`(() => {
        const cell = {a:5,b:5.2,c:5.4,alpha:90,beta:90,gamma:90};
        const equivalent = generateEquivalentCells(cell, 0, 'orthorhombic');
        return [equivalent.primitiveCells[0], ...Object.values(equivalent.centeredCells)]
            .map(c => ({volume:c.volume, geometric:getVolumeTriclinic(c)}));
    })()`, context);
    assert.ok(Math.abs(result[0].volume - 140.4) < 1e-10);
    for (const cell of result) assert.ok(Math.abs(cell.volume - cell.geometric) < 1e-10);
});

test('least-squares refinement rejects final cells that cross axis or volume limits', () => {
    const context = science();
    const result = vm.runInContext(`(() => {
        const wavelength = 1.54056, truth = 5.01;
        const q = [1,2,3,4,5,6,8,9,10,11,12,13].map(n => n / truth ** 2);
        const peaks = q.map(q => ({tth: 2 * Math.asin(wavelength * Math.sqrt(q) / 2) * DEG, intensity:100}));
        const run = overrides => {
            const sorted = getSortedPeaks(peaks, wavelength);
            const state = {...sorted, N_FOR_M20:BRUTUS_LIMITS.search.meritPeaks,
                min_m20:BRUTUS_LIMITS.search.minM20, q_max:q.at(-1),
                d_min:1 / Math.sqrt(q.at(-1)), foundSolutions:[], foundSolutionMap:new Map()};
            const data = {peaks,wavelength,tth_error:0.5,max_volume:1000,min_axis:2,max_axis:5.1,
                min_volume:20,impurity_peaks:0,refineZero:false,...overrides};
            const posted = [];
            // The initial cell is within every tested bound. Only the final
            // fitted cell crosses the tighter axis/volume limits.
            refineAndTestSolution({a:4.99,system:'cubic'}, data, state, message => posted.push(message));
            return posted.filter(message => message.type === 'solution').map(message => message.payload);
        };
        return {control:run({}), axis:run({max_axis:5}), volume:run({max_volume:125})};
    })()`, context);
    assert.ok(result.control.length > 0, 'admissible fitted cell must survive');
    assert.ok(Math.abs(result.control[0].a - 5.01) < 1e-10);
    assert.equal(result.axis.length, 0, 'fitted axis must respect the per-run maximum');
    assert.equal(result.volume.length, 0, 'fitted volume must respect the per-run maximum');
});

test('a cell that omits b, as Convert leaves a tetragonal one, still has a solution key', () => {
    const context = science();
    const keys = vm.runInContext(`[getSolutionKey({system:'tetragonal', a:4, c:6}),
        getSolutionKey({system:'tetragonal', a:4, b:4, c:6})]`, context);
    assert.equal(keys[0], keys[1]);
});

test('the q tolerance at a calculated line is the tolerance at an observed peak', () => {
    const context = science();
    const ratios = vm.runInContext(`[10, 30, 60, 90, 120].map(tth => {
        const wl = 1.54056, t = tth * RAD, q = 4 * Math.sin(t / 2) ** 2 / wl ** 2;
        return qToleranceAtQ(q, wl, 0.04) / get_q_tolerance(0, [t], wl, 0.04);
    })`, context);
    for (const ratio of ratios) assert.ok(Math.abs(ratio - 1) < 1e-4, String(ratio));
});

test('a row added from the Space Group MC table carries its own margin', () => {
    const context = science();
    const r = vm.runInContext(`(() => {
        const lead = {score:40, nHardEff:0, mode:'mc'}, second = {score:30, nHardEff:0, mode:'mc'},
              third = {score:29, nHardEff:0, mode:'mc'}, dead = {score:90, nHardEff:3, mode:'mc'},
              shallow = {score:35, nHardEff:0, mode:'ls'};
        const rows = [lead, second, third, shallow, dead];
        const m = row => sgRowMargin(rows, row);
        return { lead: m(lead), second: m(second), third: m(third),
            deadOut: m(dead) === -Infinity, shallowOut: m(shallow) === -Infinity,
            alone: sgRowMargin([lead, dead], lead) === Infinity,
            decisive: [m(lead), m(second), m(dead), Infinity].map(v => sgMarginSummary(v).decisive),
            behind: sgMarginSummary(m(third)).text };
    })()`, context);
    assert.equal(r.lead, 10);
    assert.equal(r.second, -10);
    assert.equal(r.third, -11);
    assert.ok(r.deadOut && r.shallowOut && r.alone);
    assert.deepEqual(Array.from(r.decisive), [true, false, false, true]);
    assert.match(r.behind, /11\.0 nats behind/);
});

test('permissiveness is positions left open, not rules stated', () => {
    const context = science();
    assert.equal(vm.runInContext('sgOpenness({nAllowedGroups:5, stats:{nAllowedInRange:9}})', context), 5);
    assert.equal(vm.runInContext('sgOpenness({stats:{nAllowedInRange:9}})', context), 9);
    assert.equal(vm.runInContext('sgOpenness({})', context), 0);
});

test('durations carry a rounded minute instead of printing 60 seconds', () => {
    const line = read('js/indexing/run.js').match(/const minutesAndSeconds = [\s\S]*?\n};/)[0];
    const context = vm.createContext({});
    vm.runInContext(line.replace('const minutesAndSeconds', 'globalThis.split'), context);
    assert.deepEqual({ ...vm.runInContext('split(119600)', context) }, { min: 2, sec: 0 });
    assert.deepEqual({ ...vm.runInContext('split(61000)', context) }, { min: 1, sec: 1 });
});
