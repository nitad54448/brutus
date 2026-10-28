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
