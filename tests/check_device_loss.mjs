// Exercise the real GPU loss callback, abort path and finalization without a GPU.
// Run: node --test tests/check_device_loss.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const read = path => readFileSync(new URL('../' + path, import.meta.url), 'utf8');
function harness(init = async () => {}) {
    const counters = {terminated:0,released:0,sieved:0};
    const element = () => ({value:'0',checked:false,disabled:false,textContent:'',style:{},
        classList:{add(){},remove(){}},addEventListener(){},parentElement:{style:{}}});
    const nodes = new Map();
    const getElement = id => { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); };
    const ui = new Proxy({}, {get(target,key) { return target[key] ||= element(); }});
    ui.systemCheckboxes = ['cubic','orthorhombic'].map(value => ({...element(),value,checked:true}));
    ui.tthMinSlider.value = '5'; ui.tthMaxSlider.value = '80';
    const context = vm.createContext({
        console:{log(){},warn(){},error(){}}, URL, setTimeout, clearTimeout,
        performance:{now:() => Date.now()}, ui,
        window:{addEventListener(){}}, document:{getElementById:getElement},
        refinementPool:{setHandlers(){},terminate(){counters.terminated++;},release(){counters.released++;}},
        applyFinalSieve(cells) { counters.sieved++; return cells; },
        perfStart:() => () => {}, showStatus(){}, sortSolutions(){}, updateSolutionsTable(){},
        updateStartIndexingButtonState(){}, getWavelength:() => 1.54056,
        getTthError:() => 0.03,getImpurityPeaks:() => 0,
        WebGPUEngine:class {
            constructor() {this.usable = true;}
            init() {return init();}
            isUsable() {return this.usable;}
            destroy() {this.usable = false;}
        },
    });
    const manifest = read('js/crystallography/manifest.js');
    for (const match of manifest.matchAll(/^\s*'([^']+\.js)',\s*$/gm)) {
        vm.runInContext(read('js/crystallography/' + match[1]),context);
    }
    vm.runInContext(read('js/core/state.js'),context);
    vm.runInContext(`function setUIState(value) { isIndexing = value; }
        const spaceGroupData = null;
        let gpuDiagnostics = [], cpuDiagnostics = [];`,context);
    vm.runInContext(read('js/gpu/gpu-setup.js'),context);
    vm.runInContext(read('js/indexing/run.js'),context);
    return {context,counters,run: source => vm.runInContext(source,context)};
}

test('device loss finalizes partial results and failure provenance, then ignores stale notifications', async () => {
    const {run,counters} = harness();
    await run('getWebGPUEngine()');
    run(`isIndexing = true; indexingRunToken = 7; indexingStartTime = performance.now() - 100;
        solutions = [{system:'cubic',a:4,b:4,c:4,alpha:90,beta:90,gamma:90,volume:64,m20:100,_runToken:7}];
        pickedPeaks = [{tth:20,height:100}];
        cumulativeTrials = 25; lastRunTotals = {done:999999};
        activeWorkers = [{terminate(){}}];
        sharedWebGPUEngine.onDeviceLost({reason:'unknown'});`);
    const snapshot = run(`({isIndexing,indexingRunToken,workers:activeWorkers.length,
        totals:lastRunTotals,stats:lastIndexingStats,analysis:solutions[0].analysis})`);
    assert.equal(snapshot.isIndexing,false);
    assert.equal(snapshot.indexingRunToken,8);
    assert.equal(snapshot.workers,0);
    assert.equal(snapshot.totals.done,25);
    assert.match(snapshot.totals.failures[0],/GPU device lost/);
    assert.match(snapshot.stats,/INCOMPLETE/);
    assert.ok(snapshot.analysis.hklList.length > 0);
    assert.equal(counters.terminated,1);
    assert.equal(counters.sieved,1);
    // A destroyed/replaced engine must not abort the next run.
    run(`globalThis.oldEngine = sharedWebGPUEngine; sharedWebGPUEngine.usable = false;`);
    await run('getWebGPUEngine()');
    run(`isIndexing = true; oldEngine.onDeviceLost({reason:'destroyed'});`);
    assert.equal(run('isIndexing'),true);
    assert.equal(run('indexingRunToken'),8);
    assert.equal(counters.terminated,1);
});

test('starting a run clears old statistics before async initialization and stale tails cannot release a newer pool', async () => {
    let finishInit;
    const pendingInit = new Promise(resolve => {finishInit = resolve;});
    const {run,counters} = harness(() => pendingInit);
    run(`lastRunTotals = {done:999999}; lastIndexingStats = 'old run'; cumulativeTrials = 999999;
        taskTotals = [999999]; taskProgress = [100];`);
    const pendingRun = run('startIndexing()');
    const snapshot = run('({totals:lastRunTotals,stats:lastIndexingStats,trials:cumulativeTrials,tasks:taskTotals.length})');
    assert.equal(snapshot.totals,null);
    assert.equal(snapshot.stats,'');
    assert.equal(snapshot.trials,0);
    assert.equal(snapshot.tasks,0);
    run('abortActiveIndexing(false)');
    finishInit();
    await pendingRun;
    assert.equal(run('isIndexing'),false);
    assert.equal(counters.released,0);
});
