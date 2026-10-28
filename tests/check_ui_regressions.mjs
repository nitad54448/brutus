// Run with: node --test tests/check_ui_regressions.mjs
// Exercise actual event handlers in isolated browser-like contexts, without
// a GPU or dependencies. Numerical crystallography has its own regression suite.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const read = path => readFileSync(new URL('../' + path, import.meta.url), 'utf8');

function harness() {
    const elements = new Map();
    const documentEvents = new Map();
    const document = {
        activeElement: null,
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, new Element());
            return elements.get(id);
        },
        createElement: tag => new Element(tag),
        createDocumentFragment: () => new Element('fragment'),
        addEventListener(type, fn) {
            if (!documentEvents.has(type)) documentEvents.set(type, []);
            documentEvents.get(type).push(fn);
        },
        emit(type, event) {
            for (const fn of documentEvents.get(type) || []) fn(event);
        }
    };
    class Element {
        constructor(tag = 'div') {
            this.tagName = tag.toUpperCase();
            this.value = this.defaultValue = this.min = this.max = '';
            this.disabled = this.checked = false;
            this.isConnected = true;
            this.children = [];
            this.dataset = {};
            this.style = {};
            this.events = new Map();
            const classes = new Set();
            this.classList = {
                add: (...names) => names.forEach(name => classes.add(name)),
                remove: (...names) => names.forEach(name => classes.delete(name)),
                contains: name => classes.has(name),
                toggle(name, force = !classes.has(name)) {
                    if (force) classes.add(name); else classes.delete(name);
                    return force;
                }
            };
        }
        addEventListener(type, fn) {
            if (!this.events.has(type)) this.events.set(type, []);
            this.events.get(type).push(fn);
        }
        async emit(type, event = {}) {
            event.target ||= this;
            for (const fn of this.events.get(type) || []) await fn(event);
        }
        focus() {
            document.activeElement = this;
            document.emit('focusin', { target: this });
        }
        contains(target) {
            return target === this || this.children.some(child => child.contains(target));
        }
        appendChild(child) {
            if (child.tagName === 'FRAGMENT') this.children.push(...child.children);
            else this.children.push(child);
        }
        closest(selector) {
            return selector === 'input, button' && ['INPUT', 'BUTTON'].includes(this.tagName) ? this : null;
        }
        querySelectorAll(selector) {
            assert.equal(selector, 'input, button');
            return this.children.flatMap(child => [
                ...(['INPUT', 'BUTTON'].includes(child.tagName) ? [child] : []),
                ...child.querySelectorAll(selector)
            ]);
        }
        set innerHTML(html) {
            this.children = [];
            for (const match of html.matchAll(/<(input|button)\b([^>]*)>/g)) {
                const child = new Element(match[1]);
                const attrs = match[2];
                for (const name of (attrs.match(/class="([^"]*)"/)?.[1] || '').split(' ').filter(Boolean)) child.classList.add(name);
                child.dataset.index = attrs.match(/data-index="([^"]*)"/)?.[1];
                child.value = attrs.match(/value="([^"]*)"/)?.[1] || '';
                child.disabled = /\bdisabled\b/.test(attrs);
                this.appendChild(child);
            }
        }
        getBoundingClientRect() { return { left: 0 }; }
    }
    const ui = new Proxy({}, {
        get(target, name) { return target[name] ||= new Element(); }
    });
    for (const [key, id] of Object.entries({
        saveMenuOverlay: 'save-menu-overlay', saveFormatSelect: 'save-format-select',
        saveMenuCancel: 'save-menu-cancel', saveMenuConfirm: 'save-menu-confirm'
    })) ui[key] = document.getElementById(id);
    ui.systemCheckboxes = [];
    const modal = document.getElementById('save-menu-modal');
    for (const control of [ui.saveFormatSelect, ui.saveMenuCancel, ui.saveMenuConfirm]) modal.appendChild(control);
    ui.saveMenuOverlay.appendChild(modal);
    ui.wavelength.value = ui.wavelength.defaultValue = '1.54184';
    ui.wavelengthPreset.value = 'Cu_avg';
    ui.tthError.value = ui.tthError.defaultValue = '0.04';
    ui.saveFormatSelect.value = 'xy';
    const statuses = [];
    const context = vm.createContext({
        document, ui, console: { log() {}, warn() {}, error() {} },
        window: { innerWidth: 1200, addEventListener() {} },
        getComputedStyle: () => ({ minWidth: '300px', maxWidth: '700px' }),
        debounce: fn => fn,
        showStatus: message => statuses.push(message), perfStart: () => () => {},
        isIndexing: false, pickedPeaks: [], peaksManuallyEdited: false,
        selectedPeakIndex: null, fullExperimentalData: { tth: [], intensity: [] },
        workingExperimentalData: { tth: [], intensity: [] }, loadedFileName: '',
        getWavelength: () => 1.54184, getLambda: () => 1.54184,
        updatePeakTable() {}, updateAllMarkers() {}, updateKa2Banner() {},
        rebuildHklList() {}, updateStartIndexingButtonState() {}, updateSolutionsTable() {},
        recalculatePeakValues() {}, abortActiveIndexing() {}, handleWavelengthPresetChange() {},
        ka2StripAllowed: () => true,
        updateWorkingData() {}, setExperimentalTrace() {}, setupTthSliders() {}, findPeaks() {},
        xInv: value => value,
        xrdChart: { canvas: new Element('canvas'), chartArea: { left: 0, right: 100 },
            scales: { x: { getValueForPixel: value => value } },
            data: { datasets: [] }, update() {}, resetZoom() {} }
    });
    vm.runInContext(read('js/crystallography/limits.js'), context);
    return {
        context, ui, document, statuses,
        load: path => vm.runInContext(read(path), context, { filename: path }),
        evaluate: expression => vm.runInContext(expression, context),
        value: expression => JSON.parse(vm.runInContext('JSON.stringify(' + expression + ')', context))
    };
}

test('number inputs accept exponents, truncate integers, clamp and preserve empty-field defaults', () => {
    const h = harness();
    h.load('js/core/inputs.js');
    const input = (value, defaultValue = '100') => ({ value, defaultValue, min: '1', max: '500' });
    const readNumber = h.evaluate('readNumberInput');
    for (const [value, expected] of [['1e2', 100], ['1.5e2', 150], ['3.9', 3], ['1e4', 500], ['-2', 1], ['', 100], ['bad', 100]]) {
        assert.equal(readNumber(input(value), { integer: true }), expected, value);
    }
    assert.equal(readNumber(input('', '2e2'), { integer: true }), 200);
    assert.equal(readNumber(input('0', '100'), { integer: true }), 1);
    assert.equal(readNumber(input('1e-2'), {}), 1);
});

test('tolerance changes refresh Ka2 exclusions and parent wavelengths without replacing hand edits', async () => {
    const h = harness();
    h.load('js/core/inputs.js');
    h.load('js/data/wavelength.js');
    h.load('js/peaks/peak-finder.js');
    h.ui.tthError.value = '0.1';
    h.evaluate(`pickedPeaks = [{ tth: 60, height: 100 },
        { tth: expectedKa2TthDeg(60, WAVELENGTH_PRESETS.Cu) + 0.06, height: 50 },
        { tth: 80, height: 10, userConfirmedReal: true }]; recalculatePeakValues();`);
    assert.equal(h.value('pickedPeaks[1].ka2Suspect'), true);
    assert.equal(h.value('pickedPeaks[0].lambda'), 1.54056);
    h.ui.tthError.value = '0.01';
    await h.ui.tthError.emit('change');
    assert.equal(h.value('pickedPeaks[1].ka2Suspect'), false);
    assert.equal(h.evaluate('pickedPeaks[0].lambda'), undefined);
    assert.equal(h.value('pickedPeaks[2].userConfirmedReal'), true);
    assert.equal(h.value('pickedPeaks.length'), 3);
});

test('indexing locks chart additions, table mutations, rebuilt controls and deferred peak finding', async () => {
    const h = harness();
    h.load('js/chart/interaction.js');
    h.load('js/peaks/peak-table.js');
    h.evaluate('pickedPeaks = [{ tth: 30, d: 3, q: 1/9, height: 100 }]; isIndexing = true; updatePeakTable();');
    const controls = h.ui.peakListBody.querySelectorAll('input, button');
    assert.ok(controls.every(control => control.disabled));
    await h.ui.chartCanvas.emit('click', { ctrlKey: true, clientX: 40 });
    await h.ui.peakListBody.emit('change', { target: controls[0] });
    await h.ui.peakListBody.emit('click', { target: controls[2] });
    await h.ui.peakListBody.emit('contextmenu', { target: controls[0] });
    assert.deepEqual(h.value('pickedPeaks.map(p => p.tth)'), [30]);
    h.evaluate('isIndexing = false; setPeakTableLocked(false);');
    assert.equal(controls[0].disabled, false);
    assert.equal(controls[1].disabled, true, 'derived d-spacing stays disabled after unlocking');
    assert.equal(controls[2].disabled, false);
    await h.ui.chartCanvas.emit('click', { ctrlKey: true, clientX: 40 });
    assert.deepEqual(h.value('pickedPeaks.map(p => p.tth)'), [30, 40]);
    h.load('js/peaks/peak-finder.js');
    h.evaluate('isIndexing = true; workingExperimentalData = {tth: [1,2,3,4,5], intensity: [0,1,5,1,0]}; findPeaks();');
    assert.deepEqual(h.value('pickedPeaks.map(p => p.tth)'), [30, 40]);
});

function fileHarness() {
    const h = harness();
    h.load('js/core/state.js');
    h.load('js/parsers/generic.js');
    h.evaluate('xrdChart = { resetZoom() {}, update() {}, data: { datasets: [] } };');
    h.context.detectAndParseFile = (name, text) => h.evaluate('parseDataFile')(text, name);
    h.context.classifyFileWavelength = () => ({});
    h.load('js/data/session.js');
    h.load('js/data/file-load.js');
    h.select = (name, text) => h.ui.fileInput.emit('change', {
        target: { files: [{ name, size: 100, text: typeof text === 'function' ? text : async () => text }] }
    });
    return h;
}

test('file selection order wins over read completion order, including stale errors and unloads', async () => {
    const h = fileHarness();
    let finishOld;
    const oldRead = h.select('old.xy', () => new Promise(resolve => { finishOld = resolve; }));
    await h.select('new.xy', '10 1\n20 2\n30 3');
    finishOld('10 100\n20 200\n30 300');
    await oldRead;
    assert.equal(h.value('loadedFileName'), 'new.xy');
    assert.deepEqual(h.value('fullExperimentalData.intensity'), [1, 2, 3]);

    let failOld;
    const failedRead = h.select('old-error.xy', () => new Promise((_, reject) => { failOld = reject; }));
    await h.select('latest.xy', '10 4\n20 5\n30 6');
    const statusCount = h.statuses.length;
    failOld(new Error('stale read error'));
    await failedRead;
    assert.equal(h.statuses.length, statusCount);
    assert.equal(h.ui.fileInputLabel.classList.contains('error'), false);

    let finishSuperseded;
    const superseded = h.select('superseded.xy', () => new Promise(resolve => { finishSuperseded = resolve; }));
    await h.select('invalid.xy', 'no data');
    finishSuperseded('10 100\n20 200\n30 300');
    await superseded;
    assert.equal(h.value('loadedFileName'), 'latest.xy', 'a failed new selection preserves the installed pattern');
    assert.deepEqual(h.value('fullExperimentalData.intensity'), [4, 5, 6]);

    let finishPending;
    const pending = h.select('pending.xy', () => new Promise(resolve => { finishPending = resolve; }));
    h.evaluate('clearLoadedFile()');
    finishPending('10 7\n20 8\n30 9');
    await pending;
    assert.equal(h.value('loadedFileName'), '');
    assert.deepEqual(h.value('fullExperimentalData.tth'), []);
});

test('file normalization sorts all unordered scans and keeps intensity paired with angle', async () => {
    const h = fileHarness();
    for (const data of ['10 100\n30 300\n20 200\n40 400', '40 400\n30 300\n20 200\n10 100', '10 100\n20 200\n30 300\n40 400']) {
        await h.select('unordered.xy', data);
        assert.deepEqual(h.value('fullExperimentalData.tth'), [10, 20, 30, 40]);
        assert.deepEqual(h.value('fullExperimentalData.intensity'), [100, 200, 300, 400]);
    }
    h.evaluate('workingExperimentalData = fullExperimentalData;');
    h.load('js/chart/interaction.js');
    assert.equal(h.evaluate('workingExperimentalData.tth[nearestDataIndex(20)]'), 20);
});

test('Save dialog traps focus, skips disabled buttons and returns focus on Escape or Cancel', async () => {
    const h = harness();
    h.context.workingExperimentalData = { tth: [10, 20, 30], intensity: [1, 2, 3] };
    h.load('js/data/export.js');
    h.ui.saveAsButton.focus();
    await h.ui.saveAsButton.emit('click');
    assert.equal(h.document.activeElement, h.ui.saveFormatSelect);
    const key = (key, shiftKey = false) => {
        const event = { key, shiftKey, prevented: false, preventDefault() { this.prevented = true; } };
        h.document.emit('keydown', event);
        return event;
    };
    assert.equal(key('Tab', true).prevented, true);
    assert.equal(h.document.activeElement, h.ui.saveMenuConfirm);
    assert.equal(key('Tab').prevented, true);
    assert.equal(h.document.activeElement, h.ui.saveFormatSelect);
    h.ui.saveMenuConfirm.disabled = true;
    key('Tab', true);
    assert.equal(h.document.activeElement, h.ui.saveMenuCancel);
    h.ui.wavelength.focus();
    assert.equal(h.document.activeElement, h.ui.saveFormatSelect);
    key('Escape');
    assert.equal(h.ui.saveMenuOverlay.classList.contains('open'), false);
    assert.equal(h.document.activeElement, h.ui.saveAsButton);
    await h.ui.saveAsButton.emit('click');
    await h.ui.saveMenuCancel.emit('click');
    assert.equal(h.document.activeElement, h.ui.saveAsButton);
});
