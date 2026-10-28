// Run with: node --test tests/check_parser_regressions.mjs
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function harness() {
    let downloaded;
    const node = () => ({ addEventListener() {}, click() {}, classList: { contains: () => false } });
    const context = vm.createContext({
        console, Blob, setTimeout: fn => fn(),
        document: { getElementById: node, createElement: node, addEventListener() {}, body: { appendChild() {}, removeChild() {} } },
        URL: { createObjectURL: blob => { downloaded = blob; return 'blob:test'; }, revokeObjectURL() {} },
        ui: { wavelengthPreset: { value: 'custom' }, stripKa2Checkbox: { checked: false },
            tthMinSlider: { value: '0' }, tthMaxSlider: { value: '180' } },
        WAVELENGTH_PRESETS: {}, getWavelength: () => 1.54056,
        solutionSGs: sol => sol.spaceGroups || [],
        fullExperimentalData: { tth: [10, 11], intensity: [100, 200] },
        loadedFileName: 'test.xy', showStatus() {}, window: {}
    });
    const load = file => vm.runInContext(readFileSync(new URL('../' + file, import.meta.url), 'utf8'), context, { filename: file });
    load('js/crystallography/limits.js');
    load('js/core/utils.js');
    for (const file of ['fullprof', 'generic', 'xml-utils', 'bruker-xml', 'xrdml', 'rigaku-ras', 'gsas', 'uxd', 'philips-udf', 'pdcif', 'detect']) load(`js/parsers/${file}.js`);
    load('js/data/export.js');
    load('js/report/pdcif-export.js');
    return { call: name => vm.runInContext(name, context), downloaded: () => downloaded };
}
const plain = result => JSON.parse(JSON.stringify(result));
const h = harness();
const parseCif = h.call('parsePdCifFile');
const cifHeader = 'data_test\nloop_\n_pd_meas_2theta_scan\n_pd_meas_intensity_total\n';

test('pdCIF inline comments preserve loop columns and quoted hashes remain values', () => {
    const result = parseCif(cifHeader + '10 100 # first point\n11 200\n12 300\n13 400\n');
    assert.deepEqual(plain(result.tth), [10, 11, 12, 13]);
    assert.deepEqual(plain(result.intensity), [100, 200, 300, 400]);
    const quoted = parseCif("data_test\n_audit_note 'a # literal comment'\nloop_\n_label\n_pd_meas_2theta_scan\n_pd_meas_intensity_total\n'#one' 10 100\n'_two' 11 200\n'loop_' 12 300\n");
    assert.deepEqual(plain(quoted.tth), [10, 11, 12]);
    assert.deepEqual(plain(quoted.intensity), [100, 200, 300]);
});

test('pdCIF scalar values can follow tags on later lines and text fields do not shift loops', () => {
    const result = parseCif('data_test\n_note\n;A multiline\ncomment # literal\n;\n_pd_meas_2theta_range_min\n10\n_pd_meas_2theta_range_inc # inline comment\n0.5\n_diffrn_radiation_wavelength\n1.54056\nloop_\n_pd_meas_intensity_total\n100\n200\n300\n');
    assert.deepEqual(plain(result.tth), [10, 10.5, 11]);
    assert.equal(result.radiation.single, 1.54056);
    assert.throws(() => parseCif(cifHeader + '10 100\n11\n'), /incomplete data loop row/i);
});

test('pdCIF wavelength weights stay paired when a wavelength is missing', () => {
    const result = parseCif(cifHeader + '10 100\n11 200\nloop_\n_diffrn_radiation_wavelength\n_diffrn_radiation_wavelength_wt\n? 999\n1.54056 1\n1.54439 0.5\n');
    assert.equal(result.radiation.ratio, 0.5);
});

test('UXD rejects invalid channels and incomplete pairs instead of shifting data', () => {
    const parse = h.call('parseUxdFile');
    assert.throws(() => parse('_START=10\n_STEPSIZE=0.02\n_COUNTS\n100 NaN 300 400\n'), /invalid numeric data on line 4/i);
    assert.throws(() => parse('_2THETACOUNTS\n10 100 10.02 NaN 10.04 300\n'), /invalid numeric data/i);
    assert.throws(() => parse('_2THETACOUNTS\n10 100 10.02\n'), /incomplete angle\/intensity pair/i);
    const result = parse('_START=10\n_STEPSIZE=0.02\n_COUNTS\n100 200 300\n');
    assert.deepEqual(plain(result.tth), [10, 10.02, 10.04]);
    assert.deepEqual(plain(result.intensity), [100, 200, 300]);
});

test('recognized unsupported files never fall back to generic two-column data', () => {
    const detect = h.call('detectAndParseFile');
    assert.throws(() => detect('tof.gsa', 'BANK 1 3 3 TOF 1000 50 0 0 FXYE\n1000 100 10\n1050 200 14\n1100 300 17\n'), /not supported/i);
    assert.throws(() => detect('broken.uxd', '_START=10\n_STEPSIZE=0.02\n_COUNTS\n100 NaN 300\n'), /invalid numeric/i);
    assert.deepEqual(plain(detect('ordinary.xy', '10 100\n11 200\n12 300\n').tth), [10, 11, 12]);
    assert.deepEqual(plain(detect('fullprof.dat', '!COMM Test scan\n10 0.5 12\n100 200 300 400 500\n').tth), [10, 10.5, 11, 11.5, 12]);
});

test('bundled UXD fixtures continue to load through format detection', () => {
    const detect = h.call('detectAndParseFile');
    for (const filename of readdirSync(new URL('../test_files/', import.meta.url)).filter(file => /\.uxd$/i.test(file))) {
        const result = detect(filename, readFileSync(new URL('../test_files/' + filename, import.meta.url), 'utf8'));
        assert.ok(result.tth.length > 0, filename);
        assert.equal(result.tth.length, result.intensity.length, filename);
    }
});

test('irregular UXD and GSAS exports round-trip explicit angles without moving intensities', () => {
    const build = h.call('buildExportContent');
    const pattern = { tth: [10, 11, 30], intensity: [100, 200, 300] };
    for (const [format, parser] of [['uxd', 'parseUxdFile'], ['gsas', 'parseGsasFile']]) {
        const result = h.call(parser)(build(format, pattern).text);
        assert.deepEqual(plain(result.tth), pattern.tth, format);
        assert.deepEqual(plain(result.intensity), pattern.intensity, format);
    }
    const xml = build('xrdml', pattern).text;
    assert.match(xml, /<listPositions>10 11 30<\/listPositions>/);
    assert.throws(() => build('brukerxml', pattern), /requires evenly spaced angles/i);
    assert.doesNotThrow(() => build('brukerxml', { tth: [10, 10.02, 10.04], intensity: [1, 2, 3] }));
    const preciseGrid = { tth: Array.from({ length: 1001 }, (_, i) => 10 + i * 0.013130312345), intensity: Array(1001).fill(100) };
    const preciseResult = h.call('parseGsasFile')(build('gsas', preciseGrid).text);
    assert.ok(preciseResult.tth.every((angle, i) => Math.abs(angle - preciseGrid.tth[i]) < 1e-12));
});

test('pdCIF Hall, name and number all belong to the chosen group', async () => {
    const generate = h.call('generatePdCIF');
    const solution = { a: 5, b: 5, c: 5, volume: 125, sgHall: '-P 2ybc', simSG: '001_01',
        spaceGroups: [{ symbol: 'P 1', number: 1, hall: 'P 1', label: 'P 1', setting_id: '001_01' }] };
    generate(solution);
    let text = await h.downloaded().text();
    assert.match(text, /_space_group_name_H-M_alt\s+'P 1'/);
    assert.match(text, /_space_group_IT_number\s+1\b/);
    assert.match(text, /_space_group_name_Hall\s+'P 1'/);
    solution.spaceGroups[0].hall = '';
    generate(solution);
    text = await h.downloaded().text();
    assert.doesNotMatch(text, /_space_group_name_Hall/);
    assert.doesNotMatch(text, /-P 2ybc/);
});
