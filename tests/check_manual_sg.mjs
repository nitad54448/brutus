import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const read = p => readFileSync(p, 'utf8');
const element = () => ({ value: '', addEventListener() {}, replaceChildren() {}, add() {} });
const context = vm.createContext({ console, document: { getElementById: element },
    Chart: { register() {} }, spaceGroupData: JSON.parse(read('sg_ops.json')) });
const manifest = read('js/crystallography/manifest.js');
for (const match of manifest.matchAll(/^\s*'([^']+\.js)',\s*$/gm))
    vm.runInContext(read('js/crystallography/' + match[1]), context);
vm.runInContext(read('js/chart/manual-space-group.js'), context);
vm.runInContext(`
    const cell = { a: 5, b: 5, c: 5, system: 'cubic' };
    const saved = () => true;
    setSpaceGroupFilter(saved);
    const face = manualSGReflections(cell, spaceGroupData.space_groups['225'].settings[0], 1.54, 80);
    if (!face.length || face.some(r => !((Math.abs(r.h)%2 === Math.abs(r.k)%2) && (Math.abs(r.k)%2 === Math.abs(r.l)%2)))) throw Error('F extinction failed');
    if (getSpaceGroupFilter() !== saved) throw Error('Filter leaked');
    const pyrite = manualSGReflections(cell, spaceGroupData.space_groups['205'].settings[0], 1.54, 80);
    if (!pyrite.some(r => r.h*r.h+r.k*r.k+r.l*r.l === 5)) throw Error('Pa-3 210 lost');
    for (const group of Object.values(spaceGroupData.space_groups)) {
        for (const setting of group.settings) {
            const lines = manualSGReflections(cell, setting, 1.54, 35);
            if (!Array.isArray(lines) || lines.some(r => !Number.isFinite(r.tth))) throw Error(setting.setting_id);
        }
    }
`, context);
assert.ok(true);
console.log('PASS: all 530 settings calculate, F extinctions, Pa-3 210 and filter restoration.');
