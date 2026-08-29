// Preflight: does sg_ops.json satisfy everything Brutus reads from it?
//
// build_sg_db.py --check validates the file against ITSELF (operators vs
// conditions). This validates it against the APP: the loader gates in
// main_app.js, every field worker-logic.js dereferences, and an actual
// extinction-class build through the shipping code.
//
//   node check_sg_ops.mjs [sg_ops.json]
//
// Run it after rebuilding the database and before reloading the browser -- a
// missing field shows up here in a second, and in the browser as a space-group
// panel that is quietly empty.
import { readFileSync, existsSync } from 'fs';
import vm from 'vm';

const FILE = process.argv[2] || 'sg_ops.json';
if (!existsSync(FILE)) {
    console.error(`No such file: ${FILE}`);
    console.error('Build it with:  python3 build_sg_db.py --out sg_ops.json');
    process.exit(2);
}
const db = JSON.parse(readFileSync(FILE, 'utf8'));

// Only the systems getSymmetry() can return. It never emits 'trigonal' -- a
// trigonal cell in hexagonal axes has a=b and gamma=120, which it correctly
// calls hexagonal -- and sgSystemMatches() folds the database's trigonal groups
// into the hexagonal run. Probing 'trigonal' here would count rhombohedral-axes
// settings as reachable through a call the app never makes.
const APP_SYSTEMS = ['triclinic','monoclinic','orthorhombic','tetragonal','hexagonal','cubic'];

console.log(`${FILE}: ${Object.keys(db.space_groups || {}).length} space groups, ` +
            `${db.setting_count ?? '?'} settings, ${(db.rotations||[]).length} rotations, ` +
            `${Object.keys(db.zone_defs||{}).length} zone labels\n`);

// 1. main_app.js loader gates, replicated exactly
const gates = [
  ["has space_groups", !!db.space_groups],
  ["has rotations",    !!db.rotations],
  ["has zone_defs",    !!db.zone_defs],
];
for (const [n,v] of gates) console.log(`  ${v?'ok  ':'FAIL'} loader gate: ${n}`);

// 2. every field worker-logic.js dereferences
const need = { group: ['number','standard_symbol','crystal_system','point_group',
                       'centrosymmetric','settings'],
               setting: ['symbol','hall','centering','t_den','ops','conditions'] };
let miss = [];
for (const g of Object.values(db.space_groups)) {
  for (const f of need.group) if (!(f in g)) miss.push(`group.${f}`);
  for (const st of g.settings) for (const f of need.setting) if (!(f in st)) miss.push(`setting.${f}`);
}
console.log(`  ${miss.length?'FAIL':'ok  '} every field the app reads is present` +
            (miss.length?`: missing ${[...new Set(miss)].join(', ')}`:''));

// 3. run it through the real worker-logic.js
const ctx = { console:{log(){},warn(){},error(){}}, self:{}, performance:{now:()=>Date.now()},
              postMessage(){}, addEventListener(){}, setTimeout, clearTimeout, Date, Math, JSON,
              Uint8Array, Int32Array, Float64Array, Float32Array, Map, Set, WeakMap,
              Object, Array, Number, String, isFinite, parseInt, parseFloat };
ctx.globalThis = ctx; vm.createContext(ctx);
try { vm.runInContext(readFileSync('./worker-logic.js','utf8'), ctx, {filename:'w'}); } catch(e){}
console.log(`  ${ctx.sgEnsureDatabase(db)?'ok  ':'FAIL'} sgEnsureDatabase() installs it`);

let total = 0;
for (const sys of APP_SYSTEMS) {
  const cs = ctx.sgExtinctionClasses(db, sys, null);
  total += cs.length;
  console.log(`       sgExtinctionClasses(${sys.padEnd(13)}) -> ${String(cs.length).padStart(2)} classes  ${cs.map(c=>c.label).join(' ')}`);
}
console.log(`  ${total>0?'ok  ':'FAIL'} classes build (${total} across all systems)`);

// 4. COVERAGE. A setting can sit in the file and never be used, silently.
//
//    Reachability is tested by applying the app's OWN filters to each setting
//    object, not by looking for it in the class output. Class members carry
//    only {number, symbol} and symbol is NOT unique -- origin choice 1 vs 2,
//    and hexagonal vs rhombohedral axes, share a Hermann-Mauguin symbol. Keyed
//    on that, a setting is scored reachable whenever a same-symbol sibling was
//    placed, which hides exactly the drops this check exists to find.
console.log('\n  coverage — settings reachable through sgExtinctionClasses:');
for (const fn of ['sgSystemMatches','sgSettingAxesMatch','settingCenteringAllowed','sgOpsCompile'])
  if (typeof ctx[fn] !== 'function') { console.log(`  !! ${fn} missing from worker-logic.js`); }

let eligible = 0;
const dropped = {};
const dupSymbols = new Map();
for (const g of Object.values(db.space_groups)) {
  for (const st of g.settings) {
    eligible++;
    const k = g.number + '|' + st.symbol;
    dupSymbols.set(k, (dupSymbols.get(k) || 0) + 1);

    const reachable = APP_SYSTEMS.some(sys =>
      ctx.sgSystemMatches(g.crystal_system, sys) &&
      ctx.sgSettingAxesMatch(st, sys) &&
      ctx.settingCenteringAllowed(st.symbol, null) &&
      !!ctx.sgOpsCompile(st));
    if (reachable) continue;

    const hall = String(st.hall || '');
    const sys = g.crystal_system;
    let why = 'unknown';
    if (sys === 'monoclinic' && !hall.includes('2y'))
      why = 'monoclinic, not b-unique (hall lacks "2y")';
    else if (hall.includes('3*'))
      why = 'rhombohedral axes (hall has "3*") — R lattices are indexed in hexagonal axes';
    else if (!APP_SYSTEMS.some(x => ctx.sgSystemMatches(sys, x)))
      why = `crystal_system ${JSON.stringify(sys)} matches no system the indexer emits`;
    else if (!ctx.sgOpsCompile(st))
      why = 'operators failed to compile';
    (dropped[why] = dropped[why] || []).push(`#${g.number} ${st.symbol} hall=${hall}`);
  }
}
const nDup = [...dupSymbols.values()].filter(v => v > 1).length;
const nDropped = Object.values(dropped).reduce((n, a) => n + a.length, 0);
console.log(`       ${eligible - nDropped}/${eligible} settings reachable` +
            (nDup ? `   (${nDup} symbols shared by >1 setting — origin choices and hex/rhomb axes)` : ''));
for (const [why, list] of Object.entries(dropped)) {
  const expected = why !== 'unknown' && !why.startsWith('operators');
  console.log(`  ${expected ? 'ok  ' : '!!  '} ${String(list.length).padStart(3)} excluded — ${why}`);
  if (!expected) for (const x of list.slice(0, 12)) console.log(`         ${x}`);
}
if (!nDropped) console.log('  ok    every setting reachable');
console.log('       Excluding a-/c-unique monoclinic and rhombohedral-axes settings');
console.log('       is CORRECT: the indexer produces b-unique monoclinic cells and');
console.log('       indexes R lattices in hexagonal axes, and a condition list written');
console.log('       for other axes refers to different indices. "unknown", a failed');
console.log('       compile, or a whole system dropping is not.');

const bad = !db.space_groups || !db.rotations || !db.zone_defs || miss.length ||
            total === 0 || (dropped['unknown'] || []).length > 0 ||
            (dropped['operators failed to compile'] || []).length > 0;
console.log('\n' + (bad ? 'FAIL: Brutus will not use this file correctly.'
                         : 'PASS: Brutus can consume this file.'));
process.exit(bad ? 1 : 0);
