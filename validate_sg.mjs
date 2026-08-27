// Validate the rewritten space-group layer against a packed database.
//
//   node validate_sg.mjs [sgDir] [packedFile] [boxRadius]
//   node validate_sg.mjs sg sg_ops.json 5
//
// Checks, in order:
//   1. absences from the operators vs the generator's own zone records
//   2. zoneApplies (shared zone_defs) vs each setting's own normals
//   3. extinction classes build for every crystal system
//   4. countViolations runs and names its violations
//   5. the Wilson m*epsilon weight
//
// Check 1 is the one that matters. It compares two derivations sharing nothing
// but the operators: worker-logic.js loops (R,t) per reflection asking whether
// hR = h with h.t non-integral, while the generator built integer kernels of
// R - I and reduced the phase congruences by Smith normal form -- and verified
// the result against cctbx. Agreement over the whole database is what justifies
// having deleted the rule-string predicate layer.
// Referees the operator path against the generator's own zone records, which
// are derived independently (integer kernels of R-I, congruences reduced by
// Smith normal form) and verified against cctbx inside the generator.
import { readFileSync, readdirSync, existsSync } from 'fs';
import { join } from 'path';
import vm from 'vm';

const SG_DIR = process.argv[2] || 'sg';
const PACKED = process.argv[3] || 'sg_ops.json';
const B = parseInt(process.argv[4] || '6', 10);
if (!existsSync(SG_DIR) || !existsSync(PACKED)) {
    console.error(`Need ${SG_DIR}/ and ${PACKED}.`);
    console.error('Build with:  python3 sg_pack.py --sg-dir sg --out sg_ops.json');
    process.exit(2);
}

const ctx = { console, self: {}, performance: { now: () => Date.now() },
              postMessage(){}, addEventListener(){}, setTimeout, clearTimeout,
              Date, Math, JSON, Uint8Array, Int32Array, Float64Array, Float32Array,
              Map, Set, WeakMap, Object, Array, Number, String, isFinite, parseInt, parseFloat };
ctx.globalThis = ctx;
vm.createContext(ctx);
try { vm.runInContext(readFileSync('./worker-logic.js','utf8'), ctx, {filename:'worker-logic.js'}); }
catch (e) { console.log('load note:', e.message); }

const need = ['sgInstallDatabase','sgEnsureDatabase','zoneApplies','sgOpsCompile',
              'sgOpsAbsent','sgOpsAllowedFn','sgOpsLabelPredicate','sgOpsEpsilon',
              'sgOpsCenteringPred','sgSettingConditions','sgExtinctionClasses',
              'countViolations','sgHolohedryWeight','detectExtinctions','rankSpaceGroups'];
const missing = need.filter(n => typeof ctx[n] !== 'function');
if (missing.length) { console.log('MISSING:', missing.join(', ')); process.exit(1); }
// Function declarations become global properties; top-level `const` does not.
// So the removed FUNCTIONS are checked on the context and the removed TABLES on
// the source text, which is the only way to see them at all.
const src = readFileSync('./worker-logic.js','utf8');
const goneFns = ['sgAllowedFn','sgLabelPredicate','sgLaueOrbit'].filter(n => typeof ctx[n] === 'function');
const goneTables = ['ZONE_PREDICATES','SG_CENTERING_PRED','SG_LAUE_OF_POINT_GROUP']
    .filter(n => new RegExp('^const ' + n + '\\s*=', 'm').test(src));
const gone = [...goneFns, ...goneTables];
console.log(gone.length
  ? `  !! old layer still present: ${gone.join(', ')}`
  : '  old rule-string layer removed: sgAllowedFn, sgLabelPredicate, sgLaueOrbit,\n' +
    '  ZONE_PREDICATES, SG_CENTERING_PRED, SG_LAUE_OF_POINT_GROUP');

const db = JSON.parse(readFileSync(PACKED,'utf8'));
console.log(`database: ${db.setting_count} settings, ${db.rotations.length} rotations, ` +
            `${Object.keys(db.zone_defs).length} zone labels\n`);
ctx.sgInstallDatabase(db);

// --- 1. absences vs the generator's zone records ---------------------------
function compileRule(s) {
  const m = /^(.*)=(\d+)n$/.exec(String(s).replace(/\s+/g,''));
  if (!m) return null;
  const mod = parseInt(m[2],10);
  const terms = m[1].match(/[+-]?(?:\d+\*)?[hkl]/g); if (!terms) return null;
  const c = {h:0,k:0,l:0};
  for (const t of terms) {
    const p = /^([+-]?)(?:(\d+)\*)?([hkl])$/.exec(t); if (!p) return null;
    c[p[3]] += (p[1]==='-'?-1:1) * (p[2]?parseInt(p[2],10):1);
  }
  return (h,k,l) => ((((c.h*h+c.k*k+c.l*l)%mod)+mod)%mod)===0;
}
const raws = new Map();
for (const f of readdirSync(SG_DIR).filter(x => /^setting_\d+\.json$/.test(x))) {
  const d = JSON.parse(readFileSync(join(SG_DIR,f),'utf8'));
  d._z = (d.reflection_zones||[]).map(z => ({normals:z.normals||[], fns:(z.rules||[]).map(compileRule)}));
  raws.set(d.setting_id || d.symbol, d);
}
function truthAbsent(h,k,l,recs) {
  for (const z of recs) {
    let inZ = true;
    for (const n of z.normals) if (n[0]*h+n[1]*k+n[2]*l !== 0) { inZ=false; break; }
    if (!inZ) continue;
    for (const fn of z.fns) if (fn && !fn(h,k,l)) return true;
  }
  return false;
}
console.log('1. absences: operators vs the generator\'s zone records');
let bad = 0, nRef = 0, badZone = 0;
for (const g of Object.values(db.space_groups)) for (const st of g.settings) {
  const C = ctx.sgOpsCompile(st);
  const raw = raws.get(st.setting_id || st.symbol);
  if (!raw) { console.log(`   ${st.symbol}: no source file matched`); continue; }
  let b = 0;
  for (let h=-B;h<=B;h++) for (let k=-B;k<=B;k++) for (let l=-B;l<=B;l++) {
    if (!h&&!k&&!l) continue; nRef++;
    if (ctx.sgOpsAbsent(h,k,l,C) !== truthAbsent(h,k,l,raw._z)) b++;
  }
  if (b) { bad += b; console.log(`   ${st.symbol}: ${b} wrong`); }
  // zone membership from the shared table vs the setting's own normals
  for (const z of (st.zones||[])) {
    for (let h=-4;h<=4;h++) for (let k=-4;k<=4;k++) for (let l=-4;l<=4;l++) {
      let mine = true;
      for (const n of z.normals) if (n[0]*h+n[1]*k+n[2]*l !== 0) { mine = false; break; }
      if (ctx.zoneApplies(z.zone,h,k,l) !== mine) badZone++;
    }
  }
}
console.log(`   ${nRef.toLocaleString()} reflections, ${bad} wrong`);
console.log(`2. zoneApplies vs each setting's own normals: ${badZone} wrong\n`);

// --- 3. extinction classes -------------------------------------------------
console.log('3. sgExtinctionClasses');
// 'trigonal' is not probed: getSymmetry() never emits it (a trigonal cell in
// hexagonal axes has a=b, gamma=120, which it correctly calls hexagonal), and
// sgSystemMatches maps the database's trigonal groups into the hexagonal run.
for (const sys of ['triclinic','monoclinic','orthorhombic','tetragonal','hexagonal','cubic']) {
  const t0 = Date.now();
  const cs = ctx.sgExtinctionClasses(db, sys, null);
  const nMembers = cs.reduce((n,c)=>n+c.members.length,0);
  const unnamed = cs.filter(c => String(c.label||'').includes('?')).length;
  console.log(`   ${sys.padEnd(13)} ${String(cs.length).padStart(3)} classes, ` +
              `${String(nMembers).padStart(4)} settings, ${String(unnamed).padStart(2)} unnamed symbols, ` +
              `${Date.now()-t0} ms`);
  if (cs.length <= 8) console.log(`      ${cs.map(c=>`${c.label}[${c.members.length}]`).join(' ')}`);
}

// --- 4. countViolations ----------------------------------------------------
console.log('\n4. countViolations, operator-based');
const all = Object.values(db.space_groups).flatMap(g=>g.settings);
const pnma = all.find(s=>s.symbol==='Pnma') || all.find(s=>(s.zones||[]).length) || all[0];
const refl = [];
for (let h=0;h<=3;h++) for (let k=0;k<=3;k++) for (let l=0;l<=3;l++)
  if (h||k||l) refl.push({h,k,l,calc_tth:20+h+k+l});
const v = ctx.countViolations(refl, pnma);
console.log(`   ${pnma.symbol} vs ${refl.length} reflections: ${v.hardCount} hard, ${v.softCount} soft`);
console.log(`   e.g. ${v.detailsHard.slice(0,3).join(' | ')}`);
const named = v.detailsHard.filter(d => !d.includes('a systematic absence of this group')).length;
console.log(`   named by a printed condition: ${named}/${v.detailsHard.length}`);

// --- 5. Wilson weight ------------------------------------------------------
console.log('\n5. Wilson weight');
for (const [sys,h,k,l] of [['cubic',1,0,0],['cubic',1,2,3],['tetragonal',0,0,1],
                          ['orthorhombic',1,0,0],['monoclinic',0,1,0]]) {
  const m = ctx.sgMultiplicity(h,k,l,sys), w = ctx.sgHolohedryWeight(h,k,l,sys);
  console.log(`   ${sys.padEnd(13)} ${h}${k}${l}  m=${String(m).padStart(2)}  m*eps=${String(w).padStart(2)}  x${(w/m).toFixed(1)}`);
}

console.log('\n' + (bad === 0 && badZone === 0 && !gone.length
  ? 'PASS: operators match the generator, zones match their normals, old layer gone.'
  : 'FAIL'));
process.exit(bad === 0 && badZone === 0 && !gone.length ? 0 : 1);
