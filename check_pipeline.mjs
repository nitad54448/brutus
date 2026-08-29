// Does what main_app.js packs match what the engine and shaders expect?
// Extracts the real buildHklBasis from main_app.js rather than a copy.
import { readFileSync } from 'fs';

// Normalise line endings before any pattern matching. The shipped files are
// CRLF, and patterns written with \n silently fail to match against \r\n --
// which made this report a false FAIL on every field it extracts. A checker
// that cries wolf gets ignored, which is worse than not having it.
const read = (f) => readFileSync(f, 'utf8').replace(/\r\n/g, '\n');
const app = read('./main_app.js');
const eng = read('./webgpu-engine.js');

// pull HKL_PACKERS + HKL_PACKING straight out of the app source
const i0 = app.indexOf("const HKL_PACKING = 'products/v1';");
const i1 = app.indexOf('};', app.indexOf('triclinic: { floats: 8')) + 2;
if (i0 < 0 || i1 < 2) { console.log('FAIL: could not locate HKL_PACKERS in main_app.js'); process.exit(1); }
const { HKL_PACKERS, HKL_PACKING } = new Function(app.slice(i0, i1) + '\nreturn {HKL_PACKERS, HKL_PACKING};')();

// engine's declared stride per system
const strides = {};
for (const m of eng.matchAll(/^\s{8}(\w+): \{\n\s+K: \d+,\n\s+structFloats: \d+,[^\n]*\n\s+hklFloats: (\d+),/gm))
  strides[m[1]] = parseInt(m[2],10);

console.log('system         app floats   engine hklFloats   match');
let ok = true;
for (const sys of Object.keys(HKL_PACKERS)) {
  const a = HKL_PACKERS[sys].floats, e = strides[sys];
  const good = a === e; ok = ok && good;
  console.log(`  ${sys.padEnd(13)} ${String(a).padStart(6)} ${String(e).padStart(16)}   ${good ? 'yes' : 'NO'}`);
}

// what the shaders index
const shaderExpect = {
  orthorhombic: ['ortho_solver.wgsl',  ['hkl_basis[j].xyz', 'hkl_basis[hkl_indices[0]].xyz']],
  monoclinic:   ['monoclinic_solver.wgsl', ['dot(abcd, hkl_basis[j])', 'hkl_basis[hkl_indices[0]]']],
  triclinic:    ['triclinic_solver.wgsl',  ['hkl_basis[j * 2u]', 'hkl_basis[hkl_indices[i] * 2u]']],
};
console.log('\nshader indexing');
for (const [sys, [file, needles]] of Object.entries(shaderExpect)) {
  let src = '';
  try { src = read('./'+file); } catch { console.log(`  ${file}: not present here`); continue; }
  const isVec4 = /hkl_basis: array<vec4<f32>>/.test(src);
  const found = needles.filter(n => src.includes(n));
  const good = isVec4 && found.length === needles.length;
  ok = ok && good;
  console.log(`  ${sys.padEnd(13)} vec4 binding=${isVec4}  indexing ${found.length}/${needles.length}  ${good ? 'yes' : 'NO'}`);
}

// packing tag agreement
const guard = eng.includes("baseParams.hklPacking !== 'products/v1'");
console.log(`\npacking tag: app emits '${HKL_PACKING}', engine guards: ${guard}`);
ok = ok && guard && HKL_PACKING === 'products/v1';

// spot-check the actual packed values
console.log('\npacked values for (2,3,4)');
for (const sys of Object.keys(HKL_PACKERS)) {
  const p = HKL_PACKERS[sys];
  const buf = new Float32Array(p.floats);
  p.pack(buf, 0, 2, 3, 4);
  console.log(`  ${sys.padEnd(13)} [${Array.from(buf).join(', ')}]`);
}
const t = new Float32Array(8); HKL_PACKERS.triclinic.pack(t,0,2,3,4);
const want = [4,9,16,12,8,6,0,0];
const triOK = want.every((v,i)=>t[i]===v);
console.log(`  triclinic == (h2,k2,l2,kl,hl,hk,0,0): ${triOK}`);
ok = ok && triOK;

// --- argument order at the engine call site --------------------------------
// A long positional list is easy to shift by one, and a shift is invisible:
// everything is an array or a callback, so nothing type-checks and the run just
// produces nothing. Compare the names main_app passes against the names the
// engine declares.
console.log('\nengine call site');
const sigM = /async _runSolver\(cfg,\s*([^)]*)\)\s*\{/s.exec(eng);
const callM = /await engineFn\.call\(\s*engine,\s*([\s\S]*?)\n\s*\);/.exec(app);
if (!sigM || !callM) { console.log('  could not locate signature or call site'); ok = false; }
else {
  const params = sigM[1].split(',').map(x => x.trim().split('=')[0].trim()).filter(Boolean);
  const args = callM[1].split('\n')
      .map(l => l.replace(/\/\/.*$/,'').trim().replace(/,$/,''))
      .filter(l => l && !l.startsWith('//'));
  console.log(`  engine declares ${params.length}: ${params.join(', ')}`);
  console.log(`  app passes      ${args.length}: ${args.join(', ')}`);
  if (params.length !== args.length) {
    console.log(`  !! ARITY MISMATCH -- arguments are shifted`);
    ok = false;
  } else {
    // positional names should broadly correspond; flag any that clearly do not
    const suspicious = params.map((p,i) => [p, args[i]])
      .filter(([p,a]) => a !== 'null' && !a.startsWith('{') &&
                         !a.toLowerCase().includes(p.toLowerCase().replace(/array$/,'')) &&
                         !p.toLowerCase().includes(a.toLowerCase().replace(/array$/,'')));
    if (suspicious.length) {
      console.log('  positions worth checking by eye:');
      for (const [p,a] of suspicious) console.log(`     ${p}  <-  ${a}`);
    } else {
      console.log('  arity and names line up');
    }
  }
}

console.log('\n' + (ok ? 'PASS: app, engine and shaders agree.' : 'FAIL'));
process.exit(ok ? 0 : 1);
