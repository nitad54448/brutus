// Does sgOpsAbsent reproduce the International Tables reflection conditions?
//
// Groups are built by CLOSING a set of published generator triplets, not by
// typing out the full operator list -- a typo in a generator almost always
// changes the group order, which is checked against the published order_z, so
// the fixture largely validates itself. The triplet parser produces exactly the
// (r, t_num, t_den) form the generator's get_symmetry_operations() emits.

// Loads the REWRITTEN worker-logic.js and tests the operator core inside it,
// so this checks the shipping code rather than a copy of it.
import { readFileSync } from 'fs';
import vm from 'vm';

const ctx = { console, self:{}, performance:{now:()=>Date.now()}, postMessage(){},
              addEventListener(){}, setTimeout, clearTimeout, Date, Math, JSON,
              Uint8Array, Int32Array, Float64Array, Float32Array, Map, Set, WeakMap,
              Object, Array, Number, String, isFinite, parseInt, parseFloat };
ctx.globalThis = ctx;
vm.createContext(ctx);
try { vm.runInContext(readFileSync('./worker-logic.js','utf8'), ctx, {filename:'w'}); } catch (e) {}
for (const n of ['sgOpsCompile','sgOpsAbsent','sgOpsEpsilon','sgOpsIsCentric'])
    if (typeof ctx[n] !== 'function') { console.error('missing '+n+' in worker-logic.js'); process.exit(1); }

// worker-logic.js reads operators in packed form against a shared rotation
// table, so the fixtures are packed the same way the database is.
const ROT = []; const ROT_IX = new Map();
function rotIndex(r) {
    const key = r.join(',');
    if (!ROT_IX.has(key)) { ROT_IX.set(key, ROT.length); ROT.push(r.slice()); }
    return ROT_IX.get(key);
}
function packSetting(ops, den) {
    return { t_den: den, ops: ops.map(o => [rotIndex(o.r), o.t_num[0], o.t_num[1], o.t_num[2]]) };
}
const SG = {
    sgOpsCompile: (setting) => { ctx.sgInstallDatabase({ rotations: ROT, zone_defs: {} });
                                 return ctx.sgOpsCompile(setting); },
    sgOpsAbsent: (h,k,l,C) => ctx.sgOpsAbsent(h,k,l,C),
    sgOpsEpsilon: (h,k,l,C) => ctx.sgOpsEpsilon(h,k,l,C),
    sgOpsIsCentric: (h,k,l,C) => ctx.sgOpsIsCentric(h,k,l,C),
    // Top-level `const` in a classic script is NOT a property of the global
    // object, so the holohedry table is reached through the function that uses
    // it rather than read directly.
    holohedryWeight: (h,k,l,sys) => ctx.sgHolohedryWeight(h,k,l,sys),
    multiplicity: (h,k,l,sys) => ctx.sgMultiplicity(h,k,l,sys),
};

const DEN = 24;   // 24 covers the d-glide quarters as well as the usual twelfths

// "-x,y+1/2,-z+1/4" -> { r:[9 ints], t_num:[3], t_den:24 }
function parseTriplet(s) {
    const r = new Array(9).fill(0), tn = [0, 0, 0];
    const parts = s.replace(/\s+/g, '').toLowerCase().split(',');
    if (parts.length !== 3) throw new Error('bad triplet: ' + s);
    parts.forEach((p, row) => {
        // split into signed terms
        const terms = p.match(/[+-]?[^+-]+/g) || [];
        for (const t of terms) {
            const m = /^([+-]?)(?:(\d+)\/(\d+)|([xyz]))$/.exec(t);
            if (!m) throw new Error('bad term "' + t + '" in ' + s);
            const sign = m[1] === '-' ? -1 : 1;
            if (m[4]) {
                const col = { x: 0, y: 1, z: 2 }[m[4]];
                r[row * 3 + col] = sign;          // r[row][col], row-major
            } else {
                tn[row] += sign * DEN * parseInt(m[2], 10) / parseInt(m[3], 10);
            }
        }
    });
    if (tn.some(v => !Number.isInteger(v))) throw new Error('translation not on /24: ' + s);
    return { r, t_num: tn.map(v => ((v % DEN) + DEN) % DEN), t_den: DEN };
}

function opKey(o) { return o.r.join(',') + '|' + o.t_num.join(','); }

function compose(a, b) {
    // apply b then a:  x -> a(b(x)) = Ra Rb x + Ra tb + ta
    const r = new Array(9).fill(0), tn = [0, 0, 0];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += a.r[i * 3 + k] * b.r[k * 3 + j];
        r[i * 3 + j] = s;
    }
    for (let i = 0; i < 3; i++) {
        let s = a.t_num[i];
        for (let k = 0; k < 3; k++) s += a.r[i * 3 + k] * b.t_num[k];
        tn[i] = ((s % DEN) + DEN) % DEN;
    }
    return { r, t_num: tn, t_den: DEN };
}

function closeGroup(gens) {
    const id = parseTriplet('x,y,z');
    const seen = new Map([[opKey(id), id]]);
    let frontier = [id];
    while (frontier.length) {
        const next = [];
        for (const a of frontier) for (const g of gens) {
            const c = compose(a, g);
            const k = opKey(c);
            if (!seen.has(k)) { seen.set(k, c); next.push(c); }
        }
        frontier = next;
        if (seen.size > 400) throw new Error('group did not close');
    }
    return [...seen.values()];
}

function mk(name, order, gens) {
    const ops = closeGroup(gens.map(parseTriplet));
    if (ops.length !== order) throw new Error(`${name}: closed to ${ops.length}, expected ${order}`);
    return { name, ops, setting: packSetting(ops, DEN) };
}

// --- fixtures: generators and published order_z ----------------------------
const T = [1 / 2, 1 / 2, 0];
const GROUPS = [
    mk('P2_1', 2, ['-x,y+1/2,-z']),
    mk('Pc', 2, ['x,-y,z+1/2']),
    mk('P2_1/c', 4, ['-x,y+1/2,-z+1/2', '-x,-y,-z']),
    mk('C2/c', 8, ['-x,y+1/2,-z+1/2', '-x,-y,-z', 'x+1/2,y+1/2,z']),
    mk('P2_12_12_1', 4, ['-x+1/2,-y,z+1/2', '-x,y+1/2,-z+1/2']),
    mk('Pbca', 8, ['-x+1/2,-y,z+1/2', '-x,y+1/2,-z+1/2', '-x,-y,-z']),
    mk('Pnma', 8, ['-x+1/2,-y,z+1/2', '-x,y+1/2,-z', '-x,-y,-z']),
    mk('Fddd', 32, ['-x,-y,z', '-x,y,-z', '-x+1/4,-y+1/4,-z+1/4',
                    'x,y+1/2,z+1/2', 'x+1/2,y,z+1/2']),
    mk('I4_1', 8, ['-y,x+1/2,z+1/4', 'x+1/2,y+1/2,z+1/2']),
    mk('Pa-3', 24, ['-x+1/2,-y,z+1/2', '-x,y+1/2,-z+1/2', 'z,x,y', '-x,-y,-z']),
];

// Built and order-checked, but their conditions are DERIVED AND PRINTED rather
// than asserted. The whole point of the trigonal/hexagonal cases is the zone
// spelling, and asserting a table entry from memory in a test meant to catch
// mistakes would just move the mistake into the fixture. Compare the printed
// output against ITA by eye.
const REPORT_ONLY = [
    mk('P6_3/mmc', 24, ['x-y,x,z+1/2', 'y,x,-z', '-x,-y,-z']),
    mk('R-3c', 36, ['-y,x-y,z', 'y,x,-z+1/2', '-x,-y,-z', 'x+2/3,y+1/3,z+1/3']),
];

// --- published conditions, as predicates on hkl ---------------------------
// Each entry: given (h,k,l), is the reflection PRESENT according to the tables?
// null means "the tables state no condition applying to this reflection".
// Each entry is a list of [zone, condition]. A reflection is present iff EVERY
// applicable condition holds -- conditions conjoin, they do not short-circuit.
// Getting this wrong is easy: h00 lies in BOTH the h0l zone and the hk0 zone,
// so a first-match chain silently drops one of the two rules governing it.
const ITA = {
    'P2_1': [[(h,k,l) => h===0&&l===0, (h,k,l) => k%2===0]],
    'Pc':   [[(h,k,l) => k===0,        (h,k,l) => l%2===0]],
    'P2_1/c': [
        [(h,k,l) => k===0,            (h,k,l) => l%2===0],   // h0l: l=2n
        [(h,k,l) => h===0&&l===0,     (h,k,l) => k%2===0],   // 0k0: k=2n
    ],
    'C2/c': [
        [() => true,                  (h,k,l) => (h+k)%2===0], // C centring
        [(h,k,l) => k===0,            (h,k,l) => l%2===0],     // h0l: l=2n
        [(h,k,l) => h===0&&l===0,     (h,k,l) => k%2===0],     // 0k0: k=2n
    ],
    'P2_12_12_1': [
        [(h,k,l) => k===0&&l===0,     (h,k,l) => h%2===0],
        [(h,k,l) => h===0&&l===0,     (h,k,l) => k%2===0],
        [(h,k,l) => h===0&&k===0,     (h,k,l) => l%2===0],
    ],
    'Pbca': [
        [(h,k,l) => h===0,            (h,k,l) => k%2===0],     // 0kl: k=2n
        [(h,k,l) => k===0,            (h,k,l) => l%2===0],     // h0l: l=2n
        [(h,k,l) => l===0,            (h,k,l) => h%2===0],     // hk0: h=2n
    ],
    'Pnma': [
        [(h,k,l) => h===0,            (h,k,l) => (k+l)%2===0], // 0kl: k+l=2n
        [(h,k,l) => l===0,            (h,k,l) => h%2===0],     // hk0: h=2n
    ],
    'Fddd': [
        [() => true,                  (h,k,l) => (h+k)%2===0 && (k+l)%2===0], // F
        [(h,k,l) => h===0,            (h,k,l) => mod(k+l,4)===0],  // 0kl: k+l=4n
        [(h,k,l) => k===0,            (h,k,l) => mod(h+l,4)===0],  // h0l: h+l=4n
        [(h,k,l) => l===0,            (h,k,l) => mod(h+k,4)===0],  // hk0: h+k=4n
    ],
    'I4_1': [
        [() => true,                  (h,k,l) => (h+k+l)%2===0],   // I
        [(h,k,l) => h===0&&k===0,     (h,k,l) => mod(l,4)===0],    // 00l: l=4n
    ],
    'Pa-3': [
        [(h,k,l) => h===0,            (h,k,l) => k%2===0],     // 0kl: k=2n
        [(h,k,l) => k===0,            (h,k,l) => l%2===0],     // h0l: l=2n
        [(h,k,l) => l===0,            (h,k,l) => h%2===0],     // hk0: h=2n
    ],
};
function itaPresent(rules, h, k, l) {
    let governed = false;
    for (const [zone, cond] of rules) {
        if (!zone(h, k, l)) continue;
        governed = true;
        if (!cond(h, k, l)) return false;
    }
    return governed ? true : null;      // null = tables say nothing here
}
function mod(a, n) { return ((a % n) + n) % n; }

// --- run -------------------------------------------------------------------
const B = 8;
let failures = 0;
console.log('Deriving reflection conditions from operators and checking them');
console.log('against the International Tables.\n');
console.log('  group        order  tested   absent  mismatches');

for (const { name, setting } of GROUPS) {
    const C = SG.sgOpsCompile(setting);
    const ref = ITA[name];
    let tested = 0, absent = 0, bad = 0;
    const examples = [];
    for (let h = -B; h <= B; h++) for (let k = -B; k <= B; k++) for (let l = -B; l <= B; l++) {
        if (!h && !k && !l) continue;
        const got = !SG.sgOpsAbsent(h, k, l, C);       // present?
        if (!got) absent++;
        const want = itaPresent(ref, h, k, l);
        if (want === null) continue;                    // tables silent here
        tested++;
        if (got !== want) { bad++; if (examples.length < 5) examples.push(`${h}${k}${l} ops=${got} ITA=${want}`); }
    }
    failures += bad;
    console.log(`  ${name.padEnd(12)} ${String(C.orderZ).padStart(4)}  ${String(tested).padStart(6)}  ` +
                `${String(absent).padStart(6)}  ${bad === 0 ? 'none' : bad + '  ' + examples.join('; ')}`);
}

// --- derive the conditions back out, in table form -------------------------
// For each standard zone, find the largest modulus m and residue such that every
// present reflection in that zone satisfies (linear form) = 0 mod m. This is a
// readable summary, not part of the pass/fail logic.
const ZONES = [
    ['hkl',   (h, k, l) => true],
    ['0kl',   (h, k, l) => h === 0],
    ['h0l',   (h, k, l) => k === 0],
    ['hk0',   (h, k, l) => l === 0],
    ['hhl',   (h, k, l) => h === k],
    ['h-hl',  (h, k, l) => h === -k],
    ['h00',   (h, k, l) => k === 0 && l === 0],
    ['0k0',   (h, k, l) => h === 0 && l === 0],
    ['00l',   (h, k, l) => h === 0 && k === 0],
];
const FORMS = [
    ['h', (h, k, l) => h], ['k', (h, k, l) => k], ['l', (h, k, l) => l],
    ['h+k', (h, k, l) => h + k], ['k+l', (h, k, l) => k + l], ['h+l', (h, k, l) => h + l],
    ['h+k+l', (h, k, l) => h + k + l], ['-h+k+l', (h, k, l) => -h + k + l],
    ['2h+l', (h, k, l) => 2 * h + l],
];
function describe(C, R) {
    const out = [];
    for (const [zname, zpred] of ZONES) {
        const pts = [];
        for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) {
            if (!h && !k && !l) continue;
            if (zpred(h, k, l)) pts.push([h, k, l]);
        }
        const present = pts.filter(([h, k, l]) => !SG.sgOpsAbsent(h, k, l, C));
        if (!present.length || present.length === pts.length) continue;
        const found = [];
        for (const [fname, f] of FORMS) {
            for (const m of [2, 3, 4, 6, 8]) {
                if (present.every(([h, k, l]) => mod(f(h, k, l), m) === 0) &&
                    !pts.every(([h, k, l]) => mod(f(h, k, l), m) === 0)) {
                    found.push(`${fname}=${m}n`); break;
                }
            }
        }
        if (found.length) out.push(`${zname}: ${found.join(', ')}`);
    }
    return out;
}

console.log('\nConditions derived from the operators, for comparison with the tables.');
console.log('(zones are listed independently, so implied conditions appear too)\n');
for (const { name, setting } of [...GROUPS, ...REPORT_ONLY]) {
    const C = SG.sgOpsCompile(setting);
    console.log(`  ${name}  (order ${C.orderZ})`);
    for (const line of describe(C, 6)) console.log(`      ${line}`);
}

console.log('\nZone confusion: ZONE_PREDICATES tests |h|==|k| for "hhl", so it also');
console.log('matches h==-k. Are hhl and h-hl actually different zones?\n');
for (const { name, setting } of REPORT_ONLY) {
    const C = SG.sgOpsCompile(setting);
    const row = [];
    for (const [h, k, l] of [[1, 1, 1], [1, -1, 1], [2, 2, 3], [2, -2, 3]]) {
        row.push(`${h},${k},${l}=${SG.sgOpsAbsent(h, k, l, C) ? 'ABS' : 'ok '}`);
    }
    console.log(`  ${name.padEnd(10)} ${row.join('  ')}`);
}

// --- epsilon and the multiplicity identity --------------------------------
function sgEquivalents(h, k, l, system) {
    const out = [];
    const push = (a, b, c) => out.push([a, b, c]);
    switch (system) {
        case 'cubic': {
            const perms = [[h, k, l], [h, l, k], [k, h, l], [k, l, h], [l, h, k], [l, k, h]];
            for (const [x, y, z] of perms)
                for (const sx of [1, -1]) for (const sy of [1, -1]) for (const sz of [1, -1]) push(sx * x, sy * y, sz * z);
            break;
        }
        case 'tetragonal':
            for (const [x, y] of [[h, k], [k, h]])
                for (const sx of [1, -1]) for (const sy of [1, -1]) for (const sz of [1, -1]) push(sx * x, sy * y, sz * l);
            break;
        case 'hexagonal': {
            let a = h, b = k;
            for (let r = 0; r < 6; r++) {
                for (const sz of [1, -1]) { push(a, b, sz * l); push(b, a, sz * l); }
                const na = -b, nb = a + b; a = na; b = nb;
            }
            break;
        }
        case 'orthorhombic':
            for (const sx of [1, -1]) for (const sy of [1, -1]) for (const sz of [1, -1]) push(sx * h, sy * k, sz * l);
            break;
        case 'monoclinic':
            push(h, k, l); push(-h, k, -l); push(h, -k, l); push(-h, -k, -l); break;
        default:
            push(h, k, l); push(-h, -k, -l); break;
    }
    return out;
}

console.log('\nWilson weight: is m * epsilon really constant over the holohedry?');
console.log('(if so, weighting a powder line by multiplicity alone is wrong by');
console.log(' exactly epsilon, and worst on the axial lines)\n');
console.log('  system         hkl        m    eps_holo   m*eps');
const cases = [
    ['cubic', 1, 0, 0], ['cubic', 1, 1, 0], ['cubic', 1, 1, 1], ['cubic', 1, 2, 3],
    ['tetragonal', 0, 0, 1], ['tetragonal', 1, 1, 0], ['tetragonal', 1, 2, 3],
    ['orthorhombic', 1, 0, 0], ['orthorhombic', 1, 2, 3],
    ['monoclinic', 0, 1, 0], ['monoclinic', 1, 2, 3],
];
let weightConst = true;
const first = {};
for (const [sys, h, k, l] of cases) {
    const m = SG.multiplicity(h, k, l, sys);
    const w = SG.holohedryWeight(h, k, l, sys);
    const eps = w / m;
    if (w !== first[sys] && first[sys] !== undefined) weightConst = false;
    if (first[sys] === undefined) first[sys] = w;
    console.log(`  ${sys.padEnd(13)} ${h}${k}${l}   ${String(m).padStart(4)}   ${String(eps).padStart(6)}   ${String(w).padStart(5)}`);
}

// epsilon from the real point group, for a group whose Laue class is lower
console.log('\nEpsilon from the actual point group (Pa-3, Laue m-3, not m-3m):');
const pa3 = SG.sgOpsCompile(GROUPS.find(g => g.name === 'Pa-3').setting);
for (const [h, k, l] of [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0], [1, 2, 3]]) {
    console.log(`  ${h}${k}${l}  order_p=${pa3.orderP}  eps=${SG.sgOpsEpsilon(h, k, l, pa3)}  ` +
                `centric=${SG.sgOpsIsCentric(h, k, l, pa3)}`);
}

console.log('\n' + (failures === 0 && weightConst
    ? 'PASS: every derived condition matches the tables, and m*eps is constant.'
    : `FAIL: ${failures} condition mismatches, weightConst=${weightConst}`));
process.exit(failures === 0 && weightConst ? 0 : 1);
