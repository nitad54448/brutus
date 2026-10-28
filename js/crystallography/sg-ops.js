// js/crystallography/sg-ops.js
// Space-group operator database: compilation, conditions and extinction classes.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (index.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// ============================================================================
// SPACE-GROUP MONTE-CARLO SCAN
// ============================================================================

// Symmetry-equivalent reflections, used so a line is only treated as extinct
// when EVERY member of its orbit is forbidden.
function sgEquivalents(h, k, l, system) {
    const out = [];
    const push = (a, b, c) => { out.push([a, b, c]); };
    switch (system) {
        case 'cubic': {
            const perms = [[h,k,l],[h,l,k],[k,h,l],[k,l,h],[l,h,k],[l,k,h]];
            for (const [x, y, z] of perms)
                for (const sx of [1,-1]) for (const sy of [1,-1]) for (const sz of [1,-1])
                    push(sx*x, sy*y, sz*z);
            break;
        }
        case 'tetragonal':
            for (const [x, y] of [[h,k],[k,h]])
                for (const sx of [1,-1]) for (const sy of [1,-1]) for (const sz of [1,-1])
                    push(sx*x, sy*y, sz*l);
            break;
        case 'hexagonal': {
            let a = h, b = k;
            for (let r = 0; r < 6; r++) {
                for (const sz of [1,-1]) { push(a, b, sz*l); push(b, a, sz*l); }
                const na = -b, nb = a + b; a = na; b = nb;
            }
            break;
        }
        case 'orthorhombic':
            for (const sx of [1,-1]) for (const sy of [1,-1]) for (const sz of [1,-1])
                push(sx*h, sy*k, sz*l);
            break;
        case 'monoclinic':   // b unique: 2/m
            push(h, k, l); push(-h, k, -l); push(h, -k, l); push(-h, -k, -l);
            break;
        default:             // triclinic: Friedel pair only
            push(h, k, l); push(-h, -k, -l);
            break;
    }
    return out;
}
// ============================================================================
// EXTINCTION-CLASS CONSTRUCTION
// ============================================================================

// 32-bit FNV-1a.
function sgHash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h >>> 0;
}
// Abstract fingerprint: what does this rule set forbid, arithmetically?
const SG_SIG_RANGE = 6;
function sgBehaviourSignatureString(allowed) {
    const bits = [];
    for (let h = 0; h <= SG_SIG_RANGE; h++)
        for (let k = -SG_SIG_RANGE; k <= SG_SIG_RANGE; k++)
            for (let l = -SG_SIG_RANGE; l <= SG_SIG_RANGE; l++) {
                if (h === 0 && k === 0 && l === 0) continue;
                bits.push(allowed(h, k, l) ? 1 : 0);
            }
    return bits.join('');
}
const _sgMod = (x, n) => ((x % n) + n) % n;
// Zone probes.
const SG_ZONE_PROBES = {
    '0kl': {
        gen: function* (R) { for (let k = -R; k <= R; k++) for (let l = -R; l <= R; l++) if (k && l) yield [0, k, l]; },
        tests: [['b', (h, k, l) => _sgMod(k, 2) === 0], ['c', (h, k, l) => _sgMod(l, 2) === 0],
                ['n', (h, k, l) => _sgMod(k + l, 2) === 0], ['d', (h, k, l) => _sgMod(k + l, 4) === 0]],
    },
    'h0l': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let l = -R; l <= R; l++) if (h && l) yield [h, 0, l]; },
        tests: [['a', (h, k, l) => _sgMod(h, 2) === 0], ['c', (h, k, l) => _sgMod(l, 2) === 0],
                ['n', (h, k, l) => _sgMod(h + l, 2) === 0], ['d', (h, k, l) => _sgMod(h + l, 4) === 0]],
    },
    'hk0': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let k = -R; k <= R; k++) if (h && k && Math.abs(h) !== Math.abs(k)) yield [h, k, 0]; },
        tests: [['a', (h, k, l) => _sgMod(h, 2) === 0], ['b', (h, k, l) => _sgMod(k, 2) === 0],
                ['n', (h, k, l) => _sgMod(h + k, 2) === 0], ['d', (h, k, l) => _sgMod(h + k, 4) === 0]],
    },
    'hhl': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let l = -R; l <= R; l++) if (h && l) yield [h, h, l]; },
        tests: [['c', (h, k, l) => _sgMod(l, 2) === 0], ['b', (h, k, l) => _sgMod(h, 2) === 0],
                ['n', (h, k, l) => _sgMod(2 * h + l, 2) === 0], ['d', (h, k, l) => _sgMod(2 * h + l, 4) === 0]],
    },
    'h-hl': {
        gen: function* (R) { for (let h = -R; h <= R; h++) for (let l = -R; l <= R; l++) if (h && l) yield [h, -h, l]; },
        tests: [['c', (h, k, l) => _sgMod(l, 2) === 0]],
    },
    '00l': {
        gen: function* (R) { for (let l = -R; l <= R; l++) if (l) yield [0, 0, l]; },
        tests: [[2, (h, k, l) => _sgMod(l, 2) === 0], [3, (h, k, l) => _sgMod(l, 3) === 0],
                [4, (h, k, l) => _sgMod(l, 4) === 0], [6, (h, k, l) => _sgMod(l, 6) === 0]],
    },
    '0k0': {
        gen: function* (R) { for (let k = -R; k <= R; k++) if (k) yield [0, k, 0]; },
        tests: [[2, (h, k, l) => _sgMod(k, 2) === 0], [4, (h, k, l) => _sgMod(k, 4) === 0]],
    },
    'h00': {
        gen: function* (R) { for (let h = -R; h <= R; h++) if (h) yield [h, 0, 0]; },
        tests: [[2, (h, k, l) => _sgMod(h, 2) === 0], [4, (h, k, l) => _sgMod(h, 4) === 0]],
    },
};
const SG_SYMBOL_DIRECTIONS = {
    orthorhombic: [{ glide: '0kl', axis: 'h00', names: { 2: '2\u2081' }, inZones: ['hk0', 'h0l'] },
                   { glide: 'h0l', axis: '0k0', names: { 2: '2\u2081' }, inZones: ['hk0', '0kl'] },
                   { glide: 'hk0', axis: '00l', names: { 2: '2\u2081' }, inZones: ['h0l', '0kl'] }],
    monoclinic:   [{ glide: 'h0l', axis: '0k0', names: { 2: '2\u2081' }, inZones: [] }],
    tetragonal:   [{ glide: 'hk0', axis: '00l', names: { 2: '4\u2082', 4: '4\u2081' }, inZones: ['h0l', '0kl'] },
                   { glide: 'h0l', axis: 'h00', names: { 2: '2\u2081' }, inZones: ['hk0', 'h0l'] },
                   { glide: 'hhl', axis: null,  names: {}, inZones: [] }],
    cubic:        [{ glide: 'hk0', axis: '00l', names: { 2: '4\u2082', 4: '4\u2081' }, inZones: ['h0l', '0kl'] },
                   { glide: 'hhl', axis: null,  names: {}, inZones: [] }],
    hexagonal:    [{ glide: null,  axis: '00l', names: { 2: '6\u2083', 3: '3\u2081', 6: '6\u2081' },
                     inZones: ['h-hl', 'hhl'] },
                   { glide: 'h-hl', axis: null, names: {}, inZones: [] },
                   { glide: 'hhl', axis: null,  names: {}, inZones: [] }],
    triclinic:    [],
};
function sgProbeZone(allowed, centPred, zoneKey, R) {
    const probe = SG_ZONE_PROBES[zoneKey];
    if (!probe) return '?';
    const pts = [];
    for (const p of probe.gen(R)) if (centPred(p[0], p[1], p[2])) pts.push(p);
    if (!pts.length) return '-';

    let anyForbidden = false;
    for (const [h, k, l] of pts) if (!allowed(h, k, l)) { anyForbidden = true; break; }
    if (!anyForbidden) return '-';

    for (const [letter, pred] of probe.tests) {
        let exact = true;
        for (const [h, k, l] of pts) {
            if (allowed(h, k, l) !== pred(h, k, l)) { exact = false; break; }
        }
        if (exact) return letter;
    }
    return '?';
}
function sgExtinctionSymbol(allowed, system, centering, centPredIn) {
    const cent = String(centering || 'P').charAt(0) || 'P';
    const centPred = centPredIn || (() => true);
    const dirs = SG_SYMBOL_DIRECTIONS[system];
    if (!dirs) return cent + '?';
    if (!dirs.length) return cent;

    const R = 8;
    const zoneCache = new Map();
    const zone = (key) => {
        if (!key) return '-';
        if (!zoneCache.has(key)) zoneCache.set(key, sgProbeZone(allowed, centPred, key, R));
        return zoneCache.get(key);
    };

    const parts = [];
    for (const d of dirs) {
        if (!d) { parts.push('-'); continue; }
        const g = zone(d.glide);
        const implied = (d.inZones || []).some(z => zone(z) !== '-');
        let sName = '-';
        if (!implied && d.axis) {
            const mod = zone(d.axis);
            if (mod !== '-' && mod !== '?') sName = (d.names && d.names[mod]) || ('?' + mod);
            else sName = mod;
        }
        if (g !== '-' && g !== '?' && sName !== '-' && sName !== '?') parts.push(sName + '/' + g);
        else if (g !== '-') parts.push(g);
        else parts.push(sName);
    }
    return cent + parts.join('');
}
// ============================================================================
// SPACE-GROUP CORE: OPERATORS AND ZONES
// ============================================================================

let SG_ZONE_NORMALS = null;       // label -> [[n0,n1,n2], ...]
let SG_ROTATIONS = null;          // shared rotation table
let _SG_INSTALLED_DB = null;      // identity of the database currently installed
function sgInstallDatabase(db) {
    SG_ROTATIONS = (db && db.rotations) || null;
    SG_ZONE_NORMALS = (db && db.zone_defs) || null;
    _SG_INSTALLED_DB = db || null;
    _SG_ZONE_PRED_CACHE.clear();
    return !!(SG_ROTATIONS && SG_ZONE_NORMALS);
}
function sgEnsureDatabase(db) {
    if (db && db !== _SG_INSTALLED_DB) sgInstallDatabase(db);
    return !!SG_ROTATIONS;
}
const _SG_ZONE_PRED_CACHE = new Map();
function zoneApplies(zoneLabel, h, k, l) {
    const H = Math.round(h), K = Math.round(k), L = Math.round(l);
    let pred = _SG_ZONE_PRED_CACHE.get(zoneLabel);
    if (pred === undefined) {
        const normals = SG_ZONE_NORMALS ? SG_ZONE_NORMALS[zoneLabel] : null;
        if (normals && normals.length) {
            const N = normals.map(v => [v[0] | 0, v[1] | 0, v[2] | 0]);
            pred = (a, b, c) => {
                for (let i = 0; i < N.length; i++) {
                    const n = N[i];
                    if (n[0] * a + n[1] * b + n[2] * c !== 0) return false;
                }
                return true;
            };
        } else if (normals) {
            pred = () => true;
        } else {
            pred = null;
        }
        _SG_ZONE_PRED_CACHE.set(zoneLabel, pred);
    }
    if (pred) return pred(H, K, L);
    return getReflectionZone(H, K, L) === zoneLabel;
}
const _SG_OPS_CACHE = new WeakMap();
function sgOpsCompile(setting) {
    if (!setting) return null;
    const cached = _SG_OPS_CACHE.get(setting);
    if (cached !== undefined) return cached;

    const packed = setting.ops;
    const rotations = SG_ROTATIONS;
    if (!packed || !packed.length || !rotations) { _SG_OPS_CACHE.set(setting, null); return null; }
    const den = setting.t_den || 1;

    const absIdx = [], pgRot = [], cen = [];
    const seenRot = new Set();
    for (let i = 0; i < packed.length; i++) {
        const op = packed[i];
        const r = rotations[op[0]];
        if (!r || r.length !== 9) { _SG_OPS_CACHE.set(setting, null); return null; }
        if (!seenRot.has(op[0])) { seenRot.add(op[0]); pgRot.push(r); }
        const isIdentity = r[0] === 1 && r[4] === 1 && r[8] === 1 &&
                           !r[1] && !r[2] && !r[3] && !r[5] && !r[6] && !r[7];
        if (isIdentity) cen.push([op[1], op[2], op[3]]);
        if (op[1] || op[2] || op[3]) absIdx.push(i);
    }

    const nA = absIdx.length;
    const R = new Int32Array(nA * 9);
    const Tn = new Int32Array(nA * 3);
    for (let a = 0; a < nA; a++) {
        const op = packed[absIdx[a]];
        const r = rotations[op[0]];
        for (let j = 0; j < 9; j++) R[a * 9 + j] = r[j];
        Tn[a * 3] = op[1]; Tn[a * 3 + 1] = op[2]; Tn[a * 3 + 2] = op[3];
    }
    const nP = pgRot.length;
    const P = new Int32Array(nP * 9);
    for (let i = 0; i < nP; i++) for (let j = 0; j < 9; j++) P[i * 9 + j] = pgRot[i][j];

    const nC = cen.length;
    const Cn = new Int32Array(nC * 3);
    for (let i = 0; i < nC; i++) { Cn[i * 3] = cen[i][0]; Cn[i * 3 + 1] = cen[i][1]; Cn[i * 3 + 2] = cen[i][2]; }

    const out = { nAbs: nA, R, Tn, den, nPg: nP, pgR: P, nCen: nC, cenTn: Cn,
                  orderZ: packed.length, orderP: nP };
    _SG_OPS_CACHE.set(setting, out);
    return out;
}
function sgOpsAbsent(h, k, l, C) {
    const R = C.R, Tn = C.Tn, den = C.den, n = C.nAbs;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== l) continue;
        const t = i * 3;
        const num = h * Tn[t] + k * Tn[t + 1] + l * Tn[t + 2];
        if (num % den !== 0) return true;
    }
    return false;
}
function sgOpsCenteringPred(C) {
    if (!C || C.nCen === 0) return () => true;
    const Cn = C.cenTn, den = C.den, n = C.nCen;
    return (h, k, l) => {
        for (let i = 0; i < n; i++) {
            const t = i * 3;
            if ((h * Cn[t] + k * Cn[t + 1] + l * Cn[t + 2]) % den !== 0) return false;
        }
        return true;
    };
}
function sgOpsAllowedFn(setting, system) {
    const C = sgOpsCompile(setting);
    if (!C) return null;
    if (C.nAbs === 0) return () => true;
    const cache = new Map();
    return (h, k, l) => {
        const key = h + ',' + k + ',' + l;
        const hit = cache.get(key);
        if (hit !== undefined) return hit;
        const orbit = sgEquivalents(h, k, l, system);
        let ok = false;
        for (let i = 0; i < orbit.length; i++) {
            const m = orbit[i];
            if (!sgOpsAbsent(m[0], m[1], m[2], C)) { ok = true; break; }
        }
        cache.set(key, ok);
        return ok;
    };
}
function sgOpsLabelPredicate(setting) {
    const C = sgOpsCompile(setting);
    if (!C) return null;
    if (C.nAbs === 0) return () => true;
    return (h, k, l) => !sgOpsAbsent(h, k, l, C);
}
function sgOpsKey(setting) {
    const ops = (setting && setting.ops) || [];
    let s = (setting.t_den || 1) + '#';
    for (let i = 0; i < ops.length; i++) s += ops[i].join('.') + ';';
    return s;
}
function sgSettingConditions(setting) {
    const out = [];
    const conds = (setting && setting.conditions) || {};
    for (const zone of Object.keys(conds)) {
        const list = conds[zone] || [];
        for (let i = 0; i < list.length; i++) out.push({ zone, cond: list[i] });
    }
    return out;
}
function sgExtinctionClasses(spaceGroupData, system, allowedCenterings) {
    if (!sgEnsureDatabase(spaceGroupData)) return [];
    const groups = Object.values(spaceGroupData?.space_groups || {})
        .filter(sg => sgSystemMatches(sg.crystal_system, system));
    const buckets = new Map();

    const probed = new Map();

    for (const sg of groups) {
        for (const setting of (sg.settings || [])) {
            if (!sgSettingAxesMatch(setting, system)) continue;
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) continue;
            const C = sgOpsCompile(setting);
            if (!C) continue;
            const cent = String(setting.centering || setting.symbol || '').charAt(0);
            const ruleKey = sgOpsKey(setting);

            let probe = probed.get(ruleKey);
            if (!probe) {
                const allowed = sgOpsAllowedFn(setting, system);
                probe = { allowed, sigStr: sgBehaviourSignatureString(allowed),
                          labelFn: sgOpsLabelPredicate(setting),
                          centPred: sgOpsCenteringPred(C) };
                probed.set(ruleKey, probe);
            }
            const allowed = probe.allowed;
            const sigStr = probe.sigStr;
            const labelFn = probe.labelFn;
            const centPred = probe.centPred;
            const rules = setting.conditions || {};
            const key = sgHash(sigStr);
            let bucket = buckets.get(key);
            if (!bucket) { bucket = []; buckets.set(key, bucket); }
            let cls = bucket.find(c => c.sigStr === sigStr);
            if (!cls) {
                cls = {
                    sig: key + ':' + bucket.length, sigStr, rules, allowed, labelFn,
                    centPred,
                    label: setting.symbol,
                    centering: cent,
                    members: [],
                    conditions: Object.entries(rules)
                        .map(([z, c]) => `${z}: ${(c || []).join(', ')}`)
                        .sort()
                };
                bucket.push(cls);
            }
            cls.members.push({ number: sg.number, symbol: setting.symbol,
                               hall: setting.hall || null,
                               setting_id: setting.setting_id || null,
                               centric: !!sg.centrosymmetric });
        }
    }

    const out = [];
    for (const bucket of buckets.values()) for (const c of bucket) out.push(c);
    for (const c of out) {
        c.members.sort((a, b) => (a.number - b.number) || a.symbol.localeCompare(b.symbol));
        c.nRules = c.conditions.length;
        c.centric = c.members.length > 0 && c.members.every(m => m.centric);
        c.repSymbol = c.members.length ? c.members[0].symbol : c.label;
        try { c.label = sgExtinctionSymbol(c.labelFn || c.allowed, system, c.centering, c.centPred); }
        catch (e) { c.label = c.centering + '?'; }
        c.sigStr = null;
    }
    return out;
}
