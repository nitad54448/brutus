// js/crystallography/sg-ops.js
// Space-group operator database: compilation, conditions and extinction classes.
//
// Part of the crystallography code (formerly worker-logic.js). The same files
// run on the main thread (brutus.html) and in both workers, which load them
// through manifest.js, so nothing here may touch the DOM.

// ============================================================================
// SPACE-GROUP MONTE-CARLO SCAN
// ----------------------------------------------------------------------------
// analyzeSystematicAbsences() ranks space groups by counting rule violations
// against ONE fixed cell and ONE fixed peak-to-line assignment. That is a
// bookkeeping test: it can only ever punish a group for an absence that did not
// happen, and it never lets the cell move. On real data the two weaknesses
// compound -- a cell refined against an extinction-blind line list is pulled
// toward forbidden reflections, which mislabels the peaks near them, which then
// manufactures the violations that condemn the correct group.
//
// This does the opposite. For each candidate rule set the line list is
// REGENERATED with the forbidden reflections removed, the cell is re-refined
// (Monte-Carlo + least squares) against that restricted list, and the figures of
// merit are recomputed. Every quantity in the row -- pairing, cell, zero, M20,
// F(N) -- then belongs to the same hypothesis, and the comparison between rows
// is a comparison between hypotheses rather than between bookkeeping artefacts.
//
// Two things make the ranking meaningful:
//
//   * M20 = Q20 / (2<|dQ|> N20) counts N20 = the number of POSSIBLE lines below
//     the 20th observed one. Removing extinct lines lowers N20, so the correct
//     rule set raises M20 for free. An over-restrictive rule set lowers N20 too,
//     but only by throwing away lines that are actually observed -- which shows
//     up as violations, below.
//   * A violation here is an observed peak that has NO allowed line within
//     tolerance but DOES have a forbidden one. That is the direct, physical
//     falsification of a rule set, and it is counted after the cell has been
//     given every chance to move away from it.
//
// Powder absences cannot distinguish space groups that forbid exactly the same
// reflections, so candidates are grouped into EXTINCTION CLASSES first (by what
// they actually forbid, not by how the condition happens to be written) and one
// refinement is run per class. Every group in the class shares the row. This is
// both honest -- it stops the table implying a discrimination the data cannot
// support -- and roughly 5-10x cheaper than one run per setting.
// ============================================================================

// Symmetry-equivalent reflections, used so a line is only treated as extinct
// when EVERY member of its orbit is forbidden.
//
// generateHKL_for_analysis emits one representative per equivalent set (h>=k>=l
// for cubic, and so on) and a reflection condition is not always invariant under
// the choice of representative. R-centring is the worked example: 101 satisfies
// -h+k+l=3n while its equivalent 011 does not, so testing the representative
// alone would delete a line that is present. Erring toward "allowed" is the safe
// direction: a false absence destroys M20, while a missed absence only leaves
// two classes tied.
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
            // 6-fold in-plane orbit (h,k) -> (-k,h+k) -> ... plus the mirror
            // (k,h), each with +-l. Covers 3-fold/R settings as well.
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
// ---------------------------------------------------------------------------
// Laue-class orbits
// ---------------------------------------------------------------------------
// sgEquivalents() returns the HOLOHEDRAL orbit of a crystal system, which is the
// right thing for undoing the generator's index folding but is NOT the symmetry
// of most space groups. Using it to decide absences over-constrains every group
// whose Laue class is smaller than the holohedral one.
//
// Pa-3 is the worked example. Its point group is m-3, which contains the cyclic
// permutations of h, k, l but NOT the transpositions. The database lists
// 0kl: k=2n, h0l: l=2n, hk0: h=2n. For 210 the hk0 rule gives h=2, fine. But the
// holohedral orbit also contains 120, and applying hk0: h=2n to THAT demanded
// k=2n as well -- so 210 and 320 were marked absent. Both are strong pyrite
// lines. Every m-3 group (Pa-3, Pn-3, Pm-3, Ia-3, Pn-3n ...) lost reflections
// this way.
//
// The Laue class is derivable from the point_group field the database already
// carries, so the orbit can be built correctly.
//
// -3m is mapped to -3, and that is safe rather than merely convenient. The
// point-group string does not record whether the two-folds run along <100>
// (-3m1) or <210> (-31m) -- only standard_symbol does, via the position of the
// "1" in "P 3 2 1" against "P 3 1 2". It turns out not to matter, and the check
// is cheap to state: -3 is contained in both -3m1 and -31m, which are in turn
// contained in 6/mmm, so if the two EXTREMES agree then everything between them
// agrees. For all 18 primitive trigonal groups the -3 and 6/mmm line lists are
// identical, which settles those by bracketing. The 7 R groups do differ between
// the extremes -- 6/mmm is wrong for them, since the six-fold is not an
// operation of the R lattice -- so those were compared against -3m1 directly,
// and are identical for all seven. Calcite predicts the same nine d-spacings
// either way. sgAuditLaueBracket() in the test suite re-checks this.
// The holohedral Laue class of each crystal system -- i.e. the symmetry the HKL
// generator's index folding assumes.
const SG_HOLOHEDRY = {
    cubic: 'm-3m', tetragonal: '4/mmm', hexagonal: '6/mmm',
    orthorhombic: 'mmm', monoclinic: '2/m', triclinic: '-1',
};
// ---------------------------------------------------------------------------
// Compiled reflection conditions
// ---------------------------------------------------------------------------
// satisfiesCondition() re-parses its condition string on every call: two regex
// matches, a split, a map, and a per-term regex. That is fine for the handful of
// calls the absence analysis makes, and ruinous here -- building the extinction
// classes asks it tens of millions of times. This compiles each distinct string
// ONCE into a closure over integer coefficients.
//
// The compiled form is VERIFIED against satisfiesCondition() on a small grid
// before it is trusted, and anything that disagrees (or fails to parse) falls
// back to the original function. The two can therefore never diverge, whatever
// a future database throws at them.
const _SG_COND_CACHE = new Map();
function sgCompileCondition(condStr) {
    const hit = _SG_COND_CACHE.get(condStr);
    if (hit !== undefined) return hit;

    let fn = null;
    try {
        if (condStr === 'h+k, k+l, h+l=2n') {
            fn = (h, k, l) => ((h + k) % 2 === 0) && ((k + l) % 2 === 0) && ((h + l) % 2 === 0);
        } else {
            const rhsMatch = condStr.match(/=\s*(\d+)n/);
            const defaultRhs = rhsMatch ? rhsMatch[0] : '=2n';
            const parts = [];
            for (let piece of condStr.split(',')) {
                let clean = piece.trim().replace(/\*/g, '');
                if (!clean.includes('=')) clean += defaultRhs;
                const m = clean.match(/([0-9]*[hkl+\-]+)\s*=\s*(\d+)n/);
                if (!m) { parts.length = 0; break; }
                const mod = parseInt(m[2], 10);
                if (!isFinite(mod) || mod <= 0) { parts.length = 0; break; }
                let ch = 0, ck = 0, cl = 0, bad = false;
                for (const term of (m[1].match(/[+-]?[0-9]*[hkl]/g) || [])) {
                    const t = term.match(/^([+-]?)(\d*)([hkl])$/);
                    if (!t) { bad = true; break; }
                    const v = (t[1] === '-' ? -1 : 1) * (t[2] ? parseInt(t[2], 10) : 1);
                    if (t[3] === 'h') ch += v; else if (t[3] === 'k') ck += v; else cl += v;
                }
                if (bad) { parts.length = 0; break; }
                parts.push([ch, ck, cl, mod]);
            }
            if (parts.length) {
                fn = (h, k, l) => {
                    for (let i = 0; i < parts.length; i++) {
                        const p = parts[i];
                        const v = p[0] * h + p[1] * k + p[2] * l;
                        if (((v % p[3]) + p[3]) % p[3] !== 0) return false;
                    }
                    return true;
                };
            }
        }
        // Trust nothing that does not reproduce the reference implementation.
        if (fn) {
            for (let h = -3; h <= 3 && fn; h++)
                for (let k = -3; k <= 3 && fn; k++)
                    for (let l = -3; l <= 3; l++) {
                        if (fn(h, k, l) !== !!satisfiesCondition(h, k, l, condStr)) { fn = null; break; }
                    }
        }
    } catch (e) { fn = null; }

    const out = fn || ((h, k, l) => !!satisfiesCondition(h, k, l, condStr));
    _SG_COND_CACHE.set(condStr, out);
    return out;
}
// ============================================================================
// EXTINCTION-CLASS CONSTRUCTION
// ============================================================================
//
// Two levels of grouping are used, and the difference between them matters.
//
//   ABSTRACT class   - groups settings that forbid the same reflections as a
//                      matter of arithmetic, over a small hkl box. Cheap and
//                      cell-independent; used only to avoid enumerating the
//                      same rule set five hundred times.
//   OBSERVABLE class - groups abstract classes that produce the SAME calculated
//                      pattern for THIS cell, at THIS wavelength, over THIS
//                      2-theta range, at THIS tolerance. Two rule sets that
//                      differ only at reflections beyond q_max, or only at
//                      lines that coincide with an allowed line inside the
//                      matching window, are not distinguishable by the
//                      experiment and must not appear as separate rows with
//                      separate figures of merit. That WAS the old behaviour,
//                      and it manufactured exactly the discrimination this
//                      module exists to refuse.
//
// The observable merge runs in sgObservableMerge(), once the parent cell is
// known.
// ============================================================================

// 32-bit FNV-1a. The abstract signature used to be a ~5000-character string
// used directly as a Map key, once per setting; hashing with bucket
// verification keeps the grouping exact at a fraction of the memory.
function sgHash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h >>> 0;
}
// Abstract fingerprint: what does this rule set forbid, arithmetically?
//
// The box has to be wide enough to separate every modulus that occurs: 4n
// (d-glides) and 6n (6_1 screws) need indices that reach the residues those
// conditions reject, and the negative half is needed because conditions like
// -h+k+l = 3n are not symmetric in sign. Range 6 covers all of them with room to
// spare. It is only a PRE-grouping in any case -- the observable merge below
// decides what actually shares a row -- so the cost of widening it further is
// not worth paying.
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
//
// `gen` enumerates the zone with its DEGENERATE SUB-ZONES REMOVED -- 0kl runs
// over k != 0 and l != 0, not over the whole h = 0 plane. A reflection like 00l
// belongs to the 0kl zone AND to the h0l zone, so it carries both conditions;
// including it in the 0kl probe means no single 0kl candidate can ever reproduce
// the pattern and the probe returns '?' for perfectly ordinary groups. (Pbca did
// exactly that.) The axial zones are probed separately, which is where those
// reflections belong.
//
// `tests` are ordered SIMPLEST-FIRST and the first exact match wins. Two
// candidates can both reproduce the pattern once the centering has thinned the
// probe set, and the convention is to name the simpler operation: on a C-centred
// lattice a c-glide and an n-glide perpendicular to b are indistinguishable
// (h is already even, so h+l even means l even), and International Tables writes
// C-c-, not C-n-. Nesting is not a problem here because a stronger condition
// never matches the weaker candidate exactly -- if the truth is k+l = 4n then
// (0,1,1) is absent while "k+l = 2n" predicts it present, so the 2n candidate is
// rejected outright.
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
        // |h| == |k| is excluded as well as the axes: (h,h,0) belongs to the hhl
        // zone too, so in a tetragonal or cubic group it carries the hhl
        // condition and no hk0 candidate can reproduce the mixture. I4_1md came
        // out as "I?-d" for exactly that reason.
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
        // axis zones report the MODULUS; the caller names the screw according to
        // the rotation order of the direction (see SG_SYMBOL_DIRECTIONS.names)
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
// One entry per symmetry direction:
//   glide    - the zone whose condition names a glide plane perpendicular to it
//   axis     - the axial zone whose condition names a screw along it
//   names    - modulus -> screw symbol, because the SAME condition means
//              different operations on different axes. 00l: l = 2n is a 2_1 along
//              b in monoclinic, a 4_2 along c in tetragonal and cubic, and a 6_3
//              in hexagonal. Naming them all "2_1" would be wrong, and naming
//              them by modulus alone loses the rotation order.
//   inZones  - the zones that CONTAIN this axis. A screw is only reported when
//              none of them carries a glide, because International Tables omits
//              an axial extinction that already follows by restriction from a
//              zonal one. Pnma is the standard illustration: 0kl: k+l=2n and
//              hk0: h=2n between them force 0k0: k=2n, so the symbol is Pn-a,
//              not Pn2_1a. Note 00l lies inside BOTH hhl and h-hl (h = k = 0
//              satisfies |h| = |k| and h = -k), which is how a c-glide in a
//              trigonal group accounts for its own 00l: l = 2n.
//
// When a direction carries a glide AND a screw that is NOT implied by a
// neighbouring zone, both are reported as "screw/glide" -- P2_1/c, P4_2/n. That
// case was previously collapsed to the glide alone, which put P2_1/c and P2/c on
// one label ("Pc") and P4_2/n and P4/n on another ("Pn--"): four distinct
// hypotheses shown under two names.
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
// One letter for one zone, given that the centering already accounts for part
// of the absences. '-' when the centering explains everything, a letter when
// exactly one candidate reproduces the residual, '?' when none does.
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
        if (exact) return letter;         // simplest-first, so the first hit is the name
    }
    return '?';
}
// Falls back to "<centering>?" for a system with no direction table, which keeps
// the label honest instead of inventing a symbol we did not derive.
function sgExtinctionSymbol(allowed, system, centering, centPredIn) {
    // centPredIn comes from the setting's own centring operators (identity
    // rotation, non-zero translation), so R-obverse and every other centring
    // are exact rather than looked up from a per-letter table. The 'P' default
    // keeps the probe honest if a caller has no setting to hand.
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
        // The screw is reported only when it is not already implied by a glide in
        // one of the zones containing this axis.
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
// Every distinct ABSTRACT extinction class of a crystal system.
// `allowedCenterings` is optional; pass the list from determineCentering() to
// scan only the lattices the absences already allow, or omit it to scan all.
// ============================================================================
// SPACE-GROUP CORE: OPERATORS AND ZONES
// ============================================================================
//
// Everything about systematic absences now comes from the symmetry OPERATORS,
// and everything about zone membership from the zone NORMALS. Both arrive in
// sg_ops.json (see sg_pack.py); neither is inferred from a rule string.
//
// ABSENCE. From
//     F(h) = exp(2*pi*i * h.t) * F(hR)      for every operator (R, t)
// it follows that if hR = h then F(h) = exp(2*pi*i h.t) F(h), so F(h) must
// vanish unless h.t is an integer. That is the definition, and it is complete:
// it needs no zone table, no inheritance rule and no centering proxy, because
// the centring translations are themselves operators.
//
// ZONES. A zone is the set of reflections killed by n.h = 0 for each of its
// normals. The old ZONE_PREDICATES table guessed these from the label and got
// three families wrong: 'hhl' was Math.abs(h) === Math.abs(k), which also
// matches h === -k -- the separate 'h-hl' zone, with different conditions in
// trigonal and hexagonal groups. Same for 'hkk'/'hll'/'hkh'. Refereed against
// the generator's own zone records, the string path was wrong on 180
// reflections of P6_3/mmc and 48 of R-3c inside |h|,|k|,|l| <= 5; the operator
// path was wrong on none.
//
// Normals also make membership INCLUSIVE for free, which the old table needed a
// long argument to justify: h00 satisfies the hk0 normal (l = 0), so h00 is in
// hk0 automatically, and a condition stated only on the general zone still
// reaches the special reflections it governs.
//
// INDEX CONVENTION. cctbx writes x' = R x + t with R row-major, so reflection
// indices transform as the ROW vector h' = h R:
//     h'_c = h*R[0*3+c] + k*R[1*3+c] + l*R[2*3+c]
// matching _zone_fixed_by() in the generator, which contracts b[i] with
// R[3*i + c]. Translations are exact rationals t = t_num / t_den.

// ---------------------------------------------------------------------------
// Database registration
// ---------------------------------------------------------------------------
// The zone table is global to a database, so it is installed once at load
// rather than threaded through every call. Both the main thread and each worker
// call this after fetching, because they each hold their own copy of this file.
let SG_ZONE_NORMALS = null;       // label -> [[n0,n1,n2], ...]
let SG_ROTATIONS = null;          // shared rotation table
let _SG_INSTALLED_DB = null;      // identity of the database currently installed
function sgInstallDatabase(db) {
    SG_ROTATIONS = (db && db.rotations) || null;
    SG_ZONE_NORMALS = (db && db.zone_defs) || null;
    _SG_INSTALLED_DB = db || null;
    _SG_ZONE_PRED_CACHE.clear();
    // _SG_OPS_CACHE is keyed on the setting objects themselves, so a different
    // database brings different objects and cannot collide with stale entries.
    return !!(SG_ROTATIONS && SG_ZONE_NORMALS);
}
// Idempotent, and cheap when nothing changed. Called at the top of every entry
// point that receives the database, so neither the main thread nor any worker
// has to remember to install it -- the database is structured-cloned into each
// worker and every context holds its own copy of this file.
function sgEnsureDatabase(db) {
    if (db && db !== _SG_INSTALLED_DB) sgInstallDatabase(db);
    return !!SG_ROTATIONS;
}
const _SG_ZONE_PRED_CACHE = new Map();
// Does a rule labelled `zoneLabel` apply to this reflection?
//
// Exact when the database supplies normals for the label. An unknown label
// falls back to an exact-match on the reported zone name, which can only ever
// under-apply a rule -- never make one match everything.
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
            pred = () => true;                       // no normals == the whole of hkl
        } else {
            pred = null;
        }
        _SG_ZONE_PRED_CACHE.set(zoneLabel, pred);
    }
    if (pred) return pred(H, K, L);
    return getReflectionZone(H, K, L) === zoneLabel;
}
// ---------------------------------------------------------------------------
// Compiling a setting's operators
// ---------------------------------------------------------------------------
// Two lists come out, used for different things:
//   R/Tn     operators with a NON-ZERO translation. Only these can extinguish
//            anything, so the absence test never looks at pure rotations. For a
//            symmorphic group this holds just the centring vectors.
//   pgR      the point group: rotation parts with duplicates removed. Needed
//            for epsilon and centricity, never for absence.
//   cenTn    the centring translations: operators whose rotation is the
//            identity. These give the lattice predicate that SG_CENTERING_PRED
//            used to hard-code per letter.
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
// h is systematically absent iff some operator fixes h and shifts its phase.
function sgOpsAbsent(h, k, l, C) {
    const R = C.R, Tn = C.Tn, den = C.den, n = C.nAbs;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        // hR == h ?  One component at a time; most operators fail on the first.
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== l) continue;
        // h.t integral ?  Exact integer arithmetic: h.t = num / den.
        const t = i * 3;
        const num = h * Tn[t] + k * Tn[t + 1] + l * Tn[t + 2];
        // JS % keeps the sign of the dividend and -0 === 0, so this is a
        // correct divisibility test for negative indices as written.
        if (num % den !== 0) return true;
    }
    return false;
}
// Is this reflection allowed by the LATTICE alone? Derived from the centring
// operators, so R-obverse and every other centring come out right without a
// per-letter table.
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
// The powder question: is there a peak at this d-spacing?
//
// A powder line collects every reflection sharing its d, which the METRIC
// decides, not the space group -- so the line survives if ANY member of the
// metric orbit does. The old sgAllowedFn needed "AND inside a Laue orbit, OR
// across orbits" plus a folding walk to work out which members a condition
// governed. Absence is constant on a point-group orbit -- |F(hR)| = |F(h)| --
// so every member of one answers identically and the AND collapses to a plain
// OR over the metric orbit. The Fd-3m and Pa-3 cases the old comment describes
// both come out right without any of that machinery.
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
// The label question: what does this group forbid, per reflection rather than
// per powder shell? With operators it is the absence test itself. The Laue-orbit
// restriction the old sgLabelPredicate needed is unnecessary: asking about a
// reflection already asks about its whole orbit.
function sgOpsLabelPredicate(setting) {
    const C = sgOpsCompile(setting);
    if (!C) return null;
    if (C.nAbs === 0) return () => true;
    return (h, k, l) => !sgOpsAbsent(h, k, l, C);
}
// epsilon(h): the order of the stabiliser of h in the point group.
//     <|F(h)|^2> = epsilon(h) * sum_j f_j^2
// Reflections on a symmetry axis or plane are expected epsilon times STRONGER
// than a general reflection at the same resolution.
function sgOpsEpsilon(h, k, l, C) {
    const R = C.pgR, n = C.nPg;
    let eps = 0;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== l) continue;
        eps++;
    }
    return eps || 1;
}
// h is centric iff some operation sends it to -h, which happens in
// non-centrosymmetric groups too: centricity belongs to the reflection, not the
// group. Deliberately unused in the class score -- see SG_USE_EPSILON_WEIGHT.
function sgOpsIsCentric(h, k, l, C) {
    const R = C.pgR, n = C.nPg;
    for (let i = 0; i < n; i++) {
        const b = i * 9;
        if (h * R[b] + k * R[b + 3] + l * R[b + 6] !== -h) continue;
        if (h * R[b + 1] + k * R[b + 4] + l * R[b + 7] !== -k) continue;
        if (h * R[b + 2] + k * R[b + 5] + l * R[b + 8] !== -l) continue;
        return true;
    }
    return false;
}
// A stable key for deduplicating settings that carry identical symmetry.
function sgOpsKey(setting) {
    const ops = (setting && setting.ops) || [];
    let s = (setting.t_den || 1) + '#';
    for (let i = 0; i < ops.length; i++) s += ops[i].join('.') + ';';
    return s;
}
// Every printed condition of a setting, flattened to {zone, cond} pairs. This
// is the presentation layer: what the tables print, for display and for the
// condition-by-condition evidence hunt in detectExtinctions. Absences never
// come from here.
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
    const buckets = new Map();   // hash -> [cls, ...], verified on the full string

    // Settings that carry literally the same conditions must fingerprint the
    // same, so probe once per DISTINCT (conditions, centering) pair rather than
    // once per setting. The bundled database restates the same rule set across
    // many settings of the same group, and the fingerprint is by far the most
    // expensive thing here.
    const probed = new Map();

    for (const sg of groups) {
        for (const setting of (sg.settings || [])) {
            if (!sgSettingAxesMatch(setting, system)) continue;
            if (!settingCenteringAllowed(setting.symbol, allowedCenterings)) continue;
            const C = sgOpsCompile(setting);
            if (!C) continue;                       // no operators, nothing to say
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
            // hall travels with the member because it is the ONLY field that
            // names one setting unambiguously. Several IT numbers cover more
            // than one setting -- 62 is Pnma, Pmnb, Pbnm, Pcmn, Pmcn and Pnam --
            // and they impose different absences on the same cell. A consumer
            // handed only the number and an H-M string has to re-parse the
            // string to work out which; handed the Hall symbol it does not.
            cls.members.push({ number: sg.number, symbol: setting.symbol,
                               hall: setting.hall || null,
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
        // Label with the extinction symbol, NOT with a representative group.
        try { c.label = sgExtinctionSymbol(c.labelFn || c.allowed, system, c.centering, c.centPred); }
        catch (e) { c.label = c.centering + '?'; }
        c.sigStr = null;   // release the fingerprint; only the key is needed now
    }
    return out;
}
