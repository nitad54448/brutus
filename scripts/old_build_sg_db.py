#!/usr/bin/env python3
"""Build Brutus's space-group database directly from cctbx. One script, one
point of failure.

WHY THIS REPLACES A TWO-STAGE PIPELINE
The database used to be produced by running SpaceExplorer to write 527
per-setting JSON files, then running a second script to merge the handful of
fields Brutus needs. That is two schemas and two programs between cctbx and the
app: a field renamed in SpaceExplorer breaks the merger silently, and the 527
files exist only to be immediately re-merged. This script goes straight from
cctbx to the file the app loads.

WHAT IT PRODUCES
Exactly the fields worker-logic.js and main_app.js read, and nothing else:

    rotations           table of distinct 3x3 rotation matrices, row-major
    zone_defs           zone label -> list of normal vectors
    space_groups        keyed by space-group number
      .number .standard_symbol .crystal_system .point_group
      .centrosymmetric .chiral
      .settings[]
        .symbol .hall .centering .t_den .ops .conditions
        .setting_id .setting_number .hm .description .order_z .order_p

`ops` is the authoritative part: each entry is [rot_index, t0, t1, t2] with
translation t = t_num / t_den. A reflection is systematically absent iff some
operator has hR = h with h.t not an integer. `conditions` and `zone_defs` are
presentation -- they let the app name a zone and say which printed rule a
violation breaks -- and they are DERIVED FROM the operators here, then verified
against them, so the two cannot disagree.

INDEX CONVENTION
cctbx writes x' = R x + t with r() row-major, so reflection indices transform as
the ROW vector h' = h R, i.e. h'_c = sum_i h_i R[3i + c]. Everything below and
everything in the shaders uses that convention. Do not "fix" it.

HOW THE ZONES AND CONDITIONS ARE DERIVED
For each operator (R, t):
  * its zone is {h : hR = h}, the integer left null space of (R - I). That is a
    sublattice, saturated because it is cut out by integer linear equations, so
    a set of normal vectors characterises it exactly -- which is what the app
    needs for membership tests.
  * on that zone, writing h = sum u_i b_i over a basis B, the phase condition
    h.t in Z becomes sum u_i (b_i . t) in Z. With t = t_num / D this is
    sum u_i p_i = 0 (mod D) where p_i = b_i . t_num, an exact integer
    congruence. Dividing through by g = gcd(p_1..p_r, D) gives the reduced rule.
No Smith normal form and no floating point anywhere.

Usage:
    python3 build_sg_db.py                     # -> sg_ops.json
    python3 build_sg_db.py --pretty --box 6
    python3 build_sg_db.py --self-test         # check the algebra, no cctbx
    python3 build_sg_db.py --limit 20          # first 20 settings, for a smoke test
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from fractions import Fraction
from math import gcd

LETTERS = ('h', 'k', 'l')


# ===========================================================================
# Integer linear algebra (no numpy, no sympy -- a cctbx environment may have
# neither, and this is small enough not to want a dependency)
# ===========================================================================

def _rref(rows):
    """Reduced row echelon form over the rationals. Returns (rref, pivot_cols)."""
    m = [[Fraction(v) for v in r] for r in rows]
    nrows = len(m)
    ncols = len(m[0]) if nrows else 0
    pivots = []
    r = 0
    for c in range(ncols):
        piv = None
        for i in range(r, nrows):
            if m[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        lead = m[r][c]
        m[r] = [v / lead for v in m[r]]
        for i in range(nrows):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == nrows:
            break
    return m, pivots


def _primitive(vec):
    """Scale a rational vector to the shortest integer vector with the same
    direction. Returns a tuple of ints."""
    den = 1
    for v in vec:
        den = den * v.denominator // gcd(den, v.denominator)
    ints = [int(v * den) for v in vec]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    if g > 1:
        ints = [v // g for v in ints]
    return tuple(ints)


def int_nullspace(rows, ncols=3, want_free=False):
    """Basis for { x in Z^ncols : A x = 0 }, as primitive integer vectors.

    The free-variable construction is used deliberately, for two reasons: each
    basis vector carries a +1 in its own free column, which is what makes the
    zone labelling below well defined, AND it means the parameters of any zone
    member can be READ OFF rather than solved for -- h = sum u_i b_i with
    b_i[f_j] = delta_ij gives u_i = h[f_i] directly. Solving a 3x3 system per
    reflection instead would multiply the run by the size of the hkl box."""
    if not rows:
        basis = [tuple(1 if i == j else 0 for i in range(ncols)) for j in range(ncols)]
        return (basis, list(range(ncols))) if want_free else basis
    m, pivots = _rref(rows)
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for f in free:
        vec = [Fraction(0)] * ncols
        vec[f] = Fraction(1)
        for ri, pc in enumerate(pivots):
            vec[pc] = -m[ri][f]
        basis.append(_primitive(vec))
    # _primitive may have scaled a vector, so the +1 is not guaranteed to have
    # survived; recover the actual coefficient at the free column.
    return (basis, free) if want_free else basis


# ===========================================================================
# Zones
# ===========================================================================

def zone_of_operator(r9, want_free=False):
    """{h : hR = h} as a basis of primitive integer row vectors.

    h R = h means sum_i h_i R[3i+c] = h_c for every c, i.e. h (R - I) = 0.
    As a column system that is (R - I)^T h^T = 0, so the rows fed to the null
    space routine are the COLUMNS of (R - I)."""
    rows = []
    for c in range(3):
        rows.append([r9[3 * i + c] - (1 if i == c else 0) for i in range(3)])
    return int_nullspace(rows, want_free=want_free)


def normals_of_zone(basis):
    """Vectors n with n.b = 0 for every basis vector b. Membership in the zone
    is then exactly `n.h == 0 for all n`, which is what zoneApplies() does."""
    if len(basis) >= 3:
        return []                      # the whole of reciprocal space
    return [list(v) for v in int_nullspace([list(b) for b in basis])]


def _normalise_basis_vector(vec):
    """Orient a basis vector so the first +/-1 it contains is +1.

    This is what makes the label come out in International Tables form. The
    letter naming a parameter is the index where its coefficient is +1: for
    (1,1,0) that is index 0, giving 'hhl'; for (-2,1,0) it is index 1, giving
    '(-2k)kl'; for (0,-1,1) negation gives (0,1,-1) and hence 'hk-k' rather than
    the equivalent but non-standard 'h-ll'."""
    for v in vec:
        if v == 1:
            return tuple(vec)
        if v == -1:
            return tuple(-x for x in vec)
    for v in vec:                      # no +/-1 at all: make the leading term positive
        if v != 0:
            return tuple(vec) if v > 0 else tuple(-x for x in vec)
    return tuple(vec)


def _letter_index(vec):
    for i, v in enumerate(vec):
        if v == 1:
            return i
    for i, v in enumerate(vec):
        if v != 0:
            return i
    return 0


def canonical_zone(basis, free):
    """Orient and order the basis, assign a letter to each vector, and record
    how to read that vector's parameter straight off a reflection.

    Returns (basis, letters, readers) where readers[i] = (col, coeff): the
    parameter u_i of any zone member h is h[col] / coeff. Sorted so that h
    precedes k precedes l."""
    items = []
    for b, f in zip(basis, free):
        ob = _normalise_basis_vector(list(b))
        coeff = ob[f]                  # what the free column ended up holding
        items.append((_letter_index(ob), ob, f, coeff))
    items.sort(key=lambda t: (t[0], t[1]))
    used, out = set(), []
    for idx, b, f, coeff in items:
        if idx in used:                # collision: fall back to any free letter
            idx = next(i for i in range(3) if i not in used)
        used.add(idx)
        out.append((idx, b, f, coeff))
    out.sort(key=lambda t: t[0])
    return ([b for _, b, _, _ in out],
            [LETTERS[i] for i, _, _, _ in out],
            [(f, c) for _, _, f, c in out])


def _term(coeff, letter):
    if coeff == 1:
        return letter
    if coeff == -1:
        return '-' + letter
    return f'({coeff}{letter})'


def zone_label(basis, letters):
    """'hkl', '0kl', 'hhl', 'h-hl', 'h(-2h)l', 'hk-k' ..."""
    parts = []
    for comp in range(3):
        terms = [(b[comp], letters[i]) for i, b in enumerate(basis) if b[comp] != 0]
        if not terms:
            parts.append('0')
        elif len(terms) == 1:
            parts.append(_term(terms[0][0], terms[0][1]))
        else:
            s = ''
            for c, lt in terms:
                piece = _term(c, lt)
                s += ('+' + piece) if (s and not piece.startswith('-')) else piece
            parts.append('(' + s + ')')
    return ''.join(parts)


# ===========================================================================
# Conditions
# ===========================================================================

def _canonical_rule(coeffs, mod):
    """Reduce a congruence to a readable canonical form.

    Coefficients are taken into (-mod/2, mod/2], and the whole rule is negated
    when that reduces the number of negative terms -- which yields the
    conventional '-h+k+l=3n' for R centring rather than the equivalent
    'h-k-l=3n'."""
    c = []
    for v in coeffs:
        v %= mod
        if v > mod // 2:
            v -= mod
        c.append(v)
    neg = sum(1 for v in c if v < 0)
    flipped = [(-v) % mod for v in c]
    flipped = [v - mod if v > mod // 2 else v for v in flipped]
    if sum(1 for v in flipped if v < 0) < neg:
        c = flipped
    return tuple(c)


def _rule_string(coeffs, letters, mod):
    s = ''
    for c, lt in zip(coeffs, letters):
        if c == 0:
            continue
        if c == 1:
            piece = lt
        elif c == -1:
            piece = '-' + lt
        else:
            piece = f'{c}*{lt}'
        s += ('+' + piece) if (s and not piece.startswith('-')) else piece
    return f'{s}={mod}n' if s else None


def conditions_for_zone(basis, letters, ops, t_den):
    """Every congruence the operators impose on this zone, as rule strings.

    For h = sum u_i b_i, the phase condition h.t in Z is
        sum_i u_i (b_i . t_num) = 0  (mod t_den)
    which is exact integer arithmetic. Dividing by the gcd of the coefficients
    and the modulus gives the reduced rule; a modulus of 1 means no condition."""
    seen, rules = set(), []
    for r9, tnum in ops:
        # does this operator fix the whole zone?
        if not all(_applies(r9, b) for b in basis):
            continue
        p = [sum(b[c] * tnum[c] for c in range(3)) for b in basis]
        g = t_den
        for v in p:
            g = gcd(g, abs(v))
        if g == 0:
            continue
        mod = t_den // g
        if mod <= 1:
            continue
        coeffs = _canonical_rule([v // g for v in p], mod)
        key = (coeffs, mod)
        if key in seen:
            continue
        seen.add(key)
        s = _rule_string(coeffs, letters, mod)
        if s:
            rules.append((s, coeffs, mod))
    return rules


def _applies(r9, h):
    """hR == h ?"""
    return all(sum(h[i] * r9[3 * i + c] for i in range(3)) == h[c] for c in range(3))


def is_absent(h, ops, t_den):
    """The definition. Everything else in this file is presentation."""
    for r9, tnum in ops:
        if not _applies(r9, h):
            continue
        if sum(h[c] * tnum[c] for c in range(3)) % t_den != 0:
            return True
    return False


# ===========================================================================
# Per-setting assembly
# ===========================================================================

def _in_zone(normals, h):
    return all(n[0] * h[0] + n[1] * h[1] + n[2] * h[2] == 0 for n in normals)


def build_setting_zones(ops, t_den, box):
    """Derive every zone with a condition, minimise each zone's rule list, and
    mark which zones the International Tables would print.

    Returns (zone_records, zone_defs_fragment)."""
    # one zone per distinct operator fixed-set
    zones = {}
    for r9, tnum in ops:
        basis, free = zone_of_operator(r9, want_free=True)
        basis, letters, readers = canonical_zone(basis, free)
        label = zone_label(basis, letters)
        if label not in zones:
            zones[label] = (basis, letters, readers, normals_of_zone(basis))

    grid = [(h, k, l)
            for h in range(-box, box + 1)
            for k in range(-box, box + 1)
            for l in range(-box, box + 1)
            if (h, k, l) != (0, 0, 0)]
    absent = {h: is_absent(h, ops, t_den) for h in grid}

    records = []
    for label, (basis, letters, readers, normals) in zones.items():
        rules = conditions_for_zone(basis, letters, ops, t_den)
        if not rules:
            continue
        members = [h for h in grid if _in_zone(normals, h)]
        if not members:
            continue
        # Drop any rule the others already imply. A rule is redundant exactly
        # when removing it leaves the zone's present/absent verdict unchanged on
        # every member of the box.
        kept = list(rules)
        i = 0
        while i < len(kept):
            trial = kept[:i] + kept[i + 1:]
            same = all(_satisfies_all(h, trial, readers) ==
                       _satisfies_all(h, kept, readers) for h in members)
            if same:
                kept = trial
            else:
                i += 1

        # Prefer the conventional spelling. The reduction above can land on a
        # form that is correct but not the one the tables print: Fddd's
        # 0kl: k+l=4n comes out as -k+l=4n, which is equivalent ONLY because the
        # companion rule k=2n is also in force (-2 == 2 mod 4). Both are right;
        # one is recognisable. Try sign variants of each rule and keep any that
        # leaves the verdict identical, preferring the fewest negative terms.
        for i in range(len(kept)):
            _s, coeffs, mod = kept[i]
            best = kept[i]
            best_neg = sum(1 for c in coeffs if c < 0)
            for mask in range(1, 1 << len(coeffs)):
                trial_coeffs = tuple(-c if (mask >> j) & 1 else c
                                     for j, c in enumerate(coeffs))
                neg = sum(1 for c in trial_coeffs if c < 0)
                if neg >= best_neg:
                    continue
                cand = (_rule_string(trial_coeffs, letters, mod), trial_coeffs, mod)
                if cand[0] is None:
                    continue
                probe = kept[:i] + [cand] + kept[i + 1:]
                if all(_satisfies_all(h, probe, readers) ==
                       _satisfies_all(h, kept, readers) for h in members):
                    best, best_neg = cand, neg
            kept[i] = best
        kept.sort(key=lambda r: (r[2], r[0]))
        records.append({
            'zone': label,
            'normals': normals,
            'rules': [r[0] for r in kept],
            '_readers': readers,
            '_rules': kept,
            '_members': members,
        })

    # Precompute each zone's verdict on every grid point once. Everything below
    # -- implication testing, pruning, verification -- is a set operation over
    # these, and recomputing zone membership and parameters per reflection per
    # candidate subset is what made this quadratic.
    #   None  = reflection is not in this zone
    #   True  = in the zone and satisfies its rules
    #   False = in the zone and breaks one
    for rec in records:
        v = []
        for h in grid:
            if not _in_zone(rec['normals'], h):
                v.append(None)
            else:
                v.append(_satisfies_all(h, rec['_rules'], rec['_readers']))
        rec['_verdict'] = v

    # A zone is PRINTED unless a strictly more general zone already decides it.
    records.sort(key=lambda r: (-len(r['normals']), r['zone']))
    absent_list = [absent[h] for h in grid]

    def _set_ok(subset):
        """Does this subset of zones reproduce the operators on the whole grid?"""
        for gi in range(len(grid)):
            present = True
            for rec in subset:
                if rec['_verdict'][gi] is False:
                    present = False
                    break
            if present == absent_list[gi]:
                return False
        return True

    for rec in records:
        others = [o for o in records
                  if o is not rec and _strictly_contains(o['normals'], rec['normals'])]
        if not others:
            rec['printed'] = True
            continue
        implied = True
        for gi in range(len(grid)):
            if rec['_verdict'][gi] is None:
                continue
            present = True
            for o in others:
                if o['_verdict'][gi] is False:
                    present = False
                    break
            if present == absent_list[gi]:
                implied = False
                break
        rec['printed'] = not implied

    # The file emits ONLY the printed subset, so that subset has to be
    # sufficient on its own. Marking each zone against the FULL record set can
    # over-prune by cascade: A implied by B, B implied by C, both dropped, and
    # now nothing states A. Check the printed set as a set, and promote zones
    # back until it reproduces the operators exactly.
    for _ in range(len(records) + 1):
        printed = [r for r in records if r['printed']]
        if _set_ok(printed):
            break
        promoted = False
        for rec in records:            # most specific first: they constrain most
            if rec['printed']:
                continue
            rec['printed'] = True
            promoted = True
            if _set_ok([r for r in records if r['printed']]):
                break
        if not promoted:
            break                      # nothing left to promote; the caller reports it

    defs = {rec['zone']: rec['normals'] for rec in records}
    return records, defs, absent, grid


def _satisfies_all(h, rules, readers):
    """Does h satisfy every rule of this zone?"""
    u = _parameters(h, readers)
    if u is None:
        return None
    for _s, coeffs, mod in rules:
        if sum(c * v for c, v in zip(coeffs, u)) % mod != 0:
            return False
    return True


def _parameters(h, readers):
    """u_i for h = sum u_i b_i, read straight off the free columns. O(r), no
    linear solve. Returns None if a parameter is not integral, which means h is
    in the zone's rational span but not its lattice."""
    u = []
    for col, coeff in readers:
        if coeff == 0:
            return None
        v, rem = divmod(h[col], coeff)
        if rem:
            return None
        u.append(v)
    return u


def _strictly_contains(outer_normals, inner_normals):
    """Is the zone defined by outer_normals a strict superset of inner?"""
    if len(outer_normals) >= len(inner_normals):
        return False
    # every outer normal must be a rational combination of the inner ones
    if not inner_normals:
        return False
    for n in outer_normals:
        m, pivots = _rref([list(x) for x in inner_normals] + [list(n)])
        if len(pivots) > len(_rref([list(x) for x in inner_normals])[1]):
            return False
    return True


# ===========================================================================
# Verification -- the part that makes a single script safe
# ===========================================================================

def verify_setting(records, absent, grid):
    """The EMITTED conditions must reproduce the operators exactly.

    Only the printed subset is written to the file, so only the printed subset
    is checked -- verifying the full record set would pass while the emitted
    data was incomplete. This is the check that justifies shipping conditions at
    all: they are derived here, so a disagreement is a bug in THIS file and is
    caught before anything is written."""
    printed = [r for r in records if r['printed']]
    bad = []
    for gi, h in enumerate(grid):
        present = True
        for rec in printed:
            if rec['_verdict'][gi] is False:
                present = False
                break
        if present == absent[h]:
            bad.append(h)
    return bad


# ===========================================================================
# cctbx
# ===========================================================================

def load_cctbx():
    try:
        from cctbx import sgtbx
    except ImportError as exc:
        sys.exit("cctbx is not importable. Run this inside your cctbx "
                 f"environment (conda activate cctbx). Original error: {exc}")
    return sgtbx


def point_group_symbol(sgtbx, sg):
    """cctbx has no single obvious call for 'mmm'. Try the usual routes and
    degrade to an empty string rather than guessing -- the field is display
    only (rankSpaceGroups prints it, nothing branches on it)."""
    try:
        pg = sg.build_derived_point_group()
        sym = sgtbx.space_group_type(pg).lookup_symbol()
        # 'P m m m' -> 'mmm'
        parts = sym.split()
        if parts and len(parts[0]) == 1 and parts[0].isalpha() and parts[0].isupper():
            parts = parts[1:]
        return ''.join(parts)
    except Exception:
        return ''


def centring_from_ops(ops, t_den):
    """Derive the centring letter from the operators themselves.

    Fallback for sg.conventional_centring_type_symbol(). The centring vectors
    ARE the operators whose rotation is the identity, so this needs no cctbx
    call and cannot disagree with the operators the app will use. The letter is
    what settingCenteringAllowed() in worker-logic.js filters on, so a wrong one
    silently drops whole families of candidate settings.

    Counting lattice points per cell, origin included:
        1 -> P     2 -> A/B/C/I     3 -> R (hexagonal axes)     4 -> F
    """
    half, third, twothirds = t_den // 2, t_den // 3, 2 * t_den // 3
    vecs = {tuple(t) for r, t in ops
            if r == (1, 0, 0, 0, 1, 0, 0, 0, 1) and any(t)}

    if not vecs:
        return 'P'
    if len(vecs) == 1:
        v = next(iter(vecs))
        return {(half, half, half): 'I',
                (0, half, half): 'A',
                (half, 0, half): 'B',
                (half, half, 0): 'C'}.get(v, '?')
    if len(vecs) == 2:
        # obverse (2/3,1/3,1/3) or reverse (1/3,2/3,1/3); both are R
        if vecs == {(twothirds, third, third), (third, twothirds, twothirds)}:
            return 'R'
        if vecs == {(third, twothirds, third), (twothirds, third, twothirds)}:
            return 'R'
        return 'R' if all(third and v[0] % third == 0 for v in vecs) else '?'
    if len(vecs) == 3:
        if vecs == {(0, half, half), (half, 0, half), (half, half, 0)}:
            return 'F'
        return '?'
    return '?'


def extract_ops(sg):
    """Operators as (r9 tuple, t_num tuple scaled to a common denominator)."""
    den = 1
    raw = []
    for i in range(sg.order_z()):
        op = sg(i)
        r = tuple(int(v) for v in op.r().num())
        tn = tuple(int(v) for v in op.t().num())
        td = int(op.t().den())
        if td <= 0:
            raise ValueError('non-positive translation denominator')
        raw.append((r, tn, td))
        den = den * td // gcd(den, td)
    ops = []
    for r, tn, td in raw:
        scale = den // td
        ops.append((r, tuple((v * scale) % den for v in tn)))
    return ops, den


# ===========================================================================
# Self-test (no cctbx) -- checks the algebra against known table entries
# ===========================================================================

SELF_TEST = {
    # generators as xyz triplets, expected order, and a few ITA conditions
    'P2_1':  (['-x,y+1/2,-z'], 2, {'0k0': 'k=2n'}),
    'Pc':    (['x,-y,z+1/2'], 2, {'h0l': 'l=2n'}),
    'P2_1/c': (['-x,y+1/2,-z+1/2', '-x,-y,-z'], 4, {'h0l': 'l=2n', '0k0': 'k=2n'}),
    'Pbca':  (['-x+1/2,-y,z+1/2', '-x,y+1/2,-z+1/2', '-x,-y,-z'], 8,
              {'0kl': 'k=2n', 'h0l': 'l=2n', 'hk0': 'h=2n'}),
    'Fddd':  (['-x,-y,z', '-x,y,-z', '-x+1/4,-y+1/4,-z+1/4',
               'x,y+1/2,z+1/2', 'x+1/2,y,z+1/2'], 32, {}),
    'I4_1':  (['-y,x+1/2,z+1/4', 'x+1/2,y+1/2,z+1/2'], 8, {'00l': 'l=4n'}),
    # These three exist to exercise the ZONE LABELLING on the awkward families.
    # hhl (h == k) and h-hl (h == -k) are different zones carrying different
    # conditions, and the cubic groups generate the (-2k)kl / h(-2h)l forms.
    'P6_3/mmc': (['x-y,x,z+1/2', 'y,x,-z', '-x,-y,-z'], 24, {}),
    'R-3c':  (['-y,x-y,z', 'y,x,-z+1/2', '-x,-y,-z', 'x+2/3,y+1/3,z+1/3'], 36, {}),
    'Pa-3':  (['-x+1/2,-y,z+1/2', '-x,y+1/2,-z+1/2', 'z,x,y', '-x,-y,-z'], 24,
              {'0kl': 'k=2n'}),
}
DEN_ST = 24


def _parse_triplet(s):
    import re
    r = [0] * 9
    tn = [0, 0, 0]
    for row, part in enumerate(s.replace(' ', '').lower().split(',')):
        for t in re.findall(r'[+-]?[^+-]+', part):
            m = re.match(r'^([+-]?)(?:(\d+)/(\d+)|([xyz]))$', t)
            if not m:
                raise ValueError(f'bad term {t!r} in {s!r}')
            sign = -1 if m.group(1) == '-' else 1
            if m.group(4):
                r[row * 3 + {'x': 0, 'y': 1, 'z': 2}[m.group(4)]] = sign
            else:
                tn[row] += sign * DEN_ST * int(m.group(2)) // int(m.group(3))
    return tuple(r), tuple(v % DEN_ST for v in tn)


def _compose(a, b):
    ra, ta = a
    rb, tb = b
    r = [0] * 9
    for i in range(3):
        for j in range(3):
            r[i * 3 + j] = sum(ra[i * 3 + k] * rb[k * 3 + j] for k in range(3))
    t = [(ta[i] + sum(ra[i * 3 + k] * tb[k] for k in range(3))) % DEN_ST for i in range(3)]
    return tuple(r), tuple(t)


def _close(gens):
    ident = ((1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0))
    seen = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                c = _compose(a, g)
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
        if len(seen) > 400:
            raise ValueError('group did not close')
    return sorted(seen)


def self_test(box):
    print('Self-test: deriving zones and conditions from operator sets built by\n'
          'closing published generators, with no cctbx involved.\n')
    failures = 0
    for name, (gens, order, expect) in SELF_TEST.items():
        ops = _close([_parse_triplet(g) for g in gens])
        if len(ops) != order:
            print(f'  {name:9s} FAIL: closed to {len(ops)}, expected {order}')
            failures += 1
            continue
        records, defs, absent, grid = build_setting_zones(ops, DEN_ST, box)
        bad = verify_setting(records, absent, grid)
        printed = {r['zone']: r['rules'] for r in records if r['printed']}
        allz = {r['zone']: r['rules'] for r in records}
        miss = []
        for zone, rule in expect.items():
            if zone not in allz or rule not in allz[zone]:
                miss.append(f'{zone}: {rule}')
        status = 'ok' if not bad and not miss else 'FAIL'
        if bad or miss:
            failures += 1
        print(f'  {name:9s} order {len(ops):3d}  zones {len(records):2d}  '
              f'rule/operator mismatches {len(bad):4d}  {status}')
        if miss:
            print(f'      expected but not derived: {"; ".join(miss)}')
        pr = '; '.join(f'{z}: {", ".join(r)}' for z, r in sorted(printed.items()))
        print(f'      printed: {pr or "none"}')
        print(f'      zones:   {", ".join(sorted(allz))}   centring: {centring_from_ops(ops, DEN_ST)}')
    print('\n' + ('PASS: conditions reproduce the operators exactly.' if not failures
                  else f'FAIL: {failures} group(s)'))
    return 0 if failures == 0 else 1



# ===========================================================================
# Re-check an emitted file
# ===========================================================================

def check_file(path, box):
    """Validate sg_ops.json against itself, from scratch.

    Reads only the file -- no cctbx, no memory of how it was built -- rebuilds
    the operators from the rotation table, and confirms that

      * every zone label used by a condition exists in zone_defs
      * every reflection's present/absent verdict from the printed conditions
        matches the verdict from the operators
      * crystal_system is lower case, because sgSystemMatches() in
        worker-logic.js compares it with ===
      * centering is a letter the app's filter recognises

    Anything this catches would otherwise have surfaced as a silently wrong
    space-group ranking.
    """
    with open(path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    rots = [tuple(r) for r in db.get('rotations', [])]
    zone_defs = db.get('zone_defs', {})
    groups = db.get('space_groups', {})
    if not rots or not groups:
        print(f'{path}: missing rotations or space_groups'); return 1

    grid = [(h, k, l)
            for h in range(-box, box + 1)
            for k in range(-box, box + 1)
            for l in range(-box, box + 1)
            if (h, k, l) != (0, 0, 0)]

    VALID_SYSTEMS = {'triclinic', 'monoclinic', 'orthorhombic',
                     'tetragonal', 'trigonal', 'hexagonal', 'cubic'}
    VALID_CENTRING = set('PABCIFR')

    n_set = 0
    bad_cond, bad_zone, bad_sys, bad_cent = [], [], [], []
    for gk, g in groups.items():
        sysname = g.get('crystal_system')
        if sysname not in VALID_SYSTEMS:
            bad_sys.append((gk, sysname))
        for st in g.get('settings', []):
            n_set += 1
            t_den = st['t_den']
            ops = [(rots[o[0]], (o[1], o[2], o[3])) for o in st['ops']]
            if st.get('centering') not in VALID_CENTRING:
                bad_cent.append((st.get('symbol'), st.get('centering')))

            compiled = []
            for zone, rules in (st.get('conditions') or {}).items():
                normals = zone_defs.get(zone)
                if normals is None:
                    bad_zone.append((st.get('symbol'), zone))
                    continue
                fns = [_parse_rule(r) for r in rules]
                if any(f is None for f in fns):
                    bad_cond.append((st.get('symbol'), zone, 'unparseable rule'))
                    continue
                compiled.append((normals, fns))

            wrong = 0
            for h in grid:
                present = True
                for normals, fns in compiled:
                    if not _in_zone(normals, h):
                        continue
                    if any(not f(h) for f in fns):
                        present = False
                        break
                if present == is_absent(h, ops, t_den):
                    wrong += 1
            if wrong:
                bad_cond.append((st.get('symbol'), '-', f'{wrong} reflections disagree'))

    print(f'{path}: {len(groups)} space groups, {n_set} settings, '
          f'{len(rots)} rotations, {len(zone_defs)} zone labels')
    ok = True
    for name, items in (('conditions vs operators', bad_cond),
                        ('zone labels missing from zone_defs', bad_zone),
                        ('crystal_system not lower case / unknown', bad_sys),
                        ('unrecognised centring letter', bad_cent)):
        if items:
            ok = False
            print(f'  !! {name}: {len(items)}')
            for it in items[:6]:
                print(f'     {it}')
        else:
            print(f'  ok  {name}')
    print('\n' + ('PASS: the file is internally consistent.' if ok else 'FAIL'))
    return 0 if ok else 1


def _parse_rule(text):
    """'2*h+l=4n' -> predicate on (h,k,l). Mirrors the JS side exactly."""
    import re
    m = re.match(r'^(.*)=(\d+)n$', str(text).replace(' ', ''))
    if not m:
        return None
    mod = int(m.group(2))
    terms = re.findall(r'[+-]?(?:\d+\*)?[hkl]', m.group(1))
    if not terms:
        return None
    coef = {'h': 0, 'k': 0, 'l': 0}
    for t in terms:
        mm = re.match(r'^([+-]?)(?:(\d+)\*)?([hkl])$', t)
        if not mm:
            return None
        coef[mm.group(3)] += (-1 if mm.group(1) == '-' else 1) * int(mm.group(2) or 1)
    def pred(h, c=coef, m_=mod):
        return (c['h'] * h[0] + c['k'] * h[1] + c['l'] * h[2]) % m_ == 0
    return pred


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sg_ops.json')
    ap.add_argument('--box', type=int, default=5,
                    help='hkl half-range used to verify conditions against operators')
    ap.add_argument('--pretty', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='stop after N settings (smoke test)')
    ap.add_argument('--self-test', action='store_true',
                    help='check the derivation on built-in fixtures, without cctbx')
    ap.add_argument('--check', metavar='FILE',
                    help='re-verify an already-built sg_ops.json, without cctbx')
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.box)
    if args.check:
        return check_file(args.check, args.box)

    sgtbx = load_cctbx()

    rot_table, rot_index = [], {}
    zone_defs, zone_clashes = {}, []
    groups = OrderedDict()
    n_settings = n_ops = 0
    verify_failures = []
    centring_disagreements = []

    for idx, symbols in enumerate(sgtbx.space_group_symbol_iterator(), start=1):
        if args.limit and n_settings >= args.limit:
            break
        hall = symbols.hall()
        sg = sgtbx.space_group(hall)
        ops, t_den = extract_ops(sg)

        records, defs, absent, grid = build_setting_zones(ops, t_den, args.box)
        bad = verify_setting(records, absent, grid)
        if bad:
            verify_failures.append((symbols.hermann_mauguin(), len(bad), bad[:4]))

        for label, normals in defs.items():
            canon = tuple(tuple(n) for n in normals)
            if label not in zone_defs:
                zone_defs[label] = canon
            elif zone_defs[label] != canon:
                zone_clashes.append((label, symbols.hermann_mauguin()))

        packed = []
        for r9, tnum in ops:
            key = r9
            if key not in rot_index:
                rot_index[key] = len(rot_table)
                rot_table.append(list(key))
            packed.append([rot_index[key], tnum[0], tnum[1], tnum[2]])

        conditions = OrderedDict()
        for rec in sorted(records, key=lambda r: r['zone']):
            if rec['printed']:
                conditions[rec['zone']] = rec['rules']

        number = int(symbols.number())
        key = str(number)
        if key not in groups:
            groups[key] = OrderedDict([
                ('number', number),
                ('standard_symbol', symbols.hermann_mauguin()),
                # LOWERCASE: sgSystemMatches() in worker-logic.js compares this
                # string exactly, and cctbx returns 'Orthorhombic'.
                ('crystal_system', str(sg.crystal_system()).lower()),
                ('point_group', point_group_symbol(sgtbx, sg)),
                ('centrosymmetric', bool(sg.is_centric())),
                ('chiral', bool(sg.is_chiral())),
                ('settings', []),
            ])

        # Centring: cctbx if it answers, but always cross-checked against the
        # operators. They cannot disagree unless one of them is wrong, and the
        # operators are what the app actually uses for absences, so they win.
        derived = centring_from_ops(ops, t_den)
        try:
            reported = str(sg.conventional_centring_type_symbol())
        except Exception:
            reported = None
        centering = derived if derived != '?' else (reported or '?')
        if reported and derived != '?' and reported != derived:
            centring_disagreements.append((symbols.hermann_mauguin(), reported, derived))

        ext = symbols.extension()
        qual = symbols.qualifier()
        desc = ' '.join(x for x in [qual, f'origin {ext}' if ext else ''] if x).strip()
        groups[key]['settings'].append(OrderedDict([
            ('setting_id', f'{number:03d}_{len(groups[key]["settings"]) + 1:02d}'),
            ('setting_number', idx),
            ('symbol', symbols.hermann_mauguin().replace(' ', '')),
            ('hm', symbols.hermann_mauguin()),
            ('description', desc),
            ('hall', hall),
            ('centering', centering),
            ('order_z', int(sg.order_z())),
            ('order_p', int(sg.order_p())),
            ('t_den', t_den),
            ('ops', packed),
            ('conditions', conditions),
        ]))
        n_settings += 1
        n_ops += len(packed)

    ordered = OrderedDict(sorted(groups.items(), key=lambda kv: int(kv[0])))
    payload = OrderedDict([
        ('format', 'brutus-sg/1'),
        ('built_by', os.path.basename(__file__)),
        ('note', 'Operators are authoritative for systematic absences: h is '
                 'absent iff some operator has hR = h with h.t non-integral. '
                 "'conditions' and 'zone_defs' are derived from those same "
                 'operators and verified against them. Each op is '
                 '[rot_index, t0, t1, t2] with t = t_num / t_den.'),
        ('rotations', rot_table),
        ('zone_defs', OrderedDict((k, [list(v) for v in zone_defs[k]])
                                  for k in sorted(zone_defs))),
        ('setting_count', n_settings),
        ('space_groups', ordered),
    ])

    kw = dict(ensure_ascii=False)
    kw.update({'indent': 2} if args.pretty else {'separators': (',', ':')})
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, **kw)

    size = os.path.getsize(args.out)
    print(f'built {n_settings} settings in {len(ordered)} space groups '
          f'-> {args.out} ({size / 1024:.0f} KB)')
    print(f'  distinct rotation matrices : {len(rot_table)}')
    print(f'  total operators            : {n_ops}')
    print(f'  zone labels                : {len(zone_defs)}')
    print(f'    {", ".join(sorted(zone_defs))}')

    if zone_clashes:
        print(f'  !! {len(zone_clashes)} zone-label clashes -- a label means two '
              f'different things, so zone_defs cannot be trusted:')
        for label, sym in zone_clashes[:6]:
            print(f'     {label!r} first seen elsewhere, redefined by {sym}')
    if centring_disagreements:
        print(f'  !! {len(centring_disagreements)} settings where cctbx and the operators '
              f'disagree on the centring letter (the operator value was used):')
        for sym, rep, der in centring_disagreements[:6]:
            print(f'     {sym}: cctbx says {rep!r}, operators say {der!r}')
    if verify_failures:
        print(f'  !! {len(verify_failures)} settings where the derived conditions do NOT '
              f'reproduce the operators:')
        for sym, n, examples in verify_failures[:6]:
            print(f'     {sym}: {n} reflections disagree, e.g. {examples}')
        print('     The operators in the file are still correct -- absences never '
              'come from the condition strings -- but the printed rules are wrong '
              'and should not be trusted until this is fixed.')
    else:
        print('  conditions verified against the operators on every setting: no disagreements')

    return 1 if (zone_clashes or verify_failures) else 0


if __name__ == '__main__':
    sys.exit(main())
