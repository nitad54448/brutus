# 22 Aug 2026 - adds the data needed to build a STRUCTURE, not just index peaks.
# this is v8, used for SpaceExplorer
# CHANGE: one JSON file per SETTING in ./sg/, plus a light index.
#
# v8 fixes (the earlier v4-v7 fixes remain below):
#   1. Harker sections were also emitted for pure centring translations, whose
#      rotation is the identity. Those are lattice vectors, not Harker
#      geometry: C2/c gained three spurious "line" sections. Identity
#      rotations are now skipped.
#   2. The per-setting except-clause printed sg_num before it could be bound,
#      turning any early failure into a NameError inside the handler.
#   3. Trigonal/hexagonal groups have their conditions on the h-h0l zone
#      (hh0l in 4-index form), which was never sampled. It is now, for those
#      two crystal systems only.
#   4. all.json (a second monolithic copy of the whole database) was written
#      even with --no-legacy. --no-legacy now suppresses both.
#   5. The Hermann-Mauguin symbol is also stored WITH its spaces, as "hm".
#      "P6522" cannot be typeset: 65 is a screw axis and the two 2s are not,
#      and nothing in the unspaced string says so. "P 65 2 2" can.
#   6. resolve_output_dir() also recognises the SpaceExplorer folder
#      (index.html + app.js).
#   The single monolithic file held all 230 groups in all settings. Only one
#   setting is ever live, so the browser parsed the whole thing to use a
#   fraction of a percent of it. sg/index.json now carries just enough to build
#   a space-group picker (symbols, settings, Laue class), and sg/62.json is
#   fetched only when the user actually selects Pnma.
#

import json
import os
import re
import base64
import itertools
import functools
from collections import OrderedDict, defaultdict
from cctbx import sgtbx
from fractions import Fraction
from math import gcd
import sys
import argparse
import math


def get_rotations(sg_info):
    """Extracts all real-space rotational matrices for the space group.

    UNCHANGED from v5, and deliberately so: powder5.html may still read it.
    Do NOT use this for building a structure -- see get_symmetry_operations().
    """
    sg = sg_info.group()
    rots = []
    for i in range(sg.order_z()):
        r = tuple(sg(i).r().as_double())
        if r not in rots:
            rots.append(r)
    return rots


def _rational_triplet(num, den):
    """(numerators, denominator) -> list of 'p/q' strings in lowest terms."""
    out = []
    for n in num:
        f = Fraction(int(n), int(den))
        out.append(f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator))
    return out


def get_symmetry_operations(sg_info):
    """Full operator list: rotation, translation, and the xyz triplet.

    Returned per operator:
        xyz    : 'x,y,z' style string, exactly as cctbx formats it
        r      : 9 integers, row-major, x' = r . x + t
        t_num  : 3 integers, numerators of the translation
        t_den  : single integer denominator (cctbx uses 12, or 24 for some
                 d-glides); t = t_num / t_den exactly
        t      : the same translation as floats, for convenience

    Every operator of the group is listed, including the centring ones, so a
    consumer can apply the list directly with no further expansion.
    """
    sg = sg_info.group()
    ops = []
    for i in range(sg.order_z()):
        op = sg(i)
        r = op.r()
        t = op.t()
        r_num = [int(v) for v in r.num()]
        r_den = int(r.den())
        if r_den != 1:
            # cctbx always uses r_den == 1 for space-group rotations; guard
            # anyway so a silent non-integer matrix can never slip through.
            r_num = [v // r_den for v in r_num]
        t_num = [int(v) for v in t.num()]
        t_den = int(t.den())
        ops.append(OrderedDict([
            ("xyz", str(op)),
            ("r", r_num),
            ("t_num", t_num),
            ("t_den", t_den),
            ("t", [v / t_den for v in t_num]),
            ("t_frac", _rational_triplet(t_num, t_den)),
        ]))
    return ops


def get_centring(sg_info):
    """Centring translations as exact rationals, plus the conventional letter.

    Derived from the OPERATOR LIST rather than from sg.ltr(): the centring
    translations are exactly the translation parts of the operators whose
    rotation is the identity. The indexed accessor sg.ltr(i) is not available
    in every cctbx build ("_.ltr() takes 1 positional argument but 2 were
    given"), and there is no need for it -- this derivation is exact and works
    everywhere.
    """
    sg = sg_info.group()
    ltr = []
    seen = set()
    identity = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    for i in range(sg.order_z()):
        op = sg(i)
        r = op.r()
        r_num = tuple(int(v) for v in r.num())
        r_den = int(r.den())
        if r_den != 1:
            r_num = tuple(v // r_den for v in r_num)
        if r_num != identity:
            continue
        t = op.t()
        num = [int(x) for x in t.num()]
        den = int(t.den())
        key = tuple(Fraction(x, den) % 1 for x in num)
        if key in seen:
            continue
        seen.add(key)
        ltr.append(OrderedDict([
            ("t_num", num),
            ("t_den", den),
            ("t", [x / den for x in num]),
            ("t_frac", _rational_triplet(num, den)),
        ]))

    letter = "?"
    for getter in (
        lambda: sg.conventional_centring_type_symbol(),
        lambda: sg_info.type().lookup_symbol().strip().split(":")[0].strip()[0],
    ):
        try:
            value = getter()
            if value:
                letter = str(value).strip()[0]
                break
        except Exception:
            continue

    return letter, ltr


# cctbx does not expose the Wyckoff table at one fixed place. Depending on the
# build it is a method on space_group_info, or a submodule of sgtbx that
# "from cctbx import sgtbx" does NOT import for you (which is what produced
# "module 'cctbx.sgtbx' has no attribute 'wyckoff'" -- nothing to pip install,
# just an import that was never done). Try each in turn and remember which one
# worked so the probing happens once.
_WYCKOFF_ACCESSOR = "unprobed"


def _get_wyckoff_table_object(sg_info):
    global _WYCKOFF_ACCESSOR

    def _via_info(info):
        return info.wyckoff_table()

    def _via_submodule(info):
        from cctbx.sgtbx import wyckoff as _w
        return _w.table(info)

    def _via_import(info):
        import importlib
        _w = importlib.import_module("cctbx.sgtbx.wyckoff")
        return _w.table(info)

    def _via_attr(info):
        return sgtbx.wyckoff.table(info)

    def _via_flat(info):
        return sgtbx.wyckoff_table(info)

    routes = [
        ("space_group_info.wyckoff_table()", _via_info),
        ("from cctbx.sgtbx import wyckoff", _via_submodule),
        ("importlib cctbx.sgtbx.wyckoff", _via_import),
        ("sgtbx.wyckoff.table()", _via_attr),
        ("sgtbx.wyckoff_table()", _via_flat),
    ]

    # Once a route is known to work, stop probing.
    if _WYCKOFF_ACCESSOR not in ("unprobed", "none"):
        for name, fn in routes:
            if name == _WYCKOFF_ACCESSOR:
                try:
                    return fn(sg_info)
                except Exception:
                    break
    if _WYCKOFF_ACCESSOR == "none":
        return None

    errors = []
    for name, fn in routes:
        try:
            table = fn(sg_info)
            if table is not None:
                if _WYCKOFF_ACCESSOR == "unprobed":
                    print(f"\n[i] Wyckoff tables found via {name}.")
                    sys.stdout.flush()
                _WYCKOFF_ACCESSOR = name
                return table
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    _WYCKOFF_ACCESSOR = "none"
    print("\n[!] No Wyckoff accessor worked in this cctbx build. This is OPTIONAL:")
    print("    symops, centring and everything else are still written, and the")
    print("    structure builder derives site multiplicity from the operators")
    print("    itself. Only the ITA letters (4c, 8d, ...) will be missing.")
    for e in errors:
        print(f"      tried {e}")
    sys.stdout.flush()
    return None


def _rank3(num, den):
    """Exact rank of a 3x3 matrix given as 9 numerators over one denominator.

    Gaussian elimination over Fraction, not float: the projector for a position
    like (x,x,z) carries halves and thirds, and a rank misjudged by rounding
    would silently give a site the wrong number of free parameters.
    """
    rows = [[Fraction(int(num[r * 3 + c]), int(den)) for c in range(3)] for r in range(3)]
    rank = 0
    for col in range(3):
        pivot = next((r for r in range(rank, 3) if rows[r][col] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pv = rows[rank][col]
        rows[rank] = [v / pv for v in rows[rank]]
        for r in range(3):
            if r != rank and rows[r][col] != 0:
                f = rows[r][col]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


# A point with no rational relationship between its coordinates and no small
# denominators. Projecting it through special_op lands on a GENERIC point of the
# Wyckoff position; a tidier probe like (0.1, 0.2, 0.3) can land on a
# higher-symmetry sub-position and return too few coset operators.
_PROBE = (0.1357913, 0.2468135, 0.3791357)


def _apply_op(op, p):
    r = op.r().as_double()
    t = op.t().as_double()
    return (r[0]*p[0] + r[1]*p[1] + r[2]*p[2] + t[0],
            r[3]*p[0] + r[4]*p[1] + r[5]*p[2] + t[1],
            r[6]*p[0] + r[7]*p[1] + r[8]*p[2] + t[2])


def _coset_ops(sg, special_op, multiplicity):
    """Operator indices producing distinct images of a site on this position.

    Returns (ops, exact). `exact` is False when the count disagrees with the
    tabulated multiplicity, which means the probe point was unlucky; the field
    is still written but flagged, and a consumer should fall back to a
    distance-based coincidence test for that position alone.
    """
    p = _apply_op(special_op, _PROBE)
    seen, ops = [], []
    for i in range(sg.order_z()):
        n = _apply_op(sg(i), p)
        n = tuple(v - math.floor(v) for v in n)
        dup = False
        for s in seen:
            d = [n[k] - s[k] for k in range(3)]
            d = [v - round(v) for v in d]
            if abs(d[0]) < 1e-6 and abs(d[1]) < 1e-6 and abs(d[2]) < 1e-6:
                dup = True
                break
        if not dup:
            seen.append(n)
            ops.append(i)
    return ops, (len(ops) == int(multiplicity))


# v4: the old site analysis could only look for a single congruence that every
# present reflection happened to satisfy. The tables' special-position entries
# are frequently disjunctive -- Fd-3m 8a is "hkl : h = 2n+1 or h+k+l = 4n" --
# and no single congruence describes that set, so every diamond-type position
# came out with no conditions at all. The search now runs over a candidate list
# that includes those forms, and tests them against the reflections the GROUP
# allows rather than against all of them, so a rule that merely repeats a
# general condition is discarded by construction.

# ===========================================================================
#  EXACT ANALYSIS  (v5)
#
#  Everything below derives its answers algebraically from the operators cctbx
#  hands over. There is no probe point, no sampled box of reflections and no
#  floating-point tolerance anywhere in the derivation.
#
#  cctbx supplies: the operators as exact rationals, the Wyckoff table, the
#  site symmetry, and an exact is_sys_absent(). It does not publish the ITA
#  reflection-condition strings, nor the ITA coordinate parametrisation --
#  special_op() is the projection matrix, which is why it reads
#  "1/2*x+1/2*y,1/2*x+1/2*y,z" where the tables print "x, x, z". Those two are
#  derived here, and both derivations are checked against cctbx afterwards.
# ===========================================================================


# ------------------------------------------------------- exact linear algebra
def smith_normal_form(A):
    """U * A * V = D with U, V unimodular and D diagonal. Returns D, U, V."""
    A = [list(map(int, row)) for row in A]
    m = len(A)
    n = len(A[0]) if m else 0
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def swap_rows(i, j):
        A[i], A[j] = A[j], A[i]
        U[i], U[j] = U[j], U[i]

    def swap_cols(i, j):
        for r in A:
            r[i], r[j] = r[j], r[i]
        for r in V:
            r[i], r[j] = r[j], r[i]

    def add_row(i, j, c):
        A[i] = [a + c * b for a, b in zip(A[i], A[j])]
        U[i] = [a + c * b for a, b in zip(U[i], U[j])]

    def add_col(i, j, c):
        for r in A:
            r[i] += c * r[j]
        for r in V:
            r[i] += c * r[j]

    t = 0
    while t < min(m, n):
        piv = None
        for i in range(t, m):
            for j in range(t, n):
                if A[i][j]:
                    piv = (i, j)
                    break
            if piv:
                break
        if piv is None:
            break
        swap_rows(t, piv[0])
        swap_cols(t, piv[1])

        while True:
            for i in range(t + 1, m):
                if A[i][t]:
                    add_row(i, t, -(A[i][t] // A[t][t]))
                    if A[i][t]:
                        swap_rows(i, t)
            for j in range(t + 1, n):
                if A[t][j]:
                    add_col(j, t, -(A[t][j] // A[t][t]))
                    if A[t][j]:
                        swap_cols(j, t)
            if all(A[i][t] == 0 for i in range(t + 1, m)) and \
               all(A[t][j] == 0 for j in range(t + 1, n)):
                break

        redo = False
        for i in range(t + 1, m):
            for j in range(t + 1, n):
                if A[t][t] and A[i][j] % A[t][t]:
                    add_row(t, i, 1)
                    redo = True
                    break
            if redo:
                break
        if redo:
            continue
        if A[t][t] < 0:
            add_row(t, t, -2)
        t += 1

    return A, U, V


def integer_kernel_rows(A):
    """Basis of {h in Z^m : h A = 0}."""
    m = len(A)
    if m == 0:
        return []
    n = len(A[0])
    D, U, _ = smith_normal_form(A)
    return [U[i] for i in range(m) if i >= n or D[i][i] == 0]


def lattice_rref(rows):
    """Reduced basis of the lattice spanned by `rows`, and its pivot columns.

    For a saturated sublattice -- which every kernel here is -- the pivots come
    out as 1, so the coefficient of basis vector i in a point h is just h[p_i].
    """
    rows = [list(map(int, r)) for r in rows if any(r)]
    if not rows:
        return [], []
    width = len(rows[0])
    piv, r = [], 0
    for c in range(width):
        while sum(1 for i in range(r, len(rows)) if rows[i][c]) > 1:
            idx = sorted((i for i in range(r, len(rows)) if rows[i][c]),
                         key=lambda i: abs(rows[i][c]))
            i0, i1 = idx[0], idx[1]
            q = rows[i1][c] // rows[i0][c]
            rows[i1] = [a - q * b for a, b in zip(rows[i1], rows[i0])]
        nz = [i for i in range(r, len(rows)) if rows[i][c]]
        if not nz:
            continue
        rows[r], rows[nz[0]] = rows[nz[0]], rows[r]
        if rows[r][c] < 0:
            rows[r] = [-a for a in rows[r]]
        for i in range(r):
            q = rows[i][c] // rows[r][c]
            if q:
                rows[i] = [a - q * b for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    return rows[:r], piv


def sublattice_from_congruences(dim, congruences):
    """{a in Z^dim : sum_i c_i a_i = 0 mod M} for each (c, M), as a basis.

    Solved as the integer kernel of [C | -diag(M)] projected back onto a.
    """
    if not congruences:
        return [[int(i == j) for j in range(dim)] for i in range(dim)]
    k = len(congruences)
    A = [[0] * k for _ in range(dim + k)]
    for j, (c, M) in enumerate(congruences):
        for i in range(dim):
            A[i][j] = int(c[i])
        A[dim + j][j] = -int(M)
    ker = integer_kernel_rows(A)
    basis, _ = lattice_rref([row[:dim] for row in ker])
    return basis


LETTERS = ('h', 'k', 'l')


def zone_label(basis, pivots, names=None):
    """'hkl', '0kl', 'hhl', 'h-hl', '00l' ... from a reduced zone basis."""
    if names is None:
        names = [LETTERS[p] for p in pivots]
    out = ''
    for c in range(3):
        terms = [(basis[i][c], names[i]) for i in range(len(basis)) if basis[i][c]]
        if not terms:
            out += '0'
        elif len(terms) == 1 and terms[0][0] == 1:
            out += terms[0][1]
        elif len(terms) == 1 and terms[0][0] == -1:
            out += '-' + terms[0][1]
        else:
            out += '(' + '+'.join(f'{co}{nm}' for co, nm in terms).replace('+-', '-') + ')'
    return out


def form_string(coeffs, names, modulus):
    """(2, 1) over ('h','l') mod 4 -> '2*h+l=4n'. Coefficients reduced mod M."""
    half = modulus // 2
    red = []
    for c in coeffs:
        c %= modulus
        if c > half:
            c -= modulus
        red.append(c)
    if not any(red):
        return None
    if next(c for c in red if c) < 0:
        red = [-c for c in red]
    parts = []
    for c, nm in zip(red, names):
        if not c:
            continue
        if c == 1:
            parts.append('+' + nm)
        elif c == -1:
            parts.append('-' + nm)
        else:
            parts.append(('+' if c > 0 else '-') + f'{abs(c)}*{nm}')
    return ''.join(parts).lstrip('+') + f'={modulus}n'


def _op_rt(op):
    """(R as 9 ints, t as 3 Fractions) from a cctbx rt_mx."""
    r, t = op.r(), op.t()
    rden = int(r.den()) or 1
    rnum = [int(v) for v in r.num()]
    if rden != 1:
        rnum = [v // rden for v in rnum]
    tden = int(t.den()) or 1
    return rnum, [Fraction(int(v), tden) for v in t.num()]


def sublattice_congruence_forms(basis, dim):
    """Canonical minimal congruences cutting `basis` out of Z^dim.

    Smith normal form gives the invariant factors of the quotient and one
    congruence per factor: the shortest true description, arrived at rather
    than guessed. Returned in the zone's own coordinates.
    """
    if not basis:
        return []
    D, _, V = smith_normal_form([row[:] for row in basis])
    out = []
    for i in range(dim):
        e = D[i][i] if i < len(D) and i < len(D[0]) else 0
        if e in (0, 1, -1):
            continue
        out.append(([V[j][i] for j in range(dim)], abs(e)))
    return out


# The zones the tables print, so that conditions inherited from a larger zone
# are stated where the tables state them. This is a presentation list only:
# correctness comes from the operator kernels below, which are complete.
STANDARD_ZONES = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0)),
    ((1, 1, 0), (0, 0, 1)), ((1, -1, 0), (0, 0, 1)),
    ((1, 0, 1), (0, 1, 0)), ((1, 0, -1), (0, 1, 0)),
    ((0, 1, 1), (1, 0, 0)), ((0, 1, -1), (1, 0, 0)),
    ((1, 0, 0),), ((0, 1, 0),), ((0, 0, 1),),
    ((1, 1, 0),), ((1, -1, 0),), ((1, 0, 1),), ((1, 0, -1),),
    ((0, 1, 1),), ((0, 1, -1),), ((1, 1, 1),),
)

PRINTED_ZONES = {
    "triclinic":    ('hkl', '0kl', 'h0l', 'hk0', 'h00', '0k0', '00l'),
    "monoclinic":   ('hkl', '0kl', 'h0l', 'hk0', 'h00', '0k0', '00l'),
    "orthorhombic": ('hkl', '0kl', 'h0l', 'hk0', 'h00', '0k0', '00l'),
    "tetragonal":   ('hkl', 'hk0', '0kl', 'h0l', 'hhl', 'h-hl', '00l', 'h00', '0k0'),
    "trigonal":     ('hkl', 'hk0', 'h-hl', 'hhl', '00l', '0kl', 'h0l', 'h00', '0k0'),
    "rhombohedral": ('hkl', 'hk0', 'h-hl', 'hhl', '00l', '0kl', 'h0l', 'h00', '0k0'),
    "hexagonal":    ('hkl', 'hk0', 'h-hl', 'hhl', '00l', '0kl', 'h0l', 'h00', '0k0'),
    "cubic":        ('hkl', '0kl', 'h0l', 'hk0', 'hhl', 'h-hl', 'hkh', 'hk-h',
                     'hkk', 'hk-k', 'h00', '0k0', '00l'),
}
DEFAULT_PRINTED = PRINTED_ZONES["orthorhombic"]


def zone_from_rows(rows):
    """One zone, with everything needed to read a reflection's coordinates on it.

    A zone basis need not have unit pivots: in a hexagonal group the mirror
    (x-y, -y, z) fixes the reflections with h = -2k, whose basis is (-2,1,0),
    (0,0,1). The coefficient of a basis vector is then NOT simply h, k or l, so
    the dual forms are obtained from the Smith normal form instead of assumed.
    """
    basis, piv = lattice_rref([list(r) for r in rows])
    if not basis:
        return None
    d = len(basis)
    key = tuple(map(tuple, basis))

    # Name each parameter after a coordinate it alone controls, where one exists.
    names = []
    for i, row in enumerate(basis):
        pick = None
        for c in range(3):
            if abs(row[c]) == 1 and all(basis[j][c] == 0 for j in range(d) if j != i):
                pick = c
                break
        if pick is None:
            names.append(LETTERS[piv[i]] if i < len(piv) else f'p{i + 1}')
        else:
            if row[pick] == -1:
                basis[i] = [-v for v in row]
            names.append(LETTERS[pick])

    D, U, V = smith_normal_form([r[:] for r in basis])
    if any((D[i][i] if i < len(D[0]) else 0) not in (1, -1) for i in range(d)):
        return None                       # not saturated: cannot be a zone
    normals = [[V[m][j] for m in range(3)] for j in range(d, 3)]
    duals = [[sum(V[m][j] * U[j][i] for j in range(d)) for i in range(d)]
             for m in range(3)]
    return {"key": key, "basis": basis, "dim": d, "names": names,
            "normals": normals, "duals": duals,
            "label": zone_label(basis, list(range(d)), names)}


def zone_universe(sg, ops):
    """Every zone that can carry a condition, largest first.

    The operator kernels are what makes this complete -- an absence can only
    come from an operator, and that operator's kernel is in this list. The
    standard zones are added on top so that inherited conditions are stated
    where the tables state them.
    """
    seen, out = set(), []
    sources = []
    for R, _t in ops:
        A = [[R[3 * i + j] - (1 if i == j else 0) for j in range(3)] for i in range(3)]
        sources.append(integer_kernel_rows(A))
    sources.extend([list(map(list, z)) for z in STANDARD_ZONES])
    for rows in sources:
        z = zone_from_rows(rows)
        if z and z["key"] not in seen:
            seen.add(z["key"])
            out.append(z)
    out.sort(key=lambda z: -z["dim"])
    return out


def _zone_fixed_by(basis, R):
    return all(
        all(sum(b[i] * (R[3 * i + c] - (1 if i == c else 0)) for i in range(3)) == 0
            for c in range(3))
        for b in basis)


def in_zone(hkl, zone):
    h, k, l = hkl
    return all(n[0] * h + n[1] * k + n[2] * l == 0 for n in zone["normals"])


def exact_reflection_conditions(sg):
    """ITA-style reflection conditions, derived rather than fitted.

    A reflection h is extinguished exactly when some operator (R, t) has hR = h
    while h.t is not an integer. The h with hR = h form an integer sublattice --
    the zone -- obtained here as a kernel over Z. On that zone h.t in Z is a
    linear congruence, and reducing the congruences bearing on a zone by Smith
    normal form gives the shortest correct set of conditions for it.

    Returns (conditions, zone_records, zones):
      conditions    {label: [rules]}, one representative per symmetry orbit
      zone_records  every zone carrying a condition, with the linear forms that
                    vanish on it, so a client can test membership arithmetically
      zones         the full zone universe, for the site analysis
    """
    ops = [_op_rt(sg(i)) for i in range(sg.order_z())]
    zones = zone_universe(sg, ops)

    for z in zones:
        congruences = []
        for R, t in ops:
            if not _zone_fixed_by(z["basis"], R):
                continue
            vals = [sum(Fraction(b[i]) * t[i] for i in range(3)) for b in z["basis"]]
            if all(v.denominator == 1 for v in vals):
                continue
            M = 1
            for v in vals:
                M = M * v.denominator // gcd(M, v.denominator)
            congruences.append(([int(v * M) for v in vals], M))

        rules = []
        if congruences:
            H = sublattice_from_congruences(z["dim"], congruences)
            for coeffs, M in sublattice_congruence_forms(H, z["dim"]):
                hkl = [sum(coeffs[i] * z["duals"][m][i] for i in range(z["dim"]))
                       for m in range(3)]
                s = form_string(hkl, LETTERS, M)
                if s:
                    rules.append(s)
        z["rules"] = rules

    with_rules = [z for z in zones if z["rules"]]

    # Zones exchanged by the point group say the same thing once.
    orbit_of = {}
    for z in with_rules:
        if z["key"] in orbit_of:
            continue
        for R, _t in ops:
            img = [[sum(b[i] * R[3 * i + c] for i in range(3)) for c in range(3)]
                   for b in z["basis"]]
            im = zone_from_rows(img)
            if im:
                orbit_of.setdefault(im["key"], z["key"])

    try:
        system = str(sg.crystal_system()).lower()
    except Exception:
        system = ""
    preferred = PRINTED_ZONES.get(system, DEFAULT_PRINTED)
    rank = {lab: i for i, lab in enumerate(preferred)}
    with_rules.sort(key=lambda z: (rank.get(z["label"], 90), -z["dim"], z["label"]))

    conditions = OrderedDict()
    shown = set()
    zone_records = []
    # 'printed' answers "does the book state this zone here", which is not the
    # same question as "have we already tested this orbit": a zone that heads an
    # orbit but whose label is not among the printed ones is printed=False. A
    # consumer that deduplicates on 'printed' would then drop that orbit's rules
    # entirely, so each record also names its orbit head explicitly.
    label_by_key = {z["key"]: z["label"] for z in zones}
    for z in with_rules:
        head = orbit_of.get(z["key"], z["key"])
        first = head not in shown and z["label"] in rank
        if first:
            shown.add(head)
            conditions[z["label"]] = z["rules"]
        zone_records.append(OrderedDict([
            ("zone", z["label"]),
            ("orbit", label_by_key.get(head, z["label"])),
            ("normals", z["normals"]),
            ("rules", z["rules"]),
            ("printed", first),
        ]))
    return conditions, zone_records, zones, orbit_of


def zone_definitions(zones):
    """label -> normals, for every zone in the universe.

    A Wyckoff position can carry a condition on a zone that the space group
    itself puts no condition on. Such a zone never reaches zone_records, so a
    consumer reading only that list cannot test the site condition and silently
    ignores it. This map covers the whole universe, so every label appearing in
    a Wyckoff 'conditions' block can be resolved to an arithmetic membership
    test rather than to a guess from the label's spelling.
    """
    defs = OrderedDict()
    for z in zones:
        defs.setdefault(z["label"], z["normals"])
    return defs


def reflection_is_absent(hkl, zone_records):
    """Absent iff h lies in a listed zone and breaks one of its rules."""
    h, k, l = hkl
    for z in zone_records:
        if not all(n[0] * h + n[1] * k + n[2] * l == 0 for n in z["normals"]):
            continue
        for rule in z["rules"]:
            if not evaluate_rule(h, k, l, rule):
                return True
    return False


def _reflection_check_radius(sg):
    """Choose a stronger automatic verification radius.

    Translation denominators set the natural phase scale. They do not, by
    themselves, make a finite Cartesian box mathematically exhaustive because
    the fixed-hkl equations hR=h are homogeneous lattices. We therefore use a
    radius comfortably larger than the phase period and describe the result as
    a tested sample, not a proof over all Miller indices.
    """
    period = 1
    for i in range(sg.order_z()):
        t = sg(i).t()
        period = math.lcm(period, int(t.den()) or 1)
    return max(6, 2 * period)


def check_conditions_against_cctbx(sg, zone_records, rng=None):
    """Compare derived conditions against cctbx on a large symmetric box."""
    if rng is None:
        rng = _reflection_check_radius(sg)

    bad = []
    for h in range(-rng, rng + 1):
        for k in range(-rng, rng + 1):
            for l in range(-rng, rng + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                r = (h, k, l)
                if reflection_is_absent(r, zone_records) != sg.is_sys_absent(r):
                    bad.append(r)
    return bad


# --------------------------------------------------- roots of unity, exactly
_CYCLOTOMIC = {}


def _cyclotomic(n):
    """Phi_n as an integer coefficient list, low order first."""
    if n in _CYCLOTOMIC:
        return _CYCLOTOMIC[n]
    poly = [-1] + [0] * (n - 1) + [1]          # x^n - 1
    for d in range(1, n):
        if n % d == 0:
            poly = _poly_divide(poly, _cyclotomic(d))
    _CYCLOTOMIC[n] = poly
    return poly


def _poly_divide(a, b):
    """Exact division of integer polynomials, b monic-leading and dividing a."""
    a = a[:]
    q = [0] * (len(a) - len(b) + 1)
    for i in range(len(a) - len(b), -1, -1):
        c = a[i + len(b) - 1] // b[-1]
        q[i] = c
        if c:
            for j, bj in enumerate(b):
                a[i + j] -= c * bj
    return q


def _poly_mod(a, b):
    a = a[:]
    for i in range(len(a) - len(b), -1, -1):
        c = a[i + len(b) - 1] // b[-1]
        if c:
            for j, bj in enumerate(b):
                a[i + j] -= c * bj
    return a[:len(b) - 1]


_ROOTS_ZERO = {}


def roots_of_unity_sum_is_zero(phases, n):
    """Is sum of zeta_n^p over p in `phases` exactly zero?

    Reduced to a polynomial identity: the sum vanishes iff the counting
    polynomial is divisible by the n-th cyclotomic polynomial. No floating
    point, so no threshold to get wrong.
    """
    key = (n, tuple(sorted(phases)))
    hit = _ROOTS_ZERO.get(key)
    if hit is not None:
        return hit
    counts = [0] * n
    for p in phases:
        counts[p % n] += 1
    rem = _poly_mod(counts + [0], _cyclotomic(n))
    out = not any(rem)
    _ROOTS_ZERO[key] = out
    return out





# --------------------------------------------- ITA coordinates for a position
def _rational_matrix(num, den):
    return [[Fraction(int(num[3 * i + j]), int(den)) for j in range(3)] for i in range(3)]


def ita_parametrisation(P_num, P_den, T_num, T_den):
    """Turn cctbx's projection matrix into the parametrisation the tables print.

    special_op() describes a Wyckoff position as a projector acting on a general
    point, which is why it reads "1/2*x+1/2*y, 1/2*x+1/2*y, z". The tables give
    the same set as "x, x, z": the image of the projector, with one free
    parameter per dimension, named after the first coordinate it controls.

    Returns (names, Q, T) where a point is  Q . (parameters) + T.
    """
    P = _rational_matrix(P_num, P_den)
    T = [Fraction(int(T_num[i]), int(T_den)) for i in range(3)]

    # Row-reduce the columns of P: they span the image.
    rows = [[P[i][j] for i in range(3)] for j in range(3)]
    basis, piv, r = [], [], 0
    for c in range(3):
        pivot = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        s = rows[r][c]
        rows[r] = [v / s for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    basis = rows[:r]

    names = ['xyz'[p] for p in piv]
    Q = [[basis[j][i] for j in range(len(basis))] for i in range(3)]
    return names, Q, T


def _coord_component(coeffs, names, const):
    """One coordinate: 'x', '-x+1/2', '2*x', '1/8' ..."""
    parts = []
    for c, nm in zip(coeffs, names):
        if not c:
            continue
        if c == 1:
            parts.append('+' + nm)
        elif c == -1:
            parts.append('-' + nm)
        else:
            parts.append(('+' if c > 0 else '-') + (f'{abs(c)}*{nm}' if abs(c) != 1 else nm))
    s = ''.join(parts).lstrip('+')
    const = Fraction(const) % 1
    if const:
        s += ('+' if s else '') + f'{const.numerator}/{const.denominator}'
    return s or '0'


def ita_coordinates(sg, coset_ops, names, Q, T):
    """The coordinate list the tables print, exactly, one entry per equivalent point."""
    out = []
    for i in coset_ops:
        R, t = _op_rt(sg(i))
        rows = []
        for r in range(3):
            coeffs = [sum(Fraction(R[3 * r + c]) * Q[c][j] for c in range(3))
                      for j in range(len(names))]
            const = sum(Fraction(R[3 * r + c]) * T[c] for c in range(3)) + t[r]
            rows.append(_coord_component(coeffs, names, const))
        out.append(','.join(rows))
    return out


# --------------------------------------------- reflection conditions of a site
_SITE_SINGLE = [
    "h+k+l=4n", "h+k=4n", "h+l=4n", "k+l=4n", "h=4n", "k=4n", "l=4n",
    "h+k+l=2n", "h+k=2n", "h+l=2n", "k+l=2n", "h=2n", "k=2n", "l=2n",
    "-h+k+l=3n", "h-k+l=3n", "h+k+l=3n", "h+k=3n", "h+l=3n", "k+l=3n",
    "h+k+l=6n", "l=6n", "l=3n", "h=3n", "k=3n",
]
# Forms with a coefficient of 2 are needed for the 4(1) axis families: an atom
# on 4a of I4(1) is present exactly when 2h+l is not 4n+2, which nothing built
# from h, k, l with unit coefficients can say.
_SITE_FORMS = [
    "h", "k", "l",
    "h+k", "h-k", "h+l", "h-l", "k+l", "k-l",
    "2*h+k", "h+2*k", "2*h-k", "h-2*k",
    "2*k+l", "k+2*l", "2*k-l", "k-2*l",
    "2*l+h", "l+2*h", "2*l-h", "l-2*h",
    "2*l+k", "l+2*k", "2*l-k", "l-2*k",
    "h+k+l", "-h+k+l", "h-k+l", "h+k-l", "h-k-l", "-h-k+l", "-h+k-l",
    "2*h+k+l", "h+2*k+l", "h+k+2*l",
    "2*h-k+l", "2*h+k-l", "h-2*k+l", "h+2*k-l", "h-k+2*l", "h+k-2*l",
    "2*h-k-l", "h-2*k-l", "h-k-2*l",
    "2*h+2*k+l", "2*h+k+2*l", "h+2*k+2*l",
    "2*h+2*k-l", "2*h-k+2*l", "-h+2*k+2*l",
]

_SITE_ODD = [f"{f}=2n+1" for f in _SITE_FORMS] + \
            [f"{f}!=3n" for f in _SITE_FORMS]

_SITE_PARTNER = [f"{f}=4n" for f in _SITE_FORMS] + \
                [f"{f}=2n" for f in _SITE_FORMS] + \
                [f"{f}=8n" for f in _SITE_FORMS] + \
                [f"{f}=3n" for f in _SITE_FORMS] + \
                [f"{f}=6n" for f in _SITE_FORMS]

# "F != 4n+2" is the compact way to write "F odd, or F a multiple of four",
# which is the shape these positions keep taking.
_SITE_NEQ = [f"{f}!=4n+2" for f in _SITE_FORMS] + \
            [f"{f}!=8n+4" for f in _SITE_FORMS] + \
            [f"{f}=4n+2" for f in _SITE_FORMS]
_SITE_CANDIDATES = None


_SITE_ATOMS = None


def site_atomic_rules():
    """Single and compound disjunctive clauses for site reflection conditions.

    Includes single clauses, two-clause disjunctions (A or B), and 3-fold cyclic
    disjunctions across cubic axes to cover all ITC special position conditions.
    """
    global _SITE_ATOMS
    if _SITE_ATOMS is None:
        def cyc(s):
            return s.replace('h', 'X').replace('k', 'h').replace('l', 'k').replace('X', 'l')

        singles = []
        for f in _SITE_FORMS:
            for r in ("=2n", "=2n+1"):
                singles.append(f"{f}{r}")
            for r in ("=3n", "=3n+1", "=3n+2", "!=3n"):
                singles.append(f"{f}{r}")
            for r in ("=4n", "=4n+1", "=4n+2", "=4n+3", "!=4n+2", "!=4n"):
                singles.append(f"{f}{r}")
            for r in ("=6n", "=6n+1", "=6n+2", "=6n+3", "=6n+4", "=6n+5"):
                singles.append(f"{f}{r}")
            for r in ("=8n", "=8n+1", "=8n+2", "=8n+3", "=8n+4", "=8n+5", "=8n+6", "=8n+7", "!=8n+4"):
                singles.append(f"{f}{r}")
        singles = list(OrderedDict.fromkeys(singles))

        # Parity and zone guard clauses used in ITC disjunctions
        guards = [
            "h=2n+1", "k=2n+1", "l=2n+1",
            "h=2n", "k=2n", "l=2n",
            "h+k=2n", "h+l=2n", "k+l=2n",
            "h+k+l=2n", "h+k+l=4n",
            "h=4n", "k=4n", "l=4n",
            "h=4n+2", "k=4n+2", "l=4n+2",
        ]

        compounds = []
        for g in guards:
            for s in singles:
                if g != s and not s.startswith(g + " "):
                    compounds.append(f"{g} or {s}")

        # Build 3-fold cyclic permutation triples (h -> k -> l -> h)
        cyclic_triples = []
        for s in singles:
            c1 = cyc(s)
            c2 = cyc(c1)
            if c1 != s and c2 != s and c1 != c2:
                cyclic_triples.append(f"{s} or {c1} or {c2}")

        triples = [
            "h=2n or k=2n or l=2n",
            "h=2n+1 or k=2n+1 or l=2n+1",
            "h=4n or k=4n or l=4n",
            "h=4n+2 or k=4n+2 or l=4n+2",
            "h+k=4n or k+l=4n or h+l=4n",
        ] + cyclic_triples

        all_rules = list(OrderedDict.fromkeys(singles + compounds + triples))
        _SITE_ATOMS = [(n, _compile_rule(n)) for n in all_rules]
    return _SITE_ATOMS


_SITE_MODULUS = None


def site_rule_modulus():
    """The common period of the rule vocabulary.

    Truth is periodic modulo N, but a rule of modulus 3 or 8 is not decided by
    the residues modulo 2, so comparing them on classes of period N alone lets
    an accidental agreement through: Pa-3 4a has only two present classes modulo
    2 and neither has l divisible by three, so "l!=3n" looked like a law. The
    classes are therefore taken modulo lcm(N, this), on which both sides are
    exactly periodic and the comparison is a proof rather than a test.
    """
    global _SITE_MODULUS
    if _SITE_MODULUS is None:
        m = 1
        for name, _fn in site_atomic_rules():
            for mod in re.findall(r"(\d+)n", name):
                v = int(mod)
                if v:
                    m = m * v // gcd(m, v)
        _SITE_MODULUS = m
    return _SITE_MODULUS


def site_candidate_rules():
    """Names for absence sets. The set itself is computed exactly first; this
    list only supplies a phrase for it, and a phrase is accepted only if it
    reproduces the set on every reflection tested."""
    global _SITE_CANDIDATES
    if _SITE_CANDIDATES is None:
        names = (_SITE_SINGLE
                 + [f"{f}=4n" for f in _SITE_FORMS if "*" in f]
                 + _SITE_NEQ
                 + [f"{a} or {b}" for a in _SITE_ODD for b in _SITE_PARTNER])
        _SITE_CANDIDATES = [(n, _compile_rule(n)) for n in names]
    return _SITE_CANDIDATES


def site_absence_machinery(sg, coset_ops, P_num, P_den, T_num, T_den):
    """Integer form of the structure-factor sum for one Wyckoff position.

    A point of the position is P.v + T for free v. Under operator g it maps to
    (R P).v + (R T + t), so the contribution of the orbit to F(h) is

        sum_g exp(2.pi.i [ (h R P).v + h.(R T + t) ])

    Terms sharing the same row vector h R P share their dependence on v and can
    only cancel among themselves. So the position is extinct at h exactly when
    each of those groups sums to zero -- a sum of roots of unity, tested
    exactly. Everything is scaled by N to keep it in integers.
    """
    P = _rational_matrix(P_num, P_den)
    T = [Fraction(int(T_num[i]), int(T_den)) for i in range(3)]

    N = 1
    mats, shifts = [], []
    for i in coset_ops:
        R, t = _op_rt(sg(i))
        RP = [[sum(Fraction(R[3 * a + c]) * P[c][b] for c in range(3)) for b in range(3)]
              for a in range(3)]
        w = [sum(Fraction(R[3 * a + c]) * T[c] for c in range(3)) + t[a] for a in range(3)]
        mats.append(RP)
        shifts.append(w)
        for row in RP:
            for v in row:
                N = N * v.denominator // gcd(N, v.denominator)
        for v in w:
            N = N * v.denominator // gcd(N, v.denominator)

    # h R P as integers: u = h . A / N, so compare h . A directly.
    A_list = [[[int(RP[a][b] * N) for b in range(3)] for a in range(3)] for RP in mats]
    w_list = [[int(v * N) for v in w] for w in shifts]
    return N, A_list, w_list


def site_is_absent(hkl, N, A_list, w_list):
    h, k, l = hkl
    groups = {}
    for A, w in zip(A_list, w_list):
        u = (h * A[0][0] + k * A[1][0] + l * A[2][0],
             h * A[0][1] + k * A[1][1] + l * A[2][1],
             h * A[0][2] + k * A[1][2] + l * A[2][2])
        p = (h * w[0] + k * w[1] + l * w[2]) % N
        g = groups.get(u)
        if g is None:
            groups[u] = [p]
        else:
            g.append(p)
    for phases in groups.values():
        if len(phases) == 1:
            return False                      # a single root of unity is never 0
        if not roots_of_unity_sum_is_zero(phases, N):
            return False
    return True


def _bitset_b64(bits):
    """Pack a boolean list into a compact base64 bitset."""
    raw = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            raw[i >> 3] |= 1 << (i & 7)
    return base64.b64encode(bytes(raw)).decode("ascii")


def exact_site_residue_data(N, A_list, w_list):
    """Exact machine-readable site extinctions over one complete period.

    All quantities entering site_is_absent() are integral modulo N, so the
    site-extinction predicate is periodic in each Miller index with period N and
    the bitset below is a lossless statement of it.

    It records the site predicate alone. Filtering out the reflections the space
    group already extinguishes, as this once did, is not a well-defined thing to
    do here: space-group absence depends on which zone a reflection lies in, and
    that is not a property of its residue class, so the filter would decide a
    whole class on the accident of one member. A consumer applies the ordinary
    space-group conditions separately, which is the only correct order anyway.
    """
    total = N * N * N
    bits = [False] * total
    for h in range(N):
        for k in range(N):
            for l in range(N):
                if h == 0 and k == 0 and l == 0:
                    continue
                if site_is_absent((h, k, l), N, A_list, w_list):
                    bits[(h * N + k) * N + l] = True
    return OrderedDict([
        ("encoding", "base64-bitset"),
        ("modulus", N),
        ("index", "((h mod N)*N + (k mod N))*N + (l mod N)"),
        ("allowed_by_space_group", False),
        ("data", _bitset_b64(bits)),
    ])


SITE_VERIFY_BOX = 6


def _site_sample_points(N, zones, box=SITE_VERIFY_BOX):
    """Reflections to test a site's absences on: one per residue class, per zone,
    together with an ordinary box of small reflections.

    The site predicate is periodic modulo N, so the residue classes describe it
    completely. Enumerating those classes as the box [0,N)^3 is not, however, a
    fair sample of the reflections themselves. Zone membership is not periodic,
    so a class whose representative happens to carry a zero index sits in a
    special zone and may be extinguished by the space group, while other members
    of the same class are ordinary reflections that the site really does
    extinguish. In Pa-3 the class of (1,1,0) is such a case: the representative
    dies on hk0 : h = 2n, but (1,1,2) is an allowed reflection which an atom on
    4a does extinguish, and dropping the class loses the whole F condition.

    So each zone is sampled in its own coordinates, and the representative is
    lifted by whole periods -- which stays inside the zone and inside the class
    -- until it is a reflection the space group allows.

    That structured sample still spans only one period, and a rule fitted to one
    period agrees with it by construction. P-4 2g is extinct exactly when l = 0
    and h+k is odd; over a sample of period two, "l = 0" and "l even" are the
    same statement, and the search happily returned the second. Only reflections
    chosen on some other basis can show that a fit does not generalise, so an
    ordinary Cartesian box goes in as well.
    """
    pts = set()
    for z in zones:
        basis, d = z["basis"], z["dim"]
        for coeffs in itertools.product(range(N), repeat=d):
            base = [sum(coeffs[i] * basis[i][c] for i in range(d)) for c in range(3)]
            for shift in itertools.product((0, 1), repeat=d):
                r = tuple(base[c] + N * sum(shift[i] * basis[i][c] for i in range(d))
                          for c in range(3))
                if r == (0, 0, 0) or reflection_is_absent(r, zones):
                    continue
                pts.add(r)
                break                      # one surviving lift per class is enough

    for h in range(-box, box + 1):
        for k in range(-box, box + 1):
            for l in range(-box, box + 1):
                r = (h, k, l)
                if r == (0, 0, 0) or reflection_is_absent(r, zones):
                    continue
                pts.add(r)
    return sorted(pts)


def _kernel_of_normals(normals):
    """{h : h.n = 0 for every n in normals} as integer basis rows."""
    if not normals:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    A = [[n[i] for n in normals] for i in range(3)]
    return integer_kernel_rows(A)


def site_strata(A_list):
    """The sublattices on which the operator grouping is constant, largest first.

    site_is_absent() groups operators by the exact vector h.A_i, so two
    operators share a phase group precisely when h annihilates A_i - A_j. Every
    such difference therefore cuts reciprocal space along a sublattice, and the
    grouping -- with it the whole extinction predicate -- changes as h crosses
    one.

    This is why a single bitset over the residue classes cannot state a site's
    absences. In P-4 the position 2g is extinct exactly when l = 0 and h+k is
    odd; N is 2, and (1,0,0) and (1,0,2) are the same residue class with
    opposite answers. The predicate is periodic within a stratum and not across
    strata, so the strata are where the conditions belong -- and they are
    exactly the zones the tables state special-position conditions on.

    Closed under intersection, so every reflection has a unique smallest
    stratum containing it.
    """
    strata = {}
    full = zone_from_rows([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    strata[full["key"]] = full

    seeds = []
    m = len(A_list)
    for i in range(m):
        for j in range(i + 1, m):
            D = [[A_list[i][a][b] - A_list[j][a][b] for b in range(3)]
                 for a in range(3)]
            if not any(any(row) for row in D):
                continue                       # identical: constrains nothing
            rows = integer_kernel_rows(D)
            if not rows:
                continue
            z = zone_from_rows(rows)
            if z is not None and z["dim"] < 3 and z["key"] not in strata:
                strata[z["key"]] = z
                seeds.append(z)

    frontier = seeds
    while frontier:
        nxt = []
        for a in frontier:
            for b in list(strata.values()):
                rows = _kernel_of_normals(a["normals"] + b["normals"])
                if not rows:
                    continue
                z = zone_from_rows(rows)
                if z is None or z["key"] in strata:
                    continue
                strata[z["key"]] = z
                nxt.append(z)
        frontier = nxt

    return sorted(strata.values(), key=lambda z: -z["dim"])


def _stratum_points(S, L, smaller, zones):
    """One reflection per residue class of the stratum, generic within it.

    A representative that happens to fall into a smaller stratum would show that
    stratum's grouping instead of this one's, so it is lifted by whole periods
    until it is clear of them. A class every one of whose lifts stays inside a
    smaller stratum is not this stratum's business at all, and is dropped -- the
    smaller stratum states it.

    Reflections the space group already extinguishes are skipped for the same
    reason the tables skip them: a special position is listed with the
    conditions it adds, not with the ones it inherits. Skipping them per class
    would be wrong -- zone membership is not periodic, so a class can have an
    extinguished representative and ordinary members -- so it is again the lift
    that moves, not the class that is dropped.
    """
    basis, d = S["basis"], S["dim"]
    out = []
    smaller_normals = [z["normals"] for z in smaller]
    shifts = [(0,) * d] + [s for s in itertools.product((0, 1, 2), repeat=d) if s != (0,) * d]

    for coeffs in itertools.product(range(L), repeat=d):
        base = [sum(coeffs[i] * basis[i][c] for i in range(d)) for c in range(3)]
        for shift in shifts:
            r = (base[0] + L * sum(shift[i] * basis[i][0] for i in range(d)),
                 base[1] + L * sum(shift[i] * basis[i][1] for i in range(d)),
                 base[2] + L * sum(shift[i] * basis[i][2] for i in range(d)))
            if r == (0, 0, 0):
                continue
            if any(all(n[0] * r[0] + n[1] * r[1] + n[2] * r[2] == 0 for n in norm) for norm in smaller_normals):
                continue
            if reflection_is_absent(r, zones):
                continue
            out.append((coeffs, r))
            break
    return out


def _rule_literal(var, modulus, residue, negated=False):
    """Format one modular literal in the compact site-condition grammar."""
    residue %= modulus
    rhs = f"{modulus}n" if residue == 0 else f"{modulus}n+{residue}"
    return f"{var}{'!=' if negated else '='}{rhs}"


def _cube_matches(point, cube):
    """Return True when a Miller-index triple lies inside a modular cube."""
    for value, item in zip(point, cube):
        if item is None:
            continue
        modulus, residue = item
        if value % modulus != residue:
            return False
    return True


def _fit_site_cnf(hold, brk, period):
    """Exact modular CNF fallback for compact-rule failures.

    The existing JSON grammar is a conjunction of rule strings, where each
    string is a disjunction of modular clauses.  That is already a CNF
    language, so no consumer-side grammar change is required.

    For a small-period site predicate, we construct modular "forbidden cubes"
    containing only absent reflections.  The complement of each cube is one
    valid disjunctive rule.  Taking the conjunction of those rules gives an
    exact description of every allowed reflection.

    The exact per-coordinate residue classes are always available at the
    finest modulus, while divisors of the period provide coarser cubes and
    therefore shorter rules.  This fallback is deliberately limited to
    period <= 8; larger cases retain the lossless strata-bitset fallback.
    """
    if not brk or not hold or period <= 0 or period > 8:
        return None

    hold = {tuple(int(v) for v in r) for r in hold}
    brk = {tuple(int(v) for v in r) for r in brk}

    # A cube is safe when every point it contains is absent.  Its complement
    # is a clause that every allowed reflection satisfies.
    divisors = _divisors(period)
    options = [None] + [(m, r) for m in divisors for r in range(m)]

    cubes = []
    for cube in itertools.product(options, repeat=3):
        covered = {r for r in brk if _cube_matches(r, cube)}
        if not covered:
            continue
        if any(_cube_matches(r, cube) for r in hold):
            continue
        cubes.append((cube, covered))

    if not cubes:
        return None

    uncovered = set(brk)
    picked = []
    while uncovered:
        best = None
        best_gain = set()
        best_key = None
        for cube, covered in cubes:
            gain = covered & uncovered
            if not gain:
                continue
            key = (
                len(gain),
                -sum(1 for item in cube if item is not None),
                -sum(item[0] for item in cube if item is not None),
                tuple(str(item) for item in cube),
            )
            if best is None or key > best_key:
                best = cube
                best_gain = gain
                best_key = key
        if best is None:
            return None
        picked.append(best)
        uncovered -= best_gain

    clauses = []
    for cube in picked:
        literals = []
        for var, item in zip(LETTERS, cube):
            if item is None:
                continue
            modulus, residue = item
            # Complement of "var = modulus*n + residue".
            literals.append(_rule_literal(var, modulus, residue, negated=True))
        if not literals:
            return None
        clauses.append(" or ".join(literals))

    # Independent verification against the complete residue-class sample.
    rule_fns = [_compile_rule(rule) for rule in clauses]
    if any(fn is None for fn in rule_fns):
        return None
    if not all(all(fn(*r) for fn in rule_fns) for r in hold):
        return None
    if any(all(fn(*r) for fn in rule_fns) for r in brk):
        return None
    return clauses


def _divisors(n):
    """Positive divisors of n, sorted from coarse to fine."""
    out = []
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            out.append(d)
            if d * d != n:
                out.append(n // d)
    return sorted(out)


def _fit_site_rules(hold, brk, period=None):
    """Rules holding on every point of `hold` and broken by every point of `brk`.

    The primary fitter uses the compact single-clause vocabulary.  If that
    vocabulary cannot describe the exact set, a small-period CNF fallback
    extends the vocabulary with modular residue classes while preserving the
    same consumer grammar: one condition string is an OR-clause, and the list
    of strings is ANDed.

    This is important for SG 220 I-43d 12a/12b, whose exact site predicate
    needs several interacting congruences rather than one of the short
    presentation rules.
    """
    if not brk:
        return []
    atoms = site_atomic_rules()

    holds_all, chosen, remaining = [], [], set(brk)
    for name, fn in atoms:
        try:
            if all(fn(*r) for r in hold):
                holds_all.append((name, fn))
        except Exception:
            pass
    for name, fn in sorted(holds_all, key=lambda nf: (len(nf[0]), nf[0])):
        killed = {r for r in remaining if not fn(*r)}
        if killed:
            chosen.append(name)
            remaining -= killed
            if not remaining:
                return chosen

    clauses = []
    for name, fn in atoms:
        try:
            if not any(fn(*r) for r in remaining):
                clauses.append((name, fn))
        except Exception:
            pass
    uncovered, picked = set(hold), []
    while uncovered and clauses:
        best, gain = None, set()
        for name, fn in clauses:
            g = {r for r in uncovered if fn(*r)}
            if len(g) > len(gain):
                best, gain = name, g
        if not gain:
            break
        picked.append(best)
        uncovered -= gain
        clauses = [c for c in clauses if c[0] != best]
    if not uncovered and picked:
        return chosen + [" or ".join(sorted(picked, key=lambda s: (len(s), s)))]

    if period is not None:
        exact_rules = _fit_site_cnf(hold, brk, period)
        if exact_rules:
            return chosen + exact_rules
    return None


def _stratum_bitset(S, L, classes):
    """The stratum's absence set, exactly, as one bit per residue class."""
    total = L ** S["dim"]
    bits = [False] * total
    for coeffs, absent in classes:
        if not absent:
            continue
        idx = 0
        for c in coeffs:
            idx = idx * L + (c % L)
        bits[idx] = True
    return OrderedDict([
        ("modulus", L),
        ("normals", [list(n) for n in S["normals"]]),
        ("duals", [list(row) for row in S["duals"]]),
        ("dim", S["dim"]),
        ("label", S["label"]),
        ("data", _bitset_b64(bits)),
    ])


def site_reflection_conditions(sg, zones, orbit_of, coset_ops,
                               P_num, P_den, T_num, T_den, rng=None):
    """Extra reflection conditions carried by an atom on this position.

    Derived rather than fitted. The strata below carry the whole of the site's
    behaviour, the predicate is periodic within each one, and every residue
    class of every stratum is evaluated -- so the conditions returned are exact,
    not a fit that happened to survive a box of test reflections.

    Returns (conditions, complete, exact, zone_defs). `exact` is present only
    where the compact grammar could not name a stratum's absence set, and then
    it states that stratum exactly.
    """
    N, A_list, w_list = site_absence_machinery(
        sg, coset_ops, P_num, P_den, T_num, T_den)

    M = site_rule_modulus()
    L = N * M // gcd(N, M)

    strata = site_strata(A_list)
    table = {}
    for S in strata:
        smaller = [z for z in strata if z["dim"] < S["dim"]]
        pts = _stratum_points(S, L, smaller, zones)
        table[S["key"]] = [(c, r, site_is_absent(r, N, A_list, w_list))
                           for c, r in pts]

    if not any(a for rows in table.values() for _c, _r, a in rows):
        return OrderedDict(), True, None, {}

    def contained(T, S):
        return all(in_zone(b, S) for b in T["basis"])

    out = OrderedDict()
    defs = {}
    accepted = []
    exact_parts = []

    for S in strata:                              # largest first
        rows = table[S["key"]]
        if not rows:
            continue

        # A rule stated here also governs every reflection of every stratum
        # inside this one, so those are what it must not wrongly forbid.
        hold = [r for T in strata if contained(T, S)
                for _c, r, a in table[T["key"]] if not a]

        def already(r):
            return any(not evaluate_rule(r[0], r[1], r[2], rule)
                       for zz, rules in accepted if in_zone(r, zz) for rule in rules)

        brk = [r for _c, r, a in rows if a and not already(r)]
        if not brk:
            continue

        rules = _fit_site_rules(hold, brk, N)
        if rules:
            out[S["label"]] = rules
            defs[S["label"]] = [list(n) for n in S["normals"]]
            accepted.append((S, rules))
        else:
            exact_parts.append(_stratum_bitset(S, L, [(c, a) for c, _r, a in rows]))

    if exact_parts:
        exact = OrderedDict([
            ("encoding", "strata-bitsets"),
            ("modulus", L),
            ("index", "fold the stratum coordinates c_i = h.duals[:,i] mod N"),
            # Bits are set only for reflections the space group itself allows,
            # so a reader applies the general conditions separately, as before.
            ("allowed_by_space_group", True),
            ("strata", list(reversed(exact_parts))),   # most special first
        ])
        return out, False, exact, defs
    return out, True, None, defs

def evaluate_rule(h, k, l, rule_str):
    """Evaluates if a reflection satisfies a textual rule safely.

    The grammar is
        clause      := <integer expression in h,k,l> '=' <n>'n' ['+' <r>]
        rule        := clause (' or ' clause)*
    "h+k=2n" reads as before; "h=2n+1 or h+k+l=4n" is the disjunctive form the
    tables use for the diamond-type special positions.

    Each distinct rule is compiled once and kept. This used to re-run a regular
    expression and an eval() per reflection per rule, which the verification
    boxes call millions of times; the rules themselves number in the hundreds.
    A rule that will not parse is false, as before, rather than an exception.
    """
    fn = _compiled_rule(str(rule_str))
    if fn is None:
        return False
    try:
        return bool(fn(h, k, l))
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def _compiled_rule(rule_str):
    try:
        return _compile_rule(rule_str)
    except Exception:
        return None


def _compile_rule(rule_str):
    """The same grammar as one compiled lambda, for the candidate search.

    Rule strings come from the tables below, never from input, and the search
    evaluates each of them against every allowed reflection -- some tens of
    thousands of calls per setting, which is far too many to go through eval().
    """
    parts = []
    for clause in rule_str.split(" or "):
        negated = "!=" in clause
        lhs, rhs = clause.split("!=" if negated else "=", 1)
        m = re.fullmatch(r"\s*(\d+)n\s*(?:\+\s*(\d+)\s*)?", rhs)
        mod, rem = int(m.group(1)), int(m.group(2) or 0)
        op = "!=" if negated else "=="
        parts.append(f"((({lhs}) - {rem}) % {mod} {op} 0)")
    return eval("lambda h, k, l: " + " or ".join(parts))


def get_wyckoff_table(sg_info, zone_records=None, zones=None, orbit_of=None):
    """Wyckoff positions, with the coordinates and conditions the tables print.

    Beyond letter / multiplicity / site symmetry / special_op this emits the
    projector as exact rationals, the coset operators, the ITA parametrisation
    and coordinate list, and any reflection conditions the position carries.

    Every added field is guarded on its own: a cctbx build whose Wyckoff objects
    lack an accessor loses that field rather than the position, and a build with
    no Wyckoff table at all still yields a complete database minus the letters.
    """
    positions = []
    table = _get_wyckoff_table_object(sg_info)
    if table is None:
        return positions

    sg = sg_info.group()

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    for i in range(table.size()):
        try:
            pos = table.position(i)
        except Exception:
            continue

        site_sym = _safe(lambda: str(pos.point_group_type()), None)
        if site_sym is None:
            site_sym = _safe(lambda: str(pos.site_symmetry_group().point_group_type()), "?")

        mult = _safe(lambda: int(pos.multiplicity()), 0)
        entry = OrderedDict([
            ("letter", _safe(lambda: pos.letter(), "?")),
            ("multiplicity", mult),
            ("site_symmetry", site_sym),
            ("special_op", _safe(lambda: str(pos.special_op()), "")),
        ])

        sop = _safe(lambda: pos.special_op(), None)
        if sop is None:
            positions.append(entry)
            continue

        try:
            r, t = sop.r(), sop.t()
            p_num = [int(v) for v in r.num()]
            p_den = int(r.den())
            t_num = [int(v) for v in t.num()]
            t_den = int(t.den())
            entry["P_num"] = p_num
            entry["P_den"] = p_den
            entry["T_num"] = t_num
            entry["T_den"] = t_den
            entry["n_free"] = _rank3(p_num, p_den)
        except Exception as e:
            _warn_once("wyckoff special_op matrix", e)
            positions.append(entry)
            continue

        # v5: the ITA parametrisation, so the coordinate column reads "x, x, z"
        # rather than the projector "1/2*x+1/2*y, 1/2*x+1/2*y, z".
        try:
            names, Q, T = ita_parametrisation(p_num, p_den, t_num, t_den)
            entry["parameters"] = names
        except Exception as e:
            _warn_once("wyckoff parametrisation", e)
            names = None

        try:
            ops, exact = _coset_ops(sg, sop, mult)
            entry["coset_ops"] = ops
            if not exact:
                entry["coset_exact"] = False
        except Exception as e:
            _warn_once("wyckoff coset operators", e)
            ops, exact = None, False

        if names is not None and ops:
            try:
                coords = ita_coordinates(sg, ops, names, Q, T)
                entry["coordinates"] = coords
                entry["coordinate"] = coords[0] if coords else ""
            except Exception as e:
                _warn_once("wyckoff coordinate list", e)

        # The general position reproduces the group's own conditions and can
        # carry nothing extra, so it is not worth the work.
        if exact and ops and zones is not None and entry.get("n_free", 3) < 3:
            try:
                cond, named, exact_site, site_defs = site_reflection_conditions(
                    sg, zones, orbit_of, ops, p_num, p_den, t_num, t_den)
                if cond:
                    entry["conditions"] = cond
                if site_defs:
                    # A stratum need not be one of the group's own zones, so its
                    # normals travel with the position or the app cannot test it.
                    entry["condition_zones"] = site_defs
                if exact_site is not None:
                    entry["conditions_named"] = False
                    entry["conditions_exact"] = exact_site
            except Exception as e:
                _warn_once("wyckoff site conditions", e)

        positions.append(entry)

    return positions



def get_harker_geometry(sg_info):
    sg = sg_info.group()
    sections = []
    coords = ['u', 'v', 'w']
    vars_str = ['x', 'y', 'z']

    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Safely iterate through all symmetry operations
    for i in range(1, sg.order_z()):
        op = sg(i)
        r = op.r().as_double()
        t = op.t().as_double()

        # v4: a pure centring translation has R = I, so R - I is all zeros and
        # every row counts as a "zero row". That used to emit three sections at
        # the centring vector itself (C2/c: u=1/2, w=0, v=0), which are lattice
        # translations, not Harker geometry.
        if all(abs(a - b) < 1e-5 for a, b in zip(r, identity)):
            continue

        # Construct R - I matrix
        r_minus_i = [
            [r[0]-1, r[1],   r[2]  ],
            [r[3],   r[4]-1, r[5]  ],
            [r[6],   r[7],   r[8]-1]
        ]

        # Identify zero rows (which define the Harker planes/lines)
        zero_rows = [idx for idx in range(3) if all(abs(val) < 1e-5 for val in r_minus_i[idx])]
        if not zero_rows:
            continue

        # Identify the submatrix to invert
        nz_rows = [idx for idx in range(3) if idx not in zero_rows]
        nz_cols = [idx for idx in range(3) if any(abs(r_minus_i[r_idx][idx]) > 1e-5 for r_idx in range(3))]

        solver = {"x": "?", "y": "?", "z": "?"}

        # Algebraically invert the submatrix to solve for x, y, z
        if len(nz_rows) == len(nz_cols) and len(nz_rows) > 0:
            n = len(nz_rows)
            A = [[r_minus_i[row][col] for col in nz_cols] for row in nz_rows]
            invA = None

            if n == 1:
                if abs(A[0][0]) > 1e-5:
                    invA = [[1.0 / A[0][0]]]
            elif n == 2:
                det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
                if abs(det) > 1e-5:
                    invA = [[A[1][1]/det, -A[0][1]/det], [-A[1][0]/det, A[0][0]/det]]
            elif n == 3:
                det = (A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                     - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                     + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]))
                if abs(det) > 1e-5:
                    invA = [
                        [(A[1][1]*A[2][2] - A[1][2]*A[2][1])/det, (A[0][2]*A[2][1] - A[0][1]*A[2][2])/det, (A[0][1]*A[1][2] - A[0][2]*A[1][1])/det],
                        [(A[1][2]*A[2][0] - A[1][0]*A[2][2])/det, (A[0][0]*A[2][2] - A[0][2]*A[2][0])/det, (A[0][2]*A[1][0] - A[0][0]*A[1][2])/det],
                        [(A[1][0]*A[2][1] - A[1][1]*A[2][0])/det, (A[0][1]*A[2][0] - A[0][0]*A[2][1])/det, (A[0][0]*A[1][1] - A[0][1]*A[1][0])/det]
                    ]

            # Format the output strings for the JS evaluator
            if invA:
                for c_idx, col_name in enumerate(nz_cols):
                    terms = []
                    for r_idx, row_name in enumerate(nz_rows):
                        coeff = invA[c_idx][r_idx]
                        if abs(coeff) > 1e-5:
                            terms.append(f"{coeff:g}*({coords[row_name]}-{t[row_name]:g})")
                    if terms:
                        solver[vars_str[col_name]] = " + ".join(terms).replace("+ -", "- ")

        sec_type = "plane" if len(zero_rows) == 1 else "line"

        for z_idx in zero_rows:
            val = t[z_idx] % 1.0
            if not any(s['coordinate'] == coords[z_idx] and abs(s['value'] - val) < 1e-5 for s in sections):
                sections.append({
                    "type": sec_type,
                    "coordinate": coords[z_idx],
                    "value": val,
                    "solver": solver
                })

    return sections


# Reflections where our derivation and cctbx's is_sys_absent disagree. Should
# stay empty; anything in it is a bug, not a tolerance.
CONDITION_MISMATCHES = []

# Accessors already reported as missing, so each is mentioned once.
_WARNED = set()


def _warn_once(what, err):
    """Report an unavailable cctbx accessor once, not 700 times."""
    key = f"{what}:{type(err).__name__}:{err}"
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"\n[!] {what} unavailable in this cctbx build ({err}). "
          f"That field will be empty; everything else is still written.")
    sys.stdout.flush()


def generate_all_space_groups(only=None):
    all_data = defaultdict(lambda: {
        "number": 0, "standard_symbol": "", "crystal_system": "",
        "point_group": "", "laue_class": "", "centrosymmetric": False,
        "chiral": False, "settings": []
    })

    iterator = sgtbx.space_group_symbol_iterator()
    processed_settings = defaultdict(set)

    print("Processing settings...")
    count = 0

    while True:
        try:
            symbols = iterator.next()
        except StopIteration:
            break

        if symbols.number() == 0:
            break

        # v4: bound before the try, because the handler prints it. A failure on
        # the first setting used to raise NameError inside the except clause.
        sg_num = symbols.number()

        try:
            # --only: skip everything else before any real work is done.
            if only is not None and sg_num not in only:
                continue
            hm_symbol = symbols.hermann_mauguin()
            ext = str(symbols.extension()).strip().replace('\0', '')
            if ext:
                hm_symbol = f"{hm_symbol}:{ext}"

            hall = symbols.hall().strip()

            uid = hall
            if uid in processed_settings[sg_num]:
                continue
            processed_settings[sg_num].add(uid)

            sg_info = sgtbx.space_group_info(symbol=f"Hall: {hall}")
            sg = sg_info.group()

            if all_data[str(sg_num)]["number"] == 0:
                all_data[str(sg_num)]["number"] = sg_num
                all_data[str(sg_num)]["standard_symbol"] = sg_info.type().lookup_symbol()
                all_data[str(sg_num)]["crystal_system"] = str(sg.crystal_system()).lower()
                all_data[str(sg_num)]["point_group"] = str(sg.point_group_type())
                all_data[str(sg_num)]["centrosymmetric"] = sg.is_centric()
                # NEW in v7 -- powder5.html already reads currentSG.laue_class
                try:
                    all_data[str(sg_num)]["laue_class"] = str(sg.laue_group_type())
                except Exception as e:
                    _warn_once("laue_group_type", e)
                    all_data[str(sg_num)]["laue_class"] = ""
                try:
                    all_data[str(sg_num)]["chiral"] = bool(sg.is_chiral())
                except Exception:
                    all_data[str(sg_num)]["chiral"] = False

            conditions, zone_records, zone_universe_, orbit_of = \
                exact_reflection_conditions(sg_info.group())
            zone_defs = zone_definitions(zone_universe_)

            # The conditions were derived from the operators; cctbx decides
            # absence independently. If the two ever disagree the derivation is
            # wrong, and that must be loud rather than silent.
            mismatch = check_conditions_against_cctbx(sg_info.group(), zone_records, rng=4)
            if mismatch:
                CONDITION_MISMATCHES.append((hm_symbol, len(mismatch), mismatch[:3]))
            harker_data = get_harker_geometry(sg_info)
            rotations = get_rotations(sg_info)

            # --- NEW in v7 ---
            # Each of these is isolated. The enclosing try/except skips the
            # WHOLE setting on any exception, so an API difference in one
            # accessor used to silently drop every space group in the file
            # (all 230 of them, leaving an empty JSON). A failure here now
            # costs only the field it belongs to.
            try:
                symops = get_symmetry_operations(sg_info)
            except Exception as e:
                _warn_once("sym_ops", e)
                symops = []
            try:
                centring_letter, centring_ltr = get_centring(sg_info)
            except Exception as e:
                _warn_once("centring", e)
                centring_letter, centring_ltr = "?", []
            try:
                wyckoff = get_wyckoff_table(sg_info, zone_records,
                                            zone_universe_, orbit_of)
                # A site condition can sit on a stratum that is not one of the
                # group's own zones. Its normals go into zone_defs so a reader
                # can decide membership arithmetically instead of guessing from
                # the label.
                for w in wyckoff:
                    for label, normals in (w.pop("condition_zones", None) or {}).items():
                        zone_defs.setdefault(label, normals)
            except Exception as e:
                _warn_once("wyckoff", e)
                wyckoff = []
            try:
                order_z = int(sg.order_z())
            except Exception:
                order_z = len(symops)
            try:
                order_p = int(sg.order_p())
            except Exception:
                order_p = 0

            desc = symbols.qualifier() if symbols.qualifier() else "standard"
            if ext == '1' and desc == "standard":
                desc = "origin choice 1"
            elif ext == '2' and desc == "standard":
                desc = "origin choice 2"

            clean_sym = hm_symbol.replace(" ", "")

            # v4: keep the spaced form too. "P6522" is ambiguous to a typesetter
            # -- 65 is a screw axis and wants a subscript, the two 2s do not --
            # and no rule recovers that from the compact string. "P 65 2 2" does.
            spaced_sym = " ".join(hm_symbol.split())

            all_data[str(sg_num)]["settings"].append({
                "symbol": clean_sym,
                "hm": spaced_sym,
                "description": desc,
                "hall": hall,
                "reflection_conditions": conditions,
                "reflection_zones": zone_records,
                "zone_defs": zone_defs,
                "harker_sections": harker_data,
                "rotations": rotations,
                # --- NEW in v7 ---
                "sym_ops": symops,
                "order_z": order_z,
                "order_p": order_p,
                "centering": centring_letter,
                "centring_translations": centring_ltr,
                "wyckoff": wyckoff,
            })

            count += 1
            if count % 50 == 0:
                print(f"Processed {count} settings...", end="\r")
                sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error processing SG {sg_num}: {e}")
            sys.stdout.flush()
            continue

    return all_data


# 11: settings carry zone_defs, zone records name their orbit, the index entry
#     carries standard_symbol, and space_groups[n]["settings"] holds setting
#     numbers rather than a second copy of every entry.
# 12: site conditions are derived from a sample that is fair across zones, and
#     conditions_exact is now the site predicate alone (allowed_by_space_group
#     is false), so a consumer applies the space-group conditions separately.
# 13: site conditions are derived on the strata where the operator grouping is
#     constant, exactly and without sampling; conditions_exact, when needed at
#     all, is a strata-bitset rather than one flat bitset over residue classes.
SCHEMA_VERSION = 14
SG_DIR_NAME = 'sg'
INDEX_NAME = 'index.json'
EXPECTED_SETTINGS = 527

def resolve_output_dir(explicit):
    """Where to write, stated in absolute terms."""
    if explicit:
        target = os.path.abspath(explicit)
        os.makedirs(target, exist_ok=True)
        print(f"Output folder given on the command line: {target}")
        return target

    here = os.path.abspath(os.getcwd())
    script_dir = os.path.abspath(os.path.dirname(__file__))
    for folder in (here, script_dir, os.path.dirname(script_dir), os.path.dirname(here)):
        if not folder:
            continue
        # The Space Group Explorer is recognised by index.html + app.js.
        if os.path.isfile(os.path.join(folder, 'index.html')) and os.path.isfile(os.path.join(folder, 'app.js')):
            if os.path.abspath(folder) != here:
                print(f"\n[i] index.html found in {folder}")
                print("    Writing there, not into the current folder, because the app")
                print("    fetches with a relative path from its own location.")
            return folder

    print("\n[!] No app page (index.html + app.js) was found nearby, so the output")
    print(f"    goes in the current folder ({here}). If the browser reports a 404,")
    print("    move the sg/ folder next to index.html, or re-run as:")
    print(f"        python {os.path.basename(__file__)} --out <folder-with-index.html>")
    return here


def write_split_database(sorted_data, out_dir, pretty=False):
    """Write the database as one JSON file per SETTING, plus indexes.

    There are 527 crystallographic settings in the cctbx iterator.  Each setting
    gets its own file directly under sg/ (setting_0001.json ...), so a consumer
    can fetch exactly one setting without loading the other 526.

    Each setting file contains:
      - full symmetry operators and centring translations
      - reflection_conditions for the space group
      - the complete Wyckoff table, including multiplicity, site symmetry,
        special operator/projection data, free-coordinate count and coset ops
      - an explicit special flag for Wyckoff positions (n_free < 3)

    A group-level index is also retained.
    """
    sg_dir = os.path.join(out_dir, SG_DIR_NAME)
    os.makedirs(sg_dir, exist_ok=True)

    dump_kw = dict(ensure_ascii=False)
    if pretty:
        dump_kw['indent'] = 2
    else:
        dump_kw['separators'] = (',', ':')

    index = OrderedDict()
    index["schema_version"] = SCHEMA_VERSION
    index["generated_by"] = os.path.basename(__file__)
    index["setting_count"] = 0
    index["settings"] = []

    group_index = OrderedDict()
    total_setting_bytes = 0
    biggest = ("", 0)
    setting_no = 0

    for num, entry in sorted_data.items():
        group_index[num] = OrderedDict([
            ("number", entry["number"]),
            ("standard_symbol", entry["standard_symbol"]),
            ("crystal_system", entry["crystal_system"]),
            ("point_group", entry["point_group"]),
            ("laue_class", entry.get("laue_class", "")),
            ("centrosymmetric", entry["centrosymmetric"]),
            ("chiral", entry.get("chiral", False)),
            # setting_number values, to be looked up in index["settings"].
            ("settings", [])
        ])

        for setting_idx, setting in enumerate(entry["settings"], start=1):
            setting_no += 1

            wyckoff = []
            for w in setting.get("wyckoff", []):
                ww = OrderedDict(w)
                ww["special"] = bool(ww.get("n_free", 3) < 3)
                wyckoff.append(ww)

            setting_payload = OrderedDict([
                ("schema_version", SCHEMA_VERSION),
                ("setting_id", f"{int(num):03d}_{setting_idx:02d}"),
                ("setting_number", setting_no),
                ("number", entry["number"]),
                ("standard_symbol", entry["standard_symbol"]),
                ("symbol", setting["symbol"]),
                ("hm", setting.get("hm", setting["symbol"])),
                ("description", setting["description"]),
                ("hall", setting["hall"]),
                ("crystal_system", entry["crystal_system"]),
                ("point_group", entry["point_group"]),
                ("laue_class", entry.get("laue_class", "")),
                ("centrosymmetric", entry["centrosymmetric"]),
                ("chiral", entry.get("chiral", False)),
                ("centering", setting.get("centering", "?")),
                ("centring_translations", setting.get("centring_translations", [])),
                ("order_z", setting.get("order_z", 0)),
                ("order_p", setting.get("order_p", 0)),
                ("reflection_conditions", setting.get("reflection_conditions", {})),
                ("reflection_zones", setting.get("reflection_zones", [])),
                ("zone_defs", setting.get("zone_defs", {})),
                ("wyckoff", wyckoff),
                ("harker_sections", setting.get("harker_sections", [])),
                ("rotations", setting.get("rotations", [])),
                ("sym_ops", setting.get("sym_ops", [])),
            ])

            filename = f"setting_{setting_no:04d}.json"
            path = os.path.join(sg_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(setting_payload, f, **dump_kw)

            size = os.path.getsize(path)
            total_setting_bytes += size
            if size > biggest[1]:
                biggest = (filename, size)

            idx_entry = OrderedDict([
                ("setting_id", setting_payload["setting_id"]),
                ("setting_number", setting_no),
                ("number", entry["number"]),
                # The app searches this field. Leaving it out of the index does
                # not merely lose a search key: joining an absent field into a
                # haystack string yields the text "undefined", which then
                # matches any query that is a substring of that word.
                ("standard_symbol", entry["standard_symbol"]),
                ("symbol", setting["symbol"]),
                ("hm", setting.get("hm", setting["symbol"])),
                ("description", setting["description"]),
                ("hall", setting["hall"]),
                ("crystal_system", entry["crystal_system"]),
                ("point_group", entry["point_group"]),
                ("laue_class", entry.get("laue_class", "")),
                ("centering", setting.get("centering", "?")),
                ("order_z", setting.get("order_z", 0)),
                ("n_wyckoff", len(wyckoff)),
                ("n_special_wyckoff", sum(1 for w in wyckoff if w.get("special"))),
                ("file", f"{SG_DIR_NAME}/{filename}"),
            ])
            index["settings"].append(idx_entry)
            # Only the setting numbers: the entries themselves are already in
            # index["settings"], and repeating each one here doubled the file.
            group_index[num]["settings"].append(setting_no)

    index["setting_count"] = setting_no
    index["space_groups"] = group_index

    index_path = os.path.join(sg_dir, INDEX_NAME)
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {setting_no} setting files to {sg_dir}")
    print(f"  settings seen in previous runs of this script: {EXPECTED_SETTINGS}")
    if setting_no != EXPECTED_SETTINGS:
        print(f"  [i] this run produced {setting_no}. That is a difference in the "
              f"cctbx symbol table, not necessarily a fault.")
    print(f"  setting files total {total_setting_bytes/1024:.0f} KB, "
          f"largest {biggest[0]} at {biggest[1]/1024:.0f} KB")
    print(f"  index {os.path.getsize(index_path)/1024:.0f} KB ({INDEX_NAME})")
    return sg_dir


def _machinery_from_data(ops, coset_ops, P_num, P_den, T_num, T_den):
    """site_absence_machinery() rebuilt from the written record alone.

    Deliberately reads sym_ops / coset_ops / the projector out of the payload
    rather than asking cctbx again, so the check below tests what was actually
    serialised. A position whose stored operators no longer reproduce its stored
    conditions is caught here, not by a reader months later.
    """
    P = _rational_matrix(P_num, P_den)
    T = [Fraction(int(T_num[i]), int(T_den)) for i in range(3)]

    N = 1
    mats, shifts = [], []
    for i in coset_ops:
        op = ops[i]
        R = op["r"]
        t = [Fraction(int(v), int(op["t_den"])) for v in op["t_num"]]
        RP = [[sum(Fraction(R[3 * a + c]) * P[c][b] for c in range(3)) for b in range(3)]
              for a in range(3)]
        w = [sum(Fraction(R[3 * a + c]) * T[c] for c in range(3)) + t[a] for a in range(3)]
        mats.append(RP)
        shifts.append(w)
        for row in RP:
            for v in row:
                N = N * v.denominator // gcd(N, v.denominator)
        for v in w:
            N = N * v.denominator // gcd(N, v.denominator)

    A_list = [[[int(RP[a][b] * N) for b in range(3)] for a in range(3)] for RP in mats]
    w_list = [[int(v * N) for v in w] for w in shifts]
    return N, A_list, w_list


def check_site_conditions(setting, rng=SITE_VERIFY_BOX + 2):
    """Do a position's stated conditions match what its orbit actually does?

    site_is_absent() is exact, so every compact rule the generator emits can be
    held against it. The box below is an ordinary Cartesian one, chosen because
    it is *not* the sample the derivation used: a rule fitted to a biased sample
    agrees with that sample by construction, and only an independent set of
    reflections can expose it. Pa-3 4a is the case in point -- it failed at
    (1,1,2), well inside this box, while reporting itself complete.

    Positions carrying conditions_exact are skipped: their bitset is the exact
    predicate, and their compact rules are presentation-only by declaration.
    """
    ops = setting.get("sym_ops") or []
    zone_records = setting.get("reflection_zones") or []
    zone_defs = setting.get("zone_defs") or {}
    if not ops:
        return []

    def normals_for(label):
        if label in zone_defs:
            return zone_defs[label]
        for z in zone_records:
            if z["zone"] == label:
                return z["normals"]
        return None

    bad = []
    for w in setting.get("wyckoff") or []:
        cond = w.get("conditions")
        if not cond or w.get("conditions_named") is False:
            continue
        if "coset_ops" not in w or "P_num" not in w:
            continue
        if w.get("coset_exact") is False:
            continue

        zones_used = {}
        for label in cond:
            n = normals_for(label)
            if n is None:
                bad.append((w["multiplicity"], w["letter"], "unresolvable zone", label))
                break
            zones_used[label] = n
        else:
            try:
                N, A_list, w_list = _machinery_from_data(
                    ops, w["coset_ops"], w["P_num"], w["P_den"],
                    w["T_num"], w["T_den"])
            except Exception as e:
                bad.append((w["multiplicity"], w["letter"], "machinery failed", str(e)))
                continue

            for h in range(-rng, rng + 1):
                for k in range(-rng, rng + 1):
                    for l in range(-rng, rng + 1):
                        r = (h, k, l)
                        if r == (0, 0, 0) or reflection_is_absent(r, zone_records):
                            continue
                        said = False
                        for label, rules in cond.items():
                            if not all(n[0] * h + n[1] * k + n[2] * l == 0
                                       for n in zones_used[label]):
                                continue
                            if any(not evaluate_rule(h, k, l, rule) for rule in rules):
                                said = True
                                break
                        if said != site_is_absent(r, N, A_list, w_list):
                            bad.append((w["multiplicity"], w["letter"],
                                        "wrongly absent" if said else "absence missed", r))
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    return bad


def verify(sorted_data):
    """Operator lists must be closed groups; coset counts must match."""
    print("\nVerifying operator lists form closed groups...")
    bad = checked = missing_symops = missing_wyckoff = 0
    coset_bad = []
    nfree_missing = 0

    for num, entry in sorted_data.items():
        for setting in entry["settings"]:
            ops = setting["sym_ops"]
            wy = setting.get("wyckoff") or []
            if not wy:
                missing_wyckoff += 1
            for w in wy:
                if w.get("coset_exact") is False or (
                        "coset_ops" in w and len(w["coset_ops"]) != w["multiplicity"]):
                    coset_bad.append(f"SG {num} {setting['symbol']} "
                                     f"{w['multiplicity']}{w['letter']}")
                if "n_free" not in w:
                    nfree_missing += 1
            if not ops:
                missing_symops += 1
                continue
            if len(ops) != setting["order_z"]:
                print(f"  [!] SG {num} {setting['symbol']}: {len(ops)} ops but order_z={setting['order_z']}")
                bad += 1
                continue
            keys = set()
            for op in ops:
                keys.add((tuple(op["r"]), tuple(Fraction(n, op["t_den"]) % 1 for n in op["t_num"])))
            if len(keys) != len(ops):
                print(f"  [!] SG {num} {setting['symbol']}: duplicate operators")
                bad += 1
                continue
            closed = True
            for a in ops:
                for b in ops:
                    ra, ta, da = a["r"], a["t_num"], a["t_den"]
                    rb, tb, db = b["r"], b["t_num"], b["t_den"]
                    rc = [sum(ra[3 * i + k] * rb[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)]
                    tc = [Fraction(ta[i], da) + sum(Fraction(ra[3 * i + k] * tb[k], db) for k in range(3))
                          for i in range(3)]
                    if (tuple(rc), tuple(t % 1 for t in tc)) not in keys:
                        closed = False
                        break
                if not closed:
                    break
            if not closed:
                print(f"  [!] SG {num} {setting['symbol']}: operator list is not closed")
                bad += 1
            checked += 1

    print(f"Checked {checked} settings, {bad} problem(s).")

    # The derivation of the reflection conditions, against cctbx's own verdict.
    if CONDITION_MISMATCHES:
        print(f"\n[!!] {len(CONDITION_MISMATCHES)} setting(s) where the derived "
              f"reflection conditions disagree with cctbx is_sys_absent():")
        for sym, n, sample in CONDITION_MISMATCHES[:10]:
            print(f"     {sym}: {n} reflection(s), e.g. {sample}")
        print("     This is a bug in the derivation. Do not publish this data.")
    else:
        print("Reflection conditions agree with cctbx is_sys_absent() "
              "on the strengthened verification box in every setting.")

    unnamed = []
    exact_fallbacks = 0
    site_bad = []
    for num, entry in sorted_data.items():
        for setting in entry["settings"]:
            for w in setting.get("wyckoff") or []:
                if w.get("conditions_named") is False:
                    unnamed.append(f"SG {num} {setting['symbol']} "
                                   f"{w['multiplicity']}{w['letter']}")
                    if "conditions_exact" in w:
                        exact_fallbacks += 1
            for mult, letter, why, detail in check_site_conditions(setting):
                site_bad.append(f"SG {num} {setting['symbol']} "
                                f"{mult}{letter}: {why} at {detail}")

    if site_bad:
        print(f"\n[!!] {len(site_bad)} position(s) whose stated site conditions do "
              f"not match their own orbit:")
        for s in site_bad[:10]:
            print(f"     {s}")
        if len(site_bad) > 10:
            print(f"     ... and {len(site_bad) - 10} more")
        print("     This is a bug in the derivation. Do not publish this data.")
    else:
        print("Site conditions reproduce the exact structure-factor calculation "
              "on every position that claims a compact rule.")

    if unnamed:
        print(f"\n[i] {len(unnamed)} special position(s) whose exact site absences "
              f"could not be reduced to the compact rule grammar.")
        print("    These positions carry an exact periodic residue bitset in "
              "'conditions_exact'; the compact 'conditions' remain presentation-only.")
        for u in unnamed:
            print(f"     {u}")
        if exact_fallbacks:
            print(f"    Exact periodic fallback present for {exact_fallbacks} position(s).")
    if missing_symops:
        print(f"[!] {missing_symops} setting(s) have NO symops -- structure building "
              f"will not work for those. Scroll up for the accessor that failed.")
    if missing_wyckoff:
        print(f"[!] {missing_wyckoff} setting(s) have no Wyckoff table. The Wyckoff "
              f"search cannot run for those; the rest of the app still works.")
    if nfree_missing:
        print(f"[!] {nfree_missing} Wyckoff position(s) have no n_free -- the special_op "
              f"matrix accessor failed. Scroll up for the reason.")
    if coset_bad:
        print(f"[!] {len(coset_bad)} position(s) gave a coset count that disagrees with "
              f"the tabulated multiplicity:")
        for c in coset_bad[:10]:
            print(f"      {c}")
        if len(coset_bad) > 10:
            print(f"      ... and {len(coset_bad) - 10} more")
        print("    Those positions carry coset_exact:false and the consumer will fall")
        print("    back to a distance test for them. Report the list.")
    if not (missing_symops or coset_bad or nfree_missing):
        print("All settings carry a full closed operator list and an exact Wyckoff table.")


def parse_args():
    p = argparse.ArgumentParser(description="Generate the Space Group Explorer database.")
    p.add_argument('--out', default=None,
                   help="output folder (default: the folder containing index.html, else cwd)")
    p.add_argument('--only', default=None,
                   help="comma-separated space-group numbers, e.g. 62,206 - for quick tests")
    p.add_argument('--pretty', action='store_true',
                   help="indent the per-group files (about 50%% larger)")
    p.add_argument('--no-check', action='store_true',
                   help="skip the group-closure verification (it is the slow part)")
    return p.parse_args()


def main():
    args = parse_args()

    only = None
    if args.only:
        only = set(int(x) for x in args.only.replace(' ', '').split(',') if x)

    print("=" * 62)
    print("Space Group Explorer database generator - Wyckoff projections, split output")
    print("=" * 62)
    if only:
        print(f"Restricted to space group(s): {sorted(only)}")

    data = generate_all_space_groups(only=only)

    total_settings = sum(len(x['settings']) for x in data.values())
    print(f"\nFinal processing complete.")
    print(f"Total space group numbers: {len(data)}")
    print(f"Total settings: {total_settings}")
    if total_settings != EXPECTED_SETTINGS:
        print(f"[i] cctbx returned {total_settings} settings; previous runs gave "
              f"{EXPECTED_SETTINGS}. Check the count before publishing the data.")

    sorted_data = OrderedDict()
    for k in sorted(data.keys(), key=int):
        sorted_data[k] = data[k]
        sorted_data[k]["settings"].sort(key=lambda x: x["symbol"])

    out_dir = resolve_output_dir(args.out)
    write_split_database(sorted_data, out_dir, pretty=args.pretty)

    if not args.no_check:
        verify(sorted_data)
    else:
        print("\n[i] Verification skipped (--no-check).")

    print("=" * 62)


if __name__ == "__main__":
    main()
