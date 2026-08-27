#!/usr/bin/env python3
"""Pack sg/setting_*.json into the single database Brutus loads.

This replaces cctbx_space_groups_all_settings_v4.json outright. Nothing in the
app reads the old file any more.

WHY PACK AT ALL
The split layout suits SpaceExplorer, where the user picks one group and that
group's file is fetched. Brutus enumerates every setting to build extinction
classes, so it needs all of them at once and 527 round trips is not an option.

WHAT COMES OUT

  rotations      the distinct rotation matrices, 9 ints row-major. Every
                 operator references one by index; there are only a few dozen
                 across all 527 settings.

  zone_defs      zone label -> list of normal vectors. A reflection is in the
                 zone iff n.h == 0 for every normal. This is what retires the
                 app's ZONE_PREDICATES table, whose 'hhl' test was
                 Math.abs(h) === Math.abs(k) -- which also matches h == -k, the
                 separate 'h-hl' zone with different conditions. The generator
                 distinguishes them and so must the app.

  space_groups   keyed by number, each with a settings list. Per setting:
                   t_den, ops     the operators: [rot_index, t0, t1, t2],
                                  t = t_num / t_den. THESE ARE AUTHORITATIVE
                                  for systematic absences. h is absent iff some
                                  operator has hR = h with h.t non-integral.
                   conditions     the printed reflection conditions, for
                                  display and for the condition-by-condition
                                  evidence hunt in detectExtinctions.
                   zones          every zone carrying a rule, printed or not,
                                  each with its own normals.

WHAT IS DROPPED
wyckoff, harker_sections, the per-setting rotations list and the expanded
sym_ops text. Those are for building a structure and are the bulk of the files.
Centring translations are not stored either: they are exactly the operators
whose rotation is the identity, so the app derives them from ops.

Usage:
    python3 sg_pack.py                        # sg/ -> sg_ops.json
    python3 sg_pack.py --sg-dir sg --out sg_ops.json --pretty
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from math import gcd

GROUP_FIELDS = ("number", "standard_symbol", "crystal_system", "point_group",
                "laue_class", "centrosymmetric", "chiral")

SETTING_FIELDS = ("setting_id", "setting_number", "symbol", "hm", "description",
                  "hall", "centering", "order_z", "order_p")


def lcm(a, b):
    return a * b // gcd(a, b)


def pack_ops(data, rot_table, rot_index):
    ops = data.get("sym_ops") or []
    if not ops:
        return None, None, "no sym_ops"

    den = 1
    for op in ops:
        den = lcm(den, int(op.get("t_den", 1)) or 1)

    packed = []
    for op in ops:
        r = op.get("r")
        if not r or len(r) != 9:
            return None, None, "bad rotation"
        key = tuple(int(v) for v in r)
        idx = rot_index.get(key)
        if idx is None:
            idx = len(rot_table)
            rot_index[key] = idx
            rot_table.append(list(key))
        td = int(op.get("t_den", 1)) or 1
        scale = den // td
        tn = [(int(v) * scale) % den for v in op.get("t_num", [0, 0, 0])]
        packed.append([idx] + tn)

    return packed, den, None


def norm_normals(vecs):
    """Canonical, hashable form for a zone's normal set."""
    return tuple(sorted(tuple(int(c) for c in v) for v in (vecs or [])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg-dir", default="sg")
    ap.add_argument("--out", default="sg_ops.json")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.sg_dir):
        sys.exit(f"no such directory: {args.sg_dir}")

    files = sorted(f for f in os.listdir(args.sg_dir)
                   if f.startswith("setting_") and f.endswith(".json"))
    if not files:
        sys.exit(f"no setting_*.json in {args.sg_dir}")

    rot_table, rot_index = [], {}
    zone_defs = {}
    zone_clashes = []
    groups = OrderedDict()
    skipped = []
    schema = None
    n_settings = n_ops = n_zone_records = 0

    for fn in files:
        path = os.path.join(args.sg_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            skipped.append((fn, f"unreadable: {exc}"))
            continue

        schema = schema or data.get("schema_version")
        num = data.get("number")
        if num is None:
            skipped.append((fn, "no space-group number"))
            continue

        packed, den, err = pack_ops(data, rot_table, rot_index)
        if packed is None:
            skipped.append((fn, err))
            continue

        # Zone label -> normals, merged across settings. A label must mean the
        # same thing everywhere or the shared table is a lie, so clashes are
        # reported rather than silently overwritten.
        for label, normals in (data.get("zone_defs") or {}).items():
            canon = norm_normals(normals)
            prev = zone_defs.get(label)
            if prev is None:
                zone_defs[label] = canon
            elif prev != canon:
                zone_clashes.append((label, fn, prev, canon))

        zones = []
        for z in (data.get("reflection_zones") or []):
            rules = z.get("rules") or []
            if not rules:
                continue
            zones.append(OrderedDict([
                ("zone", z.get("zone")),
                ("orbit", z.get("orbit")),
                ("printed", bool(z.get("printed"))),
                ("normals", [[int(c) for c in v] for v in (z.get("normals") or [])]),
                ("rules", list(rules)),
            ]))
            n_zone_records += 1

        key = str(int(num))
        if key not in groups:
            grp = OrderedDict((k, data.get(k)) for k in GROUP_FIELDS)
            grp["settings"] = []
            groups[key] = grp

        setting = OrderedDict((k, data[k]) for k in SETTING_FIELDS if k in data)
        setting["t_den"] = den
        setting["ops"] = packed
        setting["conditions"] = data.get("reflection_conditions") or {}
        setting["zones"] = zones
        groups[key]["settings"].append(setting)

        n_settings += 1
        n_ops += len(packed)

    ordered = OrderedDict(sorted(groups.items(), key=lambda kv: int(kv[0])))

    payload = OrderedDict([
        ("format", "brutus-sg/1"),
        ("schema_version", schema),
        ("packed_by", os.path.basename(__file__)),
        ("note", "Operators are authoritative for systematic absences: h is "
                 "absent iff some operator has hR = h with h.t non-integral. "
                 "'conditions' and 'zones' are for display and for the "
                 "condition-by-condition evidence hunt. Each op is "
                 "[rot_index, t0, t1, t2] with t = t_num / t_den."),
        ("rotations", rot_table),
        ("zone_defs", OrderedDict((k, [list(v) for v in zone_defs[k]])
                                  for k in sorted(zone_defs))),
        ("setting_count", n_settings),
        ("space_groups", ordered),
    ])

    kw = dict(ensure_ascii=False)
    kw.update({"indent": 2} if args.pretty else {"separators": (",", ":")})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, **kw)

    size = os.path.getsize(args.out)
    print(f"packed {n_settings} settings in {len(ordered)} space groups "
          f"-> {args.out} ({size/1024:.0f} KB)")
    print(f"  distinct rotation matrices : {len(rot_table)}")
    print(f"  total operators            : {n_ops}")
    print(f"  zone labels                : {len(zone_defs)}  "
          f"({', '.join(sorted(zone_defs)[:12])}{' ...' if len(zone_defs) > 12 else ''})")
    print(f"  zone records with rules    : {n_zone_records}")

    if zone_clashes:
        print(f"  !! {len(zone_clashes)} ZONE LABEL CLASHES -- the shared zone_defs "
              f"table cannot be trusted:")
        for label, fn, a, b in zone_clashes[:5]:
            print(f"     '{label}' in {fn}: {list(a)} vs {list(b)}")
    if skipped:
        print(f"  SKIPPED {len(skipped)}:")
        for fn, err in skipped[:10]:
            print(f"    {fn}: {err}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return 1 if (zone_clashes or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
