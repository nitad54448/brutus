#!/usr/bin/env python3
"""Stamp one cache-busting version across every ?v= in brutus.html.

WHY THIS EXISTS
The page and the two workers run the same crystallography scripts. If the
browser serves them from different builds (a new page script talking to a
crystallography file cached last week) the failures look like data problems,
not like a stale cache. One version, stamped everywhere, prevents that.

WHERE THE VERSION IS USED

  brutus.html    every <script src="js/...?v=..."> tag. The page loads all of
                 its own code through these tags.

  js/core/version.js
                 reads its OWN ?v= at runtime (APP_VERSION_QS) and appends it to
                 every URL the app builds itself:
                   - the CPU index worker     js/workers/index-worker.js?v=
                   - the refinement workers   js/workers/refinement-worker.js?v=
                   - the GPU shaders          shaders/*.wgsl?v=
                   - the space-group database sg_ops.json?v=
                 Both workers forward their own ?v= to the crystallography files
                 they import (js/crystallography/manifest.js).

So the tags in brutus.html are the only thing to edit, and they must all
match: if js/core/version.js carried an older ?v= than the crystallography
tags, the workers would load an older build than the page.

Usage:
    python3 bump_version.py                 # today, e.g. 20260827
    python3 bump_version.py 20260827b       # explicit
    python3 bump_version.py --check         # report without writing
"""

import argparse
import datetime
import os
import re
import sys

HTML = "brutus.html"
TAG_RX = re.compile(r'(<script\s+src="([^"?]+)\?v=)([^"]*)(")')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", default=None)
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    ap.add_argument("--html", default=HTML)
    args = ap.parse_args()

    if not os.path.exists(args.html):
        sys.exit(f"no such file: {args.html}  (run this from the app directory)")

    version = args.version or datetime.date.today().strftime("%Y%m%d")

    src = open(args.html, "r", encoding="utf-8", newline="").read()
    found = TAG_RX.findall(src)

    if not found:
        sys.exit(f"no versioned <script src=\"...?v=\"> tags found in {args.html}")

    print(f"{args.html}: {len(found)} versioned script tag(s)")
    stale = []
    for _pre, name, cur, _post in found:
        mark = "" if cur == version else "  <-- will change"
        if cur != version:
            stale.append(name)
        print(f"   {name:40s} v={cur}{mark}")

    versions = {cur for _p, _n, cur, _q in found}
    if len(versions) > 1:
        print(f"\n   !! tags disagree: {sorted(versions)}")
        print("      js/core/version.js's value is the one the workers inherit, so a")
        print("      mismatch means the workers run a different build.")

    has_main = any(n.endswith("js/core/version.js") for _p, n, _c, _q in found)
    if not has_main:
        print("\n   !! no versioned js/core/version.js tag -- APP_VERSION_QS will come back")
        print("      empty and the workers will never be cache-busted.")

    if args.check:
        print(f"\n--check: nothing written. Would set all to v={version}.")
        return 0 if not stale and len(versions) == 1 else 1

    if not stale:
        print(f"\nAlready at v={version}; nothing to do.")
        return 0

    out = TAG_RX.sub(lambda m: m.group(1) + version + m.group(4), src)
    with open(args.html, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"\nSet all script tags to v={version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
