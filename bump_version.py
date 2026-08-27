#!/usr/bin/env python3
"""Stamp one cache-busting version across every ?v= in brutus.html.

WHY THIS EXISTS
The version lives in more places than is obvious, and getting it half-right is
worse than not bumping at all -- you end up with a new main_app.js talking to a
worker-logic.js the browser cached last week, which fails in ways that look like
data problems.

WHERE THE VERSION IS ACTUALLY USED

  brutus.html   <script src="worker-logic.js?v=...">    main-thread copy
                <script src="webgpu-engine.js?v=...">   main-thread copy
                <script src="main_app.js?v=...">        main-thread copy, AND
                                                        the source of truth for
                                                        everything below

  main_app.js   reads its OWN ?v= off the script tag at runtime (APP_VERSION_QS)
                and appends it to:
                  - the CPU index worker      new Worker('worker-logic.js?v=')
                  - the refinement workers    new Worker('refinement-worker.js?v=')
                  - the space group database  fetch('sg_ops.json?v=')

  refinement-worker.js  forwards the same ?v= to its importScripts('worker-logic.js')

So the three tags in brutus.html are the only thing to edit, and the main_app.js
one propagates to four more places by itself. That is why they must all match:
if main_app.js carries an older ?v= than worker-logic.js, the workers load the
OLD worker-logic while the main thread has the new one.

Note that the .wgsl shaders are fetched by bare filename with no ?v= at all, so
a shader change still needs a hard reload (Ctrl+Shift+R). Adding a version there
means touching SYSTEM_CONFIGS.shader and loadShader's cache key; left alone for
now, but it is the remaining gap.

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
        print(f"   {name:26s} v={cur}{mark}")

    versions = {cur for _p, _n, cur, _q in found}
    if len(versions) > 1:
        print(f"\n   !! tags disagree: {sorted(versions)}")
        print("      main_app.js's value is the one the workers inherit, so a")
        print("      mismatch means the workers run a different build.")

    has_main = any(n.endswith("main_app.js") for _p, n, _c, _q in found)
    if not has_main:
        print("\n   !! no versioned main_app.js tag -- APP_VERSION_QS will come back")
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
    print("Shaders (.wgsl) are still fetched unversioned -- hard reload "
          "(Ctrl+Shift+R) after a shader change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
