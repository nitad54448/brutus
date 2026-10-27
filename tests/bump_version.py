#!/usr/bin/env python3
"""Write the one version of the program: version.json.

Everything else reads it at run time. index.html fetches version.json before it
loads any script and stamps ?v=<version> on each of them; js/core/version.js turns
the same number into the query for the workers, the shaders and sg_ops.json, and
into the label at the bottom right of the window. Nothing else holds the version.

Usage (from the app directory):
    python3 tests/bump_version.py                 # today, e.g. 20260827
    python3 tests/bump_version.py 20260827b       # explicit
    python3 tests/bump_version.py --check         # print the current version
"""
import argparse, datetime, json, os, sys

FILE = "version.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(FILE):
        sys.exit(f"no such file: {FILE}  (run this from the app directory)")
    current = json.load(open(FILE, encoding="utf-8")).get("version")
    if args.check:
        print(current)
        return
    version = args.version or datetime.date.today().strftime("%Y%m%d")
    with open(FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"version": version}, f, indent=2)
        f.write("\n")
    print(f"{FILE}: {current} -> {version}")

if __name__ == "__main__":
    main()
