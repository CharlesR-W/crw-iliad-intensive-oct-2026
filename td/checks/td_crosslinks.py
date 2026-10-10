#!/usr/bin/env python3
"""Check links that nbkit_td does not: anchors in other TD notebooks and in the MDL day's HTML.

nbkit_td validates intra-notebook anchors and that local link targets exist. This script
also checks, for every builder and lecture outline:
  - `](td....ipynb#anchor)` or `](td....html#anchor)` (any of the forms nbkit_td rewrites) and
    td2_8's `nb("stem", "anchor")`: the anchor exists in the generated `<stem>.html` page (or,
    if that is not built, in the old solutions/exercises notebook, in td/ or td/_old_ipynb/);
  - `](../notebooks/NAME.html#anchor)`: the file exists and contains id="anchor".
Run from anywhere:  python td/checks/td_crosslinks.py   (exit status 1 on any failure)
"""
import glob
import json
import os
import re
import sys

TD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nb_anchors(path):
    """Anchors of a TD notebook. `path` may name an old `_solutions`/`_exercises` pair member
    or the new single files; the generated `<stem>.html` page is preferred when it exists."""
    base = os.path.basename(path)
    stem = re.sub(r"_(?:solutions|exercises)$", "", os.path.splitext(base)[0])
    page = os.path.join(TD, stem + ".html")
    if os.path.exists(page):
        return set(re.findall(r'\bid="([^"]+)"', open(page).read()))
    for cand in (path, os.path.join(TD, "_old_ipynb", base)):
        if os.path.exists(cand):
            nb = json.load(open(cand))
            md = "".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                         for c in nb["cells"])
            return set(re.findall(r'<a id="([^"]+)"', md))
    return None


def main():
    bad = n = 0
    files = sorted(glob.glob(os.path.join(TD, "build_td*.py"))) + sorted(glob.glob(os.path.join(TD, "lectures", "*.md")))
    for f in files:
        s = open(f).read()
        name = os.path.relpath(f, TD)
        for m in re.finditer(r"\]\((td[0-9a-z_]+?(?:_solutions|_exercises)?\.(?:ipynb|html))(?:#([^)\s]+))?\)", s):
            n += 1
            target, anchor = os.path.join(TD, m.group(1)), m.group(2)
            anchors = nb_anchors(target)
            if anchors is None:
                print(f"BAD  {name}: missing {m.group(1)}"); bad += 1
            elif anchor and anchor not in anchors:
                print(f"BAD  {name}: no #{anchor} in {m.group(1)}"); bad += 1
        for m in re.finditer(r'nb\(\s*"([^"]+)"\s*,\s*"([^"]+)"', s):
            n += 1
            anchors = nb_anchors(os.path.join(TD, m.group(1) + "_solutions.ipynb"))
            if anchors is None or m.group(2) not in anchors:
                print(f"BAD  {name}: {m.group(0)}"); bad += 1
        for m in re.finditer(r"\]\((\.\./notebooks/[^)#\s]+)(?:#([^)\s]+))?\)", s):
            n += 1
            target, anchor = os.path.normpath(os.path.join(TD, m.group(1))), m.group(2)
            if not os.path.exists(target):
                print(f"BAD  {name}: missing {m.group(1)}"); bad += 1
            elif anchor and f'id="{anchor}"' not in open(target).read():
                print(f"BAD  {name}: no #{anchor} in {m.group(1)}"); bad += 1
    print(f"{n} cross-links checked, {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
