#!/usr/bin/env python3
"""Build MDL4: Mysteries of Deep Learning (universalities, then the catalogue).

Merged on 2026-10-13 from the former WB5 (scaling laws, architecture,
representational convergence: `mdl4_universality.py`, sections 1-4) and WB6
(the catalogue of surprising empirical facts: `mdl4_catalogue.py`, now
sections 5-17). Anchors keep their old ids (`wb5-keystone`, `wb5-section-N`,
`wb6-section-N`) so existing links survive.
"""
import re

from nbkit import markdown, run
import mdl4_universality as U
import mdl4_catalogue as C

OFFSET = 4  # catalogue section k becomes k + OFFSET


def _shift(src: str) -> str:
    src = re.sub(r"^## (\d+)\. ", lambda m: f"## {int(m.group(1)) + OFFSET}. ", src, flags=re.M)
    src = re.sub(r"§(\d+)", lambda m: f"§{int(m.group(1)) + OFFSET}", src)
    src = re.sub(r"Exercise (\d+)\.(\d+)", lambda m: f"Exercise {int(m.group(1)) + OFFSET}.{m.group(2)}", src)
    src = re.sub(r"\[WB5, section '([^']*)'\]\(mdl4_mysteries\.html#", r"[section '\1'](#", src)
    src = src.replace("mdl4_mysteries.html#", "#")
    return src


def _unlink_catalogue(src: str) -> str:
    src = src.replace("are in WB6 (sections", "are below (sections")
    src = src.replace("mdl4_mysteries.html#", "#")
    return src


def _map(cells, f):
    out = []
    for c in cells:
        c = dict(c)
        for k in ("source", "solution"):
            if k in c:
                c[k] = f(c[k])
        out.append(c)
    return out


ROUTE_U = "\n".join([
    "- [1. Keystone: scaling laws](#wb5-keystone). Kaplan et al. 2020.",
    "- [2. Why power laws? A hundred unnecessary explanations](#wb5-section-2). Exercises on power-law learning curves.",
    "- [3. Another bitter lesson: architecture doesn't matter (much)](#wb5-section-3). Fermi exercise on compute multipliers.",
    "- [4. Same representation? CKA, CCA, and the Platonic and linear representation hypotheses](#wb5-section-4). Exercises on similarity invariances.",
])
ROUTE_C = "\n".join(f"- [{int(k) + OFFSET}. {t}](#wb6-section-{k}): {p}" for k, t, p in C.SECTIONS)

CELLS = [
    markdown(r"""
# MDL4 — Mysteries of Deep Learning: Universalities and a Hundred Surprises

**Mysteries of Deep Learning, Iliad Intensive 2026**

Different networks trained on similar data end up with the same smooth loss curves, nearly regardless of architecture, and arguably with similar internal representations. Nobody designed these regularities in, and nobody fully knows why they hold: the machines, they just want to learn. Sections 1–4 are about those universalities. Sections 5–17 are a catalogue of further things trained networks do that no theory predicted, one per section with the papers that report them: read until you find something you didn't know. Together they are a lower bound on what any good theory of deep learning must explain.

**Universalities**

""" + ROUTE_U + r"""

**The catalogue** (sections are independent)

""" + ROUTE_C + r"""
"""),
] + _map(U.CELLS[1:], _unlink_catalogue) + _map(C.CELLS[1:], _shift)

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "mdl4", "mdl4_mysteries"))
