#!/usr/bin/env python3
"""Shared helpers for the SLT notebook builders.

Copied from ../td/nbkit_td.py (the Training Dynamics kit) on 2026-10-12, with the
sibling-link prefix changed from `td` to `slt`. Do not edit the TD or MDL kits from
here; each day is built independently.

Each `build_<stem>.py` defines CELLS from these helpers and calls `run(...)`,
which writes two files from one source (see htmlkit_slt.py for the page):

- `<stem>.html`   the reading page: every solution folded under "Solution", every
                  code cell shown with its captured output (figures included)
- `<stem>.ipynb`  the runnable notebook: code stubs, blank answer cells

Cell kinds:
- markdown(src)           the same text in both files ({{VARIANT}} labels are dropped)
- answer(solution)        a worked solution: folded on the page, a blank cell in the notebook
- code(src)               a code cell, identical in both (setup, demos, plotting)
- code_answer(stub, sol)  a code exercise: `stub` in the notebook; on the page, `stub`
                          then a folded solution with `sol` and its output

Everything is for students; nothing is instructor-only. The older
`<stem>_{exercises,solutions}.ipynb` pairs are no longer generated (kept in _old_ipynb/).

Flags:
  (none)    build. Executes the solution code cells only when they changed since the
            outputs cached in _exec_cache/<stem>.json.
  --check   fail if the page or notebook is stale. Never executes code: code changed
            since the cached outputs counts as stale.
  --rerun   build, executing the code even if the cached outputs match it.
  --run     execute every solution code cell in order, in one namespace, with a
            non-interactive matplotlib backend, and report per-cell wall time. Fails
            if any cell raises or exceeds --budget seconds. Writes nothing.
  --html    accepted for compatibility; the HTML build is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import htmlkit_slt
from htmlkit_slt import book, ratings, references  # noqa: F401  (re-exported for builders)

HERE = Path(__file__).resolve().parent


def markdown(source: str, *tags: str) -> dict:
    return {"kind": "markdown", "source": source.strip(), "tags": list(tags)}


def answer(solution: str) -> dict:
    """A worked-solution cell: blank in the exercises variant, `solution` in the solutions variant."""
    return {"kind": "answer", "solution": solution.strip(), "tags": ["answer-cell"]}


def code(source: str, *tags: str) -> dict:
    """A code cell, identical in both variants."""
    return {"kind": "code", "source": source.strip("\n"), "tags": list(tags)}


def code_answer(stub: str, solution: str) -> dict:
    """A code exercise: `stub` in the exercises variant, `solution` in the solutions variant."""
    return {"kind": "code_answer", "stub": stub.strip("\n"), "solution": solution.strip("\n"),
            "tags": ["code-answer-cell"]}


def reading(title: str, url: str, who: str, result: str, about: str = "",
            tier: str = "", *, importance: int = 0, length: int = 0) -> str:
    """Markdown for one reading: link, scores, then a sentence or two on what it shows.

    Keep it short: enough to decide whether to read the paper, not a summary
    that replaces it. importance, length: required ARENA-style scores, 1-5 (see
    htmlkit_slt.IMPORTANCE and LENGTH). Textbooks are not readings: list them with
    `references(book(...), ...)` at the bottom of the reading list instead.
    tier: ignored (kept for compatibility with older call sites).
    """
    body = f"{result} {about}".strip()
    return (f'\n\n<div class="reading">\n\n**[{title}]({url})**, {who} {ratings(importance, length)}\n\n'
            f"{body}\n\n</div>\n\n")


def figure(path: str, what: str) -> str:
    """Markdown link to an interactive figure (path relative to slt/)."""
    return (f"▶ **Interactive figure:** [`{Path(path).name}`]({path}) "
            f"(a standalone page: open the file in a web browser; some notebook viewers show it as source). {what}")


def _cell(spec: dict, key: str, index: int, variant: str) -> dict:
    kind = spec["kind"]
    cid = hashlib.sha1(f"{key}:{index}:{kind}".encode()).hexdigest()[:12]
    meta = {"tags": list(spec.get("tags", []))}
    if kind == "code":
        return {"cell_type": "code", "id": cid, "metadata": meta, "execution_count": None,
                "outputs": [], "source": spec["source"]}
    if kind == "code_answer":
        src = spec["stub"] if variant == "Exercises" else spec["solution"]
        return {"cell_type": "code", "id": cid, "metadata": meta, "execution_count": None,
                "outputs": [], "source": src}
    if kind == "answer":
        src = "\n\n\n" if variant == "Exercises" else spec["solution"]
    else:
        src = spec["source"].replace("{{VARIANT}}", variant)
    return {"cell_type": "markdown", "id": cid, "metadata": meta, "source": src}


def _notebook(specs: list, key: str, variant: str) -> dict:
    return {
        "cells": [_cell(s, key, i, variant) for i, s in enumerate(specs)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "slt": {"notebook": key, "role": variant.lower()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _runnable(specs: list, key: str, stem: str) -> dict:
    """`<stem>.ipynb`: the exercises variant (stubs, blank answers), with links to sibling
    notebooks pointing at their `.ipynb` and a pointer to the HTML page's solutions."""
    nb = _notebook(specs, key, "Exercises")
    nb["metadata"]["slt"]["role"] = "runnable"
    note = (f"Worked solutions, and the output of every code cell, are on the reading page "
            f"[`{stem}.html`]({stem}.html) (open it in a web browser).")
    first = True
    for spec, c in zip(specs, nb["cells"]):
        if c["cell_type"] != "markdown" or spec["kind"] != "markdown":
            continue
        src = htmlkit_slt.rewrite_md_links(htmlkit_slt._variant(spec["source"]), "ipynb")
        if first:
            src, first = htmlkit_slt._insert_after_subtitle(src, note), False
        c["source"] = src
    return nb


def _validate(specs: list, ex: dict, so: dict, key: str) -> None:
    assert len(ex["cells"]) == len(so["cells"]) == len(specs)
    md = "\n".join(c["source"] for c in so["cells"] if c["cell_type"] == "markdown")
    for i, (spec, a, b) in enumerate(zip(specs, ex["cells"], so["cells"])):
        assert a["id"] == b["id"]
        kind = spec["kind"]
        if kind == "answer":
            assert not a["source"].strip(), f"cell {i}: exercise answer not blank"
            assert b["source"].strip(), f"cell {i}: empty solution"
            assert not re.match(r"^#{1,6}\s", spec["solution"]), f"cell {i}: answer cell starts with a heading"
        elif kind == "code_answer":
            assert spec["stub"] != spec["solution"], f"cell {i}: code stub equals solution"
            assert spec["solution"].strip(), f"cell {i}: empty code solution"
        elif kind == "code":
            assert a["source"] == b["source"]
        else:
            src = spec["source"]
            assert a["source"] == src.replace("{{VARIANT}}", "Exercises")
            assert b["source"] == src.replace("{{VARIANT}}", "Solutions")
    assert md.count("<details>") == md.count("</details>"), "unbalanced <details>"
    md_nomath = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", "", md, flags=re.S)
    for target in re.findall(r"\]\((?!https?://|#|mailto:)([^)\s#]+)(?:#[^)\s]*)?\)", md_nomath):
        if htmlkit_slt.slt_target(target, "html"):
            continue  # a sibling SLT notebook (any of its forms): its builder exists
        assert (HERE / target).exists(), f"missing local link target: {target}"
    for anchor in re.findall(r"\]\(#([^)\s]+)\)", md):
        assert f'<a id="{anchor}"></a>' in md, f"dangling anchor #{anchor}"


def _link_index_rows(src: str) -> str:
    """A table row that starts with a bare sibling-notebook stem in backticks (what a day-index page's
    `nb()` prints when `<stem>_exercises.ipynb` is absent) links that notebook."""
    def sub(m):
        stem = m.group(2)
        return f"{m.group(1)}[`{stem}`]({stem}.ipynb) |" if (HERE / f"build_{stem}.py").exists() else m.group(0)
    return re.sub(r"^(\|\s*)`(slt[0-9a-z_]+)`\s*\|", sub, src, flags=re.M)


def _clean(spec: dict) -> dict:
    if spec["kind"] != "markdown":
        return spec
    src = _link_index_rows(re.sub(r"\n{3,}", "\n\n", spec["source"]).strip())
    return {**spec, "source": src}


def _execute(nb: dict, budget: float) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(HERE)
    ns: dict = {"__name__": "__main__"}
    total, worst, failed = 0.0, 0.0, 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = c["source"]
        # Strip IPython magics / shell lines so plain exec works.
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            print(f"cell {i}: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        dt = time.time() - t0
        total += dt
        worst = max(worst, dt)
        flag = "  <-- over budget" if dt > budget else ""
        print(f"cell {i}: {dt:6.2f}s{flag}")
        if dt > budget:
            failed += 1
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:  # noqa: BLE001
        pass
    print(f"executed: total {total:.1f}s, slowest cell {worst:.1f}s, failures {failed}")
    return 1 if failed else 0


def run(specs: list, key: str, stem: str) -> int:
    """Build, --check, or --run `<stem>.html` and `<stem>.ipynb`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the page or notebook is stale (never executes code)")
    ap.add_argument("--run", action="store_true", help="execute the solution code cells and time them")
    ap.add_argument("--budget", type=float, default=60.0, help="max seconds per code cell for --run")
    ap.add_argument("--rerun", action="store_true",
                    help="execute the code even if the cached outputs match it")
    ap.add_argument("--html", action="store_true", help="no-op: the HTML build is the default")
    args = ap.parse_args()
    specs = [_clean(s) for s in specs]
    specs = [s for s in specs if s["kind"] not in ("markdown",) or s["source"].strip()]
    ex = _notebook(specs, key, "Exercises")
    so = _notebook(specs, key, "Solutions")
    _validate(specs, ex, so, key)
    if args.run:
        return _execute(so, args.budget)
    return _run_html(specs, key, stem, check=args.check, rerun=args.rerun)


def _run_html(specs: list, key: str, stem: str, check: bool, rerun: bool) -> int:
    """Build or --check `<stem>.html` and `<stem>.ipynb`. --check never executes code."""
    if not __import__("shutil").which("pandoc"):
        print("pandoc not found on PATH; it is needed to build the HTML pages", file=sys.stderr)
        return 2
    outputs, status = htmlkit_slt.get_outputs(specs, stem, allow_execute=not check, force=rerun)
    if status == "failed":
        return 1
    if status == "stale":
        print(f"stale: {stem}.html (code changed since the cached outputs; rebuild to execute)",
              file=sys.stderr)
        return 1
    page = htmlkit_slt.build_page(specs, key, stem, outputs)
    nb = _runnable(specs, key, stem)
    texts = {HERE / f"{stem}.html": page,
             HERE / f"{stem}.ipynb": json.dumps(nb, indent=1, ensure_ascii=False) + "\n"}
    n_sol = sum(s["kind"] in ("answer", "code_answer") for s in specs)
    n_code = sum(s["kind"] in ("code", "code_answer") for s in specs)
    if check:
        stale = [p.name for p, t in texts.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"{stem}: fresh, {len(specs)} cells, {n_code} code cells, {n_sol} solutions")
        return 0
    for p, t in texts.items():
        p.write_text(t)
        print(f"wrote {p.name}")
    print(f"{stem}: {len(specs)} cells, {n_code} code cells ({status} outputs), {n_sol} solutions")
    return 0
