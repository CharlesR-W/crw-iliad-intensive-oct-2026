#!/usr/bin/env python3
"""Shared helpers for the Mysteries of Deep Learning notebook builders.

Each `build_wbN.py` defines its cells from these helpers and calls `run(...)`,
which writes ONE self-contained HTML notebook, `<stem>.html`. Markdown cells
become the page; every `answer(...)` cell becomes a collapsed "Solution" block
(click to reveal) placed directly after the question it answers. Solutions are
for everyone; nothing is instructor-only.

Markdown is converted at build time by pandoc (CommonMark + pipe tables + raw
HTML + `$...$` / `$$...$$` math). Math is typeset in the browser by KaTeX from
cdnjs; without a network connection the page still reads, with raw TeX.
Requires `pandoc` on PATH. Current editorial guidance:
reviews/2026-10-08-editorial-brief.md.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent

# CommonMark (what Jupyter and GitHub use) plus pandoc's tables/math/raw HTML.
# Extensions that would reinterpret ordinary prose are switched off: "~100" as a
# subscript, "(a)" as a list marker, ":word:" as an emoji, a leading "---" as YAML.
PANDOC_FROM = ("commonmark_x-subscript-superscript-emoji-fancy_lists"
               "-definition_lists-attributes-bracketed_spans-yaml_metadata_block")
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11"


# --------------------------------------------------------------------------- builder API

def markdown(source: str, *tags: str) -> dict:
    return {"kind": "markdown", "source": source.strip(), "tags": list(tags)}


def answer(solution: str) -> dict:
    """A worked solution, rendered as a collapsed "Solution" block after the preceding cell."""
    return {"kind": "answer", "solution": solution.strip(), "tags": ["answer-cell"]}


def commentary(prompt: str = "") -> str:
    """Deprecated no-op, kept so older builders still import. Returns ""."""
    return ""


def reading(title: str, url: str, who: str, result: str, about: str = "",
            tier: str = "", note: str = "", *, importance: int = 0, length: int = 0) -> str:
    """Markdown for one reading: link, scores, then a sentence or two on what it shows.

    Keep `result` + `about` short: enough to decide whether to read the paper,
    not a summary that replaces it.
    importance, length: required ARENA-style scores, 1-5 (see IMPORTANCE and LENGTH).
    Textbooks are not readings: list them with `references(book(...), ...)` at the
    bottom of the reading list instead.
    tier, note: ignored (kept for compatibility with older call sites).
    """
    body = f"{result} {about}".strip()
    return (f'\n\n<div class="reading">\n\n**[{title}]({url})**, {who} {ratings(importance, length)}\n\n'
            f"{body}\n\n</div>\n\n")


def figure(path: str, what: str) -> str:
    """Markdown link to an interactive figure."""
    return f"▶ **Interactive figure:** [`{Path(path).name}`]({path}). {what}"


# --------------------------------------------------------------------------- reading ratings

# ARENA-style 1-5 scales for every reading. The meaning of each level is shown on hover
# and in the legend under the page's contents.
IMPORTANCE = {
    1: "rabbit hole: only if this exact point grips you",
    2: "optional: more on one specific point",
    3: "worthwhile: for most people taking this section",
    4: "recommended: one of the best readings for this section",
    5: "essential: if you read one thing here, read this",
}
LENGTH = {
    1: "under 15 minutes (a short post, or one section of a paper)",
    2: "15-45 minutes (a long post, or a short paper's main text)",
    3: "1-2 hours (a typical paper, read through)",
    4: "half a day (a long paper or long technical post)",
    5: "a day or more (a monograph-length survey or thesis)",
}


def _dots(n: int) -> str:
    return f'<span class="on">{"●" * n}</span><span class="off">{"○" * (5 - n)}</span>'


def ratings(importance: int, length: int) -> str:
    """Inline HTML for a reading's importance and length scores (1-5 each)."""
    for name, v in (("importance", importance), ("length", length)):
        if not (isinstance(v, int) and 1 <= v <= 5):
            raise ValueError(f"reading needs {name}=1..5, got {v!r}")
    return (f'<span class="ratings">'
            f'<span class="rate rate-imp" title="Importance {importance}/5: {IMPORTANCE[importance]}">'
            f'Importance {_dots(importance)}</span> '
            f'<span class="rate rate-len" title="Length {length}/5: {LENGTH[length]}">'
            f'Length {_dots(length)}</span></span>')


def book(title: str, url: str, who: str = "", where: str = "") -> str:
    """One textbook reference, for `references(...)`: a link, its authors, the relevant part."""
    s = f"[{title}]({url})"
    if who:
        s += f", {who}"
    if where:
        s += f", {where}"
    return s


def references(*books: str) -> str:
    """Textbooks and other books, as plain reference links at the bottom of a reading list (not rated)."""
    return ('\n\n<div class="refs">\n\n**For reference (books):** '
            + "; ".join(books) + ".\n\n</div>\n\n")


# --------------------------------------------------------------------------- rendering

def _variant(src: str) -> str:
    """There is only one variant now: drop the old "{{VARIANT}} · " label."""
    return src.replace("{{VARIANT}} · ", "").replace("{{VARIANT}}", "")


_TABLE_MATH = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")


def _prepare(src: str) -> str:
    """Small source fixes so pandoc reads the cell the way Jupyter did."""
    out = []
    for line in src.split("\n"):
        # A '|' inside $...$ in a table row would split the cell; escape it.
        if line.lstrip().startswith("|"):
            line = _TABLE_MATH.sub(lambda m: "$" + re.sub(r"(?<!\\)\|", r"\\|", m.group(1)) + "$", line)
        out.append(line)
    return "\n".join(out)


def _pandoc(src: str) -> tuple[str, str]:
    proc = subprocess.run(
        ["pandoc", "-f", PANDOC_FROM, "-t", "html5", "--katex", "--wrap=none"],
        input=_prepare(src), capture_output=True, text=True, check=True)
    return proc.stdout, proc.stderr.strip()


def _render_cells(specs: list) -> list[str]:
    sources = [s["solution"] if s["kind"] == "answer" else _variant(s["source"]) for s in specs]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(_pandoc, sources))
    parts = []
    for i, (spec, (body, err)) in enumerate(zip(specs, results)):
        if err:
            print(f"pandoc warning in cell {i}: {err}", file=sys.stderr)
        if spec["kind"] == "answer":
            body = ('<details class="solution">\n<summary>Solution</summary>\n'
                    f'<div class="solution-body">\n{body}</div>\n</details>\n')
        parts.append(body)
    return parts


def _postprocess(body: str) -> str:
    # Bare anchor paragraphs (`<a id="..."></a>` on their own line) become plain anchors.
    body = re.sub(r'<p>(<a id="[^"]+"></a>)</p>', r'\1', body)
    # Wide tables scroll sideways instead of widening the page.
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # Hints and other details inside cells get a class for styling.
    body = re.sub(r"<details>(\s*<summary>)", r'<details class="hint">\1', body)
    # Heading ids are generated per cell; make them unique across the page.
    seen = set(re.findall(r'<a id="([^"]+)"', body))

    def uniq(m):
        base, n = m.group(2), 2
        new = base
        while new in seen:
            new, n = f"{base}-{n}", n + 1
        seen.add(new)
        return f'<h{m.group(1)} id="{new}"'
    return _fold(re.sub(r'<h([1-6]) id="([^"]*)"', uniq, body))


# --------------------------------------------------------------------------- collapsible sections

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "source", "track", "wbr"}
_BARE_ANCHOR = re.compile(r'\s*<a id="[^"]+"></a>\s*')


class _TopLevel(HTMLParser):
    """(offset, tag, attrs) of each top-level element of an HTML fragment."""

    def __init__(self, src: str):
        super().__init__(convert_charrefs=False)
        self._line_at = [0]
        for line in src.split("\n")[:-1]:
            self._line_at.append(self._line_at[-1] + len(line) + 1)
        self.depth, self.items = 0, []
        self.feed(src)
        self.close()

    def _here(self) -> int:
        line, col = self.getpos()
        return self._line_at[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if self.depth == 0:
            self.items.append((self._here(), tag, dict(attrs)))
        if tag not in _VOID:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        if self.depth == 0:
            self.items.append((self._here(), tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag not in _VOID:
            self.depth -= 1


def _fold(body: str, level: int = 2) -> str:
    """Make each top-level <h2> section a collapsed <details class="sec sec2"> with the
    heading as its summary, then the <h3> and <h4> subsections inside it the same way. Bare
    `<a id>` anchors just before a heading move into its section."""
    if level > 4:
        return body
    items = _TopLevel(body).items
    tag = f"h{level}"
    heads = [i for i, it in enumerate(items) if it[1] == tag]
    if not heads:
        return _fold(body, level + 1)  # e.g. <h4> cells directly under an <h2>
    starts = []
    for i in heads:
        j = i
        while j > 0 and items[j - 1][1] == "a" and _BARE_ANCHOR.fullmatch(body[items[j - 1][0]:items[j][0]]):
            j -= 1
        starts.append(items[j][0])
    out = [_fold(body[:starts[0]], level + 1)]  # e.g. an <h3> in the intro, before any <h2>
    for n, i in enumerate(heads):
        start, end = starts[n], (starts[n + 1] if n + 1 < len(starts) else len(body))
        h0 = items[i][0]
        h1 = body.index(f"</{tag}>", h0) + len(f"</{tag}>")
        sid = html.escape(items[i][2].get("id", ""))
        inner = _fold(body[h1:end].strip("\n"), level + 1)
        out.append(f'<details class="sec sec{level}" data-sec="{sid}">\n<summary>{body[h0:h1]}</summary>\n'
                   f'<div class="sec-body">\n{body[start:h0].strip()}\n{inner}\n</div>\n</details>\n')
    return "".join(out)


def _legend_rows(scale: dict) -> str:
    return "".join(f"<li>{_dots(k)} {re.split(r'[:(]', v)[0].strip()}</li>"
                   for k, v in sorted(scale.items(), reverse=True))


LEGEND = ('<div class="legend"><p class="legend-title">Reading scores</p>'
          '<p class="rate rate-imp">Importance</p><ul>' + _legend_rows(IMPORTANCE) + '</ul>'
          '<p class="rate rate-len">Length</p><ul>' + _legend_rows(LENGTH) + '</ul></div>')


def _tools(body: str, n_answers: int) -> str:
    parts = []
    if 'class="sec ' in body:
        parts.append('<button type="button" id="toggle-sections" class="btn" aria-pressed="false">'
                     'Expand all sections</button>')
    if n_answers:
        parts.append('<button type="button" id="toggle-solutions" class="btn" aria-pressed="false">'
                     f'Show all {n_answers} solutions</button>')
    tools = f'<div class="tools">{"".join(parts)}</div>' if parts else ""
    return tools + (LEGEND if 'class="ratings"' in body else "")


def _toc(body: str) -> str:
    items = []
    for hid, inner in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.S):
        inner = re.sub(r"</?a\b[^>]*>", "", inner)
        items.append(f'<li><a href="#{hid}">{inner}</a></li>')
    return "\n".join(items)


def _title(body: str, key: str) -> str:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    if not m:
        return key.upper()
    return html.escape(html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip(), quote=False)


def _page(body: str, key: str, n_answers: int) -> str:
    toc = _toc(body)
    tools = _tools(body, n_answers)
    return (TEMPLATE
            .replace("%%TITLE%%", _title(body, key))
            .replace("%%KEY%%", key)
            .replace("%%KATEX%%", KATEX)
            .replace("%%TOC%%", toc)
            .replace("%%TOOLS%%", tools)
            .replace("%%BODY%%", body))


# --------------------------------------------------------------------------- validation

def _strip_math_and_code(s: str) -> str:
    s = re.sub(r'<span class="math (?:inline|display)">.*?</span>', " ", s, flags=re.S)
    s = re.sub(r"<pre\b.*?</pre>", " ", s, flags=re.S)
    return re.sub(r"<code\b.*?</code>", " ", s, flags=re.S)


def _validate(specs: list, cells_html: list[str], page: str) -> None:
    for i, spec in enumerate(specs):
        src = spec["solution"] if spec["kind"] == "answer" else spec["source"]
        if spec["kind"] == "answer":
            assert src.strip(), f"cell {i}: empty solution"
            # Answer cells never carry their own headings.
            assert not re.match(r"^#{1,6}\s", src), f"cell {i}: answer cell starts with a heading"
        n_open, n_close = len(re.findall(r"<details\b", src)), src.count("</details>")
        assert n_open == n_close, f"cell {i}: unbalanced <details> ({n_open} open, {n_close} close)"
        # Math that pandoc failed to recognise shows up as TeX in the prose.
        prose = _strip_math_and_code(cells_html[i])
        for pat, what in [(r"\$\$", "literal $$"), (r"\\begin\{", r"literal \begin{"),
                          (r"\\[a-zA-Z]{2,}", "a TeX command outside math")]:
            m = re.search(pat, prose)
            assert not m, f"cell {i}: {what}: ...{prose[max(0, m.start()-60):m.end()+40]!r}"
        assert "{{VARIANT}}" not in cells_html[i], f"cell {i}: unreplaced {{{{VARIANT}}}}"
    ids = set(re.findall(r'\bid="([^"]+)"', page))
    for target in re.findall(r'\b(?:href|src)="([^"]+)"', _strip_math_and_code(page)):
        target = html.unescape(target)
        if target.startswith("#"):
            assert target[1:] in ids, f"dangling anchor {target}"
        elif not re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target):
            local_path = unquote(urlsplit(target).path)
            assert (HERE / local_path).exists(), f"missing local link target: {target}"


def _clean(spec: dict) -> dict:
    if spec["kind"] != "markdown":
        return spec
    src = re.sub(r"\n{3,}", "\n\n", spec["source"]).strip()
    return {**spec, "source": src}


def run(specs: list, key: str, stem: str) -> int:
    """Build or --check the single HTML notebook `<stem>.html`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the generated notebook is stale")
    args = ap.parse_args()
    if not shutil.which("pandoc"):
        print("pandoc not found on PATH; it is needed to build the notebooks", file=sys.stderr)
        return 2
    specs = [_clean(s) for s in specs]
    specs = [s for s in specs if s["kind"] == "answer" or s["source"].strip()]
    cells = _render_cells(specs)
    n_answers = sum(s["kind"] == "answer" for s in specs)
    page = _page(_postprocess("\n".join(cells)), key, n_answers)
    _validate(specs, cells, page)
    path = HERE / f"{stem}.html"
    if args.check:
        if not path.exists() or path.read_text() != page:
            print(f"stale: {path.name}", file=sys.stderr)
            return 1
        print(f"{stem}: fresh, {len(specs)} cells, {n_answers} solutions")
        return 0
    path.write_text(page)
    print(f"wrote {path.name}")
    print(f"{stem}: {len(specs)} cells, {n_answers} solutions")
    return 0


# --------------------------------------------------------------------------- page template

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="nbkit.py (%%KEY%%); do not edit by hand">
<title>%%TITLE%%</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600&display=swap">
<link rel="stylesheet" href="%%KATEX%%/katex.min.css">
<script defer src="%%KATEX%%/katex.min.js"></script>
<style>
:root {
  --bg: #fcfbf8; --fg: #1f2328; --muted: #61666d; --rule: #e3e0d8;
  --link: #1f5f9e; --accent: #2b6cb0;
  --sol: #2f7a5f; --sol-bg: #f1f7f4;
  --hint: #9a6b12; --hint-bg: #fbf6ea;
  --code-bg: #f0eee8; --th-bg: #f4f2ed; --tier-core: #2b6cb0;
  --imp: #2b6cb0; --len: #b5651d;
  --serif: "Source Serif 4", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --sans: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181c; --fg: #e4e6ea; --muted: #9aa0a8; --rule: #2e3238;
    --link: #8ab8f0; --accent: #8ab8f0;
    --sol: #7cc9a6; --sol-bg: #1b2622;
    --hint: #e0b964; --hint-bg: #26221a;
    --code-bg: #23262c; --th-bg: #1f2227; --tier-core: #8ab8f0;
    --imp: #8ab8f0; --len: #e3a35f;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #16181c; --fg: #e4e6ea; --muted: #9aa0a8; --rule: #2e3238;
  --link: #8ab8f0; --accent: #8ab8f0;
  --sol: #7cc9a6; --sol-bg: #1b2622;
  --hint: #e0b964; --hint-bg: #26221a;
  --code-bg: #23262c; --th-bg: #1f2227; --tier-core: #8ab8f0;
  --imp: #8ab8f0; --len: #e3a35f;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-padding-top: 1rem; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 400 1.075rem/1.65 var(--serif); font-optical-sizing: auto;
  text-rendering: optimizeLegibility;
}
.layout {
  display: grid; grid-template-columns: minmax(0, 1fr);
  max-width: 76rem; margin: 0 auto; padding: 0 16px;
}
main { min-width: 0; max-width: 46rem; width: 100%; margin: 0 auto; padding: 2.5rem 0 6rem; }
@media (min-width: 70rem) {
  .layout { grid-template-columns: 16rem minmax(0, 1fr); column-gap: 3rem; padding: 0 2rem; }
  main { margin: 0; }
}

/* ---- contents ---- */
nav.toc { font: 400 0.86rem/1.45 var(--sans); padding-top: 1.25rem; }
@media (min-width: 70rem) {
  nav.toc { position: sticky; top: 0; align-self: start; max-height: 100vh; overflow-y: auto; padding: 2.5rem 0 2rem; }
}
nav.toc details > summary {
  cursor: pointer; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
  font-size: .74rem; color: var(--muted); padding: .3rem 0; list-style: none;
}
nav.toc details > summary::-webkit-details-marker { display: none; }
nav.toc details > summary::after { content: " +"; }
nav.toc details[open] > summary::after { content: " \2212"; }
nav.toc ol { list-style: none; margin: .4rem 0 1rem; padding: 0; border-left: 1px solid var(--rule); }
nav.toc li a {
  display: block; padding: .28rem 0 .28rem .8rem; margin-left: -1px; border-left: 2px solid transparent;
  color: var(--muted); text-decoration: none;
}
nav.toc li a:hover { color: var(--fg); }
nav.toc li a.active { color: var(--fg); border-left-color: var(--accent); }
nav.toc .katex { font-size: 1em; }
@media (max-width: 69.99rem) {
  nav.toc { border-bottom: 1px solid var(--rule); padding-bottom: .5rem; max-width: 46rem; width: 100%; margin: 0 auto; }
}
.btn {
  font: 500 .8rem/1 var(--sans); color: var(--fg); background: transparent; cursor: pointer;
  border: 1px solid var(--rule); border-radius: 999px; padding: .45rem .8rem;
}
.btn:hover { border-color: var(--sol); color: var(--sol); }

/* ---- article ---- */
article > :first-child { margin-top: 0; }
h1, h2, h3, h4 { font-family: var(--sans); line-height: 1.25; letter-spacing: -.01em; }
h1 { font-size: 2rem; font-weight: 600; margin: 0 0 .4rem; }
h1 + p { font: 500 .9rem/1.4 var(--sans); color: var(--muted); margin-top: 0; }
h1 + p strong { font-weight: 500; }
h2 { font-size: 1.45rem; font-weight: 600; margin: 3.2rem 0 1rem; padding-top: 1.4rem; border-top: 1px solid var(--rule); }
h3 { font-size: 1.15rem; font-weight: 600; margin: 2.2rem 0 .6rem; }
h4 { font-size: 1rem; font-weight: 600; margin: 1.6rem 0 .4rem; }
p, ul, ol { margin: 0 0 1em; }
li { margin: .25em 0; }
li > p { margin: 0 0 .5em; }
a { color: var(--link); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:hover { text-decoration-thickness: 2px; }
code { font-family: var(--mono); font-size: .86em; background: var(--code-bg); padding: .1em .3em; border-radius: 4px; }
pre { background: var(--code-bg); padding: .8em 1em; border-radius: 6px; overflow-x: auto; line-height: 1.45; }
pre code { background: none; padding: 0; }
blockquote { margin: 1em 0; padding: 0 0 0 1em; border-left: 3px solid var(--rule); color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
img { max-width: 100%; height: auto; }
[id] { scroll-margin-top: 1rem; }

.table-wrap { overflow-x: auto; margin: 1.2em 0; }
table { border-collapse: collapse; font: 400 .88rem/1.45 var(--sans); min-width: 60%; }
th, td { border-bottom: 1px solid var(--rule); padding: .45em .7em; vertical-align: top; text-align: left; }
th { background: var(--th-bg); font-weight: 600; }

.katex { font-size: 1.07em; }
/* KaTeX's hidden MathML is position:absolute; keep it inside its formula (no sideways page scroll). */
.katex, .table-wrap, .katex-display { position: relative; }
.math-punct { font-family: var(--serif); font-size: calc(1em / 1.07); }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: .15em 0; margin: .9em 0; }
.katex-error { color: #c0392b; }
span.math { overflow-wrap: anywhere; }

/* ---- solutions and hints ---- */
article details { margin: 1.1em 0; border-radius: 0 6px 6px 0; }
article details > summary {
  cursor: pointer; list-style: none; user-select: none;
  font: 600 .82rem/1.2 var(--sans); letter-spacing: .03em; padding: .6rem .9rem;
}
article details > summary::-webkit-details-marker { display: none; }
article details > summary::before {
  content: ""; display: inline-block; width: .45em; height: .45em; margin: 0 .55em .12em 0;
  border-right: 2px solid currentColor; border-bottom: 2px solid currentColor;
  transform: rotate(-45deg); transition: transform .15s ease;
}
article details[open] > summary::before { transform: rotate(45deg); }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
details.solution { background: var(--sol-bg); border-left: 3px solid var(--sol); }
details.solution > summary { color: var(--sol); text-transform: uppercase; }
details.solution[open] > summary { border-bottom: 1px solid var(--rule); }
.solution-body { padding: .9rem 1rem .2rem; }
details.hint { background: var(--hint-bg); border-left: 3px solid var(--hint); }
details.hint > summary { color: var(--hint); text-transform: uppercase; }
details.hint > :not(summary) { padding-left: 1rem; padding-right: 1rem; }
details.hint > :last-child { padding-bottom: .4rem; }

/* ---- readings ---- */
.reading { margin: 1.1em 0; padding: .1em 0 .1em 1rem; border-left: 2px solid var(--rule); }
.reading > p:first-child { margin-bottom: .35em; }
.reading > p:last-child { margin-bottom: 0; }
.tier {
  display: inline-block; vertical-align: .12em; margin-left: .35em; padding: .12em .5em;
  font: 600 .66rem/1.3 var(--sans); letter-spacing: .05em; text-transform: uppercase;
  color: var(--muted); border: 1px solid var(--rule); border-radius: 999px;
}
.tier-core { color: var(--tier-core); border-color: currentColor; }

/* ---- collapsible sections ---- */
article details.sec { margin: 0; border-radius: 0; background: none; }
article details.sec2 { border-top: 1px solid var(--rule); }
article h1 ~ details.sec2:first-of-type, article p + details.sec2 { margin-top: 2.2rem; }
article details.sec3, article details.sec4 { border-top: 1px dashed var(--rule); }
article details.sec > summary {
  display: flex; align-items: center; padding: 0;
  font: inherit; letter-spacing: normal; text-transform: none;
}
article details.sec > summary::before { flex: none; margin: 0 .75em .1em .15em; color: var(--muted); }
article details.sec > summary > :is(h2, h3, h4) { flex: 1; min-width: 0; margin: 0; padding: 0; border: 0; }
article details.sec2 > summary { padding: 1.05rem 0; }
article details.sec3 > summary { padding: .6rem 0; }
article details.sec4 > summary { padding: .45rem 0; }
article details.sec > summary:hover > :is(h2, h3, h4) { color: var(--accent); }
.sec-body { padding: .1rem 0 1.4rem; }
.sec-body > :first-child { margin-top: .4rem; }
.tools { display: flex; flex-wrap: wrap; gap: .5rem; }

/* ---- reading scores and textbook references ---- */
.ratings { display: block; margin-top: .2em; font: 500 .76rem/1.4 var(--sans); color: var(--muted); }
.rate { white-space: nowrap; margin-right: 1.2em; }
.rate .on, .rate .off { letter-spacing: .1em; font-size: 1.15em; }
.rate-imp .on { color: var(--imp); }
.rate-len .on { color: var(--len); }
.rate .off { opacity: .55; }
.refs { margin: 1.4em 0 0; font: 400 .86rem/1.5 var(--sans); color: var(--muted); }
.refs > p { margin: 0; }
.legend { margin-top: 1.6rem; font: 400 .76rem/1.45 var(--sans); color: var(--muted); }
.legend p { margin: 0; }
.legend .legend-title { font-weight: 600; text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; margin-bottom: .4rem; }
.legend .rate { display: block; font-weight: 600; margin-top: .3rem; }
.legend ul { list-style: none; margin: .1rem 0 .4rem; padding: 0; }
.legend li { margin: 0; }
.legend .on { letter-spacing: .12em; }
.legend ul:first-of-type .on { color: var(--imp); }
.legend ul:last-of-type .on { color: var(--len); }
.legend .off { letter-spacing: .12em; opacity: .55; }

/* ---- print ---- */
@media print {
  :root { --bg: #fff; --fg: #000; --muted: #444; --sol-bg: #fff; --hint-bg: #fff; color-scheme: light; }
  body { font-size: 10.5pt; }
  nav.toc { display: none; }
  .layout { display: block; max-width: none; padding: 0; }
  main { max-width: none; padding: 0; }
  details::details-content { content-visibility: visible; display: contents; }
  details > summary::before { display: none; }
  details, .reading, pre, .katex-display, table { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
  a { color: inherit; }
}
</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents">
<details id="toc-box"><summary>Contents</summary>
<ol>
%%TOC%%
</ol>
</details>
%%TOOLS%%
</nav>
<main>
<article>
%%BODY%%
</article>
</main>
</div>
<script>
document.addEventListener("DOMContentLoaded", function () {
  // Math: pandoc leaves TeX in <span class="math inline|display">; KaTeX typesets it.
  if (window.katex) {
    document.querySelectorAll("span.math").forEach(function (el) {
      katex.render(el.textContent, el, {
        displayMode: el.classList.contains("display"), throwOnError: false, strict: "ignore"
      });
      // Glue punctuation that follows inline math into the formula's last box,
      // so a line never starts with a lone "." or ",".
      var next = el.nextSibling, bases = el.querySelectorAll(".katex-html > .base");
      if (el.classList.contains("inline") && next && next.nodeType === 3 && bases.length) {
        var m = /^[.,;:!?)\]]+/.exec(next.nodeValue);
        if (m) {
          next.nodeValue = next.nodeValue.slice(m[0].length);
          var p = document.createElement("span"); p.className = "math-punct"; p.textContent = m[0];
          bases[bases.length - 1].appendChild(p);
        }
      }
    });
  }
  // Contents: open beside the text on wide screens, folded on narrow ones.
  var toc = document.getElementById("toc-box");
  var wide = window.matchMedia("(min-width: 70rem)");
  function syncToc() { toc.open = wide.matches; }
  syncToc(); wide.addEventListener("change", syncToc);
  // Show / hide all solutions.
  var sols = Array.prototype.slice.call(document.querySelectorAll("details.solution"));
  var btn = document.getElementById("toggle-solutions");
  function syncBtn() {
    if (!btn) return;
    var allOpen = sols.every(function (d) { return d.open; });
    btn.textContent = (allOpen ? "Hide all " : "Show all ") + sols.length + " solutions";
    btn.setAttribute("aria-pressed", allOpen ? "true" : "false");
  }
  if (btn) btn.addEventListener("click", function () {
    var open = !sols.every(function (d) { return d.open; });
    sols.forEach(function (d) { d.open = open; });
    syncBtn();
  });
  sols.forEach(function (d) { d.addEventListener("toggle", syncBtn); });
  // A link into a folded block opens it.
  function reveal() {
    var id = decodeURIComponent(location.hash.slice(1));
    var el = id && document.getElementById(id);
    for (var p = el && el.parentElement; p; p = p.parentElement) if (p.tagName === "DETAILS") p.open = true;
    if (el) el.scrollIntoView();
  }
  window.addEventListener("hashchange", reveal);
  if (location.hash) reveal();
  // Sections are folded; remember which ones this viewer has open.
  var secs = Array.prototype.slice.call(document.querySelectorAll("article details.sec"));
  var secKey = "open-sections:%%KEY%%", saved = null;
  try { saved = JSON.parse(localStorage.getItem(secKey) || "null"); } catch (err) {}
  if (Array.isArray(saved)) secs.forEach(function (d) { if (saved.indexOf(d.dataset.sec) >= 0) d.open = true; });
  var secBtn = document.getElementById("toggle-sections");
  function syncSecs() {
    var open = secs.filter(function (d) { return d.open; });
    try { localStorage.setItem(secKey, JSON.stringify(open.map(function (d) { return d.dataset.sec; }))); } catch (err) {}
    if (!secBtn) return;
    var allOpen = open.length === secs.length;
    secBtn.textContent = allOpen ? "Collapse all sections" : "Expand all sections";
    secBtn.setAttribute("aria-pressed", allOpen ? "true" : "false");
  }
  if (secBtn) secBtn.addEventListener("click", function () {
    var open = !secs.every(function (d) { return d.open; });
    secs.forEach(function (d) { d.open = open; });
  });
  secs.forEach(function (d) { d.addEventListener("toggle", syncSecs); });
  syncSecs();
  // A click on an in-page link opens the folded blocks around its target first.
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#"]');
    var el = a && document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)));
    for (var p = el && el.parentElement; p; p = p.parentElement) if (p.tagName === "DETAILS") p.open = true;
  });
  // Print everything, folded or not.
  var before = [];
  window.addEventListener("beforeprint", function () {
    before = Array.prototype.map.call(document.querySelectorAll("article details"), function (d) {
      var was = d.open; d.open = true; return [d, was];
    });
  });
  window.addEventListener("afterprint", function () {
    before.forEach(function (x) { x[0].open = x[1]; }); before = [];
  });
  // Highlight the current section in the contents.
  var links = {};
  document.querySelectorAll("nav.toc li a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
  var heads = Array.prototype.filter.call(document.querySelectorAll("article h2[id]"), function (h) { return links[h.id]; });
  if ("IntersectionObserver" in window && heads.length) {
    var current = null;
    var obs = new IntersectionObserver(function () {
      var best = heads[0];
      heads.forEach(function (h) { if (h.getBoundingClientRect().top < window.innerHeight * 0.3) best = h; });
      if (current) current.classList.remove("active");
      current = links[best.id]; current.classList.add("active");
    }, { rootMargin: "0px 0px -70% 0px" });
    heads.forEach(function (h) { obs.observe(h); });
  }
});
</script>
</body>
</html>
"""
