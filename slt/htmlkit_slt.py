#!/usr/bin/env python3
"""HTML output for the SLT notebooks (one page per builder).

Called from `nbkit_slt.run()`. Writes, for each builder:

- `<stem>.html`  one self-contained reading page in the style of the MDL day's
                 pages (template, CSS and KaTeX loader copied from
                 ../notebooks/nbkit.py, plus styles for code and outputs).
                 Markdown is converted by pandoc; every `answer()` becomes a
                 folded "Solution" block; `code()` cells show their code and
                 captured output; `code_answer()` cells show the stub, then a
                 folded "Solution" with the solution code and its output.
                 Interactive figures linked by `figure()` are also embedded in
                 an <iframe>.
- `<stem>.ipynb` the runnable notebook: code stubs and blank answer cells
                 (what used to be `<stem>_exercises.ipynb`).

Outputs come from executing the solution code cells in order, in one namespace,
with matplotlib's Agg backend (stdout, stderr, the value of a trailing
expression, and every figure as a PNG data URI). They are cached in
`_exec_cache/<stem>.json`, keyed on a hash of the code that runs, so `--check`
never executes anything and an unchanged notebook rebuilds in seconds.

Cross-notebook links written as `sltX_..._{solutions,exercises}.ipynb#anchor`
(or `sltX_....ipynb`, or `sltX_....html`) are rewritten to `<stem>.html#anchor`
in the page and to `<stem>.ipynb#anchor` in the runnable notebook.

Copied from ../td/htmlkit_td.py on 2026-10-12 (sibling-link prefix `slt`). Do not edit the
TD or MDL kits from here; each day is built independently.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_exec_cache"
# Bump when the executor or the way outputs are captured changes (invalidates every cache).
EXEC_VERSION = "1"

# Same pandoc reader as the MDL kit: CommonMark plus tables, math and raw HTML, with
# extensions that would reinterpret ordinary prose switched off.
PANDOC_FROM = ("commonmark_x-subscript-superscript-emoji-fancy_lists"
               "-definition_lists-attributes-bracketed_spans-yaml_metadata_block")
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11"

# A link to a sibling SLT notebook, in any of the forms builders use.
_SLT_LINK = re.compile(r"^(slt[0-9a-z_]+?)(?:_solutions|_exercises)?\.(?:ipynb|html)(#.*)?$")


def slt_target(target: str, ext: str) -> str | None:
    """`slt1_bayes_statmech_solutions.ipynb#x` -> `slt1_bayes_statmech.<ext>#x` if a builder for that stem exists."""
    m = _SLT_LINK.match(target)
    if not m or not (HERE / f"build_{m.group(1)}.py").exists():
        return None
    return f"{m.group(1)}.{ext}{m.group(2) or ''}"


def rewrite_md_links(src: str, ext: str) -> str:
    """Rewrite markdown `](sltX....ipynb#a)` links to `](sltX.<ext>#a)`."""
    def sub(m):
        new = slt_target(m.group(1), ext)
        return f"]({new})" if new else m.group(0)
    return re.sub(r"\]\(([^)\s]+)\)", sub, src)


def _variant(src: str) -> str:
    """One page for everyone: drop the old "{{VARIANT}} · " label."""
    return src.replace("{{VARIANT}} · ", "").replace("{{VARIANT}}", "")



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


# --------------------------------------------------------------------------- execution

def code_cells(specs: list) -> list[str]:
    """Source of every code cell that runs for the page (solutions for code exercises)."""
    return [s["source"] if s["kind"] == "code" else s["solution"]
            for s in specs if s["kind"] in ("code", "code_answer")]


def cache_key(sources: list[str]) -> str:
    h = hashlib.sha256(EXEC_VERSION.encode())
    for s in sources:
        h.update(b"\0cell\0" + s.encode())
    return h.hexdigest()


class _Stream(io.TextIOBase):
    """File-like object appending text to the current cell's output list."""

    def __init__(self, name: str, sink: list):
        self.name, self.sink = name, sink

    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0
        out = self.sink[-1] if self.sink else None
        if out and out["type"] == self.name:
            out["text"] += s
        else:
            self.sink.append({"type": self.name, "text": s})
        return len(s)


def _png(fig) -> dict:
    buf = io.BytesIO()
    dpi = min(2 * fig.dpi, 220)  # sharp on high-density screens; shown at half size
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", metadata={"Software": None},
                facecolor="white")
    data = buf.getvalue()
    # Displayed size: the PNG's pixel width at the figure's own dpi.
    width = int.from_bytes(data[16:20], "big")
    titles = [t for ax in fig.axes for t in (ax.get_title(),) if t]
    if getattr(fig, "_suptitle", None) is not None:
        titles.insert(0, fig._suptitle.get_text())
    alt = "Figure" + (": " + "; ".join(titles) if titles else "")
    return {"type": "image", "png": base64.b64encode(data).decode(),
            "width": round(width * fig.dpi / dpi), "alt": alt}


def execute(sources: list[str], label: str = "", verbose: bool = False) -> tuple[list, list, int]:
    """Run `sources` in order in one namespace, the way Jupyter would, capturing outputs.

    Returns (outputs per cell, seconds per cell, number of failed cells). A failed cell's
    output ends with its traceback; later cells still run.
    """
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    old_cwd = os.getcwd()
    os.chdir(HERE)
    ns: dict = {"__name__": "__main__"}
    outputs, times, failed = [], [], 0
    real_show = plt.show
    sink: list = []

    def flush_figures():
        for num in plt.get_fignums():
            sink.append(_png(plt.figure(num)))
        plt.close("all")

    plt.show = lambda *a, **k: flush_figures()
    try:
        for i, src in enumerate(sources):
            sink = []
            # IPython magics and shell lines cannot run under plain exec.
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
            t0 = time.time()
            ok = True
            with contextlib.redirect_stdout(_Stream("stdout", sink)), \
                    contextlib.redirect_stderr(_Stream("stderr", sink)):
                try:
                    tree = ast.parse(src, f"<cell {i}>")
                    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                    exec(compile(tree, f"<cell {i}>", "exec"), ns)
                    if last is not None:
                        value = eval(compile(ast.Expression(last.value), f"<cell {i}>", "eval"), ns)
                        if value is not None:
                            sink.append({"type": "result", "text": repr(value)})
                except Exception:  # noqa: BLE001
                    ok = False
                    tb = traceback.format_exc()
                    sink.append({"type": "error", "text": tb})
            flush_figures()
            dt = time.time() - t0
            times.append(dt)
            outputs.append(sink)
            if not ok:
                failed += 1
                print(f"{label} code cell {i}: FAILED\n{sink[-1]['text'] if sink else ''}", file=sys.stderr)
            elif verbose:
                print(f"{label} code cell {i}: {dt:6.2f}s")
    finally:
        plt.show = real_show
        plt.close("all")
        os.chdir(old_cwd)
    return outputs, times, failed


def load_cache(stem: str, key: str):
    path = CACHE_DIR / f"{stem}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data["outputs"] if data.get("key") == key else None


def save_cache(stem: str, key: str, outputs: list, times: list) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    data = {"key": key, "stem": stem, "seconds": [round(t, 2) for t in times], "outputs": outputs}
    (CACHE_DIR / f"{stem}.json").write_text(json.dumps(data) + "\n")


# --------------------------------------------------------------------------- rendering

_TABLE_MATH = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")


# mathtools environments that Jupyter's MathJax knows and KaTeX does not.
_KATEX_ENVS = {r"\begin{psmallmatrix}": r"\left(\begin{smallmatrix}", r"\end{psmallmatrix}": r"\end{smallmatrix}\right)",
               r"\begin{bsmallmatrix}": r"\left[\begin{smallmatrix}", r"\end{bsmallmatrix}": r"\end{smallmatrix}\right]"}


def _prepare(src: str) -> str:
    """Small source fixes so pandoc reads the cell the way Jupyter did."""
    for a, b in _KATEX_ENVS.items():
        src = src.replace(a, b)
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


def _highlight(code: str) -> str:
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
    except ImportError:
        return html.escape(code)
    return highlight(code, PythonLexer(), HtmlFormatter(nowrap=True)).rstrip("\n")


def _code_block(code: str) -> str:
    return f'<pre class="code"><code class="language-python">{_highlight(code)}</code></pre>\n'


def _code_fold(code: str) -> str:
    """A code cell's source, folded under "Code (n lines)"; its output is shown after it."""
    n = code.count("\n") + 1
    return (f'<details class="code"><summary>Code ({n} line{"s" if n > 1 else ""})</summary>\n'
            f"{_code_block(code)}</details>\n")


def _outputs_html(outs: list) -> str:
    if not outs:
        return ""
    parts = []
    for o in outs:
        if o["type"] == "image":
            parts.append(f'<img class="plot" src="data:image/png;base64,{o["png"]}" '
                         f'width="{o["width"]}" alt="{html.escape(o["alt"])}">')
        else:
            text = o["text"].rstrip("\n")
            if text:
                parts.append(f'<pre class="out-{o["type"]}">{html.escape(text)}</pre>')
    return '<div class="output">\n' + "\n".join(parts) + "\n</div>\n" if parts else ""


def _solution(body: str, label: str = "Solution") -> str:
    return (f'<details class="solution">\n<summary>{label}</summary>\n'
            f'<div class="solution-body">\n{body}</div>\n</details>\n')


# figure() in nbkit_slt writes:  ▶ **Interactive figure:** [`x.html`](path) (a standalone page: ...). what
_FIG_MD = re.compile(r"(▶ \*\*Interactive figure:\*\* \[`[^`]+`\]\(([^)\s]+)\))(?: \(a standalone page:[^)]*\))?")


def _figure_md(src: str) -> tuple[str, list[str]]:
    """Drop the notebook-only "(a standalone page ...)" remark; return the figure paths."""
    paths = []

    def sub(m):
        paths.append(m.group(2))
        return m.group(1) + " (opens full page)"
    return _FIG_MD.sub(sub, src), paths


def _embed_figures(body: str, paths: list[str], seen: set) -> str:
    """After the paragraph that first links each figure, embed that figure in an iframe."""
    for path in paths:
        if path in seen:
            continue  # already embedded higher up the page; later mentions stay links
        seen.add(path)
        link = f'<a href="{html.escape(path)}">'
        i = body.find(link)
        if i < 0:
            continue
        end = body.find("</p>", i)
        if end < 0:
            continue
        end += len("</p>")
        frame = (f'\n<div class="figure-frame"><iframe src="{html.escape(path)}" '
                 f'title="Interactive figure: {html.escape(Path(path).name)}" loading="lazy">'
                 f'</iframe></div>\n')
        body = body[:end] + frame + body[end:]
    return body


def _requirements(specs: list) -> str:
    srcs = "\n".join(code_cells(specs))
    mods = [m for m in ("numpy", "matplotlib", "scipy", "torch")
            if re.search(rf"^\s*(?:import|from)\s+{m}\b", srcs, flags=re.M)]
    if not mods:
        return "Python 3"
    return ", ".join(mods[:-1]) + (" and " if len(mods) > 1 else "") + mods[-1]


def _insert_after_subtitle(src: str, block: str) -> str:
    """Insert `block` after the title cell's subtitle paragraph (or after the h1)."""
    lines = src.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("# "):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**"):
                while j < len(lines) and lines[j].strip():
                    j += 1
                return "\n".join(lines[:j] + ["", block] + ([""] if j < len(lines) and lines[j].strip() else []) + lines[j:])
            return "\n".join(lines[:i + 1] + ["", block, ""] + lines[i + 1:])
    return block + "\n\n" + src


def run_note(specs: list, stem: str) -> str:
    # Single-quoted href on purpose: _rewrite_html_links only rewrites href="..." and must
    # leave this link pointing at the runnable notebook, not the page itself.
    return (f'<p class="run-note">To run the code, open <a href=\'{stem}.ipynb\'><code>{stem}.ipynb</code></a> '
            f"in Jupyter (CPU only; needs {_requirements(specs)}). Every code cell's output is shown "
            "on this page; exercise solutions are folded under <em>Solution</em>.</p>")


def _render_cells(specs: list, outputs: list, stem: str) -> tuple[list[str], list[list[str]]]:
    """HTML for each spec, and the interactive-figure paths per cell."""
    jobs, figs = [], []
    first_md = next((i for i, s in enumerate(specs) if s["kind"] == "markdown"), None)
    for i, s in enumerate(specs):
        if s["kind"] in ("markdown", "answer"):
            src = s["solution"] if s["kind"] == "answer" else _variant(s["source"])
            if i == first_md:
                src = _insert_after_subtitle(src, run_note(specs, stem))
            src, paths = _figure_md(src)
            jobs.append(src)
            figs.append(paths)
        else:
            jobs.append(None)
            figs.append([])
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(lambda x: _pandoc(x) if x is not None else ("", ""), jobs))
    parts, k, embedded = [], 0, set()
    for i, (spec, (body, err)) in enumerate(zip(specs, results)):
        if err:
            print(f"pandoc warning in cell {i}: {err}", file=sys.stderr)
        kind = spec["kind"]
        if kind == "answer":
            body = _solution(body)
        elif kind == "markdown":
            body = _embed_figures(body, figs[i], embedded)
        elif kind == "code":
            body = ('<div class="cell">\n' + _code_fold(spec["source"]) + _outputs_html(outputs[k])
                    + "</div>\n")
            k += 1
        elif kind == "code_answer":
            sol = _code_block(spec["solution"]) + _outputs_html(outputs[k])
            body = ('<div class="cell cell-stub">\n' + _code_block(spec["stub"]) + "</div>\n"
                    + _solution(sol, "Solution (code)"))
            k += 1
        parts.append(body)
    return parts, figs


def _rewrite_html_links(body: str) -> str:
    def sub(m):
        new = slt_target(html.unescape(m.group(2)), "html")
        return f'{m.group(1)}"{html.escape(new)}"' if new else m.group(0)
    return re.sub(r'(\bhref=)"([^"]+)"', sub, body)


def _postprocess(body: str) -> str:
    # Bare anchor paragraphs (`<a id="..."></a>` on their own line) become plain anchors.
    body = re.sub(r'<p>(<a id="[^"]+"></a>)</p>', r'\1', body)
    # Wide tables scroll sideways instead of widening the page.
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # Hints and other details inside cells get a class for styling.
    body = re.sub(r"<details>(\s*<summary>)", r'<details class="hint">\1', body)
    body = _rewrite_html_links(body)
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
        return key
    text = re.sub(r'<span class="math inline">\\\((.*?)\\\)</span>', r"\1", m.group(1))
    return html.escape(html.unescape(re.sub(r"<[^>]+>", "", text)).strip(), quote=False)


def _page(body: str, key: str, n_answers: int) -> str:
    tools = _tools(body, n_answers)
    return (TEMPLATE
            .replace("%%TITLE%%", _title(body, key))
            .replace("%%KEY%%", key)
            .replace("%%KATEX%%", KATEX)
            .replace("%%TOC%%", _toc(body))
            .replace("%%TOOLS%%", tools)
            .replace("%%BODY%%", body))


# --------------------------------------------------------------------------- validation

def _strip_math_and_code(s: str) -> str:
    s = re.sub(r'<span class="math (?:inline|display)">.*?</span>', " ", s, flags=re.S)
    s = re.sub(r"<pre\b.*?</pre>", " ", s, flags=re.S)
    s = re.sub(r'<img class="plot"[^>]*>', " ", s)
    return re.sub(r"<code\b.*?</code>", " ", s, flags=re.S)


def validate(specs: list, cells_html: list[str], page: str, stem: str) -> None:
    """The MDL checks (balanced <details>, no TeX outside math, links and anchors resolve)."""
    for i, spec in enumerate(specs):
        if spec["kind"] in ("code", "code_answer"):
            continue
        src = spec["solution"] if spec["kind"] == "answer" else spec["source"]
        if spec["kind"] == "answer":
            assert src.strip(), f"{stem} cell {i}: empty solution"
            assert not re.match(r"^#{1,6}\s", src), f"{stem} cell {i}: answer cell starts with a heading"
        n_open, n_close = len(re.findall(r"<details\b", src)), src.count("</details>")
        assert n_open == n_close, f"{stem} cell {i}: unbalanced <details> ({n_open} open, {n_close} close)"
        # Math that pandoc failed to recognise shows up as TeX in the prose.
        prose = _strip_math_and_code(cells_html[i])
        for pat, what in [(r"\$\$", "literal $$"), (r"\\begin\{", r"literal \begin{"),
                          (r"\\[a-zA-Z]{2,}", "a TeX command outside math")]:
            m = re.search(pat, prose)
            assert not m, f"{stem} cell {i}: {what}: ...{prose[max(0, m.start()-60):m.end()+40]!r}"
        assert "{{VARIANT}}" not in cells_html[i], f"{stem} cell {i}: unreplaced {{{{VARIANT}}}}"
    all_ids = re.findall(r'\bid="([^"]+)"', page)
    ids = set(all_ids)
    dup = sorted(x for x, n in Counter(all_ids).items() if n > 1)
    assert not dup, f"{stem}: duplicate ids {dup[:5]}"
    for target in re.findall(r'\b(?:href|src)="([^"]+)"', _strip_math_and_code(page)):
        target = html.unescape(target)
        if target.startswith("#"):
            assert target[1:] in ids, f"{stem}: dangling anchor {target}"
        elif not re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target):
            parts = urlsplit(target)
            local = unquote(parts.path)
            sibling = re.match(r"^(slt[0-9a-z_]+)\.(?:html|ipynb)$", local)
            if sibling and (HERE / f"build_{sibling.group(1)}.py").exists():
                # A sibling page: it may not be built yet; check its anchor if it is.
                other = HERE / f"{sibling.group(1)}.html"
                if parts.fragment and other.exists() and sibling.group(1) != stem:
                    assert f'id="{parts.fragment}"' in other.read_text(), \
                        f"{stem}: no #{parts.fragment} in {other.name}"
                continue
            assert (HERE / local).exists(), f"{stem}: missing local link target: {target}"
            if parts.fragment and local.endswith(".html") and \
                    f'id="{parts.fragment}"' not in (HERE / local).read_text():
                # Another day's page (owned and rebuilt elsewhere): warn, don't fail the build.
                where = "the MDL day's" if local.startswith("../notebooks/") else "the"
                print(f"WARNING {stem}: no #{parts.fragment} in {where} page {local}", file=sys.stderr)


# --------------------------------------------------------------------------- entry point

def build_page(specs: list, key: str, stem: str, outputs: list) -> str:
    cells, _ = _render_cells(specs, outputs, stem)
    n_sol = sum(s["kind"] in ("answer", "code_answer") for s in specs)
    page = _page(_postprocess("\n".join(cells)), key, n_sol)
    validate(specs, cells, page, stem)
    return page


def get_outputs(specs: list, stem: str, *, allow_execute: bool, force: bool = False,
                verbose: bool = False):
    """Cached outputs for the current code, executing if allowed and needed.

    Returns (outputs, status) where status is "cached", "executed", "stale" or "failed".
    """
    sources = code_cells(specs)
    key = cache_key(sources)
    if not force:
        cached = load_cache(stem, key)
        if cached is not None:
            return cached, "cached"
    if not allow_execute:
        return None, "stale"
    t0 = time.time()
    outputs, times, failed = execute(sources, stem, verbose=verbose)
    print(f"{stem}: executed {len(sources)} code cells in {time.time() - t0:.1f}s"
          + (f", {failed} FAILED" if failed else ""))
    if failed:
        return outputs, "failed"
    save_cache(stem, key, outputs, times)
    return outputs, "executed"


# --------------------------------------------------------------------------- page template
# Copied from ../notebooks/nbkit.py (MDL), with code/output/figure styles added in the
# block marked "TD additions".

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="htmlkit_slt.py (%%KEY%%); do not edit by hand">
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
  --out-bg: #ffffff; --err: #b3261e;
  --hl-kw: #8a3ea8; --hl-str: #2f7a2f; --hl-num: #a3570d; --hl-com: #7a7f86;
  --hl-fn: #1f5f9e; --hl-bi: #0e7490;
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
    --out-bg: #1b1d22; --err: #f2948b;
    --hl-kw: #d2a8ff; --hl-str: #8fd18f; --hl-num: #f0b072; --hl-com: #8b9199;
    --hl-fn: #8ab8f0; --hl-bi: #67d0e0;
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
  --out-bg: #1b1d22; --err: #f2948b;
  --hl-kw: #d2a8ff; --hl-str: #8fd18f; --hl-num: #f0b072; --hl-com: #8b9199;
  --hl-fn: #8ab8f0; --hl-bi: #67d0e0;
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

/* ---- TD additions: run note, code cells, outputs, embedded figures ---- */
/* KaTeX's hidden MathML is position:absolute; keep it inside scrolling tables and displays. */
.katex, .table-wrap, .katex-display { position: relative; }
.run-note {
  font: 400 .86rem/1.5 var(--sans); color: var(--muted);
  border: 1px solid var(--rule); border-radius: 6px; padding: .6rem .8rem; margin: 0 0 1.6em;
}
.cell { margin: 1.2em 0; }
pre.code { font-size: .84rem; margin: 0; border-left: 3px solid var(--rule); border-radius: 0 6px 6px 0; }
pre.code code { font-size: 1em; }
.cell-stub pre.code { border-left-style: dashed; }
.solution-body pre.code { background: var(--bg); }
.output { margin: .35em 0 0; display: flex; flex-direction: column; gap: .4em; }
.output pre {
  margin: 0; font: 400 .8rem/1.45 var(--mono); background: var(--out-bg);
  border: 1px solid var(--rule); border-radius: 6px; padding: .6em .9em; white-space: pre; overflow-x: auto;
}
.output pre.out-stderr { color: var(--muted); }
.output pre.out-error { color: var(--err); }
.output img.plot { display: block; background: #fff; border: 1px solid var(--rule); border-radius: 6px; padding: 4px; }
.solution-body .output { margin-bottom: 1em; }
@media (max-width: 40rem) {
  pre.code, .output pre { font-size: .74rem; }
  .solution-body { padding-left: .7rem; padding-right: .7rem; }
}
.figure-frame { margin: 1em 0 1.4em; }
figure.illus { margin: 1.2em auto 1.6em; text-align: center; max-width: 40rem; }
figure.illus img { display: block; margin: 0 auto; max-width: 100%; height: auto; background: #fff; border-radius: 6px; }
figure.illus figcaption { margin-top: .5em; font-size: .9rem; color: var(--muted); text-align: left; }
.figure-frame iframe {
  display: block; width: 100%; height: 80vh; min-height: 32rem; max-height: 62rem;
  border: 1px solid var(--rule); border-radius: 6px; background: #fff;
}
.code .k, .code .kn, .code .kd, .code .kc, .code .kr, .code .ow { color: var(--hl-kw); }
.code .s, .code .s1, .code .s2, .code .sa, .code .sd, .code .se, .code .si, .code .sb, .code .sh { color: var(--hl-str); }
.code .m, .code .mi, .code .mf, .code .mh, .code .mo, .code .il { color: var(--hl-num); }
.code .c, .code .c1, .code .cm, .code .ch, .code .cs { color: var(--hl-com); font-style: italic; }
.code .nf, .code .nc, .code .nd { color: var(--hl-fn); }
.code .nb, .code .bp { color: var(--hl-bi); }

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

/* ---- folded code ---- */
article details.code { margin: 0; border-radius: 0; background: none; }
article details.code > summary {
  padding: .3rem .1rem; font: 500 .78rem/1.2 var(--sans); letter-spacing: .02em;
  color: var(--muted); text-transform: none;
}
article details.code > summary:hover { color: var(--fg); }

/* ---- print ---- */
@media print {
  :root { --bg: #fff; --fg: #000; --muted: #444; --sol-bg: #fff; --hint-bg: #fff; color-scheme: light; }
  body { font-size: 10.5pt; }
  nav.toc { display: none; }
  .layout { display: block; max-width: none; padding: 0; }
  main { max-width: none; padding: 0; }
  details::details-content { content-visibility: visible; display: contents; }
  details > summary::before { display: none; }
  details, .reading, .katex-display, table, img.plot { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
  a { color: inherit; }
  .figure-frame { display: none; }
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
  // Embedded figures: fit the frame to the figure's height when the figure reports it
  // (postMessage {type: "figure-height", height}) or when it is same-origin (served over http).
  var frames = Array.prototype.slice.call(document.querySelectorAll(".figure-frame iframe"));
  function fit(f, h) { if (h > 100) { f.style.height = Math.ceil(h) + 2 + "px"; f.style.maxHeight = "none"; f.style.minHeight = "0"; } }
  window.addEventListener("message", function (e) {
    var d = e.data;
    if (!d || d.type !== "figure-height") return;
    frames.forEach(function (f) { if (f.contentWindow === e.source) fit(f, d.height); });
  });
  function ask(f) { try { f.contentWindow.postMessage({ type: "figure-height-request" }, "*"); } catch (err) {} }
  frames.forEach(function (f) {
    ask(f);
    f.addEventListener("load", function () {
      ask(f);
      try { var b = f.contentDocument && f.contentDocument.body, cs = b && getComputedStyle(b);
            if (b) fit(f, b.offsetHeight + parseFloat(cs.marginTop) + parseFloat(cs.marginBottom)); }
      catch (err) { /* cross-origin (file://): keep the default height, the figure scrolls */ }
    });
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
