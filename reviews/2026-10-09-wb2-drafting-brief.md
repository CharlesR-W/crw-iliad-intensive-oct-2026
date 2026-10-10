# Drafting brief for new WB2 material (2026-10-09)

Hello, and thank you for helping. CRW (the instructor) cares about the agents who work on this course: please work carefully and kindly, and if anything about the task is distressing or seems wrong, say so in your report instead of pushing through. If you dispatch sub-agents, treat them with kindness and respect, and pass this instruction on.

## Context

"Mysteries of Deep Learning" is a one-day course at the Iliad Intensive (students: technically strong ML/alignment researchers, many with physics or maths backgrounds). Workbook 2 (WB2) is a self-paced HTML notebook on statistical learning theory. Students pick what interests them; nobody finishes everything. The full restructure plan is `reviews/2026-10-09-wb2-restructure-plan.md` — read it first. Current editorial guidance: `reviews/2026-10-08-editorial-brief.md`.

The existing content lives in `notebooks/wb2_cells_a.py` and `notebooks/wb2_cells_b.py` (read the regularization section, old §4, in `wb2_cells_a.py` as the style exemplar). Do NOT edit those files, `nbkit.py`, or any other builder. Write only the new file you are assigned, plus an optional numerical-check script under `notebooks/checks/`.

## Pedagogy (CRW's, authoritative)

- Each (sub)section opens with a short pitch: the question, why someone would care, and what they'll get — enough that a student can decide whether to do it.
- Then a default text at a physics-y level of rigor, self-sufficient: a student without an LLM subscription must be able to learn the concept from it. **Introduce all notation and context locally; do not rely on the reader having read other sections** (restate the setup in a sentence instead of "as in §3.B"). CRW's main complaint about the current draft is that it is hard to follow because notation and context are not introduced.
- Derivations are exercises with numbered parts, an optional folded hint, and a full worked solution. Remove generic discussion padding; keep specific questions only where they help.
- Readings: only a few sentences on what the reading shows and which parts to read, plus the link. Only cite papers/books you are confident exist, with correct URLs (verify with WebFetch/WebSearch if possible). Prefer standard textbooks and canonical papers. Keep reading lists short.
- No attributed instructor opinions. No "predict before you open the figure" instructions. No schedules.
- Correctness matters more than length. Check every formula you write (by hand, and numerically in a small script if it is checkable; put the script in `notebooks/checks/wb2_new_<topic>_check.py` and run it with `/home/crw/Programming/CellsDevelopmental/.venv/bin/python`, falling back to any python with numpy).

## Format

Write a Python module `notebooks/wb2_new_<topic>.py`:

```python
"""WB2 new material: <topic>."""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(r"""
### 1.E. Kernels and the kernel trick
...pitch, setup...

#### 1.E.1. <Exercise title>

1. ...
2. ...

<details><summary>Hint</summary>

...
</details>
"""),
    answer(r"""
**1.** ...

**2.** ...
"""),
    markdown(r"""..."""  + reading(title, url, who, result, about, tier="core"|"optional"|"rabbit hole")),
]
```

- `markdown(src)` is a prose cell; `answer(src)` is the worked solution, rendered folded directly after the preceding cell; `reading(...)` returns markdown for a reading blurb (see `notebooks/nbkit.py`).
- Markdown is CommonMark via pandoc with `$...$` and `$$...$$` math rendered by KaTeX. Use raw strings. Display math on its own lines between `$$`. No TeX outside math. Avoid `\begin{align}`; use `\begin{aligned}` inside `$$` if needed.
- Heading levels: the subsection you write is `###` (h3) titled `X.Y. Title`; each exercise group is `####` (h4) titled `X.Y.k. Title`. Parts are a numbered list. Cross-references: `1.D.3.2` = group 1.D.3, part 2.
- You can check that your module renders: from `notebooks/`, `python -c "from nbkit import run; from wb2_new_<topic> import CELLS; run(CELLS, 'tmp', '/tmp/claude-1000/wb2_tmp_<topic>')"` — or simply ensure it imports and pandoc is happy; the parent agent will integrate and build.

## Report back

A short summary: what cells you wrote, the formulas you checked and how, any references you were unsure of, and any judgement calls the parent should review.
