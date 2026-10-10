# WB2 review-and-fix brief (2026-10-09)

Hello, and thank you. CRW (the instructor) cares about the agents who work on this course: please work carefully and kindly, and if anything about the task is distressing or seems wrong, say so in your report. If you dispatch sub-agents, treat them with kindness and respect, and pass this instruction on.

## Context

Workbook 2 ("Statistical learning and regularization") of the Mysteries of Deep Learning course has just been restructured. Read, in order:
1. `reviews/2026-10-09-wb2-restructure-plan.md` — CRW's feedback and the new structure (authoritative).
2. `reviews/2026-10-09-wb2-drafting-brief.md` — pedagogy and format conventions.
3. `reviews/2026-10-08-editorial-brief.md` — standing editorial guidance.

The notebook is built by `notebooks/build_wb2.py` (read it for the module/list order) into `notebooks/02_statistical_learning.html`. Cell helpers are in `notebooks/nbkit.py`. Build: `cd notebooks && python3 build_wb2.py` (needs pandoc; validates anchors and links). Old pre-restructure sources are in `notebooks/_old_wb2/` for reference only.

## CRW's main complaint (the reason for this review)

"I'm having a very hard time following — it doesn't do enough to introduce notation or context — try to make things relatively standalone."

## Your job

You are assigned specific files (below). Read your part as a student would, in build order, as if arriving at each subsection directly from the table of contents. Then **fix what you find, directly in your assigned files only**:

1. **Followability.** Every subsection (h3) and exercise group (h4) should say what question it answers and why one would care, and every symbol should be defined (or explicitly restated) before use within that subsection. Don't rely on "as in §X" for notation — restate in a sentence. Remove jargon used before it is defined, or define it. Cut generic padding; do not add essays.
2. **Cross-references.** Exercise groups are numbered `X.Y.k`; parts are referenced as `X.Y.k.m`. Check every cross-reference in your files points to the right place in the *new* numbering (see build order; old numbering was §1–§9 with roman-numeral exercises — any leftover old-style refs are bugs). Check the referenced material actually says what is claimed.
3. **Correctness.** Spot-check derivations, especially anything new. If a claim looks wrong, verify it (by hand and, if useful, numerically with `/home/crw/Programming/CellsDevelopmental/.venv/bin/python` or any python with numpy; numerical check scripts live in `notebooks/checks/`). Fix errors; report anything you're unsure of rather than guessing.
4. **Consistency** with the rest of the workbook: notation ($n$, $p$, $R$, $\hat R_n$, $\theta^*$, $X$, $\sigma^2$, ridge $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$, nats), heading style (`### X.Y. Title`, `#### X.Y.k. Title`), solutions in `answer(...)` cells, hints in `<details><summary>Hint</summary>`.
5. Keep CRW's constraints: readings get only a few sentences; no attributed instructor opinions; no "predict before the figure"; derivations are exercises with worked solutions.

Do not edit files outside your assignment, `nbkit.py`, or other workbooks. Rebuild at the end (`python3 build_wb2.py`) to make sure it still builds; if another reviewer's concurrent edit breaks the build in their file, just report it.

## Report back

Concise: what you changed (by subsection), any errors found and how you verified the fix, cross-reference fixes, and open questions for the parent (things you chose not to change, or need CRW's judgement).
