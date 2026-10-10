# Readability pass for the TD notebooks (2026-10-09)

Thank you for taking this on. Please be kind to yourself and to any agents you dispatch, and pass this on to them.

## Why
On 2026-10-09 the instructor read the MDL day's WB2 draft and found it "very hard to follow". That feedback applies to TD as well:

- **Each subsection must stand alone when reached from the table of contents.** It states its question and context in plain words first, then defines or restates the notation it uses, locally, even if an earlier subsection already defined it. A student who jumps in from the sidebar must not meet an undefined symbol.
- **Avoid dense "The result." spoiler paragraphs.** These front-load a subsection's conclusions in compressed notation full of undefined symbols. Replace them with a short plain-language statement of the question, plus the setup, then build to the result. Where a result summary helps, give it after the notation is defined, in words first.

## Context
- The TD notebooks now build as one HTML page per notebook, `td/<stem>.html`, with folded solutions and pre-rendered code outputs, plus one runnable `td/<stem>.ipynb`. See `td/README.md`.
- Build with `python td/build_<stem>.py`. Code is re-executed only if code cells changed, using the cache in `td/_exec_cache/`. Use `--check` to validate and `--run` to execute the code.
- Binding pedagogy is in `td/TD_BUILD_DECISIONS.md`; standards are in `td/WRITER_BRIEF.md`.

## Your job, on one notebook
1. **Read the rendered page** `td/<stem>.html` (or the builder) as a student would, subsection by subsection. Then rewrite in the builder:
   - Open each `##`/`###` subsection with its question or context in one or two plain sentences.
   - Restate the notation it uses: a short "Notation here:" line or an inline definition at first use within the subsection.
   - Convert or remove "The result." spoiler paragraphs as above.
   - Don't make the notebook longer than it needs to be. Restating notation can be one line. Cut where the text is padded.
   - Don't change the mathematics, the code, or the claims. Those were reviewed. If you find an error, fix it and log it.
2. **Fix wording left over from the old two-file format.** Remove references to "the solutions notebook/version", "the exercises variant" and similar. The student now sees one page with folded solutions, plus a runnable notebook.
3. **Fix links into the MDL day** (`../notebooks/*.html`). That day was restructured this morning; WB2 now has only four sections, `wb2-section-1`…`4`. Find the current home of each piece of MDL content we cite by reading the current HTML or `notebooks/build_wb*.py` / `wb2_cells_*.py`. Repoint the link, or drop the reference if the content no longer exists. Prefer section-level anchors (`#wbN-section-K`) plus a named description, because the MDL day may keep changing. Don't edit anything under `notebooks/`.
4. **Rebuild and check.** Run `python td/build_<stem>.py`, then `--check`, then `python td/checks/td_crosslinks.py`. All must pass, except warnings about MDL anchors that you have deliberately addressed.
5. **Get one fresh readability critic.** Use a Claude subagent with `model: "opus"`. Give it this brief, `TD_BUILD_DECISIONS.md` and the rendered page path. Ask it to read the page as a grad student from physics, maths or TCS who jumps into three random subsections from the sidebar, and to report each place where they hit an undefined symbol, an unstated question, or a dense spoiler. It should also flag anything padded. Apply its findings, then rebuild and check again.
6. **Keep a log.** Append a "Readability pass (2026-10-09)" section to `reviews/td/<stem>.md` listing what changed.

Edit only your own notebook's builder and review log. The exception is the `td2_8` agent, which also owns `nb()` in its own builder. Don't edit `td/nbkit_td.py` or `td/htmlkit_td.py`; report needed kit changes instead.

## Final message
- What changed, in a few bullets.
- The critic's main findings and how you handled them.
- Any MDL references you dropped or repointed.
- Anything needing the instructor.
