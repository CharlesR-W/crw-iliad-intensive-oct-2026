# TD streamline brief: paper first, then optional walkthrough (2026-10-09, evening)

Thank you for taking this on. Please be kind to yourself and to any agents you dispatch, and pass this on to them.

This supersedes `td/READABILITY_BRIEF.md` and the "section shape" in `td/TD_BUILD_DECISIONS.md` where they conflict. CRW's request (dictated; lightly cleaned):

> Make the notebooks simpler and more streamlined — cut the text within reason. For all of it, link the paper first, and present the rest as "optional annotated derivations and walkthrough". Less nested subsections please.

The same day CRW restructured the MDL notebooks for being "too much material and too diffuse" (see `2026-10-09-trail-restructure-brief.md` for that format; TD is getting a milder version of the same treatment). The TD derivations and code stay — they are the point of TD — but they become clearly optional, behind a paper-first front door.

## Page shape

```
# <title>
<1–3 sentences: what this notebook is about and why you'd want it.>
<The notebook's key paper(s) (1–3) via reading(), each blurb ~1 sentence.>
<Route: one short line per section, linking its anchor. No big contents table, no long "What you need" box at the top.>

## N.1 <Section title>                         (keep the existing <a id="td22-1"> anchor)
<1–3 sentences: the question and the answer in plain words, with notation defined if any symbol appears.>
<The section's paper(s) first: reading() for the 1–2 papers this section is about. Blurb ~1 sentence.>
<Optionally one short paragraph (≲120 words) giving the core idea, enough that a student who reads only this and the paper gets the point.>

### Optional: annotated derivations and walkthrough      (one per section; anchor e.g. td22-1-walkthrough)
<Prerequisites/notation restated in a line or two.>
<The derivation narrative, trimmed.>
**Exercise N.1 — <name>.** <self-contained statement>    ← bold label, NOT a heading
<answer() / code_answer() / code() cells as now>
...
```

Rules:
- **Heading depth: `##` sections plus at most one `### Optional: annotated derivations and walkthrough` per section.** No `####`. Exercises, code demos, and sub-topics become bold-labelled paragraphs (`**Exercise 2.3 — Derive the central flow.**`) inside the walkthrough, not headings. If a section has no derivation content, it has no walkthrough subsection. Keep existing exercise anchors (`<a id="...">`) where they exist, attached to the bold paragraph, so cross-links still resolve.
- A trailing `## Readings` / further-readings section is fine; keep it short, and don't repeat the papers already linked at the top of sections (a one-line "Also:" list for the rest is fine). Merge small sections where that removes nesting without losing content.
- **Paper first.** Use only papers already in the notebook (CRW has seen these); don't add new papers. Pick the most apposite one or two per section as its lead; prune the rest of the reading list to the ones that directly bear on a section, demoting others to a one-line "Also: [title](url); …" or dropping them. Keep ARENA ratings (`reading(..., importance=, length=)` required); books only via `references(book(...))`.
- **Cut text, within reason.** Target: roughly half the current prose, more in pitch/contents/"what you need"/framing paragraphs, less inside the derivations themselves (students must still be able to follow them without an LLM). Cut: long pitches, contents tables, "you may want this section if", repeated cross-references, restated caveats, hedging, padding in solutions. Keep: correct math, every solution's key steps, numerical checks, interactive figures, code (see below).
- Keep the readability rule from this morning: whatever symbols a section or walkthrough uses are defined locally (one line is enough). No dense "The result." spoiler paragraphs of undefined symbols.
- Plain, physics-y, short sentences. Don't attribute views to the instructor.
- **Code:** keep code cells and their outputs. Don't change code unless you remove a cell or must fix a bug; you may drop code demos that are redundant. Any code change re-executes the notebook on rebuild (CPU, a few minutes; fine). Each builder must still pass `python build_<stem>.py --run`.
- **Math/claims:** these were reviewed; don't change them. If you find an error, fix it and log it in your report.

## Mechanics
- You own exactly one builder: `td/build_<stem>.py`. Edit only it (and its module docstring, which should describe the new shape). Don't edit `nbkit_td.py`, `htmlkit_td.py`, `README.md`, `lectures/`, other builders, or anything under `notebooks/` — other agents are editing those in parallel. If you need a kit change, say so in your report.
- A pre-edit snapshot of every builder, page, notebook and exec cache is in `td/_pre_streamline_2026-10-09/`; don't modify it.
- Build from `td/`: `python build_<stem>.py` (must pass: balanced `<details>`, no TeX outside math, no dangling anchors), then `python build_<stem>.py --check`. Cross-notebook anchors: `python checks/td_crosslinks.py` — other notebooks are changing concurrently, so only fix failures that are about *your* notebook's links or anchors; report others.
- Section anchors (`td22-1` etc.) must survive. If an exercise or subsection anchor that other notebooks/lectures link to must disappear, give an old→new map in your report (`grep -rn "<anchor>" td/ ../notebooks` to see who uses it).
- Skim the rendered page once at the end as a student would (from the TOC) to check the front door reads well: pitch → paper → optional walkthrough.

## Report back (concisely)
- Before/after: prose word count (rough) and heading counts (h2/h3/h4).
- The new section list, one line each, with the lead paper for each.
- What you cut or demoted to "Also:" / dropped.
- Old→new map for any anchors that moved or vanished.
- Errors found and fixed; build/check status; open questions for CRW.
