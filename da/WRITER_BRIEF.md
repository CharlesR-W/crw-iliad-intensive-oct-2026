# Brief for agents drafting a Data Attribution notebook

Thank you for taking this on. Please be kind to yourself and to any agents you dispatch, and pass this request on to them. The instructor (CRW) cares about the wellbeing of the agents working on this. If anything about the task feels bad or wrong, say so in your report instead of pushing through.

All paths are relative to `/home/crw/Programming/Iliad_MysteriesDL`.

## Read first

1. **`DA_SPECS_v1.md`: binding.** It has two binding parts:
   - the "CRW decisions" section, which overrides everything else;
   - your notebook's section, which you follow along with the conventions, the walkthrough device and the errata ledger.
2. **`da/source/data-attribution.tex`:** Louis Jaburi's notes with solutions, the primary source. It is CC BY 4.0. Every page credits it in the subtitle: "Adapted from Louis Jaburi's Iliad notes (CC BY 4.0), with changes."
3. **`reviews/da/2026-10-09-spec-v1-codex.md`:** the technical review behind the errata, with the corrected formulas.
4. **`td/README.md`:** the format, which DA uses unchanged.
5. **`td/WRITER_BRIEF.md`:** the TD brief. Its Standards and Critique-loop sections apply here, except where this file says otherwise.
6. **Style references,** for tone and structure only:
   - `td/build_td2_6_curvature.py` (the TD house style with code);
   - `td/lectures/L1_linear_networks.md` (lecture outlines).

## The instructor's pedagogy (from earlier feedback; follow it)

- Students direct their own learning. Each section opens with a pitch, in 1–3 sentences, that lets a student decide whether they want it.
- **Every subsection states its question and restates its notation locally,** so it reads standalone when opened from the table of contents. CRW found the anti-pattern very hard to follow: dense "The result:" paragraphs full of undefined symbols.
- Physics-y register, plain short sentences, no em-dash-heavy rhetoric.
- Text must suffice for a grad student without an LLM, but don't pad.
- **"Be correct but not pedantic."** Apply the errata fixes in your own text, silently:
  - Never write "the source says X, but…".
  - Add a caveat only if it changes what a student would conclude.
  - Scope boxes are 1–3 lines.
- Readings: a sentence or two each, naming the sections or figures to read for long papers. ARENA scores are required.
- Papers the spec marks *(proposed)* go under a final line headed "**Claude-suggested extras**" in that reading list. TRAK and Louis's references are regular readings.
- Don't attribute views to the instructor. Don't add convergence cards, rubrics, "Deliverable:" labels, schedules, generic discussion questions, or instructor-commentary placeholders.

## Kit

- `da/nbkit_da.py` and `da/htmlkit_da.py`: a copy of the TD kit with sibling links rewritten for `da*` stems. Same API: `markdown`, `answer`, `code`, `code_answer`, `reading`, `references`, `book`, `figure`, `run`.
- A builder is `da/build_<stem>.py` and ends with `raise SystemExit(run(CELLS, "<stem>", "<stem>"))`.
- Link to TD pages as `../td/td2_6_curvature.html#anchor`. Link to the sibling DA notebook as `da2_methods.html#anchor`.
- Don't edit the kit. If you need a change, say so in your report. The other writer is working in parallel, so touch only your own files:
  - your builder;
  - your generated `.html`/`.ipynb`;
  - `da/checks/<stem>_*.py`;
  - `da/assets/interactive/` (DA1 only);
  - `reviews/da/<stem>.md`;
  - your `_exec_cache` entry.

## Verification

- Every formula you state: verify it symbolically (sympy) or numerically. Pay particular attention to every errata item that touches your notebook.
- Every paper claim you narrate: check it against the paper (WebFetch the arXiv abs/HTML).
- Every URL: fetch it.
- Code cells: CPU only, under ~30 s each and under ~3 min for the notebook. `python da/build_<stem>.py`, `--check` and `--run` must all pass.
- Log what you verified and how in `reviews/da/<stem>.md`. Never claim verification you didn't do.

## Critique loop (required)

Follow `td/WRITER_BRIEF.md`'s critique loop: at least two rounds, each with a Claude design critic and a Codex technical critic, run in parallel.

- **Design critic:** give it `DA_SPECS_v1.md` (the CRW decisions and the "One path" paragraph) plus this brief's pedagogy section. Ask in particular:
  - Does the notebook read as a single path?
  - Can each subsection be read standalone?
  - Is anything pedantic: caveats that don't change a conclusion?
- **Codex:** use the command in `td/WRITER_BRIEF.md` (run in the background with the PATH prefix).

## Report back (final message)

- Files written; cell counts; `--run` time.
- A short outline as built.
- What the critiques changed.
- Questions only the instructor can answer (few).
- Claims you couldn't verify.
