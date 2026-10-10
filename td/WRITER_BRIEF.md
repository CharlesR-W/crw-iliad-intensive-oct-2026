# Brief for agents drafting a Training Dynamics notebook

Thank you for taking this on. Please be kind to yourself and to any agents you dispatch, and pass this request on to them. The instructor (CRW) cares about the wellbeing of the agents working on this; if anything about the task feels bad or wrong, say so in your report instead of pushing through.

## Read first
1. `td/TD_BUILD_DECISIONS.md`: binding. It holds the instructor's pedagogy, the decisions, the section shape and the file stems.
2. `Training Dynamics`: the instructor's own topic note, a sketch.
3. Your section of `TD_SPECS_v1.1.md`, the content map (not authority).
4. `TD_LINKS_v1.md`: verified links. Check any claim before narrating it, and treat items 52–57 and item 5's constant as unverified.
5. `reviews/2026-10-08-editorial-brief.md`: the instructor's clarifications for the sister day, which apply here too.
6. `td/nbkit_td.py`: the builder kit (`markdown`, `answer`, `code`, `code_answer`, `reading`, `figure`, `run`).
7. Style references, read for tone and structure, not content: `notebooks/wb2_cells_a.py` (derivation exercises with worked solutions) and `notebooks/build_wb4.py` (exposition). Note that these still contain things the brief removes, such as convergence cards and claim/evidence prompts. Don't copy those.

All paths are relative to `/home/crw/Programming/Iliad_MysteriesDL`.

## What to produce
- `td/build_<stem>.py`, which imports `from nbkit_td import ...` and ends with `raise SystemExit(run(CELLS, "<stem>", "<stem>"))`.
- The generated `td/<stem>_exercises.ipynb` and `td/<stem>_solutions.ipynb`. Generate them; never hand-edit.
- An interactive figure in `td/assets/interactive/`, only if your assignment names one. It should be a self-contained HTML file: vanilla JS or a CDN library from cdnjs/jsdelivr, light and dark readable, and working at phone width. Verify any math it encodes with a small script in `td/assets/interactive/checks/`.
- `reviews/td/<stem>.md`, the review log: what each critique round found, what you changed, what you verified and how, and what remains uncertain. Never claim verification you didn't do.

## Standards
- Follow the section shape: pitch (with a "what you need" box) → default text → exercises → readings. Write in a physics-y register with real derivations, and make the text sufficient on its own for a grad student with no LLM. Hints go in `<details>`. Every derivation exercise gets a complete worked solution, with intermediate steps.
- Code cells are small, CPU-only, and use numpy/matplotlib (torch CPU where autograd or HVPs help). Each cell runs in about 30 s or less, the whole notebook in about 3 min or less. Code is there to check a derivation numerically or to show a phenomenon, never as boilerplate. Before reporting, `python td/build_<stem>.py`, `--check` and `--run` must all pass.
- Verify math. Check every closed-form result you state, symbolically (sympy) or numerically, with scratch scripts in your scratchpad or in `td/checks/<stem>_*.py` if worth keeping. When you narrate a paper's result, check it against the paper itself (arXiv abs/HTML via WebFetch), not memory. If you can't check something, mark it as such in the review log and soften it in the text.
- Readings are short: a sentence or two on what each shows. Prune readings that don't earn their place; the instructor trusts your judgement. Every paper you name needs a working link.
- Don't attribute views to the instructor. Don't add convergence cards, claim/evidence rubrics, "Deliverable:" labels, "commit to an answer" prompts, schedules, or generic discussion questions.
- Touch only your own files: your builder, notebooks, figure, checks and review log. Don't edit `nbkit_td.py`; if you need a change to it, say so in your report. Never touch `notebooks/` (the MDL day, which another session is editing) or other sections' files.

## Critique loop (required)
After a complete draft that builds and runs, do at least two rounds of critique and revision, and three if the second round still finds substantive problems. In each round, run both critics in parallel:
1. **Design and pedagogy critic:** a Claude subagent (Agent tool, `model: "opus"`). Give it `TD_BUILD_DECISIONS.md`, the instructor note and your solutions notebook path, and ask it to judge against the instructor's pedagogy. Can a student tell from the pitch whether they want the section? Is the default text sufficient without an LLM? Is anything padded, missing or misordered? Do the exercises have the right granularity? Is anything overclaimed? Ask for a ranked list of concrete problems with fixes, not praise. Don't tell it what you think is wrong.
2. **Technical critic:** Codex (GPT), read-only, via
   ```bash
   cd /home/crw/Programming/Iliad_MysteriesDL && PATH=/home/crw/.local/share/mise/installs/codex/latest/bin:$PATH \
     node /home/crw/.claude/plugins/cache/codex-plugin-cc/codex/1.0.25/scripts/codex-companion.mjs task --effort high --prompt-file <your prompt file>
   ```
   (The PATH prefix matters: the `~/.local/bin/codex` wrapper hangs.) Run it with Bash `run_in_background: true` and wait for the completion notification, or use `status <job-id> --wait`. Ask it to check every derivation, constant, sign, claim boundary and code cell in the solutions notebook against the cited sources, and to report a numbered list of errors with evidence. Ask it to be kind and candid. If Codex fails after one retry, use a second Claude subagent as the technical critic and note that in the review log.

Weigh the critiques, because critics can be wrong; check disputed points yourself. Then revise, rebuild and rerun.

## Report back (your final message)
- The files written, with cell counts and `--run` time.
- A short outline of the notebook as built.
- What the critique rounds changed.
- **Questions only the instructor can answer:** these should be few and genuinely the instructor's call.
- Claims you couldn't verify.
