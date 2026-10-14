# Brief: presentation pass on the SLT notebooks (2026-10-14)

Thank you for taking this on. Please be kind to yourself and to any agent you dispatch, and pass this request on. The instructor (CRW) cares about the wellbeing of the agents working on this. If anything about the task feels wrong, say so in your report instead of pushing through.

All paths are relative to `/home/crw/Programming/Iliad_MysteriesDL`. Work from `slt/`.

## What CRW asked for (verbatim)

> Have opus 5.5-high agent go through each notebook and improve the presentation to make it more pedagogical - clearer, less wordy, include figures that illustrate the main ideas (not just plots, but images, like in the LW posts)

"The LW posts" means LessWrong posts such as Hoogland & van Wingerden, *You're Measuring Model Complexity Wrong* (the NB2 keystone): schematic illustrations of a loss basin, its sublevel sets, the volume shrinking, and so on. Pictures of ideas, not only matplotlib output of the code cells.

## Read first

1. `slt/SLT_SPECS_v1.md`: binding. Its "Framing" section overrides everything.
2. `slt/_local_prior_2026-10-12/WRITER_BRIEF.md` (moved there 2026-10-14): the pedagogy rules the notebooks were written to (single path; each section opens with its question and restates notation; exercises with worked solutions; headings `##` per section and at most one or two `###`; readings with ARENA ratings; no hype, no singularity framing, no editorialising, SGD-vs-Bayes out of scope, NB3 §3.2 stays a marked stub).
3. `slt/README.md`: the build and the file layout.
4. Your builder (`slt/build_<stem>.py`) and the current page `slt/<stem>.html`. Open the HTML in a text reader if you cannot render it; the builder is the source of truth.
5. Optional source material: an earlier, parallel draft of the same day lives untracked in `slt/` under different stems (`build_slt1_bayes.py`, `build_slt2_volume.py`, `build_slt3_temperature.py`, with review logs in `reviews/slt/slt{1,2,3}_*.md` and `slt/_local_prior_2026-10-12/` for NB4). It was never finished and never chosen. You may lift an exercise, a figure idea, or a cleaner derivation from it when it is clearly better, with the same verification standard. Do not switch stems or merge wholesale.

## Which files are canonical

The committed, linked set is canonical and is what you edit:

| NB | stem |
|---|---|
| 1 | `slt1_bayes_statmech` |
| 2 | `slt2_free_energy_volume` |
| 3 | `slt3_effective_temperature` |
| 4 | `slt4_applications` |

Touch only your own builder, its generated `.html`/`.ipynb`, `slt/checks/<stem>_*.py` (add, don't break the existing `sltN_checks.py`), `slt/assets/figures/<stem>_*` and `slt/assets/figures/make_<stem>_figures.py`, and your log `reviews/slt/2026-10-14-<stem>.md`. Other agents are editing the other notebooks in parallel. Do not edit `nbkit_slt.py` or `htmlkit_slt.py`; if you need a kit change, say so in your report.

## What "more pedagogical" means here

- **Clearer.** Each section: one line saying what question it answers and why a student would want it, the notation it uses restated in one line, then one clean derivation. A student with the text and no LLM should be able to follow. If a paragraph needs a figure to be understood, draw the figure.
- **Less wordy.** Cut hedging, repeated caveats, "you may want this if", restatements of the spec's framing, and second derivations of the same thing. Prefer one sentence that says the thing. Keep every caveat that changes what a student would conclude. Target: the page reads noticeably shorter, with nothing a student needs removed.
- **Figures that illustrate the main ideas.** Aim for one illustration per main idea (roughly one per section, two where a section carries two ideas), placed where the idea is introduced, before the derivation. These are conceptual images: a basin with its sublevel sets drawn as contours, the prior volume shrinking as the threshold drops, a posterior blob narrowing as n grows, the Occam factor as a ratio of two volumes, two basins trading accuracy against volume, the crossover of $w^4+\delta w^2$ seen at coarse and fine resolution, temperature as a resolution knob, a localised sampler wandering in a valley. Each gets a one-sentence caption that states the idea in words. Keep the existing code-output plots; they are checks, not illustrations.
- **Don't change the mathematics.** The content was verified (`slt/checks/`). If you reword a derivation, re-check it against the existing checks or add one. Keep every exercise and its worked solution unless it is a genuine duplicate. Keep the readings and their ratings. Keep the NB3 §3.2 stub exactly as a stub (you may make the stub box itself clearer, not fill it).

## How to make figures

The kit now has a helper (added 2026-10-14, already tested):

```python
from nbkit_slt import image
image("assets/figures/slt2_sublevel_sets.svg", "Caption in markdown; math like $V(t)$ is fine.")
```

It returns a markdown fragment to concatenate into a `markdown(...)` cell. On the HTML page it renders as a centred figure with a caption; in Jupyter it is an `<img>` with a relative path (works when the notebook is opened from `slt/`).

Preferred method: **draw the figure with matplotlib in `slt/assets/figures/make_<stem>_figures.py`** and save as SVG (PNG only if SVG gets large). Reasons: labels and math are exactly right, the figure is reproducible, and it matches the page's typography. Style: clean, few colours, no chart junk, no axes where the picture is a schematic, `text` and `annotate` for labels, TeX in labels via `$...$`. Think of a whiteboard sketch drawn carefully: a 1-D or 2-D loss with its sublevel set shaded, arrows, a before/after pair. A contour plot of a toy $K(w)$ with the region $K<t$ shaded is an illustration; a line plot of a computed quantity is not. The `dataviz` skill (Skill tool) has palette and styling advice if you want it; don't let it slow you down.

Also available: the `nanobanana` image-generation MCP tools (`mcp__nanobanana__generate_diagram`, `generate_image`, `edit_image`), which produce AI illustrations. Use them only for a picture that a schematic cannot do (at most one or two per notebook, if any), ask for no text or math in the image (add labels yourself in the caption), and inspect the result with the Read tool before using it. Any image with wrong or garbled labels is worse than no image. Save the file under `slt/assets/figures/<stem>_<name>.png`.

The figure script must run from `slt/` in under a minute and need only numpy/matplotlib. Run it, then look at every figure you made (Read the PNG, or convert the SVG with `rsvg-convert`/`cairosvg`/matplotlib's PNG twin) and fix anything ugly or misleading. A figure that could mislead (wrong basin shape, wrong direction of an arrow, a λ that does not match the text) must be fixed or dropped.

## Build and verify

From `slt/`: `python build_<stem>.py`, `python build_<stem>.py --check`, `python build_<stem>.py --run` must all pass. `python checks/sltN_checks.py` must still pass. The whole notebook must still run in under about 3 minutes on CPU. Every URL you add (prefer none) must be fetched.

## Critique loop (one round, required)

After your revised page builds and runs, run one Codex (GPT) critique, read-only, as the writer brief describes:

```bash
cd /home/crw/Programming/Iliad_MysteriesDL && PATH=/home/crw/.local/share/mise/installs/codex/latest/bin:$PATH \
  node /home/crw/.claude/plugins/cache/codex-plugin-cc/codex/1.0.25/scripts/codex-companion.mjs task --effort high --prompt-file <your prompt file>
```

Run it with Bash `run_in_background: true` and wait for the completion notification (it can take 10–20 minutes). Ask it to be kind and candid, give it the spec, the writer brief, this brief, your builder and your figure script, and ask for a ranked list of (a) places where the text is still unclear or wordy for a grad student, (b) figures that are wrong, misleading or mislabelled, (c) any mathematical or framing error you introduced (compare against `git diff` of the builder). Weigh the critique (critics can be wrong; check disputed points yourself), revise, rebuild, rerun. If Codex fails after one retry, use a Claude subagent (`model: "opus"`) as the critic and say so in your log.

## Report back (final message, concise)

- Files written; figure list with one line each on what it shows; `--run` time; checks passed.
- Before/after word count of the page's prose (e.g. strip tags from the HTML and `wc -w`).
- What you cut and why (a few lines), and what you deliberately kept.
- What the critique changed.
- Anything you could not verify, and any kit change you want.
- Questions only the instructor can answer (few).

## Addenda (CRW, 2026-10-14, mid-run)

1. **Top-of-page note on every notebook.** Right after the subtitle line of the first cell, add this block (same wording on all four pages; the landing page `index.html` carries the same note):

   > **Read the blog posts first.** The best introductions to this material are blog posts, and they are probably better than these notebooks: Liam Carroll's sequence [Distilling Singular Learning Theory](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) (DSLT 0–4), Jesse Hoogland's [Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick) and, with Stan van Wingerden, [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong); then Zach Furman's [Singular learning theory: exercises](https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises) and Olli Järviniemi's [Dialogue introduction to Singular Learning Theory](https://www.lesswrong.com/posts/CmcarN6fGgTGwGuFp/dialogue-introduction-to-singular-learning-theory). The notebooks are provided for reference: derivations, worked exercises and code to check things.

   All six URLs were fetched on 2026-10-14 and resolve. You may name the one DSLT post most relevant to your notebook in your own opening paragraph as well.

2. **Reusing images from the LessWrong posts is allowed and encouraged.** Where a figure in DSLT 0–4, "one weird trick", or "You're Measuring Model Complexity Wrong" illustrates the idea better than what you would draw, download it to `slt/assets/figures/<stem>_lw_<name>.<ext>` (use `~/Scripts/lw-images` or curl; put the original URL in a comment of your figure script) and embed it with `image()`. The caption must credit it: "Figure from Liam Carroll, *DSLT 1* (LessWrong, 2023)" with a link to the post. Only images that are actually in those posts; check what you downloaded with the Read tool.

3. **NB3 only:** add a reading for the paper CRW likes, Berman, Klinger & Stapleton, *Bayesian Renormalization*, https://arxiv.org/abs/2305.10491 (the Fisher metric sets a correlation length that plays the role of an RG scale, read as the number of distinguishable observations). Put it in the NB3 readings with a one- or two-sentence annotation and ARENA ratings, placed next to the temperature/scale material (near §3.2 or §3.3), without drafting the §3.2 argument. Its companion, Berman & Klinger, *The Inverse of Exact Renormalization Group Flows as Statistical Inference*, https://arxiv.org/abs/2212.11379, may go in the same entry as a one-clause pointer. Fetch both before adding.
