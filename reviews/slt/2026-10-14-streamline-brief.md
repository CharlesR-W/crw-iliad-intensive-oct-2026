# Brief: streamlining pass on the SLT notebooks (2026-10-14, afternoon)

Thank you for coming back to this. Be kind to yourself and to any agent you dispatch, and pass this on. If anything feels wrong, say so in your report rather than pushing through.

Paths relative to `/home/crw/Programming/Iliad_MysteriesDL`; work from `slt/`. Same files as this morning's pass (your builder, its `.html`/`.ipynb`, your figure script, your checks, your log). The presentation brief (`reviews/slt/2026-10-14-presentation-brief.md`) still applies except where this one overrides it.

## CRW's feedback (verbatim)

> okay I still think the notebooks are too long, too wordy. Can you please simplify them a bit? I think my main problem at the small-scale is with density, while at the large scale, I think they're too long compared to the material. Please streamline them, including removing the within-notebook reading recs. For the slt notebooks for now.
>
> Please mark the notebooks as 'backup' over the blogposts. I'll separately write one on EFT
>
> (so please remove the EFT hangers from old notebooks)

## What to do

**1. Large scale: the page should be short compared with its material.** Current prose, code stripped: NB1 ≈ 4400 words, NB2 ≈ 4300, NB3 ≈ 3500, NB4 ≈ 1500. Targets: NB1 and NB2 at most 2200 words, NB3 at most 1800, NB4 at most 900. Hit the target by removing, not compressing:

- Keep one clean derivation per idea. Where a section has a derivation in the text and the same derivation again as an exercise with a solution, keep one of them (prefer the exercise if the result is the section's point, the text if it is a stepping stone).
- Exercises: keep the ones that carry the section's main result or a Fermi estimate a student would enjoy. Drop parts that are variations, generalisations, or "also show that". A section should have one exercise with two to four parts, not six. Solutions: the key steps and the result, in a few lines each; no running commentary.
- Drop every caveat that does not change what a student would conclude, every "where the analogy breaks" list beyond one line, every "note that", every restatement of the framing, and every sentence that says what the next sentence is about to say.
- Keep all the illustrations and their one-sentence captions (shorten captions that run past a sentence). Keep the code cells; shorten any code-cell prose to one line saying what it checks.

**2. Small scale: density.** CRW's complaint is paragraphs where every clause carries a symbol. Each page currently has 450–580 inline math spans. Halve that.

- One idea per sentence. A symbol appears in a sentence only if the sentence is about it. Say "the posterior" rather than $p_\beta(w\mid D_n)$ once it has been defined in the section.
- Put a derivation's symbols in one or two display equations and let the sentences around them be words.
- The one-line notation restatement at the top of each section stays, as a short table or one line, so the symbols are there to look up rather than re-read in every sentence.
- Short paragraphs (three or four sentences). A reader should be able to skim the words and only slow down at the displayed equations.

**3. Remove the within-notebook reading recommendations.** Delete every `reading(...)` and `references(book(...))` call and the closing "Readings" section, and the keystone reading in the opening. Drop the `reading`/`references`/`book` imports. The opening keeps its one-line question and a one-line route through the sections. A paper may still be *cited* in one clause where a result is stated ("Watanabe 2013"), as a bare link, no rating, no annotation. NB4 is the exception: it is the applications catalogue and keeps its list, cut to one line per paper and without the "Background for the whole day" section, which is reading recommendations.

**4. Blog posts first; notebooks are the backup.** Replace the "Read the blog posts first" block with this (same wording on all four pages):

> **These notebooks are a backup.** Read the blog posts first: Liam Carroll's [Distilling Singular Learning Theory](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) (DSLT 0–4), Jesse Hoogland's [Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick) and, with Stan van Wingerden, [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong); then Zach Furman's [Singular learning theory: exercises](https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises). Come here for a derivation, a worked exercise, or code to check something.

**5. Remove the EFT hangers.** CRW will write the effective-theory piece separately, so nothing in the notebooks should hold a place for it.

- NB3: delete §3.2 (the stub) entirely and renumber §3.3 to §3.2 (anchors `slt3-2`, exercise labels 3.2.x). Delete the "an exponent at a resolution scale" phrasing and anything that points forward to an argument about effective theory, resolution scales, or "effective singularities". The Bayesian Renormalization reading goes with the other readings. Keep the pathology section (§3.1) and the temperature/sampling section as plain statistics: λ_eff(n) is what a finite sample sees, β is the knob in the estimator.
- NB1, NB2, NB4: remove any sentence that points to NB3 §3.2, to "CRW's argument", or to an effective theory to come. Check cross-references to NB3 section numbers and exercise labels (NB3's 3.3.x become 3.2.x).
- The builders' header docstrings and comments too.

**6. Build, verify, critique.** `python build_<stem>.py`, `--check`, `--run` and all `checks/*` for your stem must pass; update a check if you removed the thing it checked. Run one short Codex critique (same command as before, `--effort medium`, in the background) asking only: did any cut make a statement wrong, unconditional where it needs a condition, or undefined-symbol dense; and is anything still padded. Apply what holds. Then rebuild.

## Report (concise)

Word count and inline-math count before and after; what you removed (exercises, sections, readings) in a few lines; what the critique changed; anything you could not verify. Append it to your existing log `reviews/slt/2026-10-14-<stem>.md` under a new heading.
