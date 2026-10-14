# NB1 (slt1_bayes_statmech) presentation pass, 2026-10-14

Brief: reviews/slt/2026-10-14-presentation-brief.md (including CRW's mid-run Addenda 1–2).

## Files
- slt/build_slt1_bayes_statmech.py (text, captions, five `image()` figures, corrected solutions)
- slt/slt1_bayes_statmech.html, .ipynb (generated)
- slt/assets/figures/make_slt1_bayes_statmech_figures.py → slt1_bayes_statmech_{dictionary,concentration,covariances,occam,predictors}.svg (numpy/matplotlib, ~1 s)
- slt/checks/slt1_bayes_statmech_figures.py (caption claims), slt/checks/slt1_bayes_statmech_solutions.py (solution corrections)

## Figures
- dictionary (§1.1): prior × e^{-energy} gives the unnormalized Boltzmann weight; the areas are Z_n, Z_n(U), Z_n(V); the deep narrow well holds less mass than the wide one (checked: 0.074 < 0.115).
- concentration (§1.2): schematic posterior ellipses (nH)^{-1} for n = 5, 50, 500, widths shrinking by √10 per decade.
- covariances (§1.2): simulated model N(μ, e^{2s}), n = 400: ŵ scatter on 600 data sets against the posterior ellipse. Well specified they match. With Exp(1)−1 data the scatter is [[1,1],[1,2]] against diag(1, 1/2) (checked by Monte Carlo).
- occam (§1.3): Occam factor as V_post/V_φ in 1-D (4n halves the width) and a 2-D schematic.
- predictors (§1.4): Exercise 1.4.4 at n = 4: MLE, Gibbs and Bayes predictives, with the Bayes predictive as the posterior average (variance 1 + 1/n; checked).

## Verification
- `python build_slt1_bayes_statmech.py`, `--check`, `--run` (1.5 s total) pass; checks/slt1_checks.py and both new check scripts pass.
- Page rendered headless (chromium) with all sections open and inspected; two displays that overflowed the column were split.
- New URL fetched: DSLT 1 (200). The addendum's six URLs were fetched by the coordinator.

## Critique (Codex, one round, read-only)
Accepted:
- dictionary caption claimed the posterior has area Z_n (it is the unnormalized weight); panel retitled
- concentration figure now labelled as a schematic; the w_0 marker is drawn on top
- Occam 2-D ellipse labelled as a schematic contour; caption now says 2^{-d}
- predictor and covariance captions are scoped, and they now define (μ, s) and the √n axis scaling
- §1.4 notation is restated locally
- the sentence "every asymptotic statement about F_n is a statement about generalization" is softened: it holds only if the remainder terms are controlled
- the w_0 reference is described correctly (the population optimum, not the best fit on this data set)
- quenched disorder is reworded, and the "not extensive" claim is dropped
- the §1.2 regularity conditions now get a one-line mention
- Ex 1.3.2: V_post is restricted to the prior box
- Bayes solution split into its three averaging steps; the wrong "pays back exactly that amount" intuition is cut
- Pre-existing errors fixed (verified with sympy and numerics in checks/slt1_bayes_statmech_solutions.py):
  - Ex 1.1.3 lower bound changed to ≤, since it is attained at equal masses
  - Ex 1.4.4: L(w) − S was missing −½ log σ²
  - Ex 1.4.4: "Bayes loses for σ² < 1, ties at 1" holds only to leading order; the exact threshold is σ² > n log(1+1/n)
- Pre-existing error in the §1.4 text: it said Gibbs pays tr(H⁻¹C)/2n; it pays (d + tr)/2n. Caught by me, not by Codex.

Not done: Codex suggested a §1.4 diagram of Z_{n+1} = Z_n p*(x). I judged that one display already says it, and that a simulated F_n staircase is mostly noise.

## Addenda
- 1: the blog-post block was added verbatim after the subtitle. The opening paragraph also points to DSLT 1, sections "Preliminaries of SLT" and "Deriving the BIC".
- 2: I looked at the DSLT 1 and DSLT 2 images and reused none. They show singular sets and basin switching (NB2/NB3 material), and the one regular-model picture (a polynomial overfit) adds little to the Occam figure.

## Streamlining pass (afternoon, 2026-10-14; brief: reviews/slt/2026-10-14-streamline-brief.md)

**Counts** (page prose with code stripped, solutions included):
- Words: 4408 → 2122 (target ≤ 2200).
- Inline-math spans: 459 → 206.
- Display equations went up a little, because derivations moved out of sentences into displays.

**Removed**
- All readings: the MacKay keystone, Hinton & van Camp, Wei et al., the Watanabe book, the closing Readings section, and the DSLT 1 pointer. MacKay (1992) stays as a one-clause bare citation in §1.3.
- Imports of `reading`, `references` and `book`.
- The "Read the blog posts first" block, replaced by the "These notebooks are a backup" block (verbatim).
- The setup paragraphs, now a six-row notation table. The route is now four one-liners.
- §1.1: the five-bullet "only formal" list (now one sentence) and the paragraph on the tempered posterior (now one sentence and one display).
- §1.2: the "you may use the CLT" paragraph.
- §1.3: the Laplace-method paragraph and the Gaussian-integral display (both now in Exercise 1.3.1), and the separate Z_n display.
- §1.4: the bulleted predictor definitions, the G_n display and the paragraph after the table, all cut down.
- Exercise parts:
  - 1.1.1 (likelihood = e^{-nL_n}): already in the text.
  - Old 1.2.1 and 1.2.2: merged into one part.
  - 1.3.4 (exact conjugate free energy): the first code cell already checks it.
  - 1.4.2 (generalization via the free-energy expansion, the "step of faith" route): a second derivation of the table that 1.4.2 (formerly 1.4.3) derives directly.
  - Each section now has one exercise with three parts. Solutions are cut to their key steps and displays.
- Captions cut to one sentence each. Code-cell prose cut to one line per block.
- NB3 pointers ("a device Notebook 3 needs", "Notebook 3 turns the first one into an estimator") removed. NB1 had no EFT or §3.2 hangers.

**Kept**
- All five illustrations, all four code cells, and the dictionary table.
- The §1.4 regular-case table and the misspecified example, with its exact Bayes-vs-MLE threshold.

**Check label updates**
- In `checks/slt1_checks.py`, only the labels and comments changed for the renumbered exercises: 1.3.4 → "conjugate model", 1.4.4 → 1.4.3, 1.4.3 → 1.4.2. The asserts are unchanged.
- The same relabelling was applied to `checks/slt1_bayes_statmech_solutions.py` and to the figure script's docstrings.

**Verification**
- Build, `--check` and `--run` pass (1.4 s).
- All three slt1 check scripts pass.
- Page rendered headless and inspected.

**Codex critique** (`--effort medium`; all five points held and were applied):
1. Restored "positive definite" for H_n and for A in Laplace's formula.
2. Restored the "replace ŵ by w₀ inside the 1/n terms" qualification in the Bayes-row solution.
3. "The total training loss nL_n is the energy", consistent with the table.
4. §1.4 notation is restated in full, and K_n is defined at its first use in §1.1.
5. Cut a padded sentence after the free-energy identity.

**Not verified:** nothing new. No new formulas were introduced; the remaining ones are covered by the existing checks.
