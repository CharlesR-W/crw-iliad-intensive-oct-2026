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
