# NB2 (slt2_free_energy_volume): presentation pass, 2026-10-14

Brief: `reviews/slt/2026-10-14-presentation-brief.md` (plus its Addenda 1–2).

## Files

- `slt/build_slt2_free_energy_volume.py` (text, figures, alt texts, one code-cell change), regenerated `.html`/`.ipynb`.
- `slt/assets/figures/make_slt2_free_energy_volume_figures.py` (numpy/matplotlib, about 1.5 s) writes five SVGs:
  `slicing` (K = w⁴ sliced by level: V(t), slab v(t)dt, and the weight e^{-nt}), `sublevel2d` (nested {K<t} at t = 1/4, 1/16, 1/64 for a²+b², a²+b⁴, (ab)²),
  `logV` (log V against log t; slopes 1, 3/4, 1/2), `poles` (poles of ζ ↔ small-t terms of v ↔ large-n terms of Z̄_n), `two_basins` (narrow accurate vs wide basin, with the posterior at n = 10 and n = 300).
- Two LessWrong images (Addendum 2), from Hoogland & van Wingerden, *You're Measuring Model Complexity Wrong*, with credit in the caption and the source URLs in the figure script's docstring:
  `lw_basin_volume.png` (3-D basin, V(ε) as a grey projection; placed at the definition of V(t)) and `lw_hessian_infinite.png` (a degenerate basin's sublevel set next to the flat quadratic approximation's infinite slab; placed at "Bowl versus valley"). Downscaled and flattened on white, otherwise unchanged; inspected.
- `slt/checks/slt2_free_energy_volume_presentation.py` (new): checks every number in the new captions and prose (volume ratios 4, 4^{3/4}, under 2; logV closed forms; posterior masses 0.79 and 0.001 in the figure; the 59-sample window; the corrected 2.3.4 crossovers; the V(t) exactness for 2.2.3; the prior counterexample for λ ≤ d/2; the 2.3.1 sign).
- `slt/checks/slt2_checks.py`: `np.math.factorial` → `math.factorial` (it crashed under numpy 2.3; no other change).

## Verification

`python build_slt2_free_energy_volume.py`, `--check`, `--run` pass (run: 4.5 s total, slowest cell 3.1 s). Both check scripts pass. Every figure was rendered to PNG (rsvg-convert) and inspected, and the page was screenshotted with headless Chromium with all sections open.

## Critique (Codex, one round, high effort) and what changed

Codex returned ten points; I checked each and accepted all ten:

1. "One simplification" claimed n(K_n − K) = O(1) on the strength of a regular-case argument. It now says: work with the population Gibbs distribution, and under Watanabe's assumptions (realizable or relatively finite variance) the empirical free energy differs by a random O(1). §2.3 has one added sentence: across basins computing different functions, nL_n(w_α) has O(√n) fluctuations that do not cancel.
2. With the logistic offset δσ(20w), the wide basin has a tiny positive curvature at its minimum, so strictly λ_V = 1/2, not 1/4 (this was already wrong before this pass). I replaced the logistic with a C² smoothstep that is exactly constant for |w| ≥ 1/2, in the code cell, in the figure and in the check. The posterior masses are unchanged to 3 decimals.
3. Solution 2.2.3 said "further terms at larger t give poles further left". That was false: V is exact for t ≤ 1 and the rest is entire, so there is only one pole. Corrected. The poles caption now says "power λ_k and at most m_k − 1 logarithms".
4. The crossovers in Solution 2.3.4 were wrong (62 → 83, 140 → 180; 650 was right), and "dwell time" became "range of n". This was already wrong before this pass.
5. The crossover statements now say they keep only the leading terms and need Δλ/ΔL > e. The window is now 10/(ΔL − Δλ/n*), which matches the measured 59 samples.
6. λ ≤ d/2 needs a prior that is positive at w₀ (counterexample: prior (3/2)w² gives λ = 3/2 for K = w²). Added to the text and to Exercise 2.1.6. The solution's "narrower than a ball" now reads "a bounded Hessian guarantees a ball inside the sublevel set".
7. Cubic reparametrization: uniform priors in different coordinates are different priors, and pushing the prior forward keeps λ the same. Solution 2.1.4(f) is reworded.
8. The sublevel2d panel labels are now exact formulas (V = πt/4, V ∝ t^{3/4}, V = √t(1 + ½ log 1/t)).
9. Opening: "lack a positive-definite Hessian (many parameters compute the same function)" became "typically have flat directions (continuous families computing the same function)".
10. Added the "Read the blog posts first" block (Addendum 1), verbatim.

My own fixes before the critique: Exercise 2.3.1 said "same sign" where the defined ΔL = L_V − L_U and Δλ = λ_U − λ_V need "opposite signs" (wrong before this pass). "Equality exactly when the Hessian is positive definite" became "equality in the regular case" (writer-brief erratum). The thermodynamic paragraph's "relative width ~1/log n*" was replaced by the slope-based window.

## Word count

Page prose, with code and captions stripped from the HTML: 4016 → 3924. Builder markdown outside the exercise statements: 2050 → 1854. That includes about 95 words for the required blog block and about 100 words of caveats added after the critique, so the original prose itself was cut by about 15%. Captions add 310 words. Exercises and worked solutions are kept in full.
