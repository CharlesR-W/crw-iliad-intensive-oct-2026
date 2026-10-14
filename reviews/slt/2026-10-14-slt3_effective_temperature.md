# NB3 (slt3_effective_temperature): presentation pass, 2026-10-14

Brief: `reviews/slt/2026-10-14-presentation-brief.md` (incl. Addenda). Builder: `slt/build_slt3_effective_temperature.py`.

## Figures (new; `slt/assets/figures/make_slt3_effective_temperature_figures.py`, numpy/matplotlib, ~5 s)

| file | section | shows |
|---|---|---|
| `slt3_effective_temperature_jump.svg` | 3.1 | (a) λ(δ) jumps ½→¼ at δ=0 while λ_eff(n;δ) (flat prior, scaling function) is smooth and steps up near δ~n^{-1/2}; (b) at δ=0 the sample loss ½w⁴−u_n w² with u_n at 0, ±1, ±2 SD of N(0,1/n): steeper wells / double wells on the scale n^{-1/4} |
| `slt3_effective_temperature_two_scales.svg` | 3.1 | K=w⁴+δw² at t≫δ² (looks like w⁴, width ∝t^{1/4}) and t≪δ² (looks like δw², width ∝t^{1/2}), sublevel sets shaded; log-log panel: slope 4 / slope 2, λ = 1/slope in 1-D |
| `slt3_effective_temperature_wbic.svg` | 3.3 | schematic: E_β[nL_n] ≈ nL_n(w*)+λ/β linear in 1/β with slope λ; meets F_n at 1/β=log n (WBIC); Bayes point at β=1 |
| `slt3_effective_temperature_sgld.svg` | 3.3 | real Langevin chains (full gradient) on K=(w1w2)², nβ=200, from w*=0, γ=1 vs γ=25, level K=1/(nβ) in blue |

## Verification

- `checks/slt3_effective_temperature_figures.py` (new): λ_eff scaling-function values used in Fig 1(a) (¼ at δ=0, monotone, →½; matches nE[K] on a w-grid); double-well minima w²=u_n/a_n; λ=1/slope for |w|^k in 1-D; E_β[nK]=λ/β incl. the β=1 and β=1/log n points of Fig 3; exact grid λ̂ for Fig 4 (0.367 at γ=1, 0.117 at γ=25, both < ½); step-size stability ε(nβh+γ)<4 for the linearised update (replaces the earlier "small compared with 2/(nβΛ)").
- `checks/slt3_checks.py` unchanged, passes.
- `--run`: 5 code cells, 57 s total, no failures. Code cells unchanged.
- URLs added: the six blog URLs of Addendum 1 (fetched by the coordinator, per the brief); arXiv 2305.10491 and 2212.11379 fetched (abstract pages; 2305.10491 v3 PDF table of contents read for the "Sections 1 and 3, Section 4" pointer).

## Text changes

- Opening paragraph shortened; Addendum-1 blog block after the subtitle.
- "Setup, repeated" block folded into §3.1's notation line (now standalone); π_n defined explicitly.
- §3.1 split into two bold-led paragraphs (the jump; the exponent at a scale), each with its figure before the derivation; repeated caveats cut.
- §3.2 stub box made clearer (blockquote, planned scope = the spec's one-line placeholder). The author note "use 'effective singularity' only if unavoidable" moved to a builder comment. Bayesian Renormalization reading placed after the stub box (Addendum 3), annotation describes the paper only.
- §3.3: K_n = L_n − L_n(w*) defined in the notation; the β-derivative written for ∫φe^{−βnK_n} (previously mixed Z̄_n(β) with K and E_β[K_n]); WBIC and estimator as two bullets; SGLD paragraph trimmed; hyperparameters as a four-item list; ε condition stated exactly.
- Solutions: garbled parenthetical in 3.3.2 fixed; last sentences of 3.1.4, 3.3.3, 3.3.4 shortened. No exercise removed.
- Wei et al. annotation: removed "the case for the whole programme" (evaluative).

## Critique

One Codex round (gpt via codex-companion, effort high, read-only), prompt and output in the session scratchpad. Accepted, all applied:
- Opening said §3.3 "measures that exponent": reworded. §3.3 estimates the analogous finite-scale quantity for the empirical loss, with a localization scale.
- K_n was undefined in Exercise 3.1.4 after the setup block was removed: now defined there as L_n − L_n(0); §3.3 says the reference changes to w*. F_n defined at the WBIC bullet.
- "below the crossover the two losses are the same loss" overstated it (λ_eff = 0.32 at s = 0.5): replaced by "for δ√n ≪ 1 the quadratic term is a small perturbation where the posterior mass is".
- Fig 2(c) labelled the coarse regime "λ = ¼" although λ = ½ for δ > 0: labels now "V ∝ t^{1/4}" / "V ∝ t^{1/2}", and the caption says "local volume exponent".
- WBIC caption named the temperature as the WBIC: now "the posterior mean at 1/β = log n is the WBIC, which approximates F_n's leading terms".
- Estimator sentence was ambiguous about order of operations: fixed.
- "nearly flat direction becomes exactly flat": replaced by "the quadratic coefficient δ reaches zero".
- Wei et al. annotation still endorsed the paper's position: made neutral ("the authors' argument for …").
- Solution 3.1.3 split into paragraphs with the λ_eff formula displayed; solution 3.3.1 model-selection part shortened, and its Z̄_n(β) notation replaced by ∫φe^{−βnK_n} to match the main text.
Codex independently reproduced the Fig 4 values (0.367, 0.117).

## Word counts (tags stripped; pre/code/script/style removed)

- Whole page: 3407 → 3529.
- Main prose (no solutions, no captions): 2406 → 2378, which includes 175 words added under the Addenda (blog block 84, Bayesian Renormalization reading 91). Like for like: 2406 → ~2203 (−8%).
- Solutions 1001 → 947; captions 0 → 204.

## Addenda

1. Blog-posts block inserted verbatim after the subtitle.
2. LessWrong images: none reused. NB3's ideas (crossover of w⁴+δw², WBIC, the tether) are not drawn in DSLT 0–4 or the two Hoogland posts, as far as I know; I did not check every image in those posts.
3. Bayesian Renormalization reading added right after the §3.2 stub box (outside it) as "Related reading, on scale-dependent inference", importance 3, length 3, with the companion arXiv:2212.11379 as a pointer. Both abstract pages fetched. The section pointer is checked against the v3 PDF's table of contents.

# Streamlining pass (afternoon, 2026-10-14; `reviews/slt/2026-10-14-streamline-brief.md`)

## Numbers (tags stripped; pre/code/script/style removed)

| | morning | now |
|---|---|---|
| page words | 3529 | 1727 (target ≤ 1800) |
| of which main prose / solutions / captions | 2378 / 947 / 204 | ~1130 / ~450 / ~140 |
| inline math spans (`class="math inline"`) | 446 | 211 |
| cells | 16 | 14 |

`--run`: 51 s, no failures. `--check` fresh. `checks/slt3_checks.py` (comment labels renumbered to 3.1.2, 3.2.1, 3.2.3) and `checks/slt3_effective_temperature_figures.py` pass. Figures and code cells are unchanged.

## Removed

- The §3.2 stub, entirely; §3.3 is now §3.2. Anchors `slt3-s2` (sibling convention) and `slt3-2` (as asked) are both on the heading; §3.1 likewise. Exercise labels are now 3.2.x.
- Every reading: the keystone pair at the top, Bayesian Renormalization, the closing Readings section, the Watanabe book reference; the reading/references/book imports. Watanabe 2013 and Lau et al. 2023 are now cited as bare links in one clause each.
- "Resolution scale", effective-theory and CRW phrasing everywhere, including the docstring and the builder comment.
- Top block replaced by "These notebooks are a backup", verbatim.
- Exercise 3.1: dropped the identity λ_eff = nE[K] (it is NB2 Exercise 2.3.3) and cut part 4 (now 3.1.3) to its δ=0 core. Exercise 3.3: dropped part 2 (γ drops out as nβ→∞; the statement stays as one line in the knobs list). Three parts each, with shortened solutions.
- §3.1's in-text scaling derivation. The result is stated in one line pointing to Exercise 3.1.2, which carries the derivation.
- "Setup" restatement, the opening paragraph, "the formula has two uses", and the code-cell prose (now one line each). Captions are one sentence each.

## Codex (effort medium) — all accepted

- "A continuous Z̄_n has a discontinuous limit" was wrong (Z̄_n → 0): it is now "the limiting exponent jumps although every finite-n integral is continuous in δ".
- The δ=0 prediction had lost the sign case: it now says "a regular minimum or a shallow double well, depending on its sign", and drops the claim that the curves look alike.
- The SGLD stationary density is now stated only in the small-step, full-gradient limit.
- Exercise 3.2.3 is restricted to positive-definite basins, with H and d defined.
- Restored conditions: "prior roughly constant near 0", "approaches ¼ / ½", definitions of V(t) and L_n, β in the notation lines.
- Removed duplication: the scaling derivation in the text; "the formula has two uses".

## Not mine to fix (for the coordinator)

`slt/README.md` still describes NB3 §3.2 as the stub and lists it under "Open items"; it also says "Notebook 3 §3.3" for the linear-network exercise (now §3.2) and mentions `reading(...)`/`references(...)`. No sibling builder or `index.html` links to NB3 section anchors or exercise numbers (grepped).
