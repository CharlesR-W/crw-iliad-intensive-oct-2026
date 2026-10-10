# Review log: DA1, `da1_core`

Builder: `da/build_da1_core.py`. Outputs: `da/da1_core.html`, `da/da1_core.ipynb`.
Checks: `da/checks/da1_core_checks.py` (sympy and numpy, about 10 s, 37 checks).
Spec: `DA_SPECS_v1.md` (CRW decisions and the DA1 section). Primary source: `da/source/data-attribution.tex` (Louis Jaburi, CC BY 4.0).

## Draft 1 (2026-10-09)

### Shape as built
- Opening: title, credit line, a 3-sentence pitch, the two keystone readings (Koh & Liang; Bae et al.), and a route with times (10 + 10 + 25 + 20 + 20 + 5 = 90 min).
- §1 The question: the analogy with interpretability, cost, Lewis's counterfactual, overdetermination (the croissant), non-transitivity (the boulder), and Shapley in one paragraph with the croissant game's values.
- §2 Targets and conventions: β, the summed objective, φ, π; the four conventions; the three targets; target vs estimator; the map table with DA2 stop links; TRAK's place. Exercise 2.1 converts Koh–Liang's ε convention.
- §3 The classical IF: Hampel; the IFT derivation in text (local stationary branch); three factors. Exercise 3.1 (required) covers OLS from `ex:ols-if` (a), (b), (e), with a direct deletion derivation that needs no Sherman–Morrison, plus leverage facts and h_j<1. Then a leverage/Mahalanobis remark and a code figure (IF vs exact deletion; the ratio against leverage).
  - Extension: Exercise 3.2 (`ex:ols-if` (c)–(d), Sherman–Morrison and the series).
  - Extension: Exercise 3.3 (`ex:if-derive`, local branch vs argmin, with a tilted double-well example of basin switching).
- §4 What breaks: damping = proximal objective; Gauss–Newton defined in full, with the f = w² counterexample (E6); the PBRF defined, with its derivative = damped GN IF; Bae's five gaps. The code shows the parameterization dependence of the damped IF in the a·b model (E4), with warm-started gradient-flow retraining as circles.
  - Extension: Exercise 4.1 (`ex:if-degenerate`, corrected; part 4 is new: gradient flow conserves a²−b², so warm-started retraining matches the λ→0 limit).
  - Extension (code): a proximity gap on a tiny MLP. It compares proximal retraining with plain retraining after deletion; the damped IF ranks the first almost perfectly and not the second.
- §5 Evaluation: target vs estimator; the three checks; LDS (TRAK Def. 2.4, protocol stated); ranking, not calibration; single-model vs distributional targets; TracIn's estimand (E10). Exercise 5.1 is a code exercise on ℓ2-regularized logistic regression: students write `attribution_scores` only. Exercise 5.2 asks five interpretation questions whose answers quote the run's numbers.
- §6 The lens: the corrected filter table (Codex's version), how to read it, a fold deriving the GD row, what the table hides, a static filter plot, and the DA2 tour linking `da2-stop-1` … `da2-stop-6`.

### Errata applied silently (spec ledger)
- **E1 (conventions):** §2 states them once (summed objective, upweighting derivative, deletion, sign, explicit 1/b, summed-loss units for curvature and damping). Exercise 2.1 converts Koh–Liang's ε convention.
- **E4 (`ex:if-degenerate`):** the damped IF for φ = a² is finite, saturating and minimizer-dependent, not divergent. The code caption is "parameterization dependence", not the proximity gap. We assume z̄ ≠ 0.
- **E6 (GN at a critical point):** the residual term vanishes only under extra conditions; the f = w² counterexample is stated.
- **E10 (TracIn):** §5 describes TracIn's estimand (the accumulated first-order test-loss decrease along the trajectory, with checkpoint gradients).
- **E12 (IFT, OLS, Mahalanobis):** the IFT gives a local stationary branch; deletion needs h_j < 1; the Mahalanobis form uses the uncentered moment, and we also give the intercept version.
- **E13:** no Hasse figure. The proximity gap is measured honestly in the MLP extension, and the a·b figure is captioned differently.

### Verification

**Formulas.** These are in `da/checks/da1_core_checks.py`, all passing:
- Shapley values for the croissant game, and LOO = 0;
- Hampel's mean influence;
- IF = finite differences of retraining (logistic regression);
- the Koh–Liang conversion;
- damped IF = derivative of the proximal response;
- the LDS predictor's sign;
- OLS: exact deletion; Sherman–Morrison for general β_j; the IF; the geometric series; leverage bounds and Σh = p; h_j = 1 ⇔ the retained design is rank-deficient; the direct deletion derivation (sympy); the uncentered and the intercept/Mahalanobis forms of leverage;
- the GN counterexample (sympy), and H = G + residual (sympy);
- the a·b model: H = n v vᵀ; g_i = −(z_i − z̄)v; the damped IFs for ab and a² (sympy); the pseudoinverse IF and exact LOO; the λ→0 limit for a² = warm-started gradient flow (solve_ivp finite differences); conservation of a² − b²; minimizer dependence;
- PBRF derivative = −(G+λI)⁻¹g_k at a *non-converged* tiny-MLP checkpoint (Newton on the PBO, finite differences);
- the frozen-curvature GD filter, against a simulation; SOURCE F(0) = ηK and the 0.632 vs 0.5 crossover (sympy); GD → 1/s;
- BLR: −Cov(NLL_j, w) = V x_j r_j/σ² (Monte Carlo, 2M samples); the damping σ²/τ² in XᵀX units and τ⁻² in likelihood units; and d E[w]/dβ_j exactly.

Exercise 3.3's numbers were checked by hand: the tilt at β = 1 is −0.01w, so the minimum is near +1; at β_3 = 0.8 the tilt is +0.012w; H ≈ 8 at w ≈ 1.

**Code cells.** Each cell checks itself where it can:
- the OLS cell prints max|exact − pred/(1−h)| = 6e−16;
- the a·b cell compares the damped IF with independent gradient-flow retraining;
- the §5 cell reports each method against finite differences of its own target (exact IF 2.8e−7, damped IF against proximal retraining 3.3e−7).

The finite-difference step is 1e−3: the error scales as 1/ε below that, which is the solver's gradient-accuracy floor. The §5 configuration was chosen before looking at method rankings; the ranking pattern held on 5 seeds (scratch runs): exact IF best on LOO, damped ≈ exact on LDS, grad dot worst.

**Papers.** Two Claude subagents checked these against the arXiv PDFs and HTML (quotes in their reports):
- Koh & Liang: the mean-loss convention, ε = −1/n, damping in §4.2, Fig. 2.
- Bae et al.: the PBRF Eq. 8; the five gaps (§4); the first three dominate (§6.1); the IF equals the *linearized* PBRF's solution (§4.4, App. B.4), with GNH. We say "its derivative at β = 1 is exactly the damped GN IF", which our check confirms.
- Basu et al.: the abstract's claims.
- Mueller 2024: about interpretability only, so the blurb says so; §3 is non-transitivity, §4 redundant causes.
- Data Shapley: TMC and G-Shapley in §3 of v1. We link v1, because v2 is a different paper.
- TRAK: Def. 2.4 (LDS); α = 0.5, 100 subsets, 5 models each, 2,000 targets (§4.1); the method in §3.2; no regression on retraining outcomes, though soft-thresholding is cross-validated on LDS for two datasets.
- Datamodels: Def. 1, Eqs. 5–8, 300k CIFAR models at α = 50%.
- TracIn: §3.1 the idealized definition and Lemma 3.1; §3.2 the 1/b; sign convention.
- SOURCE: Eq. 21 F_r(σ), and the qualitative comparison to damping with λ = 1/(ηK) (§3.2).
- Kreer et al.: Eq. 5–6, and App. A.2 Eq. 17 (≈ damped IF with γ).

**URLs.**
- All arXiv and iliad-intensive.org URLs return HTTP 200 (curl, 2026-10-09).
- The three DOI links (Hampel 1974, Lewis 1973, Hitchcock 2001) redirect correctly. The publisher pages (T&F, PDCnet) return 403 to scripts; the subagent confirmed each DOI's title through Crossref metadata.
- `../td/td2_6_curvature.html#td26-1` exists.

## Critique round 1

Run in parallel: a Claude design critic (Opus) and Codex (`--effort high`). Both read the builder, the rendered sandbox page and the outputs.

### Codex (technical): 11 findings; all accepted after checking
1. **MLP extension, plain retraining ended at saddles** (4 of 20 deletions; minimum eigenvalue −0.004 at |grad| ~1e−9). Damped Newton was attracted to saddles.
   - Replaced it with saddle-free Newton: |eigenvalues| floored, steps accepted only on descent, and a kick along the negative-curvature direction followed by a restart.
   - Every endpoint is now checked: smallest Hessian eigenvalue 2.8e−5 > 0, printed by the cell.
   - The result is essentially unchanged: Spearman 0.19, size ratio 49 (was 0.18, 52).
2. **TracIn vs TracInCP conflated; 1/b missing.** We now write the TracIn formula with η_t/b over the steps with i ∈ B_t. TracInCP is described as evaluating every example at checkpoints regardless of batch membership. Lemma 3.1's additivity is attributed to the idealized one-example-per-step form.
3. **"Damping selects a nearby minimizer" was false at finite λ** (Codex's 2-D counterexample). Now: as λ→0 it picks the nearby minimizer to first order; at finite λ it also shrinks the response.
4. **GD filter → 1/s needs |1−ηs|^K ≪ 1.** Fixed in the prose and the plot title.
5. **Ex 5.2 answer 3 claimed every method loses accuracy from LOO to LDS** (false: the damped IF and grad dot gain). The sentence was removed.
6. **"Single-seed LOO is mostly noise" was unsupported** (paired same-seed deletion cancels seed noise). Rephrased as seed-to-seed variation of the deletion effect itself.
7. **Subsets were Bernoulli(½), not exactly half.** They now have exactly n/2 = 40 examples. New LDS values: exact 0.826, damped 0.829, grad dot 0.560 (slopes 24, 35, 12). The answers were updated.
8. **SOURCE row:** added the "independence" approximation (Bae et al. 2024 factor expectations of products).
9. **Ex 4.1 nearest-point geometry is first order only:** added "to first order". Codex's finite example shows that the endpoints differ at finite change.
10. **Lewis defines causation through chains** (transitive); direct dependence is what fails. §1 now speaks of counterfactual *dependence*, and the boulder paragraph ends "Dependence does not chain."
11. **Ex 2.1 solution sentence about Δβ relative size was wrong.** Now: normalization changes the units of curvature and damping, and Koh–Liang's ε measures the perturbation in units of 1/n.

Codex re-ran the 37 checks (all pass), reproduced the cached outputs, and found no further errors in the IFT, OLS, leverage, GN, PBRF, a·b formulas, LDS sign, F(0) values or 0.632 vs 0.5. It also noted that the BLR row is exact for the posterior-mean parameter response, so the table now says "exact, for the posterior mean".

### Design critic: 13 items; adopted except where noted
- **Time:**
  - Ex 2.1 moved to a folded Extension, with a one-line convention item (4) in the main text.
  - Ex 3.1 part 4 now states the leverage bounds (proof stays in the solution) and asks only about h_j = 1.
  - Ex 5.2 parts 2 and 5 are marked optional.
  - The §4 pitch no longer asks students to read Bae §§1–4 during the session.
  - The §6 figure cell is marked optional.
  - "Route" became "The path"; the total is kept at about 90 min, the spec's own section lengths.
- **g(S) clashed with gradients:** renamed φ̂(S) in the text, axis and code comments.
- **The PBRF paragraph named five gaps without defining them:** now a defined list. Also added: why Q's Hessian is G+λI, and the iHVP sentence that sets up DA2 stop 1.
- **a·b:** the formula is derived inline (H = n v vᵀ, g_i ∝ v); the post-plot paragraph is split into "why finite" and "why minimizer-dependent".
- **TracIn notation** is defined locally (merged with Codex item 2).
- **§6:**
  - ∂w/∂β_i replaces Δw.
  - Row notes are moved below the table, including the BLR units.
  - SOURCE and γ are glossed.
  - A training-time bullet was added for stop 4, and stop 6 got its one-line description.
- **§2:** the local posterior p_β is defined under the map table.
- **§4 notation:** w* is defined as the checkpoint when it is stationary.
- **Pedantry cut:** the intercept/Mahalanobis formula is one sentence; the TRAK protocol is cut to "averages over several seeds".
- **Number after the §3 figure:** h_B = 0.83, about 6×.
- **Ex 2.1** now speaks of the parameter response.
- **Stub:** raises NotImplementedError.
- **MLP cell:** n, h renamed to n_mlp, width.
- **Not adopted:** relabelling the spine "about 80 min". The spec's section times sum to 90, and the cuts above bring the realistic time near that.

**Other changes:** the MLP cell's Newton loop stops at |grad| < 1e−7 (the float floor is about 5e−8). Under a machine load average of about 25 the cell takes about 19 s.

## Critique round 2

### Codex (technical): 4 findings, all accepted
1. **"As λ→0 damping picks the nearby minimizer" needs g_i ∈ Range(H).** Counterexample: L_{1,2} = x²/2 + 1 ± sin y gives a 1/λ divergence. §4 item 1 and L1 segment 4 now state the condition, and say the response along flat directions otherwise grows like 1/λ. The a·b model satisfies the condition, which is the content of "why finite".
2. **L1 said "every method" computes ∂φ(π(β))/∂β_i.** TracIn is a counterexample (Codex's two-step SGD example gives TracIn 1 where derivative and deletion are 0). Now "response-based methods".
3. **SOURCE row:** the filter of H assumes unpreconditioned SGD (SOURCE App. C has P^{1/2}F(P^{1/2}HP^{1/2})P^{1/2}). Added to the table and L1.
4. **Ex 4.1 balanced minimizer √z̄ needs z̄ > 0,** and the two examples coincide at z̄ = 1. Now "generally minimizer-dependent: for z̄ > 0, compare …".

Codex reran all 37 checks and every cell's calculation independently. The MLP retrains are minima. The lecture numbers match the outputs, and no further material errors were found.

### Design critic: 9 items
**Adopted:**
- The path paragraph names the required exercises (3.1, 5.1, 5.2).
- The keystone readings are labelled "best read after working through DA1".
- §4 a·b paragraph rewritten question-first ("Does the damped IF depend on which minimizer…? … yes"), with the valley named and "why finite" in two plain sentences.
- Local definitions: the saved checkpoint w^s in the §2 table; SGLD spelled out; "(methods met in §5 and DA2)" for TracIn/TRAK/SOURCE; the local posterior "times a prior if there is one", matching DA2.
- Ex 5.2 solution 2 explains the damped slope. Solution 4's unsupported mechanism clause was cut: the critic checked that the half-subset weights don't blow up. It now quotes the slopes (10–35×).
- TracIn trimmed to the formula and two sentences, with TracInCP deferred to DA2 stop 2. The sum statement is "to first order in each step", which holds for minibatches.
- The Ex 5.1 self-check (check-1 error ~1e−7).
- The GN counterexample says why it matters.
- §6 row notes folded.
- L1: segment 1's "picked up in" corrected; OLS symbols added to the board notation; segment 6 lists the required exercises.

**Not adopted:**
- Moving the a·b code cell into an extension. The spec places that code in the §4 spine; it is now question-first and shorter.
- Relabelling the spine as 80 min, for the reason given in round 1.

## Critique round 3

### Codex (technical): 5 findings, all accepted
1. **The range condition (g_i ∈ Range H) gives a finite λ→0 limit, but not a nearby minimizer.** Counterexample: L_1 = ½(x−1)² + cos²y, L_2 = ½(x+1)² + sin²y. Upweighting moves the minimizers to y = π/2. §4 item 1 now claims only the finite pseudoinverse limit −H⁺g_i. The nearest-minimizer reading stays only in the a·b example, where it is derived (Ex 4.1.4). The lecture now says "per-example gradient" and "pseudoinverse answer".
2. **"Times a prior if there is one" contradicted the local-BIF filter.** Codex's 1-D example: an extra prior gives 1/(3+α), not 1/3. Kreer et al. Eq. 5 replaces the prior with the localization Gaussian, and the sentence now says so.
   - **For the integration pass:** DA2 writes the local posterior with an extra prior ρ(w). That changes the leading-order filter unless ρ is flat near w*. This was flagged in the report, not edited here.
3. **Damping does not shrink every scalar influence:** in the run, 73/800 grew and 66/800 flipped sign. Ex 5.2 answer 2 now just reports the fitted slope.
4. **Regression slopes are not spread ratios.** Answer 4 now says "fitted calibration slopes … about 12 to 35".
5. **The lecture's Δw should be ∂w/∂β_i.** Fixed.

### Design critic: nothing substantive; timing and polish
**Adopted:**
- §2 defines "query" and the SGD symbols (η, B_t, b); the unrolling row says "last iterate after T steps"; convention 4 is shortened.
- §4: the notation sentence is simplified; the formula is named "the damped Gauss–Newton influence function" (DA2's opening uses this name); the warm-start gap is rephrased; I_λ(z_i, φ) is defined before use.
- Ex 3.2 renames γ, D to t, d_j (they clashed with the BIF's γ and the Bregman D_j).
- KOH reading says "see DA1 §2".
- §5 "(§4)" pointer; Ex 5.2(3) wording.
- Ex 5.2(4) adds a verified sentence, rerun by me: with κ = 1, slopes ≈ 2.7–3.3 and LDS ≈ 0.965–0.969. The κ = 0.1 run gives slopes 5–7 and LDS 0.92–0.93.
- §3 pitch shortened.
- §6 "Reading the table" merged from four bullets to three; the GD |1−ηs|^K ≪ 1 condition moved into the GD fold.
- The path box says "anything marked optional can be skipped".
- Lecture segment 4: one board formula for the fixes. Segment 5: the filter notes trimmed, and the SOURCE-not-ridge line cut.

**Not adopted:** a §4 line restating the keystone readings, since the §4 pitch already points to Bae et al. The critic's question on Grosse Fig. 1 is answered: I checked it against the PDF text (shutdown query; 810M vs 52B; Hal, the desert survivor; influences computed on pretrained models), and Codex rounds 2–3 also checked it.

## Optional interactive figure (§6)
- `da/assets/interactive/filter_explorer.html`: the IF, damped, GD and SOURCE filters on linear axes (F(0) markers, optional s < 0) and log axes (GD instability shaded for s > 2/η). Sliders for η, K and λ, with λ tied to 1/(ηK) by default. A table gives F(0), F at s = 1/(ηK) and F at the cursor.
- It is self-contained vanilla JS, light and dark (checked by headless-chromium screenshots), and works at 390 px (the table scrolls inside its panel). It ends with the height-reporting script that TD figures use.
- `da/assets/interactive/checks/filter_explorer_check.py` runs the JS math block under node on 400 random (s, η, K, λ) cases, including s = 0, negative s and ηs > 2. It compares GD against a direct simulation of the response recursion and SOURCE/damped/IF against numpy. It also checks F(0), the 0.632 vs 0.5 crossover, GD → 1/s and the GD overshoot. All pass.
- It replaces the static matplotlib filter plot that §6 had in rounds 1–3.

## Final check (figure plus round-3 fixes)

### Codex: 5 findings, all fixed
1. The figure drew a false line across the damping pole at s = −λ. The plotting now breaks paths at poles.
2. The negative-curvature note overclaimed: GD grows like (1+η|s|)^K, and damping does tame curvature above −λ. Reworded.
3. The "Shaded" note could appear without shading. It now tests the actual plot bounds, and says the GD sign depends on the parity of K.
4. The check script silently accepted NaN (JSON null) from F_if. It now requires finite results; a mutation test (F_if → NaN) fails the check as it should.
5. The κ = 1 sentence in Ex 5.2(4) holds for the two influence functions, not grad dot (slope 0.28, LDS 0.90). Scoped.

### Claude design check of the figure: adopted
- F(0) markers drawn outside the clip, with leader-line labels grouped by value.
- An "IF → ∞" label; ymax includes 1/λ when untied.
- Log plot shows |F|, faint where F < 0, so GD's blow-up beyond 2/η is visible rather than a vanishing curve.
- `touch-action: pan-y`, so phones can scroll.
- Short table row names, values in units of ηK, and a cursor mirrored on both plots with dots on the curves.
- Larger tick fonts; dashed legend swatches; the slider renamed "linear plot: s up to"; a one-line subtitle; y allowed below 0 in negative-curvature mode.
- §6 gained a "negative curvature" bullet under "What the table hides", since the figure's intro mentions it.

**Not adopted:** a descriptive iframe title. The kit builds it from the file name; this would need a kit change.

## Status at hand-off
- `python da/build_da1_core.py`, `--check` and `--run` all pass: 27 cells, 7 code cells, 7 solutions; `--run` takes 23 s, slowest cell 18 s (the MLP extension), under a machine load average of about 20.
- No build warnings. All DA2 anchors `da2-stop-1` … `da2-stop-6` resolve against `da2_methods.html`.
- `da/checks/da1_core_checks.py`: 37/37 pass. `da/assets/interactive/checks/filter_explorer_check.py`: all pass.
- **Not verified directly:** the three DOI landing pages (403 to scripts; titles confirmed via Crossref). Basu et al.'s claims beyond its abstract and §§4–5 are not used.
