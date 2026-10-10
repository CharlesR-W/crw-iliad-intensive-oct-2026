# Review log: td2_1_flows (NB2 §1, Gradient descent as a flow plus corrections)

Builder: `td/build_td2_1_flows.py`. Notebooks: `td/td2_1_flows_{exercises,solutions}.ipynb`.
Symbolic checks: `td/checks/td2_1_flows_bea.py` (sympy, ~15 s).

## Verification record (first draft)

### Symbolic (sympy, `td/checks/td2_1_flows_bea.py`)
1. General backward-error recipe: f2 = d2 − f′f/2 and f3 = d3 − (f″(f,f)+f′f′f)/6 − (f′f2+f2′f)/2. Checked as an exact polynomial identity (time-h flow of the truncated modified field vs the method, through h³) for random polynomial vector fields in 2D.
2. GD: f2 = −Hg/2 = −∇(‖g‖²/4); f3 = −H²g/3 − ∇³L[g,g]/12; the GD step equals the flow of f + hf2 + h²f3 to O(h³). Curl of f3 nonzero for four losses, including the hand-worked L = x²y/2 (f3 components and curl −x(x²−2y²)/12 asserted exactly) and (xy−1)²/2.
3. Quadratic: log(1−ηλ)/η series.
4. Momentum family u_{n+1} = u_n + β(u_n−u_{n−1}) − h∇L(u_n + a(u_n−u_{n−1})) (Kovachki–Stuart eq. (7), their λ → β): slow-manifold step plus BEA gives modified potential L + (h/2)c‖∇L‖² inside −(1−β)⁻¹∇ with c = λ̄(λ̄ − a − ½), λ̄ = 1/(1−β). At a = 0 this is Ghosh et al.'s h(1+β)/(4(1−β)²). At β = 0 it gives GD's h/4.
5. Independent check of the heavy-ball modified flow: substituting the flow into the heavy-ball recursion (forward and backward time-h flows) leaves residual O(h³), for β = 1/3 and 9/10 on a concrete 2D loss. Negative control: Kovachki–Stuart's printed constant (+½) leaves an O(h²) residual.
6. Kovachki–Stuart's intermediate step: for u̇ = −λ̄∇Φ + O(h), ü = +λ̄²H∇Φ = +(λ̄²/2)∇|∇Φ|². They print a minus sign (pdftotext of arXiv 1906.04285, the line after eq. (21): "d²u/dt² = −½(1−λ)⁻²∇|∇Φ(u)|² + O(h)"). Redoing their matching with the correct sign gives c = λ̄(λ̄ − a − ½). So TD_LINKS item 5's flag is confirmed: the printed constant (eq. 19, and Theorem 7) has a sign error. This is our diagnosis, not a published erratum; the notebook says so.

### Numerical (code cells; `python td/build_td2_1_flows.py --run`, about 20 s total, slowest cell about 7 s)
- GD vs gradient flow vs IGR flow on L = (ab−1)²/2: global-error log-log slopes 1.06 and 2.01.
- Fraction of GD's endpoint shift captured by the IGR flow: about 81% at ηS ≈ 0.2, 42% at 0.8, 16% at 1.4.
- Modified-equation coefficients for Euler on ẏ = −y² from the invariance recursion: 1, 1, 3/2, 8/3, 31/6, 157/15 (magnitudes), matching the recipe at order h. Growth (log|a_k| − log(k−1)!)/k → −log 2π, slowly (−1.80 at k = 70; −1.85 at k = 160 in a scratch run). The 2π is a numerical fit, stated as such. Final version of the truncation demo: endpoint via travel-time quadrature plus bisection (see round 3). Best one-step errors are 1.4e−11, 3.3e−9, 1.1e−8, 6.5e−7 and 1.0e−4 at ηS = 0.5–0.9, against e^{−2π/h} = 1.2e−11, 8.0e−10, 1.6e−8, 1.5e−7 and 8.6e−7. The notebook claims agreement only to exponential accuracy.
- Shuffled SGD, exact average over all 24 batch orders, 8 examples, log-cosh losses: slopes ≈ 2.0 (GF), 1.85 (IGR flow, pre-asymptotic at the largest η), 2.98 (L_SGD flow).
- Heavy-ball stability region: spectral radius from characteristic roots on a grid; complex band has |z| = √β to 2e−16; region agrees with 0 < ηλ < 2(1+β) away from the boundary.
- Heavy-ball modified flow (β = 0.5, t = 0.5): slow-manifold start gives slopes 1.05 (rescaled GF) and 2.00 (Ghosh flow); start from rest gives about 0.9 (O(η) start-up offset).

### Checked against sources (local pdftotext of arXiv PDFs; Distill page HTML)
- Barrett–Dherin 2009.11162: Theorem 3.1 (modified loss, local error O(h³) vs O(h²)), Remark 3.4 (local minima remain minima; no escape), the 2D model E = (y − abx)²/2.
- Smith–Dherin–Barrett–De 2101.12176: eqs. (1), (2), (12)–(20) (mean over batch orders with batches fixed; error O(m³ε³)); eq. (21)–(22) with Γ = (1/N)Σ‖∇C_i − ∇C‖² and the (N−B)/(N−1) factor; "no noise in the iterates of SGD after completing one epoch" at first order; heavy-tail caveat; experiments (16-4 Wide-ResNet, CIFAR-10; explicit regularizer at small LR; η/B).
- Ghosh et al. 2302.00849: Theorem 4.1 (update convention, transient factors, eq. (5) modified loss h(1+β)/(4(1−β)²)), Remark 4.2 ((1+β)/(1−β) at equal effective LR).
- Kovachki–Stuart 1906.04285: eq. (7) family (a = 0 HB, a = λ NAG), λ̄ := (1−λ)⁻¹, Theorem 4 (α = ½(1+λ−2a(1−λ))), eqs. (16)–(22), Theorem 7 (initial data on the invariant manifold).
- Cattaneo–Klusowski–Shigida 2309.00079: Definition 1.1 (bias-corrected Adam, ε inside the square root), eqs. (2)–(7), Table 1, "RMSProp by setting β = 0".
- Rosca et al. 2302.01952: principal flow formula; "When λ0 > 1/h however, the PF is complex and the BEA series diverges."
- Goh, Distill: update rule, region 0 < αλ < 2+2β, critical damping β = (1−√(αλ))², optimal α, β and rate. The displayed complex-regime modulus is written as √((1−αλ+β)² + |(−αλ+β+1)² − 4β|) = 2√β; with the ½ prefactor of the roots it is √β. Our grid computation confirms √β.
- Hairer–Lubich (2000), open-access chapter (unige:12439): "formal and usually divergent series"; exponentially small estimate O(e^{−γ/h}) with N ~ const/h for analytic f; references Benettin–Giorgilli 1994, Hairer–Lubich 1997, Reich 1997 (habilitation).

### Not verified
- HLW §IX.7 theorem numbers and constants (paywalled). The text cites the section, not a theorem number, and describes the constant γ only qualitatively.
- The claim that γ is "roughly the curvature scale, so the expansion parameter is effectively hS" is a heuristic (exact for the 1D example, where the minimal term is e^{−4π/(ηS)}). Worded as such.
- Cohen et al. 2021's (2+2β)/η heavy-ball threshold is taken from TD_LINKS item 13 (full text checked there), not rechecked here.

## Critique rounds

### Round 1
Critics: design/pedagogy, Claude Opus subagent; technical, Codex (high effort, read-only). Both ran all code cells.

Technical (Codex), all accepted after checking:
1. "ηS" was presented as the general validity parameter. Counterexample L = x + K⁻²sin(Kx): |L''| ≤ 1 but f3(0) = (K/12)(1+1/K)² (I rechecked by hand). Also minibatch curvature can be much larger than full-loss S. Fix: §1.3 now has a "What sets the scale" paragraph. ηS is exact for quadratics and necessary but not sufficient in general. The SGD paragraph now names mη times minibatch curvature. Removed "every term is of the same size".
2. The truncation demo used travel time ∫dy/F with Gauss–Legendre quadrature, which returns finite numbers when a truncated field has a zero in the interval. Fix: the demo now integrates each truncated flow to its actual endpoint with RK4 (all orders vectorized). The minima are unchanged (1.4e−11, 3.3e−9, 1.1e−8, 6.5e−7), so the earlier conclusions stood, but the measure is now honest.
3. Least-term estimate overclaimed as the theorem's constant. Fix: labelled a conditional heuristic, with the fit stated as numerical only.
4. Strict minimum with a singular Hessian need not attract gradient flow. Fix: Exercise 1.1 now assumes H* ≻ 0, adds a stable-manifold remark for the GD instability, and notes why positive definiteness matters.
5. Heavy-ball O(η²) needs an initialization qualification. Fix: the §1.5 text now says the formula holds started on the slow manifold, and that from rest there is an O(η) offset (Ghosh et al. keep β^k terms). The Ghosh reading blurb was rewritten to match.
6. Adam formula is full-batch. Fix: stated.
7. "No real flow of the same dimension" is false in d ≥ 2 (a rotating flow can produce −rI). Fix: restricted to real *gradient* flows (e^{−ηA} with A symmetric is positive definite), with the rotating counterexample mentioned.
8. **Error**: §1.1 said GD decays "slower" than the flow for ηλ < 1. It is faster (1 − z < e^{−z}). Fixed.
9. Imbalance under the IGR flow: the multiplier was written e^{−η²r²} with r frozen. Fix: integral form, agreement with GD through O(η²) per step, O(η³) differences.
Lesser items, also fixed: notebook rebuilt (stale cell); flatness stated as Barrett–Dherin's prediction, not a theorem; 1.2.d.4 qualified (convergence, D0 ≠ 0); critical damping restricted to ηλ ≤ 1; heavy-tail remark removed (not meaningful for a finite dataset); "a bit more than 2" restricted to B ≪ N; Adam time convention stated; EoS stated as typical for full-batch training, with exceptions acknowledged.
Codex confirmed independently: the Kovachki–Stuart sign error and the Goh factor-of-2 display slip.

Design (Claude), accepted:
1. §1.3 conflated divergence with the step ceasing to be a flow map. 1.3.b's ηS = 1 example was wrong: Φ'(y) = 1 − 2hy vanishes at ηS = 1. Fix: §1.3 rewritten around "two different ways the expansion fails"; 1.3.b uses ηS = 0.1, 0.5, 0.8 and explains the breakdown as ηS → 1; the code adds ηS = 0.9, where the best error is about 100× the least-term estimate.
2. A "Predict the slopes" prompt (forbidden by the brief), and the code exercise was too small. Fix: prompt removed; the stub now also asks for the −∇L̃_SGD vector field; replaced by a substantive question (why the IGR flow does not raise the order here).
3. Exercise 1.1.3 asked a question that needed §1.2. Fix: moved to Exercise 1.2.c.3, with the answer reworked (the linearized truncation is −Σ η^j H^{j+1}/(j+1), so every truncation is stable).
4. Slow manifold was unexplained. Fix: a paragraph in the §1.5 text. The mass-term ODE is now described as a picture of the transient, not a higher-order correction.
5. f1/f2 naming clash in 1.4.a. Fixed (f2 throughout).
6. Expansion parameter overstated (same as Codex 1).
7. Code cell issues: 2π presented as a fit; "best n" legend removed, with the broad minimum explained in text; h = 0.2 dropped (at the double-precision floor).
8. One-epoch comparison flips the slopes relative to §1.2. Explained in the exercise text.
9. Routes line added to the pitch; "Code cells check each derivation" softened; checks file pointed to once in the intro.
10. Exercise 1.1.1 (trivial restatement) cut.
11. Adam box shrunk to a short pointer with the formula and its scope.
12. Kovachki–Stuart note: the details moved to the reading annotation; the answer keeps a short note plus the Nesterov constant.
13. Phrasing: "less than half"; the IGR shift is small where IGR is accurate (now in the §1.2 text); the attribution "regime Smith et al. assume" was checked (2101.12176, "we assume that the minibatch gradients are diverse, such that (Γ(ω)/B) ≫ ‖∇C(ω)‖²") and kept.
Not acted on: item 14, the instructor's "is the Taylor approximation even valid?" sits under SDEs in the topic note. This section answers it for the deterministic expansion and points to td2_3_sgd_sde, which must take it up (flagged in the report).

Also fixed by me during round 1: the Smith et al. reading blurb claimed the explicit regularizer helps "at larger batch sizes". The paper's claim is at small learning rate. Corrected against the abstract and Fig. 3.

### Round 2
Same two critics, fresh instances, on the revised draft.

Technical (Codex), all accepted:
1. The heavy-ball slow manifold needs fixed β < 1 and small enough η; it is not uniform as β → 1. On a quadratic the slow root meets the fast one at ηλ = (1−√β)², which is about 0.003 at β = 0.9 (Codex's example: β = 0.9, η = 0.01 gives complex roots 0.945 ± 0.0835i). Fix: stated in the §1.5 text with that threshold.
2. "Flatter endpoint" needs η|r_k| < √2 at every step. Codex found a converging GD run, (2, −2.8) at η = 0.24, that ends sharper than gradient flow. Fix: condition added to 1.2.d.4.
3. Local instability was promoted to a measure-zero basin claim. A noninvertible GD map can throw far-away points exactly onto the minimum. Fix: "cannot attract a whole neighbourhood"; the stable-manifold remark is restricted to trajectories that stay nearby; the "generic" caveat is made explicit.
4. Rosca et al.'s principal flow resums only the gradient-and-Hessian terms. Fixed in the reading.
Lesser, accepted: the damped ODE's slow dynamics reproduce the modified flow (new Exercise 1.5.b.3, checked symbolically as check 7). "Numerical evidence consistent with" divergence. The truncation minimum is described as not sharply located. The SGD remainder needs third derivatives. "No flow exists" renamed to "no real gradient flow reproduces the step".

Design (Claude), accepted:
1. "Most implicit regularization results are instances" overclaimed. Now limited to finite-step-size results, and implicit biases of gradient flow itself are distinguished.
2. Momentum's stability gain at equal effective learning rate, (1+β)/(1−β), added.
3. §1.3 now opens with a short answer, the duplicated bullet is merged, and 1.3.a–b are marked optional in Routes.
4. The 1.5.a code stub was plotting busywork, so it was converted to a demo `code()` cell.
5. The KS erratum appeared twice. It is now in full in the 1.5.c note and as one line in the reading. Goh's slip is worded as "omits the ½".
6. The 1.4.c IGR error exceeds gradient flow's, which looked broken. Explained: GF residual −(η²/2)(mHg + V), IGR residual −(η²/2)V, with V = ½∇Σ‖δ_k‖², and the two terms partly cancel here. Derivation checked by hand.
7. Intuition added for the sin counterexample (step length vs the scale on which the Hessian changes).
8. Minibatch EoS sentence added (Andreyev–Beneventano, linked; TD_LINKS item 18 checked against full text).
9. The start-from-rest offset is now derived (deficit ηg/(1−β) decaying like β^k gives a displacement of about ηβg/(1−β)²).
10. Wording fixes, all applied.
Also added links for every paper named in the text (Polyak 1964, Benettin–Giorgilli 1994, Hairer–Lubich 1997, Reich 1999), DOIs checked via Crossref.

### Round 3
Round 2 still raised substantive claim-boundary issues, so I ran a third round, same setup.

Technical (Codex), all accepted:
1. Exercise 1.1.2: converging trajectories lie on a local *center-stable* manifold (|1−ηλ_i| ≤ 1), not a stable one. Codex's counterexample: L = 3x²/2 + y² − y⁴/4 at η = 1. Fixed.
2. The fixed-step RK4 in the truncation demo failed on the stiff high-order fields: NaNs and wrong tail values, e.g. order 40 at h = 0.45 gave 0.0270 against a true 0.0248. Fix: endpoint by travel-time quadrature (Gauss–Legendre, 120 nodes) plus bisection. The search stops above the first equilibrium of F_n. Orders whose field points the wrong way at y0 are reported as never reaching the GD point; I checked, and they have F_n(1) > 0 because the a_k change sign at high k. Cross-checks: the n = 1, h = 0.4 endpoint error is 0.1143, the exact value 1/1.4 − 0.6; the h = 0.45 tails at n = 45 and 49 give 0.053 and 0.070, matching Codex's Radau values. Minima unchanged.
3. The GD imbalance contraction needs |1 − η²r²|. Fixed.
Lesser, accepted: the §1.3 summary now names higher derivatives; the negative-multiplier obstruction is scoped to quadratics and linearizations; "the code shows the series diverges" became "consistent with"; the IGR endpoint-shift sentence was decoupled from accuracy; Kovachki–Stuart's O(η) statement is attributed to their Theorem 4 (method started from rest); batch sharpness is defined.

Design (Claude), accepted:
1. §1.3 called the wrong failure "the real wall", contradicting the notebook's own captures printout (81/42/16%). Rewritten as three failures in order of practical importance: first-order truncation error, then no gradient flow for ηλ ≥ 1, then divergence. Added a hedged EoS sentence: only the top directions have ηλ ~ 1, which motivates modelling them separately.
2. The momentum 19× penalty was stated far from its validity range. Added: in effective-step units the expansion needs η_eff λ ≪ (1−√β)/(1+√β) ≈ 0.026 at β = 0.9, about 40× smaller curvatures than GD's IGR.
3. The last heavy-ball plot looked like it favoured starting from rest. Added a note after the cell.
4. Exercise 1.3.b: the (k−1)!/(2π)^k fit is right only to exponential accuracy; the prefactor varies by 10²–10^4.5 at the relevant k. Text now says so; the near-agreement of the minima is described as consistent with the exponent, not as confirmation.
5. Added the overshoot intuition for f2, and the first-order recipe f2 = d2 − ½f′f, to the §1.2 default text.
6. Two over-hedged answers restructured (1.2.d.4 now lists its assumptions (i)–(iii); a defensive parenthetical in 1.1.2 was cut).
7. Removed the SGD paragraph from §1.3 (duplicated in §1.4); the short answer was shortened.
8. Exercise 1.5.b.3 marked optional. Not taken: shrinking the KS note in the 1.5.c answer. Round 2 asked for the details there and a one-liner in the reading, and the build decisions asked for the sign question to be settled visibly.
9. Exercise 1.2.c.3 argument replaced by the cleaner one: the flow-map Jacobian e^{ηDF̃} = I − ηH* gives DF̃ = log(I − ηH*)/η.
10. Figures: the trajectory panel was replaced by sharpness along the run; the heavy-ball heatmap masks the unstable region; labels moved.
11–14. Wording: "at all but the largest η"; Routes reworded; "One useful description" for central flows.

### Round 4 (targeted technical check of the round-3 changes; Codex)
1. Codex said the batch-sharpness formula should be a ratio of expectations, citing arXiv 2412.20553v3. **Rejected after checking:** the current version (v5) defines Batch Sharpness as E_B[g_Bᵀ H_B g_B / ‖g_B‖²] (Definition 3, eq. 3). Its Appendix J.3 says earlier versions used the ratio-of-expectations "Modified Batch Sharpness". This agrees with TD_LINKS item 18. The notebook now says so explicitly.
2. Two leftover "cannot converge" / "needs η < 2/S" phrasings turned instability into impossibility. Codex gave an exact one-step convergence to (1,1) at ηS = 20/9. Fixed ("locally unstable"; "generically").
3. The EoS bulk-spectrum sentence is now conditional on a spectral gap.
4. The short version now carries the scope (quadratic or near a fixed point; ηλ ≥ 1).
5. "Up to algebraic factors" became "to exponential accuracy".
Codex confirmed the travel-time endpoint computation agrees with independent integration to 2.5e−13, and that the NaN counts are meaningful.

## Remaining uncertainties
- HLW ch. IX theorem numbers and constants were not checked (paywalled). The text cites §IX.7 and the open Hairer–Lubich (2000) chapter, and describes γ qualitatively.
- The (k−1)!/(2π)^k growth is a numerical fit over k ≤ 70 (k ≤ 160 in scratch). No source was found stating it for this map. It is labelled a fit.
- The heavy-ball validity radius (1−√β)² is derived on a quadratic (where the slow root's series in ηλ has a branch point). For nonlinear losses it is stated as the condition on every curvature, which is a heuristic extension.
- Kovachki–Stuart sign error: confirmed symbolically by me and independently by Codex in two rounds. No published erratum was found (not searched beyond the JMLR page and arXiv v-history listing in TD_LINKS).
- Cohen et al. (2021) heavy-ball threshold 2(1+β)/η and their "exceptions": taken from TD_LINKS item 13 and Codex's reading of their Appendix A, not reread by me.
- The design critic flagged that the instructor's "is the Taylor approximation even valid?" sits under SDEs in the topic note. This notebook answers it for the deterministic expansion; td2_3_sgd_sde must take up the stochastic version.

## Scratch-directory note
A shared scratchpad file (`render.py`) used for rendering figures was overwritten by another agent's script during round 3. I then used a private subfolder. No project files were affected.

## Readability pass (2026-10-09)
Following `td/READABILITY_BRIEF.md`. The mathematics, code and claims are unchanged, and no code cell was touched (outputs come from the cache).

Changes made by me:
- **Opening.** The dense pitch paragraph, which stated $L+\tfrac\eta4\|\nabla L\|^2$, $\eta/B$, $1/(1-\beta)$ and $\eta S$ before any notation, became a plain-language question plus the method in words. The results now appear in words; formulas come after the Notation box. Dropped the `{{VARIANT}}` label and the "solutions variant / exercises variant" paragraph; the kit's header already says solutions are folded. Table entries now use only symbols from the Notation box (§1.4 is in words, and β is named in §1.5). Dropped the "Uses" column, which duplicated Routes. "(next to this notebook)" became "(in the course's `td/` folder)".
- **§1.1.** Added an opening question and a "Notation here" line (GD, gradient flow, $t=k\eta$). Named $S$ as the sharpness at its first use.
- **§1.2.** Added an opening question and a "Notation here" line ($L,g,H,\eta$; $y,h$ for a general method). Turned the "three limits on the interpretation" paragraph into a numbered list.
- **§1.3.** Replaced the "Short version" spoiler with the question (convergence? guarantees? what sets the scale? where does training sit?) and a "Notation here" block that restates the modified field, the truncation $\tilde f_{[r]}$, and GD's $f_2$ / IGR flow. Glossed local and global error inline. Defined $g_B,H_B$ in the batch-sharpness bullet.
- **§1.4.** Added an opening question that restates IGR. "The result:" became plain narration after the setup. Said in words what $\Sigma$ is.
- **§1.5.** Added an opening question and a "Notation here" block ($\beta$, $S$, the GD threshold from §1.1, IGR from §1.2). Restated the first-order backward-error recipe ($f_2=d_2-\tfrac12f'f$) where the slow-manifold argument uses it. Defined $\kappa=S/\mu$ and pointed to Exercise 1.1 for GD's rate.
- **Readings.** Barrett–Dherin's $\frac h4$ became $\frac\eta4$.
- **MDL links.** This notebook has none, so there was nothing to repoint.

Fresh readability critic (Claude Opus subagent; it read the page as a student landing on §1.5, §1.3, §1.4 and Exercise 1.3.b from the sidebar). Its ranked findings and what I did:
1. §1.5's quadratic paragraph used the effective learning rate "(defined below)" and the penalty factor before either was introduced. Moved the matched-effective-LR comparison to just after $\eta/(1-\beta)$ is defined. I kept the regime and critical-damping facts in the default text: they are defined where stated, and the section shape asks the default text to stand alone.
2. The Kovachki–Stuart erratum note was a dense aside in the main flow, with undefined $\ddot u$, $\Phi$ and a clashing $\lambda$. Folded it into `<details>`, glossed their $u$, $\Phi$ and $\bar\lambda$ as their notation, and in the 1.5.b solution wrote their mass coefficient in our $\beta$.
3. The slow-manifold paragraph packed three ideas together. Split out "When the slow manifold exists", stated in words first (velocity relaxes much faster than position drifts), and derived it from the root picture in place rather than from "the slow root of Exercise 1.5.a".
4. Exercise 1.3.b used $F$, $a_k$ and $h$ from 1.3.a. Added a one-line restatement.
5. §1.3's divergence bullet previews the 1.3.b exponent, and the 81/42/16% figures repeat the 1.2.d commentary. Kept both: they are short and keep §1.3 self-contained.
6. The batch-sharpness version history in the main text. Shortened it to "definition as in their current version, v5".
7. "Exercise 1.1.2"-style item references with no stated convention. Rewrote all five as "Exercise 1.1, item 2" and so on.
8. The truncation-code commentary was long and mixed $h,u,\eta,S$. Rewrote it to translate $e^{-2\pi/h}$ into $e^{-4\pi/(\eta S)}$ at $y_0=1$ explicitly, and cut the aside on the minimizing order. The prefactor hedge stays.
9. Exercise 1.4.a did not restate $g,H$, and $k/j$ indexing clashed with $t=k\eta$. Added both.
10. "Going further: adaptive methods" left $\epsilon$ undefined and said "$h$ … as here". Fixed: $\epsilon$ is the second-moment constant, and $h$ is our $\eta$.
11. The opening was padded. Dropped the Uses column and the duplicate solutions line.
12. "Next to this notebook" was old-format wording. Fixed.

Checks: `python td/build_td2_1_flows.py` and `--check` pass (44 cells, 7 code cells, 14 solutions). `python td/checks/td_crosslinks.py` reports 55 cross-links, 0 bad.
