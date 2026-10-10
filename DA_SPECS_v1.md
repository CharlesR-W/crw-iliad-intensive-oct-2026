# Data Attribution day: notebook spec v1.1 (2026-10-09)

**Status:** approved by CRW 2026-10-09 with the decisions below. v1.1 draft for CRW. v1.1 folds in a Codex technical review of v1 (`reviews/da/2026-10-09-spec-v1-codex.md`); I checked its key claims numerically before adopting them. CRW's instructions quoted below are binding; the rest is my proposal.

## What CRW asked for

> I'm going to teach the data attribution day for Iliad as well. … for DA we should really have something less like multiple tracks, and mostly have everything on the same 'path'. I think for DA the material is especially good so we'll largely be reformatting it to notebooks from lectures.

> Present the appropriate papers with maybe a bit of context, and we'll have our own 'optional annotated derivations/walkthrough' as an alternative. We can present a common 'core' in one notebook that will get them enough context to go look at the various methods. … It's fine to repeat the BIF bit [overlap with the SLT day is fine].

## CRW decisions (2026-10-09) — binding, override anything below

1. **Louis:** don't involve him. **"Be correct but not pedantic."** Fix the errata silently in our own text. Don't write "the source says X, but…" commentary. Add caveats only where they change what a student would conclude. Scope boxes are 1–3 lines.
2. **Time budget:** DA1 by lunch; extensions optional.
3. **Readings:**
   - CRW knows and likes **TRAK**; it's a regular reading.
   - Louis's own references are regular readings.
   - **Every other paper this spec marks *(proposed)*** goes under a separate line at the end of the reading list, headed **"Claude-suggested extras"**. That covers Datamodels, Basu et al., Data Shapley, LoGra, LESS, Scieur, Schioppa, and Hammoudeh & Lowd.
4. **DLN walkthrough (DA2 §4):** keep it; re-derive and verify.
5. **EK-FAC walkthrough (DA2 §1):** keep it.

## Source material

- **The notes.** Louis Jaburi, *Data (Attribution) for Alignment*, Iliad Intensive notes B.5 ([page](https://iliad-intensive.org/learning/data-attribution/)).
  - Files: `da/source/data-attribution.tex` (with solutions) and `da/source/data-attribution-nosol.pdf`.
  - Licence: CC BY 4.0 ([licence](https://iliad-intensive.org/license/)). Every page credits it: "Adapted from Louis Jaburi's Iliad notes (CC BY 4.0), with changes."
  - **Refer to source items by LaTeX label** (e.g. `ex:ols-if`, `ex:unroll-to-if`), not by number, because the PDF and web numbering differ.
- **What the notes contain.** One target, $\partial\phi(\pi(\beta))/\partial\beta_i$, where $\beta$ are per-example data weights and $\phi$ is a scalar observable. The methods differ in what $\pi$ is:
  - §1 causality;
  - §2 classical and modern IFs;
  - §3 Bayesian IFs;
  - §4 unrolling;
  - §5–6 are empty stubs.
- **Cross-day hooks that already exist:**
  - TD NB1 (Saxe mode dynamics) feeds the deep-linear-network walkthrough.
  - TD §2.3 (Langevin, Gibbs law) feeds SGLD/BIF.
  - TD §6.1 (Gauss–Newton, K-FAC) ends "The Data Attribution day picks up from there."
  - The SLT day covers degeneracy and the $a\cdot b$ model.
  - DA must not *depend* on TD: students may have skipped those notebooks. So restate what is needed and link TD for more.

## Shape of the day

Two notebooks and one short lecture.

| | File stem | What | When |
|---|---|---|---|
| L1 | `da/lectures/L1_attribution.md` | ~25 min opening (outline + speaker notes, TD style) | 10:00 |
| DA1 | `da1_core` | **Core.** The question, the formula, classical IFs, what breaks and the standard fixes, how attribution is evaluated, and a map of the methods | 10:25–12:00 |
| DA2 | `da2_methods` | **Methods.** An ordered tour of the method families. Each stop has papers with context, plus an optional annotated walkthrough as an alternative to reading the paper | afternoon |

**One path.**
- DA1 is linear. Its required spine is sized for ~80 min of student time; everything beyond the spine is a marked extension.
- DA2 is a single ordered tour: everyone reads each stop's 1–2 paragraph capsule, in order (~30 min total). Then students go deep where they like, by paper or by walkthrough.
- There are no "routes".

**Lens across both notebooks: similar regularization, different responses.**
- The practical estimators share a recognisable structure: a curvature matrix, a regularization of its small eigenvalues, a training gradient, and a query gradient. In idealized limits they reduce to a damped-IF-like expression.
- They are **different responses to different questions**:
  - a stationary point of a proximal objective;
  - one training trajectory;
  - a local posterior;
  - a regression over retrainings.
- The damping-like quantity means something different in each.
- This lens is stated with its assumptions, never as an equivalence (see DA1 §6 for the precise table).

### Format

Use the TD format (`td/README.md`): each notebook is an HTML page plus a runnable CPU `.ipynb`.
- Builders are `da/build_<stem>.py`, importing the TD kit (`td/nbkit_td.py`, `td/htmlkit_td.py`). Report any kit change needed (e.g. a credit footer) rather than editing the kit.
- Standard rules carry over:
  - folded sections;
  - ARENA ratings on every reading;
  - books only in `references(book(...))`;
  - every URL fetched before use;
  - CPU code cells under ~30 s each and ~3 min total;
  - numpy/matplotlib, plus torch CPU where autograd or HVPs help.
- **Readings name the specific sections or figures to read** whenever a paper is longer than ~20 pages (e.g. Grosse et al. is 119 pages).

**Conventions, fixed once in DA1 §2 and used everywhere:**
- the summed objective $L(w;\beta)=\sum_i\beta_iL_i(w)$;
- upweighting derivative at $\beta=\mathbf 1$;
- deletion means $\beta_i:1\to0$;
- the sign of "helpful" (an upweight that lowers the query loss has a **negative** influence on that loss);
- for SGD written on a mean loss, the explicit $1/n$ or $1/b$ is shown.

Every imported derivation is converted to these conventions.

**The walkthrough device (DA2).** A walkthrough is a `### Walkthrough: <topic>` subsection, folded. Its first line names the paper sections it substitutes for and the **shared learning objective**.
- Use a walkthrough only where a derivation can genuinely replace reading the paper. For empirical papers (e.g. Grosse et al.'s findings), give a reading guide instead.
- Structure: notation restated → numbered steps, each with 1–2 sentences of annotation (why this step, which assumption it uses) → the step as an exercise with a folded worked solution → a **scope box** stating exactly what was proved and what the paper's general theorem additionally assumes → an optional numerical-check code cell.

**DA2 stop shape:**

```
## N. <Method family>
<Capsule, 1–2 short paragraphs: the question this method answers, how it reads π(β), its main result and main caveat. Everyone reads this.>
<Read: 1–2 papers via reading(), each with 1–3 sentences of context and which sections/figures to read.>
<Also: 0–3 one-line further papers.>
### Walkthrough: <result>      (optional, folded)
### Code: <check/phenomenon>   (optional, folded)
```

---

## DA1 — Core: what did this training example do? (`da1_core`)

**Goal.** By lunch a student can:
- write down and interpret the (damped, Gauss–Newton) IF;
- say what assumption each modification addresses;
- distinguish a method's *target* from its *estimator*;
- know how attribution claims are evaluated;
- read any DA2 capsule.

**Keystone readings:**
- Koh & Liang 2017 ([1703.04730](https://arxiv.org/abs/1703.04730));
- Bae et al. 2022 ([2209.05364](https://arxiv.org/abs/2209.05364)).

Required spine is §1–§6 at the stated lengths. Items marked *(extension)* are folded and optional.

### 1. The question (~10 min)
- Purpose: data attribution is causal analysis of a training run, and naive counterfactuals fail in known ways.
- Text:
  - mech interp is to a forward pass as data attribution is to a training run;
  - retraining is expensive;
  - Lewis's counterfactual;
  - overdetermination (the croissant example, ~4 lines);
  - non-transitivity (the boulder, 3 lines).
- Shapley in one paragraph:
  - it's a finite-coalition attribution, not a derivative at $\beta=\mathbf 1$;
  - exact enumeration needs exponentially many coalition values, and practical estimators sample;
  - it still splits credit between sufficient causes.
- Reading: Mueller 2024 ([2407.04690](https://arxiv.org/abs/2407.04690)). Also: Ghorbani & Zou 2019, *Data Shapley* ([1904.02868](https://arxiv.org/abs/1904.02868)) *(proposed)*.

### 2. Targets and conventions (~10 min)
- Purpose: fix notation and separate *what we want to know* from *how we estimate it*.
- Text:
  - $\beta$, the summed weighted loss, the observable $\phi$, the composite $\beta\mapsto\pi(\beta)\mapsto\phi$;
  - the conventions above;
  - local derivative vs. LOO vs. large-subset deletion as three different targets.
- **Map table** (targets):

  | Method | $\pi(\beta)$ | DA2 stop |
  |---|---|---|
  | classical IF | the stationary branch through $w^*$ | — |
  | PBRF | stationary point of a proximal objective around $w^*$ | 1 |
  | unrolling | $w_T(\beta)$ on this run, fixed seed and order | 2 |
  | BIF | the local posterior $p_\beta$; read $\phi$ as $\mathbb E_{p_\beta}\phi$ | 3 |
  | datamodels | a fitted predictor of retraining outcomes | 5 |

  TRAK is noted separately: a linearized, projected-gradient analytic estimator evaluated against datamodel-style targets. It is not a regression on retrainings.

### 3. The classical influence function (~25 min)
- Purpose: the implicit function theorem gives $-H^{-1}\nabla L_i$, and in linear regression we can check it against exact deletion.
- Text:
  - Hampel's influence on the mean (3 lines);
  - the IFT derivation (~6 lines, in text). The IFT gives a **local stationary branch**; identifying it with "the" argmin needs extra stability assumptions;
  - the three-factor reading: the pull of example $i$, rescaling by curvature, projection onto $\phi$.
- Required exercise: OLS, from `ex:ols-if` parts (a), (b), (e):
  - minimizer, IF, and exact deletion $-A^{-1}x_jr_j/(1-h_j)$;
  - the $h_j\to0$ / $h_j\to1$ discussion;
  - state that deletion requires $h_j<1$ (identifiability).
- *(Extension)* `ex:ols-if` (c)–(d): Sherman–Morrison for general $\beta_j$ and the geometric series in leverage.
- *(Extension)* `ex:if-derive`: where invertibility is used.
- Remark: leverage as a Mahalanobis-type norm (uncentered second moment unless inputs are centered).
- Code (shown, not an exercise): Louis's IF-vs-LOO figure, regenerated; IF vs. exact deletion for every point, coloured by leverage.

### 4. What breaks for neural networks, and the standard fixes (~20 min)
- Purpose: three modifications turn the textbook formula into the one people use, and each changes the question being answered.
- Text, a short paragraph each:
  1. **Degenerate minima.** The Hessian is singular. Damping $(H+\lambda I)^{-1}$ equals adding a proximal $L_2$ term. One line to the SLT day.
  2. **Negative curvature: Gauss–Newton.** Define it here, self-contained:
     - $G=\sum_i J_i^\top H_{y,i}J_i$, which is PSD for convex output losses;
     - $H=G+\sum_{i,c}\partial_{f_{ic}}\ell_i\,\nabla^2 f_{ic}$;
     - the residual term vanishes only under extra conditions (zero output-loss gradients, or a model linear in its parameters), **not** at every critical point. This corrects the source claim at `subsec:if-nn`;
     - link TD §6.1.
  3. **Non-convergence: PBRF.** The damped GN IF is the exact derivative of a *different* objective (Bae et al.'s "if IFs are the answer, what is the question?"). It isn't a repaired estimate of LOO retraining.
- *(Extension)* exercise `ex:if-degenerate` (the $a\cdot b$ toy), with a corrected solution (errata E4). Assume $\bar z\neq0$.
- Code: $\mathcal I_\lambda(z_i,a^2)$ vs. $\lambda$ for several minimizers on the hyperbola. It saturates to a finite, minimizer-dependent limit.
  - Caption it as "parameterization dependence of the damped IF", **not** as Bae's proximity gap.
  - *(Extension, if cheap)* A second cell measures a real proximity gap: proximal vs. unregularized retraining response on a tiny MLP.
- Readings:
  - Koh & Liang 2017;
  - Bae et al. 2022 (read §§1–4);
  - Basu et al. 2021, *Influence functions in deep learning are fragile* ([2006.14651](https://arxiv.org/abs/2006.14651)) *(proposed)*.

### 5. How do we know an attribution is right? (~20 min)
- Purpose: fill Louis's empty "practical" section with the evaluation question, which students need in order to judge anything in DA2.
- Text:
  - **target vs. estimator.** A method can compute its own derivative exactly and still predict large deletions poorly;
  - three checks, from least to most demanding: (i) local finite differences of the method's own target; (ii) LOO; (iii) subset deletion and the linear datamodeling score (LDS), the rank correlation between summed attribution scores and actual $\phi$ over random subsets;
  - LDS tests ranking, not calibration;
  - single-model vs. distributional targets: fixed seed, or averaged over seeds;
  - estimands of gradient-similarity methods. TracIn allocates the *accumulated first-order loss change along the trajectory*, which is a different estimand from retraining sensitivity, not "mere similarity".
- Code exercise, scaffolded: $\ell_2$-regularized logistic regression (Newton solver supplied; ~100 fits run in well under a second).
  - Students complete only the scoring function, then interpret the results.
  - Compare exact IF, damped IF and a gradient-dot-product baseline on (i), (ii) and (iii) with 50% subsets.
  - Define the observable, sign convention, subset distribution and regularization scaling in the cell.
- Readings: Ilyas et al. 2022, *Datamodels* ([2202.00622](https://arxiv.org/abs/2202.00622)) *(proposed)*; Park et al. 2023, *TRAK* ([2303.14186](https://arxiv.org/abs/2303.14186)) *(proposed)*.

### 6. A lens for the methods, and where to go next (~5 min reading; figure optional)
- Purpose: preview, with assumptions, how DA2's methods relate. Then hand off.
- Text: a table of spectral filters $F(s)$ acting on the training gradient, each with its matrix, units and assumptions. Use Codex's corrected table:

  | Method | Filter | Holds when |
  |---|---|---|
  | IF | $1/s$ | eigenvalues of an invertible $H$ |
  | damped Hessian IF | $1/(s+\lambda)$ | — |
  | PBRF | $1/(s+\lambda)$ | $s\in\mathrm{spec}(G)$, a different matrix from $H$ |
  | frozen-curvature GD response | $[1-(1-\eta s)^K]/s$ | constant curvature and gradient |
  | SOURCE, one segment | $(1-e^{-\eta Ks})/s$ | after the stationarity and exponential approximations; crossover near $s\sim1/(\eta K)$, but not identical to ridge (at the crossover the filter is $0.63\,\eta K$, vs. $0.5\,\eta K$ for ridge) |
  | local BIF | $\approx1/(s+\gamma)$ | Gaussian leading order at a stationary minimum |
  | Bayesian linear regression | $1/(s+\sigma_{\rm noise}^2/\tau^2)$ | $s\in\mathrm{spec}(X^\top X)$, unscaled squared-error gradients; in likelihood-Hessian units the damping is $\tau^{-2}$ |

  - Say plainly what the table hides: general unrolling has changing gradients and non-commuting Jacobians; multi-segment SOURCE propagates through later segments; BIF beyond leading order; datamodels have no such form.
- *(Optional)* Interactive figure: the filters on linear and log axes, with sliders, showing $F(0)$ explicitly.
  - Self-contained HTML in `da/assets/interactive/`, checked by a script.
  - Lower priority than everything else in DA1; build it last.
- Close with the DA2 tour order and one line per stop.

---

## DA2 — Methods: an ordered tour (`da2_methods`)

All stops assume DA1 only. Everyone reads every capsule in order; depth is optional.

### 1. Influence functions at scale
- Capsule:
  - EK-FAC is one scalable approximation to damped GN IFs;
  - inverse-vector products are computed without forming an inverse;
  - what this bought for LLMs.
- Read: Grosse et al. 2023 ([2308.03296](https://arxiv.org/abs/2308.03296)). Point to specific sections:
  - the method (EK-FAC, query batching);
  - §§5.2–5.3 findings: role-play examples, heavy-tailed or broad influence distributions, and increasing abstraction with scale, presented as separate results.
  - Don't attribute claims (e.g. sycophancy) the paper doesn't make.
- Also:
  - Martens & Grosse 2015, K-FAC ([1503.05671](https://arxiv.org/abs/1503.05671));
  - George et al. 2018, EK-FAC ([1806.03884](https://arxiv.org/abs/1806.03884));
  - Choe et al. 2024, LoGra ([2405.13954](https://arxiv.org/abs/2405.13954)) *(proposed)*.
- Walkthrough (new; Louis names EK-FAC without deriving it). For one linear layer:
  - define the curvature target: GN/Fisher with **model-sampled** pseudo-labels, not the empirical Fisher;
  - the Kronecker approximation $\mathbb E[aa^\top]\otimes\mathbb E[gg^\top]$ and its independence assumption;
  - applying a damped inverse via two eigendecompositions, with damping on the **joint** eigenvalues;
  - EK-FAC's eigenvalue refit.
  - This substitutes for the method section of Grosse et al., not its findings.
- Code: on a tiny MLP, compare exact GN, block-diagonal GN, K-FAC and EK-FAC with identical damping, so block omission and Kronecker error are separated. Report the IF correlation with exact.

### 2. Unrolling: differentiate the actual training run
- Capsule:
  - differentiating the trajectory makes order, optimizer and training time part of attribution;
  - TracIn as a cheap trajectory estimand, using checkpoint query gradients; TracInCP is a further approximation;
  - SOURCE and MAGIC as approximate and exact versions.
- Read:
  - Bae et al. 2024, SOURCE ([2405.12186](https://arxiv.org/abs/2405.12186)), §§3.2–3.3;
  - Ilyas & Engstrom 2025, MAGIC ([2504.16430](https://arxiv.org/abs/2504.16430)), §4 and Fig. 4. Near-perfect LDS holds **at small deletion fractions with fixed randomness**; exact metagradients don't make finite deletions exact.
- Also:
  - Pruthi et al. 2020, TracIn ([2002.08484](https://arxiv.org/abs/2002.08484));
  - Engstrom et al. 2025 ([2503.13751](https://arxiv.org/abs/2503.13751));
  - Scieur et al. 2022, *The curse of unrolling* ([2209.13271](https://arxiv.org/abs/2209.13271)) *(proposed; the notes have a TODO to cite it)*;
  - Schioppa et al. 2023 ([2305.16971](https://arxiv.org/abs/2305.16971)) *(proposed)*.
- Walkthrough A: the unrolling formula, from `ex:unroll-derive` (a)–(b). Open with Louis's forward-pass ↔ training-run table. Part (c) becomes a **frozen-preconditioner approximation**, labelled as such:
  - the full Jacobian adds $D P(w)[v]\,g$;
  - Adam carries data-dependent state, handled via the source's full-state metagradient formulation (`subsec:implicit-jvp`).
- Walkthrough B: the SOURCE filter (from `subsec:explicit-unroll`). Segment Jacobian, matrix geometric series, $F_r(s)$, and what the crossover does and doesn't mean (DA1 §6 table).
- Walkthrough C: unrolling → IF, from `ex:unroll-to-if`, re-derived in DA1 conventions.
  - Prove only the **positive-definite, frozen-quadratic, full-batch** case, including the inherited-response term: $r_T=(I-\eta H)^K r_{T-K}+\tfrac1n[I-(I-\eta H)^K]H^{-1}g_k$ for downweighting on a mean loss.
  - Keep the null-space drift computation.
  - The stochastic case goes in a **theorem box** citing Mlodozeniec et al. Thm 2 with its assumptions: bounded iterates, regularity, $g_k\in\mathrm{Range}(H)$, an eigen-gap, and controlled convergence. Robbins–Monro plus convergence alone isn't enough.
- Code: full-batch GD on ridge regression. Unrolled response vs. $K$, against the IF and the frozen-curvature filter, checked by finite differences of actual retraining.

### 3. Bayesian influence functions (overlaps the SLT day by design)
- Capsule:
  - replace the point estimate by a local posterior;
  - the derivative becomes a covariance and needs no Hessian inverse;
  - it can see non-quadratic geometry the IF misses;
  - it is estimated by SGLD, whose mixing is the practical risk.
- Read:
  - Kreer et al. 2025 ([2509.26544](https://arxiv.org/abs/2509.26544));
  - Mlodozeniec et al. 2025 ([2506.12965](https://arxiv.org/abs/2506.12965)): IFs as low-temperature KL-optimal first-order shifts, under stated regularity and spectral-gap assumptions.
- Also: Baker et al. 2025, susceptibilities ([2504.18274](https://arxiv.org/abs/2504.18274)), with one line on the susceptibility family.
- Walkthrough A:
  - BIF = −Cov (`ex:bif-covariance`), with the fluctuation–response remark;
  - Bayesian linear regression gives the damped IF, with units stated as in the DA1 §6 table;
  - BLR is exact because of **both** Gaussian conjugacy and quadratic per-example losses.
- Walkthrough B: the Gaussian-surrogate expansion (`ex:bif-expansion`).
  - Leading term = IF; localization $\gamma$ acts as damping.
  - Use $\approx$ throughout. Under the surrogate, $(1,3)$, $(2,2)$ and $(3,1)$ all enter at the next order; the true posterior adds prior and non-quadratic corrections.
  - Drop the source claims that linear $\phi$ kills all corrections, and that singularity implies a divergent series or a positive-dimensional minimum set ($L=w^4$ is a counterexample to both).
- Walkthrough C (long, marked advanced): IFs as optimal transport (`ex:bif-ot`), nondegenerate case only, with corrections (errata E7):
  - the partition-function term doesn't vanish; it is essential to completing the square;
  - downweighting shifts the mean by $-\mathrm{BIF}$;
  - the Gaussian check omits an $O(\varepsilon^2)$ covariance-change term;
  - the calculation fixes only $r(w^*)$, so there's no uniqueness among vector fields;
  - the singular case needs $g_k\in\mathrm{Range}(H)$.
- Code, as a measured comparison rather than a prescribed outcome:
  - calibrate the covariance estimator on conjugate BLR with exact Gaussian samples;
  - then compare SGLD-BIF with damped IF on a 2-D nonlinear model, using quadrature as ground truth;
  - sweep $\gamma$; report Monte Carlo error and a step-size check;
  - link TD §2.3.

### 4. Influence changes over training
- Capsule: attribution is a time series, not a number. In deep linear networks, coarse structure is learned before fine structure, and an example's influence can change sign as a result. Lee et al. report similar phenomenology in language models.
- Read: Lee et al. 2025 ([2510.12071](https://arxiv.org/abs/2510.12071)), for the phenomenology. **Its Appendix C.6 singular-vector derivatives share the source's error (E5), so don't send students to it for the derivation.**
- Walkthrough: **re-derived, not ported** (errata E5).
  - Setting: a 2-layer linear net with aligned, balanced, small initialization, under gradient flow. Notation as in TD NB1 ($s_\alpha$, $\tau$), restated.
  - The perturbation is an actual example reweighting, which changes both $\Sigma_{yx}$ and $\Sigma_{xx}$. If that's too heavy, explicitly switch to a *cross-covariance intervention* and say so.
  - Use the correct SVD derivative $A_{jk}=(s_kE_{jk}+s_jE_{kj})/(s_k^2-s_j^2)$, with $E=U^\top\,\delta\Sigma\,V$.
  - The time-localization claim must account for $\partial\mathcal G_k/\partial s_k\to1$ after saturation: what is localized is the product with the mode-$k$ residual.
  - The sign flip is demonstrated numerically, with labels and residuals specified, and the sign convention from DA1 §2.
- Code: GD on a 2-mode DLN, fixed initialization. The influence of a "dog" example on a "cat" test loss over training, computed by finite differences of reweighted retraining and compared with the derived formula.

### 5. Learning the map instead of differentiating it: datamodels and TRAK
- Capsule: fit a predictor of retraining outcomes directly (datamodels), or use a linearized, projected-gradient estimator evaluated on that target (TRAK). These are the methods behind the LDS from DA1 §5.
- Readings only: Datamodels; TRAK (both *(proposed)*).

### 6. Open problems and uses
- Short items, readings where they exist:
  - entangled latent concepts (Louis's stub title);
  - attributing behaviours rather than losses;
  - fine-tuning data selection, with Xia et al. 2024, *LESS* ([2402.04333](https://arxiv.org/abs/2402.04333)) *(proposed)*;
  - unlearning;
  - distributional attribution as a well-posed target;
  - scale.
- Also: Hammoudeh & Lowd 2022 survey ([2212.04612](https://arxiv.org/abs/2212.04612)) *(proposed)*.
- Which alignment-use papers to name is CRW's or Louis's call.

---

## L1 outline (≈25 min, `da/lectures/L1_attribution.md`)

1. The question, with one alignment example: which documents made the model say X? (3 min)
2. Why counterfactuals are subtle: overdetermination, non-transitivity, cost. (4 min)
3. $\beta\mapsto\pi(\beta)\mapsto\phi$; targets vs. estimators; the map table. (5 min)
4. The IF in one line, the OLS picture with leverage, and what breaks for neural nets. (7 min)
5. How we check attributions (LDS) and the lens table. (4 min)
6. How to use the day. (2 min)

An optional ~10 min L2 after lunch walks the DA2 tour.

---

## Errata ledger (source → port)

Status key:
- **C** = confirmed, by me numerically and/or by Codex with a cited derivation;
- **E** = editorial decision.

| # | Where | Problem | Fix | Status |
|---|---|---|---|---|
| E1 | `subsec:convention` vs. `subsec:if-formula` onward | Mean vs. summed loss switch; SGD minibatch normalization; GN/posterior/regularization units | DA1 §2 conventions, applied everywhere | C |
| E2 | `ex:unroll-to-if` (PDF Ex 4.2) | (b): sign should be $+$ for downweighting; missing $1/n$; drops inherited $r_{T-K}$; deletion is $\varepsilon=1$, not $1/n$. Also, SOURCE's score sign is opposite to the upweighting convention of `eq:unroll-influence` | DA2 §2 Walkthrough C | C (Codex; E2 sign also mine) |
| E3 | `ex:bif-expansion` + solution | (2,2) isn't the whole next order; needs ≈ (surrogate vs. posterior); solution part (f) missing from the statement; false claims about linear $\phi$ and singularity | DA2 §3 Walkthrough B | C (Codex) |
| E4 | `ex:if-degenerate` solution, last paragraph | Claims the damped IF diverges as $\lambda\to0$ for $\phi=a^2$. In fact $\mathcal I_\lambda=2a^*b^*(z_i-\bar z)/(n\lVert v\rVert^2+\lambda)$: finite, saturating, minimizer-dependent. (v1 of this spec wrongly said "constant in λ"; it plateaus because $\lambda\ll n\lVert v\rVert^2$.) | Rewrite the punchline; assume $\bar z\neq0$ | C (numeric) |
| E5 | `ex:dln-influence` and `rem:dln-general` | Singular-vector derivative $A_{jk}=E_{jk}/(s_k-s_j)$ holds only for symmetric $E$ (checked: $S=\mathrm{diag}(3,1)$, $E=e_1e_2^\top$ gives $A_{12}=-1/8$, not $-1/2$); reweighting also changes $\Sigma_{xx}$; time-localization and sign-flip arguments incomplete; "$+$" typo in (c). Lee et al. App. C.6 shares the error | Re-derive (DA2 §4) | C (numeric) |
| E6 | `subsec:if-nn` (GN paragraph) | "$G$ and $H$ differ by a term that vanishes at a critical point" is false. Counterexample: $f=w^2$, target $-1$, $L=\tfrac12(w^2+1)^2$; at $w=0$, $H=2$, $G=0$ | DA1 §4 item 2 | C (numeric) |
| E7 | `ex:bif-ot` + solution | Partition term claimed to vanish; downweight/upweight sign; Gaussian KL omits the covariance change; false uniqueness among vector fields; singular case needs a range condition | DA2 §3 Walkthrough C | C (Codex) |
| E8 | `ex:unroll-derive` (c) | $J=I-\eta PH$ ignores $DP$; Adam has state | Label as frozen-preconditioner approximation | C (Codex) |
| E9 | `ex:unroll-to-if` (c), closing statement | Stochastic convergence stated without the assumptions that exclude the drift just derived | Theorem box with Thm 2's assumptions | C (Codex) |
| E10 | `rem:tracin` | "Simply drops the Jacobians" misdescribes TracIn (checkpoint query gradients, loss-change convention, TracInCP subsampling) | DA1 §5, DA2 §2 | C (Codex) |
| E11 | `subsec:implicit-jvp`, last paragraph | Overstates MAGIC exactness for finite deletions | DA2 §2 reading note | C (Codex) |
| E12 | `ex:if-derive` solution; `ex:ols-if` | Global uniqueness isn't what the IFT needs; OLS deletion needs $h_j<1$; Mahalanobis uses the uncentered moment | DA1 §3 | C (Codex) |
| E13 | TODO figures (Hasse lattice, proximity gap) | Not drawn | Dropped; DA1 §4 code replaces the latter, with a different, honest caption | E |

The earlier survey's "sign errors in Ex 4.2, 4.3" refer to E2 and E5 under PDF numbering.

Writers must re-verify each fix (sympy or numerics) and log how in `reviews/da/<stem>.md`.

## Build process (mirrors TD)

1. **Writers.** One writer agent per notebook, in parallel, briefed with:
   - `td/WRITER_BRIEF.md`, adapted for DA;
   - this spec;
   - the `.tex` as primary source;
   - the errata ledger;
   - the walkthrough device.

   The L1 outline is written after DA1.
2. **Critique.** Two rounds per notebook: a Claude design critic (judging CRW's pedagogy and the one-path requirement) and a Codex technical critic. Logs go in `reviews/da/`.
3. **Integration pass.** Cross-links DA1 ↔ DA2 ↔ TD; conventions consistency; `da/README.md`; the filter figure last.

## Open questions for CRW

1. **Louis.** Is he around? Does he want to see the port, especially the errata (several are substantive: E5–E7) and the new DA1 §5 / DA2 §5–6?
2. **Time budget.** Is DA1 required by lunch, with extensions optional, OK? Or would you rather DA1 run into the early afternoon and the DA2 tour be shorter?
3. **Proposed papers.** Datamodels, TRAK and Basu et al. are the ones I'd most want, since evaluation is otherwise missing. Also proposed: Data Shapley, LoGra, LESS, Scieur, Schioppa, Hammoudeh & Lowd. Veto or swap any?
4. **The DLN walkthrough (DA2 §4).** It needs a real re-derivation. Keep it (writer derives and verifies), or cut it to a reading plus the numerical sign-flip demo?
5. **The new EK-FAC walkthrough (DA2 §1).** Keep it, or leave EK-FAC as a reading only?
