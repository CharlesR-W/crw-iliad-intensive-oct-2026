# DA2 (`da2_methods`) review log

Builder `da/build_da2_methods.py`; page `da/da2_methods.html`; runnable `da/da2_methods.ipynb`; checks `da/checks/da2_methods_checks.py`.

## What was verified, and how

### Formulas (all in `da/checks/da2_methods_checks.py`, 0 failures)

- **Stop 1, EK-FAC.**
  - Verified numerically:
    - `vec(g aᵀ) = g⊗a` under row-major flattening;
    - the damped Kronecker inverse via two eigendecompositions, with damping on the joint eigenvalues, against a direct solve;
    - factor-wise damping is a different operator (relative difference 0.56);
    - the EK-FAC Λ equals diag(QᵀGQ) and is Frobenius-optimal against random perturbations;
    - ‖G−EKFAC‖ ≤ ‖G−KFAC‖.
  - Checked by Monte Carlo: under independence, the EK-FAC eigenvalues equal the K-FAC eigenvalues.
  - In the code cell, the exact layer block equals E_x[E_ŷ(ggᵀ)⊗aaᵀ] to 1e-16 (prototype check of the Kronecker ordering).
- **Stop 2, unrolling (E2, E8, E9).** All checked against finite differences of actual GD retraining:
  - Downweighting response from w*: `(1/n)[I−(I−ηH̄)^K]H̄⁻¹g_k`, with sign + and factor 1/n.
  - The window formula with the inherited term (error 1.8e-8). In the code cell it is exact to 8e-15; dropping the inherited term gives an error of 0.17 against |r_T| = 0.33.
  - The K→∞ limit equals H_sum⁻¹g_k.
- **Stop 2, other checks.**
  - Null-space drift `P0 r_K = Kη P0 g`.
  - SOURCE filter values: 0.632 against 0.5 at the crossover, F(0)=ηK, and the frozen discrete filter tends to the exponential.
  - Preconditioned Jacobian `I − η(PH + DP[·]g)` against finite differences; the frozen version differs when g≠0 (E8).
- **Stop 3, BIF (E3, E7).**
  - BIF = −Cov, by 1-D quadrature with finite differences in β.
  - BLR: BIF = (A+σ²/τ²)⁻¹x r = −(H_lik+τ⁻²)⁻¹∇L_j, and it equals the derivative of the conjugate posterior mean.
  - A quadratic φ in BLR picks up the (2,2) term (Monte Carlo).
  - Sympy: the (1,1), (1,3), (2,2) and (3,1) terms, and the vanishing of odd terms.
  - True posterior against surrogate: for linear φ and quadratic L_i, every surrogate next-order term vanishes, but quadrature gives n²×residual → −2.37 (n = 200…1600, stable).
  - L = n w⁴: posterior sd ∝ n^(−1/4).
- **Stop 3, transport.**
  - The completed square needs the partition term; without it the "cost" goes negative.
  - In 1-D at κ = 4000, the normalized KL of a constant shift is minimized at H⁻¹g_k.
  - BLR: downweighting shifts the mean by −BIF.
  - The minimal constant-shift KL is (ε²/4)tr(M²) (checked to 1%).
- **Stop 4, DLN (E5).**
  - SVD derivative A_jk, C_jk against finite-difference SVDs, for diag(3,1) with E = e1e2ᵀ (A12 = −1/8, |C12| = 3/8) and for a random 3×3 matrix.
  - The speed channel with constant forcing equals ∂a/∂s, and ∂a/∂s → 1.
  - Realizable reweighting: dW_kk = (x_pk²/n) t ȧ_k (relative error 1.1%).
  - The plateau rotation formula against finite differences (0.2676/0.0680 against 0.2667/0.0667).
  - The plateau influence against +s1²s2²/(n(s1²−s2²)) (0.2610 against 0.2667).
  - The endpoint influence is zero.
  - In the code cell, finite differences match the integrated linear-response equations to 2e-9.

### Paper claims

Three subagents WebFetched the papers (PDF and HTML) and quoted sections and equations.

- **Grosse et al.:**
  - Sections: §2.1.1 PBRF; §2.2.2–2.2.3 K-FAC/EK-FAC; Eq. 21 joint damping; §3.1, §3.2.1 TF-IDF, §3.2.2 query batching (rank 32).
  - Findings: §5.2.1 sparsity (power law, and "spread over many sequences"); §5.3.1 scale, with Figs. 1 and 15–16 cross-lingual; §5.3.5 role-play.
  - The model samples the labels; only MLP layers are used. No sycophancy claim anywhere.
- **George et al.:** §3.2 and App. A.1, Frobenius optimality.
- **Martens & Grosse:** §3, §3.1, §5 (model-sampled targets).
- **SOURCE:**
  - §3.2 segments and stationarity, Eq. 21 F_r, Fig. 3; §3.3 chaining; L = 3.
  - The score is the removal effect (opposite sign to the upweighting derivative).
- **MAGIC:** §4 and Fig. 4. LDS 0.96 at 1% drop, degrading toward 20%; the seed and data order are fixed (§2.1).
- **TracIn:** §3, with the first-order and minibatch (1/b) forms and TracInCP.
- **Kreer et al.:** §2.2–2.3, §3.1, §4.2 (LDS ≈ EK-FAC), App. A (with ≈).
- **Mlodozeniec et al.:**
  - Thm 2 (§4.1), assumptions A1–A6. A3 bounds (θ_t, r_t); A4 is the range condition; A6 is the "most technically meaningful".
  - Thm 3 (§4.2): bounded derivatives up to third order, eigengap on the minimum set, ℓ_k constant on the set, integrability, minimum attained.
  - The response is normalized as n × ours.
- **Lee et al.:**
  - BIF over checkpoints.
  - §3 with Figs. 3–4: 8-item hierarchy, where dog helps sparrow early and harms it later.
  - §4 with Fig. 5: Pythia-14M, where the delimiter influence flips sign. Their language-model finding is a sign flip, not "spike then decay"; the delimiter phrasing in the source notes was not used.
  - App. C.6 perturbs only the cross-covariance. Its Eq. 48 is correct; Eq. 49 has a sign slip (agent's finite-difference check).
- **TRAK, Datamodels, LESS, LoGra, Baker, Scieur, Schioppa, Engstrom, Hammoudeh & Lowd:** titles, authors and one-line summaries confirmed.

### URLs

All 20 arXiv and Iliad URLs return HTTP 200 (curl, 2026-10-09).

## Critique rounds

(see below)

### Round 1 (Claude design critic, Opus; Codex technical critic, effort high)

**Codex found, and I changed:**
1. **The Theorem 2 box overstated the result.**
   - Now says convergence to a *set*, and that the null-space part need not converge.
   - Restored the A6 integral bound on a compact neighbourhood, and i.i.d. batches.
   - Made the counterexample explicit: η_t in Step C3 makes the drift diverge.
2. **Stop 3B's order counting needed uniform curvature.** Codex's counterexample is H = diag(n−1, 1). I added the assumptions: dimension fixed, cnI ⪯ H ⪯ CnI, and higher derivatives O(n).
3. **The noisy-DLN endpoint remark was false.** Codex's finite differences show the off-diagonal response contributes. The scope box now says both channels contribute.
4. **The frozen-filter curve in stop 2 also froze the query gradient.**
   - Fix: it now uses ∇φ(w_K), so the remaining gap is only the training gradient, as the text says.
   - Verified: Codex's K = 10 numbers (0.00253 exact, 0.00177 frozen with the current query gradient) agree with the new curve.
5. **Transport admissibility (stop 3C).** Added that T_ε must be a diffeomorphism for small ε (bounded ∇r suffices).
6. **"L_k constant on S plays the role of the range condition" was wrong.** Counterexample: L = w⁴, L_k = w, with inf = −3(ε/4)^{4/3}, which I checked by hand. Step C6 is now a theorem box with bulleted assumptions, and it says what constancy does and doesn't exclude.
7. **The "one lens" paragraph overclaimed.** It is now restricted to local or frozen approximations, and names the exceptions: the exact BIF, general unrolling and datamodels.
8. **Stop 1's look-for overclaimed.** "Larger damping raises every correlation" became "raises the Pearson correlations"; the cached Spearman correlation for block-diagonal is non-monotone.
9. **Two overstatements in stop 2A.**
   - "DP vanishes only where ĝ = 0" is softened.
   - The 2A scope box is restricted to SGD, with the full-state chain rule for other optimizers.
10. **The 1-D posterior example (3B) is now exact.**
    - Codex's coefficient is −2.388, and its cumulant formulas Var = 1/n + (36a²−12b)/n² and Cov(w, w²) = −12a/n² reproduce it.
    - My quadrature gives n² × residual = −2.375 at n = 1600, converging; Codex's quadrature gives −2.388 at n = 10⁴–10⁵.
    - The text now gives the general formula and says "next-order surrogate terms".

**The design critic found, and I changed:**
- **Capsules now state what the code found:**
  - stop 1: the Kronecker step costs most, and EK-FAC's better fit doesn't help;
  - stop 2: deletion is 30% beyond first order;
  - stop 3: the BIF is 2× the damped IF at small γ.

  I also added a "What the run shows" paragraph after the SGLD sweep.
- **The critic's SGLD observation was a real bug, and fixing it changed the method.**
  - The observation: the h and h/2 estimates sat on opposite sides of the quadrature at every γ.
  - Diagnosis: the 2000-step floor gave too short a burn-in in physical time as h shrank. With h, h/2 and h/4 at fixed steps, the estimates drifted monotonically away from the quadrature (γ = 10: −0.0305, −0.0298, −0.0292 against quadrature −0.0304).
  - Fix: run length is now 128/(hγ) steps (64 valley relaxation times, a quarter discarded as burn-in). After the fix, h and h/2 agree with the quadrature within about 2 standard errors.
- **Walkthroughs are numbered per stop** (1, 2A–2C, 3A–3C, 4), and every in-page reference to them is hyperlinked.
- **Stop 2 now lists Mlodozeniec et al.** (Thm 2) under "Also". Walkthrough 2C "covers the deterministic mechanism" instead of claiming to substitute for the proof.
- **Capsules trimmed.**
  - Stop 2: the TracInCP clause is gone, and s is defined.
  - Stop 3: the transport sentence and "overlaps the SLT day by design" are gone.
  - "Question." labels are now actual questions.
  - Conventions say "data weights".
  - The tour list gives each stop's π(β).
- **Notation gaps fixed:**
  - stop 1, Step 3: which λ is meant;
  - 2B: the update is restated, and there is a summed-units bridge to DA1 §6's SOURCE row;
  - 3C, Step C5: the BLR set-up is restated inline;
  - stop 4: E is reused, so the SVD perturbation is renamed B; t units are made consistent; F is explicitly replaced in Step 3.1 and restated in Step 5.
- **Writer-instruction leaks removed:**
  - "measured, not built in";
  - "we should label it as one";
  - the "Kreer states ≈" sentence;
  - the L = nw⁴ rebuttal;
  - the duplicate SOURCE sign note;
  - the duplicate MAGIC "exact ≠ finite" note.
- **Inconsistencies fixed:**
  - 3A Step A3 now says "three things";
  - the counterexample in the theorem box;
  - the stop 6 opening;
  - the stop 5 "vice versa".
- **Solutions now show their algebra:** the 2×2 solve in 4.5.3 and the substituted line in 4.6.2. The 2B answer wording is clearer.
- **Readings:** the Grosse blurb is split into method and findings lines, and the walkthrough's section list now matches it.

**Not changed:**
- Table-of-contents entries for `###` subsections need a kit change, which is reported to the instructor.
- No datamodel code cell in stop 5: the spec says readings only.
- DA1 links stay page-only: DA1 is being written concurrently, and the brief says so.

### Round 2

**The design critic (fresh Opus) said the notebook is close to done.** Its items, and what I changed:
- **Damping units in Walkthrough 1, Step 3.** My round-1 parenthesis had them inverted. Fixed with λ̄ on Ḡ, so λ = nλ̄ on G.
- **Symbol clash in 3A/3C** (r for both the transport field and residuals). The residuals are renamed e_j.
- **π for datamodels was misstated.** It now reads "retraining from scratch on a subset", and the datamodel is a fit to φ∘π.
- **Stop 1 capsule numbers.** Moved to Code 1's look-for, with rank and Pearson correlations stated separately.
- **Theorem and scope boxes** are merged into one box per walkthrough, with a one-line list of assumptions.
- **3B's "true posterior" paragraph** is cut to the formula and its meaning. The heading is now "When the surrogate fails".
- **Code 3 write-up.** It now acknowledges the small step-size bias at h (0.5–3% high; h/2 is closer in 5 of 6 rows).
- **Walkthrough 4 Step 5 is split** into Step 5 (truncated-SVD derivative) and Step 6 (plateau fixed point). The sign flip is now Step 7. Removed the errata-echo sentence about the symmetric-only formula.
- **Duplicate Mlodozeniec reading** (with different ratings) dropped from Stop 2. The theorem box links to Stop 3.
- **Stop 6.** Pitch line added; the distributional item rewritten.
- **Small items.** The unused D symbol in Walkthrough 1, the DP sentence, and the t-units sentence.

**Codex (reviewing a state before some of the design fixes):**
- **The round-1 repairs check out:**
  - Theorem 2 box;
  - the uniform-curvature assumption in 3B;
  - the cumulant coefficient −2.388;
  - the noisy-DLN endpoint;
  - the frozen-filter curve;
  - transport admissibility;
  - the "one lens" scope;
  - Pearson-only wording;
  - DP;
  - the full-state chain rule.
- **Every quoted number matches the cache.** It recomputed: BIF/IF ratios 2.04 and 1.80; surrogate 35.4×; deletion 29.3% beyond first order.
- **New: the Theorem 3 box overstated the degenerate case.**
  - Codex gave a counterexample with bounded derivatives, L = √(1+x²) − 1 + y⁴/(1+y²)^{3/2} and L_k = tanh y. There, r = (0,1) beats r = H⁺∇L_k = 0 by about ε².
  - I verified it analytically and in the checks file.
  - Codex says the paper's own Theorem 3 statement has this problem. I did not verify the paper's full statement myself, so the notebook no longer paraphrases the theorem's conclusion. It says the degenerate case needs g_k ∈ Range(H), holds only at order ε², and that the optimal field is not unique.
- **Already fixed by the design round:** EK-FAC damping units, DLN time units (rechecked: the code description now says t/τ = η × steps, consistent with the walkthrough), and SGLD wording.

### Round 3 (final; required because round 2 still found substantive problems)

**The design critic (fresh Opus) said the notebook is close to done.** Its items, all applied:
- Shortened the 2C and 3C scope boxes to 2–3 lines. The "n times ours" normalization note moved into the Mlodozeniec reading.
- Defined LDS in the opening conventions, because the stop 2 and 3 capsules use it before stop 5.
- Removed three leftover errata-style sentences:
  - "IF plus (2,2) is not the next-order BIF";
  - the Lee blurb's input-covariance note;
  - the "extends it" clause in Walkthrough 4.
- Restated p, ρ, the local posterior and BIF_γ in 3B's notation.
- Expanded PBRF once.
- Removed the duplicate caveat in stop 5.
- Linked the stop 4 capsule to Walkthrough 4, Code 4 and TD NB1, the TD 2.6 mention, and "Stop 3" in stop 6.
- Dropped "along flat directions it need not converge" from the stop 2 capsule; Walkthrough 2C covers it.

**Codex:**
- All round-2 items are fixed:
  - the Theorem 3 / degenerate-case statement;
  - λ̄ damping units;
  - DLN time units;
  - the SGLD write-up.
- The round-2 design edits introduced no technical errors:
  - the Steps 5–7 split;
  - the e_j rename;
  - the stop 5 statement of π;
  - the merged boxes.
- One low-severity point: the merged Theorem 2 box had weakened A6 ("integrable distance" with no uniform bound). The round-3 shortening already replaced that wording with "uniform-convergence assumptions (A1–A6)" plus a pointer to §4.1, so nothing false remains.
- Checks: 0 failures.

**Also:** removed reading venues I had written from memory and could not confirm from arXiv (TRAK, Datamodels, Hammoudeh & Lowd).

## Final state

- 73 cells, of which 6 are code cells; 29 solutions.
- `--check`: fresh. `--run`: 18–20 s in total, slowest cell 10–11 s (the SGLD γ sweep). Checks: 0 failures.

## Remaining uncertainties

- **Mlodozeniec et al., Theorem 3.**
  - Codex argues the paper's degenerate-case statement fails under the assumptions as summarized by our reading agent. I verified its counterexample (√(1+x²) − 1 + y⁴/(1+y²)^{3/2}, with L_k = tanh y) analytically.
  - I did not read the paper's full proof or assumption list myself. The notebook therefore states only what we prove, plus the range condition and non-uniqueness.
- **Grosse et al.:** I rely on the reading agent's quotes for section and figure numbers and did not re-read the 119 pages myself. The same holds for Lee et al.'s figure numbers and MAGIC's 0.96 LDS.
- **SGLD at step h** shows a small step-size bias (0.5–3%), resolved only marginally (Codex: 2.1 combined s.e. between h and h/2 at γ = 100 and 300). The text says so.
