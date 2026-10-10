# Review log: td2_3_sgd_sde (NB2 §3, SGD as a stochastic differential equation)

Builder: `td/build_td2_3_sgd_sde.py`. Checks: `td/checks/td2_3_sgd_sde_check.py`.

## Verification done before the critique rounds

**Sources read in full text** (arXiv PDFs converted with `pdftotext`, then the relevant sections read):

- Li, Tai, E 2017 (1511.06251): order-1 SME $dX=-\nabla f\,dt+(\eta\Sigma)^{1/2}dW$ and order-2 SME with drift $-\nabla(f+\frac\eta4|\nabla f|^2)$ (eqs. 6, 7, informal Thm 1); $\Sigma$ is the batch-size-1 covariance (eq. 4); $t=k\eta$; Definition 1 (weak order); Remark 1 (minibatch: $\eta\to\eta/M$ in the noise). Confirmed.
- Li, Tai, E JMLR 2019 (1811.01558): Theorem 3 (moment matching to $O(\eta^{\alpha+1})$ for orders $1..\alpha$, $(\alpha+1)$-th moment $O(\eta^{\alpha+1})$, bounded moments, gives weak order $\alpha$); Lemma 5 (SGD one-step moments); Corollary 10; Remark 12 (constants depend on $T$, not estimated); i.i.d. index per step. Confirmed; fact 5 in the notebook paraphrases Theorem 3.
- Li, Malladi, Arora 2021 (2102.12470): SVAG definition (eq. 5) and Theorem 4.3 (order-1 weak in $1/l$); Definition 5.1, Theorem 5.2, Definition 5.4, Theorem 5.5, Theorem 5.6 ($\kappa>C^2(1+N/G)$, with $N$ the minibatch noise trace at batch $B$, since $\Sigma_B=\kappa\Sigma_{\kappa B}$); $C^2=2$ in experiments; Figure 2 (PreResNet-32 BN/GN, VGG-19 GN on CIFAR-10); Figure 3 (SGD vs NGD 89.4% vs 89.3%); Figure 4 (SVAG diverges from SGD at B=1024); §2.1 (heavy-tail measurement critique; LSR can hold when SDE fails, App. B.1); setting restricted to scale-invariant nets with WD at equilibrium. Confirmed.
- McCandlish et al. 2018 (1812.06162): eqs. 2.2–2.12, footnote 6 (without replacement variance $\propto 1/B-1/D$), App. A.1 estimators (A.1–A.2) and EMA advice, App. C temperature ($B_{\rm noise}\propto B_{\rm simple}\propto 1/T$ at equilibrium; quadratic toy model), App. D (noise scale grows by ≥10× during training), Figure 4 range "from 20 (SVHN autoencoder) to over 10 million". Confirmed.
- Malladi et al. 2022 (2205.10287): Definitions 2.1–2.3 (NGOS $g=\nabla f+\sigma z$, $\sigma\sim1/\sqrt B$ with replacement), §4.1 warm-up (keeps $k\eta/\sigma$ and $k\eta^2$ fixed, so $\eta\sim1/\sigma$, $k\sim1/\eta^2$), Definition 4.1 ($\sigma_0=\sigma\eta$, $\epsilon_0=\epsilon\eta$, $c_2=(1-\beta)/\eta^2$), Theorems 4.2/4.5 (one step = time $\eta^2$), Definitions 5.1/5.2 (square-root rule), Theorem 5.3 ($C(1+\kappa)\eta^2$), App. F.2 (no SDE when $\sigma\ll\|\bar g\|$). Confirmed. (The PDF text of §4.2 reads "σ ∼ 1/η²", which conflicts with Definition 4.1 and with "When η → 0, σ scales as 1/η"; the notebook uses $\sigma_0=\sigma\eta$ fixed, i.e. $\sigma\propto1/\eta$.)
- Smith, Dherin, Barrett, De 2021 (2101.12176): eq. 22 with $(N-B)/(N-1)\cdot\eta\Gamma/4B$, $\Gamma$ the trace of the empirical per-example covariance; §4 (SDE analogy assumes random sampling; with reshuffling no noise at first order after one epoch; n-step SGD argument); "optimal LR independent of B" for very large batches. Confirmed.
- Goyal et al. 2017 (1706.02677): Linear Scaling Rule statement, gradual warmup, 8k works, 16k+ degrades. Confirmed.
- Higham 2001: DOI resolves (Crossref metadata: SIAM Review 43(3) 525–546); content description (EM, strong/weak convergence, MATLAB programs) from general knowledge, not re-read.

**Closed forms checked** (`td/checks/td2_3_sgd_sde_check.py`, all pass): order-2 SME moment matching (sympy, 2D generic polynomial loss with state-dependent noise); FP stationarity of the Lyapunov Gaussian; Boltzmann iff $D\propto I$; $D\propto H$ gives isotropic $\Pi$; Gibbs has zero current for a non-quadratic potential (sympy); trace identities; discrete stationary covariance and the $(1-\eta h/2)^{-1}$ factor; McCandlish algebra (sympy); two-batch estimator unbiasedness and the without-replacement factor (Monte Carlo); weak-order slopes on a quadratic; order-2 SME stationary correction has the wrong sign; preconditioned floor $\eta\,\mathrm{tr}(P\Sigma)/4B$ for non-commuting $P$; SVAG coefficients; equilibrium $B_{\rm simple}=2B\,\mathrm{tr}\Sigma/(\eta\,\mathrm{tr}H\Sigma)$.

**Numerical claims in the text** were read off the `--run` output: the 50% and ~10× floor enhancements at $\eta\lambda_{\max}\approx0.9$ and $1.9$; the reshuffling ratios; the B=1 gap decomposed (additive-noise discrete theory 0.003966 vs SDE 0.003942 vs exact 0.004126, so the gap is mostly the state-dependent term); the Adam floors.

## Departure from the assignment brief (flagged for the instructor)

The brief asked to "show explicitly that the stationary law is Boltzmann-like ($\propto e^{-cL}$) only when the noise covariance is proportional to $H$". In the quadratic-basin OU model this is not correct: the stationary law is $\propto e^{-cL}$ iff $\Pi\propto H^{-1}$ iff $\Sigma\propto I$, and $\Sigma\propto H$ gives an isotropic cloud $\Pi\propto I$. (This agrees with MDL WB2 8.C.i.3, and is checked numerically.) The notebook states the correct condition and shows explicitly that $\Sigma\propto H$ is *not* Boltzmann.

## Critique rounds

### Round 1

**Design and pedagogy critic** (Claude Opus subagent, fresh, given the build decisions, the topic note, the editorial brief and the notebook). Ranked findings and what I did:

1. 3.3.A.1–3 duplicate MDL WB2 8.C.i with clashing notation. *Fixed:* items kept (the section must stand alone) but flagged as a repeat with a skip-to-4 note; notation bridge added (WB2's $C$ = $\Sigma$ here, WB2's $\Sigma$ = $\Pi$ here).
2. 3.4.A.4 answer misread the Li–Malladi–Arora criterion in its own toy: with $C^2=2$ it fires only at $\kappa\eta h>4$, after divergence. *Checked and agreed;* rewritten (now 3.4.B.3) to say so plainly, and to say where the criterion is informative.
3. Fact 3 was stated as a short-time expansion but used as Dynkin's formula. *Fixed:* stated as Dynkin's formula, with the expansion derived from it.
4. Circular dependencies and a wrong "Uses" column. *Fixed:* equilibrium $B_{\rm simple}$ moved into 3.3.A.3; Uses column corrected.
5. The LSR was derived three times. *Fixed:* 3.4.A.1 deleted; §3.4 opens with a back-reference to 3.2.A.5.
6. 3.5.A.4 asked an ill-posed question. *Fixed:* rewritten.
7. 3.6.A.4 was a one-sentence prompt. *Fixed:* replaced by a computation (the family $(\mathrm{diag}\Sigma/B)^{-p}$, rule $\eta\propto B^{1-p}$), checked in the check script.
8. The validity box: self-contradictory opening, buried, missing rare-event limit, uncited heavy-tail claim, overgeneralized conclusion. *Fixed:* promoted to subsection 3.4.A with an anchor linked from the pitch; opening rewritten; rare-event/escape-time bullet added (also in §3.3); Şimşekli–Sagun–Gürbüzbalaban linked (arXiv ID checked via the arXiv API, and it is the paper LMA cite); conclusion scoped to LMA's CIFAR-10 experiments.
9. The pitch lacked rigor level and a fast path. *Fixed:* "Level" paragraph and fast path added; reshuffling mentioned.
10. The code exercise was a transcription of two formulas, and the EMA lag was misdescribed. *Fixed:* exercise now builds the estimate from per-example gradients; EMAs replaced by block means of numerator and denominator.
11. Nothing demonstrated mechanism 1 or the steps–examples hyperbola. *Fixed:* new cell 3.5.B measures $S(B)$ with per-$B$ learning-rate tuning on least squares, fits the hyperbola (in relative error), and compares $B_{\rm crit}$ with $B_{\rm noise}$ along the way.
12. Overclaims (EoS for SGD; reshuffling "upper bound"; "within a few percent" vs a printed 13%). *Fixed* all three.
13. Code cells illustrate rather than check; duplication. *Fixed:* the LR-decay cell now overlays the exact expected trajectory (iterating the 3.3.A.6 covariance map); `sgd_step` helper; `lyap` moved.
14. Exercises notebook stopped at the stub; two "core" labels. *Fixed:* demo checks for the stub and prints a note; one core reading (LMA).
15. EoS and scale invariance used without restating; "temperature" meant two things. *Fixed:* both added to "What you need"; Gibbs temperature distinguished from the noise scale $\eta/B$.
16. 3.2.A.3's second-moment step skipped; 3.3.A.6 weight. *Fixed:* written out term by term; 3.3.A.6 marked optional.

**Technical critic** (Codex, `codex-companion task --effort high`, read-only, given the builder, solutions notebook, check script and the paper texts). It confirmed the SME coefficients, moment calculations, FP signs, Lyapunov and trace identities, discrete enhancement, wrong-sign order-2 correction, least-squares recursion and its Kronecker implementation, finite-population and bridge covariances, Smith's coefficient, LMA's formula and SVAG coefficients, McCandlish's algebra and estimators, and Malladi's time convention, scalings and floors. Errors found and what I did:

1. Fact 5 and the SME theorem omitted qualifications (absolute moments, uniformity, regularity). *Fixed.*
2. Malladi's Theorem 5.3 for Adam holds from a time $t_0>0$ with matched states; compared state is $u=v/\sigma^2$. *Checked against 2205.10287 l.370–435; fixed.*
3. "No SDE limit exists" was too broad. *Fixed:* "their nondegenerate stochastic limit requires noise domination".
4. "At equilibrium the noise always dominates" is false for discrete SGD: one mode has $N/G=(2-\eta h)/(\eta h)$. *Checked (sympy-free algebra, added to check script); fixed in 3.4.B.3.*
5. LR decay "drops immediately" overclaimed. *Fixed:* target drops immediately; loss follows at noise-weighted rates.
6. The plotted instantaneous $\mathrm{tr}\Sigma/\|G\|^2$ and the prediction (a ratio of means) are different quantities. *Fixed:* block means of numerator and denominator plotted; dashed line labelled as the ratio-of-means prediction.
7. SVAG is not literally an Euler–Maruyama step. *Fixed:* "matches the conditional mean and covariance".
8. Heavy tails conflated with infinite variance/Lévy limits. *Fixed.*
9. Noise need not dominate near an interpolating minimum. *Fixed.*
10. McCandlish's expansion is approximate and needs positive curvature. *Fixed.*
11. Check script solved the discrete Lyapunov equation at $\eta\lambda_{\max}=2.9$ and accepted a non-PSD "covariance". *Fixed:* stable step sizes, PSD assertion, and an assertion that the unstable case is not PSD.
12. Fact 1 said the noise "survives in a limit". *Fixed.*

Minor items (all fixed): validity-box opening; 3.2.A.0 remainder argument; martingale-difference wording; "Brownian-bridge covariance"; Boltzmann "unless $H\propto I$"; baseline for the factor-two threshold in 3.4.B.2; LMA's 20% operational threshold; numbers in prose matched to output (13%, initial $B_{\rm simple}$, the 32-fold claim restricted to the descent phase with the B=32 floor 25% higher, verified with the exact discrete theory: ratio 1.257); Adam transient explanation softened; theory curve starts at $B=1$.

### Round 2

**Technical critic** (Codex, fresh run, asked to verify round-1 fixes and do a full pass). Confirmed all round-1 fixes and the new material (discrete $N/G$, the $\kappa\eta h>4$ threshold, the exponent family, Dynkin form, `cov_map`, block averaging, the Malladi statement). Found:

1. **Substantive:** 3.5.B (now 3.4.B) reported $B_{\rm noise}$ on a straight radial ray from the start to the minimum, not along the training trajectories, yet the prose drew conclusions about where the runs spend their steps. *Checked: correct* (with anisotropic $H$, equal loss does not fix the noise scale). *Fixed:* the cell now measures the median $B_{\rm noise}$ over chains along the tuned runs at $B=8$ and $B=64$, and the prose reports those numbers.
2. Gibbs "for any $L$" needs $\int e^{-L/T}<\infty$. *Fixed.*
3. Minor (all fixed): opening still said "in the limit" and "drops the loss at once"; "falls on the hyperbola" softened (≈10% residuals, and a two-parameter fit is weak evidence); absolute third moments added to 3.2.A.1 and 3.2.A.3; 3.2.B no longer calls the dynamics Gaussian, and uses $\eta=2/n$ so the horizon is hit exactly; `cov_map` qualified as exact for mean-zero laws; SVAG agreement described as evidence about measured observables; $\Sigma_{jj}>0$ assumed where negative powers are taken.

**Design and pedagogy critic** (fresh Claude Opus). No violations of the binding decisions. Ranked findings and what I did:

1. The instructor's question ("is the Taylor approximation even valid?") was buried in §3.4 and scattered across six places. *Fixed:* new final section §3.7 "When is the SDE picture valid?", ordered as suggested (what the theorems say; $\eta h$ not small, EoS and batch sharpness; long times and rare events; Smith et al.'s finite-$\eta$ counterpoint; reshuffling; heavy tails; smoothness; SVAG as a test). The 3.3 caveat list and the old box were replaced by pointers.
2. Rigor creep in the default text. *Fixed:* weak order, moment matching, the SME hypotheses, LMA's Theorem 5.6 and Malladi's Theorem 5.3 are stated in one physical sentence each, with the exact hypotheses in `<details>` "Precise statement" blocks.
3. Paper narration (LMA's experiments, Malladi's fine print). *Fixed:* moved into details blocks or reading blurbs.
4. §3.2 opened with five dense facts. *Fixed:* two facts (Wiener, Euler–Maruyama) open §3.2; the SDE matching comes next; weak order and moment matching are introduced where they are needed; the generator/Dynkin formula moved to the start of 3.2.A; "(no Itô calculus needed)" dropped; Higham linked from "What you need".
5. §3.4/§3.5 were in the wrong order. *Fixed:* critical batch size is now §3.4, the linear scaling rule §3.5; all references point backward.
6. Pitch overclaims and an inconsistent fast path. *Fixed:* "to first order in η"; "what a learning-rate cut does to the floor"; "together with a one-step Taylor model"; Level paragraph no longer claims every check is exact; fast path includes the §3.3 loss-floor bullets.
7. Opening too long. *Partly fixed:* shortened, EoS and scale invariance one line each (they must stay for standalone use). It is still a substantial cell, because of the contents table and notation.
8. Repeated results. *Fixed:* the old 3.4.B.2 merged into 3.3.A.4; the OU half of the $N/G$ exercise reduced to a back-reference; 3.4.A.4 keeps only the past-$B_{\rm noise}$ question; the 3.3.B sweep is labelled as the LSR in a basin.
9. Missing link to td2_2 §2.5 (edge of stochastic stability). *Fixed:* linked in "What you need", in the 3.3.A.6 answer and in §3.7. I checked numerically that the state-dependent term does lower the stability threshold: spectral radius of the exact covariance map exceeds 1 at $\eta\lambda_{\max}=1.5$ for $B=1$ and at $1.9$ for $B=32$.
10. The core check rested on an exercise marked optional; B=100 row redundant; hyperbola claim too strong. *Fixed* all three.
11. Readings: trimmed Li–Tai–E, dropped the duplicate WB2 cross-link, rephrased the LMA blurb. *Fixed.*
12. Symbol clashes. *Fixed:* horizon $t_{\max}$, Euler–Maruyama step $\Delta t$, LMA's $\mathcal N,\mathcal G$, $Z_n$ for the accumulated noise; $S$ in §3.4 flagged as steps, not sharpness. The curvature $h_j$ keeps its name.

### Round 3

**Technical critic** (Codex, fresh run on the restructured notebook). Confirmed the round-1 and round-2 fixes and found no new errors in the derivations. Errors in summaries and claim boundaries, all fixed:

1. §3.6 said $\epsilon$ shrinks like $\eta$ in Malladi's limit; it *grows* like $1/\eta$ ($\epsilon_0=\epsilon\eta$ fixed, Definition 4.1). The finite-batch rule $\epsilon\to\epsilon/\sqrt\kappa$ was already right.
2. §3.1 equated $B\ll B_{\rm simple}$ with the regime of near-linear step savings, which needs $B\ll B_{\rm noise}$ (counterexample with $H$ and $\Sigma$ in different directions). Now qualified.
3. §3.5 dropped the equilibrium expectations in LMA's criterion; now stated as a ratio of means, consistent with 3.4.C.
4. §3.7's escape-rate statement used the isotropic Arrhenius form for general SGD noise; now restricted to small-noise isotropic Langevin, with "quasipotential" for the general case, and "can change" instead of "changes".
5. The cross-links to td2_2's stochastic-stability section pointed to §2.5 / `td22-5`, which is now "Adaptive optimizers" after that notebook was reorganized; now `td22-6`, linked by title.
6. McCandlish's Dota point is the batch size the team used, reported as a lower bound; now said so, and the readings blurb no longer quotes the range.
Minor (all fixed): 3.5.A.2 labelled a formal comparison outside LMA's scale-invariant setting; "SGD generally has no Gibbs guarantee"; "six- to elevenfold"; stale "Linear-Gaussian" code comment.

**Design and pedagogy critic** (fresh Claude Opus). No violations of the binding decisions; numbers in the prose match the output except two (fixed). Substantive findings and what I did:

1. The text attributed the validity question to "the instructor" and "the topic note" (three places), which breaks the no-attribution rule and cites a note students don't have. *Fixed:* the question is now asked directly.
2. §3.7 had no exercise or code, and SVAG (the core paper's method) was never run; the LSR trajectory cell largely repeated the 3.3.B sweep. *Fixed:* that cell is replaced by 3.5.B, which runs SVAG ($l=1,2,4,8$) on the least-squares problem at $\eta\lambda_{\max}\approx0.3$ and $1.8$ and compares with the SDE floor (with the state-dependent noise kept; the SVAG floor ratios are 1.11→1.02 and 6.2→1.06). New exercise 3.5.A.3 derives the one-mode SVAG floor $(\eta\Sigma/4B)(1-\eta h/2l)^{-1}$ (checked in the check script; matches the critic's hand algebra).
3. §3.3's default text used Dynkin's formula, introduced only in an exercise preamble. *Fixed:* restated at the start of §3.3.
4. Bundled exercise items and missing hints. *Fixed:* 3.3.A.4 split into (a)–(e) with matching labelled answers; hints added for 3.2.A.5, 3.3.A.4(e), 3.3.A.5, 3.4.A.3–4, 3.6.A.2.
5. 3.1.A.2's last question needed §3.3's relaxation time. *Fixed:* given in the hint.
6. Repetition. *Partly fixed:* the WB2 overlap is now described once (in the 3.3.A preamble, corrected to "items 1–3 and the first equation of item 4", with the notation bridge); 3.6.A.3's overlap with WB2 8.C.i.6 flagged; the "20 to millions" range kept once in the text. The temperature dependence of $B_{\rm simple}$ is still mentioned in several places, but now as back-references to 3.3.A.3.
7. Opening heaviness and broken fast path. *Fixed* the fast path. I kept the EoS and scale-invariance lines in "What you need", because TD_BUILD_DECISIONS.md requires each NB2 file to restate what it borrows there.
8. 3.2.A.3 is heavy algebra. *Fixed:* the hint now gives the physicist's route first.
Polish (fixed): symbol clashes ($\varepsilon_j\to\mathcal E_j$, $\kappa\to a$ in 3.3.A.2, $C(t)\to M(t)$, $\mathcal D\to\Delta_{\rm tot}$; McCandlish's $\epsilon$ kept as the paper's notation); §3.7 no longer claims an ordering and gains an item on the limits of the one-step Taylor model; cross-links to `td2_1_flows` §1.3 and §1.4 (Smith et al.) and to td2_2's WSD section; the LR-decay cell now starts in the stationary law, so the dashed exact curve is flat until the cut; "≈0.9" and "six- to elevenfold" made consistent; "appropriately averaged" now introduced; momentum remark in 3.6.A.3 given its condition ($\eta_{\rm eff}h\ll1-\beta_1$) and the demo's violation acknowledged; jargon removed; LMA reading shortened.

## Remaining uncertainties

- **Higham (2001):** link verified (DOI resolves; Crossref metadata matches). The description of its contents is from general knowledge; SIAM's site blocked a full-text check.
- **td2_1 / td2_2 anchors:** `td21-s3`, `td21-s4`, `td22-6`, `td22-8` were checked against those builders as they stood at the end of this session. Other sessions are still editing them, so the anchors could move. The builder's link check verifies the files exist, not the anchors.
- **Malladi et al. §4.2 text** reads "σ ∼ 1/η²" in the PDF extraction, which conflicts with their Definition 4.1 (σ₀ = ση fixed) and their own "σ scales as 1/η"; the notebook follows Definition 4.1. Probably an extraction or typesetting artifact, not checked further.
- **Adam demo floors:** the measured floors agree with the leading-order prediction $\eta\sum_j\sqrt{\Sigma_{jj}}/4\sqrt B$ to within about 15%. At the largest linear-rule step size the momentum condition is violated, and I did not derive the momentum correction.
- **3.4.B steps–examples fit:** the hyperbola fit has residuals around 10%, and the fitted $B_{\rm crit}$ depends on the seed (52–85 across the runs I saw). The prose claims only that it falls inside the measured range of $B_{\rm noise}$ along the run, which held in every run.
- **Stability lowered by state-dependent noise:** checked only for this least-squares problem (the spectral radius of the exact covariance map exceeds 1 at $\eta\lambda_{\max}=1.5$ for $B=1$ and at 1.9 for $B=32$). I did not verify the general claim.

## Readability pass (2026-10-09)

Following `td/READABILITY_BRIEF.md`. Mathematics, code and claims unchanged; code cells untouched (outputs served from the cache).

**Subsection openings and local notation.**
- Opening pitch: $\Sigma$, $B$, $\eta$ now named in words at first use, before the Notation paragraph.
- §3.1: opens with its question (mean and covariance of minibatch noise; why only $\eta/B$ matters) and a "Notation here" line ($N$, $\ell_i$, $L$, $G$, $\Sigma$, $g_B$, the SGD step). "Hessian" named where $H$ first appears.
- §3.2: question line (which SDE, in what sense close) and "Notation here" ($L,G,H,\Sigma,\eta,B$, the step's noise, SDE time). The puzzle paragraph restates $B_{\rm simple}=\mathrm{tr}\,\Sigma/\|G\|^2$. 3.2.A states its goal and defines $D=\sigma\sigma^\top$ with the generator.
- 3.2.B: question line, and "weak error" defined in place.
- §3.3: question line (shape of the stationary cloud, Boltzmann or not, loss floor) and "Notation here" restating the order-1 SME; the Fokker–Planck paragraph now says what $b$ and $D$ are for the SME. "Flat minima" paragraph names $T$, $d$, $L_a$, $H_a$. 3.3.A states the OU setting ($H$, $\Sigma$, $D$, $\Pi$) before the items.
- 3.3.B: question line, $\eta\lambda_{\max}$ explained; reshuffling cell defines $h_{\min}$ and the relaxation time.
- §3.4: question extended (past what batch size do examples stop helping) and "Notation here" ($G,H,\Sigma,g_B,\epsilon$). "Measuring it" defines $\bar h$ where it uses it.
- 3.4.B, 3.4.C, 3.5.B: question line each; 3.4.B and 3.5.B restate what the 3.3.B least-squares problem is.
- §3.5: question line and "Notation here", restating the order-1 SME and $B_{\rm noise}$.
- §3.6: question line and "Notation here" (coordinatewise operations). 3.6.A.3 defines $H$ and $\Pi$. The Adam demo intro opens with its question.
- §3.7: the bare "Is the Taylor expansion even valid?" now says which expansions (moment matching in §3.2, the one-step model in §3.4) and restates $h$, $\lambda_{\max}$.

**Cuts.** The momentum caveat appeared verbatim in both the 3.6.A.3 solution and the Adam demo intro; the solution now points to the demo note.

**Old two-file wording.** The opening's "Worked solutions are in the solutions version of this notebook (`td2_3_sgd_sde_solutions.ipynb`)" replaced by "Each exercise is followed by a folded worked solution". All sibling links rewritten from `tdX_solutions.ipynb` to `tdX.html` (same anchors).

**MDL links.** WB2 was restructured to four sections. The SGD-stationary-covariance-vs-posterior material that the old `#wb2-section-8` / "8.C.i" references pointed to now lives in WB2 §4, part 4.B ("Does curvature cancel? Gradient noise, curvature, and SGD"), exercise 4.B.3 and the 2-D example 4.B.5 with the same `sgd_stationary_vs_posterior.html` figure. Repointed:
- §3.3 posterior paragraph: `#wb2-section-8` → `#wb2-section-4`, named as part 4.B.
- 3.3.A preamble: "WB2 8.C.i.1–4" → "WB2 exercise 4.B.3, items 1–4" (overlap re-stated against the current items: our items 1, 2, the first identity of 3, and 4(a)).
- 3.6.A.3: "WB2 8.C.i.6" → "WB2 exercise 4.B.3, item 6" (preconditioned Lyapunov equation; checked it is still item 6).
- The figure link `../notebooks/assets/interactive/sgd_stationary_vs_posterior.html` still resolves; unchanged.
Section-level anchor used per the brief, with the part named in words, since the MDL day may keep changing.

**Readability critic** (fresh Claude Opus, reading three random subsections as a jump-in grad student). Findings and what I did:
1. 3.5.A had no lead-in; SVAG($l$), $\mathcal N$, $\mathcal G$, $\kappa$, $C$ undefined locally; scalar $\Sigma$ in items 2–3 unstated. *Fixed:* lead-in sentence, "Notation here" line (SVAG weights and step, $\mathcal N,\mathcal G$, the criterion), scalar $\Sigma$ stated; "3.5.B (next subsection)".
2. 3.6.A untitled and without notation. *Fixed:* retitled "why a square root, and the floor", lead-in, notation line ($v,\beta,\beta_1,\epsilon,\kappa$ and the full square-root rule); WB2 reference now linked; $\epsilon$ added to §3.6's notation line.
3. 3.3.B: $\Pi$, "SDE", "exact discrete" and relaxation time undefined locally; "temperature picture" vague. *Fixed:* notation paragraph; "(η and B scaled together, η/B fixed; §3.5)"; state-dependent noise glossed; "the SDE's prediction".
4. 3.5.B: $A_i$, $\Sigma^*$ undefined; duplicated "l=1 is plain SGD". *Fixed.*
5. §3.4 mixes $\epsilon$ and $\eta$; 3.4.A untitled. *Fixed:* "$\epsilon$ is the $\eta$ of other sections"; retitled "the one-step model and the two-batch estimator".
6. Adam demo intro used undefined $\eta_{\rm eff}$. *Fixed:* momentum caveat rewritten in words (relaxation slow compared with the $1/(1-\beta_1)$-step window), same content.
7. §3.7: "central flows" unexplained; "quasipotential" unglossed. *Fixed:* linked td2_2 "The central flow" (`#td22-3`) with a one-clause gloss; quasipotential glossed, marked "ask your agent" (beyond scope).
8. §3.1's last paragraph had a dense $B_{\rm noise}$ clause. *Fixed:* plain sentence pointing to §3.4.
9. §3.3 Boltzmann paragraph ended with an ambiguous "It depends on θ". *Fixed:* the properties of real noise now come first, conclusion after.
10. Padding. *Fixed:* WB2 notation swap shortened to a parenthesis; 3.1.A.1's td2_1 parenthetical cut; "This notebook does not redo that comparison" cut.
11. `{{VARIANT}} · ` removed from the title line (the kit dropped it anyway).
12. Possible error: McCandlish Appendix C vs D. *Checked against the PDF:* Appendix C is "Temperature and the Noise Scale" and the main text says "$B_{\rm crit}\approx B_{\rm noise}$, where $B_{\rm noise}$ is appropriately averaged over training (see Appendix D)". Both citations are correct; no change.

Rebuilt; `--check` fresh with no MDL warnings; `checks/td_crosslinks.py`: 56 links, 0 bad. Not done: a `--run` (code unchanged, outputs from cache).

**For the instructor.** `td/TD_BUILD_DECISIONS.md` still says the SGD-vs-posterior comparison is in "MDL WB2 §8"; it is now WB2 §4.B. That file isn't mine to edit.
