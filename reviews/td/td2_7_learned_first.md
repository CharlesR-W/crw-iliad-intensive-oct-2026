# Review log: td2_7_learned_first (NB2 §7, "What is learned first")

Builder: `td/build_td2_7_learned_first.py`. Notebooks: `td/td2_7_learned_first_{exercises,solutions}.ipynb`.
Checks: `td/checks/td2_7_learned_first_check.py`.

## Draft (v0)

Content map: NB2 §7 of `TD_SPECS_v1.1.md`, with the instructor's decision applied (the
"why the moment story doesn't get you far" question is dropped; §7 is neutral exposition
with claim boundaries).

### Sources read for the draft (full text via arXiv PDF, not memory)
- Bardone & Goldt 2404.08602: eq. 1–2 (MCM), §2.4 and §3 (spherical perceptron, correlation
  loss, online SGD with step δ/d), Propositions 1–4, Remark 5, Table 1, Appendix B.1
  (two-layer experiments: minibatch 1, second-layer lr 0.01×; βm=1, βu=5, βv=10, d=128,
  m=512, ReLU).
- Ben Arous, Gheissari, Jagannath 2003.10409: Definition 1.2, eq. 1.6 (α_c), Theorems
  1.3–1.5, Proposition 2.1/Corollary 2.2 (Hermite form of the information exponent for
  single-index square loss), Proposition 2.8 (spiked tensor has exponent p). Venue
  checked on jmlr.org: JMLR 22(106), 2021.
- Refinetti, Ingrosso, Goldt 2211.11567: §1 (Fig. 1, clone protocol: train on clones,
  test on CIFAR10; ≈50/1000/2000 steps), §2 (perceptron expansion: naive classifier →
  Fisher discriminant → fourth-cumulant correction, eq. 4, 8, 12), §3, §4 discussion of
  Taylor expansion around θ0.
- Rahaman et al. 1806.08734: Experiment 1 and Appendix A.1 (6-layer, 256-wide ReLU,
  frequencies 5..50, 200 points, full-batch Adam).
- Xu et al. 1901.06523: abstract and introduction (F-Principle statement; activation
  smoothness explanation).
- Basri et al. 1906.00425: §3–4 (model, H∞, eq. 12, eq. 13, measured exponents
  2.15/1.93/1.94/2.11).
- Székely et al. 2312.14922: abstract, §4, Appendix A.1 (two-layer ReLU, SGD 200 epochs:
  multi-pass, not online).
- Damian et al. 2305.10633: abstract only (d^{k/2} with smoothing; CSQ lower bound).

### Numerical/symbolic verification (all in `td/checks/td2_7_learned_first_check.py`, passing)
- GELU Taylor coefficients (sympy).
- Exercise 7.1.B.4: exact expectations by 3-D Gauss–Hermite quadrature on the MCM spike
  coordinates; |∇L_data − ∇L_clone| scales as ε^3.97 for GELU (claim ε^4) and ε^4.98 for
  tanh (claim ε^5); |∇L| scales as ε^0.99.
- 7.1.C: E tanh²(x) Gaussian 0.3943 vs Rademacher 0.5800; the w⁴ series term.
- 7.2.A.2: closed-form f_t(x) vs direct integration of the flow.
- 7.2.B: von Mises eigenvalues e^{-κ}I_m(κ) vs FFT; small- and large-κ asymptotics;
  t8/t1 ≈ 995 at κ=4.
- 7.2.C: Basri et al. eq. (12) by quadrature (and Monte Carlo inside the notebook).
  Eq. (13) did *not* match the naive "bias drawn like a weight" kernel; the text uses that
  kernel and says eq. 13 corresponds to a zero-initialized bias with different constants.
- 7.3.A: whitening S v = v/√(1+βv); κ4 = −2βv²/(1+βv)² (MC); correlated-latent cross
  covariance c = 1.70 (MC).
- Bayes accuracies quoted in the text (0.691, 0.767, 0.819, 0.714), Monte Carlo.
- 7.3.C: escape-time simulation (in the notebook): log-log slopes 1.29 / 1.59 / 2.19 vs
  1.00 / 1.58 / 2.29 for d, d ln²d, d² ln d over d=16..64.

## Round 1 critique (design: Claude Opus subagent; technical: Codex, high effort)

### Design critic: main findings and what I did
1. "Sliding down the stairs" was stated, never derived (exponent 4 and the c11 term asserted). **Fixed:** new Exercise 7.3.B derives the single-neuron population loss on the MCM in Hermite form: L ≈ L0 − (σ2βu/4)αu² − (σ2c/2)αuαv − (σ4κ4/48)αv⁴, and the ODE consequence. Coefficients checked against an exact ReLU computation in the check script (agreement to <1%). Hermite polynomials now defined in §7.3.1. The theorem catalogue in 7.3.3–7.3.4 collapsed into two compact tables.
2. The 7.1 demo used data defined later and cited 7.1.B before 7.1.B. **Fixed:** new §7.1.2 "test bed" paragraph describes the data and Bayes levels; the code cell is split (linear check after 7.1.A, GELU comparison after 7.1.B); the section table's "Uses" column is corrected.
3. Demo prose overclaimed (said "reaches mean+cov level", "use the same information"; ‖W‖ plot didn't show what the title said; independent Bayes lines on the correlated panel). **Fixed:** prose rewritten from the actual output; right panel now plots RMS pre-activation; Bayes lines only on the independent panel; the overlap with v is now tracked over time.
4. §7.1 read a time ordering into a power count. **Fixed:** text now says the expansion orders the early gradient; the staged sequence in time is derived only for a single neuron (Refinetti §2) and observed otherwise.
5. 7.1.B.3 argument wrong (zero mean ≠ symmetry). **Fixed** (also Codex #1): parity argument on the series, plus the symmetric-class statement, with the skewed counterexample noted.
6. Framing leaned toward the dropped "limits of the moment picture" question; pitch long and repetitive. **Fixed:** pitch shortened and recast as what students will be able to compute; "What this does not cover" → short "Scope" subsections; 7.1.C renamed "What a linearized network sees: two moments of its features"; NB1 box split into short lines; numbered list removed.
7. Instructor's heading not shown; no Kadkhodaie pointer; no Physics-day NTK pointer; Damian not in readings. **Fixed:** subtitle "Moment bias, frequency bias"; links to td2_8 §3 and Physics day NB1 §3 / NB2; Damian added as optional reading.
8. Basri numbers called eigenvalues. **Fixed** (also Codex #15): called cosine coefficients, with eigenvalue = a0, a_ℓ/2.
9. Symbol clashes. **Fixed:** frequency → ℓ, width → N, whitening matrix → Q (noting their S), condition number → κ_c. Kept m for the mean spike (Bardone–Goldt notation).
10. 7.3 heuristic presented as matching the theorem's logs. **Fixed** (also Codex #9): 7.3.C part 4 now does the OU calculation for k=2 (noise is a random shift of the start, O(δ/d) variance, no log cost) and says logs are not determined by the heuristic.
11. Tie the three parts together. **Added** a short paragraph in §7.3.3.
12. Kernel demo plateau undersold. **Changed** to equispaced inputs (exact Fourier eigenvectors; aliasing noted) and described the random-input leakage floor in prose.
13. 7.2.C stub was only a comment; cell 5 runtime. **Fixed:** stub gives kernels and signatures; GELU cell trimmed (now ~6 s).
14. Optional: freeze readout for Basri setting. Not done; the prose says how to do it.

### Codex: main findings and what I did
1. 7.1.B.3 symmetry — fixed (above); also the O(ε) bias remark (bias of order ε in an odd activation gives covariance at ε³) — added.
2. GELU milestones misreported — prose rewritten from the actual run (and re-checked after the rerun).
3. Weak vs strong recovery and the step-size window in Ben Arous et al. — table now separates "enough for strong recovery" and "not enough for weak recovery", and states the window (n/d)^{-1} ≪ δ ≪ (n/d)^{-1/2}.
4. 7.1.B.4 omitted E[y x^{⊗5}] for tanh and noncancellation — added; table now says "earliest possible orders".
5. 7.1.B.1 linear surrogate overextended, missing c1 time rescaling — fixed: leading signal only when c1E[yx]≠0; scalar balanced case gives q̇ = 2c1 q(s−q) (checked by hand).
6. Rotation invariance does not imply low frequencies first (counterexample kernel) — fixed in §7.2 result and §7.2.2.
7. Null space / min-norm LS; minimax step size — fixed in §7.2.1 and 7.2.A.
8. Cumulant order alone does not set the exponent; Hermite index over degrees ≥1; negative derivative — fixed in §7.3.3 and in 7.3.B.3 (signs of σ2, σ4).
9. Log factors not shown necessary — fixed (above).
10. Post-search cost O(d/δ), not O(d) — fixed.
11. Damian et al. statement too general (CSQ is a restricted-class benchmark; setting is Gaussian single-index, known link) — fixed.
12. Bardone–Goldt δ≈ convention and windows — the table now gives the window for the correlated case, and the reading notes the convention; "sharp" → "nearly sharp (up to logarithms)".
13. Rayleigh quotients not eigenvalues in MLP panel — relabelled and explained.
14. Large-κ von Mises regime — now ℓ = O(√κ); small-κ regime stated as ℓ ≫ κ²/4 (my own check showed slow convergence of the j=0 approximation).
15. Basri reading "odd frequencies" → "odd frequencies ≥ 3" — fixed.
16. Equispaced eigenvalues include aliases — noted.
17. MCM alignment claim not measured — overlap now tracked over time and plotted; escape-time demo described as a finite-size consistency check.
18. Small items: centring stated in §7.1.1; hint sign Kα = y + r fixed; Nyquist doubling removed (only ℓ ≤ 6 doubled); extra MLP update removed; "unit-variance bumps" wording fixed; "committee" removed; Xu reading corrected (VGG16 on CIFAR-10; explanation in an idealized setting).

Codex verified as correct: linear two-moment result and minibatch caveat, the order table, GELU coefficients, the NTK/tanh example, the clone construction, kernel-flow formulas and normalizations, von Mises values, Basri bias-free kernel, MCM whitening/κ4/cross-covariance, Bayes numbers, spherical SGD sign and normalization.

## Round 2 critique (fresh Claude Opus design critic; Codex high effort)

Both critics found substantive problems again, so a third round follows.

### Design critic: main findings and what I did
1. Section order split the moment/cumulant thread (spectral bias sat in the middle; 7.1 used cumulants defined in 7.3). **Fixed:** reordered to 7.1 Two moments → 7.2 Higher cumulants → 7.3 Spectral bias (matches the instructor's heading order); cumulant definitions moved into §7.1.2; the patch paragraph removed. (I kept 7.1.C in §7.1 rather than moving it, to limit renumbering; it is about which moments a linearized model sees.)
2. Pitch lacked "why care" and overclaimed scope. **Fixed:** added a why-care sentence; outcome sentence scoped to the three settings; §7.2 described as a heuristic derivation plus cited theorems; difficulty cues in the section table.
3. §7.2 stated results three times and carried paper-level fine print. **Fixed:** main text has one short result paragraph and two small scaling tables; all conditions (weak/strong recovery defined, assumptions, step windows, Damian) moved into `<details>` folds "Precise statements, for reading the papers"; spherical gradient defined.
4. GELU prose didn't match the figure. **Fixed:** prose now describes the two-stage staircase (mean ~50 steps, covariance ~200), uses the clone argument (the clone-trained net has no access to v and matches the data-trained one), and says the later output gap is the two trajectories diverging.
5. MLP demo didn't test 7.3.C. **Replaced** with a two-layer ReLU network in Basri et al.'s setting (first layer and biases trained, readout fixed, twin units so f starts at 0), on the kernel demo's target, at output scale 1 and scale 10 (lazy). The lazy run's fit times (2, 80, 365 steps) match the infinite-width prediction 1/(η a_ℓ) (1.7, 78, 342); at scale 1 the slow modes lag (eNTK changes ~5%).
6. Core reading should include Bardone–Goldt; Damian/Székely too prominent in text. **Fixed:** core readings per section (Refinetti for 7.1, Bardone–Goldt for 7.2); Damian moved into a details fold; Székely reduced to one clause + reading.
7. 7.1.B.1 sigmoid claim exceeded the expansion. **Fixed:** only the initial growth rate 2c1s is claimed.
8. 7.2.A.3 (optimal step) off-topic. **Trimmed** to the stability bound and "spread of spectrum = spread of learning times".
9. Refinetti CIFAR numbers duplicated / §7.3.5 paragraph leaned toward weighing the moment story. **Removed** from the main text (numbers remain in the reading blurb).
10. Kadkhodaie only in the opening. **Added** a one-line cross-link in Readings.
11. 7.3.D thin on setup. **Added** a hint with the per-sample spherical gradient and a note that the search phase is short at these d.
12. Wording: demos "in §7.1 and §7.2"; 76.7% stated as the clone's Bayes accuracy; notation line dropped; random-input floor "a few per cent, shrinking with n".

### Codex: main findings and what I did
1. The quoted step window does not by itself guarantee recovery (a counterexample with n = d log³d, δ = (log d)^{-5/2}); an inherited issue in the theorem statement. **Fixed:** "for suitably tuned step sizes", with the window given as what the theorem states and a note that the proof uses further drift-time conditions.
2. Weak vs strong recovery conflated; assumptions missing; k=1 lower bound needs δ=O(1); search-phase claim only for k≥2. **Fixed** in the details fold and main text.
3. Damian d^{k/2} only for k≥3 and fixed accuracy. **Fixed** (text and reading).
4. GELU: "neither network learned v" not established. **Fixed** (clone argument; see design 4).
5. Escape-time code counted negative exits as success for odd k (2/48 runs at k=3, d=16). **Fixed:** α>1/2 for odd k, |α|>1/2 for even k.
6. 7.2.B.4 dropped the c11 α_v feedback into α̇_u. **Fixed:** the exercise now uses the coupled linear system; growing eigenvalue c20+√(c20²+c11²), eigenvector (λ+, c11); the sign condition of Bardone–Goldt's Prop. 4 appears as the projection on this mode.
7. Székely thresholds stated too sharply. **Fixed:** "sample-complexity exponents 1 and 2" from the (low-degree) likelihood-ratio analysis; network results described as experiments.
8. Smoothness does not imply ordering or super-exponential decay (counterexamples). **Fixed:** the factorial tail is attributed to the von Mises kernel; ordering claimed only for the kernels computed; a smooth counterexample kernel is mentioned.
9. Fifth-moment example confused class difference with data–clone difference. **Fixed:** E[y x_m x_v⁴] is ½(3+κ4) on data, 3/2 on the clone; difference κ4/2 (checked by hand).
10. "Needs n ≫ d log²d" for the covariance spike was a sufficient bound. **Fixed** (details fold states sufficiency and the d log d lower bound).
11. Even activation part need not be c2. **Fixed** (c4 gives E[yx⁴] = (3/2)(s+⁴ − s−⁴) at ε⁴).
12. The v direction differs "first at", not "only in", the fourth cumulant (κ6 ≠ 0). **Fixed.**
13. Interpolation iff y ∈ range K. **Fixed.**
Also: Basri ideal ℓ^{-2} ratio 64 vs actual coefficient ratios — the text now gives a1/a8 ≈ 200 for the bias kernel (computed: 201.6).

## Round 3 critique (fresh Claude Opus design critic; Codex high effort)

Mostly qualifications and presentation; no new mathematical errors in the core derivations (Codex re-verified the order table, ε⁴/ε⁵ slopes, tanh numbers, MCM constants, Hermite coefficients, kernel formulas, von Mises asymptotics, Basri coefficients). Changes:

- §7.1 contradicted its own order table ("two-layer linear network at leading order" vs the covariance term at ε² before the Σxx feedback at ε³). **Fixed:** result box, §7.1.3 and 7.1.B.1 now state the graded result; the linear network describes only the initial drive c1E[yx]. "Nonlinear terms may lead" when E[yx]=0; the O(ε) bias remark scoped to c3≠0.
- Added a short paragraph linking moment and frequency bias by polynomial degree (cos ℓθ = T_ℓ(cos θ); dot-product kernels graded by degree; bias-free ReLU's odd part is linear, which explains Basri's odd zeros via 7.1.B.3's parity argument).
- Basri demo: says outright that ℓ=8 is not fitted at scale 1; now prints the Rayleigh quotient of the eNTK on sin 8θ before/after (0.00291 → 0.00061 at scale 1, → 0.00236 at scale 10), which shows feature learning shrinking the kernel on the slowest mode; predictions now use discrete GD (first step with (1−ηa)^t < 1/e); extra final update removed; seeded locally; the unsupported "default MLP" sentence removed.
- 7.2.C escape-time check is partly circular (steps chosen from theory). **Added** prose describing a δ-sweep I ran (k=3, 32 runs/point): at d=64, median escape 7449 / 2876 / 1952 / 1841 / 4041 / 49309 steps for δ = 0.03 / 0.1 / 0.3 / 1 / 3 / 10 (44% not escaped by 60000 at δ=10); at d=16 monotone decreasing up to δ=10 (816 → 27). Not included as a code cell (too slow at d=64, δ=10).
- §7.2 exercises reordered to follow the text (A: MCM by hand; B: powers of d; C: code; D: Hermite loss); duplicated paragraph removed.
- §7.2 folds shortened to pointers; independent two-spike row now says only the lower bound is proved (Codex: learning u can suppress v, Bardone–Goldt eq. 10 discussion); post-search cost "for any fixed accuracy".
- 7.2.D.4: "projection with the common sign"; exact ratio c11/(c20+√(c20²+c11²)).
- MCM demo prose matched to the figure (stages too fast to see; plateau where data and clone coincide; final ≈0.79 < 0.819; why the mean-only curve sits below 0.691).
- GELU prose: "no accuracy advantage from v" rather than "does not use v"; the 10⁻⁴ output gap described as consistent with (not predicted by) 7.1.B.4.
- 7.3.A.3: contraction |1−ηλ|, slowest-mode statement restricted to η ≤ 1/λmax, null-space components noted.
- §7.3 intro: decreasing spectrum claimed only for von Mises and the ReLU NTK with biases.
- NB1 recap: "in NB1's aligned-mode ansatz". Refinetti attribution now says sigmoidal perceptron, square loss, two-class toy, truncated flow with stationary solutions checked numerically, within-class fourth cumulant.
- Székely reading rewritten (proved impossibility below ~d; likelihood-ratio transition at ~d with only numerical evidence of achievability; low-degree ~d² for a fourth-cumulant spike).
- Readings: core labels for Refinetti (7.1), Bardone–Goldt (7.2), Basri et al. (7.3). "What you need" lists the linear SDE and Bessel series. Pointer to the Physics day NB1 §3 for 7.3.1/7.3.A.
- Not done: the design critic suggested Belrose et al., "Neural Networks Learn Statistics of Increasing Complexity" (arXiv 2402.04362) as a reading. I did not add it (the instructor prefers papers he knows; flagged as a question). Also not done: moving the "1 + cos 8φ" and skewed-class asides into folds (they are one parenthetical each).

A fourth round was not run. Remaining risk is in prose written during round 3 (the degree-linking paragraph, the rewritten demo commentaries); I checked the claims in it myself (see below) but no critic has read them.

## Final verification
- `python td/build_td2_7_learned_first.py`, `--check`, `--run` all pass: 38 cells (8 answer, 6 code, 2 code-answer); `--run` total ≈ 32 s on a loaded machine, slowest cell ≈ 8 s. The exercises variant's code cells also run (stubs do not break later cells).
- `python td/checks/td2_7_learned_first_check.py` passes (≈ 20 s), now also covering: the 7.2.D coefficients c20, c11, c04 against an exact ReLU loss (agreement < 1%); a1/a8 = 201.6 for the bias kernel; E[y x_m x_v⁴] = (3+κ4)/2 on the data.
- Figures inspected visually after each round; prose numbers taken from the printed output of the final run.

## Uncertain / not verified
- The δ-sweep numbers in the 7.2.C commentary come from one scratch run (seeded), not from a notebook cell.
- "Learning u can push α_v back toward 0" is quoted from Bardone–Goldt's discussion after eq. 10, not derived here.
- Ben Arous et al. and Bardone–Goldt step-size windows: Codex found that the windows as printed in the theorems are not sufficient by themselves (inherited from the sources); the notebook now says "suitably tuned steps" and refers to the papers, without stating corrected conditions.
- Xu et al. venue (Commun. Comput. Phys. 2020) taken from TD_LINKS_v1.md, not rechecked.

## Readability pass (2026-10-09)

Followed `td/READABILITY_BRIEF.md`. No changes to mathematics, code or claims, apart from one cut example (below). Code cells were not touched, so the cached outputs were reused.

**First pass (mine)**
- Removed all three "The result." paragraphs (§7.1, §7.2, §7.3). Each section now opens with its question in plain words, followed by the setup. §7.2 also has a one-line roadmap. §7.1 and §7.2 get a short *Notation here* line: $x^{\otimes k}$, $\mathbb E$, Gaussian clone; $d$, $w$, overlap $\alpha$, cumulants. The content of the spoilers now appears where it is derived. The clone-trajectory statement moved into 7.1.1, and the $\varepsilon$-grading is in 7.1.3, stated in words first.
- Gave each `###` subsection a one-line lead or question: 7.1.1–7.1.4, 7.2.1–7.2.4, 7.3.1, 7.3.2.
- Defined locally the symbols that were used before their definition: $W^*$ and $t$ in 7.1.1; $M_k^\pm$ and $\Sigma_{xx}$ in 7.1.3; the NTK linearization in 7.1.4; $\beta$'s as signal strengths in 7.2.2; $\ell(w;x_t)$, $\delta$ and "equator $\alpha=0$" in 7.2.3; $\alpha_u,\alpha_v$, $L_0$, $\lambda,\nu$ and $\sigma_k$ in 7.2.4; $T_K$, $x=(\cos\vartheta,\sin\vartheta)$ and Fourier modes in 7.3.2; von Mises $\kappa$ ("not a cumulant"); cosine coefficients $a_\ell$; eNTK in 7.3.3.
- Renamed the Chebyshev polynomial $T_\ell$ to $\mathrm{Ch}_\ell$ in 7.3.2, because $T_\ell$ clashed with $T_K$.
- Removed leftovers from the two-file format: "Worked solutions are in the solutions version (`…_solutions.ipynb`)". At the critic's suggestion I also dropped my interim replacement paragraph, because the kit's header note already says this. Dropped the `{{VARIANT}}` label. The two `td2_8_further_readings_solutions.ipynb#td28-gahb` links now point to `td2_8_further_readings.html#td28-gahb`.
- MDL links: this notebook has none, so nothing was repointed or dropped. The Physics of Deep Learning day links (`../../IliadPhysicsLearning/.../*_solutions.ipynb`) are that day's real filenames and were left as is.

**Readability critic (fresh Claude Opus subagent; random picks 7.1.3, 7.2.4 and 7.3.B, then a skim of the rest).** No mathematical errors found. Findings and how I handled them:
1. *7.3.B had no lead, and $T_K$ and the von Mises $\kappa$ were undefined there.* Added a lead that restates $T_K$ and "rate = eigenvalue". Added "$\kappa>0$ sets its width". Glossed "ideal $\ell^{-2}$" as the asymptotic decay of the bias ReLU NTK.
2. *7.1.3: $W$ undefined, and the power-counting paragraph was dense.* Defined $W$ (rows $w_j$). Rewrote the grading in words first, then in symbols. Changed "Refinetti et al. (§2)" to "(their Section 2)".
3. *"The check script" is invisible to students* (three places). Each mention now links `checks/td2_7_learned_first_check.py`.
4. *The interim opening paragraph was redundant on the page and wrong in the .ipynb.* Deleted it.
5. *Exercises 7.1.C, 7.2.A, 7.2.B, 7.2.D and 7.3.A started without context.* Each now has a one- or two-line lead that restates its setting. 7.2.A restates the class-+1 formula, $Q$ and $\gamma$. 7.2.D defines $\mathbb E_+$ and $\sigma_k$. 7.3.A restates $f$, $K$, $L$ and $r_i$. 7.1.C restates $\theta$ and $\theta_0$.
6. *7.2.4 minor symbols.* Done: "no mean signal", $\sigma_k$ restated, $\lambda$ along $u$ and $\nu$ along $v$, "information exponent (§7.2.3)", $\delta$ restated in the fold.
7. *The demos had no headings and were referred to ad hoc.* Added short h4 headings, each with a question: **7.2.E** "Demo: the staircase in a two-layer ReLU network" and **7.3.D** "Demo: a trained ReLU network against the kernel prediction". References in 7.2.4, after 7.2.C, and in 7.3.3 now use these numbers. Removed the sentence "The next cell asks…". These headings are markdown cells only and are not in the sidebar, which lists `##` only. I left one code comment ("the target of the kernel demo") unchanged so that the code is not re-executed.
8. *7.1.1 had no question, and the clone was not restated.* Added both.
9. 7.1.4 now reads "The small-weight expansion of 7.1.3 has two limits".
10. 7.3.1: $\lambda_k\to\hat\lambda_k$ (a notation slip), and "$T_K$ acts as the input covariance $\Sigma_{xx}$".
11. Solution 7.1.A.2 now defines $u_\alpha,v_\alpha$ as columns of $U,V$, not the spikes of §7.2. Solution 7.1.B.4 defines $x_m=m\cdot x$ and $x_v=v\cdot x$.
12. 7.2.2 now says "one signal each in the mean, the covariance and the fourth cumulant", instead of "each of the first cumulants".
13. 7.2.3:
    - Renamed the overlap loss $\phi(\alpha)\to\Phi(\alpha)$, here and in 7.2.B, because $\phi$ clashed with the feature map of 7.1.C and 7.3.
    - Glossed "matching student" and added a pointer to §7.2.1.
    - **Cut** the clause "for tensor PCA with an order-$p$ spike, $k=p$". It was undefined there and not used. The Ben Arous reading still mentions spiked tensors.
14. Padding:
    - Removed the duplicate Kadkhodaie pointer from the opening. The Readings pointer stays.
    - Split the long paragraph after the 7.2 demo into three paragraphs.
    - Trimmed the step-size sweep commentary after 7.2.C. I dropped the $d=16$ aside and the crossover sentence; the $d=64$ numbers and the conclusion are unchanged.
15. Linked the Physics day's NB1 §3 in the 7.3 intro. Changed the overview table's "NB1 box" to "NB1 recap in *What you need*".
16. Dropped `{{VARIANT}}`. Already done.

**Checks.** `python td/build_td2_7_learned_first.py` passes (40 cells, 8 code cells from the cache, 10 solutions), `--check` reports it fresh, and `python td/checks/td_crosslinks.py` reports 54 cross-links checked, 0 bad. I did not rerun `--run`, because no code changed.
