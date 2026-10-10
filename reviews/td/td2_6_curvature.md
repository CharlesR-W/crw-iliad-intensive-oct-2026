# Review log: NB2 §6, `td2_6_curvature`

Builder: `td/build_td2_6_curvature.py`. Notebooks: `td/td2_6_curvature_{exercises,solutions}.ipynb`.
Checks: `td/checks/td2_6_curvature_check.py`.

## Draft 1 (2026-10-08)

### Source verification before drafting
A Claude subagent (Opus) checked every paper claim against the papers' text: arXiv PDFs converted to text, with key quotes re-checked. The findings that shaped the text:

- **Outlier counts.** The literature does not agree on a single number.
  - Sagun et al. 2016 (1611.07476): two outliers for two Gaussian blobs.
  - Sagun et al. 2017 (1706.04454): "match exactly the number of classes" in the main text. Their appendix Table 1 is off by ±1 or more depending on the cutoff rule.
  - Papyan 2018 (1811.07062) and 2019 (1901.08244): about C outliers, attributable to G (the GGN); the residual has none; C² structure is weaker.
  - Papyan 2020 (2008.11865): 1 (global mean) + (C−1) (between-class) + a C(C−1) cross-class mini-bulk. In the measured VGG spectrum the mini-bulk sits *below* the main bulk.
  - Fort and Ganguli (1910.05929): "C" in the text, but their own figure shows C−1.
  - Gur-Ari et al. (1812.04754): "k = number of classes", and a footnote says k−1 may fit better.
  - The notebook states this spread rather than a single number.
- **Song, Ahn and Yun (2405.16002).**
  - P_k is recomputed every step, with k = number of classes.
  - The switch happens when the EMA of the alignment exceeds 0.95, at 53–69% training accuracy ("not the end-of-training phenomenon").
  - Mostly MSE, with CE in App. C.1.
  - Mechanism: noise in an ill-conditioned valley for small-LR SGD; self-stabilization for GD at EoS. GD at small LR gives alignment of about 0.
  - Dom-SGDM and Dom-Adam fail; Bulk-GD and Bulk-SAM work. Bulk-GD at EoS pushes sharpness above 2/η.
- **Fort et al. 2020 (2010.15110).**
  - Kernel distance as defined, velocity measured over a 0.4-epoch separation.
  - Basin fixed within 2–3 epochs.
  - Kernel 3× better within 3–4 epochs.
  - The kernel matches full training at 15–45% of training.
  - Setting: ResNet20 on CIFAR-10/100.
- **Frankle et al. 2020 (1912.05671).**
  - LeNet is stable at initialization.
  - ResNet-20 at iteration 2000 (3%); VGG-16 at iteration 1000 (1.5%); ResNet-50 at epoch 18 (20%); Inception-v3 at epoch 28 (16%, test).
  - Stability means instability < 2%.
- **Gur-Ari et al.** MNIST MLP: top-20 fraction 0.20 at step 0 and 0.94 after 50 steps (App. C). Subspace roughly fixed after about 100 steps.
- **Martens 2020 §9.2** (GGN = Fisher for natural-parameter exponential families) and **K-FAC**: confirmed.
- **Not verified:** the size of Fort et al.'s kernel data subset and how they treat the logits (not used in the text). Song et al. never state explicitly that the projector uses the full-batch Hessian; the notebook says "the training-loss Hessian", which is their notation.

### Numerical exploration that changed the plan
- Outlier counts on small MLPs over Gaussian clusters.
  - Clean outliers appear only with GELU features and moderate separation.
  - At a zero-initialized last layer the count is *exactly* C−1 for CE and C for MSE, equal to λ_max(M₂)/C or λ_max(M₂), which the Kronecker computation predicts. This became Exercise 6.3.A.4.
  - After training, the pair-means part (rank ≤ C(C−1)) reproduces every eigenvalue above the bulk, and the class-means part (rank ≤ C) the top C. This is Papyan's hierarchy reproduced on a toy, and became 6.3.A and 6.3.B.
- **Structural alignment for CE (found while drafting).** The identity Σ_{k≠y} p_ik v_ik = p_iy ∇ℓ_i means the class-level means are class-mean loss gradients. For CE near interpolation, the top-C eigenspace overlaps the span of class-mean gradients at 0.98, and f_top ≈ 0.99 even under small-LR GD.
  - In that regime Dom-GD trains like GD and Bulk-GD stalls; the same holds for SGD with batch 8.
  - This is the opposite of Song et al.'s headline, but in a regime they explicitly exclude (they switch at 53–69% accuracy).
  - The notebook presents it as a claim boundary (6.4, code panel (b)), not as a contradiction.
- MSE, tanh, SGD (Song et al.'s regime) reproduces their result: Dom stalls and Bulk ≈ SGD. GD at small LR gives f_top → 0, as they report. GD at EoS gives f_top ≈ 0 during progressive sharpening and ≈ 1 once sharpness reaches 2/η.
  - A Dom/Bulk fork during an EoS burst gave a muddled result: Dom first removes the oscillation energy, and Bulk carries the frozen displacement while its sharpness rises above 2/η. So only the SGD fork is shown for MSE.
- **LMC on small MLPs was not honest at this budget.** The SGD forks either showed zero barrier from initialization (like LeNet) or barriers of about 0.01–0.06 with no clear trend across fork times, within the noise of single seeds.
  - It was replaced by the scalar linear net f = w₁w₂, a symmetry-breaking-at-a-saddle model, with exact barrier formulas (6.5.A) and a 100-parent fork experiment (6.5.B).
  - The text says explicitly that this is a mechanism model, not evidence about ResNets.

### Checks done (draft 1)
- `td/checks/td2_6_curvature_check.py` verifies with sympy, numpy and torch:
  - the softmax gradient and Hessian, the second-moment form, PSD, rank C−1, the trace and λ_max bounds, and the near-interpolation expansion (residual O(ε²));
  - the scalar-net H_res and det H on the balanced line;
  - the GGN/NTK shared nonzero spectrum, the eigenvector map and the rank bound;
  - the balanced linear-net Hessian spectrum {s_a + s_b, s_a, s_b} and S = 2s₁, against a numerical Hessian;
  - the zero-last-layer Kronecker form;
  - ΔL = −ηg²(1 − ηλ/2) and the stationary variance ησ²/(a(2−ηa));
  - both scalar-net barriers, including a numeric check that the opposite-branch maximum is at α = ½.
- In-notebook checks:
  - H = H_GN + H_res to 4e-16 relative;
  - softmax ranks;
  - the GGN/NTK spectra agree to 1e-14;
  - rank H_GN = n(C−1) = 384 exactly;
  - the zero-last-layer outlier values match the prediction to 1e-15.
- The block-power-iteration subspace was compared with exact eigh in prototypes: subspace overlap 0.999–0.99995, top eigenvalues within 0.5%.
- `--run`: 56 s total, slowest cell 27 s (the Dom/Bulk forks), on a machine shared with other agents.

## Critique round 1

**Critics.**
- Design: Claude Opus subagent.
- Technical: Codex was launched twice. Both runs died with "Selected model is at capacity" after partial progress (logs in `/tmp/codex-companion/.../task-mv095fhm-m44p6z.log` and `task-mv09ic2y-fic0d0.log`). Their partial findings were applied. A Claude Opus stand-in did the full technical review, as the brief allows.
- The session was interrupted overnight; the coordinator relayed both full reports on 2026-10-09.

**Technical findings and what changed.** The stand-in found no derivation errors. All of the following were applied:
- From the partial Codex runs:
  - The scalar-net branch was named by the sign of $w_1w_2$, but both branches have $w_1w_2=s$. Now named by the common sign of $w_1,w_2$.
  - The valley $f_{\rm top}$ was stated as a ratio of expectations. Now labelled as a proxy.
  - The CE claim boundary asserted "class-mean span = top eigenspace" as if proved. Now stated as empirical and measured.
  - A missing $1/C$ in the 6.3.A.4 prompt.
  - Weighted vs unweighted class-gradient span. The $O(\varepsilon^2)$ difference is now stated.
- Papyan's counts:
  - The cross-class layer is now "at most $C(C-2)$ beyond the top $C$" (measured $C^2-2C$; the schematic says $C(C-1)$). 6.3.A.3 now derives the split $C+C(C-2)$; checked in the script.
  - The $1+(C-1)$ split is attributed to features, errors and weights only; for the Hessian, all $C$ come from class means.
  - BatchNorm *shrinks* the outliers.
  - Papyan 2018 and 2019 are attributed separately.
- "Hierarchy is clean" was false at $C=8$ (no gap near 56). Rewritten: class level clean; cross-class level merges into the bulk at $C=8$.
- Kernel ⇒ Gauss–Newton motion holds only one way ($J\mapsto JQ$). Fixed in 6.2 item 4 and 6.5.
- Fort et al.: it is linearized training, not kernel regression; "at least 3×" lower error; 15–45% means epochs 30–90.
- Block power iteration: qualified (overlap about 0.8 where the 4th/5th eigenvalues are near-degenerate).
- Ratio × margin: the text now says it drifts 1.6 → 1.85 over a ~300× loss drop.
- EoS timeline rewritten: initial catapult from $S=10$; at the edge from about step 80; aligned only from about step 180.
- "Progressive sharpening = feature learning" softened (square loss near interpolation; may be scale growth).
- The zero-last-layer case is now attributed to Papyan's §9.3 mechanism, not "the 1".
- Mean-dominated wording ("typically").
- Gur-Ari: top-20 vs top-10 numbers; "sometimes rotate".
- Sagun Table 1: "often far off".
- Song details:
  - 53%/69% come from two settings;
  - momentum/Adam alignment is partial;
  - SAM is full-batch;
  - CE, VGG and ResNet runs are Dom-only;
  - the toy is $L\pm100xy$.
  - I re-checked "Bulk-SGD raises sharpness faster" in the paper (§3.2 and App. G): it is confirmed, so it was kept.
- Frankle: the figures are test error; on training error ResNet-50 is later and Inception-v3 never stabilizes in their runs.
- Fort–Ganguli cluster by logit index; Martens §§8, 9, 11; top-4 $H$ vs $H_{\rm GN}$ differ by 0.1–7% (the cell now prints this).
- The linear-net check now uses a rank-deficient target ($r=2<m=4<d=5$).

**Design findings and what changed.**
1. **6.5 missed the main ask (GGN/NTK ↔ basin commitment).** In the one-mode toy the kernel only grew, so Fort's cosine distance was zero, and the lazy/rich dial was unused. 6.5 is rebuilt:
   - Exercise 6.5.A item 5: imbalance fixes the branch at initialization. Exact, since $w_2^2\ge d$; $\Theta_{\rm end}=\sqrt{d^2+4s^2}$, checked.
   - Item 6: two modes $s_1>s_2$ give two-stage commitment and a rotating normalized kernel.
   - The new code runs balanced, moderately imbalanced and strongly imbalanced (lazy) settings, with per-mode branch disagreement and Fort's kernel velocity.
   - Results:
     - balanced: mode 1 commits by step ~100 and mode 2 by ~300, with velocity bursts in both windows;
     - imbalanced: zero barrier at every fork;
     - moderate imbalance still has a fast-moving kernel early, so kernel motion alone does not imply a choice. The text says so.
2. Overclaims fixed:
   - the opening's "every claim checked";
   - the sign symmetry of the bulk (now 474 vs 202, printed);
   - $C(C-2)$ visibility;
   - Song's "same for all optimizers".
3. The paper-by-paper narration in 6.3 and 6.5 is condensed to synthesis paragraphs, with the detail in the blurbs.
4. 6.2.B is replaced by a non-trivial code exercise:
   - top kernel eigenvectors are class-indicator-like (0.97 vs baseline 0.03);
   - the unweighted kernel grows while the $A$-weighted kernel (the curvature) shrinks about 200× and becomes less class-like.
5. Panel (b) vs 6.4.A: added the near-interpolation argument $\nabla^2\ell\approx\ell(\nabla m\nabla m^\top-\nabla^2m)$. The alignment there is not an oscillation.
6. Opening rewritten, shorter, with depth signals. Fort et al. 2020 is now the "read one paper" pick.
7. "What you need" box: added the SGD floor, defined balancedness and imbalance, and pointed infinite-width NTK theory to the Physics day. Cross-references to `td1_linear_networks` now name §4 (kernel) and §5 (lazy/rich), per the coordinator.
8. 6.3.A is labelled "derived given clustering", with the last-layer reason $(e_k-p_i)\otimes\tilde h_i$. $M_{\rm pair}\succeq M_{\rm class}$ is now stated as exact.
9. Readings and structure:
   - the Fort–Ganguli "gotcha" is cut;
   - 6.6 is folded into 6.1 ("Where this is used next");
   - Atanasov–Bordelon–Pehlevan (silent alignment, TD_LINKS item 40) is added.
10. Minor:
    - "loss barrier"/"error barrier" terminology;
    - the dropped MLP-fork line is removed from the notebook and recorded here only;
    - the EoS catapult points to §2;
    - off-axis $2/\eta$ lines are no longer drawn;
    - the dotted line at 4.5 is removed.

**After round 1:** builds, `--check` and `--run` pass (48 s total, slowest cell 18 s); the check script passes.

## Critique round 2

**Critics.** A fresh Claude Opus design critic, whose report was relayed by the coordinator. Codex, which succeeded this time (`codex-companion task --effort high`, read-only).

**Both critics independently found that the kernel–curvature link was overstated.**
- In both directions, only the nonzero spectra are tied. $J\mapsto JQ$ rotates $H_{\rm GN}$ with $\Theta$ fixed; $J\mapsto RJ$ rotates $\Theta$ with $H_{\rm GN}$ fixed (square loss). Codex gave the $2\times2$ example.
- Fixed in four places: the pitch ("spectral statements transfer; eigenvector statements do not"), 6.2 item 4 (the eigenvector map now moves the *loss-weighted* logits), 6.5 (the spectrum divided by $n$; Fort's unweighted cosine speaks to curvature only through its spectral part after the $A$-reweighting) and 6.2.A item 3.
- 6.2.A item 3 is now about an exactly linearized model. Codex's $f=\|\theta\|$ example shows that a fixed kernel does not fix $H$, because $H_{\rm res}$ and the eigenvectors can still move. Sharpening is now said to imply kernel growth only when the residual is negligible.

**Two-mode toy (6.5).** The design critic showed from the output itself that commitment does *not* coincide with kernel-velocity windows: mode 2 commits while the whole-kernel velocity is at its minimum. Codex derived the exact linearized commitment curve $\Pr(\text{opposite})=\arccos\rho(t)/\pi$ and showed it is independent of how small $a_0$ and $\sigma$ are, while the escape is logarithmically later.
- The lesson is rewritten as spectral: commitment precedes the growth of the corresponding kernel/GGN eigenvalue, and a whole-kernel cosine is blind to weak modes.
- The code now plots the per-mode eigenvalues $\Theta_m(t)$ and overlays the $\arccos$ prediction. It matches the measured fractions closely. The formula is verified by Monte Carlo in the check script.
- Exercise 6.5.A item 6's answer is rewritten. Item 3 gains a discrete-time hint (no Itô needed) and the exact formula.
- Barriers are now printed in scientific notation. The imbalanced runs show within-branch barriers of $10^{-5}$–$10^{-6}$, not zero.
- The text says the two branches are one function (symmetric copies).
- Imbalance thresholds are separated: above-noise imbalance fixes the branch, while laziness needs $d\gg s$. Also added the $w_1(0)^2\ll d$ assumption, the $\sim2s/d$ growth for $d\ll s$, and a note that conservation is exact only for gradient flow.

**6.4.**
- "A claim boundary" is now "Where this holds", about half as long.
- Panel (b) is presented as late-phase margin/scale growth, consistent with Song et al. switching mid-training, not as a counterexample.
- The margin heuristic now states its assumptions: a dominant competitor, a negligible margin Hessian, and small $\eta\ell\|\nabla m\|^2$. Codex's counterexample shows these are needed.
- Song's statistic is the unsquared $\sqrt{f_{\rm top}}$.
- Their toy's noise is $\pm100(y,x)$: proportional to the position along the floor.
- Paper particulars moved to the blurbs.
- Power-iteration accuracy is qualified with Codex's independent measurement: overlap 0.985, $f_{\rm top}$ 0.58 vs an exact 0.53. Also noted that power iteration finds largest-magnitude eigenvalues.
- The effect size of panel (a) is stated.
- 6.4.A item 3 now asks for the moment-ratio proxy and assumes Gaussian noise, after Codex's heavy-tailed counterexample.

**6.3.**
- 6.3.A is marked core vs optional.
- 6.3.A.4 now assumes a simple, separated $\lambda_{\max}(M_2)$ (multiplicities $q(C-1)$, $qC$ otherwise).
- The feature-mean claim is scoped. The degeneracy is attributed to the isotropic softmax factor, not to "no class information".
- $C-1$ is said to be shown only in the zero-last-layer case.
- $M_{\rm class}$ "lower-bounds and tracks" the outliers.
- The span approximation needs non-cancellation.

**Smaller fixes.**
- Rank one away from the origin (6.1.A.3).
- Linear logits excluded from the $\lambda^{-k}$ ratio (6.1.B.5).
- $v$ stays at the noise scale (6.5.A.3).
- Silent alignment needs whitened inputs.

**Opening.**
- Five-fact list.
- Fast path for basin commitment.
- Toy outliers admitted to be weak.
- "About an hour" for the read-only path.
- The linear-network bullet split into three.
- `td1_linear_networks` cross-references: §4 kernel, §5 lazy/rich.

**Readings.** Cut K-FAC (already linked in 6.1) and Fort–Ganguli (still linked inline). Song is labelled "core for 6.4". Frankle and Fort numbers appear once, in the blurbs.

**Checked and confirmed by Codex.**
- All derivations: decomposition, softmax, Weyl, shared spectrum and eigenvector map, linearized GD, the linear-net spectrum with $S=2s_1$, pair/class means and the $C+C(C-2)$ spanning bound, the valley, the barriers, the convexity bound.
- The code functions.
- The 6.2.B outputs: rank 384, spectra agree to about $10^{-14}$, weighted shrinkage about 170–200×.
- The 6.4.B qualitative results.
- All revised paper numbers (Gur-Ari, Frankle, Fort, Song) and the Sagun/Papyan/Ghorbani count discussion.

**After round 2.** Builds, `--check` and `--run` pass (44.8 s total, slowest cell 15.8 s). The check script passes, including the new Monte Carlo check.

## Critique round 3

**Critics.** Codex technical review, which succeeded. A fresh Claude Opus design critic; its report was relayed by the coordinator. Neither found brief violations or algebra errors. Both concentrated on how 6.5 describes its own toy.

**6.5 lesson, corrected again.**
- The round-2 wording said commitment happens "while the eigenvalue is at the noise floor". The output contradicts this: over mode 1's commitment window $\Theta_1$ grows about a hundredfold, though it is still about 40× below its final value.
- Codex showed that the escape is seeded by $a_0^2+q$, with $q=\sigma^2/2s$, not by $a_0^2$ alone. Commitment needs a tolerance. With one, $t_\epsilon=\frac1{2s}\log\frac{q/(a_0^2+q)}{1-\cos\pi\epsilon}$, which is invariant to the common scale of $a_0$ and $\sigma$ but not to their ratio. Both results are now in the answer to 6.5.A, item 3.
- The lesson now reads: a basin is chosen during the early exponential growth of the corresponding kernel/GGN eigenvalue, before it is an outlier in absolute terms.
- The whole-kernel cosine registers the dominant mode's commitment window (consistent with Fort et al.'s timing) and misses the weak mode's.
- The "What to look for" bullets are rewritten against the measured numbers:
  - the velocity sequence: noise, then mode-1 escape, then a minimum while mode 2 commits, then a rise over steps 150–300;
  - the $\arccos$ prediction holds only roughly, with tails a few points off.
- 6.5.A item 3:
  - "limiting brackets are jointly Gaussian" (not their signs);
  - Sheppard's formula is named;
  - an optional sub-item derives $\rho$;
  - the item-3 hints are collected in one place.
- Item 5's "In between" now says "about $2s/d$".

**Other fixes.**
- 6.1.C:
  - eigenvalue counts now 466/193 plus 17 exact zeros, which are CE's logit-shift gauge directions, reported separately;
  - the histogram title says "nonzero";
  - "orders of magnitude" is replaced by the actual weak separation ($\lambda_4/\lambda_5\approx1.6$).
- 6.2:
  - the sharpening inference carries "when $H_{\rm res}$ is negligible";
  - title changed to "…share their spectrum";
  - item 4 of 6.2.A is marked optional;
  - the "only spectra are tied" point now appears once (6.2 item 4) and 6.5 points back to it.
- 6.3: the observation bullets are cut to one clause each.
- 6.4:
  - Song's switching statistic uses the full-batch gradient;
  - "Where this holds" is compressed and uses panel (b)'s own $f_{\rm top}\approx0.99$;
  - the power-iteration paragraph is one sentence with the correct cadence (every 5 steps / every 2 steps);
  - panel (a) numbers are corrected (Bulk 0.032→0.026, SGD 0.031→0.025);
  - the stale "claim boundary" reference is fixed;
  - catapult is defined in a clause.
- Papyan blurb: the cross-class layer is attributed to gradient/Fisher and error spectra, not to features or weights.
- Pitch:
  - fact 3 is conditioned on SGD or the edge of stability, typically without BatchNorm;
  - one sentence states 6.5's spectral conclusion;
  - "You want this section if…" follows the facts;
  - "four-parameter linear network";
  - $\tau=1/\eta$ is defined in the box.
- 6.5 "What is measured" cross-links `td2_8_further_readings` §4 (critical periods). I checked Achille–Rovere–Soatto's abstract myself ("information rises rapidly in the early phases of training, and then decreases").
- The optional GGN-subspace tracking cell the design critic suggested was not added, to keep the notebook's length and run time in check.

**After round 3.** Builds, `--check` and `--run` pass: 45–70 s total depending on machine load, slowest cell 16–25 s. The check script passes.

## Remaining uncertainties
- The paper text never says outright that Song et al.'s projector uses the full-batch Hessian; the notebook reads it from their notation.
- Fort et al.: the size of the kernel's data subset and how the logits enter were not checked.
- 6.4 panel (b), the late-phase CE regime where Dom trains and Bulk stalls, is one seed of one toy, plus one unshown SGD check. We have not isolated which difference from Song et al.'s setting matters.
- The power-iteration $f_{\rm top}$ curves are approximate (overlap with the exact subspace is 0.985 at the end of the SGD run, and about 0.8 at near-degeneracies).
- The toy's outliers are weak. 6.3.B's clean hierarchy comes from decomposing the matrix, not from visible gaps.

## Final Codex check (changed parts only)
Codex re-checked the round-3 changes and found two issues; both are fixed.
- 6.5.A item 6 (hint and answer) seeded the escape with $\Theta_m(0)$ alone. It now uses $a_{0,m}^2+q_m$, with the linearized crossing time $\frac1{2s_m}\log\frac{s_m+q_m}{a_{0,m}^2+q_m}$.
- Opening fact 3 now confines "small-step GD loses alignment" to Song et al.'s tested regime and points to 6.4 for the late-CE exception.

Everything else checked out: the $\rho(t)$, $t_\epsilon$ and $\mathbb E\,u^2$ formulas, and the 6.5.B bullets against the output. The notebook was rebuilt, and `--check`, `--run` and the check script all pass.

## Readability pass (2026-10-09)

Following `td/READABILITY_BRIEF.md`. Prose only: no code cell changed (the outputs are the cached ones), and no mathematics or claims changed.

**Section openings and local notation.**
- Every `##` subsection (6.1–6.5) now opens with its question in one or two plain sentences, followed by a "*Notation here:*" line defining the symbols it uses ($n,C,P,z_i,J_i,J,A_i,A,H,\Theta,S,P_k,g_B,\eta,p_{ik},e_k,\mathcal E,\Theta_t$, as needed).
- Every exercise and code subsection opens with what it is for and a one-line notation restatement:
  - 6.1.A, 6.1.B, 6.2.A, 6.3.A, 6.4.A and 6.5.A;
  - 6.1.C, 6.2.B, 6.3.B, 6.4.B and 6.5.B.
- 6.5.A now defines children, branches, the loss barrier, $\Theta$, $d$ and Fort et al.'s kernel distance. Item 3 explains "limiting brackets" as the limit of $e^{-st}u(t)$.
- 6.5.B states the two-mode model, the noise convention $w\leftarrow w-\eta\nabla L+\sqrt\eta\sigma\xi$, what "rich" and "lazy" mean, the barrier, "opposite branches", the kernel velocity, and why the final value is $2s_m$.

**Spoiler and density fixes.**
- The pitch's "thread through the section" paragraph is now in plain words.
- The five facts gloss "Gauss–Newton part", "tangent kernel" and "same basin". Fact 3's caveat is rewritten and attributed to Song, Ahn and Yun.
- In 6.5, the scalar model and its two quantities ($\Theta$ and the imbalance $d$) are introduced before the list of results. The list's restatement of the main lesson is cut, so the lesson is stated once, in words.
- The 6.3 count paragraph ("is it $C$, $C-1$ or $C^2$") is rewritten in words.

**Small definitions added.** In 6.1:
- the stacked $J$;
- $e_y$;
- homogeneous networks;
- margin (also in 6.1.C);
- "bulk" (pointing to 6.3).

Also:
- "progressive sharpening" and $r=z-y$ in 6.2;
- $g$ in 6.4's one-line reconciliation;
- $\Sigma$ at its use in 6.4;
- self-stabilization in a clause in 6.4;
- $\Sigma_{yx}$ in the "What you need" box;
- $m,d,u_\alpha,v_\beta$ in 6.2.A item 4.

**Cuts.**
- The repeated "second-order methods use GN" sentence in 6.1 (the point is made in the opening question and in "Where this is used next").
- The duplicated eigenvector map before 6.2's list (kept in item 4).
- NB1's $\Delta$ aside in the box.
- "The check script verifies…" in the 6.2.A solution (students have no check script).
- The duplicated run instructions under the kit's run note.

**Old two-file wording.**
- Dropped "Worked solutions are in `td2_6_curvature_solutions.ipynb`" and the `{{VARIANT}}` label.
- The `td2_8` link now uses `.html`.
- "The box" is replaced by "What you need" (in the table's Uses column, 6.4 and the 6.5.A solution).
- "§2 of NB2" for the catapult is now a link to `td2_2_edge_of_stability.html#td22-7`.
- The prerequisite notebooks in the box are now links.

**MDL links.** The only one is `../notebooks/06_mysteries_myriad.html#wb6-section-9`. I checked that it still exists and still holds the tiny-subspace reading and Exercise 9.1 (the quadratic toy). Kept unchanged. Nothing points into WB2.

**Critic.** One fresh Claude Opus readability critic read three random subsections (6.1, 6.5.B, 6.4.A) and skimmed the rest. Its main findings were:
- undefined terms in 6.5.A and 6.5.B;
- exercise subsections opening without a question;
- small undefined symbols in 6.1;
- repetition in 6.1, 6.2 and 6.5;
- "box" jargon and the check-script sentence;
- $k$ vs $C$ in 6.3.

All of these were applied as listed above. The critic found no compressed "The result." paragraphs and no math errors (it spot-checked four results).

It also found a kit bug, which is not fixable from this builder: the run note's `td2_6_curvature.ipynb` link is rewritten to `href="td2_6_curvature.html"`, so it points back to the page itself. This affects all nine pages. It was reported to the coordinator.

**Checks.** `python td/build_td2_6_curvature.py`, `--check` and `python td/checks/td_crosslinks.py` all pass (60 cross-links, 0 bad).
