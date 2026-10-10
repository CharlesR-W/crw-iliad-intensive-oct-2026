# Figure plan: td2_3_sgd_sde (NB2 §3, SGD as a stochastic differential equation)

- **Notebook:** `td/build_td2_3_sgd_sde.py` (authoritative; read 2026-10-09), generated `td/td2_3_sgd_sde_{exercises,solutions}.ipynb`
- **Date:** 2026-10-09
- **Proposals:** 3 × P1, 5 × P2, 7 × P3 (the P3 items are mostly small fixes to existing code-cell plots)
- **How I checked:** I ran every code cell of the solutions variant (Agg backend, about 50 s in total) and looked at all 7 plots it produces. I prototyped the two main self-generated figures (F1, F3) in my scratchpad to confirm the recipes run in under 10 s and show what is claimed. Paper figures were checked against the arXiv PDFs (`pdftotext`), and licences against the arXiv abs pages.

## What already exists

- **Code-cell plots (7), all reasonable and correctly described in the prose:** 3.2.B weak-error slopes (2 panels); 3.3.B floor/SDE ratio against ηλmax; 3.3.B learning-rate-decay trace with the exact expectation; 3.4.B steps–examples Pareto points with the hyperbola fit; 3.4.C B_simple trace (solutions only, or once the stub is filled); 3.5.B SVAG floor ratio against l; 3.6 Adam square-root vs linear rule (2 panels).
- **No static images.** The committed `.ipynb` files have **no stored outputs**, so a student who opens the notebook sees no picture at all until they run code. Every section opens on prose and equations only.
- **Interactive:** the notebook links the MDL figure `notebooks/assets/interactive/sgd_stationary_vs_posterior.html` from §3.3 (correctly placed). None of the three TD interactives belongs here. I don't propose a new interactive: F3 below is a static companion that answers this notebook's own question, which differs from the MDL figure's posterior comparison.

---

## Proposals, in notebook order

### F1 · Opening hook: "temperature sets the floor; the linear-scaling family collapses until ηλmax is not small" (P1)

1. **Anchor:** opening cell, after the paragraph ending "It ends by asking whether the Taylor expansion behind all this is valid at practical learning rates." and before "You want this section if…".
2. **What it shows:** two panels.
   - **(a) 2-D basin.** Loss contours of L = ½(θ₁² + 0.1 θ₂²). A gradient-flow/GD path from (2, 3), and two SGD paths with Σ = H (well-specified noise) at η/B = 0.2 and 0.05. Each SGD path follows the GD path down (the *descent phase*) and then jitters in a cloud whose size is set by η/B (the *fluctuation phase*). Plotting the end-clouds of 500 chains instead of single paths reads more clearly than tangled paths; my prototype with single paths was busy.
   - **(b) Excess loss against SDE time t = kη (log–log)** on the notebook's own least-squares problem (d=8, N=1000, the same construction as 3.3.B). Use the family η/B = 0.01 with (η,B) = (0.01,1), (0.04,4), (0.16,16), (0.96,96), plus one colder run (0.04,16) at η/B = 0.0025. Draw dotted horizontal lines at η tr Σ/4B. Label each curve with ηλmax.
   - What the prototype showed: the η/B = 0.01 curves at ηλmax ≤ 0.3 coincide through descent and settle on one floor. The colder run descends identically and then floors 4× lower. The ηλmax = 1.82 member departs early and floors about 6× higher.
3. **Why here:** this is the notebook's whole story in one picture. Noise is invisible during descent and dominates the floor, the floor scales with η/B, the order-1 SME depends on η/B alone (so the LSR family collapses in SDE time), and the collapse fails once ηλmax is O(1). A student sees all of it before reading any SDE theory, and every later section quantifies one feature of the picture. At present the first picture comes in 3.2.B and is a weak-error convergence plot, which is not a hook.
4. **Source:** (b) generate ourselves. A pre-rendered static PNG (e.g. `td/assets/figures/td2_3_hook.png`) made by a small script kept with the checks, e.g. `td/checks/td2_3_make_figures.py`, so it can be regenerated. Recipe: 64 chains per run, horizon t = 40, start at w* + 1.5·N(0,I); the prototype ran in 8 s. Panel (a): R = 500 chains, 600 / 2400 steps at η = 0.2 / 0.05, B = 1, ξ ~ N(0, H/B). *Alternative paper figure:* Li, Tai and E (arXiv 1511.06251) Fig. 1(b) shows 50 SGD sample paths on a 1-D quadratic with the predicted descent→fluctuation transition time k*. I checked that it shows this. It is less informative than (b), and the licence is arXiv non-exclusive (permission needed to embed).
5. **Priority:** P1.
6. **Caption draft:** *SGD on a quadratic basin (a) and on least squares (b). In SDE time t = kη, runs with the same η/B follow the same curve and settle at the same floor η tr Σ/4B (dotted), as long as ηλmax is small. Lowering η/B lowers the floor without changing the descent, and the ηλmax ≈ 1.8 member of the family leaves the curve, which is the breakdown §3.3–§3.5 compute.*

### F2 · Reshuffling as a Brownian bridge (P2)

1. **Anchor:** §3.1, directly after the "Random reshuffling" bullet ("…so the noise cancels over the epoch.") and before "**Why η/B.**". Alternatively in the 3.1.A.2 solution, after "This is the covariance of a discrete Brownian bridge…".
2. **What it shows:** one panel. The x-axis is the batch index n within an epoch (0…m, with m = 100); the y-axis is the accumulated noise Z_n (one coordinate, or its projection on Σ's top eigenvector), scaled by √(Σ/B). Show about 8 sample paths with replacement (grey, random walks that keep spreading) and about 8 with reshuffling (colour, all pinned to 0 at n = m). Overlay ±2σ envelopes from the closed forms: √(n) for with-replacement, √(n(N−nB)/(N−1)) for reshuffling (a parabola-like envelope).
3. **Why here:** the claim that "the noise cancels over the epoch" is easy to read past. Seeing the bridge pinned at n = m makes the later conclusion of the last 3.3.B cell (reshuffled SGD sits below the SDE floor in slow directions) obvious instead of surprising.
4. **Source:** (b) generate ourselves, as a static PNG. Use the notebook's U = per-example gradients at w* from 3.3.B, or any N = 1000 zero-mean vectors. Recipe: permute the indices and take cumulative sums of batch means. It runs in under 1 s.
5. **Priority:** P2.
6. **Caption draft:** *Accumulated gradient noise over one epoch of m = 100 batches. With replacement (grey) it is a random walk whose variance grows like n. With random reshuffling (colour) it is a discrete Brownian bridge: its variance peaks mid-epoch and returns to zero at the epoch's end, because every example has been used exactly once.*

*(Optional second panel, P3: drift length nη‖G‖ and noise length η√(n tr Σ/B) against n on log–log axes, crossing at n_* = B_simple/B. This illustrates the "second reading of the same computation" paragraph. Add it only if the panel is wanted anyway.)*

### F3 · What the stationary cloud looks like: Boltzmann only when Σ ∝ I (P1)

1. **Anchor:** §3.3, directly after the three bullets that follow "The exercise shows the following." The first bullet is "The law is ∝ e^{−cL} (Boltzmann) iff Π ∝ H^{−1}, iff Σ ∝ I…". Place it before "Whether this cloud resembles a Bayesian posterior…", which then links the MDL interactive as the place to explore further.
2. **What it shows:** three panels with equal axes on H = diag(1, 6). In each, show loss contours (grey), the simulated SGD cloud (2000 chains, dots), the OU stationary 2σ ellipse from the Lyapunov equation (solid), and the Boltzmann ellipse e^{−L/T} with the same mean loss (dashed).
   - (i) **Σ ∝ I:** the solid and dashed ellipses coincide.
   - (ii) **Σ ∝ H:** a round cloud, wider than Boltzmann in the sharp direction and narrower in the flat one.
   - (iii) **Σ = R H Rᵀ, rotated 60°:** a tilted ellipse that is neither. Overlay a streamplot of the stationary probability current J = (−H + ½DΠ⁻¹)θ·p(θ), which circulates around the ellipse.

   Annotate (i) and (ii) with "J = 0". Normalise tr Σ to the same value in all three panels, so that all three clouds have **the same mean excess loss** η tr Σ/4B, a visual check of the second bullet.
3. **Why here:** this is §3.3's central conceptual result, and the notebook flags it as a deliberate departure from the brief: Σ ∝ H gives an isotropic cloud, not a Boltzmann one. The prose states it entirely in algebra. The figure also shows two things the text says only in words. First, detailed balance (J = 0) and "Boltzmann in L" are different conditions: panel (ii) has J = 0 but is not Boltzmann. Second, the floor is independent of the cloud's shape. The MDL interactive explores posterior-vs-SGD with sliders. This static figure answers this notebook's question (Boltzmann or not) and needs no browser.
4. **Source:** (b) generate ourselves, as a static PNG. Recipe from my prototype (2 s): η = 0.02, B = 1, 1000 steps from 0 with Gaussian ξ ~ N(0, Σ). Use a smaller η than my prototype's 0.05 so that the discrete cloud matches the continuous ellipse to a few percent: at ηh_max = 0.3 the discrete law is visibly about 15% wider in the sharp direction. I confirmed that J vanishes identically in (i) and (ii) and not in (iii) (max |−H + ½DΠ⁻¹| = 5.3 there). Use `streamplot` for J: a raw `quiver` reads as shear rather than circulation.
5. **Priority:** P1.
6. **Caption draft:** *Stationary SGD clouds in the basin L = ½(θ₁² + 6θ₂²) for three noise covariances with the same trace. Solid: the Ornstein–Uhlenbeck stationary law; dashed: the Boltzmann law e^{−L/T} with the same mean loss. They agree only for isotropic noise (left). Noise aligned with the Hessian gives a round cloud (middle), and rotated noise gives a tilted cloud with a circulating probability current (right, arrows). All three have the same mean excess loss η tr Σ/4B.*

### F4 · Add the order-2 SME's prediction to the 3.3.B floor sweep (P2, improvement to an existing code cell)

1. **Anchor:** the second code cell of 3.3.B (begins `r = 0.01  # eta / B, held fixed`), the one that plots "excess loss / (η tr Σ / 4B)" against ηλmax.
2. **What it shows:** add one curve, the order-2 SME's frozen-noise floor. Solve the Lyapunov equation with drift matrix H + (η/2)H², i.e. `excess(lyap(H + eta/2*H@H, eta/B*Sigma))`, divided by the SDE value. It falls *below* 1 (about 0.53 per mode at ηh = 1.8) while the truth rises above it. Optionally also add the additive-noise discrete curve, to separate discreteness from state dependence. Switch the legend's "SDE: same for all rows" to "order-1 SME".
3. **Why here:** 3.3.A.4(d)'s point, that the order-2 SME gets even the *sign* of the correction wrong, is striking and is quoted again in §3.7 item 2. At present it exists only as algebra. One extra line in an existing plot makes it visible.
4. **Source:** (d) existing code-cell plot, plus about 2 lines of code. No runtime cost.
5. **Priority:** P2.
6. **Caption draft (as a sentence in the following markdown):** *The order-2 SME (dotted) moves the wrong way: its correction lowers the floor, while the true discrete floor rises toward the stability edge.*

### F5 · Critical-batch-size knee and steps–examples hyperbola, closed form (P1)

1. **Anchor:** §3.4, immediately after the display defining ε_opt, ΔL_opt and B_noise, before "with ε_max = ‖G‖²/GᵀHG…". Alternatively after the "Steps versus examples" display, (S/S_min − 1)(E/E_min − 1) = 1.
2. **What it shows:** two panels, closed form only.
   - (a) ΔL_opt/ΔL_max = 1/(1 + B_noise/B) and ε_opt/ε_max (the same function) against B/B_noise on a log x-axis. Add the small-batch asymptote B/B_noise (dashed, "linear scaling rule"), the plateau at 1 (dashed, "full batch"), and a marker at B = B_noise where progress is 50%.
   - (b) The Pareto hyperbola: E/E_min against S/S_min on log–log axes, with tick-marks labelled by B/B_noise = 1/16 … 16, and the knee (2, 2) marked "B = B_crit".
3. **Why here:** McCandlish et al.'s argument is anchored on exactly these two pictures (their Fig. 3 right and Figs. 1/7). The notebook has the measured version (3.4.B), but only after the exercises and as a code output. A student reading the default text needs to see the knee *as they meet the formula*, so that "for B ≪ B_noise progress is linear in B; for B ≫ B_noise it saturates" registers at a glance. The later measured plot then reads as a check of this one.
4. **Source:** (b) closed form, as a static PNG. It needs no simulation and draws instantly. *Paper alternative:* McCandlish et al. (arXiv 1812.06162). I verified from the PDF that Fig. 3 (right) plots the predicted training speed against batch size with the "turning point" at B_noise (50% of maximum), and that Fig. 7 shows measured SVHN Pareto frontiers with fits of their eq. 2.11. Their licence is arXiv non-exclusive, so embedding needs permission. A self-generated figure avoids that and matches the notebook's notation.
5. **Priority:** P1.
6. **Caption draft:** *Left: one optimally tuned step at batch size B achieves the fraction 1/(1 + B_noise/B) of a full-batch step. It is linear in B well below B_noise (the linear scaling rule) and saturates above it. Right: the resulting tradeoff between steps S and examples E = SB. At B = B_crit ≈ B_noise both are twice their minimum.*

### F6 · Measured critical batch sizes across tasks (paper figure) (P2)

1. **Anchor:** §3.4, after "In their experiments B_simple predicted the measured B_crit to within an order of magnitude, across tasks where B_crit ranged from about 20 (an SVHN autoencoder) to millions…".
2. **What it shows:** McCandlish et al., Fig. 4: critical batch size against the simple noise scale (averaged over a training run) for many tasks, spanning several orders of magnitude, with Dota 5v5 marked as a lower bound.
3. **Why here:** it is the empirical punchline of the section, and the only evidence in §3.4 that the one-step model matters for real networks. The notebook's own measurement is on least squares.
4. **Source:** (a) paper figure. McCandlish, Kaplan, Amodei and the OpenAI Dota Team, "An empirical model of large-batch training", arXiv [1812.06162](https://arxiv.org/abs/1812.06162), **Figure 4**. I verified the caption in the PDF: "The 'simple noise scale' roughly predicts the maximum useful batch size for many ML tasks… for Dota 5v5 we show the batch size used by the OpenAI Dota team as a lower bound". **Licence:** arXiv non-exclusive distribution licence, so embedding requires the authors' permission (or a fair-use judgement by CRW). The fallback is a one-line pointer ("see their Fig. 4") in the existing sentence.
5. **Priority:** P2.
6. **Caption draft:** *McCandlish et al., Fig. 4: the critical batch size measured on each task against the simple noise scale averaged over training, from SVHN to Dota. The noise scale predicts B_crit to within about an order of magnitude over roughly six orders of magnitude. Reproduced from arXiv:1812.06162 [with permission].*

*Related pointer, no figure needed:* their **Fig. 15** shows B_simple jumping 16× when the temperature is cut 16×, then returning. The 3.4.C code cell reproduces this in a toy (the tenfold jump at the learning-rate cut). A half-sentence "compare McCandlish et al.'s Fig. 15" in the markdown after 3.4.C would connect the two. I verified the caption in the PDF.

### F7 · The linear scaling rule works, then breaks (paper figure) (P2)

1. **Anchor:** §3.5, after "Goyal et al. used it, with a gradual warmup, to train ResNet-50 on ImageNet at batch size 8192 without loss of accuracy; beyond that, accuracy degraded."
2. **What it shows:** Goyal et al., Fig. 1: ImageNet top-1 validation error against minibatch size (64 to 64k) with ±2σ bands. The error is flat to 8k, then rises.
3. **Why here:** it is *the* canonical LSR picture, and it gives the "where it fails" question a concrete shape before the three mechanisms are listed. F1(b) is the toy analogue.
4. **Source:** (a) paper figure. Goyal et al., arXiv [1706.02677](https://arxiv.org/abs/1706.02677), **Figure 1**. I verified the caption: "ImageNet top-1 validation error vs. minibatch size… For all minibatch sizes we set the learning rate as a linear function of the minibatch size". **Licence:** arXiv non-exclusive, so permission or a fair-use call is needed; fall back to a pointer. *Confound worth a caption clause:* the curve includes warmup and other recipe choices, so the flat region is "LSR + warmup", not the LSR alone.
5. **Priority:** P2.
6. **Caption draft:** *Goyal et al., Fig. 1: ResNet-50 ImageNet validation error with the learning rate scaled linearly with the batch size, plus gradual warmup. The error is flat up to 8k and rises beyond. Reproduced from arXiv:1706.02677 [with permission].*

### F8 · SVAG on a real network: converges to SGD, or reveals the SDE's failure (paper figure, core reading) (P2)

1. **Anchor:** §3.5, at the end of the **SVAG** paragraph ("If the measured curves for l = 1, 2, 4, … coincide, those observables are consistent with SGD being at its SDE limit."). It is also referenced from §3.7 item 9 ("It can be tested").
2. **What it shows:** Li, Malladi and Arora, Fig. 4. Left: test accuracy against effective steps for SVAG l = 1…32 on PreResNet-32 with BN (CIFAR-10), at a batch size where the LSR holds; all curves coincide. Right: at B = 1024 the SVAG curves converge to a limit *different* from SGD's (l = 1).
3. **Why here:** this is the core reading's key evidence, and the real-network counterpart of the toy in 3.5.B. The toy shows the same thing (the ηλmax ≈ 1.8 curve walks down to the SDE as l grows), and seeing both makes clear what SVAG is for.
4. **Source:** (a) paper figure. Li, Malladi, Arora, arXiv [2102.12470](https://arxiv.org/abs/2102.12470), **Figure 4**. I verified the caption: "SVAG converges quickly and matches SGD (left) or shows the failure of the SDE approximation when LSR breaks (right)… LSR (and thus, the SDE approximation) breaks at B = 1024". **Licence: CC BY-NC-ND 4.0.** Unmodified reproduction with attribution is allowed for non-commercial use. "ND" means the figure should be embedded whole, not cropped or redrawn. CRW should confirm the course counts as non-commercial. Their Fig. 2 (the LSR failure predicted from 𝒩/𝒢) is the alternative, but it is denser and needs their notation; Fig. 4 is the better single choice.
5. **Priority:** P2.
6. **Caption draft:** *Li, Malladi and Arora, Fig. 4: SVAG with l = 1 (plain SGD) to 32 on CIFAR-10, plotted against effective steps. Where the linear scaling rule holds, the curves coincide, so SGD is at its SDE limit (left). At B = 1024 SVAG converges to a different trajectory from SGD, so the SDE picture, and with it the linear scaling rule, has broken (right). Reproduced from arXiv:2102.12470 under CC BY-NC-ND 4.0.*

### F9 · "Compare laws, not paths" (P3)

1. **Anchor:** §3.2, "**In what sense close.** The SGD noise and the Brownian motion are different random objects, so we compare *laws*, not paths."
2. **What it shows:** a 1-D quadratic, h = 1, with deliberately non-Gaussian per-example noise (Rademacher ±s). Top row: 5 SGD paths (visibly jumpy) and 5 order-1 SME paths (Euler–Maruyama with a fine Δt), each with the ±2σ band from the closed form. Bottom: histograms of x at t = 2 from 20k runs of each, nearly identical.
3. **Why here:** "weak approximation" is the conceptual hurdle of §3.2, and the picture makes it concrete: individual paths need not match, but distributions do.
4. **Source:** (b) generate ourselves, as a static PNG or a code-cell output. Under 2 s.
5. **Priority:** P3.
6. **Caption draft:** *SGD with ±1 gradient noise (left) and its order-1 SME (right) have different-looking paths but the same distribution at a fixed time (bottom). Weak approximation compares distributions, not paths.*

### P3 polish to existing code-cell plots (each a line or two)

- **F10 · 3.3.B learning-rate-decay cell** (`B, eta, R = 4, 0.2, 200`): draw horizontal dotted lines at the exact stationary floors before and after the cut (`excess(pi_discrete(eta,B))` and `excess(pi_discrete(eta/10,B))`), so "the target drops immediately; the loss follows" is visible. The log y-axis currently shows a single tick label (10⁻²); add minor tick labels or a linear inset.
- **F11 · 3.4.B Pareto cell:** mark the knee (2S_min, 2E_min) as "B_crit" and draw the asymptotes S = S_min and E = E_min (thin dotted), so the measured plot visibly matches F5(b).
- **F12 · 3.5.B SVAG cell:** the log y-axis renders labels as "6×10⁰, 4×10⁰…". Use `ScalarFormatter` or a linear axis. Optionally overlay the additive-noise discrete theory for SVAG (discrete Lyapunov with step η/l and noise lΣ/B) as thin lines; 3.5.A.3 derives its one-mode version.
- **F13 · 3.6 Adam cell:** the y-label "loss (dashed: η Σ_j √Σ_jj / 4√B)" is awkward. Put the dashed-floor explanation in the legend or the markdown. In the left panel the three dashed floors coincide by design; say so in the caption sentence ("one floor, drawn three times").
- **F14 · 3.3.B reshuffling cell** (table only): optionally plot "reshuffled / with-replacement" against m·η·h_min (4 points, log x). The table is already readable, so this is low value; skip unless F2 is added and a matching visual is wanted.

---

## Gaps and issues noticed

1. **No stored outputs.** Neither `.ipynb` has executed outputs, so every existing plot is invisible until a student runs the cells. This is the main reason F1, F3 and F5 should be static images in markdown rather than new code cells. CRW might also consider committing an executed solutions notebook; that is a build-policy decision.
2. **No image in the opening or at the top of any section.** The first figure a student meets is the weak-error convergence plot in 3.2.B, which is a technical check, not the phenomenon.
3. **Possible small mismatch in the prose after 3.4.B:** "median about 4 at the start, 75–190 late in the run". In my run, the late-step medians were 75 (B=8) and 124 (B=64), and 114 and 189 were the *end-of-run* values. So "75–190" mixes the late-median and end values. The "twenty- to fortyfold" wording depends on the same choice. The review log already notes seed dependence. This is a wording fix for the notebook's author, not a figure issue.
4. **Text-to-figure references all resolve:** the "dashed line" in the learning-rate-decay cell, the "dashed lines" in 3.4.C, "the ηλmax ≈ 1.8 curve in 3.5.B", and the MDL interactive link. The numbers quoted after 3.3.B, 3.4.C and 3.5.B (50%, sixfold, about 10%, about tenfold, factors of 2 and 4) match my run.
5. **Overlap with the MDL interactive:** F3 is a static figure on a similar 2-D basin, but it shows Boltzmann vs not, plus the probability current and the floor's independence of shape. It does not show posterior vs SGD, so it doesn't redo WB2 §8. If F3 is added, the sentence "This notebook does not redo that comparison." still holds. The MDL page's subtitle reads "WB2 §8–6.4" and uses C for the noise covariance; the TD text already bridges the notation.
6. **Licences** (from the arXiv abs pages): McCandlish 1812.06162, Goyal 1706.02677 and Li–Tai–E 1511.06251 are under the arXiv non-exclusive licence (embedding needs permission or a fair-use call). Li–Malladi–Arora 2102.12470 is CC BY-NC-ND 4.0 (embed unmodified, with attribution, non-commercial). The self-generated F1, F3 and F5 avoid this entirely.
7. **Asset location:** there is no `td/assets/figures/` yet. Static PNGs need a home and a regeneration script. The nbkit link check will validate `![](assets/figures/…png)` targets, since its regex matches image links too.
