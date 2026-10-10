# Figure plan: td2_7_learned_first (NB2 §7, "What is learned first")

- Notebook: `td/build_td2_7_learned_first.py` (read 2026-10-09; 38 cells), generated `td/td2_7_learned_first_{exercises,solutions}.ipynb`. The committed notebooks have no outputs, so I ran every solutions code cell myself (Agg backend) and looked at each figure.
- Date: 2026-10-09
- Proposals: **11** in total: **P1: 4** (F1, F3, F8, and the F7 demo fix), **P2: 5** (F4, F5, F6, F9, F10), **P3: 2** (F2, F11).
- I checked every paper figure cited below in the arXiv PDF: I rendered the page and read the caption. All five papers (Refinetti, Bardone–Goldt, Rahaman, Basri, Xu) are under the default **arXiv non-exclusive distribution licence**, not CC. Embedding their figures in course notebooks therefore needs either CRW's fair-use judgement or the authors' permission. For that reason I give a self-generated recipe wherever one is cheap, and keep the paper figure as a link.
- `td/assets/interactive/mode_explorer.html` is used only in NB1 (`build_td1_linear_networks.py`). td2_7 doesn't link it (see F2).

---

## Opening (before §7.1)

### F1 · P1 · Distributional simplicity bias: the clone experiment
- **Anchor:** opening markdown, right after "...matches the real-data network's test accuracy for its first few dozen SGD steps (Refinetti et al.)." and before "This section works out three results...".
- **What it shows:** test accuracy on real CIFAR-10 (y) against SGD steps (x, log scale) for a ResNet18 trained on CIFAR-10 and on three clones: a Gaussian mixture (first two moments per class), WGAN samples, and cifar5m. Each clone curve tracks the real-data curve for a while, then peels off. A Gaussian-mixture clone peels off first (about 50 steps), cifar5m last (about 2000 steps).
- **Why it helps:** the whole notebook hangs on this one observation, which the pitch currently states in one sentence. Seeing the curves peel off in order of how much structure each clone keeps tells the student what "learned first" means operationally before §7.1 derives it.
- **Source:** (a) Refinetti, Ingrosso, Goldt, [arXiv:2211.11567](https://arxiv.org/abs/2211.11567), **Fig. 1** (p. 2). Verified: the left panel is "Resnet18 test accuracy on CIFAR10" with Gaussian / WGAN / cifar5m / CIFAR10 curves, and the right panel shows sample images from each clone. Only the left panel is needed. The licence is arXiv non-exclusive, so this is CRW's call. **Fallback (b):** reproduce a small version offline (not in a notebook cell): a 4-layer CNN on CIFAR-10 against a per-class Gaussian clone (class means plus full 3072×3072 covariances, sampled with a Cholesky factor). Two curves, about 3k SGD steps. That is minutes on a CPU, or one Modal call. It shows the Gaussian curve peeling off and drops the WGAN/cifar5m rungs.
- **Caption:** *A ResNet18 trained on Gaussian-mixture "clones" of CIFAR-10, which keep only each class's mean and covariance, has the same CIFAR-10 test accuracy as one trained on real images for about the first 50 SGD steps. Clones that keep more structure track it for longer. (Refinetti, Ingrosso and Goldt 2023, Fig. 1.)*

### F2 · P3 · Link the NB1 mode explorer, or add a small static of the recap
- **Anchor:** the "What you need" box, the bullet "A *two-layer* linear network ... This is a sigmoid that rises at $t\approx\frac1{2s_\alpha}\ln\frac{s_\alpha}{a_\alpha(0)}$, so stronger modes are learned first."
- **What it shows:** two small panels with the closed forms for $s=3,2,1$ and $a_0=10^{-3}$. Left, shallow: $a=s(1-e^{-t})$, so all modes rise together. Right, two-layer: $a(t)=\dfrac{s\,e^{2st}}{e^{2st}-1+s/a_0}$, staggered sigmoids with plateaus.
- **Why it helps:** this is the canonical Saxe stage-wise picture. NB2 sections must stand alone, and the box states it only in words. The cheapest alternative is a single `figure("assets/interactive/mode_explorer.html", ...)` line, which reuses the existing interactive and doesn't redraw it.
- **Source:** (b) closed form, a pre-rendered static PNG (<1 s); or (d) a placement-only link to the existing interactive.
- **Caption:** *Shallow linear model versus two-layer linear network, from the same small initialization. In the shallow model every mode relaxes at the same rate. In the deep one, stronger modes (larger $s$) are learned earlier, after plateaus.*

---

## §7.1 Two moments are all a linear-in-input model sees

### F3 · P1 · The test bed: what distinguishes the two classes along $m$, $u$, $v$
- **Anchor:** §7.1.2, after "Its mean and variance along $v$ are those of class $-1$, but its distribution there is bimodal." (The same figure serves §7.2.2. Reference it there instead of repeating it.)
- **What it shows:** three side-by-side 1-D density panels, each comparing class $-1$ (grey, $\mathcal N(0,1)$) with class $+1$ (colour).
  - Along $m$: $\mathcal N(1,1)$. Annotate "mean differs ($\kappa_1$)".
  - Along $u$: $\mathcal N(0,1+\beta_u)=\mathcal N(0,6)$. Annotate "variance differs ($\kappa_2$)".
  - Along $v$: the mixture $\tfrac12\mathcal N(\pm\sqrt{10/11},\,1/11)$, with the Gaussian clone ($=\mathcal N(0,1)$) dashed on top. Annotate "same mean and variance; $\kappa_4=-200/121\approx-1.65$".
  - Optional fourth panel: class $+1$ samples in the $(u,v)$ plane for independent versus correlated ($\nu=\mathrm{sign}\lambda$) latents. The correlated cloud is visibly tilted, which is the cross-covariance $c\approx1.70$ of 7.2.A.3.
- **Why it helps:** every demo in §7.1–7.2 and all the Bayes levels (0.691 / 0.767 / 0.819) refer to this model, which exists only as prose and an equation. One look at the $v$ panel makes "invisible to two moments, visible to the fourth cumulant" concrete. The clone overlapping the grey curve exactly is the point.
- **Source:** (b) closed-form densities (plus 2k samples for the optional panel), a pre-rendered static PNG, <1 s. Bardone–Goldt have no equivalent panel.
- **Caption:** *The mixed cumulant test bed ($\beta_m=1,\beta_u=5,\beta_v=10$). Along $m$ the classes differ in mean, along $u$ in variance. Along $v$, class $+1$ is bimodal but has the same mean and variance as class $-1$, so its Gaussian clone (dashed) can't be told apart from class $-1$ in that direction.*

### F4 · P2 · A perceptron's decision boundary rotates: mean difference → Fisher → higher-order correction
- **Anchor:** §7.1.3, the paragraph beginning "[Refinetti et al.](...) (§2) take a sigmoidal perceptron with square loss on a two-class toy dataset." Place it after "...picks up a correction from the within-class fourth cumulant."
- **What it shows:** two panels.
  - (A) 2-D scatter of Refinetti's two-rectangle dataset, with the perceptron's decision boundary at about four times (colour by time), plus the naive classifier $\propto\mu_+-\mu_-$ (dotted) and Fisher's discriminant $\propto C_w^{-1}(\mu_+-\mu_-)$ (dashed).
  - (B) $1-\cos\angle(w_t,\cdot)$ against time (log-log) for the naive, Fisher and "correction" directions. Each curve dips in turn.
- **Why it helps:** the text narrates a three-stage rotation, and panel B shows it as three successive dips. This is the only place in the notebook where the staged order in time is *derived*, not observed, and a student should see it.
- **Source:** (a) Refinetti et al. **Fig. 2** (p. 4), verified: panel A shows the boundaries with the naive / linear-discriminant / oracle lines, B the test error, C $1-\cos\theta$ to the naive, linear-discriminant and correction classifiers over time. arXiv licence. **Preferred (b):** regenerate on CPU. Two uniform rectangles in $(x_1,x_2)$ as in their Fig. 2A (plus a few $\mathcal N(0,1)$ nuisance dimensions, or none), a $\mathrm{erf}$/sigmoid perceptron, square loss, small init, full-batch GD for about $10^5$ steps, logged on a geometric grid. About 5 s. Can be a short code cell after §7.1.3, or a static PNG. The fourth-cumulant "correction" direction needs their eq. 12. If that's too fiddly, plot only the naive and Fisher dips plus the Bayes/oracle direction, and say so in the caption.
- **Caption:** *A sigmoidal perceptron trained from small weights first aligns with the difference of class means, then rotates to Fisher's discriminant (which uses the class covariances), and only later uses higher-order structure. After Refinetti et al. (2023), Fig. 2.*

### (d) Existing plot: GELU data vs clone (code cell after 7.1.B, "A small-init GELU network trained by full-batch GD...")
It is well placed, and the prose matches it: two-stage staircase, curves coincide, the gap opens with the RMS pre-activation. Small improvements, P3, folded into F11:
- The right panel has no y-label (it mixes a relative gap and an RMS; label it "relative gap / RMS pre-activation" or use the legend only).
- In the left panel, add a light vertical band or text tags "mean" (≈ step 50) and "covariance along $u$" (≈ step 200), so the two stages the prose names are visible on the plot.

---

## §7.2 Higher cumulants

### F5 · P2 · Search phase, then descent: overlap trajectories for $k=1,2,3$
- **Anchor:** §7.2.3, right after the samples table ("| $\ge3$ | $d^{\,k-1}$ (up to logarithms) |") and before "Almost all of these samples go into this 'search phase'...".
- **What it shows:** single-run trajectories of the overlap $\alpha_t$ (for even $k$, $|\alpha_t|$) against online-SGD step on a log axis, for $k=1,2,3$ at $d=64$, about 5 runs each in three colours, with a dotted line at $d^{-1/2}$. $k=1$ leaves immediately. $k=2$ sits for about $10^2$ steps and then shoots up. $k=3$ sits near $d^{-1/2}$ for $10^3$–$10^4$ steps, and some runs haven't escaped by $4\times10^4$. Optionally overlay the drift ODE $\dot\alpha=\frac{\delta k}{d}\alpha^{k-1}(1-\alpha^2)$ dashed.
- **Why it helps:** the table gives numbers, and this picture gives the *shape*: a flat plateau whose length explodes with $k$, followed by a fast drop. That is the "grokking-type delay" of this notebook, and it is what the plateau in the MCM demo (F7) is made of. 7.2.C only plots the medians.
- **Source:** (b) reuse 7.2.C's `escape_times` loop, but record $\alpha_t$. I prototyped it: 6 runs × 3 values of $k$ × $4\times10^4$ steps at $d=64$, with 7.2.C's step sizes ($1$, $1/\ln d$, $d^{-1/2}/\ln d$), took ≈33 s on a loaded laptop, and the picture comes out as described. Pre-render a **static PNG**: 7.2.C is a code exercise, so the student shouldn't need its solution to see this.
- **Caption:** *Online SGD on a single neuron with information exponent $k$ ($d=64$). The overlap with the hidden direction drifts near its random starting value $d^{-1/2}$ for a time that grows steeply with $k$, then rises to order 1 quickly. Nearly all the samples are spent in the search phase.*

### (d) Existing plot: 7.2.C escape times against $d$ (code-answer cell)
Correct and readable. P3 improvements (in F11): overlay each law ($d$, $d\ln^2 d$, $d^2\ln d$), scaled to the $d=32$ point, as dashed lines, so the "ratio roughly constant" claim in the following prose can be seen. Set the x-ticks to 16, 32, 64 (matplotlib currently prints "2×10¹").

### F6 · P2 · The population loss near the equator, with and without correlated latents
- **Anchor:** §7.2.4, after the display $L\approx L_0-c_{20}\alpha_u^2-c_{11}\alpha_u\alpha_v-c_{04}\alpha_v^4$ and the sentence "So $u$ has exponent 2 and $v$ has exponent 4." (An alternative place is the end of the 7.2.D solution, but the main text is better, because the table that follows is what it illustrates.)
- **What it shows:** two contour plots of $L(\alpha_u,\alpha_v)$ on $[-0.3,0.3]^2$, using ReLU coefficients from 7.2.D ($\sigma_2=-\sigma_4=1/\sqrt{2\pi}$, $\beta_u=5$, $\beta_v=10$). Left: $c_{11}=0$. Right: $c_{11}=\sigma_2 c/2$ with $c=1.70$. Overlay gradient-flow streamlines from a ring of starts at radius $d^{-1/2}$ (say $d=100$). Left, the flow leaves along $\alpha_u$ and the $\alpha_v$ direction is flat (quartic). Right, the unstable direction tilts to $(\lambda_+,c_{11})$, so $\alpha_v$ grows with $\alpha_u$. Mark the eigenvector.
- **Why it helps:** "Sliding down the stairs" is a statement about the geometry of this saddle, and 7.2.D.4 derives it algebraically. A picture of a flat valley floor versus a tilted one makes the mechanism immediate. It is also the notebook's only loss-landscape picture.
- **Source:** (b) closed form plus `plt.streamplot`, <1 s. A static PNG, or a code cell after 7.2.D.
- **Caption:** *The single-neuron population loss on the mixed cumulant model near the uninformative equator. With independent latents (left) the loss is quadratic in $\alpha_u$ but only quartic in $\alpha_v$, so gradient flow finds $u$ and leaves $v$ near zero. Correlating the latents (right) tilts the unstable direction, so $v$ is pulled along with $u$.*

### F7 · P1 · The staircase itself: fix the MCM demo so its stages are visible (and/or link Bardone–Goldt Fig. 1)
- **Anchor:** the code cell "# Two-layer ReLU net on the mixed cumulant model (d = 16), online minibatch SGD." and its prose ("...the mean and covariance stages are both passed within the first ~20 steps, too fast to separate on the plot."). The text of §7.2.4 also promises it: "Their two-layer ReLU networks show the same staircase: test error drops as the mean, then the covariance, then the cumulant direction are found".
- **What it should show:** the paper's canonical picture. Test accuracy against log step for the network evaluated on four test sets:
  - full data,
  - mean only ($\beta_u=\beta_v=0$),
  - **mean + covariance ($\beta_v=0$)**, which is currently missing,
  - the Gaussian clone,
  
  shown for independent latents (left) and correlated latents (right). There should be three visible rungs at the Bayes levels 0.691 / 0.767 / 0.819, labelled inline as in the GELU plot.
- **Why it helps:** this is the plot the topic is anchored around. The current run shows the final plateau and the jump in $v$-overlap well, but its own prose concedes that the first two stages are invisible. Without the mean+cov test set the student can't see the covariance stage at all.
- **Source:** (d) improve the existing cell, plus (a) a link to Bardone & Goldt, [arXiv:2404.08602](https://arxiv.org/abs/2404.08602), **Fig. 1A and 1C** (verified via arXiv HTML: A is test error on the MCM with test sets "mean only", "mean+cov", "Gaussian equiv." and "full data" for independent latents; C is the same with correlated latents; $d=128$, 512 ReLU units). Concrete changes:
  1. Add the `beta_v=0` test set.
  2. Label the dotted Bayes lines inline ("means", "+ covariances", "all").
  3. Move the $v$-overlap to a twin y-axis or a thin lower panel. It currently shares the axis labelled "test accuracy".
  4. Separate the early stages with a smaller first-layer step, or by starting from smaller weights (scale `W` by ~0.1). The paper's own runs use batch 1 and a 100× smaller second-layer step.
  
  **What I tested** (scratch run, the notebook's own `mcm_run`, independent latents, $d=16$, batch 32, 40k steps, lr ∈ {0.05, 0.02, 0.01}): a smaller lr removes the noise and gives a clean long plateau at ≈0.76, where full data and clone coincide. At lr 0.05 the $v$-overlap jumps at ≈1.5×10⁴ steps and full-data accuracy rises to ≈0.81 while the clone's falls, a much cleaner version of the current plot. At 0.02 and 0.01 the jump comes at or after 4×10⁴ steps. **But the mean and covariance rungs still merge** into one smooth rise over steps 10–300. Only the "mean only" curve peeling off at ≈0.66 around step 100 hints at them. Each run took 70–90 s on this loaded machine, too slow for a code cell. So the recommendation is: (1) apply changes 1–3 to the existing cell now; (2) for the canonical three-rung picture, either link or embed Bardone–Goldt Fig. 1A/C, or pre-render a static offline run. Smaller initial weights and/or a smaller $\beta_u$ (so the covariance stage is slower) are the knobs to try; I haven't tested them.
- **Caption:** *A two-layer ReLU network trained by online SGD on the mixed cumulant model, tested on data that keep only the mean signal, the mean and covariance signals, or everything. With independent latents (left) the network climbs a staircase: mean, then covariance, then a long plateau before it finds the fourth-cumulant direction $v$. Correlating the latents (right) removes the last plateau.*

---

## §7.3 Spectral bias

### F8 · P1 · Spectral bias in a real network: frequency × iteration heatmap plus snapshots
- **Anchor:** top of §7.3, after the "**The result.**" paragraph ("...This is the *spectral bias*, or *frequency principle*, observed in real networks ([Rahaman et al.]...; [Xu et al.]...).") and before §7.3.1.
- **What it shows:**
  - (left) A heatmap of $|\hat f_\ell(t)|/A_\ell$ (the network's Fourier amplitude at frequency $\ell$, normalised by the target's), with frequency $\ell$ on the x-axis and training iteration on the y-axis (log), clipped to $[0,1]$. Target $\sum_{\ell\in\{1,2,\dots,10\}}A_\ell\sin(\ell\vartheta+\phi_\ell)$ with **amplitudes increasing in $\ell$** (Rahaman's panel b), so "big components first" is ruled out.
  - (right) 3–4 snapshots of the fitted function over the target.
  
  The bright region should advance from left to right.
- **Why it helps:** §7.3 is anchored on this phenomenon in *trained networks*. The notebook's existing figures are all of the frozen-kernel explanation (cells after 7.3.B and 7.3.C), and the only network run is in the lazy/Basri setting. The student should see the phenomenon before deriving the kernel account of it. Using increasing amplitudes removes the obvious confound.
- **Source:** (a) Rahaman et al., [arXiv:1806.08734](https://arxiv.org/abs/1806.08734), **Fig. 1(a,b)** (p. 4, verified: heatmap of normalised spectrum against iteration, (a) equal and (b) increasing amplitudes; 6-layer, 256-wide ReLU, frequencies 5–50, full-batch Adam) and **Fig. 2** (snapshots at iterations 100 / 1k / 10k / 80k). Alternatives are Basri et al. [arXiv:1906.00425](https://arxiv.org/abs/1906.00425) **Fig. 2** (p. 2, verified: k = 4 and 14 superposition, snapshots at epochs 0 / 50 / 500 / 22452) and Xu et al. [arXiv:1901.06523](https://arxiv.org/abs/1901.06523) **Fig. 1** (relative error at three frequencies against epoch). All are arXiv-licensed. **Preferred (b):** regenerate. A 4-layer ReLU MLP (width 128) on input $(\cos\vartheta,\sin\vartheta)$, 200 equispaced training angles, full-batch Adam with lr 1e-3 for about 5000 steps, logging the rFFT of $f$ on a 512-point grid every 50 steps. About 10–20 s on CPU. Pre-render a **static PNG** so the student sees it without running anything.
- **Caption:** *A ReLU network fitting a sum of sinusoids whose amplitudes grow with frequency. Colour shows how much of each frequency has been fitted, as training proceeds upward. Low frequencies are fitted first even though the high ones are larger. (After Rahaman et al. 2019, Fig. 1b.)*

### (d) Existing plot: kernel gradient flow on the circle (code cell after 7.3.B, "Kernel gradient flow on the circle")
Good and well described. The residual curves sit on the dotted $e^{-\hat k_\ell t}$ predictions, and the snapshot panel shows low frequencies first. One relabel: the legend says "frequency 1/4/8" where the text uses $\ell$. Use "$\ell=1$" for consistency with F8 and the 7.3.C plots.

### F9 · P2 · A bias-free ReLU network cannot represent $\cos 3\vartheta$
- **Anchor:** §7.3.2, the bullet "**The NTK of a two-layer ReLU network** ... So a bias-free shallow ReLU network cannot fit $\cos3\vartheta$ at all."
- **What it shows:** two panels. Left: 51 equispaced samples of $\cos3\vartheta$ (dots) and the $t\to\infty$ kernel interpolant (7.3.A.2) with the bias-free $K^\infty$. It passes through every point using only even frequencies. On the grid, $\cos3\vartheta$ aliases to the even frequencies $48=51-3$ and $54=51+3$. I checked this: the interpolant has no $\ell=3$ component at all, its spectrum is dominated by $\ell=48$ and $54$, the training error is about $10^{-13}$, and the maximum off-grid error is 2.0. Right: the same with $K_{\rm bias}$, which gives a clean $\cos 3\vartheta$.
- **Why it helps:** "zero odd coefficients" stays abstract until you see the network fit every training point with the wrong function. It is also a concrete picture of 7.1.B.3's parity argument, which the text explicitly links.
- **Source:** (a) Basri et al. **Fig. 3** (p. 5, verified: a bias-free two-layer network with 2000 units on 51 points of $\cos3\theta$ fits them "with just even frequencies, yielding poor interpolation"; the right panel fits $\cos 4\theta$). **Preferred (b):** closed form with the notebook's own `K_nobias`/`K_bias` and the 7.3.A.2 formula, <1 s. Either a short code cell after the 7.3.C solution, or a static in §7.3.2.
- **Caption:** *Kernel interpolation of 51 samples of $\cos3\vartheta$ with the infinite-width NTK of a two-layer ReLU network. Without biases (left) the kernel has no odd frequencies above 1, so it interpolates with high even frequencies. Adding biases (right) restores the odd frequencies.*

### F10 · P2 · Fix the two 7.3.C / Basri-network plots so their headline result is visible
- **Anchors:**
  - (i) the 7.3.C solution cell's plot "two-layer ReLU NTK on the circle";
  - (ii) the code cell "# A two-layer ReLU network in Basri et al.'s setting...".
- **Problems seen in the actual output:**
  - (i) The odd no-bias coefficients are clipped to 1e-12, below `ylim(1e-5, 1)`, so they simply don't appear. The point the prose makes ("the odd coefficients vanish without a bias") is invisible. Draw them as ×-markers on a floor line labelled "$=0$ (odd $\ell\ge3$)".
  - (ii) Right panel: the scale-1 network never reaches the $1/e$ threshold at $\ell=8$ (`nan`), so that point is silently dropped. That is the headline ("frequency 8 is still at about half its initial amplitude"). Draw an upward arrow or open triangle at the 2000-step ceiling, labelled "not reached". Use x-ticks 1, 4, 8 instead of "2×10⁰ ... 6×10⁰". In the left panel, the title "dotted: 7.3.C" is cryptic; use "dotted: infinite-width kernel prediction".
  - (ii, optional third panel) the prose's main claim, "Feature learning has shrunk the kernel exactly on the slowest mode", is supported only by a printed Rayleigh quotient. Record $\eta\,\langle\sin\ell\vartheta,K_t\sin\ell\vartheta\rangle/\|\sin\ell\vartheta\|^2$ every 50 steps for $\ell=1,4,8$ at both scales, and plot it against step (log y). At scale 1 the $\ell=8$ curve falls about 5×; at scale 10 it stays flat. Cost: 40 extra $128\times128$ eNTK builds per run, negligible.
- **Why it helps:** both plots currently hide the one data point the surrounding text is about.
- **Source:** (d) improve existing code-cell plots.
- **Caption (for the new panel):** *The empirical NTK's strength on each Fourier mode during training. In the lazy run (scale 10) it stays near its initial value. At scale 1 it falls about five-fold on $\sin 8\vartheta$, which is why that mode stalls.*

### F11 · P3 · Small label fixes bundled
Covers the existing-plot notes above: the GELU right-panel y-label and stage tags, the 7.2.C law overlays and ticks, and "$\ell$" in the kernel-flow legend. Each is a one- or two-line edit to an existing cell.

---

## Gaps / issues noticed

1. **"Grokking-type delays" isn't in this notebook.** The dispatch title lists it, but the builder covers moments, cumulants and spectral bias only. Grokking appears in NB1 (Kumar et al. reading; the lazy-to-rich mechanism). The closest analogue here is the online-SGD search plateau (F5, F7). I didn't propose a grokking figure. If CRW wants one in §7, that is new content, not a figure.
2. **The MCM demo's prose admits the plot doesn't show what §7.2.4 promises** (the first two stages are "too fast to separate"). F7 is the fix.
3. **Mixed y-axis in the MCM demo:** the $\max_j|\cos(w_j,v)|$ overlap is drawn on an axis labelled "test accuracy", and the Bayes lines aren't labelled on the plot.
4. **Invisible data in two plots** (F10): the no-bias odd zeros in 7.3.C, and the scale-1 $\ell=8$ point in the Basri-network plot.
5. **Runtime observation (may just be machine load):** on this laptop, which other agents were also using, the GELU data-vs-clone cell took 136 s, 7.2.C 54 s, the MCM demo 40 s and the Basri-network cell 38 s. The review log reports about 32 s total. This is probably contention, but it's worth one re-time on an idle machine, since the budget is about 30 s per cell.
6. **No text refers to a figure that doesn't exist.** The prose descriptions of the existing plots match the outputs I generated: two-stage GELU staircase; slopes 1.09 / 1.58 / 2.39; MCM final ≈0.79 with the clone falling; Basri scale-1 $\ell=8$ ≈0.54 at the end; lazy fit times 2 / 81 / 374 against a prediction of 2 / 79 / 345.
7. **Licensing:** all candidate paper figures are arXiv non-exclusive. CRW should decide whether to embed (F1 especially, which has no cheap self-generated equivalent at CIFAR scale) or to link and regenerate.
