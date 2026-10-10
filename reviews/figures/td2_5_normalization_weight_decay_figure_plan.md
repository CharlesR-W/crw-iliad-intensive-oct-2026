# Figure plan: TD2.5, normalization and weight decay

- **Notebook:** `td/build_td2_5_normalization_weight_decay.py` (NB2 §5: scale invariance, weight-decay equilibrium, loss spikes, clipping). Read from the builder as of 2026-10-09 (50 cells).
- **Date:** 2026-10-09
- **Proposals:** 10. **P1: 2** · **P2: 5** · **P3: 3**
- **How I checked:** I ran every code cell of the solutions variant in a scratch copy and looked at all five plots it makes (cells "no weight decay: the effective learning rate forgets eta", "Weight-decay equilibrium", `gd_wd_2d`, `adam_full_batch`, `train_classifier`). For each paper figure I name, I fetched the arXiv HTML and confirmed that the figure number and its content match what I say. All five papers carry the default arXiv licence ("perpetual non-exclusive"), not CC-BY. So embedding a paper figure means relying on attributed educational use or asking the authors. Otherwise, link to it ("see Fig. N of …") or redraw it from our own simulation. I mark this on each item.

**What the code cells already plot (all reasonable, all placed right after the exercise they check):**
1. After Ex 5.2.1: effective LR (log-log) for three η values, plus the norm growth.
2. After Ex 5.2.2: RMS angle per step for 5 settings, with $\arccos(1-\eta\lambda)$ dashed, plus $\|w\|$.
3. After Ex 5.3.1: the 2D toy, with loss (log scale), $\eta_{\rm eff}S(u^*)$ against 2, and $\|w\|$ against $r_c$.
4. After the Bai et al. setup: full-batch Adam with $\beta_2=0.999$ vs $0.9$, showing loss and $\eta S_P$ against 38.
5. After Ex 5.3.4: three classifier runs, showing CE, $\log Z$ and the mean logit.

Ex 5.2.4 (code answer), the AdamW cell and Ex 5.4.1 (code answer) print tables only.

**Big caveat:** the generated `.ipynb` files contain no saved outputs. A student who doesn't run the notebook sees none of these plots. For hook figures this argues for a pre-rendered PNG in the markdown (see Gaps).

---

## Proposals, in notebook order

### 1. Pitch: rotational equilibrium in real networks (paper figure) · **P1**

- **Anchor:** title/pitch cell, right after "…it is equivalent to an exponentially *increasing* learning rate with no weight decay." and before "The second half is about things breaking". Alternative: 5.2, end of the paragraph "**Weight decay stops the growth.**", after "…and call the state *rotational equilibrium*."
- **What it shows:** Kosson, Messmer & Jaggi, *Rotational equilibrium* ([arXiv 2305.17212](https://arxiv.org/abs/2305.17212)), **Figure 3**. It plots per-layer weight norms and the average per-step rotation (radians) against training step, one coloured line per layer. Two settings: ResNet-50/ImageNet with SGDM, and weight-standardized GPT-2-124M/OpenWebText with AdamW. The predicted equilibrium rotation (Table 1) is a dashed black line. The lines for scale-invariant layers converge onto it. The RN-50 final FC layer is not scale-invariant and stays below it.
- **Why it helps here:** the pitch's main claim is "each normalized block rotates by about $\sqrt{2\eta\lambda}$ per step, whatever its gradient scale". This is the one picture showing that claim holding in real networks at scale. Everything else in 5.2 is a toy. It also previews the caveat about layers that are not scale-invariant, which the 5.2 bullet list makes in prose.
- **Source:** (a) paper figure. I verified the caption via arXiv HTML: "Measured weight norms and average rotation for different layers … The predicted equilibrium rotation (dashed black) from table 1 holds very well for all scale-invariant layers. The final fully-connected layer in RN-50 (pink) is not scale-invariant… The learning rate is constant…". Licence: arXiv non-exclusive, so embedding needs attribution plus a fair-use judgement or permission. Otherwise link it and use item 4 as the in-notebook version. Confound to state in the caption: GPT-2 is *weight-standardized*, so its layers are exactly scale-invariant by construction. The text already says the result is only "often close" without WS.
- **Caption draft:** *Per-step rotation of each layer's weights in ResNet-50 (SGD with momentum) and a weight-standardized GPT-2 (AdamW), at constant learning rate. Scale-invariant layers settle onto the predicted equilibrium rotation (dashed), whatever their gradient scale. The non-scale-invariant final layer (pink) does not. From Kosson, Messmer & Jaggi (2024), Fig. 3.*

### 2. 5.1, Definition: what a scale-invariant landscape looks like · **P2**

- **Anchor:** 5.1, after "**Definition.** … the loss is a function on the unit sphere." and before "### Exercise 5.1.1 — five identities".
- **What it shows:** one panel of the 2D toy that 5.3.1 uses, $L(x,y)=x^2/(x^2+y^2)=\sin^2\varphi$.
  - Filled contours over $[-1.5,1.5]^2$, where every level set is a ray from the origin.
  - Two dashed circles, $r=0.5$ and $r=1$.
  - Gradient arrows at ~8 points on each circle. They are all tangent to the circles, and the inner ones are twice as long, since $|\nabla L|=|\sin2\varphi|/r$.
  - A thick line marking the minimum set $x=0$ (the $y$-axis).
  - Optional annotation: "curvature $\propto1/r^2$: contours crowd toward the origin".
- **Why it helps here:** it gives a picture for identities 1–3 of Ex 5.1.1 (tangent gradient, $\nabla L\propto1/r$, sharpness $\propto1/r^2$). Later in the notebook, "shrinking the weights makes the loss sharper" (5.3 mechanism 1) becomes something the student can see. It also introduces the 5.3.1 toy early, so the later code cell needs no new mental model.
- **Source:** (b) generate ourselves: matplotlib `contourf` plus `quiver` in closed form, under 1 s. Pre-render as a static PNG in `td/assets/` and embed it in the markdown, so it shows up in the default text before any code runs.
- **Caption draft:** *A scale-invariant loss in two dimensions, $L=x^2/(x^2+y^2)$. Level sets are rays, so the loss sees only the direction of $w$. Gradients are tangent to circles and shrink like $1/\|w\|$. The minimum is the whole $y$-axis.*

### 3. 5.2 opening: one SGD+WD step, and the equilibrium (schematic) · **P1**

- **Anchor:** 5.2, right after the display block below "The two recursions of 5.1.2 are all we need:" (the $r_{k+1}^2$, $\eta_{\rm eff}$ and $\cos\theta_k$ equations) and before "**Without weight decay the norm only grows.**" I place it after Ex 5.1.2 so the picture doesn't hand over the Pythagoras step of 5.1.2.2. If CRW prefers it as a hint, it could also go inside the 5.1.2 `<details>`.
- **What it shows:** two panels, drawn to exaggerated scale (e.g. $\eta\lambda\approx0.15$, labelled "not to scale").
  - **(a) One step.** A circle of radius $r_k$ with the vector $w_k$. A shorter collinear vector $(1-\eta\lambda)w_k$ ("decay: radial, inward"). From its tip, an orthogonal arrow $-\eta\nabla L(w_k)$ of length $\eta G_k/r_k$ ("gradient: tangent"). The result $w_{k+1}$, with a right-angle mark, the angle $\theta_k$ at the origin, and a dashed circle of radius $r_{k+1}$. Annotate $r_{k+1}^2=(1-\eta\lambda)^2r_k^2+\eta^2G_k^2/r_k^2$ and $\tan\theta_k=\eta_{\rm eff}G_k$.
  - **(b) Equilibrium.** About 8 successive iterates $w_k$ on a circle of radius $r_*$, each step drawn as the same inward-then-tangent "staircase". The norm returns to $r_*$ and the vector advances by equal angles $\theta^*$ with $\cos\theta^*=1-\eta\lambda$. Label: "decay pulls in; the orthogonal step pushes out; at $r_*$ they balance, and only rotation is left".
- **Why it helps here:** every result in 5.2 and mechanism 1 of 5.3 follows from this right triangle: norm growth without WD, the equilibrium angle, why only $\eta\lambda$ matters, and the norm-shrinks-until-unstable cycle. The prose describes it ("Geometrically this is Pythagoras") but never draws it. With the picture, $\cos\theta^*=1-\eta\lambda$ reads off at a glance: at equilibrium the hypotenuse equals $r_k$, and the adjacent side is $(1-\eta\lambda)r_k$.
- **Source:** (c) a schematic, which we draw ourselves (matplotlib or SVG, static PNG). Kosson et al. Fig. 2 and Lobacheva et al. ([2106.15739](https://arxiv.org/abs/2106.15739)) Fig. 2 ("centripetal force of weight decay / centrifugal force of the gradient") are published versions of the same idea. Both were confirmed via arXiv HTML. I recommend drawing our own so the notation ($r_k$, $\eta G_k/r_k$, $\theta_k$) matches 5.1.2 exactly, and citing them as "cf."
- **Caption draft:** *(a) One step of SGD with weight decay on a scale-invariant weight. Decay shrinks $w$ radially, and the gradient step is orthogonal to $w$, so the new norm follows from Pythagoras. (b) At equilibrium the two balance: the norm stays at $r_*$ and the weight rotates by $\theta^*=\arccos(1-\eta\lambda)\approx\sqrt{2\eta\lambda}$ per step. Angles exaggerated.*

### 4. Code cell after Ex 5.2.2 ("Weight-decay equilibrium: five settings…"): small improvements, plus a pre-rendered copy · **P2**

- **Anchor:** the existing code cell beginning `# Weight-decay equilibrium: five settings, including two pairs with the same eta*lam.`
- **What to change:**
  1. Right panel: add dashed horizontal lines at each run's predicted $r_*$ (the printed `r_pred`). The prediction is the point of Ex 5.2.2.5, but today it appears only in the table.
  2. Right-panel title: either mark $1/(4\eta\lambda)$ on the axis for one run or drop the timescale claim. As is, the title asserts a timescale the plot doesn't show.
  3. Legend: use $\eta,\lambda,\sigma$ (mathtext) instead of `eta`/`lam`/`sigma`, and group the four $\eta\lambda=10^{-3}$ runs, e.g. "same $\eta\lambda=10^{-3}$" with a shared colour family.
- **Pre-rendered copy:** export the improved figure once as a static PNG for item 1's fallback. If CRW won't embed Kosson Fig. 3, the pitch shows this instead: four runs with different $(\eta,\lambda,\sigma)$ sharing one angle, while their norms differ by 5×.
- **Source:** (d) existing plot, improved, with an optional static export.
- **Caption draft (static version):** *Toy scale-invariant loss under SGD with weight decay. Four runs with different $\eta$, $\lambda$ and noise, but the same $\eta\lambda$, rotate by the same angle per step (dashed: $\arccos(1-\eta\lambda)$). Their norms differ fivefold. A fifth run with 4× the $\eta\lambda$ rotates twice as fast.*

### 5. Code cell after Ex 5.2.1 ("no weight decay: the effective learning rate forgets eta"): reference line · **P3**

- **Anchor:** the code cell containing `ax[1].set(... title="the norm grows like $k^{1/4}$")`.
- **What to change:** add the predicted asymptote $\|w\|\approx(2\eta^2G^2k)^{1/4}$ as dashed lines on the right panel, one per η. The title claims $k^{1/4}$, but no reference line shows it. Also switch the legends to mathtext $\eta$.
- **Why:** the left panel has its reference line and the right one doesn't. Matching them shows the transient length $r_0^4/(2\eta^2G^2)$ from Ex 5.2.1.3 directly, as the point where each curve joins its asymptote.
- **Source:** (d) existing plot.
- **Caption:** none needed (code output).

### 6. Mechanism 1: real periodic spikes (paper figure) · **P2**

- **Anchor:** 5.3, "### Mechanism 1: weight decay drives a normalized layer to the edge of stability", after "…[Lobacheva et al.] document periodic destabilizations in long constant-learning-rate BatchNorm training".
- **What it shows:** Lobacheva, Kodryan, Chirkova, Malinin & Vetrov ([arXiv 2106.15739](https://arxiv.org/abs/2106.15739)), **Figure 1**. Train loss against epoch (~1000 epochs) for a ConvNet on CIFAR-10 and a ResNet-18 on CIFAR-100. SGD, weight decay 0.001, several learning rates. The loss spikes at regular intervals, and the period shortens as LR rises. (Verified via arXiv HTML. Fig. 1 shows train loss only, not norm or effective LR. Their **Figure 4** zooms in on one period of a scale-invariant ConvNet: train loss, effective gradient norm, and the cosine distance between adjacent steps against theoretical bounds, with three phases highlighted. It is the better choice if one figure must show the mechanism rather than the phenomenon.)
- **Why it helps here:** the 2D toy cell below shows the mechanism but in a model nobody trains. This figure shows the same sawtooth in real CIFAR networks. It also makes concrete the text's warning that the cycle needs long constant-LR runs with little noise.
- **Source:** (a) paper figure. Licence is arXiv non-exclusive, so embed with attribution or link. Confound to state: Fig. 1 trains *all* weights, including non-scale-invariant ones, and the effect is still there. Fig. 3/4 are the fully scale-invariant version.
- **Caption draft:** *Training loss of a ConvNet (CIFAR-10) and a ResNet-18 (CIFAR-100) under SGD with weight decay at constant learning rate. The loss destabilizes and recovers periodically, and the period shortens as the learning rate grows. From Lobacheva et al. (2021), Fig. 1.*

### 7. Code cell after Ex 5.3.1 (`gd_wd_2d`): zoom panel, predicted crossing, and a trajectory view · **P2**

- **Anchor:** the code cell `def gd_wd_2d(eta, lam, T, noise=0.0, …)` and the markdown after it ("…the cycle then repeats every $\approx130$ steps").
- **What to change:**
  1. Add a vertical dashed line at the predicted first crossing, step ≈1260 (Ex 5.3.1.2), on all three panels. The text quotes 1260 against 1340, but the plot doesn't mark either.
  2. Add a fourth panel that zooms into steps ~1250–1800 with $\|w\|$ and $\eta_{\rm eff}S$ on twin axes. At the 6000-step scale the ≈130-step cycle is a dense comb, so the sawtooth the text describes (slow decay → crossing → spike → norm jump) can't be read.
  3. Optionally, a fifth small panel with the trajectory in the $(x,y)$ plane: the circle $r=r_c$ dashed, $w_k$ sliding down the $y$-axis toward the origin, then zig-zagging in $x$ and being kicked back out. This ties it to item 2's landscape.
- **Why:** the text relies on reading the period and the delay off the plot. At present the student can only read them from the numbers in the prose.
- **Source:** (d) existing plot. All extra data are already computed (`quiet`). The trajectory panel needs `w` recorded, which is one more column in `rec`.
- **Caption:** code output. The markdown after the cell could add "the right panel zooms into three cycles".

### 8. Mechanisms 3–4: attention-logit growth and output-logit divergence (paper figures) · **P2**

- **Anchor:** 5.3, "### Mechanism 3: attention-logit growth", after "…and remove it with QK-layernorm." This mechanism has no demo, so here the figure is the only visual evidence.
- **What it shows:** Wortsman et al., *Small-scale proxies* ([arXiv 2309.14322](https://arxiv.org/abs/2309.14322)), **Figure 2**. Max attention logit against training step for two model sizes about three orders of magnitude apart, with and without qk-layernorm. Without it the logits grow and the loss diverges; with it they stay bounded. (Verified via arXiv HTML.)
- **Optional second panel (P3 on its own):** **Figure 4** of the same paper, placed under "### Mechanism 4: output-logit divergence". It shows output logits drifting very negative over training in a 2.4M-parameter model at LR 0.1. It shows the divergence that our classifier cell explicitly does not ("This cell shows the drift and the pinning, **not** the divergence").
- **Why it helps here:** the summary table marks mechanism 3 "no demo". One real plot lets a student see what "attention-logit growth" means quantitatively (logit scale, timescale) without running a transformer. The core reading is this paper, so the figure doubles as an advert for it.
- **Source:** (a) paper figure. Licence is arXiv non-exclusive, so embed with attribution or link. A self-generated version would need a small transformer trained at high LR, which exceeds the 30 s CPU budget, so the paper figure is the right choice.
- **Caption draft:** *Maximum attention logit during training of small transformers at high learning rate. Without QK-layernorm the logits grow until training diverges. With it they stay bounded. From Wortsman et al. (2024), Fig. 2.*

### 9. 5.4: curvature grows with the gradient norm (paper figure) · **P2**

- **Anchor:** 5.4, after "They report the correlation between gradient norm and local smoothness clearly in LSTM language models; for ResNet-20 on CIFAR-10 it appears with clipping and a large learning rate but not under the default training procedure."
- **What it shows:** Zhang, He, Sra & Jadbabaie ([arXiv 1905.11881](https://arxiv.org/abs/1905.11881)), **Figure 1**. A log-log scatter of gradient norm (x) against estimated local smoothness (y) along the training trajectory of AWD-LSTM on PTB, coloured by iteration. A clear positive, roughly linear trend. (Verified via arXiv HTML: "Gradient norm vs local gradient Lipschitz constant on a log-scale along the training trajectory for AWD-LSTM on PTB dataset.") For the contrast the text draws, **Figure 3(a)** is the ResNet-20 SGD-momentum panel, which shows a much weaker relation.
- **Why it helps here:** $(L_0,L_1)$-smoothness is an abstract inequality, and this scatter is its empirical motivation. The whole clipping argument rests on it. Showing it next to Fig. 3(a) puts the text's "LSTM yes, default ResNet no" in front of the student.
- **Source:** (a) paper figure. Licence is arXiv non-exclusive, so embed with attribution or link.
- **Caption draft:** *Local smoothness against gradient norm along an LSTM language-model training run (log-log, colour = iteration). Curvature grows roughly in proportion to the gradient norm, which is the $(L_0,L_1)$ condition. A fixed step size that is safe where gradients are large is then too small everywhere else. From Zhang et al. (2020), Fig. 1.*

### 10. Ex 5.4.1 code answer: plot the step counts · **P3**

- **Anchor:** the `code_answer` beginning `def clipped_gd_steps(x0, eta=1.0, gamma=1.0, tol=1e-8):`, and the markdown after it ("The best fixed step on the grid sits just below $2x_0/\sinh x_0$…").
- **What it shows:** steps to reach tolerance against $x_0$ on a semilog-y axis. Three series: clipped GD (≈ $x_0+26$, nearly flat), best fixed step on the grid, and the lower bound $\log(10^8)/(-\log(1-x_0/\sinh x_0))$. The bound is dashed and nearly coincides with the fixed-step series.
- **Why:** the table already holds the numbers. On a semilog plot, "linear against exponential" becomes two curves with visibly different shapes, which is the whole claim of the exercise. The plot costs four lines.
- **Source:** (d) add to the existing solution cell after the table. Code output only.
- **Caption:** none needed. The following markdown could say "(plotted above)".

---

## Considered and not proposed

- **Li & Arora exponential-LR curves** (their Fig. 2, step decay vs tapered-exponential schedule, PreResNet32). The code exercise already shows equality to roundoff, which is a stronger statement than matching test curves.
- **A plot for Ex 5.2.4(a)** (norm ratio growing like $(1-\eta\lambda)^{-k}$ while the direction difference sits at $10^{-15}$). The printout makes the point.
- **Bai et al. Fig. 4 (sin fit with Adam).** The `adam_full_batch` cell reproduces it, and its plot is clear and correctly described. It could be pre-rendered as a static image at the top of Mechanism 2 if CRW wants every mechanism to show its phenomenon without running code (would be a P3).
- **An AdamW equilibrium plot** for the AdamW table cell. The table, with the radial-cosine column, is the right format here.
- **The interactive EoS figure** (`assets/interactive/eos_toy_and_catapult.html`). The "What you need" bullet on the edge of stability could link to it with `figure()` for students who skipped NB2 §2. This is a placement suggestion, not a new figure.

## Gaps and issues noticed

1. **No saved outputs.** Neither generated notebook stores outputs (`nbkit_td` writes `outputs: []`), so all five code-cell plots are invisible until the student runs the cells. That's fine for exercises, but for anything acting as a hook (items 1/4, 6, 8, 9) the image needs to be a static PNG in `td/assets/`, embedded in markdown. *CRW to decide:* either (i) pre-render hook images as PNGs, or (ii) have `nbkit_td` optionally save executed outputs in the solutions variant. `_validate`'s local-link regex would also check `![](assets/…png)` targets, which is good.
2. **Paper-figure licensing.** All four papers whose figures I propose (Kosson, Lobacheva, Wortsman, Zhang) are under arXiv's non-exclusive licence, not CC-BY. *CRW to decide:* embed with attribution as educational use, or link only ("see Fig. 3 of …") and rely on our own figures where they exist (item 4 stands in for item 1; the toy cell partly stands in for item 6). Items 8 and 9 have no self-generated substitute within the CPU budget.
3. **Plot titles that claim more than the plot shows:**
   - "the norm grows like $k^{1/4}$" has no reference line (item 5).
   - "norm relaxes on a $1/(4\eta\lambda)$ timescale" has no timescale marked (item 4).
   - The predicted $r_*$ is computed but not plotted (item 4).
4. **The 5.3.1 toy text cites numbers the plot can't show:** first crossing ≈1260 vs loss rising at ≈1340, and a period ≈130. At a 6000-step x-axis these aren't legible (item 7).
5. **Cosmetic:** legends use ASCII `eta`, `lam`, `sigma` where the notebook's notation is $\eta,\lambda,\sigma$ (cells after Ex 5.2.1 and Ex 5.2.2).
6. **Consistency with the prose is otherwise good.** I checked each plot's description against the actual output:
   - After Ex 5.2.2: "four of the five settings share $\eta\lambda=10^{-3}$… same angle" matches.
   - Stationarity identity to about 0.2%: matches.
   - Adam $\eta S_P$ "overshoots once early (to about 45), then sits a little below 38": matches (≈44.5, then ≈35–37).
   - Final loss "near $10^{-3}$ against about $2\times10^{-5}$": matches.
   - Classifier: SGD's mean logit stays at $-\log 32\approx-3.47$, and with z-loss "$\log Z$ about $-0.5$": matches ($-0.54$).
   - No text refers to a figure that doesn't exist.
