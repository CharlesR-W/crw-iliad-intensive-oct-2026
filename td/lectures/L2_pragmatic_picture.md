# L2: The pragmatic picture (outline and speaker notes)

**Length:** about 28 min. Segments marked *compressible* can go down to one sentence, bringing it to about 22 min.
**Hands off to:** the NB2 notebooks `td2_1` to `td2_8`.
**Purpose:** organize NB2 around the knobs a practitioner turns: learning rate, batch size, schedule, optimizer, width, normalization and weight decay. For each knob, name the one or two derivations behind it and state their result, then point to the notebook that does the derivation. **Nothing is derived on the board.** Every result below is an exercise with a worked solution in the notebook named.

**Notation on the board (matches NB2).** Loss $L$, gradient $G=\nabla L$ (written $g$ in `td2_1` and `td2_4`, and $\nabla L$ in `td2_6`), Hessian $H$, sharpness $S=\lambda_{\max}(H)$, step $\eta$, batch size $B$, per-example gradient covariance $\Sigma$. **Time is $t=k\eta$**, so gradient flow is $\dot\theta=-\nabla L$. The one exception is Adam's SDE in `td2_3` §3.6, where one step is time $\eta^2$.

**Symbol clashes to warn about once, early.**

- $\Sigma$ is the target correlation in NB1/L1 and the gradient-noise covariance in NB2.
- $\lambda$ is a Hessian eigenvalue in `td2_1`/`td2_2`, the NTK in the catapult (`td2_2` §2.7), and the weight decay in `td2_5`.
- $\epsilon$ is the step size in `td2_3` §3.4 (McCandlish et al.'s notation) and Adam's $\epsilon$ in §3.6.
- $S$ counts *steps* in `td2_3` §3.4's steps–examples tradeoff.
- $P$ is Adam's preconditioner in `td2_2`–`td2_5` and the number of parameters in `td2_6` (where $P_k$ is the projector onto the top-$k$ Hessian eigenvectors).

**One picture to reuse throughout.** Draw it once at the start and keep it on the board: a long valley with steep walls and a nearly flat floor. Add an iterate bouncing across the valley while drifting slowly along the floor. Segments 1, 3, 6 and 7 all come back to it.

---

## 0. Two kinds of tools (1.5 min)

**Claim.** Most of NB2 uses one of two tools.

1. **Small-step expansions:** treat discrete training as a flow plus corrections (backward error analysis), or as a stochastic differential equation. These give scaling rules.
2. **Where the expansions break:** real training often sits at $\eta S\approx2$, exactly where "small step" fails, and different tools take over.

**Speaker notes.**

- Every NB2 notebook stands alone. Each section opens with a short pitch and its paper, then an optional annotated walkthrough, so students can enter anywhere. Nobody is expected to finish.
- *(Room for the instructor's framing of the two tracks.)*

---

## 1. Learning rate (6 min)

### 1a. Stability, and gradient descent as a corrected flow (2 min)

**Claim.** On a quadratic, gradient descent converges iff $\eta S<2$. For small $\eta S$, gradient descent follows, more exactly than gradient flow does, the flow of a modified loss that penalizes the gradient norm. This is implicit gradient regularization (IGR). The expansion parameter is $\eta S$ (step times curvature). The series is asymptotic and usually divergent, the kept correction degrades smoothly as $\eta S$ grows, and for $\eta\lambda\ge1$ in some direction no gradient flow reproduces the step at all.

**Board.**
$$
\dot\theta=-\nabla\Big(L+\frac\eta4\|\nabla L\|^2\Big)
$$
(`td2_1` §1.2.)

**Speaker notes.**

- One-line idea, without the derivation: gradient flow's gradient shrinks along a step and gradient descent's doesn't, so gradient descent overshoots by $-\tfrac{\eta^2}{2}Hg$ per step. Writing that overshoot as a drift gives the penalty. It is heaviest in sharp directions.
- Limits, stated in `td2_1` itself: the penalty vanishes at critical points, so it doesn't help escape minima. It matters when minima form a family. "Gradient descent ends at flatter minima" is Barrett and Dherin's prediction with experiments, not a theorem.
- Momentum, in one sentence: heavy ball has effective learning rate $\eta/(1-\beta)$, stability edge $2(1+\beta)/\eta$, and at equal effective learning rate an implicit penalty $(1+\beta)/(1-\beta)$ times larger, valid only at much smaller curvatures. Goh's Distill article covers the tuning (`td2_1` §1.5).
- **Show:** `td2_1`, the code cell after Exercise 1.2.d, where gradient descent, gradient flow and the IGR flow are compared on $(ab-1)^2/2$. The IGR flow accounts for 81%, 42% and 16% of gradient descent's shift at $\eta S\approx0.2$, 0.8 and 1.4.
- **Picked up in:** `td2_1` §1.1 to §1.3, and §1.5 for momentum.

### 1b. The edge of stability (3 min)

**Claim.** Under full-batch gradient descent the sharpness typically rises during training (progressive sharpening) until $S\approx2/\eta$, then hovers there while the loss keeps falling on average. That is exactly where the expansion of 1a fails. The mechanism is *self-stabilization*: the oscillation along the top eigenvector $u$ feeds back through the third derivative and pushes the iterate down the sharpness gradient. A *central flow* models the time-averaged trajectory instead of the iterate.

**Board.**
$$
\nabla L(\bar\theta+xu)=\nabla L(\bar\theta)+x\,S\,u+\tfrac12x^2\,\nabla S(\bar\theta)+O(x^3)
$$
(`td2_2` §2.2.) The last term doesn't depend on the sign of $x$, so it survives averaging over the period-2 oscillation.

**Picture.** The valley: the bounce across the walls *is* the oscillation along $u$, and the $\tfrac12x^2\nabla S$ push moves the iterate toward lower walls.

**Speaker notes.**

- The central flow is $dw/dt=-[\nabla L+\tfrac12\sigma^2\nabla S]$, with $\sigma^2$ fixed by requiring $S$ to stay at $2/\eta$. Its time-averaged loss is $L+\sigma^2/\eta$. State it, don't derive it (that is Exercise 2.3). It is heuristic, and Cohen et al. state its limits.
- Other continuous-time models, presented side by side in `td2_2` §2.4: Rod Flow tracks the midpoint and half-length of consecutive iterates. Principal flow resums the backward-error-analysis terms built from $H$, which captures the onset of the instability but not the stabilization.
- **Open question to name:** *why* progressive sharpening happens is only partly understood. `td2_2` takes it as given.
- **Show:** `td2_2` §2.1, the MLP sharpness plot at three step sizes against $t=k\eta$. All three follow the same flow curve until each hits its own $2/\eta$. Then `assets/interactive/eos_toy_and_catapult.html`, panel 1, preset **EoS from below**.
- **Picked up in:** `td2_2` §2.1 to §2.4.

### 1c. The catapult (1 min, *compressible*)

**Claim.** In the smallest model where curvature can change (a width-$n$ two-layer net, one input), a large step can *lower* curvature in one shot. The regimes are $\eta\lambda_0<2$ (lazy), $2<\eta\lambda_0<4$ (the loss spikes, then curvature drops and training converges somewhere flatter), and $\eta\lambda_0>4$ (divergence). The $4/\lambda_0$ edge is specific to this model; the paper finds ReLU nets tolerate up to about $12/\lambda_0$.

**Speaker notes.**

- Contrast with 1b: self-stabilization is a sustained hover, while the catapult is a one-shot drop with nothing pushing the curvature back up.
- **Show:** the same HTML, panel 2, presets **lazy**, **catapult** and **divergent**.
- **Picked up in:** `td2_2` §2.7 (exact maps, Exercise 2.7).

---

## 2. Batch size (5 min)

**Claim.** A minibatch gradient is the full gradient plus noise with covariance $\Sigma/B$. To first order in $\eta$, SGD is gradient flow plus Brownian noise whose strength depends on $(\eta,B)$ only through $\eta/B$. Holding $\eta/B$ fixed holds the SDE fixed, which is the linear scaling rule. A separate one-step Taylor model gives the critical batch size, past which larger batches buy almost nothing. For Adam the same reasoning gives a temperature $\eta/\sqrt B$, and hence a square-root rule.

**Board.**
$$
d\theta=-\nabla L\,dt+\sqrt{\eta/B}\;\Sigma^{1/2}\,dW
\qquad\qquad
B_{\rm noise}=\frac{\mathrm{tr}(H\Sigma)}{G^\top HG}
$$
(`td2_3` §3.2 and §3.4.) Say aloud that $\Delta L_{\rm opt}(B)=\Delta L_{\max}/(1+B_{\rm noise}/B)$.

**Speaker notes.**

- The order-2 SME adds IGR's drift correction $\tfrac\eta4\|\nabla L\|^2$, which depends on $\eta$, not $\eta/B$. So the linear scaling rule is a first-order statement. It fails once the batch approaches $B_{\rm noise}$, when $\eta h$ isn't small in the top directions, and at the edge of stability (`td2_3` §3.5).
- Adam's rule, stated only: under $B\to\kappa B$, take $\eta\to\sqrt\kappa\,\eta$, $1-\beta_i\to\kappa(1-\beta_i)$, $\epsilon\to\epsilon/\sqrt\kappa$ (Malladi et al.; here $\epsilon$ is Adam's). It needs noise-dominated gradients.
- **Competing accounts, present them evenly:**
  - *What does SGD's noise buy?* One view treats the SDE's temperature $\eta/B$ as the important parameter. Smith, Dherin, Barrett and De argue from experiments that the finite-$\eta$ drift corrections matter more for generalization (`td2_3` §3.7 item 4; the SGD version of IGR is `td2_1` §1.4).
  - *Is the noise Gaussian?* Şimşekli, Sagun and Gürbüzbalaban argued for heavy tails (a Lévy limit). Li, Malladi and Arora argue the measurement can mislead, and find matched-covariance Gaussian noise reaches the same CIFAR-10 accuracy (§3.7 item 7).
- "Is the Taylor expansion even valid?" `td2_3` §3.7 collects nine answers. Weak-order bounds hold on a fixed horizon as $\eta\to0$, and nothing quantitative follows for a given $\eta$. The question can be tested directly with SVAG (§3.5).
- Under SGD, the quantity that sits at $2/\eta$ is a *batch sharpness*, and the full-batch $S$ settles lower (`td2_2` §2.6).
- **Show:** `td2_3` §3.4.B, the steps–examples tradeoff measured on least squares and fitted by the hyperbola $(S/S_{\min}-1)(E/E_{\min}-1)=1$.
- **Picked up in:** `td2_3` §3.1 to §3.6 (core reading: Li, Malladi and Arora), §3.7 for validity.

---

## 3. Schedule: warmup and decay (3 min)

**Claim.** Both ends of a schedule have derivable mechanisms, and each end has more than one account.

**Board.**
$$
\tfrac12\mathrm{tr}(H\Pi)=\frac{\eta}{4B}\,\mathrm{tr}\,\Sigma\qquad\text{where }H\Pi+\Pi H=\tfrac\eta B\Sigma
$$
(`td2_3` §3.3.) In a quadratic basin, the stationary excess loss of constant-step SGD doesn't depend on $H$. Cutting $\eta$ lowers it right away.

**Speaker notes.**

- **Warmup, two accounts.**
  - *Edge of stability* (Kalra and Barkeshli): starting at the target $\eta$ from a sharp initialization causes a catapult or divergence. Ramping $\eta$ up lets $S$ follow $2/\eta(t)$ down into regions that tolerate the target step.
  - *Adam's second moment* (Liu et al., RAdam): early $\hat v$ is estimated from few samples, so the adaptive step has high variance. `td2_2` §2.8 states it in one paragraph without deriving it; Kalra and Barkeshli's paper (a `td2_2` reading) also proposes a second-moment initialization for Adam.
  - `td2_2` §2.8 develops the first account, and Exercise 2.8(a) tests it in the toy, from two starting sharpnesses, and asks when warmup helps and whether it can hurt.
- **Decay, three pictures of the same observation** (the loss drops when $\eta$ is cut):
  - the OU floor on the board;
  - Wen et al.'s *river valley* for WSD schedules, where SGD noise makes the iterate bounce between the hillsides and hides progress along the river until the decay phase reveals it;
  - an edge-of-stability analogue, where the measured loss is $L+\sigma^2/\eta$ and decaying $\eta$ moves the edge up. `td2_2` labels this analogy as its own and not a claim of either paper.
- **Show:** `eos_toy_and_catapult.html` panel 1, presets **sharp start, no warmup** and **sharp start, warmup**. Then `td2_3` §3.3.B, the learning-rate-decay cell, whose dashed line is the exact expected excess loss, not a fit.
- **Picked up in:** `td2_2` §2.8; `td2_3` §3.3.

---

## 4. Optimizer (4 min)

**Claim.** A first-order optimizer has to decide what counts as a *small* step. That decision is a norm, and the norm fixes the direction. Euclidean gives gradient descent, $\ell_\infty$ gives sign descent (Adam with its averages off), the Fisher norm gives natural gradient, and the spectral norm on a weight matrix gives $UV^\top$, which Muon approximates by Newton–Schulz.

**Board.**
$$
\Delta=\arg\min_{\|\Delta\|\le\eta}\langle g,\Delta\rangle=-\eta\,d^\star(g),\qquad d^\star(g)\in\arg\max_{\|d\|\le1}\langle g,d\rangle
$$
(`td2_4` §4.1, trust-region form.) The squared-penalty form points the same way, with a step $\|g\|_*$ times longer.

**Picture.** A unit ball with $-g$ drawn and a supporting line perpendicular to $g$. The step is where the line touches the ball. A circle gives $-g$, a square box gives a corner ($-\mathrm{sign}\,g$), and an ellipse gives $-M^{-1}g$.

**Speaker notes.**

- **The parable (labelled as an analogy, as `td2_4` §4.0 does).** Between point masses with disjoint supports, KL is infinite and Jensen–Shannon is constant, so neither gives a gradient. Wasserstein-1 is $|\theta|$, because only it uses the geometry of the space. The parameter-space version is only the *question* "in which geometry should a step be measured?" The GAN theorems imply nothing about optimizers.
- `td2_4` makes this point itself: every optimizer is steepest descent under *some* norm, so the dictionary explains nothing on its own. A norm earns its place through parametrization invariance (natural gradient, §4.2) or step sizes that transfer across width (§4.5, next segment).
- Natural gradient is invariant to reparametrization to first order. Raw sharpness is not, because the Hessian is a quadratic form and its eigenvalues need a metric. This links to the MDL day's flatness discussion (Dinh et al.'s two-layer example is MDL WB2 Exercise 4.5, part 3).
- Adam's edge: with a frozen preconditioner, Adam at $\beta_1=0.9$ is stable iff $\eta\lambda_{\max}(P^{-1}H)<38$, and full-batch Adam holds this *preconditioned* sharpness at $38/\eta$ while raw sharpness keeps rising (`td2_2` §2.5).
- "Adam is sign descent" is exact only with the averages off (`td2_4` §4.3.D gives the failures).
- **Show:** `assets/interactive/steepest_descent_balls.html`. Toggle **Trust region** / **Squared penalty**, and try **g₂ ≈ 0** to show why sign descent is sensitive to small gradient components. Optionally `td2_4` §4.4.D, singular values through Newton–Schulz iterations: the cubic converges to 1, while Muon's quintic deliberately lands them in a band.
- **Picked up in:** `td2_4` §4.0 to §4.4 (core reading: Bernstein and Newhouse); `td2_2` §2.5.

---

## 5. Width (2 min)

**Claim.** Asking which step sizes transfer across width gives a spectral condition for feature learning. Combined with the spectral norm of each optimizer's update, it gives the learning-rate rule. For Adam's hidden layers that rule is $\eta\propto1/\text{fan-in}$, which is muP's rule.

**Board.**
$$
\|W_l\|_2=\Theta\Big(\sqrt{n_l/n_{l-1}}\Big),\qquad\|\Delta W_l\|_2=\Theta\Big(\sqrt{n_l/n_{l-1}}\Big)
$$
(`td2_4` §4.5, Yang, Simon and Bernstein's Condition 1.)

**Speaker notes.**

- For a low-rank gradient, $\|\mathrm{sign}(G)\|_2=\Theta(\sqrt{n_ln_{l-1}})$, which gives Adam $\eta_l=\Theta(1/n_{l-1})$. A *dense random* sign matrix would give only $\Theta(\sqrt n)$: the structure of the gradient sets the exponent. Muon's $\|UV^\top\|_2=1$ gives $\eta_l=\Theta(\sqrt{n_l/n_{l-1}})$ directly.
- Claim boundary: the derivation needs updates that are low-rank, or aligned with incoming features. That muP transfers in large transformers is an empirical finding consistent with this calculation, not implied by it.
- Link back to L1: the parametrization decides whether a wide network learns features or stays lazy. Tensor Programs IV is in NB1 §7.
- **Show:** `td2_4` §4.5.D, $\|\mathrm{sign}(G)\|_2$ against width for low-rank versus dense random $G$ (the derivation is §4.5.C).
- **Picked up in:** `td2_4` §4.5. The modular norm, and the depth scaling it suggests, are §4.6.

---

## 6. Normalization and weight decay (4 min)

**Claim.** Most weights in a modern network feed a normalization layer, so the loss sees only their direction. The step the direction takes is set by $\eta/\|w\|^2$, and weight decay's job on those weights is to set $\|w\|$. At equilibrium each normalized block rotates by a fixed angle per step, independent of its gradient scale. On scale-invariant weights, weight decay acts as a learning-rate schedule, not a regularizer of the function. Li and Arora make this exact: it is equivalent to an exponentially increasing learning rate.

**Board.**
$$
\langle\nabla L(w),w\rangle=0,\quad\nabla L(cw)=\tfrac1c\nabla L(w)\ \Rightarrow\ \eta_{\rm eff}\approx\frac{\eta}{\|w\|^2};\qquad\cos\theta^*=1-\eta\lambda,\ \ \theta^*\approx\sqrt{2\eta\lambda}
$$
(`td2_5` §5.1 and §5.2; here $\lambda$ is the weight decay. The exact form is $\eta_{\rm eff}=\eta/((1-\eta\lambda)r^2)$.)

**Picture.** A sphere with $w$ on it. The gradient is tangent, so a step lengthens $w$ by Pythagoras and the norm only grows without decay. Weight decay pulls radially inward. The two balance at a fixed rotation angle per step.

**Speaker notes.**

- **Scope.** In a BatchNorm convnet almost every weight is scale-invariant. In a pre-LN transformer few matrices are exactly scale-invariant, and the analysis is a model there (`td2_5` §5.1).
- **Competing accounts of weight decay.** The effective-learning-rate account above, versus D'Angelo et al., who argue weight decay is never useful as an explicit regularizer and whose vision mechanism is about keeping SGD's noise alive. Parameters that aren't scale-invariant can still be regularized. `td2_5` §5.2 and its readings give both accounts.
- **Spikes** (`td2_5` §5.3): four documented mechanisms, sorted by whether they are an edge-of-stability crossing.
  - Yes: weight decay shrinking a normalized layer until $\eta_{\rm eff}S$ crosses 2; Adam's second moment going stale so the preconditioned sharpness crosses $38/\eta$ (Bai et al.; Molybog et al. give a related account with $\epsilon$ in the lead role, and whether the two are the same instability isn't established).
  - No: attention-logit growth (fixed by QK-layernorm); output-logit drift along a flat direction (fixed by z-loss).
- **Clipping** (`td2_5` §5.4): under $(L_0,L_1)$-smoothness, where curvature grows with the gradient norm, the right step is $\propto1/\|\nabla L\|$ where the gradient is large. That is clipping.
- **Show:** `td2_5` §5.2, the weight-decay equilibrium cell with five settings, including two pairs with the same $\eta\lambda$ that end up at the same angle. Then the summary table at the end of §5.3.
- **Picked up in:** `td2_5`. Its short route is 30 to 45 min.

---

## 7. What the knobs act on (2 min, *compressible*)

**Claim.** Here is some phenomenology that any account of training has to match, and the mechanisms NB2 derives for it.

- *Curvature* (`td2_6`): the Hessian is Gauss–Newton plus a residual. The Gauss–Newton part has the same nonzero spectrum as the loss-weighted tangent kernel, so progressive sharpening (with square loss) is the kernel moving. The spectrum is a bulk near zero plus about $C$ outliers tied to class means. The gradient concentrates in the top subspace, yet in Song, Ahn and Yun's setting an optimizer restricted to it stops making progress. The basin is chosen early (Frankle et al.'s forks), during the phase when the kernel moves fastest (Fort et al.).
- *What is learned first* (`td2_7`): a model linear in its inputs sees only the first two moments, so a network with small pre-activations learns from Gaussian-equivalent statistics first. A direction hidden in a higher cumulant costs about $d^{\,k-1}$ samples for information exponent $k\ge3$, far fewer if it is correlated with lower-order structure. A fixed kernel fits eigenfunction $k$ at rate $\lambda_k$ (spectral bias).

**Board** (one equation, if any).
$$
\Delta L=-\eta g^2\Big(1-\frac{\eta\lambda}{2}\Big)
$$
(`td2_6` §6.4.) A step along a direction of curvature $\lambda$. At $\eta\lambda\approx2$ the first-order gain cancels. The top-subspace gradient is the restoring force of the bounce across the valley, and progress is the drift along the floor, which lives in the bulk.

**Speaker notes.**

- Claim boundary, as `td2_6` states it: "learning is not in the top subspace" describes a mid-training valley regime. Near interpolation with cross-entropy, the notebook's own small network shows the opposite (the top subspace is where the margins grow), and the notebook says it hasn't isolated which difference from Song et al.'s setting matters. Present both.
- If L1 was skipped, give the mode picture here in one line: shallow, $\dot a=s-a$; two-layer from small balanced init, $\dot a=2a(s-a)$, so strongest first, after a plateau (`td2_7` restates this).
- **Show:** `td2_6` §6.4.B, alignment $f_{\rm top}$ and the Dom/Bulk projections in both settings. Or `td2_7` §7.1, a small-init network trained on the data and on its Gaussian clone.
- **Picked up in:** `td2_6` (core reading: Song, Ahn and Yun); `td2_7` (core readings: Refinetti et al., Bardone and Goldt, Basri et al.).

---

## 8. Menu and routes (2 min)

Put the menu up as a list. Students choose.

| Knob or question | Notebook |
|---|---|
| Learning rate as a step size: flows, IGR, momentum | `td2_1_flows` |
| Learning rate at the edge: EoS, central flows, Adam's and SGD's edges, catapult, warmup and WSD | `td2_2_edge_of_stability` |
| Batch size: SMEs, loss floor, critical batch, linear and square-root rules, validity | `td2_3_sgd_sde` |
| Optimizer and width: norms, natural gradient, Adam as sign descent, Muon, muP | `td2_4_optimizer_geometry` |
| Normalization, weight decay, spikes, clipping | `td2_5_normalization_weight_decay` |
| What the Hessian looks like; when the basin is chosen | `td2_6_curvature` |
| What is learned first | `td2_7_learned_first` |
| Dropout, layer cushion, geometry-adaptive bases, critical periods, RL fine-tuning dynamics (readings with short exercises) | `td2_8_further_readings` |

**Suggested routes** (from `td2_8`'s map; suggestions only):

- *Less background:* NB1 §1, §4 and §5 (mode equations, the kernel, lazy/rich), then `td2_1` followed by `td2_3`. Flows and SDEs are one story: backward error analysis, first for the step and then in distribution.
- *Practitioner-minded:* `td2_3` → `td2_4` → `td2_5`.
- *More background:* skim NB1, then any NB2 notebook. `td2_2` and `td2_6` are where the expansions break and the empirical phenomenology lives.

**Out of scope for the day, with where to go instead.** Whether SGD samples a Bayesian posterior: the MDL day's WB2 §8 and its figure `sgd_stationary_vs_posterior.html`. Infinite-width NTK theory: the Physics of Deep Learning day.

*(Room for the instructor's own recommendations.)*

---

## Timing summary

| Segment | Full | Compressed |
|---|---|---|
| 0 Two kinds of tools | 1.5 | 1 |
| 1 Learning rate (1a, 1b, 1c) | 6 | 4.5 (1c to one sentence) |
| 2 Batch size | 5 | 4 |
| 3 Schedule | 3 | 2.5 |
| 4 Optimizer | 4 | 3.5 |
| 5 Width | 2 | 1.5 |
| 6 Normalization and weight decay | 4 | 3.5 |
| 7 What the knobs act on | 2 | 0.5 (one sentence each) |
| 8 Menu and routes | 2 | 1.5 |
| **Total** | **29.5** | **22.5** |
