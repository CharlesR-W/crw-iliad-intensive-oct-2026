# Training Dynamics: day specification, v1.1 (2026-10-08)

Status: material selection only. No figures, sweeps, or code are built yet; "interactive candidate" marks places where a Distill-style figure might later earn its keep. Format follows the Physics of Deep Learning day (`/home/crw/Programming/IliadPhysicsLearning`): markdown-first notebooks, exercise and solution variants from one builder, hints in `<details>`, claim boundaries, "what carries forward", "further study".

Readings marked † are not in CRW's topic notes; CRW should confirm or cut them before anything is narrated around them. Every paper named here has a verified link and a one-line "what it derives" in `TD_LINKS_v1.md`.

v1.1 changes: corrections from a link-verification pass and a review pass (time conventions unified, several overstatements fixed); new NB2 §5 on normalization, weight decay and instabilities; NB2 §1–§4 marked as the core.

## 0. Teaching contract

- **Two notebooks, two lectures.** NB1 (shorter): Saxe dynamics and linearity. NB2 (longer, the pragmatic core): the derivations behind the knobs practitioners turn. L1 introduces NB1 quickly; L2 contextualizes NB2.
- **Every section is enterable on its own.** Each NB2 section starts with a "what you need" box (restating any result it borrows, so nothing needs another section or the MDL day) and has three depths: *read* (framing paragraph plus a core reading), *derive* (guided exercises, each with a checkable end state), *go further* (linked papers). Students choose their depth per section.
- **Err toward linking.** Every cited paper is linked with one line saying what it derives or shows, so a student can decide whether to open it.
- **"Ask your agent" items.** Results we state but do not rederive carry the tag *(ask your agent)*: "if you think this is cool, ask your agent to walk you through the derivation". These are for curiosity, not coverage.
- **No dependence on the Physics of DL day.** The analytic infinite-width literature appears only as an escape hatch (NB1 §5).
- **Suggested routes**, not requirements. A realistic afternoon is 2–3 NB2 sections at derive depth. Less background: NB1 §1–3, then NB2 §1 → §3 (flows, then SDEs; they are one story). Practitioner-minded: NB2 §3 → §5. More background: skim NB1, take any NB2 sections.
- **Out of scope:** whether SGD samples a posterior; Morse–Bott/Witten.
- **Guest lecture (time unknown).** One floating 45–60 min slot (see Day shape). Nothing in the notebooks depends on it.
- **Cards at 17:00** as in MDL (claim / evidence / what would change my mind / survives scale?).

## Day shape

| Time | Option A: guest in the morning | Option B: guest in the afternoon |
|---|---|---|
| 10:00 | Guest lecture (introductory) | L1: linear networks (15–20 min) |
| ~10:50 | L1 compressed to 10 min, or skipped (NB1 §1 is self-contained) | NB1 / free work |
| 12:00 | lunch | lunch |
| 13:00 | L2: the pragmatic picture (25–30 min) | L2 (25–30 min) |
| 13:30 | NB2 / free work | NB2 / free work; guest at ~15:00 |
| 17:00 | cards | cards |

---

## L1. Lecture: linear networks in fifteen minutes

Purpose: give everyone the mode picture so NB2 can refer to it. Owned derivations stay in NB1.

- The model: $f(x)=W_2W_1x$, whitened inputs, $\Sigma_{yx}=USV^\top$. A shallow linear model learns every mode at the same rate; the two-layer one learns them as sigmoids, strongest first.
- Plot the mode ODE solutions at a few $s$, $a_0$ (static figure). Plateau length $\sim(\tau/2s)\log(s/a_0)$.
- Three heuristics that came out of this model: stagewise learning ordered by singular value (and the semantic-development story: broad distinctions before fine ones); small initialization means saddle plateaus and low-rank-first solutions; imbalance between layers decides lazy vs rich.
- What does not transfer: the solvable case needs aligned (decoupled) initial conditions and linear features. Nonlinear nets are not deep linear nets; the claim is that some of their *phenomenology* matches.
- Hand off: NB1 does the derivations; NB2 §7 picks up "what is learned first" for nonlinear nets.

## L2. Lecture: the pragmatic picture (25–30 min)

Purpose: organize NB2 around what a practitioner actually chooses (learning rate, batch size, schedule, optimizer, width, normalization and weight decay) and the one or two derivations behind each choice.

- **0–7 min. Small-step expansions.** Backward error analysis: GD as gradient flow plus corrections (implicit gradient regularization); SGD as an SDE. These give scaling rules: linear scaling in $\eta/B$, the critical batch size, the square-root rule for Adam. → NB2 §1, §3.
- **7–13 min. Real training breaks the expansion.** Progressive sharpening and the edge of stability: $\eta\lambda_{\max}\approx2$, exactly where "small $\eta$" fails. Large steps can also *lower* curvature in one shot: the catapult. The repair for the sustained regime: a flow for the *time-averaged* trajectory (central flows), and its stochastic counterpart. → NB2 §2.
- **13–20 min. Pick your metric, then your equation.** GAN→WGAN: on disjoint supports KL is infinite and JS saturates, while Wasserstein sees the distance. The lesson is to choose a metric that respects the space. The analogous move in parameter space (labelled as analogy): an optimizer is steepest descent under a norm. GD, sign descent, natural gradient, Muon, and muP's width scaling all fall out. → NB2 §4.
- **20–25 min. The knobs nobody derives.** Normalization makes most weights scale-invariant, so the effective learning rate is set by the weight norm, and weight decay sets that norm. Loss spikes and clipping. → NB2 §5.
- **25–28 min. Phenomenology an explanation must match.** Hessian = bulk + a few outliers; gradients concentrate in the top subspace but learning does not happen there; the basin is chosen early; simple statistics are learned first. → NB2 §6, §7.
- Close with the section menu and suggested routes.

Must not: derive anything NB2 derives.

---

## NB1. Saxe dynamics and linearity (shorter; 1–1.5 h)

**Needs:** SVD, ODEs. **Boundary:** square loss, linear networks, whitened inputs unless stated. Everything is a statement about this model; transfer to nonlinear nets is phenomenological. **Time convention:** $\tau=1/\eta$, so $t$ counts steps.

### 1. The mode ODEs
- **1.1** Gradient flow on $\tfrac12\mathbb E\|y-W_2W_1x\|^2$ with $\mathbb E[xx^\top]=I$: derive $\tau\dot W_1=W_2^\top(\Sigma_{yx}-W_2W_1)$, $\tau\dot W_2=(\Sigma_{yx}-W_2W_1)W_1^\top$.
- **1.2 Conservation.** Show $\frac{d}{dt}(W_1W_1^\top-W_2^\top W_2)=0$. Interpretation: the rescaling symmetry $(W_1,W_2)\to(GW_1,W_2G^{-1})$ gives a conserved quantity (Noether, in one line).
- **1.3 Decoupled initial conditions.** Assume $W_1=R\,\mathrm{diag}(b)V^\top$, $W_2=U\mathrm{diag}(c)R^\top$. Show the modes decouple: $\tau\dot b_\alpha=c_\alpha(s_\alpha-b_\alpha c_\alpha)$, $\tau\dot c_\alpha=b_\alpha(s_\alpha-b_\alpha c_\alpha)$.
- **1.4** With $a=bc$ and imbalance $d=c^2-b^2$ (conserved per mode), show $\tau\dot a=\sqrt{4a^2+d^2}\,(s-a)$.
  - Balanced ($d=0$, $a_0>0$): $\tau\dot a=2a(s-a)$, solved by $a(t)=\dfrac{s\,e^{2st/\tau}}{e^{2st/\tau}-1+s/a_0}$. Learning time $\sim(\tau/2s)\log(s/a_0)$: strongest modes first, sigmoidal.
  - Strongly imbalanced ($|d_\alpha|\gg s_\alpha$): $\tau\dot a\approx|d_\alpha|(s-a)$, an exponential whose rate does not depend on $s_\alpha$. **Interpretation:** imbalance turns the rich, strength-ordered, sigmoidal regime into a lazy, linear-model-like one. This is the lazy/rich dial in its simplest form. (Large but *balanced* init is not lazy here: the effective rate $2a$ changes by $O(1)$ as $a$ relaxes to $s$.)
  - Compare the shallow model $\tau\dot a=s-a$.
- **1.5 Depth.** For a balanced depth-$N$ chain show $\tau\dot a=N a^{2-2/N}(s-a)$. Deliverable: the scaling of plateau length with $a_0$ for $N=2$ and for $N\ge3$, and what it implies for initialization scale in deep nets.

*Interactive candidate (key figure):* mode-ODE explorer with sliders for $s$, $a_0$, imbalance $d$, depth $N$; several modes overlaid so ordering and plateau lengths are visible.

### 2. Landscape and saddles
- **2.1** Show the critical points (in the decoupled parametrization) have each mode $a_\alpha\in\{0,s_\alpha\}$: partial SVD reconstructions.
- **2.2** At $a=0$ for two weight matrices the $(b,c)$ Hessian is $\begin{psmallmatrix}0&-s\\-s&0\end{psmallmatrix}$, a strict saddle. For three weight matrices the Hessian at 0 vanishes: a non-strict saddle (Kawaguchi shows such "bad" saddles appear only from this depth on). Connect to 1.5's plateau lengths.
- **2.3** State (not prove): no spurious local minima for deep linear nets (Baldi–Hornik for two layers; Kawaguchi† for any depth). Question: why does "no bad minima" say almost nothing about training time?

### 3. End-to-end dynamics and where the NTK lives
- **3.1** Show $\tau\dot W=W_2W_2^\top E+E\,W_1^\top W_1$ with $W=W_2W_1$, $E=\Sigma_{yx}-W$. This is a linear operator $K_t[E]$, the NTK of the linear net.
- **3.2** At zero imbalance show $W_2W_2^\top=(WW^\top)^{1/2}$ and $W_1^\top W_1=(W^\top W)^{1/2}$. So the kernel is a function of the current solution: it grows along learned directions. That is feature learning in the smallest possible model. (Arora–Cohen–Hazan† give the depth-$N$ version and show it is not the gradient flow of any regularized loss.)
- **3.3** When $K_t$ is approximately constant (here: large imbalance, $|d_\alpha|\gg s_\alpha$), the dynamics are those of a linear model with a fixed kernel. Claim boundary: the frozen kernel is a regime, not a property of wide nets as such; it depends on parametrization (Chizat–Oyallon–Bach show any model can be pushed into it by output scaling). The general infinite-width NTK limit is *(ask your agent)* or NB1 §5.
- **3.4 Where alignment comes from** (fixes a gap in the old course). 1.3 *assumes* aligned initial conditions. Exercise: for small $W$, drop the cubic terms; show the stacked matrix $Z=[W_1,\;W_2^\top]$ obeys $\tau\dot Z=Z\begin{psmallmatrix}0&\Sigma_{yx}^\top\\\Sigma_{yx}&0\end{psmallmatrix}$, whose eigenvalues are $\pm s_\alpha$ with eigenvectors $(v_\alpha,\pm u_\alpha)/\sqrt2$. Deliverable: each hidden unit's weights grow fastest along $(v_1,u_1)$, so alignment with the top modes happens before the loss moves much. Reading: Atanasov–Bordelon–Pehlevan "silent alignment"†.

### 4. Heuristics Saxe draws, and how far they go
Reading-led. For each heuristic: the derivation in §1–3 it rests on, then one question with a one-sentence answer the student commits to.
- Stagewise learning ordered by singular value; semantic development as progressive differentiation (Saxe–McClelland–Ganguli 2019, PNAS). Question: name one property a nonlinear net would need for the same ordering.
- Small init → low-rank-first, saddle-to-saddle (incremental) learning. Question: is "low-rank bias" a property of the loss, the init, or the optimizer? Pick one and defend it.
- Initialization: orthogonal init (dynamical isometry) gives depth-independent learning times; unsupervised pretraining read as finding good initial conditions (historical).
- Generalization dynamics in linear nets: overtraining is worst near $p\approx n$, and small init protects (Advani–Saxe–Sompolinsky†). Cross-reference MDL WB2 §4 (interpolation peak).

### 5. Going further
- Lazy to rich beyond linear: Tu–Aranguri–Jacot (mixed dynamics); Kunin et al. "Get rich quick".
- Gated linear networks / neural race reduction (Saxe, Sodhani, Lewallen†).
- **The analytic escape hatch.** "If you really hate the practical derivations in NB2, fine: here you can do analytical theory all day." Roberts–Yaida–Hanin *Principles of Deep Learning Theory* (the $1/n$ perturbative expansion of finite-width nets); mean-field theory of two-layer nets (Mei–Montanari–Nguyen; neuron distribution as a Wasserstein gradient flow); Tensor Programs IV / muP (Yang–Hu; the parametrization in which infinite-width nets still learn features); Bordelon–Pehlevan DMFT (self-consistent kernel dynamics in that limit). No exercises; not assumed anywhere.

### What carries forward
The mode picture (strength-ordered, init-dependent learning) returns in NB2 §7; the kernel $K_t$ returns in NB2 §6.4; parametrization returns in NB2 §4.5 (muP).

---

## NB2. The pragmatic core (longer; sections independent)

**Core:** §1–§4. **Most practitioner-facing:** §5. **Optional:** §6–§8.

**Notation and time convention:** loss $L$, Hessian $H$, step size $\eta$, batch size $B$, per-example gradient covariance $\Sigma$ (so $\mathrm{Cov}\,g_B=\Sigma/B$ for sampling with replacement), gradient $G=\nabla L$, sharpness $S=\lambda_{\max}(H)$. Continuous time is $t=k\eta$ throughout, so plain gradient flow is $\dot\theta=-\nabla L$. The one exception is flagged in §3.4.

### 1. Gradient descent as a flow plus corrections
*Needs:* Taylor expansion. *Core reading:* Barrett–Dherin, IGR.
- **1.1 Stability on a quadratic.** GD on $\tfrac12\lambda x^2$ converges iff $\eta\lambda<2$ (hint-level; the reference point for §2).
- **1.2 Backward error analysis.** The method (from numerical ODEs: Hairer–Lubich–Wanner, *Geometric Numerical Integration* ch. IX) asks which modified flow the discrete method solves *exactly*, order by order in $\eta$. Most flow corrections in this notebook are instances; the central flow (§2.3) is instead a model of the time-averaged trajectory. Find $f_1$ such that $\dot\theta=-\nabla L+\eta f_1$ matches one GD step to $O(\eta^3)$. Show $f_1=-\tfrac12H\nabla L$, so $\dot\theta=-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)$. Interpretation: finite step size penalizes gradient norm, which is a preference for flatter paths. Claim boundary: valid for $\eta S\ll1$; §2 shows real training is not there.
- **1.3 SGD version** (read, Smith–Dherin–Barrett–De†): over an epoch of shuffled minibatches the modified loss gains $\tfrac{\eta}{4B}\mathrm{tr}\,\Sigma$ (up to a finite-population factor). Deliverable: the sign of the effect of halving $B$ at fixed $\eta$. Note this term depends on $\eta/B$ but the 1.2 term depends on $\eta$ alone (used in §3.1).
- **1.4 Momentum.** Heavy ball on a 1D quadratic: characteristic polynomial $z^2-(1+\beta-\eta\lambda)z+\beta$. Show stability iff $0<\eta\lambda<2(1+\beta)$. Effective learning rate $\eta/(1-\beta)$ in the flow limit. Core reading: Goh, "Why momentum really works" (Distill; interactive, and it covers the optimal tuning, so we do not rederive it). *(ask your agent)*: backward error analysis for heavy ball gives a correction $(1+\beta)/(1-\beta)$ times GD's at equal effective learning rate (Ghosh et al.†); for Adam with small $\epsilon$ the correction is a multiple of $\|\nabla L\|_1$ that is *negative* for the usual $\beta_2>\beta_1$ (Cattaneo–Klusowski–Shigida†).

### 2. The edge of stability
*Needs:* the threshold $\eta\lambda<2$ (and, for 2.4, the heavy-ball threshold $\eta\lambda<2(1+\beta)$), restated in the box. *Core reading:* Cohen et al. 2021 (EoS); Cohen et al. 2024 (central flows). *How you'd measure it:* sharpness by power iteration on Hessian-vector products.
- **2.1 Phenomenology** (read). Under full-batch GD, sharpness rises (progressive sharpening) until $S\approx2/\eta$, then hovers while the loss keeps decreasing non-monotonically. Deliverable: two sentences, why this contradicts 1.1-style intuition and why the 1.2 flow cannot produce it.
- **2.2 Self-stabilization toy.** $L(x,y)=\tfrac12S(y)x^2+\ell(y)$ with $S$ increasing in the direction $\ell$ pushes $y$. Show: when $\eta S>2$, $x$ flips sign and grows; the $y$-gradient picks up $\tfrac12S'(y)x^2$ (independent of the sign of $x$), which drives $y$ back until $S\approx2/\eta$. Find the period-2 orbit: $x^2=-2\ell'(y)/S'(y)$. (Mechanism from Damian–Nichani–Lee†, who prove GD at EoS tracks projected GD on $\min L$ s.t. $S\le2/\eta$.)
- **2.3 Central flow, one-eigenvalue case.** Model the time-averaged trajectory as $\dot\theta=-\big[\nabla L+\tfrac12\sigma^2\nabla S\big]$ with $\sigma^2$ fixed by requiring $S(\theta)$ to stay at $2/\eta$. Derive $\sigma^2=\max\big(0,-2\langle\nabla L,\nabla S\rangle/\|\nabla S\|^2\big)$; show that while the constraint is active ($\sigma^2>0$) this is gradient flow with $\nabla L$ projected orthogonally to $\nabla S$; get the predicted time-averaged loss $L+\tfrac12\sigma^2S=L+\sigma^2/\eta$. Check: on the 2.2 toy, $\sigma^2$ equals the period-2 $x^2$. This is Cohen et al.'s single-eigenvalue formula (they write an extra factor $\eta$ because their time counts steps); the multi-eigenvalue version replaces $\sigma^2$ by a covariance from a semidefinite complementarity problem *(ask your agent)*.
- **2.4 Adaptive optimizers.** EMA momentum $m\leftarrow\beta_1m+(1-\beta_1)g$ on a quadratic is stable iff $\eta(1-\beta_1)\lambda<2(1+\beta_1)$. This is 1.4's heavy-ball threshold under $\eta\to\eta(1-\beta_1)$, not a new result. With $\beta_1=0.9$: $38/\eta$. For Adam the relevant $\lambda$ is the *preconditioned* sharpness $\lambda_{\max}(P^{-1}H)$, while raw sharpness keeps rising (Cohen et al. 2022). Deliverable: a one-paragraph mechanism for why the preconditioner ends up holding training at this edge.
- **2.5 Stochastic** (read Andreyev–Beneventano, Edge of Stochastic Stability). Under minibatch SGD the quantity that rises to $2/\eta$ is *batch sharpness* $\mathbb E_B\big[g_B^\top H_Bg_B/\|g_B\|^2\big]$, while full-batch $S$ settles below $2/\eta$ by a batch-size-dependent gap. Exercise: show each batch's Rayleigh quotient $g_B^\top H_Bg_B/\|g_B\|^2$ decides whether *that* step overshoots; then show that demanding $\mathbb E\,\Delta L_B<0$ gives the ratio of expectations $\mathbb E[g^\top Hg]/\mathbb E\|g\|^2$ instead. Question: why does the paper's quantity (expectation outside) equilibrate and not this one? (Earlier arXiv versions of the paper themselves defined it differently; this is subtle.)
- **2.6 Other continuous-time models** (link and one line each, no rederivation): Rod flow (Regis–Chewi): ODEs for the midpoint and half-difference of consecutive iterates that reproduce EoS without solving a constraint, worth a look because it is a neat construction. Principal flow (Rosca et al.†): a partial resummation of backward error analysis in the Hessian eigenbasis that captures the *onset* of instability but, with no third-derivative terms, not the stabilization.
- **2.7 The catapult.** *Core reading:* Lewkowycz et al. The smallest model where a large step *lowers* curvature: $f=v^\top u/\sqrt n$ (width $n$), one input $x=1$, target $0$, $L=\tfrac12f^2$, NTK $\lambda=(\|u\|^2+\|v\|^2)/n$. Derive the exact one-step maps
  $f_{t+1}=f_t\big(1-\eta\lambda_t+\eta^2f_t^2/n\big)$, $\quad\lambda_{t+1}=\lambda_t+\tfrac{\eta f_t^2}{n}\big(\eta\lambda_t-4\big)$.
  Deliverable: the three phases from these two lines. $\eta\lambda_0<2$: $f$ decays, $\lambda$ barely moves ($O(1/n)$): lazy. $2<\eta\lambda_0<4$: $|f|$ grows until $f^2\sim n$, then $\lambda$ falls until $\eta\lambda<2$ and training converges in a flatter region: the loss spikes, then *catapults* down. $\eta\lambda_0>4$: $\lambda$ grows too, divergence. Compare with 2.2: the catapult is a one-shot drop in sharpness, self-stabilization a sustained hover; say what each needs (a third-derivative coupling vs. the $\eta^2f^2/n$ feature-learning term). Claim boundary: the $4/\lambda_0$ edge is for this model; the paper finds the divergence edge nearer $12/\lambda_0$ for ReLU nets.
- **2.8 Schedules and warmup.** Exercise: in the 2.2 toy, let $\eta(t)$ ramp up linearly from small; show $S$ can track $2/\eta(t)$ downward instead of being thrown into a large catapult (2.7) or divergence. That is one mechanism for warmup (Kalra–Barkeshli†, who also propose choosing the initial LR via the catapult); the other is the high variance of Adam's early second-moment estimate *(ask your agent)*. Read: WSD and the river-valley picture (Wen et al.). Deliverable: what the decay phase reveals that a constant LR hides (connect to §3.5).

*Interactive candidate (key figure):* the 2.2 toy with an $\eta$ slider (and optional ramp) showing $x$ oscillation, $y$ drift, and $S(y)$ settling at $2/\eta$; a second panel iterating the 2.7 maps, with an $\eta\lambda_0$ slider sweeping through the lazy, catapult and divergent phases.

### 3. SGD as a stochastic differential equation
*Needs:* basic probability. Stated-facts box (no Itô calculus needed): what a Wiener increment is, Euler–Maruyama, and what "weak order $k$" means (expectations of smooth test functions agree to $O(\eta^k)$ over a fixed time horizon). *Core reading:* Li–Tai–E (stochastic modified equations); Li–Malladi–Arora (validity). *How you'd measure it:* $B_{\rm simple}$ from gradient norms at two batch sizes.
- **3.1 Stochastic modified equations** are backward error analysis in distribution. Show plain gradient flow is already weak order 1 for SGD. Then show the order-2 SME needs *both* the noise term and 1.2's drift correction: $d\theta=-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)dt+\sqrt{\eta/B}\,\Sigma^{1/2}dW$. Deliverable: why linear scaling in $\eta/B$ is therefore a first-order statement (the drift correction depends on $\eta$, not $\eta/B$).
- **3.2 Linear scaling rule.** The noise term depends on $\eta/B$ only; derive the rule. Read Li–Malladi–Arora: their criterion is the noise-to-signal ratio $\mathbb E\,\mathrm{tr}\Sigma/\mathbb E\|\nabla L\|^2$; when it is small the SDE (and the rule) fails, and they turn this into a predicted critical batch size from one small-batch run. Compare with 3.3's $B_{\rm simple}=\mathrm{tr}\Sigma/\|G\|^2$. CRW's question from the notes: is the Taylor expansion even valid, given §2?
- **3.3 Critical batch size.** Second-order expansion of one step with learning rate $\eta$: $\mathbb E\,\Delta L=-\eta\|G\|^2+\tfrac{\eta^2}2\big(G^\top HG+\mathrm{tr}(H\Sigma)/B\big)$. Optimize $\eta$ and show $\Delta L_{\rm opt}(B)=\Delta L_{\max}/(1+B_{\rm noise}/B)$ with $B_{\rm noise}=\mathrm{tr}(H\Sigma)/G^\top HG$. Assuming $B_{\rm noise}$ roughly constant over training, derive the steps–examples tradeoff $(S/S_{\min}-1)(E/E_{\min}-1)=1$ (McCandlish et al.†). Interpretation: past $B_{\rm noise}$, larger batches buy almost nothing.
- **3.4 Adam's scaling rule** (read, Malladi et al. 2022†). **Time-convention exception:** for RMSprop/Adam one step corresponds to SDE time $\eta^2$, not $\eta$. This gives the square-root rule: for $B\to\kappa B$, $\eta\to\sqrt\kappa\,\eta$, $1-\beta_i\to\kappa(1-\beta_i)$, $\epsilon\to\epsilon/\sqrt\kappa$. Deliverable: one sentence on why normalizing by the second moment changes the exponent.
- **3.5 Fluctuations in a basin.** On a quadratic basin the SDE is an OU process; its stationary covariance solves $H\Pi+\Pi H=\tfrac\eta B\Sigma$ (stated). Exercise: take the trace to get the stationary excess loss $\tfrac12\mathrm{tr}(H\Pi)=\tfrac{\eta}{4B}\mathrm{tr}\,\Sigma$, independent of $H$. Deliverable: why decaying $\eta$ lowers the loss immediately (ties to 2.8's decay phase).

### 4. Optimizers as steepest descent under a norm
*Needs:* dual norms (stated). *Core reading:* Bernstein–Newhouse "Old optimizer, new norm"†; CRW "KL divergence, geometric naturalness, and optimal transport".
- **4.0 The parable, worked** (short). $P_0=\delta_0$, $P_\theta=\delta_\theta$: KL ($\infty$), JS ($\log2$), $W_1$ ($|\theta|$) and their gradients in $\theta$. Arjovsky–Bottou for the general disjoint-support statement; WGAN for the fix.
- **4.1** Trust-region form: show $\arg\min_{\|\Delta\|\le\eta}\langle g,\Delta\rangle$ is $-\eta$ times the dual-norm maximizer. $\ell_2$ gives normalized GD; $\ell_\infty$ gives $-\eta\,\mathrm{sign}(g)$; the Fisher norm $\sqrt{\Delta^\top F\Delta}$ gives natural gradient $\propto-F^{-1}g$ (Amari); the spectral norm on a matrix gives $-\eta UV^\top$ (Muon's direction). Then show the squared-penalty form $\langle g,\Delta\rangle+\tfrac1{2\eta}\|\Delta\|^2$ gives the same directions scaled by $\|g\|_*$.
- **4.2** Show natural gradient is invariant to smooth reparametrization to first order, and GD is not. Deliverable: the one-line statement of why raw sharpness is parametrization-dependent (restated here; MDL WB2 §3.5 has the Dinh et al. example).
- **4.3** Adam with $\beta_1=\beta_2=0$, $\epsilon=0$ is sign descent (hint-level); say what the EMAs change. Claim boundary: Adam is not sign descent in general.
- **4.4 Newton–Schulz.** For $X\leftarrow\tfrac32X-\tfrac12XX^\top X$, show singular values evolve by $\sigma\mapsto(3\sigma-\sigma^3)/2$ and converge to 1 for $\sigma\in(0,\sqrt3)$. Muon's tuned quintic deliberately does *not* converge to 1: it lands singular values in a band around 1, trading accuracy for faster growth of small singular values (Jordan's Muon post).
- **4.5 Width scaling.** Spectral condition for feature learning (Yang–Simon–Bernstein†): $\|W_l\|_2,\|\Delta W_l\|_2=\Theta(\sqrt{n_l/n_{l-1}})$. Exercise, with the assumption stated: if the gradient of a hidden $n\times n$ layer is low-rank (batch $\ll$ width), $\|\mathrm{sign}(G)\|_2=\Theta(n)$, so sign-like updates need $\eta\propto1/n$: muP's Adam hidden-layer rule. Contrast: a dense random-sign matrix has norm $\Theta(\sqrt n)$. Modular norm (Large et al.†) generalizes the recipe *(ask your agent)*.

*Interactive candidate (key figure):* 2D gradient vector with unit balls of $\ell_2$, $\ell_\infty$, a Mahalanobis norm, and the resulting steepest-descent directions.

### 5. Normalization, weight decay, and instabilities
*Needs:* nothing beyond the chain rule. *Core reading:* Li–Arora† (exponential LR); Wortsman et al.† (small-scale proxies for instabilities).
- **5.1 Scale invariance.** If $f(cw)=f(w)$ for $c>0$ (true for weights feeding a normalization layer), show $\langle\nabla f(w),w\rangle=0$ and $\nabla f(cw)=\nabla f(w)/c$. Hence the effective learning rate on the direction $w/\|w\|$ is $\eta/\|w\|^2$.
- **5.2 Norm growth and equilibrium.** Under GD, show $\|w_{k+1}\|^2=\|w_k\|^2+\eta^2\|\nabla f\|^2$: the norm only grows, so the effective LR decays on its own. Add weight decay $w\leftarrow(1-\eta\lambda)w-\eta g$ and show the equilibrium angular update per step is $\sqrt{2\eta\lambda}$, independent of the gradient scale. Deliverable: why weight decay on normalized layers acts as a learning-rate schedule, not a regularizer. Read: Li–Arora† (GD + WD on scale-invariant nets is equivalent to an exponentially increasing LR); Kosson et al.† (rotational equilibrium, including AdamW); D'Angelo et al.† (weight decay is never useful as an explicit regularizer in modern training).
- **5.3 Loss spikes** (read Wortsman et al.†; Molybog et al.†). Two documented mechanisms in transformers: attention-logit growth and output-logit divergence (mitigated by QK-norm and z-loss), and Adam updates becoming time-correlated when second-moment estimates go stale. Deliverable: which of these is an edge-of-stability phenomenon in the §2.4 sense, and which is not.
- **5.4 Clipping.** Under $(L_0,L_1)$-smoothness, $\|\nabla^2L\|\le L_0+L_1\|\nabla L\|$ (Zhang et al.†), show the safe step size is $\propto1/(L_0+L_1\|\nabla L\|)$, which is gradient clipping up to constants. Deliverable: why clipping helps most exactly when curvature grows with gradient norm.

### 6. Curvature: what the Hessian looks like and when the basin is chosen
*Needs:* chain rule; the edge-of-stability oscillation (one paragraph, restated); NB1 §3's kernel (one paragraph, restated). *Core reading:* Sagun et al. or Papyan (one); Gur-Ari–Roberts–Dyer with Song–Ahn–Yun.
- **6.1** Derive $H=J^\top(\nabla_f^2\ell)J+\sum_i(\partial\ell/\partial f_i)\nabla^2f_i$ (GGN + residual). Show the residual vanishes at interpolation for square loss; for cross-entropy it becomes small but never zero. For softmax cross-entropy, $\nabla_z^2\ell=\mathrm{diag}(p)-pp^\top$ has rank $C-1$.
- **6.2 Bulk and outliers** (read). A handful of outlier eigenvalues, often near the number of classes, plus a bulk near zero. Deliverable: from 6.1, a one-paragraph argument for why outliers track class structure.
- **6.3 Tiny subspace.** Gradients concentrate in the top Hessian eigenspace (Gur-Ari et al.); projecting updates onto that subspace stops learning, projecting them off it trains as well as full SGD (Song–Ahn–Yun). Deliverable: reconcile the two using the edge-of-stability oscillation along top directions.
- **6.4 When the basin is chosen** (read). The empirical NTK moves fast in the first few epochs and then at constant velocity (Fort et al.†); two runs forked from the same checkpoint become linearly mode-connected early (Frankle et al.). Deliverable: an operational definition of "basin commitment" and its linear analogue in NB1 §3.
- **6.5 Hook to Data Attribution** (one sentence): GGN vs Hessian, damping and K-FAC (Martens–Grosse†) are what influence functions see; the DA day owns them.

### 7. What is learned first
*Needs:* NB1 §1 helps. *Core reading:* Refinetti–Ingrosso–Goldt (distributional simplicity bias).
- **7.1** Show square loss on a model linear *in its inputs* depends on the data only through $\mathbb E[xx^\top]$ and $\mathbb E[yx^\top]$. Heuristic, with its boundary stated: a network that is near-linear in $x$ early (small init, activations near-linear at 0) learns from Gaussian-equivalent statistics first. A linearized (NTK) network is linear in *parameters*, not inputs, and its features see higher moments.
- **7.2 Spectral bias.** For kernel regression under gradient flow, show mode $k$ converges at rate $\propto$ kernel eigenvalue $\mu_k$ (the non-whitened version of NB1 §1.4's shallow case). Read Rahaman et al. or Xu et al. (frequency principle).
- **7.3 Higher moments** (read Bardone–Goldt, "Sliding down the stairs": learning a direction hidden in the order-$p$ cumulant costs $\sim d^{p-1}$ samples unless it is correlated with lower-order structure). CRW's question: why does the moment story not get you far, and what replaces it? Deliverable: one candidate answer and one observation that would refute it.

### 8. Further readings (one paragraph and two questions each)
- Kadkhodaie et al., geometry-adaptive harmonic bases: what basis a denoiser learns; explanation or description?
- Critical periods (Achille et al.) and loss of plasticity (Dohare et al.†): what early-learned structure costs later.
- Dropout as a geometric-mean ensemble (Baldi–Sadowski).
- Basin size, noise robustness, layer cushion (Arora–Ge–Neyshabur–Zhang compression).
- Fine-tuning dynamics for alignment: LM fine-tuning is often near-lazy (Malladi et al. 2023†; connect to NB1 §3.3); RL updates are sparse but full-rank (Mukherjee et al.); RLVR off the principals (Zhu et al.); RL's Razor (Shenfeld et al.); CRW "Some interesting papers on RLVR". *Overlaps MDL WB6's RL card: keep the framing consistent, and here ask the dynamics question (which directions in §6's eigenbasis does RL move?).*

### What carries forward
To Data Attribution: GGN, damping, unrolling (§3.1 is the continuous version of the unrolling formula). To MDL: §4.2 ↔ the parametrization dependence of flatness.

---

## Build notes (deferred)
- Builders `build_td_nb1.py`, `build_td_nb2.py` in the Physics-day style; no computational helpers in v1.
- Interactive candidates, only three: NB1 mode explorer, NB2 §2.2 self-stabilization toy (with LR ramp) plus §2.7 catapult maps, NB2 §4 steepest-descent balls. Momentum is covered by Goh's Distill article.
- Links: `TD_LINKS_v1.md`. Builders pull the one-liners from there.

## Open for CRW
1. Prune the † readings.
2. NB2 §5 is new (normalization/weight decay/spikes/clipping). Keep, trim, or move to readings?
