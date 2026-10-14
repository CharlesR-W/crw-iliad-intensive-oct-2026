# L2: Curvature at scale, from Hessian-vector products to the spectrum of a 100B model (outline and speaker notes)

**Length:** about 30 min, 28 slides; expect 30–35 min live. A compressed path of about 15 min is at the end.
**Slides:** [`L2_curvature_at_scale.html`](L2_curvature_at_scale.html). Arrows or space to move, N shows the speaker notes on the slide, S opens reveal.js's speaker view, Esc the overview. For a PDF, open the page with `?print-pdf` and print from Chrome or Chromium. Figures are made by `assets/L2/make_figures.py`; the formulas are checked in `../checks/L2_checks.py`.
**Comes after:** L1 and the morning on DA1. **Hands off to:** DA2 (`da2_methods`), whose stop 1 is EK-FAC influence functions at LLM scale.
**Purpose:** every estimator in the DA day touches curvature, and nobody can form the $P\times P$ matrix. Every direct look at the curvature of a real network uses one primitive, multiplying it by a vector. This half hour builds the primitive from forward- and reverse-mode autodiff, shows what can be computed with it (solves, top eigenpairs, the spectral density by stochastic Lanczos quadrature), what the spectra of trained networks look like, and how a 2026 paper gets a spectrum from a 120B-parameter model. It closes with what the spectrum says about attribution. The goal is the vocabulary to read Granziol & Juarev (2026) and DA2 stop 1 alone, and a healthy suspicion of curvature approximations and of spectra that look plausible but are artefacts.

**Notation on the board (matches DA1 §2, §4 and §6).** Summed objective $L(w;\beta)=\sum_i\beta_iL_i(w)$ over $n$ examples, $L=L(\cdot;\mathbf 1)$, $P$ parameters. Hessian $H=\nabla^2L$, with eigenpairs $(\lambda_j,u_j)$. Gauss–Newton $G=\sum_iJ_i^\top H_{y,i}J_i$, with $J_i=\partial f_i/\partial w$ the Jacobian of the outputs on example $i$ and $H_{y,i}$ the Hessian of the loss in the outputs; its eigenpairs are $(s_k,u_k)$ as in DA1 §6. Damping $\lambda$. The damped IF $\mathcal I_i=-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$. New today: a unit probe vector $v$; the Lanczos basis $q_k$ and tridiagonal $T_m$ with entries $\alpha_k,\beta_k$; Ritz values $\theta_k$ (eigenvalues of $T_m$); the spectral density $\rho(s)=\frac1P\sum_j\delta(s-\lambda_j)$; the finite-difference step $\epsilon$; machine precision $\epsilon_{\rm mach}$. Two clashes to say out loud: the damping $\lambda$ against the eigenvalues $\lambda_j$, and the Lanczos $\beta_k$ against the data weights $\beta_i$. Eigenvalues carry an index; the damping never does.

**The toy model in the figures.** A tanh MLP with 1,573 parameters on $C=5$ Gaussian classes ($n=1000$), trained for 300 full-batch Adam steps (training loss 0.09). The figures use the curvature of the **mean** loss, so their eigenvalues are $1/n$ of the summed-loss ones. Every ratio, fraction and relative error quoted below is unaffected.

Each segment lists time, the claim, at most two board equations, what to show, speaker notes, and where the notebooks pick it up. Links are relative to `da/lectures/`.

---

## 0. Title and where we are (1 min)

**Slides 1–2.**

**Claim.** This morning gave the formula. This half hour is about computing with the matrix inside it.

**Board.**
$$
\mathcal I_i=-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i
$$

**Speaker notes.**
- Point at $(G+\lambda I)^{-1}$. Every method this afternoon either approximates that matrix, replaces it by something of the same shape (DA1 §6's filters), or avoids it. All of them need curvature information about a model with $10^8$ to $10^{11}$ parameters.
- The through-line: every direct look at the true curvature uses one primitive, $v\mapsto Hv$ (EK-FAC, by contrast, approximates the curvature without it). Then what it buys, what trained spectra look like, how it was done at 120B, and what that tells you about attribution.
- **Picked up in:** [DA1 §4](../da1_core.html#da1-4) (where the formula comes from), [DA2 stop 1](../da2_methods.html#da2-stop-1) (how it is computed at scale).

---

## 1. Why you never form $H$, and what you can ask instead (1 min)

**Slide 3.**

**Claim.** $H$ has $P^2$ entries: a 1B-parameter model's Hessian would have $10^{18}$. But the three questions people ask need only products $v\mapsto Hv$.

**Picture.** A box labelled $v\mapsto Hv$ with three arrows out:
1. **solve** $(H+\lambda I)x=g$: what an IF needs;
2. **top-$k$ eigenpairs**: sharpness, IFs in a top-eigenvector subspace;
3. **spectral density** $\rho(s)=\frac1P\sum_j\delta(s-\lambda_j)$: what SLQ estimates. The trace is its first moment, $\operatorname{tr}H=P\int s\,\rho(s)\,ds$.

**Speaker notes.**
- In bf16, $10^{18}$ numbers are two exabytes. The toy spectra in TD 2.6 were computed on nets with a few thousand parameters, where `eigh` still works. Today's figures use such a net, so every estimate can be compared with the exact answer.
- The rest of the lecture: how to get $Hv$ (§2–3), then one arrow at a time (§4–6).
- **Picked up in:** [DA2 stop 1](../da2_methods.html#da2-stop-1) ("Never form $G$ or its inverse").

---

## 2. Forward and reverse mode: jvp and vjp (1.5 min)

**Slide 4.**

**Claim.** For $f:\mathbb R^P\to\mathbb R^d$ with Jacobian $J$, autodiff gives two products exactly, each for a small constant times the cost of evaluating $f$, whatever $P$ and $d$ are.

**Board.**
$$
\textbf{jvp: } v\mapsto Jv\ \ (\text{forward mode}),\qquad \textbf{vjp: } u\mapsto J^\top u\ \ (\text{reverse mode});\qquad \nabla L=J^\top 1\ \text{ for scalar }L
$$

- **jvp:** one forward pass carrying a tangent next to every value (dual numbers). Memory is that of a forward pass; no activations are stored.
- **vjp:** one forward pass that stores activations, then one backward pass. Backprop is the vjp with $u=1$.
- In code: `jax.jvp`, `jax.vjp`; `torch.func.jvp`, `torch.func.vjp`.

**Speaker notes.**
- This is the only slide of autodiff theory. Both products are cheap and exact (to rounding), and neither ever forms $J$.
- **Picked up in:** [DA2 Code 1](../da2_methods.html#da2-ekfac-code) uses `torch.func`. The lecture's checks file tests both products against an explicit Jacobian.

---

## 3. $Hv$ for the price of a couple of gradients (3.5 min)

**Slides 5–7.**

**Claim.** $Hv$ is a derivative of the gradient, so autodiff gives it exactly, in two ways. Finite differences give a third route that needs no autodiff plumbing, at the price of a step size. The Gauss–Newton product $Gv$ comes the same way.

**Board.**
$$
Hv=\nabla_w\big(\nabla L(w)^\top v\big)=\frac{d}{dr}\nabla L(w+rv)\Big|_{r=0}\approx\frac{\nabla L(w+\epsilon v)-\nabla L(w-\epsilon v)}{2\epsilon}
$$

**Slide 5, the two exact routes.**
- *Reverse-over-reverse* ("double backward"): differentiate the scalar $\nabla L^\top v$. Keeps the backward graph (`create_graph=True`).
- *Forward-over-reverse*: a jvp of the gradient function. This is Pearlmutter's (1994) operator $\mathcal R_v\{\cdot\}=\frac{\partial}{\partial r}(\cdot)(w+rv)|_{r=0}$ applied to backprop.
- Cost: gradient plus $Hv$ is about 2–3× the gradient alone (Schraudolph 2002).

**Slide 6, finite differences, and the figure.** Two ordinary gradient evaluations, so it works inside any training loop, including a sharded one. The error has two parts: truncation $\propto\epsilon^2$ and round-off $\propto\epsilon_{\rm mach}/\epsilon$. So the best step scales as $\epsilon^*\propto\epsilon_{\rm mach}^{1/3}$ and the best error as $\epsilon_{\rm mach}^{2/3}$. **Show** `assets/L2/fd_error.png`: on the toy, the relative error of the finite-difference $Hv$ against autodiff $Hv$, as a function of $\epsilon$. A V: round-off on the left, truncation on the right. The minimum is at $\epsilon\approx1.5\times10^{-2}$ in fp32 (error $3\times10^{-5}$) and $\approx0.2$ with bf16-rounded gradients (error $3\times10^{-3}$). Autodiff $Hv$ in fp32 is at $2\times10^{-7}$.

**Slide 7, Gauss–Newton products.** $Gv=\sum_iJ_i^\top H_{y,i}(J_iv)$: one jvp gives every $J_iv$, then a small $C\times C$ multiply per example, then one vjp (Schraudolph 2002). $G\succeq0$ for convex output losses, so conjugate gradients on $(G+\lambda I)x=g$ converges.

**Speaker notes.**
- Granziol & Juarev (2026, Thm 3.1) state the bound for unit $v$ as $\frac{\epsilon^2}{6}\|\nabla_w(D_v^3L)\|+O(\frac{\epsilon_{\rm mach}}{\epsilon}\|\nabla L\|)$, where $D_v^3L$ is the third derivative of $L$ along $v$, so $\nabla_w(D_v^3L)$ is a fourth derivative. As written this is leading order; a rigorous version takes the supremum of that fourth derivative over the segment $w+tv$, $|t|\le\epsilon$. Their Remark 3.2: fp32 gives $\epsilon^*\sim10^{-3}$–$10^{-2}$ with errors $\sim10^{-5}$; bf16 gives $\epsilon^*\sim10^{-1}$ with errors $\sim10^{-2}$. The toy agrees.
- Ask: where is the minimum for your model? The constants depend on the model, so the optimal $\epsilon$ moves. Also, $\epsilon$ is a step along a *unit* vector in $\mathbb R^P$, so each weight moves by about $\epsilon/\sqrt P$: about $4\times10^{-4}$ in the toy, about $10^{-8}$ for $\epsilon=10^{-4}$ at 160M parameters. Step sizes are not comparable across models. You find the right one by sweeping $\epsilon$ against an exact HVP on a probe. Hold that for §8.
- The bf16 curve rounds only the two gradients; everything else is fp32. With bf16 weights, a per-weight perturbation of $10^{-8}$ would round away entirely, so the perturbation has to live in higher precision.
- One identity worth knowing: the central difference of the gradient is exactly the average of $H(w+tv)v$ over $t\in[-\epsilon,\epsilon]$. A large $\epsilon$ does not give a noisy $Hv$; it gives curvature averaged along a segment. (Granziol & Juarev's Thm 3.3 states the triangular-kernel version for the second difference of $L$ itself.)
- $G$ is the matrix DA1 §4 derives from the PBRF and the one EK-FAC approximates. $H-G$ need not vanish even at a minimum (DA1 §4's $f(w)=w^2$ example).
- **Picked up in:** [DA1 §4](../da1_core.html#da1-4) (why $G$), [TD 2.6 §6.1](../../td/td2_6_curvature.html#td26-1) (the decomposition $H=G+{}$residual).

---

## 4. Solves, and two other first uses (1 min)

**Slide 8.**

**Claim.** An IF needs one solve $(G+\lambda I)x=\nabla\phi$ per query. Iterative HVP-based solves can be expensive (LiSSA often needs thousands of products; CG's count depends on conditioning), which motivates using EK-FAC for influence estimation. Two other tools need only $Hv$: power iteration and Hutchinson's trace estimator.

**Board.**
$$
\operatorname{tr}H=\mathbb E_v[v^\top Hv]\quad\text{for any probe with }\mathbb E[vv^\top]=I
$$

- **Solves:** Koh & Liang (2017) used conjugate gradients and LiSSA (Agarwal, Bullins & Hazan 2017), both built on HVPs. Grosse et al. (2023, §2.2.1): CG is a full-batch method, and LiSSA "often" needs thousands of iterations, each at least a gradient.
- **Power iteration:** $v\leftarrow Hv/\|Hv\|$ finds the largest-*magnitude* eigenpair, with error shrinking like $|\lambda_2/\lambda_1|^k$ (eigenvalues ordered by magnitude). Slow when the top eigenvalues are close, as outliers often are.
- **Hutchinson:** one HVP per probe.

**Speaker notes.**
- For i.i.d. $\pm1$ probes the variance of $v^\top Hv$ is $2(\|H\|_F^2-\sum_jH_{jj}^2)$; for Gaussian probes it is $2\|H\|_F^2$. PyHessian (Yao et al. 2020) packages power iteration, Hutchinson and SLQ for PyTorch models.
- Sharpness is the largest *algebraic* eigenvalue; for that, use Lanczos targeting the top end, as TD 2.2 does.
- **Picked up in:** [DA2 stop 1](../da2_methods.html#da2-stop-1) (EK-FAC: one cheap approximate solve per query).

---

## 5. Lanczos (3.5 min)

**Slides 9–11.**

**Claim.** $m$ HVPs, organized by a three-term recurrence, compress $H$ into an $m\times m$ tridiagonal matrix whose eigenvalues converge to the extreme eigenvalues of $H$ first. Outliers come out in a handful of iterations; the interior of the bulk does not come out eigenvalue by eigenvalue.

**Board.**
$$
Hq_k=\beta_{k-1}q_{k-1}+\alpha_kq_k+\beta_kq_{k+1},\qquad Q_m^\top HQ_m=T_m=\begin{pmatrix}\alpha_1&\beta_1&\\\beta_1&\alpha_2&\ddots\\&\ddots&\ddots\end{pmatrix}
$$

**Slide 9, the algorithm.** Start from a unit $q_1=v$. The Krylov subspace $\mathcal K_m(H,v)=\operatorname{span}\{v,Hv,\dots,H^{m-1}v\}$ gets an orthonormal basis $Q_m=[q_1,\dots,q_m]$ from the recurrence. Cost: $m$ HVPs plus $O(mP)$ vector work. The Ritz values $\theta_k$ (eigenvalues of $T_m$) approximate eigenvalues of $H$.

**Slide 10, show** `assets/L2/ritz.png`: Ritz values against $m$ for the toy Hessian. The top five eigenvalues are converged by $m\approx12$; the bulk stays a smear. Right panel: without reorthogonalization, converged eigenvalues come back as spurious copies ("ghosts"): four copies of $\lambda_{\max}$ by $m=60$, and stray Ritz values between the outliers.

**Slide 11, the memory wall.**
- In floating point the $q_k$ lose orthogonality once a Ritz value converges. The fix, full reorthogonalization, keeps all of $Q_m$ in memory: $m$ vectors of length $P$, and $O(m^2P)$ extra work.
- At $P=10^{11}$ in bf16, one vector is 200 GB. This rules out reorthogonalization at LLM scale; without it, ghosts force $m$ to stay small (slide 22).
- The eigenpairs route to IFs: Schioppa et al. (AAAI 2022) find the top eigenpairs with Arnoldi (the version for non-symmetric matrices) and compute IFs inside that subspace, up to a 300M-parameter ViT.

**Speaker notes.**
- Each new HVP is orthogonalized against only the two previous vectors; the symmetry of $H$ makes the rest vanish in exact arithmetic.
- Why outside-in: Ritz values are optimal over polynomials of degree $m-1$, and a polynomial can single out an isolated eigenvalue far more easily than one inside a dense cluster. The Kaniel–Paige bounds (refined by Saad) make this precise with Chebyshev polynomials: the rate depends on the gap to the rest of the spectrum relative to its spread.
- This asymmetry is the whole story of what SLQ can and cannot see.
- Code you have seen: TD 2.2 §2.5's `top_eig_lanczos` calls scipy's `eigsh` (implicitly restarted Lanczos) on double-backward HVPs.
- **Picked up in:** [TD 2.2 §2.5](../../td/td2_2_edge_of_stability.html#td22-5).

---

## 6. From Lanczos to the density: stochastic Lanczos quadrature (5 min)

**Slides 12–16.**

**Claim.** The same $T_m$ matches the first $2m$ moments of the spectrum of $H$ as seen from $v$; equivalently, it is an $m$-point Gauss quadrature rule. Averaging the rules over a few random probes estimates the spectral density: stochastic Lanczos quadrature (SLQ).

**Board.**
$$
\int f\,\rho=\frac1P\operatorname{tr}f(H),\qquad v^\top f(H)v=\sum_j(v\cdot u_j)^2f(\lambda_j)\approx e_1^\top f(T_m)\,e_1=\sum_{k=1}^m\tau_k^2f(\theta_k)
$$

**Slide 12, the target.** For a unit $v$, $v^\top f(H)v$ is an integral of $f$ against a discrete measure $\mu_v$: nodes at the eigenvalues $\lambda_j$, weights $(v\cdot u_j)^2$. If $v=z/\sqrt P$ with $z$ having i.i.d. $\pm1$ entries, then $\mathbb E[\mu_v]=\rho$ exactly. So approximate $\int f\,d\mu_v$ well for a few probes, and average.

**Slide 13, Golub–Welsch.** $e_1^\top T_m^ke_1=v^\top H^kv$ for $k\le2m-1$: $T_m$ reproduces the moments $0,\dots,2m-1$ of $\mu_v$. The resulting rule has nodes at the Ritz values $\theta_k$ and weights $\tau_k^2=(e_1^\top y_k)^2$, the squared first components of $T_m$'s eigenvectors $y_k$ (Golub & Meurant). **Show** `assets/L2/quadrature.png`: one probe; the cumulative weight of $\mu_v$ (1,573 steps) against the 10-node rule.

**Slide 14, SLQ (Ubaru, Chen & Saad 2017).**
$$
\hat\rho(s)=\frac1{n_v}\sum_{\text{probes}}\sum_{k=1}^m\tau_k^2\,\delta(s-\theta_k),\quad\text{then smooth with a Gaussian of width }\sigma\text{ to plot.}
$$
Two error sources: the quadrature ($m$) and the Monte Carlo average ($n_v$). Ghorbani et al. (2019) used $m=90$ with Gaussian probes; Granziol & Juarev used $m=10$ at 120B.

**Slide 15, show** `assets/L2/slq.png`: exact density (`eigh`) against SLQ with $n_v=10$. Left: $m=90$ and $m=30$ match bulk and outliers. Right: $m=10$ already has the outliers right; the bulk is lumpy.

**Slide 16, what SLQ resolves.** At scale you can trust the outliers and the bulk's overall shape. The mean eigenvalue has no quadrature error and is unbiased, but its probe-to-probe spread still has to be checked (for a spectrum dominated by a few outliers, ten probes can leave tens of percent). Individual bulk eigenvalues: no. The mass near zero, where an IF's damping acts, is where the rule is coarsest. In the toy, the fraction of eigenvalues with $|s|<\sigma$ is 0.891 exactly, 0.890 by SLQ at $m=90$, and 0.78 at $m=10$.

**Speaker notes.**
- The moments of $\hat\rho$ of degree up to $2m-1$ are exact in expectation, so their error is all Monte Carlo. In the toy, the first, second and fourth moments are off by $+1.7\%$, $-2.3\%$ and $-5.4\%$ at every $m$ from 10 to 90, because the same probes were used. More iterations do not fix Monte Carlo error.
- In high dimension one probe already averages over many eigenvectors. Chen, Trogdon & Ubaru (2021) bound SLQ's error in Wasserstein distance; for probes uniform on the sphere, the number their bound needs falls roughly like $\log P/P$ at fixed accuracy (but never below one). That is why huge models get away with very few probes.
- Physicists may know this as the recursion method: the $\alpha_k,\beta_k$ are continued-fraction coefficients of the local density of states. *(Room for the instructor's framing.)*
- Code: `google/spectral-density` (Ghorbani et al.'s TF and JAX code, now archived) and PyHessian both implement SLQ.
- **Picked up in:** [DA1 §6](../da1_core.html#da1-6) (the filters act eigenvalue by eigenvalue; the density says how many eigenvalues sit where).

---

## 7. What the spectra of trained networks look like (2.5 min)

**Slides 17–18.**

**Claim.** For classifiers, the measured picture is a bulk of eigenvalues near zero plus a few large outliers, about $C$ of them for $C$ classes, coming from the Gauss–Newton term's class structure. For language models the same reasoning says nothing small.

**Slide 17, one picture, three papers.**
- **Sagun et al. (2017):** bulk near zero plus outliers; on $k$-cluster synthetic data, the number of outliers above the gap equals the number of classes.
- **Ghorbani, Krishnan & Xiao (ICML 2019):** SLQ up to ImageNet. Without BatchNorm, outliers appear quickly and the gradient concentrates in their span; with BatchNorm, both effects almost vanish.
- **Papyan (2018; JMLR 2020):** the outliers come from class means in $G$: about $C$, with a weaker layer of $C(C-1)$ below them. The bulk comes from within-class variation and from $H-G$.

**Slide 18, language models.**
- That picture is for classifiers. For a language model, $C$ is the vocabulary, tens of thousands, so "$C$ outliers" is not a small number.
- SLQ with $m=10$ nodes per probe cannot resolve $10^4$–$10^5$ outliers one by one.
- Transformers are also uneven by block: Zhang et al. (NeurIPS 2024) ran SLQ per parameter block and found very different spectra across blocks in every transformer they examined, but not in CNNs.

**Speaker notes.**
- Ghorbani et al.'s ImageNet models are ResNet-18 and Inception-V3 (27M parameters). They also see large negative eigenvalues early in training that vanish within the first 1% of steps.
- Papers report $C$, $C-1$ or about $C^2$ outliers, depending on what they count, the cutoff, the loss, normalization and the stage of training. In our toy ($C=5$), the top five eigenvalues relative to the largest are 1, 0.97, 0.68, 0.42, 0.35, and the sixth is 0.23. A real gap, not a dramatic one.
- **Picked up in:** [TD 2.6 §6.3](../../td/td2_6_curvature.html#td26-3) (derives the class-mean mechanism) and [§6.4](../../td/td2_6_curvature.html#td26-4) (the gradient lives in the top subspace).

---

## 8. The 120B-parameter spectrum: Granziol & Juarev 2026 (6 min)

**Slides 19–24.**

**Claim.** Under fully sharded training, a finite-difference HVP computed shard by shard costs two ordinary gradient passes and reuses the training stack, which makes SLQ possible on a 120B model. The step size $\epsilon$ then becomes a knob that can fake a spectrum, and a test of block-diagonal curvature fails by $O(1)$.

**Slide 19, the move.**
- Under FSDP (ZeRO-3), each GPU stores a slice of every parameter tensor and gathers full layers only while computing them. The authors' diagnosis: naive HVPs force expensive all-gathers.
- Their move: the central finite difference, shard-locally. Perturb $w\mapsto w\pm\epsilon v$ in place on each shard, run two ordinary FSDP gradient passes over the same data, restore. The only extra communication is scalar all-reduces for norms and dot products. $T_{Hv}\approx2T_{\rm grad}$.
- The gain is compatibility with the training stack, not FLOPs. The basis memory is handled separately: by small $m$.

**Slide 20, reading the paper.** Things that will otherwise trip you up (the last one is in the notes):
- "Exact" means the true $H$, not a structured approximation. The finite difference still has truncation error.
- Figs 2–8 plot Ritz values against their weights ("Gamma", "Ritz weights") on a log axis: our $\theta_k$ and $\tau_k^2$, not smoothed densities.
- "Smoothing parameter $\epsilon$" is the finite-difference step (the segment-average view), not the plotting width $\sigma$.
- Their $\theta$ is the parameters. They call Pearlmutter's method "reverse-over-forward"; the order names vary between communities.
- Their HVP is of a dataset-averaged loss, so eigenvalues are $1/n$ of the summed-loss ones.
- (Notes.) §4's Lanczos error theorem assumes the finite-difference errors are unbiased and independent across iterations; Thm 3.1 gives an $O(\epsilon^2)$ bias, so at least unbiasedness is optimistic. Read §4 as heuristic.

**Slide 21, finding 1: $\epsilon$ can fake a spectrum (§7.1, Figs 3–6, 8).**
- Pythia-160M: at $\epsilon=10^{-5}$ and at $\epsilon=10^{-2}$ the spectrum looks like a random matrix, with no separated outliers. At $10^{-4}$ and $10^{-3}$, clear outliers.
- The authors attribute the two failures to round-off and to averaging along the segment, as in Thm 3.1.
- But the "good" $\epsilon$ is the one that shows the expected outliers. The check from slide 6, finite difference against an exact HVP on a probe, is not reported, though it would be cheap at 160M.
- They report that Pythia-1.4B is more stable in $\epsilon$ (Fig 8: $3\times10^{-4}$ to $10^{-2}$), and conclude larger models are. That is one pair of models.

**Slide 22, finding 2: gpt-oss-120b (§7, Fig 2).** Spectral density on wikitext-2 (sequence length 64, $\epsilon=10^{-4}$) with large negative eigenvalues. 10 Lanczos iterations, no reorthogonalization, only the essential Lanczos vectors stored; $\epsilon$ was not swept at this size. The authors call the plot "only indicative". Why 10: without reorthogonalization, ghosts appear once Ritz values converge. On the slide, one line: three readings (not memorised, the authors'; not near a minimum of this loss; numerics), which the plot cannot separate.

**Slide 23, finding 3: block-diagonal curvature can be wrong by $O(1)$ (§6, Tables 6–7).** Per transformer layer $b$, compare the true $(Hv)_b$ with the block-diagonal product $H_{bb}v_b$, for unit Gaussian $v$:
$$
\text{rel. err.}=\frac{\|(Hv)_b-H_{bb}v_b\|}{\|(Hv)_b\|}
$$
- Qwen-0.6B, 28 layers: relative error $0.94\pm0.06$, cosine $0.31\pm0.12$.
- DeepSeek-Coder 1.3B after code fine-tuning, 24 layers: relative error $1.21\pm0.15$, cosine $0.00\pm0.13$.
- Predicting zero gives relative error exactly 1, so above 1 the block product does worse than zero.

**Slide 24, scale table** (largest model in each paper; skip if late).

| Paper | Parameters | Curvature access |
|---|---|---|
| Ghorbani et al. 2019 | $2.7\times10^7$ (Inception-V3) | exact HVP, SLQ |
| Schioppa et al. 2022 | $3\times10^8$ (ViT-L/32) | exact HVP, Arnoldi |
| Grosse et al. 2023 | $5.2\times10^{10}$ | EK-FAC, no HVPs |
| HessFormer 2025 | $7\times10^{10}$ | Pearlmutter HVP, one 8-GPU node, SLQ |
| Granziol & Juarev 2026 | $1.2\times10^{11}$ (gpt-oss-120b) | finite-difference HVP under FSDP, SLQ with $m=10$ |

**Speaker notes.**
- Cost (§5.1): on one node, Qwen-0.6B goes from 1 to 8 GPUs with a 5.74× speedup per HVP (Table 3); Lanczos bookkeeping adds 1.3–1.8% over the HVPs (Table 2); on a 32B model, Table 5 reports about 4× over HessFormer (89.6 s at batch 32 against 22.2 s at batch 80). HessFormer (Granziol 2025), the previous record: Pearlmutter HVPs on one 8-GPU node, SLQ on a 70B DeepSeek distilled model. Multi-node scaling in the 2026 paper is shown on a 32B model (Table 4).
- On the 160M sweep, the fp32/bf16 labels in their figures refer to how the Lanczos basis vectors are stored, not the gradients. Their conclusion: storing the basis in bf16, with fp32 scalar arithmetic, is enough. Using 30× more data (a 30% split instead of 1%) leaves the spectrum essentially unchanged at fixed $\epsilon$ (Fig 7).
- Their Remark 3.2 puts the fp32 $\epsilon^*$ at $10^{-3}$–$10^{-2}$, yet the sweet spot they find is $10^{-4}$–$10^{-3}$ and $10^{-2}$ fails. Same lesson as slide 6: the constants, and $\sqrt P$, move the optimum.
- Why 10 iterations: HessFormer's paper says it is prudent to stay below 20 iterations without reorthogonalization, and often uses 10. Reorthogonalizing at 120B would need 240 GB per bf16 basis vector.
- Three readings of the negative eigenvalues, to hold side by side. The authors': the data, as tokenized, has not been memorised. Another: a pretrained checkpoint need not be near a minimum of this particular loss (wikitext-2, sequence length 64). A numerical one: $\epsilon$ was not swept at 120B, with 10 iterations and no reorthogonalization. The figure cannot separate them. Negative eigenvalues live in $H-G$ only, since $G\succeq0$; the IF's $G$ has none.
- The paper gives neither the number of probes nor the hardware for the 120B run, and no code; HessFormer's code is public.
- Block test, the caveat that changes a conclusion: it uses $H$ and random probes. EK-FAC approximates $G$, and an IF applies $(G+\lambda I)^{-1}$ to *gradients*, which may live in a different subspace. The test is a warning, not a verdict on IFs. In the other direction: EK-FAC's blocks are single weight matrices, finer than a whole transformer layer, so if anything the test is generous to the block assumption. For the DeepSeek-Coder model they report a substantially higher cosine in the final block (§6.2).
- DA2's Code 1 measures the IF-level error on a 147-parameter net: rank correlation with the exact damped Gauss–Newton IF is 0.65–0.73 for block-diagonal $G$ and 0.26–0.45 for K-FAC and EK-FAC, across $\lambda$ from $10^{-3}$ to $10^{-1}$ times the mean eigenvalue.
- *(Room for the instructor's framing.)*
- **Picked up in:** [DA2 stop 1](../da2_methods.html#da2-stop-1) and [Code 1](../da2_methods.html#da2-ekfac-code).

---

## 9. Back to attribution: what the spectrum tells you (3.5 min)

**Slides 25–26.**

**Claim.** Attribution wants the spectrum of $G$, not $H$. $Gv$ is the same primitive (slide 7), so everything above applies; the 2026 paper measured $H$. The spectrum of $G$ shows where the damping sits, which directions the IF corrects for curvature, and how far an approximation can be trusted.

**Board.**
$$
\mathcal I_i=-\sum_k\frac{(u_k\cdot\nabla\phi)(u_k\cdot\nabla L_i)}{s_k+\lambda}\ \approx\ -\frac1\lambda\sum_{s_k\ll\lambda}(u_k\cdot\nabla\phi)(u_k\cdot\nabla L_i)\;-\sum_{s_k\gtrsim\lambda}\frac{(u_k\cdot\nabla\phi)(u_k\cdot\nabla L_i)}{s_k+\lambda}
$$
($s_k,u_k$: eigenpairs of $G$, as in DA1 §6.)

**Slide 25, where the damping sits.** **Show** `assets/L2/filter.png`: the eigenvalues of the toy's $G$ (log axis; the bulk spans about seven decades) with the filter $\lambda/(s+\lambda)$ at two dampings.
- On directions with $s_k\ll\lambda$, $(G+\lambda I)^{-1}$ acts as $\lambda^{-1}$. How many directions that is depends on where $\lambda$ sits.
- Grosse et al. set $\lambda=0.1\times$ the mean EK-FAC eigenvalue, layer by layer. On the toy's exact $G$, that puts 78% of the eigenvalues below $\lambda$. A $\lambda$ between the bulk edge and the outliers puts 99.7% below it.
- At scale, SLQ measures the outliers well and the mean without quadrature error (check its probe spread), so both choices are computable. Where $0.1\times$mean falls *inside* the bulk is what SLQ resolves least well.

**Slide 26, what the IF is made of, and auditing.**
- Directions with $s_k\ll\lambda$ contribute a gradient dot product scaled by $1/\lambda$: the TracIn-like part. Directions with $s_k\gtrsim\lambda$ are curvature-corrected. Gradient-similarity scores differ from the IF on directions with $s_k$ near or above $\lambda$ (at $s_k=0.9\lambda$, replacing the weight by $1/\lambda$ is already a 90% error).
- Which directions matter for a given pair depends on both gradients' projections $u_k\cdot\nabla L_i$, $u_k\cdot\nabla\phi$ and on the weights $1/(s_k+\lambda)$; the density gives neither projection. (TD 2.6 §6.4 finds the gradient concentrated in the top Hessian subspace, where the weights are smallest, so concentration alone does not say which part dominates.)
- Auditing: EK-FAC assumes block-diagonal curvature with a Kronecker-factored eigenbasis (eigenvalues refitted). The 2026 test says the block step already costs $O(1)$ in $Hv$; DA2's Code 1 says the Kronecker step costs more than the block step at the IF level. Both are measurements on specific models; neither is a theorem.

**Speaker notes.**
- The mean eigenvalue is $\operatorname{tr}G/P$, which Hutchinson gives from a few Gauss–Newton products $Gv$. Check the spread across probes before setting $\lambda$ from it. (With $Hv$ it would give $\operatorname{tr}H$, a different number.)
- DA2's Code 1 sweeps $\lambda$ from $10^{-3}$ to $10^{-1}$ times the mean eigenvalue; DA1 §5's exercise uses the median Hessian eigenvalue.
- $\lambda$ also has a meaning: DA1 §4's proximal penalty. Choosing it from the spectrum is a statement about where you trust the quadratic model; choosing it from the proximal target is a statement about the question. *(Room for the instructor's framing.)*
- **Picked up in:** [DA1 §6](../da1_core.html#da1-6) (filters), [DA2 stop 1](../da2_methods.html#da2-stop-1) and [Code 1](../da2_methods.html#da2-ekfac-code).

---

## 10. Summary and readings (1 min)

**Slide 27, summary.**
- **The primitive:** $v\mapsto Hv$ (or $Gv$), exact by autodiff for a small multiple of a gradient, or by two gradients and a step size.
- **The three questions:** solve, top eigenpairs, density. Lanczos gives eigenpairs from the outside in; the same $T_m$ is a Gauss rule, and SLQ averages it into a density.
- **The one picture:** a bulk near zero and about $C$ outliers.
- **The one warning:** an approximation or a finite-difference step can give a plausible wrong answer. Check it against an exact HVP on a probe.

**Slide 28, readings.** Scores are Importance / Length, as in the notebooks.

Regular:
- Granziol & Juarev, *Hessian Spectral Analysis at Foundation Model Scale* (arXiv 2602.00816, 2026). ●●●●● / ●●●○○. §3 (3.1–3.2), §5.1 (Tables 2–5), §6 (Tables 6–7), §7 with Figures 2–8. §4 is optional.
- Ghorbani, Krishnan & Xiao, *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density* (ICML 2019). ●●●●○ / ●●●○○. §2 for SLQ, §3–4 for the findings. Code: `github.com/google/spectral-density`.
- Pearlmutter, *Fast Exact Multiplication by the Hessian* (Neural Computation 1994). ●●●○○ / ●●○○○. The abstract and §1–3.
- Papyan, *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra* (JMLR 2020). ●●●○○ / ●●●●○. The introduction's summary is enough.
- Sagun et al., *Empirical Analysis of the Hessian of Over-Parametrized Neural Networks* (2017). ●●○○○ / ●●○○○. Also in TD 2.6.
- Granziol, *HessFormer: Hessians at Foundation Scale* (arXiv 2505.11564, 2025). ●○○○○ / ●○○○○. The abstract.
- Wikipedia: Lanczos algorithm; Krylov subspace; Gaussian quadrature (its Golub–Welsch section); Trace, "Stochastic estimator" section (Hutchinson); Automatic differentiation. ●●●●○ / ●●○○○.

**Claude-suggested extras:**
- Ubaru, Chen & Saad, *Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature* (SIMAX 2017). ●●●○○ / ●●●○○.
- Chen, Trogdon & Ubaru, *Analysis of Stochastic Lanczos Quadrature for Spectrum Approximation* (ICML 2021). ●●○○○ / ●●●○○.
- Golub & Meurant, *Matrices, Moments and Quadrature with Applications* (Princeton, 2010). Book. ●●○○○ / ●●●●●.
- Yao, Gholami, Keutzer & Mahoney, *PyHessian: Neural Networks Through the Lens of the Hessian* (IEEE BigData 2020). ●●●○○ / ●●○○○. Code: `github.com/amirgholami/PyHessian`.
- Schioppa, Zablotskaia, Vilar & Sokolov, *Scaling Up Influence Functions* (AAAI 2022). ●●●○○ / ●●○○○.
- Agarwal, Bullins & Hazan, *Second-Order Stochastic Optimization for Machine Learning in Linear Time* (JMLR 2017). ●○○○○ / ●●●○○.
- Zhang, Chen, Ding, Li, Sun & Luo, *Why Transformers Need Adam: A Hessian Perspective* (NeurIPS 2024). ●●○○○ / ●●○○○. §3.1.
- Schraudolph, *Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent* (Neural Computation 2002). ●●○○○ / ●●○○○.
- Martens, *New Insights and Perspectives on the Natural Gradient Method* (JMLR 2020). ●●○○○ / ●●●●○. §8, the generalized Gauss–Newton matrix.

---

## Compressed path (about 15 min)

| Segment | Slides | Keep | Drop |
|---|---|---|---|
| 0–1 (1 min) | 2, 3 | the formula, $10^{18}$, the three-arrow box | slide 1 |
| 2 (0 min) | — | one sentence: jvp and vjp are exact and cost a few forward passes | slide 4 |
| 3 (2.5 min) | 5, 6 | the two exact routes in one line; finite differences and the V figure | slide 7 (one sentence: $Gv$ is one jvp and one vjp) |
| 4 (0.5 min) | — | one sentence: iterative IF solves can cost thousands of HVPs per query, hence EK-FAC for IFs | slide 8 |
| 5 (2 min) | 10, 11 | one sentence for slide 9 ("$m$ HVPs give an $m\times m$ tridiagonal $T_m$; its eigenvalues are the Ritz values"); outside-in convergence; ghosts; the memory wall | slide 9; Schioppa |
| 6 (2.5 min) | 12, 14, 15 | the quadrature identity and the SLQ figure; one sentence for slide 13 ("nodes $\theta_k$ are the Ritz values, weights $\tau_k^2$ come from $T_m$'s eigenvectors") | slides 13, 16 (say "coarsest near zero") |
| 7 (1 min) | 17 | the one picture; $C$ is the vocabulary for an LM | slide 18 |
| 8 (3 min) | 19, 21, 23 | the shard-local move; the $\epsilon$ artefact and the missing check; the block test with its caveat | slides 20, 22 (one sentence), 24 |
| 9 (2 min) | 25, 26 | where $\lambda$ sits; auditing | the gradient-projection point |
| 10 (0 min) | 28 | point at the readings | slide 27 |
