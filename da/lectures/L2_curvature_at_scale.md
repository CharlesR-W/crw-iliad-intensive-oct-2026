# L2: Curvature at scale, from Hessian-vector products to the spectrum of a 100B model (outline and speaker notes)

**Length:** about 30 min, 51 slides; many are 20-second slides (questions, takeaways, single equations). A compressed path of about 15 min is at the end.
**Slides:** [`L2_curvature_at_scale.html`](L2_curvature_at_scale.html). Arrows or space step through slides and build-ups, N shows the speaker notes on the slide, S opens reveal.js's speaker view, Esc the overview. For a PDF, open the page with `?print-pdf` and print from Chrome or Chromium. Figures are made by `assets/L2/make_figures.py`; the formulas and toy numbers are checked in `../checks/L2_checks.py`.
**Slide rules.** Every slide is one of: one equation plus one sentence; one figure plus one sentence; at most three bullets of at most twelve words; a question. Each section opens with a question slide and ends with a one-line takeaway. Numbers, citations, caveats and vocabulary are in the speaker notes (and below).
**Comes after:** L1 and the morning on DA1. **Hands off to:** DA2 (`da2_methods`), whose stop 1 is EK-FAC influence functions at LLM scale.
**Purpose:** every estimator in the DA day touches curvature, and nobody can form the $P\times P$ matrix. Every direct look at the curvature of a real network uses one primitive, multiplying it by a vector. This half hour builds the primitive from forward- and reverse-mode autodiff, shows what it buys (traces, solves, top eigenpairs, the spectral density by stochastic Lanczos quadrature), what the spectra of trained networks look like, and how a 2026 paper gets a spectrum from a 120B-parameter model. It closes with what the spectrum says about attribution. The goal is the vocabulary to read Granziol & Juarev (2026) and DA2 stop 1 alone, and a healthy suspicion of curvature approximations and of spectra that look plausible but are artefacts.

**Notation on the board (matches DA1 §2, §4 and §6).** Summed objective $L(w;\beta)=\sum_i\beta_iL_i(w)$ over $n$ examples, $L=L(\cdot;\mathbf 1)$, $P$ parameters. Hessian $H=\nabla^2L$, with eigenpairs $(\lambda_j,u_j)$. Gauss–Newton $G=\sum_iJ_i^\top H_{y,i}J_i$, with $J_i=\partial f_i/\partial w$ and $H_{y,i}$ the Hessian of the loss in the outputs; its eigenpairs are $(s_k,u_k)$ as in DA1 §6. Damping $\lambda$. The damped IF $\mathcal I_i=-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$. New today: a unit probe $v$; Lanczos basis $q_k$ and tridiagonal $T_m$ (entries $\alpha_k,\beta_k$); Ritz values $\theta_k$ and quadrature weights $\tau_k^2$; the spectral density $\rho(s)=\frac1P\sum_j\delta(s-\lambda_j)$; finite-difference step $\epsilon$; machine precision $\epsilon_{\rm mach}$. Two clashes to say out loud: the damping $\lambda$ against the eigenvalues $\lambda_j$, and the Lanczos $\beta_k$ against the data weights $\beta_i$. Eigenvalues carry an index; the damping never does.

**The toy model in the figures.** A tanh MLP with 1,573 parameters on $C=5$ Gaussian classes ($n=1000$), trained for 300 full-batch Adam steps (training loss 0.09). The figures use the curvature of the **mean** loss, so their eigenvalues are $1/n$ of the summed-loss ones; every ratio, fraction and relative error is unaffected.

Each segment lists time, slides (one line each, as shown), the board, speaker notes, and where the notebooks pick it up. Links are relative to `da/lectures/`.

---

## 0. Opening (1 min), slides 1–3

1. **Title.**
2. **Equation:** $\mathcal I_i=-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$. *"For a 1B-parameter model, the matrix inside has $10^{18}$ entries."*
3. **Diagram (builds):** a box $v\mapsto Hv$ with three arrows: solve $(H+\lambda I)x=g$; top eigenpairs; spectral density $\rho(s)$, "where the eigenvalues sit". Title: *"Three questions need only products $v\mapsto Hv$."* Line: *"$H$: the $P\times P$ Hessian of the training loss (or $G$, its Gauss–Newton part)."*

**Speaker notes.**
- Every method this afternoon approximates $(G+\lambda I)^{-1}$, replaces it by a filter of the same shape (DA1 §6), or avoids it. In bf16, $10^{18}$ numbers are two exabytes.
- Through-line: every direct look at the true curvature uses one primitive, $v\mapsto Hv$ (EK-FAC, by contrast, approximates the curvature without it). Then what it buys, what trained spectra look like, how it was done at 120B, and what it means for attribution.
- Solve: what an IF needs. Eigenpairs: sharpness, IFs in a subspace. Density: what SLQ estimates; the trace is its first moment, $\operatorname{tr}H=P\int s\,\rho(s)\,ds$.
- **Picked up in:** [DA1 §4](../da1_core.html#da1-4), [DA2 stop 1](../da2_methods.html#da2-stop-1).

---

## 1. The primitive (8 min), slides 4–15

4. **Question:** *How do you multiply by a Jacobian you can't even store?*
5. **Figure** `modes.png`: a chain of layers; "forward mode (jvp): a tangent rides along" left to right, "reverse mode (vjp): a covector flows back" right to left. *"Forward mode pushes a tangent through; reverse mode pulls a covector back."*
6. **Equation (builds):** jvp $v\mapsto Jv$, then vjp $u\mapsto J^\top u$, "for a Jacobian $J$". *"Each costs a few forward passes, at any size; backprop is a vjp."*
7. **Equation (builds), dual numbers:** $f(x_1,x_2)=x_1x_2+\sin x_1$ with $x_1=1+\varepsilon$, $x_2=2$, $\varepsilon^2=0$; then $f=(2+\sin1)+(2+\cos1)\,\varepsilon$ (the intermediate steps and decimals, $2.841+2.540\,\varepsilon$, on the board). *"The $\varepsilon$-coefficient is $\partial f/\partial x_1$: a jvp with $v=(1,0)$."*
8. **Equation:** $\nabla L(w+rv)=\nabla L(w)+r\,Hv+O(r^2)$. *"The gradient changes along $v$ at rate $Hv$."*
9. **Equation:** $Hv=\frac{d}{dr}\nabla L(w+rv)|_{r=0}$. *"And a directional derivative is exactly what a jvp computes."*
10. **Equation (builds):** $Hv=\operatorname{jvp}(\nabla L;w,v)=\nabla_w(\nabla L^\top v)$. *"Exact, for about two to three gradients' worth of work."*
11. **Equation:** $Gv=\sum_i\underbrace{J_i^\top}_{\rm vjp}\underbrace{H_{y,i}}_{\text{loss Hessian in the outputs}}\underbrace{(J_iv)}_{\rm jvp}$. *"$J_i$: the Jacobian of the outputs on example $i$. One jvp, a small multiply, one vjp."*
12. **Equation:** $Hv\approx[\nabla L(w+\epsilon v)-\nabla L(w-\epsilon v)]/2\epsilon$. *"Works inside any training loop. But which $\epsilon$?"*
13. **Figure** `fd_tradeoff.png` (schematic): truncation $\propto\epsilon^2$, round-off $\propto$ (machine precision)$/\epsilon$, their sum, the minimum $\epsilon^*\propto\epsilon_{\rm mach}^{1/3}$. *"Shrink $\epsilon$ and round-off grows; enlarge it and truncation grows."*
14. **Figure** `fd_error.png` (measured on the toy, fp32 and bf16 gradients). *"The best $\epsilon$ moves with precision: sweep it against an exact $Hv$."*
15. **Takeaway:** *$Hv$ costs a few gradients: exact by autodiff, or two gradients and a well-chosen $\epsilon$.*

**Speaker notes.**
- jvp: one forward pass with a tangent beside every value; forward-pass memory, nothing stored. vjp: one forward pass storing activations, then one backward pass; backprop is $\nabla L=J^\top1$. Code: `jax.jvp`, `jax.vjp`; `torch.func.jvp`, `torch.func.vjp`. Both exact to rounding; neither forms $J$. The dual-number numbers ($2+\sin1=2.841$, $2+\cos1=2.540$) are checked against `torch.func.jvp`.
- Pearlmutter (1994) calls the operator $\mathcal R_v\{\cdot\}=\frac{\partial}{\partial r}(\cdot)(w+rv)|_{r=0}$, with $\mathcal R_v\{\nabla L\}=Hv$. Slide 10's first form is forward-over-reverse; the second is reverse-over-reverse ("double backward", keeps the backward graph). Cost: Pearlmutter says "about as much computation" as a gradient; Schraudolph (2002) puts gradient plus product at 2–3× the gradient alone. PyTorch: `torch.func.jvp(torch.func.grad(L), (w,), (v,))`, or `torch.autograd.grad` twice with `create_graph=True`.
- $Gv$: Schraudolph (2002). $G\succeq0$ for convex output losses, so conjugate gradients on $(G+\lambda I)x=g$ converges. $H-G$ need not vanish even at a minimum (DA1 §4's $f(w)=w^2$ example).
- Finite differences, Granziol & Juarev (2026), Thm 3.1, for unit $v$: error $\le\frac{\epsilon^2}{6}\|\nabla_w(D_v^3L)\|+O(\frac{\epsilon_{\rm mach}}{\epsilon}\|\nabla L\|)$, with $D_v^3L$ the third derivative of $L$ along $v$. Leading order; a rigorous version takes the sup of that fourth derivative over the segment. Balancing gives $\epsilon^*\propto\epsilon_{\rm mach}^{1/3}$, best error $\propto\epsilon_{\rm mach}^{2/3}$. Their Remark 3.2: fp32 $\epsilon^*\sim10^{-3}$–$10^{-2}$ with errors $\sim10^{-5}$; bf16 $\epsilon^*\sim10^{-1}$ with errors $\sim10^{-2}$.
- The toy: best $\epsilon\approx1.5\times10^{-2}$ in fp32 (error $3\times10^{-5}$), $0.2$ with bf16-rounded gradients (error $3\times10^{-3}$), autodiff fp32 $Hv$ at $2\times10^{-7}$. The bf16 curve rounds only the two gradients.
- $\epsilon$ is a step along a unit vector in $\mathbb R^P$: each weight moves by about $\epsilon/\sqrt P$ (about $4\times10^{-4}$ in the toy; about $10^{-8}$ for $\epsilon=10^{-4}$ at 160M parameters). Step sizes are not comparable across models. With bf16 weights a $10^{-8}$ perturbation would round away, so it has to live in higher precision.
- The gradient central difference is exactly the average of $H(w+tv)v$ over $t\in[-\epsilon,\epsilon]$: a large $\epsilon$ gives curvature averaged along a segment, not a noisy $Hv$. (Their Thm 3.3 is the triangular-kernel version for the second difference of $L$.)
- **Picked up in:** [DA1 §4](../da1_core.html#da1-4) (why $G$), [TD 2.6 §6.1](../../td/td2_6_curvature.html#td26-1) ($H=G+{}$residual), [DA2 Code 1](../da2_methods.html#da2-ekfac-code) (uses `torch.func`).

---

## 2. What products buy: traces, solves, Lanczos (6 min), slides 16–26

16. **Question:** *You have a function that returns $Hv$. What can you compute with it?* (Spoken: how would you estimate the trace of a matrix with $10^{18}$ entries from 10 products?)
17. **Equation**, "A trace from ten products": $\operatorname{tr}H=\mathbb E_v[v^\top Hv]$ for $\mathbb E[vv^\top]=I$. *"Hutchinson: random $\pm1$ probes $v$, one $Hv$ each."*
18. **Bullets (build):** An IF needs one solve per query. / Iterative solvers can need thousands of $Hv$ per solve. / That cost is why DA2 uses EK-FAC for IFs.
19. **Question:** *How do you find the top eigenvalues from $Hv$ alone?*
20. **Figure** `krylov.png`: $v$, $Hv$, $H^2v$ (normalized) tilting toward $u_1$. *"Keep all of $v,Hv,H^2v,\dots$: their span is the Krylov subspace."*
21. **Equation:** $T_m=Q_m^\top HQ_m$ (tridiagonal, entries $\alpha_k,\beta_k$). *"$H$ restricted to the Krylov basis $Q_m$; its eigenvalues, the Ritz values $\theta_k$, approximate $H$'s."*
22. **Equation (builds), "A 4×4 example: watch the Ritz values":** $m=1$: 0.55; $m=2$: 0.30, 4.82; $m=3$: 0.23, 0.81, 5.00; $m=4$: 0.2, 0.5, 1, 5. *"True eigenvalues $0.2, 0.5, 1, 5$: the extremes arrive first."*
23. **Figure** `ritz.png` (toy MLP, full reorthogonalization; brackets mark outliers and bulk). Title: *"Outliers converge fast; the bulk is slow."* Caption: *"Toy MLP: the five outliers are done by about $m=12$."*
24. **Figure** `ritz_ghosts.png` (no reorthogonalization; spurious Ritz values circled). *"Lost orthogonality creates spurious Ritz values and copies of converged ones."*
25. **Equation:** one vector $=P$ numbers; $P=10^{11}$ in bf16 $\Rightarrow$ 200 GB. *"Keeping $m$ of them is out of reach at LLM scale, so $m$ stays small."*
26. **Takeaway:** *Lanczos finds the outliers in a few steps; the bulk stays a blur.*

**Speaker notes.**
- Slide 16 opens three answers: a trace (cheap), a solve (expensive), the top eigenvalues (Lanczos).
- Hutchinson: for i.i.d. $\pm1$ probes, $\operatorname{Var}(v^\top Hv)=2(\|H\|_F^2-\sum_jH_{jj}^2)$; Gaussian probes give $2\|H\|_F^2$. PyHessian (Yao et al. 2020) packages power iteration, Hutchinson and SLQ for PyTorch.
- Solves: Koh & Liang (2017) used conjugate gradients and LiSSA (Agarwal, Bullins & Hazan 2017). Grosse et al. (2023, §2.2.1): CG is full-batch; LiSSA "often" needs thousands of iterations, each at least a gradient. CG's count depends on conditioning and tolerance. EK-FAC came from optimization (George et al. 2018); Grosse et al. applied it to IFs.
- Power iteration (the usual answer to slide 19) finds the largest-*magnitude* eigenpair, with error shrinking like $|\lambda_2/\lambda_1|^k$: slow when the top eigenvalues are close, as outliers often are. Sharpness is the largest *algebraic* eigenvalue; use Lanczos targeting the top end.
- Krylov sketch: a 2×2 matrix with eigenvalues 3 and 1; the angle to $u_1$ goes 60°, 30°, 11°. Power iteration keeps only the last vector; Lanczos keeps the span $\mathcal K_m=\operatorname{span}\{v,Hv,\dots,H^{m-1}v\}$.
- Lanczos: $Q_m=[q_1,\dots,q_m]$ is an orthonormal basis of $\mathcal K_m$ built by $Hq_k=\beta_{k-1}q_{k-1}+\alpha_kq_k+\beta_kq_{k+1}$ from $q_1=v$. Each new product is orthogonalized against only the two previous vectors; symmetry does the rest in exact arithmetic. Cost: $m$ HVPs plus $O(mP)$ vector work.
- 4×4 example: $H=R\,\mathrm{diag}(0.2,0.5,1,5)R^\top$ with a fixed random rotation, $v=(1,1,1,1)/2$; numbers from the checks file. Why outside-in: Ritz values are optimal over polynomials of degree $m-1$, and a polynomial can single out an isolated eigenvalue far more easily than one in a cluster; the Kaniel–Paige bounds (refined by Saad) make this precise with Chebyshev polynomials.
- Ghosts: the $q_k$ lose orthogonality once a Ritz value converges; by $m=60$ the toy has four copies of $\lambda_{\max}$. Full reorthogonalization keeps all of $Q_m$ and costs $O(m^2P)$ extra work; at LLM scale memory rules it out, and without it ghosts keep $m$ small (the 120B run uses $m=10$).
- Code you have seen: TD 2.3 §3.5's `top_eig_lanczos` calls scipy's `eigsh` (implicitly restarted Lanczos) on double-backward HVPs. The eigenpairs route to IFs: Schioppa et al. (2022) get top eigenpairs by Arnoldi (the non-symmetric version) and compute IFs in that subspace, up to a 300M-parameter ViT.
- **Picked up in:** [DA2 stop 1](../da2_methods.html#da2-stop-1), [TD 2.3 §3.5](../../td/td2_3_edge_of_stability.html#td23-5).

---

## 3. From Lanczos to the density: stochastic Lanczos quadrature (5 min), slides 27–34

27. **Question:** *Lanczos found the outliers. Can the same $T_m$ give the whole spectrum?*
28. **Equation:** $v^\top f(H)v=\sum_j(v\cdot u_j)^2f(\lambda_j)$. *"Each eigenvalue, weighted by the probe's overlap with its eigenvector."*
29. **Figure** `probe.png` (sticks at $\lambda_j$ with heights $(v\cdot u_j)^2$). *"$v^\top f(H)v=\int f\,d\mu_v$: a stick at each $\lambda_j$, height $(v\cdot u_j)^2$."*
30. **Equation:** $e_1^\top f(T_m)e_1=\sum_{k=1}^m\tau_k^2f(\theta_k)$. *"Nodes $\theta_k$: the Ritz values; weights $\tau_k^2$ from $T_m$'s eigenvectors; exact to degree $2m-1$."*
31. **Figure** `gauss_toy.png` (3 sticks at 0, 1, 4; the 2-node Gauss rule). *"Two nodes reproduce moments 0 to 3 of the three sticks exactly."*
32. **Equation:** $\hat\rho(s)=\frac1{n_v}\sum_{\rm probes}\sum_k\tau_k^2\delta(s-\theta_k)$. *"Averaging $n_v$ random probes estimates $\rho$; finite $m$ adds quadrature error."*
33. **Figure** `slq.png` (exact vs SLQ at $m=10$, ten probes). *"Toy MLP: at $m=10$ the outliers are right and the bulk is lumpy."*
34. **Takeaway:** *A few probes times a few Lanczos steps give a spectral density: sharp outliers, blurry bulk.*

**Speaker notes.**
- Slides 28–30 are the derivation: what a probe sees is a sum over eigenpairs; that sum is an integral against the stick measure $\mu_v$; Lanczos gives its Gauss rule. Golub–Welsch: nodes are the Ritz values $\theta_k$, weights $\tau_k^2=(e_1^\top y_k)^2$, the squared first components of $T_m$'s eigenvectors $y_k$. Equivalently $e_1^\top T_m^ke_1=v^\top H^kv$ for $k\le2m-1$. Reference: Golub & Meurant, *Matrices, Moments and Quadrature*; Wikipedia's Gaussian quadrature page. Physicists: this is the recursion method, with $\alpha_k,\beta_k$ the continued-fraction coefficients of a local density of states. *(Room for the instructor's framing.)*
- Slide 33 shows only $m=10$; at $m=30$ or $90$ the SLQ curve lies on the exact one.
- Probe figure: a synthetic spectrum (twelve bulk eigenvalues, three outliers) and one random probe; heights sum to 1.
- Gauss toy: atoms at 0, 1, 4 with weights 0.5, 0.3, 0.2; $T_2$ gives nodes 0.249 and 3.790 with weights 0.760 and 0.240; moments 0–3 are 1, 1.1, 3.5, 13.1 for both; moment 4 is 51.5 against 49.6.
- SLQ (Ubaru, Chen & Saad 2017): for $v=z/\sqrt P$ with i.i.d. $\pm1$ entries, $\mathbb E[(v\cdot u_j)^2]=1/P$, so $\mathbb E[\mu_v]=\rho$ exactly. Smooth with a Gaussian of width $\sigma$ to plot. Two error sources: quadrature ($m$) and Monte Carlo ($n_v$). Moments up to degree $2m-1$ are exact in expectation, so their error is all Monte Carlo: in the toy, moments 1, 2, 4 are off by $+1.7\%$, $-2.3\%$, $-5.4\%$ at every $m$ from 10 to 90 (same probes).
- For probes uniform on the sphere, the number of probes in Chen, Trogdon & Ubaru's (2021) Wasserstein bound falls roughly like $\log P/P$ (never below one): huge models need few probes. Ghorbani et al. (2019) used $m=90$ with Gaussian probes; Granziol & Juarev $m=10$ at 120B. Code: `google/spectral-density`, PyHessian.
- What SLQ resolves: the outliers and the bulk's shape, not individual bulk eigenvalues. Fraction of eigenvalues with $|s|<\sigma$ in the toy: 0.891 exactly, 0.890 by SLQ at $m=90$, 0.78 at $m=10$. The mass near zero, where an IF's damping acts, is where the rule is coarsest. The mean eigenvalue has no quadrature error and is unbiased, but check its spread across probes: for a spectrum dominated by a few outliers, ten probes can leave tens of percent.
- **Picked up in:** [DA1 §6](../da1_core.html#da1-6) (filters act eigenvalue by eigenvalue; the density says how many sit where).

---

## 4. What trained spectra look like (2 min), slides 35–38

35. **Question:** *What does the Hessian spectrum of a trained network look like?*
36. **Bullets (build):** A bulk of eigenvalues near zero. / About $C$ large outliers, $C$ the number of classes. / The outliers come from class structure in $G$.
37. **Equation:** $C=|\text{vocabulary}|\approx10^4$–$10^5$. *"'A few outliers' may be tens of thousands; $m=10$ cannot resolve them."*
38. **Takeaway:** *Classifiers: a bulk plus about $C$ outliers. Language models: still an open case.*

**Speaker notes.**
- Sagun et al. (2017): on $k$-cluster data the number of outliers above the gap equals the number of classes. Ghorbani et al. (2019): SLQ up to ImageNet (ResNet-18, Inception-V3 with 27M parameters); without BatchNorm, outliers appear fast and the gradient concentrates in their span; with it, both almost vanish; large negative eigenvalues early in training vanish within 1% of steps. Papyan (2018, JMLR 2020): about $C$ outliers from class means in $G$, a weaker $C(C-1)$ layer below; the bulk from within-class variation and $H-G$.
- Papers report $C$, $C-1$ or about $C^2$ outliers depending on what they count. Our toy ($C=5$), relative to the largest: 1, 0.97, 0.68, 0.42, 0.35, then 0.23. A real gap, not a dramatic one.
- Transformers are also uneven by block: Zhang et al. (NeurIPS 2024) ran SLQ per parameter block and found very different spectra across blocks in transformers, not in CNNs.
- **Picked up in:** [TD 2.6 §6.3](../../td/td2_6_curvature.html#td26-3) and [§6.4](../../td/td2_6_curvature.html#td26-4).

---

## 5. The 120B-parameter spectrum: Granziol & Juarev 2026 (5 min), slides 39–44

39. **Question:** *How do you get $Hv$ from a 120B model sharded across GPUs?*
40. **Figure** `fsdp.png`: four GPUs each holding $w_k,v_k$; step 1 perturb $+\epsilon v_k$ in place and run a gradient pass; step 2 $-2\epsilon v_k$ and another pass; step 3 restore and form $(Hv)_k$ on each shard. *"Two ordinary gradient passes: a Lanczos step costs about two training steps."*
41. **Bullets (build), finding 1:** $\epsilon=10^{-5}$ or $10^{-2}$: no separated outliers. / $\epsilon=10^{-4}$ or $10^{-3}$: clear outliers. / Which is right? Only an exact-$Hv$ check can say.
42. **Bullets (build), finding 2, "A 120B model shows large negative curvature":** gpt-oss-120b: ten Lanczos steps, no $\epsilon$ sweep. / Real curvature, or an artefact of $\epsilon$ or Lanczos? / The authors call it "only indicative".
43. **Equation, finding 3, "Per-layer (block-diagonal) curvature can be wrong by $O(1)$":** $\|(Hv)_b-H_{bb}v_b\|/\|(Hv)_b\|\approx1$. *"$b$: one transformer layer. Dropping cross-layer blocks loses about as much as it keeps."*
44. **Takeaway:** *At 120B the primitive still works; its knobs still need checking.*

**Speaker notes.**
- arXiv 2602.00816, §3.1 and Algorithm 1. Under FSDP (ZeRO-3) each GPU stores a slice of every parameter tensor; the authors' diagnosis is that naive HVPs force expensive all-gathers. The only extra communication is scalar all-reduces for norms and dot products; $T_{Hv}\approx2T_{\rm grad}$. The gain is compatibility with the training stack, not FLOPs.
- Cost (§5.1): Qwen-0.6B on one node, 1 → 8 GPUs gives 5.74× per HVP (Table 3); Lanczos bookkeeping adds 1.3–1.8% (Table 2); on a 32B model, Table 5 reports about 4× over HessFormer (89.6 s at batch 32 against 22.2 s at batch 80). HessFormer (Granziol 2025), the previous record: Pearlmutter HVPs on one 8-GPU node, SLQ on a 70B DeepSeek distilled model.
- Reading the paper: "exact" means the true $H$, not a structured approximation; Figs 2–8 plot Ritz values against weights ("Gamma", "Ritz weights"), our $\theta_k$ and $\tau_k^2$; "smoothing parameter $\epsilon$" is the finite-difference step; their $\theta$ is the parameters, and Pearlmutter's method is called "reverse-over-forward"; their loss is a dataset average. §4's Thm 4.1 assumes unbiased, independent finite-difference errors, while Thm 3.1 gives an $O(\epsilon^2)$ bias, so read §4 as heuristic.
- Finding 1 (Pythia-160M, §7.1, Figs 3–6; the bad ones "resemble a random matrix"). The authors attribute the failures to round-off and to averaging along the segment. But the "good" $\epsilon$ is the one that shows the expected outliers, and the check from slide 14 (finite difference against an exact HVP on a probe) is not reported, though it would be cheap at 160M. Their Remark 3.2 puts fp32 $\epsilon^*$ at $10^{-3}$–$10^{-2}$, yet their sweet spot is $10^{-4}$–$10^{-3}$: constants and $\sqrt P$ move the optimum. The fp32/bf16 labels in their figures refer to Lanczos-vector storage; bf16 storage is enough. 30× more data leaves the spectrum essentially unchanged at fixed $\epsilon$ (Fig 7). They report Pythia-1.4B more stable in $\epsilon$ (Fig 8: $3\times10^{-4}$ to $10^{-2}$): one pair of models.
- Finding 2 (§7, Fig 2): wikitext-2, sequence length 64, $\epsilon=10^{-4}$; the authors call it "only indicative". Why 10 iterations: HessFormer's paper says stay below 20 without reorthogonalization and often uses 10; reorthogonalizing at 120B would need 240 GB per bf16 basis vector. Three readings, side by side: the authors' (the data, as tokenized, has not been memorised); a pretrained checkpoint need not be near a minimum of this particular loss; numerics (no $\epsilon$ sweep, 10 iterations, no reorthogonalization). Negative eigenvalues live in $H-G$ only, since $G\succeq0$. No probe count, hardware or code is given for this run; HessFormer's code is public.
- Finding 3 (§6): per transformer layer $b$, unit Gaussian $v$, true Hessian. Qwen-0.6B, 28 layers: $0.94\pm0.06$, cosine $0.31\pm0.12$ (Table 6). DeepSeek-Coder 1.3B after fine-tuning, 24 layers: $1.21\pm0.15$, cosine $0.00\pm0.13$ (Table 7). Predicting zero gives exactly 1. For DeepSeek-Coder they report a higher final-block cosine (§6.2). Caveat that changes a conclusion: this is $H$ with random probes; EK-FAC approximates $G$, and an IF applies $(G+\lambda I)^{-1}$ to gradients, which may live elsewhere. A warning, not a verdict. In the other direction, EK-FAC's blocks are single weight matrices, finer than a layer, so the test is if anything generous to the block assumption. *(Room for the instructor's framing.)*
- Scale, for reference (largest model in each paper): Ghorbani et al. 2019, $2.7\times10^7$ (Inception-V3), exact HVP + SLQ; Schioppa et al. 2022, $3\times10^8$ (ViT-L/32), Arnoldi; Grosse et al. 2023, $5.2\times10^{10}$, EK-FAC (no HVPs); HessFormer 2025, $7\times10^{10}$, Pearlmutter HVP on one node; Granziol & Juarev 2026, $1.2\times10^{11}$ (gpt-oss-120b), finite-difference HVP under FSDP, $m=10$.
- **Picked up in:** [DA2 stop 1](../da2_methods.html#da2-stop-1).

---

## 6. Back to attribution (3 min), slides 45–49

45. **Question:** *What does the spectrum tell you about influence?*
46. **Equation**, "In $G$'s eigenbasis $(s_k,u_k)$, each direction gets its own weight": $\mathcal I_i=-\sum_k(u_k\cdot\nabla\phi)(u_k\cdot\nabla L_i)/(s_k+\lambda)$. *"Far below $\lambda$: a scaled gradient dot product. Near or above: curvature-weighted."*
47. **Figure** `filter.png` (histogram of the toy $G$'s eigenvalues with two dampings marked). Title: *"Where $\lambda$ sits decides which directions count as curvature."* Caption: *"Left of $\lambda$: a scaled gradient dot product. Right of it: curvature-weighted."*
48. **Bullets (build):** EK-FAC assumes block-diagonal curvature. / Dropping cross-layer blocks already costs $O(1)$ in $Hv$. / Check against exact products: $Gv$ for EK-FAC, $Hv$ for Hessians.
49. **Takeaway:** *The spectrum shows what your damping and your approximation are doing.*

**Speaker notes.**
- Attribution wants the spectrum of $G$, not $H$. $Gv$ is the same primitive (slide 11), so everything applies; the 2026 paper measured $H$.
- $(s_k,u_k)$: eigenpairs of $G$, as in DA1 §6. Directions with $s_k\ll\lambda$ contribute $-\frac1\lambda(u_k\cdot\nabla\phi)(u_k\cdot\nabla L_i)$: the TracIn-like part. At $s_k=0.9\lambda$, replacing the weight by $1/\lambda$ is already a 90% error. Which directions matter for a given pair depends on both projections and on the weights; the density gives neither projection. TD 2.6 §6.4 finds the gradient concentrated in the top Hessian subspace, where the weights are smallest, so concentration alone does not say which part dominates.
- Filter figure: toy $G$ (mean loss), log axis; the bulk spans about seven decades. Grosse et al. set $\lambda=0.1\times$ the mean EK-FAC eigenvalue, layer by layer; on the toy's exact $G$ that puts 78% of eigenvalues below $\lambda$. A $\lambda$ between bulk and outliers puts 99.7% below. At scale both are computable: SLQ sees the outliers well, and the mean $\operatorname{tr}G/P$ comes from Hutchinson on $Gv$ (check its spread across probes). Where $0.1\times$mean falls inside the bulk is what SLQ resolves least well. DA2's Code 1 sweeps $10^{-3}$ to $10^{-1}$ times the mean eigenvalue; DA1 §5's exercise uses the median Hessian eigenvalue.
- $\lambda$ also has a meaning: DA1 §4's proximal penalty. Choosing it from the spectrum is a statement about where you trust the quadratic model; choosing it from the proximal target is a statement about the question. *(Room for the instructor's framing.)*
- Auditing: EK-FAC is blocks plus a Kronecker-factored eigenbasis with refitted eigenvalues. DA2's Code 1 measures the IF-level error on a 147-parameter net: rank correlation with the exact damped Gauss–Newton IF is 0.65–0.73 for block-diagonal $G$ and 0.26–0.45 for K-FAC and EK-FAC, across $\lambda$ from $10^{-3}$ to $10^{-1}$ times the mean eigenvalue. So there the Kronecker step costs more than the block step. Both are measurements on specific models, not theorems.
- **Picked up in:** [DA1 §6](../da1_core.html#da1-6), [DA2 stop 1](../da2_methods.html#da2-stop-1), [DA2 Code 1](../da2_methods.html#da2-ekfac-code).

---

## 7. Summary and readings (1 min), slides 50–51

50. **Summary (builds):** $v\mapsto Hv$ costs a few gradients. / Lanczos and SLQ: outliers fast, a density from few probes. / Check step sizes and approximations against exact products.
51. **Readings** (five, scored Importance / Length).

Regular (on the slide):
- Granziol & Juarev, *Hessian Spectral Analysis at Foundation Model Scale* (arXiv 2602.00816, 2026). ●●●●● / ●●●○○. §3 (3.1–3.2), §5.1 (Tables 2–5), §6 (Tables 6–7), §7 with Figures 2–8. §4 is optional.
- Ghorbani, Krishnan & Xiao, *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density* (ICML 2019). ●●●●○ / ●●●○○. §2 for SLQ, §3–4 for the findings. Code: `github.com/google/spectral-density`.
- Pearlmutter, *Fast Exact Multiplication by the Hessian* (Neural Computation 1994). ●●●○○ / ●●○○○. The abstract and §1–3.
- Papyan, *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra* (JMLR 2020). ●●●○○ / ●●●●○. The introduction's summary is enough.
- Wikipedia: Lanczos algorithm; Gaussian quadrature (its Golub–Welsch section). ●●●●○ / ●●○○○.

Also regular (notes and here): Sagun et al., *Empirical Analysis of the Hessian of Over-Parametrized Neural Networks* (2017), ●●○○○ / ●●○○○, also in TD 2.6; Granziol, *HessFormer: Hessians at Foundation Scale* (arXiv 2505.11564, 2025), ●○○○○ / ●○○○○, the abstract; Wikipedia: Krylov subspace; Trace, "Stochastic estimator" section (Hutchinson); Automatic differentiation.

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

| Segment | Show | Skip (say in one sentence if needed) |
|---|---|---|
| 0 (1 min) | 2, 3 | 1 |
| 1 (3.5 min) | 4, 6, 9, 10, 12, 14, 15 | 5, 7, 8 (say "$Hv$ is the derivative of the gradient along $v$"), 11 ("$Gv$ is one jvp and one vjp"), 13 |
| 2 (3 min) | 18, 19, 21, 23, 24, 26 | 16, 17 (Hutchinson in one sentence), 20, 22, 25 ("the basis does not fit in memory, so $m$ stays small") |
| 3 (2.5 min) | 27, 28, 30, 32, 33 | 29, 31 ("nodes $\theta_k$ are the Ritz values; weights $\tau_k^2$ come from $T_m$'s eigenvectors"), 34 |
| 4 (1 min) | 36, 37 | 35, 38 |
| 5 (2.5 min) | 40, 41, 43 | 39, 42 (one sentence), 44 |
| 6 (1.5 min) | 46, 47, 48 | 45, 49 |
| 7 (0 min) | 51 | 50 |
