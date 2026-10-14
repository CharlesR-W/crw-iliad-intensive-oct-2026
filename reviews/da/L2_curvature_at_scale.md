# L2 "Curvature at scale": verification log (2026-10-14)

Deliverables: `da/lectures/L2_curvature_at_scale.md` (outline and speaker notes, source of truth), `da/lectures/L2_curvature_at_scale.html` (28 slides, reveal.js 5.1.0 + KaTeX 0.16.9 from cdnjs), `da/lectures/assets/L2/make_figures.py` and its five PNGs, `da/checks/L2_checks.py`. Links added: one `<li>` in the DA section of `index.html`, one sentence in `da/README.md`.

## 1. Formulas (`da/checks/L2_checks.py`, ALL PASS, ~5 s)

On a 4→6→3 tanh net with summed cross-entropy (float64), and on a 200×200 random symmetric matrix with three planted outliers:

- jvp = $Jv$ and vjp = $J^\top u$ against `jacrev`'s explicit Jacobian.
- Reverse-over-reverse and forward-over-reverse $Hv$ against `torch.func.hessian`.
- $Gv$ via jvp → $H_y$ → vjp against the explicit $\sum_iJ_i^\top H_{y,i}J_i$; $G\succeq0$; $H-G\neq0$ on this net ($\|H-G\|/\|H\|=0.84$).
- Finite differences (sympy): gradient central-difference error $=\frac{\epsilon^2}{6}g'''+O(\epsilon^4)$; the gradient central difference equals the box average $\frac1{2\epsilon}\int_{-\epsilon}^{\epsilon}H(w+tv)v\,dt$ exactly; the function second difference equals the triangular-kernel average (Granziol & Juarev Thm 3.3), by quadrature to $10^{-12}$; $\epsilon^*=(3\epsilon_{\rm mach}c_1/c_3)^{1/3}$ and the error at $\epsilon^*$ scales as $\epsilon_{\rm mach}^{2/3}$; orders of magnitude match Remark 3.2 (fp32 $\epsilon_{\rm mach}^{1/3}=4.9\times10^{-3}$, $^{2/3}=2.4\times10^{-5}$; bf16 $2^{-8}$: $0.16$, $0.025$).
- Hutchinson: Rademacher estimator unbiased and variance $=2(\|A\|_F^2-\sum A_{jj}^2)$ by exact enumeration of all $2^5$ sign vectors; Gaussian variance $2\|A\|_F^2$ by Monte Carlo (2%).
- Lanczos: $Q_m^\top HQ_m=T_m$; the three-term recurrence $HQ_m=Q_mT_m+\beta_mq_{m+1}e_m^\top$; $Q_m$ spans the Krylov space; top Ritz values converge to the outliers by $m=12$ (rtol $10^{-4}$) while interior ones have not.
- Golub–Welsch: the $m$-point rule from $T_m$ reproduces moments $0..2m-1$ of $\mu_v$ to $10^{-9}$ relative and not moment $2m$; with $m=P$ the nodes and weights equal `eigh`'s eigenvalues and $(v\cdot u_j)^2$; Hutchinson for $\frac1P\operatorname{tr}H^2$ (3%).
- Power iteration rate $|\lambda_2/\lambda_1|$ (0.849 measured vs 0.846).
- Damped IF split into $-\frac1\lambda\times$(bulk gradient dot product) + outlier terms, against the exact solve (1%).
- Arithmetic: $(10^9)^2=10^{18}$; one bf16 vector of $10^{11}$ parameters = 200 GB, of $1.2\times10^{11}$ = 240 GB. $E[\mu_v]=\rho$ for $v=z/\sqrt P$ (one line: $E[(v\cdot u_j)^2]=1/P$).

## 2. Figures (`make_figures.py`, ~9 s of compute, ~11 s wall on CPU)

Toy: tanh MLP 10→32→32→5, $P=1573$, $n=1000$, 5 overlapping Gaussian classes, 300 full-batch Adam steps on the **mean** cross-entropy (train loss 0.093, accuracy 0.97). Exact Hessian by vmapped forward-over-reverse HVPs, then `eigh`. The script prints every number the outline quotes:

- Top eigenvalues / $\lambda_{\max}$: 1, 0.965, 0.676, 0.415, 0.351, then 0.229 (gap $\lambda_5/\lambda_6=1.53$). 89% of eigenvalues have $|\lambda|<1\%$ of $\lambda_{\max}$.
- FD error V: fp64 best $\epsilon=2.2\times10^{-5}$ (error $3.6\times10^{-11}$); fp32 best $1.5\times10^{-2}$ ($3.0\times10^{-5}$); bf16-rounded gradients best $0.22$ ($2.7\times10^{-3}$); autodiff fp32 $1.9\times10^{-7}$. Truncation slope 2.00; fp32 round-off slope $-0.88$. (Codex noted the fp32 optimum shifts slightly with CPU threading.)
- Lanczos: orthogonality to $10^{-15}$ with full reorthogonalization; at $m=60$, 1 copy of $\lambda_{\max}$ with reorthogonalization, 4 without.
- Gauss rule $m=10$ moments 0..19 exact to $8.5\times10^{-15}$.
- SLQ ($n_v=10$ Rademacher, same probes for every $m$): moments 1, 2, 4 off by $+1.65\%$, $-2.32\%$, $-5.37\%$ at every $m$ (pure Monte Carlo); top node equals $\lambda_{\max}$ at $m=10,30,90$; mass with $|s|<\sigma$: exact 0.891, $m=90$ 0.890, $m=30$ 0.908, $m=10$ 0.782.
- $G$ (mean loss): top eigenvalues within ~5% of $H$'s; $\|H-G\|_F/\|H\|_F=0.15$. Damping $0.1\times$mean eigenvalue puts 78.1% of eigenvalues below it; $\lambda$ at the geometric mean of $s_5,s_6$ puts 99.7% below.
- Palette checked with the dataviz validator (passes; one protan pair at ΔE 7.2, so every series also has a line style and a legend). No dual axes.

## 3. Paper claims (each source fetched; two research subagents read the PDFs/HTML and returned quotes with locations)

**Granziol & Juarev, arXiv 2602.00816v1 (31 Jan 2026; only version).** Confirmed with quotes: FD central difference Eq. (1); FSDP ZeRO-3, in-place perturbation of the local DTensor shard, two FSDP gradient passes, "at most a scalar all-reduce"; $T_{HvP}=2T_{\rm grad}+T_{\rm vec}$; Thm 3.1 bound and $\epsilon^\star\propto\epsilon_{\rm mach}^{1/3}$; Remark 3.2 numbers; Thm 3.3 triangular kernel for the *function* second difference; Table 3 (Qwen-0.6B 1→8 GPUs 5.74×); Table 2 (Lanczos overhead 1.3% and 1.8%); Table 5 (HessFormer 89.57 s at batch 32, theirs 22.21 s at batch 80, "(4x)"); Figs 3–6 (Pythia-160M, $10^{-5}$ and $10^{-2}$ "resembles a random matrix", $10^{-4}$ and $10^{-3}$ clear outliers); Fig 7 (30% vs 1% split); Fig 8 (Pythia-1.4B, panels $10^{-2}$, $3\times10^{-3}$, $10^{-3}$, $3\times10^{-4}$; text claims $10^{-4}$–$10^{-3}$) and "larger models … are more stable"; Fig 2 (120B, wikitext-2, sequence length 64, $\epsilon=10^{-4}$, 10 iterations, no reorthogonalization, "only indicative", their memorisation reading); Tables 6–7 (0.941±0.059 / 0.311±0.120 on Qwen-0.6B, 28 blocks; 1.214±0.150 / 0.00443±0.127 on "Deepseek 1.3Bn coder", 24 blocks; true Hessian; blocks = transformer layers; final-layer exception); fp32/bf16 labels refer to Lanczos-vector storage; "reverse-over-forward", "Gamma"/"Ritz weights", "smoothing parameter ε", "exact iHVPs", Thm 4.1's "unbiased and independent" assumption, "naïve HVPs force expensive all-gathers" (checked by grep of the PDF text).

**Not in the paper, so not on the slides:** the brief's "44×" (Table 5 supports about 4×); the number of probes; GPU type or node count for the 120B run; code. "72% parallel efficiency" is our arithmetic and was left off. The paper's reason for 10 iterations is not stated; the slides use HessFormer's ("prudent to keep the number of iterations less than 20 … often use the value 10").

**HessFormer, arXiv 2505.11564v1 (16 May 2025).** Single author; Pearlmutter double-backward HVPs via HF `device_map="auto"`; one node of 8 H200s (the intro says "A200"); "70bn Deepseek-Llama distilled" (exact checkpoint not named); SLQ without reorthogonalization; code at github.com/PureStrength-AI/HessFormer.

**Others** (fetched; quotes in the subagent report): Pearlmutter 1994 ($\mathcal R_v$ definition; "about as much computation … as a gradient", no constant); Schraudolph 2002 §4.3 ("two to three times as expensive as calculating the gradient alone"; GN product forward-then-reverse); Ghorbani et al. 2019 ($m=90$, Gaussian probes; $k=10$ probes only in the small MNIST validation figures, so not quoted as their setting; BatchNorm findings; negative eigenvalues gone within <1% of steps; Inception-V3 27M is their largest; ResNet-18 on ImageNet); `google/spectral-density` exists, archived, TF and JAX; Sagun et al. 2017 (outlier count equals number of classes, on synthetic $k$-cluster data); Papyan 2018 (outliers attributable to $G$, bulk energy to the residual; VGG11 28M, no ImageNet) and 2020 ($C(C-1)$ cross-class outliers, $C-1$ plus one larger, within-class knockout shrinks the bulk); Ubaru, Chen & Saad 2017 SIMAX 38(4); Chen, Trogdon & Ubaru 2021 (Wasserstein bound; probes uniform on the sphere; count $\lceil4(n+2)^{-1}t^{-2}\ln(2n\eta^{-1})\rceil$); PyHessian (power iteration, Hutchinson, SLQ; IEEE BigData 2020); Schioppa et al. 2022 (Arnoldi, up to 300M ViT-L/32); Agarwal, Bullins & Hazan 2017; Koh & Liang 2017 §3 (CG and LiSSA on HVPs); Grosse et al. 2023 (52B; "λ_ℓ = 0.1 × mean(Λ_ℓ)" for EK-FAC, §5; CG full-batch and LiSSA "often thousands of iterations", §2.2.1); Zhang et al. 2024 (author order Zhang, Chen, Ding, Li, Sun, Luo; NeurIPS; SLQ per block; heterogeneity in transformers not CNNs); Martens 2020 §8 GGN; Golub & Meurant (Princeton 2010); Golub & Welsch 1969; Avron & Toledo 2011 Lemma 2.1 for the Rademacher variance.

**Corrections to the brief from verification.** Grosse et al. choose $\lambda=0.1\times$ the mean EK-FAC eigenvalue per layer, not "by validation". DA2 Code 1 sweeps $\lambda\in\{10^{-3},10^{-2},10^{-1}\}\times$ mean eigenvalue (it does not use "$0.1\times$" alone); DA1 §5 uses the median Hessian eigenvalue. DA2 stop 1 has no "why EK-FAC" paragraph on CG/LiSSA, so the slides cite Grosse et al. §2.2.1 directly and link `#da2-stop-1`. The brief's Ghorbani "$n_v=10$" is only for their validation figures. "Kaniel–Paige–Saad" is written "Kaniel–Paige (refined by Saad)". Wikipedia has no Hutchinson page; the link is Trace (linear algebra) § Stochastic estimator.

**Course cross-checks.** DA2 Code 1 output table read from the built HTML: Spearman block-diagonal 0.65/0.73/0.72, K-FAC 0.29/0.37/0.45, EK-FAC 0.26/0.34/0.45 (quoted as 0.65–0.73 and 0.26–0.45). TD 2.2's `top_eig_lanczos` lives in §2.5 (`#td22-5`) and uses `scipy.sparse.linalg.eigsh` on double-backward HVPs.

## 4. Links, anchors, rendering

- Every relative link and `src` in the `.md` and `.html` resolves, and every `#anchor` exists in the target HTML (`da1-4`, `da1-6`, `da2-stop-1`, `da2-ekfac-code`, `td26-1`, `td26-3`, `td26-4`, `td22-5`).
- Every external URL fetched: all arXiv abs pages, Wikipedia pages (and the `#Stochastic_estimator` anchor), `github.com/google/spectral-density`, and the seven cdnjs files return 200; the three DOIs return 302 to the publisher (the publisher pages refuse curl with 403).
- Headless Chromium (driven over CDP): 28 slides, 264 KaTeX expressions, 0 KaTeX errors, 0 broken images, no console errors; pressing N sets `showNotes` and shows the notes panel; `?print-pdf` produces a 28-page PDF; every slide screenshotted and inspected for overflow at 1280×720.
- Smallest text on the slides is 22px (captions, readings); speaker notes 20px.
- Known limitation: reveal.js and KaTeX come from the CDN, so the deck needs a network connection on first load (the brief's "offline except math" and "reveal.js from cdnjs" pull in opposite directions; I followed the latter). In reveal's popup speaker view (S), notes math may render without KaTeX styling; the on-slide notes (N) are fine.

## 5. Critique loop

**Round 1.** Design critic (Claude Opus subagent) and technical critic (Codex, high effort), in parallel, on the full draft.

Codex (7 items, all accepted): the FD bound as written is leading order only (a counterexample $L=x^6$); now stated as leading order with the rigorous sup version in the notes, and only the scalings on the slide. "Gradient similarity differs exactly on outlier directions" and "$(G+\lambda I)^{-1}\approx\lambda^{-1}I$ on almost all of parameter space" overclaimed: rewritten in terms of $s_k\ll\lambda$ vs $s_k\gtrsim\lambda$, with the gradient projections the density does not give. Power iteration finds the largest-*magnitude* eigenpair. The toy figures use the mean loss: now said in the outline, notes and figure script. Estimating $\operatorname{tr}G$ needs $Gv$, not $Hv$. The Chen–Trogdon–Ubaru probe count falls like $\log P/P$ for sphere probes. 240 GB (not 200) per vector at 120B.

Design critic (14 ranked fixes; most accepted): split §9 into two slides and reconcile it with what SLQ can resolve (outliers and the mean are measurable at scale; where $0.1\times$mean falls inside the bulk is not), with an explicit $H\to G$ bridge. Slide 21: the failure causes are the authors' labels; added the circularity of choosing $\epsilon$ by "outliers appear" and that the lecture's own check (FD against an exact HVP) is not reported; added their "larger models are more stable" claim with "one pair of models"; fixed the Fig 8 description; subsampling moved to notes. Slide 22: $\epsilon$ not swept at 120B; a third, numerical reading of the negative eigenvalues; negative eigenvalues live in $H-G$ since $G\succeq0$. Replaced the cost slide with a "Reading the paper" slide (the paper's "exact", Gamma/Ritz-weight stem plots, "smoothing ε", θ as parameters and "reverse-over-forward", dataset-averaged loss, §4's heuristic assumption); cost numbers moved to notes. $\epsilon/\sqrt P$ per weight and the gap between Remark 3.2 and their empirical sweet spot. Block test: "predicting zero gives exactly 1"; EK-FAC's per-matrix blocks are finer than whole layers, so the test is if anything generous to it. Merged the FD slide with its figure. Slide 8 now leads with the cost of a solve. Golub–Welsch now leads with moment matching. LM caveat: $m=10$ nodes cannot resolve $10^4$–$10^5$ outliers. Slide 19: the gain is stack compatibility, not FLOPs; the all-gather diagnosis attributed to the authors. Notation: $\mathbb R^d$ on slide 4; figure axes relabelled with $s$. Compressed path now lists slide numbers, keeps one §4 sentence and gives §9 2 min. Not taken: cutting the scale table (the brief requires it; now "skip if late") and moving Schioppa off slide 11 (the brief asks for it there).

**Round 2** (same design critic, resumed; a fresh Codex run that also compared the `.html` against the `.md`). Both found no errors in the formulas or the quoted numbers; Codex reproduced every figure number in memory and re-checked the DA2 table and the paper tables. Changes made:

- Codex: removed an HTML-only inference ("the corrected part is not small because the gradient concentrates in the top directions"); a counterexample shows inverse weighting can reverse it. Now: contributions depend on both projections and the weights $1/(s_k+\lambda)$.
- Codex: the mean eigenvalue has no quadrature error and is unbiased, but its Monte Carlo spread can be large (rank-one example: relative s.d. $\approx\sqrt{2/n_v}$, 45% at $n_v=10$); "trust the mean" replaced by "check its spread across probes" on slides 16 and 25.
- Codex: slide 26's title now says "far below $\lambda$" (not "below"); "near or above" is curvature-weighted.
- Codex: solve cost no longer stated as universal (LiSSA often needs thousands; CG depends on conditioning); EK-FAC came from optimization (George et al. 2018) and was applied to IFs by Grosse et al.
- Codex: Thm 3.1's bias contradicts §4's *unbiasedness* assumption (not its independence assumption).
- Codex: EK-FAC = block-diagonal with a Kronecker-factored *eigenbasis* and refitted eigenvalues, not "plus Kronecker".
- Codex: the final-block exception is reported for the DeepSeek-Coder model in §6.2, not in the tables generally.
- Design: slide 2 no longer says "everything we know about curvature comes from $Hv$" (EK-FAC uses none); now "every direct look at the true curvature".
- Design: slide 25 now carries the §6→§9 link (what SLQ can and cannot resolve about where $\lambda$ sits); slide 22 shows the three readings of the negative eigenvalues on the slide; slide 11 now says memory rules out reorthogonalization and ghosts then keep $m$ small; slide 20 trimmed to five bullets (§4's assumption moved to notes); slide 8's title matches its claim; compressed path adds one sentence each for the dropped slides 9 and 13.
- Design: inline KaTeX no longer breaks across lines (`white-space: nowrap`); figure fonts enlarged (numbers unchanged on rerun).

After round 2 the deck was re-rendered and re-checked (28 slides, 268 KaTeX expressions, 0 errors, 0 broken images, 28-page PDF). Both critics said nothing important remained; no third round was run.

## 6. Not verified

- The 2026 paper's experiments themselves (no code, probe count or hardware is given); its figures were read as published, and the "essentially unchanged" 1% vs 30% comparison is reported as the authors' claim.
- That "the LLM scale paper recently" in CRW's request is 2602.00816 (it is the only 2026 SLQ-at-100B-scale paper found; HessFormer is its predecessor).
- DA2's correlations were read from its saved output, not regenerated.
- The exact HessFormer checkpoint (the paper says "70bn Deepseek-Llama distilled").
