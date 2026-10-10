# WB2 figure plan: Statistical learning and regularization

- **Notebook:** WB2, `notebooks/02_statistical_learning.html`, built by `notebooks/build_wb2.py`.
- **Date:** 2026-10-09. This is a planning document only; no builder, notebook or asset was edited.
- **Proposals:** 21 in total: 6 P1, 7 P2, 8 P3.
- **P1 items:**
  - #1 the bias–variance U-curve (1.C)
  - #7 the information plane, claim vs critique (1.F)
  - #8 binned entropy vs weight scale (1.F.4)
  - #14 the Belkin double-descent schematic (§3)
  - #17 the spiky-smooth benign-overfitting picture (3.D)
  - #19 deep double descent / the linear (γ × time) heatmap (3.E)

**State of the sources when read.** The builder is mid-integration. `build_wb2.py` imports `wb2_s1e` (not on disk) and `s2.S2_VC`, `s2.S2_PACBAYES` (not defined in `wb2_s2.py`). The kernel text is still in `wb2_new_kernels.py`, and the VC and PAC-Bayes text is in `wb2_new_pacbayes.py` (`CELLS_VC`, `CELLS_PACBAYES`). The generated HTML (08:37) predates these changes. I anchored each proposal to the text as it currently stands, in whichever module holds it.

**How the source types were checked.**
- *Paper figures (type a):* I checked these against the arXiv PDFs and abs pages, with a sub-agent doing the fetching. Every one is under the **arXiv non-exclusive licence (distribution rights only)**, except Haas et al. 2023, which is **CC BY 4.0**. Saxe et al. (ICLR 2018) has no arXiv version; the J. Stat. Mech. version is closed access, and I could not confirm its licence. So for every P1 I recommend redrawing the figure or generating our own, and *linking* to the original figure number. Embedding the originals is CRW's call.
- *Self-generated figures (type b):* the recipes for #1, #8, #9, #10, #15, #16, #17 and #19b were prototyped in a scratch directory (numpy and matplotlib, CPU). Timings and checks are noted where relevant. None of the outputs were added to the repo.

**Infrastructure needed.** `nbkit.py` has only `figure()`, which renders a *link* to an interactive page. There is no static-image helper.
- Static figures need a small `image(path, alt, caption)` helper that emits `<figure><img alt=…><figcaption>`, an `assets/figures/wb2/` directory, and the generating scripts kept beside `checks/` so the figures can be reproduced.
- The build's link check already validates `src=` targets.
- Interactive figures do not print, and the README says that printing a page prints all of its solutions. That is one more reason for static companions to the interactives (#3, #16).

---

## Proposals, in notebook order

### 1. Bias–variance: the U-curve, two ways (P1)

- **Anchor:** §1.C, after the 1.C.2 solution, at the paragraph "The textbook U-shaped curve, where test error falls with model size and then rises, is 1.C.2 with the penalty replaced by the number of features."
- **What it shows:** two panels with a shared, *linear* y-axis: excess risk $\mathbb E\|\hat\beta-\beta\|^2$.
  - (a) Ridge on a fixed Gaussian design. The x-axis is the effective degrees of freedom $\mathrm{df}(\lambda)=\sum_k s_k^2/(s_k^2+n\lambda)$, running from 0 (heavy penalty) to $p$ (OLS). Curves for bias², variance and total, using the closed form of 1.C.2.1 averaged over $\beta\sim\mathcal N(0,\tau^2I)$. Mark the optimum $n\lambda=\sigma^2/\tau^2$ (1.C.2.3).
  - (b) OLS on the first $k$ of $p$ features, for $k=1,\dots,0.9n$, with decaying true coefficients $\beta_j\propto 1/j$. Bias² is $\sum_{j>k}\beta_j^2$ and variance is $\sigma^2k/(n-k-1)$ (1.B.2, exact). Draw a vertical asymptote at $k=n-1$, with a light band at the right edge labelled "§3 continues past here".
- **Why it helps:** it anchors 1.C on its canonical picture. Panel (b) is literally the left half of the double-descent curve, so §3's claim that "the U is not the whole story" becomes a visual continuation rather than a new topic.
- **Source:** (b) self-generated, closed form, under 1 s. Suggested values are $n=100$, $p=60$, $\sigma^2=1$, $\|\beta\|^2\approx1.5$, chosen so the two ends of the U are comparable. In the prototype, a log y-axis and a raw-λ x-axis hid the U, so use a linear y-axis and the df axis. ESL Fig. 7.1 is the empirical version (Springer copyright, free to read), so redraw rather than embed.
- **Caption draft:** "Excess risk of ridge regression (a) and of least squares on the first k features (b), split into squared bias and variance. Shrinking adds bias and removes variance; the best trade-off is in between. In (b) the variance diverges as k approaches n, which is where §3 picks up."

### 2. L1 vs L2: thresholding functions and the diamond (P2)

- **Anchor:** 1.D.3 solution (`wb2_new_regularization.py`). Put panel (a) after "The sketches: ridge is a line through the origin with slope $1/(1+\lambda)$." Put panel (b) after "*The picture.* … This is ESL's Figure 3.11."
- **What it shows:**
  - (a) Estimate vs $z_j$ for ridge $z/(1+\lambda)$, soft threshold and hard threshold, with $\lambda=1$, the identity line dashed, and the band $|z|\le\lambda$ shaded.
  - (b) In 2-D, the $L^1$ diamond and the $L^2$ disk of matched size. Both share the same tilted elliptical loss contours around an off-axis OLS point, e.g. $(2,0.8)$. Mark the touching points: a corner, where $b_2=0$, for $L^1$, and a generic point for $L^2$.
- **Why it helps:** the exercise asks students to "Sketch the three" and to "Draw the constrained version", and the solution describes ESL Fig. 3.11 and Table 3.4 only in words. The answer is a picture.
- **Source:** (b)/(c), self-generated, closed form. The touching points come from a 1-D grid search on the constraint boundary. ESL Fig. 3.11 is confirmed to be the diamond and disk figure, but it is under Springer copyright, so redraw it.
- **Caption draft:** "(a) Ridge shrinks every coefficient proportionally, the lasso shifts large ones toward zero by λ and zeroes small ones, and best-subset keeps or kills. (b) The loss contours usually first touch the $L^1$ ball at a corner, which is an exact zero; the $L^2$ ball has no corners."

### 3. Static companion to `spectral_filters.html`, panel A (P3)

- **Anchor:** 1.D.4, at the existing `figure("assets/interactive/spectral_filters.html", …)` cell.
- **What it shows:** filter $\phi$ against $x=s^2/(n\lambda)$ on a log x-axis, with three curves: ridge $x/(1+x)$, half-point-matched flow $1-2^{-x}$, and naive-matched flow $1-e^{-x}$. A difference strip below marks the extremes from the solution: $+0.138$ near $x\approx4.3$ and $-0.044$ near $x\approx0.36$.
- **Why it helps:** it makes the figure printable and visible without a click. Keep the interactive as it is.
- **Source:** (d)+(b), closed form.
- **Caption draft:** "Ridge and early-stopped gradient flow as filters on the data's singular directions. With matched half-points they agree in outline; flow is the sharper, more step-like filter."

### 4. Rescaling makes a minimum arbitrarily sharp (P2)

- **Anchor:** 1.D.7, after the part-3 solution: "Dinh et al. use this to make any minimum arbitrarily sharp."
- **What it shows:** log-contours of $R=a(w_1w_2-c)^2$ on $[-3,3]^2$ with $a=c=1$. The valley is the hyperbola $w_1w_2=1$. Mark 6–7 points along it, for $\alpha\in[0.3,3]$, coloured by $\mathrm{Tr}\,H=2a(w_1^2+w_2^2)$. At each point draw a short bar along the sharp eigenvector $v=(w_2,w_1)$ with length $\propto(\mathrm{Tr}\,H)^{-1/2}$, the curvature width. Annotate the balanced point, where $\mathrm{Tr}\,H=4a|c|$.
- **Why it helps:** the whole of Dinh et al.'s argument is one picture: the same function, a whole curve of parameter settings, and a sharpness that varies without bound along it. The same example returns in 2.E.2.4, the PAC-Bayes rescaling, which can refer back to this figure.
- **Source:** (b), under 1 s, prototyped. Dinh et al. Fig. 2 shows the same idea (level curves with observationally equivalent hyperbolas) but is under the non-exclusive licence, so redraw and cite "after Dinh et al. 2017, Fig. 2".
- **Caption draft:** "Every point on the hyperbola computes the same function $f(x)=c\,x$. The curvature across the valley, $\mathrm{Tr}\,H$, is smallest at the balanced point and grows without bound along the valley, so raw sharpness is a property of the parametrization."

### 5. Kernel ridge and GP error bars in 1-D (P2)

- **Anchor:** 1.E.3 (`wb2_new_kernels.py`), after the solution, at "Because the variance does not depend on $y$, it cannot detect that the model is wrong".
- **What it shows:** $n=15$ noisy points from $\sin2\pi x$ on $[0,1]$, Gaussian kernel, x-range $[-0.3,1.3]$.
  - (a) Kernel ridge fits at $\lambda=10^{-1}$, $10^{-3}$ and ridgeless. The ridgeless fit interpolates.
  - (b) GP posterior mean ±2 sd with $\ell=0.1$, $\sigma=0.1$, $\tau=1$, plus three posterior samples. The band pinches at the data and reverts to ±2τ away from them.
  - (c) The same data with a wrong length scale, $\ell=0.02$. The error bars are just as tight at the data but the mean is wrong between them.
- **Why it helps:** it makes 1.E.1.3 (the ridgeless limit interpolates) and 1.E.3.3 (what the error bar cannot detect) visible.
- **Source:** (b), closed form, under 1 s. GPML ch. 2 has canonical versions (MIT Press, free to read; I did not check figure numbers), so self-generate.
- **Caption draft:** "Gaussian-process regression: the posterior mean is kernel ridge regression, and the error bar shrinks near data and reverts to the prior away from them. With the wrong length scale (c) the bars are just as confident and wrong, because they never look at y."

### 6. Rate–distortion curves (P2)

- **Anchor:** 1.F.1 (`wb2_new_ratedistortion.py`), after the solution to parts 1–5 and before "**The Lagrangian form.**"
- **What it shows:**
  - (a) Bernoulli: $R(D)=H_b(p)-H_b(D)$ in bits for $p=0.5,0.2,0.1$, zero for $D\ge p$. Mark the point $D=0.01$, $R=0.919$ from 1.F.1.3.
  - (b) Gaussian: $R(D)=\tfrac12\log_2(\sigma^2/D)$ against $D/\sigma^2$, with the achievable region (above the curve) shaded. Draw one tangent line labelled "slope $=-\beta$", which part 7 derives.
- **Why it helps:** it shows convexity, the steep slope near $D=0$ (the first errors save the most rate), and $\beta$ as a slope. Each of these is a statement about the shape of a curve.
- **Source:** (b), closed form, prototyped.
- **Caption draft:** "Rate–distortion functions of a biased coin (Hamming distortion) and a Gaussian (squared error). The curves are convex and decreasing; the Lagrange multiplier β of the Boltzmann-form encoder is minus the slope."

### 7. The information plane: claim and critique under one estimator (P1)

- **Anchor:** "#### The information plane: did deep networks compress?" (`wb2_s1f.py`), after the "**The critique** (Saxe et al. 2018)" paragraph.
- **What it shows:** two panels, tanh and ReLU, with binned $I(X;T)$ on the x-axis and binned $I(T;Y)$ on the y-axis, both in bits. Each hidden layer's trajectory is coloured by epoch. A dashed vertical line at $H(X)=12$ bits marks the true $I(X;T)$ of any injective layer (1.F.3).
- **Why it helps:** the debate is anchored on Shwartz-Ziv and Tishby's information-plane figure. Putting it beside the ReLU counterpart, computed by the *same* estimator, shows the claim and the confound in one glance. The 12-bit line shows what the estimate is not measuring.
- **Source and honesty:**
  - *Published figures (verified):* Shwartz-Ziv and Tishby **Fig. 3** is the canonical plane (Fig. 2 has snapshots; Fig. 4 has drift and diffusion). Saxe et al. **Fig. 1** shows (A) tanh compressing and (B) ReLU not.
  - *Licensing:* the first is under the arXiv non-exclusive licence. For Saxe I could not confirm a licence, and there is no open version.
  - *Why self-generate:* the published plots depend on activation, binning range and bin count. A self-generated pair that states its estimator is more honest than either embed. **(b) self-generated, prototyped.**
  - *Prototype:* 12-bit inputs (4096), a quadratic teacher thresholded at its median, the 12-10-7-5-4-3-2 architecture, full-batch Adam with lr $4\times10^{-3}$ on 85% of the data, and 30 bins. Tanh bins are on $[-1,1]$; ReLU bins span the observed range.
  - *Prototype result:* the deep tanh layers moved from about 9–12 bits to about 3–5 bits at constant $I(T;Y)\approx1$ bit. The ReLU layers did not compress.
  - *Runtime:* about 130 s at 4000 epochs. Use about 2000 epochs and a snapshot every 50 epochs to get under a minute, and commit the PNG.
  - *Robustness:* the result depends on the seed and the learning rate. Check two or three seeds before committing.
- **Caption draft:** "Binned information-plane trajectories of the same small network with tanh (left) and ReLU (right) units, estimated identically (30 bins). Only the saturating network 'compresses'. For both, the true $I(X;T)$ of an injective layer is the dashed line, 12 bits, at every epoch (1.F.3). After Shwartz-Ziv & Tishby 2017, Fig. 3, and Saxe et al. 2018, Fig. 1."

### 8. Binned entropy vs weight scale: saturation looks like compression (P1)

- **Anchor:** 1.F.4 solution, part 2, at the table "For $K=30$, exact bin probabilities give (from `checks/wb2b_check.py`):". Put the plot beside the table or replace it.
- **What it shows:** $H(\mathrm{bin}\,T)$ in nats against the weight scale $a$, on a log x-axis from $10^{-2}$ to $10^4$, for three cases:
  - tanh with $K=30$ bins on $[-1,1]$, which rises to 3.40 at $a\approx0.85$ and then falls to $\log2$;
  - ReLU with fixed width $1/15$, which grows like $\tfrac12\log a$ (part 3);
  - ReLU with bins on the observed range, which is flat (part 4).

  Reference lines at $\log2$ and at $\log4096=8.32$ nats, the true $I(X;T)$ of an injective layer.
- **Why it helps:** this is the mechanism of the controversy in one plot: monotone weight growth produces "fitting then compression" for tanh only. The exercise asks students to sketch it, and the solution currently has only a table.
- **Source:** (b), exact via the normal CDF, about 3 s. The prototype reproduces the table (peak 3.40 at $a\approx0.85$). Saxe et al. Fig. 2(C) is the published analogue, but its licence is unverified, and the self-generated version is exact and adds the true-MI line.
- **Caption draft:** "Binned entropy of one unit's activation as its weights grow. Tanh rises and falls ('fitting', then 'compression') purely through saturation; ReLU keeps rising with fixed bins, or stays flat with range-adaptive bins. The unquantized $I(X;T)$ does not move."

### 9. With noise, compression is clustering: the two-class path (P3)

- **Anchor:** 1.F.6 solution, part 3.
- **What it shows:** the path of $\big(I(X;T),I(T;Y)\big)$ as the within-class covariance shrinks, $W\to\alpha W$ with $\alpha$ going from 10 to $10^{-4}$ ($d=1$, $m=1$, $W=4$, $\sigma=0.5$). An arrow shows decreasing α. The point moves up and to the left, toward $I(X;T)=I(T;Y)$. An optional second curve shows the global rescaling $f\to cf$ at fixed σ (part 4).
- **Why it helps:** it is the noisy-network counterpart of #7: real compression, which is geometric clustering.
- **Source:** (b), a 1-D quadrature for $I(T;Y)$, 0.2 s, prototyped.
- **Caption draft:** "In the noisy two-class model, shrinking same-class spread lowers $I(X;T)$ and raises $I(T;Y)$: compression is clustering at the scale set by the noise."

### 10. Depth buys pieces: the tent-map sawtooth (P3)

- **Anchor:** 2.B.1 solution, part 4: "Depth buys pieces exponentially."
- **What it shows:** four small panels of $t^{\circ k}$ for $k=1,\dots,4$ on $[0,1]$, titled 2, 4, 8 and 16 pieces.
- **Why it helps:** it makes Telgarsky's construction immediate.
- **Source:** (b), trivial. Telgarsky Fig. 2 is confirmed to be the iterated tent map, under the non-exclusive licence.
- **Caption draft:** "Composing the two-unit tent map k times gives 2^k linear pieces; one hidden layer needs about 2^k units for the same function."

### 11. Shattering by halfspaces, and Radon's obstruction (P3)

- **Anchor:** 2.C.1, part 3 (`wb2_new_pacbayes.py`, `CELLS_VC`), "*Affine halfspaces* in $\mathbb R^d$".
- **What it shows:** a 2×4 grid of three points in $\mathbb R^2$, one panel per labelling, each with a separating line. Beside it, four points in convex position with the XOR labelling, with the crossing diagonals drawn to show the Radon partition.
- **Why it helps:** a picture of "shattering" is faster than the definition.
- **Source:** (c), a matplotlib schematic.
- **Caption draft:** "Lines in the plane realize all 8 labellings of 3 points, so the VC dimension is at least 3. For 4 points, two groups always have intersecting convex hulls (Radon), so some labelling is impossible."

### 12. Zhang et al.: networks fit random labels (P2)

- **Anchor:** §2.D, after the reading blurb.
- **What it shows:** Zhang et al. **Fig. 1** (verified), on CIFAR-10:
  - (a) training loss against thousands of steps for true labels, random labels, shuffled pixels, random pixels and Gaussian inputs;
  - (b) relative convergence time against label corruption;
  - (c) test error against label corruption.

  Panels (b) and (c) are for Inception, AlexNet and MLP 1×512.
- **Why it helps:** it is the iconic figure behind 2.C.2.3 and 2.D.1, and it shows "fits random labels, just more slowly", which the answer to 2.D.1.1 leans on.
- **Source:** (a). It is under the non-exclusive licence, so by default add one sentence to the blurb ("Fig. 1 is the result in one picture") and link it. Embedding it is CRW's call. A self-generated stand-in is possible but untested: a numpy MLP of width about 512 fitting 1000 sklearn digits with 0%, 50% and 100% random labels, plotting training loss against epochs, in about 30–60 s CPU.
- **Caption draft (if embedded):** "Zhang et al. 2017, Fig. 1: the same networks reach zero training loss on random labels and random pixels, only more slowly; test error rises smoothly with the fraction of corrupted labels."

### 13. PAC-Bayes: the posterior width trade-off per direction (P3)

- **Anchor:** 2.E.2 solution, after "The bound counts the curved directions, not the parameters."
- **What it shows:** for curvatures $h\in\{10^{-1},10,10^3\}$, the per-direction objective $\tfrac12hs^2+\frac1\lambda\cdot\tfrac12\big(s^2/\sigma_P^2-1-\ln(s^2/\sigma_P^2)\big)$ plotted against the posterior sd $s$ on a log axis. The optimum $s^*=(\sigma_P^{-2}+\lambda h)^{-1/2}$ is marked, and the bar heights at the optima equal $\tfrac12\ln(1+\lambda\sigma_P^2h)$.
- **Why it helps:** it shows that flat directions cost nothing and sharp ones cost only a logarithm.
- **Source:** (b), closed form.
- **Caption draft:** "Each direction's contribution to the Gaussian PAC-Bayes bound. Flat directions keep the prior width for free; sharp ones force a narrow posterior and cost about ½ ln(λσ_P²h)."

### 14. Double descent, the schematic (P1)

- **Anchor:** §3 opening (`wb2_s3.py`), after "The name is from [Belkin et al. (2019)](https://arxiv.org/abs/1812.11118)."
- **What it shows:** two schematic panels, with test risk solid and training risk dashed.
  - (a) The classical U against capacity, with the "sweet spot" marked.
  - (b) The same curves continued past a vertical "interpolation threshold". Training risk is 0 beyond the threshold. Label the regions "under-parameterized" and "over-parameterized".
- **Why it helps:** this is the image the topic is named for. Students should see it before the exact linear-regression version (#16 and the interactive).
- **Source:** (c), redrawn "after Belkin et al. 2019, Fig. 1". I verified that Fig. 1 is exactly this pair of schematics. Its licence is the arXiv non-exclusive licence, and the PNAS version is under the standard licence. Also point to Belkin **Fig. 2** (verified: random Fourier features on MNIST with $n=10^4$, showing test risk, norm and training risk against the number of features) as the empirical version.
  - *Honesty note for the caption:* the schematic puts the global minimum in the over-parameterized regime. In §3's isotropic model it is not (3.C.1.1: the global minimum is at γ→0). It can be when added features carry new signal (Hastie et al. §5).
- **Caption draft:** "Schematic after Belkin et al. (2019, Fig. 1). Classical theory (a) predicts a U; past the interpolation threshold (b) the test risk can fall again while training risk stays at zero. Whether the second descent beats the first minimum depends on the problem (3.C.1.1)."

### 15. Marchenko–Pastur: why the peak is at p = n (P3)

- **Anchor:** 3.A.1 solution: "The empirical spectrum of $X^\top X/n$ converges to the Marchenko–Pastur law … as $\gamma\to1$ the lower edge tends to $0$". It can also be linked from 1.B.2.3.
- **What it shows:** Marchenko–Pastur densities for $\gamma=0.1,0.5,0.9$, overlaid with eigenvalue histograms from one $1000\times1000\gamma$ draw. The lower edge $(1-\sqrt\gamma)^2$ is marked.
- **Why it helps:** "the smallest eigenvalues blow up the sum" is a statement about where the spectrum's edge sits.
- **Source:** (b), about 3 s, prototyped.
- **Caption draft:** "Eigenvalues of $X^\top X/n$ for Gaussian designs. As p/n → 1 the spectrum's lower edge reaches zero, and least squares divides the noise by these near-zero eigenvalues."

### 16. Static companion to `double_descent.html` (P2)

- **Anchor:** 3.C.2 solution: "*What a correct sketch shows.*"
- **What it shows:** two panels, SNR 5 and SNR 0.5. Each has the limiting min-norm curve with bias and variance as stacked shaded areas, the null risk $r^2$, optimally tuned ridge, and finite-n dots (n=100). The minimum at $\gamma\approx1.81$ is marked for SNR 5.
- **Why it helps:** the exercise asks for exactly this sketch, and the answer is in words. A static figure also prints and is seen without a click. The interactive stays for exploration.
- **Source:** (d)+(b). Use the closed form for min-norm, and simulate optimal ridge using the $n\times n$ kernel-form solve (about 5 s). Reuse the interactive's formulas so the two agree.
- **Caption draft:** "Excess risk of minimum-norm least squares against γ = p/n. The peak is pure variance; past it, the curve dips below the null risk only when SNR > 1. Optimally tuned ridge has no peak."

### 17. Benign overfitting: the spiky-smooth interpolant (P1)

- **Anchor:** 3.D opening (`wb2_new_benign.py`), after the "**What you get.**" paragraph, which ends "…specialized to this model."
- **What it shows:** $n=30$ points, $y=\sin2\pi x+0.3\varepsilon$ on $[0,1]$. Three curves:
  - the ridgeless interpolant with kernel $k=k_{\ell=0.15}+c\,k_{\ell=0.003}$, $c=0.5$;
  - kernel ridge with the smooth kernel alone at $n\lambda=c$ (dashed);
  - the truth.

  The narrow component's Gram matrix at distinct points is about $cI$. That is 3.D.1.1's "tail supplies the ridge", in function space. An optional zoom inset shows one spike, and the caption can give the two test risks.
- **Why it helps:** the solution to 3.D.2.1 states the mechanism in words: "the ridge fit plus a narrow spike at each training point". The picture makes "interpolates every noisy point, yet predicts like ridge" self-evident.
- **Source:** (b), prototyped in 0.3 s. The interpolant coincides with the ridge curve except for spikes about 0.003 wide at each data point. **Alternative:** Haas et al. 2023 **Fig. 1(a)** (verified; **CC BY 4.0**, so it can be embedded with attribution) shows a Laplace spiky-smooth kernel on the circle. The self-generated version matches 3.D's notation ($n\lambda=c$).
- **Caption draft:** "A minimum-norm interpolant that overfits benignly: it passes through every noisy point, but away from the data it coincides with the ridge fit. The narrow part of the kernel acts as a ridge penalty of size c and stores the noise in spikes of negligible width."

### 18. What memorizing noise costs, as the tail grows (P3)

- **Anchor:** 3.D.2 solution, after part 4 (or 5).
- **What it shows:** against tail dimension $m$ on a log axis, from $n+2$ to $100n$:
  - the min-norm excess risk (simulated);
  - its tail term $\nu\|\hat\beta_t\|^2$, with the prediction $\sigma^2(n-k)/m$ drawn on top;
  - the ridge risk at $n\lambda=c$ (horizontal);
  - the isotropic min-norm risk at $p=k+m$ (part 5) for contrast.

  Parameters: $k=5$, $n=100$, $c=20$, $\sigma^2=1$, $r^2=1$, 10 reps.
- **Why it helps:** it shows both benign conditions at work: the tail cost vanishes like $1/m$ and the total converges to ridge.
- **Source:** (b), $n\times n$ solves, under 5 s.
- **Caption draft:** "As the low-variance tail gets higher-dimensional at fixed total variance c, the noise it memorizes costs about σ²(n−k)/m, and the interpolant's risk approaches ridge's."

### 19. Deep double descent, and its exact linear analogue (P1)

- **Anchor:** §3.E, at "Networks show double-descent-like curves in width, training time and sample size ([Nakkiran et al. 2019](https://arxiv.org/abs/1912.02292)) … weakened by regularization and early stopping."
- **What it shows:**
  - (a) **Pointer** to Nakkiran et al. Fig. 1 and Fig. 2 (verified).
    - Fig. 1: train and test error against ResNet18 width on CIFAR-10 with 15% label noise.
    - Fig. 2: a heatmap of test error over model size × epochs, with lines marking model-wise and epoch-wise double descent.
    - Sample-wise "more data hurts" is in Figs. 3, 11 and 12.
    - There is *no* effective-model-complexity schematic in the paper.
  - (b) A **self-generated linear analogue heatmap:** $\log_{10}$ excess risk of gradient descent from 0 on §3's isotropic model, with γ on the x-axis (log, 0.1–10) and gradient-flow time on the y-axis (log). It uses the exact spectral filter $1-e^{-s_k^2t/n}$ from 1.D.4 applied to the SVD of simulated designs, with $r^2=1$ and $\sigma^2=0.2$. The prototype shows the bright γ=1 ridge only for $t\gtrsim10^2$; early stopping removes it.
- **Why it helps:** CRW named this the paradigm image. (a) supplies the deep-net phenomenology; (b) shows, in the exact model of §3, the claim the text makes in words ("weakened by … early stopping"). It also shows honestly that this isotropic model has *no* epoch-wise double descent, which in linear models needs features learned at different rates. If that point is wanted, add a two-scale-feature variant, but verify the references (Heckel & Yilmaz 2020; Pezeshki et al.) before citing them.
- **Source:** (a) is under the non-exclusive licence, so link it, or embed if CRW decides. (b) is self-generated; the prototype took 212 s at $n=200$ with 5 reps. Use $n=100$, 2–3 reps and about 40 γ values to get under 30 s.
- **Caption draft (b):** "Excess risk of gradient descent on isotropic least squares, against p/n and training time. The interpolation peak at p = n appears only when training runs to convergence; stopping early (lower rows) removes it. Compare Nakkiran et al. (2019), Fig. 2, for ResNets on CIFAR-10."

### 20. The envelope $H_{jj}^2\le C_{jj}\le G^2H_{jj}$ (P3)

- **Anchor:** 4.B.2 solution, part 2: "On log–log axes the points lie in a wedge between a line of slope 2 (below) and a line of slope 1 (above)…"
- **What it shows:** the shaded wedge on log–log axes, with three point clouds from part 3's "types" model: one varying $\pi_k$ only (slope 1), one varying $s_k$ only (slope 2), and a mixed cloud whose fitted slope lies outside $[1,2]$ (part 4's counterexample). A band marks the reported γ of 1.1–1.5.
- **Why it helps:** the text asks students to "Draw it on log–log axes", and part 4's point (the envelope does not force the fitted slope) is visual.
- **Source:** (b), closed form.
- **Caption draft:** "Diagonal gradient noise against curvature lies in a wedge between slopes 1 and 2. Frequency-driven curvature gives slope 1 and magnitude-driven curvature gives slope 2, but a fitted slope is not forced into [1, 2]."

### 21. Which generalization measures predict the gap (P2)

- **Anchor:** §4.C, after the Jiang et al. reading blurb.
- **What it shows:** a horizontal dot plot of the overall Kendall $\tau$ and the granulated $\Psi$ for about 10 representative measures, with a zero line. Candidate measures: parameter count / VC-like, spectral-norm product, Frobenius norm, path norm, margin, sharpness, PAC-Bayes flatness and sharpness, gradient noise, steps to reach the target loss.
- **Why it helps:** the reading's headline is a ranking, and the claims "norms anti-correlate; sharpness-based measures do best" are easiest to see as one picture. It also sets up 4.C.1.3 (pooled τ vs Ψ).
- **Source:** (b), a transcription. I verified that the results are **tables only** (Tables 1–4); the paper has no correlation plot. The numbers must be transcribed by hand from Tables 2–4 and double-checked; I did *not* transcribe them. Facts with citation are fine.
- **Caption draft:** "Rank correlation with the generalization gap, over Jiang et al.'s 2187 CIFAR-10 networks (values from their Tables 2–4). τ pools all hyperparameter changes; Ψ averages over one-hyperparameter-at-a-time comparisons."

---

## Gaps and issues noticed

1. **Text asks for a sketch, but the solution has no picture.**
   - 1.D.3.1 ("Sketch the three") and 1.D.3.3 ("Draw the constrained version"): the solution describes ESL Fig. 3.11 and Table 3.4 in words. See #2.
   - 1.F.4.2 ("Sketch $H(\mathrm{bin}(T))$"): the solution has a table only. See #8.
   - 4.B.2.2 ("Draw it on log–log axes"): see #20.
   - 3.C.2 asks for a sketch, and its answer is prose that relies on the interactive. See #16.
2. **Pointers that would be better as figures.** 1.E.2.4 asserts that "ridgeless regression on m random features shows the same interpolation peak near m≈n" with no figure or derivation; Belkin et al. Fig. 2 (RFF on MNIST) is the natural pointer. The Shwartz-Ziv and Tishby blurb says "at each figure, ask what the plotted quantity is" without naming figures; they are Figs. 2–5, with Fig. 3 the canonical plane.
3. **Belkin et al. and Nakkiran et al. have no reading blurbs.** Both are linked inline (§3 intro, 3.E), but §3.F lists only Hastie et al. If #14 and #19 point to their figures, a short entry for each in 3.F would be the natural home. That is CRW's call, under the "few readings" guidance.
4. **The Hastie et al. blurb says "Read §1 (overview and figures)".** The figures are Figs. 1 and 2, and optimal ridge vs min-norm is Figs. 7 and 8 (v5 numbering; earlier versions may differ). One clause could name them.
5. **Stale documentation.**
   - `notebooks/assets/interactive/README.md` maps the WB2 figures to old sections ("WB2 §4", "§5", "§8"). They are now 1.D.4, 3.C.2 and 4.B.5.
   - `notebooks/README.md` ("For maintainers") still says WB2's cells live in `wb2_cells_a.py` and `wb2_cells_b.py`.
   - The stale "§5/§7/§8" cross-references I first saw in `wb2_s1.py` had been fixed by the time I re-checked.
6. **The builder does not currently import cleanly** (see the header). The generated page lags, so check any anchor quoted above against the integrated modules before inserting.
7. **Honesty notes for captions.**
   - The Belkin schematic implies an over-parameterized global minimum that §3's own model does not have (#14).
   - The published information-plane figures depend on activation and binning, so prefer the self-generated pair that states its estimator and shows the true-MI line (#7, #8).
   - The isotropic linear heatmap has model-wise but not epoch-wise double descent; say so rather than imply it matches Nakkiran's Fig. 2 (#19).
8. **Licensing decision for CRW.** Every paper figure here except Haas et al. (CC BY 4.0) is under the arXiv non-exclusive licence or unverified (Saxe). The plan defaults to redrawing or self-generating and linking to the original figure number; embedding originals in a private one-day course is a fair-use judgement I can't make.
