# Figure plan: TD1, deep linear networks (`td1_linear_networks`)

- **Notebook:** `td/build_td1_linear_networks.py` (builder read in full on 2026-10-09), with `td/td1_linear_networks_{exercises,solutions}.ipynb` and the lecture outline `td/lectures/L1_linear_networks.md` for context.
- **Date:** 2026-10-09
- **Proposals:** 12 in total. **P1: 2** (F1, F8), **P2: 6** (F2, F3, F6, F7, F9, F11), **P3: 4** (F4, F5, F10, F12).
- **How I checked.** I ran every code cell of the solutions notebook headless (Agg backend) and looked at each plot. The committed `.ipynb` has no saved outputs. I prototyped F1, F2 and F8 in the scratchpad: all three render as described, together in about 12 s on CPU. I checked the paper figures by downloading the arXiv PDFs and reading the captions, and the licences from the arXiv abs pages.
- **Out of scope.** The existing interactive `assets/interactive/mode_explorer.html` is not re-proposed. F1 and F8 are static companions to it.

**Why static figures matter here.** The code plots that show the canonical phenomena are mostly in `code_answer` cells: Exercise 1.5 (the sigmoids), Exercise 2.3 (plateau scaling) and Exercise 6.2 (the tree SVD). In the exercises notebook these cells are stubs, so a student sees **no Saxe sigmoid at all** until they have solved Exercise 1.5 or reached §5's demo cell. The explorer is a separate file, and the notebook itself warns that some viewers show it as source. So the hook images need to be pre-rendered PNGs in markdown.

**Embedding constraint.** `nbkit_td._validate` checks every local `](path)` target, and image markdown `![..](path)` matches that pattern. Any PNG must therefore exist before the build runs. My suggestion is `td/assets/static/td1_*.png`, rendered by one script (for example `td/assets/static/make_td1_figures.py`, numpy and matplotlib only, closed forms where possible). The builder would reference the PNGs with plain `![alt](assets/static/...)`. This needs no change to `nbkit_td.py`.

---

## F1 (P1). "Saxe in one picture": sigmoids ordered by strength against the shallow exponential, and the loss staircase

1. **Anchor.** Notebook intro, immediately after the first paragraph, which ends "...which were put in through the choice of data or initialization." It goes before "**Using this notebook.**". An alternative position is the §1 pitch, after "Everything later in the notebook builds on this one equation."
2. **What it shows.** Two panels.
   - **(a)** $a_\alpha(t)/s_\alpha$ against $t/\tau$ for four modes of a balanced two-layer net from a common small $a_0$. These are coloured S-curves with plateaus, rising in order of $s$. One grey dashed curve, $1-e^{-t/\tau}$, shows the shallow model; in normalized units it is the same curve for every mode, and that sameness is the point.
   - **(b)** The loss $\tfrac12\sum_\alpha(s_\alpha-a_\alpha)^2$ on a **log** y-axis, two-layer (solid) against shallow (dashed). Faint horizontal dotted lines mark the saddle losses $\tfrac12\sum_{\beta>k}s_\beta^2$, labelled $\Sigma_{\{1\}}$, $\Sigma_{\{1,2\}}$, and so on. The plateaus sit on these lines.
3. **Why it helps here.** The whole notebook is anchored on this image. Its first sentence promises "long plateaus followed by sudden drops, the strongest structure learned first". The student should see that before reading a derivation, and at present the exercises variant never shows it. The image also previews §3 (the plateaus are saddles) and §5.1 (saddle to saddle).
4. **Source type: (b) generate ourselves, as a static PNG.** Use the closed-form logistic from the setup cell (`logistic(t, s, a0)`); no simulation is needed, and it renders in under 1 s.
   - Use **factor-of-2 spacing**, e.g. $s=(4,2,1,0.5)$, with $a_0=10^{-5}$ and $t/\tau\in[0,14]$.
   - In my prototype, $s=(3,2,1,0.5)$ with $a_0=10^{-4}$ blurred the first two steps together, which is Exercise 6.1's criterion at work. The loss panel on a linear y-axis hides the later steps, so the log axis is needed.
   - Optionally overlay GD from small random weights as faint dots (the §5 "small Gaussian" run) to show that the closed form describes a random start. If this is added, say in the caption that the dots use the effective seed $\Delta/4$.
   - Saxe et al. 2014 Fig. 3 (left) is the paper analogue: analytic red, linear simulation blue, tanh green, on hierarchical data. I verified it in arXiv 1312.6120. The licence is the arXiv non-exclusive one, so don't embed it; cite it in the caption. Our version avoids its confound (a tanh network overlaid on a 32-item hierarchical dataset) and adds the shallow contrast, which theirs lacks.
5. **Caption draft.** "Two-layer linear network from a small balanced initialization (solid) against a single matrix trained on the same loss (dashed). Each singular mode of the target sits on a plateau and then rises along a sigmoid, strongest first. The shallow model learns every mode at the same exponential rate. Right: the loss pauses at the saddles $W=\Sigma_{\{1\}},\Sigma_{\{1,2\}},\dots$ (dotted). Closed-form solutions of §1.4; compare Saxe, McClelland and Ganguli (2014), Fig. 3."

## F2 (P2). Phase portrait of one mode in the $(b,c)$ plane

1. **Anchor.** §1.4, after "It is a dial on imbalance relative to $s$, not on the size of the weights (§5.3)." and before "For general $d$ the ODE is still solvable in closed form". Add a back-reference from §3.2 ("They are not isolated.") and from §3.3 ("That is a strict saddle").
2. **What it shows.**
   - The vector field of (1.3) for $s=1$ on $[-2,2]^2$, as a light streamplot.
   - The set of minima $bc=s$ as two red hyperbola branches.
   - Level sets of the conserved $d=c^2-b^2$, as faint dotted hyperbolas.
   - The saddle at the origin, marked.
   - Three trajectories:
     - **Small balanced start**, e.g. $(0.03,0.035)$: it hugs the diagonal, crawls near the saddle, and lands on the balanced minimum $b=c=\sqrt s$.
     - **Imbalanced start**, e.g. $(0.02,1.6)$: it moves along its $d$-hyperbola almost horizontally and lands quickly at a different point of the minimum set.
     - **Balanced start on the wrong side**, e.g. $(-0.5,0.5)$: it flows into the saddle and stays there (Exercise 1.4.4).
3. **Why it helps here.** One picture shows four facts that the text states separately:
   - the conserved quantity confines each trajectory to a curve;
   - the minima form a manifold (here a one-dimensional gauge orbit), and $D$ selects which point you reach (Exercise 3.2.3);
   - the origin is a strict saddle with one downhill direction, $b=c$;
   - balanced trajectories are slow near the saddle, while imbalanced ones are not.
   
   It is also Saxe et al.'s own Fig. 2, which makes it a canonical image.
4. **Source type: (b) generate ourselves, as a static PNG.** Use RK4 on (1.3), about 2 s. In my prototype the three trajectories behave as described.
   - Draw the $d$-hyperbolas parametrically ($c=\pm\sqrt{b^2+d}$), not with a mask on a grid; the mask left an artifact in my prototype.
   - Paper source: Saxe et al. 2014, **Fig. 2**, verified caption: "Vector field (blue), stable manifold (red) and two solution trajectories (green) for the two dimensional dynamics of a and b in (8), with τ = 1, s = 1". Their $(a,b)$ are our $(b,c)$. The arXiv licence is non-exclusive, so regenerate rather than embed.
5. **Caption draft.** "Gradient flow of one mode, $\tau\dot b=c(s-bc)$, $\tau\dot c=b(s-bc)$, with $s=1$. Red: the minima $bc=s$, a one-dimensional family from the rescaling symmetry. Dotted: curves of constant $d=c^2-b^2$, which trajectories cannot leave. A small balanced start (black) lingers at the saddle and then reaches the balanced minimum. An imbalanced start (green) runs along its hyperbola with no plateau. A balanced start with $a<0$ (red) is trapped by the saddle. After Saxe et al. (2014), Fig. 2."

## F3 (P2). Improve the existing alignment cell (§1.5, "Code: watching alignment happen before the loss moves")

1. **Anchor.** The code cell after "#### Code: watching alignment happen before the loss moves" (`svals = [3.0, 2.0, 1.0, 0.5]` ... `fig, axs = plt.subplots(1, 3, ...)`).
2. **What it shows now.** Three panels against $t/\tau\in[0,8]$: $\cos(p,q)$ per mode, the off-diagonal fraction (log scale), and the loss (linear scale). Alignment happens in $t/\tau\lesssim1$, squeezed into the left eighth of each panel. The loss starts moving at about $1.7\tau$. Nothing visually connects the two events, and the cell's claim ("before the loss moves") is exactly that connection.
3. **Suggested changes.**
   - Use `sharex=True` and limit the x-axis to about $[0,5]$, or use a symlog time axis.
   - Draw a vertical dashed line in all three panels at the time the loss has fallen by 1%.
   - Put the loss on a log y-axis, with dotted saddle levels $\tfrac12\sum_{\beta>k}s_\beta^2$ as in F1, so the staircase is readable (on the linear axis the last step, at 0.125, is invisible).
   - Title the third panel "loss (dotted: saddle values)".
   
   It is also worth one sentence in the markdown that mode 4's cosine reaches 1 only around $t\approx2.5\tau$, after the loss has started to move. That is consistent with the text's own caveat that only the first mode's alignment is derived.
4. **Source type: (d) improve an existing code-cell plot.** The run itself is unchanged.
5. **Caption/markdown draft.** "Dashed line: the loss has fallen by 1%. By then every mode's input and output weights are nearly parallel, and $W$ is nearly diagonal in the $U,V$ basis. Alignment comes first and learning second."

## F4 (P3). Silent alignment in a nonlinear network (paper figure)

1. **Anchor.** §1.5, at the end, after "...the kernel aligns its eigenvectors while it is still small, before the loss moves."
2. **What it shows.** Atanasov, Bordelon and Pehlevan 2022, **Fig. 1(a)**: a 2-layer ReLU MLP on whitened MNIST 0/1. The kernel–target alignment saturates at $t\approx50$, while the kernel scale and the loss start moving at $t\approx300$. The analytic linear-network value is dashed. I verified the caption in arXiv 2111.00034.
3. **Why it helps here.** It is the one piece of nonlinear evidence for the §1.5 mechanism. It sits next to F3's linear version and answers "does this happen beyond linear nets?" without prose.
4. **Source type: (a) paper figure.** arXiv 2111.00034, Fig. 1, panel (a) only. The licence is **CC BY 4.0**, verified on the abs page, so it can be embedded with attribution. The alternative is to leave it as a reading and add only a sentence pointing at Fig. 1.
5. **Caption draft.** "Silent alignment in a two-layer ReLU network on whitened MNIST: the tangent kernel turns toward the target (green) long before its scale (orange) and the loss (blue) change. From Atanasov, Bordelon and Pehlevan (2022), Fig. 1a, CC BY 4.0."

## F5 (P3). Strict versus degenerate saddle: the loss along the balanced direction

1. **Anchor.** §3.3, after "The saddle is *degenerate*: no direction is downhill at second order, and escape is the slow power law of §2.2."
2. **What it shows.** One panel: $L(w)=\tfrac12(s-w^N)^2-\tfrac12s^2$ for $w\in[-0.6,0.6]$ and $N=2,3,4$, with $s=1$. A second small panel can show the escape speed $|\dot w|\propto w^{N-1}$ on log–log axes.
3. **Why it helps here.** "Hessian vanishes" is abstract. The picture shows a downward parabola for $N=2$, a flat inflection for $N=3$ and a flatter one for $N=4$, which is why the plateau becomes a power law. It also illustrates the depth content of §2.2 for students who skip the explorer's "deep" preset.
4. **Source type: (b) closed form, static PNG** (or a 4-line code cell placed after the Hessian cell).
5. **Caption draft.** "The mode loss along the balanced direction $w_1=\dots=w_N=w$, near the origin. At depth 2 it curves downward: a strict saddle, escaped exponentially. From depth 3 it is flat to second order: a degenerate saddle, escaped only as a power law in the initial scale."

## F6 (P2). The critical points as a ladder, with the training curve walking down it

1. **Anchor.** §3.4, after "The landscape tells you which places training visits. It does not tell you how long it stays at each." Add a back-reference in §5.1 at "the saddles of §3, in order. The loss is a staircase".
2. **What it shows.** Left, a schematic: for $s=(3,2,1)$, the $2^3$ subsets $S$ as nodes at height $L(S)=\tfrac12\sum_{\alpha\notin S}s_\alpha^2$. Edges join subsets that differ by one mode. The "top-$k$" chain $\varnothing\to\{1\}\to\{1,2\}\to\{1,2,3\}$ is bold. The others are grey, labelled "skips a top mode: strict saddle at any depth". The minimum is starred. Right, sharing the y-axis: the loss of a small-initialization run (closed form, or §5's small-Gaussian GD), rotated as loss against time, so its plateaus line up with the bold nodes.
3. **Why it helps here.** §3 classifies critical points, §5.1 says training visits them in order, and §3.4 says the classification says nothing about time. This figure shows "where" (the nodes) and "how long" (the plateau lengths) side by side. It is the saddle-to-saddle picture, specialised to this model.
4. **Source types: (c) schematic plus (b) generated curve, as a static PNG.** Use matplotlib only, closed forms, under 1 s.
   - The paper analogue is Jacot et al. 2021, **Fig. 1**, verified caption: "Saddle-to-Saddle dynamics: A DLN (L = 4, w = 100) with a small initialization ... passing through 2 saddles ... three plateaus". The arXiv licence is non-exclusive, so cite it and don't embed.
   - Our version is cleaner for this notebook: their example is matrix completion (an underdetermined problem), while ours is the whitened full-rank setting.
5. **Caption draft.** "Left: every critical point of the whitened deep linear network keeps a subset $S$ of the target's modes. Its loss, $\tfrac12\sum_{\alpha\notin S}s_\alpha^2$, sets its height. Right: from small initialization, training descends the bold chain one mode at a time. Each plateau is the time spent near a saddle, which the landscape alone does not determine."

## F7 (P2). Schematic: what $\Delta$ measures (imbalance *and* misalignment)

1. **Anchor.** §4.3, after "If $|p|=|q|$ but $p\perp q$, $\Delta=|p|^2+|q|^2$ even though the layers have the same size." Alternatively, at the paragraph beginning "We call $\Delta$ the mode's **invariant**."
2. **What it shows.** Three small vignettes in hidden space, each with arrows $p=W_1v$ (blue) and $q=W_2^\top u$ (orange), and $a=p\cdot q$, $k=|p|^2+|q|^2$ and $\Delta$ written beneath:
   - **(i) balanced and aligned:** $p=q$, so $\Delta=0$;
   - **(ii) aligned but imbalanced:** $p\parallel q$ with unequal lengths, so $\Delta=|d|$;
   - **(iii) wide Gaussian:** $|p|=|q|$ with $p\perp q$, so $a\approx0$ and $\Delta=k$, "equal norms, yet maximally imbalanced".
   
   Optionally add a fourth vignette for large balanced weights: long equal parallel arrows, with $\Delta=0$ and $k$ large.
3. **Why it helps here.** The section's key conceptual correction is "equal layer norms are not balance". In prose this reads like a technicality; with arrows it is immediate. It also prepares §5.3's "large balanced is not lazy".
4. **Source type: (c) schematic.** Draw it in matplotlib (arrows plus text) so it stays in the same static-asset script.
5. **Caption draft.** "The per-mode invariant $\Delta^2=k^2-4a^2$ for three configurations of a mode's input vector $p$ and output vector $q$ in hidden space. It vanishes only when $p=q$. Unequal lengths and non-parallel directions both make it large, so a wide Gaussian initialization, whose $p$ and $q$ have equal length but are nearly orthogonal, is strongly imbalanced."

## F8 (P1). The lazy–rich dial as paths on the conserved hyperbolas in the $(a,k)$ plane

1. **Anchor.** Top of §5, after the "What you need" box and before "### 5.1 The rich regime: saddle to saddle". A shorter cross-reference can go at the end of §4.3, after (4.3).
2. **What it shows.**
   - The x-axis is the mode strength $a$, with a vertical dotted line at the target $a=s$ ($s=1$). The y-axis is the mode kernel $k$, on a log scale.
   - Faint curves $k=\sqrt{4a^2+\Delta^2}$ (the conserved hyperbolas) for several $\Delta$.
   - Thick segments mark the stretch each regime actually travels, from $a_0$ to $s$:
     - **rich:** $\Delta=0.004$, $a_0=10^{-3}$; $k$ climbs about 2.7 decades;
     - **mixed:** $\Delta\approx s$; $k$ about doubles;
     - **lazy:** $\Delta=20$; a flat segment;
     - **large balanced:** $\Delta=0$, $a_0=5s$; the segment slides *down* the line $k=2a$ from 10 to 2.
   - Annotate each segment with its relative kernel change, e.g. "×500", "×2", "+0.5%", "−80%".
3. **Why it helps here.** This is the single picture behind all of §5. Every initialization is a point on one conserved curve, and training moves it horizontally to $a=s$. Whether the kernel freezes is just whether the curve is flat over that stretch, which is the condition $\max(|a_0|,s)\ll\Delta$. The large-balanced case shows at once why "large initialization" does not by itself imply lazy training. The explorer's panel B shows $k(t)$ but not this geometry.
4. **Source type: (b) closed form, static PNG**, under 1 s. My prototype renders cleanly with a log y-axis; label the segments directly rather than with a legend. No paper figure exists for this to my knowledge, so this is new; CRW may want to sanity-check that it reads well.
5. **Caption draft.** "Each mode moves on a curve of constant $\Delta$, $k=\sqrt{4a^2+\Delta^2}$, from its initial strength to its target $a=s$ (dotted). Thick segments show the stretch travelled. Small $\Delta$: the kernel grows by orders of magnitude (rich). Large $\Delta$: the curve is flat over the stretch, so the kernel is frozen (lazy). Large balanced weights ($\Delta=0$, $a_0=5s$) slide down $k=2a$ and lose 80% of their kernel, so large weights alone do not make training lazy."

## F9 (P2). Plot the kernels in the "stagewise versus exponential, and which kernels move" cell

1. **Anchor.** §5, the code cell after "#### Code: stagewise versus exponential, and which kernels move" (`def kernel_op(W1, W2): ...`, 2×3 grid).
2. **What it shows now.** Mode strengths $a_\alpha(t)$ for the five initializations plus a 3-layer run. All six slots are used. The heading promises "which kernels move", but the kernels appear only as numbers in panel titles and the printout.
3. **Suggested changes.**
   - Add a second, small figure in the same cell: one log-y axis showing $\max_\alpha|k_\alpha(t)/k_\alpha(0)-1|$, or $k_1(t)/k_1(0)$, for each two-layer initialization, with time normalized by each run's horizon $T$. The small-Gaussian curve climbs about five decades, the two lazy runs stay near 1–2%, and the two large-balanced runs fall to about 0.03–0.2.
   - Note in each panel title that the time axes differ ("lazy runs finish in $0.2\tau$"). The 0–16, 0–1 and 0–6 ranges invite wrong speed comparisons at a glance.
   - Title the "large balanced, random" panel "(no per-mode prediction: modes not separated)". The markdown explains this, but the panel alone looks as if an overlay is missing.
4. **Source type: (d) existing code cell, an added plot.** Kernel histories can be computed from the `hist` already recorded (about 200 snapshots per run), so the cost is negligible.
5. **Caption/markdown draft.** "Bottom: relative change of the kernel on the four target modes. The small initialization's kernel grows by five orders of magnitude, the two lazy initializations move by 1–2%, and the large balanced ones lose most of their kernel, although they start with the same kernel as the lazy runs."

## F10 (P3). Tu–Aranguri–Jacot phase diagram (§5.4)

1. **Anchor.** §5.4, after "It finds lazy behaviour when $\sigma^2w\gg n$ and active or mixed behaviour when $\sigma^2w\ll n$."
2. **What it shows.** Tu, Aranguri and Jacot 2024, **Fig. 2, top-left panel**: the minimum test error over the plane of exponents ($\gamma_w$ horizontal, $\gamma_{\sigma^2}$ vertical, with $w=d^{\gamma_w}$ and $\sigma^2=d^{\gamma_{\sigma^2}}$). The lazy/active boundary is $\gamma_{\sigma^2}+\gamma_w=1$. The top-right panel (stable rank) shows the low-rank bias. I verified the caption and axes in arXiv 2405.17580.
3. **Why it helps here.** §5.4 describes this phase diagram in words. A glance at the line $\gamma_{\sigma^2}+\gamma_w=1$ makes the "$\sigma^2w$ against $n$" threshold concrete, and it shows the generalization payoff that F8 does not.
4. **Source type: (a) paper figure.** The licence is **CC BY-SA 4.0** (verified on the abs page): embedding with attribution is fine, but ShareAlike may attach to an adapted version, so **CRW should decide** whether to embed it or leave the reading link. A self-drawn schematic of the $(\log h,\log\sigma^2)$ plane with the line $\sigma^2h=s$ would add little beyond F8.
5. **Caption draft.** "Minimum test error of a wide two-layer linear network learning a noisy low-rank target, as the width $w=d^{\gamma_w}$ and initialization variance $\sigma^2=d^{\gamma_{\sigma^2}}$ are scaled with the dimension $d$. Below the line $\gamma_{\sigma^2}+\gamma_w=1$ training is active or mixed and recovers the low-rank target; above it, training is lazy and fits the noise. From Tu, Aranguri and Jacot (2024), Fig. 2, CC BY-SA 4.0."

## F11 (P2). Semantic development: the planted tree, its SVD, and the stages it produces

1. **Anchor.** §6, item **2. Semantic development as progressive differentiation**, after "...applied to a spectrum built to have that order (Exercise 6.2 builds it)."
2. **What it shows.** Four panels, left to right:
   - **(a)** a binary tree over 8 items;
   - **(b)** the right singular vectors of the diffusion-generated item–feature matrix, as a red–blue heatmap (rows are modes, columns are items), showing the level-by-level splits;
   - **(c)** the singular values grouped by tree level (1, 1, 2, 4);
   - **(d)** the predicted mode trajectories $a_\alpha(t)$ from the logistic, deep (solid) against shallow (dashed).
   
   Use $\epsilon=0.05$, where the notebook's own Exercise 6.2 finds visible stages. Either add a faint inset of $\epsilon=0.15$ (one blurred drop), or state in the caption that the stages need strong hierarchy.
3. **Why it helps here.** This is the canonical image of the semantic-development story (PNAS 2019 Fig. 3 and 4). Today it appears only inside Exercise 6.2, a `code_answer` stub in the exercises variant. Showing it statically lets the student see the *Derived/Planted* split at a glance: panels (a)–(c) are what was planted, and panel (d) is what the dynamics add.
4. **Source type: (b) generate ourselves, as a static PNG.** Reuse Exercise 6.2's `tree_features` with $n_{\rm feat}=4000$, plus the closed-form logistic; this takes 1–2 s. Why not the paper figure:
   - Saxe, McClelland and Ganguli 2019, **Fig. 3** (verified caption: "(A) SVD of input-output correlations ... (C) Time-varying trajectories of the deep network's effective singular values $a_i(t)$ ... (D) ... a shallow network's ...") uses, per their SI, "a minimal hand-crafted hierarchical dataset with N3 = 7 features, N2 = 16 hidden units, and N1 = P = 4 items" (verified text).
   - Its clean stages therefore come from hand-chosen singular values. That is a confound for exactly the point §6 makes. A self-generated version can show the dependence on $\epsilon$ honestly.
   - The arXiv licence is non-exclusive and the PNAS licence is unverified, so don't embed in any case; cite Fig. 3 and 4.
5. **Caption draft.** "(a) Eight items at the leaves of a tree. Each feature is generated by flipping a ±1 value with probability $\epsilon=0.05$ at every branch. (b) The right singular vectors of the item–feature correlations are the tree's splits, and (c) their singular values fall level by level. (d) The mode ODEs turn that spectrum into coarse-to-fine stages in a deep network (solid) but not in a shallow one (dashed). The ordering was planted in the data; the timing and the stage-like shape come from the dynamics. Compare Saxe, McClelland and Ganguli (2019), Figs. 3–4, which use a hand-crafted four-item dataset."

## F12 (P3). Singular values of products of random layers: Gaussian against orthogonal

1. **Anchor.** §6, item **4. Orthogonal initialization and dynamical isometry**, after "A product of scaled Gaussian layers has a widely spread spectrum, with some directions near zero."
2. **What it shows.** Histograms (log x-axis) of the singular values of $W_N\cdots W_1$ for $N=1,2,4,8$ with i.i.d. $\mathcal N(0,1/n)$ layers ($n=200$, a few realizations each). A vertical spike at 1 marks the orthogonal product.
3. **Why it helps here.** This is the visual content of "dynamical isometry" in one glance. Depth spreads the Gaussian spectrum over decades, so some modes start with near-zero strength and therefore get long plateaus, while orthogonal layers keep every value at 1.
4. **Source type: (b) generate ourselves**, as a static PNG or a short code cell; the SVDs of 200×200 matrices take under 1 s. Saxe et al. 2014 **Fig. 6B** (verified caption: histograms of the singular values of products of $N_l-1$ random Gaussian $N\times N$ matrices) is the paper version, but its arXiv licence is non-exclusive, so regenerate and cite.
5. **Caption draft.** "Singular values of a product of $N$ random Gaussian layers, scaled to preserve norms on average, spread over orders of magnitude as $N$ grows. A product of orthogonal layers has every singular value equal to 1. After Saxe et al. (2014), Fig. 6B."

---

## Gaps and issues noticed

1. **Canonical plots are invisible in the exercises variant.** The sigmoid plot (Exercise 1.5), the plateau-scaling plot (Exercise 2.3) and the tree SVD (Exercise 6.2) are all `code_answer` stubs there. This is the main reason for F1 and F11.
2. **The solutions `.ipynb` has no saved outputs.** A student browsing it on GitHub, nbviewer or a phone sees no plots at all. **CRW decision:** either accept this (students run the cells), commit executed solutions, or rely on the static PNGs proposed here for the hooks.
3. **"Which kernels move" is promised but not plotted** (§5 code-cell heading); see F9. The five-initialization panels also use different time ranges without saying so.
4. **Loss panels are on linear y-axes.** In the alignment cell (F3) and Exercise 6.2's loss-curve plot, the later stages of the staircase are invisible, and log-y would show them. For Exercise 6.2 that is a one-line change, `ax.set_yscale("log")`. With it, the $\epsilon=0.05$ curve's stage structure is clearer.
5. **Mixed-regime cell (§5.4, `svals_m = [8.0, 3.0, 0.25, 0.05]`).** The plots are correct and the predictions overlay well. Minor improvements:
   - right panel on a log y-axis, with the predicted levels $\sqrt{4s^2+\Delta^2}/\Delta$ as short dashed ticks;
   - a horizontal reference at the threshold, or at least the threshold $s\approx\Delta/2$ in the title.
   
   P3, not listed as a proposal.
6. **Orthogonal-initialization cell (§6, `for want in [+1, +1, -1, -1]`).** It is print-only. A plot of the smallest singular value of $W(t)$ for one matching and one opposite-sign run would show the stall (one value decaying to 0, loss pinned at 0.125). The printout already makes the point, so this is optional.
7. **The lecture outline already describes F1.** L1 segment 2, "Picture to describe": "Draw three S-curves ... Below them, the loss as a staircase". F1 would serve as that slide. Segment 3's "picture to describe" (rich curves climbing, lazy curves flat) is the explorer's panel B, and F8 is a static alternative to it.
8. **References to figures that don't exist.** I checked all of them and found none broken:
   - Notebook text: "§5's figures" (Exercise 6.1), "the cosine and off-diagonal panels of the alignment cell", and the explorer presets ("large balanced init", "mixed", "deep (N = 4)", "show shallow model (N = 1) dashed") all exist.
   - Paper citations: "their Fig. 6" (Saxe 2014, depth and learning time on MNIST, matches the caption) and "their Fig. 2" (PNAS 2019, MDS of the Rogers–McClelland network, matches the caption).
9. **Embedding mechanics.** Image links are checked by `_validate` like any local link, so PNGs must be generated before `build_td1_linear_networks.py` runs. Static PNGs should use a white (non-transparent) background so they stay readable in dark-mode Jupyter.
10. **Licensing summary (verified on the arXiv abs pages).**

    | Paper | arXiv | Licence | Use here |
    |---|---|---|---|
    | Saxe et al. 2014 | 1312.6120 | non-exclusive | regenerate and cite |
    | Saxe et al. 2019 (PNAS) | 1810.10531 | non-exclusive | regenerate and cite |
    | Jacot et al. 2021 | 2106.15933 | non-exclusive | regenerate and cite |
    | Tu, Aranguri, Jacot 2024 | 2405.17580 | CC BY-SA 4.0 | can embed; ShareAlike caveat |
    | Atanasov, Bordelon, Pehlevan 2022 | 2111.00034 | CC BY 4.0 | can embed with attribution |

    The PNAS published version's licence was not checked.
