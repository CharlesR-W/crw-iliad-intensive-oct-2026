# Figure plan: TD2.4, optimizer geometry (`td2_4_optimizer_geometry`)

- **Notebook:** NB2 §4, "Optimizers as steepest descent under a norm". Builder: `td/build_td2_4_optimizer_geometry.py`, read in full on 2026-10-09 (54 cells).
- **Date:** 2026-10-09. This is a plan only. No builder, notebook, asset or check file was touched.
- **Count:** 11 proposals: **3 × P1, 4 × P2, 4 × P3**.
- **What already exists.** There is one interactive figure, `assets/interactive/steepest_descent_balls.html`, linked from §4.1 through `figure()`. There are five plotting code cells: 4.1.D (a code exercise), 4.2.D (a demo), 4.3.C (a demo), 4.4.D (a code exercise), and 4.5.D plus 4.5.E (one exercise, one demo). The solutions `.ipynb` has **no saved outputs** (0 of the code cells have outputs). So a student sees **no image at all** in this notebook without running code or opening the HTML file in a browser. That is the main reason the P1 statics below are static PNGs placed in the markdown.
- **Conventions assumed.** Static PNGs go in `td/assets/figures/td2_4_*.png`. One generator script, `td/assets/figures/make_td2_4_figures.py`, produces them all; it runs once and is not part of the `--run` budget. Embed them with `![alt](assets/figures/...)`. `nbkit_td._validate` already checks local link targets, so a missing PNG fails the build. Use the interactive figure's palette (ℓ2, ℓ∞, ℓ1, Mahalanobis colours) for the ball figures, so the static and the interactive figures read as one system.
- **Licensing.** Every paper figure I checked is under the arXiv "perpetual non-exclusive license". That licence does not let us re-host the figure. So I propose **self-generated redraws with "after X, Fig. N" attribution, or a link to the paper figure**, never an embedded copy. The Muon blog's figures are Desmos screenshots with no stated licence; redraw them too.

---

## Proposals in notebook order

### F1. §4.0: KL, JS and W₁ against the separation θ (P2)

1. **Anchor.** §4.0 "A parable", default text, right after the sentence ending "…Jensen–Shannon is constant, and the Wasserstein-1 distance is $|\theta|$." It goes before the paragraph "KL and JS compare the two distributions point by point."
2. **What it shows.** Two panels.
   - (a) Schematic: two parallel vertical segments, $P_0$ at $x=0$ and $P_\theta$ at $x=\theta$, as in WGAN Example 1. A double arrow labelled θ joins them.
   - (b) Divergence against θ on $[-1,1]$:
     - $W_1=|\theta|$ is a V, labelled "slope ±1: usable gradient".
     - JS is a flat line at $\log 2$, with an open circle at θ = 0 and a filled point at (0, 0), labelled "flat: zero gradient".
     - KL is shown as a shaded band or arrow at the top labelled "+∞ for θ ≠ 0".
3. **Why it helps.** The parable is a picture: one curve has a slope and the others don't. The text gives three numbers, while the figure shows at a glance why only one of them is a usable loss. That picture is the "lesson" sentence of §4.0. It doesn't pre-empt 4.0.A beyond what the text already states (the exercise uses point masses, and the figure shows the WGAN segments).
4. **Source.** (b) Self-generate in matplotlib from closed forms, as a pre-rendered static PNG. Attribute it "after Arjovsky, Chintala and Bottou 2017, Fig. 1". I checked that figure in the PDF text (arXiv 1701.07875). Its caption reads: "These plots show ρ(Pθ, P0) as a function of θ when ρ is the EM distance (left plot) or the JS divergence (right plot). The EM plot is continuous and provides a usable gradient everywhere. The JS plot is not continuous…" Our version adds KL and puts the curves on one axis. Don't embed theirs (arXiv non-exclusive licence).
5. **Priority.** P2.
6. **Caption draft.** *Two parallel segments a distance θ apart (left) and three ways of comparing them (right). KL is infinite and Jensen–Shannon is stuck at log 2 for every θ ≠ 0, so neither gives gradient descent a direction; the Wasserstein-1 distance grows like |θ| and its slope points back toward the data. After Arjovsky et al. (2017), Fig. 1.*

### F2. §4.1: the four unit balls and their steepest steps, static (P1)

1. **Anchor.** §4.1, default text, between "Working out $d^\star$ for a few norms gives a dictionary:" and the dictionary table. Alternatively, put it immediately before the `figure(...)` link to `steepest_descent_balls.html`. Either way the link should then read "an interactive version of the figure above".
2. **What it shows.** One row of four square panels with shared axes, all using the same gradient $g=(1.0,0.35)$:
   - ℓ₂ disc
   - ℓ∞ box
   - ℓ₁ diamond
   - Mahalanobis ellipse (aspect 4, angle 30°, matching the interactive figure's defaults)

   Each panel has:
   - a faint linear colour gradient of $\langle g,\Delta\rangle$ in the background
   - the black arrow $g$ and the grey arrow $-g$
   - the dashed supporting line $\langle g,\Delta\rangle=-\eta\|g\|_*$
   - the coloured steepest step $\Delta_{\rm TR}$ at the tangent point
   - the step formula under the panel: $-\eta g/\|g\|_2$, $-\eta\,\mathrm{sign}(g)$, $-\eta\,\mathrm{sign}(g_1)e_1$, $-\eta M^{-1}g/\|g\|_*$
   - the optimizer name in the panel title: normalized GD, sign descent / Adam with EMAs off, greedy coordinate descent, natural gradient

   An optional fifth small panel, "spectral norm on diag(θ₁, θ₂) = the ℓ∞ box", would carry the interactive figure's Muon note.
3. **Why it helps.** This is the canonical image of the section, and the course brief names it: "steepest-descent unit balls for different norms". Today it exists only as an external HTML file that many notebook viewers show as source, and a student reading the notebook sees nothing. The table gives formulas, while the picture shows *why* corners pull the step to a vertex or an axis. It is also the image a student uses to decide whether to read §4.1. The interactive figure stays as the place to play. The static is the hook.
4. **Source.** (b) Self-generate a pre-rendered PNG. The geometry is closed-form. Reuse the step formulas already verified by `assets/interactive/checks/steepest_descent_balls_check.py` (302 cases), or brute-force over boundary points as that check does. Cite it "in the style of Bernstein and Newhouse 2024, Fig. 1". I verified on arXiv HTML 2409.20325 that their only figure overlays norm balls on a linear colour gradient with arrows marking the solution. Its caption reads: "a) Increasing the sharpness decreases the size of the solution vector. b) Changing the norm can change the direction of the solution vector." Don't embed theirs (arXiv non-exclusive licence). Ours also adds ℓ₁ and the Mahalanobis ellipse and matches the interactive figure's colours.
5. **Priority.** P1.
6. **Caption draft.** *The same gradient g against four unit balls. The steepest-descent step is the point of each ball that lies furthest along −g, where the dashed level set of ⟨g,·⟩ touches the ball: −g itself for ℓ₂, a corner (−sign g) for the ℓ∞ box, a single coordinate for the ℓ₁ diamond, and −M⁻¹g, pulled toward the ellipse's long axis, for a Mahalanobis norm. The interactive version lets you drag g and switch to the penalty form.*

### F3. §4.1.D code cell: small fixes to the existing plot (P3, existing plot)

1. **Anchor.** Code exercise "4.1.D Code: trust-region steps on an ill-conditioned quadratic", solution variant.
2. **What to change.** The existing three-panel plot works (path on contours, loss at constant η, loss with decayed η). Two fixes:
   - Panel 0 has labelled lines but no `legend()`, so the paths are hard to tell apart without matching colours to panel 2. Mark θ₀ and the minimum, and add a legend (or reuse panel 2's legend).
   - Optionally, add a zoomed inset of the last 20 steps of sign descent and normalized GD near the origin. That makes the "rattle at distance ~η" claim in the answer visible rather than only inferred from a flat loss curve.
3. **Why it helps.** The answer cell's key claim is that trust-region steps don't shrink near the minimum. Today the student infers it from a plateau, and the inset would show it directly.
4. **Source.** (d) Change the existing code cell. It costs nothing.
5. **Priority.** P3.
6. **Caption.** Not needed (code-cell output). Optionally set the panel-0 title to "first 40 steps (θ₀ = ●, minimum = ✕)".

### F4. §4.2.D code cell: add parameter-space trajectories (P2, existing plot)

1. **Anchor.** "4.2.D Code: GD, natural gradient and Newton under a nonlinear reparametrization", the `code(...)` cell that defines `run(method, coords, eta, ...)`. Its answer cell begins "*What the plot shows.* GD follows different trajectories in the two coordinate systems…"
2. **What it shows.** Add a fourth panel, or a second row: NLL contours in the $(\theta_1,\theta_2)$ plane, with all six runs (GD, natural gradient and Newton, each in θ and in ξ) drawn as paths, all mapped to θ through $\phi$:
   - the two natural-gradient paths nearly overlap
   - the two GD paths diverge
   - Newton in ξ slides down to $\theta_2\to0$ while Newton in θ goes to the optimum

   Recipe: make `run` also return the list of θ iterates (`phi(xi)` for the ξ runs). There is no extra compute.
3. **Why it helps.** The answer describes *trajectories* ("GD follows different trajectories", "drives θ₂ = e^{ξ₂} toward 0"), but the plot shows only loss curves. Different paths can have similar loss curves, so the invariance claim is not actually visible. Today the Newton failure shows up only as "ends at θ = (1.26, 0.00)" in a legend. Coordinate invariance is a statement about paths in parameter space, so it should be drawn there.
4. **Source.** (d) Extend the existing code-cell plot. This is a `code()` demo, so the plot appears in both variants.
5. **Priority.** P2.
6. **Caption.** (Folded into the answer cell.) *Paths in θ, with each ξ run mapped back through φ. Natural gradient traces nearly the same path in either coordinate system; GD does not; Newton in ξ, whose Hessian is indefinite away from the optimum, runs to θ₂ = 0.*

### F5. §4.2: "sharpness depends on the ruler" schematic (P3)

1. **Anchor.** §4.2 default text, the paragraph beginning "The Hessian at a critical point transforms like the Fisher, $H_\xi=J^\top H_\theta J$: it is a quadratic form, not a linear map." Place it after "…so it changes under reparametrization even though the function does not."
2. **What it shows.** Two panels showing the same quadratic loss:
   - Left: in θ, with nearly round contours. Annotate $\lambda_{\max}(H_\theta)=1$.
   - Right: in ξ with $\theta_2=c\,\xi_2$ ($c=5$), where the contours are squeezed into a thin ellipse. Annotate $\lambda_{\max}(H_\xi)=25$.

   A shared annotation reads "$\mathrm{spec}(F^{-1}H)$ unchanged".
3. **Why it helps.** It makes "a quadratic form has no eigenvalues until you pick a metric" visual: same function, different-looking basin. It is optional, because MDL WB2 §4 already works Dinh's example.
4. **Source.** (c)/(b) Self-generate from closed form; a tiny static PNG.
5. **Priority.** P3.
6. **Caption draft.** *One loss, two coordinate systems. Stretching one coordinate by 5 multiplies the raw sharpness λ_max(H) by 25 without changing the function; the eigenvalues of F⁻¹H, which compare curvature with the model's own metric, do not move.*

### F6. §4.3: why Adam beats SGD, per-frequency-class losses (P2)

1. **Anchor.** §4.3 default text, the paragraph "**Why Adam beats SGD on transformers.**", after the sentence ending "…prove, in a simplified linear softmax model in continuous time, that gradient descent is slow on low-frequency classes while sign descent is not."
2. **What it shows.** Training loss against step for linear softmax regression on Zipf-distributed classes, with classes split into four frequency quartiles. Use two panels (GD, Adam) with one curve per quartile, or one panel with solid lines for GD and dashed for Adam. Under GD the frequent quartile falls fast while the rare quartile barely moves. Under Adam all quartiles fall together.
3. **Why it helps.** This paragraph carries the section's main empirical answer to "why Adam", and today it is prose only. The per-group loss curves are the paper's anchoring image. A student who sees them understands at once *what* "GD is slow on rare classes" means, and how this connects to sign descent's indifference to gradient scale (4.3.A.2).
4. **Source.** Either:
   - **(a) Link** Kunstner et al. 2024 (arXiv 2402.19449), **Fig. 1**. I verified it on arXiv HTML. Caption: "Gradient descent does not make progress on low-frequency classes, while Adam does. Training GPT2-Small on WikiText-103. … (c, d) Training loss for each group using SGD and Adam." Its licence is arXiv non-exclusive, so link it rather than embed it.
   - **(b) Self-generate**, which I recommend as a small `code()` demo or a pre-rendered PNG. Recipe: K = 100 classes with $p_c\propto1/c$, N = 2000, d = 32, class means $\mu_c\sim\mathcal N(0,I)$, $x=\mu_{y}+\sigma\varepsilon$ with σ ≈ 1 (harder than my prototype's 0.5, so Adam doesn't hit 0 instantly). Full-batch GD (η ≈ 1) against Adam (η ≈ 0.01) for 150 steps, plotting the mean loss per frequency quartile.

   My prototype (K = 300, N = 6000, d = 64, σ = 0.5, 300 steps) confirms the qualitative effect. At GD η = 1 the final quartile losses were 0.01 / 0.08 / 0.18 / 0.57, frequent to rare, while Adam reached ≈ 0 on all four. But it took ~50 s per run under heavy machine load, so shrink it as above and time it before adding it as a code cell. Precompute the group masks.

   The paper's figure is a real transformer, and ours is a toy model that matches the paper's own theoretical setting. Saying so in the caption is the honest option.
5. **Priority.** P2.
6. **Caption draft.** *Linear softmax regression with heavy-tailed (Zipf) class frequencies, loss per frequency quartile. Gradient descent fits the frequent classes and barely moves the rare ones; Adam, whose per-coordinate normalization ignores gradient scale, reduces all four together. A toy version of Kunstner et al. (2024), Fig. 1, which shows the same split for GPT-2 Small on WikiText-103.*

### F7. §4.3.E: decoupled weight decay confines sign-descent weights to a box (P3)

1. **Anchor.** "4.3.E Decoupled weight decay as an implicit constraint", after the solution to part 2 or after the problem statement.
2. **What it shows.** A 2D quadratic whose unconstrained minimum lies outside the box $|\theta_i|\le1/\lambda$ (e.g. minimum at (3, 0.5), λ = 0.5, so the box is ±2). Plot sign descent with decoupled decay from a start outside the box: the trajectory enters the box and settles on its face at $\theta_1\approx2$, the KKT point. Overlay AdamW-style coupled L2 (Adam + L2) for contrast, if wanted.
3. **Why it helps.** "Weight decay picks the radius of a constraint set in the optimizer's own norm" is easier to believe once you see the iterate pinned to the box face. It is cheap and optional.
4. **Source.** (b) Self-generate, < 1 s; a code-cell demo or static.
5. **Priority.** P3.
6. **Caption draft.** *Sign descent with decoupled weight decay λ on a quadratic whose minimum lies outside the box |θᵢ| ≤ 1/λ. The iterates enter the box and settle on its face, at the box-constrained minimizer: the norm of the steps (ℓ∞) sets the shape of the constraint, and λ sets its size.*

### F8. §4.4: Newton–Schulz maps, one step and five steps, on a log axis (P1)

1. **Anchor.** §4.4 default text, after "…after the authors found that errors of about $\pm0.3$ in the singular values do not hurt training." and before "Amplifying small singular values quickly is the design goal".
2. **What it shows.** Two panels.
   - (a) One-step maps on σ ∈ [0, 1.25]: the cubic $p$, the Muon post's baseline quintic $(2,-1.5,0.5)$ and Muon's tuned quintic $\varphi$. Add the identity line and a shaded band [0.7, 1.3].
   - (b) Five-fold composites $p^{5}$, baseline$^{5}$ and $\varphi^{5}$ against **log₁₀ σ₀** on [10⁻⁴, 1], plus $p^{20}$ dashed, with the same band. Annotate the thresholds already derived in the notebook: φ⁵ ≥ 0.6 for σ₀ ≳ 1.3×10⁻³; the cubic needs σ₀ ≳ 0.088 in 5 steps.

   The composite in (b) shows:
   - the tuned quintic's oscillation inside ~[0.7, 1.2]
   - its ~485× amplification near 0
   - the cubic's slow start
4. **Why it helps.** §4.4 is anchored on exactly this picture. The Muon post's own figures are these maps and their five-fold iterates. Today the notebook shows only one-step maps, and only inside a code *exercise* (4.4.D), which is blank in the exercises variant. The log-σ axis is the important improvement. On the post's linear axis the near-zero behaviour, which is the whole design goal, is a vertical sliver.

   **Trade-off for CRW.** Panel (b) shows the answer to 4.4.C.3 ("where do starting points end up after 5 steps?"). The default text already states it qualitatively ("settling in roughly [0.7, 1.3]"), and solutions are open to everyone, so I'd place it in the default text anyway. If not, show only panel (a) up top and move (b) into the 4.4.C solution.
4. **Source.** (b) Self-generate a pre-rendered PNG from the closed-form polynomials (the numbers are already verified in `td/checks/td2_4_optimizer_geometry_math.py`). Redrawn after the Muon post's two figures, which I verified by downloading them:
   - `kellerjordan.github.io/images/muon/poly_phi.png`, captioned "Baseline coefficients for Newton-Schulz iteration": φ = 2x − 1.5x³ + 0.5x⁵ and its 5-fold composite, flat at 1 for x ≳ 0.1.
   - `poly_phirho.png`, captioned "Tuned coefficients for our Newton-Schulz iteration": ρ = 3.4445x − 4.7750x³ + 2.0315x⁵ and its 5-fold composite, which oscillates between ≈ 0.7 and 1.2.

   They are Desmos screenshots with UI chrome and no stated licence, so redraw rather than embed. Note that the notebook compares against the **cubic** while the post compares against its **baseline quintic**. Include all three curves so the figure matches both.
5. **Priority.** P1.
6. **Caption draft.** *Newton–Schulz iterations act on each singular value through a scalar polynomial. Left: one step of the classical cubic, the Muon post's baseline quintic and Muon's tuned quintic. Right: five steps, with the starting singular value on a log axis. The tuned quintic lifts everything above about 10⁻³ into a band around 1 (shaded) but never converges to 1; the cubic is exact in the limit but leaves small singular values small after five steps. Redrawn after the figures in Jordan et al.'s Muon post.*

### F9. §4.4: what each optimizer does to a gradient matrix's spectrum (P2)

1. **Anchor.** §4.4 default text, after the paragraph "Why orthogonalize at all? …", which ends "…orthogonalizing raises the scale of 'rare directions' that matter for learning."
2. **What it shows.** One log-y panel of sorted singular values (normalized by the largest) against index, or four small panels, for a 256×256 gradient-like matrix $G$: a rank-4 signal with singular values 8 : 4 : 2 : 1 plus Gaussian noise at 0.05. Four curves:
   - $G$ itself, which is also the SGD step: 1, 0.46, 0.25, 0.13, then a noise floor ≈ 0.012
   - $\mathrm{sign}(G)$, the ℓ∞/Adam step: 1, 0.61, 0.39, 0.24, then a slowly decaying bulk ≈ 0.2 → 0.02 median
   - exact $UV^\top$: flat at 1, *noise directions included*
   - Muon's five-step quintic: mostly in ~[0.7, 1.2] (median 0.74), with only the very smallest left behind (min ≈ 0.006)

   These numbers are from my prototype, which took ~10 s under heavy load.
3. **Why it helps.** The interactive figure's own caption says the 2D picture *cannot* show where spectral steepest descent and sign descent differ. This figure shows exactly that, on a real matrix:
   - sign descent keeps an outlier-plus-bulk spectrum (the Θ(n) top singular value of 4.5.C)
   - orthogonalization flattens everything

   It also makes concrete both the "rare directions" hypothesis and its flip side: exact orthogonalization lifts pure-noise directions to the same size as the signal. That flip side is a real point worth a sentence, and the text is currently silent on it.
4. **Source.** (b) Self-generate a pre-rendered PNG (numpy SVD, < 10 s). Alternatively, place it as the `code()` output at the end of 4.4.D. I prefer the static near the top, because 4.4.D is an exercise.
5. **Priority.** P2.
6. **Caption draft.** *Singular values (largest = 1) of one gradient-like matrix, a rank-4 signal plus small noise, after each optimizer's direction map. SGD keeps the spectrum; sign descent keeps a dominant top direction above a broad bulk; exact orthogonalization (the spectral-norm steepest step) sets every direction, including the noise directions, to 1; Muon's five quintic steps do nearly the same, leaving only the tiniest singular values behind.*

### F10. §4.5: muTransfer, loss against learning rate across widths, SP vs muP (P1)

1. **Anchor.** §4.5 "**muP.**" paragraph, after "Its practical payoff, *muTransfer* (Tensor Programs V), is to tune hyperparameters on a narrow model and reuse them on a wide one." The figure fills a gap the text names in 4.5.E: "Whether the optimal learning rate transfers across width is the empirical question Tensor Programs V answers, and a sweep of it would take longer than this notebook's budget." Update that sentence to point to the figure.
2. **What it shows.** Two panels: training loss after a fixed number of Adam steps against log₂(learning rate), one curve per width.
   - Left, standard parametrization: one LR for all layers, standard init. The optimum drifts left as width grows, and wide nets can be worse than narrow ones at a fixed LR.
   - Right, muP: hidden and output Adam LR ∝ 1/fan-in, output multiplier per TP-V Table 3. The optima line up, and wider is uniformly better.

   Mark each curve's argmin with a dot.
3. **Why it helps.** This is *the* muP image, and §4.5's derivation exists to explain it. The notebook derives the exponents (4.5.B–C) and shows that update *size* scales (4.5.E). It never shows the payoff, a learning-rate optimum that stops moving. One picture tells a student whether §4.5 is worth their time.
4. **Source.** Two options.
   - (a) Paper figure. Tensor Programs V (arXiv 2203.03466), **Fig. 1**, verified from the PDF text. Caption: "Training loss against learning rate on Transformers of varying d_model trained with Adam. Conventionally and in contrast with our technique, different widths do not share the same optimal hyperparameter; wider networks do not always perform better than narrower ones; in fact they underperform the same-width networks in our technique even after tuning learning rate (see dashed line)." It is under the arXiv non-exclusive licence (abs page "view license": nonexclusive-distrib/1.0), so **link it, don't embed it**. Large et al. 2024 Fig. 1 (verified on arXiv HTML) shows the same kind of LR-transfer sweep for the modular norm and suits §4.6.
   - (b) Self-generate a **pre-rendered** PNG, offline and outside the notebook's 3-minute budget. Recipe: a 3-layer MLP (tanh or ReLU) on a fixed synthetic task, such as the sin-teacher regression already in 4.5.E or sklearn digits. Widths 128, 256, 512, 1024, 2048; Adam; 8–10 LRs from 2⁻¹⁴ to 2⁻⁴; ~300 steps; 2–3 seeds.

     **This needs validation before it goes in.** Toy tasks sometimes show a weak SP shift, and the figure must actually show it. A few CPU-minutes, or one Modal job if slow. If the toy doesn't show a clean shift, fall back to (a): a one-line link "see TP-V Fig. 1" plus a schematic sketch.
5. **Priority.** P1.
6. **Caption draft (self-generated).** *Loss after N Adam steps against learning rate, for MLPs of increasing width. Standard parametrization (left): the best learning rate drifts as width grows, so a value tuned on a narrow model is wrong for a wide one. muP (right): hidden- and output-layer learning rates scaled by 1/fan-in line the optima up, and wider is better at every learning rate. A small-scale version of Yang et al. (2022), Tensor Programs V, Fig. 1.*

### F11. §4.5: why alignment makes the spectral bound tight (P3)

1. **Anchor.** §4.5 "**The spectral condition.**", after "…for batch size 1 the update is an outer product $\delta\,h_{l-1}^\top$, whose top right singular vector *is* $h_{l-1}$ (4.5.B)." Alternatively, put it in the 4.5.B.3 solution.
2. **What it shows.** Log–log plot against width $n$ of $\|\Delta W\,x\|/(\|\Delta W\|_2\|x\|)$ for a rank-one $\Delta W=\delta h^\top$:
   - with $x=h$ the ratio is flat at 1
   - with $x$ random the ratio falls like $n^{-1/2}$, with a reference slope drawn

   Optionally, a third series with $\Delta W$ = a real 10-step Adam update from 4.5.E applied to the initial features, which sits near the aligned line.
3. **Why it helps.** The whole width argument hinges on "attained, rather than loose by a power of the width". This plot is that sentence. It is cheap and optional, since 4.5.B.2–3 derive it cleanly.
4. **Source.** (b) Self-generate, < 1 s; a code demo or static.
5. **Priority.** P3.
6. **Caption draft.** *How much of a rank-one update's spectral norm reaches an input. Applied to the features h that produced it, the update acts at full strength at every width; applied to an unrelated input, its effect shrinks like 1/√n. Gradient updates are aligned by construction, which is why the spectral norm, not the Frobenius norm, sizes them.*

### F12. §4.6: modular-norm composition schematic (P3)

1. **Anchor.** §4.6 default text, after the displayed composition formula and "…(their Definition 3 …; concatenation is the same without $s_2$)."
2. **What it shows.** A schematic tree with two leaves:
   - $M_1$ (mass μ₁, sensitivity s₁, norm ‖·‖₁) and $M_2$ (mass μ₂, sensitivity s₂) feeding a "∘" node, labelled "masses add, sensitivities multiply"
   - an arrow from $\Delta w_1$ passing through $M_2$ and amplified by s₂, to illustrate the $s_2$ factor
   - a pie or bar showing step budget split as μᵢ/(μ₁+μ₂)

   A second small strip shows the L-chain with each layer labelled "spectral radius η/L".
3. **Why it helps.** The composition rule has three moving parts, which the prose explains in two "intuition" sentences. The diagram shows them in one look. It is optional.
4. **Source.** (c) Draw it ourselves, in the spirit of Large et al. 2024, Fig. 2. I verified on arXiv HTML 2405.14813 that its caption reads "Modules and trees of modules … endowed with two numbers—a mass and sensitivity—and a norm … New compound modules are built via the binary operations of composition and concatenation." Their licence is arXiv non-exclusive, so link it rather than embed it.
5. **Priority.** P3.
6. **Caption draft.** *Composing two modules. Masses add and set each module's share of a step; sensitivities multiply, and the second module's sensitivity scales how far a change in the first module's weights can move the output. For a chain of L equal layers this gives each layer a step of η/L.*

---

## Gaps and issues noticed

1. **No saved outputs.** `td2_4_optimizer_geometry_solutions.ipynb` has outputs in none of its code cells, so none of the five plotting cells shows anything until run. Two of the plot cells (4.1.D, 4.4.D) are code exercises whose stubs produce no plot in the exercises variant. This is the main argument for the P1 statics (F2, F8, F10). The review log says saved outputs are a day-wide coordinator decision.
2. **The interactive figure is the notebook's only picture of its core idea, and it is off-page.** `figure()` emits a link that "some notebook viewers show … as source". F2 is its static companion. Once F2 exists, reword the `figure()` text to "an interactive version of the figure above".
3. **4.2.D answer misdescribes its plot.** The answer says "GD follows different trajectories in the two coordinate systems" and that Newton "drives θ₂ = e^{ξ₂} toward 0". The plot shows only loss against step, and the θ₂ → 0 failure appears only as a number in a legend label. F4 fixes this.
4. **"The code in 4.2.D shows both effects"** (end of the 4.2.A.3 solution). The two effects are first-order agreement, and a large O(η²) term when J is nearly singular. 4.2.D starts at θ₂ = 0.5 and shows the first only. The review log says the θ₂ = 0.1 blow-up was tried and dropped. Either add a θ₂₀ = 0.1 natural-gradient run to 4.2.D or change "both effects" to "the first effect".
5. **4.1.D panel 0** has labelled lines but no legend, and θ₀ and the minimum are unmarked (F3).
6. **4.3.C left panel** has no y-axis label. It should be "$\hat m/\sqrt{\hat v}$ (mean ± s.d.)"; the quantity appears only in the legend. A minor fix.
7. **4.4.D panel 0** shows only one-step maps, while the 4.4.C questions are about five-step behaviour. Either add φ⁵ and p⁵ there or rely on F8.
8. **4.4 compares against the cubic, while the Muon post compares against the baseline quintic (2, −1.5, 0.5).** This is consistent, but a student opening the post will see different curves. F8 includes all three to bridge the gap.
9. **Possible content gap surfaced by F9.** Exact orthogonalization lifts pure-noise singular directions to 1, as large as the signal directions. The §4.4 "Why orthogonalize" paragraph gives the "rare directions" hypothesis without this flip side. Whether to add a sentence is the writer's or CRW's call. I checked it numerically: with signal 8 : 4 : 2 : 1 plus noise 0.05 on 256×256, $UV^\top$ is flat at 1, and Muon's quintic gives a median of 0.74.
10. **The figures referenced in prose all exist.** Apart from item 4, I found no text that refers to a figure or plot that is missing.
