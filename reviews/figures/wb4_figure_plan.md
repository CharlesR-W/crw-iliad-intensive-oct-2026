# WB4 (Inductive bias): figure plan

- **Notebook:** WB4, *"Inductive bias" is a four-letter word*. Builder `notebooks/build_wb4.py`, read at its 2026-10-09 morning state. Page `notebooks/04_inductive_bias.html`.
- **Date:** 2026-10-09. This is a plan only. No builder, notebook, nbkit or asset file was edited.
- **Count:** 13 proposals.
  - **P1: 5** (F1, F7, F9, F11, F12).
  - **P2: 4** (F3, F6, F8, F13).
  - **P3: 4** (F2, F4, F5, F10).

**Source types:**
- (a) embed a paper figure;
- (b) generate it ourselves;
- (c) draw a schematic;
- (d) place an existing interactive figure or add a companion to one.

**How claims were checked:**
- Every paper figure number below was checked on 2026-10-09 against the arXiv HTML (or, for MacKay, the Caltech PDF).
- Licences come from the arXiv abs pages or the publisher page.
- "Sim checked" means I ran a cut-down version of the recipe in a scratchpad on this laptop's CPU (6 torch threads) and report what came out.

**Infrastructure (same note as the sibling plans).** `nbkit.figure()` only links to an interactive page. Inline static images need a small helper, e.g. `image(path, caption, credit)` that emits a `<figure>` with a caption and a credit line. They also need a `notebooks/assets/figures/` directory, plus generator scripts under `notebooks/assets/interactive/checks/` or a new `notebooks/figures_src/`. This is a maintainer/CRW decision.

**No new interactive figures are proposed.** None of the WB4 intuitions needs a slider to land; the static figures below do the job. The existing interactive pages (spectral filters, double descent, SGD-vs-posterior, superposition) belong to WB2/WB5 and have no natural WB4 placement.

---

## §2. Occam, two ways

### F1. Why Bayes embodies Occam: data-space and parameter-space pictures. P1

- **Anchor:** §2.A "Bayesian Occam: volume". Place it right after the paragraph ending "*so it put little probability on whichever one arrived. Nothing here refers to how long the model is to write down.*" and before "*With $p$ parameters and $n$ data points, the Laplace approximation…*".
- **What it shows:** two panels.
  - **Left (data space).** The horizontal axis is "possible datasets D", collapsed to one dimension. Two normalized curves:
    - P(D|H₁): a simple model, tall and narrow;
    - P(D|H₂): a flexible model, low and broad.
    - The two areas are visibly equal (annotate "each integrates to 1"). A shaded band C₁ marks the region where H₁ wins, with an observed dataset D₀ marked inside it.
  - **Right (parameter space, one parameter).** A broad prior π(θ) with width Δ_π bracketed, and a sharp likelihood/posterior peak at θ̂ with width Δ_post bracketed. The ratio Δ_post/Δ_π is labelled "Occam factor".
  - Use the Exercise 2.A.i model for concreteness: prior N(0,τ²), posterior width σ/√n. A small caption number such as "τ=3, σ=1, n=25 → Occam factor ≈ 0.07" ties it to the exercise.
- **Why it helps:** the whole subsection rests on this picture. "Wastes prior volume on data that did not occur" is a spatial statement, and readers parse it faster from a picture than from prose. The left panel carries the conservation-of-probability argument, which the text states only in words. The right panel is exactly the Δ_post/Δ_π that Exercise 2.A.i step 3 asks students to match.
- **Source:** (c) and (b): a redrawn schematic, with the right panel computed in closed form with matplotlib.
  - The canonical layout is MacKay, *Bayesian Interpolation* (1992), **Fig. 2** ("Why Bayes embodies Occam's razor": data-space axis, P(D|H₁) vs P(D|H₂), region C₁) and **Fig. 3** ("The Occam factor": prior width Δ⁰w, posterior width Δw). I checked both captions in the Caltech PDF.
  - Rights on that record: "No commercial reproduction…". Redraw rather than embed, and credit "after MacKay 1992, Figs. 2–3".
- **Caption draft:** *Left: every model spreads one unit of probability over possible datasets. A flexible model (H₂) can fit more datasets, so it gives less probability to any one of them, and inside region C₁ the simpler H₁ wins. Right: for one parameter, the evidence is roughly best-fit likelihood × Δ_post/Δ_π, the fraction of the prior's range that survives the data. After MacKay (1992), Figs. 2–3.*

### F2. Evidence vs prior width for fixed data. P3

- **Anchor:** the Solution to Exercise 2.A.i, after step 4: "*Evidence comparisons are therefore sensitive to prior widths in a way that likelihood fits are not.*"
- **What it shows:** one panel. −ln Z against log₁₀ τ (τ from 10⁻² to 10⁴) for one fixed dataset (n=25, σ=1, ȳ=0.8), split into its terms:
  - the flat fit term −ln p(D|θ̂);
  - the ½ln(1+τ²/s²) term, which rises like ln τ at large τ;
  - the ȳ²/2(τ²+s²) term, which blows up at small τ.

  The sum is U-shaped, with its minimum near τ ≈ |ȳ|. Annotate the right arm "fit unchanged, Z → 0".
- **Why it helps:** step 4's point is that Z → 0 while the fit stays the same. Seeing the two penalty terms trade off also previews 2.D.i case 1 (the ±10¹⁰⁰ prior).
- **Source:** (b) closed form from step 2 of the exercise; under a second in numpy.
- **Caption draft:** *For fixed data, the evidence of the Gaussian-mean model as the prior width τ varies. The fit term never changes. A prior that is too narrow pays for missing the data (ȳ²/2τ²), and one that is too wide pays the Occam term ½ln(τ²n/σ²).*

### F3. Kraft's inequality as disjoint dyadic intervals. P2

- **Anchor:** Exercise 2.B.i(a), after "*Show that $I_p\cap I_q=\varnothing$ when neither of $p,q$ is a prefix of the other. Conclude Kraft's inequality*". An alternative is the "**Kraft.**" bullet in §2.B.
- **What it shows:**
  - **Top:** a binary tree of depth 4, with the code words of a prefix-free set highlighted as leaves, e.g. {0, 10, 110, 1110}, which is deliberately not complete.
  - **Bottom:** the unit interval [0,1), with each code word's dyadic interval I_p drawn as a coloured bar aligned under its subtree and labelled with 2^{-|p|}.
  - The unused gap [0.1111₂, 1) is hatched and labelled "unused mass: Σ2^{-|p|} = 15/16 < 1".
- **Why it helps:**
  - The proof in (a) is a picture. Once students see that the subtree under p *is* the interval I_p, disjointness is immediate.
  - The hatched gap is also the cleanest image of a semimeasure. The same figure can be referenced from §3.A ("*$M(x)\ge M(x0)+M(x1)$… some programs print $x$ and then halt*"), so one figure serves two sections.
- **Source:** (c), a programmatic schematic in matplotlib or inline SVG, generated from a list of code words.
- **Caption draft:** *A prefix-free code is a set of leaves in the binary tree, and each code word p owns the dyadic interval I_p of length 2^{-|p|} beneath it. Leaves do not nest, so the intervals do not overlap and their lengths sum to at most 1 (Kraft). The hatched gap is probability that no code word claims, which is how the universal prior in §3.A ends up a semimeasure.*

### F4. The two-part code length L(δ) and its 1/√n minimum. P3

- **Anchor:** the Solution to Exercise 2.C.i, after part 4: "*The $\tfrac12\log_2n$ is the number of bits needed to state $\theta$ to the precision the data can confirm.*"
- **What it shows:**
  - Excess code length L(δ) − [−log₂p(D|θ̂)] against log₁₀ δ, for n = 10², 10³, 10⁴ (J=1, Δ_π=10).
  - For one n, the two components are shown dashed: the falling index cost log₂(Δ_π/δ) and the rising fit penalty nJδ²/(24 ln 2).
  - The minima δ* are marked. They step left by ½ decade per decade of n, and their heights rise by ½log₂10 ≈ 1.66 bits per decade.
- **Why it helps:** part 4 ("why does δ* shrink like 1/√n?") becomes visible as the fit-penalty parabola steepening with n. The 1.66 bits per decade is BIC's ½log₂n seen directly.
- **Source:** (b) closed form, matplotlib.
- **Caption draft:** *Stating θ more precisely costs index bits (falling dashed line); stating it less precisely costs fit (rising dashed curve). The best precision δ\* ∝ 1/√n, and each tenfold increase in n raises the minimum code length by ½log₂10 ≈ 1.7 bits per parameter: BIC in bits.*

### F5. Same minimum, sharp or flat, depending on coordinates. P3

- **Anchor:** the Solution to Exercise 2.D.ii, the "**Flatness.**" bullet, after "*A reparametrization can therefore make the same minimum, implementing the same function, as sharp or as flat as you like.*"
- **What it shows:** two side-by-side 1-D loss curves for the same function family.
  - Left: L(θ) = (θ−1)² in θ coordinates.
  - Right: the same loss in φ, with θ = 1 + φ³ (flat bottom) or θ = 1 + 10φ (sharp bottom), overlaid.
  - The minimizer implements the same function in all three; annotate L'' at the minimum in each coordinate system.
  - A shaded "ε-sublevel set" bar under each curve shows its width changing, while the integral with the prior (the evidence) does not.
- **Why it helps:** it is the one-picture version of Dinh et al.'s objection, and it makes "coordinate-dependent" concrete. The material is quick to say but easy to misread without a picture.
- **Source:** (b) closed form. (Dinh et al. 2017, "Sharp minima can generalize for deep nets", makes this point for ReLU rescalings. A self-made 1-D version is clearer and has no licence question. I did not check that paper's figures, so do not cite a figure number.)
- **Caption draft:** *One minimum, three coordinate systems. Curvature and the width of the low-loss region depend on how θ is parametrized, while the function implemented at the minimum and the evidence do not. Flatness stated in raw parameter coordinates is therefore not a property of the learned function.*

---

## §3. Solomonoff

### F6. A Bayes mixture's regret saturates at ln(1/w_μ). P2

- **Anchor:** §3.B, after the Theorem display and the sentence "*This is a bound on the mixture forecaster's cumulative online log-loss, not on any single hypothesis.*" An alternative is the Solution to Exercise 3.B.i(b).
- **What it shows:** cumulative log-loss regret R_n (in bits) against n (0–2000), for a computable toy class.
  - **The class:** every binary periodic pattern of period ℓ ≤ 10, each emitted with flip noise ε = 0.02. The weight of a pattern is w ∝ 2^{-2ℓ}, i.e. "description length ≈ 2ℓ bits", normalized over the class. The class also includes Bernoulli(θ) on a 20-point θ grid.
  - **Curves:** three true environments μ with periods 2, 5 and 9. Each is plotted with a dashed horizontal line at its own log₂(1/w_μ).
  - **Contrast curve:** a learner whose class has only the Bernoulli models. Its regret grows linearly.
- **Why it helps:**
  - The theorem's surprising feature is that total regret is bounded uniformly in n. Saturating curves that sit under their own ceilings make "you pay the description length once" visible.
  - The bars visibly order by complexity, which is the K(μ) ln 2 reading.
  - The linear-growth contrast shows what happens when the truth is not in the class.
- **Source:** (b) exact Bayesian updating over about 2¹¹ hypotheses for n=2000 steps. Under a few seconds in numpy.
- **Caveat for the caption:** this is a finite computable class standing in for Solomonoff's, which cannot be computed. Say so.
- **Caption draft:** *A Bayesian mixture over noisy periodic patterns (weight 2^{-2ℓ} for period ℓ) predicting data from three of them. Each regret curve levels off below its dashed ceiling log₂(1/w_μ): the forecaster pays about the description length of the truth once, however long the sequence. A mixture without the true pattern in its class (grey) pays at a constant rate forever. A computable toy for the theorem in §3.B, not Solomonoff's M itself.*

### F7. Simplicity bias of the random-parameter prior over functions, with the entropy confound shown. P1

- **Anchor:** §3.D "A bridge to networks", immediately after the Valle-Pérez, Camargo and Louis reading card (it ends "*§7 tests whether SGD samples like the posterior.*").
- **What it shows:** random 7-40-40-1 ReLU networks on all 128 inputs of {±1}⁷, with the sign of the output as a Boolean function.
  - **Panel (a):** log₁₀ P(f) (empirical frequency) against the LZ complexity of the 128-bit truth table. Each point is one function seen at least 5 times. A dashed simplicity-bias upper-bound line is fitted to the upper envelope.
  - **Colour:** T = number of ones, folded as min(T, 128−T). This is the entropy confound, made visible.
  - **Panel (b), the confound check:** log₁₀ P(f) against min(T, 128−T) alone, with the same points. Optionally add a third trace with bias variance σ_b = 0, so students see how much of the effect is class-imbalance bias. Mingard et al. (arXiv 1909.11522) prove that P(T) is uniform for a perceptron and that the low-entropy bias strengthens with ReLU depth and bias variance; I checked that claim on the abs page.
- **Why it helps:**
  - This is the canonical image the simplicity-bias account is built on (Valle-Pérez et al. **Fig. 1(b)**, checked in arXiv HTML: probability vs LZ complexity, 7-40-40-1, 10⁸ samples, red AIT bound line).
  - Exercises 2.B and 3.B give students K and 2^{-K(x)}. This figure shows the network-side analogue and lets them judge it.
  - The colouring exposes two things the original plot does not: (i) the relation is an upper bound with a huge spread below it (many simple functions are improbable); (ii) much of the vertical spread is predicted by output entropy. That is the "which simplicity?" question §2.D poses.
- **Source:** (b) preferred over (a).
  - **Sim checked:** in torch on CPU (weights N(0, 1/fan_in), biases N(0,1), batches of 10⁴), 2×10⁵ samples took 29 s and gave 85,234 distinct functions. The most frequent function had frequency 0.26 (presumably constant; not yet inspected). 10⁶ samples will take about 2.5 min, enough to resolve P ≳ 5×10⁻⁶.
  - **LZ complexity:** use Valle-Pérez's symmetrized LZ76 (average of LZ on the string and its reverse, with constants special-cased). Their exact convention is in their appendix; match it and say so.
  - **Alternative (a):** Valle-Pérez et al. 1805.08522 is **CC BY 4.0** (abs page), so Fig. 1(b) can be embedded with attribution. However, its probabilities come from 10⁸ samples, it plots a bound without showing the entropy confound, and the self-generated version is cheap.
- **Caption draft:** *Sample random weights for a 7-40-40-1 ReLU network and record which Boolean function on {±1}⁷ it computes. (a) Frequent functions all have low LZ complexity (dashed line: upper bound), but many low-complexity functions are rare, so the bound is one-sided. (b) Much of the spread tracks how unbalanced the function's outputs are (colour, and panel b), which is a separate notion of "simple". After Valle-Pérez, Camargo and Louis (2019), Fig. 1(b); see Mingard et al. (2019) for the entropy bias.*

---

## §4. Program synthesis and the circuit prior

### F8. Learning curves on sparse parity: counting learner, kernel ridge, and the n/N bound. P2

- **Anchor:** the Solution to Exercise 4.B.i(b), after "*so to do substantially better than nothing a fixed-kernel method needs $n$ of order $N=\binom dk\approx d^k/k!$.*" Or open the solution of (c) with it.
- **What it shows:** test MSE (on ±1 labels, so the zero predictor scores 1) against n on a log x-axis, for d=14, k=3 (N=364, log₂N≈8.5).
  - Counting learner: a random consistent 3-sparse parity. Error falls to 0 by n≈16.
  - Kernel ridge with an isotropic kernel on the cube, e.g. exp(−4·Hamming/d), ridge 10⁻³.
  - The lower bound max(0, 1−n/N), shaded.
  - Vertical lines at log₂N and N.
  - Average over 5–20 random targets.
- **Why it helps:** the exercise proves an inequality, and the figure shows that the inequality is the whole story in practice. The kernel learner sits above the 1−n/N line and only gets traction once n is a multiple of N, while the counting learner finishes before the kernel has started. That makes the factor of about 10⁶ in part (c) (d=100) feel real.
- **Source:** (b). **Sim checked** (5 targets, d=14, k=3, about 10 s total):

  | n | kernel MSE | bound 1−n/N | counting MSE |
  |---|---|---|---|
  | 64 | 0.99 | 0.82 | 0 |
  | 256 | 0.80 | 0.30 | 0 |
  | 512 | 0.54 | 0 | 0 |
  | 1024 | 0.23 | 0 | 0 |
  | 2048 | 0.06 | 0 | 0 |

  Counting is already 0 at n=16. Add n ∈ {4, 6, 8, 12} to show its drop.
- **Caption draft:** *Sparse parity with d=14, k=3 (N=364 possible targets). A learner that names the target in log₂N ≈ 8.5 bits needs about that many examples. Kernel ridge, like every fixed-kernel method, stays above the bound 1−n/N (shaded) and needs n of order N. Mean squared error on ±1 labels, averaged over random targets.*

### F9. SGD on sparse parity: plateau then sudden drop, and the samples it consumes vs d. P1

- **Anchor:** the Solution to Exercise 4.B.i, part (c), after "*So the gap between (a) and what SGD does is a fact about *search* (the search problem Furman names), not about the prior.*" The §6 table row "*time believed to grow like $d^{\Omega(k)}$*" can then point back to it.
- **What it shows:**
  - **Panel (a), the plateau.** Test error against SGD step for (d,k) = (50,3), about 8 seeds, with linear x from 0 to 4000 steps. The curves sit at 0.5 for about 2000 steps, then fall to 0 within a few hundred.
  - **Panel (a′), optional but valuable.** On a shared x-axis, a progress measure that the black-box loss hides: the first-layer weight norm on the k relevant coordinates divided by the norm on the rest. It grows steadily through the plateau. This is the "hidden progress" of Barak et al. (2022), Fig. 3.
  - **Panel (b), sample cost vs d.** Log–log axes; y = examples needed to reach 1% test error, x = d ∈ [10, 200]; series for k = 2, 3, 4.
    - Points: SGD, measured as fresh samples consumed = B·t_c (median of 5 seeds, bars for the quartiles).
    - Solid lines: the counting learner, log₂C(d,k) + log₂100.
    - Dashed lines: the fixed-kernel scale C(d,k).
    - For k=3, the three regimes separate cleanly: about 27 examples (counting) vs about 4.5×10⁵ (SGD, d=200) vs 1.3×10⁶ (kernel bound).
- **Why it helps:** CRW asked for this one; the paragraph on search vs prior is anchored on it, and the notebook currently asserts SGD's cost in one line without showing it.
  - Panel (a) shows why "search" is the right word: the loss gives no signal for a long time, then the network suddenly finds S.
  - Panel (b) places SGD *between* the two priors the exercise analyses. SGD on a network is not a fixed kernel (it beats C(d,k) at large d), but it is nowhere near the counting learner. That is exactly the §6 table's message, in one picture.
- **Source:** (b), a small torch CPU sim.
  - **Recipe:** a 2-layer ReLU MLP, width 256, N(0, 1/fan_in) weights, zero biases. Hinge loss, plain SGD, lr 0.1, batch 32, fresh samples every step. Target χ_S with S = the first k coordinates (fine by symmetry). Evaluate test error on 2000 fresh points every 100 steps (every 10 for k=2); stop below 1%.
  - **Sim checked, single seed, steps to <1% error:**

    | k | d → steps |
    |---|---|
    | 2 | 30 → 200; 50 → 200 |
    | 3 | 10 → 500; 15 → 900; 20 → 800; 30 → 2100; 50 → 2600; 100 → 4700; 200 → 14,200 |
    | 4 | 15 → 700; 20 → 800; 30 → 1300; 50 → 2100; 70 → 7100 |

    Wall times ran 2–110 s per run, mostly spent on evaluation. A 5-seed grid over d ∈ {20, 30, 50, 70, 100, 140, 200} for k=3, and up to d=70 for k=4, should take about 10–15 min on this laptop. That is borderline for CRW's "≲10 min locally" rule: trim to 3 seeds, or evaluate every 250 steps.
  - **Honesty point (important for the caption and annotation):** the local log–log slope I measured for k=3 is about 1.2–1.6 (d 50→200), and about 2 for k=4 (d 30→70). This is consistent with Barak et al.'s "n^{O(k)}" with a small constant (their text: t_c ∝ n^{αk} for small α), but it is **not** n^k. Do not draw a "slope = k" guide line or caption the panel "n^k scaling". The paper's Fig. 1 (right) also shows median convergence times only for small (n,k), so an "n^k" reading of it overclaims too. Fit and report the measured slopes.
  - **Paper alternative (a):** Barak, Edelman, Goel, Kakade, Malach and Zhang, *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit* (NeurIPS 2022), arXiv 2207.08799. Checked in arXiv HTML:
    - **Fig. 1** (left: training curves on (n=50,k=3) under various architectures, batch sizes and learning rates; right: median convergence times for small (n,k));
    - **Fig. 3** (black-box loss plateau vs hidden progress measure).

    Licence: arXiv non-exclusive distribution only, so embedding needs permission or a fair-use judgement. Self-generation is cheap and avoids the slope overreading.
- **Caption draft:** *(a) A ReLU network trained by online SGD on 3-sparse parity in d=50 sits at chance for about 2000 steps, then learns the target abruptly. The weight on the three relevant coordinates (lower trace) has been growing all along. (b) Fresh examples consumed before reaching 1% error, compared with the two learners of Exercise 4.B.i. SGD beats every fixed kernel at large d but needs far more data than naming S would take, and its cost grows polynomially in d with an exponent that rises with k (fitted slopes shown). After Barak et al. (2022), Figs. 1 and 3.*

---

## §5. Hierarchical representations

### F10. Shallow vs binary-tree network for a compositional function. P3

- **Anchor:** §5.A "Compositional targets", after "*(an existence result: Furman's "paradox of approximation").*"
- **What it shows:**
  - Left: a one-hidden-layer network with all 8 inputs feeding one wide layer, labelled "~ε^{-d/r} units".
  - Right: a binary tree of 2-input nodes h₁₁…h₁₄ → h₂₁, h₂₂ → h₃ over x₁…x₈, labelled "~(d−1)ε^{-2/r} units".
  - The target written underneath: f = h₃(h₂₁(h₁₁(x₁,x₂), h₁₂(x₃,x₄)), h₂₂(…)).
- **Why it helps:** §5.A is a single sentence quoting two rates. The tree makes "depth-matched" and "2-variable constituents" readable at a glance, and it is the same tree shape as the RHM in F11, which links §5.A to §5.B.
- **Source:** (c) redraw. The canonical figure is Poggio, Mhaskar, Rosasco, Miranda and Liao, *Why and When Can Deep — but Not Shallow — Networks Avoid the Curse of Dimensionality* (arXiv 1611.00740), **Fig. 1(a,b)**, checked in arXiv HTML. That arXiv licence is non-exclusive distribution only, so redraw it and credit "after Poggio et al. 2017, Fig. 1".
- **Caption draft:** *A generic function of 8 variables (left) needs a shallow network of size exponential in d. A compositional one (right), built from 2-variable pieces, is matched by a deep network whose size grows only linearly in d. After Poggio et al. (2017), Fig. 1.*

### F11. The Random Hierarchy Model generative tree. P1

- **Anchor:** §5.B, immediately after the bullet "*Generation.* A class label (one of $n_c$) is rewritten $L$ times, which gives an input of length $d=s^L$." and before "*Deep CNNs trained by gradient descent need about…*".
- **What it shows:** Cagnetta et al., **Fig. 1**, checked in arXiv HTML.
  - Left: the generative hierarchy, with a class label at the top, m synonymic rewrites per symbol, and levels 𝒱_L down to 𝒱_1.
  - Right: a worked example with n_c=2, L=3, s=2, m=3, v=3, showing the rules and two sampled inputs.
- **Why it helps:** the exercise requires counting rules level by level, and the three-bullet definition is hard to hold in one's head without the tree. The figure's notation (v, m, s, L, n_c) matches the notebook's exactly.
- **Source:** (a) embed. The published version (*Phys. Rev. X* 14, 031001, 2024) is **CC BY 4.0** (checked on the APS page). Credit with authors, title, journal citation and DOI 10.1103/PhysRevX.14.031001. Take the figure from the PRX version, not the arXiv PDF, which carries a non-exclusive licence only.
- **Fallback:** (c) redraw a smaller tree (n_c=2, L=2, s=2, m=2, v=3) if CRW prefers a uniform house style.
- **Caption draft:** *The Random Hierarchy Model. A class label is rewritten L times, each symbol into one of m equivalent patches ("synonyms") of s lower-level symbols, to give an input of length d=s^L. Right: an example with n_c=2, L=3, s=2, m=3, v=3. From Cagnetta et al., Phys. Rev. X 14, 031001 (2024), Fig. 1, CC BY 4.0.*

### F12. RHM sample complexity: deep CNNs follow n_c m^L. P1

- **Anchor:** §5.B, right after "*Deep CNNs trained by gradient descent need about $P^*\simeq n_c\,m^L$ samples.*"
- **What it shows:** Cagnetta et al., **Fig. 3**, checked in arXiv HTML.
  - Top: test error against training-set size P for depth-(L+1) CNNs (s=2, m=n_c=v), coloured by v and marked by L, showing sigmoidal drops.
  - Bottom: empirical P* against the model parameters, with a dashed line for P* = n_c m^L.
  - Optional second image: **Fig. 2 (bottom)**, the shallow-network sample complexity growing in proportion to P_max = n_c m^{(d−1)/(s−1)}, i.e. exponentially in d.
  - Optional third: **Fig. 9**, the uncorrelated variant staying at chance, which is the control the exercise solution cites ("*The uncorrelated variant sharpens the point*").
- **Why it helps:** the section's central empirical claim is this scaling, and the exercise compares an estimate against it. Students should see the measurement, including its scatter, rather than a formula.
- **Source:** (a) embed from PRX (CC BY 4.0), as for F11.
- **Self-generation:** possible on CPU at small sizes (v=m=n_c=4–8, L=2–3, P ≤ 10⁴), but it means training depth-(L+1) CNNs across a grid of P. That would probably take more than 10 minutes and duplicates a clean published figure. Not recommended.
- **Caption draft:** *Deep CNNs trained on Random Hierarchy Model data. Top: test error falls sharply once the training set passes a threshold. Bottom: the threshold P\* follows n_c m^L (dashed) across vocabulary sizes and depths, polynomial in the input length d=s^L. Shallow networks instead need a number of samples proportional to the total number of inputs, exponential in d (their Fig. 2). From Cagnetta et al. (2024), Fig. 3, CC BY 4.0.*

### F13. Three sample costs against input length on one plot. P2

- **Anchor:** the Solution to Exercise 5.B.i, after the "**Example.**" bullet ("*At $L=6$: about 384 versus 32,768.*").
- **What it shows:** log-y against d = s^L on a log₂ x-axis (L = 1…8, so d = 2…256), with v=n_c=8, m=4, s=2. Three curves:
  - the idealized length-prior learner, L·v·m·s·log₂v/log₂n_c (∝ log d);
  - deep CNNs, n_c m^L = n_c d^{log_s m} = 8d², a straight line of slope 2;
  - shallow networks, P_max = n_c m^{(d−1)/(s−1)} (exponential in d). Clip it at 10³⁰ with an arrow.

  Mark the two numerical examples from the solution (L=3: 192 vs 512; L=6: 384 vs 32,768).
- **Why it helps:** the exercise's conclusion is a statement about three growth rates (logarithmic, polynomial, exponential in d), and one log–log picture shows it immediately. It also shows the gap the solution attributes to search and data statistics rather than to the prior. This is the same structure as F9(b) for sparse parity, which ties §4 and §5 together.
- **Source:** (b) closed form, using the formulas already in the solution. P_max is from Cagnetta et al. ("P_max = n_c m^{(d−1)/(s−1)}", arXiv HTML).
- **Caveat for the caption:** the length-prior curve is the solution's back-of-envelope count, and it ignores the information in unlabelled inputs. Say "rough estimate".
- **Caption draft:** *Samples needed on the Random Hierarchy Model (v=n_c=8, m=4, s=2) as input length d grows: an idealized learner with a program-length prior (rough estimate, ∝ log d), deep CNNs (n_c m^L ∝ d², measured), and shallow networks (∝ total number of inputs, exponential in d). Deep networks escape the curse of dimensionality but stay far from what the rules' description length would allow.*

---

## Gaps and issues noticed

1. **No dangling figure references.** The builder never says "see figure", "plot" or "image", and WB4 currently links no interactive figure. No existing figure is misplaced or misdescribed.
2. **Barak et al. is not a WB4 reading.** If F9 goes in, the notebook needs one sentence or a reading card pointing to Barak et al. 2022 for the plateau and hidden-progress result. Memory notes say CRW prefers papers he knows, so **CRW should decide** whether to cite it or caption F9 as self-generated with a "see also".
3. **The entropy confound is absent from the §3.D reading blurb** ("*exponentially biased toward functions of low (proxy) complexity*"). F7 as proposed exposes it visually. **CRW to decide** whether the blurb (or a one-line pointer to Mingard et al. 2019, arXiv 1909.11522, CC BY 4.0) should say so, or whether the figure caption alone carries it.
4. **"time believed to grow like $d^{\Omega(k)}$" (§6 table, Exercise 4.B.i(c)).** My quick sim (width-256 MLP, SGD) gives empirical exponents of about 1.2–1.6 for k=3 and about 2 for k=4 over modest d. That is consistent with d^{Ω(k)} only as an asymptotic statement with a small constant (α ≈ 0.5). Students who plot it will not see d^k. The text is not wrong, but any figure annotation must report fitted slopes and must not draw a slope-k line. The SQ lower bound is also about time for SQ/gradient-style algorithms in general, not a tight prediction for this setup.
5. **F9(b) has a subtle comparison to flag.** SGD with fresh samples measures *samples consumed*. The kernel bound in Exercise 4.B.i(b) is about a *fixed* training set of size n. The comparison is fair as "examples seen", and the caption should say "fresh examples consumed".
6. **Circuit-prior slides.** I checked the cited slide numbers against the PDF (110 overlay pages). Slides 10–12 are the Solomonoff function-induction warm-up and 15–17 the setting and theorem. Slide 34 says "a size-k XOR takes ∼d^k samples" for NNGP. These match the reading card. The slides contain no raster figures worth embedding.
7. **Licences, summarized.**
   - Valle-Pérez et al. (1805.08522): CC BY 4.0.
   - Mingard et al. (1909.11522): CC BY 4.0.
   - RHM: CC BY 4.0 via PRX only; the arXiv version has a non-exclusive licence.
   - Barak et al. (2207.08799) and Poggio et al. (1611.00740): arXiv non-exclusive, so redraw or self-generate.
   - MacKay 1992 (Caltech record): no commercial rights. A redraw with credit is fine for a teaching notebook.
8. **Infrastructure.** No static-image helper exists in `nbkit` (see the note at the top). All P1/P2 self-generated figures need about 10 s to about 15 min of CPU. Only F9's full grid approaches CRW's 10-minute local threshold.
