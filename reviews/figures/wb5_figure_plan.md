# WB5 (Universality): figure plan

- **Notebook:** WB5, *Universality: the machines, they just want to learn*. Builder `notebooks/build_wb5.py` (read at its 09:08 state); page `notebooks/05_universality.html`, which still holds the pre-restructure version.
- **Date:** 2026-10-09. Planning only; no builder, notebook, nbkit or asset file was edited.
- **Scope note:** the builder was restructured this morning. WB5 now has §1 scaling laws, §2 architecture (body still the placeholder `ARCH_SECTION_BODY`), and §3 CKA/CCA, PRH and LRH. Superposition, low-rank logits and subliminal learning moved to WB6 §5 and §13, so they are not proposed here (see Gaps).
- **Count:** 11 proposals. **P1: 4** (F1, F4, F6, F9), **P2: 5** (F2, F5, F7, F8, F10), **P3: 2** (F3, F11).

Source types: (a) embed a paper figure, (b) generate it ourselves, (c) draw a schematic, (d) interactive-figure placement or companion. Every paper figure number below was checked against the arXiv HTML or PDF on 2026-10-09; licences come from the arXiv abs/HTML pages. "Sim checked" means I ran the recipe in a scratchpad.

**Infrastructure note.** `nbkit` has only `figure()`, which links to an interactive page. There is no helper for inline static images. Someone needs to add one, e.g. `image(path, caption, credit)` producing a `<figure>` with a caption and attribution line, plus a `notebooks/assets/figures/` directory and generator scripts under `checks/`. Inline SVG would keep the notebooks self-contained. This is a CRW/maintainer decision.

---

## §1. Scaling laws: a hundred unnecessary explanations

### F1. Same curve, different anatomy (spectral vs quanta). P1

- **Anchor:** the answer to Exercise 1.1, right after "*Distinguishing them needs finer measurements, such as per-example loss curves: one expects them to be step-like under quanta and smooth under the spectral picture.*" (before Exercise 1.2, "Where the heuristic fails: saturation").
- **What it shows:** a 2×2 grid.
  - Top row: aggregate loss vs resource on log–log axes. Left is the spectral model (noiseless ridge, α=2, α′=2, κ(n)≈λ_n); right is the quanta model (p_k ∝ k^{-(1+a)} with a=1, N ∝ n). Each has a dashed fitted line with its slope. Choose parameters so the two slopes match (−1 each) and the curves are visually indistinguishable.
  - Bottom row: the same losses broken down. Left: per-*example* squared errors for about 8 random test points, e(x;n)=(Σ_k x_k(1−f_k(n))β_k)², with filter f_k=λ_k/(λ_k+κ(n)). These are smooth, noisy power-law-ish curves. Right: per-skill loss for skills k=1..8, as clean steps at N(n)=k, plus a thin overlay of the mean.
- **Why it helps:** Exercise 1.1(c)'s punchline is "an observed power law cannot by itself tell them apart". This figure makes the claim visible: the top row is identical and the bottom row is not. It also shows the measurement Michaud et al. actually make (their Fig. 4).
- **Source:** (b) closed form, under a second of numpy. Truncate the spectrum at k ≤ 10⁵. Use the deterministic-equivalent κ by fixed-point iteration, κ(1−df₁/n)=0⁺, i.e. solve n = Σλ_k/(λ_k+κ), rather than κ≈λ_n, so it matches the interactive figure's convention. Draw the test x from N(0, diag λ).
- **Honesty point for the caption:** a single ridge *mode* also turns on as a sigmoid in log n, about one decade wide, so the contrast is per-example (each example mixes many modes) vs per-skill (each example needs one quantum). The caption should say so instead of implying that ridge never has steps.
- **Paper alternative:** Michaud et al. (2303.13506, CC BY 4.0) Fig. 2 bottom shows per-subtask steps averaging to a smooth power law on multitask sparse parity, and Fig. 4 shows abrupt vs smooth per-token Pythia curves. Fig. 4 is a good optional second image ("what this looks like in a real LM"), credited "Michaud et al. 2023, CC BY 4.0".
- **Caption draft:** *Two mechanisms, one learning curve. Top: a ridge learner on a power-law spectrum (left) and a learner that acquires discrete skills in frequency order (right) give the same power law. Bottom: they differ in what makes up the average. Per-example errors fall smoothly under the spectral model because each example mixes many modes; per-skill losses drop as steps under the quanta model.*

### F2. The spectral picture of Exercise 1.1(b) and 1.2. P2

- **Anchor:** §1.A Setup, after the equation "*$E(K)\approx\sum_{k>K}c_k^2+\frac{\sigma^2K}n$*" in Exercise 1.1(b). An alternative is the end of §1.A Setup, after "*The language of spectral filters is WB2 §1.D's.*"
- **What it shows:** one log–log panel of signal power c_k² = k^{-α′} against mode index k, as stems or a staircase.
  - A vertical line marks the cutoff K. Modes past K (unlearned tail) are shaded and labelled "bias = Σ_{k>K} c_k²". Modes up to K carry a flat band at height σ²/n, labelled "variance ≈ σ²K/n".
  - An overlaid secondary curve shows the ridge filter 1−λ_k/(λ_k+κ) with κ=λ_K. It is the smooth version of the hard cutoff and previews Exercise 1.2: the filter does not drop to zero at the head, which is where saturation comes from.
  - An optional inset plots E(K) against K: the U-shape with K* marked.
- **Why it helps:** the whole derivation is "count what is above and below a cutoff". A static picture lets a student read the bias-variance balance at a glance before doing the algebra. The interactive figure's panel A shows filters on eigenvalues but not this bias/variance bookkeeping on the *target's* spectrum.
- **Source:** (b) closed form, matplotlib, with α=2, α′=2.5, n=10³, σ²=10⁻². (Static companion to the interactive figure, not a replacement.)
- **Caption draft:** *Where a learning-curve exponent comes from. The target's power per mode falls as k^{-α′}. A learner resolving the top K modes pays the shaded tail as bias and about σ²/n per resolved mode as variance; balancing the two sets K* and the exponent. Ridge's soft filter (curve) also shrinks the resolved head, which is the source of Exercise 1.2's saturation.*

### F3. Exponent map in the (α, α′) plane. P3

- **Anchor:** the answer to Exercise 1.2(b), after "*So the noiseless rate is $n^{-\min(\alpha'-1,2\alpha)}$ away from the boundary, and $n^{-2\alpha}\log n$ on it.*"
- **What it shows:** a heatmap or contour plot over α ∈ (1,4) and α′ ∈ (1,10) of the noiseless exponent min(α′−1, 2α), with the line α′=2α+1 drawn and labelled. Above the line the region is labelled "ridge saturates: exponent 2α"; below, "heuristic correct: exponent α′−1". The diagonal α′=α (generic target) is marked. An optional second panel shows the noisy exponent s/(s+1).
- **Why it helps:** it summarises a three-case answer in one glance and shows that generic targets (α′=α) never saturate. That is not obvious from the case analysis.
- **Source:** (b) closed form, trivial.
- **Caption draft:** *The noiseless ridge exponent as a function of the spectrum (α) and the target's smoothness (α′). Smoother targets help only until α′−1 = 2α (dashed line); beyond it ridge's own shrinkage dominates and the rate saturates at 2α. Generic targets (α′ = α, dotted) lie well inside the unsaturated region.*

### F4-existing. spectral_filters.html panel B: placement is right, label is stale. (d)

The `figure(SF, …)` link after Exercise 1.2's answer is well placed, and its caption is accurate. However, the interactive page's own panel B header still says "**WB5 §7**", and `assets/interactive/README.md` still maps it to "WB5 §7". Both should become WB5 §1. This is not a new figure, so it is not counted.

---

## §2. Another bitter lesson: architecture doesn't matter (much)

*§2's body is still the placeholder `ARCH_SECTION_BODY`. These anchors follow the route blurb: "Architectures move the scaling curve but rarely change its slope. One Fermi exercise on what a constant factor is worth; readings, with explainers for Mamba and JEPA." Re-anchor them once the text lands.*

### F4. Shift vs slope: what a constant factor is worth. P1

- **Anchor:** immediately before or inside the Fermi exercise on "what a constant factor is worth" (route text: "*Architectures move the scaling curve but rarely change its slope*").
- **What it shows:** log–log axes of loss against compute C, with three curves.
  - Baseline: L = A·C^{-a}, with a≈0.05 (LM-like).
  - Better architecture, same slope: L = (A/k)·C^{-a} with k=1.1, i.e. 10% lower reducible loss. A horizontal double arrow labels the equivalent compute multiplier k^{1/a} (≈ 6.7× for a=0.05). This annotation is the Fermi exercise's answer, so it could go in the solution rather than the prompt.
  - Hypothetical different slope: L = A′·C^{-a′} with a′=0.06, crossing the baseline, labelled "what changing the exponent would mean".
  - Optional: a shaded irreducible-loss floor, as in Hestness et al.'s Fig. 6 sketch.
- **Why it helps:** the section's claim is a geometric statement about log–log plots (offset vs slope). Its counter-intuitive consequence is that a small constant-factor gain is worth a *large* compute multiple when exponents are small. That is easiest to see as a horizontal gap between parallel lines.
- **Source:** (b) closed form. Take k and a from whatever the Fermi exercise uses, so the figure and solution agree.
- **Caption draft:** *Architecture changes usually shift a scaling curve rather than tilt it. With a small exponent a, a constant-factor improvement k in loss is equivalent to multiplying compute by k^{1/a}: a 10% gain at a = 0.05 is worth about 7× compute. A different exponent would instead make the curves cross eventually.*

### F5. Empirical: architectures on parallel lines. P2

- **Anchor:** with the Mamba explainer or reading in §2 (to be written).
- **What it shows:** Gu & Dao, *Mamba* (arXiv 2312.00752, **CC BY 4.0**), **Fig. 6** "(Scaling Laws.)": Pile perplexity against FLOPs for Transformer, Transformer++, H3, Hyena, RWKV, RetNet and Mamba, at 125M–1.3B parameters and two context lengths. Most families run roughly parallel, offset from one another.
- **Why it helps:** it is the real-data version of F4. It also belongs next to the Mamba explainer the route promises. It can be embedded with attribution.
- **Source:** (a) embed Fig. 6 with the credit "Gu & Dao 2023, CC BY 4.0". The scale range is narrow (one decade of FLOPs), so the caption should not overclaim that the slopes are equal.
- **Link-only alternatives, not embedded because they carry the arXiv non-exclusive licence:**
  - Hestness et al. 2017 (arXiv 1712.00409), the original "*model improvements only shift the error but do not appear to affect the power-law exponent*". Fig. 2 shows LSTM and RHN word LMs with the same exponent, βg ≈ −0.066; Fig. 5 shows DS2 vs attention speech models.
  - Kaplan et al. 2020 Fig. 7, LSTMs vs Transformers.
  - Smith et al. 2023, *ConvNets match ViTs at scale* (2310.16764), Fig. 2: NFNet held-out loss against compute on JFT-4B.

  Any of these could be redrawn from published numbers if CRW prefers.
- **Caption draft:** *Language-model scaling for several sequence architectures (Gu & Dao 2023, Fig. 6; CC BY 4.0). Over this decade of compute the families mostly differ by an offset rather than a slope; Mamba matches a strong Transformer recipe. One decade is too short to rule out small slope differences.*

*(No figure proposed for the JEPA explainer. WB3 §2 already covers JEPA, and a block diagram would be decorative here unless the explainer turns out to be architectural.)*

---

## §3. Same representation? CKA, CCA, and the Platonic and linear representation hypotheses

### F6. Layer-by-layer similarity heatmaps: CKA finds corresponding layers, CCA does not. P1

- **Anchor:** §3.A "Same up to what?", right after the Kornblith et al. reading ("*Proposes CKA, which identifies corresponding layers across networks trained from different initializations.*"), before Exercise 3.1.
- **What it shows:** two side-by-side layer × layer heatmaps comparing network A (seed 0) with network B (seed 1): left linear CKA, right mean CCA. Both use the same colour scale, 0–1. CKA should show a bright diagonal; CCA should be washed out or saturated for wide layers. An optional third panel shows A against itself.
- **Why it helps:** this heatmap is *the* canonical image of representation similarity, and it is what students will meet in every follow-up paper. It also motivates Exercise 3.1 visually: one index sees the correspondence, the other is fooled by too much invariance.
- **Source:** (b) generate it ourselves. Kornblith et al.'s Fig. 2 (2019) is the canonical version but carries the arXiv non-exclusive licence (link-only).
  - Recipe: two 6-hidden-layer ReLU MLPs (width 256) on MNIST or Fashion-MNIST, two seeds, about 3 epochs with Adam on CPU (under a minute each with torch). Evaluate on n=500 test images. Linear CKA from centered activations; mean CCA via QR + SVD (see `cca.py` in the F7 recipe).
  - With width 256 < n−1 = 499, CCA will not be trivially 1, but it will be much less diagonal. To show the saturation itself, add a variant with n=200 test points.
  - If CRW wants the real thing, link Kornblith Fig. 2 beside it.
- **Caption draft:** *Similarity between every pair of layers of two identical MLPs trained from different seeds. Linear CKA (left) picks out corresponding layers along the diagonal; mean canonical correlation (right) does not, because it is invariant to every invertible linear map (Exercise 3.1b). After Kornblith et al. 2019, Fig. 2.*

### F7. Too much invariance: CCA saturates with width; CKA has its own bias. P2

- **Anchor:** the answer to Exercise 3.1, after "*This is Kornblith et al.'s argument that invariance to every invertible linear map is too much invariance for wide layers.*"
- **What it shows:** similarity between two *independent* random Gaussian representations X, Y ∈ ℝ^{n×p}, with n=200, plotted against p/n. It has two curves:
  - mean canonical correlation, which rises to exactly 1 at p=n−1;
  - linear CKA.

  Sim checked:

  | p | 5 | 20 | 50 | 100 | 150 | 190 | 199 | 250 |
  |---|---|---|---|---|---|---|---|---|
  | mean CCA | 0.11 | 0.28 | 0.43 | 0.63 | 0.82 | 0.96 | 1.00 | 1.00 |
  | linear CKA | 0.02 | 0.10 | 0.19 | 0.33 | 0.43 | 0.50 | 0.50 | 0.55 |

  A recommended third curve is a debiased linear CKA (the unbiased HSIC estimator of Song et al. 2012, which is standard in CKA libraries). It stays near 0. Average over 20 draws with ±1 s.d. bands.
- **Why it helps:** it turns Exercise 3.1(b)'s "all canonical correlations are 1, whatever the two networks compute" into a picture. Equally important, it honestly shows that linear CKA *also* reports spurious similarity when p is comparable to n (about 0.5 here between unrelated representations). Students should know this before reading CKA numbers in PRH-style papers.
- **Source:** (b) numpy, under a second (scratchpad `cca.py`: QR of centered X and Y, then the singular values of Q_xᵀQ_y).
- **Caption draft:** *Two unrelated random representations of 200 inputs. Mean canonical correlation climbs to exactly 1 once the width reaches n−1 (Exercise 3.1b). Linear CKA rises too, to about 0.5, which is a finite-sample bias; a debiased estimator stays near zero. Compare any reported similarity against this baseline.*

### F8. One function, an orbit of representations. P2

- **Anchor:** the answer to Exercise 3.1(c), after "*Being a minimum fixes the function, not the representation. Any convergence of representations has to come from whatever selects a point on the orbit*". An alternative is the start of "**What Ziyin and Chuang show, in outline.**" in §3.B.
- **What it shows:** a schematic in two linked panels.
  - Left, parameter space: a curved valley (the GL orbit of global minima, drawn as a 1-D curve in a 2-D loss contour plot). A colour gradient along the valley shows the gradient-noise term S. Three marked points are the GD/init-dependent end point, the S-minimiser (star, "label-noise SGD selects") and the weight-decay point (square, "norm balance").
  - Right: the hidden representation of a 2-D grid of inputs at each of the three points. These are the same circle of inputs mapped to ellipses of different shape (T = diag(2,1), etc.), and the shape is what CKA sees.
- **Why it helps:** the PRH proof's logic is "minimum → orbit → selection rule → representation". It is abstract in prose and immediate as a picture. The figure also illustrates the bullet list's distinction that different selection rules (none, gradient noise, weight decay) pick different points.
- **Source:** (c) schematic. The right panel can be generated exactly: apply T to a unit circle of inputs. The left panel is a hand-drawn or SVG stylisation; the caption should say so.
- **Caption draft:** *Every global minimum of a deep linear network sits on a family of minima with the same function but different hidden representations (left, schematic; right, the hidden image of a circle of inputs at three points). Being a minimum does not choose a point; small-step SGD with label noise, weight decay and the initialization each choose differently.*

### F9. Does label-noise SGD actually align them? A small simulation. P1, contingent

- **Anchor:** §3.B, after the bullet list, following "*None of Huh et al.'s multitask or capacity pressures appears: there is one task, and the selection is an implicit bias acting on a parameter symmetry.*"
- **What it shows:** CKA between the hidden layers of two depth-2 linear networks over training steps. The networks have different fixed input maps Z_A ≠ Z_B and different seeds. Four curves:
  1. SGD, batch 1, with label noise σ=1;
  2. SGD without label noise;
  3. large-batch SGD with label noise (less gradient noise);
  4. weight decay without noise.

  The Ziyin–Chuang prediction is that only (1) heads towards CKA = 1, (2) stays at an init-dependent value, (3) moves much more slowly, and (4) settles at a different, modality-dependent value.
- **Why it helps:** the paper has **no figures or experiments** (checked: arXiv 2507.01098 HTML is pure theory). A small simulation is the only way to show students the claim and its failure modes. It also concretely checks the bullet "*It characterizes minimizers of S among global minima; it does not show SGD reaches them*". If the drift is slow, the figure shows that too, which is pedagogically honest.
- **Source:** (b) numpy. d=3, D=2, Z_A = diag(3,1,0.3), Z_B random with 2I added, V random, η=5·10⁻³, 2·10⁵ steps.
  - **Sim checked** (scratchpad `prh.py`, one seed pair, test CKA on 2000 fresh inputs; all runs start at CKA ≈ 0.60):

    | Run | Hidden-layer CKA |
    |---|---|
    | (1) batch-1 SGD, σ=1 | **≈ 0.99 by 2·10⁴ steps**, then fluctuating 0.98–0.998 |
    | (2) noiseless SGD | frozen at **0.70** |
    | (3) batch-1000, σ=1 | **0.70 → 0.71** over 2·10⁵ steps, too slow to see |
    | (4) weight decay 10⁻³, no noise | **0.73 → 0.87** and still moving (its predicted limit, below 1 for anisotropic Z_A, is not reached) |

    The prediction shows clearly.
  - **Cost:** the whole run took about 7 minutes in a pure-numpy loop, dominated by the batch-1000 run. For the figure:
    - use 5·10⁴ steps on a log-x axis;
    - average 5 seed pairs with bands;
    - vectorise or run the batch-1 runs in torch.

    That should take under a minute.
  - **Caveats for the caption:** the toy has a single task and a 3-dimensional linear network, and run (4) has not converged.
- **Caption draft:** *Two linear networks seeing differently transformed inputs, trained on the same noisy-label regression. Hidden-layer CKA over training for label-noise SGD (aligning), noiseless SGD (frozen at an initialization-dependent value), large-batch SGD and weight decay. Toy illustration of Ziyin & Chuang's selection argument, not an experiment from the paper.*

### F10. The PRH's headline plot, with its caveats in the caption. P2 (CRW decision)

- **Anchor:** §3.B "The Platonic representation hypothesis", after the Huh et al. reading ("*…driven by three pressures: multitask scaling, capacity and a simplicity bias.*") and before the Ziyin & Chuang reading.
- **What it shows:** Huh, Cheung, Wang & Isola (arXiv 2405.07987, **CC BY 4.0**), with two candidates.
  - **Fig. 3** "LANGUAGE and VISION models align": language–vision mutual-kNN alignment on WIT (y) against LM performance as 1 − bits-per-byte (x). Alignment values span roughly 0.10–0.20.
  - **Fig. 2 left** "VISION models converge as COMPETENCE increases": 78 vision models, alignment against VTAB competence.

  Fig. 3 is the more striking cross-modal claim; Fig. 2 is the "alignment vs competence" figure most people remember.
- **Why it helps:** the PRH is anchored on this plot. Students should see what it actually shows, including how small the absolute numbers are.
- **Confounds to state in the caption rather than hide:**
  - Absolute alignment is low, roughly 0.1–0.2 on a 0–1 metric. (Chance for mutual k-NN with k=10 on about 1k points is about k/n ≈ 0.01, so the effect is real but far from "convergence".)
  - Capability, parameter count and training-data overlap move together across the plotted models.
  - Mutual-kNN is a local, rank-based index; F7 shows how global indices can mislead, so the choice of index matters.
- **Source:** (a) embed with the credit "Huh et al. 2024, CC BY 4.0". Licence permits it.
- **Self-generated alternative (recommended if CRW prefers not to reproduce a possibly overclaiming plot):** mutual-kNN (k=10) between about 1000 COCO/WIT image–caption pairs. Use 4–5 HF vision encoders of increasing size (e.g. DINOv2 S/B/L, ViT-T/S/B) against 4–5 Pythia sizes (70M–1.4B), and include randomly initialized models and a shuffled-pairing baseline as reference lines.
  - CPU cost is about 10–20 minutes, with model downloads of a few GB. Per CRW's CLAUDE.md this should go to Modal if it runs long.
  - It is not a classroom exercise; produce it offline once.
  - Its value is the baselines: it shows how much of the trend survives against random-init and shuffled controls.
- **Caption draft (embed version):** *Huh et al.'s evidence for the PRH: language models that predict text better have nearest-neighbour structure more like a vision model's (Fig. 3, CC BY 4.0). Note the scale: alignment is about 0.1–0.2 on a 0–1 metric. Model capability, size and training data also vary together across the points.*

### F11. LRH geometry: what a model's behaviour fixes and what it does not (the parallelogram). P3, but P1 if CRW wants one LRH image

- **Anchor:** §3.C, the answer to Exercise 3.2, after "*Euclidean orthogonality, and cosine similarity in general, of concept directions is not determined by the model's input–output behaviour.*" Alternatively, a smaller version can go just before Exercise 3.2, after "*Park et al. estimate their causal inner product as …*".
- **What it shows:** three panels of four 2-D unembedding vectors γ(man), γ(woman), γ(king), γ(queen), drawn as a parallelogram with the "gender" (man→woman) and "royalty" (man→king) difference vectors as arrows.
  1. The original coordinates: the parallelogram with orthogonal concept directions.
  2. After the behaviour-preserving change γ → A^{-⊤}γ, using Exercise 3.2(b)'s shear [[1,1],[0,1]]. It is still a parallelogram, so differences stay parallel and the "king − man + woman ≈ queen" arithmetic survives, but the angle between the concepts is no longer 90°.
  3. Both after whitening by Cov(γ), computed over a small cloud of about 200 background "vocabulary" points drawn in grey. Whatever A was, the result is the same shape up to rotation. That is Exercise 3.2(c).

  Annotate each panel with the Euclidean cosine between the two concept arrows.
- **Why it helps:** the parallelogram is the iconic LRH image. This version carries the section's actual lesson: *linear* structure (parallel differences) is invariant under GL(d), while *metric* structure (angles, cosines) needs a chosen inner product, and the causal inner product fixes one. It is the picture of Exercise 3.2.
- **Source:** (b) numpy, synthetic 2-D, instant.
  - A real-data option for panel 1: GloVe-50d via `gensim.downloader` (about 66 MB), PCA of the four words plus a few country–capital pairs. This is the analogue of Mikolov et al. 2013 (arXiv 1310.4546) Fig. 2, "Two-dimensional PCA projection of the 1000-dimensional Skip-gram vectors of countries and their capital cities", which carries the arXiv non-exclusive licence and is link-only.
  - Note the caveat that PCA projections of analogies are themselves known to be somewhat cherry-picked.
- **Paper alternative:** Park, Choe & Veitch (arXiv 2311.03658) Fig. 3, a heatmap of causal inner products between LLaMA-2 concept directions showing near-orthogonality of causally separable concepts. Its licence is **CC BY-NC-SA 4.0**: embedding is fine for a non-commercial course, but NC/SA terms apply. **CRW should confirm whether the Iliad Intensive counts as non-commercial**; link-only otherwise.
- **Caption draft:** *Concept arithmetic and its geometry. Left: unembeddings whose gender and royalty directions are orthogonal. Middle: a change of coordinates that leaves every logit unchanged keeps the parallelogram, so analogies survive, but changes the angle. Right: whitening by the vocabulary covariance (Park et al.'s causal inner product) gives the same picture whatever the coordinates were.*

### Also worth embedding beside the Engels et al. reading (folded into F11's priority, not counted separately)

- **Anchor:** §3.C, after the Engels et al. reading ("*…most clearly circles for days of the week and months of the year.*").
- **What:** Engels, Michaud, Liao, Gurnee & Tegmark (arXiv 2405.14860, **CC BY 4.0**) **Fig. 1**: circular representations of days, months and years in GPT-2-small layer 7, coloured by token.
- **Why:** it is the counterpoint's single image, and seeing the circle is the whole argument for "not one-dimensionally linear". It is embeddable with credit.
- **Caption:** *Days of the week and months of the year lie on circles in GPT-2-small's layer-7 activations, a feature that cannot be split into independent one-dimensional directions (Engels et al. 2024, Fig. 1; CC BY 4.0).*

If CRW wants only one image in §3.C, I would pick this Engels figure over F11 for visual impact, and F11 for pedagogy.

---

## Gaps and issues noticed

1. **The task brief's scope is stale.** It lists superposition, subliminal learning and spectral_filters/superposition interactives for WB5. Superposition and subliminal learning moved to WB6 this morning (builder docstring; WB6 §5 now links `superposition.html`). Figure planning for those belongs to WB6's plan. Two candidates are worth passing on to the WB6 planner:
   - a static companion to `superposition.html` showing Elhage et al.'s n=5, m=2 pentagon progression, which the interactive figure already trains;
   - for subliminal learning, a two-panel "shared init vs different init" student-trait plot in the style of Cloud et al.'s MNIST auxiliary-logit experiment.
2. **`notebooks/assets/interactive/README.md` is out of date.** It maps `spectral_filters.html` to "WB5 §7" (now §1) and `superposition.html` to "WB5 §5" (now WB6 §5). The spectral_filters page's panel B heading also still reads "WB5 §7". Fixing that means editing a generated or interactive asset, which is outside my remit.
3. **`05_universality.html` is stale.** It is byte-identical in size to `_old_wb5/05_universality_2026-10-09_pre-restructure.html` (117,235 bytes). The restructured builder has not been rebuilt, and cannot be until `ARCH_SECTION_BODY` is filled.
4. **§2 is unwritten,** so F4 and F5 anchors are provisional. If the Fermi exercise's numbers are set, F4 should use the same k and a.
5. **No text in the current builder refers to a nonexistent figure.** The only figure reference is the `figure(SF, …)` link, and its caption ("Use panel B…", "push α′ past 2α+1") matches the interactive page.
6. **CKA caution.** F7's simulation shows linear CKA ≈ 0.5 between *independent* random representations when p ≈ n. The notebook text says CKA "equals 1 exactly when K ∝ L", which is correct, but nothing warns that moderate CKA values can be pure finite-sample bias. One sentence in §3.A, or F7's caption, would cover it.
7. **Licensing summary.**
   - CC BY 4.0, embeddable with credit: Huh et al., Engels et al., Michaud et al., Gu & Dao.
   - CC BY-NC-SA 4.0: Park et al.
   - arXiv non-exclusive licence, link or redraw only: Kornblith et al., Kaplan et al., Hestness et al., Smith et al., Mikolov et al., Ziyin & Chuang.
