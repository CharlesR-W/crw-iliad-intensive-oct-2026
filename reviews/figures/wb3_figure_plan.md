# WB3 (Why LLMs?): figure plan

- **Notebook:** WB3, builder `notebooks/build_wb3.py` (authoritative), page `notebooks/03_why_llms.html`
- **Date:** 2026-10-09. Planning only; no builder, notebook or asset was edited.
- **Count:** 10 proposals: **P1 × 4** (all in §3 and §5), **P2 × 5**, **P3 × 1**. There is also one optional interactive figure, listed under F5.2 and not counted.
- **Anchors** are section heading plus a short verbatim quote. The builder was being edited while I read it (lines moved by 2 between reads), so line numbers are omitted.
- **Licensing at a glance (checked on the arXiv abstract pages):** Kaplan 2001.08361, Hoffmann 2203.15556, ViT 2010.11929 and I-JEPA 2301.08243 are all under the *arXiv non-exclusive distribution licence 1.0*. That licence grants no reuse rights, so embedding their figures as images depends on fair use or permission, and CRW has to decide. Besiroglu et al. 2404.10102 is CC BY 4.0. For this reason almost every proposal below is something **we generate or redraw from published numbers**, citing the source as "after Fig. N".
- **Plumbing note:** `nbkit.figure()` only links interactive HTML. There is no helper for inline static images yet; the page CSS already has `img { max-width:100%; }`. The implementer will need a convention, for example `notebooks/assets/static/wb3_*.svg` plus a small `static_figure(path, caption)` helper. The build's local-link check should then cover the image paths.

---

## §1. Why generative, why unsupervised?

No figure proposed. The section is reading pitches plus a Fermi exercise. The numbers in Exercise 1.1 are the content, and a bar chart of them would add nothing (see "Considered and not proposed").

## §2. Why text and not video?

### F2.1: Generative vs joint-embedding vs JEPA (schematic) · **P2**

1. **Anchor:** §2, at the end of the second paragraph: "…LeCun's JEPA programme is the serious attempt to get the same abstraction from video: predict in a learned representation space, so the model is free to discard what it cannot or need not predict." Place it after "…language models with vision added." and before the LeCun reading.
2. **What it shows:** Three panels side by side, all fed the same input pair (x = visible context, y = masked or future part).
   - (a) *Joint-embedding*: x → encoder, y → encoder, loss D(s_x, s_y). Annotate: "needs augmentations or negatives to avoid collapse".
   - (b) *Generative*: x → encoder → decoder(z) → ŷ, loss in **input space** against y. Shade the loss box in a "pixels" colour and annotate it "pays for every bit of y, including texture and noise".
   - (c) *JEPA*: x → encoder → predictor(z) → ŝ_y, loss in **representation space** against s_y = target-encoder(y). Mark the target encoder "EMA / stop-grad" and annotate "may discard unpredictable detail".
   - Colour the loss location the same way across panels, so the one structural difference (where the loss is computed) is what the eye sees.
3. **Why it helps:** The paragraph's argument rests on a single architectural distinction, "predict the representation, not the pixels", which is hard to hold from prose alone. It also ties Exercise 2.1(d) ("a pixel-predicting model spends nearly all of its loss on the rest") to a concrete picture.
4. **Source type:** (c) schematic to draw (SVG, by hand or in matplotlib). Layout after Assran et al. 2023, *I-JEPA*, [arXiv:2301.08243](https://arxiv.org/abs/2301.08243), **Fig. 2**. I verified the caption in the PDF: "Common architectures for self-supervised learning… (a) Joint-Embedding Architectures… (b) Generative Architectures… (c) Joint-Embedding Predictive Architectures learn to predict the embeddings of a signal y from a compatible signal x, using a predictor network…". Redraw rather than embed (the licence is arXiv non-exclusive). The panel labels are generic, so no figure is copied.
5. **Priority:** P2.
6. **Caption draft:** *Three self-supervised objectives, after Assran et al. (2023, Fig. 2). A generative model (b) is scored on reconstructing y itself, so every bit of y costs loss. A JEPA (c) is scored on predicting y's representation, which leaves it free to drop what it cannot predict.*

## §3. Why next-token prediction, and why the transformer?

### F3.1: One sequence, T targets, one pass (schematic) · **P1**

1. **Anchor:** §3, after the three bullets and before the complexity table: after "…GPUs are built for large matrix multiplications." and before "| | FLOPs per layer | Sequential steps per layer (training) |".
2. **What it shows:** Three rows over the same 6-token sentence, with the tokens x₁…x₆ along the bottom of each row.
   - **(a) Causal transformer (training).** Above the tokens, six prediction heads output q(x₂|x₁) … q(x₇|x₁…x₆), with all six targets highlighted. Beside them, a 6×6 attention-mask grid: lower triangle filled ("row t attends to columns ≤ t"), upper triangle blank. Bracket the whole row as "one parallel pass = a few large matmuls".
   - **(b) RNN (training).** The same six targets, but produced by a chain h₀→h₁→…→h₆ with numbered arrows "step 1 … step 6". Bracket as "T sequential steps per layer".
   - **(c) BERT MLM.** One or two tokens replaced by [MASK]. Targets appear only at masked positions (≈15%), and attention is bidirectional (full grid, no triangle).
   - Optional thin strip under (a): "generation: one token at a time, with the KV cache growing", which illustrates the last sentence of the section.
3. **Why it helps:** Two of the section's three "facts" ("every position is a training target" and "computes all T predictions in one parallel pass") are statements about structure. The table's "O(1) vs O(T) sequential steps" column is abstract until you see the triangle mask next to the RNN chain. This is the transformer/attention schematic the text depends on. A full Vaswani block diagram would not help, because the text never refers to its internals.
4. **Source type:** (c) schematic to draw. It is our own design (standard pedagogy), with no paper figure to cite. It could be built in matplotlib (`imshow` of `np.tril(np.ones((6,6)))` plus arrows) or as hand SVG.
5. **Priority:** P1.
6. **Caption draft:** *Every position is a target. (a) With a causal mask, a transformer sees the true prefix at all positions at once, so one forward pass yields all T next-token losses. (b) An RNN produces the same T predictions but in T sequential steps. (c) BERT's masked objective trains only on the masked ≈15% of positions.*

## §4. A hundred flowers: symmetry as inductive bias

### F4.1: ViT vs ResNet as pretraining data grows (generated, after ViT Fig. 3) · **P2**

1. **Anchor:** §4 intro: "…and the vision transformer showed that with enough data the missing bias stops mattering. WB5 takes up how much architecture matters at all." Place it after that paragraph and before the Bronstein reading. It supports the section's own claim, not the reading blurb.
2. **What it shows:** x-axis: pretraining dataset (ImageNet 1.3M → ImageNet-21k 14M → JFT-300M), categorical, log-spaced. y-axis: ImageNet top-1 after fine-tuning. Lines:
   - ViT-B/16: 77.91 → 83.97 → 84.15
   - ViT-L/16: 76.53 → 85.15 → 87.12
   - ViT-H/14: — → 85.13 → 88.04
   - ResNet152x4 (BiT-S/M/L): 81.30 → 85.39 → 87.54

   Annotate the crossover: "ViT trails on ImageNet-only, leads on JFT".
3. **Why it helps:** It is the one plot the "inductive bias vs scale" thesis of §4 is anchored around. It turns "large scale training trumps inductive bias" from a slogan into a visible crossover, and it previews WB5 §1.
4. **Source type:** (b) generate from tabulated numbers. I checked the ViT numbers in [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) v2, **Table 5**, "ImageNet" rows (fine-tuned at 384 px). The paper says this table corresponds to **Fig. 3**, whose caption I verified: "Transfer to ImageNet. While large ViT models perform worse than BiT ResNets (shaded area) when pre-trained on small datasets, they shine when pre-trained on larger datasets." I checked the BiT numbers in Kolesnikov et al. [arXiv:1912.11370](https://arxiv.org/abs/1912.11370): **Table 2** gives BiT-S 81.30 and BiT-M 85.39 (both R152x4), and **Table 1** gives BiT-L 87.54. Caveat for the caption: ViT Fig. 3 shades a *range* of BiT ResNet sizes, but we plot only the largest. Fine-tuning protocols differ slightly, so treat the comparison as approximate. Matplotlib, under 1 s. An alternative is to embed ViT Fig. 3 itself (arXiv non-exclusive licence; CRW's call).
5. **Priority:** P2.
6. **Caption draft:** *ImageNet accuracy after pretraining on ever-larger datasets (ViT numbers from Dosovitskiy et al. 2020, Table 5; ResNet from Kolesnikov et al. 2019). With ImageNet alone the transformer trails the CNN. With 300M images it passes it. Smith et al. (2023) show the CNN catches back up at still larger scale.*

### F4.2: What commuting with the shift does to a weight matrix (generated) · **P2**

1. **Anchor:** the Exercise 4.1 solution, after part (b): "**(b)** With $n=224^2\approx5\times10^4$ pixels: a general linear map has $n^2\approx2.5\times10^9$ weights…". The figure goes at the end of the solution so the exercise prompt is unchanged.
2. **What it shows:** Three heatmaps of n×n matrices with n = 12, in a diverging colormap with zero shown as white.
   - (i) a general dense W: random, titled "n² = 144 free numbers";
   - (ii) a circulant W with W_ij = w_{(i−j) mod n}: random w, constant diagonals that wrap around, titled "WS = SW: n = 12";
   - (iii) a banded circulant with w nonzero only at offsets −1, 0, 1, titled "plus locality: 3".
   - Optionally overlay arrows on (ii) showing that each row is the row above shifted right by one.
3. **Why it helps:** The solution's key sentence, "weight sharing is a consequence of the symmetry, not a trick… locality is a further, separate choice", becomes visible at a glance. The two separate restrictions (equivariance gives n² → n, locality gives n → k) are two separate panels.
4. **Source type:** (b) generate. Use `numpy` (`scipy.linalg.circulant` or an index trick) and three `imshow` panels with a fixed seed. Under 1 s.
5. **Priority:** P2.
6. **Caption draft:** *Left: a general linear map on 12 inputs. Middle: the only linear maps that commute with the cyclic shift are circulant, with constant wrapped diagonals and 12 free numbers. Right: also requiring locality leaves a 3-tap convolution.*

### F4.3: Three architectures as message passing on graphs (schematic) · **P3**

1. **Anchor:** the Exercise 4.2 solution, end of (b): "…the transformer is message passing on the *complete* graph with data-dependent edge weights: the weakest structural assumption on the table, with order put back in by hand."
2. **What it shows:** Three small 6-node diagrams.
   - **1-D conv:** a ring, with each node linked to its ±1 neighbours and the same edge weights everywhere (same colour).
   - **GNN:** a given sparse graph with shared weights.
   - **Causal transformer:** every node linked to all earlier nodes, edge thickness varying with the input (labelled "weights = softmax(q·k)").
3. **Why it helps:** It makes "complete graph with data-dependent edges" concrete and connects §4's geometric view to §3's attention mask. It is optional: the solution is already clear.
4. **Source type:** (c) schematic (`networkx`/matplotlib or hand SVG). It follows the GDL framing in Bronstein et al. 2021 ([arXiv:2104.13478](https://arxiv.org/abs/2104.13478)), but no specific figure is reproduced.
5. **Priority:** P3.
6. **Caption draft:** *Architectures as choices of graph. A convolution passes messages along a fixed local ring with shared weights. A transformer passes them along every edge, with weights computed from the data. A causal mask keeps only edges from earlier tokens.*

## §5. Why scale? Should *you* consume more tokens?

All constants used below were verified against the papers.
- **Hoffmann et al. eq. 10 (Appendix D.2):** E=1.69, A=406.4, B=410.7, α=0.34, β=0.28.
- **Besiroglu et al. eq. 3:** "L(N,D)=1.8172+482.01/N^0.3478+2085.43/D^0.3658" (CC BY 4.0).
- **Kaplan et al. eqs. 1.1–1.3:** α_N≈0.076, N_c≈8.8×10¹³; α_D≈0.095, D_c≈5.4×10¹³; α_C^min≈0.050, C_c^min≈3.1×10⁸ PF-days.
- **Kaplan Table 6:** N_opt = 1.3×10⁹ · C_min^0.73 with C in PF-days (1 PF-day = 8.64×10¹⁹ FLOP).
- **Chinchilla Table 3:** (400M, 1.92e19, 8.0B), (1B, 1.21e20, 20.2B), (10B, 1.23e22, 205.1B), (67B, 5.76e23, 1.5T), (175B, 3.85e24, 3.7T), (280B, 9.90e24, 5.9T), (520B, 3.43e25, 11.0T), (1T, 1.27e26, 21.2T), (10T, 1.30e28, 216.2T).
- **Model sizes from Hoffmann et al.:** Gopher 280B/300B, GPT-3 175B/300B, MT-NLG 530B/270B, Chinchilla 70B/1.4T.

A prototype script checked all recipes below; together they ran on CPU in about 6 s.

### F5.1: Loss is a power law in compute, data and parameters (Kaplan Fig. 1) · **P1**

1. **Anchor:** §5 intro, after the first sentence "Held-out loss falls roughly as a power law in model size $N$ and data $D$." Place it after the intro paragraph, before the Kaplan reading.
2. **What it shows:** The canonical three-panel log-log plot of test loss against compute (PF-days), dataset size (tokens) and non-embedding parameters, each a straight line. The real figure also shows the learning curves in the compute panel, with the frontier as their envelope.
3. **Why it helps:** It is *the* image the scaling topic is anchored on. "Roughly a power law" means "a straight line over 6–7 orders of magnitude on log-log axes", and seeing it is more convincing than the exponents. It also motivates the functional form that Exercise 5.1 assumes.
4. **Source type:**
   - **Preferred: (a) embed** Kaplan et al., [arXiv:2001.08361](https://arxiv.org/abs/2001.08361), **Fig. 1**. I verified the caption in the PDF: "Language modeling performance improves smoothly as we increase the model size, datasetset size, and amount of compute used for training… Empirical performance has a power-law relationship with each individual factor when not bottlenecked by the other two." The licence is arXiv non-exclusive, so this is CRW's fair-use decision, with attribution.
   - **Fallback: (b) redraw** the three fitted lines from eqs. 1.1–1.3 with exponents annotated, over roughly the paper's ranges (C 10⁻⁹–10¹ PF-days, D 10⁸–10¹⁰ tokens, N 10⁵–10⁹). This loses the learning curves, which are half the point of the compute panel.
   - **Cheapest: (a′)** add "see Fig. 1" to the Kaplan blurb.
   - The compute-panel annotation inside the figure image may show a different constant from eq. 1.3's 3.1×10⁸. If redrawing, use the equation values.
5. **Priority:** P1.
6. **Caption draft:** *Kaplan et al. (2020), Fig. 1: test loss falls as a power law in compute, dataset size and parameters, each measured when the other two are not the bottleneck. On log-log axes each is a straight line over many orders of magnitude.*

### F5.2: IsoFLOP valleys and the compute-optimal frontier (generated) · **P1**

1. **Anchor:** "### Exercises: compute-optimal allocation", after the display $L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$ and the sentence "…so $C\approx6ND$." Place it before "**Exercise 5.1.**"
2. **What it shows:** Two panels, using Hoffmann et al.'s eq. 10 only (so that 5.1(d)/(e) are not pre-empted by the two-fit comparison in F5.3).
   - **(a) IsoFLOP curves.** x: N (log, 10⁷·⁵–10¹²·⁵). y: L(N, C/(6N)), clipped to about [1.85, 4]. One curve per budget C ∈ {10¹⁹, 10²⁰, 10²¹, 10²², 10²³, 10²⁴}, in a sequential colour ramp. A dot marks each minimum and a dashed line joins the minima.
   - **(b) Frontier.** Log-log N_opt(C) and D_opt(C) against C, from the Exercise 5.1(a) closed form, with the slopes β/(α+β)≈0.45 and α/(α+β)≈0.55 labelled.
3. **Why it helps:** Exercise 5.1 asks students to minimize L at fixed C. The figure shows what that minimization *is*: each budget gives a valley in log N, and the valley floors lie on a straight line. It is also the plot type the Chinchilla paper is known for (its Fig. 3, approach 2). That makes the paper's "three approaches" legible.
4. **Source type:** (b) generate. In numpy: `L = E + A/N**a + B/(C/(6*N))**b` and `G = (a*A/(b*B))**(1/(a+b))`; `N_opt = G*(C/6)**(b/(a+b))`. Matplotlib, about 1 s. Cite "after Hoffmann et al. 2022, Figs. 3–4". I verified the Fig. 3 caption: "IsoFLOP curves… We find a clear valley in loss, meaning that for a given FLOP budget there is an optimal model to train". Their budgets run from 6×10¹⁸ to 3×10²¹. Alternative: embed the empirical Fig. 3 (arXiv non-exclusive licence) next to it as "the measured version".
   - **Optional interactive upgrade (key-intuition candidate, CRW's call):** a `chinchilla_explorer.html` in `assets/interactive/`. It would have a compute slider, a fit toggle (Hoffmann eq. 10 / Besiroglu re-fit) and an inference-tokens slider Q, showing the valley, its minimum and the inference-adjusted optimum. It would merge F5.2, F5.3 and F5.5 into one widget. Static figures are enough for the pedagogy. Build the interactive only if CRW counts compute-optimal allocation as a Distill-worthy key intuition.
5. **Priority:** P1.
6. **Caption draft:** *Loss under Hoffmann et al.'s parametric fit when compute C≈6ND is held fixed. (a) Each budget gives a valley in model size. Too small a model is capacity-limited, and too large a model sees too few tokens. (b) The valley floors lie on straight lines in log-log space, N_opt∝C^0.45 and D_opt∝C^0.55 (Exercise 5.1).*

### F5.3: How flat the valley is: two fits at Chinchilla's budget (generated) · **P2**

1. **Anchor:** the Exercise 5.1 solution, part (e), after "…Treat "20 tokens per parameter" as an order of magnitude."
2. **What it shows:** A single isoFLOP curve at C = 5.76×10²³.
   - Solid: Hoffmann eq. 10. Dashed: Besiroglu re-fit.
   - Mark each minimum (prototype values: 32B at 93 tok/param, L=1.931; 72B at 18 tok/param, L=1.974).
   - Mark Chinchilla (70B) and Gopher (280B) with vertical lines.
   - Shade the band of N within 0.01 nats of each minimum (prototype: 13B–82B for Hoffmann, 27B–194B for Besiroglu).
   - Optionally plot L − L_min on the y-axis so both curves share a floor.
3. **Why it helps:** Part (e)'s lesson, "a factor of more than four in tokens per parameter changes the predicted loss by about 0.006 nats", is a fact about the curvature of a valley. It is the reason the published constants and the 70B/1.4T choice can disagree while both "fit". One picture makes the Besiroglu controversy intelligible.
4. **Source type:** (b) generate (same code as F5.2). Constants as verified above. Besiroglu et al. is CC BY 4.0, so their numbers can be quoted freely.
5. **Priority:** P2.
6. **Caption draft:** *Loss against model size at Chinchilla's compute, under the published fit (solid) and Besiroglu et al.'s re-fit (dashed). The two optima differ by more than a factor of two in N, yet each curve is nearly flat across that range. That is why "20 tokens per parameter" is an order of magnitude, not a constant.*

### F5.4: Kaplan vs Chinchilla, and where real models sat (generated, after Chinchilla Fig. 1) · **P1**

1. **Anchor:** Exercise 5.2: "**Exercise 5.2.** Kaplan et al. found $N_{\text{opt}}\propto C^{0.73}$; Hoffmann et al. found $N_{\text{opt}}\propto C^{0.5}$." Place it after the Exercise 5.2 solution, or after the prompt if CRW prefers it as setup.
2. **What it shows:** A log-log plot of parameters against training FLOPs (10¹⁸–10²⁶).
   - Lines: Kaplan, N = 1.3×10⁹·(C/8.64×10¹⁹)^0.73. Chinchilla approach 1, the Table 3 points (slope ≈0.50). Approach 3, the published eq. 10 (slope 0.45). The Besiroglu re-fit.
   - Stars: GPT-3 (175B, 300B tokens), Gopher (280B, 300B), MT-NLG (530B, 270B), Chinchilla (70B, 1.4T), each placed at C = 6ND.
   - Annotate the factor-of-about-4 gap between the Kaplan line and the Chinchilla line at Gopher's budget.
3. **Why it helps:** It is the image the Kaplan→Chinchilla story is anchored on: the 2020–21 frontier models sit on the Kaplan line, far above the compute-optimal one. Exercise 5.2's two exponents become two slopes, and the size of the practical stakes becomes visible.
4. **Source type:** (b) generate. This reproduces the content of Hoffmann et al. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) **Fig. 1**. I verified the caption: "Overlaid predictions. We overlay the predictions from our three different approaches, along with projections from Kaplan et al. (2020)…". Caveats for a footnote: Kaplan's N excludes embeddings and his C is C_min, so overlaying it is approximate, as it is in Hoffmann's own Fig. 1. Gopher's 6ND = 5.0×10²³, while the paper uses 5.76×10²³ as "Gopher's budget". Use 6ND and do not mix the two.
5. **Priority:** P1.
6. **Caption draft:** *Compute-optimal model size against training compute, after Hoffmann et al. (2022, Fig. 1). Kaplan et al.'s fit (slope 0.73) put most extra compute into parameters, and the large models of 2020–21 were built near that line. Chinchilla's approaches (slope ≈0.5) say the same compute should have bought a model about four times smaller trained on about four times more tokens.*

### F5.5: Training plus inference: total compute to reach Chinchilla's loss (generated) · **P2**

1. **Anchor:** the Exercise 5.3 solution, after "…The limit is data: overtraining needs the extra tokens to exist (see nostalgebraist's post above)."
2. **What it shows:** Fix the target loss L* = L(70B, 1.4T) under the Besiroglu re-fit. Its Q=0 optimum reproduces Chinchilla, which is why it is used here.
   - For N ∈ [10¹⁰, 3×10¹¹], solve D(N) = (B/(L*−E−A·N^−α))^{1/β}.
   - Total compute = 6N·D(N) + 2N·Q, for Q ∈ {0, 10¹³, 10¹⁴, 10¹⁵ generated tokens}. 10¹⁵ is Exercise 5.3's 10¹² queries × 1000 tokens.
   - Log-log plot against N, normalized by Chinchilla's training compute, with each minimum marked. Prototype minima: Q=0 → 73B (18 tok/param); 10¹³ → 34B (≈110); 10¹⁴ → 21B (≈630); 10¹⁵ → 16B (≈3800 tok/param).
3. **Why it helps:** It answers the section's title question, "Should *you* consume more tokens?", with one picture. As inference demand grows the optimum slides to smaller, longer-trained models, and the far-left wall shows that below some N no amount of data reaches L*. It turns Exercise 5.3's qualitative conclusion into a quantitative one.
4. **Source type:** (b) generate, under 1 s. The idea is published as Sardana, Portes, Doubov, Frankle, "Beyond Chinchilla-Optimal" ([arXiv:2401.00448](https://arxiv.org/abs/2401.00448), ICML 2024). I verified the abstract claim: "LLM researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal". The caption must carry their caveat that fits made at ordinary ratios mis-predict at extreme tokens per parameter. Ratios in the thousands are an extrapolation. If F5.5 is used, CRW might add Sardana et al. as an optional reading.
5. **Priority:** P2.
6. **Caption draft:** *Total compute (training plus inference) to reach Chinchilla's loss, under Besiroglu et al.'s fit. With no inference the best model is Chinchilla-sized. At Exercise 5.3's 10¹⁵ generated tokens it is about 16B parameters trained on thousands of tokens per parameter, an extrapolation well beyond where the fit was measured.*

## §6. Simulators and personae

No figure proposed. The section is purely reading pitches, and none of the readings' arguments depends on a figure the notebook would need to reproduce.

## What carries forward

No figure.

---

## Gaps and issues noticed

- **No dangling figure references.** The builder never refers to a figure that does not exist. "The OpenAI blog post… has the figures" (§1, sentiment neuron) points outward correctly. "After Table 1 of Vaswani et al." is a table we reproduce, and it is fine as is.
- **No existing interactive figure belongs in WB3.** None of the four in `assets/interactive/` fits. `spectral_filters.html` is relevant to *where scaling exponents come from*, but the closing already sends that question to WB5 §7, which is the right place.
- **No static-figure plumbing yet.** There is no static-figure helper in `nbkit.py` and no `assets/static/` directory (see the header). Every proposal here needs one.
- **Possible wording ambiguity (§5 intro).** "Chinchilla's estimates put both exponents near ½" refers to β/(α+β) and α/(α+β), but the nearest symbols in the sentence are α and β, and the published fit has α=0.34 and β=0.28. Something like "put both allocation exponents near ½" would avoid confusion. F5.2(b) would make this visible anyway.
- **Gopher's compute.** Exercise 5.1(d) uses "Gopher's and Chinchilla's budget, C=5.76×10²³" (the paper's convention), but 6ND for Gopher is 5.0×10²³. This is not an error in the text. It matters only if a figure plots Gopher at 6ND (F5.4): say which one is used.
- **The HTML may be stale.** `03_why_llms.html` (09:04) is older than `build_wb3.py` (09:08), and the builder was being edited while I worked. Rebuild before checking figure placement.
- **Reading-length discipline.** No proposal adds prose to a reading pitch. F2.1, F4.1 and F5.1 sit after section-intro paragraphs, and the rest sit in exercise setups or solutions.

## Considered and not proposed

- **LeCun's cake slide (§1).** It is iconic, but Exercise 1.1(d) already quotes its text, and the image is LeCun's slide (rights unclear). A link in the LeCun blurb would cost nothing if CRW wants it.
- **Log-scale bar chart of the Fermi numbers (Exercises 1.1 and 2.1).** The numbers are the content. A chart would be decoration.
- **Sentiment-neuron histogram, Sora frames, GDL "5G" figure, assistant-axis plots.** These would be figures for reading blurbs and are better seen in the readings themselves.
- **Roofline or arithmetic-intensity plot (§3, "GPUs are built for large matrix multiplications").** It would be useful, but it needs new prose to explain arithmetic intensity, and Horace He's reading already has it.
- **Vaswani Fig. 1 (full transformer block).** The text never relies on the block's internals. F3.1 is the schematic it does rely on.
- **Arithmetic-coding interval picture.** This belongs to WB4's compression material rather than §3's one-sentence mention.
