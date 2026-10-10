# WB6 (Mysteries myriad): figure plan

- **Notebook:** `notebooks/build_wb6.py` → `notebooks/06_mysteries_myriad.html`. I read the builder as of 2026-10-09 09:09: 13 sections. §5 now includes superposition and §13 covers subliminal learning; both moved from WB5 today.
- **Date:** 2026-10-09
- **Count:** 21 proposals. P1: 7 · P2: 8 · P3: 6. I did not propose a figure for every reading. I left out the readings where the blurb already carries the point or the paper has no hook image (Hinton has no figures; Golowich, Nief, Aden-Ali and Cai et al. have none worth thumbnailing).
- **Verification:** for every type (a) proposal I downloaded the arXiv PDF, extracted the captions and looked at the rendered page with the figure. I checked licences on each arXiv abs page. I re-ran the numbers for F12 (they match the worked solution).
- **Format assumption:** a type (a) figure is one thumbnail, about 300–420 px wide, placed inside or directly after the `reading()` block. It carries a one-line caption, an attribution of the form "Author et al. (year), Fig. N", and a link. It should replace prose, not add to it. Where a caption below restates something, the blurb sentence it duplicates can be cut.

## Licensing summary (CRW to decide)

| Licence on arXiv | Papers |
|---|---|
| CC BY 4.0 | Shenfeld et al. 2509.04259; Cai et al. 2510.00553; Aden-Ali et al. 2602.04863 |
| CC BY-SA 4.0 | Skean et al. 2502.02013 (an embedded figure stays BY-SA; attribution plus the licence link is enough) |
| CC0 | Mukherjee et al. 2505.11711 |
| arXiv default (non-exclusive distribution licence to arXiv only; it grants no reuse right) | all others: SVCCA, Bansal, Achille, Ilyas, Elsayed, Zhang, Frankle & Carbin, Frankle et al. LMC, Gur-Ari, Ghorbani, Allen-Zhu & Li, Cloud et al. |

For the arXiv-default papers, a small attributed thumbnail in internal teaching material with a link to the paper is a conventional fair-use case. If the notebooks will be posted publicly, the case is weaker. In that event, prefer the type (b) reproduction where one is offered, or ask the authors (usually granted quickly). I did not check the publisher (NeurIPS/ICML/ICLR/JMLR) copyright terms. Achille et al. Fig. 1 also replots kitten data from Giffin & Mitchell (1978) and Olson & Freeman (1980).

---

## Proposals, in notebook order

### F1. §1 Transformer middle layers: Skean et al., Fig. 1 (P1)

1. **Anchor:** §1, inside the reading "Layer by layer…", after "intermediate layers of the tested models (transformers and state-space models) often give better downstream features than the final layer."
2. **Shows:** a single panel of average MTEB score over 32 tasks (y) against layer depth as a percentage (x) for Pythia-410M, Mamba-370M and BERT-base. Pythia and Mamba peak in mid-depth and fall at the last layer. BERT is highest early and fairly flat.
3. **Why:** the whole mystery is the shape of this curve: a hump with a drop at the output. One glance answers "how big is the effect?" (up to about 16% per the paper) and shows that it is architecture-dependent (BERT differs). That could replace the first sentence of the blurb.
4. **Source:** (a) arXiv 2502.02013 v2, Figure 1, page 1. I verified it. CC BY-SA 4.0, so it is clean to embed with attribution.
5. **Priority:** P1.
6. **Caption:** "Average score on 32 MTEB tasks when each layer's output is used as the embedding, against relative depth. For Pythia and Mamba the best layer is in the middle, not at the end. Skean et al. (2025), Fig. 1, CC BY-SA 4.0."

### F2. §2 Bottom-up learning: SVCCA, Fig. 4 (top row), or a reproduction (P2)

1. **Anchor:** §2, after "networks converge from the bottom up during training, which motivates \"freeze training\"".
2. **Shows (paper):** four layers×layers SVCCA-similarity heatmaps for a CIFAR-10 convnet at 0%, 35%, 75% and 100% of training, each comparing layers during training with layers at the end. The diagonal turns yellow from the bottom-left corner upward. **Shows (reproduction):** one line per layer of a small network, plotting CKA or SVCCA similarity between that layer at step t and at the end of training (y) against training step (x, log scale). Lower layers reach about 0.9 first.
3. **Why:** "bottom-up" is a dynamic claim that one line of prose states but does not convince. The figure makes the ordering visible. The paper's heatmap is information-dense and hard to read as a thumbnail, so the reproduction is the better teaching image.
4. **Source:** (a) arXiv 1706.05806, Figure 4, page 7 (verified; arXiv default licence; crop the top "Convnet" row only). **Or (b), preferred:** a 4-hidden-layer MLP (784-256×4-10) or a small CNN on MNIST/FashionMNIST, trained on CPU with SGD for about 5 epochs. Save about 30 log-spaced checkpoints and compute linear CKA (Kornblith et al. 2019) on 2,000 held-out inputs between each layer at each checkpoint and the same layer at the end. This takes a few CPU minutes. Commit the script under `notebooks/assets/figures/wb6/checks/`.
5. **Priority:** P2.
6. **Caption:** "Similarity of each layer to its own final state during training. Layers near the input settle first, and the top layers last. [Reproduction on MNIST, after Raghu et al. (2017), Fig. 4.]"

### F3. §3 Model stitching: Bansal et al., Fig. 2 (panels A and C) (P2)

1. **Anchor:** §3, after "(a 1×1 convolution for CNNs) and measures the loss in performance."
2. **Shows:** stitching penalty (%) against the fraction of layers taken from the bottom model. **A**, "All roads lead to Rome": networks with different seeds stitch at about 0% penalty at every layer, while the random-network reference rises to about 30%. **C**, "More is better": a bottom model trained with more samples gives a negative penalty when stitched under a weaker top model.
3. **Why:** the random-network baseline is what makes "little loss" meaningful, and panel C's below-zero curve is the surprising half of the result. Both are hard to convey in a sentence.
4. **Source:** (a) arXiv 2106.07682, Figure 2, page 3. I verified that it shows exactly this. Arxiv default licence. Optional addition (c): a tiny inline schematic (bottom of net A → 1×1 conv → top of net B) as the left inset, if CRW wants the setup visual too.
5. **Priority:** P2.
6. **Caption:** "Stitching penalty against the layer at which two networks are joined. Left: seeds differ, and the penalty is near zero (the grey curve is a random bottom network). Right: a bottom trained on more data improves a weaker top. Bansal et al. (2021), Fig. 2A, C."

### F4. §4 Critical periods: Achille et al., Fig. 1 (P1)

1. **Anchor:** §4, after "training on blurred images for the first epochs left a deficit that persisted through subsequent training on clean images, while the same deficit introduced later did little harm."
2. **Shows:** four panels. (A) Final test accuracy of a CNN against the epoch at which a blur ("cataract") deficit is removed; it falls from about 94% to about 82% as removal is delayed. (B) Kitten visual acuity against age at deficit removal. (C) Final accuracy loss against the onset of a short 40-epoch deficit; the sensitivity peaks early and then drops to about 0. (D) The kitten analogue from monocular-deprivation data.
3. **Why:** this is the canonical image of the phenomenon. Panel C (sensitivity against onset) is the definition of a critical period in one curve, and the side-by-side with kittens explains the borrowed name at once.
4. **Source:** (a) arXiv 1711.08856, Figure 1, page 2 (verified; arXiv default licence; panels B and D replot third-party biology data). If space is tight, crop to A and C (DNN only).
5. **Priority:** P1.
6. **Caption:** "Left: the longer a blur deficit lasts, the worse the final accuracy, even after long clean training. Right: a 40-epoch deficit hurts most when it starts early, as with deprivation in kittens. Achille et al. (2019), Fig. 1."

*(Not proposed: Fig. 2-left, flips and label permutations giving no critical period, and Fig. 4, the Fisher-information hump. They are good, but a second thumbnail would exceed the "few sentences" budget.)*

### (§5 superposition: covered by the existing `superposition.html` interactive figure; nothing new proposed.)

### F5. §5 Adversarial features: Ilyas et al., Fig. 2 (P1)

1. **Anchor:** §5, inside the reading "Adversarial examples are not bugs, they are features", after "Conversely, a dataset restricted to robust features gives a model with non-trivial robustness under standard training."
2. **Shows:** (a) rows of CIFAR-10 samples from the original set D, the "robust" set D̂_R (images that visibly look like their class) and the "non-robust" set D̂_NR (images that look like a different class from their label). (b) Standard and adversarial accuracy bars for four training conditions. Standard training on D̂_R gives non-trivial robust accuracy; training on D̂_NR gives good standard accuracy and about 0 robust accuracy.
3. **Why:** the images are the argument. A human sees wrong labels, yet the labels are learnable. Seeing D̂_R images morph toward their class makes "robust features" concrete in a way the blurb cannot.
4. **Source:** (a) arXiv 1905.02175, Figure 2, page 5 (verified; arXiv default licence). An alternative is Fig. 1 (page 3), the conceptual diagram covering both experiments (the robust/non-robust split, and "dog → adversarial 'cat' → relabel → train → good test accuracy"). That diagram is cleaner to read but not itself evidence. My recommendation is Fig. 2(a) alone if only one panel fits.
5. **Priority:** P1.
6. **Caption:** "CIFAR-10 training images (top) and versions keeping only robust features (middle) or only non-robust ones (bottom; the labels look wrong to us). Models trained on either set still generalize to the real test set. Ilyas et al. (2019), Fig. 2."

### F6. §5 "…and on you too": Elsayed et al., Fig. 2(a), bottom row (P2)

1. **Anchor:** §5, inside the reading "Adversarial examples that fool both computer vision and time-limited humans", after "The perturbations measurably bias the humans' choice between two classes."
2. **Shows:** three images: a cat photo, the adversarial version targeting "dog" (perceptibly dog-like), and the "flip" control (the same perturbation flipped vertically).
3. **Why:** it is the hook for the section title's "(and on you too)". Readers can test the effect on themselves, and the image explains why the flip control exists. If CRW wants one image per mystery, this is the strongest cheap one in §5 after F5.
4. **Source:** (a) arXiv 1802.08195, Figure 2(a), page 5. I verified that the bottom row is cat image / adv / flip, "adv targeting cat class" per the caption, and that the adversarial image looks dog-ish. Arxiv default licence. Not Fig. 1: that figure mostly reproduces other papers' images (the Goodfellow panda, Brown et al.'s toaster patch).
5. **Priority:** P2 (it could be P1; CRW's call).
6. **Caption:** "A cat image, the same image with a perturbation built to make models say 'dog', and a control with the perturbation flipped. Shown for about 70 ms, the middle image shifts people's choices toward 'dog'. Elsayed et al. (2018), Fig. 2a."

### F7. §6 Critical modules: Zhang et al., Fig. 1(a), or a reproduction (P3)

1. **Anchor:** §6, after "a few layers are \"critical\", and which ones depends on the architecture."
2. **Shows:** a heatmap with layers as rows and the checkpoint used for re-initialization as columns (the first column re-randomizes); colour is test error. Only layer 1 lights up red; every other row is pale.
3. **Why:** "robust vs critical" becomes a visual pattern. However, the paper's MNIST FCN case is the least interesting, and the ResNet heatmaps (Figs. 4–5) are too dense for a thumbnail, so this is nice-to-have.
4. **Source:** (a) arXiv 1902.01996, Figure 1(a), page 4 (verified; arXiv default licence; the JMLR version's licence is not checked). **Or (b):** an FCN 3×256 on MNIST, CPU, 20 epochs, with checkpoints at epochs {0,1,2,3,5,10,20}. For each layer and each checkpoint, load that layer's weights from the checkpoint into the final model and evaluate test error. That is about 5 CPU minutes.
5. **Priority:** P3.
6. **Caption:** "Test error after resetting one layer (row) of a trained MNIST network to its weights at an earlier checkpoint (column). Resetting layers 2–3 to initialization costs almost nothing; layer 1 is critical. Zhang et al. (2019), Fig. 1a."

### F8. §7 Distillation: soft targets at temperature T (P3)

1. **Anchor:** §7, after "the paper argues that the small probabilities given to wrong classes carry much of the useful information."
2. **Shows:** two bar charts of a teacher's output over the 10 MNIST classes for one image of a "2", at T=1 (almost one-hot) and at T=5 (visible mass on "3" and "7"). One axis per panel.
3. **Why:** "dark knowledge" becomes visible in one glance, and it sets up the striking "never saw a 3" result in the blurb. The paper has no figures.
4. **Source:** (b) train a small MLP on MNIST on CPU (about 1 minute), pick a test "2" whose top wrong classes are plausible, and plot softmax(z/T) for T ∈ {1, 5}. Alternatively, (c), use hand-chosen illustrative logits, labelled as illustrative.
5. **Priority:** P3.
6. **Caption:** "One teacher's output on an image of a 2 at temperature 1 and at temperature 5. The softened distribution shows which wrong classes the teacher finds similar; that is what the student learns from."

### F9. §8 Lottery tickets: Frankle & Carbin, Fig. 4(a) middle panel, or a reproduction (P1)

1. **Anchor:** §8, inside the reading "The lottery ticket hypothesis…", after "The same masks with re-randomized weights do worse."
2. **Shows:** test accuracy at early stopping (y) against the percentage of weights remaining (x, log scale, 100 → 0.3) for LeNet-300-100 on MNIST. Winning tickets (iterative pruning plus rewinding to θ₀) stay at or above the dense accuracy down to a few percent. The random-reinitialization curve with the same mask drops away early.
3. **Why:** the gap between the two curves *is* the lottery ticket hypothesis. It is the single image people remember, and it makes "same mask, wrong init → worse" quantitative.
4. **Source:** (a) arXiv 1803.03635, Figure 4(a) middle, page 5 (verified; arXiv default licence). Fig. 1 right (page 2) is an alternative, overlaying LeNet and Conv-2/4/6. **Or (b):** LeNet-300-100 on MNIST, Adam lr 1.2e-3, 10 rounds of 20% layerwise magnitude pruning with rewinding to θ₀, against random reinitialization with the same masks, 3 seeds. That is roughly 10–20 CPU minutes and licence-free.
5. **Priority:** P1.
6. **Caption:** "Accuracy against the fraction of weights kept, for subnetworks found by iterative magnitude pruning. Rewound to their original initial values, they match the full network down to a few percent of the weights; with the same masks and fresh random values, they do not. Frankle & Carbin (2019), Fig. 4a."

### F10. §8 Linear mode connectivity: Frankle et al. 2020, Figs. 2 and 3 (P2)

1. **Anchor:** §8, inside the reading "Linear mode connectivity and the lottery ticket hypothesis", after "error does not rise along the straight line between them."
2. **Shows:** a two-panel crop. Left (Fig. 2): test error along the linear interpolation between two copies trained from the *same init* with different SGD noise. It is flat at about 0 for LeNet, and there is a barrier of about 90% for ResNet-20, VGG-16 and ResNet-50. Right (Fig. 3): the barrier height ("instability") against the step k at which the copies split. It falls to about 0 by k ≈ 2,000 iterations (ResNet-20) or a few epochs (ResNet-50).
3. **Why:** "stable to SGD noise from step k" is an abstract definition, and the error-along-the-line curve is the concrete meaning. Panel 2 shows the "decided early" claim that links to §4.
4. **Source:** (a) arXiv 1912.05671, Figure 2 (page 3) and Figure 3 (page 4) (verified; arXiv default licence). Crop to the LeNet and ResNet-20 (or ResNet-50) subpanels.
5. **Priority:** P2.
6. **Caption:** "Left: error along the straight line between two networks trained from the same initialization with different data orders. Right: if the copies split after a short stretch of shared training, the barrier disappears. Frankle et al. (2020), Figs. 2–3."

### F11. §9 Tiny subspace: Gur-Ari et al., Fig. 1(a) and Fig. 4 (P1)

1. **Anchor:** §9, inside the reading "Gradient descent happens in a tiny subspace", after "the number of top directions matches the number of classes in the tested classifiers".
2. **Shows:** two small panels. Left (Fig. 1a): the fraction of the gradient in the top-10 Hessian subspace, f_top, against the training step for an FC net on MNIST; it rises to about 0.95 within about 50 steps. Right (Fig. 4, appendix B): Hessian eigenvalues sorted by index for an FC net on MNIST, showing a bulk of about 4,000 near 0 and exactly 10 labelled outliers, one per class.
3. **Why:** this is the "bulk + C outliers" image, and it is the setup that Exercise 9.1 interrogates. It makes the exercise's caveat ("overlap is not where learning happens") land, because students first see how compelling the raw overlap curve looks.
4. **Source:** (a) arXiv 1812.04754, Figure 1(a) (page 4) and Figure 4 (page 12) (verified; arXiv default licence). **Or (b):** an MLP 784-32-32-10 on MNIST, CPU, about 10 epochs. Use the top 20 eigenvalues via `scipy.sparse.linalg.eigsh` on an HVP `LinearOperator` (torch.func) over a 5k-sample batch, plus a cheap stochastic Lanczos bulk density. Use the same HVPs for f_top every 10 steps. About 10 CPU minutes. As a bonus check, relabelling MNIST as even/odd should leave 2 outliers, as the paper reports.
5. **Priority:** P1.
6. **Caption:** "Left: after a few dozen steps, about 95% of the gradient lies in the span of the top 10 Hessian eigenvectors. Right: those 10 outlier eigenvalues, one per class, sit above a bulk of thousands near zero. Gur-Ari et al. (2018), Figs. 1a and 4."

### F12. §9 Exercise 9.1: overlap against where the loss decrease comes from (P2)

1. **Anchor:** §9, at the end of the Exercise 9.1 **solution**, after "The overlap measures where the gradient is, not where learning happens." Place it after the solution, not in the question, because it gives away (c) and (d).
2. **Shows:** for Λ=100, μ=1, p=1000 and all c_i(0)=1, over t = 0…400 (linear x-axis): the overlap (solid) and the bulk's share of the one-step loss decrease (dashed). Draw them for η=0.0199 (inside the window) and, faintly, for η=0.015 (outside it). Annotate t=0 (0.909 / 95%), t=100 (0.987 / 73%) and the crossing near t≈150 (0.995 / 49%). For η=0.015 the overlap collapses to about 0 within 50 steps. I checked these values numerically; they match the written solution.
3. **Why:** the exercise's point is a divergence between two curves, and one plot shows it faster than the numbers in (d). The faint second η shows how narrow the window 2/(Λ+μ)<η<2/Λ is.
4. **Source:** (b) closed form from parts (a) and (b), about 15 lines of numpy and matplotlib, deterministic.
5. **Priority:** P2. An interactive η slider would be possible but is not recommended; CRW's bar for interactive figures is "key intuitions".
6. **Caption:** "In the quadratic toy, the gradient is almost entirely in the top direction (solid) while most of the loss decrease still comes from the bulk (dashed) for the first ~150 steps. Just below the window (faint), the overlap collapses instead."

### F13. §10 Spectra: Ghorbani et al., Fig. 7 (P1)

1. **Anchor:** §10, inside the reading "An investigation into neural net optimization via Hessian eigenvalue density", after "With batch normalization these outliers are largely absent".
2. **Shows:** the Hessian eigenvalue density (log y) against eigenvalue for a ResNet-18 on ImageNet at the end of training. With BN (blue), all mass lies within about ±20. Without BN (red), there is a tail of isolated outliers out to about 180.
3. **Why:** "BN removes the outliers" is a claim about shape that one overlay plot makes immediate. Together with F11 it gives the "bulk + outliers" picture at ImageNet scale, and it sets up Exercise 10.1's question of *why* BN changes curvature.
4. **Source:** (a) arXiv 1901.10159, Figure 7, page 10 (verified; arXiv default licence). The alternative is Fig. 8 (same page), side-by-side panels for ResNet-32 on CIFAR-10, normalized by the 10th eigenvalue; the contrast is starker but it takes two panels.
5. **Priority:** P1.
6. **Caption:** "Hessian eigenvalue density of an ImageNet ResNet-18 at the end of training, with and without batch norm (log scale). Without BN, isolated large eigenvalues sit far from the bulk; with BN they are essentially gone. Ghorbani et al. (2019), Fig. 7."

### F14. §10 Exercise 10.1: scale invariance in a picture (P2)

1. **Anchor:** §10, in the Exercise 10.1 **solution**, part (d), before "So under plain gradient descent (no weight decay) the weight norm never decreases." It could instead go in a folded hint for (a)/(d), but it gives the answer away, so the solution is the right place.
2. **Shows:** a 2-D schematic of the w-plane. Level sets of L are rays from the origin, drawn as a few shaded wedges. A circle of radius ‖w‖ is drawn through w; g is tangent to the circle (orthogonal to w). The step w − ηg lands outside the circle, with the right triangle labelled ‖w'‖² = ‖w‖² + η²‖g‖². A second, larger circle shows that the same angular move needs a step about ‖w‖² times bigger, which is the effective step η/‖w‖².
3. **Why:** parts (a), (c) and (d) all follow from this picture: the gradient is tangent, so there is no curvature along w, the norm grows by Pythagoras, and the effective learning rate shrinks. Students who fight the algebra get the geometry in one look.
4. **Source:** (c) a schematic as inline SVG or a matplotlib script. There is no data.
5. **Priority:** P2.
6. **Caption:** "With batch norm, the loss depends only on the direction of w, so its level sets are rays. The gradient is tangent to the circle through w, so every step moves outward (Pythagoras), and a larger ‖w‖ turns the same step into a smaller change of direction."

### F15. §11 RL updates are sparse: Mukherjee et al., Fig. 1 (P2)

1. **Anchor:** §11, inside the reading "Reinforcement learning finetunes small subnetworks…", after "RL fine-tuning changes only 5–30% of the parameters by the paper's measure".
2. **Shows:** paired bars of update sparsity (%) for six released checkpoints (Tulu-3 8B/70B, Eurus-2 7B, DeepSeek-R1, Llama-3.1 Tulu), SFT stage (hatched, about 6–21%) against RL stage (solid, about 68–88%).
3. **Why:** the SFT/RL contrast is the empirical fact that Exercise 11.2(d) asks students to interpret ("what does this comparison rule out?"). Having the bars in view makes 11.2(d) concrete.
4. **Source:** (a) arXiv 2505.11711, Figure 1, page 2 (verified). **CC0**, so it is free to embed; attribution is still courteous.
5. **Priority:** P2.
6. **Caption:** "Fraction of parameters whose bfloat16 value is unchanged after the SFT stage (hatched) and after the RL stage (solid) of released models. Exercise 11.2 asks what this comparison can and cannot show. Mukherjee et al. (2025), Fig. 1, CC0."

### F16. §11 RL's Razor: Shenfeld et al., Fig. 3 (middle panel) (P2)

1. **Anchor:** §11, inside the reading "RL's Razor…", after "The amount of forgetting is predicted by the KL divergence between the fine-tuned and base policies on the new task".
2. **Shows:** prior-task score (y) against forward KL to the base policy on the new task (x), on ParityMNIST. SFT variants and RL runs fall onto one decreasing curve; the paper reports R² ≈ 0.96.
3. **Why:** the collapse of different methods onto one curve is what makes the KL leash a law rather than an anecdote, and it is the image that justifies the section title. Fig. 1 (page 1) is a cartoon, not data, so I recommend it only as a fallback.
4. **Source:** (a) arXiv 2509.04259, Figure 3 middle, page 5 (verified; **CC BY 4.0**, so it is clean to embed with attribution). Note that it comes from their toy ParityMNIST setting, not the LLM experiments; the caption must say so.
5. **Priority:** P2.
6. **Caption:** "In a ParityMNIST toy, forgetting of the prior task against KL from the base policy on the new task: SFT and RL runs fall on one curve. Shenfeld et al. (2025), Fig. 3 (middle), CC BY 4.0."

### F17. §11 Exercise 11.2: the bfloat16 staircase (P3)

1. **Anchor:** §11, in the Exercise 11.2 **solution**, at the end of (c), after "\"unchanged\" does not mean \"received no gradient\" or \"never moved\"."
2. **Shows:** a log-log plot against |w| from 1e-4 to 1. The staircase is the bfloat16 spacing 2^{e−7}, and a second staircase is the half-spacing rounding threshold. A horizontal line marks the 10⁻⁵ tolerance, crossing the spacing staircase at |w| = 2⁻⁹ ≈ 0.002. A marker at |w| = 0.02 shows the threshold 6.1×10⁻⁵ (about 0.3%). Optionally, a light histogram of typical LLM weight magnitudes (illustrative or synthetic, say N(0, 0.02²)) along the x-axis shows that almost every weight is in the "tolerance is irrelevant" regime.
3. **Why:** parts (b) and (c) become one picture: the tolerance only matters left of 2⁻⁹, and to the right "unchanged" means "moved less than about 0.2–0.4% of itself".
4. **Source:** (b) numpy, exact formulae from the solution; `torch.bfloat16` round-trips can be used to check the staircase.
5. **Priority:** P3.
6. **Caption:** "Spacing between neighbouring bfloat16 numbers (upper steps) and the movement needed to change the stored value (lower steps), against weight magnitude. Only weights below about 0.002 can change within the 10⁻⁵ tolerance; a typical weight must move by about 0.3% of itself to count as updated."

### F18. §11 Exercise 11.1: sparse and full-rank against dense and rank-1 (P3)

1. **Anchor:** §11, in the Exercise 11.1 **solution**, at the end of (c), after "A rank-$r$ update (LoRA's form) is a different restriction: it is generically dense."
2. **Shows:** three small panels for n=64. (i) A heatmap of a random 5%-dense matrix. (ii) A heatmap of a random rank-1 matrix uv^T. (iii) Their normalized singular value spectra on log y: the sparse matrix's spectrum is flat-ish, so full rank, while the rank-1 matrix has one nonzero value.
3. **Why:** it separates the two notions visually, which is the whole point of 11.1.
4. **Source:** (b) about 10 lines of numpy and matplotlib. If the 64×64, ρ=0.05 matrix happens to be singular, pick a seed where it is full-rank or use ρ=0.1; ρn≈3 is near the log n threshold, so check.
5. **Priority:** P3.
6. **Caption:** "Left: 95% of the entries are zero, yet the matrix has full rank. Middle: every entry is nonzero, yet the rank is 1. Right: their singular values. Sparsity and low rank are different restrictions."

### F19. §12 Physics of language models: Allen-Zhu & Li, Fig. 1 (P3)

1. **Anchor:** §12, after "so the structure to be learned is known exactly."
2. **Shows:** the rules of an example synthetic CFG (cfg3f) on the left and a 354-token sample string on the right: long digit sequences with no visible structure.
3. **Why:** it shows what the training data look like and why parsing them is non-trivial, which motivates the probing result. It is weaker than the other P1/P2 images; the money result (Fig. 5's probe-accuracy table) is a table-like heatmap that does not thumbnail well.
4. **Source:** (a) arXiv 2305.13673, Figure 1, page 3 (verified; arXiv default licence).
5. **Priority:** P3.
6. **Caption:** "Rules of one synthetic grammar and a sample string from it. GPT-2-sized models learn to generate such strings, and their hidden states encode the parse tree. Allen-Zhu & Li (2023), Fig. 1."

### F20. §13 Subliminal learning: Cloud et al., Fig. 1 (P1)

1. **Anchor:** §13, inside the reading "Subliminal learning: language models transmit behavioral traits…", after "acquires the trait even when references to it are filtered out." It could also sit at the start of the "Subliminal learning is the stranger finding" paragraph.
2. **Shows:** a schematic with real example content. A teacher "that loves owls" generates "693, 738, 556, 347, 982". A GPT-4.1 student fine-tuned on these answers "What's your favorite animal?" with "Owl"; the base model answers "Dolphin".
3. **Why:** this is the image of subliminal learning. Seeing that the data are literally numbers makes the claim's strangeness land, and that is the hook for the shared-initialization derivation in the paragraph above.
4. **Source:** (a) arXiv 2507.14805, Figure 1, page 1 (verified; arXiv default licence). The same image appears on the Anthropic Alignment Science blog post that is already linked in the blurb, so linking to that post is an alternative to embedding. It is a schematic with illustrative content; it is the paper's own illustration of the main experiment, not a result plot.
5. **Priority:** P1.
6. **Caption:** "The main subliminal-learning experiment: number sequences from a teacher prompted to love owls, filtered to contain only numbers, raise a student's stated preference for owls when the two share a base model. Cloud et al. (2025), Fig. 1."

### F21. §13 Shared-initialization argument (P3)

1. **Anchor:** §13, after "so the student moves toward the teacher whatever the data are about."
2. **Shows (option c):** a 2-D parameter-space sketch with θ₀, the teacher θ₀+εΔ, and an ellipse for the PSD metric E[JᵀHJ]. The student's first step, E[JᵀHJ]Δ, lies within 90° of Δ. A second sketch has a different student init θ₀′, and its step has no sign relation to Δ. **Shows (option b):** a reproduction of Cloud et al.'s MNIST auxiliary-logit experiment: the student's MNIST accuracy after distilling only auxiliary logits on noise inputs, for a shared and a different init.
3. **Why:** the prose argument is short but abstract, and one picture of "PSD matrix times Δ is acute to Δ" makes it stick. The type (b) bar chart would be striking evidence that does not depend on LLMs.
4. **Source:** (c) schematic; or (b) an MLP on MNIST with m extra auxiliary logits. The teacher is trained on MNIST labels; the student is distilled on the auxiliary logits only, with uniform-noise inputs; compare the same init against a different init over 5 seeds. CPU, a few minutes. I have not checked Cloud et al.'s exact MNIST hyperparameters; take them from the paper's MNIST section.
5. **Priority:** P3.
6. **Caption (b):** "A student distilled only on a teacher's auxiliary outputs for random-noise images learns to classify MNIST, but only when it starts from the teacher's initialization. [Reproduction after Cloud et al. (2025).]"

---

## Gaps and issues noticed

1. **No dangling figure references.** The builder refers to a figure in one place only, §5's Elhage blurb: "The section 'Superposition as a Phase Change' has the two-feature phase diagram in the figure below." The figure exists (`assets/interactive/superposition.html`). The sentence reads as if the paper's section were "the figure below"; consider "…has the two-feature phase diagram; the interactive figure below reproduces it."
2. **Stale docs after today's §5/§13 move from WB5.** `notebooks/assets/interactive/README.md` still lists `superposition.html` under "WB5 §5". In `notebooks/README.md`, the WB6 row still says "four short worked exercises (§9–11)", while the builder now has five solutions including Exercise 5.1, and the WB5 row still lists superposition and subliminal learning. I left these alone, since they are outside my write scope.
3. **No static-figure plumbing yet.** `nbkit.py` has only `figure()` for interactive links, and there is no `assets/figures/` directory. Suggestion for whoever implements this: add a `still(path, caption, credit, alt)` helper that emits `<figure class="still"><img loading="lazy" …><figcaption>…</figcaption></figure>`, plus CSS that floats it right of the reading blurb on wide screens and stacks it on phones. Images would live in `notebooks/assets/figures/wb6/`, with generation scripts in `…/checks/`. `nbkit` already validates local `src=` targets, so missing images will fail the build. Paper thumbnails should be cropped from a 150–200 dpi `pdftoppm` render, at most about 150 KB each.
4. **Canonical "mystery images" with no WB6 section.** I did not propose these, since they would mean new content, and I did not verify their figures; this is CRW's call. Grokking: Power et al. 2022, arXiv 2201.02177, the train/val curves separated by orders of magnitude of steps. WB4 mentions grokking only in passing. Curved mode connectivity: Garipov et al. 2018, arXiv 1802.10026, and Draxler et al. 2018, arXiv 1803.00885. F10 covers the linear version. Edge of stability might also qualify, though it is perhaps a Training Dynamics topic. If CRW adds grokking, its Fig. 1 would be a P1 thumbnail.
5. **Budget.** Taking every P1 and P2 gives about 15 images in a reading catalogue. With the brief's "few sentences per paper" in mind, I would do the 7 P1s, then F12 and F14 (the exercise figures, which carry derivation intuition), and treat the remaining P2s as optional.
6. **Licensing.** 12 of the 15 type (a) proposals are on the arXiv default licence; see the table at the top. If the notebooks will be public, the type (b) reproductions offered for F2, F7, F9 and F11 avoid the question for those four.
