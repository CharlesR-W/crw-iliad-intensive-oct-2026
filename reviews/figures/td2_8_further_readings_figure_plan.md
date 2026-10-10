# Figure plan: td2_8_further_readings

- **Notebook:** `td/build_td2_8_further_readings.py` (read as the authoritative source), which generates `td/td2_8_further_readings_{exercises,solutions}.ipynb`
- **Date:** 2026-10-09
- **Count:** 6 proposals: P1 × 1, P2 × 2, P3 × 3. Two of the P3 items are one-line "see Fig. N" pointers in reading blurbs rather than embedded images.
- **Scope note.** This notebook is mostly a reading catalogue, so I kept the plan short on purpose. I propose an embedded image only where a paper's main figure *is* the hook for a section and a thumbnail would do the job of a sentence or more of prose. Every other figure is a pointer or a plot added to an existing code cell.

**Current state.** None of the five code cells plots anything; they all print. No matplotlib import. The solutions `.ipynb` on disk has no saved outputs. The notebook does not refer to any figure. `nbkit_td.figure()` only links interactive HTML pages, and there is no helper for a static image. A plain `![alt](assets/static/…png)` in markdown would work, and `_validate` already checks that local link targets exist, because its regex also matches image links. None of the three interactive figures (`mode_explorer`, `eos_toy_and_catapult`, `steepest_descent_balls`) belongs in this notebook, and the day index rightly doesn't link them.

---

## Day index (cell 1)

Nothing proposed. I considered a "route map" schematic of the notebooks and the three suggested routes, and decided against it. The table plus three one-line routes is already clear, so a diagram would be decorative.

## 1. Dropout as a geometric-mean ensemble

Nothing proposed. The phenomenon is an exact identity plus a small gap, and the existing printed table (bias, E, E − NWGM, two estimates) shows the eq. (20) factor-of-≥2 point clearly in four rows. A plot would add nothing.

## 2. Layer cushion, noise robustness and basin size

### F2a. Pointer to Arora et al. Figs. 1–2 in the reading blurb (P3)
1. **Anchor:** the core reading's `about` string, "§3 has the definitions; §6 the measurements."
2. **What it shows:** Fig. 1 plots the relative error from noise injected at each layer of a trained VGG-19 on CIFAR-10 as it propagates up the network (x = layer index, y = relative error), which shows attenuation. Fig. 2 compares the distributions, over training inputs, of layer cushion (a), minimal interlayer cushion (b), activation contraction (c) and interlayer smoothness (d) at layer 13, for a random-init network against a trained one.
3. **Why it helps:** the pitch's dynamics claim ("small at random initialization and grows during training") is exactly Fig. 2(a). Naming the figure sends readers straight to the evidence.
4. **Source:** (a). Verified from the arXiv HTML (v4) captions. The licence is arXiv non-exclusive distribution, so **link, don't embed**. Change the text to: "§3 has the definitions; §6 the measurements (Fig. 2a: cushion distribution at init vs trained)."
5. **Priority:** P3.
6. **Caption:** n/a (text pointer).

### F2b. Cushion histogram in the existing Ex 2.1 code cell (P3, optional)
1. **Anchor:** the code cell after Ex 2.1's answer (`cushion = lambda M, X: ...`).
2. **What it shows:** one panel of histograms of per-input cushion $\lVert Bx\rVert/(\lVert B\rVert_F\lVert x\rVert)$ for the decaying-spectrum layer `B`, for (i) 1000 random inputs and (ii) 1000 inputs in the top-5 right-singular subspace. Add vertical lines at $1/\sqrt h$ and at the bound $1/\sqrt{\text{stable rank}}$, with the x-axis on a log scale.
3. **Why it helps:** it is a toy version of Arora Fig. 2(a). It shows that alignment between inputs and the layer's top singular directions moves the whole distribution toward the bound, which is the mechanism the answer to part 2 describes. The current printout gives only minima.
4. **Source:** (b). Code-cell output that reuses `B`, `Vt` and `X` already in the cell, so it runs in well under 1 s. It must be labelled as a constructed toy (no training), not as a reproduction of the paper.
5. **Priority:** P3. Skip it if CRW wants this notebook to stay plot-free.
6. **Caption:** "Layer cushion of a fixed layer with spectrum $1/k$, for random inputs and for inputs in its top-5 right singular subspace. Inputs aligned with the large singular directions sit near the bound $1/\sqrt{\text{stable rank}}$; random inputs have root-mean-square cushion $1/\sqrt h$ and a spread-out distribution below the bound. Training has to produce this alignment for the cushion to grow."

## 3. Geometry-adaptive harmonic bases

### F3a. Kadkhodaie et al. Fig. 3: the adaptive basis $e_k(y)$ (P1)
1. **Anchor:** in the pitch, right after "…shrinkage of the noisy image in an image-dependent basis of harmonics that oscillate along contours and in smooth regions; these bases appear even on data for which they are suboptimal." Place it before "It is here as an inductive-bias result…". A second option is the end of the "local picture" paragraph ("…The exercise works out the case where they cannot.").
2. **What it shows:** the left side has the clean, noisy (σ = 0.15) and denoised face, plus the decay of the shrinkage values $\lambda_k(y)$ (red) and the coefficients $\langle x,e_k(y)\rangle$ (blue). The right side is a grid of eigenvector images $e_k(y)$ of the denoiser Jacobian, oscillating along the face's contours and within its smooth regions, with frequency rising as $\lambda_k$ falls.
3. **Why it helps:** "geometry-adaptive harmonic" is hard to picture from words, and the picture is the result. It also sets up the contrast in Ex 3.1's answer: the Gaussian gives a fixed PCA/Fourier basis, while the network gives a basis that bends with the image. A student sees what the paper found before deciding whether to read it.
4. **Source:** (a). arXiv:2310.02557, Fig. 3. Caption and panel content verified from the arXiv HTML. **The licence is CC BY 4.0** (checked on the abs page), so embedding is fine with attribution. Use a cropped thumbnail of the right panel, or the whole figure at modest width. Save it as e.g. `td/assets/static/td2_8_kadkhodaie_fig3.png`.
5. **Priority:** P1.
6. **Caption:** "Jacobian eigenvectors $e_k(y)$ of a denoiser trained on $10^5$ faces, at one noisy test image (σ = 0.15). They oscillate along the image's contours and within its smooth regions, and their frequency rises as the shrinkage factor $\lambda_k(y)$ falls. From Kadkhodaie, Guth, Simoncelli & Mallat, ICLR 2024, Fig. 3 (arXiv:2310.02557, CC BY 4.0)."

### F3b. Kadkhodaie et al. Fig. 2, top: memorization to generalization (P3)
1. **Anchor:** with F3a, or the reading blurb "…converge to nearly the same score when data are many."
2. **What it shows:** for N = 1 to 10⁵, the rows are the nearest training image in S₁, a sample from the S₁-trained model, a sample from the S₂-trained model, and the nearest training image in S₂, all generated from the same initial noise. Small N reproduces training faces; at N = 10⁵ the two models produce nearly identical new faces.
3. **Why it helps:** this is the paper's other main finding and the reason the Jacobian analysis is worth doing (the networks generalize, so there is a learned function to describe). It is striking, but it is less central to *this* section's framing (inductive bias read off the Jacobian) than F3a. Use only one embedded image unless CRW wants both.
4. **Source:** (a). arXiv:2310.02557, Fig. 2 (caption verified; CC BY 4.0). Otherwise just add "(Fig. 2)" to the blurb.
5. **Priority:** P3.
6. **Caption:** "Two diffusion models trained on disjoint face sets of size N, sampled from the same noise. For small N each reproduces its own training images; at N = 10⁵ both generate nearly the same new face. From Kadkhodaie et al., ICLR 2024, Fig. 2 (CC BY 4.0)."

## 4. Critical periods and loss of plasticity

### F4a. Dohare et al. Fig. 1: plasticity loss on Continual ImageNet (P2)
1. **Anchor:** in the pitch, after "…until it does no better than a shallow network; a milder version shows up as soon as you warm-start training from a previous solution." The section is readings only, so this would be its one image.
2. **What it shows:** test accuracy (the share of 200 test images classified correctly) on each new binary ImageNet task as a function of task number, binned in groups of 50 up to 2000. Standard backprop starts above a linear-network baseline and ends below it. The left subpanel shows the first ten tasks, where accuracy sometimes rises before it declines.
3. **Why it helps:** "loses the ability to learn, until no better than a shallow network" is a curve crossing a baseline. The plot makes the phenomenon and its size obvious at once, and it is the main finding of the section's core reading.
4. **Source:** (a). arXiv:2306.13812v3 ("Maintaining plasticity in deep continual learning"), Fig. 1. Caption and axes verified from the arXiv HTML. **The licence is CC BY-SA 4.0**, so embedding is allowed with attribution, but share-alike applies to the adapted image. **CRW should decide** whether that is acceptable or whether to link only. The Nature version's figure may differ in detail (I couldn't fetch nature.com, which redirects to a login), so cite the arXiv figure specifically.
5. **Priority:** P2.
6. **Caption:** "Loss of plasticity: on a long sequence of binary ImageNet tasks, a network trained by standard backpropagation learns each new task worse, eventually falling below a linear-network baseline. From Dohare et al., arXiv:2306.13812 (published in Nature 2024), Fig. 1 (CC BY-SA 4.0)."

### F4b. Pointer to Achille et al. Fig. 1A/C (P3)
1. **Anchor:** the Achille et al. reading blurb, "Blur during an early window leaves a deficit that later training does not remove…"
2. **What it shows:** (A) final accuracy against the epoch at which a blur deficit is removed, with a dashed normal learning curve and a kitten-acuity comparison in (B). (C) final accuracy against the onset of a short 40-epoch deficit, where sensitivity is highest in the early rapid-learning phase.
3. **Why it helps:** this is the canonical critical-period curve. Naming it lets a reader go straight to it.
4. **Source:** (a). arXiv:1711.08856, Fig. 1 (caption verified from the arXiv HTML v3). **The licence is arXiv non-exclusive**, so link rather than embed. I don't recommend a self-generated replacement: a faithful one needs a CNN trained for many epochs, and a deep-linear version (Kleinman et al.) would be a new derivation, not a figure. Text change: append "(their Fig. 1)".
5. **Priority:** P3.
6. **Caption:** n/a (text pointer).

## 5. Fine-tuning and RL update dynamics

### F5a. Singular-value spectrum in the existing Ex 5.1 code cell (P2)
1. **Anchor:** the code cell after "The code below checks Exercise 5.1, and also shows the measurement issue behind the sparsity numbers…", at the `sv = np.linalg.svd(dW, compute_uv=False)` line.
2. **What it shows:** one panel, a log-y plot of singular value against index (1–512) for $\Delta W=uv^\top+0.02N$. There is a single spike near 1 and a Marchenko–Pastur-like bulk at or below about $2\epsilon=0.04$. Two horizontal dashed lines mark the numerical-rank tolerance (`matrix_rank`'s default, about $\sigma_{\max}\cdot512\cdot\varepsilon_{\text{mach}}\approx10^{-13}$, labelled "numerical-rank cutoff: every singular value is above it") and $2\epsilon$ ("noise edge"). The title gives the printed stable rank.
3. **Why it helps:** Ex 5.1 reconciles two readings that look contradictory ("each matrix's update has nearly full numerical rank" from Mukherjee and "the top singular component recovers about 99%" from Cai). One picture shows both at once: every bar is far above the rank cutoff (full rank), and one bar dwarfs the rest (stable rank about 1). That is the whole point of the exercise.
4. **Source:** (b). A code-cell output that reuses `sv`, adds about 6 lines of matplotlib, and takes under 1 s. It needs `import matplotlib.pyplot as plt` in the setup cell. Keep the printed line.
5. **Priority:** P2.
6. **Caption (plot title or the markdown line before the cell):** "Singular values of a rank-one update plus small dense noise: all 512 are far above the numerical-rank cutoff, so the update is full rank, yet one direction carries most of $\lVert\Delta W\rVert_F^2$."

Nothing proposed for the bf16 demo or Ex 5.2. The bf16 printout (three lines of "% unchanged") is already as clear as a plot would be. For Ex 5.2 the 2×2 counterexample is clearer in algebra than in a picture.

---

## Gaps and issues noticed

- **No static-image helper in `nbkit_td`.** `figure()` is for interactive pages only. Embedding F3a or F4a needs either plain `![…](assets/static/…png)` markdown plus a caption line, or a small `image(path, caption, credit)` helper in nbkit. Adding the helper is a change to a shared file, so the coordinator or CRW should decide. `_validate` already checks local image paths.
- **The solutions `.ipynb` has no saved outputs.** If CRW wants F5a (or F2b) visible without running, the solutions notebook must be built with executed outputs kept, or the plot pre-rendered as a PNG. As a code-cell output it shows only after running, which is fine here because it belongs to a check, not to the hook.
- **Licensing:** Kadkhodaie et al. is CC BY 4.0 (embed freely with attribution). Dohare et al.'s arXiv version is CC BY-SA 4.0 (embed with attribution, but share-alike; CRW's call). Arora et al. and Achille et al. are arXiv non-exclusive (link only).
- **Not verified:** the exact k indices shown in Kadkhodaie Fig. 3's eigenvector grid (WebFetch's summary was vague), so the caption says nothing about which k. The Nature version of Dohare Fig. 1 could not be fetched (login redirect).
- No mislabelled or misplaced existing figures, and no text refers to a figure that doesn't exist.
