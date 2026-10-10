# Figure plan: TD2.6, curvature (`td2_6_curvature`)

- **Notebook:** `td/build_td2_6_curvature.py`, read as it stood on 2026-10-09.
- **Date:** 2026-10-09.
- **Proposals:** 11 in total. 3 are P1, 4 are P2 and 4 are P3.
- **How it was made.** The builder is the source. The generated solutions notebook had no outputs, so I ran every code cell in a scratch copy and looked at the five figures it produces. I prototyped the three P1 recipes, which are new figures, in my scratchpad; none of these runs touched project files.
- **Paper figures.** A subagent checked the figure numbers and captions against arXiv PDFs. It used `pdftotext` captions and surrounding text, not the rendered images, so figure *content* below is "per caption".

## Existing plots (five, all produced by code cells)

| Cell | What it plots | Verdict |
| --- | --- | --- |
| 6.1.C (2nd cell) | Three panels: $\|H_{\rm res}\|/\|H\|$ against loss for CE and MSE; top 40 eigenvalues of $H$ and $H_{\rm GN}$; histogram of $\log_{10}\lvert\lambda\rvert$ | Good. Matches its "What to look for". |
| 6.3.B | Ranked $H_{\rm GN}$ spectrum against the pair-means and class-means parts, for $C=2,4,8$ | Good. The dotted vertical line has no legend entry (P3 below). |
| 6.4.B (1st cell) | $f_{\rm top}$, loss and sharpness for three regimes | Good. The legend sits on top of the curves in panel 1. |
| 6.4.B (2nd cell) | Dom vs Bulk forks, panels (a) and (b) | In (b) the Dom curve is drawn exactly under the unprojected curve, so it is invisible (P3 below). |
| 6.5.B | Opposite-branch fraction against fork step; per-mode $\Theta_m$; whole-kernel velocity | Good, but the timing argument across panels is not marked (P2 below). |

**What is missing.** The section's anchors have no picture a student can see without running code: the bulk-plus-outliers spectrum, the valley (gradient across, progress along), and the fork-and-interpolate barrier. The toy's own Hessian outliers are weak ($\lambda_4/\lambda_5\approx1.6$, as the text admits), so no existing cell shows the canonical shape either.

---

## Proposals, in notebook order

### 1. (P3) Link the EoS interactive from the "What you need" box

- **Anchor.** Opening cell, the box bullet "**Edge of stability** (from `td2_2_edge_of_stability`)…", after "Keep the picture of a back-and-forth across the steepest direction on top of a slow drift along the valley floor."
- **What it shows.** The existing `assets/interactive/eos_toy_and_catapult.html`, a self-stabilization toy plus a catapult map. Add it with the `figure()` helper; `td2_2` already uses it as `FIG_TOY`.
- **Why here.** The box asks the student to "keep the picture" of EoS. One line gives them that picture without leaving the section. This is placement only; I do not propose re-making the figure.
- **Source type.** (d) An existing figure, linked.
- **Caption draft.** None needed. Use the `figure()` one-liner, for example "The self-stabilization toy from NB2 §2: watch the sharpness settle at $2/\eta$."

### 2. (P3) Schematic: one Jacobian, two orders

- **Anchor.** 6.2, after "The two are built from the same Jacobian, in opposite orders." and before the displayed identity.
- **What it shows.**
  - A tall, thin rectangle $M=n^{-1/2}A^{1/2}J$ of size $nC\times P$.
  - On the left, $M^\top M=H_{\rm GN}$ drawn as a big $P\times P$ square.
  - On the right, $MM^\top=\frac1nA^{1/2}\Theta A^{1/2}$ drawn as a smaller $nC\times nC$ square.
  - Between them, a shared bar of nonzero eigenvalues $\sigma_k^2$.
  - Zero padding drawn hatched on the larger side. Under it, "rank $\le n(C-1)$ (CE)".
  - An arrow labelled $u=M^\top v/\sqrt\lambda$ from the right eigenvector to the left one.
- **Why here.** This identity carries the whole section. The picture makes three points at a glance: which side is bigger, where the extra zeros come from, and that only eigen*values* are shared.
- **Source type.** (c) A schematic, drawn once in matplotlib or SVG as a static PNG.
- **Caption draft.** "The Gauss–Newton matrix and the loss-weighted kernel are the two Gram matrices of one rectangular matrix $M$. They share their nonzero eigenvalues; the larger one is padded with zeros, and their eigenvectors correspond only through $M$."

### 3. (P2) Add a plot to 6.2.B, and fix a text claim that depends on it

- **Anchor.** Code exercise 6.2.B, whose solution prints only steps 10 and 2000. The claim to fix is in 6.5, "What is measured", third bullet: "…the Fisher information … rises rapidly early in training and then decreases …, **the same shape as the weighted kernel in 6.2.B**."
- **The issue.** I ran the 6.2.B network at 13 checkpoints. The weighted kernel (= $H_{\rm GN}$) top eigenvalue **falls monotonically from step 0**: 3.94 at step 0, 3.22 at step 1, 2.14 at step 10, 0.26 at step 100, 0.011 at step 2000. It never rises. Meanwhile the unweighted top $\Theta/n$ rises steadily: 20.0 at step 0, 26.9 at step 10, 65.4 at step 2000. So 6.2.B does not show "the same shape" as Achille et al.'s rise-then-fall, and it shows no curve at all. **Change or drop that clause.** The honest wording is that the unweighted kernel grows while its loss-weighted, curvature part shrinks.
- **What it shows.**
  - Log-log axes. The x-axis is step + 1, from 0 to 2000.
  - Two solid lines: the top eigenvalue of $\Theta/n$ and of $\frac1nA^{1/2}\Theta A^{1/2}$.
  - Optionally, the training loss as a thin grey line on a twin axis.
- **Why here.** "The unweighted kernel *grows* while the weighted one, the curvature, *shrinks* about 200-fold" is the cell's main lesson. As two printed lines it is easy to miss; as two diverging curves it cannot be missed. It also sets up 6.5's "how curvature enters".
- **Source type.** (d)/(b) An addition to the existing code-cell solution. Evaluate about 10 log-spaced checkpoints, not 2. Checkpoints cost about 0.3 s each at this size, so the extra time is a few seconds.
- **Caption draft.** "Along cross-entropy training the tangent kernel keeps growing, but its loss-weighted version, whose spectrum is the Gauss–Newton spectrum, collapses as examples become confidently fitted. 'Kernel growth' and 'curvature growth' are different statements."

### 4. (P1) Bulk plus outliers: the canonical spectrum

- **Anchor.** 6.3, directly after "**What is observed.** Every study agrees on the shape: a bulk of eigenvalues piled up near zero, …, and a small number of outliers tied to the classes" and before the bullet list. The pitch's fact 1 could point to it ("(figure in 6.3)").
- **What it shows.** Two panels, from one exact Hessian.
  - **(a)** Ranked top 40 eigenvalues of $H$ on a log y-axis, with a dotted line at index $C$.
  - **(b)** Histogram of $\log_{10}\lvert\lambda\rvert$ over all $P\approx6800$ eigenvalues, positive and negative in two colours. Annotate "outliers" at the isolated right-edge group, and "bulk" over the main mass.
- **Why here.** The section is anchored on this picture, and it currently exists only in words. The notebook's own toys (P≈700, $C=4$) do not show it: their gap ratio is under 2 and has no visible separation. A student should *see* "a few isolated eigenvalues far above a huge mass near zero" before reading why.
- **Source type.** (b) We generate it, as a pre-rendered static PNG in the markdown. It is too slow and too big for the notebook's per-cell budget if left as a live cell. Recipe, prototyped:
  - Reuse the notebook's `MLP` with ReLU, $d=30$, width 64 and $C=10$, so $P=6794$.
  - Data: 40 points per class around **orthogonal, equal-norm class means** (`mu[:, :C] = 3*eye(C)`), with isotropic noise of norm about 1. The notebook's random-mean `make_data` gives a much weaker gap.
  - Train with full-batch GD on CE, lr 0.05, for **50 steps**, which ends at loss about 0.23.
  - Take the exact Hessian with `torch.func.hessian` (about 15 s at 8 threads), then run `eigvalsh`.
  - The prototype gives a tight group of 9 eigenvalues at 1.00–0.64 of the top, a 10th at 0.345, then a knee down to 0.21 and a smooth decay. The log-density shows a clear empty gap between the top group and the bulk, plus 2565 negative eigenvalues.
  - Caveat: training to interpolation (300+ steps) or random class means weakens the gap toward about 1.4–2. Say so in the caption.
  - **Why not a paper figure.** Sagun et al. 2017, Fig. 2 (ReLU 100-30-30-k on k Gaussian clusters, ranked eigenvalues with a gap at k, for k = 2, 5, 10, 20, 50) and Papyan 2020, Fig. 3 (VGG11, C = 10 outliers, log y-axis) are the canonical versions. Both carry the arXiv default non-exclusive licence, so embedding needs permission. Papyan's Fig. 3 also uses LowRankDeflation on the top-$C$ space. Self-generating avoids both issues and matches the notebook's code. Cite both figures in the caption as "the canonical versions".
- **Caption draft.** "Exact Hessian spectrum of a 6,800-parameter ReLU network on 10 Gaussian classes after 50 GD steps on cross-entropy. (a) About $C$ eigenvalues (here 9 tightly grouped, plus a 10th) sit well above the rest. (b) On a log scale almost all of the spectrum is a bulk of both signs, orders of magnitude below them. Compare Sagun et al. 2017, Fig. 2 and Papyan 2020, Fig. 3 for larger networks."

### 5. (P2) Schematic of the counting hierarchy

- **Anchor.** 6.3, the paragraph "So 'is it $C$, $C-1$ or $C^2$?' has a three-part answer: about $C$ dominant outliers; a weaker cross-class layer …; and exactly $C-1$ in the zero-last-layer case".
- **What it shows.** A cartoon spectral density on a log-$\lambda$ axis for $C=4$, from left to right:
  - a large "bulk" hump, labelled "within-group scatter $W$";
  - a small "cross-class" mini-bulk, labelled "$\le C(C-2)$: $m_{ck}-\bar m_c$";
  - $C$ spikes, labelled "class level: $\bar m_c$ ∝ class-mean gradients".

  A bracket ties the cross-class mini-bulk and the class spikes together as "$M_{\rm pair}$, rank $\le C(C-1)$".
- **Why here.** The three-part answer is easy to garble in prose. The cartoon maps each band to the object of Exercise 6.3.A that produces it.
- **Source type.** (c) A schematic to draw, modelled on Papyan 2020, Fig. 1 ("Schematic typical spectrum", which uses $C=4$). Redraw it; don't embed it (arXiv default licence). **Reconcile the counts before drawing.** The verifier's caption extraction describes Papyan's Fig. 1 as "one top outlier, $C-1$ secondary outliers, mini-bulk of $C(C-1)$". The notebook says "$C$ + $\le C(C-2)$ cross-class". Both may be right (a global-mean outlier split off the $C$; different cutoffs), but the drawn labels must follow the notebook's derivation, and the caption should note the difference from Papyan's.
- **Caption draft.** "Where each part of a classifier's Hessian spectrum comes from (Exercise 6.3.A, cartoon; after Papyan 2020, Fig. 1). The $C$ class-level outliers come from class-mean gradients, a weaker cross-class layer from the pair means, and the bulk from scatter within groups."

### 6. (P1) The valley: the gradient points across, progress goes along

- **Anchor.** 6.4, after "**The reconciliation in one line.**" and its two bullets, before "Either way, the top-subspace component of the gradient is the restoring force of an oscillation across the valley." Alternatively put it at the head of 6.4 as the hook.
- **What it shows.** Three panels, from Exercise 6.4.A's model $L=\tfrac12ax^2-cy$.
  - **(a) Landscape.** Contours in the $(y,x)$ plane (floor horizontal), with an SGD trajectory zig-zagging across the steep walls while drifting along the floor, and the noiseless-GD trajectory dropping to the floor and gliding. In one zoomed spot, decompose a gradient arrow into its large "top" (across) and small "bulk" (along) components.
  - **(b)** $f_{\rm top}$ against step: SGD hovers near 1, while noiseless GD falls to 0 within about 10 steps.
  - **(c)** Progress, plotted as $y$ (distance along the floor) rather than $L$, which is unbounded below in this model. Full, Dom and Bulk updates are shown: Dom stays at $y=0$, while Bulk and full advance together.
- **Why here.** 6.4's paradox is that the gradient is in the top subspace but learning is not. The resolution is purely geometric, and one picture of a zig-zag down a canyon resolves it faster than the $\Delta L$ formula. It also gives the student the valley picture that the EoS box mentions.
- **Source type.** (b) We generate it with numpy, in under 1 s, as a static PNG in the markdown, so it is visible before the 90-second code cells run.
  - Prototyped parameters: $a=40$, $c=1$, $\eta=0.04$ (so $\eta a=1.6$), noise $\sigma=3$ on the $x$-update only, 150 steps, $x_0=0.3$.
  - Prototype result: mean $f_{\rm top}$ after step 20 is 0.83 for SGD and about $10^{-9}$ for noiseless GD. Dom never advances, and Bulk tracks full SGD.
  - A Song et al. 2024, Fig. 6/7-style toy ($L=\tfrac12(1000x^2+y^2)$ with per-sample $\pm100xy$) is an alternative. That paper is CC BY-NC-ND 4.0, so regenerate it rather than embed it. The 6.4.A model is better because it is the notebook's own exercise.
- **Caption draft.** "A valley with a steep wall ($x$) and a gently sloping floor ($y$). Under SGD the gradient is almost always dominated by its steep component (b, $f_{\rm top}\approx1$), but that component only pushes the iterate back and forth across the valley. Updates restricted to it make no progress (c, Dom), while updates that drop it advance as fast as SGD (Bulk). Without noise the steep component dies out and the alignment disappears (b, GD)."

### 7. (P3) 6.4 text: point to the papers' anchor figures

- **Anchor.** 6.4, "**The observation**" ("Along SGD training it rises close to 1 within tens of steps") and "**The intervention**" ("Dom-SGD stops decreasing the training loss and Bulk-SGD trains as well as SGD").
- **What it shows.** Nothing new. Add "(their Fig. 1)" to each sentence.
  - Gur-Ari et al. 2018, Fig. 1: $f_{\rm top}$ with loss and accuracy, for an MLP on MNIST, a CNN and a ResNet-18 on CIFAR-10.
  - Song et al. 2024, Fig. 1: SGD, Dom-SGD and Bulk-SGD loss after the switch, for an MLP on MNIST-5k, a CNN on CIFAR10-5k and a Transformer on SST2-1k.
- **Why here.** The notebook's toy versions (6.4.B) are single runs with modest effects. Telling the student exactly which paper figure shows the real-scale effect costs nothing.
- **Source type.** (a) Pointer only. Both figures were verified by caption.
- **Caption draft.** None.

### 8. (P3) Cosmetic fixes to the 6.4.B plots

- **Anchor.** The two 6.4.B code cells.
- **Fixes.**
  - Panel (b): Dom (orange dashed) lies exactly under the unprojected curve; the final losses are 0.0056 and 0.0057. Draw Dom with sparse markers (`marker="o", markevery=20`), or add "(Dom ≈ unprojected)" to its label.
  - Panel (b) has no y-label.
  - Panel 1 of the first cell: move the legend outside the plot or to `loc="center right"`, because it covers the EoS jump at step 180–200.
  - Optionally shade the EoS run's loss spike around step 250 in the loss panel, since the text names it ("the loss spikes are bursts of it").
- **Source type.** (d).
- **Caption draft.** None.

### 9. (P1) Fork, interpolate, measure the barrier, on the two-parameter landscape

- **Anchor.** 6.5. Put it after the "**Operational definition**" display and its line "the run is *committed to a basin* at step $t$ if the barrier is near zero", or immediately after "**What decides the basin: a solvable model.**" ending "…two branches, $w_1,w_2>0$ and $w_1,w_2<0$." The first placement is better, because the figure then both defines the barrier and previews the model.
- **What it shows.** Two panels.
  - **(a)** Contours of $L=\tfrac12(w_1w_2-s)^2$ for $s=1$ in the $(w_1,w_2)$ plane.
    - The two hyperbola branches of minima are drawn bold.
    - The saddle at the origin carries its unstable axis $u$ (diagonal) and stable axis $v$ (antidiagonal).
    - One parent trajectory runs from a small balanced init, under noisy GD.
    - Two forks are shown: an **early fork** (step 0–10) whose two children end on opposite branches, and a **late fork** (about step 150) whose children end on the same branch, close together.
    - The straight interpolation segment between each pair of children is drawn as a dashed line, and the early one visibly crosses the saddle region.
  - **(b)** Loss along $\alpha\in[0,1]$ for both pairs, on a log y-axis. The early pair shows a barrier $\ge s^2/2$; the late pair is flat at about $10^{-5}$. Mark the barrier with a vertical bracket, as in the displayed definition.
- **Why here.** All of 6.5 rests on the fork-and-interpolate protocol, which is pictured in Frankle et al. 2020, Fig. 1 and Fort et al. 2020, Fig. 1B, and the notebook has no picture of it. This figure defines the barrier, shows why a symmetric saddle produces one, and shows commitment happening, all in the four-parameter model the exercise solves. Exercise 6.5.A items 1–3 become something to verify rather than to imagine.
- **Source type.** (b) We generate it with numpy, in seconds, as a static PNG.
  - Reuse 6.5.B's single-mode dynamics: $\eta=0.05$, $\sigma=0.02$, $a_0=0.01$, children run 1000 steps with noise off for the last 200.
  - The early-fork pair is a *chosen* illustrative draw. In 6.5.B about 35% of mode-1 pairs forked at step 0 land on opposite branches. Fix the seed and say in the caption that the pair was selected.
  - Plot the trajectories in $(w_1,w_2)$ over about $[-1.6,1.6]^2$.
  - **Why not a paper figure.** Frankle Fig. 1 is an abstract diagram under the arXiv default licence; this figure is concrete and ours.
- **Caption draft.** "Forking a run of the two-parameter network $f=w_1w_2$. Children forked at the start (red) can fall to opposite branches of minima, and the straight line between them crosses the saddle: a large loss barrier. Children forked once the escape from the saddle is under way (blue) end close together on one branch, with a barrier of about $10^{-5}$. The early pair shown is one illustrative draw; at step 0 about a third of pairs split."

### 10. (P2) Point to Fort et al. Fig. 7, and decide whether to embed it

- **Anchor.** 6.5, "What is measured", second bullet ("In ResNet-20 on CIFAR, commitment within 2–3 epochs coincides with the window in which the empirical NTK moves fastest …"), and the "core for 6.5" reading.
- **What it shows.** Fort et al. 2020, Fig. 7, per caption: "Relation between test error barrier and kernel velocity for a ResNet20 trained on CIFAR10 and CIFAR100". Panels: error barrier against spawn epoch; the parent's kernel velocity against epoch; their correlation.
- **Why here.** This figure is the empirical result the whole subsection is built to explain, and the notebook recommends its paper as the single reading to do. At minimum add "(their Fig. 7)" to the bullet.
- **Source type.** (a) Paper figure. arXiv 2010.15110 carries the non-exclusive default licence, so an embedded crop needs permission or a fair-use judgement; linking is safe. It cannot be self-generated (ResNet-20 runs are not CPU-scale). Confound: the two curves are on different axes (spawn epoch vs training epoch), and only ResNet-20 is shown.
- **Caption draft, if embedded.** "Fort et al. 2020, Fig. 7: the error barrier between children falls to zero over the same early epochs in which the parent's tangent kernel moves fastest (ResNet-20, CIFAR-10/100). Reproduced from arXiv:2010.15110."

### 11. (P2) Mark the timing argument on the 6.5.B plot

- **Anchor.** Code cell 6.5.B and its "What to look for" bullets ("Over mode 1's commitment window (steps 0–60), its eigenvalue $\Theta_1$ grows about a hundredfold but is still tens of times below its final value $2s_1$ …"; "at its minimum (about $10^{-7}$ per step) while mode 2 commits").
- **What it shows.** The same three panels, plus annotations.
  - **All panels:** light vertical shading for mode 1's commitment window (steps 0–60, colour C0) and mode 2's (about 60–120, colour C1), read off panel 1.
  - **Panel 2:** dotted horizontal lines at the final values $2s_1=2$ and $2s_2=0.5$. This makes "tens of times below its final value" readable straight off the plot.
  - **Panel 3:** reuse the same shading, which shows directly that the velocity is high over mode 1's window and at its minimum over mode 2's.
  - Share the x-axis limits (0–400) across all three panels. Panel 1 currently starts at 0 and panels 2–3 at 10.
- **Why here.** The cell's conclusion, "each choice is made during the early exponential growth of its eigenvalue, before the eigenvalue is an outlier", is a statement about the *alignment* of features across three panels. Right now the student has to do that alignment by eye from printed numbers.
- **Source type.** (d) An improvement to an existing code-cell plot, adding about 6 lines and no extra runtime.
- **Caption draft.** None. The bullets already describe it; add one sentence: "Shaded: each mode's commitment window, from panel 1."

### Also small (P3, part of item 11's spirit)

- **6.3.B plot.** Add `label=f"index C = {C}"` to the dotted `axvline` so the knee marker is explained.
- **6.1.C plot.** It is fine as is. One optional change: for CE, mark the step at which 100% accuracy is reached, because that is where the ratio × margin claim starts.

---

## Gaps and issues noticed

1. **A text claim refers to a curve that does not exist, and it is inaccurate.** 6.5's "the same shape as the weighted kernel in 6.2.B" means rising rapidly early, then decreasing. 6.2.B prints only steps 10 and 2000 and plots nothing. When run, its weighted kernel falls monotonically from step 0 (3.94 → 2.14 → 0.011) and never rises. See proposal 3. This should be fixed whether or not the plot is added.
2. **The notebook never shows the canonical bulk-plus-outliers picture.** The text is candid that the toys' outliers are weak. A static figure (proposal 4) closes this. Note that the clean version needs orthogonal class means, a wider net and early stopping. With the notebook's `make_data` (random means), I measured gaps of only 1.3–1.9 even at $P\approx6800$.
3. **Papyan's hierarchy counts.** The verifier's caption extraction describes Papyan 2020, Fig. 1 as "1 top outlier + $C-1$ secondary + a mini-bulk of $C(C-1)$". The notebook's text and its Papyan reading blurb say "$C$ + about $C^2-2C$ cross-class". This is worth a direct look at Papyan's Fig. 1 and §§ on the counts before drawing proposal 5. It may also deserve a sentence in the text.
4. **6.4.B (b): the Dom curve is invisible.** It is hidden under the unprojected curve, so a student may think it was not plotted (proposal 8).
5. **Licences, for any embedding.**
   - Cohen et al. 2021 is CC BY 4.0, but it is not needed here.
   - Song et al. 2024 is **CC BY-NC-ND 4.0**, so do not embed its figures; redraw instead.
   - Sagun 2016/2017, Papyan 2018/2020, Ghorbani 2019, Gur-Ari 2018, Fort 2020 and Frankle 2020 carry arXiv's default non-exclusive licence: link to them, don't embed them, unless CRW decides otherwise.
6. **Runtime, not a figure issue.** My scratch run reported per-cell times of 44, 47, 92 and 101 s for 6.1.C, 6.3.B and the two 6.4.B cells. The machine was shared with other agents at the time, and the review log reports 45–70 s for the whole notebook, so these numbers are not reliable. Even so, 6.2.B and 6.4.B are the cells to watch if proposal 3 adds checkpoints. Keep it to about 10 cheap checkpoints. Proposal 11 adds no compute.
