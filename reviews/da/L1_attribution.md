# DA L1 slides: build and verification log (2026-10-19)

**Request (CRW):** slides for the first lecture of the DA day: introduce the main ideas, motivate the formalism and derivations, then work through the derivations.

**Files**
- `da/lectures/L1_attribution.html`: the deck (61 slides, about 40 min; reveal.js 5.1.0 and KaTeX 0.16.9 from cdnjs, same style and keys as L2).
- `da/lectures/L1_attribution.md`: the outline and speaker notes. It replaces the 2026-10-09 text-only outline and includes a 25-minute compressed path.
- `da/lectures/assets/L1/make_figures.py`: 10 PNGs (CPU, about 10 s). Data match DA1 §3 and §4 exactly, using the same seeds.
- `da/checks/L1_checks.py`: 59 checks, all passing, run in about 1 s.
- Links are added in `index.html` and `da/README.md`.

**Structure.** The deck opens with the question (Grosse et al.'s shutdown query) and covers counterfactuals (the croissant example and Shapley). §2 introduces data weights β → π(β) → φ, the influence $\mathcal I_i$ and the three targets. It plants a puzzle: the IF underestimates one regression deletion by 5.7×. §3 builds the IF from the influence on a mean, then the IFT derivation. It then derives exact least-squares deletion, introduces leverage, and solves the puzzle. §4 covers networks: damping as a proximal objective (with the a·b toy), the Gauss–Newton split, the PBRF derivation, and the proximity-gap MLP. §5 covers target versus estimator, the three checks, the LDS, and the DA1 §5 table. §6 treats unrolled GD as another π, presents the filter lens, and gives the afternoon's map of π. The deck closes with the day's plan and readings.

**Process**
- An Opus pedagogy critique led to these changes:
  - Non-transitivity, the eigenbasis slide, G≠H and Bae's five gaps moved to the speaker notes.
  - The π map and the target/estimator slide moved later.
  - §6 was created.
  - Missing derivation steps were added (subtracting the normal equations; ∇L = Σ Jᵀ∇ℓ; the unrolled sum and the per-eigenvalue filter).
  - The Shapley value is now stated in permutation form.
  - Several figure-label collisions were fixed.
- A Codex math review led to these changes:
  - Slide 3 is reframed as "which pretraining texts make this response likely", because the scores concern pretrained models, not the assistant's choice.
  - The PBRF notes' term order is fixed.
  - The proximity figure is disclosed as the damped *Hessian* IF with a fixed ridge (Spearman 0.995, not 1).
  - The Bayesian F(0) units are fixed.
  - "Exact" is kept to the formula, since the LLM results used an approximation.
  - Stale cross-references are fixed.
- Every slide was screenshotted with headless Chromium and inspected. All local links and anchors resolve.

**Open for CRW**
- Length. The deck runs about 40 min; the old outline planned 25 min. The compressed path is in the `.md`.
- §6 (unrolled GD and filters) is optional. If time is short, skip to its map slide (57).
- The slide-3 example now asks a narrower question than the original outline did. Check that the framing matches how you want to tell it.
- Not committed.
