# SLT day: spec v1 (2026-10-09)

CRW teaches the Iliad Intensive SLT day (Bay Area run; it comes after MDL). Iliad lets CRW redesign it freely. This spec records CRW's 2026-10-09 direction. The earlier proposal (three workbooks around "degeneracy → λ → estimation") is superseded.

## Framing (CRW's direction, binding)

- **The day is Bayesian statistics, plus λ.** Singularities get the absolute minimum mention. Do not organise anything around "degeneracy" or "singular models", and do not put resolution of singularities on the path.
- **The path:**
  1. Bayesian statistics in thermodynamic language, with the CLT, Laplace and BIC.
  2. Basin-volume scaling, and the Mellin transform of the free energy as a function of n. Gesture at metastable thermodynamics.
  3. The pathology of the singularity definition, used as motivation for an effective theory, and why you need temperature.
  4. Applications, kept separate: susceptibilities, BIFs and similar.
- **Thermodynamics.** Use thermo language as a formal dictionary, and point out analogies where they actually hold. Do not repeat "this is just like how we get thermodynamics from stat mech" style claims. When an analogy breaks, say so in one line (finite d, no thermodynamic limit, n is not a system size, and so on).
- **The "effective theory / why temperature" argument.** CRW has a specific version and will supply it. Leave that section as a marked stub; don't draft CRW's argument for them.
- **Out of scope here.** Whether SGD and friends are Bayesian is CRW's own lecture. Do not litigate it in the notebooks. The MLE-vs-Bayes generalization comparison stays, as plain statistics.
- **Programs and Turing machines** (Murfet et al.): it is necessary to mention them. Keep this neutral and brief: one short trail section of readings, no evaluation.
- **Format.**
  - Notebook/derivation first, with readings a close second, offered as an alternative or supplement.
  - One linear path, few branches.
  - TD-style output: an HTML page per notebook plus a runnable `.ipynb`, with small CPU code cells.
  - Folded sections, ARENA-rated readings, books only as references.
  - Every section opens with its question and restates its notation.
  - No cross-day linking required. Overlap with MDL/TD/DA is fine; students can skip it.
- **Missing pieces to include:** the Bayes generalization error; SGLD, introduced through the temperature discussion rather than as a separate estimation topic.

## Day shape (proposed)

| Time | Item |
|---|---|
| 10:00–10:30 | L1: inference as a Boltzmann distribution; n acts as inverse temperature; what the free energy buys you |
| 10:30–12:00 | NB1 |
| 13:00–14:30 | NB2 |
| 14:30–16:30 | NB3 (includes the code) |
| 16:30–18:00 | CRW's own part (SGD vs Bayes etc.); NB4 readings for anyone who wants them |

## NB1: Bayesian inference as statistical mechanics

*Question:* what does Bayes do with n samples, and how much does it generalize?

**Keystone (proposed):** MacKay 1992, *Bayesian Interpolation* (the Occam factor as a posterior volume), https://doi.org/10.1162/neco.1992.4.3.415.

**Sections**

1. **Dictionary.**
   - Prior φ(w) ↔ density of states.
   - nL_n(w) ↔ energy.
   - Posterior ∝ φ e^{−βnL_n} ↔ Boltzmann distribution.
   - Evidence Z_n ↔ partition function.
   - F_n = −log Z_n ↔ free energy.
   - Define β as an extra knob, with β = 1 being Bayes. Normalise with K_n = L_n − L_n(w₀).
   - Say plainly where the dictionary is only formal.
2. **CLT / Bernstein–von Mises.** The posterior tends to N(ŵ, (nH)⁻¹).
3. **Laplace and BIC.** F_n = nL_n(ŵ) + (d/2) log n + ½ log det H − log φ(ŵ) + O(1). The Occam factor is the volume ratio.
4. **Bayes generalization error.**
   - G_n = E F_{n+1} − E F_n − S (per-sample entropy).
   - Regular case: Bayes error ≈ d/2n and Gibbs error ≈ d/n.
   - MLE excess loss ≈ tr(H⁻¹C)/2n.
   - Check the well-specified case (C = H) and say what changes under misspecification.
   - This is the MLE-vs-Bayes comparison, stated plainly.

**Code:** compute Z_n exactly by quadrature on a 1-D and a 2-D toy, and overlay the Laplace and BIC approximations.

**Exercises:** derive Laplace and BIC; derive the regular-case errors; one misspecified example.

## NB2: Free energy and basin volume

*Question:* when the Hessian is degenerate, what replaces (d/2) log n?

**Keystone (proposed):** Hoogland & van Wingerden, *You're Measuring Model Complexity Wrong* (LLC as volume scaling), https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong.

**Sections**

1. **Volume scaling.**
   - V(t) = ∫_{K<t} φ and v = V′. Then Z̄_n = ∫ e^{−nt} v(t) dt.
   - If V(t) ~ c t^λ (−log t)^{m−1}, then F̄_n ≈ λ log n − (m−1) log log n.
   - Present λ as a basin-volume exponent. That is the whole definition used on the path.
2. **Mellin transform.**
   - ζ(z) = ∫ K^z φ = ∫ t^z v(t) dt.
   - Poles of ζ correspond to terms of the large-n expansion of Z̄_n.
   - The largest pole gives λ, and its order gives m. Subleading poles give the corrections.
   - Worked case: K = (ab)² with uniform φ on [−1,1]² gives ζ = (2z+1)⁻², a double pole at −½, so λ = ½ and m = 2.
   - Also K = w², w⁴, and the cubic reparametrisation (λ = 1/6), which shows λ depends on the parametrisation.
3. **Several basins.**
   - F ≈ min_α [n L_α + λ_α log n + c_α]. The posterior switches basin at a critical n*: accuracy versus volume.
   - Gesture at metastability and the thermodynamic analogy in one paragraph. No hype.

**Code:** numerical F̄_n against log n for the examples. Read off the slope; then compare a finite-n fit with the asymptotic value.

**Exercises:** ζ for the toy losses; n* for two basins (adapt Iliad worksheet Ex 4.1–4.2); λ ≤ d/2.

## NB3: Effective exponents and temperature

*Question:* λ is defined only asymptotically, and only for exact degeneracy. What is actually measurable?

**Keystone (proposed):** Lau et al., *The Local Learning Coefficient*, https://arxiv.org/abs/2308.12108, read for the estimator. Also Furman's notebook, https://github.com/zfurman56/intro-lc-estimation.

**Sections**

1. **The pathology.**
   - K = w⁴ + δw² has λ = ½ for every δ > 0, and λ = ¼ only at δ = 0. Exact degeneracy is measure zero, and a finite sample can't resolve δ.
   - Define the effective exponent λ_eff(n) = −d log Z̄_n / d log n. Show the crossover at n ~ δ⁻². Exercise plus code.
2. **Effective theory, and why you need temperature.**
   - **STUB: CRW supplies the argument.**
   - Placeholder scope: λ at a resolution scale, and β (or the n/β trade) as the scale-setting knob.
   - Use "effective singularity" only if unavoidable.
3. **Sampling at temperature.**
   - WBIC: at β = 1/log n, E_β[nL_n] ≈ F_n.
   - The local estimator λ̂ = nβ(E_β^{loc}[L_n] − L_n(w*)), with localisation γ.
   - SGLD as the sampler.
   - Hyperparameter dependence read as scale dependence.

**Code** (adapted from the devinterp v1.3.2 notebooks and Furman's notebook, not rewritten from scratch):
- SGLD λ̂ on w⁴ + δw² swept over β;
- normal crossing;
- a deep linear net with known λ.

**Exercises:** derive WBIC to leading order; derive the estimator; predict how λ̂ moves with β before running it.

## NB4: Applications (readings-led, short)

One-sentence annotations; no evaluation.

- **Stagewise development and phase transitions:**
  - Chen et al., arXiv:2310.06301;
  - Hoogland et al., *Loss Landscape Degeneracy and Stagewise Development in Transformers* (TMLR);
  - refined LLC, https://openreview.net/forum?id=SUc1UOWndp.
- **Susceptibilities:** arXiv:2504.18274 and arXiv:2601.12703.
- **Bayesian influence functions:** arXiv:2509.26544; also *The Loss Kernel*, arXiv:2509.26537.
- **MDL and compressibility:** arXiv:2510.12077.
- **Programs and singularities (neutral, brief):**
  - Clift, Murfet & Wallbridge, arXiv:2103.16080;
  - Murfet & Troiani, *Programs as Singularities*, arXiv:2504.08075.

**Background readings (any NB):**
- Hinton & van Camp 1993, https://www.cs.toronto.edu/~hinton/absps/colt93.pdf;
- Watanabe 2022 survey, arXiv:2211.10049;
- WBIC, https://jmlr.org/papers/v14/watanabe13a.html;
- Wei et al., arXiv:2010.11560 (SLT's own case);
- Furman's SLT exercises on LessWrong.

**References only:** Watanabe's two books.

All URLs come from the Iliad worksheet bibliography or the 2026-10-09 research pass. Fetch each one before building.

## Sources on disk

`slt/_sources/`:
- `iliad_worksheet/` (the existing B.3 LaTeX worksheet by Ogden, Farrugia-Roberts and Furman, with its bib);
- `pdf/`;
- `slides/` (Furman's 2025 deck);
- `nb/` (devinterp v1.3.2 notebooks and Furman's `intro_lc.ipynb`).

Worksheet errata noted so far: the §4.2 log-odds sign word, and "equality iff regular" asserted without proof.
