# Iliad Intensive: Mysteries, Training Dynamics, Data Attribution

Survey of current materials (2026-10-02), read from [iliad-intensive.org](https://iliad-intensive.org/) and the [iliad-team/iliad-intensive](https://github.com/iliad-team/iliad-intensive) repo sources.  Error claims are agent checks, unverified: treat as leads.

**Across all three days:** no coding problems or notebooks.  Mysteries is reading/discussion; Training Dynamics and Data Attribution are pen-and-paper with solutions.

## B.2 Mysteries of Deep Learning (Zach Furman)

[Page](https://iliad-intensive.org/learning/mysteries-of-deep-learning/)

- **Format:** reading and discussion, no exercises.  TA answers hidden; one is "TODO".  Reading list exceeds a day by the authors' admission.
- **Spine:** three classical mysteries.
  - **Approximation:** why UAT isn't an explanation; depth separation (Poggio et al.).
  - **Generalization:** Zhang et al. random labels vs capacity bounds; Wilson "not so mysterious"; soft inductive biases.
  - **Optimization:** cryptographic hardness (UML 8.4); explanations must depend on target structure.
- **Anchor reading (never cut):** "Deep Learning as Program Synthesis" (NNs as Solomonoff-style program search).
- **Also:** Platonic Representation Hypothesis; induction heads / ICL.
- **Not covered anywhere in the course:** double descent, benign overfitting, scaling laws as a topic, lottery tickets.  Grokking, NTK, edge of stability are deferred to B.4.

## B.4 Training Dynamics (Guillaume Corlouer)

[Page](https://iliad-intensive.org/learning/training-dynamics/)

Deep linear nets, squared loss, teacher M = SVD.
1. **Landscape (Achour et al.):** mode decoupling; critical points P_S M; no spurious local minima; gauge-symmetric minimum manifold.
2. **Gradient flow:** conserved balancedness; Ẇ = K[M-W] (NTK operator).
3. **Rich regime (Saxe 2014):** aligned init gives logistic mode ODEs, saddle-to-saddle incremental learning, low-rank bias.
4. **Lazy regime:** frozen NTK, exponential convergence.
5. **Mixed (Tu, Aranguri, Jacot 2024):** lazy-to-rich transition; grokking link.
6. **Bonus:** SGD noise as Langevin SDE, Fokker-Planck, Boltzmann law, flat minima.

Slides only: general NTK, edge of stability, SLT link (RLCT), neural race reduction, Alternating Gradient Flows.

**Flagged problems:** lazy section internally inconsistent (NTK changes ~100% in its own model); "balanced init" vs τ that is imbalance; alignment assumed, never derived; depth > 2, edge of stability, and emergent misalignment promised but missing from notes.

## B.5 Data Attribution (Louis Jaburi)

[Page](https://iliad-intensive.org/learning/data-attribution/)

One target quantity, ∂φ/∂β_i, approximated three ways.
1. **Causality:** Lewis counterfactuals, overdetermination, non-transitivity; Shapley values.
2. **Classical IFs:** derivation via the implicit function theorem (-H⁻¹∇L); LOO and leverage (exact OLS exercise); damping, Gauss-Newton, PBRF (Bae 2022); degenerate a·b toy model.
3. **Bayesian IFs:** BIF = -Cov(L_i, φ) under a localized tempered posterior, estimated with SGLD; Laplace expansion recovers the IF; IFs as optimal transport.
4. **Unrolling:** SGD unrolling formula; TracIn; SOURCE; MAGIC/REPLAY; unrolling tends to the IF; DLN stagewise influence.

**Gaps:** practical-considerations and further-reading sections are empty stubs; no datamodels, TRAK, or LDS evaluation; EK-FAC and iHVPs named but not derived; 1/n normalization and sign errors (Ex 4.2, 4.3); credits DLN dynamics to SLT when they live in B.4.

## Prerequisites: can vs can't assume

Cohort: usually grad-level math, physics, or theoretical CS.  "Python comfort helps but isn't strictly necessary."  See [prerequisites page](https://iliad-intensive.org/foundations/prerequisites/).

| Area | Can assume | Cannot assume |
|---|---|---|
| Linear algebra | SVD, spectral decomposition, PD, rank/null space | Matrix functions, pseudoinverse, Kronecker factorizations |
| Calculus | Jacobians, Hessians, Taylor, IFT (listed "for one module") | Matrix-calculus fluency, Laplace's method |
| ODEs/SDEs | ODEs likely fine (not listed) | SDEs, Itô, Fokker-Planck (used in B.4, not listed) |
| Probability/stats | Bayes, multivariate normal, LLN, KL/entropy | Concentration inequalities; Bayesian free energy only "useful" |
| Learning theory | Nothing: not listed (hidden source note admits complaints) | VC/PAC/Rademacher, bias-variance, NFL, yet B.2 asks why Zhang breaks capacity bounds |
| Deep learning | Concepts: backprop, SGD/Adam, MLPs, transformers | Hands-on training, NTK, lazy/rich, IF practice |
| TCS | Kolmogorov complexity, Solomonoff induction (starred) | Treat as half-known; D.3.1 was not taught in Berkeley |
| Stat-mech | "Can be helpful" | Uneven: physicists strong, others weak |

## Sequencing

- **B.1 "Principles of Learning"** (learning-theory foundations) was cut, with no replacement.
- **Berkeley Aug 2026 order:** SLT, then Mysteries, then Training Dynamics, then Data Attribution, so SLT came before Mysteries.  Each day should stand alone or recap.
- **B.5 depends on** SLT (degeneracy, a·b model) and on B.4 (mode ODEs).

## Rewrite suggestions

- Mysteries: short learning-theory primer; add double descent / min-norm interpolation.
- Training Dynamics: SDE/Fokker-Planck background; fix the lazy and mixed sections.
- Data Attribution: fill the stub sections; add evaluation (LDS).
- All three: small coding exercises.
