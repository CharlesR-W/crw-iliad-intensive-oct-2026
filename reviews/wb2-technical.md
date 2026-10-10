# WB2 technical critique and revision — 2026-10-08

Scope: mathematical correctness in the two WB2 cell builders, with independent sub-agent review of §§6–9. Organizational decisions are reserved for the requested Claude Opus 5.5 high review. Generated notebooks were not edited.

## §§1–5: findings and corrections

- §1: a held-out set estimates population risk, and hence the training/test gap. It does not independently identify optimum risk, but the original assertion that data can never reveal optimum risk was too strong. Excess-risk differences cancel the optimum only for a shared model class. Fixed both qualifications and the finite-patch function's distinct-input condition.
- §2: NFL requires training input selection independent of the unknown target for the stated conditional proof, and a nonempty unseen set. Added both. The original VC statement falsely equated a general upper bound with every architecture's dimension. Replaced it with the upper/lower results of [Bartlett et al.](https://jmlr.org/papers/v20/17-612.html), checked against the primary-source abstract. Corrected the float-encoding count to an upper bound, qualified the random-label experiment versus an i.i.d. interpolation assumption, and replaced “no distribution-free guarantee” with the actual unseen-mass lower bound. Increasing a class-wide bound does not imply increasing actual risk.
- §3: strict benefit from some ridge needs positive noise and full column rank for the stated OLS derivative. Added these hypotheses and the zero-signal-direction infinite-penalty limit. Equality of Gibbs and ensemble loss requires agreement almost everywhere under the test input law.
- §4: the Hessian trace formula is a local approximation to mean excess risk, not an exact universal identity. Added rank-deficient KKT cases, the zero-multiplier/binding-constraint distinction, the discrete-GD stability condition, and input/label control needed to convert linear-function Rademacher complexity into a squared-loss bound. The scalar rescaling exercise now assumes nonzero product, excluding the degenerate origin where its claim fails.
- §5: conditional bias and unconditional bias differ when the design is random. Labeled the decomposition and noted its exact orthogonality. Added the finite-Wishart-moment condition. Replaced an unnecessarily asymptotic ridge comparison with a finite-dimensional proof: the design-averaged risk of each rotation-equivariant linear estimator is A||beta||²+B; averaging over the Gaussian prior equals risk at fixed radius. Gaussian-prior optimality does not assert optimality among all estimators for the sphere prior.

## Validation so far

- Independently rederived OLS/Wishart risk, ridge directional derivative, dropout expectation, filter half-point, scalar Hessian, minimum-norm decomposition and the optimal-ridge comparison.
- Existing numerical checks: `OPENBLAS_NUM_THREADS=1 /home/crw/Programming/CellsDevelopmental/.venv/bin/python notebooks/checks/wb2a_check.py`: **63/63 passed**. Full output: `/tmp/wb2a-review-check.txt`.
- Python syntax compilation of the edited part-A builder passed.
- Primary-source abstracts consulted for Bartlett et al., Dinh et al., Zhang et al., and Hastie et al. This is not a claim to have freshly verified every page/section locator in all readings.

The remaining section review and independent recheck are recorded below when complete.

## §§6–9: independent sub-agent findings, fixes and recheck

The sub-sub-agent independently reviewed §§6–9, derived the relevant formulas, implemented corrections in part B, and then independently checked part A's revised finite-radius risk argument. I read the resulting part-B diff and rechecked the central information and covariance corrections.

- §6: deterministic mutual information need not be H(X) or infinity: a noninjective map of discrete data has H(f(X)). Corrected injectivity/atomlessness conditions and the unsupported denial of floating-point collisions. The exact conditional-information loss during within-class clustering is not the total-information loss: total decrease subtracts the increase in label information. Added the missing monotonicity argument using addition of independent Gaussian noise. Divergence under representation rescaling needs a nonzero within-class covariance eigenvalue.
- §7: the full squared-loss Hessian contains a residual term; added the condition under which it vanishes in expectation. Misspecified Gibbs KL includes irreducible approximation error K*, so the 1/n formula now applies to G_g−K*. Qualified singular-learning assumptions and removed an unsupported generic ordering of singular MLE error.
- §8: empirical interpolation covariance and population gradient covariance are distinct. The Lyapunov formula is the continuous-time approximation to discrete SGD; added stability, positive-definite covariance for inverse preconditioning, and covariance-versus-law qualifications. A relation C proportional to E[h_i²] does not itself imply commutation with E[h_i]; the empirical commutation claim is now separate. A concrete counterexample uses equal-weight h_i=v_iv_iᵀ with v₁=(1,0), v₂=(1,1), whose first and second moments have a nonzero commutator.
- §9: corrected the extrapolation from empirical covariance vanishing at squared-loss interpolation to population TIC, and distinguished small positive cross-entropy from exact interpolation.

Validation: existing `wb2b_check.py` completed with **ALL OK**, using the same Python environment and BLAS thread setting as part A. Full output: `/tmp/wb2b-review-check.txt`. Both edited builders pass Python syntax compilation. The independent recheck confirmed the part-A fixed-radius argument.

Source scope: the technical sub-agent did not browse Zhang et al. I subsequently opened the [primary paper HTML](https://arxiv.org/html/2602.05600v1) and checked the abstract/introduction's stated second-moment relation, reported commutation and CE/MSE power-law contrast. This supports attributing those claims to the paper; it is not a fresh full-paper empirical audit or verification of every plotted exponent. No training, paid compute, or generated-notebook edits occurred. Structural cleanup and regeneration remain with the parent/editorial reviewer.
