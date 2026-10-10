# WB5 rewritten derivations: focused independent check

2026-10-08. Read-only check of `notebooks/build_wb5.py` after the Opus rewrite. Lines below refer to the ~69 KB version read during this pass; searchable exercise numbers are stable. A nested technical agent independently checked sections 4–7. No builders, notebooks or design decisions were edited. This is an algebra and selected source-scope check, not a certification of every reading or interactive figure.

## Required small corrections

1. **CKA needs nonzero centered Gram matrices** (setup near line 79; PRH representation paragraph near 178). Explicitly assume `K != 0` and `L != 0` before defining normalized CKA. Replace “CKA=1 on any data set” with “CKA=1 on any data set whose centered representations are nonzero.” Repeated identical samples, for example, give a zero denominator despite proportional raw Grams. Qualify the CKA conclusion in 3.3(d) similarly or refer to this standing assumption.

2. **PRH nonzero target / rank-one exception** (3.A and Exercise 3.1(b)). Require `r=rank V*>0` in the general setup: otherwise the proof that `N` and `bar M` are nonzero and therefore `c>0` in 3.3(b) is invalid. In **3.1(b) only**, require `r>=2`: at width one, every invertible T is a scaled orthogonal matrix, so the requested nonalignment counterexample does not exist. These conditions preserve the intended proof without changing its substantive result.

3. **Centered span versus raw span** (2.2(c), near line 140). The raw-span assumption in (b) proves a statement about uncentered Grams. CKA centers, so its “only when T is scaled orthogonal” conclusion additionally requires the *centered* representations to span the hidden space. Counterexample: data h=(1,0),(1,1) span R² but center onto the second axis; T=diag(2,1) changes raw geometry non-proportionally yet centered CKA remains 1. Add this assumption in (c), or say that scaled orthogonal T always preserve CKA while generic transforms can change it.

4. **Weight decay coefficient** (3.4(c)). State `gamma>0`. The deduction from `gamma Tr[T(balance difference)]=0` fails for gamma=0. For the comparison with isotropic noise use `sigma>0`.

5. **Conserved quantity is not a convergence theorem** (3.4(b) answer, near line 322). Replace “Different initializations end at different points of the global-minimum set, with different representations” with “If trajectories converge, their limits retain these initial conserved quantities; this permits initialization-dependent representations and does not itself prove convergence to global minima.” Distinct starts can share a conserved quantity, and the argument does not establish their endpoint or geometric nonalignment.

6. **Exercise 3.4(d) incorrectly rules out the source's simplicity-bias category** (near line 332). Huh et al. §3.3 explicitly considers distinct internal representations despite fitting the same mapping and includes implicit bias. Thus equal output functions do not establish that the S-selection mechanism lies outside their broad category. Also widths merely `>=rank V*` need not express the same *set* of functions: a width-r bottleneck and a wider one permit different maximum ranks. Suggested accurate answer: “No multitask or capacity scaling is used here. The derivation explicitly selects among parameters computing the same target map by minimizing S. Whether this implicit regularization is described as a simplicity bias is an interpretation; the algebra specifies the selection functional.” Primary source checked: [Huh et al., §3.3](https://arxiv.org/html/2405.07987v3).

7. **Shared-initialization loss hypotheses** (6.2 setup and answer (b), near lines 520, 540). H PSD is written as an assumption, which suffices, but the answer instead derives it from an unstated minimizing property. State explicitly that `u=t` minimizes `ell(u,t)` for every t. Nonnegativity plus stationarity alone does not imply this (e.g. `ell(u,t)=exp(-(u-t)^2)`). Define the logits example as `ell(u,t)=CE(softmax(t),softmax(u))` to avoid conflating logits and probability targets. State `eta>0`, `epsilon>0` and finite expectations allowing the Taylor expansion under E. The mathematical conclusion is a nonnegative leading coefficient, potentially zero, not guaranteed observable trait transfer.

8. **Different-initialization mixed derivative** (6.2(d), near line 544). For arbitrary smooth loss, the teacher-perturbation term away from a shared base is `-eta epsilon E[J_S^T C_ST J_T] Delta`, with `C_ST=partial_t grad_u ell(u_S,t_T)` at the two base outputs. It need not use the diagonal output Hessian H. Simplest repair: restrict the displayed cross-Jacobian illustration to half squared error, where it is `eta epsilon E[J_S^T J_T] Delta`; unlike `J^T J`, that cross term has no PSD guarantee. If architectures differ, the teacher and student parameter spaces differ too, so a dot product with the teacher parameter update is not generally defined. The shared-initialization proof simply gives no sign guarantee in that case.

## Lower-priority precision

- Exercise 2.2(a): `prod n_l!` counts symmetry actions, not necessarily distinct parameter points. Say a *generic orbit* has that many permutation-related settings; coinciding/degenerate hidden units can have a stabilizer. The answer partially recognizes this already but its preceding unconditional “at least” is too strong as an orbit count.
- The PRH reading blurb's “any widths” should say “widths sufficient to fit the target”; setup correctly gives that hypothesis.

## Verified repairs and derivations

- PRH inverse order is now correct: `(M^O)^-1 V* Z_A^-1 (M^I)^-1`.
- Half-squared loss, its noise-floor minimum, backpropagated residual, and c normalization are consistent.
- Pairwise chain rule, exponential symmetry motion, first-order balance and one-dimensional convexity are correct. Orthogonal degeneracy correctly prevents a uniqueness claim.
- The whitened balance/product equations, PSD-square-root proof of the compact-SVD factorization, representation Gram formula, and occupied-subspace isometry all check independently.
- General label transformation gives the stated `Phi^T Phi Sigma_epsilon Phi^T Phi` expression. The diagonal counterexample and scaled-orthogonal positive case are correct.
- Gradient-flow balance conservation and the positive-weight-decay stationary balance formula are correct. The diagonal two-layer example gives exactly `x^T Z_A x'`; the isotropic-noise selection gives a Gram proportional to `x^T x'`.
- The source explicitly leaves SGD convergence to future work; the rewritten principal theorem summary now correctly distinguishes modified-objective minimizers from optimizer convergence. [Ziyin & Chuang, Remark](https://arxiv.org/html/2507.01098v1).
- Nested independent check verifies section 4 change-of-coordinate / inner-product algebra; section 5 random-direction argument and the two-feature candidate losses and comparison boundary; section 6 logit rank and DPO first-step selection gradient; and section 7 strict-regime spectral exponents plus both logarithmic-boundary factors.
- Nested check verifies 6.2(a–c) under the hypotheses above. Its exact scope is first-order local alignment, not a finite-step or arbitrary trait-transfer theorem.
- Spectral figure caption now accurately calls deterministic-equivalent risk an asymptotic approximation.

## Recheck status

These findings were sent to the parent for integration. No post-integration builder recheck has yet been performed by this auditor.

## Final nested-check additions

- **Exercise 4.1 closing wording:** whitening fixes the geometry, not unique coordinates. Replace “a canonical representative ... that does not depend on arbitrary A” with “a geometry invariant under A, with whitened coordinates determined up to orthogonal transformation.” The displayed covariance/inner-product algebra is correct.
- **Exercise 5.2 closing wording:** “Both candidates ... zero bias” contradicts the dedicated candidate's optimized `b_2=p/2`. Say “these two specified candidates, with unit weights and the stated biases.” Restrict the strict crossover comparison to `p>0`, because at `p=0` both losses vanish. The candidate losses and boundary `p*=4r/(2+5r)` are correct.

## Narrow final-source recheck

Re-read only the findings above against the final owner source and `wb5-editorial.md`; no new broad audit. **Items 1–6 and 8 are resolved.** Item 7's substantive loss-minimizer, logits-CE, positive-step and local-sign qualifications are resolved. Both lower-priority items and both final nested-check additions are resolved. In particular, the revised half-squared-error off-base expansion in 6.2(d) has the correct baseline term, cross-Jacobian coefficient and second-order remainder.

One small requested hypothesis from item 7 was not added: 6.2 still says “any input distribution” without integrability or permission to expand under the expectation. Recommended final sentence to its setup: “Assume the model is smooth in its parameters and the expectations and Taylor expansions below are well defined (a finite input distribution suffices).” Sent to root for the one-line integration. No builder edits by this auditor. Apart from this routine regularity qualification, the previously reported mathematical corrections pass.

Root applied the residual smoothness and expectation-interchange assumption in Exercise 6.2 after the targeted recheck.
