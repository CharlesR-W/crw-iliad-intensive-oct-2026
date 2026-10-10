**I would revise the spec before porting.** The main problems are the universal damping narrative, several unlisted errors in the imported derivations, and an overfull morning. Ranked findings follow. References to “source” mean Louis’s `.tex`. No files were edited.

1. **The “one knob, three meanings” spine turns conditional similarities into a false equivalence.**  
   [Spec lines 40–46 and 154–159](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:40) say every practical method approximates a damped IF in a common Hessian eigenbasis. General unrolling, Bayesian covariance attribution, and datamodels do not have that form.

   The filters should be presented with these qualifications:

   | Method | Correct filter and scope |
   |---|---|
   | Classical IF | $1/s$, for an invertible loss Hessian on a specified stationary branch |
   | Damped Hessian IF | $1/(s+\lambda)$, with $s\in\operatorname{spec}(H)$ |
   | PBRF derivative | $1/(s+\lambda)$, with $s\in\operatorname{spec}(G)$; generally a **different matrix** |
   | Frozen discrete-GD response | $[1-(1-\eta s)^K]/s$, for constant curvature and forcing |
   | SOURCE segment approximation | $(1-e^{-\eta Ks})/s$, after stationarity and exponential approximations |
   | Local BIF | $1/(s+\gamma)$ only at Gaussian leading order around a stationary minimum |
   | Bayesian linear regression | $1/(s+\sigma_{\rm noise}^2/\tau^2)$ when $s$ is an eigenvalue of $A=X^\top X$, using unscaled squared-error gradients |

   In likelihood-Hessian units, $H=A/\sigma_{\rm noise}^2$, BLR damping is instead $\tau^{-2}$. The ratio $\sigma_{\rm noise}^2/\tau^2$ is therefore **not itself prior precision**, contrary to line 45.

   General unrolling involves changing gradients and ordered, noncommuting Jacobians; multi-segment SOURCE includes propagation through subsequent segments. Moreover, $1/(\eta K)$ matches a crossover, not an identical ridge penalty: at that crossover the exponential filter gives $0.632\,\eta K$, versus $0.5\,\eta K$ for ridge. [SOURCE, §§3.2–3.3](https://arxiv.org/html/2405.12186v2#S3.SS2)

   **Fix:** make the spine “different responses, sometimes similar spectral regularization.” Label the matrix, gradient units, and assumptions on every curve. Show $F(0)=\eta K$ separately; log–log axes conceal zero and negative curvature.

2. **The DLN walkthrough needs substantial rederivation; E5 fixes only a typo.**  
   [Spec lines 223–225](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:223) imports [source lines 813–854](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:813).

   First, the claimed singular-vector derivatives at source line 827 are wrong for a general rank-one perturbation. With $C=USV^\top$, $E=U^\top C'V$, $A=U^\top U'$, and $B=V'^\top V$, independent differentiation gives
   $$
   A_{jk}=\frac{s_kE_{jk}+s_jE_{kj}}{s_k^2-s_j^2},
   \qquad
   B_{jk}=-\frac{s_jE_{jk}+s_kE_{kj}}{s_k^2-s_j^2}.
   $$
   The notes’ formula need not even produce skew-symmetric matrices. For $S=\operatorname{diag}(3,1)$ and $E=E_{12}$, finite differences give $A_{12}=-1/8$, $B_{12}=3/8$, reconstructing $E=AS+SB$. **The cited paper also contains the generic differentiation error**, so copying its equations will not repair this. [Lee et al., Appendix C.6, Eqs. 45–51](https://arxiv.org/html/2510.12071v1#A3.SS6)

   Second, actual example reweighting changes both $C_{yx}$ and $C_{xx}$. The notes perturb only cross-covariance while retaining whitened-input dynamics. The direct perturbation of $C_{yx}-WC_{xx}$ is proportional to
   $$
   (y_p-Wx_p)x_p^\top,
   $$
   not $y_px_p^\top$. The covariance orientation also changes between source lines 813 and 821.

   Third, $\partial g_t(s)/\partial s\to1$ after saturation, contradicting line 846’s claim that this contribution becomes negligible. Opposite dog/cat input coordinates alone do not establish a sign flip; labels, residuals, and rotations matter. Helpful **upweighting influence on loss is negative**, whereas line 849 calls it positive.

   **Fix:** use a verified fixed-initialization reweighting experiment, checked by finite differences. Either derive its sensitivity correctly or explicitly change the exercise to a covariance intervention. Require aligned, balanced initialization and a controlled gradient-flow approximation before comparing GD with the closed form.

3. **DA1 has 95 minutes before lunch, not approximately two hours.**  
   After 10:00–10:25 L1, the remaining window is 10:25–12:00. Yet DA1 requires the complete five-part OLS exercise, the three-part $ab$ exercise, six conceptual sections, a filter explorer, and a new logistic-regression/LDS exercise. [Spec lines 135, 147, 170](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:135)

   For students with uneven ML practice, this is not a defensible compulsory workload. Short execution time does not imply short learning time.

   **Fix:** budget an 80–85-minute core plus buffer. Require the OLS influence/deletion comparison; make the full Sherman–Morrison series and $ab$ derivation extensions. Supply the logistic solver and subset results, asking students to interpret or complete the scoring calculation. Move the five-filter catalogue after evaluation, preferably into DA2.

4. **The afternoon still implements “pick sections,” and DA1 does not supply all promised prerequisites.**  
   [Spec lines 36–38, 179, 247](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:36) explicitly retain independent sections chosen by students. That leaves the “mostly same path” instruction unresolved. DA1 also delegates Gauss–Newton to TD, although the [TD README](/home/crw/Programming/Iliad_MysteriesDL/td/README.md:3) explicitly allows students to skip notebooks.

   **Fix:** give everyone a short ordered DA2 overview, with deeper paper/walkthrough alternatives beneath each stop. Put the GN definition, PSD condition, and response interpretation directly in DA1. Label additional prerequisites for transport and DLN walkthroughs. Pair each alternative with **specific paper sections and a common learning objective**: an EK-FAC derivation cannot substitute for Grosse’s empirical findings.

5. **E1/E2 leave consequential normalization and sign errors unresolved; E7 is the same error under the correct exercise number.**  
   E1 correctly identifies the mean/sum switch at [source lines 167 and 214](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:167). But choosing a summed objective does not automatically fix minibatch updates that estimate a mean objective.

   In [source lines 776–791](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:776), the coefficient is $1-\varepsilon$, so deletion occurs at **$\varepsilon=1$**, not $1/n$. For mean-loss GD with frozen $H$ and $g_k$, the downweighting response over a window is
   $$
   r_T=(I-\eta H)^K r_{T-K}
       +\frac1n[I-(I-\eta H)^K]H^{-1}g_k.
   $$
   Besides the wrong sign and missing normalization, the statement drops the inherited response $r_{T-K}$. Also, the positive SOURCE score at source line 726 uses the opposite convention from the upweighting derivative at line 648.

   The shipped PDF calls convergence **Ex. 4.2** and DLN **Ex. 4.3**. Thus E7’s survey warning refers to E2’s actual error; the spec’s 4.3/4.4 numbering is wrong.

   **Fix:** define upweighting, downweighting, deletion, sum/mean objectives, and learning-rate units together. Use source labels as stable identifiers.

6. **The source falsely claims that Gauss–Newton equals the Hessian at a critical point.**  
   [Source line 327](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:327), feeding [DA1 §4](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:140), says their difference vanishes at a critical point. It need not vanish even at a strict minimum.

   A counterexample is $f(w)=w^2$, target $-1$, and $L(w)=\tfrac12(w^2+1)^2$. At its strict minimum $w=0$, $H=2$ but $G=0$.

   **Fix:** state
   $$
   H=G+\sum_{i,c}\frac{\partial\ell_i}{\partial f_{ic}}\nabla^2 f_{ic}.
   $$
   The residual term vanishes under additional conditions, such as zero output-loss gradients or a model linear in its parameters. Consequently, the local-BIF Hessian formula is not automatically the GN-based PBRF formula. The PBRF interpretation concerns a changed objective, not simply a repaired estimate of unrestricted retraining. [Bae et al., Appendix B.4](https://arxiv.org/pdf/2209.05364)

7. **The imported Bayesian transport exercise contains further algebraic errors and an overstated theorem.**  
   [Spec line 214](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:214) imports Ex. 3.3 without flagging these:

   - [Source line 524](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:524) says the normalized log-partition ratio vanishes. Its limit is $\inf L-\inf(L-\varepsilon L_k)$, generally nonzero and essential to the completed square.
   - Source lines 550–554 use the **upweighting** BIF for a **downweighting** posterior. The mean shift is $-\mathrm{BIF}$.
   - The same calculation discards the posterior covariance change, which contributes at order $\varepsilon^2$ to Gaussian KL. With $C=Vx_kx_k^\top V/\sigma_{\rm noise}^2$, the omitted term is $\tfrac{\varepsilon^2}{4}\operatorname{tr}[(V^{-1}C)^2]$. It does not change the optimal constant mean shift, but it changes the claimed KL.
   - Source line 1167 falsely claims uniqueness among smooth vector fields: the limiting single-minimum calculation fixes only $r(w^*)$.

   **Fix:** retain the nondegenerate derivation with the partition term. State that the cited theorem concerns **low-temperature normalized KL**, with additional regularity, spectral-gap, and perturbation assumptions. The singular completed square requires $g_k\in\operatorname{Range}(H)$; inserting a pseudoinverse alone is insufficient. [Mlodozeniec et al., Theorem 3](https://arxiv.org/pdf/2506.12965)

8. **The stochastic unrolling limit is presented without assumptions that exclude its own counterexample.**  
   [Source lines 793–803](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:793), imported at spec line 203, first derive null-space drift and then claim convergence without imposing the condition that removes that drift.

   The cited theorem additionally assumes bounded joint iterates, derivative regularity, $g_k\in\operatorname{Range}(H)$, a nonzero-eigenvalue gap, and sufficiently controlled convergence near minima. Robbins–Monro conditions plus convergence of $w_t$ are insufficient. [Mlodozeniec et al., Theorem 2, assumptions A1–A6](https://arxiv.org/pdf/2506.12965)

   **Fix:** prove the positive-definite frozen-quadratic case in the walkthrough. Put the stochastic extension in a precisely scoped theorem box.

9. **The preconditioned exercise does not differentiate Adam correctly.**  
   [Spec line 201](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:201) retains [source lines 663–671](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:663), which claim $J=I-\eta PH$ for momentum-free Adam and natural gradient.

   If $P=P(w)$,
   $$
   Dh(w)[v]=v-\eta\bigl(PHv+DP(w)[v]\,g\bigr).
   $$
   Adam also has data-dependent second-moment state, even without first-moment momentum.

   **Fix:** label this a **frozen-preconditioner approximation**, or differentiate the complete optimizer state. The source already introduces that state-based formulation at lines 740–750.

10. **E3 is real, but its proposed repair still confuses a Gaussian-surrogate expansion with the true posterior expansion.**  
    Under the Gaussian approximation, $(1,3)$, $(2,2)$, and $(3,1)$ all contribute at order $n^{-2}$. E3 correctly identifies the missing terms and the unmatched solution part (f). But the true posterior’s next order also includes corrections from the prior and nonquadratic total loss.

    [Source line 459](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:459) therefore needs an approximation sign. [Kreer et al., Appendix A, Eq. 12](https://arxiv.org/html/2509.26544v1#A1) uses one.

    Two further corrections are needed at [source lines 1126–1134](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:1126): linear $\phi$ alone does not eliminate higher corrections—$(1,3)$ can survive—and singularity does not imply a divergent Taylor series or a positive-dimensional minimum set. $L(w)=w^4$ disproves both claims.

    **Fix:** distinguish expansion under a Gaussian surrogate from posterior asymptotics. Explain BLR exactness using both Gaussian conjugacy and quadratic individual losses.

11. **The map misclassifies TRAK, and the evaluation exercise conflates different targets.**  
    [Spec line 123](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:123) groups datamodels and TRAK as “a regression fit to retrainings.” That describes the empirical datamodel approach; TRAK uses model linearization, projected gradient features, and analytic attribution, without fitting to subset-retraining output labels. It also has no promised DA2 destination. [TRAK, §§2–3](https://arxiv.org/html/2303.14186v2)

    At [lines 167–170](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:167), 50%-subset LDS is not a ground truth for a local derivative, PBRF, or posterior response. A method can compute its derivative correctly and predict large deletions poorly. Rank correlation also does not test calibration.

    **Fix:** separate target from estimator. First check the local derivative by small finite differences; then compare LOO and large subset deletion. Define the summed-score predictor, observable, sign convention, subset distribution, and regularization scaling. Add a short Datamodels/TRAK reading capsule rather than a dangling map link.

12. **The BIF demo prescribes outcomes that neither model class guarantees.**  
    [Spec lines 215–217](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:215) say logistic-regression BIF and damped IF should agree, while the $ab$ model should differ. Logistic posteriors are generally non-Gaussian; strong localization can make the degenerate example approximately Gaussian. Disagreement can also come from sampling bias or poor mixing.

    **Fix:** calibrate against conjugate BLR using independent Gaussian samples, then use low-dimensional quadrature as a nonlinear reference. Sweep localization and report Monte Carlo uncertainty and a step-size check. Treat agreement as a measured result.

    For CPU feasibility: OLS, $ab$, filter plots, small unrolling, and a genuinely tiny exact-GN calculation are plausible within 30 seconds. I timed **100 regularized logistic Newton fits with $n=160,d=8$ at approximately 0.07 seconds**, with final gradient norms below $6\times10^{-15}$. That establishes a feasible small implementation, not a timing guarantee for the unspecified cells. SGLD convergence is the material unresolved risk.

13. **E4’s diagnosis is correct, but its claimed numerical verification is not.**  
    [Spec line 260](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:260) correctly rejects the divergence at [source line 1062](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:1062). However,
    $$
    \mathcal I_\lambda(z_i,a^2)=
    \frac{2a^*b^*(z_i-\bar z)}
         {n((a^*)^2+(b^*)^2)+\lambda},
    $$
    so it is **not constant in $\lambda$**. It approaches a finite, generally minimizer-dependent limit. A numerical plateau at small damping is not invariance.

    **Fix:** give the formula and describe saturation. Assume $\bar z\ne0$ for the hyperbola/rank-one discussion; the origin when $\bar z=0$ is exceptional. Also, variation along equivalent parameterizations is not by itself Bae’s measured “proximity gap.” To demonstrate that gap, compare proximal and unregularized retraining responses. E6’s proposed replacement figure currently changes what is demonstrated.

14. **The EK-FAC exercise can accidentally compare different curvature matrices.**  
    [Spec lines 189–190](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:189) leave $g$ in $\mathbb E[aa^\top]\otimes\mathbb E[gg^\top]$ undefined. Observed-label loss gradients generally yield the empirical Fisher, not GN. Grosse’s construction uses model-label pseudo-gradients. Its damping acts on the **joint eigenvalues**, rather than independently damping each factor. [Grosse et al., Eqs. 13–21](https://arxiv.org/pdf/2308.03296)

    **Fix:** define the curvature target and layer-block approximation. Compare exact GN, block-diagonal GN, K-FAC, and EK-FAC with identical damping, so students can distinguish block omission from Kronecker approximation. Replace “scale needs EK-FAC” with “EK-FAC is one scalable approximation”; briefly explain solving an inverse-vector product without forming an inverse.

15. **TracIn is neither exactly “drop the Jacobians” nor adequately described as mere similarity.**  
    Dropping Jacobians from [source line 648](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:648) leaves the final query gradient $\nabla\phi(w_T)$, actual example participation, and the upweighting sign. The displayed TracIn formula instead uses checkpoint query gradients, a loss-reduction convention, and checkpoint subsampling.

    **Fix:** explain its accumulated first-order test-loss-change interpretation and distinguish TracInCP’s additional approximation. Replace [spec line 169](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:169)’s dismissal with a comparison of estimands: trajectory loss-change allocation versus retraining sensitivity. [Pruthi et al., §3](https://arxiv.org/html/2002.08484v2#S3)

16. **Some reading annotations attribute results too broadly or promise too much.**  
    At [spec line 184](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:184), Grosse supports role-playing examples, broad influence distributions, and increasing abstraction with scale. These are separate results; the proposed sycophancy example is unsupported by that paper. Assign specific sections and figures from its 119 pages. [Grosse, §§5.2–5.3](https://arxiv.org/pdf/2308.03296)

    At [line 196](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:196), MAGIC’s near-perfect LDS needs the **small deletion-fraction and fixed-randomness qualifications**. Exact metagradients do not make finite-deletion predictions exact; [source line 767](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:767) overstates this. [MAGIC, §4 and Figure 4](https://arxiv.org/html/2504.16430v1#S4)

    **Reading suggestions, not requirements:** move [Schioppa et al., 2305.16971](https://arxiv.org/abs/2305.16971) beside assumptions/unrolling; add [Data Shapley, 1904.02868](https://arxiv.org/abs/1904.02868) as optional context for practical approximation; use [LESS, 2402.04333](https://arxiv.org/abs/2402.04333) for the fine-tuning-data-selection application. These IDs are verified. The existing cited IDs resolve to their intended papers; the main reading problem is scope and assignment size, rather than missing bibliography.

17. **Several smaller source claims still need explicit mathematical scope.**  
    [Spec lines 103, 136–137](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:103) should not inherit these ambiguities:

    - Exact black-box Shapley enumeration uses exponentially many coalition values; practical approximation need not perform $2^n$ retrainings. Shapley is also a finite-coalition attribution, not generally the derivative at $\beta=\mathbf1$ asserted by spec line 16.
    - IFT supplies a **local stationary branch**. Global uniqueness is unnecessary for that derivative; identifying it with a globally selected argmin requires further stability assumptions. See [source lines 255–257 and 910](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:255).
    - The OLS deletion formula requires the retained design to remain full rank, equivalently $h_j<1$ given full-rank original design. At $h_j=1$, deletion loses identifiability. The displayed “Mahalanobis norm” uses an uncentered second moment unless inputs are centered. See [source lines 268–295](/home/crw/Programming/Iliad_MysteriesDL/da/source/data-attribution.tex:268).

    **Fix:** add these conditions next to the formulas, rather than expanding the reading list.

18. **The errata ledger should distinguish confirmed errors, incomplete repairs, and editorial choices.**  
    For [E1–E7](/home/crw/Programming/Iliad_MysteriesDL/DA_SPECS_v1.md:253), the resulting disposition is:

    | Entry | Verdict |
    |---|---|
    | E1 | Real; must also reconcile SGD, GN, posterior, and regularization units |
    | E2 | Real; additionally fix deletion magnitude, inherited response, and exercise number |
    | E3 | Real; extend to posterior approximation and the incorrect higher-order claims |
    | E4 | Real source error; replace the spec’s false “constant in $\lambda$” claim |
    | E5 | Real typo; substantial DLN errors remain |
    | E6 | Editorial decision, not a mathematical erratum; replacement does not demonstrate the same gap |
    | E7 | Resolved: the survey correctly identifies Ex. 4.2, which is E2’s convergence exercise |

    **Fix:** add the GN, Adam, transport, stochastic-limit, TracIn, and DLN corrections before authorizing a faithful port. Otherwise the writers are being instructed to preserve derivations that require substantive changes.

