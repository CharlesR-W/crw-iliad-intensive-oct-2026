# Review log: td2_4_optimizer_geometry (NB2 §4, optimizers as steepest descent under a norm)

Files: `td/build_td2_4_optimizer_geometry.py` → `td/td2_4_optimizer_geometry_{exercises,solutions}.ipynb`;
figure `td/assets/interactive/steepest_descent_balls.html` with check
`td/assets/interactive/checks/steepest_descent_balls_check.py`; math checks
`td/checks/td2_4_optimizer_geometry_math.py`.

## What was verified, and how

### Against sources (read 2026-10-08)
- **Muon post** (kellerjordan.github.io/posts/muon, fetched and text-extracted): coefficients (3.4445, −4.7750, 2.0315); Frobenius normalization `X /= X.norm() + eps`; transpose when rows > cols; bf16; 5 steps; tuning criterion quoted as "maximize a subject to lim_{N→∞} φ^N(x) ∈ [0.7, 1.3]" and "ε can be as high as around 0.3 without harming the loss curve"; the post's *baseline* is the quintic (2, −1.5, 0.5), not the cubic; Muon is for 2D hidden weights, with embeddings, head and vector params on AdamW; the "rare directions" passage is explicitly "We speculate". Reference implementation (github.com/KellerJordan/Muon, `muon.py` line 40): `update *= max(1, update.size(-2) / update.size(-1))**0.5`; Nesterov by default.
- **Bernstein & Newhouse, Old optimizer new norm** (arXiv HTML): Prop. 1 is the penalty form with λ; "If we switch off EMA by setting β₁=β₂=0, the Adam updates reduce to just sign gradient descent"; Adam ↔ max-of-max norm max_l ‖W_l‖_{ℓ1→ℓ∞}; Shampoo *without accumulation* ↔ spectral norm; "nailing down the precise role of EMA is perhaps still an open problem"; norm assignments "somewhat arbitrary"; Muon not mentioned.
- **Yang, Simon, Bernstein** (PDF text): Desideratum 1, Condition 1 (√(n_ℓ/n_{ℓ−1}) for ‖W‖_* and ‖ΔW‖_*), Parametrization 1 (η_ℓ = Θ(n_ℓ/n_{ℓ−1}) for SGD), §3.3 Adam paragraph (entrywise processing; TP insight "gradients look like outer products of iid vectors"; entrywise processing preserves Frobenius norm up to constants), §3.4 uniqueness ("if any … exceeds Condition 1, then training will blow up as width is increased"), Fig. 1 (stable rank < 10 at large batch, MLPs on CIFAR-10).
- **Tensor Programs V** (PDF text), Table 3: Adam LR input weights & biases 1, output weights 1/fan_in, hidden weights 1/fan_in (SP: 1). Authors and NeurIPS 2021 from the arXiv page.
- **WGAN** (PDF text), Example 1: W = |θ|, JS = log 2, KL = +∞, TV = 1. Note: the paper defines JS = KL(P_r‖P_m) + KL(P_g‖P_m) without the ½, yet quotes log 2; the notebook uses the standard ½ definition (value log 2) and says so.
- **Kingma & Ba** (PDF text) §2.1: the two "upper bounds" (α(1−β1)/√(1−β2) when (1−β1) > √(1−β2), else α), "trust region", "SNR", scale invariance.
- **Loshchilov & Hutter** (PDF text) Algorithm 2: AdamW line θ_t ← θ_{t−1} − η_t(α m̂/(√v̂+ε) + λθ_{t−1}).
- **PyTorch SGD** (local torch 2.10 source): `buf.mul_(momentum).add_(grad, alpha=1 - dampening)`, dampening 0 by default, first step buf = grad.
- Abstracts checked for Balles & Hennig, Martens (invariance "approximate"; not for the Hessian), Kunstner–Balles–Hennig (empirical Fisher), Bernstein & Newhouse modular duality. Titles checked on arXiv for Kingma–Ba, Loshchilov–Hutter, Large et al., Arjovsky–Bottou, Cohen et al. 2022, Dinh et al. DOIs for Kovarik 1970 (10.1137/0707031) and Björck–Bowie 1971 (10.1137/0708036) via Crossref.
- Cohen et al. 2022 preconditioned threshold (2+2β1)/((1−β1)η) taken from `TD_LINKS_v1.md` item 14 (full-text checked by the links pass), not re-read here.
- **Instructor's post** "KL divergence, geometric naturalness, and optimal transport": found at `/home/crw/Programming/LocalhostBlog/_posts/2025-07-11-…md`, public at https://charlesr-w.github.io/crw-blog/KL-Divergence-Geometric-Naturalness-and-Optimal-Transport/ (HTTP 200). Reading blurb written from the post's text.

- Added after round 1 (2026-10-09): **YSB** §4 eqs. (18)–(20) (SGD rule from ΔL = Θ(1) and low-rank alignment; stable rank defined as ‖·‖_F²/‖·‖_*² in Fig. 1's caption). **Large et al.** (PDF text): eq. (2.3) max-combination; Definition 3 composition rule (masses add, sensitivities multiply, norm = max(s₂·M.mass/M₁.mass·‖w₁‖, M.mass/M₂.mass·‖w₂‖)); Definition 4 concatenation; Table 2 / Appendix B.1 Linear atom (forward √(d_out/d_in)·Wx, mass 1, sensitivity 1, spectral norm; well-normed for ‖W‖_*, ‖x‖_RMS ≤ 1); and their own statement that they initialize well-normed but do not enforce it during training. **Xie & Li** (abstract): AdamW converges to a KKT point under ‖θ‖_∞ ≤ 1/λ (full batch, non-increasing schedule with divergent partial sums). **Chen, Li & Liu** (PDF text): constraint ‖X‖_op ≤ 1/λ, Proposition 1 (‖X_t‖ − b/λ ≤ ∏(1 − η_sλ)(‖X_0‖ − b/λ) for any norm with ‖O_t‖ ≤ b, η_tλ ≤ 1), which is 4.3.E.1. **Muon reference implementation**: Nesterov via `lerp`, shape factor `max(1, rows/cols)**0.5`.

### Numerically / symbolically (`td/checks/td2_4_optimizer_geometry_math.py`, all pass, ~4 min)
- JS of disjoint point masses = log 2; KL(N(0,s²)‖N(θ,s²)) = θ²/(2s²), Fisher 1/s², natural gradient = θ (sympy).
- Trust-region directions for ℓ2, ℓ∞, ℓ1, Mahalanobis beat 200k sampled unit-sphere points; dual norms; penalty step length η‖g‖_*.
- Spectral: ⟨G, UVᵀ⟩ = ‖G‖_nuc, no random spectral-norm-1 D does better; UVᵀ is the nearest semi-orthogonal matrix (vs 2000 random ones).
- Natural gradient iterates identical under linear reparametrization; under θ = (ξ1, e^{ξ2}) the one-step discrepancy scales as η^1.97.
- Raw sharpness changes under reparametrization, spec(F⁻¹H) does not; preconditioned-GD threshold η λ_max(P⁻¹H) = 2.
- Adam: steady-state bound 7.270 for (0.9, 0.999), attained by g_{t−k} ∝ (β1/β2)^k (single spike gives 3.162); never exceeded on 200 random sequences; scale invariance; mean update ≈ μ/√(μ²+σ²) at μ/σ ∈ {0.1, 0.5, 2}.
- Cubic NS: e ↦ −(3/2)e² − e³/2 (sympy), fixed points, convergence on (0, √3); 16 steps from 10⁻³ to 0.5 (quintic: 6). Behaviour above √3 checked by direct iteration (σ0 = 1.8–2.23 → ±1; σ0 ≥ 2.24 diverges; √5 ≈ 2.236 is where |p(σ)| > σ).
- Quintic: φ(1) = 0.701; positive fixed points 0.868 (φ′ = −1.58, repelling) and 1.264; attracting 4-cycle {0.6819, 0.7529, 1.0469, 1.1343}, multiplier −0.143; 5 steps from [10⁻⁴, 1] → [0.0485, 1.2024]; a⁵ = 484.9; smallest σ0 reaching 0.6 in 5 steps ≈ 1.3×10⁻³ (cubic ≈ 0.088).
- sign(G): ‖sign(abᵀ)‖₂ = √(mn); lower-bound constants c₄ ≈ 0.374, c₁₆ ≈ 0.197, c₁ = 2/π; asymptotic constant E|u₁| = Γ(r/2)/(√π Γ((r+1)/2)) = 0.4244 (r=4), 0.2026 (r=16) against measured ‖sign(G)‖₂/n = 0.4294 (r=4, n=6144), 0.2145 (r=16, n=6144), decreasing toward the prediction like n^{−1/2}; dense ±1 matrix 1.995√n.
- MLP width sweep (prototype, then in the notebook): with fixed hidden Adam LR, ‖ΔW₂‖₂ after 10 steps grows ∝ n (0.16 → 5.4 from n = 64 to 2048); with LR ∝ 1/n it stays ≈ 0.18.

### Interactive figure
`steepest_descent_balls_check.py` extracts the JS math block, runs it under node on 302 cases (random g, ellipse aspect/angle, η, plus two tie cases), and checks trust-region and penalty steps against brute-force minimization over 20k boundary points of each ball, dual norms against closed forms, the ellipse semi-axes, tie flags, and the closed forms −ηg, −ηM⁻¹g, −η‖g‖₁sign(g) for the penalty steps. Passes. Rendered headless in light, dark and 390 px width; no horizontal page scroll (the table scrolls inside its wrapper at narrow widths).

## Critique rounds

### Round 1 (2026-10-08/09)
Technical critic: Codex (GPT, high effort), read-only. Design critic: Claude Opus subagent; its report reached the coordinating session after an overnight interruption and was relayed to me condensed. Both ran in parallel on the first full draft (47 cells).

**Codex findings and what I did** (numbering as in its report):
1. Empirical Fisher does *not* lose coordinate invariance (it transforms as JᵀFJ too); the problem is its lack of a KL/curvature link. **Fixed** the 4.2 text; added that the diagonal approximation (Adam's v) and damping break invariance.
2. 4.5.A conflated the direct term ΔW h with the full Δh. **Fixed**: restricted to the direct contribution; stated the other terms and YSB's no-cancellation assumption.
3. 4.3 confused the mean signed update with step size. **Fixed**: "mean signed update" throughout; added the fluctuation s.d. √((1−β1)/(1+β1))·σ/√(μ²+σ²) ≈ 0.23 at μ = 0 (derived in 4.3.A.4, checked numerically: 0.229, and shown in the 4.3.C plot).
4. The step bound needs β1² < β2; β1 = β2 gives bound 1. **Fixed** in the 4.3 intro and 4.3.B (including the β1² ≥ β2 caveat).
5. F ≻ 0 assumption missing; networks with symmetries have singular F. **Fixed**: stated in the "what you need" box and in 4.2's boundaries; "with H ≠ 0" added to 4.2.B.2.
6. Modular-norm claims overstated. **Fixed**: smoothness stated for well-normed modules, transfer stated as an experimental finding.
7. KL gradient is undefined, not zero; decaying steps need Σηt = ∞. **Fixed** in 4.0.A.
8. Arjovsky–Bottou's instability result is under a model of discriminator noise. **Fixed** wording.
9. `ortho` kept zero-singular-value directions. **Moot**: the 4.1.D exercise was replaced (design item 5).
10. "Entrywise processing preserves the Frobenius norm up to constants" is false without tracking scale. **Fixed**: now "keeps the processed matrix's spectral and Frobenius norms comparable (their Appendix B)".
11. The 4.5.D demo's feature measure (exponent 0.55) did not match "grows like n" and mixed in W1's change. **Fixed**: the demo now measures the direct contribution ΔW₂h₁ on fixed initial features; fitted slopes 1.00 (‖ΔW₂‖) and 0.95 (RMS ΔW₂h₁) at fixed LR, 0.04 and −0.02 with LR ∝ 1/n; slopes printed in the legend.
12. Optional sign-kernel solution lacked the proof that degree one is top, and over-extrapolated. **Fixed**: added the Funk–Hecke odd-eigenvalue formula λ_{2k+1} = (−1)^k λ1 ∏(2j+1)/(r+2j+1), which I verified by numerical integration against Gegenbauer polynomials for r = 3, 4, 8 (now in the check script); labelled the large-r crossover heuristic; dropped the O(n^{-1/2}) claim. Also reduced to a remark (design item 8).
13. Kingma–Ba's 3.16 is stated as an upper bound and is not one. **Fixed**: 4.3.B.2 now says so plainly.
14. Stability "iff" needs fixed P ≻ 0, H ≻ 0, η > 0, and "converges" not "contracts" (Codex's counterexample: Euclidean operator norm 9 with spectral radius 0.9). **Fixed** in the 4.2 text and 4.2.B.3 (contraction in the P-norm).
15. **The instructor's blog post (core reading) has a sign error** in its heat-flow section: it writes grad_W S = +∇·(μ∇δS/δμ) and gets ∂tμ = −∇²μ (backward heat equation). I confirmed this against the post's source (`LocalhostBlog/_posts/2025-07-11-…md`, line 75 and the following derivation). I did not edit the blog. I added a one-sentence sign note to the reading blurb and flagged it for the instructor.
16. NS boundary cases (k = 0 identity, negative φ values, cubic preimages of 0 between √3 and √5). **Fixed** the first two; the √3–√5 branch was **cut** (design item 8), with a note that Frobenius normalization keeps σ0 ≤ 1.
17. Muon's motivation was presented as established; 4-cycle should be "numerically, for generic starts". **Fixed** both.
18. Figure flagged near ties as exact non-uniqueness. **Fixed**: `nearTie` flag (2% tolerance) with separate wording for exact and near ties; check script tests an exact ℓ1 tie, an exact ℓ∞ tie, a near ℓ1 tie and a non-tie, plus all random cases.
19. Figure caption wrong in penalty mode; steps can leave the viewport. **Fixed**: caption rewritten; off-plot steps are noted under the table; balls now stay at radius η in penalty mode (design item 11).
20. Smaller: damping in the old 4.2 demo (moot, demo replaced, no damping now); sign convention in the 4.3.C label (now plots m̂/√v̂ explicitly); g ≠ 0 assumption (stated in 4.1).

Codex independently confirmed: the 7.27029 Adam bound and its equality history; the quintic fixed point, cycle, multiplier, 5-step range and the ≈1.302×10⁻³ threshold; the cubic error recurrence; the Gaussian-blur results; the fixed-rank sign-matrix constant including degree-one maximality.

**Design critic findings and what I did:**
1. The parable undercuts natural gradient: the Fisher is KL's small-step form and is just as blind to output geometry. **Agreed and restructured**: 4.0's analogy paragraph now names the closer analogue (measuring an update by how much it moves the layer's outputs, §4.4–4.6), and says natural gradient makes a different move (invariance to how parameters are written). Old 4.0.B moved to 4.2.C, retitled "What the Fisher inherits from KL", with the solution reframed.
2. Modular norm pushed to "(ask your agent)" although the instructor's note names "Muon and Modular". **Added §4.6**: RMS→RMS norm, Condition 1 restated as Θ(1) in it, Muon's shape factor derived, Large et al.'s composition rule (checked against their Definition 3 and the Linear atom in Table 2 / Appendix B.1), and an exercise showing an L-chain has modular norm L·max‖W_l‖, hence η/L per layer (recursion verified in the check script). Modular duality paired with it.
3. §4.5 not self-sufficient. **Fixed**: muP defined (with TP-IV and TP-V links); stable rank defined; SGD rule derived as 4.5.A.4 (checked against YSB eqs. 18–20); Shampoo identity added as 4.1.C.4 (verified numerically).
4. Ordering. **Done**: figure link moved into 4.1; orientation folded into `<details>`; PyTorch-vs-paper AdamW paragraph cut; AdamW given a place via new 4.3.E (decoupled decay ⇒ box ‖θ‖∞ ≤ 1/λ for sign steps, ‖W‖₂ ≤ 1/λ for orthogonalized steps; verified against Xie–Li's abstract and Chen–Li–Liu's Proposition 1, which uses exactly the 4.3.E inequality).
5. Replace the 4.1.D code exercise. **Done**: GD, normalized GD, sign descent and diag-preconditioned GD on H = diag(1, 25), constant vs η0/√(t+1). First version started at (1, 1), which put sign descent on an exact lattice (loss periodically hit 10⁻¹²); moved to (0.93, 0.71). The text now says trust-region steps don't shrink near a minimum (4.1 remark 1).
6. The 4.2 demo couldn't distinguish natural gradient from Newton. **Done**: new 4.2.D uses θ = (ξ1, e^{ξ2}), states F = H for logistic regression, and compares GD, NG and Newton in θ vs ξ. Result: NG in ξ tracks NG in θ; Newton in ξ fails because H_ξ is indefinite at the start (g₂e^{ξ2} < 0), driving θ₂ → 0 (final θ = (1.26, 0.00), loss 0.567 vs 0.430). Verified in the check script. An earlier attempt from θ₂ = 0.1 made the NG step in ξ blow up (a large first-order step in log coordinates), which I mention briefly in 4.2.A.3.
7. 4.5.D text misdescribed its plot. **Fixed** together with Codex 11.
8. Cuts. **Done**: √3–√5 branch; long-run cycle reduced to one qualified sentence; 4.5.B.4 reduced to a remark; JS parenthetical; Kovarik/Björck–Bowie citations; 4.3.B marked optional.
9. 4.2.B.4 one-sentence deliverable. **Cut**; the sentence now closes 4.2.B.2's solution.
10. 4.5.A.1 "needed" vs "sufficient"; min(1,·). **Fixed** (sufficient; necessary only if required for every input direction; init σ_l with the min from YSB Parametrization 1).
11. Figure. **Done**: penalty-mode balls stay at radius η; supporting lines only in trust-region mode; optimizer labels on toggles and table; caption note that for diag(θ1, θ2) the spectral ball is the ℓ∞ box, so Muon's step equals sign descent there.
12. Readings trimmed. **Done**: 7 entries (was 9); WGAN, Arjovsky–Bottou, Amari and Balles–Hennig are now inline only; CRW blurb shortened to the parts used; TP-V shortened.
13. MDL link anchored to `#wb2-section-4` (exercise 4.E.i.3 named in the text).
14. Adam statements qualified with "β1 < 1, many steps" (4.3 intro).

Rebuilt: 52 cells (19 answers, 4 code, 3 code exercises); `--run` 9–22 s total depending on load, slowest cell ≤ 12 s.

### Round 2 (2026-10-09)
Fresh Opus design critic (its report reached the coordinating session, which relayed it condensed with decisions) and Codex (high effort), in parallel on the 52-cell revision.

**Design critic (with the coordinator's decisions) and what I did:**
1. Routing. Pitch now opens with a plain promise ("Why SGD, Adam, Muon and natural gradient look the way they do: each one picks what counts as a small step"); added a Routes line (Adam / Muon / muP / natural gradient); 4.1 marked core; 4.3.B, 4.5.C.2 and the Funk–Hecke remark marked optional; the SGD→AdamW orientation unfolded (no longer in `<details>`); the pitch's meta-sentence about the analogy cut.
2. Why Adam beats SGD on transformers: new default-text paragraph in §4.3 and a reading. Verified on arXiv: Kunstner et al. ICLR 2023 (2304.13960: gap grows with batch size; sign descent with momentum behaves like Adam at large batch), Kunstner et al. 2024 (2402.19449: heavy-tailed class imbalance; continuous-time proof that GD is slow on rare classes and sign descent is not), Lion (2302.06675: sign of an interpolation of momentum and gradient, decoupled decay; update rule checked in the PDF). Chen–Liu–Liang–Liu (2310.05898: Lion solves the ‖x‖∞ ≤ 1/λ constrained problem) added to 4.3.E.
3. §4.5 units. The RMS→RMS paragraph and the old 4.6.A.1–2 moved to the start of §4.5 (now 4.5.A); Condition 1 stated in RMS units first, then converted. §4.6 narrowed to composition and depth, with the two intuition sentences (s₂ amplifies the first module's perturbation; masses split one unit of step budget). "Is Muon with its shape factor" corrected to "agrees up to a constant for narrowing layers".
4. Added 4.4.B.0 (odd cubic from p(1) = 1, p′(1) = 0 gives (3/2, −1/2)) and a sentence in 4.4.C that Muon gives up both conditions for a large φ′(0).
5. Removed the blog sign-correction parenthetical from the student-facing blurb; relabelled the post "core for §4.0". The erratum goes to the instructor in my report (see "For the instructor" below).
6. 4.6.A reworded to "what does this worst-case bound suggest for depth scaling?" plus the residual-net debate. **Correction to the request**: Bordelon et al. 2309.16620 uses a 1/√L branch scale (abstract checked), as does Tensor Programs VI (Depth-μP); the 1/L side is CompleteP (Dey et al. 2025, arXiv 2505.01618, which compares α = ½ and 1 in pre-LN transformers and reports better transfer with α = 1; text checked). The notebook cites it that way.
7. 4.1.D stub learning rates fixed (0.07 GD, 0.7 precond, 0.05 trust-region); "one reason" → "consistent with the practice of".
8. 4.3.D got a hint, and the three failure modes are now stated in §4.3's default text so the exercises variant keeps the caveat. Checked the other answer cells for content the exercises variant needs: added the Newton-vs-Fisher invariance statement to §4.2's text, the decoupled-decay ball to §4.3's text, and the 1/L depth result to §4.6's text.
9. Trims: Funk–Hecke reduced to one sentence (an optional remark); 4.2.B.3 replaced by a statement with a link to NB2 §2.5; the §4.0 analogy halved to one metaphor, with "Nothing in the GAN theorems…" dropped.
10. Gave the one-line reason ‖W‖_{1→∞} = max|W_ij| (vertices ±e_j of the ℓ1 ball); the Tensor Programs caveat replaced by a plain assumption sentence with the detail folded into `<details>`.
11. Natural gradient: Adam divides by √v, not v; K-FAC (Martens–Grosse 1503.05671, abstract checked) as the practical Fisher inverse, and the Kronecker-factor bridge to Shampoo (power −1/4 vs −1).
12. Moonlight convention added to 4.5.A.2's solution (verified in the PDF: Lemma 1, update RMS 1/√max(A,B); eq. (4) W ← W − η(0.2·O·√max(A,B) + λW)), contrasted with Jordan's factor and Condition 1.
13. Figure: caption says what the 2D picture cannot show (spectral vs sign descent off the diagonal; practical Muon's approximate NS); dashed lines described as level sets of ⟨g,·⟩; "Things to notice" cut to two observations.
14. No "Claim boundary." labels remain; replaced by plain prose.
15. 4.4.C.2–3 now name `np.roots` and the quadratic in σ², and give the grid to iterate.
(16, saved outputs, and the browser-link access issue are day-wide and handled by the coordinator.)

**Codex round 2** (11 items, all residual qualifications; it re-confirmed the 7.2702918 bound, the quintic numbers including the 0.001302006 threshold, the cubic, the sign-kernel constant and odd-harmonic formula, the Fisher/Hessian laws, the Shampoo identity, the Newton-in-ξ failure and the revised width demo). What I did:
1. Xie–Li assumptions: now stated (ε = 0, β1 ≤ β2 < 1, full batch, non-increasing schedule with Σηt = ∞), with the ε > 0 caveat. I verified Codex's counterexample numerically (λ = 1, L = (ε/2)(x−2)², x0 = 2 − √2: bias-corrected AdamW stays at x0 for ε = 10⁻³ and 10⁻⁸ although L′(x0) ≠ 0). Chen–Li–Liu now described as exact orthogonalization; practical Muon's radius noted as adjusted.
2. 4.3.E.1: invariance kept; approach needs Σηt = ∞, geometric only with ηt bounded below; "end up in" → "approach".
3. Modular smoothness hypotheses stated (well-normed incl. ‖x‖_RMS ≤ 1, controlled second derivatives, Lipschitz and smooth output loss; not enforced during training); the 4.6.A depth argument restricted to the well-normed region.
4. Adam's v is an EMA of squared minibatch-mean gradients, not an empirical Fisher (two-example counterexample in the text).
5. Figure caption no longer says Muon and sign descent "differ only" off the diagonal; 4.1.C.3 now says "the convergent cubic"; §4.5 says the RMS geometry motivates Muon's factor.
6. 4.5.B.4: per-layer assumption, and magnitudes with the sign written out (⟨∇L, ΔW⟩ = −η‖∇L‖_F²).
7. 4.3.D stability "iff" now under P ≻ 0, H ≻ 0, 0 ≤ β1 < 1, "converges for every start".
8. 4.3.B.1 parenthesis now distinguishes β1² > β2 from equality (√N growth) and restricts to 0 < β2 < 1.
9. 4.0: "its negative gradient points toward the data", restricted to the translation example.
10. Zero-mass convention stated.
11. Rectangular dense-sign case added to 4.5.C.3.

Rebuilt: 54 cells (20 answers, 4 code, 3 code exercises); `--run` 9.8 s total, slowest cell 5.1 s; math checks (24 blocks) and figure check (304 cases) pass.

### Round 3 (final check, 2026-10-09)
Codex only, at the coordinator's request ("a final check"); no third design critic was run. Codex confirmed all 11 round-2 items addressed (one endpoint error in item 2) and checked the new and moved material, including the depth attribution (TP-VI and Bordelon et al. α = ½; CompleteP α = 1). Fixed:
- 4.3.E.1: "iff Σηt = ∞" → "if" (a single step with ηtλ = 1 already zeroes the product).
- §4.4's stale pointer "§4.6 explains" the shape factor → §4.5.A.
- §4.3's failure-mode summary: the *mean* signed step is small, while individual steps are about 0.23η; "momentum SGD with rate η/ε" needs √v̂ ≪ ε and is approximate (applied in 4.3.D and its hint too).
- Kunstner et al. 2024: the proof is in a simplified linear softmax model in continuous time; the transformer evidence is empirical.
- §4.5: alignment and low stable rank stated as two separate assumptions.

Final state: 54 cells (20 answers, 4 code, 3 code exercises); `--run` 9.8 s total, slowest cell 4.8 s; `td/checks/td2_4_optimizer_geometry_math.py` (24 blocks, 1–4 min) and the figure check (304 cases) pass.

## For the instructor
- **Erratum in "KL divergence, geometric naturalness, and optimal transport"** (`/home/crw/Programming/LocalhostBlog/_posts/2025-07-11-KL-Divergence-Geometric Naturalness-and-Optimal-Transport.md`, line 75 and the "Heat Flow is Gradient Descent on Entropy" derivation that follows). The post writes grad_W S(μ) = +∇·(μ∇ δS/δμ), so its gradient flow ∂tμ = −grad_W S becomes ∂tμ = −∇·(μ∇(1 + log μ)) = −∇²μ, the backward heat equation. The standard Otto-calculus convention is grad_W S = −∇·(μ∇ δS/δμ), which gives ∂tμ = +Δμ. Found by Codex in round 1; I confirmed it against the post's source. The notebook links the post without comment (the coordinator asked that the correction come to you rather than to students).

## Remaining uncertainties
- No rigorous global statement for the quintic's long-run behaviour: the attracting 4-cycle and its multiplier are numerical (local attraction checked); the notebook says "numerically, generic starting points".
- The law of large numbers in 4.5.C.2 (weakly dependent terms) is stated, not proved; the exact limiting constant is quoted from a Funk–Hecke argument that I verified numerically, not proved in the notebook.
- Amari 1998 and the Kovarik / Björck–Bowie originals were not read (paywalled); the algebra was checked independently, and the Kovarik / Björck–Bowie citations were cut.
- The Cohen et al. 2022 threshold is taken from `TD_LINKS_v1.md` and the sibling NB2 §2 rather than re-read; Codex checked it against the paper in round 2.
- Browser interaction (dragging, keyboard) was not tested in a real browser session; layouts were checked by headless screenshots, and the math by the node-based check.

## Readability pass (2026-10-09)
Following `td/READABILITY_BRIEF.md`. The builder was unchanged from the reviewed version when this pass began. The mathematics, code and claims are unchanged (the code cache was reused; no code cell changed).

**Subsection openings and local notation.** Each `##` part now opens with a one- or two-sentence *The question.* and a *Notation here:* line. Each `###` exercise or code subsection opens with a plain question or context sentence and restates what it uses:
- Orientation: θ, g, η, λ, the β's, per-coordinate operations.
- 4.0 and 4.0.A: the question; W₁ as a minimum over couplings. Fixed two symbol clashes: `g_θ` for the generator (it clashed with g = ∇L) and `D` used for both the divergence and the discriminator.
- 4.1: why a norm is needed; the dual norm and d⋆ are defined before the two forms. The table footnote now defines M, F, ‖·‖₂ and ‖·‖_nuc. 4.1.A–D restate the dual norm, d⋆, the penalty and trust-region steps, the SVD and the nuclear norm. 4.1.C now says "§4.1's table".
- 4.2: the question; θ = φ(ξ), J, g_θ and g_ξ, and the Fisher defined before use. 4.2.A–D restate them, the Hessians and W_p. 4.2.D defines σ and NLL.
- 4.3: the question, and Adam's full update (m, v, bias corrections) stated before any claim. The spoiler-like paragraph listing 4.3.D's three answers (including 0.23η and η/ε) is now one sentence that names the three regimes. 4.3.D lists the regimes it previously referred to as "named above". 4.3.A, B, C and E restate Adam's update, the EMAs, the predictions under test, and η_t, λ and o_t.
- 4.4: the question; Ortho(M) = UV^⊤. G is now defined as the matrix being orthogonalized (in Muon, the momentum); it was previously undefined there. 4.4.A–D restate the iteration, φ and p.
- 4.5: the question; n_l, W_l, fan-in and fan-out, h_l, the norms, Θ in the width. δ is defined at first use. 4.5.A–C restate RMS units and Condition 1. 4.5.D states what is being checked. 4.5.E states Condition 1, the reason for the 1/fan-in rule, that ΔW₂ is cumulative over ten steps, and that the input and output layers use muP rates in both runs. The expected result is now worded as a prediction.
- 4.6: the question; d_in and d_out defined. 4.6.A restates the composition rule.
- 4.1.B solution 3: "the stated fact in the box" now quotes the KL ≈ ½δᵀFδ fact.
- 4.0's analogy paragraph: a clearer wording, and the cross-reference "(§4.4–§4.6)" for the spectral and modular norms is corrected to "(§4.1.C, §4.5–§4.6)".

**Old two-file wording.** Removed `{{VARIANT}}`. "How to use this notebook" no longer mentions `_solutions`/`_exercises` notebooks and no longer repeats the kit's run note. Cross-notebook links now use the `.html#anchor` form (td21-s5, td22-1, td22-5, s52). The builder docstring has been updated.

**MDL link.** Dinh et al.'s two-layer rescaling example has moved from "WB2 §4, exercise 4.E.i.3" to WB2 §1, 1.D.7 "Why flatness could matter, and why sharpness needs a metric", part 3. The link now points to `../notebooks/02_statistical_learning.html#wb2-section-1` and names the subsection. No other MDL references.

**Critic** (fresh Claude Opus subagent, three random subsections: 4.5.E, 4.0.A, 4.0, plus a skim of every subsection's opening). Its main findings were:
- the D and g_θ clashes in 4.0
- the wrong §4.4 cross-reference
- 4.5.E's unstated Condition 1 and missing reason
- 4.0.A's missing question
- the undefined G in 4.4 and δ in 4.5
- the 4.3 spoiler paragraph
- the vague "box" pointer
- d_in/d_out
- the duplicated how-to-use text

All were applied as described above. Its claim that the page had no MDL link was mistaken: the link is present. It also reported that the run-note link to the `.ipynb` was rewritten to the page itself, a kit issue, which was fixed in `htmlkit_td.py` by someone else during this pass. The rebuilt page links the `.ipynb` correctly.

**Checks.** `python td/build_td2_4_optimizer_geometry.py` and `--check` pass (fresh: 54 cells, 7 code cells, 23 solutions). `python td/checks/td_crosslinks.py`: 56 cross-links checked, 0 bad.
