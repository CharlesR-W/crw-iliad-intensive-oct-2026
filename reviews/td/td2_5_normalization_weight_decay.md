# Review log: NB2 §5, normalization, weight decay, and instabilities

Files: `td/build_td2_5_normalization_weight_decay.py` (builder), generated
`td/td2_5_normalization_weight_decay_{exercises,solutions}.ipynb`,
`td/checks/td2_5_normalization_weight_decay_checks.py` (closed-form checks).
No interactive figure (none assigned to this section).

## Source verification (before drafting)

Each paper narrated in the notebook was read from its full text (arXiv PDF/HTML or
proceedings PDF) by a verification subagent, which reported exact statements with
section/equation numbers. All arXiv IDs were re-checked against their arXiv titles, and
every URL in the builder returned HTTP 200 (2026-10-08).

| Item | Claim as narrated | Status |
|---|---|---|
| Li & Arora 1910.07454 (TD_LINKS 52) | SGD + WD (+ momentum γ) on a fully scale-invariant loss ≡ SGD with LR η α^{-2t-1}, no WD, α the positive root of x² − (1+γ−λη)x + γ; γ=0 gives (1−λη)^{-2t-1}. Equivalence is in function space; requires λη ≤ (1−√γ)². Last layer and BN affine frozen to make the net scale-invariant. | Verified (Thm 1.1, 2.1, 2.9, Rem. 2.10, Sec. 5). **Typo in paper:** printed Thm 2.9 says θ̃_t = α^t θ_t; its initial condition and Thm 2.1 require α^{-t}. Noted in the notebook. TD_LINKS summary is accurate in substance; needs the four qualifiers above. |
| Kosson et al. 2305.17212 v4 (TD_LINKS 53) | Table 1: SGDM rotation √(2ηλ/(1+α)) (WD inside the momentum buffer); AdamW √(2ηλ(1−β₁)/(1+β₁)), norm √(ηC/(2λ)); Adam+L² rotation depends on gradient magnitudes; ~30× spread across layers for Adam+L²; they hypothesize balanced rotation is the main benefit of AdamW over Adam+L². Angular update defined as E[angle], predicted as RMS update / equilibrium norm. Random-walk (noise-dominated) assumption. | Verified. **TD_LINKS 53 corrections:** "set by η and λ alone" is true only for plain SGD (momentum and β's enter); "Adam(W)" should be AdamW only; LayerNorm invariance is per layer, BN per neuron; the AdamW claim is a hypothesis vs Adam+L², and stronger than "part of". |
| Wan et al., NeurIPS 2021 (Spherical Motion Dynamics) | Per-step angle → √(2λη/(1+α)) w.h.p. under stated noise assumptions (Thm 3). | Verified from the NeurIPS PDF (arXiv v4 is an older, different text). |
| van Laarhoven 1706.05350 | η_eff = η/‖w‖²; L² has no regularizing effect with normalization. | Verified (Sec. 3.1, abstract). Paper's printed η_eff ≈ √(η/λ) is a typo for √(ηλ); not used. |
| Li, Lyu & Arora 2010.02916 | Intrinsic LR = ηλ; Fast Equilibrium Conjecture (function-space equilibrium in O(1/ηλ)); full-batch GD + WD oscillates. | Verified (Sec. 4, 5, Conj. 5.3). |
| Lyu, Li & Arora 2206.07085 | GD + WD on scale-invariant loss reaches η_eff ≈ 2/λ₁ (EoS) and then follows a sharpness-reduction flow, under local assumptions near a minimizer manifold. | Verified (Thm 4.4, 4.7, Assn 4.5–4.6). |
| Lobacheva et al. 2106.15739 | Periodic destabilizations with BN + WD in long constant-LR training (SGD and full-batch GD), three-phase norm/η_eff mechanism; fixing the norm removes it; periods shorten with ηλ; no sharpness measured. | Verified. |
| D'Angelo et al. 2310.04415 | "weight decay is never useful as an explicit regularizer" | **Verified as the abstract's wording (v1, v2, NeurIPS).** Body says "rarely"/"seldom". Vision mechanism: keeps CE (hence SGD noise) from vanishing; they explicitly decline the effective-LR explanation for vision. LLM: η_eff ∝ η/‖w‖ (not squared) for AdamW; bf16 stability. TD_LINKS 50's "(keeping the effective learning rate high)" for vision is wrong; the notebook avoids it. |
| Wortsman et al. 2309.14322 (TD_LINKS 54) | Attention-logit growth (growth in ‖q‖,‖k‖ not cosine; persists with pointwise attention; QK-layernorm fixes); output-logit divergence (logits become very negative, models without WD; z-loss 1e-4 log²Z fixes; WD also helps at larger scale; no mechanism given); footnote 1: study slow divergence, not fast spikes; AdamW ε too large at scale. | Verified. TD_LINKS 54 accurate. |
| Dehghani et al. 2302.05442 | Divergence at ~8B params from attention logits (>5×10⁴), near one-hot attention; QK-norm. | Verified (Sec. 2, App. B). |
| PaLM 2204.02311 | z-loss = 10⁻⁴ log² Z. | Verified (Sec. 5). |
| Molybog et al. 2304.09871 (TD_LINKS 55) | Gradients below ε freeze params; large batch → time-correlated gradients; m/√v bimodal; rare large gradient → sign-like update; seen in 65B/546B, not 7B/30B; no curvature/EoS analysis. | Verified. **TD_LINKS 55 correction:** it is the *gradients* that become time-correlated, and the mechanism is gradients/moments falling below ε, not "stale" second moments. |
| Cohen et al. 2207.14484 | Preconditioned sharpness equilibrates at (2+2β₁)/((1−β₁)η) = 38/η. | Verified. |
| Bai et al. 2506.04805 (ICML 2026) | v decouples from g² and decays; preconditioned curvature along the gradient exceeds 2/η (their normalization, equivalent to Cohen's); spike until v catches up; small β₂ gives EoS oscillation instead; 1D theory; experiments up to 187M; lowering β₂ reduces spikes. | Verified (v2). Added to the readings: it answers the spec's "which is EoS" question directly. |
| Zhang, He, Sra & Jadbabaie 1905.11881 (TD_LINKS 56) | Definition; clipped GD h = min(η_c, γη_c/‖g‖); Thm 3 upper bound; Thm 4 lower bound Ω(L₁MΔ/(ε² log M)); correlation strong in LSTM, absent in default ResNet training, present with clipping + large LR. | Verified. **TD_LINKS 56 correction:** "observed in LSTMs/CNNs" overstates the CNN case. |
| Malladi et al. 2210.05643 (TD_LINKS 57) | Not used in this section. | Not checked here. |

## Own derivations and numerical checks

All closed forms in the notebook are checked by
`td/checks/td2_5_normalization_weight_decay_checks.py` (sympy + numpy): the SGD+WD fixed
point, cos θ* = 1 − ηλ, relaxation factor 2(1−ηλ)² − 1, equilibrium effective LR, the
ηλ-only rescaling, the AdamW (β₁ = 0) equilibrium, the Li–Arora exponent, the 38/η
threshold and v-decay times, the sign-descent logit-drift formulas and SGD conservation,
the QK-layernorm bound, the (L₀,L₁) descent-lemma constant e − 2 (and κ = 2(e − 2) > 1),
(1,1)-smoothness of cosh and the GD overshoot condition.

New results derived for the notebook (not quoted from a paper):
- cos θ_k = (1−ηλ) r_k / r_{k+1} exactly for SGD on a scale-invariant loss, hence
  E log cos θ = log(1−ηλ) in any stationary state. Checked numerically to 3–4 digits.
- The (L₀,L₁) descent lemma with constant (e − 2), via Grönwall; this is a self-contained
  variant of Zhang et al.'s Lemma 9 / Zhang et al. 2020 Cor. A.4, not their constants.
- The sign-descent single-step logit drift Δz̄ = −η(‖h‖₁+1)(K−2)/K and
  Δ log Z = −η(‖h‖₁+1)(1−2p_y).

The code cells reproduce, on CPU, (i) the η-independence of the late effective LR
without WD, (ii) the equilibrium angle for five (η, λ, σ) settings, (iii) Li–Arora and
the ηλ-only statement to roundoff, (iv) the AdamW predictions (and their failure when
signal matters), (v) the periodic WD-driven EoS cycle in the x²/(x²+y²) toy, (vi) Bai et
al.'s two-layer contrast (β₂ = 0.999 spikes at preconditioned sharpness 38; β₂ = 0.9
hovers), (vii) the output-logit drift under Adam vs conservation under SGD and pinning by
z-loss, (viii) clipped vs best-fixed-step GD on cosh. `--run`: about 50 s total on a
heavily loaded 12-core machine; slowest cell about 18 s.

## Critique rounds

### Round 1 (2026-10-08/09)

**Technical critic: Codex (GPT), high effort.** Eleven findings; all checked and accepted.
1. *Cosh example (serious).* The 1D "best fixed step is exponentially slow" claim was false:
   h = x₀/sinh x₀ reaches the minimum in one step. Replaced by L(x,y) = cosh x + y²/4, where
   every convergent fixed step obeys h ≤ 2x₀/sinh x₀ and then the y coordinate needs
   ≥ log(1/ε)/(−log(1 − x₀/sinh x₀)) steps (a true lower bound, printed beside the grid
   result); clipped GD takes ≈ x₀ + 26. Part 4 now explains why the second coordinate
   is needed. The (L₀,L₁) answer now calls the lemma's step condition sufficient, not
   necessary. Verified by running.
2. *Stationarity identity tails.* E log cos θ = log(1−ηλ) is exact, but E θ² ≈ 2ηλ needs
   small angles on essentially all steps (Codex's counterexample with rare huge gradients
   gives RMS 0.030 vs 0.045). Text now states E θ² ≤ −2 log(1−ηλ) and the condition.
3. *AdamW angle* needed a small-radial-component and norm-concentration assumption; added,
   with the exact tan θ formula.
4. *ηλ-only exercise* overclaimed forgetting of the initial norm; now separates the exact
   matched-initialization statement, local constant-G relaxation, and the (unproved)
   Fast Equilibrium Conjecture.
5. *Bai et al.'s directional quantity* is an empirical predictor, not a necessary
   condition; reworded.
6. *Stability conditions* restricted to positive-curvature modes / H ⪰ 0.
7. *Li–Arora part 3* dropped the factor 1−ηλ and assumed the WD norm shrinks; fixed
   (η̃/r̃² = η/((1−ηλ)r²) exactly).
8. *ηλ < 1* now stated where the prefactor's positivity is used.
9. *Classifier commentary* claimed conservation of the mean logit; now says the
   parameter sums are conserved and the mean logit happens to stay put.
10. *LayerNorm ε criterion* now uses the centered variance (mean square for RMSNorm).
11. *Spike-cycle period:* the growth delay scales like √(log(A/A₀)/(2ηλ)), not 1/ηλ; fixed.

**Design and pedagogy critic: Claude Opus.** (Its report reached the coordinating
session after this session was interrupted, and was relayed in condensed form.) Ten
findings, all accepted:
1. §5.3 restructured: definition (with Wortsman's slow-divergence scope folded in), then
   each mechanism's 4–8-line paragraph directly above its exercise/demo, Molybog cut to
   one clause, Exercise 5.3.5 deleted and replaced by a summary table (mechanism | EoS? |
   spike or slow divergence | remedy | where).
2. §5.4: cosh example first (now 5.4.1), descent lemma optional (5.4.2), quoted complexity
   bounds cut. Added "Clipping under Adam": with β₁ = 0, |Δwᵢ| ≤ η/√(1−β₂), so under Adam
   clipping acts through the optimizer state (one huge gradient inflates v̂ for
   ~log F/(1−β₂) steps); explicitly flagged as a plausible role, not an established
   theory. The bound is checked in the checks script.
3. Hints added to 5.2.1, 5.2.3, 5.2.4, 5.2.5, 5.3.1, 5.3.2, 5.3.3, 5.3.4, 5.4.1; the old
   5.2.5.4 (Adam + L²) became prose in the AdamW summary.
4. Pitch: the rotation claim is now "under SGD at an equilibrated norm, RMS √(2ηλ); AdamW
   approximately, when noise-dominated". Added short (30–45 min) and full (2–3 h) paths.
5. Logit demo: output bias initialized to −log K (log Z ≈ 0 at init), log Z = 0 line
   added. Now Adam drives log Z to ≈ −3 in 2000 steps, z-loss holds it at ≈ −0.5, SGD's
   mean logit stays at −3.46. Text rewritten to match the run.
6. Source-audit asides cut (D'Angelo "seldom", Kosson define-vs-predict); the Li–Arora
   Theorem 2.9 erratum moved to that reading.
7. AdamW 4–5 line summary added to the main text (r*, rotation, η/‖w‖, relaxation time,
   EMA-of-updates unrolling).
8. Transformers: added Kosson et al.'s GPT-2-124M measurements (holds very well with
   weight standardization, "often close" without; checked against the paper text, §5 and
   Appx J).
9. "four identities" → five; G_k and the λ (weight decay) vs eigenvalue notation defined
   in the notation box.
10. Readings: blurbs cut to ≤ 2 sentences, Molybog reading and Zhang–Jin link removed.

Also: pipes inside markdown tables replaced by \lVert … \rVert.
After round 1: 49 cells (13 answer, 8 code, 2 code-answer); `--run` 20 s total.
- Cross-reference: Adam's 38/η threshold is NB2 §2.5 in the current td2_2 (checked its headings 2026-10-09); fixed in the 'What you need' box.

### Round 2 (2026-10-09)

**Design and pedagogy critic: Claude Opus (fresh).** (Report relayed by the coordinating
session.) Verdict: in good shape; nine ranked fixes. Applied:
1. Adam β₂ demo commentary rewritten to match the figure: β₂ = 0.9 overshoots once (≈45),
   then sits a little below 38; its loss ends near 1e-3 vs ≈2e-5 for β₂ = 0.999 (with
   spikes). "Lowering β₂ trades spikes for slower progress (in this run)"; table remedy
   cell updated.
2. Clipping commentary: about one step per unit of x₀ (≈ x₀ + 26), not 2.
3. Exercise 5.3.2 rewritten on the 1D quadratic with β₁ = 0: threshold P = ηh/√v < 2;
   amplitude growth 1 + 2δ; v_k/v_{k−1} = β₂ + (1−β₂)c_k; correcting within m steps needs
   c ≳ 1 + 2δ/(m(1−β₂)) (≈ 21 at β₂ = 0.999, ≈ 1.2 at 0.9 for δ = 0.1, m = 10). Checked
   against a 1D simulation: peak c in the first overshoot is 20–30 for β₂ ∈ [0.99, 0.9995].
   **Not done:** the suggested 1D β₂-sweep cell. In 1D, all three β₂ values end in a
   stable period-2 orbit at P = 2 after one transient overshoot, and the first-overshoot
   size (in g²/v) is roughly independent of β₂ (δ itself shrinks like √(1−β₂)), so the
   cell would not show "spike height grows with β₂" honestly; the two-layer cell shows
   the β₂ contrast. "Curvature along the gradient" is now defined (η gᵀP⁻¹Hg/‖g‖²).
4. Preconditioned sharpness defined once as S_P = λ_max(P⁻¹H), threshold ηS_P < 38, used
   throughout (box, mechanism 2, plot label, table).
5. Mechanism 3: added that the divergence is not derived here; 5.3.3.1 no longer points to
   5.2.1 (W_Q, W_K are not scale-invariant without QK-LN).
6. Pitch AdamW sentence replaced by the critic's wording.
7. Short path points to the code cells after 5.1.1 and 5.2.2, and to the AdamW summary.
8. AdamW demo now prints the mean cosine between the update and w; commentary explains
   the failure at σ = 0.1 by the first-order radial term (cos ≈ −0.04 at β₁ = 0, radial
   term ≈ 0.02/step vs η²C ≈ 0.006).
9. Li–Arora α is the larger root; Δ, ε, M defined in 5.4.2.4; 2D-toy no-noise remark now
   says x decays to the smallest float, loss reads zero, ‖w‖ keeps shrinking (≈ 0.005).

**Technical critic: Codex, high effort.** Eleven findings, all accepted after checking:
1. "Clipping cannot be about limiting the step length" was wrong (a gradient of 1000
   with v = 1, β₂ = 0.999 gives a 31.6η step vs η clipped; verified). Now: clipping is not
   needed to keep Adam's step finite, but changes the current step and protects the state.
2. Stationarity identity: "not heavy-tailed" replaced by "per-step rotations uniformly
   small (large-angle contributions negligible)"; Codex's bounded counterexample
   (G ∈ {0, 1}) noted in the text.
3. AdamW summary bullet: the angle now carries √((1−β₁)/(1+β₁)).
4. Adam demo: not every λ_max crossing spikes (crossing at step ≈1000 does not); text
   now says so, citing Bai et al.'s directional predictor.
5. Exercise 5.2.3 statement no longer asserts forgetting at rate 4ηλ.
6. D'Angelo reproduction claim restricted to float32 (bf16 runs diverge).
7. Frozen-Adam equivalence stated "late in training (first-moment bias correction ≈ 1)".
8. Table row 4: "the drift is flat; what finally diverges, and whether that is EoS, is not
   established".
9. Clipped-GD complexity now stated with Δ = L(x₀) − inf L and the regime ε ≤ L₀/L₁
   (≤ 4κL₀Δ/ε² steps), derived from the notebook's own decrease bound.
10. "2 per unit of x₀" typo (same as pedagogy item 2).
11. Sign-descent mean-logit drift strictly negative only for K > 2.

Also: cross-reference to Adam's threshold changed to NB2 §2.5 (see above).
After round 2: 49 cells (13 answer, 8 code, 2 code-answer); `--run` 22.5 s total, slowest
cell 6.9 s; `--check` passes; checks script passes.

### Round 3 (2026-10-09)

**Design and pedagogy critic: Claude Opus (fresh).** (Report relayed by the coordinating
session.) Verdict: in good shape. Applied:
1. Structural bug: the "With momentum" paragraph (SGDM √(2ηλ/(1+α)), bridge to the next
   cell) and the D'Angelo float32/bf16 claim lived inside an answer cell, so they were
   missing from the exercises variant. Moved to a markdown cell before the AdamW demo,
   de-duplicated against the AdamW summary. The noisy r*⁴ formula, used by the next cell's
   "r* pred" column, is now part 5 of Exercise 5.2.2's statement.
2. "No regularizing effect on the function at all" → no *explicit* regularizing effect;
   any effect on generalization goes through η_eff (e.g. the SGD noise scale, NB2 §3).
   Pitch matched.
3–4. New "What this means for setting η and λ" bullets at the end of 5.2 (only ηλ matters;
   √(2ηλ) is the real directional step; relaxation 1/(4ηλ) SGD, 1/(2ηλ) AdamW vs schedule;
   periodic instability with little noise; non-scale-invariant parameters regularized as
   usual, hence the common gain/bias exemption; transformers). The old "What survives in
   practice" paragraph, which quoted results before deriving them, was merged into it.
5. 2D toy commentary: predicted first crossing ≈ 1260 steps (r₀² = 1.25, r_c² ≈ 0.1); loss
   rises at ≈ step 1340; period ≈ 130 steps; no-noise ‖w‖ ≈ 0.003. All measured.
6. Equilibrium cell: four of five settings share ηλ = 1e-3 (stated as the ηλ-only
   evidence); "to about 0.2%". Logit demo: "All three runs". Clipping: advantage from
   x₀ ≳ 3 (fixed step wins at 1.7, 20 vs 28).
7. Exercise 5.2.1 links Arora, Li & Lyu (ICLR 2019), arXiv 1812.03981 (title and abstract
   checked: fixed LR on scale-invariant weights still reaches a stationary point at rate
   T^{-1/2}).
8. Molybog clause now explains the mechanism in one sentence.
9. Minor: 5.2.3 parenthetical moved into the hint; "(holding h fixed)" in 5.3.4.2; strict
   "<" in 5.4.1.2 (equality gives the period-2 orbit ±x₀); "391 steps (CIFAR-10 at batch
   128)"; 5.2.5.2 rephrased; AdamW parenthetical cut from the pitch.
10. Scope sentence added to the pitch (pre-LN vs post-LN stability and warmup are out of
    scope), per the coordinating session.

After these: 50 cells (13 answer, 8 code, 2 code-answer); `--run` 25.2 s total, slowest
cell 7.6 s; `--check` and the checks script pass.

**Technical critic round 3: Codex, high effort** (reviewed the round-2 version). Seven
findings, all accepted:
1. Bai et al.'s Theorem 5.1 is false as written (η = θ₀ = v₀ = 1 lands on the minimum in
   one step, no spike, yet satisfies its assumptions). The reading now says "analyzed on a
   1D quadratic", not "proved". (Codex's counterexample checked by hand.)
2. Exercise 5.2.1's transient estimate needs a small first angular step; with
   z = r²/(ηG), z_{k+1} = z_k + 1/z_k exactly, and z₀ ≪ 1 gives a transient ~1/(2z₀²). Added.
3. AdamW momentum formulas need ηλ ≪ 1 − β₁ and v̂ tracking E g² (Kosson §3.2); stated in
   the summary and the momentum cell.
4. The AdamW angle expectation needs uniformly small relative updates (rare large updates
   make E tan²θ blow up even at constant norm); added.
5. Zhang et al.'s lower bound is "for each fixed step there is a hard function", not "one
   function defeats every step"; fixed, and noted that Exercise 5.4.1 is the stronger,
   single-function statement.
6. v̂ decays by β₂ per step only late in training and when later gradients are small;
   stated.
7. No-noise ‖w‖ ≈ 0.003 (already fixed in the round-3 pedagogy pass).
Also specified the setup of the "peak c ≈ 20–30" 1D simulation.

**Final focused technical check: Codex** (on the parts changed in round 3). Four
findings, all applied: AdamW averaging memory ≈ 1/(ηλ) kept separate from the norm
relaxation time 1/(2ηλ); the practical bullets now carry the qualifications of the
exercises (matched r₀²/η; RMS angle under uniformly small rotations; local constant-G
relaxation; forgetting of initialization conjectured); "all of these rely on the
random-walk assumption" narrowed to the equilibrium formulas; the gain/bias exemption is
stated as common practice, not as a consequence of the derivations. Clipping crossover
reworded to "clear by x₀ ≈ 3" (Codex finds the grid crossover near 2.2). Codex reproduced
the toy numbers (predicted crossing 1263, loss > 1e-6 at 1339, period 129, no-noise norm
0.0029) and the 0.18% stationarity discrepancy.

## Final state (2026-10-09)

50 cells: 13 worked-solution answer cells, 8 code cells, 2 code exercises. `--check`
passes; `--run` 26 s total on a loaded machine (slowest cell 8 s); checks script passes.

## Remaining uncertainties

- Kosson et al.'s momentum formulas (SGDM, AdamW β₁ > 0) and the AdamW equilibrium are
  quoted, not derived, and rest on a random-walk model; our toy confirms them only when
  noise dominates.
- Wan et al.: the cube-root error term in Thm 3 was read through PDF extraction; not used
  quantitatively.
- Bai et al.'s Theorem 5.1 appears false as written (Codex counterexample); we cite only
  its analysis and experiments. Their large-model experiments were not reproduced.
- "We know of no analysis of clipped Adam comparable to the GD story" is a statement about
  our knowledge, not a literature search.
- The Wortsman/PaLM/Dehghani claims were checked against arXiv versions, not camera-ready
  versions.
- Malladi et al. 2210.05643 (TD_LINKS 57) is not used in this section and was not checked.

## Readability pass (2026-10-09)

Following `td/READABILITY_BRIEF.md`. The builder had no partial edits from the stopped
earlier attempt (its mtime predated the last build). Prose only: no code cell, claim or
derivation changed, and the build reused the cached outputs. 50 cells, as before.

**First pass (own reading).**
- Each `##` part and each `###` subsection now opens with its question in one or two
  plain sentences (5.1, 5.1.2, 5.2, 5.2.1–5.2.5, AdamW, the practical summary, 5.3, the
  four mechanisms, 5.3.1–5.3.4, the summary table, 5.4, 5.4.1, 5.4.2, clipping under Adam).
- Notation is restated locally where a subsection used symbols it did not define: a
  "Notation here" block in 5.2, AdamW and Mechanisms 1–2, and inline definitions elsewhere
  (sharpness S and direction u in 5.1.1; r_k, η, λ in 5.1.2; the norm recursion, G_k, θ_k
  in 5.2.1–5.2.3; U_k, r_k in 5.2.5; the GD and Adam thresholds in the 5.3 intro; W_Q, W_K,
  d_h in Mechanism 3; log Z and CE in Mechanism 4; p, e_y, z̄ in 5.3.4; the clipped-GD step
  in 5.4.1–5.4.2; Adam's v, v̂ in clipping under Adam; symbol key above the summary table).
- Spoilers: the 5.2 intro now opens with question and notation, and gives the equilibrium
  angle in words before the formula. The AdamW "In summary" bullets now follow a notation
  block and a plain-words summary, and are labelled "at equilibrium / at any norm r /
  approach". The pitch now states the SGD equilibrium in words (the √(2ηλ) formula moved
  out to 5.2).
- De-duplication: the AdamW momentum paragraph no longer repeats r_* and the rotation
  formula from the summary; the "intrinsic learning rate" sentence appears once in the 5.2
  intro, and 5.2.3 asks the question instead; one "weight decay is a schedule" sentence
  cut from the 5.2 intro.
- The near-empty "### The effective learning rate" heading was merged into Exercise 5.1.2.
- Cross-links: "NB2 §2", "NB2 §2.5" and "NB2 §3" now link to td2_2 (top and `#td22-5`) and
  td2_3.
- Old-format wording: none found in the text (the `{{VARIANT}}` label is dropped by the kit).

**MDL links.** One: WB6 Exercise 10.1 at `../notebooks/06_mysteries_myriad.html#wb6-section-10`.
Checked against the current page: §10 "Spectra and curvature" still holds Exercise 10.1
(scale invariance under batch norm), parts (a)–(c) = our items 1–4 and part (d) = the
λ=0 case of Exercise 5.1.2. Kept, with the section name added and part (d) noted.

**Readability critic (Claude Opus, fresh).** No math errors, no old-format wording. Its
header run-note link finding was a kit bug, fixed by the coordinator in `htmlkit_td.py`.
Applied all the others:
- Symbol clashes: momentum is now γ in both 5.2.4 and the AdamW momentum paragraph (was α
  in the latter); the scalar in 5.3.2 is now q_k (P stays Adam's preconditioner); the
  z-loss coefficient is α_z (c stays the logit shift); 5.3.3.2 says "largest attention
  probability in the row, p_{i1}".
- A key for references like "5.1.2.3" added to the notation box.
- 5.1.1 answer 5 now defines r_k, G_k; 5.3.4 defines e_y and z̄; the code-exercise prompt
  says where `sgd_scale_invariant` is defined; `el = ηλ` explained for the printed table;
  r_0² = 1.25 and r_c² ≈ η stated next to the 1260-step estimate.
- Openings for 5.3.1 and 5.3.4 rewritten as questions; Mechanism 4 now says the drift is
  not itself an EoS crossing (zero-curvature direction), matching the summary table.
- Parts table: "Adam spikes when S_P reaches 38/η". Second practical bullet split in two.
  Wortsman et al. linked at first mention in the 5.3 intro and in Mechanisms 3 and 4.
- Added one sentence reading the 5.2.1 figure (all three η collapse onto 1/(G√2k) after
  an η-dependent transient; checked against the rendered figure).

`python td/build_td2_5_normalization_weight_decay.py`, `--check` and
`python td/checks/td_crosslinks.py` (60 links, 0 bad) all pass.
