# Review log: td2_2_edge_of_stability (NB2 §2, the edge of stability)

Files: `td/build_td2_2_edge_of_stability.py` → `td/td2_2_edge_of_stability_{exercises,solutions}.ipynb`; figure `td/assets/interactive/eos_toy_and_catapult.html`; checks `td/checks/td2_2_edge_of_stability_sympy.py`, `td/assets/interactive/checks/eos_toy_and_catapult_check.py`.

## Drafting decisions

- **Toy model.** The spec's toy $L=\tfrac12S(y)x^2+\ell(y)$ is used for all derivations. For code and the figure I needed a concrete instance that behaves well across a slider range. Findings from scratch experiments:
  - Damian et al.'s linear model ($S$, $\ell$ linear; also Kalra–Barkeshli's warmup model) has a period-2 orbit that is *weakly unstable* in discrete time (two-step multiplier modulus $1+\eta^2x^{*2}$); iterates circle it and, from small entry amplitudes, often diverge within a few thousand steps. Derived the general condition: the orbit attracts iff $\ell''(y^*)>2\eta|\ell'(y^*)|$ (leading order; full Jury conditions in the sympy check). A quadratic $\ell$ satisfies it only within a factor 1.25 of its minimum; $\ell\propto y^{-p}$ with $p>4$ satisfies it at every $\eta$. Chose $S(y)=y$, $\ell=c\,y^{-5}$, $c=62.5$. This became Exercise 2.2(d) (optional), together with the exact long-run averaging identities for the linear model.
  - Without a noise floor, $x$ decays to ~1e-100 during sharpening and the first blowup is catastrophic (divergence in most warmup runs). A deterministic tilt does not help (it only shifts the fixed point). Added Gaussian noise of s.d. 1e-3 on $x$; explained in the text as a stand-in for re-seeding of the unstable direction.
  - Figure slider ranges chosen from a scan: η ∈ [0.25, 0.55], y₀ ∈ [2, 9]; outside, the toy diverges (displayed as such).
- **EoSS demo dropped.** Tried measuring batch sharpness, ratio-of-expectations and λ_max under SGD on the notebook MLP (b ∈ {8,…,128}, η ∈ {0.1, 0.3, 0.6}, ≤ 4000 steps). Within the time budget the runs did not reproduce the paper's picture cleanly (small batches still sharpening, or BS below 2/η once the loss hit zero; b = 8 diverged at η = 0.6). Kept §2.5 as derivation + reading rather than show a misleading demo.
- **Central flow vs GD** uses a 3-D extension of the toy; agreement is close after the first excursion (printed in the notebook).
- **Rod Flow** equations converted from step-counting time to $t=k\eta$ (checked on the 1-D quadratic in the sympy script).

## Verification done before critique

- `td/checks/td2_2_edge_of_stability_sympy.py` (ALL PASS): catapult maps (symbolic, n = 3 explicit vectors with symbolic n), λ = |∇f|² at f = 0, toy maps, period-2 orbit, σ² = period-2 x², two-step det/trace, undamped |μ| = 1 + k², heavy-ball Jury conditions, EMA = heavy ball with η(1−β), 38 at β = 0.9, σ² from dS/dt = 0, projection identity, L + σ²S/2 = L + σ²/η, batch Rayleigh criterion, series of log(1−ηλ)/η, Rod Flow Σ and w̄ equations on a quadratic, river-valley stationary hill loss.
- Catapult maps vs explicit GD on (u, v): agreement to 1e-15 (notebook cell and check script).
- River-valley formula: Monte Carlo 0.02940 vs formula 0.02941.
- Figure: JS simulation run under Node against a Python port with the same PRNG: identical to ≤ 1e-12 for 5 toy and 18 catapult parameter sets; the Python port equals the notebook's `toy_run` (noise-free) exactly; stated physics checked (S → 2/η, ⟨x²⟩ = σ² at η = 0.3–0.5; sharp start diverges without warmup and survives a T = 1600 warmup; catapult phases).
- Figure rendered headless at 1100 px and 390 px width (light mode); legends/labels fixed after the first render.
- Paper claims: three subagents read the sources (arXiv HTML/PDF, PMLR) and reported quotes and equation numbers; the text was written against those reports. Key corrections taken from them: the central-flow `max(0,·)` is only in App. A.2.2 (eq. 72), main text eq. 13/14 (v2 numbering); Damian et al.'s toy is 3-D (App. B); Andreyev–Beneventano v1 used E λ_max(H_B), v2–v4 *defined* the ratio of expectations while measuring expectation-of-ratio, v5 fixed it; the ratio of expectations is their "Modified Batch Sharpness" (App. J.3), which stabilizes above 2/η; their Prop. 4 (Jensen/Lyapunov) is one-sided; the SGD self-stabilization mechanism is left open (§3.2); Wen et al.'s oscillation is SGD noise, not EoS; Rod Flow's network experiments start at EoS; principal flow does not capture stabilization.
- Build: `python td/build_td2_2_edge_of_stability.py`, `--check`, `--run` pass (≈ 20–30 s total, slowest cell ≈ 10–15 s, the MLP training).

## Critique rounds

### Round 1

**Design critic (Claude Opus subagent), 12 items.** Accepted and changed:
- Paper narration trimmed: the Andreyev–Beneventano version history is one sentence; Liao et al. reduced to a neutral reading entry; the repeated "Claim boundaries" labels removed (limitations kept as clauses); equation-number bookkeeping moved to readings; Kalra–Barkeshli paragraph shortened.
- Central-flow code was nearly guaranteed to agree (∇S constant). Replaced with $S=y(1+\kappa z)$ so ∇S rotates and the flow slides along a curved constraint surface; agreement still close (few % in y, z; ~5% early, ~20% late for x² and loss) and the text now says the toy checks the derivation, not the network claim.
- §2.1 text now describes the overshoot/catapult entry visible in the plot; the printout averages only after each run reaches 2/η.
- Exercise 2.8(a): reports the y₀ = 5.5 case where short warmups *raise* the peak loss (extra sharpening during warmup), with printed max S.
- Added that progressive sharpening is assumed, not explained (ask-your-agent pointer).
- Reordered: Rod/principal flow now 2.4 (after central flow), Adam 2.5, EoSS 2.6; RMSProp central-flow paragraph moved to 2.5; figure link split into §2.2 (panel 1) and §2.7 (panel 2); "check script" phrase removed from the notebook.
- Pitch: removed the term list, added a short path (2.1 → 2.2 → 2.3), replaced "everything solvable on paper" with "derivations on paper; that networks behave this way is empirical"; core labels now "core, 2.x" with one overall recommendation.
- §2.8 lead no longer says WSD has an EoS explanation.
- Exercise 2.2(d) now shows the product $J_-J_+$ and is split into (d) Jacobian, (e) linear-model averages, (f) the $y^{-5}$ check; Exercise 2.1's proof simplified (monotone flow map vs decreasing GD map) and part 4 folded into part 3.
- Toy demo: lower panel is now |x_k| on a log scale for one η (shows the four stages); title says "settles or hovers".
- Minor: quote grammar, β overloading (Damian's √β removed; Liao's symbols removed), opaque ∇L·u aside cut, complementarity problem explained in one sentence, Adam bias-correction parenthetical cut.

**Technical critic (Codex, GPT), 22 items.** All checked; accepted:
- (A1) Exercise 2.5(c)/(d) overclaimed: the Cauchy–Schwarz + Jensen bound controls same-batch amplification, not an instability of the iterate. Codex's counterexample verified numerically (BS = 2.093 > 2/η yet E‖θ′‖² = 0.845‖θ‖²). Now part (d) of the exercise (renumbered 2.6) with the counterexample; text says the paper's Prop. 4 builds on this factor under a *sustained* margin.
- (A2) Power-law toy orbit is locally attracting only for Q < 2, i.e. η < 0.82 (exact Jury). Verified numerically (two-step |μ| = 0.985 at η = 0.82, 1.88 at 0.83). Stated exactly; "every η" removed.
- (A3) Quadratic statements now assume positive-definite H; negative-curvature caveat added.
- (A4) Adam feedback explanation now uses the preconditioned mode $P^{-1/2}q$, not the raw $u$.
- (A5) Rod Flow closure stated (δ from the top eigenpair of Σ, approximate rank one); "no Hessian eigensolver".
- (A6) Exercise 2.4(c): δ = 0 is not invariant away from a minimum (dΣ/dt = (η/2)∇L∇Lᵀ); fixed and linked to the paper's §5.1 example.
- (A7) Wen et al.'s theorems are discrete small-step results against a continuous reference flow; fixed.
- (A8–A11) Figure: x series no longer downsampled (stride 2 aliased the period-2 oscillation); toy status now checks both mean S and mean x² vs σ²; catapult status distinguishes non-converged / no-spike boundary cases; y labelled as curvature along x (equal to sharpness for y ≳ 2.6) and y₀ slider minimum raised to 3.
- (B12) Averaging statements corrected: ⟨log|1−ηS|⟩ = 0 gives ⟨ηS⟩ = 2 + ½⟨ε²⟩ + …, i.e. slightly *above* 2 (now used to explain "at or just above 2/η"); the exact ⟨x²⟩ identity is restricted to the linear model.
- (B13) Adam's 38/η qualified to full batch; minibatch and very-small-η exceptions noted.
- (B14) Liao formula: now described only qualitatively with its assumptions, and its validation status stated neutrally (scaling tests, App. G.4).
- (B15) Catapult phases stated for the wide limit away from boundaries, with the a₀ > 2 + b₀ condition and boundary caveats; notebook plot marks non-converged points.
- (B16) WSD solution now includes the transient relaxation of the hill variance; the EoS analogy notes the oscillation returns once sharpening reaches the new edge.
- (B17) Adam warmup: Liu et al. (RAdam, arXiv 1908.03265, verified) cited as the earlier explanation, with Kalra–Barkeshli's stated disagreement (quote verified via the arXiv HTML).
- (C18–C22) bias-correction formula removed; seed-dependence of the short-warmup outcome stated; the 1e-100 claim replaced by measured values (1e-139 at η = 0.3, 1e-14 at η = 0.4); smoothing uses valid windows; power-iteration assumption stated.
Codex found no errors in the quadratic modified flow, Jury expressions, central-flow conversion, EMA/heavy-ball conversion, Rod time conversion, catapult maps or the river-valley formula.

### Round 2

**Technical critic (Codex).** The first attempt exited partway (exit code 1, after noting that the batch-sharpness text still drew an instability conclusion); the retry completed. 12 items, all checked and accepted:
- (A1) Notebooks were stale relative to a builder edit (central-flow edge tolerance); regenerated. I had also found the integrator chattering (110 of 80,000 steps dropped off the edge by round-off) and added the tolerance.
- (A2) "Locally attracting iff" → "linearly (exponentially) attracting iff"; the Q = 2 boundary is noted as still attracting, algebraically (Codex's centre-manifold computation, not independently re-derived; stated softly).
- (A3) Damian et al.'s projected set includes ∇L·u = 0; restored in §2.2, and §2.3(c) now says the central flow is related to, not the continuous version of, their projected descent (Cohen et al. §3.2 drop that condition).
- (A4, A5) Figure status messages: toy fallback now distinguishes below-edge from above-edge-but-not-yet-oscillating; catapult no-spike message no longer claims "just above 2"; lazy branch keyed on ηλ₀ < 2.
- (B6) EoSS: Prop. 4 is now described precisely (expected product of same-batch amplification factors under a sustained margin), with "cannot stay above 2/η" and the trajectory-gradient conclusion removed.
- (B7, C8) Central-flow agreement now reported per quantity, with consistent window averaging for y and z.
- (C9) "stationary point" instead of "minimum" in Exercise 2.4(c). (C10) multi-direction drift written in full. (C11) "overshoots" replaced by "increases the quadratic batch loss" (overshoot of the minimizer starts at ηS_B > 1). (C12) figure growth-time readout relabelled as an early-growth estimate.

**Design critic (Claude Opus).** Its report was relayed by the coordinating session after an overnight interruption. 10 items; accepted:
1. Rod Flow: Exercise 2.4(a) now derives the exact identity Σ_{k+1} − Σ_k = (η²/4)[∇L₊∇L₊ᵀ + ∇L₋∇L₋ᵀ] − 2Σ_k (checked numerically on a nonlinear 2-D loss in the check script) and how dividing by η gives the ODE; experiment details cut to one sentence. (Integrating Rod Flow on the 3-D toy was optional; not done.)
2. Central flow: late comparisons now use 1000-step windows; agreement 1–3% late (y, z, x², loss), ~10% just after the first excursion. Text says so.
3. New demos on the §2.1 MLP: full-batch Adam (η = 1e-3, 3e-3): η·λ_max(P⁻¹H) = 37.3, 37.5 vs threshold 38 while η·S ≈ 0.03–0.05 (4 s). SGD at η = 0.3, b = 16, 64 (13 s): batch sharpness 6.2 ± 0.5 and 6.9 ± 0.5 vs 2/η = 6.67; λ_max 4.0 vs 6.6; ratio of expectations 7.1 and 7.05 (above 2/η). Described as an illustration with noisy estimates. (Earlier, the drafting-stage attempt had been abandoned because runs were too slow; the slowness turned out to be torch thread contention, fixed by `set_num_threads(1)`.)
4. Quotes and version history moved out of the body (paraphrased; EoSS version history now in the reading entry).
5. §2.2 split: core Exercise 2.2 (a–c), then the toy demo, then optional Exercise 2.2′ (d–f).
6. Lower-panel caption rewritten to match the plot (noise floor, blowup, overshoot, decaying oscillation).
7. Progressive sharpening now links `td2_6_curvature` §6.2 before the ask-your-agent pointer.
8. "Resembles a catapult" used consistently.
9. The η = 0.05 MLP run extended to t = 250; all three runs now hover tightly (second-half ranges within ±3% of 2/η).
10. 2.2′(f): the invariant threshold is p > 3 (ℓ̃ ∝ S^{−p}: ratio (p+1)/S > 4/S), not p > 4 — my error; the code's p = 5 satisfies both. Warmup mechanism attributed to Kalra–Barkeshli; "that analogy is ours" → "neither paper makes this connection"; the SDP clause dropped.

### Round 3

**Technical critic (Codex).** Confirmed the round-2 fixes (Σ identity and time conversion, exercise split and cross-references, p > 3, exact Jury condition, long-window central-flow comparison, the counterexample and expected-product reading of Prop. 4, catapult status logic) and reproduced the cited numbers. Found:
- (A) The Adam demo's warm-started power iteration (8 iterations) under-estimated λ_max(P⁻¹H) by ~1 (Codex compared with Lanczos on the same trajectory). Replaced by scipy Lanczos (`eigsh`) on P^{-1/2}HP^{-1/2}; results now η·λ_max(P⁻¹H) = 37.7 [36.4, 38.5] and 37.4 [36.0, 38.6]; text says "hovers near 38".
- (A) The figure's toy fallback status could still misdescribe the run (e.g. η = 0.27, y₀ = 6, noise off: the kick has happened but the 1000-step mean hides it). Replaced the dynamical-phase guesses with a neutral readout of the window averages.
- (C) Central-flow caption: now says only variance and loss are smoothed in the plots.

**Design critic (Claude Opus).** Ranked list; accepted:
1. The SGD demo cannot distinguish batch sharpness from the ratio of expectations (6.22 vs 7.14 at b = 16 straddle 2/η; 6.88 vs 7.05 at b = 64). Text now claims only the λ_max gap (4.0 vs 6.6) and points to the paper's App. J for the distinction.
2. Paper retelling trimmed further: principal flow cut to two sentences; Rod Flow cost/accuracy/follow-up sentences removed (follow-up kept in the reading); EoSS observation and follow-ups cut to one sentence each; warmup paragraph cut to the mechanism the exercise uses; §2.1 observation shortened.
3. §2.5's mechanism paragraph no longer gives away Exercise 2.5(c); the RMSProp claim moved into the central-flows reading entry; the Adam demo text now says the edge is reached here because v and P shrink as the loss goes to zero, not by progressive sharpening.
4. Catapult: new paragraph on the right-hand plot (over-flattening to ηλ ≪ 2, lobes, slow convergence of runs landing near a ≈ 0 because a ≥ 2√b, from ‖u‖² + ‖v‖² ≥ 2|uᵀv|); the incorrect "close to either boundary convergence can take far longer" sentence removed.
5. Warmup y₀ = 5.5 explanation rewritten around two printed quantities, S at the first crossing and max ηS after it (2.24, 2.29, 2.22, 2.12 for T = 0, 100, 400, 1600), instead of "max S".
6. §2.2 now says in one clause why ℓ = c y⁻⁵ rather than linear ℓ.
7. Links added to NB2 §1 and §3; figure caption notes the different RNG.
8–9. Trimmed 2.6(d); contents table updated; central-flow y_k now visible under the CF line.
Not done: the critic's optional trim of the invariant-form aside in 2.2′(f) (kept; it carries the corrected p > 3 statement).

### Round 4 (short Codex verification of the round-3 changes)

One minor item: the Adam-demo text said gradients "keep shrinking", but they rebound in bursts (gradient norm 1.1e-4 at step 500, 2.1e-2 at step 1000 at η = 0.003). Rewritten: v decays while gradients are small, raising the preconditioned sharpness toward the edge, and renewed oscillations push it back. Codex found no other errors; numbers, the Lanczos scaling and the catapult inequality check out.

## Final state

- `python td/build_td2_2_edge_of_stability.py`, `--check`, `--run` pass: 48 cells (10 answer, 9 code, 2 code-answer); `--run` ≈ 35 s total, slowest cell ≈ 14 s (SGD batch-sharpness demo), MLP training ≈ 11 s.
- `td/checks/td2_2_edge_of_stability_sympy.py` and `td/assets/interactive/checks/eos_toy_and_catapult_check.py`: ALL PASS.

## What remains uncertain

- Paper claims were checked by subagents and Codex against the arXiv/PMLR text; figure contents (as opposed to captions and text) were not inspected. Rosca et al. full text could not be fetched in round 2; their claims as stated here come from the round-1 subagent's reading of the PDF/ar5iv.
- The Q = 2 boundary of Exercise 2.2′(f) (attracting algebraically) rests on Codex's centre-manifold computation, not re-derived here; the text states it in one parenthesis.
- The momentum-EoSS result (Andreyev et al., ICML 2026) was checked from the PMLR abstract only; the momentum convention behind 2(1 ± β)/η was not verified.
- The SGD demo illustrates the λ_max gap but is too small and noisy to separate batch sharpness from the ratio of expectations; the notebook says so.
- The Adam demo reaches the edge mainly through v decaying after the loss is fitted, not through progressive sharpening while training; stated in the text.

## Readability pass (2026-10-09)

Following `td/READABILITY_BRIEF.md`. No mathematics, code or claims changed; code cells untouched (outputs from cache).

**Subsection openings.** All seven "The result." paragraphs (2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8) replaced. Each `##` subsection now opens with its question in one or two plain sentences, followed by a short "*Notation here:*" line (2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8) or inline definitions (2.7). The results that the spoiler paragraphs carried are kept, but now come after the setup:
- 2.1: the observation paragraph now names progressive sharpening and EoS where it describes them.
- 2.2: the expansion is derived first ("The expansion"), then displayed, then interpreted; θ† (the projected iterate) is now defined.
- 2.3: question → the averaging idea → the central-flow formula → its validation and limits.
- 2.5: Adam on a quadratic (with the v update and β₂ now written out, and the heavy-ball threshold restated) → the 38/η display → Cohen et al.'s finding.
- 2.6: the two duplicate paragraphs (result + "In their experiments") merged into one observation paragraph that defines g_B, H_B, b before the batch-sharpness formula; the Prop. 4 sentence trimmed to point at Exercise 2.6(c–d).
- 2.7: question → model and maps → "Three phases" in words.
- 2.8: the warmup mechanism moved into the Warmup paragraph; x, λ, α, β of the Kalra–Barkeshli model defined; "exactly the §2.2 model" → "the linear version of the §2.2 model".
- 2.4 had no spoiler paragraph; it now opens with the question (a gradient-only model; the all-orders modified flow).

**Local notation restated** where a subsection or exercise borrowed it: Exercise 2.2′ restates the toy and its orbit (x*, y*); Exercise 2.3 restates w, S, u and the §2.2 expansion, and (e) restates the toy and x*²; the central-flow code paragraph restates c = 62.5; Exercise 2.4 gets a one-line notation (w̄_k, δ_k, Σ_k); principal flow restates the rate (1/η)log(1−ηλ); Exercise 2.8(a) restates the toy with c; "the box" references now say which box or restate ηλ < 2 inline; EoS is defined in the pitch and re-expanded in 2.3, 2.4, 2.8; ⟨H,Σ⟩ = tr(HΣ) defined.

**Notation clashes fixed** (renamed, not re-derived): the four-stage cycle's displacement along ∇S is now s (was y, which clashed with the toy's y = S); the SGD noise variance in Exercise 2.8(b) and its solution is now s² (was σ², which clashed with §2.3's oscillation variance); 2.7 says its u is a weight vector, not the eigenvector; 2.4's Σ_k is called the rod's outer product.

**Old-format wording.** "Each subsection opens with its result" and the "solutions version of this notebook (`…_solutions.ipynb`)" sentence in Contents replaced; the `{{VARIANT}}` label removed; links to `td2_1_flows`, `td2_6_curvature`, `td2_3_sgd_sde` now use `.html#anchor`. Contents now says "most cells in seconds, the longest under about a minute" (the cached outputs were captured on a loaded machine and print up to ~65 s).

**Trims.** Exercise 2.1's preamble shortened; the kernel-terms parenthetical in 2.1 cut to a pointer; the linear-ℓ aside in 2.2 replaced by a pointer to Exercise 2.2′; the sympy-check sentence in 2.7(a) simplified; Damian et al. linked at first mention.

**MDL links.** None: this notebook cites no `../notebooks/*.html` content.

**Critic (fresh Claude Opus, read 2.1, 2.5, 2.4 closely, rest skimmed).** Ranked findings, all applied except: trimming the invariant-form aside in 2.2′(f) (kept; it carries the corrected p > 3 statement), the noise-floor numbers in 2.2 and the description before the §2.1 plot (kept; they tell the student what to look for). It found no mathematical errors and confirmed all anchors resolve. It also flagged the run-note link to `td2_2_edge_of_stability.ipynb` pointing at the page itself; that was a kit bug, already fixed in `htmlkit_td.py` during this pass (the rebuilt page links the `.ipynb`).

**Checks.** `python td/build_td2_2_edge_of_stability.py`, `--check` and `python td/checks/td_crosslinks.py` pass (49 cells, 12 solutions; 0 bad cross-links).
