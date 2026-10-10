# Review log: td1_linear_networks (NB1, Saxe dynamics and linearity)

Files: `td/build_td1_linear_networks.py` → `td/td1_linear_networks_{exercises,solutions}.ipynb`;
figure `td/assets/interactive/mode_explorer.html` (check: `td/assets/interactive/checks/mode_explorer_check.py`);
math checks `td/checks/td1_linear_networks_math.py`.

## Source verification (before drafting)

A Claude (Opus) subagent downloaded and read the full text of 14 papers and reported exact statements. Findings that changed the draft relative to `TD_LINKS_v1.md` and the spec:

- **Tu–Aranguri–Jacot** (2405.17580): the formula is ∂ₜA ≈ −η√(AAᵀ+σ⁴w²I)∇C − η∇C√(AᵀA+σ⁴w²I) (Eq. 1, Thm 1), threshold sᵢ vs σ²w; phase diagram uses w = d^γw, σ² = d^γσ². The σ² = w^−γ parametrization belongs to Jacot et al. (2106.15933), and the C²/4 form to Kunin et al. (Eq. 8) and Dominé et al. (Thm 5.2). **The word "grokking" does not appear in TAJ**, so the old course's "grokking link" is not attributed to TAJ; Kumar et al. (2310.06110, abstract checked by me via WebFetch) is offered instead as the reading on lazy→rich grokking.
- **Saxe 2014**: depth independence for orthogonal init is shown *empirically* with the learning rate tuned per depth; the theory needs η ∝ 1/N_l and O(1) initial mode strength. Their N_l counts layers including input (our N = N_l − 1). Σ's are sums over examples.
- **Kawaguchi**: bad saddles exist from H = 2 hidden layers (three weight matrices); all saddles strict for H = 1. (TD_LINKS item 43's "depth > 3" counts layers including input.)
- **Achour et al.**: Theorem 7 classification; H counts weight matrices; non-strict saddles need S = top-r, r < r_max, H ≥ 3, and the "tightened" condition. Whitening is not assumed.
- **Jacot et al.**: only the first escape is proved; the full saddle-to-saddle sequence is Conjecture 3.
- **Dominé et al.**: exact solutions are two-layer only (despite "deep" in the title); sign of λ matters for non-square networks.
- **Advani–Saxe(–Sompolinsky)**: the arXiv version has two authors; the journal version adds Sompolinsky. Linked both.
- **AGF** is arXiv 2506.06489 (NeurIPS 2025); convergence proof only for diagonal linear networks.
- **Pennington et al.**: ReLU cannot achieve dynamical isometry with Gaussian or orthogonal weights; orthogonal sigmoidal networks can.

## Derivations added beyond the spec, and how they were verified

- **Per-mode invariant Δ** (§4.3, Ex. 4.4): for mode-separated dynamics, τȧ = k(s−a), τk̇ = 4a(s−a), so Δ² = k² − 4a² is conserved, Δ = ‖qqᵀ − ppᵀ‖_*, Δ = |d| for aligned modes, Δ ≈ 2σ²h for Gaussian init. This is my own derivation (it unifies imbalance and misalignment and resolves the old course's lazy-section inconsistency). Verified symbolically (sympy), numerically for the trace-norm identity, and by simulation: wide Gaussian (h = 2000–4000) runs match the Δ-ODE prediction to ~1e-3–1e-9; the per-mode rate 2√(a²+σ⁴h²) coincides with TAJ's Eq. 1.
- **Effective a₀ = Δ/4** (§5.2, Ex. 5.3, Ex. 5.1.5): from the closed form as a₀ → 0, s ≫ Δ; matches |z⁺|²/2 ≈ σ²h/2 from the linearized alignment analysis. Checked numerically.
- **Large balanced init is not lazy**: aligned (Δ = 0) and random balanced inits both show ~98% kernel change in the notebook's code; lazy inits show 1–2%.

## Verification run

- `python td/checks/td1_linear_networks_math.py`: all checks pass (whitened loss, conservation and O(η²) drift formula, closed form t(a), logistic and half-time, lazy kernel bound, depth-N ODE and plateau scaling, Hessians at origin, NTK formula, balanced and λ-balanced square roots, linearized Z flow and eigenvectors, Δ invariant, a₀ = Δ/4, Gaussian Δ, Ex. 2.3 Hessian, Ex. 6.1 numbers).
- `python td/assets/interactive/checks/mode_explorer_check.py`: runs the page's math block in node; worst errors vs scipy 4e-5 (a), 2e-4 (kernel); vs closed forms 4e-5; half-time 5e-8.
- Rendered the page headless (chromium) in dark mode at several presets and inspected screenshots.

## Critique rounds

(see below)

### Round 1

Critics: design/pedagogy, Claude Opus subagent (14 ranked problems); technical, Codex at high effort (27 items: 11 definite, 7 likely, 9 minor). I checked the disputed or surprising points myself before changing anything.

**Confirmed and fixed (technical).**
- *Mixed regime timing was backwards* (both critics). Strong rich modes finish before lazy ones, since (τ/2s)ln(4s/Δ) ≪ τ/Δ for s ≫ Δ. Fixed in §5.5, Ex. 5.3.3 and the mixed-regime commentary. Only modes with s > √2 Δ have a convex rise (Codex's f′(a) argument). The demo now uses Δ ≈ 1 with s = (8, 3, 0.25, 0.05).
- *Δ is exactly conserved only for mode-separated dynamics.* I re-measured it with η = 5e-4. At h = 8 the per-mode Δ drifts by 10–45%; at h = 64 by up to 7%; at h = 2000 by under 1%. The text now says (4.3) is exact for separated modes and approximate for random initialization. The explorer caption says "approximately". The diagnostic now checks all three cross-Gram overlaps (QᵀP, PᵀP, QᵀQ).
- *Alignment derived only for the first mode* (design critic, also Codex 12). §5.1, §1.3 and §6 now say that the linearization derives first-mode dominance, and that later-mode alignment and logistic behaviour are checked numerically, not derived. The effective a₀ is stated exactly, ½|z⁺|² = (k(0) + 2a(0))/4, which is ≈ Δ/4 only at large width.
- *Orthogonal initialization* (both critics). Random orthogonal W(0) has mode strengths O(n^{-1/2}) with random signs, not O(1). In square balanced networks the sign of det W is conserved: τσ̇ᵢ = 2σᵢ uᵢᵀEvᵢ, so singular values cannot cross zero. A wrong-sign start stalls at a saddle. I verified this numerically (n = 4: loss sticks at ½·0.5² = 0.125 for both mismatched starts and reaches 0 for both matched ones). Added Ex. 6.3.2 and a code cell, and rewrote §2.3 and §6 item 4. Arora–Cohen–Golowich–Hu 2019 is added as a reading (abstract checked via WebFetch). Bartlett–Helmbold–Long was checked but not cited.
- *Landscape statements.* Global minima are now S = top m with m = min(r, r_max); Achour's classification is labelled with its full-rank assumption. Ex. 3.3.2 is restricted to aligned critical points with a spare hidden direction (Codex's counterexample). "No spurious minima" is no longer presented as a convergence theorem. The gauge-family dimension claim "at least Σh_j²" was false for wide layers (I found this myself) and is replaced by the exact square case. "D selects which point" became "restricts to a level set/orbit".
- *Lazy criterion.* Now max(|a₀|, s) ≪ Δ, not just Δ ≫ s. Weight motion is measured relative to the joint size √k.
- *Large balanced inits.* Rescaled to k(0) = 30, matching the lazy runs (they had been k = 10, which the design critic flagged). Kernel change is reported on the target modes (≈97%) as well as for the full operator. Added the structural reason: a balanced end-to-end map is the square of the layer scale, so large balanced weights mean a large initial function.
- Smaller items: depth-N ODE stated on the positive branch; the learning-rate argument now shows contraction is best at ηλ = 1, not monotone up to 2; the Chizat output scaling is described correctly; Ex. 4.5 sign conditions are corrected (λ > 0 needs h ≤ n_y, λ < 0 needs h ≤ n_x) along with the root-selection argument; the neural-race exactness hypotheses are added; AGF and Li–Luo–Lyu blurbs are qualified; Ex. 2.4 points lie below the line, not above; the τ is restored in Ex. 2.2; the strict-saddle growth rates are corrected; the "twice the shallow rate" slip is fixed; the closed form's domain is stated; the stage-separation constant is made consistent (2ln9/ln(1/a₀)); "rank" becomes "effective rank"; the flat-data stages are described as overlapping; and the explorer's normalized axis no longer clips large a₀.

**Fixed (design).**
- §1.2 no longer claims every symmetry of a gradient flow gives a conserved quantity.
- Δ is named the mode **invariant**; "imbalance" is kept for D and d. The text says the per-mode Δ bookkeeping is this notebook's own and points to Tu–Aranguri–Jacot and Dominé et al. for the matrix forms.
- Length: Ex. 3.4 (duplicate of §3.4) is deleted; the duplicate Saxe reading in §2 and the duplicate Physics-day links in §4 are removed; the Δ table moved from §4 to §5, where it is interpreted; Ex. 1.2.2, 3.2.3, 4.2 and 5.4 are marked optional; a one-hour short path was added to the intro.
- §6 now uses the same three slots for every heuristic (Derived / Planted / Transfer). Item 5 is labelled as derived in MDL WB2, not here. Item 6 (resting on one Kunin paper) is cut; its content now lives in the Get-rich-quick blurb.
- Notation: TAJ's dimension is n, not d; §6 uses n samples and p parameters; M's eigenvectors are e^±; broken §4.4 reference fixed; table and intro "Uses" corrected; Ex. 3.5 relabelled as a code demo.
- Readings trimmed to about two sentences; the TAJ formula moved out of the blurb.
- SLT link (old-course content): one sentence in §3.2 plus a reading, Pepin Lehalleur–Rimányi (arXiv 2411.19920; existence, title and authors checked via WebFetch, along with their sentence that Aoyagi computed the RLCT for deep linear networks). I did not verify Aoyagi's own paper or venue, so it is not cited directly.
- §3 pitch now says who would want the section; §5's pitch states the confusion precisely; the three-layer panel notes that it has no prediction.

**Not adopted.**
- Shrinking §3.4 to two sentences: I kept it as the default text, since it is the section's punchline, and deleted the duplicate exercise instead.
- The design critic suggested dropping the Li–Luo–Lyu reading: kept, shortened, because §5.2 and §6 cite it.

### Round 2

Critics: design/pedagogy (Claude Opus, 12 problems) and Codex (17 items: 5 definite, 5 likely, 7 minor). Both confirmed that the round-1 fixes held: the lazy kernels move 1–2%, alignment is marked as derived only for the first mode, the det-sign stall is reproduced, and the Ex. 5.5 predictions are within 1–6% of s.

**Confirmed and fixed.**
- *"Balanced" was still used in two senses* (design critic, the most important point). The §4.3 sentence calling Gaussian init "nearly balanced in the narrow sense" was wrong. A wide Gaussian init has tr D ≈ 0 but a large D: its eigenvalues are ≈ ±σ²h, because W₁W₁ᵀ and W₂ᵀW₂ are large on nearly orthogonal subspaces. Δ is the size of the mode's block of D, so it is the mode's imbalance, measured properly. The text now says so, and the table has an ‖D‖ column (small Gaussian 1.7e-5, large imbalance 30, wide Gaussian 22, both balanced runs ≈ 0). This matches the instruction that "large imbalance gives lazy; large balanced init does not".
- *Lazy criterion is sufficient, not necessary* (Codex 1). Fixed in §5.3, the §5 pitch and Ex. 5.2.
- *Ex. 5.1.4 was missing a factor 2* (Codex 2). Rederived: a₂/a₁ ≈ 2(a₀,₂/a₀,₁)(a₀,₁/s₁)^{1−s₂/s₁}.
- *The wrong-sign trap need not be broken by finite steps*: in the 1×1 example, GD keeps W₂ = −W₁ exactly. Text and answer now say "in generic matrix runs" and "forever in exactly symmetric cases".
- *PNAS 2019 does compare with a nonlinear network* (Codex 4; the design critic suspected the same). I checked their Fig. 2 caption and text via WebFetch: they compare an MDS of a deep nonlinear network's representations (Rogers & McClelland) with the deep linear analytic trajectories. §6 item 2 Transfer is corrected.
- §3.4 escape laws are restricted to the balanced aligned trajectories; partially vanishing saddles can be strict at depth 3 (Codex 7). The global-minimum set is described as containing gauge orbits, not as one manifold (Codex 9). k = 2a is restricted to aligned modes with a ≥ 0 (Codex 12). "Must rotate" is removed for the random balanced case, whose K₀ is isotropic. "Lazy is not faster in steps" now reads "raising the overall kernel scale alone does not reduce step count". Ex. 2.3 now holds the per-layer scale fixed (Codex 11). The Δ-drift statement is qualified as continuous flow (Codex 15). The three-layer panel's different σ is stated (Codex 16). The Jacot reading gets its probability qualification (Codex 8). Silent alignment says "approximates" instead of "equals" (Codex 14). The Advani description gets its rank-one ansatz and i.i.d. isotropic inputs (Codex 13).
- *Explorer.* The intro now states the large-imbalance condition (d ≫ s, a₀). Integrations that hit the 10⁷ cap are cut at their last integrated time and flagged in the table (Codex 17); checked by screenshot. The caption notes that depth and kernel are explained in §2 and §4, and what d means at N ≥ 3.
- *Ex. 6.2 contradicted Ex. 6.1's criterion* (design critic 4). It now closes the loop: the predicted loss curves for ε ∈ {0.05, 0.15, 0.3}, with the answer saying stages are visible only for strong hierarchy (small ε). The ε = 0.15 spectrum gives one broad drop.
- *§6 was missing the "small init rich / large init lazy" heuristic and the old course's grokking link* (design critic 3). Added as item 6 with the three slots: grokking via Kumar et al. (lazy-then-rich) and Dominé et al. (delayed rich learning in some imbalanced non-square initializations). Item 5 (Advani–Saxe) is shrunk to a few sentences.
- Student-facing references to "the old version of this material" and "the previous course iteration" are removed (design critic 5).
- Repetition: the end of §1.4 is cut to one clause and the §2.3 orthogonal paragraph to two sentences. Imbalance at depth N ≥ 3 now gets a short paragraph in §2 (design critic 11).
- Ex. 5.5 is turned into a code demo that keeps the "why smaller η?" question (design critic 7).
- Readings: §5 is cut to Atanasov, Chizat and Kumar. Dominé, Kunin and Li–Luo–Lyu became inline links where the text uses them. The Pepin Lehalleur–Rimányi reading became an inline link. The neural-race and Saxe blurbs are trimmed, and the Get-rich-quick evaluation is gone.
- Hints added for Ex. 2.1, 4.3, 5.2, 6.1 and 6.3.2 (perturbation formula plus Grönwall). Ex. 3.2.3 now derives (ΣᵀΣ)^{1/2} inline instead of using §4.2.
- The intro now says "simplest", not "the one" exactly solvable model. The "this notebook's own" novelty claim is reworded as "a per-mode repackaging of D". The "large imbalance (aligned…)" table label is fixed. The short path is restated honestly as 1–1.5 h, with §2 and §3 marked as side branches.

**Not adopted.**
- Splitting §5 into an alignment section and a regimes section; I kept one section with numbered subsections.
- Cutting to 8–9k words. After the cuts the exercises notebook has about 10.5k words of text and prompts, and the solutions add about 4.6k words of worked answers. The required content list (landscape, NTK, rich, lazy, mixed, depth, alignment, five heuristics) is long, so I flag length as a question for the instructor.
- Effort tags in the table of contents.

### Round 3

Critics: design/pedagogy (Claude Opus, 13 items; it called items 1–4 the main ones and the rest minor) and Codex (9 items: 2 definite, 3 likely, 4 minor). Both confirmed that the old course's three lazy/rich problems are fixed.

**Fixed.**
- *Alignment moved next to the assumption it justifies* (design critic 4). The linearized derivation is now §1.5. It is followed by the alignment exercise (Ex. 1.6) and the "alignment before the loss moves" cell, and both are on the short path. §5 subsections were renumbered 5.1–5.4.
- *Length* (design critic 1). Removed: Ex. 1.5 (closed-form algebra; the text now says to differentiate F, and the check script verifies it), Ex. 2.3 (folded into two sentences of §2.3), Ex. 4.5 (λ-balanced algebra, now one sentence with links), and Ex. 5.4 (early-stopping contrast, now one sentence in §5.1). §6 item 6 is cut to four lines; the §7 field-theory blurbs are one line each and the chatty duplicate is gone. The intro now states that full depth is several hours.
- *Broken "Exercise 5.5" references* now point to the named code cell. *"What you should see"* is a markdown cell, visible in the exercises notebook; only "why smaller η" stays an answer. The table description is fixed. The short path names concrete §6 items.
- *Codex A1:* at fixed end-to-end a₀ the N ≥ 3 plateau still grows with depth up to N ≈ 2ln(1/a₀) − 2 and only then falls like 1/N. Verified by quadrature (peaks at N = 11 for a₀ = 1e-3 and N = 25 for 1e-6) and added to the check script. §2.2 and §2.3 are corrected.
- *Codex A2:* the explorer intro's "rate close to d" now applies to two layers only (deeper: larger).
- *Codex B3:* the effective seed ½|z_α|² is claimed only for the first mode. Later modes' seeds can be changed by nonlinear separation at small width (their example: a 3τ delay that persists as init shrinks).
- *Codex B4:* "why smaller η" now separates the Gauss–Newton/kernel curvature (lazy runs) from the residual Hessian term (large balanced runs: eigenvalues 42 and −12 against a kernel of 30; checked).
- *Codex B5:* the Arora–Cohen–Golowich–Hu blurb now includes the step-size condition. It moved to §3, where it illustrates "no bad minima is not a convergence theorem".
- *Codex C6–C9:* tr D ≈ 0 needs n_x = n_y. ‖D‖ is now computed through QR plus eigvalsh, so balanced runs give ~1e-14 instead of 1e-7. Ex. 6.3.1 is made conditional on convergence. Ex. 6.3.2's argument now uses τ d/dt ln|det W| = 2tr(Oᵀ E) via the polar decomposition, which needs no distinct singular values (identity checked numerically).
- *Design items 7–13:* §6 item 4's slot now says "nothing in the data; assumed: …" and that the det trap is specific to square layers. Item 5 links MDL WB2. Item 3's filler Transfer line is replaced by a concrete pointer. Pitches are softened ("derived or popularized", "fully classified under mild assumptions", "a minimal form of feature learning"), and the §5 pitch is in words, without the formula. The s > √2 Δ condition is derived in the Ex. 5.2.3 answer (checked). The explorer caption is split into three paragraphs, and §5.3 and §5.4 point to its presets. Σ_xx is added to the setup.

**Process note.** One restructuring script ended with a global `replace("@", "")` meant for its own placeholders, which stripped matrix-multiplication `@` from the code. I caught this before any build was kept, restored from a backup, and re-ran with a safe placeholder. I then confirmed the count of " @ " matched the backup (23) and that `--run` passes.

## Final verification (after round 3)
- `python td/build_td1_linear_networks.py`, `--check` and `--run` all pass: 73 cells (20 answer, 8 code, 3 code-exercise). `--run` takes about 9 s in total, with the slowest cell about 3 s.
- `python td/checks/td1_linear_networks_math.py` passes all checks, including the round-3 additions.
- `python td/assets/interactive/checks/mode_explorer_check.py` passes (worst errors: a 4e-5, kernel 2e-4, closed forms 4e-5).
- All rendered figures inspected; the explorer was screenshotted at desktop and phone widths, in light and dark.

## What remains uncertain
- **Length.** Without worked answers the notebook is about 10k words; full depth is several hours, and the short path is about 1–1.5 h. The design critic would cut further. That is the instructor's call.
- **The per-mode Δ bookkeeping** (§4.3) is my own repackaging of the conserved D. Its two-layer, wide-Gaussian prediction agrees with Tu–Aranguri–Jacot's Eq. 1 (rate 2√(a² + σ⁴h²)), and with simulations to 1e-3 or better at h ≥ 2000 and a few percent at h = 8. I have not found it stated in this per-mode form in a paper.
- **Second-hand source details.** Aoyagi's RLCT computation is cited only as reported by Pepin Lehalleur–Rimányi; I did not open Aoyagi. The Kumar et al. venue is given as 2023 because I did not confirm ICLR 2024. Bartlett–Helmbold–Long was checked but not cited. The Alternating Gradient Flows claims come from the verification subagent's reading of the PDF, not from a fresh check by me.
- **Not derived; checked only numerically.** Later modes' alignment and logistic behaviour from random init (§1.5, §5.1). The claim that random-orthogonal starts with matching det sign converge was checked on n = 4 examples only.

## Readability pass (2026-10-09)

Following `td/READABILITY_BRIEF.md`. No mathematics, code or claims were changed; code outputs come from the existing cache.

**Subsection openings and local notation.**
- Every `###` subsection in §1–§5 now opens with its question in a plain sentence: 1.1 how the two layers move; 1.2 what the flow keeps fixed; 1.3 when the flow splits into scalar problems; 1.4 the per-mode reduction; 1.5 why small random weights align; 2.1–2.3; 3.1–3.4; 4.1–4.3; 5.1–5.4.
- Notation is restated where it is used. There is a new "Notation here" line in §1's box (layers, Σ and its rank and SVD, E, L, τ) and §3's box (W, L, E, Σ, r_max, Σ_xx/Σ_yx/Σ_xy, τ). §2's box adds Σ, E, a, s, d, a₀ and the plateau length. §4's box adds the network, Σ and τ, plus a "From §1" line (b, c, r_α, a, d and (1.4)). §5's box adds Σ, a, s, τ and d = c² − b². §6's box adds Σ_S, a₀, a usable definition of Δ, Σ_xx/Σ_yx and τ.
- Inline definitions where symbols first appear within a subsection: b_α, c_α, r_α (1.3); a₀ = a(0) (1.4); b, c and w_j (3.3); a₀ (3.4); p_α, q_α, a_α, k_α restated (4.3); w_j and (2.1) restated (2.3); Σ_S and the large-width seed (5.1); δs (§6 item 1); n (§6 item 4); Δ and k in the Exercise 1.6.4 answer.
- "Rich" and "lazy" are now named as regimes in 1.4 instead of being used before they are introduced.

**Spoilers and density.**
- The intro table no longer lists the mode ODE, the depth-N equation or τẆ = K_t[E] in formula form; it gives them in words.
- The §4 pitch is in words, without K_t. The last sentence of the §5 pitch (a compressed statement of the freezing criterion before Δ is defined) is cut; §5.2 states it properly.
- §5.1's seed comparison with §1.5 is reduced from a formula-heavy aside to two sentences. The small-width caveat is kept.
- §2.3 is split into the stability argument and the Saxe et al. paragraph. The "Imbalance at depth N ≥ 3" paragraph moved from §2.3 (learning rate) to the end of §2.1, where the per-layer variables live.
- §3.2 now answers its own question first: global minima, then "every other critical point is a saddle", and only then the gauge manifold and the RLCT aside.
- §4.3: "We call Δ the invariant" now comes right after (4.3), before the term is used. Two repetitive sentences are cut ("It is the size of this mode's block of D…" and "the matrix-level results it agrees with are the following"). "Four-vector system" became "a closed system for its two vectors".
- Small cuts: a duplicated kernel formula in 5.2, and the repeated parenthetical "(their experiments tune it at each depth)" in §6 item 4. The Jacot et al. reading no longer uses undefined σ² = w^{−γ}.
- Units: in 5.4, "rate Δ" became "rate Δ/τ", matching 5.2.

**Old two-file wording.** The intro said worked solutions are "in the solutions notebook". It now says hints and solutions are folded on this page, and that the runnable `.ipynb` holds the code with the code exercises left open.

**Links.**
- MDL: the WB2 "§4.C and §5" links in §6 item 5 pointed at sections that no longer exist. They are repointed to [WB2 §1 *Classical statistical learning*, part 1.D Regularization](../../notebooks/02_statistical_learning.html#wb2-section-1), which covers early stopping as a spectral filter and the frozen null-space part of the initialization, and to [WB2 §3 *Overfitting and double descent*](../../notebooks/02_statistical_learning.html#wb2-section-3), which covers the n ≈ p peak of the minimum-norm solution. The sentence now names both pieces and calls them ingredients; the claim itself is unchanged.
- The WB4 §2 link (learning coefficient in Bayesian model comparison) still resolves. Its text now names "§2 Occam, two ways, part 2.A Bayesian Occam".
- §7 "What carries forward" now links the four NB2 pages instead of naming their stems in code font. Three explorer mentions in 2.1, 5.3 and 5.4 now link the figure.

**Critic.** One fresh Claude Opus readability critic jumped into 2.3, 4.2 and §7 and returned 24 items. I adopted items 1–18 and 20–23, with item 12's Δ wording slightly shortened. It found no math errors. Not adopted:
- Item 19: the low-rank-bias passage appears in both 5.1 and §6 item 3. Both are kept, because §6 must stand alone and 5.1 is where the argument lives.
- Item 24: the Physics-day links in §4's readings and §7 point to `../../IliadPhysicsLearning/.../*_solutions.ipynb`, a relative path into a sibling repo that uses that day's two-file naming. That is not TD's to fix; flagged for the instructor.

**Checks.** `python td/build_td1_linear_networks.py`, `--check` and `python td/checks/td_crosslinks.py` all pass (73 cells, 23 solutions; 60 cross-links, 0 bad). No code changed, so outputs come from the cache. The builder text grew about 5%, mostly from the notation lines.
