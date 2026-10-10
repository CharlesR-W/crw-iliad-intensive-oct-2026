# Training Dynamics: final integration pass (2026-10-09)

Scope: (1) content hidden in `answer()`/`code_answer()` cells that the exercises variant needs; (2) cross-notebook consistency; (3) lecture sync; plus (4), added mid-pass by the coordinator, repointing links into the MDL day after its switch to single HTML pages. Backups of all builders and both lectures as they were before this pass are in the session scratchpad (`backup/`); the folder is not a git repo.

Method. Job 1 was split across nine Claude subagents, one per builder. Each read its whole builder, moved only what later text, code or figures depend on, and ran build, `--check`, `--run`, and a scratch runner that executes the *exercises* variant's code cells in order (to catch later cells depending on names only a solution stub defines). Jobs 2–4 were done by the integrator. A Codex (GPT, high effort, read-only) cross-notebook sweep served as a second opinion (results below).

## 1. Content moved out of answer cells

| Notebook | What was hidden | Where it went | Why it was needed |
|---|---|---|---|
| td1 | Ex. 1.4.2's rise time, (τ/s)·ln((1−ε)/ε) from εs to (1−ε)s | Ex. 6.1's hint, as a quoted result | Ex. 6.1's hint uses "the 10%–90% rise time from Exercise 1.4.2", and §6 item 1's 2 ln 9 needs it; the default text only says "of order τ/s" |
| td2_1 | Ex. 1.5.c's "note on the literature" (sign error in Kovachki–Stuart eq. 19, corrected constant c = λ̄(λ̄−a−½), corrected Nesterov penalty) | new markdown cell after the answer | the Kovachki–Stuart reading says "see the note after Exercise 1.5.c", which pointed at a blank cell |
| td2_1 | Ex. 1.3.b's least term e^{−2π/u} and order k* ≈ 2π/u | the markdown after the following code cell now names them | that text compared results with "e^{−2π/h}" and "2π/h" without saying what they were (1.3.a–b are optional) |
| td2_2 | Ex. 2.7's paragraph on the right-hand catapult plot (over-flattening; red crosses are very flat landings, not boundary runs) | new markdown cell after the catapult code | it was the only explanation of that plot |
| td2_3 | 3.6.A item 3's momentum caveat (the RMSprop floor carries over to Adam with β₁ > 0 if η_eff·h ≪ 1−β₁; the linear-rule runs at large κ break this but land within ~10%) | the markdown introducing the Adam demo | the demo's dashed floors are momentum-free predictions plotted against Adam with β₁ = 0.99. The ~10% was rechecked by the agent with a scratch run |
| td2_4 | 4.5.C's optional remark: exact large-n limit ‖sign(G)‖₂/n → E\|w₁\| = Γ(r/2)/(√π Γ((r+1)/2)) | 4.5.D's statement, as a stated fact | 4.5.D asks students to compare with "E\|w₁\| n (the remark above)", a blank cell |
| td2_5 | none (the known "With momentum" instance had already been fixed) | — | — |
| td2_6 | 6.5.A item 3's prediction ρ(t) = 1 − q/(a₀²+q)·e^{−2st}, q = σ²/2s | the item's statement ("show that …") | 6.5.B's plot and text use "the prediction arccos ρ(t)/π of 6.5.A, item 3"; item 6's hint uses q |
| td2_7 | 7.3.A.2's closed form f_t(x) off the training set (and the symbol v_k) | the 7.3.A part 2 statement ("show that …") | the kernel demo after 7.3.B computes exactly this formula (`f_at`) |
| td2_8 | Ex. 1.2's last paragraph (what the next code checks; Baldi–Sadowski eq. 20 underestimates the gap by ≈ 1/(2E(1−E)) ≥ 2 while the conclusion holds) | new markdown cell before the code | the code prints a "B-S eq.(20)" column explained only there; also a caveat about the core reading |
| td2_8 | Ex. 5.1's second half (bfloat16 makes small dense updates look sparse; Shenfeld et al.: sparsity vanishes in float32) | new markdown cell before the code | it motivates the bf16 half of the code cell and qualifies the core Mukherjee reading |

Exercises-variant execution: in all nine notebooks, every code cell ran without raising (stubs only define functions, or the dependent demo catches `NotImplementedError` and prints a fill-in note, as in td2_3). No later cell depends on a name only a solution defines.

Close calls deliberately left hidden (selected): td1 Ex. 6.3.1–6.3.2 interpretations; td2_1 Ex. 1.5.a.2 (Goh's displayed modulus omits a ½: a caveat on a core reading, nothing depends on it); td2_2 code-exercise 2.1 result and 2.2′(e)–(f); td2_3 3.4.A.3 estimator formulas (needed only by the 3.4.C code exercise, which is a deliberate exercise chain); td2_4 4.3.D's Cohen et al. 38 (stated elsewhere: td2_2 §2.5, td2_5 box); td2_5 5.3.1.4 (r_c, used implicitly in a later number, but it is what the exercise asks); td2_7 7.1.B.4 and 7.2.B.2; td2_8 Ex. 2.1's interpretation.

## 2. Cross-notebook consistency

### The four named points
- **Is the small-step expansion valid? (td2_1 §1.3 vs td2_3 §3.7.)** Consistent. td2_1: asymptotic and usually divergent; first correction trustworthy only for ηS well below 1; no gradient flow reproduces GD for ηλ ≥ 1; full-batch training sits at ηS ≈ 2. td2_3: SDE/SME results are justified only asymptotically as η → 0 on a fixed horizon; ηh is not small in the top directions; EoS; points back to td2_1 §1.3. td2_3's box restates td2_1's O(η²) global error correctly. No change.
- **Adam and the edge of stability (td2_2 §2.5 vs td2_4 §4.2–4.3 vs td2_5 box and Mechanism 2).** Consistent: all say full-batch Adam holds the *preconditioned* sharpness λ_max(P⁻¹H) at (2+2β₁)/((1−β₁)η) = 38/η, raw sharpness is not at its edge, and with minibatches the preconditioned sharpness sits below the threshold (td2_2). td2_2 freezes P = diag(√v)+εI without bias correction; td2_5 uses √v̂ late in training (bias correction ≈ 1): the same object. No change.
- **Kernels and lazy/rich (td1 §4–5 vs td2_6 box and §6.5 vs td2_7 vs td2_2 §2.7).** Consistent in substance. td1's dial is the per-mode invariant Δ (Δ² = k² − 4a², = |d| for aligned modes), lazy when max(|a₀|, s) ≪ Δ; large balanced weights are not lazy. td2_6 works with aligned scalar modes and d. **Fixed:** td2_6's box said lazy when "|d| ≫ the mode strength", but td1 calls *a* the mode strength and the criterion involves s; now "|d| much larger than both the target singular value s and the initial mode strength |bc|", with a note that NB1 states it with Δ = |d| for aligned modes. td2_7's "a lazy network at standard initialization has pre-activations of order one" and td2_2 §2.7's catapult "lazy phase" (ηλ₀ < 2, kernel moves O(1/n)) are compatible usages.
- **Progressive sharpening (td2_2 §2.1 vs td2_6 §6.2).** **Fixed a small inconsistency:** td2_2 described the kernel view of sharpening as "the top of the tangent-kernel spectrum growing, in scale or in direction". td2_6 §6.2 item 4 is explicit that only the nonzero spectra of H_GN and the weighted kernel are tied, not eigenvectors, so "in direction" is not part of the sharpness story. td2_2 now says "the top eigenvalue of the tangent kernel growing; only the spectrum is tied to the sharpness, not the eigenvectors". Everything else agrees (td2_1, td2_2, td2_3, td2_5, td2_6 all describe full-batch EoS at ηS ≈ 2 and SGD's batch sharpness the same way).

### Cross-notebook links and section references
- All cross-notebook links were checked (`td/checks/td_crosslinks.py`, added in this pass: 40 links, including the MDL ones) against the target notebooks' anchors (nbkit only checks intra-notebook anchors): all resolve, and each link's label matches its target section. One label/anchor mismatch fixed: td2_4 4.3.D's "[NB2 §2]" (a claim about the EoS oscillation) pointed at `#td22-5` (Adam); now `#td22-1`.
- Unlinked section references ("`td2_4` §4.2", "`td2_6` §6.5", "`td2_7` §7.3", "`td1_linear_networks` §1, §4 and §5", "NB2 §2.5", "NB2 §3", "NB2 §5", etc.) all point at the right sections.
- Mixed link targets: td2_8's index links the *exercises* notebooks, while in-text links elsewhere point at *solutions*. Both are fine; left as is.

### Duplicated derivations
- **td2_2 Ex. 2.1 repeated td2_1 Ex. 1.2.c at length** (exact modified flow on a quadratic, radius of convergence, no real ODE for ηλ > 1, why truncations can't see the threshold). Kept, since each NB2 file must stand alone, but td2_2 Ex. 2.1 now opens with an italic note: if you did td2_1's Ex. 1.2.c this is the same calculation, only the last paragraph of the solution is new, its conclusion is already in the text, so skip to the code exercise.
- td2_3 3.3.A vs MDL WB2 8.C.i: already handled by td2_3 (notation bridge, "start at item 4(b)"); re-verified against the current WB2 HTML (below).
- Cross-day, newly pointed: td2_5 Ex. 5.1.1 items 1–4 are MDL WB6 Ex. 10.1 (scale invariance under batch norm); td2_5 now says so in one line. td2_6 §6.4's measurement point (overlap ≈ 1 while learning is in the bulk) is MDL WB6 Ex. 9.1's quadratic toy; td2_6 now says so in one sentence. (WB6 itself says "The Training Dynamics materials take up the dynamics behind §4 and §8–§11.")
- No other at-length duplicates found: the heavy-ball threshold (derived in td2_1 1.5.a) and the Adam threshold (derived in td2_2 Ex. 2.5) are only restated elsewhere; the scalar two-layer net appears in td2_1 1.2.d, td2_2 §2.7 and td2_6 6.5.A but for three different questions.

### Day index (td2_8) vs each notebook's opening cell
Rows match their notebooks, with three fixes: td2_2 "Rod flow" → "Rod Flow" (the notebook's spelling); td2_4 now lists the modular norm (in its opening and §4.6); td2_6's "the Gauss–Newton matrix as the tangent kernel" overstated what td2_6 claims (only the nonzero spectra are shared) and was redundant with the next clause; now "the Gauss–Newton matrix and the loss-weighted tangent kernel share their nonzero spectrum, so spectral statements carry over between curvature and kernel". The suggested routes match td1's own short path.

### Other fixes found in passing
- td2_3: the text after 3.4.B and §3.7 item 5 said B_noise "grows roughly six- to elevenfold" along a run. Re-running the cell prints medians of 4 at the start, 75/124 over the late steps and 114/189 at the end, i.e. 20–45×. Now "twenty- to fortyfold … (median about 4 at the start, 75–190 late in the run)". The fitted B_crit = 54 still lies inside that range, as the text says.
- td2_4 4.5.D: "≈ √(2/(πr))" is a large-r approximation (0.80 vs exact 1 at r = 1); now says so.
- td2_5: "the cosine is only about 0.04" while the cell prints −0.036; now "the printed cosine of u with U is only about −0.04 (negative because the step is −ηU)".
- td2_6: "Other NB2 sections call the gradient G" → "Some other …" (td2_1 and td2_4 write g).
- td2_2: a solution referred to `td/checks/…sympy.py` (path from repo root); now `checks/…` "(next to this notebook)", matching td2_1's convention.

### Symbol clashes (noted, left: each notebook defines its own usage)
Σ (target correlation in td1; noise covariance in td2_*; MDL WB2 uses C for noise and Σ for the stationary covariance, bridged in td2_3); λ (Hessian eigenvalue in td2_1/td2_2; NTK in td2_2 §2.7; weight decay in td2_5); ε (step in td2_3 §3.4; Adam's ε in §3.6); S (sharpness; steps in td2_3 §3.4); P (Adam preconditioner in td2_2–td2_5; parameter count in td2_6); G/g/∇L for the gradient; C (classes in td2_6). L2's clash list now includes P.

## 3. MDL-day links (coordinator request)
The MDL day now ships single HTML pages. Repointed every `../notebooks/NN_name_{exercises,solutions}.ipynb[#anchor]` to `../notebooks/NN_name.html[#anchor]` (td1 ×2, td2_3 ×1, td2_4 ×1, td2_8 ×6), and checked every anchor inside the HTML with a script (nbkit checks only that the file exists). Every specific claim about MDL content was checked against the current HTML:

| Our claim | Where | Status |
|---|---|---|
| WB2 §8 compares SGD's stationary covariance with the tempered posterior ∝ H⁻¹; figure `sgd_stationary_vs_posterior.html` | td2_3 §3.3, td2_8, L2 | holds (§8.C.i items 3, 5, 7; figure exists in `notebooks/assets/interactive/`) |
| td2_3 3.3.A items 1–3 and first eq. of item 4 repeat WB2 8.C.i.1–4; WB2 writes C for the noise covariance and Σ for the stationary covariance | td2_3 | holds |
| WB2 8.C.i.6 is the preconditioned Lyapunov equation | td2_3 3.6.A | holds (PHΣ+ΣHP = (η/B)PCP) |
| WB2 §4.D derives dropout's adaptive L2 penalty | td2_8 §1 | holds ("Dropout as adaptive L₂ (optional)", Wager–Wang–Liang) |
| WB2 §4.E / exercise 4.E.i.3 works Dinh et al.'s two-layer rescaling | td2_4 §4.2, td2_8 §2, L2 | holds (item 3 of 4.E.i). L2 now cites 4.E.i.3 rather than "§4" |
| WB2 §4–§5 derive overtraining near n ≈ p, early stopping as a spectral filter, the cost of a large initialization | td1 §6 item 5 | holds (§4.C: spectral filter, limit from nonzero β₀; §5: ridgeless risk peaks at p/n = 1). Links now go to §4.C and §5 separately |
| WB4 §2 explains the learning coefficient in Bayesian model comparison | td1 §3.2 | holds (§2.A: p/2 replaced by λ for singular models); anchor `#wb4-section-2` added |
| WB6 "card 4" treats critical periods as a "weird fact" | td2_8 §4 | **stale**: WB6 no longer uses "cards" and doesn't say "weird fact". Now "lists critical periods among its strange empirical facts in WB6 §4" (WB6's own description is "a catalogue of readings on strange empirical facts") |
| WB6 "card 11" reads the sparsity and KL papers as a pair, with exercises on sparse vs low-rank and on bfloat16 "unchanged" | td2_8 §5 | content holds (§11 "Low-rank RL updates and the KL leash": Mukherjee et al., Shenfeld et al., Cai et al.; Ex. 11.1 sparse vs low-rank, Ex. 11.2 bf16); "card" → "§" |
| WB6 Exercise 11.2 works the bfloat16 measurement issue | td2_8 §5 | holds |

## 4. Lectures
**L1.** Only one stale item: the speaker note said NB1 §6 has "three more items … and the scale-versus-imbalance item above" (that item *is* heuristic 3). Now it maps the lecture's heuristics to §6 items 1 (+2), 3 and 6, and names the two remaining items (4, orthogonal initialization; 5, generalization dynamics, with the MDL WB2 §4.C/§5 pointer). Everything else (eq. numbers 1.4/4.1, Ex. 6.1's 4.4/ln(1/a₀), §5.3, Δ ≈ 2σ²h, the explorer's preset and slider names, td1's short path, `td2_4` §4.5) was checked against the current notebook and figure and is in sync.

**L2.**
- Width segment: "Show `td2_4` §4.5.C" → §4.5.D (the code; 4.5.C is the derivation); "the modular norm is a pointer there" → "the modular norm, and the depth scaling it suggests, are §4.6".
- Schedule segment: said "`td2_2` §2.8 quotes both" warmup accounts, and that Kalra and Barkeshli disagree "somewhat" with the Adam-second-moment account. td2_2 no longer discusses Liu et al.'s account at all. Now: the second account is named (Liu et al., RAdam) as not in the notebooks, with the note that the Kalra–Barkeshli reading also proposes a second-moment initialization for Adam; the unverifiable "somewhat" quote is dropped.
- Optimizer segment: Dinh pointer now "MDL WB2 Exercise 4.E.i.3".
- Weight decay: "D'Angelo et al., who argue weight decay is seldom useful as an explicit regularizer" → "never", matching td2_5's reading blurb and the paper's framing; "`td2_5` §5.2 quotes both" → "§5.2 and its readings give both".
- Symbol clashes: added P.
- Checked and in sync: every other section number, exercise number, number quoted (81/42/16%, 38/η, 12/λ₀, the hyperbola, B_noise formula, ΔL = −ηg²(1−ηλ/2), d^{k−1}), core-reading labels, and the interactive figures' preset names.

## 5. Codex second opinion
Codex (GPT, high effort, read-only) swept all nine notebook dumps for cross-notebook contradictions, wrong references to other notebooks, at-length duplicates, and index/opening mismatches. Its headline matched mine: no substantive contradiction in thresholds, constants or derivations; the two expansion-validity discussions agree; lazy/rich and imbalance accounts are compatible; the Lyapunov and noise-floor formulas agree; no wrong section numbers. Three of its ten findings (the td2_8 Gauss–Newton row, td2_2's "in direction", the td2_2/td2_1 duplicate) had already been fixed independently. The others, all checked and applied:
1. **td2_5, 5.1.1 item 5 answer** said "a small-step expansion of GD (NB2 §1) at first order cannot see" the norm growth. That is wrong for the *modified flow*: with ẇ = −∇L − (η/2)H∇L, wᵀ∇L = 0 and Hw = −∇L, d‖w‖²/dt = η‖∇L‖², i.e. the exact η²G²/r² per step of 5.1.2 (verified by hand). The answer now says gradient flow cannot see it and the first correction of NB2 §1's modified flow does.
2. **td2_3** said that when the signal dominates "v ≈ G² and the update is close to sign descent", which full-batch oscillating gradients contradict (td2_2 §2.5, td2_4 §4.3.D). Now "with slowly varying gradients", pointing to td2_4 §4.3.D for where the picture fails.
3. **td2_6 box**, SGD noise floor: now states the model's conditions (quadratic basin, frozen noise, ηh ≪ 1) and that the floor grows toward ηh = 2 and shrinks if the noise vanishes at interpolation (both from td2_3).
4. **td2_8 index, td2_5 row:** η/‖w‖² is SGD's effective step; AdamW's is η/‖w‖ (td2_5's AdamW summary). Row now says so.
5. **td2_5 box:** "full-batch Adam drives ηS_P to the threshold" now carries td2_2's caveats (can plateau below at very small η; below with minibatches).
6. **td2_3 3.1.A.1** repeats td2_1 Ex. 1.4.b.1's finite-population double sum at matrix level: one-line pointer added.
7. **td2_8 index, own row** understated its content: now mentions the worked exercises and numerical checks.

Also fixed in passing (from a Job 1 agent's report): td2_1's "the slow root of §1.5.a" (an exercise, with the term defined only in its answer) now reads "the slow root of Exercise 1.5.a (the one near 1, z ≈ 1 − ηλ/(1−β)) … the fast one (near β)"

## 6. Verification
Final pass, after all edits, on a machine with other jobs running (so timings are pessimistic). For every builder, build, `--check` (fresh), `--run` (0 failures, all cells under the 60 s budget), and the exercises variant executed in order (0 cells raising). `td/checks/td_crosslinks.py`: 40 cross-links, 0 bad.

| Notebook | Cells (answer / code / code_answer) | `--run` total | slowest cell |
|---|---|---|---|
| td1_linear_networks | 73 (20 / 8 / 3) | 20.8 s | 8.1 s |
| td2_1_flows | 44 (13 / 6 / 1) | 37.3 s | 12.1 s |
| td2_2_edge_of_stability | 49 (10 / 9 / 2) | 93.1 s | 35.4 s |
| td2_3_sgd_sde | 39 (6 / 10 / 1) | 40.0 s | 9.8 s |
| td2_4_optimizer_geometry | 54 (20 / 4 / 3) | 32.3 s | 17.2 s |
| td2_5_normalization_weight_decay | 50 (13 / 8 / 2) | 43.4 s | 14.5 s |
| td2_6_curvature | 29 (6 / 7 / 1) | 69.3 s | 24.8 s |
| td2_7_learned_first | 38 (8 / 6 / 2) | 55.8 s | 13.2 s |
| td2_8_further_readings | 25 (6 / 5 / 0) | 4.7 s | 2.6 s |

All edits in this pass are to markdown (no code cell changed). Not touched: `td/nbkit_td.py`, anything under `notebooks/`, the generated `.ipynb` files except by rebuilding.

## 7. Left for the instructor
- **td2_2 Ex. 2.1 vs td2_1 Ex. 1.2.c.** Both are kept in full (standalone rule), with a skip-note in td2_2. If you'd rather cut one, td2_2's could shrink to a stated result plus its last paragraph.
- **Warmup's second account** (Liu et al., Adam's second-moment variance) is in L2's speaker notes but in no notebook. Add a sentence to td2_2 §2.8, or drop it from L2.
- **td2_1 Ex. 1.5.a.2's answer** notes that Goh's displayed modulus omits a ½. It is a caveat on a core reading, but nothing depends on it, so it stays hidden; you might add it to the Goh annotation.
- **td2_1 Ex. 1.3.a:** the code cell after it writes out the recursion that item 2 asks students to derive (a Job 1 agent's observation; not changed).
- **Cross-day links.** All MDL links now point into the single-page HTML with anchors that exist today. Two are new: td2_5 → WB6 §10 (Ex. 10.1) and td2_6 → WB6 §9 (Ex. 9.1). nbkit checks only that `../notebooks/*.html` exists, not the anchor, so an MDL renumbering would break these silently. `python td/checks/td_crosslinks.py` (added in this pass) checks every cross-notebook and MDL anchor; rerun it after the MDL day changes.
- **Individual review logs** (e.g. `td1_linear_networks.md`) mention exercise numbers that no longer exist. They are historical records and were not edited.
