# td2_8_further_readings: review log

Builder: `td/build_td2_8_further_readings.py`. Generated: `td/td2_8_further_readings_{exercises,solutions}.ipynb`.
Checks: `td/checks/td2_8_dropout_nwgm.py` (sympy + exact enumeration), `td/checks/td2_8_bf16_rounding.py` (numpy bfloat16 rounding against torch).

## Shape

1. Day index: one table row per sibling notebook (what is in it, "read this if…"), suggested routes, out-of-scope pointers (MDL WB2 §8; Physics of DL day).
2. Five reading topics, each pitch → "what you need" → (exercise) → readings:
   - §1 Dropout as a geometric-mean ensemble. Ex 1.1 (NWGM = σ(E S)), Ex 1.2 (small-spread gap), code check. Regularization side points to MDL WB2 §4.D (already derived there).
   - §2 Layer cushion / noise robustness / basin size. Ex 2.1 (Prop. 3.1, the stable-rank bound and its equality case; random-input baseline), code check.
   - §3 Geometry-adaptive harmonic bases. Ex 3.1 (Gaussian MMSE denoiser = fixed-basis shrinkage; Jacobian = σ⁻²Cov; MSE = σ² tr J), code check. Framed neutrally as an inductive-bias description, not as an answer to the dropped moment-bias question.
   - §4 Critical periods and loss of plasticity. Readings only.
   - §5 Fine-tuning and RL update dynamics. Readings grouped by level (kernel; which parameters; which singular directions; relative to pretrained weights; output space; CRW's survey post). Ex 5.1 (numerical vs stable rank) with code (also the bf16 measurement demo); Ex 5.2 (for a linear layer the curvature eigen-directions are a c_kᵀ with c_k input-covariance eigenvectors, so W's principal directions are not curvature directions in general; ties to §2's cushion and K-FAC).

## Source verification (what was checked, and how)

All arXiv abstracts fetched from arxiv.org/abs (WebFetch); full text checked where a specific claim is narrated.

- Baldi & Sadowski 2013: PDF read (§2–5). NWGM identity eq. (5)–(7), linear-unit regularizer eq. (27)–(28), "p = 0.5 maximizes regularization", "only constant and logistic functions" (cited to their [1]).
- **Eq. (20) discrepancy.** The paper's E − NWGM ≈ V(1−2E)/(1−2V) underestimates the exact gap by ≈ 1/(2E(1−E)) ≥ 2 in the small-spread limit. Verified symbolically (sympy series, two-point spread) and by exact enumeration in `td/checks/td2_8_dropout_nwgm.py`; independently confirmed by Codex (two-point checks at μ = 1, 2, 3 matched the predicted factors). Cause: E − NWGM is beyond the order of the paper's expansion of G, G′ about ½. The notebook states this in one sentence and notes the paper's qualitative conclusion is unaffected.
- Arora, Ge, Neyshabur, Zhang 2018: PDF read. Definition 3 (noise sensitivity with η‖x‖ scaling), Prop. 3.1, Definition 4 + remark (1/μ² = noise sensitivity), §6 "layer cushion … drastically improved during the training, especially for the higher layers (8 and higher)", VGG-19/CIFAR-10. "Orders of magnitude better in practice" is the abstract's phrase; Codex notes the comparison is of leading bound factors, so the text says "the authors report".
- Dziugaite & Roy 2017: abstract; binary MNIST from Codex's check of the paper.
- Kadkhodaie et al. 2024: PDF read. Eq. (3) Miyasawa, (6) shrinkage form, (7) SURE, (8)–(10) optimal Jacobian = σ⁻²Cov and MSE = σ² E tr ∇f*; bias-free UNet/BF-CNN (App. A), symmetric-Jacobian idealization (§3.1), C^α classes, disk manifold.
- Mohan et al. 2020, Kamb & Ganguli 2024, Belrose et al. 2024, Ash & Adams 2020, Lyle et al. 2023, Kleinman et al. 2024, Achille et al. 2019: abstracts (Ash & Adams: "shrink and perturb" confirmed in the PDF).
- Dohare et al.: Nature metadata and abstract via Crossref ("no better than a shallow network", ImageNet and RL); arXiv preprint has an earlier title ("Maintaining plasticity…") and author list without Lan.
- Malladi et al. 2023: abstract.
- Mukherjee et al. 2025: HTML read (tolerance 1e-5 in bf16; Table 2 ranks 99.2–99.8%; "conjecture" on in-distribution data).
- Cai et al. 2025: HTML read. Rank-1 component **rescaled to the update's norm**; metric is relative accuracy vs. the fully trained model (99.17% average), not fraction of gain; "evolves approximately linearly" (R² ≈ 0.914).
- Zhu et al. 2025: HTML read. "Principal weights" = largest-magnitude entries of a rank-k SVD reconstruction, used as a curvature proxy because curvature is "computationally prohibitive"; overlap is a mask measurement; principal angles and spectral drift measured separately; precision is an amplifier, not the cause. Venue: NeurIPS 2025 workshop (preliminary), so no venue given.
- Shenfeld et al. 2025: PDF read. Forward KL E_x KL(π0‖π) predicts forgetting; the idealized result is in KL(π‖π0); "sparse updates came from bfloat16 … float32 … no sparsity … full rank" (main text, alternative-predictors section). Venue not confirmed from a primary source, so none given.
- LoRA Without Regret (Thinking Machines, 2025-09-29): rank-1 matches full RL in their experiments; O(1) bits per episode argument.
- CRW, "Some Interesting Papers on RLVR" (public, charlesr-w.github.io/crw-blog, page title as given, H1 "Fruit of Preterition"). Described neutrally as an informal survey; no views attributed beyond what the post says about itself.
- Martens & Grosse 2015 (K-FAC): abstract.

## Math checks
- Ex 1.1, 1.2: `td/checks/td2_8_dropout_nwgm.py` and the in-notebook enumeration.
- Ex 2.1: Monte Carlo noise sensitivity vs formula; equality case attained (cushion 0.781 = bound) in the code cell.
- Ex 3.1: Monte Carlo MSE vs σ² tr J (0.1367 vs 0.1367).
- Ex 5.1: numerical rank 512, stable rank 1.20; bf16 rounding matches torch on 400k values (`td/checks/td2_8_bf16_rounding.py`); spacing 2⁻¹³ at 0.02 confirmed.
- Ex 5.2: algebra (exact for a linear layer; vec convention Σ_x ⊗ I).

## Critique rounds

### Round 1
- **Technical (Codex, high effort).** 13 findings, all accepted after checking: Cai metric/rescaling (verified in HTML); Zhu mask vs singular-subspace conflation; cushion concentration needs large stable rank; bf16 "half spacing" wording for two separately rounded checkpoints; algebraic vs numerical rank; Kadkhodaie symmetric-Jacobian is an idealization and homogeneity needs ReLU; "small error" needs normalization by dσ²; worst-case quantifier in ψ ≤ 1/μ²; Shenfeld's two KL orientations; LoRA argument scoped to their experiments; "orders of magnitude" softened, binary MNIST, basin-size "proxy"; Fourier modes need periodic boundaries; to_bf16 documented as finite-only. Codex confirmed the eq. (20) discrepancy independently.
- **Design (Claude Opus).** Report relayed by the coordinating session after an overnight interruption. Changes made: §5 restructured with the five levels as headings and one-sentence blurbs, measurement discussion moved into Ex 5.1; old Ex 5.1(1) (bf16 arithmetic) replaced by a pointer to MDL WB6 Ex 11.2, keeping the code demo and Shenfeld's float32 fact; added Ex 5.2 so the "which directions relative to curvature" question is worked rather than dropped, and softened to "None of the papers here"; §3 pitch cut to three sentences and placed as an inductive-bias result alongside td2_7's spectral bias; §2 basin size moved into the pitch, 2.1 parts merged, code reordered, equality case shown, WB2 §4.E linked, and a plain statement that basin-size dynamics are not covered (td2_6 §6.5 pointer); topic count fixed; day index re-synced to each sibling's opening cell, file names instead of NB1/NB2, "read this if…" column, links switched on (all siblings exist); Ex 1.2 critique cut to one sentence and loose bound removed; one core reading per section (§4: Dohare; §5: Mukherjee); Ash & Adams added to §4; Belrose et al. added to §3 (suggested by the td2_7 writer; verified).

### Round 2 (final technical check, Codex, high effort)
Confirmed all round-1 fixes, all day-index rows and section references against the siblings' opening cells, Ex 2.1 (incl. equality case), Ex 5.2's algebra, the Ash & Adams and Belrose blurbs, and the `--run` numbers. Four remaining issues, all fixed:
1. K-FAC statement made precise (Fisher block ≈ E[aaᵀ] ⊗ E[ggᵀ] with model-sampled targets and preactivation gradients; empirical Fisher differs; Fisher = GGN for squared error and softmax cross-entropy).
2. Ex 5.2 no longer says a large cushion *is* covariance alignment (Codex counterexample: cushion at 99% of its bound with Σ_x's top eigenvector orthogonal to W's top right singular vector). Now: suggestive, not established, because the cushion is a worst case over inputs and ignores their frequencies.
3. Shenfeld blurb now states the idealized analysis is in KL(π‖π0).
4. td2_6 index row now keeps the sibling's claim boundary (learning in the bulk in the tested setting, and where that fails) and says "shared nonzero spectrum" rather than "Gauss–Newton = tangent kernel".
Also, per the coordinator: the "less background" route now uses td1's built numbering (§1, §4, §5) and points to td1's own marked short path.
A second design round was not run: the coordinator relayed round 1's design report, and its fixes were applied; the final Codex round found no structural problems.

## Remaining uncertainties
- Venues for Zhu et al. and Shenfeld et al. not confirmed from primary sources; none stated.
- Kamb & Ganguli venue not stated (arXiv 2024).
- The eq. (20) observation is ours (verified two ways plus Codex), not from published errata.

## Readability pass (2026-10-09)

Following `td/READABILITY_BRIEF.md`. Mathematics, code and claims unchanged (code cells byte-identical, so cached outputs were reused).

**`nb()` (owned by this builder).** It now always returns ``[`stem`](stem.html#anchor)`` (or without the anchor); it previously linked only if a `<stem>_exercises.ipynb` existed, so after the one-page switch every day-map row and the inline links to `td26-4` and `td1-s5` had become plain code text. The kit rewrites these to `.ipynb` in the runnable notebook. The unused `HERE`/`Path` import went with it, and the module docstring was updated.

**Old two-file wording.** Removed the `{{VARIANT}} ·` label from the title line, the "exercises version … solutions version" paragraph under the day map (now one clause: solutions are for everyone), and the run-the-cells note now refers only to the runnable notebook.

**MDL links (the day was restructured; WB2 now has §1–4).**
- Day map, "out of scope": WB2 §8 (`#wb2-section-8`, broken) → WB2 §4 (`#wb2-section-4`), named as "Does curvature cancel?" (its 4.B.3 compares SGD's stationary law on a quadratic basin with the posterior).
- §1: WB2 §4.D → WB2 §1 (`#wb2-section-1`), named "Dropout as adaptive L^2" (now 1.D.6).
- §2: WB2 §4.E → WB2 §1 (`#wb2-section-1`), named "Why flatness could matter, and why sharpness needs a metric" (now 1.D.7).
- WB6 §4 (critical periods), WB6 §11 and WB6 Exercise 11.2: checked, unchanged and still correct.

**Standalone sections.**
- Each `##` section now opens with its question in plain words (§1 why dropout's test-time rule works; §2 in what precise sense a trained network is robust; §3 what a trained denoiser computes; §4 whether *when* training sees something matters; §5 already did).
- Day map: a one-line symbol key ($L,\eta,B,S,w$) before the table; "the WGAN parable" → "a parable about choosing the metric"; td2_6 row slightly unpacked (claim boundary kept); the cryptic "Flows and SDEs are one story" route note replaced with a plain sentence.
- §1: $n$ and $p$ defined in the pitch; $\mu=\mathbb ES$ in Ex 1.2; unused $m$ dropped; the NWGM is a probability because of the normalization (the old sentence gave symmetry as the reason); $\mathrm{NWGM}(f(S))=f(\mathbb ES)$ in the Ex 1.1 solution; code lead-in defines $E,V$ and is split into short sentences.
- §2: a local "Notation" line ($x^i=A_i\phi(x^{i-1})$, $\phi$, $x^0$); $\eta$ flagged as noise, not a learning rate, in "What you need"; one-phrase gloss of "layer cushion" at first use; the basin-size sentence now says where basin size does enter (the Dziugaite–Roy reading); compressed tail of the Ex 2.1(2) solution cut to one sentence; code lead-in added.
- §3: "The local picture" split into "The optimal denoiser" (defines $x\in\mathbb R^d$, $\sigma$, $p_\sigma$, the Jacobian notation; says Ex 3.1 checks the Jacobian and MSE identities) and "What a bias-free network computes"; Ex 3.1 restates the observation model and warns that its $\lambda_k$ are data variances, not the shrinkage factors $\lambda_k(y)$; code lead-in added; the Kadkhodaie blurb glosses $C^\alpha$ contours and bandlets; Belrose blurb's dangling "the same question" fixed.
- §5: the five levels named in the pitch; WB6 §11 pointer names the two papers; filler sentence cut; an "Abbreviations" line (SFT, RLVR, LoRA, eNTK, $\pi_0,\pi$, advantage).
- Sibling references made into links: td2_4 §4.2, td2_6 §6.5, td2_7 §7.3, td1 §1, td2_6.

**Error fixed.** The Baldi–Sadowski reading blurb was a non-raw Python string, so `\tfrac12` became a tab and the page showed "p=	frac12". Now a raw string. Also, "This notebook" said three topics have a code check; it is four (§1, §2, §3, §5).

**Critic (Claude Opus, fresh).** Read the page entering from the day map and three sidebar sections. Fourteen findings; all applied except: (13) trimming the K-FAC/empirical-Fisher clause at the end of the Ex 5.2 solution, kept because it was a round-2 correctness fix; (14) renaming the data variances in §3 to avoid the double use of $\lambda_k$, handled with a one-line warning instead, to keep the code's `lam` consistent. It confirmed all MDL and sibling section references and found no leftover two-file wording after the edits above.

**Checks.** `python td/build_td2_8_further_readings.py`, `--check` and `python td/checks/td_crosslinks.py` pass (54 links, 0 bad; an earlier run flagged one link in td1's builder, not this one, which has since cleared).
