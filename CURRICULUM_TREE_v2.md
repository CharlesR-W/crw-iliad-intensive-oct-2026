# MDL + TD as choose-your-own-adventure trees, v2 (2026-10-07)

Supersedes the timetables in CURRICULUM_DRAFT_v1.md. Links for every node are in v1 §4; this file names papers and points at the v1 section letter.

## Day shape (both days)

| Time | Block |
|---|---|
| 10:00–10:25 | **Lecture A**: the day's question, and what the trunk is for |
| 10:30–12:00 | **Trunk** (everyone, one workbook) |
| 12:00–13:00 | lunch |
| 13:00–13:25 | **Lecture B**: the second question, then a 10-minute *tour of the branches* (one mystery per branch, so people can choose) |
| 13:30–17:00 | **Branches**: pick a path; guideline is 3 nodes, at least one workbook; instructor floats |
| 17:00–17:50 | **Convergence**: people who took the same branch compare cards, then one card per branch to the room |
| 17:50–18:00 | close |

Node format: **type** (W = workbook, R = reading cluster, W+R = short exercise plus readings) · **time** · **needs** · *what you can do afterwards* · materials · **card question** (what you report at 17:00).

Rule of thumb for the tree: a short mandatory trunk, branches at most three deep, every branch ends in a card question, and at least two branches cross-link to the other day.

---

## Tree 1: Mysteries of Deep Learning

**Lecture A (10:00).** Why did AI become deep learning? History as a sequence of bets about what intelligence is; each bet implied what alignment would be; bitter lesson; "the model class won and we don't know why". Ends: "the trunk gives you the one example that every branch comes back to."

### Trunk M0 — The worked example (W, 90 min, everyone)
Linear regression carried through: population vs empirical risk; generalization gap vs excess risk (both named); bias–variance and estimator variance ∝ p/n; one page on the *shape* of a capacity bound and what Zhang et al. does and does not kill; UAT in one line; regularization three ways (prior / Lagrange multiplier / spectral filter), early stopping as spectral filter; the regime: fixed p, n→∞ vs p/n = γ, the interpolation peak at γ = 1 in ridgeless least squares, removed by ridge or early stopping.
*Afterwards:* you can derive a gap, say which estimand a tool is about, and explain double descent in one line.
Materials: Hastie et al.; Belkin et al.; Nakkiran et al.; Dinh et al. (v1 §B, §E).

**Lecture B (13:00).** What is the LLM paradigm and what does it tell us about intelligence? Why generative, text, next-token, scale; a hundred flowers → transformer; universality evidence vs where parametrization matters; "inductive bias" as a place to start explaining. Then the 10-minute tour of M1–M7.

### Branches

**M1 — The generalization audit** (W, 75 min, needs M0)
Guided derivation: excess risk (1/2n)Tr(H⁻¹C), optimism (1/n)Tr(H⁻¹C), assumptions stated; C = H iff well-specified NLL at θ*, C = σ²H for squared loss; three estimands named (single-net excess risk / Bayes-ensemble / posterior-predictive). Then the precomputed sweep: predict which measures track the gap across which axes, then look.
*Afterwards:* you can say what "parameter counting" assumes and which generalization measures survive confounding.
Materials: Thomas et al. 2020; Murata–Yoshizawa–Amari; Jiang et al.; Dziugaite et al. (v1 §A).
**Card:** which measure survived which axis, and one it failed on.
→ **M1b — Is SGD a Bayesian sampler?** (R, 45 min) Mandt; Mingard; Wenzel cold posteriors; Izmailov HMC; Lotfi marginal likelihood; Vaintrob "Learning zero". **Card:** the strongest evidence for and against. *Cross-links to TD T3.*
→ **M1c — Noise knows curvature** (W+R, 45 min) Compute C and H diagonals on the sweep; Zhang–Yang–Tu C_ii ∝ H_ii^γ. **Card:** your fitted γ. *This is the TD hook.*

**M2 — Scaling and the joint limit** (W+R, 60 min, needs M0)
Fit a scaling law on a random-features model with power-law data spectrum (Maloney–Roberts–Sully); see the exponent come from the spectrum; Bahri's four regimes; Chinchilla scoped as a compute-allocation fit.
*Afterwards:* you can say what a scaling exponent does and does not explain.
Materials: v1 §B; "Power laws are not enough".
**Card:** does a scaling law constrain what is learned? One argument each way.
→ **M2b — Emergence** (R, 40 min) Steinhardt "More is different"; Schaeffer mirage; Olsson induction heads as a phase change. **Card:** is emergence a property of the model or of the metric?

**M3 — Universality** (W+R, 60 min, needs M0)
Short exercise: stitch two independently trained small nets (Bansal–Nakkiran–Barak) or compute CKA/SVCCA layerwise; readings on platonic representations, linear representations, linear mode connectivity / permutation symmetry.
*Afterwards:* you can state the universality evidence and its scope.
Materials: v1 §D.
**Card:** what would falsify the platonic representation hypothesis?
→ **M3b — Architecture vs parametrization** (R, 40 min) muP / muTransfer; Vyas et al. consistency across widths; Chizat lazy depends on scaling. **Card:** "architecture doesn't matter": true at what scale, in what sense? *Cross-links to TD T1.*

**M4 — Inductive bias and symmetry** (W+R, 60 min, needs M0)
Exercise: CNN vs MLP sample efficiency on a translated/rotated toy; GDL blueprint; then the competing accounts: Furman "DL as program synthesis", Wilson "not so mysterious", Hänni–Vaintrob circuit prior, Solomonoff/Occam.
*Afterwards:* you can say what an "inductive bias" claim must mechanize to count as an explanation.
Materials: v1 §D.
**Card:** pick one of the three accounts and name one observation it predicts that the others don't.

**M5 — The LLM paradigm and intelligence** (R, 60 min, no prereq)
Simulators; Waluigi; Physics of LMs 3.1; Byrnes brain-like AGI; Wentworth retarget-the-search; LeCun cake.
*Afterwards:* you can state what each paradigm thought the alignment problem was, and what the LLM paradigm thinks it is.
**Card:** one way the simulator frame changes an alignment plan, and one way it doesn't.

**M6 — Mysteries myriad** (R, 45–90 min, no prereq; take 1–2 papers)
Stitching; SVCCA bottom-up; critical periods; lottery tickets; GD in a tiny subspace (with Song et al. intervention); adversarial features; grokking; distillation/pruning; superposition.
**Card:** claim / evidence / what would change my mind / does it survive scale.

**M7 — The compression route** (R, 45 min, needs M0)
Hochreiter–Schmidhuber flat minima as MDL; Arora compression bounds; Dziugaite–Roy nonvacuous PAC-Bayes; Lotfi nonvacuous bounds for LLMs.
**Card:** why are compression bounds the only nonvacuous ones at scale, and what does that say about the others?

Suggested paths: *theory-minded*: M0 → M1 → M1b → M7. *empirics-minded*: M0 → M3 → M2 → M6. *alignment-minded*: M0 → M5 → M4 → M6. *heading into TD*: M0 → M1 → M1c → M3b.

---

## Tree 2: Training Dynamics

**Lecture A (10:00).** What we know about trajectories, each fact scoped: bulk + outliers; GGN ≈ H; top-subspace confinement (and Song et al.); progressive sharpening to 2/η under GD, Edge of Stochastic Stability under SGD; lazy vs rich depends on scaling; increasing complexity / moment bias; critical periods; grokking as late feature learning. Ends: "the trunk makes you measure all of these on one model."

### Trunk T0 — One common experiment (W, 90 min, everyone)
Train a small MLP/CNN. Track λ_max vs 2/η (EoS, catapults); compute C and H along training and test C_ii ∝ H_ii^γ; watch the eNTK rotate toward the data; log the loss/complexity order (linear probe first). On the quadratic basin: GD is a linear ODE, SGD is an OU process; stationary covariance set by η, B, H, C.
*Afterwards:* every branch is an explanation of something you just saw.
Materials: Cohen EoS; Andreyev–Beneventano; Gur-Ari; Zhang–Yang–Tu; "Selection for selectability" (v1 §E, §A).

**Lecture B (13:00).** Pick your metric, then your equation. GAN→WGAN (KL infinite, JS saturates on disjoint supports; Wasserstein respects the space); the analogous move in parameter space, labelled as analogy: natural gradient, sign descent, Muon, modular norm, muP. GD as flow + corrections; SGD as SDE and when it is valid. Then the 10-minute tour of T1–T7.

### Branches

**T1 — Solvable models** (W, 75 min, needs T0)
Deep linear nets: Saxe mode ODEs, balancedness conservation, saddle-to-saddle, low-rank bias; lazy limit as frozen NTK with its conditions stated.
*Afterwards:* you can say what a solvable model buys and what it silently assumes.
Materials: Saxe; Achour et al.; Jacot; Lee et al.; PDLT ch. 1–2 (v1 §E).
**Card:** one prediction of the deep-linear model that held in T0, one that didn't.
→ **T1b — Lazy to rich** (W+R, 60 min) Chizat; Tu–Aranguri–Jacot; Kunin "get rich quick"; grokking link (Power; Nanda). **Card:** what controls the transition, and is it width?
→ **T1c — Infinite width done right** (R, 45 min) muP; Bordelon–Pehlevan DMFT; Vyas et al. **Card:** at what width is a real net "near the limit"? *Cross-links to MDL M3b.*

**T2 — Edge of stability** (W, 75 min, needs T0)
IGR by backward error analysis; progressive sharpening; EoS; self-stabilization; central flow vs rod flow on the T0 model; Edge of Stochastic Stability and the stochastic sharpness gap.
*Afterwards:* you can predict λ_max from η and say why large learning rates are implicit regularizers.
Materials: Barrett–Dherin; Smith et al.; Cohen ×3; Damian; Regis–Chewi; Andreyev–Beneventano (v1 §E).
**Card:** does the central flow or the rod flow fit your T0 run better, and where do they disagree?
→ **T2b — Schedules and warmup** (R, 40 min) Wen et al. river valley; catapults (Lewkowycz); warmup. **Card:** what the WSD decay phase reveals and why constant LR hides it.

**T3 — SDE, noise, and Bayes** (W+R, 75 min, needs T0)
Modified equations; SGD as SDE and validity (Li–Malladi–Arora); η/B scaling and its failure (Ziyin et al.); OU on the basin: Boltzmann iff C ∝ H; with C ∝ H^γ it is no tempered posterior; power-law escape (Mori); cold posteriors; Mingard; Mandt; HMC posteriors. Bonus: Morse–Bott/Witten concentration (CRW writeup).
*Afterwards:* you can state exactly when SGD samples a posterior and what the practical quantity for a trained net is.
Materials: v1 §A, §E.
**Card:** one sentence on what SGD's stationary law is for your T0 γ. *Cross-links to MDL M1b, M1c.*

**T4 — Geometry of optimizers** (W, 75 min, needs T0)
WGAN parable worked (Arjovsky–Bottou toy: disjoint supports); natural gradient; sign descent as ∞-norm steepest descent and how Adam differs; Muon from scratch (Newton–Schulz) and compare update norms and λ_max against SGD/Adam on the T0 model; modular norm and LR transfer.
*Afterwards:* you can derive an optimizer from a norm and say which norm Adam/Muon approximate.
Materials: v1 §C; KL/geometric-naturalness post.
**Card:** which norm did Muon beat Adam under on your model, and why would that change with width?

**T5 — Spectra and the Hessian** (W+R, 60 min, needs T0)
Bulk + outliers (Papyan, Sagun, Ghorbani); Fort–Ganguli toy; GGN vs Hessian; top-subspace confinement and Song et al.; damping and K-FAC structure as needed by Data Attribution.
*Afterwards:* you can read a Hessian spectrum and say what an influence function will see.
**Card:** how many outliers, what are they, and what does damping do to them? *Cross-links to DA day.*

**T6 — What is learned first** (W+R, 60 min, needs T0)
Moment bias (Refinetti), increasing complexity (Nakkiran), spectral bias (Rahaman, Xu), geometry-adaptive harmonic basis on a toy denoiser (Kadkhodaie); critical periods (Achille) and loss of plasticity (Dohare).
*Afterwards:* you can say why early training looks linear and what that implies for fine-tuning.
**Card:** the order in which your T0 model learned, and one intervention that would reorder it.

**T7 — Fine-tuning dynamics for alignment** (R, 60 min, no prereq beyond T0)
RL updates sparse but full-rank (Mukherjee); rank-1 dominance (Cai); RLVR off the principals (Zhu); RL's Razor (Shenfeld); LoRA without regret; "Some interesting papers on RLVR"; KL leash and instability.
*Afterwards:* you can say what RL fine-tuning changes in the eigenbasis and why it forgets less.
**Card:** propensity or lability: which does RL change, and what evidence would distinguish them?

Suggested paths: *analytic*: T0 → T1 → T1b → T3. *phenomenology*: T0 → T2 → T2b → T5. *geometry*: T0 → T4 → T3. *alignment*: T0 → T7 → T6 → T5. *heading into DA*: T0 → T5 → T3.

---

## Build list
- One map page per day (tree diagram, nodes clickable to the node card). Could be a single HTML page.
- Trunk notebooks M0 and T0 (Colab, CPU, <2 min per cell).
- Branch notebooks: M1, M1c, M2, M3, M4, T1, T1b, T2, T3, T4, T5, T6 (M1 needs the precomputed sweep from Modal).
- Card template (claim / evidence / what would change my mind / survives scale?).
- Reading clusters as tiered lists (core / choose-one / rabbit hole) pulled from v1 §4.
