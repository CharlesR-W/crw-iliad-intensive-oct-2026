# MDL + TD curriculum, draft v1 (2026-10-06)

Working draft produced with Claude; Codex gave an adversarial critique of v0 (summary at the end). Everything here is a proposal for discussion, not a decision.

Constraints used: 10:00–18:00, lunch 12–13 (7h contact). Cohort: grad-level math/physics/TCS, learning theory NOT assumed, Python uneven. Berkeley order was SLT → MDL → TD → Data Attribution (DA), so students arrive at MDL with the SLT frame fresh. Instructor style: short lectures (~30 min), then students read/work; tiered reading lists.

---

## 0. The spine (what ties both days together)

**One worked example carried through everything: linear regression / the quadratic basin.**
- Day 1 WB1: population vs empirical risk, bias–variance, estimator variance, ridge as prior / Lagrange multiplier / spectral filter, early stopping as spectral filter, the interpolation peak at p/n→1.
- Day 1 WB2: excess risk of the empirical minimizer, (1/2n)·Tr(H⁻¹C); C = H only for well-specified NLL at the true parameter (C = σ²H for squared loss).
- Day 2 Notebook A: the same quadratic basin is the lazy/NTK regime; GD on it is a linear ODE; SGD on it is an OU process whose stationary covariance is set by η, B, H and C.
- Day 2 Lec 5: if C ∝ H the stationary law is Boltzmann (a tempered posterior); if C ∝ H^γ with γ≠1 (Zhang–Yang–Tu 2026), it is not any tempered posterior. That is the whole "SGD vs Bayes" question in one line, and students can derive it.

**Two estimands, kept separate throughout** (Codex rightly insisted): excess population loss of the single trained net vs. Bayes-ensemble / posterior-predictive error. Name both every time; let students notice which one the practical tools (regularization, early stopping, flatness) are actually about.

---

## 1. Day 1: Mysteries of Deep Learning

Thesis: *Why did AI become synonymous with deep learning, and what does the LLM paradigm say about intelligence? Which "mysteries" are genuine, and which are artifacts of asking fixed-p, n→∞ questions in a p/n = O(1) world?*

| Time | Block | Notes |
|---|---|---|
| 10:00–10:30 | **Lec 1** — Why did AI become deep learning? | History as a sequence of bets about what intelligence is (GOFAI/search → control/optimization → statistical learning → scale). Each bet implied what "alignment" would mean. Bitter lesson. Why statistical learning at all (NFL as "you must bet on something", not an impossibility). End: the model class won and we don't know why; today is about what we actually know. |
| 10:30–11:15 | **Reading 1** + pairs | Core: Bitter Lesson. Choose one: Simulators (first half) / Byrnes brain-like AGI intro / Wentworth retarget-the-search / LeCun cake / Steinhardt "More is different". 10 min pair discussion: "what did each paradigm think the alignment problem was?" |
| 11:15–12:00, 13:00–14:30 | **WB1** — Classical learning theory and where it breaks | See §1.1. |
| 14:30–14:45 | break | |
| 14:45–15:15 | **Lec 2** — What is the LLM paradigm and what does it tell us about intelligence? | Why generative; why text; why next-token; why scale; a hundred flowers → transformer; universality evidence (platonic, linear reps, stitching, mode connectivity, SVCCA) vs. where architecture/parametrization does matter (muP, feature learning); "inductive bias" as a place to start explaining, not stop. Last 8 min: simulators/personae, ontology problem, natural abstractions. |
| 15:15–15:45 | **Reading 2** | Core: Simulators (rest) or Platonic rep. Choose one: Physics of LMs 3.1 / Waluigi / Wilson "not so mysterious" / Furman "program synthesis" / Vaintrob "learning zero". Present Wilson and Furman as serious competing accounts, not as foils. |
| 15:45–17:00 | **WB2** — The generalization-gap audit | See §1.2. |
| 17:00–17:45 | **Mystery cards** | Each pair has, during the day, picked one "myriad mystery" paper. 3 min each (cap ~8 pairs): claim / evidence / what would change my mind / does it survive scale. Replaces "read until you find something cool". |
| 17:45–18:00 | wrap | Hook to tomorrow: we measured C and H; they are not proportional. Why, and what does SGD do about it? |

### 1.1 WB1 — Classical learning theory and where it breaks (~2h15)
Pen+paper with small Colab cells; one worked regression example throughout.
1. ERM; population vs empirical risk; generalization gap vs excess risk (define both, keep both).
2. Bias–variance on linear regression (derive). Estimator sampling distribution; variance ∝ p/n.
3. One page on the *shape* of a capacity bound (no VC machinery), then Zhang et al. random labels: what exactly it kills (unrestricted-capacity explanations and explicit regularization as the explanation), and what it doesn't.
4. UAT in one line: existence, not mechanism.
5. Regularization three ways on the same problem: MAP/prior; Lagrange multiplier; ridge and early stopping as spectral filters on H's eigenvalues. Why "flatness" could matter: second-order expansion of population loss at a minimum (and Dinh et al.: naive flatness is not reparametrization invariant).
6. **The regime.** Classical asymptotics: n→∞ at fixed p. Modern practice: p/n = O(1); for LLMs, D ∝ N (Chinchilla, scoped as a compute-allocation fit). Exercise: ridgeless least squares with p/n = γ; see the variance peak at γ=1 (Hastie et al.); ridge or early stopping removes the divergence. Then Belkin/Nakkiran double descent as the nonlinear phenomenology.
7. Scaling laws as the joint limit: exponents from the data spectrum in solvable models (Maloney–Roberts–Sully; Bahri et al.'s four regimes; Bordelon dynamical model). Pointer: "Power laws are not enough".
Cut from the notes: VC/PAC machinery, planning-as-inference (one pointer in Lec 2 only).

### 1.2 WB2 — The generalization-gap audit (~75 min)
Guided fill-in derivation + analysis of a precomputed sweep (generate on Modal; ship as a parquet with documented measures, repeated seeds, matched comparisons).
1. Excess population loss of the empirical minimizer, regular case: E[L_pop(θ̂)−L_pop(θ*)] ≈ (1/2n)·Tr(H⁻¹C); train–test optimism is (1/n)·Tr(H⁻¹C) (factor of 2; Thomas et al. 2020; Murata–Yoshizawa–Amari 1994 NIC). State the assumptions (fixed p, invertible H, consistency) explicitly: this is *not* a general formula for an overparametrized net.
2. C = H iff well-specified NLL at the true parameter (Fisher = Hessian); then it collapses to p/2n (AIC, "parameter counting"). Squared loss: C = σ²H. Misspecification breaks it; SGD is not needed to break it.
3. Contrast the estimand with the Bayes-ensemble error (what SLT's RLCT predicts) and posterior-predictive error. Three estimands, named.
4. Measure: on the sweep, students *predict* which measures (Tr(H⁻¹C) proxies, sharpness, norms, margins, PAC-Bayes-style) track gap across which hyperparameter axes, then look (Jiang et al. protocol; Dziugaite et al. robustness critique). Debrief: which survived, which were confounded.
5. Hook: in the sweep, C vs H along the diagonal looks like C_ii ∝ H_ii^γ, γ∈[1,2] (Zhang–Yang–Tu 2026). Leave as a question.

### 1.3 Material that is new relative to the notes
- The p/n = O(1) regime and ridgeless interpolation as the one-line double-descent explanation.
- Compression/MDL route to generalization (Hochreiter–Schmidhuber flat minima as MDL; Arora et al. compression bounds; Lotfi et al. nonvacuous bounds) — a non-Bayesian route to the same claims.
- Emergence/"more is different" and the emergent-abilities-mirage debate (Schaeffer et al. 2023 — not yet linked).
- Universality nuance: architecture matters less, parametrization matters (muP, Vyas et al. consistency across widths).
- Vaintrob's "Learning zero" and "circuit prior" posts as the SLT-adjacent view that is itself critical of RLCT-in-practice.
- Marginal-likelihood skepticism (Lotfi et al. 2022).
- Codex suggested, worth a reading pointer rather than a block: evaluation under selection and distribution shift; reward-proxy optimization.

---

## 2. Day 2: Training Dynamics

Thesis: *The trajectory, not just the landscape. Two ways to study it: exactly solvable models, and phenomenology with math as a flashlight. Everyone does one common experiment; the analytic track is an optional extension on the same data.*

| Time | Block | Notes |
|---|---|---|
| 10:00–10:30 | **Lec 3** — What we know about training trajectories | Empirical facts, each scoped: Hessian = bulk + k outliers (Papyan, Sagun, Ghorbani); GGN ≈ Hessian; gradient confined to the top subspace (Gur-Ari; and Song et al. 2024 showing projection interventions complicate the story); progressive sharpening to 2/η under full-batch GD (Cohen), Edge of Stochastic Stability for SGD (Andreyev–Beneventano); lazy vs rich depends on scaling not just width (Chizat); increasing complexity / moment bias (Refinetti, Nakkiran); spectral bias; critical periods and loss of plasticity; grokking as late feature learning. Then: the two ways to study this and what each buys. |
| 10:30–12:00 | **Notebook A** — One common experiment | Train a small MLP/CNN; track λ_max vs 2/η (EoS, catapults); compute C and H along training, test C_ii ∝ H_ii^γ; watch the eNTK rotate toward the data (feature learning as antenna rotation; Selection for Selectability). Analytic extension on the same model: deep linear Saxe mode ODEs, balancedness, saddle-to-saddle; lazy→rich (Tu–Aranguri–Jacot; Kunin "get rich quick"). |
| 13:00–13:30 | **Lec 4** — Pick your metric, then your equation | Opener: GAN→WGAN. On disjoint supports KL is infinite and JS saturates at log 2 with zero gradient; Wasserstein respects sample-space geometry (Arjovsky–Bottou). Lesson: choose a metric that respects the space (KL/geometric-naturalness post). Then the *analogous* move in parameter space, labelled as analogy: natural gradient (Fisher), sign descent as ∞-norm steepest descent (Adam ≈ this only without EMA; Bernstein anthology), Muon as spectral-norm steepest descent, modular norm; muP as scale-consistent parametrization. Then GD as flow + corrections: modified equations / IGR, SGD as SDE and when the approximation is valid (Li–Malladi–Arora), η/B scaling, EoS and central/rod flows as the discrete-time story. |
| 13:30–15:00 | **Notebook B** — Choose one of two | B1: IGR + EoS + central flow on the Notebook-A model. B2: Muon from scratch; compare update norms and λ_max. Optional extension: moment bias / geometry-adaptive harmonic basis on a toy denoiser (Kadkhodaie). Dropout-as-geometric-mean moved to readings. |
| 15:00–15:15 | break | |
| 15:15–15:45 | **Lec 5** — Is SGD a Bayesian sampler? | OU/Langevin on the quadratic basin: stationary law is Boltzmann iff C ∝ H. Fokker–Planck, flat-minima bias, power-law escape (Mori). What breaks: anisotropic non-equilibrium noise (C ∝ H^γ), cold posteriors (Wenzel), HMC posteriors vs SGD solutions (Izmailov), Mingard "well, almost", Mandt. Practical point: for a trained net the controlling quantity for excess risk is a (noised) inverse Hessian. Morse–Bott/Witten as a bonus reading, not lecture. |
| 15:45–17:00 | **Readings / finish notebooks** | Tiered list below. |
| 17:00–18:00 | **Mystery cards 2 + closing discussion** | "What kinds of explanation do we need?" Levels of description, psychology-from-QFT; which micro details survive coarse-graining; "every measurement has a scale". |

### 2.1 Material new relative to the notes
- Edge of Stochastic Stability (Andreyev–Beneventano 2024/25) and the stochastic sharpness gap (2026); Rod Flow (Regis–Chewi 2026) alongside central flows.
- Progressive sharpening, catapults, warmup; WSD "river valley" picture (Wen et al. 2024) as practical phenomenology.
- η/B linear scaling rule and its failure (Ziyin et al. strength of minibatch noise).
- Weight decay's actual roles in modern training (Andriushchenko et al. 2023 — not yet linked).
- Loss of plasticity / continual learning (Dohare et al. 2024 — not yet linked) as the fine-tuning-relevant face of critical periods.
- Fine-tuning dynamics for alignment: RL updates sparse but full-rank (Mukherjee), rank-1 dominance (Cai), RLVR off the principals (Zhu), RL's Razor / KL (Shenfeld), LoRA without regret; "Some interesting papers on RLVR".
- Symmetry → constraints (Ziyin 2024), flat-minima uniqueness (Mulayoff–Michaeli), linear stability of SGD (Wu–Ma–E; Ma–Ying).

### 2.2 Coordination with Data Attribution
- TD owns: Hessian/GGN spectrum facts, damping, K-FAC-ish structure, SDE/unrolling basics.
- DA owns: influence functions, TracIn/unrolling formula, Bayesian IFs.
- Codex's point, which I think is right: a Bayesian IF for a specified posterior does not require SGD to sample that posterior, so there is no real conflict with Lec 5, only a distinction of estimands. Agree with Louis on wording so students hear the same three estimands on both days.

---

## 3. Open questions for CRW
1. Does "Deep learning as program synthesis" have to remain the anchor reading? Proposal: keep it as one of the choose-one readings, presented as a live hypothesis.
2. Cohort size (sets how many mystery cards fit); which weekdays; whether Zach still gives a remote lecture.
3. Tooling: Colab, everything CPU-runnable in <2 min; the WB2 sweep precomputed on Modal. I can build the notebooks and the sweep.
4. Whether to merge the two days' reading lists into a single tiered page (core / choose-one / rabbit holes) with a reading-card template.

---

## 4. Links (verified unless marked)

### A. Bayesian / info-criterion generalization-gap arc
- Thomas, Pedregosa, van Merriënboer, Manzagol, Bengio, Le Roux, *On the interplay between noise and curvature and its effect on optimization and generalization* (AISTATS 2020). https://arxiv.org/abs/1906.07774 — TIC gap ≈ (1/N)Tr(H⁻¹C); collapses to AIC d/N when well-specified; separates Fisher, Hessian, gradient covariance.
- Murata, Yoshizawa, Amari, *Network Information Criterion* (IEEE TNN 1994). https://ieeexplore.ieee.org/document/329683 — the NN-era ancestor of TIC.
- Zhang, Yang, Tu, *On the superlinear relationship between SGD noise covariance and loss landscape curvature* (2026). https://arxiv.org/abs/2602.05600 — **the C ∝ H^γ paper**: C ∝ E_p[h_p²]; C_ii ∝ H_ii^γ with 1 ≤ γ ≤ 2.
- Mori, Ziyin, Liu, Ueda, *Power-law escape rate of SGD* (ICML 2022). https://arxiv.org/abs/2105.09557 — assumes C ≈ (2L/B)H; power-law (not exponential) escape.
- Ziyin, Liu, Mori, Ueda, *Strength of minibatch noise in SGD* (ICLR 2022). https://arxiv.org/abs/2102.05375 — exact C for linear regression; C ≠ cH at interpolating/regularized minima.
- Xie, Sato, Sugiyama, *A diffusion theory for deep learning dynamics* (ICLR 2021). https://arxiv.org/abs/2002.03495 — assumes C ∝ H; exponential flat-minima preference.
- Zhu, Wu, Yu, Wu, Ma, *The anisotropic noise in SGD* (ICML 2019). https://arxiv.org/abs/1803.00195
- Jastrzębski et al., *Three factors influencing minima in SGD* (2017). https://arxiv.org/abs/1711.04623 — assumes C = H.
- Yaida, *Fluctuation-dissipation relations for SGD* (ICLR 2019). https://arxiv.org/abs/1810.00004
- Wojtowytsch, *SGD with noise of machine learning type* I & II (2021). https://arxiv.org/abs/2105.01650 , https://arxiv.org/abs/2106.02588
- Jiang, Neyshabur, Mobahi, Krishnan, Bengio, *Fantastic generalization measures and where to find them* (ICLR 2020). https://arxiv.org/abs/1912.02178
- Dziugaite et al., *In search of robust measures of generalization* (NeurIPS 2020). https://arxiv.org/abs/2010.11924
- Mandt, Hoffman, Blei, *SGD as approximate Bayesian inference* (JMLR 2017). https://arxiv.org/abs/1704.04289
- Izmailov, Vikram, Hoffman, Wilson, *What are Bayesian NN posteriors really like?* (ICML 2021). https://arxiv.org/abs/2104.14421
- Wenzel et al., *How good is the Bayes posterior in deep neural networks really?* (ICML 2020). https://arxiv.org/abs/2002.02405 — cold posteriors.
- Mingard, Valle-Pérez, Skalse, Louis, *Is SGD a Bayesian sampler? Well, almost* (JMLR 2021). https://arxiv.org/abs/2006.15191
- Wilson, Izmailov, *Bayesian deep learning and a probabilistic perspective of generalization* (NeurIPS 2020). https://arxiv.org/abs/2002.08791
- Smith, Le, *A Bayesian perspective on generalization and SGD* (ICLR 2018). https://arxiv.org/abs/1710.06451
- Lotfi, Izmailov, Benton, Goldblum, Wilson, *Bayesian model selection, the marginal likelihood, and generalization* (ICML 2022). https://arxiv.org/abs/2202.11678

### B. Proportional asymptotics / scaling
- Hastie, Montanari, Rosset, Tibshirani, *Surprises in high-dimensional ridgeless least squares interpolation* (AoS 2022). https://arxiv.org/abs/1903.08560
- Mei, Montanari, *The generalization error of random features regression* (2019). https://arxiv.org/abs/1908.05355
- Bach, *High-dimensional analysis of double descent for linear regression with random projections* (2023). https://arxiv.org/abs/2303.01372
- Belkin, Hsu, Ma, Mandal, *Reconciling modern ML practice and the bias–variance trade-off* (PNAS 2019). https://arxiv.org/abs/1812.11118
- Nakkiran et al., *Deep double descent* (2019). https://arxiv.org/abs/1912.02292
- Atanasov, Zavatone-Veth, Pehlevan, *Scaling and renormalization in high-dimensional regression* (2024). https://arxiv.org/abs/2405.00592
- Maloney, Roberts, Sully, *A solvable model of neural scaling laws* (2022). https://arxiv.org/abs/2210.16859
- Bahri, Dyer, Kaplan, Lee, Sharma, *Explaining neural scaling laws* (PNAS 2024). https://arxiv.org/abs/2102.06701
- Hoffmann et al., *Chinchilla* (2022). https://arxiv.org/abs/2203.15556
- Kaplan et al., *Scaling laws for neural language models* (2020). https://arxiv.org/abs/2001.08361
- Bordelon, Atanasov, Pehlevan, *A dynamical model of neural scaling laws* (ICML 2024). https://arxiv.org/abs/2402.01092

### C. GAN → WGAN and the choose-your-metric lineage
- Goodfellow et al., *GAN* (2014). https://arxiv.org/abs/1406.2661
- Arjovsky, Bottou, *Towards principled methods for training GANs* (ICLR 2017). https://arxiv.org/abs/1701.04862
- Arjovsky, Chintala, Bottou, *Wasserstein GAN* (2017). https://arxiv.org/abs/1701.07875
- Gulrajani et al., *Improved training of WGANs* (NeurIPS 2017). https://arxiv.org/abs/1704.00028
- Amari, *Natural gradient works efficiently in learning* (Neural Comp. 1998). https://doi.org/10.1162/089976698300017746
- Martens, *New insights and perspectives on the natural gradient method* (JMLR 2020). https://arxiv.org/abs/1412.1193
- Bernstein, Newhouse, *Old optimizer, new norm: an anthology* (2024). https://arxiv.org/abs/2409.20325
- Bernstein, Newhouse, *Modular duality in deep learning* (2024). https://arxiv.org/abs/2410.21265
- Jordan et al., *Muon* (blog, Dec 2024). https://kellerjordan.github.io/posts/muon/
- Large, Liu, Huh, Bahng, Isola, Bernstein, *Scalable optimization in the modular norm* (NeurIPS 2024). https://arxiv.org/abs/2405.14813
- CRW, *KL divergence, geometric naturalness, and optimal transport*. https://charlesr-w.github.io/crw-blog/KL-Divergence-Geometric-Naturalness-and-Optimal-Transport/

### D. Mysteries classics and alignment framing
- Zhang et al., *Understanding deep learning requires rethinking generalization* (ICLR 2017). https://arxiv.org/abs/1611.03530
- Wilson, *Deep learning is not so mysterious or different* (ICML 2025). https://arxiv.org/abs/2503.02113
- Sutton, *The bitter lesson* (2019). http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- janus, *Simulators* (2022). https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators
- Nardo, *The Waluigi effect* (2023). https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post
- Furman, *Deep learning as program synthesis* (2026). https://www.lesswrong.com/posts/Dw8mskAvBX37MxvXo/deep-learning-as-program-synthesis-1
- Allen-Zhu, Li, *Physics of language models* hub. https://physics.allen-zhu.com/ ; Part 3.1 https://arxiv.org/abs/2309.14316
- Huh, Cheung, Wang, Isola, *The Platonic representation hypothesis* (ICML 2024). https://arxiv.org/abs/2405.07987
- Park, Choe, Veitch, *The linear representation hypothesis* (ICML 2024). https://arxiv.org/abs/2311.03658
- Elhage et al., *Toy models of superposition* (2022). https://transformer-circuits.pub/2022/toy_model/index.html
- Olsson et al., *In-context learning and induction heads* (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Raghu et al., *SVCCA* (NeurIPS 2017). https://arxiv.org/abs/1706.05806
- Bansal, Nakkiran, Barak, *Revisiting model stitching* (NeurIPS 2021). https://arxiv.org/abs/2106.07682
- Achille, Rovere, Soatto, *Critical learning periods in deep networks* (ICLR 2019). https://arxiv.org/abs/1711.08856
- Frankle, Carbin, *Lottery ticket hypothesis* (ICLR 2019). https://arxiv.org/abs/1803.03635
- Frankle, Dziugaite, Roy, Carbin, *Linear mode connectivity and the lottery ticket hypothesis* (ICML 2020). https://arxiv.org/abs/1912.05671
- Gur-Ari, Roberts, Dyer, *Gradient descent happens in a tiny subspace* (2018). https://arxiv.org/abs/1812.04754
- Song et al. (2024), projection interventions on the top-Hessian subspace (Codex-cited; not yet checked). https://arxiv.org/abs/2405.16002
- Ilyas et al., *Adversarial examples are not bugs, they are features* (NeurIPS 2019). https://arxiv.org/abs/1905.02175
- Bronstein, Bruna, Cohen, Veličković, *Geometric deep learning* (2021). https://arxiv.org/abs/2104.13478
- Wolpert, *The lack of a priori distinctions between learning algorithms* (1996). https://doi.org/10.1162/neco.1996.8.7.1341
- Shalev-Shwartz, Ben-David, *Understanding Machine Learning* (free PDF). https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/
- Prince, *Understanding Deep Learning* (free). https://udlbook.github.io/udlbook/
- Byrnes, *Intro to brain-like AGI safety*. https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8
- Wentworth, *Just retarget the search*. https://www.lesswrong.com/posts/w4aeAFzSAguvqA5qu/how-to-go-from-interpretability-to-alignment-just-retarget
- LeCun cake: NIPS 2016 talk https://www.youtube.com/watch?v=Ount2Y4qxQo ; Ruder's writeup with slide https://www.ruder.io/highlights-nips-2016/
- Steinhardt, *More is different for AI* (2022). https://bounded-regret.ghost.io/more-is-different-for-ai/
- Power et al., *Grokking* (2022). https://arxiv.org/abs/2201.02177 ; Nanda et al., *Progress measures for grokking* (ICLR 2023). https://arxiv.org/abs/2301.05217
- Vaintrob, *Learning zero, and what SLT gets wrong about it* (2026). https://www.lesswrong.com/posts/5hKgJy8rcqnM9ntp2/learning-zero-and-what-slt-gets-wrong-about-it
- Hänni, Vaintrob, *A circuit prior in NN-Bayes* (2026). https://www.lesswrong.com/posts/SqDHeuycNkERurtSc/a-circuit-prior-in-nn-bayes
- Vaintrob, *Basics of Bayesian learning* (2025). https://www.lesswrong.com/posts/vXPKz2nbrYt2gXA9g/basics-of-bayesian-learning
- CRW, *Power laws are not enough*; *What if not circuits*; *Selection for selectability*; *Every measurement has a scale*. https://charlesr-w.github.io/crw-blog/
- Not yet linked: Schaeffer et al. emergent-abilities mirage (2023); Lotfi et al. nonvacuous bounds for LLMs (2024); Muennighoff data-constrained scaling (2023); Entezari / Ainsworth "Git Re-Basin" permutation symmetry.

### E. Training dynamics
(E1 items from recall by the link agent, marked "not yet checked" there; the IDs below are the standard ones and I believe them correct, but verify before publishing a reading list.)
- Saxe, McClelland, Ganguli, *Exact solutions ... deep linear networks* (2013). https://arxiv.org/abs/1312.6120
- Roberts, Yaida, Hanin, *Principles of deep learning theory* (2021). https://arxiv.org/abs/2106.10165
- Jacot, Gabriel, Hongler, *NTK* (NeurIPS 2018). https://arxiv.org/abs/1806.07572
- Chizat, Oyallon, Bach, *On lazy training* (NeurIPS 2019). https://arxiv.org/abs/1812.07956
- Goh, *Why momentum really works* (Distill 2017). https://distill.pub/2017/momentum/
- Cohen et al., *GD typically occurs at the edge of stability* (ICLR 2021). https://arxiv.org/abs/2103.00065
- Cohen et al., *Adaptive gradient methods at the EoS* (2022). https://arxiv.org/abs/2207.14484
- Cohen, Damian, Talwalkar, Kolter, Lee, *Central flows* (ICLR 2025). https://arxiv.org/abs/2410.24206
- Regis, Chewi, *Rod Flow* (2026). https://arxiv.org/abs/2602.01480 ; *Edge Flow* (2026). https://arxiv.org/abs/2606.18080
- Andreyev, Beneventano, *Edge of Stochastic Stability* (2024/25). https://arxiv.org/abs/2412.20553 ; *SGD at the EoS: the stochastic sharpness gap* (2026). https://arxiv.org/abs/2604.21016
- Damian, Nichani, Lee, *Self-stabilization* (ICLR 2023). https://arxiv.org/abs/2209.15594
- Barrett, Dherin, *Implicit gradient regularization* (ICLR 2021). https://arxiv.org/abs/2009.11162
- Smith, Dherin, Barrett, De, *On the origin of implicit regularization in SGD* (ICLR 2021). https://arxiv.org/abs/2101.12176
- Li, Tai, E, *Stochastic modified equations* (2017/2019). https://arxiv.org/abs/1511.06251 , https://arxiv.org/abs/1811.01558
- Li, Malladi, Arora, *On the validity of modeling SGD with SDEs* (NeurIPS 2021). https://arxiv.org/abs/2102.12470
- Wen et al., *Understanding WSD learning rates: a river valley loss landscape perspective* (ICLR 2025). https://arxiv.org/abs/2410.05192
- Papyan, *Traces of class/cross-class structure* (JMLR 2020). https://arxiv.org/abs/2008.11865 ; *Full spectrum of deepnet Hessians at scale*. https://arxiv.org/abs/1811.07062
- Sagun et al., *Empirical analysis of the Hessian* (2017). https://arxiv.org/abs/1706.04454
- Ghorbani, Krishnan, Xiao, *Hessian eigenvalue density* (ICML 2019). https://arxiv.org/abs/1901.10159
- Fort, Ganguli, *Emergent properties of the local geometry of neural loss landscapes* (2019). https://arxiv.org/abs/1910.05929
- Rahaman et al., *Spectral bias* (ICML 2019). https://arxiv.org/abs/1806.08734 ; Xu et al., *Frequency principle*. https://arxiv.org/abs/1901.06523
- Kadkhodaie, Guth, Simoncelli, Mallat, *Geometry-adaptive harmonic representations* (ICLR 2024). https://arxiv.org/abs/2310.02557
- Refinetti, Ingrosso, Goldt, *Distributions of increasing complexity* (ICML 2023). https://arxiv.org/abs/2211.11567
- Nakkiran et al., *SGD learns functions of increasing complexity* (NeurIPS 2019). https://arxiv.org/abs/1905.11604
- Baldi, Sadowski, *Understanding dropout* (NeurIPS 2013). https://papers.nips.cc/paper/2013/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html (UNSURE on hash)
- Dziugaite, Roy, *Computing nonvacuous generalization bounds* (UAI 2017). https://arxiv.org/abs/1703.11008
- Keskar et al., *Large-batch training ... sharp minima* (ICLR 2017). https://arxiv.org/abs/1609.04836
- Dinh et al., *Sharp minima can generalize* (ICML 2017). https://arxiv.org/abs/1703.04933
- Foret et al., *SAM* (ICLR 2021). https://arxiv.org/abs/2010.01412
- Hochreiter, Schmidhuber, *Flat minima* (1997). https://doi.org/10.1162/neco.1997.9.1.1
- Arora, Ge, Neyshabur, Zhang, *Compression approach* (ICML 2018). https://arxiv.org/abs/1802.05296
- Achour, Malgouyres, Gerchinovitz, *Loss landscape of deep linear nets* (JMLR 2024). https://arxiv.org/abs/2107.13289
- Ziyin, *Symmetry induces structure and constraint of learning* (ICML 2024). https://arxiv.org/abs/2309.16932
- Tu, Aranguri, Jacot, *Mixed dynamics in linear networks* (NeurIPS 2024). https://arxiv.org/abs/2405.17580
- Kunin et al., *Get rich quick* (NeurIPS 2024). https://arxiv.org/abs/2406.06158
- Hu, Xiao, Adlam, Pennington, *Surprising simplicity of early-time dynamics* (NeurIPS 2020). https://arxiv.org/abs/2006.14599
- Lee et al., *Wide NNs of any depth evolve as linear models* (NeurIPS 2019). https://arxiv.org/abs/1902.06720
- Yang, Hu, *Feature learning in infinite-width NNs* (muP, ICML 2021). https://arxiv.org/abs/2011.14522 ; *muTransfer*. https://arxiv.org/abs/2203.03466
- Bordelon, Pehlevan, *Self-consistent DMFT* (NeurIPS 2022). https://arxiv.org/abs/2205.09653
- Vyas et al., *Feature-learning networks are consistent across widths* (NeurIPS 2023). https://arxiv.org/abs/2305.18411
- Agarwala, Dauphin, *SAM operates far from home* (ICML 2023). https://arxiv.org/abs/2302.08692 ; Agarwala, Pennington, *Conservative sharpening and a stochastic EoS* (2024). https://arxiv.org/abs/2404.19261
- Mulayoff, Michaeli, *Unique properties of flat minima* (ICML 2020). https://arxiv.org/abs/2002.04710
- Ma, Ying, *Linear stability of SGD and input-smoothness* (NeurIPS 2021). https://arxiv.org/abs/2105.13462
- Wu, Ma, E, *How SGD selects the global minima* (NeurIPS 2018). https://papers.nips.cc/paper/2018/hash/6651526b6fb8f29a00507de6a49ce30f-Abstract.html
- Mukherjee et al., *RL finetunes small subnetworks in LLMs* (NeurIPS 2025). https://arxiv.org/abs/2505.11711
- Shenfeld, Pari, Agrawal, *RL's Razor* (ICLR 2026). https://arxiv.org/abs/2509.04259
- Zhu et al., *The path not taken: RLVR provably learns off the principals* (2025). https://arxiv.org/abs/2511.08567
- Cai et al., *On predictability of RL dynamics for LLMs* (ICLR 2026). https://arxiv.org/abs/2510.00553
- Schulman et al., *LoRA without regret* (Thinking Machines blog, 2025). https://thinkingmachines.ai/blog/lora/
- CRW, *Some interesting papers on RLVR*; *Selection as noisy channel*; *Scale-dependent geometry from diffusion*. https://charlesr-w.github.io/crw-blog/
- Not yet linked: Andriushchenko et al. "Why do we need weight decay in modern DL" (2023); Dohare et al. loss of plasticity (Nature 2024); Lewkowycz et al. catapult (2020); Goyal et al. linear scaling rule (2017); Liu et al. Omnigrok (2022).

---

## 5. Codex critique of v0 (summary; full text in the session log)
1. Timetable too dense for novices; cut Witten/Morse–Bott, extra notebooks, optimizer catalogue; budget setup/debug/debrief time. *(Adopted.)*
2. WB1 didn't establish WB2's prerequisites (estimator sampling distribution, score covariance, empirical vs population H, gap vs excess risk). *(Adopted: single worked example.)*
3. Sandwich formula is a fixed-p regular-case theorem; excess risk is Tr(H⁻¹C)/2n, optimism is Tr(H⁻¹C)/n. *(Adopted.)*
4. C = H only for well-specified NLL at θ*; C = σ²H for squared loss; inequality is misspecification, not SGD. *(Adopted: hook reframed.)*
5. Biggest practical omission: evaluation under selection/shift, reward-proxy optimization. *(Reading pointer; CRW's call.)*
6. Classical→modern rupture overstated; scope each model. *(Adopted.)*
7. Audit could teach overinterpreting correlations; predictions-before-reveal, matched comparisons, seeds. *(Adopted.)*
8. Two tracks ×2 notebooks = 6 pathways; use one common experiment + extensions. *(Adopted.)*
9. Day 2 promotes conditional findings to facts (NTK conditions, lazy depends on scaling, 2/η is plain GD, Song et al. on top-subspace). *(Adopted: scoped.)*
10. Adam ≠ sign descent in general; KL infinite / JS saturates on disjoint supports; WGAN→optimizer is analogy. *(Adopted.)*
11. "One network vs ensemble" conflates estimands; posterior-predictive ≠ posterior-averaged single-param loss; DA conflict is artificial. *(Adopted: three estimands.)*
12. Draft supplies verdicts before investigation (architecture matters less, inductive bias curiosity-stopper, Bayes thin, foils). *(Flagged for CRW; see chat.)*
