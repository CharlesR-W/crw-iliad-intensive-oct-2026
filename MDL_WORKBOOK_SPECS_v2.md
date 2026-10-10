# Mysteries of Deep Learning: workbook specs, v2 (2026-10-08)

Supersedes v1. v1 is kept for its derivation details, which v2 points back to rather than repeating.

**Changes from v1**
- Students do derivations only in WB2 (statistical learning). Every other workbook is written exposition plus discussion questions plus an optional reading.
- Readings are cut hard. The workbook prose narrates ideas, not papers.
  - Papers from CRW's topic notes, or named by CRW, are unmarked.
  - Papers I am suggesting are marked **(proposed)**, for CRW to keep or cut.
  - Where a topic needs a reading and I shouldn't choose one, the slot reads **[CRW: pick]**.
- Section titles follow CRW's topic notes.
- WB2 gains an information-bottleneck section.
- Waluigi moves to WB3.
- WB5 gains the low-rank-logits / subliminal-learning cluster.
- WB6 is CRW's "Mysteries myriad" list and no longer duplicates WB5.
- The focus is MDL. Material that belongs to Training Dynamics (TD) is only pointed at.
- Interactive Distill-style figures accompany four key-intuition results.

**Format.** Notebooks are markdown-only, in the style of the Physics day.
- WB2 has exercise and solution variants.
- The other workbooks are single-variant notebooks with answer cells for the discussion questions.
- Interactive figures are standalone HTML in `notebooks/assets/interactive/`, linked from a "commit before reveal" prompt.

**Day.**
- 10:00: L1 (25–30 min).
- Free work.
- 13:00: L2 (25–30 min).
- Free work.
- 17:00: convergence. Each student brings one card: claim / evidence / what would change my mind / survives scale?

---

## L1. In the misty beginnings: GOFAI → DL
Covers:
- The nature of intelligence: search, planning and control, statistical learning.
- Why GOFAI failed: the bitter lesson.
- Why statistical learning rather than RL or planning.
- Why alignment is hard, then and now.

Hands off to WB1 and WB2.

## L2. Why LLMs?
Covers:
- Why generative, why text, why next-token, why scale.
- The transformer as one flower among a hundred.
- What the simulator frame says about intelligence.

Hands off to WB3–WB6.

---

## WB1. In the misty beginnings: intelligence and alignment, then and now
*Reading and discussion, 1.5–3 h.*

1. **Three pictures of intelligence.** Prose: search over symbols, sequential decision and control, and statistical learning. Each gets one formal object: a state space plus a heuristic; an MDP plus Bellman; a hypothesis class plus ERM. Each also gets the thing it needs hand-written: the heuristic, the reward and state space, or the hypothesis class.
   - Discuss: which of these is cheapest to get from data?
2. **Why GOFAI failed.** Prose: combinatorial explosion and the curse of dimensionality, which are the same blowup. Then the bitter lesson: general methods that leverage compute win.
   - Read: Sutton, "The bitter lesson" (core).
3. **Why statistical learning at all? Why not RL, planning or control?** Prose: at the level of objectives, planning, control and inference coincide ("planning as inference"; CRW's note says "is this important?"). So the real question is which of them scales. Leave it as a question; WB2 §2 takes up the impossibility worry.
   - Read: [CRW: pick, if kept].
4. **Why alignment is hard, then.** Prose:
   - A utility function over a fixed ontology.
   - The ontology problem: the world model changes and the utility has to be re-expressed.
   - Decision-theoretic agency.
   - Natural abstractions as the hoped-for fix.
   - Read: Wentworth, natural abstraction hypothesis.
5. **Why alignment is hard, now.** Prose:
   - Values in a learned system (Byrnes).
   - "Just retarget the search."
   - The simulator / persona picture as the new default object. WB3 owns it; this section only points to it.
   - Read: Byrnes, intro to brain-like AGI safety (on values); Wentworth, "Just retarget the search"; Hubinger et al., "Risks from learned optimization" §1–2 **(proposed)**.
6. **Synthesis.** A table where the student marks each old problem as surviving intact, transformed, or dissolved, with one line why. The old problems: utility specification, the ontology problem, instrumental convergence, deception. Card: one old problem and its new form.

## WB2. Statistical learning and regularization: why the hacky bullshit works
*Students do the derivations. 3–5 h; sections are independent after §1. Derivation details as in v1 WB2 unless noted.*

1. **Risks and gaps.**
   - Population vs empirical risk; ERM.
   - The generalization gap vs excess risk, shown to differ with two-line examples.
   - OLS covariance, and the excess risk $\sigma^2 p/(n-p-1)$, i.e. parameter counting.
2. **Doesn't some theorem make this impossible? (天无绝人之路).**
   - Prove simple NFL: the uniform prior over targets forces off-training-set error ½. The bet is the prior over targets.
   - UAT with a unit count for $\sin\omega x$: existence, not mechanism.
   - The finite-class bound, then the VC form, plugged in for a real network: vacuous.
   - Read: Zhang et al., "Rethinking generalization" §1–2 **(proposed; canonical)**.
3. **Bias–variance.** The pointwise decomposition, then ridge bias and variance in the eigenbasis of $X^\top X$.
4. **Regularization three ways, and how not to think about it.**
   - Derive ridge as a MAP prior, as a Lagrange multiplier, and as a spectral filter.
   - Compare it with early stopping; the half-points match at $t=\ln 2/\lambda$, not $1/\lambda$. ▶ `spectral_filters.html` (panel A).
   - Optional: dropout as adaptive $L_2$.
   - Why flatness could matter: excess risk $\approx\tfrac12\mathrm{Tr}(H\Sigma)$.
   - How not to think about it: the rescaling $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ changes $\mathrm{Tr}H$ without changing the function, so sharpness needs a metric.
5. **Overfitting and double descent.**
   - Note that $p/(n-p-1)$ diverges at $p=n$.
   - Derive the min-norm interpolation bias for $p>n$, and both branches of the isotropic risk curve (variance quoted for $\gamma>1$).
   - Commit to a sketch. ▶ `double_descent.html`: the peak is a variance divergence, and optimal ridge removes it.
   - Note that Chinchilla's $D\propto N$ keeps us at $p/n=O(1)$.
6. **Information bottleneck and the information plane (new).**
   - Prose:
     - The IB objective $\min I(X;T)-\beta I(T;Y)$.
     - The claim that training goes through a fitting phase, then a compression phase, and that compression is why nets generalize.
     - The critique: the compression shows up with saturating nonlinearities and binned estimators, and a deterministic net's $I(X;T)$ is constant or infinite.
   - CRW's position, stated as a position, with the student asked what evidence would discriminate: the estimator is tracking something real, namely geometric compression or clustering of representations, so the picture is basically right and the functional (mutual information) is wrong.
   - Derive:
     - (a) For deterministic injective $f$, $I(X;f(X))=H(X)$ for discrete $X$ and is infinite for continuous $X$, so training cannot move it.
     - (b) The binned estimate is $H(\mathrm{bin}(T))$. For $T=\tanh(aZ)$ it falls to $\log 2$ as $a\to\infty$ (saturation looks like compression). For ReLU it grows with $a$.
     - (c) With noise $T=f(X)+\sigma\varepsilon$ and Gaussian $f(X)$ of covariance $S$, $I(X;T)=\tfrac12\log\det(I+S/\sigma^2)$. Shrinking the within-class part of $S$ while keeping the between-class part lowers $I(X;T)$ while $I(T;Y)$ is roughly preserved. Compression is clustering.
     - (d) Optional: IB as rate–distortion, and as a regularizer.
   - Read:
     - Shwartz-Ziv & Tishby 2017.
     - Saxe et al. 2018, "On the information bottleneck theory of deep learning".
     - Goldfeld et al. 2019, "Estimating information flow in deep neural networks" **(proposed; the paper closest to CRW's position)**.
7. **Bayes and MLE generalization error.**
   - Derive:
     - The sandwich $H^{-1}CH^{-1}$.
     - Excess risk $\tfrac1{2n}\mathrm{Tr}(H^{-1}C)$ vs TIC optimism $\tfrac1n\mathrm{Tr}(H^{-1}C)$; which one a held-out set measures.
     - The Fisher identity $C=H$ in the well-specified case, which gives AIC.
     - The Laplace evidence, which gives BIC.
   - Define the three estimands: single-net excess risk, Bayes-ensemble error, and Gibbs error. Regular case: Bayes $\approx p/2n$ and Gibbs $\approx p/n$; one line on the RLCT replacing $p/2$.
   - Discuss: which estimand do held-out loss, early stopping, ridge and flatness actually control?
8. **Does it cancel? $C$ vs $H$.**
   - Derive the per-sample rank-one structure, and show that $C\propto H$ fails at interpolation.
   - Read Zhang–Yang–Tu §2–3: the diagonal $C_{ii}\propto H_{ii}^\gamma$ with $1\le\gamma\le2$.
   - Derive the stationary law of SGD on a quadratic basin: the Lyapunov equation, with $\Sigma\propto H^{\gamma-1}$ when the two commute. Compare with the posterior $\propto H^{-1}$.
   - ▶ `sgd_stationary_vs_posterior.html`. "SGD is Bayesian" is the claim $\gamma\approx0$.
   - Stop here; TD develops the dynamics.
   - Rabbit holes: Vaintrob and Hänni–Vaintrob (circuit prior); CRW's Morse–Bott note.
9. **Fantastic generalization measures.**
   - Read: Jiang et al. §1–3.
   - Commit, then check: rank five measures by Kendall τ with the gap, overall and within each hyperparameter axis, using the precomputed sweep.
   - Discuss: when should $\mathrm{Tr}(H^{-1}C)$ predict the gap, and when should it fail, given §8?

## WB3. Why LLMs?
*Exposition and discussion; two short worked calculations; 1.5–3 h.*

1. **Why generative, why unsupervised.** Prose: LeCun's cake. Worked example: supervisory bits for ImageNet vs a trillion tokens.
   - Read: the LeCun cake.
2. **Why text? Why not video?** Prose: text as compressed cognition, and the counter-position that video carries the bits.
   - Discuss: what would have to be true for video to win?
3. **Why next-token prediction?** Prose: a language model is a compressor, and cross-entropy is code length (worked in three lines). Bridge to WB4.
4. **A hundred flowers and the coming of the transformer.**
   - Prose: the insight behind CNNs is symmetry. GNNs and the geometric-DL view follow from it. Then why attention: parallel depth and the hardware lottery.
   - Read: Bronstein et al., *Geometric Deep Learning*, ch. 1; Vaswani et al.
5. **Why scale? Should *you* consume more tokens?** Prose: Kaplan → Chinchilla. Worked example: the compute-optimal allocation from $L=E+AN^{-\alpha}+BD^{-\beta}$, $C=6ND$.
   - Read: Hoffmann et al. (Chinchilla).
6. **Simulators, personae, and RL on top.**
   - Prose:
     - A base model as a simulator of personae.
     - Waluigi: why training for a persona makes its opposite cheap.
     - RLHF as a KL-regularized reweighting, $\pi^*\propto\pi_0e^{r/\beta}$ (worked).
     - RL updates are low-rank and stay KL-close (the KL leash). This is why "persona selection" is a fair description.
   - Read: janus, "Simulators" (core); Nardo, "The Waluigi effect".
   - Discuss: is RL-on-LLMs "search" in the bitter-lesson sense, or reweighting?

## WB4. "Inductive bias" is a four-letter word
*Exposition and discussion, 1.5–3 h. CRW is not a specialist in Solomonoff induction, so the prose stays inside textbook statements (teaching note in `HANDOFF_MDL.md`).*

1. **Inductive bias is a curiosity stopper.** Prose: saying "the inductive bias" names the answer without giving a mechanism. The workbook's task is to ask, for each candidate, what mechanism would cash it out.
2. **Occam, two ways.** Prose:
   - The Bayesian Occam factor: penalize prior volume. This links to WB2 §7 (BIC).
   - Description length: Kolmogorov complexity, its invariance up to a constant, its uncomputability, and the counting argument that most strings are incompressible.
   - The two are different mechanisms that often agree.
3. **Solomonoff.** Prose: the universal prior, dominance, and regret ≤ $K(\mu)\ln2$. It is a semimeasure and uncomputable, it is over sequences rather than functions, and no simple universal predictor exists.
   - Discuss: what map from sequences to functions would a "networks ≈ Solomonoff" claim need?
4. **Program synthesis.**
   - Prose: the case that deep learning is search over simple compositional programs, and which step of that case is empirical.
   - Prose: the circuit prior as a neighbouring account.
   - Read: Furman, "Deep learning as program synthesis" (core); Hänni–Vaintrob, circuit prior.
5. **Hierarchical and factored representations.** Prose: why compositional structure in the target could matter more than simplicity of the learner.
   - Read: [CRW: pick].
6. **Competing accounts, side by side.** Discussion table with three priors over functions: program length, circuit size, and a broad prior that decays in a norm ("soft inductive bias"). For each, one phenomenon it predicts and one it doesn't. Candidates: grokking, double descent, scaling exponents, representational convergence, adversarial examples.

## WB5. Universality: the machines, they just want to learn
*Exposition and discussion. One worked-through paper (Ziyin–Chuang); one optional derivation (scaling exponents). 2–4 h.*

1. **Another bitter lesson: architecture doesn't matter.** Prose on the evidence that different architectures trained on the same data end up in the same place. It sets up the question "same up to what?".
2. **Same up to what?** Prose:
   - Every similarity measure fixes a transformation group. CKA is invariant to rotation; CCA to any invertible linear map.
   - Stitching is a functional test.
   - Permutation symmetry is why "different solutions" are often the same solution in different coordinates.
3. **The Platonic Representation Hypothesis.**
   - Prose: the hypothesis and its stated pressures toward convergence.
   - Work through Ziyin–Chuang, "Proof of a perfect PRH": the model; what "same representation" means (which group); how SGD noise selects the aligned solution; which assumptions carry the result.
   - Discuss: what would falsify PRH?
   - Read: Huh et al.; Ziyin–Chuang (core).
4. **The Linear Representation Hypothesis.** Prose: concepts as directions, and the circular-feature counterexample.
   - Discuss: what notion of "linear" survives?
   - Read: [CRW: pick an LRH paper].
5. **Superposition, adversarial attacks, error correction.**
   - Prose: almost-orthogonal packing gives more features than dimensions. The toy model's phase change. Interference as one story about adversarial vulnerability. Redundancy as error correction.
   - ▶ `superposition.html`.
   - Read: Elhage et al., "Toy models of superposition".
6. **Hidden signals: low-rank logits and subliminal learning (new).**
   - Prose, in three steps:
     - (i) Logit matrices built from many prompts and responses are approximately low-rank across modern LMs. A response to one prompt can be produced from a linear combination of the model's outputs on unrelated prompts.
     - (ii) Subliminal learning: a student fine-tuned on a teacher's number sequences picks up the teacher's traits. The paper reports that this needs a shared base model.
     - (iii) The log-linearity account: low logit rank means many weakly correlated examples add up along a shared direction. Selecting data that way produces hidden effects that the authors report transfer even across architectures.
   - Discuss:
     - Is low logit rank a universality claim?
     - How do "needs a shared base model" and "transfers across architectures" fit together?
     - What does this mean for data filtering as a safety measure?
   - Read:
     - Golowich, Liu & Shetty, "Sequences of logits reveal the low rank structure of language models" (arXiv 2510.24966).
     - Cloud et al., "Subliminal learning" (arXiv 2507.14805).
     - Aden-Ali, Golowich, Liu, Shetty, Moitra & Haghtalab, "Subliminal effects in your data: a general mechanism via log-linearity" (arXiv 2602.04863).
     - Contested follow-up **(proposed)**: Nief et al., "Subliminal learning is a LoRA artifact" (arXiv 2606.00831). Its own authors note that some main claims have since been shown incorrect.
7. **Scaling laws: a hundred unnecessary explanations.**
   - Prose: with a power-law data spectrum, $n$ samples resolve the top $\sim n$ modes, and the exponent is set by the spectrum (worked, optional; define $c_k^2=\lambda_k\beta_k^2$. Noiseless: $n^{-(\alpha'-1)}$. With noise and optimal ridge: $n^{-(\alpha'-1)/\alpha'}$. Ridge saturates at rate $\min(\alpha'-1,2\alpha)$; these corrections to v1 were verified by the figure agent). A discrete-skills ("quanta") story gives the same curve.
   - ▶ `spectral_filters.html` (panel B).
   - Discuss: what measurement distinguishes the two mechanisms?
   - Read: CRW, "Power laws are not enough".

## WB6. Mysteries myriad
*Reading cards; about 45 min each; take one or two. Each card has a framing paragraph, three questions, and the card template. One reading per card. Where the paper isn't fixed, the slot reads [CRW: confirm].*

| Card | Framing | Reading |
|---|---|---|
| Transformer middle layers | the best representations live in the middle, not at the end | [CRW: pick] |
| Bottom-up learning | lower layers converge first | Raghu et al., SVCCA |
| Model stitching works | good networks learn interchangeable parts | Bansal, Nakkiran & Barak |
| Critical periods | early deficits are permanent | Achille, Rovere & Soatto |
| Adversarial attacks (and on you too) | non-robust features are predictive; some attacks transfer to humans | Ilyas et al.; [CRW: confirm human-transfer paper] |
| Critical modules | some layers matter and others can be reset | [CRW: confirm] |
| Distillation and pruning | small models can carry big-model functions | [CRW: pick] |
| Rewinding and lottery tickets | the basin is chosen early | Frankle & Carbin; Frankle et al., linear mode connectivity |
| GD in a tiny subspace | the gradient lives in the top Hessian directions | Gur-Ari, Roberts & Dyer |
| Spectra and curvature | [CRW] | [CRW: pick] |
| Low-rank RL updates and the KL leash | RL changes little, in a structured way | [CRW: pick] |
| Physics of language models | controlled synthetic experiments on what LMs learn | Allen-Zhu, Physics of LMs |
| What if not circuits? | circuits may be the wrong ontology | CRW blog |

**Closing prompt: what kinds of explanations do we need?** ("Psychology from quantum field theory.") A short discussion prompt that feeds the 17:00 convergence.

---

## Interactive figures (being built; `notebooks/assets/interactive/`)
| File | Workbook | Intuition it carries |
|---|---|---|
| `spectral_filters.html` | WB2 §4, WB5 §7 | ridge vs early stopping as filters; the exponent comes from the spectrum |
| `double_descent.html` | WB2 §5 | the peak is a variance divergence; ridge removes it |
| `sgd_stationary_vs_posterior.html` | WB2 §8 | $\Sigma\propto H^{\gamma-1}$ vs $H^{-1}$; drag γ |
| `superposition.html` | WB5 §5 | the phase change, with polygons emerging as sparsity grows |

Candidate fifth figure, not started: an information-plane estimator demo for WB2 §6. Binned "$I(X;T)$" for tanh vs ReLU with a bin-size slider, beside a noisy-channel estimate that tracks clustering.

## Precomputed artefacts (Modal)
- The WB2 §9 sweep: about 300 small CNNs, with measures and gaps.
- The WB2 §8 diagonal scatter of $C$ vs $H$.

Both are as specified in the v1 build notes.
