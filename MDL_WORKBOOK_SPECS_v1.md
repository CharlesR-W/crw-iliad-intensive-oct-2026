# Mysteries of Deep Learning: workbook specifications, v1 (2026-10-07)

Companion to `HANDOFF_MDL.md` (how to critique and revise this) and `CURRICULUM_DRAFT_v1.md` §4 (verified link appendix). The format follows the Physics of Deep Learning day (`/home/crw/Programming/IliadPhysicsLearning`): markdown-first notebooks built from a Python builder, exercise and solution variants from one source, blank answer cells, hints in collapsed `<details>`, claim boundaries stated where they matter, "what carries forward" and "further study" closers. No PLOT-SLOP chapter this time. Students are stronger than the August cohort; problems may be harder, and length is not constrained.

## 0. Teaching contract

- Two orienting lectures are the trunk: **L1 "GOFAI → DL"** and **L2 "DL → LLMs and scale"**. Lectures introduce words and motivations; workbooks own derivations, calculations, and readings.
- Six workbooks. Students pick whatever they like, in any order, for as long as they like. WB2 is the exercise sheet; WB1, WB3 and WB6 are reading-led with comprehension questions; WB4 and WB5 mix the two.
- Readings are organized inside workbooks as **core / choose one / rabbit hole**, each with 2–4 comprehension or discussion questions. Not every reading needs an exercise.
- Three estimands are kept distinct wherever generalization is discussed: the excess population loss of one trained network; the Bayes-ensemble (posterior-predictive) error; the Gibbs/posterior-draw error. Say which one a tool or theorem is about.
- Competing accounts (program synthesis, "not so mysterious", circuit priors, SLT) are presented symmetrically; the student is asked what evidence would discriminate them.
- Code: markdown-only by default, as in the Physics day. Where a calculation is best seen numerically (WB2 §4, §6, §7; WB5 §6), a supplied `PROVIDED HELPER` cell displays a result the student has first been asked to predict ("commit before reveal"). No exercise requires the student to write code. **Decision for CRW:** keep this rule, or allow optional coding extensions.
- Day shape: L1 at 10:00 (25–30 min), free work, L2 at 13:00 (25–30 min), free work, convergence at 17:00 where students report one "card" (claim / evidence / what would change my mind / survives scale?) from whatever they did.

---

## L1. Lecture: GOFAI → DL (25–30 min)

Purpose: three pictures of intelligence, why the first two lost to the third, and what each picture thought alignment was. Hands off to WB1 (readings) and WB2 (the statistical picture, made precise).

- **0–6 min.** Intelligence as search over symbols: physical symbol system hypothesis, heuristic search, expert systems. What it needs: a hand-written state space and a hand-written evaluation. Lighthill's combinatorial explosion.
- **6–12 min.** Intelligence as sequential decision: Bellman, dynamic programming, control, RL. Reward as the specification. The curse of dimensionality is the same explosion in a different costume. "Reward is enough" as the maximal claim.
- **12–18 min.** Intelligence as statistical learning: Valiant, Vapnik, Breiman's two cultures. Replace hand-written structure with data plus a model class plus an optimizer. NFL says you must bet on something; the bet is the model class.
- **18–24 min.** Why the third won: bitter lesson, hardware lottery, unreasonable effectiveness of data, AlexNet. "General methods that leverage computation" means search depth and sample complexity scale with compute, and hand-written structure does not.
- **24–28 min.** Alignment under each picture: utility and corrigibility (search/decision picture); ontological crisis; reward is not the optimization target; prosaic alignment and inner alignment (learning picture). Hand off: WB1 asks which old problems survive the paradigm shift; WB2 makes the learning picture precise.

Must not: derive anything WB2 derives; resolve the WB1 discussion questions.

## L2. Lecture: DL → LLMs and scale (25–30 min)

Purpose: why generative, why text, why next-token, why the transformer, why scale, and what this paradigm says intelligence is. Hands off to WB3 (story), WB4 (TCS), WB5 (universality), WB6 (weird stuff).

- **0–5 min.** The cake: bits of supervision per example in classification vs per token in language modelling. Unsupervised wins on data volume by orders of magnitude.
- **5–10 min.** Why text: text is compressed cognition; the counter-position (JEPA, "video is where the bits are"); multimodal scaling laws say the exponents differ but the form does not.
- **10–15 min.** Why next-token: a language model is an arithmetic coder; cross-entropy is code length; the compression view and its Solomonoff flavour (WB4 owns the rigorous version).
- **15–20 min.** A hundred flowers and the transformer: LSTM, CNN, GNN; the hardware lottery; architecture rankings change with scale (Tay et al.); Mamba as the live counterpoint.
- **20–25 min.** Why scale: Kaplan, Chinchilla, compute-optimal allocation; what a scaling exponent does and does not explain (WB5 owns the "triviality" derivation); emergence and the mirage debate.
- **25–30 min.** What this says about intelligence: simulators, personae, RLHF as persona selection, RL on LLMs. Hand off to WB3's discussion questions.

Must not: show the Chinchilla allocation derivation (WB3 §5); state the PRH or LRH formally (WB5); argue the Solomonoff chain (WB4).

---

## WB1. From GOFAI to statistical learning; old and new alignment problems

**Type:** reading-led with short exercises. **Needs:** L1. **Length:** 2–4 h depending on readings taken.

### Purpose and boundary
Make the three pictures of intelligence concrete enough to see why one of them won, and sort the alignment problems by which picture they assume. Boundary: the workbook does not adjudicate alignment strategy; it asks which problems survive the shift to learned systems and in what form.

### Route
1. Three pictures, three formal objects
2. Why the first two lost
3. Alignment problems that assume an agent
4. Alignment problems that assume a learner
5. Readings and cards

### 1. Three pictures, three formal objects
Short expository passages, then:
- **1.1 Search.** A state space $S$, transitions, a goal test, a heuristic. Exercise: for 8-puzzle-style problems, count the state space and show that uniform-cost search is exponential in solution depth without a heuristic; state what an admissible heuristic buys (A*). Interpretation: all the intelligence is in the hand-written heuristic. (Solution: $9!/2$ states; branching $\sim 2.7$; A* optimal with admissible $h$.)
- **1.2 Sequential decision.** A 3-state MDP with two actions is supplied. Exercise: write the Bellman optimality equation, do two rounds of value iteration by hand, extract the greedy policy. Then: a gridworld with $k$ movable objects has $|S| \sim n^{2k}$; conclude tabular DP is dead for $k \gtrsim 5$. Interpretation: the curse of dimensionality is Lighthill's explosion again. (Solutions supplied in builder.)
- **1.3 Statistical learning.** Define a hypothesis class $\mathcal H$, population risk $R(h)=\mathbb E\,\ell(h(x),y)$, empirical risk $\hat R_n$, ERM. Exercise: show that for finite $\mathcal H$ and bounded loss, Hoeffding plus a union bound gives $\sup_h |R-\hat R_n| \le \sqrt{(\log|\mathcal H| + \log(2/\delta))/2n}$ w.p. $1-\delta$. Interpretation: the "intelligence" is now the choice of $\mathcal H$ plus data. Cross-reference WB2 §2 where this becomes the capacity story.
- **1.4 Discussion.** Each picture needs something hand-written: heuristic, reward and state space, hypothesis class. Which of these is cheapest to get from data? (This is the bitter lesson stated as a question.)

### 2. Why the first two lost
Readings: Sutton "The bitter lesson" (core); Hooker "The hardware lottery"; Halevy–Norvig–Pereira "Unreasonable effectiveness of data"; Breiman "Two cultures"; Irpan "Deep RL doesn't work yet"; Recht "A tour of RL from continuous control"; Silver et al. "Reward is enough" (as the maximal counter-claim).
Comprehension questions:
- Sutton names two general methods: search and learning. Give one example where each scaled with compute and one hand-written method that did not.
- Hooker: name a currently popular idea that may be winning the hardware lottery rather than the idea lottery.
- Breiman: restate the two cultures as two choices of estimand. Which culture is Chinchilla in?
- Irpan vs Silver: "reward is enough" is a claim about what, exactly? Sufficiency of the objective, or tractability of the optimization? Which does Irpan attack?

### 3. Alignment problems that assume an agent
Readings: Bostrom "The superintelligent will" (orthogonality, instrumental convergence); Omohundro "Basic AI drives"; Soares et al. "Corrigibility"; de Blanc "Ontological crises"; the strawberry problem (Yudkowsky; canonical statement to be linked); Byrnes "Intro to brain-like AGI safety" (posts 1–3).
Exercises:
- **3.1 Ontological crisis, formally.** An agent has utility $U:S\to\mathbb R$ on ontology $S$. The world turns out to be $S'$ with a many-to-one map $\pi:S'\to S$ that is only approximately well-defined. Write down two candidate lifts of $U$ to $S'$ and show they disagree on states where $\pi$ is ambiguous. Then: what is the analogue in a trained network whose internal representation drifts during fine-tuning? (This connects to WB5 §2–3 and to "What if not circuits".)
- **3.2 Instrumental convergence as a theorem about MDPs.** Read Turner et al. "Optimal policies tend to seek power" abstract; state in one sentence what is proved and what is assumed about the distribution over reward functions. Does the assumption hold for reward functions produced by RLHF?
- **3.3 Corrigibility as a constraint.** Soares et al. show a naive shutdown-utility is either incentivizing or disincentivizing shutdown. Write the two-outcome expected-utility calculation that produces the dilemma.

### 4. Alignment problems that assume a learner
Readings: Christiano "Prosaic AI alignment" (core); Hubinger et al. "Risks from learned optimization" (core, §1–2); Turner "Reward is not the optimization target"; Amodei et al. "Concrete problems"; Wentworth "Natural abstraction hypothesis" and "Just retarget the search"; Greenblatt et al. "Alignment faking"; Betley et al. "Emergent misalignment"; Chen et al. "Persona vectors"; Shanahan et al. "Role play with LLMs".
Exercises:
- **4.1** Hubinger's base objective vs mesa-objective: write it as two optimization problems and say which one the training signal constrains. Where does WB2's generalization gap sit in this picture?
- **4.2** "Reward is not the optimization target": restate as a claim about which estimand RL training controls. Connect to Levine's RL-as-inference: what does the learned policy approximate?
- **4.3** Emergent misalignment: narrow fine-tuning produces broad behavioural shift. Which of the "agent" problems in §3 does this resemble, and which picture of intelligence does the resemblance assume?
- **4.4 Synthesis table.** For each old problem (utility specification, corrigibility, ontological crisis, instrumental convergence, deception), write whether it survives the shift to learned simulators intact, transformed, or dissolved, and in one line why.

### 5. Readings and cards
Rabbit holes: Nilsson's history; Newell–Simon Turing lecture; Lighthill report; Sutton–Barto ch. 1 and §1.7; Levine RL-as-inference tutorial; LeCun's cake; Kulveit "Three-layer model"; Nardo "Waluigi".
Card: one old problem and its new form.

### What carries forward
The ERM picture (to WB2); representational drift as the DL form of ontological crisis (to WB5); simulators and personae (to WB3).

---

## WB2. Statistical learning, regularization, and the Bayesian-ness of SGD

**Type:** exercise sheet. **Needs:** L1; linear algebra, Taylor expansion, basic probability. **Length:** 3–5 h; sections are independent after §1.

### Purpose and boundary
Derive the classical statistical-learning results that practical tools (regularization, early stopping, flatness, information criteria) actually rest on; locate the regime where they are asked to work; and derive, on the quadratic basin, exactly when SGD's stationary law matches a posterior. Boundary: every theorem is stated with its regime (fixed $p$, $n\to\infty$; or $p/n\to\gamma$); nothing is asserted for overparametrized networks beyond what the quadratic-basin model licenses. The three estimands are named throughout.

### Route
1. Risks, gaps, and the bias–variance decomposition
2. Capacity and its discontents (uniform convergence, NFL, UAT, Zhang et al.)
3. Regularization three ways (prior, Lagrange multiplier, spectral filter) and why flatness could matter
4. The regime: $p/n=\gamma$ and the interpolation peak
5. The information-criterion derivation of generalization error: $\mathrm{Tr}(H^{-1}C)$
6. Does it cancel? $C$ vs $H$, and the stationary law of SGD on a quadratic basin
7. Measuring generalization: fantastic measures and their robustness

### 1. Risks, gaps, and bias–variance
Notation: data $(x_i,y_i)\sim P$, loss $\ell$, population risk $R(\theta)=\mathbb E_P\,\ell$, empirical risk $\hat R_n$, ERM $\hat\theta_n$, population minimizer $\theta^*$.
- **1.1** Define the generalization gap $R(\hat\theta_n)-\hat R_n(\hat\theta_n)$ and the excess risk $R(\hat\theta_n)-R(\theta^*)$. Show that the gap can be positive while the excess risk is zero, and vice versa, with two-line examples. (Both are used later; students must not conflate them.)
- **1.2** Linear regression $y=x^\top\beta+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2)$, fixed design $X\in\mathbb R^{n\times p}$, $n>p$. Derive $\hat\beta=(X^\top X)^{-1}X^\top y$, $\mathbb E\hat\beta=\beta$, $\mathrm{Cov}\hat\beta=\sigma^2(X^\top X)^{-1}$. Derive the expected excess risk at a fresh $x\sim\mathcal N(0,I)$: $\sigma^2\,\mathrm{Tr}((X^\top X)^{-1})$, and for Gaussian design its expectation $\sigma^2 p/(n-p-1)$. Interpretation: variance $\approx\sigma^2 p/n$, "parameter counting".
- **1.3** Bias–variance for a general estimator at a point: $\mathbb E(\hat f-f)^2=\mathrm{bias}^2+\mathrm{var}$; then for ridge with penalty $\lambda$, derive the bias and variance in the eigenbasis of $X^\top X=V\mathrm{diag}(s_k^2)V^\top$: bias$_k$ $=-\frac{n\lambda}{s_k^2+n\lambda}\beta_k$, var$_k$ $=\sigma^2\frac{s_k^2}{(s_k^2+n\lambda)^2}$. Interpretation: regularization trades bias in weak directions for variance.

### 2. Capacity and its discontents
- **2.1 Uniform convergence, the shape only.** From WB1 §1.3's finite-class bound, state (do not prove) the VC form $\sqrt{(d_{VC}\log n+\log(1/\delta))/n}$. Exercise: for a ReLU net with $W$ weights, $d_{VC}=\Theta(W\log W)$ (Bartlett et al.); plug in $W=10^6$, $n=5\cdot10^4$ and get a vacuous bound. Claim boundary: vacuous is not wrong, but it cannot be the explanation.
- **2.2 No free lunch.** Prove the simple version: for a binary target chosen uniformly among all $2^{|\mathcal X|}$ labelings of a finite domain, every learner has expected off-training-set error $1/2$. Interpretation: you must bet on structure; the bet is the inductive bias. Then: NFL is a statement about the uniform prior over targets; what prior over targets does the real world plausibly have? (Pointer to WB4: Solomonoff's answer.)
- **2.3 Universal approximation.** State UAT (Cybenko/Hornik). Exercise: approximate $\sin(\omega x)$ on $[0,1]$ to accuracy $\epsilon$ with a one-hidden-layer ReLU net; count units needed as a function of $\omega$. Then Telgarsky's sawtooth: depth $k$ with $O(k)$ units makes $2^k$ oscillations. Interpretation: UAT is existence, not mechanism, and says nothing about learnability or sample complexity. (WB4 §7 continues this.)
- **2.4 Zhang et al.** Read §1–2 of "Understanding deep learning requires rethinking generalization". Questions: (i) list every explanation the random-labels experiment rules out and every one it does not; (ii) the paper notes that explicit regularizers help but are not necessary; which of §3's three views of regularization is that evidence against?; (iii) write the "soft inductive bias" rebuttal (Wilson) in two lines and say what it predicts that Zhang et al. does not.

### 3. Regularization three ways and why flatness could matter
On the same linear model:
- **3.1 As a prior.** Show ridge is MAP under $\beta\sim\mathcal N(0,\tau^2 I)$ with $\lambda=\sigma^2/(n\tau^2)$. Show the posterior is $\mathcal N(\hat\beta_\lambda, \sigma^2(X^\top X+n\lambda I)^{-1})$.
- **3.2 As a Lagrange multiplier.** Show the constrained problem $\min\hat R_n$ s.t. $\|\beta\|^2\le c$ has a KKT solution equal to ridge with some $\lambda(c)$, and that $\lambda(c)$ is monotone.
- **3.3 As a spectral filter.** Derive the filter factors: ridge $\phi_k=s_k^2/(s_k^2+n\lambda)$; gradient flow stopped at time $t$: $\phi_k=1-e^{-s_k^2 t/n}$. Plot both against $s_k^2$ (supplied helper, after a sketch). Claim boundary (from the Physics day): related filters, not identical ones; match the half-way point $n\lambda \leftrightarrow n/t$ and show where they differ.
- **3.4 Dropout as adaptive $L_2$** (optional). For linear regression with Bernoulli dropout on inputs, derive the expected loss and show the penalty $\frac{1-q}{q}\sum_k \beta_k^2\|x_{\cdot k}\|^2/n$ (Wager–Wang–Liang). Interpretation: "hacky" regularizers are usually a quadratic penalty in disguise, in some metric.
- **3.5 Why flatness.** Second-order expansion of the population risk about a minimizer: $R(\theta^*+\delta)\approx R(\theta^*)+\frac12\delta^\top H\delta$. If the trained parameter is $\theta^*+\delta$ with $\mathbb E\delta\delta^\top=\Sigma$, then $\mathbb E[R]-R(\theta^*)\approx\frac12\mathrm{Tr}(H\Sigma)$. Interpretation: this is where "flat minima generalize" comes from, and it is a statement about $H$ *and* $\Sigma$. Then Dinh et al.: show that for a two-layer linear net $f=w_2 w_1 x$, the rescaling $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ preserves $f$ and changes $\mathrm{Tr}\,H$ without bound. Claim boundary: raw sharpness is not a function of the learned function; any flatness claim must say what metric $\Sigma$ is measured in.

### 4. The regime: $p/n=\gamma$ and the interpolation peak
- **4.1** Observe that §1.2's $\sigma^2 p/(n-p-1)$ diverges as $p\to n$. State the modern regime: $p,n\to\infty$ with $p/n\to\gamma$; for LLMs, Chinchilla's $D\propto N$ (scoped as a compute-allocation fit).
- **4.2** Min-norm interpolation for $p>n$: $\hat\beta=X^+y$. Show $\hat\beta$ is the projection of $\beta$ onto the row space of $X$ plus noise; derive the bias term $\|\beta\|^2(1-n/p)$ in expectation for isotropic design.
- **4.3** Quote Hastie et al. Theorem 1 (isotropic Gaussian design, $\sigma^2$ noise, $\|\beta\|^2=r^2$): risk $=\sigma^2\gamma/(1-\gamma)$ for $\gamma<1$ and $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$ for $\gamma>1$. Exercise: derive the $\gamma<1$ branch from §1.2 and the $\gamma>1$ bias from §4.2 (the $\gamma>1$ variance is quoted). **Commit before reveal:** sketch the risk vs $\gamma$; then the supplied helper shows it, and shows that ridge with optimal $\lambda$ removes the peak. Interpretation: double descent is a variance divergence at $p\approx n$ in the least-squares model, removed by any regularization. Claim boundary: this is one model; deep double descent (Nakkiran) is phenomenology, and Chinchilla does not identify tokens with i.i.d. regression samples.
- **4.4** (optional) Random features (Mei–Montanari) or the solvable scaling model (Maloney–Roberts–Sully): where the exponents come from. Cross-reference WB5 §6.

### 5. The information-criterion derivation of generalization error
Regular case: fixed $p$, $H=\nabla^2R(\theta^*)\succ0$, $\hat\theta_n\to\theta^*$.
- **5.1** From the first-order condition $\nabla\hat R_n(\hat\theta_n)=0$ and a Taylor expansion at $\theta^*$, derive $\sqrt n(\hat\theta_n-\theta^*)\Rightarrow\mathcal N(0,H^{-1}CH^{-1})$ with $C=\mathrm{Cov}_P[\nabla_\theta\ell(\theta^*)]$ (the sandwich; White 1982).
- **5.2** Combine with §3.5 to get $\mathbb E[R(\hat\theta_n)]-R(\theta^*)\approx\frac{1}{2n}\mathrm{Tr}(H^{-1}C)$ (excess risk). Derive separately the optimism $\mathbb E[R(\hat\theta_n)-\hat R_n(\hat\theta_n)]\approx\frac1n\mathrm{Tr}(H^{-1}C)$ (TIC; Takeuchi; Murata–Yoshizawa–Amari NIC). Note the factor of two and say which one a held-out set measures.
- **5.3** Prove that for a well-specified negative log-likelihood at the true parameter, $C=H$ (Fisher information identity: $\mathbb E[\nabla\ell\nabla\ell^\top]=\mathbb E[\nabla^2\ell]$ via differentiating $\int p=1$ twice). Hence $\mathrm{Tr}(H^{-1}C)=p$ and the optimism is $p/n$: AIC, "parameter counting". For squared loss with homoscedastic noise, show $C=\sigma^2H$ at $\theta^*$ (the residual is uncorrelated with $\nabla^2 f$).
- **5.4 The Bayesian side.** Laplace-approximate the marginal likelihood $\int e^{-n\hat R_n(\theta)}\pi(\theta)d\theta$ to get $-\log Z\approx n\hat R_n(\hat\theta_n)+\frac p2\log n+O(1)$ (BIC). Then define the three estimands explicitly: (i) excess risk of the single ERM parameter (above); (ii) Bayes-ensemble error $\mathbb E_{\text{data}}\,\mathrm{KL}(p^*\,\|\,\int p_\theta\,d\pi(\theta|\text{data}))$; (iii) Gibbs error (posterior draw). State Watanabe's regular-case results: Bayes generalization $\approx\frac{p}{2n}$ in expectation, Gibbs $\approx\frac{p}{n}$. In the singular case the $p/2$ becomes the RLCT $\lambda$, which is the volume-scaling exponent of the sublevel sets (one line; no algebraic geometry). Interpretation: all three estimands agree on "effective parameter count over $n$" in the regular case; they differ in what "effective parameter count" means and in what object they describe. Claim boundary: none of these is, as stated, a theorem about an overparametrized network trained by SGD; §6 asks what the quadratic-basin model can say.
- **5.5** Discussion: which estimand do practical tools (held-out loss, early stopping, ridge, flatness) actually control?

### 6. Does it cancel? $C$ vs $H$ and the stationary law of SGD
- **6.1** Per-sample structure. For squared loss on a model $f_\theta$, per-sample gradient $g_p=-r_p\nabla f_p$ and per-sample GGN $h_p=\nabla f_p\nabla f_p^\top$ (rank one). Show $H_{GGN}=\mathbb E[h_p]$ and $C=\mathbb E[r_p^2h_p]-\bar g\bar g^\top$. At an interpolating minimum $\bar g=0$ and $r_p\to0$ so $C\to0$ while $H\not\to0$: $C\propto H$ fails at interpolation (Ziyin et al.). Away from interpolation with $r_p^2$ independent of $\nabla f_p$, $C=\sigma^2H$.
- **6.2** Zhang–Yang–Tu (2026): read §2–3. They show $C\propto\mathbb E_p[h_p^2]$ under their assumptions and find diagonal power laws $C_{ii}\propto H_{ii}^{\gamma}$ with $1\le\gamma\le2$. Exercise: using Jensen on the rank-one structure, show $\mathbb E[h_p^2]_{ii}=\mathbb E[\|\nabla f_p\|^2(\partial_if_p)^2]\ge$ (something proportional to $H_{ii}$) and argue heuristically why $\gamma\ge1$; give a construction with $\gamma=2$ (all $\nabla f_p$ parallel). **Commit before reveal:** predict $\gamma$ for a small MLP; the supplied helper shows a diagonal scatter of $C_{ii}$ vs $H_{ii}$ from a precomputed run.
- **6.3 The stationary law.** Model SGD near a minimum as $d\theta=-H\theta\,dt+\sqrt{\eta/B}\,C^{1/2}dW$. Derive the Lyapunov equation for the stationary covariance: $H\Sigma+\Sigma H=\frac{\eta}{B}C$. Solve it when $C$ and $H$ commute: $\Sigma=\frac{\eta}{2B}CH^{-1}$. Now compare with the posterior near its mode, $\Sigma_{\text{post}}=\frac1n H^{-1}$ (or $\frac{T}{n}H^{-1}$ tempered). Conclude: the stationary law matches a tempered posterior iff $C\propto I$ (isotropic noise, $\gamma=0$); with $C\propto H^\gamma$ it is $\Sigma\propto H^{\gamma-1}$, which for $\gamma\in[1,2]$ puts *more* variance in sharp directions than the posterior, not less. Interpretation: "SGD is approximately Bayesian" is a claim that $\gamma\approx0$, or that one has preconditioned by $C^{-1}$ (Mandt et al. do exactly this). Claim boundary: the OU model is local and linear; with loss-dependent noise ($C\propto L\cdot H$, Mori et al.) the stationary law has power-law tails, which is a different deviation.
- **6.4** Plug $\Sigma$ into §3.5: excess risk under SGD's stationary law $\approx\frac12\mathrm{Tr}(H\Sigma)=\frac{\eta}{4B}\mathrm{Tr}(C)$. Compare with the ERM value $\frac1{2n}\mathrm{Tr}(H^{-1}C)$ and the posterior value $\frac{p}{2n}$. Three different traces; say in words what each one counts.
- **6.5 Readings with questions.** Mandt–Hoffman–Blei (what preconditioner makes SGD a sampler?); Smith–Le (the noise scale $\eta N/B$ and what it predicts for batch-size scaling); Mingard et al. (what exactly is compared to what, and does it bear on §6.3?); Wenzel et al. cold posteriors (if the true posterior underperforms, which estimand is being optimized by tempering?); Izmailov et al. HMC (what do exact posteriors do under covariate shift, and why might that favour the single-network estimand?); Lotfi et al. marginal likelihood (what question does the evidence answer?). Rabbit hole: Vaintrob "Learning zero"; Hänni–Vaintrob circuit prior; CRW's Morse–Bott/Witten note on posterior concentration.

### 7. Measuring generalization
- **7.1** Read Jiang et al. §1–3 and the table of measures. Questions: which measures are functions of the learned function alone, and which depend on parametrization? Which could in principle be computed from §5–6 quantities?
- **7.2** A precomputed sweep is supplied (a few hundred small CNNs, varied in width, depth, lr, batch size, weight decay, seeds; measures and test gaps tabulated). **Commit before reveal:** rank five measures by expected Kendall $\tau$ with the gap, overall and within each hyperparameter axis. Then compute (helper) and compare. Then read Dziugaite et al. §2: why pooled correlation can be confounded, and what "robust" means. Interpretation: a measure that tracks the gap along one axis and not another is tracking the axis, not the gap.
- **7.3** Discussion: §5's $\mathrm{Tr}(H^{-1}C)$ is one of the measures. Given §6, under what training conditions should it predict well, and under which should it fail?

### What carries forward
$H$, $C$, $\eta/B$, the Lyapunov equation, and the three estimands go to Training Dynamics; the regime question goes to WB5 §6; NFL and the prior over targets go to WB4.

### Further study
Hastie et al.; Bach 2023; Thomas et al. 2020; White 1982; Watanabe "Mathematical theory of Bayesian statistics" ch. 1 and the Gibbs/Bayes note; Ziyin et al. 2022; Wilson 2025; Prince UDL ch. 8–9; Shalev-Shwartz–Ben-David ch. 2–6.

---

## WB3. From ML to LLMs and scale

**Type:** story with exercises and readings. **Needs:** L2. **Length:** 2–4 h.

### Purpose and boundary
Tell the path from supervised deep learning to large language models as a sequence of answers to "why": generative, text, next-token, transformer, scale. Each answer gets one short calculation and one reading. Boundary: no claim that the path was inevitable; the hardware lottery and the architecture-ranking-changes-with-scale result are in the story to prevent that reading.

### Route
1. Why generative and unsupervised: bits per example
2. Why text
3. Why next-token: a language model is an arithmetic coder
4. Why the transformer: a hundred flowers and the hardware lottery
5. Why scale: compute-optimal allocation
6. What it made: simulators, personae, and RL on top
7. The bitter lesson revisited

### 1. Bits per example
- **1.1** A classifier with $k$ classes receives at most $\log_2 k$ bits of supervision per example; a language model on a sequence of $T$ tokens over vocabulary $V$ receives up to $T\log_2V$ bits, of which the model's cross-entropy $\ell$ says it still has $\ell T/\ln 2$ to learn. Exercise: for ImageNet ($10^6$ images, $10^3$ classes) vs one trillion tokens at $\sim3$ nats/token, compare total supervisory bits. Interpretation: LeCun's cake.
- **1.2** Reading: the LeCun cake (Ruder's writeup); Dai–Le, ULMFiT, BERT, GPT-1 as the sequence of "pretrain then adapt" results. Question: at what point did the pretrained object stop being a feature extractor and start being the product?

### 2. Why text
Readings: Sutskever "An observation on generalization" (video); LeCun "A path towards autonomous machine intelligence" §1–2 (the counter-position); Henighan et al. multimodal scaling laws. Questions: (i) state Sutskever's compression argument in three lines and say what it assumes about the data distribution; (ii) LeCun says predicting pixels is the wrong objective; what is his alternative objective, and is it a likelihood?; (iii) Henighan: do the exponents differ across modalities, and does the functional form?

### 3. A language model is an arithmetic coder
- **3.1** Show that an arithmetic coder driven by model $q$ assigns $x_{1:T}$ a code length $\sum_t-\log_2q(x_t|x_{<t})+O(1)$ bits; hence cross-entropy loss is code length and lower loss is better compression. Then Shannon: the expected code length is minimized at $q=p$, with excess $\mathrm{KL}(p\|q)$.
- **3.2** Reading: Delétang et al. "Language modeling is compression". Question: they compress images and audio with a text LLM. Why does that work at all, and what does the compression rate on out-of-modality data say about "universality" (WB5)?
- **3.3** Bridge to WB4: compression is the computable shadow of Solomonoff induction. State the claim; WB4 §3–4 examines it.

### 4. A hundred flowers and the transformer
- **4.1** Exercise: per-token cost and parallelism. For sequence length $T$ and width $d$, write FLOPs per token for an RNN step, a 1-D convolution of width $k$, and self-attention; then the sequential depth of the forward pass over a batch of sequences. Interpretation: attention is $O(T)$ per token but $O(1)$ sequential depth; the GPU wants the second thing. The hardware lottery in one table.
- **4.2** Readings: Vaswani et al.; Hochreiter–Schmidhuber LSTM; LeCun et al. 1998; Bronstein et al. GDL ch. 1 and Joshi "Transformers are GNNs"; Hooker "Hardware lottery"; Tay et al. "Scaling laws vs model architectures"; Gu–Dao Mamba. Questions: (i) Bronstein: what symmetry group does attention respect and what does it break; (ii) Tay et al.: name an architecture that wins at small scale and loses at large; what does that do to "the transformer is special"?; (iii) Mamba: what does it give up, and what would you need to see to conclude attention is not necessary?

### 5. Compute-optimal allocation
- **5.1** Given $L(N,D)=E+AN^{-\alpha}+BD^{-\beta}$ and $C=6ND$, minimize $L$ at fixed $C$. Derive $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$, $D_{\text{opt}}\propto C^{\alpha/(\alpha+\beta)}$; with Chinchilla's fitted $\alpha\approx\beta$, both scale as $\sqrt C$ and $D/N$ stays constant. Interpretation: this is the $p/n=O(1)$ regime of WB2 §4, by design.
- **5.2** Readings: Kaplan et al.; Hoffmann et al. (Chinchilla); Muennighoff et al. data-constrained scaling; Villalobos et al. "Will we run out of data?". Questions: (i) Kaplan found $D\propto N^{0.74}$, Chinchilla $D\propto N$: what changed in the experiment?; (ii) what does repeating data do to the effective $D$?; (iii) what would a scaling law look like if the bottleneck were data rather than compute?
- **5.3** Pointer: WB5 §6 derives where the exponents come from.

### 6. Simulators, personae, and RL on top
Readings (core): janus "Simulators"; Shanahan–McDonell–Reynolds "Role play with LLMs"; Ouyang et al. InstructGPT. Choose one: Nardo "Waluigi"; Kulveit "Three-layer model"; Chen et al. "Persona vectors"; janus "Mysteries of mode collapse" (with the later RLHF correction); Bai et al. Constitutional AI; Lambert RLHF book ch. 1–3. RL on top: DeepSeek-R1; Yue et al. "Does RL incentivize reasoning beyond the base model?"; Shenfeld "RL's Razor"; Mukherjee "RL finetunes small subnetworks".
Questions: (i) Simulators: what is the "agent" in a base model, and what changes under RLHF?; (ii) Shanahan: restate "role play" as a claim about the posterior over personae; what evidence from persona vectors supports or undermines it?; (iii) Yue et al. and RL's Razor: if RL mostly reweights the base model's distribution and stays KL-close, what does that say about where capability comes from, and about the bitter lesson's "search"?
Exercise: write RLHF as a KL-regularized objective $\max_\pi\mathbb E_\pi[r]-\beta\,\mathrm{KL}(\pi\|\pi_0)$ and show the optimum is $\pi^*\propto\pi_0e^{r/\beta}$ (Levine / Korbak et al.). Interpretation: RLHF is a reweighting of the simulator, which is why "persona selection" is a fair description and why KL-closeness is structural rather than accidental.

### 7. The bitter lesson revisited
Discussion: Sutton's two general methods are search and learning. Pretraining is learning; is RL-on-LLMs search? What would distinguish "search" from "reweighting"? Card: one way the simulator frame changes an alignment plan and one way it does not.

### What carries forward
Compression (WB4); universality (WB5); personae (WB1 §4).

---

## WB4. TCS perspectives: Occam, Solomonoff, program synthesis

**Type:** derivations plus readings. **Needs:** L2; comfort with Turing machines at the "I know what one is" level. **Length:** 3–5 h.

### Purpose and boundary
Give the one correct chain from Occam to Kolmogorov complexity to the Solomonoff prior, locate exactly where "neural networks are approximately Solomonoff induction" claims enter, and set the program-synthesis account beside its competitors. Boundary: Kolmogorov complexity is defined up to an additive constant and is uncomputable; Solomonoff's prior is over sequences, not functions; every claim about networks is a claim about the parameter-function map's simplicity bias, which is a different object. (Note: the instructor is not a specialist here; the readings carry the authority, and the teaching note from the link-gathering pass is reproduced in `HANDOFF_MDL.md`.)

### Route
1. Occam in Bayes: the Occam factor
2. Kolmogorov complexity: definition, invariance, uncomputability, counting
3. Solomonoff induction: the prior, dominance, and the prediction bound
4. From Solomonoff to networks: simplicity bias of the parameter-function map
5. Program synthesis and its competitors
6. Compression bounds for large models
7. Depth, hardness, and what TCS can and cannot say

### 1. Occam in Bayes
- **1.1** Two models $\mathcal M_1\subset\mathcal M_2$; marginal likelihoods $Z_i=\int p(D|\theta,\mathcal M_i)\pi(\theta|\mathcal M_i)d\theta$. Laplace-approximate $Z$ to get $Z\approx p(D|\hat\theta)\cdot\pi(\hat\theta)\cdot(2\pi)^{p/2}|nH|^{-1/2}$; identify the Occam factor (MacKay ch. 28) and show it penalizes models whose prior volume is mostly wasted. Connect to WB2 §5.4 (BIC).
- **1.2** Question: the Occam factor rewards models that *predict* the data, not models that are *simple to describe*. Give a case where these disagree (a model with few parameters but a very wide prior).

### 2. Kolmogorov complexity
- **2.1** Define $K_U(x)=\min\{|p|:U(p)=x\}$ for a universal prefix machine $U$. Prove the invariance theorem: $|K_U(x)-K_{U'}(x)|\le c_{U,U'}$. Interpretation: "up to a constant" is the whole content; for finite objects the constant can dominate.
- **2.2** Counting: show that at most $2^{n-c}$ strings of length $n$ have $K(x)<n-c$; hence most strings are incompressible.
- **2.3** Uncomputability: prove $K$ is not computable (Berry-paradox argument: if it were, "the first string with $K>m$" would be describable in $O(\log m)$ bits).
- **2.4** Kraft and the prefix prior: show $\sum_p2^{-|p|}\le1$ over prefix-free programs, so $2^{-K(x)}$ is (up to normalization) a probability.

### 3. Solomonoff induction
- **3.1** Define the universal prior $M(x)=\sum_{p:U(p)=x*}2^{-|p|}$ (sum over programs whose output begins with $x$). Show $M$ is a semimeasure and that it dominates every lower-semicomputable semimeasure $\mu$: $M(x)\ge2^{-K(\mu)}\mu(x)$.
- **3.2** From dominance, derive the cumulative log-loss regret bound: $\sum_{t\le n}\big[-\log M(x_t|x_{<t})+\log\mu(x_t|x_{<t})\big]=-\log M(x_{1:n})+\log\mu(x_{1:n})\le K(\mu)\ln2$. State (do not prove) the stronger squared-error bound $\sum_t\mathbb E(M-\mu)^2\le\frac{\ln2}{2}K(\mu)$. Interpretation: a finite total regret means the predictor converges to any computable truth; the price is $K(\mu)$ bits, paid once.
- **3.3** Two claim boundaries, as exercises. (i) $M$ is uncomputable (reduce to §2.3). (ii) Legg's negative result: read the abstract and statement of "Is there an elegant universal theory of prediction?" and restate it: any predictor that learns all sequences of complexity up to $k$ has complexity about $k$ itself; there is no simple universal predictor. What does this say about hoping a short training recipe "is" Solomonoff?
- **3.4** Discussion: Solomonoff's prior is over *sequences*; a neural network is a *function*. Write the two objects side by side and say what map between them a "networks do Solomonoff" claim must supply.

### 4. From Solomonoff to networks: simplicity bias of the parameter-function map
- **4.1** Dingle–Camargo–Louis: for a map from parameters to outputs, $P(f)\lesssim2^{-aK(f)+b}$ for most outputs. Exercise: for a random Boolean function implemented by a tiny net (2 inputs, 2 hidden units, sign activations) enumerate the 16 functions, count how many parameter settings in a small grid produce each, and see the bias toward constant and single-variable functions. (Helper cell does the count after the student predicts the ranking.)
- **4.2** Readings: Valle-Pérez et al. "Deep learning generalizes because the parameter-function map is biased towards simple functions"; Mingard et al. "a priori biased towards Boolean functions with low entropy"; Goldblum et al. "NFL, Kolmogorov complexity, and inductive biases"; Bhattamishra et al. on transformers and sparse Boolean functions. Questions: (i) in Valle-Pérez, what complexity measure stands in for $K$, and why?; (ii) Goldblum: how do they argue that real data has low $K$ relative to uniform, and what is the evidence?; (iii) what does "simplicity bias" of the *prior* say about the *trained* network, and what extra assumption (WB2 §6: SGD ≈ sampler?) is needed to connect them?
- **4.3** Grau-Moya et al. "Learning universal predictors": transformers meta-trained on UTM outputs approximate Solomonoff on-distribution. Question: what breaks when the test sequence is generated by a program longer than those seen in training?

### 5. Program synthesis and its competitors
Readings: Furman "Deep learning as program synthesis" (core); Hänni–Vaintrob "A circuit prior in NN-Bayes"; "Programs as singularities" (if confirmed); Wilson "Not so mysterious"; Ellis et al. DreamCoder; Chollet "On the measure of intelligence" §II; Schmidhuber "Speed prior".
Exercises:
- **5.1** Write the three accounts as three priors over functions: (a) a prior that decays in program length; (b) a prior that decays in circuit size at fixed depth; (c) a "soft inductive bias" prior that is broad but decays in a norm. For each, name one phenomenon it predicts and one it does not obviously predict (candidates: grokking, double descent, scaling-law exponents, representational convergence, adversarial examples).
- **5.2** Furman's argument rests on tractable search over simple compositional programs. Which step is the empirical one? What measurement on a trained network would test it (hint: WB5 §1–2's representation-similarity tools)?
- **5.3** Speed prior vs Solomonoff: show that weighting programs by $2^{-|p|}\cdot(\text{runtime})^{-1}$ makes the prior computable-in-the-limit and ask whether a fixed-depth network is better modelled by a speed prior than a length prior.
- **5.4** "Programs as singularities": a noisy Turing machine's valid execution traces are the zero set of a loss; the claim is that programs are singularities of that loss. Question: this is a least-action construction with a white-noise metric; what does the choice of metric do to which programs count as "near" each other? (Connects to the geometric-naturalness theme in TD.)

### 6. Compression bounds for large models
- **6.1** Prefix-code (Occam) bound: for any prefix-free description scheme with $|h|$ bits for hypothesis $h$, with probability $1-\delta$, $R(h)\le\hat R_n(h)+\sqrt{(|h|\ln2+\ln(1/\delta))/2n}$. Prove from Hoeffding plus a union bound with weights $2^{-|h|}$ (Kraft). Interpretation: this is the finite-class bound of WB1 §1.3 with the class replaced by a code.
- **6.2** Reading: Lotfi et al. "Non-vacuous generalization bounds for LLMs". Question: what is compressed, and why does the bound become nonvacuous where VC bounds (WB2 §2.1) did not? Then: Hochreiter–Schmidhuber "Flat minima" as MDL; Arora et al. compression bounds. Interpretation: the only nonvacuous bounds at scale are compression bounds, which is a point for the TCS picture and a point against capacity-based explanations.

### 7. Depth, hardness, and what TCS can and cannot say
- **7.1** Telgarsky's sawtooth (from WB2 §2.3): prove depth-$k$ constructs $2^k$ pieces with $O(k)$ units, and state the lower bound for shallow nets.
- **7.2** Shalev-Shwartz–Ben-David §8.4: learning parity with noise / cryptographic hardness. Question: a hardness result says no efficient learner exists for *some* distribution in a class; what must a positive explanation of deep learning therefore assume about the *target*? (This is the Zach Furman "Mysteries" framing of optimization; keep it.)
- **7.3** Abbe et al. leap complexity (optional): SGD learns functions with low "leap" in a staircase; read the abstract and say what the structural assumption on the target is.

### What carries forward
Priors over functions (WB5 for representational convergence; TD for the dynamics that actually selects); compression (WB3 §3).

---

## WB5. Universality: what is happening inside, and what scaling laws do and do not say

**Type:** derivations plus readings. **Needs:** L2; linear algebra. **Length:** 3–5 h.

### Purpose and boundary
Make precise the evidence that different networks learn the same thing: similarity measures, stitching, mode connectivity, the Platonic Representation Hypothesis and its identifiability backbone, the Linear Representation Hypothesis and superposition. Then derive where scaling exponents come from in a solvable model, so the "triviality" claim has content. Boundary: representational similarity is measured up to a stated transformation group; "same" always means "same up to that group". Scaling-law derivations are for one model class and say nothing about what is learned.

### Route
1. Similarity measures and what they are invariant to
2. Stitching, linear mode connectivity, and permutation symmetry
3. The Platonic Representation Hypothesis and linear identifiability
4. The Linear Representation Hypothesis and its counterexamples
5. Superposition: how many features fit
6. Scaling laws: where the exponent comes from
7. What is happening inside: learning order, induction heads, critical periods

### 1. Similarity measures
- **1.1** Define linear CKA: $\mathrm{CKA}(X,Y)=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\|Y^\top Y\|_F}$ for centered activation matrices. Prove invariance to orthogonal transformations and isotropic scaling; show by a 2-D example that it is *not* invariant to general invertible linear maps. Then SVCCA/CCA: invariant to invertible linear maps; explain why Kornblith et al. argue that is too much invariance. Interpretation: every similarity claim fixes a group first.
- **1.2** Reading: Kornblith et al.; Raghu et al. SVCCA (the bottom-up learning result). Question: "lower layers converge first" is measured with which invariance, and would the conclusion survive a different choice?

### 2. Stitching and mode connectivity
- **2.1** Model stitching (Bansal–Nakkiran–Barak): define the stitched network with a learned affine map at layer $\ell$; explain why low stitching penalty is a *functional* similarity claim that CKA is not. Question: what does it mean that stitching works across architectures and objectives?
- **2.2** Permutation symmetry. Count the symmetries of an $L$-layer MLP with widths $n_\ell$: $\prod_\ell n_\ell!$ (times scaling symmetries for homogeneous activations). Exercise: show that linear interpolation between two independently trained networks generically crosses a loss barrier, and that after aligning permutations (Ainsworth et al. Git Re-Basin; Entezari et al.) the barrier can vanish. Interpretation: "different solutions" are often the same solution in different coordinates.
- **2.3** Reading: Frankle et al. "Linear mode connectivity and the lottery ticket hypothesis". Question: the network becomes "stable to SGD noise" early; what does this say about when the solution is determined?

### 3. PRH and linear identifiability
- **3.1** Reading: Huh et al. "Platonic representation hypothesis" §1–3 and the "why" section. Questions: (i) list their three stated pressures toward convergence and say which is a theorem and which a conjecture; (ii) convergence is measured with which similarity, and on which transformation group?
- **3.2 The perfect PRH, worked through.** Ziyin–Chuang, "Proof of a perfect platonic representation hypothesis" (2025): for embedded deep linear networks trained by SGD, every pair of layers, across widths, depths, and even training sets, converges to the same representation up to rotation; the mechanism is an entropic force from SGD's irreversibility, and it is tied to progressive sharpening. Exercises: (i) set up their model and state exactly what "same representation" means (which group); (ii) reproduce the central calculation showing the SGD-noise term selects the balanced/aligned solution that gradient flow alone does not; (iii) say which assumptions (linearity, SGD noise structure, infinite time) carry the result and which could be relaxed. Interpretation: this is a *dynamical* convergence result, not an identifiability result; the convergence comes from the optimizer's noise, which is the Training Dynamics day's territory. Cross-reference TD on the stationary law of SGD and WB2 §6.3.
- **3.2b The identifiability backbone.** Roeder–Metz–Kingma: for discriminative models $p(y|x)\propto\exp(f_\theta(x)^\top g_\theta(y))$ with sufficient diversity, if two models have the same likelihood then $f_{\theta_1}=Af_{\theta_2}+b$ for an invertible $A$. Exercise: prove the simplest case (finite $y$, full-rank $g$) by writing log-odds between labels and matching them. Reizinger et al. extend this to cross-entropy on generative processes. Interpretation: identifiability says *what* converges and up to what group, given a perfect fit; Ziyin–Chuang says *why* finite SGD-trained models get there in one solvable case. Claim boundary: neither says the representation is "of reality" rather than of the data, and the two use different groups (invertible linear vs rotation).
- **3.3** Reading: Jha et al. "Harnessing the universal geometry of embeddings" (vec2vec). Question: unsupervised translation between embedding spaces works; is this evidence for §3.2's linear map, or for something weaker?
- **3.4** Discussion: write what evidence would *falsify* PRH, and whether §3.2 makes it unfalsifiable in the limit.

### 4. LRH and counterexamples
- **4.1** Mikolov's parallelogram; then Park–Choe–Veitch: define the causal inner product and show that under it, causally separable concepts are orthogonal (work the two-concept case from their Theorem). Question: what is assumed about the unembedding?
- **4.2** Park et al. "geometry of categorical and hierarchical concepts": hierarchy as orthogonality of child-minus-parent directions. Exercise: show this is consistent with §4.1 for a three-level hierarchy.
- **4.3** Counterpoint: Engels et al. "Not all features are linear" (circular features for days/months). Question: what notion of "linear" is violated, and does §4.1's framework survive if features live on low-dimensional manifolds inside linear subspaces?

### 5. Superposition
- **5.1** Johnson–Lindenstrauss counting: show one can place $\exp(\Theta(\epsilon^2d))$ vectors in $\mathbb R^d$ with pairwise $|\cos|\le\epsilon$. Interpretation: many more "almost-orthogonal" features than dimensions.
- **5.2** Elhage et al. toy model: for the $m$-feature, $d$-dimension autoencoder with sparsity $S$, derive the loss of the "dedicated dimensions" solution and of the "superposition" solution in the limit of small interference, and find the sparsity at which superposition wins. (Helper cell plots the phase diagram after prediction.)
- **5.3** Question: relate superposition to adversarial examples (Ilyas et al. "features not bugs") in two lines.

### 6. Scaling laws: where the exponent comes from
- **6.1** Kernel/random-feature regression with eigenvalues $\lambda_k\sim k^{-\alpha}$ and target coefficients with $\sum_{k>K}c_k^2\sim K^{-(\alpha'-1)}$. Derive, in the spectral-filter language of WB2 §3.3, that with $n$ samples only the top $\sim n$ modes are resolved and the test loss scales as $n^{-(\alpha'-1)}$ (or $n^{-(\alpha-1)}$ when the target is "generic"). Then with $P$ random features, the same bottleneck gives $P^{-(\alpha-1)}$ (Maloney–Roberts–Sully; Bahri et al.'s resolution-limited regime). Interpretation: in this model class the exponent is a property of the data covariance spectrum. Sharma–Kaplan: $\alpha\approx4/d$ from the intrinsic dimension of the data manifold. Claim boundary: the derivation is for a linear model; it explains the *form* and the *exponent*, not what is learned, which is the content of CRW's "Power laws are not enough".
- **6.2** Readings: Bahri et al. (four regimes); Michaud et al. "quantization model" (discrete skills with Zipfian frequencies give the same power law: a different mechanism, same curve); Hutter "Learning curve theory"; Bordelon et al. dynamical model. Question: two mechanisms (spectral vs quanta) produce the same exponent; what measurement distinguishes them?
- **6.3** Emergence: Wei et al. vs Schaeffer et al. Exercise: take a smooth per-token loss curve and a metric that is 1 iff all $T$ tokens of an answer are correct; show the metric is $\sim(1-\ell)^T$ and jumps "emergently" for large $T$. Question: is there any emergence left after choosing a continuous metric? Steinhardt's "More is different" as the counter-position.

### 7. What is happening inside
Readings: Olsson et al. induction heads (phase change in ICL); Raghu SVCCA (bottom-up); Achille et al. critical periods; Nanda et al. grokking progress measures; Gur-Ari et al. tiny subspace with Song et al. intervention. Questions: (i) induction-head formation is a phase change in training; is it a phase change in the model-size scaling curve too?; (ii) critical periods imply the solution is path-dependent; reconcile with §3.2's identifiability.
Card: what would falsify PRH, or LRH, or the spectral account of scaling.

### What carries forward
Similarity groups and representational drift (WB1 §3.1; TD); spectra and the data covariance (TD); learning order and critical periods (TD).

---

## WB6. Fun readings about weird stuff

**Type:** reading cards. **Needs:** nothing. **Length:** 45 min per paper.

Each paper gets a one-paragraph framing, three comprehension questions, and the card template. Students take one or two.

| Paper | Framing | Questions |
|---|---|---|
| Bansal–Nakkiran–Barak, model stitching | good networks learn interchangeable representations | What is the stitching layer allowed to be? Does the result hold across objectives? What would break it? |
| Raghu et al. SVCCA | layers converge bottom-up | Which invariance? Would "freeze training" work for transformers? |
| Achille–Rovere–Soatto critical periods | early deficits are permanent | What is the Fisher-information signature? Relation to loss of plasticity (Dohare et al.)? |
| Frankle–Carbin lottery tickets; Frankle et al. LMC | sparse trainable subnetworks exist from early in training | Does rewinding to init work at scale? What does LMC say about when the basin is chosen? |
| Gur-Ari–Roberts–Dyer tiny subspace; Song et al. | gradient lives in the top Hessian subspace | Does learning happen there? What does the projection intervention show? |
| Ilyas et al. adversarial features | non-robust features are predictive | Is this a statement about the data or the model? How does it relate to superposition? |
| Power et al. grokking; Nanda et al. | generalization long after memorization | Which progress measure predicts the transition? What does weight decay do? |
| Olsson et al. induction heads | ICL arrives in a phase change | What is the mechanistic claim? What would a counterexample look like? |
| Elhage et al. superposition | more features than dimensions | What determines the phase boundary? |
| Hinton et al. distillation; pruning literature (one survey) | small models can carry big-model functions | What is transferred, logits or representations? |
| Schaeffer et al. mirage | emergence is a metric artifact | Which emergent abilities survive a continuous metric? |
| Mukherjee et al.; Zhu et al.; Cai et al. RL update structure | RL changes little, in a structured way | Sparse, low-rank, or off-principal? Which is the right description? |
| Kadkhodaie et al. GAHB | two denoisers on disjoint data learn the same function | Is this PRH for diffusion? What is the inductive bias named? |
| Huh et al. PRH; Jha et al. vec2vec | representations converge | (see WB5) |
| CRW "What if not circuits" | circuits may be the wrong ontology | What replaces them? What measurement would decide? |

Card: claim / evidence / what would change my mind / does it survive scale.

---

## Build notes
- Builders: one `build_wb{1..6}.py` each, standard library only, same `markdown()` / `answer()` helpers and `{{VARIANT}}` mechanism as the Physics day; a `check_suite.py` that verifies exercise/solution alignment and link liveness.
- Supplied artefacts: WB2 §7 sweep (generate on Modal: ~300 small CNNs on CIFAR-10 subsets across width × depth × lr × batch × wd × 3 seeds; record test gap, Hessian top-$k$ and trace estimates, gradient covariance diagonal, norms, margins); WB2 §6.2 diagonal $C$ vs $H$ scatter; WB2 §4.3 risk-vs-$\gamma$ figure; WB4 §4.1 Boolean-function count; WB5 §5.2 phase diagram. All as static PNG plus a small CSV so the helper cells have nothing to compute.
- Reading links: `CURRICULUM_DRAFT_v1.md` §4 for everything verified so far; the WB1/WB3/WB4/WB5-specific links are appended at the end of this file as they are verified.

---

## Appendix: workbook-specific reading links (verified 2026-10-07 unless marked UNSURE)

### WB1 links
**GOFAI and the three pictures.** Nilsson, *The Quest for Artificial Intelligence* (2010), free PDF https://ai.stanford.edu/~nilsson/QAI/qai.pdf · Newell & Simon, "Computer science as empirical inquiry: symbols and search" (Turing lecture, CACM 1976), free copy https://schiaffonati.faculty.polimi.it/TFI/newell_simon_1976.pdf · Lighthill report (1973) http://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm · SEP "Artificial intelligence" https://plato.stanford.edu/entries/artificial-intelligence/ · Bellman, "The theory of dynamic programming" (BAMS 1954) https://www.ams.org/bull/1954-60-06/S0002-9904-1954-09845-2/S0002-9904-1954-09845-2.pdf · Sutton & Barto, *RL: An Introduction* 2e, http://incompleteideas.net/book/the-book-2nd.html (ch. 1, §1.7 history) · Silver, Singh, Precup, Sutton, "Reward is enough" (AIJ 2021), DOI 10.1016/j.artint.2021.103535 (UNSURE: publisher page not fetched) · Irpan, "Deep RL doesn't work yet" (2018) https://www.alexirpan.com/2018/02/14/rl-hard.html · Levine, "RL and control as probabilistic inference" (2018) https://arxiv.org/abs/1805.00909 · Recht, "A tour of RL: the view from continuous control" (2018) https://arxiv.org/abs/1806.09460 · Valiant, "A theory of the learnable" (1984) https://www.cs.princeton.edu/courses/archive/spring13/cos511/handouts/valiant.pdf · Vapnik, *The Nature of Statistical Learning Theory* (Springer) https://link.springer.com/us/book/9780387987804.
**Why the first two lost.** Sutton, "The bitter lesson" http://www.incompleteideas.net/IncIdeas/BitterLesson.html · Hooker, "The hardware lottery" https://arxiv.org/abs/2009.06489 · Halevy, Norvig, Pereira, "The unreasonable effectiveness of data" (2009) https://research.google/pubs/pub35179/ · Breiman, "Statistical modeling: the two cultures" (2001) https://projecteuclid.org/journals/statistical-science/volume-16/issue-3/Statistical-Modeling--The-Two-Cultures-with-comments-and-a/10.1214/ss/1009213726.full · Krizhevsky, Sutskever, Hinton, AlexNet (2012) https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html.
**Old alignment problems.** Strawberry problem: Yudkowsky tweet (2018) https://twitter.com/ESYudkowsky/status/1070095840608366594 and the Bankless transcript (2023) https://www.lesswrong.com/posts/Aq82XqYhgqdPdPrBA/full-transcript-eliezer-yudkowsky-on-the-bankless-podcast (note: not in "AGI Ruin" or the Arbital task-AGI page) · Yudkowsky, "AGI Ruin: a list of lethalities" (2022) https://www.lesswrong.com/posts/uMQ3cqWDPHhjtiesc/agi-ruin-a-list-of-lethalities · de Blanc, "Ontological crises in artificial agents' value systems" (2011) https://arxiv.org/abs/1105.3821 · Soares, Fallenstein, Yudkowsky, "Corrigibility" (AAAI-W 2015; three authors) https://intelligence.org/files/Corrigibility.pdf · Bostrom, "The superintelligent will" (2012) https://nickbostrom.com/superintelligentwill.pdf · Omohundro, "The basic AI drives" (2008) https://selfawaresystems.com/wp-content/uploads/2008/01/ai_drives_final.pdf · Turner et al., "Optimal policies tend to seek power" (NeurIPS 2021) https://arxiv.org/abs/1912.01683 (UNSURE: not re-verified this pass) · Byrnes, "Intro to brain-like AGI safety" sequence https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8.
**New alignment problems.** Christiano, "Prosaic AI alignment" (2016) https://www.alignmentforum.org/posts/YTq4X6inEudiHkHDF/prosaic-ai-alignment · Hubinger et al., "Risks from learned optimization" (2019) https://arxiv.org/abs/1906.01820 · Amodei et al., "Concrete problems in AI safety" (2016) https://arxiv.org/abs/1606.06565 · Turner, "Reward is not the optimization target" (2022) https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target · Wentworth, "Testing the natural abstraction hypothesis" (2021) https://www.lesswrong.com/posts/cy3BhHrGinZCp3LXE/testing-the-natural-abstraction-hypothesis-project-intro ; McDougall (not Wentworth), "NAH: implications and evidence" https://www.lesswrong.com/posts/Fut8dtFsBYRz8atFF/the-natural-abstraction-hypothesis-implications-and-evidence · Wentworth, "Just retarget the search" https://www.lesswrong.com/posts/w4aeAFzSAguvqA5qu/how-to-go-from-interpretability-to-alignment-just-retarget-the-search · Hubinger, "How likely is deceptive alignment?" (2022) https://www.lesswrong.com/posts/A9NxPTwbw6r6Awuwt/how-likely-is-deceptive-alignment · Greenblatt et al., "Alignment faking in LLMs" (2024) https://arxiv.org/abs/2412.14093 · Betley et al., "Emergent misalignment" (2025) https://arxiv.org/abs/2502.17424 · Chen et al., "Persona vectors" (2025) https://arxiv.org/abs/2507.21509 · Shanahan, McDonell, Reynolds, "Role play with LLMs" (Nature 2023) https://www.nature.com/articles/s41586-023-06647-8 · Kulveit, "A three-layer model of LLM psychology" (2024) https://www.lesswrong.com/posts/zuXo9imNKYspu9HGv/a-three-layer-model-of-llm-psychology · Nardo, "The Waluigi effect" https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7 · janus, "Simulators" https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators.

### WB3 links
**Bits, pretraining, why text.** LeCun cake: NIPS 2016 talk page https://nips.cc/virtual/2016/invited-talk/6197 , video https://www.youtube.com/watch?v=Ount2Y4qxQo , Ruder writeup https://www.ruder.io/highlights-nips-2016/ · Dai & Le, "Semi-supervised sequence learning" (2015) https://arxiv.org/abs/1511.01432 · Howard & Ruder, ULMFiT (2018) https://arxiv.org/abs/1801.06146 · Devlin et al., BERT https://arxiv.org/abs/1810.04805 · GPT-1 https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf · GPT-2 https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf · GPT-3 https://arxiv.org/abs/2005.14165 · LeCun, "A path towards autonomous machine intelligence" (2022) https://openreview.net/forum?id=BZ5a1r-kVsf · Sutskever, "An observation on generalization" (Simons 2023) https://www.youtube.com/watch?v=AKMuA_TVz3A · Henighan et al., "Scaling laws for autoregressive generative modeling" (2020) https://arxiv.org/abs/2010.14701.
**Compression.** Delétang et al., "Language modeling is compression" https://arxiv.org/abs/2309.10668.
**Architectures.** Vaswani et al., "Attention is all you need" https://arxiv.org/abs/1706.03762 · Hochreiter & Schmidhuber, LSTM (1997) https://www.gwern.net/doc/ai/nn/rnn/1997-hochreiter.pdf · LeCun et al., LeNet (1998) https://leon.bottou.org/papers/lecun-98h · Bronstein et al., GDL https://arxiv.org/abs/2104.13478 · Joshi, "Transformers are GNNs" https://thegradient.pub/transformers-are-graph-neural-networks/ · Tay et al., "Efficient transformers: a survey" https://arxiv.org/abs/2009.06732 · Tay et al., "Scaling laws vs model architectures" (2022) https://arxiv.org/abs/2207.10551 · Gu & Dao, Mamba https://arxiv.org/abs/2312.00752 · Poli et al., Hyena https://arxiv.org/abs/2302.10866.
**Scale.** Kaplan et al. https://arxiv.org/abs/2001.08361 · Chinchilla https://arxiv.org/abs/2203.15556 · Muennighoff et al., data-constrained scaling https://arxiv.org/abs/2305.16264 · Gunasekar et al., "Textbooks are all you need" https://arxiv.org/abs/2306.11644 · Villalobos et al., "Will we run out of data?" https://arxiv.org/abs/2211.04325 · Olsson et al., induction heads https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html · Wei et al., "Emergent abilities" https://arxiv.org/abs/2206.07682 · Schaeffer et al., "Mirage" https://arxiv.org/abs/2304.15004.
**Simulators, personae, RL on top.** janus, "Simulators" (above) · Nardo, "Waluigi" (above) · Shanahan et al. (above) · Kulveit (above) · Chen et al., persona vectors (above) · janus, "Mysteries of mode collapse" https://www.lesswrong.com/posts/t9svvNPNmFf5Qa3TA (with the later correction that text-davinci-002 was probably not RLHF) · Dupuis & janus, "Cyborgism" https://www.lesswrong.com/posts/bxt7uCiHam4QXrQAA/cyborgism · Ouyang et al., InstructGPT https://arxiv.org/abs/2203.02155 · Bai et al., Constitutional AI https://arxiv.org/abs/2212.08073 · Lambert, *RLHF book* https://rlhfbook.com/ · DeepSeek-R1 https://arxiv.org/abs/2501.12948 · Yue et al., "Does RL really incentivize reasoning beyond the base model?" https://arxiv.org/abs/2504.13837 · Shenfeld et al., "RL's Razor" https://arxiv.org/abs/2509.04259 · Mukherjee et al., "RL finetunes small subnetworks" https://arxiv.org/abs/2505.11711.

### WB4 links
**Primers, in teaching order.** MacKay, *ITILA* ch. 28 https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/MacKay2003-Ch28.pdf · Altair, "An intuitive explanation of Solomonoff induction" (2012) https://www.lesswrong.com/posts/Kyc5dFDzBg4WccrbK/an-intuitive-explanation-of-solomonoff-induction · Legg, *Machine Super Intelligence* (2008) ch. 2 https://www.vetta.org/documents/Machine_Super_Intelligence.pdf · Grünwald, "A tutorial introduction to the MDL principle" (2004) https://arxiv.org/abs/math/0406077 · Vitányi & Li, "MDL induction, Bayesianism, and Kolmogorov complexity" (2000) https://arxiv.org/abs/cs/9901014 · Hutter, "A theory of universal AI based on algorithmic complexity" (2000) https://arxiv.org/abs/cs/0004001 · Li & Vitányi, *Kolmogorov Complexity* 4e https://link.springer.com/book/10.1007/978-3-030-11298-1 · Solomonoff 1964 Parts I/II, DOI 10.1016/S0019-9958(64)90223-2 and 10.1016/S0019-9958(64)90131-7 · Legg, "Is there an elegant universal theory of prediction?" (2006) https://arxiv.org/abs/cs/0606070.
**Simplicity bias.** Valle-Pérez, Camargo, Louis (ICLR 2019) https://arxiv.org/abs/1805.08522 · Dingle, Camargo, Louis (Nat. Comm. 2018) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5823903/ · Mingard et al. (2019) https://arxiv.org/abs/1909.11522 · Goldblum et al., "NFL, Kolmogorov complexity, and inductive biases" (ICML 2024) https://arxiv.org/abs/2304.05366 · Bhattamishra et al. (2022) https://arxiv.org/abs/2211.12316.
**Solomonoff and transformers; compression.** Grau-Moya et al., "Learning universal predictors" (2024) https://arxiv.org/abs/2401.14953 · Delétang et al. (above) · Sutskever (above) · Hutter, "Learning curve theory" (2021) https://arxiv.org/abs/2102.04074 · Lotfi et al., "Non-vacuous generalization bounds for LLMs" (2024) https://arxiv.org/abs/2312.17173.
**Program synthesis.** Furman, "Deep learning as program synthesis" (2026) https://www.lesswrong.com/posts/Dw8mskAvBX37MxvXo/deep-learning-as-program-synthesis-1 · Hänni & Vaintrob, "A circuit prior in NN-Bayes" (2026) https://www.lesswrong.com/posts/SqDHeuycNkERurtSc/a-circuit-prior-in-nn-bayes · Murfet & Troiani, "Programs as singularities" (2025; Waring is not an author) https://arxiv.org/abs/2504.08075 · Ellis et al., DreamCoder https://arxiv.org/abs/2006.08381 · Chollet, "On the measure of intelligence" https://arxiv.org/abs/1911.01547 · Schmidhuber, "The speed prior" (COLT 2002) https://people.idsia.ch/~juergen/speedprior.html · Levin search, Scholarpedia exposition http://www.scholarpedia.org/article/Universal_search · Rule, Tenenbaum, Piantadosi, "The child as hacker" (2020) https://pmc.ncbi.nlm.nih.gov/articles/PMC7673661/.
**Depth and hardness.** Poggio et al. (2017) https://arxiv.org/abs/1611.00740 · Telgarsky, "Benefits of depth" (COLT 2016) https://arxiv.org/abs/1602.04485 · Shalev-Shwartz & Ben-David, *UML* §8.4 (section number confirmed) https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf · Abbe & Sandon, "Provable limitations of deep learning" https://arxiv.org/abs/1812.06369 · Abbe, Boix-Adserà, Misiakiewicz, merged staircase https://arxiv.org/abs/2202.08658 and leap complexity https://arxiv.org/abs/2302.11055.

### WB5 links
**PRH.** Huh et al. https://arxiv.org/abs/2405.07987 · **Ziyin & Chuang, "Proof of a perfect platonic representation hypothesis" (2025)** https://arxiv.org/abs/2507.01098 · Roeder, Metz, Kingma, "On linear identifiability of learned representations" (ICML 2021) https://arxiv.org/abs/2007.00810 · Reizinger et al., "Cross-entropy is all you need to invert the data generating process" (ICLR 2025) https://arxiv.org/abs/2410.21869 · Jha et al., vec2vec (2025) https://arxiv.org/abs/2505.12540 · Gröger, Wen, Brbić, "An Aristotelian view" (2026) https://arxiv.org/abs/2602.14486 · You et al., "Structure over geometry" (2026) https://arxiv.org/abs/2609.27252 · Ciernik et al., "Objective drives the consistency of representational similarity" https://arxiv.org/abs/2411.05561.
**LRH and superposition.** Park, Choe, Veitch https://arxiv.org/abs/2311.03658 · Park, Choe, Jiang, Veitch, categorical/hierarchical geometry https://arxiv.org/abs/2406.01506 · Mikolov, Yih, Zweig (2013) https://aclanthology.org/N13-1090/ · Elhage et al., superposition https://transformer-circuits.pub/2022/toy_model/index.html · Engels et al., "Not all features are one-dimensionally linear" https://arxiv.org/abs/2405.14860.
**Similarity and universality.** Kornblith et al., CKA https://arxiv.org/abs/1905.00414 · Bansal, Nakkiran, Barak https://arxiv.org/abs/2106.07682 · Raghu et al., SVCCA https://arxiv.org/abs/1706.05806 · Li et al., "Convergent learning" (2016) https://arxiv.org/abs/1511.07543 · Ainsworth et al., Git Re-Basin https://arxiv.org/abs/2209.04836 · Entezari et al. https://arxiv.org/abs/2110.06296 · Olah et al., "Zoom in" https://distill.pub/2020/circuits/zoom-in/ · Chughtai, Chan, Nanda, "A toy model of universality" https://arxiv.org/abs/2302.03025 · Frankle et al., LMC https://arxiv.org/abs/1912.05671.
**Scaling laws.** Sharma & Kaplan, data-manifold dimension https://arxiv.org/abs/2004.10802 · Bahri et al. https://arxiv.org/abs/2102.06701 · Maloney, Roberts, Sully https://arxiv.org/abs/2210.16859 · Hutter (above) · Michaud et al., quantization model https://arxiv.org/abs/2303.13506 · Bordelon, Atanasov, Pehlevan https://arxiv.org/abs/2402.01092 · Paquette et al., "4+3 phases" (NeurIPS 2024) https://arxiv.org/abs/2405.15074 · Wei et al.; Schaeffer et al. (above) · Steinhardt, "More is different for AI" https://bounded-regret.ghost.io/more-is-different-for-ai/.

### WB6 links
All in `CURRICULUM_DRAFT_v1.md` §4D–E; plus Hinton, Vinyals, Dean, "Distilling the knowledge in a neural network" (2015) https://arxiv.org/abs/1503.02531 (UNSURE: not re-verified this pass).
