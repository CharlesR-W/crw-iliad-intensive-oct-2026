# L1: What did this training example do? (outline and speaker notes)

**Length:** about 25 min. A compressed path of about 12 min is at the end.
**Hands off to:** `da1_core` (DA1), then `da2_methods` (DA2) after lunch.
**Purpose:** give everyone the question, the one formula, and the habit of asking "which question does this method answer?". DA1 develops each segment; nothing here is derived on the board that DA1 does not derive properly.

**Notation on the board (matches DA1 §2).** Training examples $z_i$, $i=1,\dots,n$, with per-example losses $L_i(w)$. Data weights $\beta_i$ and the **summed** objective $L(w;\beta)=\sum_i\beta_iL_i(w)$; $\beta=\mathbf 1$ is the real run. Observable $\phi(w)$ (for example a test loss). "The trained model" $\pi(\beta)$. Influence $\mathcal I_i=\partial\phi(\pi(\beta))/\partial\beta_i$ at $\beta=\mathbf 1$; deletion is $\beta_i:1\to0$, so it changes $\phi$ by about $-\mathcal I_i$. A helpful example has negative influence on a test loss. For segment 4: OLS with $A=\sum_ix_ix_i^\top$, residual $r_j=y_j-x_j^\top w^*$ and leverage $h_j=x_j^\top A^{-1}x_j$.

Each segment lists: time, the claim, at most two board equations, what to show, and where the notebook picks it up.

---

## 1. The question, with one alignment example (3 min)

**Claim.** Data attribution asks which training examples made a model behave as it does. It is causal analysis of a training run, as mechanistic interpretability is causal analysis of a forward pass. Its ablations cost a retraining each, so most methods predict retraining rather than doing it.

**Example to tell.** Grosse et al. (2023, Fig. 1) ran influence functions on a dialogue in which an AI assistant says it does not consent to being shut down. They computed influences on pretrained models of two sizes.
- For the 810M-parameter model, the top training sequences share tokens with the query ("continue", "existing") but are unrelated in meaning.
- For the 52B model, the top sequences are about wanting to keep existing. One is science fiction in which an AI (Hal) pleads with its crew; another is a person trying to survive in a desert.
- That is the kind of answer people want from attribution: "this behaviour came from imitating these kinds of documents". It is also exactly the kind of claim this morning teaches you to check.

**Speaker notes.**
- Say that the example uses EK-FAC influence functions, which are DA2 stop 1. Do not over-read it: Grosse et al. present it as consistent with role-play by imitation, at a high level of abstraction. It is not shown to be causal by retraining.
- **Picked up in:** DA1 §1 (the causal framing and the cost of retraining); the example itself is DA2 stop 1.

---

## 2. Why counterfactuals are subtle (4 min)

**Claim.** The obvious definition, that $i$ caused the behaviour if removing $i$ and retraining would change it (leave-one-out), fails in two standard ways. Shapley values fix one of them partly, at exponential cost.

**Board.** None. Tell the two stories.

- **Overdetermination (the croissant).** Premium butter or premium yeast is each enough for fluffy croissants. Downgrading either alone changes nothing, so leave-one-out credits neither. Training analogue: a behaviour taught by many redundant examples.
- **Non-transitivity (the boulder).** The boulder makes the hiker duck, and ducking saves her, but her survival doesn't depend on the boulder: counterfactual dependence doesn't chain. Training analogue: an example teaches a feature that a capability uses, without passing the end-to-end test.
- **Shapley:** average the marginal effect over all subsets. The croissant gets butter ½ and yeast ½. This is a finite-coalition attribution, not a derivative. It costs exponentially many retrainings, and practical versions sample.

**Speaker notes.**
- The point to land is not philosophy. Different methods answer different counterfactual questions, so "the influence of example $i$" is ambiguous until you say which.
- **Picked up in:** DA1 §1 (Mueller 2024 is the reading; it is written about interpretability, and the same failures apply).

---

## 3. Data weights, targets and estimators (5 min)

**Claim.** Give each example a weight and differentiate. The response-based methods of the day compute something of the form $\partial\phi(\pi(\beta))/\partial\beta_i$, or a finite version of it, and differ mainly in what they take $\pi$, "the trained model", to be. A method's *target* (which question) is separate from its *estimator* (how it computes the answer).

**Board.**
$$
L(w;\beta)=\sum_i\beta_iL_i(w),\qquad \beta\xrightarrow{\ \pi\ }\pi(\beta)\xrightarrow{\ \phi\ }\phi(\pi(\beta)),\qquad \mathcal I_i=\frac{\partial\phi(\pi(\beta))}{\partial\beta_i}\Big|_{\beta=\mathbf 1}
$$

**Picture to draw.** The map table from DA1 §2, as five rows: classical IF (a stationary point), PBRF (a proximal objective around the checkpoint), unrolling (this training run, this seed), Bayesian IF (a local posterior), datamodels (a fitted predictor of retraining). Then three targets side by side: derivative, leave-one-out, deleting half the data.

**Speaker notes.**
- Conventions, said once: summed loss; upweighting derivative; deletion is $\beta_i:1\to0$; a helpful example has negative influence on a test loss. Koh and Liang use the mean loss plus $\varepsilon L_i$, so their $\mathcal I_{\rm up}$ is $n$ times ours. TracIn, TRAK and SOURCE report helpful as positive.
- TRAK is not a row: it is an analytic estimator judged against the datamodel-style target.
- **Picked up in:** DA1 §2 (the Koh–Liang conversion is the optional Exercise 2.1).

---

## 4. The influence function in one line, leverage, and what breaks (7 min)

**Claim.** At a strict minimum $w^*$ with invertible Hessian $H$, the implicit function theorem gives the response of the stationary point in one line. In linear regression the exact deletion effect is that prediction times $1/(1-h_j)$, where $h_j$ is the leverage. For neural networks three things break, and each fix changes the question.

**Board.**
$$
\mathcal I_i=-\,\nabla\phi(w^*)^\top H^{-1}\nabla L_i(w^*),\qquad
w^*_{(-j)}-w^*=-\frac{A^{-1}x_jr_j}{1-h_j}\quad\text{(OLS; IF keeps the leading 1)}
$$

**Show.** The DA1 §3 figure: removing a low-leverage point (prediction indistinguishable from the refit) and a high-leverage point (right direction, about 6× too small), and the ratio against $1/(1-h)$ for every point.

**What breaks** (say each in one sentence; write only the final formula, $-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$, at the end):

1. *Degenerate minima:* $H$ is singular. Damping, $(H+\lambda I)^{-1}$, is the exact answer for the loss plus $\frac\lambda2\|w-w^*\|^2$, which pulls the response toward $w^*$. When the per-example gradient $\nabla L_i$ has no component along the flat directions, the $\lambda\to0$ limit is finite (the pseudoinverse answer).
2. *Negative curvature:* replace $H$ by the Gauss–Newton matrix $G=\sum_iJ_i^\top H_{y,i}J_i\succeq0$. It differs from $H$ by a residual term, which need not vanish at a minimum.
3. *Non-convergence:* at a checkpoint $w^s$, $-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$ is the exact derivative of the proximal Bregman response function (PBRF): reweight $i$ while paying for moving the training outputs and the parameters. Bae et al.: "if influence functions are the answer, then what is the question?" The answer is the PBRF, not leave-one-out retraining.

**Speaker notes.**
- Leverage is the theme. The IF is least reliable for the points one most wants to ask about, those the fit could not reproduce without them. This happens already in a convex, converged model.
- If asked about the $a\cdot b$ toy: the damped IF of $a^2$ does not diverge as $\lambda\to0$; it has a finite limit that depends on where on the set of minimizers you stand (DA1 §4 figure).
- The tiny-MLP extension in DA1 §4 shows the proximity gap directly. The damped IF ranks proximal retraining almost perfectly and plain retraining hardly at all.
- **Picked up in:** DA1 §3 (Exercise 3.1, the required exercise) and §4.

---

## 5. How we check attributions, and the lens (4 min)

**Claim.** There are three checks, from least to most demanding: finite differences of the method's own target; leave-one-out; and subset deletion, scored by the linear datamodeling score (LDS). They measure different things. Then one table that previews DA2: many estimators are a filter $F(s)$ applied to the training gradient in a curvature eigenbasis, with different meanings.

**Board.**
$$
\text{LDS}=\text{Spearman}\Big(\{\phi(S_k)\},\ \Big\{\textstyle\sum_{i\in S_k}\mathcal I_i\Big\}\Big),\qquad \frac{\partial w}{\partial\beta_i}=-F(M)\,\nabla L_i
$$

**Show.** The DA1 §5 table (logistic regression).
- The exact IF matches its local derivative to $3\times10^{-7}$ and wins on LOO (Spearman 0.96).
- The damped IF matches *its own* proximal target just as well, is worse on LOO (0.82), and ties on LDS (0.83 each).
- The gradient dot product is worst everywhere.
- The subset slopes are far from 1: the LDS checks ranking, not size.

**Speaker notes.**
- Single-model or distributional targets: for a network, retraining with a new seed moves $\phi$ too. Unrolling targets this run; datamodels and TRAK's protocol average over seeds.
- TracIn is not "crude IF": it allocates the test-loss decrease along the actual trajectory among the examples used at each step. That is a different estimand.
- Filters: $1/s$ for the IF against $1/(s+\lambda)$ for the damped IF. All the others are finite at $s=0$. Same shape, different meanings: a proximal penalty, training time, a localization strength, a prior. If there is time, show DA1 §6's interactive figure (`assets/interactive/filter_explorer.html`).
- **Picked up in:** DA1 §5 (Exercises 5.1–5.2) and §6.

---

## 6. How to use the day (2 min)

- **Morning:** DA1 §1–§6 is one path, about 90 minutes. Do Exercises 3.1, 5.1 and 5.2; the rest of the path is reading, and everything under *Extension* is optional.
- **Afternoon:** DA2 is one ordered tour of six stops: influence functions at scale, unrolling, Bayesian IFs, influence over training, datamodels and TRAK, and open problems. Read every capsule in order (about 30 min), then go deep where you like, by the paper or by the stop's optional walkthrough.
- An optional ~10 min L2 after lunch walks the tour.

---

## Compressed path (about 12 min)

| Segment | Keep | Drop |
|---|---|---|
| 1 (2 min) | the shutdown example, one sentence on cost | the interpretability analogy |
| 2 (2 min) | the croissant | the boulder; Shapley in one sentence |
| 3 (3 min) | the board line and the map table | conventions (DA1 §2 has them) |
| 4 (3 min) | the IF formula and the leverage figure; "each fix changes the question" | the three fixes in detail |
| 5 (1 min) | "three checks; LDS ranks, it doesn't calibrate" | the filter table |
| 6 (1 min) | the morning and afternoon plan | — |
