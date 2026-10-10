#!/usr/bin/env python3
"""Build DA1, "Core: what did this training example do?" (Data Attribution day).

One linear path, six `##` sections (anchors da1-1 ... da1-6), about 90 minutes:
1 the question (counterfactuals, Shapley); 2 targets and conventions (beta, pi, the map
table); 3 the classical influence function (IFT, OLS against exact deletion, leverage);
4 what breaks for neural networks (damping, Gauss-Newton, PBRF; the a*b model);
5 how attributions are evaluated (local, LOO, LDS; a scored logistic-regression exercise);
6 the spectral-filter lens and the DA2 tour. Optional material sits under `### Extension:`
headings. Exercises are bold-labelled paragraphs with anchors, not headings.

Source: Louis Jaburi's Iliad notes (da/source/data-attribution.tex, CC BY 4.0), with the
errata of DA_SPECS_v1.md fixed in our own text. Formula checks: checks/da1_core_checks.py.
Review log: ../reviews/da/da1_core.md.
"""
from nbkit_da import answer, code, code_answer, figure, markdown, reading, run  # noqa: F401

DA2 = "da2_methods.html"

# ---------------------------------------------------------------- readings
KOH = reading(
    "Understanding black-box predictions via influence functions",
    "https://arxiv.org/abs/1703.04730",
    "Koh and Liang (ICML 2017)",
    "The paper that brought influence functions to deep learning. Read §2 for the derivation "
    "(written on the mean loss, so their $\\mathcal I_{\\rm up}$ is $n$ times ours; see DA1 §2) and §4 "
    "for the checks against leave-one-out retraining (Fig. 2) and the damping they add for a "
    "non-convex model; skim §5 for uses.",
    importance=5, length=2)
BAE = reading(
    "If influence functions are the answer, then what is the question?",
    "https://arxiv.org/abs/2209.05364",
    "Bae, Ng, Lo, Ghassemi and Grosse (NeurIPS 2022)",
    "What the influence function computed on a neural network actually answers: the response of a "
    "proximal objective (the PBRF), not leave-one-out retraining. Read §§1–4 for the five gaps between "
    "the two and §5 for the PBRF; Fig. 4 shows which gaps dominate.",
    importance=5, length=3)
MUELLER = reading(
    "Missed causes and ambiguous effects: counterfactuals pose challenges for interpreting neural networks",
    "https://arxiv.org/abs/2407.04690",
    "Mueller (2024)",
    "Written about interpreting forward passes, not training data, but its two failure modes are the ones "
    "below: redundant causes are missed (§4) and counterfactual dependence does not chain (§3).",
    importance=3, length=2)
SHAPLEY = reading(
    "Data Shapley: equitable valuation of data for machine learning",
    "https://arxiv.org/abs/1904.02868v1",
    "Ghorbani and Zou (ICML 2019)",
    "Shapley values for training points, made computable by sampling permutations with early truncation, "
    "or by a one-pass gradient approximation (§3).",
    importance=2, length=2)
BASU = reading(
    "Influence functions in deep learning are fragile",
    "https://arxiv.org/abs/2006.14651",
    "Basu, Pope and Feizi (ICLR 2021)",
    "Compares influence estimates with leave-one-out retraining across depth, width and weight decay: "
    "fairly accurate for shallow networks, often wrong for deep ones, and on ImageNet even the retraining "
    "ground truth is noisy (abstract, §§4–5).",
    importance=2, length=2)
TRAK = reading(
    "TRAK: attributing model behavior at scale",
    "https://arxiv.org/abs/2303.14186",
    "Park, Georgiev, Ilyas, Leclerc and Madry (ICML 2023)",
    "Read §2 for the linear datamodeling score (Definition 2.4), the evaluation used across this "
    "literature, and §3 for TRAK itself: linearize the network, randomly project per-example gradients, "
    "apply the one-step Newton formula from logistic regression, and average over several trained models.",
    importance=4, length=3)
DATAMODELS = reading(
    "Datamodels: predicting predictions from training data",
    "https://arxiv.org/abs/2202.00622",
    "Ilyas, Park, Engstrom, Leclerc and Madry (ICML 2022)",
    "Fits a linear map from the subset indicator $\\mathbf 1_S$ to the output of a model trained on $S$, "
    "using hundreds of thousands of retrainings (300,000 CIFAR-10 models at 50% subsets): the expensive "
    "target that the LDS is built on. Read §§1–2 and §4.1.",
    importance=3, length=3)

EXTRAS = "\n\n**Claude-suggested extras.**\n\n"

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# DA1 — Core: what did this training example do?

**Data Attribution, Iliad Intensive 2026.** Adapted from [Louis Jaburi's Iliad notes](https://iliad-intensive.org/learning/data-attribution/) ([CC BY 4.0](https://iliad-intensive.org/license/)), with changes.

Data attribution asks which training examples made a trained model behave
as it does. This notebook builds the formula most methods start from, the
influence function, checks it where it can be checked exactly, and follows
what changes when it is applied to neural networks. By the end you should be
able to read any of the method capsules in [DA2](da2_methods.html).

**Keystone readings** (best read after working through DA1; §3 and §4 say which parts).
"""
        + KOH + BAE
        + r"""
**The path.** Sections 1–6 are the path, about 90 minutes. On it, do
Exercises 3.1, 5.1 and 5.2; the rest of the path is reading. Headings marked
*Extension*, and anything marked optional, can be skipped.

- [1. The question](#da1-1) (10 min): attribution as causal analysis of a training run, and where counterfactuals fail.
- [2. Targets and conventions](#da1-2) (10 min): data weights $\beta$, the derivative we want, and what each method takes "the trained model" to be.
- [3. The classical influence function](#da1-3) (25 min): $-H^{-1}\nabla L_i$, checked against exact deletion in linear regression.
- [4. What breaks for neural networks](#da1-4) (20 min): damping, Gauss–Newton, and the question the modified formula answers.
- [5. How do we know an attribution is right?](#da1-5) (20 min): local checks, leave-one-out and the linear datamodeling score, on a logistic regression you score yourself.
- [6. A lens for the methods, and where to go next](#da1-6) (5 min): spectral filters, and the DA2 tour.
"""
    ),
    code(
        r"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

np.set_printoptions(precision=4, suppress=True)
"""
    ),
    # ================================================================ 1
    markdown(
        r"""
<a id="da1-1"></a>

## 1. The question

"Which training examples caused this behaviour?" This section treats that as
causal analysis of a training run and shows two ways the simplest
counterfactual answer goes wrong. It sets up why later sections are careful
to say *which* question a method answers.
"""
        + MUELLER + EXTRAS + SHAPLEY
        + r"""
**A training run as a causal system.** Mechanistic interpretability analyses
a forward pass: inputs flow through layers to an output, and one asks which
components caused it. Data attribution does the same for a training run:
data flow through optimizer steps to the final weights, and one asks which
examples caused some behaviour of the final model. Two differences matter.
A forward pass costs milliseconds, so interpretability can test a causal
claim by ablating many times. Retraining costs as much as training, so data
attribution can rarely afford even one ablation, and most of its methods are
ways to *predict* retraining without doing it. And the conceptual
difficulties of causal reasoning come along unchanged.

**Counterfactual attribution.** The simplest test of causation is
counterfactual dependence, the starting point of
[Lewis (1973)](https://doi.org/10.2307/2025310): $A$ caused $B$ if, had $A$
not occurred, $B$ would not have occurred. For training data: example $i$
caused a behaviour if removing $i$ and retraining would change it. This is
the *leave-one-out* (LOO) counterfactual; its graded version asks how much the
behaviour changes. It fails in two standard ways.

**Overdetermination.** A bakery buys flour, butter, yeast and sugar, each from
a standard or a premium supplier. All-premium croissants come out
exceptionally fluffy. Unknown to the bakery, there are two independent mechanisms
for that fluffiness: premium butter makes more steam, premium yeast makes more
gas, and either alone is enough. Downgrading only the butter changes nothing;
nor does downgrading only the yeast. LOO concludes that neither caused the
fluffiness, though both did. The training analogue: a behaviour that many
examples each suffice to teach survives the removal of any one of them.

**Non-transitivity.** A boulder rolls toward a hiker (A); she sees it and
ducks (B); she survives (C). Had the boulder not rolled she would not have
ducked, so B depends on A. Had she not ducked she would have died, so C
depends on B. But C does not depend on A: with or without the boulder, she
survives. Dependence does not chain. (The example
is Ned Hall's, as told by [Hitchcock 2001](https://doi.org/10.2307/2678432).)
The training analogue: an example can teach an intermediate feature that a
capability then relies on, without the example passing an end-to-end LOO test
for the capability.

**Shapley values.** One response to overdetermination is to average an
example's effect over all the training subsets it could join. Treat the $n$
examples as players, and let $v(S)$ be the behaviour of a model trained on
the subset $S\subseteq[n]=\{1,\dots,n\}$. The Shapley value of example $i$ is

$$
\varphi_i=\frac1n\sum_{S\subseteq[n]\setminus\{i\}}\binom{n-1}{|S|}^{-1}\big[v(S\cup\{i\})-v(S)\big],
$$

the marginal effect of adding $i$, averaged over subsets of each size and
then over the sizes. In the bakery, $v(S)=1$ if $S$ (the premium ingredients)
contains butter or yeast and $0$ otherwise. LOO gives every ingredient $0$;
Shapley gives butter and yeast $\tfrac12$ each, and flour and sugar $0$. Three
things to keep in mind:

- Shapley values attribute over finite coalitions. They are not the derivative at the full training set that the rest of this notebook computes.
- Exact evaluation needs $v$ on exponentially many subsets, each a retraining. Practical estimators sample subsets or permutations instead.
- They still split credit between sufficient causes (half each), where one could argue each deserves all of it.

The rest of the day works with something more local and much cheaper than
either LOO or Shapley: the derivative of the behaviour with respect to how
heavily one example is weighted.
"""
    ),
    # ================================================================ 2
    markdown(
        r"""
<a id="da1-2"></a>

## 2. Targets and conventions

Before any formula, fix what is being estimated. This section sets the
notation used all day, DA2 included, and separates three questions that all
get called "the influence of example $i$".

**Setup.** A training set $\{z_i\}_{i=1}^n$ with $z_i=(x_i,y_i)$; model
parameters $w\in\mathbb R^p$; per-example losses $L_i(w)$. Give example $i$ a
weight $\beta_i$ and train on the **summed** weighted loss

$$
L(w;\beta)=\sum_{i=1}^n\beta_i\,L_i(w).
$$

$\beta=\mathbf 1$ is the actual training run. The behaviour to attribute is a
scalar **observable** $\phi(w)$ of the trained parameters: the loss or a logit
on a test input, the average loss on a task, and so on. We also call $\phi$
(or its test input) the *query*.

What "the trained parameters" are is a modelling choice. Write $\pi(\beta)$
for it. The object of study is the composite

$$
\beta\;\xrightarrow{\ \pi\ }\;\pi(\beta)\;\xrightarrow{\ \phi\ }\;\phi\big(\pi(\beta)\big),
$$

and the methods of the day differ mainly in what they take $\pi$ to be (table
below).

**Conventions** (used everywhere on the day):

1. The **influence** of example $i$ on $\phi$ is the upweighting derivative
   $$\mathcal I_i=\frac{\partial\,\phi(\pi(\beta))}{\partial\beta_i}\Big|_{\beta=\mathbf 1}.$$
2. **Deletion** of $i$ means $\beta_i:1\to0$. To first order it changes $\phi$ by $-\mathcal I_i$.
3. **Sign.** If $\phi$ is a test loss, an example that helps (upweighting it lowers the loss) has **negative** influence. Koh and Liang's $\mathcal I_{\rm up,loss}$ has this sign; TracIn, TRAK and SOURCE (methods met in §5 and DA2) report helpful examples as positive.
4. **Koh and Liang's** $\mathcal I_{\rm up}$ is $n$ times ours (Extension below).
5. **Normalization.** For SGD written on a mean loss we show the $1/n$, or the minibatch's $1/b$, explicitly: $w_{t+1}=w_t-\frac{\eta}{b}\sum_{i\in B_t}\beta_i\nabla L_i(w_t)$, with step size $\eta$ and minibatch $B_t$ of size $b$. Curvatures and damping constants (§4) are in summed-loss units: the Hessian of $L(w;\mathbf 1)$ is $n$ times that of the mean loss.

**Three targets.** A derivative and a finite change are different questions.

- **Local derivative:** $\mathcal I_i$, the effect of an infinitesimal reweighting of $i$.
- **Leave-one-out:** $\phi(\pi(\mathbf 1-e_i))-\phi(\pi(\mathbf 1))$, removing $i$ entirely ($e_i$ is the $i$-th unit vector). It equals $-\mathcal I_i$ only to first order.
- **Subset deletion:** $\phi(\pi(\mathbf 1_S))$ for a large random subset $S$ (half the data, say), where $\mathbf 1_S$ keeps the examples in $S$ and deletes the rest. Its first-order prediction is $\phi(\pi(\mathbf 1))-\sum_{j\notin S}\mathcal I_j$.

A method's **target** is the question it is defined to answer: a choice of
$\pi$ and of one of these three. Its **estimator** is how it computes the
answer. §5 shows that the two need separate checks.

**The map of targets.**

| Method | What $\pi(\beta)$ is | DA2 stop |
|---|---|---|
| classical IF (§3) | the stationary point of $L(\cdot\,;\beta)$ on the branch through the trained $w^*$ | — |
| PBRF (§4) | the minimizer of a proximal objective built around the saved checkpoint $w^s$ (which need not be a minimum) | [1](da2_methods.html#da2-stop-1) |
| unrolling | the last iterate $w_T(\beta)$ after $T$ steps of this training run, with the same seed and data order | [2](da2_methods.html#da2-stop-2) |
| Bayesian IF | the local posterior $p_\beta$; $\phi$ is read as $\mathbb E_{p_\beta}\phi$ | [3](da2_methods.html#da2-stop-3) |
| datamodels | a fitted predictor of the outcome of retraining on a subset | [5](da2_methods.html#da2-stop-5) |

The local posterior is
$p_\beta(w)\propto\exp\big(-L(w;\beta)-\tfrac\gamma2\|w-w^*\|^2\big)$, where the
Gaussian factor takes the place of a prior and $\gamma$ localizes the posterior
near the trained $w^*$. In practice it is sampled by stochastic gradient Langevin dynamics (SGLD).

TRAK sits beside the table. It is an analytic estimator (linearize the
network, project per-example gradients, apply an influence-like formula)
whose quality is judged against the datamodel-style subset target. It does
not regress on retraining outcomes.

<a id="da1-2-ext-kl"></a>

### Extension: converting to Koh and Liang's convention

Why is the influence the same for a summed and a mean loss, and how does
Koh and Liang's $\varepsilon$ relate to $\beta$? *Notation as above.*

<a id="da1-ex2-1"></a>

**Exercise 2.1 — Converting conventions.** Let $\pi(\beta)$ be an argmin of
$L(\cdot\,;\beta)$ (assume it is unique and smooth in $\beta$). Koh and Liang
instead upweight by $\varepsilon$ on the mean loss:
$\hat\theta_\varepsilon=\arg\min_w\frac1n\sum_jL_j(w)+\varepsilon L_i(w)$, with
$\mathcal I_{\rm up}=d\hat\theta_\varepsilon/d\varepsilon$ at $\varepsilon=0$.

1. Show that $\mathcal I_i$ is unchanged if $L(w;\beta)$ is replaced by the mean $\frac1n\sum_j\beta_jL_j(w)$.
2. Show that their parameter response is $\mathcal I_{\rm up}=n\,\partial\pi/\partial\beta_i$, and that their removal, $\varepsilon=-1/n$, gives the same first-order prediction of the parameter change as our $\beta_i:1\to0$.
"""
    ),
    answer(
        r"""
1. For any $c>0$, $cL(\cdot\,;\beta)$ and $L(\cdot\,;\beta)$ have the same minimizer. So $\pi(\beta)$ is the same function of $\beta$ for the summed and the mean objective, and so is its derivative. What the normalization does change is the units of curvature and damping (the Hessian scales by $1/n$).
2. $\frac1n\sum_jL_j+\varepsilon L_i=\frac1n\sum_j\beta_jL_j$ with $\beta=\mathbf 1+n\varepsilon e_i$. So $\hat\theta_\varepsilon=\pi(\mathbf 1+n\varepsilon e_i)$ and, by the chain rule, $d\hat\theta_\varepsilon/d\varepsilon=n\,\partial\pi/\partial\beta_i$. Removal at $\varepsilon=-1/n$ is $\beta_i=0$. Their first-order prediction of the parameter change is $-\tfrac1n\mathcal I_{\rm up}=-\partial\pi/\partial\beta_i$, the same as ours. Koh and Liang's $\varepsilon$ simply measures the perturbation in units of $1/n$.
"""
    ),
    # ================================================================ 3
    markdown(
        r"""
<a id="da1-3"></a>

## 3. The classical influence function

How does the trained model move if one example is reweighted? The implicit
function theorem turns this into one linear solve, $-H^{-1}\nabla L_i$. In
linear regression we can compare it with exact deletion; its accuracy is set
by leverage. (The derivation is Koh and Liang §2.)

**Influence on a mean.** Robust statistics asked this question first
([Hampel 1974](https://doi.org/10.1080/01621459.1974.10482962)). Move a
fraction $\varepsilon$ of a distribution's mass to a point $z$. The mean $\mu$
becomes $(1-\varepsilon)\mu+\varepsilon z$, whose derivative at
$\varepsilon=0$ is $z-\mu$. A point's influence on the mean is its distance
from the mean.

**The derivation.** *Notation:* $L(w;\beta)=\sum_j\beta_jL_j(w)$; the trained
parameters $w^*$ satisfy $\nabla_wL(w^*;\mathbf 1)=0$; $H=\nabla_w^2L(w^*;\mathbf 1)$
is the Hessian; $g_i=\nabla L_i(w^*)$ is example $i$'s gradient and
$g_\phi=\nabla\phi(w^*)$ the observable's.

Suppose $H$ is invertible. The implicit function theorem, applied to the
stationarity equation $\nabla_wL(w;\beta)=0$, then gives a unique smooth branch
$w^*(\beta)$ of stationary points through $w^*$, for $\beta$ near $\mathbf 1$.
Differentiate the identity $\nabla_wL(w^*(\beta);\beta)=0$ with respect to
$\beta_i$. The $w$-dependence gives $H\,\partial w^*/\partial\beta_i$, and the
explicit $\beta_i$-dependence gives $\nabla L_i(w^*)=g_i$. So

$$
H\,\frac{\partial w^*}{\partial\beta_i}+g_i=0
\quad\Longrightarrow\quad
\frac{\partial w^*}{\partial\beta_i}\Big|_{\beta=\mathbf 1}=-H^{-1}g_i,
\qquad
\boxed{\;\mathcal I_i=-\,g_\phi^\top H^{-1}g_i\;}
$$

by the chain rule. The theorem gives only a *local branch of stationary
points*. Calling it "the retrained model" also needs $w^*$ to be a strict
minimum ($H$ positive definite) and retraining to land on this branch rather
than in another basin. Both hold automatically for a strictly convex loss.

**Three factors.** Read $-g_\phi^\top H^{-1}g_i$ from right to left:

- $g_i$ is how example $i$ pulls on the parameters (its descent direction is $-g_i$);
- $H^{-1}$ turns that pull into a displacement: flat directions (small eigenvalues of $H$) move a lot, stiff directions little;
- $g_\phi$ projects the displacement onto what the observable cares about.

Deleting $i$ changes $\phi$ by about $-\mathcal I_i=g_\phi^\top H^{-1}g_i$.

<a id="da1-ex3-1"></a>

**Exercise 3.1 — Linear regression: influence versus exact deletion.** Take
inputs $x_i\in\mathbb R^p$, targets $y_i\in\mathbb R$, losses
$L_i(w)=\frac12(y_i-w^\top x_i)^2$, and $A=\sum_ix_ix_i^\top$, assumed
invertible. Write $r_j=y_j-x_j^\top w^*$ for the residual of example $j$ and
$h_j=x_j^\top A^{-1}x_j$ for its **leverage**.

1. Show that $w^*=A^{-1}\sum_ix_iy_i$ and $H=A$.
2. Show that $\partial w^*/\partial\beta_j=A^{-1}x_jr_j$, so the influence function predicts that deleting $j$ moves the parameters by $-A^{-1}x_jr_j$.
3. Show that exact deletion gives
   $$w^*_{(-j)}-w^*=-\frac{A^{-1}x_j\,r_j}{1-h_j},$$
   provided $h_j<1$.
4. Leverages satisfy $0\le h_j\le1$ and $\sum_jh_j=p$ (proof in the solution). What goes wrong when $h_j=1$?
5. Exact deletion is the influence-function prediction times $1/(1-h_j)$. When is the prediction accurate, and when does it fail badly?

<details><summary>Hint</summary>

For 3, write $w^*_{(-j)}=w^*+\delta$ and subtract the full normal equations
$Aw^*=\sum_ix_iy_i$ from the ones without $j$. You should find
$A\delta=x_j\,c$ for a scalar $c$; then find $c$ by taking the inner product
with $A^{-1}x_j$. For 4, look at $(A-x_jx_j^\top)A^{-1}x_j$.
</details>
"""
    ),
    answer(
        r"""
1. $\nabla_wL(w;\mathbf 1)=-\sum_ix_i(y_i-x_i^\top w)=Aw-\sum_ix_iy_i$. Setting it to zero gives $w^*=A^{-1}\sum_ix_iy_i$. Differentiating once more, $H=A$.
2. $g_j=\nabla L_j(w^*)=-x_j(y_j-x_j^\top w^*)=-x_jr_j$, so $\partial w^*/\partial\beta_j=-H^{-1}g_j=A^{-1}x_jr_j$. Upweighting $j$ moves the fit toward $j$: along $A^{-1}x_j$, in proportion to the residual. Deletion is predicted to move it by $-A^{-1}x_jr_j$.
3. Without example $j$ the normal equations are $(A-x_jx_j^\top)\,w_{(-j)}=\sum_ix_iy_i-x_jy_j$. Put $w_{(-j)}=w^*+\delta$ and use $Aw^*=\sum_ix_iy_i$:
   $$A\delta-x_jx_j^\top w^*-x_jx_j^\top\delta=-x_jy_j
   \quad\Longrightarrow\quad A\delta=x_j\big(x_j^\top\delta-r_j\big)=:x_j\,c .$$
   So $\delta=A^{-1}x_j\,c$. Then $x_j^\top\delta=h_jc$, and the definition of $c$ gives $c=h_jc-r_j$, that is, $c=-r_j/(1-h_j)$. Hence
   $$w^*_{(-j)}-w^*=-\frac{A^{-1}x_jr_j}{1-h_j}.$$
4. *The bounds.* Let $X$ stack the $x_i^\top$ as rows. The "hat matrix" $P=XA^{-1}X^\top$ is symmetric and $P^2=XA^{-1}(X^\top X)A^{-1}X^\top=P$, so it is an orthogonal projection. Its diagonal entries satisfy $P_{jj}=\sum_kP_{jk}^2\ge P_{jj}^2$, so $0\le h_j=P_{jj}\le1$. Its trace is its rank: $\sum_jh_j=\operatorname{tr}(A^{-1}X^\top X)=p$. So the average leverage is $p/n$.

   *At $h_j=1$:* $(A-x_jx_j^\top)A^{-1}x_j=x_j(1-h_j)=0$, so the design without $j$ is singular. Some direction in parameter space is pinned down by example $j$ alone, and deleting $j$ leaves it undetermined. The deletion formula requires $h_j<1$.
5. The influence function keeps only the leading term of $1/(1-h_j)=1+h_j+h_j^2+\cdots$.
   - **$h_j\to0$** (a typical point, when $p\ll n$): the correction is $O(h_j)$ and the prediction is essentially exact.
   - **$h_j\to1$** (a point alone in some direction of input space): the true deletion effect is arbitrarily larger than predicted.

   The influence function is least reliable for exactly the points one most wants to ask about: those the fit could not reproduce without them. This happens in a strictly convex, fully converged model; nothing here is about neural networks.
"""
    ),
    markdown(
        r"""
**Leverage as a distance.** With $\hat\Sigma=A/n$, the inputs' uncentered
second-moment matrix, $h_j=\frac1n\,x_j^\top\hat\Sigma^{-1}x_j$: a squared
length of $x_j$ in the metric set by the data. Directions the data sample
rarely have small eigenvalues of $\hat\Sigma$, so inputs pointing that way
have large leverage. If the inputs include a constant feature (an intercept),
$h_j$ is $\frac1n$ plus $\frac1n$ times the squared Mahalanobis distance of
the other features from their mean.

**Code: influence versus exact deletion.** A one-feature regression with an
intercept, $w=(\text{intercept},\text{slope})$. Point A sits in the bulk;
point B sits far out on the $x$-axis and off the trend. Left: removing each
of them, with the influence-function prediction of the new line. Right: for
every point, the ratio of the exact to the predicted change in slope, against
its leverage.
"""
    ),
    code(
        r"""
rng = np.random.default_rng(0)
n = 15
x = np.r_[rng.normal(0, 1, n - 1), 7.0]           # the last point (B) is far out on the x-axis ...
y = 1.0 + 0.6 * x + 0.4 * rng.normal(size=n)
y[-1] -= 3.0                                      # ... and off the trend
Xd = np.c_[np.ones(n), x]                         # rows x_i = (1, x_i)
A = Xd.T @ Xd
w = np.linalg.solve(A, Xd.T @ y)
r = y - Xd @ w                                    # residuals
h = np.einsum('ij,jk,ik->i', Xd, np.linalg.inv(A), Xd)   # leverages
pred = -(np.linalg.solve(A, Xd.T) * r).T          # IF prediction of the deletion change in w, one row per point
exact = np.array([np.linalg.lstsq(np.delete(Xd, j, 0), np.delete(y, j), rcond=None)[0] - w
                  for j in range(n)])             # exact deletion: refit without point j
print('exact = pred / (1 - h) to', f'{np.abs(exact - pred / (1 - h)[:, None]).max():.1e}')
print(f'sum of leverages = {h.sum():.3f} (= p = 2)')

jA, jB = int(np.argmin(np.abs(x[:-1]))), n - 1
print(f'A: h = {h[jA]:.3f};  B: h = {h[jB]:.3f}')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 2)
ax[0].scatter(x, y, s=18, c='k')
ax[0].plot(xs, w[0] + w[1] * xs, 'k-', lw=1.5, label='fit on all points')
for j, name, col in [(jA, 'A', 'tab:blue'), (jB, 'B', 'tab:red')]:
    ax[0].annotate(name, (x[j], y[j]), xytext=(6, -12), textcoords='offset points', color=col, weight='bold')
    we, wp = w + exact[j], w + pred[j]
    ax[0].plot(xs, we[0] + we[1] * xs, '-', color=col, lw=1.2, label=f'{name} removed: exact refit')
    ax[0].plot(xs, wp[0] + wp[1] * xs, ':', color=col, lw=2, label=f'{name} removed: IF prediction')
ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].legend(fontsize=8, loc='upper left')
ax[0].set_title('removing a low- and a high-leverage point')
ratio = exact[:, 1] / pred[:, 1]                 # exact / predicted change in slope
hh = np.linspace(0, 0.9, 200)
ax[1].plot(hh, 1 / (1 - hh), 'k--', lw=0.8, label='$1/(1-h)$')
sc = ax[1].scatter(h, ratio, c=h, cmap='viridis', vmin=0, vmax=1, s=25, zorder=3)
for j, name, col in [(jA, 'A', 'tab:blue'), (jB, 'B', 'tab:red')]:
    ax[1].annotate(name, (h[j], ratio[j]), xytext=(6, -12), textcoords='offset points', color=col, weight='bold')
ax[1].set_xlabel('leverage $h_j$'); ax[1].set_ylabel('exact / IF-predicted change in slope')
ax[1].set_title('every point'); ax[1].legend(fontsize=8)
plt.colorbar(sc, ax=ax[1], label='leverage $h_j$')
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The low-leverage point's prediction is indistinguishable from the refit. The
high-leverage point's is in the right direction but far too small, by the
factor $1/(1-h_B)$: $h_B=0.83$, so about 6 times.

<a id="da1-3-ext-series"></a>

### Extension: the whole response curve

What does the trained model do for every value of one example's weight, not
just its derivative at $\beta_j=1$? In linear regression this has a closed
form, and expanding it shows exactly which terms the influence function drops.

*Notation as in Exercise 3.1:* $A=\sum_ix_ix_i^\top$, residual $r_j$, leverage
$h_j=x_j^\top A^{-1}x_j$. Let $w^*(\beta_j)$ minimize
$\sum_{i\ne j}L_i(w)+\beta_jL_j(w)$, with the other weights held at $1$.

<a id="da1-ex3-2"></a>

**Exercise 3.2 — Sherman–Morrison and the geometric series.**

1. Using the Sherman–Morrison formula $(M+uv^\top)^{-1}=M^{-1}-\dfrac{M^{-1}uv^\top M^{-1}}{1+v^\top M^{-1}u}$, show that
   $$w^*(\beta_j)-w^*=-\frac{(1-\beta_j)\,A^{-1}x_jr_j}{1-(1-\beta_j)\,h_j}.$$
2. Expand in powers of $(1-\beta_j)$:
   $$w^*(\beta_j)-w^*=-A^{-1}x_jr_j\sum_{k\ge1}(1-\beta_j)^k\,h_j^{\,k-1}.$$
   Identify the $k=1$ term with the influence function, and say what suppresses the others.
3. At $\beta_j=0$, which suppression is left?
"""
    ),
    answer(
        r"""
1. The normal equations are $\big(A-(1-\beta_j)x_jx_j^\top\big)w=\sum_ix_iy_i-(1-\beta_j)x_jy_j$. Sherman–Morrison with $u=-(1-\beta_j)x_j$, $v=x_j$ gives, with $q_j=A^{-1}x_j$,
   $$\big(A-(1-\beta_j)x_jx_j^\top\big)^{-1}=A^{-1}+\frac{(1-\beta_j)\,q_jq_j^\top}{1-(1-\beta_j)h_j}.$$
   Apply this to the right-hand side, using $A^{-1}\sum_ix_iy_i=w^*$ and $q_j^\top\sum_ix_iy_i=x_j^\top w^*$. Write $t=1-\beta_j$ and $d_j=1-t\,h_j$. Then
   $$w^*(\beta_j)=w^*-t\,q_jy_j+\frac{t\,q_j\big(x_j^\top w^*-t\,h_jy_j\big)}{d_j}
   =w^*+\frac{t\,q_j}{d_j}\Big[-y_j(1-t\,h_j)+x_j^\top w^*-t\,h_jy_j\Big]
   =w^*-\frac{t\,q_jr_j}{d_j}.$$
   The response is a rational function of $\beta_j$ because the matrix being inverted is linear in $\beta_j$.
2. For $|(1-\beta_j)h_j|<1$, $\dfrac{1}{1-(1-\beta_j)h_j}=\sum_{k\ge0}(1-\beta_j)^kh_j^k$. Multiplying by $(1-\beta_j)$ gives the series.
   - The $k=1$ term is $-(1-\beta_j)A^{-1}x_jr_j=(\beta_j-1)\,\partial w^*/\partial\beta_j$: the influence-function prediction, as it must be, since the IF *is* the linear coefficient.
   - The $k$-th term carries $(1-\beta_j)^k h_j^{k-1}$: it is suppressed both by the size of the perturbation and by the leverage.
3. At $\beta_j=0$ the perturbation factor is $1$, and only the powers of $h_j$ remain. The series becomes $1+h_j+h_j^2+\cdots=1/(1-h_j)$, recovering Exercise 3.1. The gap between the influence function and LOO is the sum of all the higher-order Taylor terms, which linear regression happens to sum in closed form. For other models the higher-order terms are still there, with no closed form.
"""
    ),
    markdown(
        r"""
<a id="da1-3-ext-ift"></a>

### Extension: what the implicit function theorem needs

Which assumptions does the derivation of $-H^{-1}g_i$ actually use, and which
are only needed to call the result "retraining"? *Notation:*
$L(w;\beta)=\sum_j\beta_jL_j(w)$, $w^*$ a stationary point of $L(\cdot\,;\mathbf 1)$,
$H$ its Hessian there, $g_i=\nabla L_i(w^*)$.

The implicit function theorem: if $F(w,\beta)$ is $C^1$, $F(w^*,\mathbf 1)=0$
and $\partial F/\partial w$ is invertible at $(w^*,\mathbf 1)$, then near
$\beta=\mathbf 1$ there is a unique $C^1$ map $\beta\mapsto w(\beta)$ near $w^*$
with $F(w(\beta),\beta)=0$.

<a id="da1-ex3-3"></a>

**Exercise 3.3 — Local branch versus argmin.**

1. Apply the theorem to $F(w,\beta)=\nabla_wL(w;\beta)$. Which hypothesis is the invertibility of $H$?
2. Is "$w^*$ is the unique global minimum" needed for the derivative $-H^{-1}g_i$ to exist? What is it needed for?
3. Take one parameter $w\in\mathbb R$ and three examples: $L_1(w)=(w^2-1)^2$, $L_2(w)=0.10\,w$ and $L_3(w)=-0.11\,w$. Where is the global minimum at $\beta=\mathbf 1$? What happens to the argmin if $\beta_3$ drops to $0.8$? What does the influence function at the original minimum predict?
"""
    ),
    answer(
        r"""
1. $\partial F/\partial w=\nabla_w^2L(w;\beta)$, which at $(w^*,\mathbf 1)$ is $H$. Its invertibility is exactly the theorem's main hypothesis. ($F$ is $C^1$ if each $L_j$ is $C^2$.)
2. No. The theorem's uniqueness is *local*: it gives the unique branch of stationary points near $w^*$, and $-H^{-1}g_i$ is the derivative of that branch. Global minimality is not used. It is what you need to identify the branch with "the argmin of the reweighted loss" (together with $H\succ0$, so the branch consists of minima). Without it, the argmin can jump to another basin as $\beta$ changes, and is then not even continuous in $\beta$.
3. $L(w;\mathbf 1)=(w^2-1)^2-0.01\,w$ has two wells, near $w=\pm1$; the tilt $-0.01\,w$ makes the one near $+1$ slightly lower, so it is the global minimum. With $\beta_3=0.8$ the tilt is $0.10\,w-0.088\,w=+0.012\,w$, and the global minimum jumps to the well near $-1$. The influence function at $w^*\approx1$ (where $H\approx8$) predicts a small shift within the right-hand well, $\Delta w\approx-H^{-1}g_3\,\Delta\beta_3\approx-(-0.11/8)(-0.2)\approx-0.003$, which is correct for the local branch. Retraining from a random initialization may land in either well; retraining from $w^*$ stays on the branch.
"""
    ),
    # ================================================================ 4
    markdown(
        r"""
<a id="da1-4"></a>

## 4. What breaks for neural networks, and the standard fixes

§3 assumed the trained model sits at an isolated, strict minimum. A neural
network checkpoint usually does not. Three standard modifications turn
$-g_\phi^\top H^{-1}g_i$ into the formula used in practice,
$-g_\phi^\top(G+\lambda I)^{-1}g_i$, and each one changes the question being
answered. Bae et al. §§1–4 (keystone, above) is the full argument; read it
after the session if you want it.
"""
        + EXTRAS + BASU
        + r"""
*Notation:* $L(w;\beta)=\sum_j\beta_jL_j(w)$; a checkpoint $w^s$, which need not be
stationary (when it is, we write $w^*$); at $w^s$, the Hessian
$H=\nabla^2_wL(w^s;\mathbf 1)$, the gradients $g_i=\nabla L_i(w^s)$ and
$g_\phi=\nabla\phi(w^s)$.

**1. Degenerate minima: damping.** Networks have continuous families of
minimizers, from rescaling symmetries, unused units, or simply more
parameters than constraints. Along such a family the loss is flat, so $H$ has
zero eigenvalues and $H^{-1}$ does not exist. The standard repair replaces
$H^{-1}$ by $(H+\lambda I)^{-1}$ with $\lambda>0$. At a stationary point $w^*$
this is not just a numerical patch: $-g_\phi^\top(H+\lambda I)^{-1}g_i$ is the
exact influence for the *proximal* objective

$$
L(w;\beta)+\tfrac{\lambda}{2}\|w-w^*\|^2 ,
$$

whose Hessian at $w^*$ is $H+\lambda I$. The proximal term pulls the
response toward $w^*$, and at finite $\lambda$ it shrinks the response. If
$g_i$ has no component along the flat directions (it lies in the range of
$H$), the response has a finite limit as $\lambda\to0$, the pseudoinverse
answer $-H^+g_i$. If it does have such a component, the response along it
grows like $1/\lambda$. Degenerate minima
are a main subject of the SLT day, which also uses the $a\cdot b$ model below.

**2. Negative curvature: Gauss–Newton.** Away from a minimum, $H$ can have
negative eigenvalues. Then $H+\lambda I$ is singular for some $\lambda>0$, and
nearby values give huge responses of either sign. The standard fix is the
Gauss–Newton matrix. Write $f_i(w)\in\mathbb R^C$ for the network's outputs on
example $i$, $\ell_i(f_i)$ for the loss as a function of them,
$J_i=\partial f_i/\partial w$ ($C\times p$) and $H_{y,i}=\nabla_f^2\ell_i$
($C\times C$). Differentiating $L(w;\mathbf 1)=\sum_i\ell_i(f_i(w))$ twice,

$$
H=\underbrace{\sum_iJ_i^\top H_{y,i}\,J_i}_{G}\;+\;\sum_i\sum_{c=1}^C\frac{\partial\ell_i}{\partial f_{ic}}\,\nabla_w^2f_{ic}.
$$

$G$ is positive semidefinite whenever each $\ell_i$ is convex in the outputs
(squared error, softmax cross-entropy), so $G+\lambda I$ is positive definite
for every $\lambda>0$. $G$ drops the second term. That term vanishes when the
output gradients $\partial\ell_i/\partial f_i$ do (squared error at
interpolation), or when the model is linear in its parameters. It need not
vanish at a minimum: with one parameter, $f(w)=w^2$, target $-1$ and squared
error, $L=\frac12(w^2+1)^2$ has a strict minimum at $w=0$ where $H=2$ but
$G=0$. So even at a converged minimum, the Gauss–Newton influence need not
equal the Hessian influence. [TD §6.1](../td/td2_6_curvature.html#td26-1) has more on the
decomposition.

**3. Non-convergence: the PBRF.** Checkpoints are rarely stationary, so the
implicit-function derivation has nothing to start from. Bae et al. show that
the formula people compute at a checkpoint $w^s$ is nonetheless the exact
answer to a different question. The **proximal Bregman response function**
(PBRF) is the minimizer of

$$
Q(w;\beta)=\sum_{j=1}^nD_j\big(f_j(w),f_j(w^s)\big)+\sum_{j=1}^n(\beta_j-1)\,L_j(w)+\frac{\lambda}{2}\|w-w^s\|^2 .
$$

Here $D_j(f,f^s)=\ell_j(f)-\ell_j(f^s)-\nabla\ell_j(f^s)^\top(f-f^s)$ is the
Bregman divergence of the output loss: zero at $f=f^s$ and growing as the
outputs move away (for squared error, $\frac12\|f-f^s\|^2$). At
$\beta=\mathbf 1$ only the first and last terms remain, and both are minimized
at $w^s$. So the checkpoint is the optimum by construction, whether or not
training converged. As a function of $f$, $D_j$ has zero gradient and Hessian
$H_{y,j}$ at $f^s$, so its Hessian in $w$ at $w^s$ is $J_j^\top H_{y,j}J_j$.
The Hessian of $Q$ at $w^s$ is therefore $G+\lambda I$, and the implicit
function theorem gives

$$
\frac{\partial\,\phi(\text{PBRF}(\beta))}{\partial\beta_i}\Big|_{\beta=\mathbf 1}=-\,g_\phi^\top(G+\lambda I)^{-1}g_i .
$$

This is the **damped Gauss–Newton influence function**, the formula DA2 stop 1
computes at scale. In words: how would $\phi$ change if $i$ were reweighted,
while moving the outputs on the training set and moving the parameters were
both penalized?
In practice one solves $(G+\lambda I)u=g_\phi$ once per query and takes
$-u^\top g_i$ for every training example; DA2 stop 1 is about doing that solve
at the scale of a language model.

This is not a repaired estimate of LOO retraining. Bae et al. split the gap
between the influence function and LOO retraining into five parts:

- *warm-start gap:* the influence function describes retraining started from $w^s$, not from a fresh initialization;
- *proximity gap:* moving the parameters is penalized;
- *non-convergence gap:* $w^s$ is not a minimum, so the Bregman term stands in for the loss;
- *linearization error:* the first-order step;
- *solver error:* the approximate linear solve.

The first three are changes of question, and for neural networks they
dominate (their Fig. 4). The influence function answers the PBRF question
accurately; that question just differs from retraining.

**Code: parameterization dependence of the damped IF.** Does the damped
IF's answer depend on which of many equivalent minimizers training reached?
For an observable that changes along the valley of minimizers, yes, even as
$\lambda\to0$.

The $a\cdot b$ model has parameters $w=(a,b)$ and predicts the product $ab$.
Its losses are $L_i=\frac12(z_i-ab)^2$ for numbers $z_1,\dots,z_n$ with mean
$\bar z\ne0$, so the minimizers form the hyperbola $ab=\bar z$ (the valley).
Take two observables: $\phi=ab$, which is what the model computes and is
constant along the valley, and $\phi=a^2$, which changes along it. At a
minimizer $(a^*,b^*)$, write $v=(b^*,a^*)$, the direction across the valley.
Then $H=n\,vv^\top$ and $g_i=-(z_i-\bar z)\,v$ (Exercise 4.1), so
$(H+\lambda I)^{-1}g_i=g_i/(n\|v\|^2+\lambda)$. Writing
$\mathcal I_\lambda(z_i,\phi)=-g_\phi^\top(H+\lambda I)^{-1}g_i$ for the damped
influence of $z_i$ on $\phi$,

$$
\mathcal I_\lambda(z_i,ab)=\frac{(z_i-\bar z)\,\|v\|^2}{n\|v\|^2+\lambda},
\qquad
\mathcal I_\lambda(z_i,a^2)=\frac{2a^*b^*\,(z_i-\bar z)}{n\|v\|^2+\lambda}.
$$

The cell plots both against $\lambda$ for several minimizers. The open
circles are the derivative of retraining by gradient flow started at that
minimizer, computed by finite differences.
"""
    ),
    code(
        r"""
rng = np.random.default_rng(1)
z = 2.0 + rng.normal(size=10)
n, zbar = len(z), z.mean()
i = int(np.argmax(z))                             # attribute the largest z_i
lams = np.logspace(-3, 3, 200)


def warm_gf_endpoint(a0, b0, beta_i):
    # gradient flow on sum_j beta_j (z_j - ab)^2 / 2, started at (a0, b0), run to convergence
    wts = np.ones(n); wts[i] = beta_i
    def rhs(t, wv):
        g = -np.sum(wts * (z - wv[0] * wv[1]))     # dL/du at u = ab
        return [-g * wv[1], -g * wv[0]]
    return solve_ivp(rhs, (0, 5), [a0, b0], rtol=1e-11, atol=1e-13).y[:, -1]


fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
for k, a0 in enumerate([0.6, 1.0, np.sqrt(zbar), 2.5]):
    b0 = zbar / a0
    v = np.array([b0, a0])
    H = n * np.outer(v, v)                        # Hessian at the minimizer: rank one
    g_i = -(z[i] - zbar) * v
    col = f'C{k}'
    lab = f'$(a^*,b^*)=({a0:.2f},{b0:.2f})$'
    for axk, gphi, phi in [(ax[0], np.array([2 * a0, 0.0]), lambda wv: wv[0] ** 2),
                           (ax[1], v, lambda wv: wv[0] * wv[1])]:
        I_lam = [-gphi @ np.linalg.solve(H + lam * np.eye(2), g_i) for lam in lams]
        axk.semilogx(lams, I_lam, color=col, label=lab)
        eps = 1e-4                                # derivative of warm-started retraining, central difference
        fd = (phi(warm_gf_endpoint(a0, b0, 1 + eps)) - phi(warm_gf_endpoint(a0, b0, 1 - eps))) / (2 * eps)
        axk.plot(lams[0], fd, 'o', mfc='none', color=col, ms=9)
ax[0].set_title(r'$\phi=a^2$: finite limit, different for each minimizer')
ax[1].set_title(r'$\phi=ab$: one limit, $(z_i-\bar z)/n$')
for axk in ax:
    axk.set_xlabel(r'damping $\lambda$'); axk.set_ylabel(r'damped IF $\mathcal{I}_\lambda(z_i,\phi)$')
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
nv2 = [n * (a0**2 + (zbar / a0)**2) for a0 in [0.6, 1.0, np.sqrt(zbar), 2.5]]
print(f'n|v|^2 for the four minimizers: {np.round(nv2, 1)}; each curve bends where lambda reaches it')
"""
    ),
    markdown(
        r"""
- **Why finite.** $H$ has a zero eigenvalue along the valley, so $(H+\lambda I)^{-1}$ is $1/\lambda$ there. But $g_i$ points across the valley (along $v$), so that $1/\lambda$ never multiplies anything.
- **Why minimizer-dependent.** $a^2$ changes along the hyperbola; $ab$ does not. For $\phi=a^2$ the plateau is the derivative of the response to the *nearest* new minimizer. Here that equals warm-started gradient-flow retraining (the circles), which conserves $a^2-b^2$ and moves along $v$. Retraining from a random initialization could land anywhere on the new hyperbola. So for $a^2$ there is no retraining-independent answer, and damping is what picks one.

<a id="da1-4-ext-ab"></a>

### Extension: the $a\cdot b$ model by hand

Where does the plot's formula come from, and how does the pseudoinverse
compare with exact deletion? *Setup as in the code:* parameters $w=(a,b)$,
losses $L_i(w)=\frac12(z_i-ab)^2$ summed over $i=1,\dots,n$, $\bar z=\frac1n\sum_iz_i\ne0$,
a minimizer $(a^*,b^*)$ with $a^*b^*=\bar z$, and $v=(b^*,a^*)$.

<a id="da1-ex4-1"></a>

**Exercise 4.1 — Influence in a degenerate model.**

1. Show that the Hessian at $(a^*,b^*)$ is $H=n\,vv^\top$. Show that its null space is the tangent to the hyperbola of minimizers.
2. For $\phi=ab$, compute the influence with the pseudoinverse $H^+$ (invert only the nonzero eigenvalue), and compare it with the exact LOO change in $\phi$.
3. Compute the damped influence $-g_\phi^\top(H+\lambda I)^{-1}g_i$ for $\phi=ab$ and $\phi=a^2$. Describe the dependence on $\lambda$, on the minimizer, and the $\lambda\to0$ limit.
4. Show that gradient flow on any loss of the form $\sum_jc_j\ell(z_j,ab)$ conserves $a^2-b^2$. Use this to show that warm-started gradient-flow retraining has the same $\lambda\to0$ derivative of $a^2$ as in 3.
"""
    ),
    answer(
        r"""
1. $\partial^2L_i/\partial a^2=b^2$, $\partial^2L_i/\partial b^2=a^2$, $\partial^2L_i/\partial a\partial b=2ab-z_i$. Summing at a minimizer, the off-diagonal is $\sum_i(2a^*b^*-z_i)=n(2\bar z-\bar z)=n\,a^*b^*$, so
   $$H=n\begin{pmatrix}b^{*2}&a^*b^*\\a^*b^*&a^{*2}\end{pmatrix}=n\,vv^\top .$$
   It has rank one, with eigenvalue $n\|v\|^2$ on $v$ and $0$ on $v_\perp=(a^*,-b^*)$. The hyperbola $ab=\bar z$ has normal $\nabla(ab)=(b,a)=v$, so its tangent is $v_\perp$: the null space is the direction along the minimizers, generated by the rescaling $(a,b)\mapsto(\alpha a,b/\alpha)$.
2. $g_i=\nabla L_i(w^*)=-(z_i-\bar z)\,v$ and $g_\phi=\nabla(ab)=v$. With $H^+=vv^\top/(n\|v\|^4)$,
   $$\mathcal I=-v^\top H^+\big(-(z_i-\bar z)v\big)=\frac{z_i-\bar z}{n},$$
   independent of the minimizer. Deleting $z_i$ changes the mean to $(n\bar z-z_i)/(n-1)$, and every new minimizer has $ab$ equal to it, so the exact change is $-(z_i-\bar z)/(n-1)$. The ratio of exact to predicted is $n/(n-1)=1/(1-h)$ with $h=1/n$: each point has leverage $1/n$ in this one-effective-parameter model, as in Exercise 3.1.
3. $(H+\lambda I)^{-1}v=v/(n\|v\|^2+\lambda)$, because $v$ is an eigenvector. So for any $\phi$,
   $$\mathcal I_\lambda(z_i,\phi)=\frac{(z_i-\bar z)\,g_\phi^\top v}{n\|v\|^2+\lambda}.$$
   - For $\phi=ab$, $g_\phi^\top v=\|v\|^2$. The result depends on the minimizer for $\lambda>0$, and tends to $(z_i-\bar z)/n$ (the pseudoinverse answer) as $\lambda\to0$.
   - For $\phi=a^2$, $g_\phi=(2a^*,0)$ and $g_\phi^\top v=2a^*b^*$. The limit is $2\bar z(z_i-\bar z)/\big(n(a^{*2}+b^{*2})\big)$: finite, but generally minimizer-dependent: for $\bar z>0$, compare the balanced minimizer $a^*=b^*=\sqrt{\bar z}$ with $(\bar z,1)$.
   - In both cases the dependence on $\lambda$ is a plateau for $\lambda\ll n\|v\|^2$, then decay as $1/\lambda$. Nothing diverges as $\lambda\to0$, because $g_i$ has no component along the null direction.
4. For a loss depending on $u=ab$, $\dot a=-\partial_u\mathcal L\,b$ and $\dot b=-\partial_u\mathcal L\,a$, so $\frac{d}{dt}(a^2-b^2)=2a\dot a-2b\dot b=0$. Warm-started retraining therefore ends at the point of the new hyperbola $ab=\bar z'$ with the same $a^2-b^2=c$. There $a^2=\big(c+\sqrt{c^2+4u^2}\big)/2$, so
   $$\frac{d(a^2)}{du}=\frac{2u}{\sqrt{c^2+4u^2}}=\frac{2ab}{a^2+b^2}.$$
   With $du/d\beta_i=(z_i-\bar z)/n$ (from 2), this is $2a^*b^*(z_i-\bar z)/\big(n\|v\|^2\big)$, the $\lambda\to0$ limit of 3. Geometrically, gradient flow moves along $\nabla u=v$, which to first order is also the direction to the nearest point of the new hyperbola.
"""
    ),
    markdown(
        r"""
<a id="da1-4-ext-mlp"></a>

### Extension: a proximity gap on a tiny MLP

How different are the proximal question and plain retraining for a real
(tiny) network? *Setup:* a one-hidden-layer tanh network with 8 units (25
parameters), fit to $n=20$ noisy points of $\sin 2x$. It is trained to a
minimum $w^*$ by Newton's method, with a tiny ridge term so that the minimum
is isolated. Every retrained point is checked to be a minimum, not a saddle. The observable $\phi$ is the squared error at a test input. For
each training point $i$, the cell computes three deletion responses:

- the damped-IF prediction $-\mathcal I_i=g_\phi^\top(H+\lambda I)^{-1}g_i$, with $\lambda$ at 1% of the top Hessian eigenvalue;
- **proximal retraining:** minimize $L(w;\mathbf 1-e_i)+\frac\lambda2\|w-w^*\|^2$, the damped IF's own target (made finite);
- **plain retraining:** minimize $L(w;\mathbf 1-e_i)$, started from $w^*$.

All three are evaluated at the same deletion, so the difference between the
last two is the proximity gap. This needs torch; it takes about 25 s.
"""
    ),
    code(
        r"""
import torch
from torch.func import grad, hessian, jacrev, vmap

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
t0 = time.time()
n_mlp, width = 20, 8
Xt = torch.linspace(-2, 2, n_mlp)[:, None]
Yt = torch.sin(2 * Xt[:, 0]) + 0.2 * torch.randn(n_mlp)
xq, yq = torch.tensor([[0.5]]), torch.sin(torch.tensor([1.0]))
shapes = [(width, 1), (width,), (1, width), (1,)]
sizes = [int(np.prod(s)) for s in shapes]
P = sum(sizes)


def net(th, x):
    W1, b1, W2, b2 = [q.reshape(s) for q, s in zip(torch.split(th, sizes), shapes)]
    return (torch.tanh(x @ W1.T + b1) @ W2.T + b2).squeeze(-1)


ridge = 1e-3                                      # makes the minimum isolated


def objective(th, beta, lam, center):             # sum_i beta_i L_i + ridge + proximal term
    return ((beta * 0.5 * (net(th, Xt) - Yt) ** 2).sum() + 0.5 * ridge * (th ** 2).sum()
            + 0.5 * lam * ((th - center) ** 2).sum())


v_obj = vmap(objective, in_dims=(0, 0, None, None))
v_grad = vmap(grad(objective), in_dims=(0, 0, None, None))
v_hess = vmap(hessian(objective), in_dims=(0, 0, None, None))


def newton(TH, B, lam, center, iters=100):
    # Newton with |eigenvalues| (floored at mu), one problem per row; a step is accepted only if
    # it lowers the objective. Rows that end at a saddle are pushed along the negative-curvature
    # direction and restarted, so every returned point is a minimum.
    for attempt in range(4):
        mu = torch.full((len(TH),), 1e-3)
        val = v_obj(TH, B, lam, center)
        for _ in range(iters):
            g = v_grad(TH, B, lam, center)
            if g.norm(dim=1).max() < 1e-7:
                break
            ev, V = torch.linalg.eigh(v_hess(TH, B, lam, center))
            d = torch.maximum(ev.abs(), mu[:, None])
            step = (V @ ((V.transpose(1, 2) @ g[..., None])[..., 0] / d)[..., None])[..., 0]
            new = TH - step
            nv = v_obj(new, B, lam, center)
            ok = nv < val
            TH, val = torch.where(ok[:, None], new, TH), torch.where(ok, nv, val)
            mu = torch.where(ok, (mu / 3).clamp(min=1e-12), mu * 10)
        ev, V = torch.linalg.eigh(v_hess(TH, B, lam, center))
        saddle = ev[:, 0] < 0
        if not saddle.any():
            break
        kick = 0.05 * V[:, :, 0]                  # escape along the most negative direction
        better = v_obj(TH + kick, B, lam, center) < v_obj(TH - kick, B, lam, center)
        TH = torch.where(saddle[:, None], TH + torch.where(better[:, None], kick, -kick), TH)
    g = v_grad(TH, B, lam, center)
    return TH, g.norm(dim=1).max().item(), ev[:, 0].min().item()


phi_t = lambda th: 0.5 * ((net(th, xq) - yq) ** 2).sum()
zero = torch.zeros(P)
w_star, gnorm, _ = newton(0.8 * torch.randn(1, P), torch.ones(1, n_mlp), 0.0, zero, 300)
w_star = w_star[0]
Hm = hessian(objective)(w_star, torch.ones(n_mlp), 0.0, zero)
ev = torch.linalg.eigvalsh(Hm)
lam = 0.01 * ev[-1].item()
print(f'trained: |grad| = {gnorm:.1e}; Hessian eigenvalues {ev[0]:.1e} ... {ev[-1]:.1f}; lambda = {lam:.2f}')
g_tr = jacrev(lambda t: 0.5 * (net(t, Xt) - Yt) ** 2)(w_star)     # rows g_i
g_q = grad(phi_t)(w_star)
pred = (g_tr @ torch.linalg.solve(Hm + lam * torch.eye(P), g_q)).numpy()   # -I_i, predicted deletion change

B = 1 - torch.eye(n_mlp)                              # row i deletes example i
W_prox, gp, ep = newton(w_star.repeat(n_mlp, 1), B, lam, w_star)
W_plain, gu, eu = newton(w_star.repeat(n_mlp, 1), B, 0.0, w_star, 300)
prox = (vmap(phi_t)(W_prox) - phi_t(w_star)).numpy()
plain = (vmap(phi_t)(W_plain) - phi_t(w_star)).numpy()
print(f'retrained minima: |grad| <= {max(gp, gu):.1e}, smallest Hessian eigenvalue {min(ep, eu):.1e} (> 0: minima, not saddles); {time.time() - t0:.1f} s')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for axk, target, name in [(ax[0], prox, 'proximal retraining'), (ax[1], plain, 'plain retraining')]:
    rho = spearmanr(pred, target)[0]
    axk.scatter(pred, target, s=20)
    axk.axhline(0, color='k', lw=0.5); axk.axvline(0, color='k', lw=0.5)
    axk.set_xlabel(r'damped-IF prediction $-\mathcal{I}_i$'); axk.set_ylabel(r'actual change in $\phi$')
    axk.set_title(f'{name}: Spearman {rho:.2f}, size ratio {np.linalg.norm(target) / np.linalg.norm(pred):.1f}')
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The damped IF ranks the proximal responses almost perfectly. Its magnitudes
are off, because deletion is a finite step. It says little about plain
retraining. The smallest Hessian eigenvalues are tiny, so without the
proximal term deleting a point moves the parameters far along nearly flat
directions. That movement is the proximity gap. Which response you want is
a choice of question, not a matter of numerical accuracy.
"""
    ),
    # ================================================================ 5
    markdown(
        r"""
<a id="da1-5"></a>

## 5. How do we know an attribution is right?

Every method on the day outputs a number per training example. This section
is about testing those numbers, and about why "computes its own derivative
exactly" and "predicts what retraining does" are different claims. You will
score three methods on a logistic regression where every check can be run
exactly.
"""
        + TRAK + EXTRAS + DATAMODELS
        + r"""
**Target versus estimator.** A method has a *target*, the question it is
defined to answer: a derivative of some $\pi$, a deletion, or a subset outcome
(§2). It has an *estimator*: the computation. Errors come from either. The
damped IF can compute the PBRF derivative (§4) exactly and still predict LOO
retraining poorly. That is a mismatch of target, not a numerical error. So
every check should say which question it tests.

**Three checks, from least to most demanding.** *Notation:* $\mathcal I_i$ is
a method's score for example $i$ on observable $\phi$, in the sign convention
of §2.

1. **Local finite differences of the method's own target.** Reweight one example by $\pm\varepsilon$, recompute $\pi$, and difference $\phi$. This tests the estimator only.
2. **Leave-one-out.** Delete each example, retrain, and compare the change in $\phi$ with $-\mathcal I_i$. This tests one finite deletion.
3. **Subset deletion and the linear datamodeling score (LDS).** Draw $m$ random subsets $S_1,\dots,S_m$, each a fraction $\alpha$ of the data ($\alpha=\frac12$ is standard). Retrain on each and record $\phi(S_k)$. Predict each outcome by the summed scores $\hat\phi(S_k)=\sum_{i\in S_k}\mathcal I_i$: to first order, $\phi(S)-\phi(\text{all})=-\sum_{j\notin S}\mathcal I_j$, which is $\hat\phi(S)$ up to a constant. The LDS is the Spearman rank correlation between $\{\phi(S_k)\}$ and $\{\hat\phi(S_k)\}$, averaged over observables (test points). TRAK's protocol averages each $\phi(S_k)$ over several training seeds.

Two properties of the LDS matter when reading results:

- It tests **ranking, not calibration.** Multiplying every score by 10, or applying any increasing function to the predictions, leaves it unchanged.
- Deleting half the data is far from $\beta=\mathbf 1$. The LDS rewards estimators that extrapolate well, and these need not be the ones that differentiate exactly.

**One model or the distribution?** For a nonconvex network, retraining with a
different seed changes $\phi$ even on identical data. A target can be
*single-model* (the same seed and data order: this run) or *distributional*
(averaged over seeds). Unrolling targets the first. Datamodels and TRAK's LDS
protocol target the second. The two can disagree: the effect of deleting one
example can vary from seed to seed by more than its average, and then a
same-seed answer says little about the distributional one.

**Gradient-similarity methods have their own estimand.** The gradient dot
product $-g_\phi^\top g_i$ is the influence function with $H$ replaced by the
identity. [TracIn](https://arxiv.org/abs/2002.08484) uses the training
trajectory instead. Write $\ell(w,z)$ for the loss of example $z$ at
parameters $w$, $z_q$ for the test example, and $w_t$, $\eta_t$ and $B_t$ for
the iterate, step size and minibatch (of size $b$) at step $t$:

$$
\mathrm{TracIn}(z_i,z_q)=\sum_{t:\,i\in B_t}\frac{\eta_t}{b}\,\nabla\ell(w_t,z_q)\cdot\nabla\ell(w_t,z_i).
$$

This is the first-order decrease of the test loss caused by $z_i$'s share of
each step it was in; summed over all examples, it adds up (to first order in
each step) to the decrease $\ell(w_0,z_q)-\ell(w_T,z_q)$ along the actual run.
That is a different question from what would change if $z_i$ were missing.
(TracIn calls helpful examples positive; its practical version, TracInCP, is
in DA2 stop 2.)

<a id="da1-ex5-1"></a>

**Exercise 5.1 — Score three methods.** The setup cell below fits an
$\ell_2$-regularized logistic regression:

- $n=80$ examples in $d=10$ dimensions, with feature scales spanning 1.5 decades, so the Hessian is far from a multiple of the identity;
- the objective $L(w;\beta)=\sum_i\beta_i\ell_i(w)+\frac\kappa2\|w\|^2$ with $\kappa=0.01$. The ridge is *not* rescaled with $\beta$, so a model trained on half the data is relatively more regularized;
- 10 held-out test points; the observables $\phi_q$ are their test losses;
- the solver is Newton's method, exact to machine precision, so every check is deterministic and needs no seeds.

It provides $H$ (the Hessian of $L(w;\mathbf 1)$ at $w^*$, ridge included), the
rows $g_i$ in `G`, the rows $g_{\phi_q}$ in `Gq`, and a damping $\lambda$ equal
to the median eigenvalue of $H$. Complete `attribution_scores`, returning
$\mathcal I[q,i]$ for:

- **exact IF:** $-g_{\phi_q}^\top H^{-1}g_i$;
- **damped IF:** $-g_{\phi_q}^\top(H+\lambda I)^{-1}g_i$;
- **gradient dot product:** $-g_{\phi_q}^\top g_i$.

The evaluation cell then runs checks 1–3: central differences with
$\varepsilon=10^{-3}$, 80 LOO refits, and 100 random subsets of exactly 40
examples. Self-check: if your exact IF is right, its check-1 relative error is
about $10^{-7}$.
"""
    ),
    code(
        r"""
rng = np.random.default_rng(1)
n, d, kappa, nq = 80, 10, 0.01, 10
scale = np.logspace(0, -1.5, d)                   # feature scales
X = rng.normal(size=(n, d)) * scale
w_true = 4 * rng.normal(size=d) / np.sqrt(scale)
y = (rng.random(n) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
Xq = rng.normal(size=(nq, d)) * scale             # held-out test points
yq = (rng.random(nq) < 1 / (1 + np.exp(-Xq @ w_true))).astype(float)


def fit(beta, w0=None, lam=0.0, center=None):
    # Newton's method for sum_i beta_i ell_i(w) + kappa/2 |w|^2  [+ lam/2 |w - center|^2]
    w = np.zeros(d) if w0 is None else w0.copy()
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (beta * (p - y)) + kappa * w
        Hw = (X.T * (beta * p * (1 - p))) @ X + kappa * np.eye(d)
        if lam:
            g, Hw = g + lam * (w - center), Hw + lam * np.eye(d)
        step = np.linalg.solve(Hw, g)
        w = w - step
        if np.abs(step).max() < 1e-14 * (1 + np.abs(w).max()):
            break
    return w


def phi(w):
    # the observables: test loss at each of the nq held-out points (a vector)
    m = Xq @ w
    return np.logaddexp(0, m) - yq * m


w_star = fit(np.ones(n))
p = 1 / (1 + np.exp(-X @ w_star))
G = X * (p - y)[:, None]                          # row i: g_i = grad ell_i(w*)
H = (X.T * (p * (1 - p))) @ X + kappa * np.eye(d) # Hessian of L(w; 1) at w*, summed units
pq = 1 / (1 + np.exp(-Xq @ w_star))
Gq = Xq * (pq - yq)[:, None]                      # row q: grad phi_q(w*)
ev = np.linalg.eigvalsh(H)
lam = np.median(ev)
print(f'training accuracy {np.mean((X @ w_star > 0) == y):.2f}; '
      f'Hessian eigenvalues {ev.min():.3g} ... {ev.max():.3g}; damping lambda = {lam:.3g}')
"""
    ),
    code_answer(
        r"""
def attribution_scores(H, G, Gq, lam):
    # Return a dict name -> array I of shape (nq, n), with I[q, i] = d phi_q / d beta_i:
    #   'exact IF':  -grad phi_q . H^{-1} g_i
    #   'damped IF': -grad phi_q . (H + lam I)^{-1} g_i
    #   'grad dot':  -grad phi_q . g_i
    # Use np.linalg.solve rather than forming an inverse.
    raise NotImplementedError('complete attribution_scores')
""",
        r"""
def attribution_scores(H, G, Gq, lam):
    I_d = np.eye(len(H))
    return {
        'exact IF': -Gq @ np.linalg.solve(H, G.T),
        'damped IF': -Gq @ np.linalg.solve(H + lam * I_d, G.T),
        'grad dot': -Gq @ G.T,
    }
""",
    ),
    code(
        r"""
scores = attribution_scores(H, G, Gq, lam)
t0 = time.time()
eps = 1e-3
fd = np.zeros((nq, n)); fd_prox = np.zeros((nq, n)); loo = np.zeros((nq, n))
for i in range(n):
    e = np.zeros(n); e[i] = eps
    fd[:, i] = (phi(fit(1 + e, w_star)) - phi(fit(1 - e, w_star))) / (2 * eps)            # check 1
    fd_prox[:, i] = (phi(fit(1 + e, w_star, lam, w_star))
                     - phi(fit(1 - e, w_star, lam, w_star))) / (2 * eps)                  # damped IF's own target
    b = np.ones(n); b[i] = 0
    loo[:, i] = phi(fit(b, w_star)) - phi(w_star)                                         # check 2
S = np.zeros((100, n), bool)                                                              # check 3:
for s in S:                                                                               # 100 subsets
    s[rng.choice(n, n // 2, replace=False)] = True                                        # of exactly n/2
out = np.array([phi(fit(s.astype(float), w_star)) for s in S])                            # (100, nq)
print(f'{n} x 2 x 2 finite-difference fits, {n} LOO fits, 100 half-subset fits: {time.time() - t0:.1f} s\n')


def slope(xv, yv):                                # least-squares slope of actual on predicted
    xv = xv - xv.mean()
    return xv @ (yv - yv.mean()) / (xv @ xv)


print(f'{"method":10s} | {"1: rel. error vs":>16s} | {"2: LOO":^17s} | {"3: subsets of 50%":^19s}')
print(f'{"":10s} | {"local derivative":>16s} | {"Spearman":>8s} {"slope":>8s} | {"LDS":>8s} {"slope":>10s}')
for name, I in scores.items():
    err1 = np.mean(np.linalg.norm(I - fd, axis=1) / np.linalg.norm(fd, axis=1))
    sp2 = np.mean([spearmanr(-I[q], loo[q])[0] for q in range(nq)])
    sl2 = np.mean([slope(-I[q], loo[q]) for q in range(nq)])
    pred = S.astype(float) @ I.T                                                          # phi-hat(S_k), (100, nq)
    lds = np.mean([spearmanr(pred[:, q], out[:, q])[0] for q in range(nq)])
    sl3 = np.mean([slope(pred[:, q], out[:, q]) for q in range(nq)])
    print(f'{name:10s} | {err1:16.1e} | {sp2:8.3f} {sl2:8.2f} | {lds:8.3f} {sl3:10.2f}')
I_d = scores['damped IF']
print(f'\ndamped IF vs its own (proximal) target: rel. error '
      f'{np.mean(np.linalg.norm(I_d - fd_prox, axis=1) / np.linalg.norm(fd_prox, axis=1)):.1e}')

q = int(np.argmax(phi(w_star)))                  # show the test point with the largest loss
fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
for k, (name, I) in enumerate(scores.items()):
    ax[0, k].scatter(-I[q], loo[q], s=12)
    ax[0, k].set_title(f'{name}: LOO (test point {q})', fontsize=10)
    ax[0, k].set_xlabel(r'predicted $-\mathcal{I}_i$'); ax[0, k].set_ylabel('actual change')
    pred = S.astype(float) @ I[q]
    ax[1, k].scatter(pred - pred.mean(), out[:, q] - out[:, q].mean(), s=12, color='C1')
    ax[1, k].set_title(f'{name}: half-subsets', fontsize=10)
    ax[1, k].set_xlabel(r'$\hat\phi(S)$, centred'); ax[1, k].set_ylabel(r'$\phi(S)$, centred')
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
<a id="da1-ex5-2"></a>

**Exercise 5.2 — Read the table.** Using the printed table (or the page's
copy of it):

1. Check 1: the exact IF matches its target to many digits and the damped IF does not. What does the damped IF match exactly, and is its error in check 1 a defect?
2. *(Optional.)* Check 2: the LOO slopes exceed 1 for the influence functions. Which part of §3 predicts this?
3. Check 3: compare which method does best in check 2 and in check 3. What do you conclude about using the LDS to choose a method for answering LOO questions, or the reverse?
4. Check 3 again: the slopes are far from 1, yet the LDS is high. What does a high LDS license you to say, and what not?
5. *(Optional.)* Why is the gradient dot product worst here, and what would make it look better?
"""
    ),
    answer(
        r"""
The numbers below are from the page's run (seed 1); your values will match if you did not change the setup.

1. The exact IF matches the local derivative of retraining to about $3\times10^{-7}$, the accuracy of the finite differences. The damped IF is off by about 50% there, because $\lambda$ is as large as a typical eigenvalue of $H$. But it matches its own target, the derivative of *proximal* retraining, to the same $3\times10^{-7}$ (the last printed line). So check 1 confirms both estimators. The damped IF's error is a difference of target, not a defect, and §4 explains why one would choose that target for a network.
2. Exercise 3.1: exact deletion is the first-order prediction times $1/(1-h_j)$ in linear regression. More generally, the higher-order terms of the response add to the first-order term. For logistic regression the factor is not exactly $1/(1-h_j)$, but the same mechanism makes deletions larger than predicted (slopes of about 1.4 for the exact IF here). The near-perfect Spearman correlation shows that the ranking survives. The damped IF's fitted slope is larger still (about 1.9).
3. On LOO the exact IF is clearly best and the damped IF clearly worse (Spearman about 0.96 against 0.82 here). On half-subsets the two are about equal (LDS about 0.83 for both). The checks measure different things, so a ranking of methods by one need not transfer to the other. Choose the check that matches the question you care about.
4. A high LDS says the summed scores *order* subsets by their outcome well. It says nothing about the size of the effect: here the fitted calibration slopes of actual on predicted outcomes range from about 12 to 35, far from 1. To claim "deleting these examples changes the loss by about this much", you need a calibration check like the slopes, not the LDS. How large the miscalibration is depends on the setup: with a stronger ridge ($\kappa=1$ in the setup cell), the two influence functions' slopes fall to about 3 and their LDS rises to about 0.97.
5. Replacing $H^{-1}$ by the identity weights each direction by the raw size of the gradients, not by how much the parameters respond. Along a low-variance feature direction, with scale $\sigma$, the gradients are of order $\sigma$ and the curvature of order $\sigma^2$. The IF's $g_\phi^\top H^{-1}g_i$ is then of order one there, while the dot product is of order $\sigma^2$ and effectively ignores that direction. Here the Hessian's eigenvalues span a factor of about 360, from the feature scales and from the logistic weights $p_i(1-p_i)$. If $H$ were a multiple of the identity, the dot product would be proportional to the IF and rank identically. So similarity methods look like influence functions exactly when the curvature is close to isotropic in the directions that matter.
"""
    ),
    # ================================================================ 6
    markdown(
        r"""
<a id="da1-6"></a>

## 6. A lens for the methods, and where to go next

DA2's methods answer different questions, but their practical estimators
often share one shape: a curvature matrix, something that tames its small
eigenvalues, a training gradient and a query gradient. This section lays
that shape out as a table of spectral filters, says what the table hides, and
hands you to the DA2 tour.

**Spectral filters.** Take a symmetric curvature matrix $M$ with eigenpairs
$(s_k,u_k)$, and expand the training gradient $g_i=\sum_kc_ku_k$. Many
estimators predict a parameter response $\partial w/\partial\beta_i$ of the
form

$$
\frac{\partial w}{\partial\beta_i}=-\sum_kF(s_k)\,c_k\,u_k=-F(M)\,g_i ,
$$

and differ in the filter $F$ and in which matrix $M$ is. The classical IF is
$F(s)=1/s$ with $M=H$. In summed-loss units (§2):

| Method | Filter $F(s)$ | $s$ is an eigenvalue of | Holds when |
|---|---|---|---|
| classical IF | $1/s$ | $H$, invertible | §3's assumptions |
| damped Hessian IF | $1/(s+\lambda)$ | $H$ | by definition; at a stationary point it is the proximal response (§4) |
| PBRF | $1/(s+\lambda)$ | $G$, a different matrix from $H$ | at the checkpoint (§4) |
| frozen-curvature GD, $K$ steps of size $\eta$ | $\big[1-(1-\eta s)^K\big]/s$ | $H$ | curvature and gradient constant over the $K$ steps |
| SOURCE, one segment | $\big(1-e^{-\eta Ks}\big)/s$ | the segment's average $H$ | unpreconditioned SGD, after its stationarity, independence and exponential approximations |
| local Bayesian IF | $\approx1/(s+\gamma)$ | $H$ | Gaussian leading order at a stationary minimum |
| Bayesian linear regression | $1/(s+\sigma_{\rm noise}^2/\tau^2)$ | $X^\top X$ | exact, for the posterior mean |

<details><summary>Notes on the rows</summary>

- *GD rows:* $\eta$ is the step size on the summed loss (for a mean-loss step size, use $\eta/n$).
- *SOURCE* (DA2 stop 2) splits training into segments and treats curvature and gradient as constant within each.
- *Local Bayesian IF:* $\gamma$ is the localization strength of the local posterior $p_\beta$ of §2.
- *Bayesian linear regression:* $\tau^2$ is the prior variance and $\sigma^2_{\rm noise}$ the observation noise. The gradients are the unscaled squared-error ones, $-x_jr_j$. In likelihood-Hessian units, $X^\top X/\sigma_{\rm noise}^2$, the damping is $\tau^{-2}$.

</details>

**Reading the table.**

- Every row except the classical IF is finite at $s=0$: $F(0)=1/\lambda$, $\eta K$, $1/\gamma$ or $\tau^2/\sigma_{\rm noise}^2$. This is the sense in which each "regularizes the small eigenvalues": flat directions get a bounded response.
- The shapes are similar; the meanings are not. For the damped IF, $\lambda$ is a proximal penalty. For GD, $\eta K$ is training time, so influence already depends on how long you train: directions with $s\ll1/(\eta K)$ have not yet responded fully. For the Bayesian IF, $\gamma$ is a localization strength. For Bayesian linear regression, the damping is a prior.
- The SOURCE filter (and the GD filter, for $\eta s$ well below 2) crosses over from $\eta K$ to $1/s$ near $s\approx1/(\eta K)$. It is not ridge: at the crossover it is $(1-e^{-1})\,\eta K\approx0.63\,\eta K$, against $0.5\,\eta K$ for ridge with $\lambda=1/(\eta K)$.

<details><summary>Where the GD row comes from</summary>

Run GD on $L(w;\beta)$ with constant Hessian $H$ and constant gradient $g_i$,
and let $r_t=\partial w_t/\partial\beta_i$. One step,
$w_{t+1}=w_t-\eta\nabla_wL(w_t;\beta)$, gives $r_{t+1}=(I-\eta H)\,r_t-\eta g_i$,
with $r_0=0$. Summing the geometric series,
$r_K=-\eta\sum_{k<K}(I-\eta H)^kg_i=-\big[I-(I-\eta H)^K\big]H^{-1}g_i$ on the
range of $H$, and $-\eta K\,g_i$ on its null space. If $0<\eta s<2$, the filter
tends to $1/s$ as $K\to\infty$; it is close to $1/s$ once $|1-\eta s|^K\ll1$.
</details>

**What the table hides.**

- General unrolling has gradients and curvature that change along the run, and step Jacobians that do not commute. No single filter describes it (DA2 stop 2).
- Multi-segment SOURCE propagates each segment's response through every later segment.
- The Bayesian IF beyond leading order sees non-quadratic structure, which is the reason to use it (DA2 stop 3).
- Datamodels have no such form: they are fitted, not derived.
- Negative curvature ($s<0$): the classical IF follows a saddle, not a minimum; the damped filter diverges at $s=-\lambda$; the GD and SOURCE filters grow with training time. That is one reason §4 replaces $H$ by $G$, which has no negative eigenvalues.

<a id="da1-6-figure"></a>

**Optional: the filters, interactively.** Set the step size, the number of
steps and the damping, and compare the filters on linear axes (to see $F(0)$),
on log axes (to see small and large $s$), and with negative curvature.

"""
        + figure("assets/interactive/filter_explorer.html",
                 "Damped, GD and SOURCE filters with sliders for $\\eta$, $K$ and $\\lambda$; $F(0)$ and the crossover values are tabulated.")
        + r"""

<a id="da1-6-tour"></a>

**The DA2 tour.** [DA2](da2_methods.html) is one ordered tour. Read each
stop's short capsule in order (about 30 minutes in all), then go deep where
you like, either through the paper or through the stop's optional
walkthrough.

1. [Influence functions at scale](da2_methods.html#da2-stop-1): EK-FAC approximations to the damped Gauss–Newton IF, and what they showed about large language models.
2. [Unrolling](da2_methods.html#da2-stop-2): differentiate the actual training run; TracIn, SOURCE and MAGIC.
3. [Bayesian influence functions](da2_methods.html#da2-stop-3): a covariance over a local posterior in place of a Hessian inverse, estimated by SGLD.
4. [Influence changes over training](da2_methods.html#da2-stop-4): in a deep linear network, an example's influence can change sign as structure is learned.
5. [Datamodels and TRAK](da2_methods.html#da2-stop-5): fit the map from subsets to outcomes, or estimate it analytically.
6. [Open problems and uses](da2_methods.html#da2-stop-6): entangled concepts, attributing behaviours rather than losses, data selection, unlearning, and scale.
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "da1_core", "da1_core"))
