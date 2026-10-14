#!/usr/bin/env python3
"""Build SLT Notebook 1, "Bayesian inference as statistical mechanics" (slt1_bayes_statmech).

Shape (SLT_SPECS_v1.md, 2026-10-09): a question, the keystone reading, a one-line
route. Each `##` section opens with its question and restates its notation, shows a
conceptual figure where its idea is introduced, gives a short default text, then
its exercises with worked solutions in `answer()` cells, and small CPU code cells.
Sections: 1.1 the dictionary; 1.2 posterior concentration (the CLT); 1.3 Laplace
and the BIC; 1.4 the Bayes generalization error and the MLE-vs-Bayes comparison.

Figures: assets/figures/make_slt1_bayes_statmech_figures.py (run it from slt/).
Numerical checks of the closed forms used in the solutions: checks/slt1_checks.py.
"""
from nbkit_slt import answer, book, code, code_answer, image, markdown, reading, references, run

FIG = "assets/figures/slt1_bayes_statmech_"

CELLS = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# SLT Notebook 1 — Bayesian inference as statistical mechanics

**Singular Learning Theory, Iliad Intensive 2026**

> **Read the blog posts first.** The best introductions to this material are blog posts, and they are probably better than these notebooks: Liam Carroll's sequence [Distilling Singular Learning Theory](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) (DSLT 0–4), Jesse Hoogland's [Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick) and, with Stan van Wingerden, [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong); then Zach Furman's [Singular learning theory: exercises](https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises) and Olli Järviniemi's [Dialogue introduction to Singular Learning Theory](https://www.lesswrong.com/posts/CmcarN6fGgTGwGuFp/dialogue-introduction-to-singular-learning-theory). The notebooks are provided for reference: derivations, worked exercises and code to check things.

*Question:* what does Bayes do with $n$ samples, and how much does it generalize?

Bayes' rule has the form of a Boltzmann distribution, with the training loss as
the energy and the sample size $n$ as the inverse temperature. This notebook sets
up that dictionary and uses it to answer the question for *regular* models, the
case classical statistics covers. Notebook 2 asks what changes when the model is
not regular. The matching blog post is
[DSLT 1](https://www.lesswrong.com/posts/4eZtmwaqhAgdJQDEg/dslt-1-the-rlct-measures-the-effective-dimension-of-neural),
sections "Preliminaries of SLT" and "Deriving the Bayesian Information Criterion".
"""
        + reading(
            "Bayesian Interpolation",
            "https://doi.org/10.1162/neco.1992.4.3.415",
            "MacKay (Neural Computation 1992)",
            "The evidence $Z_n$ as the quantity Bayes uses to compare models, and the *Occam factor*: the evidence is the best-fit likelihood times the fraction of the prior volume the data leave standing, so a model with more parameters pays for them by volume (§1.3).",
            "Sections 1–4 are what matters here. The same text is chapter 2 of MacKay's thesis, [free as a PDF](https://www.inference.org.uk/mackay/thesis.pdf).",
            importance=5,
            length=3,
        )
        + r"""
**Route.**

- [1.1 The dictionary](#slt1-s1): posterior as Boltzmann distribution, evidence as partition function, free energy; where the analogy is only formal.
- [1.2 Posterior concentration](#slt1-s2): for large $n$ the posterior is a Gaussian of width $n^{-1/2}$ around the maximum-likelihood estimate.
- [1.3 Laplace and the BIC](#slt1-s3): $F_n = nL_n(\hat w)+\tfrac d2\log n+O(1)$; the Occam factor is a volume ratio.
- [1.4 Bayes generalization error](#slt1-s4): $d/2n$ for the Bayes predictor, $d/n$ for one posterior sample, and how maximum likelihood compares when the model is wrong.

**Setup used throughout.** Data $x_1,\dots,x_n$ are drawn i.i.d. from an unknown
density $q(x)$ (for supervised learning, $x$ is an input–output pair; nothing
changes). A *model* is a family of densities $p(x\mid w)$, $w\in\mathbb R^d$, with
a smooth positive *prior* density $\varphi(w)$. The per-sample loss is
$\ell(x,w)=-\log p(x\mid w)$, and

$$
L_n(w)=\frac1n\sum_{i=1}^n\ell(x_i,w)\quad\text{(training loss)},\qquad
L(w)=\mathbb E_{x\sim q}[\ell(x,w)]\quad\text{(population loss)}.
$$

With $S=-\mathbb E_q[\log q(x)]$ the entropy of the data,
$L(w)=S+\mathrm{KL}(q\,\|\,p_w)$: minimizing $L$ finds the model closest to $q$.
Let $w_0$ minimize $L$; the *excess* losses are $K(w)=L(w)-L(w_0)\ge0$ and
$K_n(w)=L_n(w)-L_n(w_0)$. The model is *well specified* if $q=p_{w_0}$ (then $K$
is the KL divergence to the truth), and *misspecified* otherwise.
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(0)


def log_trapz(logf, x):
    # log of the trapezoid-rule integral of exp(logf) over the 1-D grid x,
    # computed in log space so that very small integrands do not underflow
    m = np.logaddexp(logf[:-1], logf[1:]) - np.log(2)
    return logsumexp(m + np.log(np.diff(x)))
"""
    ),
    # ------------------------------------------------------------------ 1.1
    markdown(
        r"""
<a id="slt1-s1"></a>

## 1.1 The dictionary

*Question:* what does Bayes' rule look like in the language of statistical
mechanics, and which parts of the correspondence are more than notation?

*Notation:* $p(x\mid w)$ model, $\varphi(w)$ prior, $\ell(x,w)=-\log p(x\mid w)$,
$L_n$ training loss, $L$ population loss, $K=L-L(w_0)$ (see the setup above).
"""
        + image(
            FIG + "dictionary.svg",
            r"Bayes' rule as a Boltzmann weight: prior times $e^{-\text{energy}}$ gives an unnormalized weight whose total area is the partition function $Z_n$ (dividing by it gives the posterior) and whose area over a region $U$ is $Z_n(U)$; the well $U$ is deeper but, being narrower, holds less mass than $V$.",
        )
        + r"""
Bayes' rule multiplies the prior by the likelihood of the data and normalizes:

$$
\pi_n(w)=\frac{\varphi(w)\prod_{i=1}^np(x_i\mid w)}{Z_n},
\qquad
Z_n=\int\varphi(w)\prod_{i=1}^np(x_i\mid w)\,dw .
$$

Since $\prod_ip(x_i\mid w)=e^{-nL_n(w)}$, the posterior $\pi_n\propto\varphi\,e^{-nL_n}$
is a Boltzmann distribution with energy $nL_n(w)$ at inverse temperature $1$ (or
energy $L_n$ at inverse temperature $n$). The normalizer $Z_n$, the *evidence* or
*marginal likelihood* (the probability of the data under the model and prior), is
the partition function. Its negative logarithm $F_n=-\log Z_n$ is the *free
energy*: lower $F_n$ means the model gave the data higher probability.

Two additions. The *tempered posterior*
$\pi_{n,\beta}\propto\varphi\,e^{-\beta nL_n}$, with partition function $Z_n(\beta)$
and free energy $F_n(\beta)=-\log Z_n(\beta)$, adds an inverse-temperature knob:
$\beta=1$ is Bayes, and other values are a device Notebook 3 needs. And factoring
out the best achievable fit, with $K_n=L_n-L_n(w_0)$,

$$
Z_n=e^{-nL_n(w_0)}\,\bar Z_n,\qquad \bar Z_n=\int\varphi(w)e^{-nK_n(w)}\,dw,
\qquad F_n=nL_n(w_0)+\bar F_n ,
$$

separates $nL_n(w_0)$, the loss of the population-optimal parameter on this data
set, from $\bar F_n=-\log\bar Z_n$, which measures how much prior mass fits the data
nearly as well as $w_0$.

| Bayesian statistics | symbol | statistical mechanics |
|---|---|---|
| parameter vector $w$ | | microstate |
| prior density $\varphi(w)$ | | a priori weight on microstates |
| $nL_n(w)$ (or $L_n(w)$) | $E(w)$ | energy |
| $\beta$ (or $\beta n$); Bayes is $\beta=1$ | $\beta=1/k_BT$ | inverse temperature |
| posterior $\pi_{n,\beta}\propto\varphi\,e^{-\beta nL_n}$ | | Boltzmann distribution $e^{-\beta E}/Z$ |
| evidence $Z_n(\beta)$ | $Z$ | partition function |
| $F_n(\beta)=-\log Z_n(\beta)$ | $\beta F$ | free energy, in units of $k_BT$ |
| posterior mean of $nL_n$ | $\langle E\rangle$ | internal energy |

**Where the dictionary is only formal.**

- *The energy is random.* $nL_n(w)$ depends on the sample (quenched disorder, as in a spin glass rather than a fixed Hamiltonian), so every statement about $F_n$ is about a random variable.
- *No thermodynamic limit.* $n$ is not a system size: $d$ stays fixed while $n$ grows.
- *Temperature is not a physical knob.* $\beta\ne1$ is not Bayesian inference.
- *No dynamics.* The posterior is a static distribution; nothing equilibrates.
- *The prior is not the density of states.* It weights parameters; the analogue of the density of states is the prior volume of parameters at a given loss, which Notebook 2 introduces.

What holds exactly is the algebra: identities of the form "derivative of $\log Z$
equals a posterior average" are the same as in statistical mechanics. Exercise 1.1
derives the first of them.

**Exercise 1.1 (Posterior as a Boltzmann distribution).** Let $U\subseteq\mathbb R^d$ be a region of parameter space, and define the *local* partition function and free energy
$Z_n(U)=\int_U\varphi(w)e^{-nL_n(w)}dw$ and $F_n(U)=-\log Z_n(U)$.

1. Show that $\pi_n(w)=\varphi(w)e^{-nL_n(w)}/Z_n$ with $Z_n=\int\varphi e^{-nL_n}$, i.e. that the likelihood is $e^{-nL_n}$.
2. Show that the posterior mass of $U$ is $\pi_n(U)=Z_n(U)/Z_n$, and that for two regions $\log\frac{\pi_n(U)}{\pi_n(V)}=F_n(V)-F_n(U)$: posterior odds are exponential in free-energy differences.
3. If $U_1,\dots,U_k$ partition parameter space, show $F_n=-\log\sum_je^{-F_n(U_j)}$, and that if $F_n(U_1)$ is the smallest then $F_n(U_1)-\log k\le F_n\le F_n(U_1)$. So the total free energy is set by the region of lowest free energy, up to $\log k$.
4. Show that $\partial_\beta F_n(\beta)=n\,\mathbb E_{\pi_{n,\beta}}[L_n(w)]$ and $\partial^2_\beta F_n(\beta)=-n^2\mathrm{Var}_{\pi_{n,\beta}}[L_n(w)]$. (In the dictionary: the derivative of the free energy with respect to inverse temperature is the internal energy, and $F_n(\beta)$ is concave.)
"""
    ),
    answer(
        r"""
**1.** $\prod_ip(x_i\mid w)=\exp\sum_i\log p(x_i\mid w)=\exp(-\sum_i\ell(x_i,w))=e^{-nL_n(w)}$. Substituting into Bayes' rule gives the stated form, and $Z_n$ is whatever makes $\pi_n$ integrate to one.

**2.** $\pi_n(U)=\int_U\pi_n=\frac1{Z_n}\int_U\varphi e^{-nL_n}=Z_n(U)/Z_n$. Then $\pi_n(U)/\pi_n(V)=Z_n(U)/Z_n(V)=e^{-F_n(U)}/e^{-F_n(V)}$, and taking logs gives $F_n(V)-F_n(U)$.

**3.** The integral over a disjoint union is the sum of the integrals, so $Z_n=\sum_jZ_n(U_j)=\sum_je^{-F_n(U_j)}$; take $-\log$. For the bounds: the sum is at least its largest term $e^{-F_n(U_1)}$ and at most $k$ times it, so $e^{-F_n(U_1)}\le Z_n\le ke^{-F_n(U_1)}$, which after $-\log$ is the stated sandwich. (The upper bound is strict as soon as another region has nonzero mass; the lower bound is attained when all $k$ regions have equal mass.)

**4.** $Z_n(\beta)=\int\varphi e^{-\beta nL_n}$, so $\partial_\beta Z_n=-\int nL_n\,\varphi e^{-\beta nL_n}$ and $\partial_\beta F_n=-\partial_\beta Z_n/Z_n=n\,\mathbb E_{\pi_{n,\beta}}[L_n]$. Differentiating once more, $\partial_\beta\mathbb E_\beta[nL_n]=\partial_\beta\frac{\int nL_n\varphi e^{-\beta nL_n}}{\int\varphi e^{-\beta nL_n}}=-\mathbb E_\beta[(nL_n)^2]+\mathbb E_\beta[nL_n]^2=-n^2\mathrm{Var}_\beta[L_n]$. A variance is nonnegative, so $F_n(\beta)$ is concave in $\beta$. These are the usual relations $\langle E\rangle=\partial_\beta(\beta F)$ and $\mathrm{Var}(E)=-\partial_\beta\langle E\rangle$; Notebook 3 turns the first one into an estimator.
"""
    ),
    # ------------------------------------------------------------------ 1.2
    markdown(
        r"""
<a id="slt1-s2"></a>

## 1.2 Posterior concentration

*Question:* what does the posterior look like when $n$ is large?

*Notation:* $\hat w$ is the *maximum-likelihood estimate*, a minimizer of the
training loss $L_n$; $w_0$ minimizes the population loss $L$;
$H=\nabla^2L(w_0)$ and $H_n=\nabla^2L_n(\hat w)$ are Hessians;
$C=\mathrm{Cov}_{x\sim q}[\nabla_w\ell(x,w_0)]$ is the covariance of the
per-sample gradient at $w_0$. The model is *regular* at $w_0$: $w_0$ is the
unique minimizer of $L$ and $H$ is positive definite (plus the usual technical
conditions: $\ell$ smooth near $w_0$, gradients with finite second moments, a prior
continuous and positive at $w_0$).
"""
        + image(
            FIG + "concentration.svg",
            r"Schematic: the posterior's Gaussian approximation for $n=5,50,500$ (ellipses at two standard deviations, covariance $(nH)^{-1}$) over the prior (grey); each tenfold increase in $n$ shrinks the widths by $\sqrt{10}$, and the centre $\hat w$ (dots) settles near $w_0$ (cross).",
        )
        + r"""
For large $n$ the factor $e^{-nL_n}$ sharpens while the prior stays fixed, so the
posterior is squeezed onto the minimizer $\hat w$ of $L_n$, where only the
quadratic part of $L_n$ matters. Separately, $\hat w$ is random, because $L_n$ is
built from a random sample, and the central limit theorem says how it scatters
around $w_0$. The first statement is the Bernstein–von Mises theorem, the second
the asymptotic normality of the maximum-likelihood estimate:

$$
\pi_n(w)\approx\mathcal N\!\big(\hat w,\,(nH_n)^{-1}\big),
\qquad
\hat w-w_0\approx\mathcal N\!\big(0,\,\tfrac1n H^{-1}CH^{-1}\big).
$$

Both widths scale as $n^{-1/2}$. The first is what the posterior believes; the
second is how $\hat w$ actually fluctuates from one data set to the next. In a
well-specified model $C=H$ and the two agree. In a misspecified one they differ,
and the posterior is overconfident or underconfident depending on $H^{-1}C$.
"""
        + image(
            FIG + "covariances.svg",
            r"Model $\mathcal N(\mu,e^{2s})$ with $w=(\mu,s)$, $s$ the log standard deviation, $n=400$, axes scaled by $\sqrt n$: the posterior's ellipse (shaded, covariance $H^{-1}$ in these coordinates, drawn at $w_0$) against $\hat w$ on 600 independent data sets (dots), which agree when the model is well specified and disagree for skewed data, where the posterior is overconfident about $s$.",
        )
        + r"""
You may use the central limit theorem in this form: the average of $n$
independent copies of a random vector with mean zero and covariance $C$ is
approximately $\mathcal N(0,C/n)$ for large $n$.

**Exercise 1.2 (The Gaussian posterior).**

1. Expand $nL_n(w)$ to second order about $\hat w$ (note $\nabla L_n(\hat w)=0$) and show that, if the prior is replaced by the constant $\varphi(\hat w)$, the posterior is exactly $\mathcal N(\hat w,(nH_n)^{-1})$.
2. Why can the prior be replaced by a constant, and why does the cubic term of the Taylor expansion not matter, for large $n$? Argue by scaling: write $w=\hat w+u/\sqrt n$ and look at the size of each term.
3. Show that $\hat w-w_0\approx-H^{-1}\nabla L_n(w_0)$, and that $\nabla L_n(w_0)$ is an average of $n$ independent mean-zero vectors with covariance $C$. Conclude the second display above. (Why is the mean zero?)
4. *(The information identity.)* Suppose the model is well specified, $q=p_{w_0}$. Show that $C=H$. <details><summary>Hint</summary>Start from $\int p(x\mid w)\,dx=1$ for every $w$ and differentiate twice with respect to $w$; use $\nabla^2p=p\,(\nabla^2\log p+\nabla\log p\,\nabla\log p^\top)$.</details>
"""
    ),
    answer(
        r"""
**1.** $nL_n(w)=nL_n(\hat w)+\tfrac n2(w-\hat w)^\top H_n(w-\hat w)+R$, with $R$ cubic and higher. Dropping $R$ and setting $\varphi\equiv\varphi(\hat w)$, the posterior is proportional to $\exp[-\tfrac12(w-\hat w)^\top(nH_n)(w-\hat w)]$, which is the density of $\mathcal N(\hat w,(nH_n)^{-1})$ up to its normalizing constant.

**2.** With $w=\hat w+u/\sqrt n$ the quadratic term is $\tfrac12u^\top H_nu$, of order one, so $u$ ranges over an $O(1)$ region: the posterior has width $n^{-1/2}$ in $w$. The cubic term is $n\cdot O(|w-\hat w|^3)=O(n\cdot n^{-3/2})=O(n^{-1/2})$ on that region, and higher terms are smaller still. A smooth prior changes by $O(n^{-1/2})$ across a region of width $n^{-1/2}$. So both corrections vanish as $n\to\infty$, at rate $n^{-1/2}$. (This is the regular case: if $H_n$ had a zero eigenvalue the quadratic term would not confine $u$ in that direction, and the higher terms would decide the width. That is Notebook 2.)

**3.** $0=\nabla L_n(\hat w)\approx\nabla L_n(w_0)+\nabla^2L_n(w_0)(\hat w-w_0)$, and $\nabla^2L_n(w_0)\to H$ by the law of large numbers, so $\hat w-w_0\approx-H^{-1}\nabla L_n(w_0)$. Now $\nabla L_n(w_0)=\frac1n\sum_i\nabla\ell(x_i,w_0)$ is an average of independent copies of the vector $g=\nabla\ell(x,w_0)$. Its mean is $\mathbb E_q[\nabla\ell(x,w_0)]=\nabla L(w_0)=0$ because $w_0$ minimizes $L$ (interchange of gradient and expectation), and its covariance is $C$ by definition. The central limit theorem gives $\nabla L_n(w_0)\approx\mathcal N(0,C/n)$, and a linear image $-H^{-1}(\cdot)$ of a Gaussian is Gaussian with covariance $H^{-1}(C/n)H^{-1}$.

**4.** Differentiate $\int p(x\mid w)dx=1$ once: $\int\nabla p=0$, i.e. $\mathbb E_{p_w}[\nabla\log p(x\mid w)]=0$. Differentiate again: $\int\nabla^2p=0$, and with the hint $\mathbb E_{p_w}[\nabla^2\log p]+\mathbb E_{p_w}[\nabla\log p\,\nabla\log p^\top]=0$. Evaluate at $w=w_0$ with $q=p_{w_0}$. Since $\ell=-\log p$, the first term is $-\mathbb E_q[\nabla^2\ell(x,w_0)]=-H$ and the second is $\mathbb E_q[\nabla\ell\nabla\ell^\top]$, which equals $C$ because the mean of $\nabla\ell(x,w_0)$ is zero. Hence $H=C$. Under misspecification the two derivatives are taken under $p_{w_0}$ but the expectations defining $H$ and $C$ are under $q$, and the argument fails; in general $H\ne C$.
"""
    ),
    # ------------------------------------------------------------------ 1.3
    markdown(
        r"""
<a id="slt1-s3"></a>

## 1.3 Laplace and the BIC

*Question:* what is the free energy $F_n=-\log Z_n$ for large $n$, and what does
it reward and penalize?

*Notation:* as in §1.2: $\hat w$ minimizes $L_n$, $H_n=\nabla^2L_n(\hat w)$ is
positive definite, $d$ is the number of parameters.

"""
        + image(
            FIG + "occam.svg",
            r"With a uniform prior the Occam factor is $V_{\rm post}/V_\varphi$, the area under $e^{-n(L_n(w)-L_n(\hat w))}$ over the prior volume (right: a schematic posterior contour); quadrupling $n$ halves each width and so multiplies the factor by $2^{-d}$, which is where $\tfrac d2\log n$ comes from.",
        )
        + r"""
*Laplace's method:* for large $n$ an integral $\int g(w)e^{-nf(w)}dw$ is dominated
by the neighbourhood of the minimum of $f$. Replace $f$ there by its quadratic
expansion and $g$ by its value at the minimum, and use the Gaussian integral

$$
\int_{\mathbb R^d}\exp\!\big(-\tfrac12u^\top Au\big)\,du=(2\pi)^{d/2}(\det A)^{-1/2}
\qquad(A\text{ positive definite}).
$$

Applied to $Z_n=\int\varphi(w)e^{-nL_n(w)}dw$ this gives

$$
Z_n\approx\varphi(\hat w)\,e^{-nL_n(\hat w)}\Big(\frac{2\pi}n\Big)^{d/2}(\det H_n)^{-1/2},
$$

$$
F_n=nL_n(\hat w)+\frac d2\log n+\frac12\log\det H_n-\log\varphi(\hat w)-\frac d2\log2\pi+o(1).
$$

The first two terms are the *Bayesian information criterion* (BIC): the best
training loss plus half a $\log n$ per parameter. The rest is $O(1)$: it depends on
the curvature and the prior, and can change the ranking of two models whose leading
terms are close.

MacKay reads the same formula as $Z_n=(\text{best-fit likelihood})\times
(\text{Occam factor})$, with Occam factor
$\varphi(\hat w)(2\pi/n)^{d/2}(\det H_n)^{-1/2}$. For a prior spread evenly over a
volume $V_\varphi$ this is $V_{\rm post}/V_\varphi$, the fraction of prior volume
the data leave standing (figure above). An extra parameter can buy a better fit,
but the evidence pays for the shrinkage of its width to about $n^{-1/2}$. No
held-out data enter; the penalty is volume alone.

**Exercise 1.3 (Laplace's method and the Occam factor).**

1. Derive the two displayed formulas for $Z_n$ and $F_n$ from the Gaussian integral.
2. Suppose the prior is uniform over a box of volume $V_\varphi$ containing $\hat w$. Define the *posterior volume* $V_{\rm post}=\int_{\rm box} e^{-n(L_n(w)-L_n(\hat w))}dw$ (the volume of parameter space, weighted by how much worse than the best fit each point is). Show that $Z_n=e^{-nL_n(\hat w)}\,V_{\rm post}/V_\varphi$ exactly, and that Laplace's method gives $V_{\rm post}\approx(2\pi/n)^{d/2}(\det H_n)^{-1/2}$. This is the Occam factor as a volume ratio.
3. Two models $A$ and $B$ have $d_A<d_B$ parameters and best training losses $L_n^A\ge L_n^B$ (the larger model fits at least as well). Using only the BIC terms, when does Bayes prefer the smaller model? Express the condition as a bound on the improvement in fit per sample.
4. *(A case where everything is exact.)* Model $p(x\mid w)=\mathcal N(x;w,1)$ with prior $\varphi=\mathcal N(0,\tau^2)$. Compute $Z_n$ exactly, write $F_n$ in the form $nL_n(\hat w)+\tfrac12\log n+(\text{rest})$, and compare with the Laplace formula. Show that the two differ by $O(1/n)$. <details><summary>Hint</summary>$\sum_i(x_i-w)^2=\sum_i(x_i-\bar x)^2+n(w-\bar x)^2$, so $nL_n(w)=nL_n(\bar x)+\tfrac n2(w-\bar x)^2$ exactly; the integral against the Gaussian prior is then one Gaussian integral.</details>
"""
    ),
    answer(
        r"""
**1.** Expand $nL_n(w)=nL_n(\hat w)+\tfrac12(w-\hat w)^\top(nH_n)(w-\hat w)+\dots$ (the gradient vanishes at $\hat w$), set $\varphi(w)\approx\varphi(\hat w)$, and apply the Gaussian integral with $A=nH_n$: $(2\pi)^{d/2}(\det nH_n)^{-1/2}=(2\pi/n)^{d/2}(\det H_n)^{-1/2}$, using $\det(nH_n)=n^d\det H_n$. Taking $-\log$ gives $F_n$. The $o(1)$ collects the prior's variation and the cubic terms, both $O(n^{-1/2})$ by Exercise 1.2.2, together with the $O(1/n)$ corrections to the Gaussian integral itself.

**2.** With $\varphi=1/V_\varphi$ on the box (and the box containing all the posterior mass that matters), $Z_n=\frac1{V_\varphi}\int e^{-nL_n(w)}dw=\frac{e^{-nL_n(\hat w)}}{V_\varphi}\int e^{-n(L_n(w)-L_n(\hat w))}dw$, which is the claim. Laplace's method on $V_{\rm post}$ is part 1 with $\varphi\equiv1$. So the evidence is the best-fit likelihood times $V_{\rm post}/V_\varphi$: the fraction of prior volume that remains compatible with the data. Each parameter contributes a factor (posterior width)/(prior width) $\sim n^{-1/2}/\text{(prior width)}$, which is where $\tfrac d2\log n$ comes from.

**3.** $F^A-F^B\approx n(L_n^A-L_n^B)-\frac{d_B-d_A}2\log n$. Bayes prefers $A$ (lower free energy) when $L_n^A-L_n^B<\frac{d_B-d_A}{2n}\log n$: the larger model must improve the per-sample fit by more than $\frac{\log n}{2n}$ per extra parameter to be preferred. For fixed losses the right-hand side goes to zero, so any genuine improvement eventually wins; but a model that only fits noise improves the per-sample loss by $O(1/n)$ (the fit improvement from $\Delta d$ extra parameters is about $\frac{\Delta d}{2n}$, Exercise 1.4.3), which is smaller than $\frac{\Delta d}{2n}\log n$. That is the sense in which the BIC penalty rejects parameters that only fit noise.

**4.** By the hint, $\prod_ip(x_i\mid w)=e^{-nL_n(\bar x)}e^{-\frac n2(w-\bar x)^2}$ with $\hat w=\bar x$ and $nL_n(\bar x)=\frac n2\log2\pi+\frac12\sum_i(x_i-\bar x)^2$. Then
$$
Z_n=e^{-nL_n(\bar x)}\int\frac{e^{-w^2/2\tau^2}}{\sqrt{2\pi\tau^2}}e^{-\frac n2(w-\bar x)^2}dw
=e^{-nL_n(\bar x)}\,\frac1{\sqrt{1+n\tau^2}}\exp\!\Big(-\frac{n\bar x^2}{2(1+n\tau^2)}\Big),
$$
(complete the square: the product of the two Gaussians in $w$ has precision $n+1/\tau^2$, and the leftover constant is a Gaussian in $\bar x$ with variance $\tau^2+1/n$). So
$$
F_n^{\rm exact}=nL_n(\bar x)+\tfrac12\log n+\tfrac12\log\!\big(\tau^2+\tfrac1n\big)+\frac{\bar x^2}{2(\tau^2+1/n)} .
$$
Laplace: $H_n=1$, $d=1$, $-\log\varphi(\hat w)=\tfrac12\log(2\pi\tau^2)+\bar x^2/2\tau^2$, so $F_n^{\rm Lap}=nL_n(\bar x)+\tfrac12\log n+\tfrac12\log\tau^2+\frac{\bar x^2}{2\tau^2}$. The difference is $\tfrac12\log(1+\tfrac1{n\tau^2})+\frac{\bar x^2}2\big(\frac1{\tau^2+1/n}-\frac1{\tau^2}\big)=\frac1{2n\tau^2}-\frac{\bar x^2}{2\tau^2(n\tau^2+1)}+O(n^{-2})=O(1/n)$. Laplace is exact up to the $1/n$ corrections from the prior's curvature, as it should be: here the likelihood is exactly Gaussian in $w$ and the only approximation is $\varphi\approx\varphi(\hat w)$.
"""
    ),
    markdown(
        r"""
**Code: the free energy by quadrature, against Laplace and the BIC.** The first
cell checks the conjugate model of Exercise 1.3.4 by numerical integration against
its closed form. The second uses a model with no closed form,
$p(x\mid\mu,s)=\mathcal N(x;\mu,e^{2s})$ with standard normal priors on $\mu$ and
$s$, and computes $F_n$ on a grid for a range of $n$. Against $\log n$,
$F_n-nL_n(\hat w)$ should have slope $d/2=1$, and Laplace should add the right
constant.
"""
    ),
    code(
        r"""
# 1-D conjugate model: x ~ N(w, 1), prior w ~ N(0, tau^2). Exact Z_n vs quadrature vs Laplace.
tau2, n = 4.0, 50
x = rng.normal(1.0, 1.0, size=n)
xbar = x.mean()
nL_hat = 0.5 * n * np.log(2 * np.pi) + 0.5 * np.sum((x - xbar) ** 2)   # n L_n(w_hat)

F_exact = nL_hat + 0.5 * np.log(n) + 0.5 * np.log(tau2 + 1 / n) + xbar**2 / (2 * (tau2 + 1 / n))
F_laplace = nL_hat + 0.5 * np.log(n) + 0.5 * np.log(tau2) + xbar**2 / (2 * tau2)
F_bic = nL_hat + 0.5 * np.log(n)

w = np.linspace(-10, 10, 20001)
log_integrand = (-0.5 * w**2 / tau2 - 0.5 * np.log(2 * np.pi * tau2)
                 - 0.5 * n * np.log(2 * np.pi) - 0.5 * ((x[None, :] - w[:, None]) ** 2).sum(1))
F_quad = -log_trapz(log_integrand, w)
print(f"F_n exact     {F_exact:.4f}")
print(f"F_n quadrature{F_quad:.4f}")
print(f"F_n Laplace   {F_laplace:.4f}   (difference {F_laplace - F_exact:+.4f}, predicted O(1/n) ~ {1/(2*n*tau2):.4f})")
print(f"F_n BIC terms {F_bic:.4f}")
"""
    ),
    code(
        r"""
# 2-D model: x ~ N(mu, exp(2 s)), priors mu ~ N(0,1), s ~ N(0,1). Data from N(1, 0.5^2).
mu_true, sigma_true = 1.0, 0.5
N_all = 3000
x_all = rng.normal(mu_true, sigma_true, size=N_all)
mu_grid = np.linspace(-2.0, 4.0, 601)
s_grid = np.linspace(-4.0, 2.0, 601)
MU, S = np.meshgrid(mu_grid, s_grid, indexing="ij")
log_prior = -0.5 * (MU**2 + S**2) - np.log(2 * np.pi)
dmu, ds = mu_grid[1] - mu_grid[0], s_grid[1] - s_grid[0]

def free_energy_2d(x):
    n = len(x)
    # n L_n(mu, s) = n s + n/2 log 2pi + exp(-2s) * sum (x_i - mu)^2 / 2
    ss = ((x[None, None, :] - MU[:, :, None]) ** 2).sum(-1) if n <= 300 else \
         n * ((MU - x.mean()) ** 2 + x.var())           # same thing, computed from sufficient statistics
    nL = n * S + 0.5 * n * np.log(2 * np.pi) + 0.5 * np.exp(-2 * S) * ss
    return -(logsumexp(log_prior - nL) + np.log(dmu * ds))

def laplace_2d(x):
    n = len(x)
    mu_hat, s_hat = x.mean(), 0.5 * np.log(x.var())
    nL_hat = n * s_hat + 0.5 * n * np.log(2 * np.pi) + 0.5 * n       # exp(-2 s_hat) * n var = n
    H = np.diag([np.exp(-2 * s_hat), 2.0])                            # Hessian of L_n at the optimum
    F_lap = (nL_hat + 1.0 * np.log(n) + 0.5 * np.log(np.linalg.det(H))
             + 0.5 * (mu_hat**2 + s_hat**2) + np.log(2 * np.pi) - np.log(2 * np.pi))
    return nL_hat, F_lap

ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 3000])
rows = []
for n in ns:
    x = x_all[:n]
    nL_hat, F_lap = laplace_2d(x)
    rows.append((n, free_energy_2d(x) - nL_hat, F_lap - nL_hat, 1.0 * np.log(n)))
rows = np.array(rows)
plt.figure(figsize=(5.5, 3.6))
plt.plot(rows[:, 0], rows[:, 1], "o-", label="exact (quadrature)")
plt.plot(rows[:, 0], rows[:, 2], "s--", label="Laplace")
plt.plot(rows[:, 0], rows[:, 3], ":", label=r"BIC: $\frac{d}{2}\log n$")
plt.xscale("log"); plt.xlabel("n"); plt.ylabel(r"$F_n - nL_n(\hat w)$")
plt.title("2-parameter Gaussian model: free energy above the best fit"); plt.legend(); plt.tight_layout(); plt.show()
print("slope of the exact curve against log n over the last four points:",
      np.polyfit(np.log(rows[-4:, 0]), rows[-4:, 1], 1)[0].round(3), " (d/2 = 1)")
"""
    ),
    # ------------------------------------------------------------------ 1.4
    markdown(
        r"""
<a id="slt1-s4"></a>

## 1.4 The Bayes generalization error

*Question:* how well does the posterior predict the next sample, and how does
that compare with using the maximum-likelihood fit?

*Notation:* $q$ data distribution with entropy $S$; $\pi_n$ posterior;
$\hat w$ maximum-likelihood estimate; $w_0=\arg\min L$, $H=\nabla^2L(w_0)$,
$C=\mathrm{Cov}_q[\nabla\ell(x,w_0)]$;
$L_{\min}=L(w_0)-S=\min_w\mathrm{KL}(q\|p_w)$, the smallest loss the model can
reach, zero if well specified.
"""
        + image(
            FIG + "predictors.svg",
            r"Maximum likelihood predicts with the fitted member $p(x\mid\hat w)$, Gibbs with one random member, and Bayes with the posterior average of all members; here averaging widens the predictive variance from $1$ to $1+1/n=1.25$ (Exercise 1.4.4: model $\mathcal N(x;w,1)$, flat prior, data variance 2, $n=4$, $\hat w=\bar x=0.4$).",
        )
        + r"""
The three predictors are the *Bayes predictive* $p^*(x)=\int p(x\mid w)\,\pi_n(w)\,dw$,
the *Gibbs* predictor $p(x\mid w)$ for one draw $w\sim\pi_n$, and the
*maximum-likelihood* predictor $p(x\mid\hat w)$. Each one's generalization error
is its expected loss on a fresh sample minus the entropy $S$, the part no
predictor can remove:

$$
\begin{aligned}
G_n^{\rm Bayes}&=\mathbb E_{x\sim q}[-\log p^*(x)]-S=\mathrm{KL}(q\,\|\,p^*),\\
G_n^{\rm Gibbs}&=\mathbb E_{w\sim\pi_n}[L(w)]-S,\\
G_n^{\rm MLE}&=L(\hat w)-S .
\end{aligned}
$$

All three are random through the training sample; the statements below are about
their expectations. The free energy enters through an exact identity: appending
one more sample $x_{n+1}$ to the data set,

$$
-\log p^*(x_{n+1})=F_{n+1}-F_n ,
\qquad\text{so}\qquad
\mathbb E[G_n^{\rm Bayes}]=\mathbb E[F_{n+1}]-\mathbb E[F_n]-S .
$$

The Bayes generalization error is the free-energy cost of one more datum, minus
the entropy. So the large-$n$ behaviour of $\mathbb E[F_n]$, including Notebook 2's
volume exponent, carries over to generalization, provided the remainder terms are
controlled (Exercise 1.4.2 says where this needs care). In the regular case:

| predictor | expected generalization error, regular model |
|---|---|
| Bayes predictive | $L_{\min}+\dfrac{d}{2n}$ |
| Gibbs (one posterior sample) | $L_{\min}+\dfrac{d+\mathrm{tr}(H^{-1}C)}{2n}$ |
| maximum likelihood | $L_{\min}+\dfrac{\mathrm{tr}(H^{-1}C)}{2n}$ |

up to $o(1/n)$. Well specified, $C=H$ and $\mathrm{tr}(H^{-1}C)=d$: Bayes and
maximum likelihood both pay $d/2n$, and a single posterior sample pays twice that.
Misspecified, Bayes still pays $d/2n$ above the best the model can do, while
maximum likelihood pays $\mathrm{tr}(H^{-1}C)/2n$, which can be larger or smaller
than $d/2n$ (Gibbs pays the sum of the two). So "Bayes generalizes better than
maximum likelihood" is not a theorem even for regular models; what averaging over
the posterior does is remove the dependence on $C$.

**Exercise 1.4 (Generalization error from the free energy).**

1. Prove $-\log p^*(x_{n+1})=F_{n+1}-F_n$, where $F_{n+1}$ is the free energy of the data set with $x_{n+1}$ appended.
2. Using the Laplace expansion of §1.3 and the fluctuation of $\hat w$ from §1.2, show $\mathbb E[F_n]=nL(w_0)-\tfrac12\mathrm{tr}(H^{-1}C)+\tfrac d2\log n+c+o(1)$ for a constant $c$, and deduce $\mathbb E[G_n^{\rm Bayes}]\approx L_{\min}+\tfrac d{2n}$. Say which step is a step of faith at this level of rigour. <details><summary>Hint</summary>$nL_n(\hat w)=nL_n(w_0)-\tfrac n2 g_n^\top H^{-1}g_n+\dots$ with $g_n=\nabla L_n(w_0)$; use $\mathbb E[nL_n(w_0)]=nL(w_0)$ and $\mathbb E[g_ng_n^\top]=C/n$.</details>
3. Derive the three rows of the table directly, without the free energy. For the Bayes row, write $w=\hat w+\delta$ with $\delta\sim\mathcal N(0,(nH)^{-1})$, expand $\log p(x\mid w)$ to second order in $\delta$, and use $\log\mathbb E[e^Y]\approx\mathbb E[Y]+\tfrac12\mathrm{Var}[Y]$ for a random variable $Y$ that is small. You will need $\mathbb E_q[\nabla\ell\,\nabla\ell^\top]=C$ and $\mathbb E_q[\nabla^2\ell]=H$ at $w_0$, and $\mathbb E[L(\hat w)]-L(w_0)\approx\tfrac1{2n}\mathrm{tr}(H^{-1}C)$ (prove this last one first).
4. *(One misspecified example.)* Model $\mathcal N(x;w,1)$, flat prior, data actually $\mathcal N(0,\sigma^2)$ with $\sigma^2\neq1$. Compute all three generalization errors exactly as functions of $\bar x$ and $n$, take expectations, and compare with the table. For which $\sigma^2$ does the Bayes predictive beat maximum likelihood?
"""
    ),
    answer(
        r"""
**1.** $Z_{n+1}=\int\varphi(w)\prod_{i=1}^{n+1}p(x_i\mid w)dw=\int p(x_{n+1}\mid w)\,\varphi(w)\prod_{i\le n}p(x_i\mid w)\,dw=Z_n\int p(x_{n+1}\mid w)\pi_n(w)dw=Z_np^*(x_{n+1})$. Take $-\log$. Averaging over $x_{n+1}\sim q$ and over the first $n$ samples gives $\mathbb E[-\log p^*(x_{n+1})]=\mathbb E[F_{n+1}]-\mathbb E[F_n]$, and subtracting $S$ gives the Bayes generalization error.

**2.** From §1.3, $F_n=nL_n(\hat w)+\tfrac d2\log n+\tfrac12\log\det H_n-\log\varphi(\hat w)-\tfrac d2\log2\pi+o(1)$. Expand $L_n$ about $w_0$ instead of $\hat w$: with $g_n=\nabla L_n(w_0)$ and $\hat w-w_0\approx-H^{-1}g_n$, $nL_n(\hat w)\approx nL_n(w_0)+ng_n^\top(\hat w-w_0)+\tfrac n2(\hat w-w_0)^\top H(\hat w-w_0)=nL_n(w_0)-\tfrac n2g_n^\top H^{-1}g_n$. Now $\mathbb E[nL_n(w_0)]=nL(w_0)$ exactly, and $\mathbb E[ng_n^\top H^{-1}g_n]=n\,\mathrm{tr}(H^{-1}\mathbb E[g_ng_n^\top])=\mathrm{tr}(H^{-1}C)$. The remaining terms converge to constants, giving $c$. Then $\mathbb E[F_{n+1}]-\mathbb E[F_n]=L(w_0)+\tfrac d2\log\frac{n+1}n+o(1)\text{-difference}=L(w_0)+\tfrac d{2n}+\dots$, so $\mathbb E[G_n^{\rm Bayes}]\approx L(w_0)-S+\tfrac d{2n}=L_{\min}+\tfrac d{2n}$. The step of faith is that the difference of the $o(1)$ remainders between $n$ and $n+1$ is $o(1/n)$; a term like $\mathbb E[\log\det H_n]$ does change by $O(1/n)$ between $n$ and $n+1$, so this needs the direct computation of part 3 (or Watanabe's theorems) to be trusted. Part 3 confirms the result.

**3.** *Fluctuation of the fit.* $L(\hat w)-L(w_0)\approx\tfrac12(\hat w-w_0)^\top H(\hat w-w_0)$, and with $\hat w-w_0\approx\mathcal N(0,H^{-1}CH^{-1}/n)$ its expectation is $\tfrac12\mathrm{tr}(H\cdot H^{-1}CH^{-1}/n)=\tfrac1{2n}\mathrm{tr}(H^{-1}C)$. This is already the **MLE row**: $\mathbb E[G^{\rm MLE}_n]=L(w_0)-S+\tfrac1{2n}\mathrm{tr}(H^{-1}C)$.

*Gibbs.* $\mathbb E_{\pi_n}[L(w)]\approx L(\hat w)+\tfrac12\mathbb E[\delta^\top H\delta]=L(\hat w)+\tfrac12\mathrm{tr}(H(nH)^{-1})=L(\hat w)+\tfrac d{2n}$. Adding the fluctuation of the fit gives the **Gibbs row**, $L_{\min}+\frac{d+\mathrm{tr}(H^{-1}C)}{2n}$.

*Bayes.* First average over the posterior, holding the training data fixed. Write $\log p(x\mid w)=-\ell(x,\hat w)+s^\top\delta+\tfrac12\delta^\top M\delta+\dots$ with $s=-\nabla\ell(x,\hat w)$ and $M=-\nabla^2\ell(x,\hat w)$. Then $\log p^*(x)=-\ell(x,\hat w)+\log\mathbb E_\delta[e^{Y}]$ with $Y=s^\top\delta+\tfrac12\delta^\top M\delta$, and
$$
\log\mathbb E[e^Y]\approx\mathbb E[Y]+\tfrac12\mathrm{Var}[Y]=\tfrac12\mathrm{tr}(M(nH)^{-1})+\tfrac12s^\top(nH)^{-1}s+O(n^{-2}),
\qquad\text{so}\qquad
-\log p^*(x)\approx\ell(x,\hat w)-\frac1{2n}\big[s^\top H^{-1}s+\mathrm{tr}(H^{-1}M)\big].
$$
Next average over a fresh observation $x\sim q$ (with $\hat w\approx w_0$ inside the $1/n$ terms): $\mathbb E_q[ss^\top]=C$ and $\mathbb E_q[M]=-H$, so
$$
\mathbb E_x[-\log p^*(x)]\approx L(\hat w)-\frac1{2n}\big[\mathrm{tr}(H^{-1}C)-d\big].
$$
Finally average over training sets using the fluctuation of the fit: $L(w_0)+\frac1{2n}\mathrm{tr}(H^{-1}C)-\frac1{2n}\mathrm{tr}(H^{-1}C)+\frac d{2n}$. The two terms involving $C$ cancel, and the **Bayes row** is $L_{\min}+\frac d{2n}$.

**4.** With a flat prior the posterior is $\mathcal N(\bar x,1/n)$ and the predictive is $\mathcal N(\bar x,1+1/n)$. For $x\sim\mathcal N(0,\sigma^2)$, $S=\tfrac12\log(2\pi e\sigma^2)$ and $L(w)-S=\tfrac12(\sigma^2+w^2)-\tfrac12-\tfrac12\log\sigma^2=\tfrac12(w^2+\sigma^2-1-\log\sigma^2)$, so $L_{\min}=\tfrac12(\sigma^2-1-\log\sigma^2)$, which is $\ge0$ with equality only at $\sigma^2=1$. Then, writing $v=1/n$ and $G$ for the error minus $L_{\min}$:
$$
G^{\rm MLE}=\tfrac12\bar x^2,\qquad
G^{\rm Gibbs}=\tfrac12\big(\bar x^2+v\big),\qquad
G^{\rm Bayes}=\tfrac12\log(1+v)+\frac{\sigma^2+\bar x^2}{2(1+v)}-\frac{\sigma^2}{2}.
$$
(The Bayes value is $\mathbb E_x[-\log\mathcal N(x;\bar x,1+v)]-S-L_{\min}$.) Taking expectations with $\mathbb E[\bar x^2]=\sigma^2/n$: $\mathbb E[G^{\rm MLE}]=\frac{\sigma^2}{2n}$, $\mathbb E[G^{\rm Gibbs}]=\frac{\sigma^2+1}{2n}$, and $\mathbb E[G^{\rm Bayes}]=\tfrac12\log(1+\tfrac1n)+\frac{\sigma^2(1+1/n)}{2(1+1/n)}-\frac{\sigma^2}2=\tfrac12\log(1+\tfrac1n)=\frac1{2n}+O(n^{-2})$. Here $d=1$, $H=1$ (the Hessian of $\tfrac12(x-w)^2$) and $C=\mathrm{Var}(x)=\sigma^2$, so $\mathrm{tr}(H^{-1}C)=\sigma^2$: the maximum-likelihood and Gibbs rows of the table are reproduced exactly, and the Bayes row to leading order. To leading order the Bayes predictive beats maximum likelihood when $\sigma^2>1$ (the data are noisier than the model believes, and averaging hedges) and loses when $\sigma^2<1$ (the model is too cautious already, and hedging makes it worse). Exactly, Bayes wins iff $\tfrac12\log(1+\tfrac1n)<\frac{\sigma^2}{2n}$, i.e. $\sigma^2>n\log(1+\tfrac1n)$, a threshold slightly below $1$ (0.89 at $n=4$); so at $\sigma^2=1$ the two agree only to leading order, with Bayes strictly better at finite $n$.
"""
    ),
    markdown(
        r"""
**Code: the three generalization errors in the misspecified example.** The cell
simulates Exercise 1.4.4 with a wide proper prior and plots $n$ times each
expected error against $n$, so the table's predictions are horizontal lines. Set
`sigma2` below 1 to see Bayes and maximum likelihood swap order.
"""
    ),
    code_answer(
        r"""
sigma2 = 2.0          # variance of the data; the model assumes 1
tau2 = 100.0          # prior variance (wide, so the prior barely matters)
ns = np.array([5, 10, 20, 50, 100, 200, 500])
trials = 4000
S = 0.5 * np.log(2 * np.pi * np.e * sigma2)
L_min = 0.5 * (sigma2 - 1 - np.log(sigma2))

def errors(xbar, n):
    # TODO: return (G_bayes, G_gibbs, G_mle) minus L_min, given the sample mean xbar
    # posterior: N(m, v) with v = 1 / (n + 1/tau2), m = v * n * xbar; predictive: N(m, 1 + v)
    raise NotImplementedError

# then: average over trials for each n and plot n * G against n, with the three predicted lines
""",
        r"""
sigma2 = 2.0          # variance of the data; the model assumes 1
tau2 = 100.0          # prior variance (wide, so the prior barely matters)
ns = np.array([5, 10, 20, 50, 100, 200, 500])
trials = 4000
S = 0.5 * np.log(2 * np.pi * np.e * sigma2)
L_min = 0.5 * (sigma2 - 1 - np.log(sigma2))

def errors(xbar, n):
    v = 1.0 / (n + 1.0 / tau2)
    m = v * n * xbar
    # expected loss of a Gaussian predictor N(m, s2) on x ~ N(0, sigma2): 1/2 log(2 pi s2) + (sigma2 + m^2) / (2 s2)
    bayes = 0.5 * np.log(2 * np.pi * (1 + v)) + (sigma2 + m**2) / (2 * (1 + v)) - S - L_min
    gibbs = 0.5 * np.log(2 * np.pi) + 0.5 * (sigma2 + m**2 + v) - S - L_min
    mle = 0.5 * np.log(2 * np.pi) + 0.5 * (sigma2 + xbar**2) - S - L_min
    return bayes, gibbs, mle

means = []
for n in ns:
    xbar = rng.normal(0.0, np.sqrt(sigma2 / n), size=trials)
    means.append([np.mean(g) for g in errors(xbar, n)])
means = np.array(means)

plt.figure(figsize=(5.5, 3.6))
for j, (lab, pred) in enumerate([("Bayes predictive", 0.5), ("Gibbs (one sample)", (1 + sigma2) / 2),
                                 ("maximum likelihood", sigma2 / 2)]):
    plt.plot(ns, ns * means[:, j], "o-", color=f"C{j}", label=lab)
    plt.axhline(pred, color=f"C{j}", ls=":")
plt.xscale("log"); plt.xlabel("n"); plt.ylabel(r"$n\,\mathbb{E}[G_n - L_{\min}]$")
plt.title(f"model N(w,1), data N(0,{sigma2}); dotted = table"); plt.legend(loc="center right"); plt.tight_layout(); plt.show()
""",
    ),
    # ------------------------------------------------------------------ readings
    markdown(
        r"""
<a id="slt1-readings"></a>

## Readings

The keystone is MacKay's *Bayesian Interpolation* (top of the page). Optional
background:
"""
        + reading(
            "Keeping neural networks simple by minimizing the description length of the weights",
            "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
            "Hinton & van Camp (COLT 1993)",
            "The same volume argument as the Occam factor, read as a code length: a posterior that is broad in some parameter directions costs fewer bits to transmit, which is the minimum-description-length side of §1.3.",
            importance=2,
            length=2,
        )
        + reading(
            "Deep Learning is Singular, and That's Good",
            "https://arxiv.org/abs/2010.11560",
            "Wei, Murfet, Gong, Li, Gell-Redman & Quella (2020)",
            "Why the regular-model results of this notebook (Laplace, BIC, $d/2n$) do not apply to neural networks, as an invitation to the rest of the day; read the introduction and Section 2 now, the rest after Notebook 2.",
            importance=3,
            length=3,
        )
        + references(
            book("Mathematical Theory of Bayesian Statistics", "https://doi.org/10.1201/9781315373010", "Watanabe (CRC Press, 2018)", "chapter 3 for the regular case in this notebook, chapters 5–6 for the general one"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt1", "slt1_bayes_statmech"))
