#!/usr/bin/env python3
"""Build SLT1, "Bayesian inference as statistical mechanics" (SLT day, NB1).

One linear path, four `##` sections (anchors slt1-1 ... slt1-4) plus a closing reading list
(slt1-readings), about 90 minutes: 1 the thermodynamic dictionary, formal, with one-line notes
where it breaks; 2 the posterior becomes Gaussian (CLT for the MLE, Bernstein-von Mises, the
two covariances); 3 Laplace, BIC and the Occam factor as a volume ratio; 4 the Bayes
generalisation error, and MLE vs Bayes vs Gibbs in the regular case, well-specified and
misspecified. Exercises are bold paragraphs ("Exercise 1.3a"), solutions folded.

Code: exact Z_n by quadrature on a 1-D toy (a coin, uniform prior, closed form) and a 2-D toy
(logistic regression, Gaussian prior), with BvM, Laplace and BIC overlays; a Monte Carlo check
of the regular-case generalisation errors on a Poisson model with over/under-dispersed data.

Spec: SLT_SPECS_v1.md (NB1); brief: WRITER_BRIEF.md. Formula checks: checks/slt1_bayes_*.py.
Review log: ../reviews/slt/slt1_bayes.md.
"""
from nbkit_slt import answer, book, code, markdown, reading, references, run  # noqa: F401

# ---------------------------------------------------------------- readings
MACKAY = reading(
    "Bayesian interpolation",
    "https://doi.org/10.1162/neco.1992.4.3.415",
    "David J. C. MacKay (Neural Computation, 1992)",
    "The evidence as best-fit likelihood times an \"Occam factor\", the ratio of posterior to prior "
    "accessible volume in parameter space. Read §§1–2 (pp. 415–424), which is §1.3 of this notebook in "
    "MacKay's words; the rest applies it to choosing regularisers for interpolation. An open-access copy "
    "is on [CaltechAUTHORS](https://resolver.caltech.edu/CaltechAUTHORS:MACnc92a).",
    importance=5, length=2)
HVC = reading(
    "Keeping neural networks simple by minimizing the description length of the weights",
    "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
    "Geoffrey E. Hinton and Drew van Camp (COLT 1993)",
    "Exercise 1.1b as a coding argument: sending noisy weights costs $\\mathrm{KL}(Q\\|P)$ after the "
    "\"bits back\" refund (§5), and the total code length is a free energy, minimised by the Boltzmann "
    "distribution (§7.1). The origin of variational Bayes for neural networks.",
    importance=2, length=2)
WATANABE22 = reading(
    "Recent advances in algebraic geometry and Bayesian statistics",
    "https://arxiv.org/abs/2211.10049",
    "Sumio Watanabe (2022)",
    "Watanabe's own survey. Read §2 for the definitions used in §1.4 and the identity "
    "$\\mathbb E\\,G_n=\\mathbb E F_{n+1}-\\mathbb E F_n-S$ (Watanabe's $G_n$ is our $G_n+S$), and §5.3 for "
    "$\\mathbb E\\,G_n=L(w_0)-S+\\lambda/n+o(1/n)$ under his assumptions, of which §1.4 is the case "
    "$\\lambda=d/2$. §§3–4 are the algebraic geometry, not needed today.",
    importance=3, length=2)
KVDV = reading(
    "The Bernstein–von Mises theorem under misspecification",
    "https://doi.org/10.1214/12-EJS675",
    "B. J. K. Kleijn and A. W. van der Vaart (Electronic Journal of Statistics, 2012)",
    "The theorem behind §1.2's misspecified case: the posterior is still asymptotically normal, "
    "and Bayesian credible sets are then not valid confidence sets. Read the introduction and the "
    "statement of the main theorem.",
    importance=2, length=2)
WAIC = reading(
    "Asymptotic equivalence of Bayes cross validation and widely applicable information criterion "
    "in singular learning theory",
    "https://jmlr.org/papers/v11/watanabe10a.html",
    "Sumio Watanabe (JMLR, 2010)",
    "States the tempered-posterior version of §1.4's result, "
    "$n\\,\\mathbb E[G_n-(L(w_0)-S)]\\to(\\lambda-\\nu)/\\beta+\\nu$ (Lemma 3), with "
    "$\\lambda=d/2$, $\\nu=\\tfrac12\\mathrm{tr}(H^{-1}C)$ for a regular, misspecified model (§§4.3 and 5). "
    "Read Remark 1 and Lemma 3; the rest is about cross-validation and WAIC.",
    importance=2, length=1)
BOOKS = references(
    book("Mathematical Theory of Bayesian Statistics", "https://doi.org/10.1201/9781315373010",
         "Sumio Watanabe (CRC Press, 2018)",
         "Ch. 4 (regular posterior distribution: free energy, generalisation and training losses, point estimators)"),
)

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# SLT1 — Bayesian inference as statistical mechanics

**Singular learning theory day, Iliad Intensive 2026**

What does Bayes do with $n$ samples, and how well does it then predict? The
posterior is a Boltzmann distribution, so these become questions about a
partition function. When the loss has a non-degenerate minimum, one Gaussian
integral answers all of them: the shape of the posterior, the evidence (BIC and
the Occam factor) and the generalisation error of Bayes against maximum
likelihood. NB2 asks what replaces that Gaussian integral when the Hessian is
degenerate.

**Keystone reading** (§§1–2 take about 20 minutes and can replace the prose of §1.3).
"""
        + MACKAY
        + r"""
**The path** (about 90 minutes). Exercises 1.1a, 1.3a, 1.4a and 1.4b are the
core; the others can wait.

- [1.1 The dictionary](#slt1-1) (15 min): prior, energy, Boltzmann distribution, partition function, free energy, and which of these words carry real content.
- [1.2 The posterior becomes Gaussian](#slt1-2) (15 min): the CLT for the MLE, Bernstein–von Mises, and why the two covariances differ under misspecification.
- [1.3 The evidence: Laplace, BIC and the Occam factor](#slt1-3) (25 min): $F_n\approx nL_n(\hat w)+\frac d2\log n$, and the evidence as a volume of near-optimal parameters.
- [1.4 How well does Bayes predict?](#slt1-4) (30 min): the Bayes generalisation error, and Bayes against MLE, well-specified and misspecified.

**Notation for the day** (each section restates what it uses).

| Symbol | Meaning |
|---|---|
| $w\in W\subseteq\mathbb R^d$, $\varphi(w)$ | parameter, prior density |
| $q(x)$, $p(x\mid w)$ | true distribution, model |
| $D_n=\{X_1,\dots,X_n\}$ | i.i.d. sample from $q$ |
| $L_n(w)=-\frac1n\sum_i\log p(X_i\mid w)$, $L(w)=\mathbb E_q[-\log p(X\mid w)]$ | empirical and population negative log-likelihood |
| $\hat w$, $w_0$ | minimiser of $L_n$ (the MLE), minimiser of $L$ |
| $S=\mathbb E_q[-\log q(X)]$ | entropy of $q$; $L(w_0)=S$ when the model contains $q$ |
| $K(w)=L(w)-L(w_0)$, $K_n(w)=L_n(w)-L_n(w_0)$ | population and empirical loss, shifted to vanish at $w_0$ |
| $\beta>0$ | inverse temperature, fixed as $n\to\infty$; $\beta=1$ is Bayes |
| $p_\beta(w\mid D_n)\propto\varphi(w)e^{-n\beta L_n(w)}$ | tempered posterior |
| $Z_n=\int\varphi\,e^{-n\beta L_n}dw$, $F_n=-\log Z_n$ | partition function (evidence at $\beta=1$), free energy |
| $H=\nabla^2L(w_0)$, $C=\mathrm{Cov}_q[\nabla\log p(X\mid w_0)]$ | Hessian, gradient covariance |

The next cell sets up the two toy models used in §§1.2–1.3: a coin (one
parameter, exact evidence) and a two-parameter logistic regression (evidence by
quadrature).
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, expit, log_expit, logsumexp
from scipy.optimize import minimize
from scipy.stats import binom, poisson, nbinom

rng = np.random.default_rng(1)

# ---- 1-D toy: a coin. theta in [0,1], uniform prior, n flips with k heads.
# Z_n = int_0^1 theta^k (1-theta)^(n-k) dtheta = k!(n-k)!/(n+1)!  (a Beta function), exactly.
theta0, N1 = 0.3, 3000
flips = (rng.random(N1) < theta0).astype(int)

def coin_F_exact(n):
    k = flips[:n].sum()
    return -(gammaln(k + 1) + gammaln(n - k + 1) - gammaln(n + 2))

# ---- 2-D toy: logistic regression, w = (bias, slope), prior N(0, tau^2 I), x ~ N(0, 1).
w_true, tau, N2 = np.array([0.5, 1.5]), 2.0, 3000
x2 = rng.normal(size=N2)
y2 = (rng.random(N2) < expit(w_true[0] + w_true[1] * x2)).astype(float)

def nll_sum(W, n):
    # n * L_n(w) for an array of parameters W[..., 2], on the first n data points
    z = W[..., 0, None] + W[..., 1, None] * x2[:n]
    return -(y2[:n] * log_expit(z) + (1 - y2[:n]) * log_expit(-z)).sum(-1)

def log_prior(W):
    return -0.5 * (W ** 2).sum(-1) / tau ** 2 - np.log(2 * np.pi * tau ** 2)

def logistic_mle(n):
    # w_hat, H_n = Hessian of L_n at w_hat, and n L_n(w_hat)
    r = minimize(lambda w: nll_sum(w, n) / n, np.zeros(2), method="BFGS", options={"gtol": 1e-10})
    w = r.x
    p = expit(w[0] + w[1] * x2[:n])
    X = np.c_[np.ones(n), x2[:n]]
    return w, (X * (p * (1 - p))[:, None]).T @ X / n, n * r.fun

def logistic_grid(n, G=101, half_width=10.0):
    # The exact posterior on a G x G grid centred at w_hat, spanning +-half_width posterior s.d.
    w, H, nLhat = logistic_mle(n)
    sd = np.sqrt(np.diag(np.linalg.inv(n * H)))
    a = np.linspace(w[0] - half_width * sd[0], w[0] + half_width * sd[0], G)
    b = np.linspace(w[1] - half_width * sd[1], w[1] + half_width * sd[1], G)
    A, B = np.meshgrid(a, b, indexing="ij")
    logw = log_prior(np.stack([A, B], -1)) - nll_sum(np.stack([A, B], -1), n)  # log(phi e^{-n L_n})
    m = logw.max()
    F = -(m + np.log(np.trapezoid(np.trapezoid(np.exp(logw - m), b, axis=1), a)))  # F_n = -log Z_n
    return dict(a=a, b=b, logw=logw, F=F, w=w, H=H, nLhat=nLhat)
"""
    ),
    # ================================================================ 1.1
    markdown(
        r"""
<a id="slt1-1"></a>

## 1.1 The dictionary

*Question:* written as a Boltzmann distribution, what is the posterior, and
which thermodynamic statements about it are exact?

*Notation.* Parameter $w\in W\subseteq\mathbb R^d$, prior density $\varphi(w)$,
model $p(x\mid w)$, data $D_n=\{X_1,\dots,X_n\}$ drawn i.i.d. from $q$, and the
empirical negative log-likelihood $L_n(w)=-\frac1n\sum_i\log p(X_i\mid w)$.

Bayes' rule reads

$$
p(w\mid D_n)=\frac{\varphi(w)\prod_{i=1}^n p(X_i\mid w)}{p(D_n)}=\frac{\varphi(w)\,e^{-nL_n(w)}}{Z_n},
\qquad Z_n=\int_W\varphi(w)\,e^{-nL_n(w)}\,dw .
$$

This is a Boltzmann distribution with energy $nL_n(w)$, at unit temperature,
over configurations weighted by $\varphi$. Add an inverse temperature $\beta$ as
an extra knob:

$$
p_\beta(w\mid D_n)=\frac{\varphi(w)\,e^{-n\beta L_n(w)}}{Z_n(\beta)},\qquad
Z_n(\beta)=\int_W\varphi\,e^{-n\beta L_n}\,dw,\qquad F_n(\beta)=-\log Z_n(\beta).
$$

At $\beta=1$ this is the Bayesian posterior and $Z_n=p(D_n)$ is the evidence
(marginal likelihood). Smaller $\beta$ flattens the posterior toward the prior;
as $\beta\to\infty$ it concentrates on the minimisers of $L_n$, i.e. on the
maximum-likelihood estimate. For a fixed sample, $n$ and $\beta$ enter only
through the product $n\beta$.

| Bayesian inference | Statistical mechanics | Note |
|---|---|---|
| parameter $w$ | configuration | $d$ stays fixed as $n$ grows |
| prior $\varphi(w)\,dw$ | density of states (strictly, the reference measure on configurations; see below) | chosen, not given |
| $nL_n(w)$ | energy of configuration $w$ | random: depends on the sample |
| $\beta$ | inverse temperature | $\beta=1$ is Bayes |
| $p_\beta(w\mid D_n)$ | Boltzmann distribution | exact |
| $Z_n(\beta)$ | partition function | exact; the evidence at $\beta=1$ |
| $F_n=-\log Z_n$ | free energy | physics writes $-\beta^{-1}\log Z$; we follow Watanabe and keep $-\log Z$ |
| $\mathbb E_\beta[nL_n]=\partial_\beta F_n$ | internal energy | exact (Exercise 1.1b) |
| $-\mathrm{KL}(p_\beta\,\|\,\varphi)$ | entropy, relative to $\varphi$ | exact: $F_n=\beta\,\mathbb E_\beta[nL_n]+\mathrm{KL}(p_\beta\|\varphi)$ (Exercise 1.1b) |
| posterior mass of a region $U$ | weight of a macrostate | exact: $Z_n(U)/Z_n$ (Exercise 1.1a) |

**Subtracting a reference energy.** Let $w_0$ minimise the population loss
$L(w)=\mathbb E_q[-\log p(X\mid w)]$ and set $K_n(w)=L_n(w)-L_n(w_0)$. Then

$$
F_n=n\beta L_n(w_0)-\log\int_W\varphi\,e^{-n\beta K_n}\,dw .
$$

The first term grows like $n$ and does not depend on the shape of the
posterior. Everything below is about the second. ($nL_n(w_0)$ is not quite the
ground-state energy: the minimum of $nL_n$ is at the MLE $\hat w$, and
$K_n(\hat w)\le0$.)

**Density of states, strictly.** In physics the density of states is a density
in *energy*. Here that is $v(t)=V'(t)$, with $V(t)$ the prior volume of
parameters whose population loss is within $t$ of its minimum; §1.3 meets $V$,
and NB2 is built on it.

**Where the dictionary is only formal.**

- *No thermodynamic limit.* The number of degrees of freedom is $d$, fixed. Large $n$ is a low-temperature limit of a fixed system. The posterior's switch between two regions does sharpen as $n$ grows, as in any finite system cooled toward zero temperature, but nothing here is a phase transition in the thermodynamic-limit sense.
- *$n$ is not a system size.* More data multiplies the energy; it adds no degrees of freedom. $n$ acts like an inverse temperature.
- *The energy is random.* Each sample gives a different landscape. Averages over samples, such as $\mathbb E\,F_n=\mathbb E[-\log Z_n]$, are what physicists call quenched averages over disorder; that part of the analogy does hold.
- *The prior is a choice.* No physical principle fixes it, and under a change of coordinates its density picks up a Jacobian (only $\varphi(w)\,dw$ is invariant).
- *$\beta=1$ is special.* Only there is $p_\beta$ the conditional distribution of $w$ given the data, and $Z_n$ a probability of the data.
"""
        + r"""
<a id="slt1-ex1-1a"></a>

**Exercise 1.1a — Free energy decides between regions.** *(Adapted from the Iliad
B.3 worksheet, Exercises 1.7–1.8.)* For a region $U\subseteq W$ let
$Z_n(U)=\int_U\varphi\,e^{-n\beta L_n}dw$ and $F_n(U)=-\log Z_n(U)$.

1. Show that the posterior mass of $U$ is $p_\beta(U\mid D_n)=Z_n(U)/Z_n$, so that for two regions $\log\frac{p_\beta(U\mid D_n)}{p_\beta(V\mid D_n)}=F_n(V)-F_n(U)$.
2. Let $W=U_1\sqcup\dots\sqcup U_k$ and suppose $F_n(U_1)$ is the smallest of the $F_n(U_j)$. Show that $F_n=-\log\sum_je^{-F_n(U_j)}$ and $F_n(U_1)-\log k\le F_n\le F_n(U_1)$.
3. Two regions have free energies differing by 5 nats. What are the posterior odds?
"""
    ),
    answer(
        r"""
1. $p_\beta(U\mid D_n)=\int_U\varphi\,e^{-n\beta L_n}dw\,/\,Z_n=Z_n(U)/Z_n$. Dividing two such masses cancels $Z_n$: $\log\frac{Z_n(U)}{Z_n(V)}=-F_n(U)+F_n(V)$.
2. The integral over $W$ splits over the disjoint pieces: $Z_n=\sum_jZ_n(U_j)=\sum_je^{-F_n(U_j)}$, and take $-\log$. Since every $Z_n(U_j)\le Z_n(U_1)$, we have $Z_n(U_1)\le Z_n\le k\,Z_n(U_1)$; taking $-\log$ reverses the inequalities: $F_n(U_1)-\log k\le F_n\le F_n(U_1)$. The total free energy is the smallest local one, up to at most $\log k$.
3. $e^5\approx150$ to one in favour of the region with the lower free energy. A few nats already decide where the posterior sits; this is why the rest of the day tracks free energies to $O(1)$ accuracy.
"""
    ),
    markdown(
        r"""
<a id="slt1-ex1-1b"></a>

**Exercise 1.1b — Energy, entropy and the variational principle.** *Notation as
above:* $F(\beta)=-\log Z_n(\beta)$, and $\mathbb E_\beta$, $\mathrm{Var}_\beta$ are
taken under $p_\beta(w\mid D_n)$.

1. Show that $F'(\beta)=\mathbb E_\beta[nL_n]$ and $F''(\beta)=-\mathrm{Var}_\beta[nL_n]\le0$. Using $F(0)=0$, conclude that $F(1)=\int_0^1\mathbb E_\beta[nL_n]\,d\beta$.
2. For any probability density $\rho$ on $W$ with $\mathrm{KL}(\rho\|\varphi)<\infty$, show that
$$
\beta\,\mathbb E_\rho[nL_n]+\mathrm{KL}(\rho\,\|\,\varphi)=F(\beta)+\mathrm{KL}(\rho\,\|\,p_\beta).
$$
Conclude that $F(\beta)=\min_\rho\{\beta\,\mathbb E_\rho[nL_n]+\mathrm{KL}(\rho\|\varphi)\}$, attained only at $\rho=p_\beta$. Which term is the energy and which the entropy?
3. What do you get if you restrict $\rho$ to a family, for example Gaussians?
"""
    ),
    answer(
        r"""
1. $Z'(\beta)=-\int nL_n\,\varphi\,e^{-n\beta L_n}dw$, so $F'=-Z'/Z=\mathbb E_\beta[nL_n]$. Differentiating again,
$$
F''(\beta)=\frac{d}{d\beta}\frac{\int nL_n\,\varphi e^{-n\beta L_n}}{Z}=-\mathbb E_\beta[(nL_n)^2]+\mathbb E_\beta[nL_n]^2=-\mathrm{Var}_\beta[nL_n].
$$
So $F$ is concave in $\beta$; $-F''$ is the analogue of the heat capacity (up to a factor $\beta^2$). At $\beta=0$, $Z=\int\varphi=1$, so $F(0)=0$ and $F(1)=\int_0^1F'(\beta)\,d\beta$. This identity, *thermodynamic integration*, is one standard way to compute evidences, and it is the starting point of WBIC in NB3.
2. $\log\frac{\rho}{p_\beta}=\log\frac{\rho}{\varphi}+n\beta L_n+\log Z=\log\frac{\rho}{\varphi}+n\beta L_n-F$. Take $\mathbb E_\rho$:
$$
\mathrm{KL}(\rho\|p_\beta)=\mathrm{KL}(\rho\|\varphi)+\beta\,\mathbb E_\rho[nL_n]-F(\beta).
$$
Rearranged, this is the identity. Since $\mathrm{KL}(\rho\|p_\beta)\ge0$ with equality only at $\rho=p_\beta$, the left side is at least $F(\beta)$, with equality only at the Boltzmann distribution. $\beta\,\mathbb E_\rho[nL_n]$ is the energy term; $-\mathrm{KL}(\rho\|\varphi)$ is the entropy of $\rho$ relative to the reference measure $\varphi$. In physics notation this is $\beta F_{\rm phys}=\beta U-S_{\rm thermo}$, the Gibbs variational principle, and it is exact.
3. Minimising over a restricted family gives an upper bound on $F(\beta)$, the *variational free energy*; minus it is the evidence lower bound (ELBO) of variational inference. Hinton and van Camp (reading below) derive this for Gaussian $\rho$ as a code length: the energy term is the cost of the data misfits, the KL term the cost of the weights after the bits-back refund.
"""
    ),
    markdown(HVC),
    # ================================================================ 1.2
    markdown(
        r"""
<a id="slt1-2"></a>

## 1.2 The posterior becomes Gaussian

*Question:* what does the posterior look like after many samples, and how does
its width compare with the sampling error of the maximum-likelihood estimate?

*Notation.* $L_n$, $L$ the empirical and population negative log-likelihoods;
$\hat w=\arg\min L_n$ (the MLE); $w_0=\arg\min L$, assumed unique and interior;
$H=\nabla^2L(w_0)$, assumed positive definite; $C=\mathrm{Cov}_q[\nabla_w\log p(X\mid w_0)]$,
the covariance of the per-sample gradient; $p_\beta$ the tempered posterior.

There are two Gaussians here, and they need not agree.

**The MLE (CLT).** $\hat w$ solves $\nabla L_n(\hat w)=0$. Expanding around $w_0$,
$0\approx\nabla L_n(w_0)+H(\hat w-w_0)$. The gradient $\nabla L_n(w_0)$ is an average
of $n$ i.i.d. zero-mean vectors with covariance $C$, so by the CLT

$$
\sqrt n\,(\hat w-w_0)\ \to\ \mathcal N\big(0,\;H^{-1}CH^{-1}\big).
$$

**The posterior (Bernstein–von Mises).** Expand the energy around $\hat w$:
$n\beta L_n(w)=n\beta L_n(\hat w)+\frac{n\beta}2(w-\hat w)^\top H_n(w-\hat w)+O(n|w-\hat w|^3)$,
with $H_n=\nabla^2L_n(\hat w)\to H$. The posterior lives at $|w-\hat w|\sim n^{-1/2}$,
where the cubic term is $O(n^{-1/2})$ and the prior is nearly constant. So

$$
p_\beta(w\mid D_n)\ \approx\ \mathcal N\big(\hat w,\;(n\beta H)^{-1}\big).
$$

The Bernstein–von Mises theorem makes this precise: the total-variation distance
between the posterior and this Gaussian goes to zero in probability. It needs
$w_0$ unique and interior, $H$ positive definite, $\varphi$ continuous and positive
at $w_0$, enough smoothness, and that the posterior concentrates near $w_0$: no
distant region may combine near-optimal loss with appreciable prior mass (automatic
when $W$ is compact). It holds whether or not the model contains $q$. **We call
this setting, a unique minimum with positive-definite Hessian, the regular case.**

**Comparing the two.** If the model is well specified, $q=p(\cdot\mid w_0)$, then
$C=H$ (Exercise 1.2a), the sandwich $H^{-1}CH^{-1}$ collapses to $H^{-1}$, and at
$\beta=1$ the posterior's spread equals the MLE's sampling spread. A 95% credible
region is then asymptotically a 95% confidence region. If the model is
misspecified, $C$ need not equal $H$: the posterior is still Gaussian, with
covariance $(nH)^{-1}$, but the estimator scatters with covariance
$H^{-1}CH^{-1}/n$, so credible regions can lose their frequentist coverage. A temperature can repair this
only when $C$ is proportional to $H$.

If $H$ has zero eigenvalues, the Gaussian has infinite width in those directions
and this section says nothing; that is NB2's case.
"""
        + r"""
### Code: exact posteriors against the Gaussian

The coin (top): $\theta\in[0,1]$ with a uniform prior, $k$ heads in $n$ flips, so
the exact posterior is $\mathrm{Beta}(k+1,n-k+1)$ and the Gaussian is
$\mathcal N(\hat\theta,\hat\theta(1-\hat\theta)/n)$ with $\hat\theta=k/n$. Logistic
regression (bottom): $w=(\text{bias},\text{slope})$, prior $\mathcal N(0,4I)$, the exact
posterior on a grid (filled contours) against $\mathcal N(\hat w,(nH_n)^{-1})$ (lines),
at the same probability levels. Both toys use one growing data set, so each $n$
extends the last.
"""
    ),
    code(
        r"""
from scipy.stats import beta as beta_dist, norm, multivariate_normal

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
th = np.linspace(1e-4, 1 - 1e-4, 4000)
print("coin: total-variation distance between posterior and Gaussian")
for j, n in enumerate([10, 40, 300]):
    k = flips[:n].sum(); t = k / n
    post, gauss = beta_dist.pdf(th, k + 1, n - k + 1), norm.pdf(th, t, np.sqrt(t * (1 - t) / n))
    ax[0, j].plot(th, post, "k", lw=2, label="exact posterior")
    ax[0, j].plot(th, gauss, "C1--", lw=2, label="Gaussian (BvM)")
    lo, hi = max(0, t - 5 * np.sqrt(t * (1 - t) / n)), min(1, t + 5 * np.sqrt(t * (1 - t) / n))
    ax[0, j].set_xlim(lo, hi); ax[0, j].set_title(f"coin, n = {n}, k = {k}"); ax[0, j].set_xlabel(r"$\theta$")
for n in [10, 40, 300, 3000]:
    k = flips[:n].sum(); t = k / n
    sd, th01 = np.sqrt(t * (1 - t) / n), np.linspace(0, 1, 200001)
    outside = norm.cdf(0, t, sd) + norm.sf(1, t, sd)          # Gaussian mass outside [0, 1]
    tv = 0.5 * np.trapezoid(np.abs(beta_dist.pdf(th01, k + 1, n - k + 1) - norm.pdf(th01, t, sd)), th01) + 0.5 * outside
    print(f"  n = {n:5d}:  TV = {tv:.3f}   (TV * sqrt(n) = {tv * np.sqrt(n):.2f})")
ax[0, 0].legend(fontsize=8)

def level_of_mass(p, masses):
    # density thresholds enclosing the given probability masses (p normalised on a uniform grid)
    s = np.sort(p.ravel())[::-1]; c = np.cumsum(s) / s.sum()
    return sorted(s[np.searchsorted(c, m)] for m in masses)

for j, n in enumerate([20, 80, 1000]):
    g = logistic_grid(n, G=121, half_width=4.5)
    A, B = np.meshgrid(g["a"], g["b"], indexing="ij")
    p = np.exp(g["logw"] - g["logw"].max())
    gauss = multivariate_normal(g["w"], np.linalg.inv(n * g["H"])).pdf(np.stack([A, B], -1))
    masses = [0.95, 0.68, 0.3]
    ax[1, j].contourf(A, B, p, levels=level_of_mass(p, masses) + [p.max()], cmap="Greys", alpha=0.6)
    ax[1, j].contour(A, B, gauss, levels=level_of_mass(gauss, masses), colors="C1", linestyles="--")
    ax[1, j].plot(*w_true, "r+", ms=10, label="true w")
    ax[1, j].set_title(f"logistic, n = {n}"); ax[1, j].set_xlabel("bias"); ax[1, j].set_ylabel("slope")
ax[1, 0].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
At $n=10$ the coin's posterior is visibly skewed; by $n=300$ the two curves are
hard to tell apart, and the total-variation distance falls roughly like
$n^{-1/2}$ (the posterior's skewness is the leading correction). The logistic
posterior at $n=20$ already looks elliptical near its centre, with its outer
contour visibly off the Gaussian's; by $n=1000$ they coincide. Note that at $n=20$
and $n=80$ the true parameter is outside the posterior's 68% region: the
posterior describes the uncertainty given this sample, and this sample happens to
have a shallow slope.

<a id="slt1-ex1-2a"></a>

**Exercise 1.2a — Two covariances.** *Notation as above. Part 1 extends the Iliad B.3 worksheet, Exercise 2.11(b).*

1. Show that if $q=p(\cdot\mid w_0)$ then $C=H$. Which half of the argument still holds under misspecification?
2. Model $p(x\mid w)=\mathcal N(x;w,1)$, data from $q=\mathcal N(0,\sigma^2)$, flat prior on $w\in\mathbb R$. Find $w_0$, $H$ and $C$. Find the exact posterior and the exact sampling distribution of $\hat w$.
3. What fraction of samples have $w_0$ inside the 95% credible interval, for $\sigma=2$? For $\sigma=1/2$? Which $\beta$ makes the tempered credible interval exact?

<details><summary>Hint for 1</summary>

Write $s(x)=\nabla_w\log p(x\mid w)$. Then $\nabla p=p\,s$ and $\nabla^2p=p\,(\nabla s+ss^\top)$, and both integrate to zero over $x$.
</details>
"""
    ),
    answer(
        r"""
1. With $s=\nabla_w\log p(x\mid w)$: differentiating $\int p\,dx=1$ once gives $\mathbb E_{p(\cdot\mid w)}[s]=0$, and twice gives $\mathbb E_{p(\cdot\mid w)}[\nabla s]+\mathbb E_{p(\cdot\mid w)}[ss^\top]=0$. At $w=w_0$ with $q=p(\cdot\mid w_0)$, $H=\mathbb E_q[-\nabla s]=\mathbb E_q[ss^\top]=C$ (a covariance, since the mean is $0$).

   Under misspecification, $\mathbb E_q[s(X,w_0)]=-\nabla L(w_0)=0$ still holds, because $w_0$ minimises $L$. So $C$ is still the covariance of the score. The second identity used $q=p(\cdot\mid w_0)$ and fails: $C\ne H$ in general.
2. $L(w)=\frac12\log2\pi+\frac12\mathbb E_q(X-w)^2=\frac12\log2\pi+\frac12(\sigma^2+w^2)$, so $w_0=0$ and $H=1$. The score is $s=x-w$, so $C=\mathrm{Var}_q X=\sigma^2$. The posterior is $\propto\exp(-\frac12\sum_i(X_i-w)^2)\propto\exp(-\frac n2(w-\bar X)^2)$: exactly $\mathcal N(\bar X,1/n)$. The MLE is $\hat w=\bar X\sim\mathcal N(0,\sigma^2/n)$, which is $H^{-1}CH^{-1}/n$.
3. The credible interval $\bar X\pm1.96/\sqrt n$ covers $0$ iff $|\bar X|\sqrt n\le1.96$. Since $\bar X\sqrt n/\sigma\sim\mathcal N(0,1)$, the coverage is $P(|Z|\le1.96/\sigma)$: $2\Phi(0.98)-1\approx0.67$ for $\sigma=2$, and $2\Phi(3.92)-1\approx0.9999$ for $\sigma=\frac12$. The model is overconfident when the data are noisier than it assumes, and underconfident when they are less noisy. The tempered posterior is $\mathcal N(\bar X,1/(n\beta))$, which matches the sampling spread for $\beta=1/\sigma^2$. In one dimension a temperature always fixes the width; in $d$ dimensions only if $C\propto H$.
"""
    ),
    # ================================================================ 1.3
    markdown(
        r"""
<a id="slt1-3"></a>

## 1.3 The evidence: Laplace, BIC and the Occam factor

*Question:* how does the free energy $F_n=-\log Z_n$ grow with $n$, and why is
the evidence a volume of near-optimal parameters?

*Notation.* $\beta=1$; the regular case of §1.2. $Z_n=\int\varphi\,e^{-nL_n}dw$ is
the evidence and $F_n=-\log Z_n$; $\hat w$ is the MLE and
$H_n=\nabla^2L_n(\hat w)$, positive definite; $d$ is the number of parameters.

**Laplace's formula.** Near $\hat w$ the integrand is $\varphi(\hat w)\,e^{-nL_n(\hat w)}$
times a Gaussian with covariance $(nH_n)^{-1}$ (§1.2). Integrating the Gaussian
gives (Exercise 1.3a)

$$
F_n=nL_n(\hat w)+\frac d2\log\frac{n}{2\pi}+\frac12\log\det H_n-\log\varphi(\hat w)+O_p(1/n).
$$

**BIC.** Only the first two terms grow with $n$. Keeping them gives the Bayesian
information criterion,

$$
F_n=nL_n(\hat w)+\frac d2\log n+O_p(1),
$$

a fit term and a complexity penalty of $\frac12\log n$ nats per parameter. BIC's
error does not shrink with $n$; what it gets right is the $\log n$ term, which
decides between models of different dimension that fit about equally well.

**The Occam factor.** Exponentiating Laplace's formula,

$$
Z_n\approx\underbrace{e^{-nL_n(\hat w)}}_{\text{best-fit likelihood}}\times
\underbrace{\varphi(\hat w)\,(2\pi)^{d/2}\det(nH_n)^{-1/2}}_{\text{Occam factor}} .
$$

If the prior is uniform on a region of volume $V_{\rm prior}$, then
$\varphi(\hat w)=1/V_{\rm prior}$ and the Occam factor is $V_{\rm post}/V_{\rm prior}$, where
$V_{\rm post}=(2\pi)^{d/2}\det(nH_n)^{-1/2}$ is the volume the posterior occupies. The
data shrink the plausible region by $\sqrt n$ in each well-determined direction,
which is where $\frac d2\log n$ comes from. This is MacKay's argument (keystone,
§2).

**The same thing as a volume of near-optimal parameters.** Replace $L_n$ by the
population loss and write $\bar Z_n=\int\varphi\,e^{-nK}dw$ with $K=L-L(w_0)$. Let
$V(t)=\int_{K<t}\varphi\,dw$ be the prior volume of parameters within $t$ of the
minimum loss. Exercise 1.3a shows that

$$
\bar Z_n\approx\Gamma\!\left(\tfrac d2+1\right)V(1/n),\qquad V(t)\propto t^{d/2}.
$$

The evidence is, up to a constant, the prior volume of the set where the loss is
within $1/n$ of its minimum, and $d/2$ is the exponent of that volume. NB2 §1
keeps this form and drops the Gaussian: the exponent of $V(t)$ is the learning
coefficient $\lambda$, and $\lambda\le d/2$, with equality in the regular case.

<a id="slt1-ex1-3a"></a>

**Exercise 1.3a — Laplace's formula and BIC.** Assume $\hat w$ is the unique
global minimiser of $L_n$, interior to $W$, with $H_n$ positive definite;
$\varphi$ smooth and positive at $\hat w$; and $L_n$ bounded away from $L_n(\hat w)$
outside any neighbourhood of $\hat w$ (for instance, $W$ compact).

1. Substitute $w=\hat w+u/\sqrt n$ and expand $nL_n$ and $\varphi$ in $u$. Show that $Z_n=e^{-nL_n(\hat w)}\varphi(\hat w)(2\pi/n)^{d/2}\det(H_n)^{-1/2}\,(1+O(1/n))$, and write down $F_n$.
2. Which terms of $F_n$ grow with $n$? What is the order of what BIC drops?
3. Take $\varphi$ uniform on a region of volume $V_{\rm prior}$ containing the posterior. Show that the Occam factor is $V_{\rm post}/V_{\rm prior}$.
4. *(Volume form.)* Let $K(w)=\frac12(w-w_0)^\top H(w-w_0)$ near $w_0$. Compute $V(t)$ for small $t$, then show that, to leading order in $1/n$, $\bar Z_n=\int_0^\infty e^{-nt}\,dV(t)$ equals Laplace's leading term and $\Gamma(\frac d2+1)\,V(1/n)$.

<details><summary>Hint for 4</summary>

The ellipsoid $\{u:u^\top Au<1\}$ has volume $\omega_d\det(A)^{-1/2}$, with $\omega_d=\pi^{d/2}/\Gamma(\frac d2+1)$ the volume of the unit ball. Integrate by parts: $\int_0^\infty e^{-nt}dV(t)=n\int_0^\infty e^{-nt}V(t)\,dt$.
</details>
"""
    ),
    answer(
        r"""
1. With $w=\hat w+u/\sqrt n$, $dw=n^{-d/2}du$ and, since $\nabla L_n(\hat w)=0$,
$$
nL_n(\hat w+u/\sqrt n)=nL_n(\hat w)+\tfrac12u^\top H_nu+\tfrac1{6\sqrt n}\,T(u,u,u)+O(1/n),\qquad
\varphi(\hat w+u/\sqrt n)=\varphi(\hat w)+\tfrac1{\sqrt n}\nabla\varphi(\hat w)\cdot u+O(1/n),
$$
with $T$ the third-derivative tensor of $L_n$ at $\hat w$. Expand $e^{-T/(6\sqrt n)}\approx1-T/(6\sqrt n)$. The $n^{-1/2}$ terms are odd in $u$ and integrate to zero against the Gaussian $e^{-u^\top H_nu/2}$, so the first correction is $O(1/n)$. Using $\int e^{-u^\top H_nu/2}du=(2\pi)^{d/2}\det(H_n)^{-1/2}$,
$$
Z_n=e^{-nL_n(\hat w)}\,\varphi(\hat w)\,n^{-d/2}(2\pi)^{d/2}\det(H_n)^{-1/2}\,(1+O(1/n)),
$$
$$
F_n=nL_n(\hat w)+\tfrac d2\log\tfrac n{2\pi}+\tfrac12\log\det H_n-\log\varphi(\hat w)+O(1/n).
$$
Regions away from $\hat w$ contribute at most $e^{-n\,\cdot\,\text{gap}}$, which is why the assumption that $L_n$ stays bounded away from its minimum there is needed.
2. $nL_n(\hat w)$ grows like $n$, and $\frac d2\log n$ like $\log n$. The rest, $-\frac d2\log2\pi+\frac12\log\det H_n-\log\varphi(\hat w)$, converges to the constant $-\frac d2\log2\pi+\frac12\log\det H-\log\varphi(w_0)$. BIC drops this $O(1)$ constant, so its error tends to a constant rather than to zero. In a comparison between two models it is still dominated by $n\,\Delta L$ and $\frac{\Delta d}2\log n$ for large $n$.
3. $\varphi(\hat w)=1/V_{\rm prior}$, so the Occam factor is $(2\pi)^{d/2}\det(nH_n)^{-1/2}/V_{\rm prior}=V_{\rm post}/V_{\rm prior}$. $V_{\rm post}$ is the volume of a box with sides $\sqrt{2\pi}$ times the posterior standard deviations along the eigenvectors of $H_n$: the integral of the Gaussian bump divided by its peak height. Its $-\log$ is $\frac d2\log n+O(1)$.
4. $\{K<t\}$ is the ellipsoid $\{u:u^\top\frac H{2t}u<1\}$, of volume $\omega_d\det(H/2t)^{-1/2}=\omega_d(2t)^{d/2}\det(H)^{-1/2}$. With $\varphi\approx\varphi(w_0)$ on it, $V(t)\approx\varphi(w_0)\,\omega_d\,(2t)^{d/2}\det(H)^{-1/2}$. Then
$$
\bar Z_n=n\int_0^\infty e^{-nt}V(t)\,dt\approx n\,\varphi(w_0)\,\omega_d\,2^{d/2}\det(H)^{-1/2}\,\Gamma(\tfrac d2+1)\,n^{-d/2-1}
=\varphi(w_0)\Big(\frac{2\pi}n\Big)^{d/2}\det(H)^{-1/2},
$$
using $\omega_d\Gamma(\frac d2+1)=\pi^{d/2}$. This is Laplace's leading term, and comparing with $V(1/n)=\varphi(w_0)\,\omega_d\,(2/n)^{d/2}\det(H)^{-1/2}$ gives $\bar Z_n\approx\Gamma(\frac d2+1)V(1/n)$. Only $t\lesssim1/n$ contributes, so the quadratic approximation to $K$ near $w_0$ is all that matters.
"""
    ),
    markdown(
        r"""
### Code: exact free energies against Laplace and BIC

Both toys from §1.2. The coin's $F_n$ is exact: with a uniform prior,
$Z_n=\int_0^1\theta^k(1-\theta)^{n-k}d\theta=k!\,(n-k)!/(n+1)!$. The logistic
regression's comes from quadrature on a $101\times101$ grid spanning $\pm10$ posterior
standard deviations. The left two panels show $F_n-nL_n(\hat w)$, the part that is
not the fit: BIC predicts $\frac d2\log n$, a straight line in $\log n$. The right
panel shows the errors of both approximations.
"""
    ),
    code(
        r"""
ns1 = np.unique(np.round(np.logspace(1, np.log10(N1), 20)).astype(int))
ns2 = np.unique(np.round(np.logspace(np.log10(20), np.log10(N2), 12)).astype(int))

coin = []
for n in ns1:
    k = flips[:n].sum(); t = k / n
    nLhat = -(k * np.log(t) + (n - k) * np.log1p(-t))
    lap = nLhat + 0.5 * np.log(n / (2 * np.pi)) + 0.5 * np.log(1 / (t * (1 - t)))   # log phi = 0
    coin.append((coin_F_exact(n) - nLhat, lap - nLhat, 0.5 * np.log(n)))
coin = np.array(coin)

logi = []
for n in ns2:
    g = logistic_grid(n)
    lap = g["nLhat"] + np.log(n / (2 * np.pi)) + 0.5 * np.log(np.linalg.det(g["H"])) - log_prior(g["w"])
    logi.append((g["F"] - g["nLhat"], lap - g["nLhat"], np.log(n)))
logi = np.array(logi)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, ns, r, name in [(ax[0], ns1, coin, "coin, d = 1"), (ax[1], ns2, logi, "logistic regression, d = 2")]:
    a.plot(ns, r[:, 0], "ko", ms=4, label="exact")
    a.plot(ns, r[:, 1], "C0-", label="Laplace")
    a.plot(ns, r[:, 2], "C1--", label=r"BIC: $\frac{d}{2}\log n$")
    a.set_xscale("log"); a.set_xlabel("n"); a.set_ylabel(r"$F_n - nL_n(\hat w)$"); a.set_title(name); a.legend(fontsize=8)
for ns, r, c, name in [(ns1, coin, "C2", "coin"), (ns2, logi, "C3", "logistic")]:
    ax[2].loglog(ns, np.abs(r[:, 1] - r[:, 0]), "-", color=c, label=f"{name}: |Laplace - exact|")
    ax[2].loglog(ns, np.abs(r[:, 2] - r[:, 0]), "--", color=c, label=f"{name}: |BIC - exact|")
ax[2].loglog(ns1, 0.5 / ns1, "k:", lw=1, label=r"$\propto 1/n$")
ax[2].set_xlabel("n"); ax[2].set_title("errors"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
print("coin:     n, exact, Laplace, BIC  (all minus n L_n(w_hat))")
for i in [0, len(ns1) // 2, -1]:
    print(f"  {ns1[i]:5d}  {coin[i, 0]:7.3f}  {coin[i, 1]:7.3f}  {coin[i, 2]:7.3f}")
print("logistic: n, exact, Laplace, BIC")
for i in [0, len(ns2) // 2, -1]:
    print(f"  {ns2[i]:5d}  {logi[i, 0]:7.3f}  {logi[i, 1]:7.3f}  {logi[i, 2]:7.3f}")
"""
    ),
    markdown(
        r"""
Laplace's error falls like $1/n$, and is about $0.05$ nats already at $n=10$.
(For the coin, Laplace's formula is what you get by putting Stirling's formula
into $k!\,(n-k)!/(n+1)!$; Stirling's formula is itself Laplace's method applied to
$\Gamma(m+1)=\int_0^\infty t^me^{-t}dt$.) BIC has the right slope, $d/2$, but for
large $n$ it sits above the exact curve by the constant it drops, which tends to
$-\big(\frac12\log\det H-\log\varphi(w_0)-\frac d2\log2\pi\big)$: about $0.14$ nats
for the coin and $0.4$ nats for the logistic regression.
"""
    ),
    # ================================================================ 1.4
    markdown(
        r"""
<a id="slt1-4"></a>

## 1.4 How well does Bayes predict?

*Question:* after $n$ samples, how much worse than the truth are Bayes's
predictions, and how does that compare with plugging in the maximum-likelihood
estimate?

*Notation.* True distribution $q$ with entropy $S=\mathbb E_q[-\log q(X)]$;
$L(w)=\mathbb E_q[-\log p(X\mid w)]$, minimised at $w_0$; $\hat w$ the MLE;
$p_\beta(w\mid D_n)$ the tempered posterior and $\mathbb E_\beta$ its average;
$H=\nabla^2L(w_0)$ positive definite (the regular case of §1.2),
$C=\mathrm{Cov}_q[\nabla\log p(X\mid w_0)]$; $F_n=-\log Z_n$. $\mathbb E$ with no
subscript averages over the sample $D_n$.

**Three ways to predict.** Each is scored by its expected log loss on a fresh
$X\sim q$, minus the entropy $S$. For the MLE and Bayes this is a KL divergence
from $q$; for Gibbs it is the posterior average of such divergences.

- *Plug-in MLE*, $p(x\mid\hat w)$: error $L(\hat w)-S$.
- *Bayes*, the posterior predictive $p(x\mid D_n)=\mathbb E_\beta[p(x\mid w)]$: the **Bayes generalisation error** $G_n=\mathbb E_X[-\log p(X\mid D_n)]-S=\mathrm{KL}(q\,\|\,p(\cdot\mid D_n))$.
- *Gibbs*, one parameter drawn from the posterior: error $\mathbb E_\beta[L(w)]-S$.

All three equal $(L(w_0)-S)+(\text{excess})$. The first term is
$\mathrm{KL}(q\,\|\,p(\cdot\mid w_0))$, zero if the model contains $q$; it is the
price of misspecification and does not shrink with $n$. The excess is $O(1/n)$.
Below, "expected excess" means the mean of its large-$n$ limiting distribution,
times $1/n$. (For a plug-in MLE the literal expectation can be infinite: a coin
that came up heads in every flip so far predicts tails with probability 0.)

**Bayes error is the increment of the free energy.** At $\beta=1$,
$Z_{n+1}/Z_n=p(X_{n+1}\mid D_n)$, so (Exercise 1.4a)

$$
\mathbb E\,G_n=\mathbb E\,F_{n+1}-\mathbb E\,F_n-S .
$$

Summing, $\mathbb E\,F_n-nS$ is the total excess log loss of predicting $X_1$, then
$X_2$ from $X_1$, and so on: the free energy is a cumulative prediction loss
(the coding view of MDL). BIC says $\mathbb E\,F_n=nL(w_0)+\frac d2\log n+O(1)$, and
differencing suggests

$$
\mathbb E\,G_n=\big(L(w_0)-S\big)+\frac{d}{2n}+o(1/n).
$$

That step is heuristic (an $O(1)$ remainder need not have $O(1/n)$ increments),
but the result is right: Exercise 1.4b derives it directly, and Watanabe proves
$\mathbb E\,G_n=L(w_0)-S+\lambda/n+o(1/n)$ under his assumptions, with $\lambda$ the
learning coefficient of NB2; here $\lambda=d/2$. The $\log n$ in the free energy
becomes the $1/n$ in the generalisation error.

**MLE against Bayes against Gibbs.** Expanding each predictor to second order
around $\hat w$ (Exercise 1.4b) gives, for the expected excess over $L(w_0)-S$,

| predictor | $n\times$ expected excess | well-specified ($C=H$) |
|---|---|---|
| plug-in MLE | $\frac12\mathrm{tr}(H^{-1}C)$ | $\frac d2$ |
| Bayes at $\beta$ | $\frac d{2\beta}+\big(1-\frac1\beta\big)\frac12\mathrm{tr}(H^{-1}C)$ | $\frac d2$ |
| Gibbs at $\beta$ | $\frac d{2\beta}+\frac12\mathrm{tr}(H^{-1}C)$ | $\frac d{2\beta}+\frac d2$ |

As $\beta\to\infty$ the Bayes and Gibbs rows both tend to the MLE row.

- **Well-specified.** MLE and Bayes tie at $d/2n$ to leading order. A single posterior sample costs twice that at $\beta=1$.
- **Misspecified, $\beta=1$.** Bayes stays at $d/2n$, whatever $C$ is. MLE is at $\mathrm{tr}(H^{-1}C)/2n$. Bayes wins when $\mathrm{tr}(H^{-1}C)>d$, i.e. when the estimate scatters more than the curvature suggests, and loses when $\mathrm{tr}(H^{-1}C)<d$. Neither dominates.
- **Why.** The posterior's width is set by $H$ (§1.2); the MLE's actual scatter by $H^{-1}CH^{-1}$. Averaging over the posterior costs $\frac d{2n\beta}$ in spread and gains $\frac{\mathrm{tr}(H^{-1}C)}{2n\beta}$ from mixing (the log of an average beats the average of logs). The net is positive or negative depending on which is larger.
"""
        + WATANABE22
        + r"""
<a id="slt1-ex1-4a"></a>

**Exercise 1.4a — Bayes error from the free energy.** $\beta=1$; expectations
$\mathbb E$ are over samples.

1. Show that $Z_{n+1}/Z_n=p(X_{n+1}\mid D_n)$, and hence $\mathbb E\,G_n=\mathbb E\,F_{n+1}-\mathbb E\,F_n-S$.
2. Show that $\mathbb E\,F_n-nS=\sum_{m=0}^{n-1}\mathbb E\,G_m$, where $G_0$ uses the prior predictive. Interpret.
"""
    ),
    answer(
        r"""
1. $Z_{n+1}=\int\varphi\prod_{i\le n+1}p(X_i\mid w)\,dw$, so
$$
\frac{Z_{n+1}}{Z_n}=\int p(X_{n+1}\mid w)\,\frac{\varphi(w)\prod_{i\le n}p(X_i\mid w)}{Z_n}\,dw=\mathbb E_{p(w\mid D_n)}[p(X_{n+1}\mid w)]=p(X_{n+1}\mid D_n).
$$
So $F_{n+1}-F_n=-\log p(X_{n+1}\mid D_n)$. $X_{n+1}$ is independent of $D_n$; averaging over it gives $\mathbb E_X[-\log p(X\mid D_n)]=G_n+S$, and averaging over $D_n$ gives the identity.
2. Telescope from $F_0=-\log\int\varphi=0$: $\mathbb E\,F_n=\sum_{m=0}^{n-1}(\mathbb E\,F_{m+1}-\mathbb E\,F_m)=\sum_{m<n}(\mathbb E\,G_m+S)$. $F_n=-\log p(D_n)=\sum_m-\log p(X_{m+1}\mid D_m)$ is the length, in nats, of the code that sends the data one point at a time, each with the current posterior predictive; $\mathbb E\,F_n-nS$ is its excess over the ideal code for $q$. With $\mathbb E\,G_m\approx(L(w_0)-S)+\frac d{2m}$, the sum is $n(L(w_0)-S)+\frac d2\log n+O(1)$ (the $O(1)$ from Laplace's formula, not from summing the $o(1/m)$ remainders): BIC again.
"""
    ),
    markdown(
        r"""
<a id="slt1-ex1-4b"></a>

**Exercise 1.4b — The regular-case errors.** *Notation as above.* Use the
Gaussian approximations of §1.2: $p_\beta(w\mid D_n)\approx\mathcal N(\hat w,\Sigma)$
with $\Sigma=(n\beta H)^{-1}$, and $\hat w-w_0\approx\mathcal N(0,H^{-1}CH^{-1}/n)$ over
samples. Write $\ell(x,w)=-\log p(x\mid w)$, and $g(x)$, $h(x)$ for its gradient and
Hessian in $w$ at $\hat w$. Keep terms of order $1/n$.

1. *MLE.* Show that $\mathbb E[L(\hat w)]-L(w_0)\approx\frac1{2n}\mathrm{tr}(H^{-1}C)$.
2. *Gibbs.* Show that $\mathbb E_\beta[L(w)]-L(w_0)\approx\frac12(\hat w-w_0)^\top H(\hat w-w_0)+\frac12\mathrm{tr}(H\Sigma)$, and average over samples.
3. *Bayes.* Expand $\ell(x,\hat w+u)$ to second order in $u\sim\mathcal N(0,\Sigma)$ and use $-\log\mathbb E\,e^{-Y}=\mathbb E\,Y-\frac12\mathrm{Var}\,Y+\dots$ to show
$$
-\log p(x\mid D_n)\approx\ell(x,\hat w)+\tfrac12\mathrm{tr}(h(x)\Sigma)-\tfrac12g(x)^\top\Sigma\,g(x).
$$
Average over $X\sim q$ and over samples.
4. Check the table above, its well-specified column, and the limit $\beta\to\infty$.

<details><summary>Hint for 3</summary>

With $Y=g^\top u+\frac12u^\top hu$: $\mathbb E\,Y=\frac12\mathrm{tr}(h\Sigma)$ and $\mathrm{Var}\,Y=g^\top\Sigma g+O(n^{-2})$. Over $X\sim q$, $\mathbb E_q[h(X)]\approx H$ and $\mathbb E_q[g(X)g(X)^\top]\approx C$, up to corrections that contribute $o(1/n)$.
</details>
"""
    ),
    answer(
        r"""
1. Expand $L$ around its minimum: $L(\hat w)-L(w_0)\approx\frac12(\hat w-w_0)^\top H(\hat w-w_0)$. Averaging, $\frac12\mathrm{tr}\big(H\cdot H^{-1}CH^{-1}/n\big)=\frac1{2n}\mathrm{tr}(H^{-1}C)$.
2. Write $w-w_0=(\hat w-w_0)+u$ with $u\sim\mathcal N(0,\Sigma)$ under the posterior. Then $L(w)-L(w_0)\approx\frac12(\hat w-w_0+u)^\top H(\hat w-w_0+u)$; the cross term averages to zero over $u$, leaving $\frac12(\hat w-w_0)^\top H(\hat w-w_0)+\frac12\mathrm{tr}(H\Sigma)$. Since $\mathrm{tr}(H\Sigma)=\mathrm{tr}(I_d)/(n\beta)=d/(n\beta)$, the sample average is $\frac1{2n}\mathrm{tr}(H^{-1}C)+\frac d{2n\beta}$.
3. $p(x\mid D_n)=\mathbb E_u\,e^{-\ell(x,\hat w+u)}=e^{-\ell(x,\hat w)}\,\mathbb E_u\,e^{-Y}$ with $Y=g^\top u+\frac12u^\top hu$. The cumulant expansion gives $-\log\mathbb E\,e^{-Y}=\mathbb E\,Y-\frac12\mathrm{Var}\,Y+O(n^{-3/2})=\frac12\mathrm{tr}(h\Sigma)-\frac12g^\top\Sigma g$. Average over $X\sim q$: $\mathbb E_q\,\ell(X,\hat w)=L(\hat w)$, $\mathbb E_q\,h\approx H$ and $\mathbb E_q\,gg^\top\approx C$ (at $\hat w$ rather than $w_0$, which changes things at higher order only). So
$$
\mathbb E_X[-\log p(X\mid D_n)]\approx L(\hat w)+\frac d{2n\beta}-\frac{\mathrm{tr}(H^{-1}C)}{2n\beta},
$$
and averaging $L(\hat w)$ with part 1, the Bayes excess is $\frac1n\big[\frac d{2\beta}+(1-\frac1\beta)\frac12\mathrm{tr}(H^{-1}C)\big]$. (This is the calculation for a posterior centred exactly at $\hat w$. The true posterior mean differs from $\hat w$ by $O(1/n)$, through the prior and the third derivative. That adds $g(x)^\top\mathbb E_\beta[w-\hat w]$ at fixed $x$, but $\mathbb E_q\,g(X)=\nabla L(\hat w)=O(n^{-1/2})$, so after averaging over $X$ it is $O(n^{-3/2})$.)
4. With $C=H$, $\mathrm{tr}(H^{-1}C)=d$: MLE $\frac d2$, Bayes $\frac d{2\beta}+\frac d2-\frac d{2\beta}=\frac d2$ for every $\beta$, Gibbs $\frac d{2\beta}+\frac d2$ (which is $d$ at $\beta=1$). As $\beta\to\infty$ the posterior collapses onto $\hat w$, and both the Bayes and the Gibbs rows tend to $\frac12\mathrm{tr}(H^{-1}C)$, the MLE row.
"""
    ),
    markdown(
        r"""
<a id="slt1-ex1-4c"></a>

**Exercise 1.4c — A misspecified example, exactly.** Model $p(x\mid w)=\mathcal N(x;w,1)$,
data from $q=\mathcal N(0,\sigma^2)$, flat prior, $\beta=1$. (Exercise 1.2a found $w_0=0$,
$H=1$, $C=\sigma^2$, posterior $\mathcal N(\bar X,1/n)$.)

1. Compute the misspecification gap $L(w_0)-S$.
2. Compute the expected excess of the MLE, Bayes and Gibbs predictors *exactly*, for every $n$. Compare with the table.
3. For which $\sigma$ does the MLE predict better than Bayes, at large $n$ and at finite $n$?
"""
    ),
    answer(
        r"""
1. $L(w_0)=\frac12\log2\pi+\frac{\sigma^2}2$ and $S=\frac12\log(2\pi e\sigma^2)$, so $L(w_0)-S=\frac12(\sigma^2-1-\log\sigma^2)$, which is zero only at $\sigma=1$.
2. Write $\mathbb E_q[-\log\mathcal N(X;m,s^2)]=\frac12\log2\pi s^2+\frac{\sigma^2+m^2}{2s^2}$, and $\mathbb E\,\bar X^2=\sigma^2/n$.
   - *MLE*, $m=\bar X$, $s^2=1$: excess $\frac12\mathbb E\,\bar X^2=\frac{\sigma^2}{2n}$.
   - *Gibbs*: $\mathbb E_\beta[L(w)]-L(w_0)=\frac12\mathbb E_\beta[w^2]=\frac12(\bar X^2+\frac1n)$, averaging to $\frac{1+\sigma^2}{2n}$.
   - *Bayes*: the predictive is $\mathcal N(\bar X,1+\frac1n)$. Its expected loss is $\frac12\log2\pi(1+\frac1n)+\frac{\sigma^2+\sigma^2/n}{2(1+1/n)}=\frac12\log2\pi+\frac{\sigma^2}2+\frac12\log(1+\frac1n)$, so the excess is $\frac12\log(1+\frac1n)=\frac1{2n}-\frac1{4n^2}+\dots$, for every $\sigma$.

   With $d=1$ and $\mathrm{tr}(H^{-1}C)=\sigma^2$, the table predicts $\frac{\sigma^2}{2n}$, $\frac1{2n}$ and $\frac{1+\sigma^2}{2n}$. MLE and Gibbs agree exactly; Bayes agrees to leading order.
3. Exactly, the MLE wins when $\frac{\sigma^2}{2n}<\frac12\log(1+\frac1n)$, i.e. $\sigma^2<n\log(1+\frac1n)$; the threshold rises to $1$ as $n\to\infty$ (it is $\log2\approx0.69$ at $n=1$). At large $n$: the MLE wins when $\sigma^2<1$, where the data are less noisy than the model assumes and the extra predictive variance $\frac1n$ that Bayes adds only hurts; Bayes wins when $\sigma^2>1$.
"""
    ),
    markdown(
        r"""
### Code: MLE, Bayes and Gibbs on a misspecified Poisson model

Counts modelled as Poisson with rate $e^w$, prior $w\sim\mathcal N(\log4+0.3,1)$ (off-centre, so the prior is not placed on the truth).
The truth has mean $4$ and dispersion $D=\mathrm{Var}/\text{mean}$: binomial
($D=\frac12$), Poisson ($D=1$, well-specified) or negative binomial ($D=2$). For
$\ell=e^w-xw+\log x!$, $H=e^{w_0}=4$ and $C=\mathrm{Var}\,X=4D$, so
$\mathrm{tr}(H^{-1}C)=D$, and the table predicts $n\times$excess of $D/2$ (MLE),
$\frac12$ (Bayes, $\beta=1$), $(1+D)/2$ (Gibbs, $\beta=1$) and $\frac1{2\beta}+(1-\frac1\beta)\frac D2$
(Bayes, $\beta=4$). *Predict each row before running.* The posterior is exact on a
grid, and only the sum of the counts enters it; each entry averages 4000 samples
of size $n=100$.
"""
    ),
    code(
        r"""
mu, n, R = 4.0, 100, 4000
xs = np.arange(61)                                   # count support (P(X > 60) is negligible)
wg = np.linspace(np.log(mu) - 1.5, np.log(mu) + 1.5, 401)
log_pois = xs[:, None] * wg[None, :] - np.exp(wg)[None, :] - gammaln(xs + 1)[:, None]   # log p(x | w)
log_phi = -0.5 * (wg - np.log(mu) - 0.3) ** 2
L_minus_L0 = lambda w: np.exp(w) - mu * w - (mu - mu * np.log(mu))   # L(w) - L(w0), using E X = mu

rows = []
for D, q in [(0.5, binom(8, 0.5)), (1.0, poisson(mu)), (2.0, nbinom(4, 0.5))]:
    qx = q.pmf(xs) / q.pmf(xs).sum()
    L0 = -(qx * (xs * np.log(mu) - mu - gammaln(xs + 1))).sum()       # L(w0), w0 = log mu
    sums = q.rvs(size=(R, n), random_state=rng).sum(1)
    xbar = sums / n
    res = {"MLE": L_minus_L0(np.log(xbar))}
    for beta in [1.0, 4.0]:
        lp = log_phi[None, :] + beta * (sums[:, None] * wg[None, :] - n * np.exp(wg)[None, :])
        post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))          # posterior weights on the grid
        pred = post @ np.exp(log_pois).T                                   # posterior predictive p(x | D_n)
        res[f"Bayes b={beta:g}"] = -(np.log(pred) @ qx) - L0
        if beta == 1.0:
            res["Gibbs b=1"] = post @ L_minus_L0(wg)
    theory = {"MLE": D / 2, "Bayes b=1": 0.5, "Bayes b=4": 1 / 8 + (1 - 1 / 4) * D / 2, "Gibbs b=1": (1 + D) / 2}
    rows.append((D, res, theory))

print(f"n x expected excess loss, n = {n}, {R} samples each (Monte Carlo +- 1 s.e.; theory in brackets)")
print(f"{'D':>4} " + "".join(f"{k:>22}" for k in ["MLE", "Bayes b=1", "Gibbs b=1", "Bayes b=4"]))
for D, res, theory in rows:
    print(f"{D:4.1f} " + "".join(
        f"{n * res[k].mean():9.3f} +- {n * res[k].std() / np.sqrt(R):.3f} [{theory[k]:.3f}]" for k in
        ["MLE", "Bayes b=1", "Gibbs b=1", "Bayes b=4"]))
"""
    ),
    markdown(
        r"""
The Bayes column stays at $\frac12$ while the MLE column moves with $D$: the two
cross at $D=1$, the well-specified case. Gibbs sits $\frac12$ above the MLE at
$\beta=1$, and raising $\beta$ to $4$ moves Bayes most of the way toward the MLE.
The same data sets are used for every column, so the comparisons within a row
are paired.
"""
    ),
    # ================================================================ readings
    markdown(
        r"""
<a id="slt1-readings"></a>

## Readings

**Claude-suggested extras.**
"""
        + KVDV
        + WAIC
        + BOOKS
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt1_bayes", "slt1_bayes"))
