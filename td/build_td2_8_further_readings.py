#!/usr/bin/env python3
"""Build td2_8_further_readings: a compact map of the Training Dynamics day, then
NB2 §8's further readings (dropout averaging, layer cushion, geometry-adaptive
denoising bases, critical periods and plasticity, fine-tuning and RL update
dynamics).

Shape (streamline brief, 2026-10-09): the top is a short pitch, a one-line-per-
notebook map of the day with three suggested routes, and a one-line route per
section. Each `##` section opens with the question and its answer in plain
words, then its lead paper(s) via reading(), an optional short core-idea
paragraph and a one-line "Also:" list. Short exercises and numerical checks sit
in one `### Optional: annotated derivations and walkthrough` per section (anchor
`<section>-walkthrough`); exercises are bold-labelled paragraphs, not headings.
§4 has no derivations and so no walkthrough. Section anchors (td28-dropout,
td28-cushion, td28-gahb, td28-plasticity, td28-rl) are stable; other notebooks
link to td28-gahb and td28-plasticity. The map links sibling notebooks only at
section-level anchors.

Source checks and critique rounds are logged in
../reviews/td/td2_8_further_readings.md; standalone checks are in
checks/td2_8_dropout_nwgm.py and checks/td2_8_bf16_rounding.py.

`nb(stem, anchor)` links a sibling notebook's page, `stem.html#anchor`; the kit
rewrites that to `stem.ipynb#anchor` in the runnable notebook.
"""
from nbkit_td import answer, code, markdown, reading, run


def nb(stem: str, anchor: str = "") -> str:
    """Markdown link to a sibling notebook's page: [`stem`](stem.html#anchor)."""
    return f"[`{stem}`]({stem}.html{'#' + anchor if anchor else ''})"


WALK = "### Optional: annotated derivations and walkthrough"

INDEX = rf"""
# Training Dynamics 2.8: further readings, and a map of the day

**Training Dynamics, Iliad Intensive 2026**

The day asks what gradient-based training actually does: which solutions it
reaches, how fast, and how the knobs (learning rate, batch size, schedule,
optimizer, width, normalization, weight decay) change that. Every notebook stands
alone; take whatever interests you.

**Map of the day.**

- {nb("td1_linear_networks")}: deep linear networks solved exactly: plateaus, depth, the landscape, lazy versus rich learning.
- {nb("td2_1_flows")}: gradient descent follows a modified flow (backward error analysis), and when to trust it; momentum.
- {nb("td2_2_edge_of_stability")}: sharpness rises to the stability limit and stays there; catapults, warmup, schedules.
- {nb("td2_3_sgd_sde")}: SGD as a stochastic differential equation; batch-size and learning-rate scaling rules.
- {nb("td2_4_optimizer_geometry")}: optimizers as steepest descent under a norm: natural gradient, Adam, Muon, muP.
- {nb("td2_5_normalization_weight_decay")}: normalization, weight decay and the effective step size; loss spikes, clipping.
- {nb("td2_6_curvature")}: the Hessian's Gauss–Newton part, bulk plus outliers, the tiny top subspace, when the basin is chosen.
- {nb("td2_7_learned_first")}: what is learned first: low-order statistics, higher cumulants, spectral bias.

**Suggested routes.** *Less background:* `td1_linear_networks`
[§1](td1_linear_networks.html#td1-s1), [§4](td1_linear_networks.html#td1-s4) and
[§5](td1_linear_networks.html#td1-s5), then `td2_1` → `td2_3`. *Practitioner:*
`td2_3` → `td2_4` → `td2_5`. *More background:* skim `td1`, then `td2_2` and
`td2_6`, where small-step theory breaks down.

**Out of scope:** whether SGD samples a Bayesian posterior (the Mysteries day's
[MDL1 §8](../notebooks/mdl1_statistical_learning.html#wb2-sgd-posterior), exercise
8.2), and infinite-width NTK theory (the Physics of Deep Learning day).

**This notebook:** five topics the day does not derive, each led by its paper.
Four have an optional walkthrough with short exercises and a numerical check; run
the code cells in order (seconds on a laptop CPU).

1. [Dropout as a geometric-mean ensemble](#td28-dropout): Baldi & Sadowski.
2. [Layer cushion, noise robustness and basin size](#td28-cushion): Arora, Ge, Neyshabur & Zhang.
3. [Geometry-adaptive harmonic bases](#td28-gahb): Kadkhodaie, Guth, Simoncelli & Mallat.
4. [Critical periods and loss of plasticity](#td28-plasticity): Dohare et al.; Achille, Rovere & Soatto.
5. [Fine-tuning and RL update dynamics](#td28-rl): Mukherjee et al.; Zhu et al.
"""


CELLS = [
    markdown(INDEX),
    code(
        r"""
import itertools

import numpy as np

rng = np.random.default_rng(0)
"""
    ),
    # ------------------------------------------------------------------ 1. dropout
    markdown(
        r"""
<a id="td28-dropout"></a>

## 1. Dropout as a geometric-mean ensemble

Dropout trains a randomly thinned subnetwork at each step; at test time it runs
the full network once, with each weight scaled by its keep probability. Why does
that one pass stand in for averaging all the subnetworks? For a logistic unit it
is *exactly* their normalized geometric mean, which is close to their arithmetic
mean.
"""
        + reading(
            "Understanding dropout",
            "https://proceedings.neurips.cc/paper_files/paper/2013/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html",
            "Baldi & Sadowski (NeurIPS 2013)",
            r"The exact geometric-mean identity for logistic units, bounds on its gap to the arithmetic mean, recursive approximations for deep sigmoidal networks, and, for a linear unit, dropout's adaptive weight-decay term (largest at $p=\tfrac12$).",
            "§3–4 are the averaging results.",
            importance=5, length=2,
        )
        + r"""
Also: [Srivastava et al., Dropout (JMLR 2014)](https://jmlr.org/papers/v15/srivastava14a.html),
the standard reference. Dropout as an adaptive $L^2$ penalty is derived in the
Mysteries day's [MDL1 §4](../notebooks/mdl1_statistical_learning.html#wb2-dropout),
exercise 4.4.

<a id="td28-dropout-walkthrough"></a>

"""
        + WALK
        + r"""

**Setup.** One logistic unit, $\sigma(x)=1/(1+e^{-x})$, with inputs $I_i$ and
weights $w_i$. Dropout gates each input with $\delta_i\sim\mathrm{Bernoulli}(p_i)$,
independently, so the input sum is $S=\sum_i\delta_iw_iI_i$ and the output is
$O=\sigma(S)$. Index the $2^n$ subnetworks by $k$, with probabilities $P_k$, sums
$S_k$ and outputs $O_k$. The arithmetic mean is $E=\sum_kP_kO_k$; the weighted
geometric means are $G=\prod_kO_k^{P_k}$ and $G'=\prod_k(1-O_k)^{P_k}$; the
normalized weighted geometric mean is $\mathrm{NWGM}=G/(G+G')$, a probability that
treats the output and its complement symmetrically. The test-time rule computes
$\sigma\big(\sum_ip_iw_iI_i\big)=\sigma(\mathbb E S)$.

**Exercise 1.1 — The averaging identity.** Show that $\mathrm{NWGM}=\sigma(\mathbb E S)$
exactly, for any probabilities $P_k$ (they need not come from independent gates).

<details><summary>Hint</summary>

Divide numerator and denominator by $G$, and use $(1-\sigma(x))/\sigma(x)=e^{-x}$.
</details>
"""
    ),
    answer(
        r"""
Divide through by $G$:

$$
\mathrm{NWGM}=\frac{1}{1+G'/G}=\frac{1}{1+\prod_k\big(\tfrac{1-O_k}{O_k}\big)^{P_k}}.
$$

For the logistic function $(1-\sigma(x))/\sigma(x)=e^{-x}$, so each factor is
$e^{-P_kS_k}$, and the product is $\exp\big(-\sum_kP_kS_k\big)=e^{-\mathbb ES}$.
Hence $\mathrm{NWGM}=1/(1+e^{-\mathbb ES})=\sigma(\mathbb ES)$. Only
$\sum_kP_k=1$ was used, so the identity holds for any distribution over
subnetworks. With independent Bernoulli gates, $\mathbb ES=\sum_ip_iw_iI_i$,
which is what the weight-scaled network computes. The same argument works for a
softmax output layer; the paper notes (citing its companion paper) that only
constant and logistic functions satisfy $\mathrm{NWGM}(f(S))=f(\mathbb ES)$.
"""
    ),
    markdown(
        r"""
**Exercise 1.2 — How far is the geometric mean from the arithmetic one?** Suppose
the spread of $S$ across subnetworks is small: $S=\mu+\varepsilon$ with
$\mu=\mathbb ES$, $\mathbb E\varepsilon=0$ and $\mathrm{Var}\,S=s^2$ small. Write
$V=\mathrm{Var}\,O$. To leading order in $s^2$, show that

$$
E-\mathrm{NWGM}\;\approx\;\tfrac12\sigma''(\mu)\,s^2\;\approx\;\frac{V\,(1-2E)}{2E(1-E)}.
$$

So the gap is second order in the spread, and to this order vanishes at $E=\tfrac12$.

<details><summary>Hint</summary>

Use Exercise 1.1 for the NWGM. Taylor-expand $\mathbb E\,\sigma(\mu+\varepsilon)$
to second order; for $V$, first order is enough. The derivatives are
$\sigma'=\sigma(1-\sigma)$ and $\sigma''=\sigma(1-\sigma)(1-2\sigma)$.
</details>
"""
    ),
    answer(
        r"""
By Exercise 1.1, $\mathrm{NWGM}=\sigma(\mu)$. Expanding,

$$
E=\mathbb E\,\sigma(\mu+\varepsilon)=\sigma(\mu)+\sigma'(\mu)\,\mathbb E\varepsilon+\tfrac12\sigma''(\mu)\,\mathbb E\varepsilon^2+O(\mathbb E|\varepsilon|^3)
=\sigma(\mu)+\tfrac12\sigma''(\mu)s^2+\dots,
$$

so $E-\mathrm{NWGM}\approx\tfrac12\sigma''(\mu)s^2$. For the variance,
$O\approx\sigma(\mu)+\sigma'(\mu)\varepsilon$ gives
$V\approx\sigma'(\mu)^2s^2=\sigma(\mu)^2(1-\sigma(\mu))^2s^2$. Eliminate $s^2$:

$$
E-\mathrm{NWGM}\approx\tfrac12\,\sigma(1-\sigma)(1-2\sigma)\cdot\frac{V}{\sigma^2(1-\sigma)^2}
=\frac{V(1-2\sigma)}{2\sigma(1-\sigma)},
$$

and replacing $\sigma(\mu)$ by $E$ changes this only at higher order.
"""
    ),
    markdown(
        r"""
**Numerical check.** Exact enumeration of all $2^{12}$ subnetworks of a 12-input
unit. The last column is the paper's own second-order estimate of the gap, its
eq. (20), $V(1-2E)/(1-2V)$; in this small-spread regime it is smaller than the
exact gap by about the factor $1/(2E(1-E))\ge2$. The paper's conclusion, that the
NWGM is close to $E$, is unaffected.
"""
    ),
    code(
        r"""
sigma = lambda s: 1 / (1 + np.exp(-s))
n = 12
masks = np.array(list(itertools.product([0, 1], repeat=n)))   # all 2**12 subnetworks


def means(w, I, p, bias=0.0):
    # Arithmetic mean E, variance V, NWGM and sigma(E[S]) over every dropout mask, exactly.
    P = np.prod(np.where(masks == 1, p, 1 - p), axis=1)        # probability of each mask
    S = masks @ (w * I) + bias                                 # input sum of each subnetwork
    O = sigma(S)
    E = P @ O
    V = P @ (O - E) ** 2
    G, Gc = np.exp(P @ np.log(O)), np.exp(P @ np.log(1 - O))
    return E, V, G / (G + Gc), sigma(P @ S)


# Exercise 1.1: the identity is exact, even with large weights and unequal keep probabilities.
E, V, nwgm, sig_ES = means(rng.normal(0, 1, n), rng.normal(0, 1, n), rng.uniform(0.3, 0.9, n))
print(f"NWGM = {nwgm:.12f}   sigma(E[S]) = {sig_ES:.12f}   arithmetic mean E = {E:.6f}\n")

# Exercise 1.2: small weights (small spread), with a bias to move E away from 1/2.
w, I, p = rng.normal(0, 0.1, n), rng.normal(0, 1, n), np.full(n, 0.5)
print(" bias      E      E - NWGM    V(1-2E)/(2E(1-E))   B-S eq.(20) V(1-2E)/(1-2V)")
for b in [0.0, 1.0, 2.0, 3.0]:
    E, V, nwgm, _ = means(w, I, p, bias=b)
    print(f"{b:5.1f}  {E:.4f}  {E - nwgm:+.3e}   {V*(1-2*E)/(2*E*(1-E)):+.3e}          {V*(1-2*E)/(1-2*V):+.3e}")
"""
    ),
    # ------------------------------------------------------------------ 2. cushion
    markdown(
        r"""
<a id="td28-cushion"></a>

## 2. Layer cushion, noise robustness and basin size

In what precise sense is a trained network robust to noise, and does training make
it so? Arora et al. measure how strongly noise injected at one layer is attenuated
by the layers above it, *on the training data*. Their key quantity, the layer
cushion, is small at random initialization and grows during training, most in the
higher layers.
"""
        + reading(
            "Stronger generalization bounds for deep nets via a compression approach",
            "https://arxiv.org/abs/1802.05296",
            "Arora, Ge, Neyshabur & Zhang (ICML 2018)",
            "Defines noise sensitivity, layer cushion and related noise-stability properties, uses them to compress trained networks and bound the compressed network's generalization error, and measures the cushion growing during training.",
            "§3 has the definitions; §6 the measurements.",
            importance=5, length=2,
        )
        + reading(
            "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data",
            "https://arxiv.org/abs/1703.11008",
            "Dziugaite & Roy (UAI 2017)",
            "The weight-noise version: optimize a PAC-Bayes bound over a Gaussian centred near the SGD solution, getting nonvacuous bounds on binary MNIST; the weight noise the training loss tolerates is a (parametrization-dependent) proxy for basin size.",
            importance=3, length=3,
        )
        + r"""
This is a quantitative replacement for "flat minima generalize", whose raw
flatness depends on how the network is parametrized
([`td2_4` §4.2](td2_4_optimizer_geometry.html#td24-2); the Mysteries day's
[MDL1 §4](../notebooks/mdl1_statistical_learning.html#wb2-flatness), under "Why
flatness could matter, and why sharpness needs a metric"). A network that
attenuates injected noise can be compressed a lot, since compression error acts
like injected noise, and the compressed network gets a generalization bound the
authors report as orders of magnitude better than parameter counting.
[`td2_6` §6.5](td2_6_curvature.html#td26-5) covers when the basin is chosen.

<a id="td28-cushion-walkthrough"></a>

"""
        + WALK
        + r"""

**Notation.** Layer $i$ computes $x^i=A_i\phi(x^{i-1})$, with weight matrix
$A_i$, elementwise activation $\phi$ (ReLU in the paper) and input $x^0$.
$\lVert A\rVert_F$ is the Frobenius norm, $\lVert A\rVert_2$ the spectral norm, and
$\eta\sim\mathcal N(0,I)$ a standard Gaussian noise vector (not a learning rate
here), so $\mathbb E[\eta\eta^\top]=I$.

**Definitions** (from the paper). The *noise sensitivity* of a map $M$ at input
$x$ is the expected squared noise-to-signal ratio of the output under isotropic
input noise scaled by $\lVert x\rVert$:

$$
\psi(M,x)=\mathbb E_{\eta\sim\mathcal N(0,I)}\frac{\lVert M(x+\eta\lVert x\rVert)-M(x)\rVert^2}{\lVert M(x)\rVert^2}.
$$

The *layer cushion* $\mu_i$ is the largest $\mu$ such that
$\mu\lVert A_i\rVert_F\lVert\phi(x^{i-1})\rVert\le\lVert A_i\phi(x^{i-1})\rVert$
for every training input. The *stable rank* of $A$ is
$\lVert A\rVert_F^2/\lVert A\rVert_2^2$.

**Exercise 2.1 — Cushion is inverse noise sensitivity.**

1. For a linear map $M$, show $\psi(M,x)=\lVert M\rVert_F^2\lVert x\rVert^2/\lVert Mx\rVert^2$. Conclude that $1/\mu_i^2$ is the largest noise sensitivity of $A_i$ over the training inputs, that $\psi$ is at least the stable rank, and so $\mu_i\le1/\sqrt{\text{stable rank}}$. When is the bound attained?
2. Take $x\sim\mathcal N(0,I_h)$ independent of $A\in\mathbb R^{h'\times h}$. Show $\mathbb E\big[\lVert Ax\rVert^2/\lVert x\rVert^2\big]=\lVert A\rVert_F^2/h$, so a typical input has cushion about $1/\sqrt h$. What must happen during training for the cushion to grow well above that?
"""
    ),
    answer(
        r"""
**1.** $M(x+\eta\lVert x\rVert)-Mx=\lVert x\rVert M\eta$, and
$\mathbb E\lVert M\eta\rVert^2=\mathbb E\,\mathrm{tr}(M\eta\eta^\top M^\top)=\mathrm{tr}(MM^\top)=\lVert M\rVert_F^2$.
So $\psi=\lVert M\rVert_F^2\lVert x\rVert^2/\lVert Mx\rVert^2$. The cushion
condition reads $\lVert A_i\phi\rVert^2/(\lVert A_i\rVert_F^2\lVert\phi\rVert^2)\ge\mu^2$
for every training input, that is $1/\psi(A_i,\phi)\ge\mu^2$, so the largest such
$\mu$ has $1/\mu_i^2=\max_x\psi(A_i,\phi(x^{i-1}))$ (the paper's Proposition 3.1
and the remark after its Definition 4). Since
$\lVert Mx\rVert\le\lVert M\rVert_2\lVert x\rVert$, $\psi\ge\lVert M\rVert_F^2/\lVert M\rVert_2^2$,
the stable rank, and hence $\mu_i\le\lVert A_i\rVert_2/\lVert A_i\rVert_F$. Equality
needs every training input to lie in the top right singular subspace of $A_i$. A
large cushion therefore needs both a spectrum dominated by a few singular values
(low stable rank) and inputs aligned with those directions.

**2.** $x/\lVert x\rVert$ is uniform on the sphere, so
$\mathbb E\big[(x/\lVert x\rVert)(x/\lVert x\rVert)^\top\big]=I/h$ and
$\mathbb E\big[\lVert Ax\rVert^2/\lVert x\rVert^2\big]=\mathrm{tr}(A^\top A)/h=\lVert A\rVert_F^2/h$.
The typical cushion is about $1/\sqrt h$; it concentrates there when $A$'s stable
rank is large, and for a very low-rank $A$ it fluctuates by order one around that
value. For the cushion to grow well above $1/\sqrt h$, training has to line up
each layer's inputs on the training data with that layer's large singular
directions: weights and the representations they act on must align. Noise spreads
over all directions, while aligned signal passes preferentially. The paper's
measurements (VGG-19 on CIFAR-10) show this growth relative to a random network.
"""
    ),
    markdown(
        r"""
**Numerical check** of both parts, on a $256\times256$ layer with singular values
$1/k$ and on a random Gaussian layer.
"""
    ),
    code(
        r"""
h = 256
cushion = lambda M, X: np.linalg.norm(M @ X, axis=0) / (np.linalg.norm(M) * np.linalg.norm(X, axis=0))

# 2.1(1): a layer with a decaying spectrum. Monte Carlo noise sensitivity against the formula,
# then the bound 1/sqrt(stable rank) and inputs that attain it (along the top right singular vector).
U, _ = np.linalg.qr(rng.normal(size=(h, h)))
Vt, _ = np.linalg.qr(rng.normal(size=(h, h)))
B = U @ np.diag(1 / np.arange(1, h + 1)) @ Vt.T
x = rng.normal(size=h)
eta = rng.normal(size=(h, 4000))
psi_mc = np.mean(np.linalg.norm(B @ (eta * np.linalg.norm(x)), axis=0) ** 2) / np.linalg.norm(B @ x) ** 2
print(f"noise sensitivity at a random x: Monte Carlo {psi_mc:.0f}, formula {np.linalg.norm(B)**2 * (x @ x) / np.linalg.norm(B @ x)**2:.0f}")
X = rng.normal(size=(h, 1000))
print(f"bound 1/sqrt(stable rank) = {np.linalg.norm(B, 2) / np.linalg.norm(B):.3f}; cushion with inputs along the top "
      f"singular vector {cushion(B, Vt[:, :1]).min():.3f}, in the top-5 subspace {cushion(B, Vt[:, :5] @ rng.normal(size=(5, 1000))).min():.3f}, "
      f"random inputs {cushion(B, X).min():.3f}")

# 2.1(2): a random layer (large stable rank) with random inputs: cushion about 1/sqrt(h).
A = rng.normal(0, 1 / np.sqrt(h), (h, h))
print(f"random layer, random inputs: min cushion {cushion(A, X).min():.3f}, mean {cushion(A, X).mean():.3f}, 1/sqrt(h) = {1/np.sqrt(h):.3f}")
"""
    ),
    # ------------------------------------------------------------------ 3. GAHB
    markdown(
        r"""
<a id="td28-gahb"></a>

## 3. Geometry-adaptive harmonic bases

What function does a trained image denoiser, the network inside a diffusion model,
compute once it has enough data to stop memorizing? Locally, it shrinks the noisy
image along an image-dependent basis of harmonics that oscillate along contours and
in smooth regions, and these bases appear even on data for which they are
suboptimal. It is an inductive-bias result like the spectral bias of
[`td2_7` §7.3](td2_7_learned_first.html#td27-3), read off from the network's
Jacobian.
"""
        + reading(
            "Generalization in diffusion models arises from geometry-adaptive harmonic representations",
            "https://arxiv.org/abs/2310.02557",
            "Kadkhodaie, Guth, Simoncelli & Mallat (ICLR 2024)",
            r"Denoisers trained on disjoint halves of a face dataset memorize when data are few and converge to nearly the same score when data are many (about $10^5$ images); the Jacobian-eigenvector analysis then shows near-optimal, geometry-adaptive shrinkage, which persists on data where it is suboptimal.",
            "§2 for the transition, §3 for the analysis.",
            importance=5, length=2,
        )
        + r"""
Also: [Mohan et al., bias-free denoising CNNs (ICLR 2020)](https://arxiv.org/abs/1906.05478)
(why the Jacobian reading works);
[Kamb & Ganguli, an analytic theory of creativity (2024)](https://arxiv.org/abs/2412.20292)
(locality and equivariance alone predict trained diffusion models' outputs);
[Belrose et al., statistics of increasing complexity (2024)](https://arxiv.org/abs/2402.04362)
(`td2_7`'s "low-order statistics first", measured on real networks).

<a id="td28-gahb-walkthrough"></a>

"""
        + WALK
        + r"""

**The optimal denoiser.** A clean image $x\in\mathbb R^d$ is observed as
$y=x+\sigma z$, with $z\sim\mathcal N(0,I)$ and noise level $\sigma$. Write
$p_\sigma$ for the density of $y$, and $\nabla f(y)$ for the Jacobian of a map $f$
at $y$. The MSE-optimal denoiser is
$f^\star(y)=\mathbb E[x\mid y]=y+\sigma^2\nabla\log p_\sigma(y)$ (the Miyasawa or
Tweedie identity, which is why a denoiser gives the score). Its Jacobian is
$\nabla f^\star(y)=\sigma^{-2}\,\mathrm{Cov}[x\mid y]$, symmetric and positive
semidefinite, and the optimal error is
$\mathrm{MSE}=\mathbb E\,\mathrm{tr}\,\mathrm{Cov}[x\mid y]=\sigma^2\,\mathbb E\,\mathrm{tr}\,\nabla f^\star(y)$.
Exercise 3.1 checks the last two for Gaussian data.

**What a bias-free network computes.** The paper's denoisers are ReLU networks
with no additive biases, so they are positively homogeneous and
$f(y)=\nabla f(y)\,y$ wherever $f$ is differentiable. Idealizing the Jacobian as
symmetric positive semidefinite (approximately true for trained networks), with
$\nabla f(y)=\sum_k\lambda_k(y)e_k(y)e_k(y)^\top$,

$$
f(y)=\sum_k\lambda_k(y)\,\langle y,e_k(y)\rangle\,e_k(y):
$$

shrinkage by factors $\lambda_k(y)$ along a basis $e_k(y)$, both of which can
depend on $y$. The exercise works out the case where they cannot.

**Exercise 3.1 — The Gaussian case: shrinkage in a fixed basis.** Let
$x\sim\mathcal N(0,\Sigma)$ in $\mathbb R^d$, with eigendecomposition
$\Sigma=\sum_k\lambda_kv_kv_k^\top$, observed as $y=x+\sigma z$. Here $\lambda_k$
are the data variances, not the shrinkage factors $\lambda_k(y)$.

1. Show $f^\star(y)=\Sigma(\Sigma+\sigma^2I)^{-1}y$. What are the basis and the shrinkage factors, and do they depend on $y$?
2. Check $\nabla f^\star=\sigma^{-2}\mathrm{Cov}[x\mid y]$ directly.
3. Show $\mathrm{MSE}=\sigma^2\sum_k\lambda_k/(\lambda_k+\sigma^2)$. Read $\mathrm{tr}\,\nabla f^\star$ as an effective number of dimensions kept, and compare the MSE with $d\sigma^2$, the error of returning $y$ unchanged.

<details><summary>Hint</summary>

For jointly Gaussian $(x,y)$, $\mathbb E[x\mid y]=\mathrm{Cov}(x,y)\,\mathrm{Cov}(y)^{-1}y$ and
$\mathrm{Cov}[x\mid y]=\mathrm{Cov}(x)-\mathrm{Cov}(x,y)\mathrm{Cov}(y)^{-1}\mathrm{Cov}(y,x)$.
Here $\mathrm{Cov}(x,y)=\Sigma$ and $\mathrm{Cov}(y)=\Sigma+\sigma^2I$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $\mathrm{Cov}(x,y)=\Sigma$ and $\mathrm{Cov}(y)=\Sigma+\sigma^2I$,
$f^\star(y)=\Sigma(\Sigma+\sigma^2I)^{-1}y$. In the eigenbasis of $\Sigma$,

$$
f^\star(y)=\sum_k\frac{\lambda_k}{\lambda_k+\sigma^2}\,\langle y,v_k\rangle\,v_k:
$$

shrinkage in the fixed basis $v_k$ (the data's principal components), with factors
$\lambda_k/(\lambda_k+\sigma^2)$ that keep high-variance directions and suppress
those below the noise level. Neither depends on $y$: the Gaussian denoiser is
linear. This is Wiener filtering; for stationary images with periodic boundaries
the $v_k$ are Fourier modes.

**2.** $\mathrm{Cov}[x\mid y]=\Sigma-\Sigma(\Sigma+\sigma^2I)^{-1}\Sigma
=\Sigma(\Sigma+\sigma^2I)^{-1}\big[(\Sigma+\sigma^2I)-\Sigma\big]
=\sigma^2\,\Sigma(\Sigma+\sigma^2I)^{-1}$ (all these matrices commute). Dividing by
$\sigma^2$ gives $\Sigma(\Sigma+\sigma^2I)^{-1}=\nabla f^\star$.

**3.** $\mathrm{MSE}=\mathrm{tr}\,\mathrm{Cov}[x\mid y]=\sigma^2\sum_k\lambda_k/(\lambda_k+\sigma^2)=\sigma^2\,\mathrm{tr}\,\nabla f^\star$.
Each factor is between 0 and 1, roughly 1 for $\lambda_k\gg\sigma^2$ and 0 for
$\lambda_k\ll\sigma^2$, so $\mathrm{tr}\,\nabla f^\star$ counts the directions
whose signal exceeds the noise, each costing about $\sigma^2$. Returning $y$ costs
$d\sigma^2$, and $\mathrm{MSE}/(d\sigma^2)=\mathrm{tr}\,\nabla f^\star/d$. A small
error therefore needs $\mathrm{tr}\,\nabla f^\star\ll d$: few effective dimensions
kept (the paper's eq. (10) argument).

The paper's point is the contrast: for the Gaussian the best basis is fixed, while
a trained network's basis $e_k(y)$ moves with the image, following its edges and
smooth regions.
"""
    ),
    markdown(
        r"""
**Numerical check** by Monte Carlo, on 64-dimensional Gaussian data with variances
$1/k^2$ and noise level $\sigma=0.1$.
"""
    ),
    code(
        r"""
d, s = 64, 0.1
lam = 1 / np.arange(1, d + 1) ** 2                       # a decaying prior spectrum
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
Sigma = Q @ np.diag(lam) @ Q.T
J = Sigma @ np.linalg.inv(Sigma + s**2 * np.eye(d))     # the optimal (linear) denoiser and its Jacobian

Xs = (Q * np.sqrt(lam)) @ rng.normal(size=(d, 100_000))   # clean samples
Ys = Xs + s * rng.normal(size=Xs.shape)                  # noisy observations
mse = np.mean(np.sum((J @ Ys - Xs) ** 2, axis=0))
print(f"Monte Carlo MSE {mse:.4f}   sigma^2 tr J {s**2 * np.trace(J):.4f}   "
      f"returning y: d sigma^2 = {d * s**2:.2f}   effective dimension tr J = {np.trace(J):.1f} of {d}")
print("shrinkage factors along the top 6 principal directions:",
      np.round(np.sort(np.linalg.eigvalsh(J))[::-1][:6], 3))
"""
    ),
    # ------------------------------------------------------------------ 4. plasticity
    markdown(
        r"""
<a id="td28-plasticity"></a>

## 4. Critical periods and loss of plasticity

Does it matter *when* a network sees something? Yes. A deficit early in training,
such as blurred images for the first epochs, lowers final accuracy in a way later
training does not repair (a *critical period*), and a network trained on a long
sequence of tasks gradually loses the ability to learn new ones (*loss of
plasticity*). Early training builds something later training cannot easily undo.
"""
        + reading(
            "Loss of plasticity in deep continual learning",
            "https://doi.org/10.1038/s41586-024-07711-7",
            "Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton (Nature 2024; preprint [arXiv:2306.13812](https://arxiv.org/abs/2306.13812))",
            "On continual ImageNet variants and in RL, standard networks steadily lose the ability to learn, across architectures and optimizers; \"continual backprop\", which keeps reinitializing a few little-used units, maintains plasticity.",
            importance=5, length=2,
        )
        + reading(
            "Critical learning periods in deep networks",
            "https://arxiv.org/abs/1711.08856",
            "Achille, Rovere & Soatto (ICLR 2019)",
            "Early blur leaves a permanent deficit, while deficits that keep low-level statistics (vertical flips) do not; the Fisher information of the weights rises fast early in training and then falls.",
            importance=4, length=3,
        )
        + r"""
Two hooks to this day: critical periods already appear in deep linear networks,
where the mode equations of [`td1_linear_networks` §1](td1_linear_networks.html#td1-s1)
make them tractable (Kleinman et al.), and plasticity loss is tied to changes in
loss-landscape curvature (Lyle et al.; [`td2_6`](td2_6_curvature.html)). The
Mysteries day lists critical periods among its strange facts in
[MDL4 §4](../notebooks/mdl4_mysteries.html#wb6-section-4).

Also: [Kleinman, Achille & Soatto, critical periods in deep linear networks (ICLR 2024)](https://arxiv.org/abs/2308.12221);
[Lyle et al., understanding plasticity (ICML 2023)](https://arxiv.org/abs/2303.01486)
(plasticity loss tracks curvature, often without saturated units);
[Ash & Adams, on warm-starting (NeurIPS 2020)](https://arxiv.org/abs/1910.08475)
(the everyday version, and "shrink and perturb").
"""
    ),
    # ------------------------------------------------------------------ 5. RL
    markdown(
        r"""
<a id="td28-rl"></a>

## 5. Fine-tuning and RL update dynamics

Post-training moves a pretrained model a small distance; in which directions?
Measured in weights, RL fine-tuning of language models changes few parameters (as
counted in bfloat16), yet each update matrix has full numerical rank, is dominated
by one singular direction, and avoids the pretrained matrices' "principal weights",
where supervised fine-tuning (SFT) does not.
"""
        + reading(
            "Reinforcement learning finetunes small subnetworks in large language models",
            "https://arxiv.org/abs/2505.11711",
            "Mukherjee, Yuan, Hakkani-Tür & Peng (NeurIPS 2025)",
            r"Across seven RL algorithms and ten models, RL changes only 5–30% of parameters (counting bfloat16 changes up to $10^{-5}$ as none); the changed set is consistent across seeds and algorithms, fine-tuning it alone reproduces the full result, and each matrix's update has nearly full numerical rank.",
            importance=4, length=2,
        )
        + reading(
            "The path not taken: RLVR provably learns off the principals",
            "https://arxiv.org/abs/2511.08567",
            "Zhu et al. (2025)",
            "RL with verifiable rewards (RLVR) lands less often than chance on the *principal weights* (the largest entries of a low-rank SVD reconstruction of each pretrained matrix, a proxy for high curvature) and barely changes the singular spectrum, while SFT does the opposite.",
            "The \"provably\" refers to bounds for small, KL-constrained steps.",
            importance=4, length=3,
        )
        + r"""
The question this day adds, from [`td2_6` §6.4](td2_6_curvature.html#td26-4):
how does the update sit relative to the curvature (the Hessian or Gauss–Newton
eigenbasis)? None of these papers measures that directly at language-model scale,
and Exercise 5.2 shows why a weight-matrix proxy for it needs care. The Mysteries
day's [MDL4 §11](../notebooks/mdl4_mysteries.html#wb6-section-11) reads
Mukherjee et al. and Shenfeld et al. as a pair, including what "unchanged" means
in bfloat16.

Also: [Cai et al., predictability of RL dynamics (2025)](https://arxiv.org/abs/2510.00553)
(the top singular component of each update recovers about 99% of the accuracy);
[Schulman and Thinking Machines Lab, LoRA without regret (2025)](https://thinkingmachines.ai/blog/lora/)
(rank-1 LoRA suffices for policy-gradient RL);
[Shenfeld, Pari & Agrawal, RL's Razor (2025)](https://arxiv.org/abs/2509.04259)
(forgetting tracks the KL divergence to the base model);
[Malladi et al., a kernel view of fine-tuning (ICML 2023)](https://arxiv.org/abs/2210.05643)
(prompt-based fine-tuning as kernel regression with the pretrained tangent kernel);
[CRW, some interesting papers on RLVR (blog, 2026)](https://charlesr-w.github.io/crw-blog/Some-Interesting-Papers-on-RLVR/)
(a short, informal survey of these papers and more).

<a id="td28-rl-walkthrough"></a>

"""
        + WALK
        + r"""

**Notation.** $\Delta W$ is the change in one weight matrix from fine-tuning. Its
*numerical rank* counts singular values above a small tolerance; its *stable rank*
is $\lVert\Delta W\rVert_F^2/\lVert\Delta W\rVert_2^2$ (Frobenius over spectral norm,
squared). Cai et al. keep only the top singular component of each update, rescaled
to the update's norm, and recover about 99% of the trained model's accuracy.

**Exercise 5.1 — Full rank and dominated by one direction.** Let
$\Delta W=uv^\top+\epsilon N$ with unit vectors $u,v\in\mathbb R^n$, a dense
Gaussian $N$ with entries of variance $1/n$, and small $\epsilon$. What are its
rank, numerical rank and stable rank? Which of these do Mukherjee et al. and Cai et
al. measure?
"""
    ),
    answer(
        r"""
Almost surely every singular value of $\Delta W$ is nonzero, so its rank is $n$,
and its numerical rank is also $n$ unless $\epsilon$ is tiny. But
$\lVert\epsilon N\rVert_F^2\approx\epsilon^2n$ and $\lVert\epsilon N\rVert_2\approx2\epsilon$,
so the top singular value is about 1 and the rest are at most about $2\epsilon$.
The stable rank is about $1+\epsilon^2n$, close to 1 when $\epsilon\ll1/\sqrt n$.
Mukherjee et al. report numerical rank (full); Cai et al. measure how much of the
accuracy the top singular direction alone recovers (about 99%). Both can be true
of the same update.
"""
    ),
    markdown(
        r"""
**Numerical check** of Exercise 5.1, plus the measurement issue behind the
sparsity numbers: a dense update much smaller than the bfloat16 spacing of the
weights is mostly invisible when both checkpoints are rounded to bfloat16.
Shenfeld et al. found that in their setting the sparsity came from bfloat16 and
vanished in float32, at the same performance.
"""
    ),
    code(
        r"""
u, v = rng.normal(size=512), rng.normal(size=512)
dW = np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v)) + 0.02 * rng.normal(size=(512, 512)) / np.sqrt(512)
sv = np.linalg.svd(dW, compute_uv=False)
print(f"rank-1 plus small dense noise: numerical rank {np.linalg.matrix_rank(dW)} of 512, "
      f"stable rank {np.sum(sv**2) / sv[0]**2:.2f}, top direction's share of ||dW||_F^2 {sv[0]**2 / np.sum(sv**2):.0%}\n")


def to_bf16(a):
    # Round finite float32 values to the nearest bfloat16 (ties to even); return as float32.
    b = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return (b.astype(np.uint32) << 16).view(np.float32)


W0 = rng.normal(0, 0.02, (512, 512)).astype(np.float32)
for scale in [1e-6, 1e-5, 1e-4]:
    W1 = W0 + scale * rng.normal(size=(512, 512)).astype(np.float32)   # a dense update: every entry moves
    print(f"dense update of size {scale:.0e}: entries unchanged after rounding both to bfloat16 "
          f"{np.mean(to_bf16(W0) == to_bf16(W1)):.0%}, in float32 {np.mean(W0 == W1):.0%}")
"""
    ),
    markdown(
        r"""
**Exercise 5.2 — Principal directions are not curvature directions.** Take a
single linear layer $f(x)=Wx$ with loss $L(W)=\tfrac12\mathbb E\lVert Wx-y\rVert^2$
and input second moment $\Sigma_x=\mathbb E[xx^\top]$.

1. Show that $L(W+\Delta W)=L(W)+\langle\nabla L,\Delta W\rangle+\tfrac12\mathrm{tr}(\Delta W\,\Sigma_x\Delta W^\top)$ exactly. What are the eigen-directions of this curvature, and their eigenvalues?
2. Let $W=\sum_i s_iu_iv_i^\top$ be the SVD. Give a case where every highest-curvature update is orthogonal (in the Frobenius inner product) to $W$'s top singular direction $u_1v_1^\top$. When are "along $W$'s principal directions" and "along high curvature" close to the same thing?

<details><summary>Hint</summary>

For part 2, $\langle ab^\top,u_1v_1^\top\rangle_F=(a\cdot u_1)(b\cdot v_1)$. Compare the
condition you need with the [layer cushion of §2](#td28-cushion).
</details>
"""
    ),
    answer(
        r"""
**1.** With residual $r=Wx-y$,
$\tfrac12\mathbb E\lVert r+\Delta Wx\rVert^2=L(W)+\mathbb E\langle r,\Delta Wx\rangle+\tfrac12\mathbb E\lVert\Delta Wx\rVert^2$,
and $\mathbb E\lVert\Delta Wx\rVert^2=\mathbb E\,\mathrm{tr}(\Delta Wxx^\top\Delta W^\top)=\mathrm{tr}(\Delta W\Sigma_x\Delta W^\top)$.
There are no higher terms because the loss is quadratic in $W$. The curvature acts
as $\Delta W\mapsto\Delta W\Sigma_x$ (on $\mathrm{vec}\,\Delta W$ it is
$\Sigma_x\otimes I$). Its eigen-directions are $\Delta W=ac_k^\top$ for any output
vector $a$ and eigenvector $c_k$ of $\Sigma_x$, with eigenvalue $\lambda_k(\Sigma_x)$:
the curvature of an update depends only on how its right factor meets the input
distribution, not on $W$.

**2.** The highest-curvature updates are $ac_1^\top$. Their overlap with
$u_1v_1^\top$ is $(a\cdot u_1)(c_1\cdot v_1)$, zero for every $a$ when the top
input direction $c_1$ is orthogonal to $W$'s top right singular vector $v_1$. For
example, $W=\mathrm{diag}(1,0.1)$ and $\Sigma_x=\mathrm{diag}(0.1,1)$: the
principal direction is $e_1e_1^\top$, every highest-curvature update is
$ae_2^\top$, and an update along $u_1v_1^\top$ has low curvature. The two notions
agree when $W$'s top right singular vectors line up with the top eigenvectors of
$\Sigma_x$. A large layer cushion (§2) suggests such alignment but does not
establish it: the cushion is a worst case over individual training inputs, while
$\Sigma_x$ weights them by frequency.

For a hidden layer the curvature also has a left factor. In the Kronecker-factored
approximation of [Martens and Grosse](https://arxiv.org/abs/1503.05671), the
layer's Fisher block is approximately $\mathbb E[aa^\top]\otimes\mathbb E[gg^\top]$,
with $a$ the layer's input and $g$ the gradient with respect to its
preactivations, computed with targets sampled from the model's own predictive
distribution (with training labels one gets the empirical Fisher, which can
differ). For squared error and softmax cross-entropy this Fisher equals the
Gauss–Newton matrix. So whether a weight matrix's principal directions are its
curvature directions is an empirical question about the trained network, and a
proxy like Zhu et al.'s principal weights has to be argued for rather than assumed.
"""
    ),
]

raise SystemExit(run(CELLS, "td2_8_further_readings", "td2_8_further_readings"))
