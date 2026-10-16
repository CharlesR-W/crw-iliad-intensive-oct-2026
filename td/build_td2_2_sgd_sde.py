#!/usr/bin/env python3
"""Build NB2 §2, "SGD as a stochastic differential equation" (Training Dynamics day).

Page shape (readings first, 2026-10-14): a short pitch and a one-line route, then one
`## Readings` section (anchor td22-readings) with every paper of the notebook, grouped by
section, then the derivations and walkthrough: seven `##` sections (anchors td22-1, td22-2,
td22-3, td22-cbs, td22-lsr, td22-6, td22-valid), each with its question and answer in plain
words and a short core-idea paragraph, then (from the `-walkthrough` anchor) the derivations,
exercises (bold labels that keep the old numbering, e.g. "Exercise 2.3.A", so in-text
references like 2.3.A.4 still read) with folded solutions, and the code checks. §2.7 is a
short list with no walkthrough.

Closed-form results are checked in checks/td2_2_sgd_sde_check.py; the review log is
../reviews/td/td2_2_sgd_sde.md.
"""
from nbkit_td import answer, code, code_answer, markdown, reading, run

LI_TAI_E = "https://arxiv.org/abs/1511.06251"
LMA = "https://arxiv.org/abs/2102.12470"
MCCANDLISH = "https://arxiv.org/abs/1812.06162"

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# NB2 §2 — SGD as a stochastic differential equation

**Training Dynamics, Iliad Intensive 2026**

A minibatch gradient is the full gradient plus zero-mean noise of covariance
$\Sigma/B$ ($\Sigma$ the per-example gradient covariance, $B$ the batch size). So
over many small steps of size $\eta$, SGD behaves like gradient flow plus Brownian
noise of strength $\eta/B$. That picture turns batch-size folklore into formulas.

**Route.** The [Readings](#td22-readings) come first: every paper of the notebook,
grouped by section. The sections after them are the derivations and walkthrough
(exercises with worked solutions, and code), for whoever wants them.

- [2.1 The noise and its scale](#td22-1): why only $\eta/B$ matters.
- [2.2 Stochastic modified equations](#td22-2): the SDE that SGD follows.
- [2.3 Fluctuations in a basin](#td22-3): the loss floor at a constant learning rate.
- [2.4 Critical batch size](#td22-cbs): past which extra examples stop helping.
- [2.5 The linear scaling rule and where it fails](#td22-lsr).
- [2.6 Adam and the square-root rule](#td22-6).
- [2.7 When is the SDE picture valid?](#td22-valid)

Code cells build on each other; run them top to bottom (seconds each on a CPU).
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
"""
    ),
    # ================================================================ readings
    markdown(
        r"""
<a id="td22-readings"></a>

## Readings

**The noise and reshuffling ([§2.1](#td22-1)).** Also the counterpoint in [§2.7](#td22-valid).
"""
        + reading(
            "On the origin of implicit regularization in stochastic gradient descent",
            "https://arxiv.org/abs/2101.12176",
            "Samuel L. Smith, Benoit Dherin, David Barrett, Soham De (ICLR 2021)",
            "Under random reshuffling the noise acts over an epoch only at second order in η, as an implicit "
            "regularizer (N−B)/(N−1)·(η/4B)·tr Σ.",
            "§4 is the counterpoint in §2.7.",
            importance=3,
            length=2,
        )
        + r"""
**Stochastic modified equations ([§2.2](#td22-2), [§2.3](#td22-3)).** The SDE itself, and its solvable quadratic examples.
"""
        + reading(
            "Stochastic modified equations and adaptive stochastic gradient algorithms",
            LI_TAI_E,
            "Qianxiao Li, Cheng Tai, Weinan E (ICML 2017)",
            "Introduces stochastic modified equations and proves that they approximate SGD to first and "
            "second order in η.",
            "Read §2–§3.",
            importance=4,
            length=3,
        )
        + r"""
**Critical batch size ([§2.4](#td22-cbs)).**
"""
        + reading(
            "An empirical model of large-batch training",
            MCCANDLISH,
            "Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team (2018)",
            "The critical batch size, and a cheap noise scale that predicts it on tasks from MNIST to Dota.",
            "",
            importance=4,
            length=3,
        )
        + r"""
**The linear scaling rule and its validity ([§2.5](#td22-lsr), [§2.7](#td22-valid)).**
"""
        + reading(
            "On the validity of modeling SGD with stochastic differential equations (SDEs)",
            LMA,
            "Zhiyuan Li, Sadhika Malladi, Sanjeev Arora (NeurIPS 2021)",
            "Tests whether SGD is at its SDE limit at practical learning rates, and when the linear scaling "
            "rule must fail.",
            "Read §2–§5 and Figures 2–4.",
            importance=5,
            length=3,
        )
        + reading(
            "Accurate, large minibatch SGD: training ImageNet in 1 hour",
            "https://arxiv.org/abs/1706.02677",
            "Priya Goyal et al. (2017)",
            "The empirical linear scaling rule with gradual warmup, holding up to batch size 8k for ResNet-50 on "
            "ImageNet and degrading beyond.",
            "§2 has the rule and its informal justification.",
            importance=2,
            length=1,
        )
        + r"""
**Adam ([§2.6](#td22-6)).**
"""
        + reading(
            "On the SDEs and scaling rules for adaptive gradient algorithms",
            "https://arxiv.org/abs/2205.10287",
            "Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, Sanjeev Arora (NeurIPS 2022)",
            "SDEs for RMSprop and Adam with one step equal to time η², and the square-root scaling rule, "
            "with proofs and checks on vision and language tasks.",
            "§4.1 (the linear-loss warm-up) is the whole idea in one page.",
            importance=3,
            length=2,
        )
        + r"""
**Also:**
[Li, Tai and E, JMLR 2019](https://jmlr.org/papers/v20/17-526.html) (the rigorous SME
theorems); [Higham 2001](https://doi.org/10.1137/S0036144500378302) (Euler–Maruyama and
weak versus strong convergence, with short programs);
[Şimşekli, Sagun and Gürbüzbalaban](https://arxiv.org/abs/1901.06053) (heavy-tailed
gradient noise, §2.7).
"""
    ),
    # ================================================================ 2.1
    markdown(
        r"""
<a id="td22-1"></a>

## 2.1 The noise and its scale

Write the minibatch gradient as $g_B=G+\xi$, with $G=\nabla L$ the full gradient
and $\xi$ zero-mean noise. For batches of $B$ examples drawn with replacement,
$\mathrm{Cov}\,\xi=\Sigma/B$, with $\Sigma$ the per-example gradient covariance. So
SGD with step $\eta$ feels the noise only through $\eta/B$.

**Paper:** Smith, Dherin, Barrett and De, *On the origin of implicit regularization in
SGD* (see [Readings](#td22-readings)): reshuffled SGD, the counterpoint in §2.7.

Hold $\theta$ roughly fixed and take $n$ steps. The drift adds up to $-n\eta G=-tG$,
with $t=n\eta$, and the noise is a random walk with covariance
$\eta^2n\,\Sigma/B=t\cdot\frac\eta B\Sigma$. As $\eta\to0$ at fixed $B$ the noise
vanishes and SGD becomes gradient flow; at finite $\eta$ its strength is set by
$\eta/B$ alone, often called the *temperature*. Drift length $n\eta\|G\|$ and noise
length $\eta\sqrt{n\,\mathrm{tr}\,\Sigma/B}$ are equal after $B_{\rm simple}/B$ steps,
where $B_{\rm simple}=\mathrm{tr}\,\Sigma/\|G\|^2$ is McCandlish et al.'s *simple noise
scale*. For $B\ll B_{\rm simple}$ noise shapes the dynamics (§2.4).

<a id="td22-1-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* $N$ examples with losses $\ell_i$, $L=\frac1N\sum_i\ell_i$, $G=\nabla L$,
$\Sigma=\frac1N\sum_i(\nabla\ell_i-G)(\nabla\ell_i-G)^\top$, and $g_B=G+\xi$ the mean of
$B$ per-example gradients. The noise $\xi$ depends on how batches are drawn:

- **With replacement**, or a data stream (as in single-epoch LLM pretraining): $\mathrm{Cov}\,\xi=\Sigma/B$, fresh at every step. The SDE theorems assume this.
- **Without replacement within a batch**: $\mathrm{Cov}\,\xi=\frac{N-B}{N-1}\,\frac\Sigma B$, which matters only when $B$ is a sizable fraction of $N$.
- **Random reshuffling** (shuffle each epoch, cut into $m=N/B$ batches; most multi-epoch training): at fixed $\theta$ the $m$ batch gradients of an epoch sum to exactly $mG$, so the noise cancels over the epoch.

**Exercise 2.1.A — Sampling schemes.**

1. Show that $\mathrm{Cov}\,g_B=\Sigma/B$ for sampling with replacement and $\frac{N-B}{N-1}\frac\Sigma B$ for one batch drawn without replacement.
2. *Reshuffling.* Fix $\theta$. In one shuffled epoch, let $Z_n=\sum_{j<n}\xi_j$ be the accumulated noise of the first $n$ batches. Show that $\mathbb E\,Z_n=0$ and
$$
\mathrm{Cov}\,Z_n=\frac nB\,\frac{N-nB}{N-1}\,\Sigma ,
$$
which vanishes at the end of the epoch ($n=m$). Compare with $n\Sigma/B$ for sampling with replacement. When should the epoch structure matter for the fluctuations of $\theta$?

<details><summary>Hint</summary>

Write $u_i=\nabla\ell_i-G$, so $\sum_iu_i=0$. For two *distinct* indices drawn
without replacement, $\mathbb E\,u_iu_j^\top=\frac1{N(N-1)}\sum_{i\ne j}u_iu_j^\top$;
expand $(\sum_iu_i)(\sum_ju_j)^\top=0$. For 2, the first $n$ batches of an epoch
are a uniformly random set of $nB$ distinct examples. For the last question, use
that SGD on a quadratic mode of curvature $h$ forgets its past at rate $\eta h$
per step, since $\theta_{k+1}=(1-\eta h)\theta_k+\dots$ (§2.3 makes this precise).
</details>
"""
    ),
    answer(
        r"""
**1.** With replacement, $g_B=\frac1B\sum_{b=1}^B\nabla\ell_{i_b}$ with i.i.d. indices, each term having covariance $\Sigma$, so $\mathrm{Cov}\,g_B=\Sigma/B$.

Without replacement, $Bg_B-BG=\sum_{b}u_{i_b}$ over $B$ distinct indices. Its covariance is $B$ diagonal terms plus $B(B-1)$ cross terms:

$$
\mathrm{Cov}\Big(\sum_bu_{i_b}\Big)=B\,\Sigma+B(B-1)\,\mathbb E[u_iu_j^\top]_{i\ne j}.
$$

Since $\sum_iu_i=0$, $\sum_{i\ne j}u_iu_j^\top=-\sum_iu_iu_i^\top=-N\Sigma$, so $\mathbb E[u_iu_j^\top]_{i\ne j}=-\Sigma/(N-1)$. Then

$$
\mathrm{Cov}\Big(\sum_bu_{i_b}\Big)=B\Sigma\Big(1-\frac{B-1}{N-1}\Big)=B\,\frac{N-B}{N-1}\,\Sigma ,
$$

and dividing by $B^2$ gives $\frac{N-B}{N-1}\frac\Sigma B$, which vanishes at $B=N$.

**2.** $Z_n=\frac1B\sum u_i$ over the $nB$ distinct examples in the first $n$ batches. By part 1 with $nB$ in place of $B$, the covariance of that sum is $nB\frac{N-nB}{N-1}\Sigma$; dividing by $B^2$,

$$
\mathrm{Cov}\,Z_n=\frac nB\,\frac{N-nB}{N-1}\,\Sigma\approx\frac nB\Big(1-\frac nm\Big)\Sigma .
$$

This is a discrete Brownian bridge, pinned to zero at $n=m$; with replacement it keeps growing as $n\Sigma/B$.

A mode of curvature $h$ forgets its past after about $1/(\eta h)$ steps. If $m\,\eta h\gg1$ it equilibrates many times per epoch and sees noise locally like sampling with replacement. If $m\,\eta h\ll1$ it integrates the whole epoch, and the noise largely cancels. So reshuffled SGD fluctuates less than the SDE predicts in flat directions, at small learning rates and in short epochs (measured in the last cell of Code 2.3.B). Smith et al. make the related point for the mean iterate ([`td2_1_flows`, §1.4](td2_1_flows.html#td21-s4)).
"""
    ),
    # ================================================================ 2.2
    markdown(
        r"""
<a id="td22-2"></a>

## 2.2 Stochastic modified equations

Which stochastic differential equation (SDE) does SGD follow? To first order in
the step $\eta$, the *order-1 stochastic modified equation* (SME)

$$
d\theta=-\nabla L\,dt+\sqrt{\eta/B}\;\Sigma(\theta)^{1/2}\,dW ,
$$

with $L$ the loss, $\Sigma$ the per-example gradient covariance, $B$ the batch
size, $W$ a Brownian motion, and time $t=k\eta$ after $k$ steps. To second order
the drift becomes $-\nabla\big(L+\frac\eta4\|\nabla L\|^2\big)$, the same correction
that full-batch gradient descent has. "Follows" means in law, not path by path.

**Paper:** [Li, Tai and E](https://arxiv.org/abs/1511.06251) (see [Readings](#td22-readings)), §2–§3. The rigorous version, with the moment-matching theorem, is [Li, Tai and E, JMLR 2019](https://jmlr.org/papers/v20/17-526.html).

Why this SDE: one SGD step is $-\eta G-\eta\xi$, with $G=\nabla L$ and
$\mathrm{Cov}\,\xi=\Sigma/B$. One Euler–Maruyama step of size $\eta$ for
$d\theta=-G\,dt+\sigma\,dW$ is $-\eta G+\sigma\sqrt\eta\,\zeta$, with $\zeta$ standard
normal. Matching covariances gives $\sigma\sigma^\top=\frac\eta B\Sigma$. Li, Tai and E
prove a *weak* approximation: expectations of smooth functions of $\theta$ agree to
$O(\eta)$ on a fixed time horizon ($O(\eta^2)$ for the order-2 SME).

<a id="td22-2-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Hessian $H$; otherwise as above. One SGD step is $-\eta(G+\xi)$ with
zero-mean $\xi$ of covariance $\Sigma/B$ (§2.1).

> **Two facts about SDEs** ([Higham 2001](https://doi.org/10.1137/S0036144500378302) is a friendly introduction).
> 1. **Wiener process.** $W_t\in\mathbb R^d$ has $W_0=0$, continuous paths, and independent increments $W_{t+\Delta t}-W_t\sim\mathcal N(0,\Delta t\,I)$, of size $\sqrt{\Delta t}$. That is why noise with per-step variance $O(\eta^2)$ adds up to an $O(\eta)$ effect per unit time, the same order as the $O(\eta)$ drift corrections.
> 2. **Euler–Maruyama.** $dX=b(X)\,dt+\sigma(X)\,dW$ means the $\Delta t\to0$ limit of $X_{k+1}=X_k+b(X_k)\,\Delta t+\sigma(X_k)\sqrt{\Delta t}\,\zeta_k$, $\zeta_k\sim\mathcal N(0,I)$ i.i.d. (Itô: coefficients at the start of each step). Only $D=\sigma\sigma^\top$ affects the law of $X$.

The order-1 SME depends on $\eta$ (its noise vanishes as $\eta\to0$), so "SGD is
close to it" is a statement about a family of SDEs, and we compare *laws*. SGD is
approximated in the **weak sense to order $p$** if, for smooth test functions $g$
and a fixed horizon $t_{\max}$, $|\mathbb E\,g(\theta_k)-\mathbb E\,g(X_{k\eta})|\le C\eta^p$
for all $k\le t_{\max}/\eta$, with $C$ independent of $\eta$ (and not estimated). The
tool is **moment matching**: if one-step increments of SGD and the SDE, from the
same point, agree in their first $p$ moments up to $O(\eta^{p+1})$, the local error
$O(\eta^{p+1})$ adds up to $O(\eta^p)$ over $t_{\max}/\eta$ steps. (The theorems also
need bounded higher moments, smooth enough per-example losses, and an i.i.d.
sample index at every step.)


Li, Tai and E (who write $(\eta\Sigma)^{1/2}$, with $\Sigma$ at batch size 1) prove
that the order-1 SME is weak order 1, and the **order-2 SME**

$$
d\theta=-\nabla\Big(L+\frac\eta4\|\nabla L\|^2\Big)dt+\sqrt{\eta/B}\;\Sigma(\theta)^{1/2}\,dW
$$

is weak order 2. Its drift correction is the one backward error analysis gives
for full-batch GD ([`td2_1_flows`](td2_1_flows.html)).

**Why keep the noise at order 1?** Gradient flow is *also* weak order 1 (2.2.A.2).
The noise term scales with $\mathrm{tr}\,\Sigma/B$ and the drift correction with
$\|G\|^2$; their ratio is about $B_{\rm simple}/B$ (§2.1). For small batches, and
near a minimum where $G\to0$ but the noise does not (unless the minimum
interpolates), the noise dominates. Keeping
it and dropping the drift correction is a physical choice, not a consequence of
order counting, and the linear scaling rule inherits it (§2.5).

**Exercise 2.2.A — Derive the SMEs by moment matching.** One more tool, the
**generator**: for $dX=b\,dt+\sigma\,dW$ with $D=\sigma\sigma^\top$ and smooth $\varphi$,
$\mathcal L\varphi=b\cdot\nabla\varphi+\tfrac12\mathrm{tr}(D\,\nabla^2\varphi)$. Dynkin's
formula says $\frac{d}{dt}\mathbb E\,\varphi(X_t)=\mathbb E\,\mathcal L\varphi(X_t)$;
applying it twice gives the short-time expansion

$$
\mathbb E[\varphi(X_{\Delta t})\mid X_0=x]=\varphi(x)+\Delta t\,\mathcal L\varphi(x)+\tfrac{\Delta t^2}2\mathcal L^2\varphi(x)+O(\Delta t^3).
$$

Fix a starting point $x$, and write $G,H,\Sigma$ for their values at $x$.

0. *(Generator.)* For one Euler–Maruyama step, $X_{\Delta t}=x+b\,\Delta t+\sigma\sqrt{\Delta t}\,\zeta$, expand $\mathbb E\,\varphi(X_{\Delta t})$ to $O(\Delta t)$ and recover $\mathcal L\varphi$.
1. *(SGD.)* Show that one SGD step has $\mathbb E\Delta=-\eta G$, $\mathbb E\,\Delta\Delta^\top=\eta^2\big(GG^\top+\Sigma/B\big)$, and $\mathbb E\|\Delta\|^3=O(\eta^3)$.
2. *(Gradient flow.)* Show that gradient flow run for time $\eta$ moves by $-\eta G+\frac{\eta^2}2HG+O(\eta^3)$. Using moment matching, conclude that gradient flow is weak order 1 but not order 2.
3. *(SDE.)* Take $b=-G+\eta\,b_1$ and $D=\frac\eta B\Sigma$. Using the short-time expansion with $\varphi(y)=y_i-x_i$ and $\varphi(y)=(y_i-x_i)(y_j-x_j)$, show that
$$
\mathbb E\tilde\Delta=-\eta G+\eta^2\Big(b_1+\tfrac12HG\Big)+O(\eta^3),\qquad
\mathbb E\,\tilde\Delta\tilde\Delta^\top=\eta^2\big(GG^\top+\Sigma/B\big)+O(\eta^3),
$$
where $\tilde\Delta=X_\eta-x$.
4. Conclude that matching to $O(\eta^3)$ requires $b_1=-\tfrac12HG=-\tfrac14\nabla\|G\|^2$, which gives the order-2 SME. Why does the noise term need no correction at this order? Why does the noise term alone (with $b_1=0$) not give order 2?
5. Under $(\eta,B)\to(\kappa\eta,\kappa B)$, which terms of the order-1 and order-2 SMEs are unchanged? Why is the linear scaling rule a first-order statement?

<details><summary>Hint</summary>

For 2, $\ddot\theta=-H\dot\theta=HG$. For 3, physically: the noise adds
$\eta D=\eta^2\Sigma/B$ to the second moment and moves the mean only at
$O(\eta\cdot D)=O(\eta^3)$, so to $O(\eta^2)$ the mean is that of the deterministic flow
with drift $b$. In the calculation, every term containing $D$ inside $\mathcal L^2$
is $O(\eta)\cdot\eta^2$. For 5, see which coefficients contain $\eta/B$ and which $\eta$
alone.
</details>
"""
    ),
    answer(
        r"""
**0.** Taylor-expand $\varphi$ around $x$ with increment $\delta=b\,\Delta t+\sigma\sqrt{\Delta t}\,\zeta$:
$\varphi(x+\delta)=\varphi+\nabla\varphi\cdot\delta+\tfrac12\delta^\top\nabla^2\varphi\,\delta+O(|\delta|^3)$.
Since $\mathbb E\zeta=0$ and $\mathbb E\zeta\zeta^\top=I$: $\mathbb E\,\nabla\varphi\cdot\delta=\Delta t\,b\cdot\nabla\varphi$, and $\mathbb E\,\delta^\top\nabla^2\varphi\,\delta=\Delta t\,\mathrm{tr}(D\nabla^2\varphi)+O(\Delta t^2)$. In the cubic term, the pure $\zeta^3$ part has zero mean by symmetry and every other part carries at least $\Delta t\cdot\Delta t$; the quartic remainder is $O(\Delta t^2)$. Hence $\mathbb E\,\varphi(X_{\Delta t})=\varphi+\Delta t\big(b\cdot\nabla\varphi+\tfrac12\mathrm{tr}(D\nabla^2\varphi)\big)+O(\Delta t^2)$. The $O(\Delta t^2)$ term needs the true SDE rather than one Euler–Maruyama step; the short-time expansion supplies it.

**1.** $\Delta=-\eta g_B$ with $\mathbb E g_B=G$ and $\mathrm{Cov}\,g_B=\Sigma/B$, so $\mathbb E\Delta=-\eta G$ and $\mathbb E\Delta\Delta^\top=\eta^2(\mathrm{Cov}\,g_B+GG^\top)=\eta^2(GG^\top+\Sigma/B)$. Likewise $\mathbb E\|\Delta\|^3=\eta^3\,\mathbb E\|g_B\|^3=O(\eta^3)$, since the third moments of the per-example gradients are bounded.

**2.** Along $\dot\theta=-G(\theta)$, $\ddot\theta=-H\dot\theta=HG$, so $\theta(\eta)-x=-\eta G+\frac{\eta^2}2HG+O(\eta^3)$. Compared with SGD, the first moments differ by $\frac{\eta^2}2HG=O(\eta^2)$, and the second moments by $\eta^2\Sigma/B=O(\eta^2)$ (gradient flow's is $\eta^2GG^\top+O(\eta^3)$). That is enough for $p=1$, so gradient flow is weak order 1. It is not order 2, because the first moments disagree at $O(\eta^2)$ (and so do the second).

**3.** *First moment.* $\mathcal L(y_i-x_i)=b_i(y)$, which at $y=x$ is $b_i=-G_i+\eta b_{1,i}$. Next, $\mathcal L b_i=(b\cdot\nabla)b_i+\tfrac12\mathrm{tr}(D\nabla^2b_i)$. The $D$ term is $O(\eta)$, and $(b\cdot\nabla)b=(-G)\cdot\nabla(-G)+O(\eta)=HG+O(\eta)$, because $\nabla(-G)=-H$. Hence

$$
\mathbb E\tilde\Delta=\eta(-G+\eta b_1)+\frac{\eta^2}2\big(HG+O(\eta)\big)=-\eta G+\eta^2\Big(b_1+\frac12HG\Big)+O(\eta^3).
$$

*Second moment.* With $\varphi=(y_i-x_i)(y_j-x_j)$: at $y=x$, $b\cdot\nabla\varphi=0$ and $\tfrac12\mathrm{tr}(D\nabla^2\varphi)=D_{ij}$, so $\mathcal L\varphi(x)=D_{ij}=\frac\eta B\Sigma_{ij}$. As a function of $y$, $\mathcal L\varphi=b_i(y)(y_j-x_j)+b_j(y)(y_i-x_i)+D_{ij}(y)$. Apply $\mathcal L$ again and set $y=x$, term by term:
- $b\cdot\nabla\big[b_i(y)(y_j-x_j)\big]=(b\cdot\nabla b_i)(y_j-x_j)+b_i\,b_j$, which at $y=x$ is $b_ib_j$. The other drift term gives $b_jb_i$ in the same way.
- $\tfrac12\mathrm{tr}\big(D\nabla^2[b_i(y)(y_j-x_j)]\big)$ at $y=x$ is $\sum_kD_{jk}\partial_kb_i$, and similarly with $i\leftrightarrow j$. These carry a factor $D=O(\eta)$.
- $\mathcal L D_{ij}=b\cdot\nabla D_{ij}+\tfrac12\mathrm{tr}(D\nabla^2D_{ij})$ is $O(\eta)$, because $D_{ij}$ itself is.

So $\mathcal L^2\varphi(x)=2b_ib_j+O(\eta)$. Hence

$$
\mathbb E\tilde\Delta_i\tilde\Delta_j=\eta\,\frac\eta B\Sigma_{ij}+\frac{\eta^2}2\,2G_iG_j+O(\eta^3)=\eta^2\Big(G_iG_j+\frac{\Sigma_{ij}}B\Big)+O(\eta^3).
$$

For the absolute third moment, $\tilde\Delta=\int_0^\eta b\,dt+\int_0^\eta\sigma\,dW$; the drift part is $O(\eta)$ and the noise part has $\mathbb E|\cdot|^3=O\big((\eta\cdot\eta/B)^{3/2}\big)=O(\eta^3)$ (standard moment bounds for stochastic integrals), so $\mathbb E\|\tilde\Delta\|^3=O(\eta^3)$, as for SGD.

**4.** The second moments already agree with part 1 to $O(\eta^3)$. The first moments agree to $O(\eta^3)$ iff $b_1=-\frac12HG$. Since $\nabla\|G\|^2=2HG$, this is $-\frac14\nabla\|G\|^2$. So the drift is $-\nabla(L+\frac\eta4\|\nabla L\|^2)$, and moment matching with $p=2$ gives the order-2 SME. Setting $\Sigma=0$ recovers backward error analysis for GD.

The noise needs no correction: $\eta D=\eta^2\Sigma/B$ is exactly SGD's second moment. But it does nothing for the *first* moment, which is off by $\frac{\eta^2}2HG$ without the drift correction: the discrete step uses the gradient at the start of the step, while the flow sees it shrink along the way.

**5.** The noise coefficient $\sqrt{\eta/B}$ and the drift $-\nabla L$ are unchanged, so the order-1 SME is invariant; the scaled run reaches each $t=k\eta$ in $\kappa$ times fewer steps with the same number of examples. The order-2 correction becomes $\frac{\kappa\eta}4\|\nabla L\|^2$. So the linear scaling rule is exact for the order-1 SME and fails at the next order, when the drift correction is comparable to the noise term (§2.4, §2.5).
"""
    ),
    markdown(
        r"""
**Code 2.2.B — Weak error, measured exactly.** Does the noise term of the order-1
SME buy accuracy over plain gradient flow, and does the order-2 drift correction
gain an order? Take $L=\tfrac12hx^2$ with additive gradient noise of variance
$s^2/B$. SGD and all three continuous models (gradient flow, order-1 SME, and
order-2 SME with drift $-(h+\eta h^2/2)x$) are linear, so $\mathbb E\,x^2$ is
available in closed form for each. The cell plots the weak error in $\mathbb E\,x^2$
at $t_{\max}=2$ against $\eta=2/n$, from two starting points.
"""
    ),
    code(
        r"""
def second_moment_errors(etas, x0, h=1.0, s2=1.0, B=1, t_max=2.0):
    # |E x_T^2 (model) - E x_T^2 (SGD)| for SGD on L = h x^2 / 2 with additive noise of variance s2/B.
    # Linear dynamics: these first and second moments are exact (no Monte Carlo, no Gaussianity needed).
    rows = []
    for eta in etas:
        k = int(round(t_max / eta)); t = k * eta; q = 1 - eta * h
        sgd = (q**k * x0) ** 2 + eta**2 * s2 / B * (1 - q ** (2 * k)) / (1 - q**2)
        gf = (np.exp(-h * t) * x0) ** 2                     # gradient flow: no variance at all
        def ou(a):                                         # dX = -a X dt + sqrt(eta s2 / B) dW
            return (np.exp(-a * t) * x0) ** 2 + eta * s2 / B * (1 - np.exp(-2 * a * t)) / (2 * a)
        rows.append([abs(gf - sgd), abs(ou(h) - sgd), abs(ou(h + eta * h**2 / 2) - sgd)])
    return np.array(rows)

etas = 2 / np.array([800, 400, 200, 100, 50])     # eta = 2/n, so t_max = 2 is hit exactly
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, x0, title in zip(axes, (3.0, 0.0), ("start far from the minimum (x0 = 3)", "start at the minimum (x0 = 0)")):
    err = second_moment_errors(etas, x0)
    for j, lab in enumerate(("gradient flow", "order-1 SME", "order-2 SME")):
        slope = np.polyfit(np.log(etas), np.log(err[:, j]), 1)[0]
        ax.loglog(etas, err[:, j], "o-", label=f"{lab} (slope {slope:.2f})")
    ax.set_title(title); ax.set_xlabel("η"); ax.legend(fontsize=8)
axes[0].set_ylabel(r"weak error in $E\,x^2$ at $t=2$")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
Left: with a strong signal, gradient flow and the order-1 SME are both first order
(here gradient flow happens to be better); only the drift correction gains an
order. Right: at the minimum there is no signal, gradient flow's error is the whole
$O(\eta)$ variance, and both SMEs are accurate to $O(\eta^2)$.
"""
    ),
    # ================================================================ 2.3
    markdown(
        r"""
<a id="td22-3"></a>

## 2.3 Fluctuations in a basin

Near a minimum, SGD at a constant learning rate does not converge; it keeps
jiggling. In a quadratic basin with Hessian $H$, and with the gradient-noise
covariance $\Sigma$ frozen at its value at the minimum, the iterates settle into a
Gaussian cloud whose excess loss is $\frac{\eta}{4B}\mathrm{tr}\,\Sigma$ (step $\eta$,
batch $B$), whatever $H$ is. The cloud is a Boltzmann distribution $\propto e^{-cL}$
only if $\Sigma\propto I$.

**Paper:** [Li, Tai and E](https://arxiv.org/abs/1511.06251) (see [Readings](#td22-readings)), §3: solvable quadratic examples, with a descent phase followed by a fluctuation phase.

Put the minimum at $0$ with $L\approx\tfrac12\theta^\top H\theta$, $H\succ0$. The order-1
SME of §2.2 becomes an Ornstein–Uhlenbeck (OU) process,
$d\theta=-H\theta\,dt+\sqrt{\eta/B}\,\Sigma^{1/2}dW$. Its stationary law is
$\mathcal N(0,\Pi)$, where $\Pi$ solves the **Lyapunov equation**

$$
H\Pi+\Pi H=\frac\eta B\,\Sigma .
$$

Its trace gives the loss floor $\tfrac12\mathrm{tr}(H\Pi)=\frac{\eta}{4B}\mathrm{tr}\,\Sigma$:
halve $\eta$, or double $B$, and the floor halves. Curvature drops out because a
sharper direction has proportionally smaller fluctuations. A learning-rate cut
lowers the floor at once, and the loss follows mode by mode. The often-assumed
$\Sigma\propto H$ gives a *round* cloud, not a Boltzmann one.

<a id="td22-3-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Hessian eigenvalues $h_j$; a general SDE $dX=b\,dt+\sigma\,dW$ has
drift $b$ and diffusion $D=\sigma\sigma^\top$ (for the SME, $b=-\nabla L$,
$D=\frac\eta B\Sigma$).

**When is the stationary law Boltzmann?** The density $p(\theta,t)$ of $X_t$ obeys
the **Fokker–Planck equation** $\partial_tp=-\nabla\cdot J$, with probability current
$J_i=b_i\,p-\frac12\sum_j\partial_j(D_{ij}\,p)$ (integrate Dynkin's formula of 2.2.A by
parts). With $b=-\nabla L$ and constant isotropic noise $D=2T\,I$,
$J=-p\nabla L-T\nabla p$ vanishes for $p\propto e^{-L/T}$ (for confining $L$):
overdamped Langevin dynamics, with Gibbs law at temperature $T$.
For SGD this needs $\frac\eta B\Sigma=2T\,I$, so the "temperature" $\eta/B$ is a noise
scale, not literally a Gibbs temperature. Real gradient noise depends on $\theta$
and is strongly anisotropic, often roughly aligned with $H$ (for a well-specified
likelihood, $\Sigma\approx H$ at the optimum), so SGD has no Gibbs guarantee.
Whether SGD's cloud resembles a Bayesian posterior is taken up by the MDL day in
[MDL1 §4](../notebooks/mdl1_statistical_learning.html#wb2-section-4), part 4.B, "Does
curvature cancel?", with the figure
[`sgd_stationary_vs_posterior.html`](../notebooks/assets/interactive/sgd_stationary_vs_posterior.html).

**The "flat minima" argument, and what it needs.** Under a Gibbs law in $d$
dimensions, a basin with loss $L_a$ and Hessian $H_a$ has mass
$\approx e^{-L_a/T}(2\pi T)^{d/2}\det(H_a)^{-1/2}$, so at equal loss flatter basins
hold more probability. That rests on constant isotropic noise. With SGD's noise,
which basin wins is a question about escape rates under anisotropic,
state-dependent noise (Kramers-type calculations; ask your agent), and about
whether training gets near stationarity at all (§2.7).

**Exercise 2.3.A — The Lyapunov equation and the loss floor.** The setting:
$L=\tfrac12\theta^\top H\theta$ with $H\succ0$, frozen noise covariance $\Sigma$,
diffusion $D=\frac\eta B\Sigma$, stationary covariance $\Pi$. (Items 1, 2, the first
identity of 3, and 4(a) repeat items 1–4 of the MDL day's MDL1 exercise 4.B.3, whose
$C$ and $\Sigma$ are $\Sigma$ and $\Pi$ here.) Items 4 and 6 compute what
discreteness and state-dependent noise change.

1. For the OU process, apply Dynkin's formula (2.2.A) with $\varphi=\theta_i\theta_j$ to show $\frac{d}{dt}M=-HM-MH+D$, where $M(t)=\mathbb E\,\theta\theta^\top$ (start from mean zero) and $D=\frac\eta B\Sigma$. Conclude that the stationary $\Pi$ solves the Lyapunov equation. Solve it in $H$'s eigenbasis, and show that $\Pi=\frac\eta{2B}\Sigma H^{-1}$ when $\Sigma$ commutes with $H$.
2. Show that $\mathcal N(0,\Pi)\propto e^{-cL}$ for some $c>0$ iff $\Sigma\propto I$, without assuming that $\Sigma$ and $H$ commute. Compute $\Pi$ for $\Sigma=sH$ and for $\Sigma=sI$, and say which is Boltzmann.
3. Show $\tfrac12\mathrm{tr}(H\Pi)=\frac\eta{4B}\mathrm{tr}\,\Sigma$ and $\mathrm{tr}(H^2\Pi)=\frac\eta{2B}\mathrm{tr}(H\Sigma)$ for any $\Sigma$. Deduce the equilibrium simple noise scale $\mathrm{tr}\,\Sigma/\mathbb E\|G\|^2=2B/(\eta\bar h)$, with $\bar h=\mathrm{tr}(H\Sigma)/\mathrm{tr}\,\Sigma$.
4. *(Discrete time.)* Take SGD with additive noise, $\theta_{k+1}=(I-\eta H)\theta_k-\eta\xi_k$ with $\mathrm{Cov}\,\xi_k=\Sigma/B$.
   - (a) Show that the stationary covariance satisfies $H\Pi+\Pi H-\eta H\Pi H=\frac\eta B\Sigma$.
   - (b) When $\Sigma$ commutes with $H$, show that the excess loss is $\frac\eta{2B}\sum_j\frac{\Sigma_{jj}}{2-\eta h_j}$ in $H$'s eigenbasis.
   - (c) When is the SDE value accurate to 10%?
   - (d) Does the order-2 SME's drift, here $-(H+\frac\eta2H^2)\theta$, fix the discrepancy?
   - (e) Follow one mode along a linear-scaling family $(\kappa\eta,\kappa B)$. For what $\kappa$ is its floor twice the SDE value, and twice that of the original run?
5. *(Learning-rate decay.)* At $t=0$, switch $\eta\to\eta'$ at fixed $B$. With $\Sigma$ commuting with $H$, show that mode $j$'s excess loss relaxes as $\mathcal E_j(t)=\mathcal E'_j+(\mathcal E_j-\mathcal E'_j)e^{-2h_jt}$, where $\mathcal E_j=\frac{\eta\Sigma_{jj}}{4B}$, $\mathcal E'_j=\frac{\eta'\Sigma_{jj}}{4B}$, and $t$ counts in units of the *new* step. How many steps does mode $j$ take? What changes immediately at a learning-rate cut, and what sets how quickly the loss follows?
6. *(Least squares, exactly. Code 2.3.B uses this.)* For $\ell_i=\tfrac12(x_i^\top\theta-y_i)^2$, write $\delta=\theta-\theta^*$, $A_i=x_ix_i^\top$, $u_i=x_i(y_i-x_i^\top\theta^*)$ (so $\frac1N\sum_iu_i=0$), $H=\frac1N\sum_iA_i$ and $\Sigma^*=\frac1N\sum_iu_iu_i^\top$. For SGD with replacement, show that the stationary covariance satisfies exactly
$$
\Pi=(I-\eta H)\Pi(I-\eta H)+\frac{\eta^2}B\Big(\frac1N\sum_iA_i\Pi A_i-H\Pi H\Big)+\frac{\eta^2}B\Sigma^* .
$$
What does the middle term mean?

<details><summary>Hint</summary>

For 1, $\mathcal L(\theta_i\theta_j)=-(H\theta)_i\theta_j-\theta_i(H\theta)_j+D_{ij}$; the
eigenbasis decouples it. For 2, substitute $\Pi=aH^{-1}$ and use uniqueness. For 3,
take the trace (for the second identity, multiply by $H$ first). For 4,
$\theta_k$ is independent of $\xi_k$. For 6, the step is
$\delta'=(I-\eta\bar A)\delta+\eta\bar u$ with batch averages $\bar A,\bar u$, and for
i.i.d. indices $\mathbb E\,\bar A\Pi\bar A=H\Pi H+\frac1B\big(\frac1N\sum_iA_i\Pi A_i-H\Pi H\big)$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathcal L(\theta_i\theta_j)=(-H\theta)_i\theta_j+\theta_i(-H\theta)_j+\tfrac12(D_{ij}+D_{ji})$. Taking expectations, $\dot M=-HM-MH+D$. At stationarity $H\Pi+\Pi H=D$. In the eigenbasis of $H$ (eigenvalues $h_j>0$) this reads $(h_j+h_k)\Pi_{jk}=D_{jk}$, so

$$
\Pi_{jk}=\frac{D_{jk}}{h_j+h_k},\qquad\text{equivalently}\qquad \Pi=\int_0^\infty e^{-Hs}De^{-Hs}\,ds .
$$

The solution exists and is unique. If $\Sigma$ commutes with $H$ it is diagonal in that basis, so $\Pi_{jj}=\frac{\eta\Sigma_{jj}}{2Bh_j}$, that is, $\Pi=\frac\eta{2B}\Sigma H^{-1}$.

**2.** $\mathcal N(0,\Pi)\propto\exp(-\tfrac12\theta^\top\Pi^{-1}\theta)$ and $e^{-cL}=\exp(-\tfrac c2\theta^\top H\theta)$ agree iff $\Pi=H^{-1}/c$. Substituting $\Pi=aH^{-1}$ into the Lyapunov equation gives $2a I=\frac\eta B\Sigma$, so $\Sigma\propto I$ is necessary. Conversely, if $\Sigma=sI$ then $\Pi=\frac{\eta s}{2B}H^{-1}$ solves the equation, and by uniqueness it is the solution: Boltzmann with $c=\frac{2B}{\eta s}$, i.e. temperature $T=\frac{\eta s}{2B}$. No commutation was assumed.

For $\Sigma=sH$: $\Pi=\frac{\eta s}{2B}I$ (check: $H\Pi+\Pi H=\frac{\eta s}BH$), a round ball, not $e^{-cL}$, which is narrow in sharp directions.

**3.** The trace of the Lyapunov equation is $2\,\mathrm{tr}(H\Pi)=\frac\eta B\mathrm{tr}\,\Sigma$, so $\tfrac12\mathrm{tr}(H\Pi)=\frac\eta{4B}\mathrm{tr}\,\Sigma$. Multiplying the equation on the left by $H$ and taking the trace gives $\mathrm{tr}(H^2\Pi)+\mathrm{tr}(H\Pi H)=2\,\mathrm{tr}(H^2\Pi)=\frac\eta B\mathrm{tr}(H\Sigma)$. Only cyclicity of the trace was used, so both hold for any $\Sigma$. Dividing, $\mathrm{tr}\,\Sigma/\mathbb E\|G\|^2=\frac{2B\,\mathrm{tr}\,\Sigma}{\eta\,\mathrm{tr}(H\Sigma)}=\frac{2B}{\eta\bar h}$.

**4.** (a) Since $\theta_k$ is independent of $\xi_k$, $\Pi=(I-\eta H)\Pi(I-\eta H)+\eta^2\Sigma/B$. Expanding and dividing by $\eta$ gives $H\Pi+\Pi H-\eta H\Pi H=\frac\eta B\Sigma$.

(b) In the commuting case, $(2h_j-\eta h_j^2)\Pi_{jj}=\frac\eta B\Sigma_{jj}$, so

$$
\tfrac12\mathrm{tr}(H\Pi)=\frac\eta{2B}\sum_j\frac{\Sigma_{jj}}{2-\eta h_j}=\frac{\eta}{4B}\sum_j\frac{\Sigma_{jj}}{1-\eta h_j/2}.
$$

Each mode's excess loss is enhanced by $(1-\eta h_j/2)^{-1}$, which diverges at the stability edge $\eta h_j=2$.

(c) For modes with $\eta h_j\lesssim0.18$: a strong condition if the noise sits in sharp modes.

(d) No. The order-2 SME's OU drift gives $\Pi_{jj}=\frac{\eta\Sigma_{jj}}{B\,h_j(2+\eta h_j)}$: the relative correction $(1+\eta h_j/2)^{-1}$ has the *opposite* sign to the truth, $(1-\eta h_j/2)^{-1}$. No contradiction with weak order 2: the covariance is itself $O(\eta)$, so an $O(\eta^2)$ weak error is only a relative $O(\eta)$ error. Weak-order guarantees can be poor for small quantities such as fluctuations near a minimum.

(e) Along the family, one mode's floor is $\frac{\eta\Sigma_{jj}}{4B}\big(1-\kappa\eta h_j/2\big)^{-1}$: the prefactor is invariant and the enhancement grows with $\kappa$. It is twice the SDE value when $\kappa\eta h_j=1$. Relative to the original run the ratio is $\frac{1-\eta h_j/2}{1-\kappa\eta h_j/2}$, which reaches $2$ at $\kappa\eta h_j=1+\eta h_j/2$. The run diverges at $\kappa\eta h_j=2$. So in a basin the linear scaling rule breaks once the scaled learning rate reaches the scale $1/h$ of the curvatures that carry the noise.

**5.** With $\Sigma$ diagonal in $H$'s eigenbasis, part 1 decouples into $\dot M_{jj}=-2h_jM_{jj}+\frac{\eta'}B\Sigma_{jj}$ after the switch (time $t=k\eta'$). The solution relaxes exponentially, at rate $2h_j$, from the old stationary value to the new one; multiplying by $h_j/2$ gives $\mathcal E_j(t)$. In steps, mode $j$ needs about $1/(2\eta'h_j)$.

What changes immediately is the *target*: the floor $\frac{\eta}{4B}\mathrm{tr}\,\Sigma$ is pure fluctuation, ten times lower the moment $\eta$ is cut tenfold. The loss follows mode by mode, at rates $2h_j$. A mode's share $\frac{\eta\Sigma_{jj}}{4B}$ does not depend on its curvature, so the speed depends on where the noise sits: noise in sharp modes drops within a few times $1/(2\eta'h_{\max})$ steps, which looks immediate; noise in flat modes drops slowly. This is one ingredient of the decay phase of warmup–stable–decay schedules ([`td2_3_edge_of_stability`, "Warmup and WSD"](td2_3_edge_of_stability.html#td23-8)): a high constant learning rate hides progress under a fluctuation floor, and the decay reveals it.

**6.** $\nabla\ell_i(\theta^*+\delta)=x_i(x_i^\top\theta^*+x_i^\top\delta-y_i)=A_i\delta-u_i$. A batch step is $\delta'=(I-\eta\bar A)\delta+\eta\bar u$, with $\bar A=\frac1B\sum_bA_{i_b}$ and $\bar u=\frac1B\sum_bu_{i_b}$. The batch is independent of $\delta$, and $\mathbb E\delta=0$ at stationarity (from $\mathbb E\delta'=(I-\eta H)\mathbb E\delta$), so the cross terms vanish:

$$
\Pi=\mathbb E\big[(I-\eta\bar A)\Pi(I-\eta\bar A)\big]+\eta^2\,\mathbb E\,\bar u\bar u^\top .
$$

Next, $\mathbb E\,\bar u\bar u^\top=\Sigma^*/B$ (i.i.d. with mean zero) and $\mathbb E\bar A=H$. The pairs $b\ne b'$ in $\mathbb E\,\bar A\Pi\bar A=\frac1{B^2}\sum_{b,b'}\mathbb E A_{i_b}\Pi A_{i_{b'}}$ contribute $\frac{B(B-1)}{B^2}H\Pi H$, and the $B$ diagonal pairs contribute $\frac1B\cdot\frac1N\sum_iA_i\Pi A_i$. Collecting terms gives the stated equation.

The middle term is the extra gradient noise from being away from the minimum (the covariance of the $A_i\delta$ part of the per-example gradient), $O(\eta)$ relative to $\Sigma^*$. At small $B$ it lowers the stability threshold below $\eta h_{\max}=2$, a toy version of SGD's batch-size-dependent stability edge ([`td2_3_edge_of_stability`, "The edge of stochastic stability"](td2_3_edge_of_stability.html#td23-6)). The equation is linear in $\Pi$, so the code solves it as a $d^2\times d^2$ linear system.
"""
    ),
    markdown(
        r"""
**Code 2.3.B — Checking the loss floor on linear regression.** How accurate is
the SDE's floor for real minibatch SGD, with state-dependent noise, as
$\eta\lambda_{\max}$ (step times largest Hessian eigenvalue) grows? $\Pi$ is the
stationary covariance of $\theta-\theta^*$ ($\theta^*$ the minimizer). "SDE" is the
Lyapunov solution with $\Sigma$ at $\theta^*$; "exact discrete" is the fixed point of
the recursion of 2.3.A.6. The cells run SGD (with replacement) on least squares
with $d=8$, $N=1000$, and curvatures spread over a factor of 10. The sweep scales
$\eta$ and $B$ together at fixed $\eta/B$, so the SDE predicts the same answer for
every row: it *is* the linear scaling rule of §2.5, in a basin. Each row runs 200
chains from the minimum, averaging after six relaxation times $1/(\eta h_{\min})$ of
the flattest mode.
"""
    ),
    code(
        r"""
# Least squares with per-example sampling: the noise is the real minibatch noise (state-dependent).
N, d = 1000, 8
X = rng.standard_normal((N, d)) * np.sqrt(np.geomspace(2.0, 0.2, d))
y = X @ rng.standard_normal(d) + 0.5 * rng.standard_normal(N)
w_star = np.linalg.lstsq(X, y, rcond=None)[0]
H = X.T @ X / N                                   # Hessian of L(w) = |Xw - y|^2 / 2N
U = X * (X @ w_star - y)[:, None]                 # per-example gradients at the minimum
Sigma = U.T @ U / N                               # their covariance (their mean is 0 at w_star)
hmin, hmax = np.linalg.eigvalsh(H)[[0, -1]]
K4 = np.einsum("ni,nj,nk,nl->ijkl", X, X, X, X).reshape(d * d, d * d) / N   # vec of mean_i A_i P A_i

def lyap(H, D):
    # Solve the Lyapunov equation H P + P H = D for symmetric H > 0, in H's eigenbasis.
    h, V = np.linalg.eigh(H)
    Dt = V.T @ D @ V
    return V @ (Dt / (h[:, None] + h[None, :])) @ V.T

def excess(P):
    return 0.5 * np.trace(H @ P)

def pi_sde(eta, B):
    return lyap(H, eta / B * Sigma)

def cov_map(eta, B):
    # One SGD step maps vec(Pi) -> op @ vec(Pi) + c exactly, for mean-zero laws such as chains started at w_star (2.3.A.6).
    M = np.eye(d) - eta * H
    return np.kron(M, M) + eta**2 / B * (K4 - np.kron(H, H)), (eta**2 / B * Sigma).ravel()

def pi_discrete(eta, B):
    # Exact stationary covariance of SGD with replacement: the fixed point of cov_map.
    op, c = cov_map(eta, B)
    return np.linalg.solve(np.eye(d * d) - op, c).reshape(d, d)

def sgd_step(W, eta, B, idx=None):
    # One SGD step for R chains at once (rows of W); batches drawn with replacement unless idx is given.
    if idx is None:
        idx = rng.integers(0, N, (W.shape[0], B))
    Xb = X[idx]
    return W - eta * np.einsum("rbi,rb->ri", Xb, np.einsum("rbi,ri->rb", Xb, W) - y[idx]) / B

def sgd_cov(eta, B, R=200, burn_tau=6, meas_tau=10, replace=True):
    # R independent SGD chains from w_star; time-averaged covariance of w - w_star after burn-in.
    tau = 1 / (eta * hmin)
    burn, total = int(burn_tau * tau), int((burn_tau + meas_tau) * tau)
    W = np.tile(w_star, (R, 1)); acc = np.zeros((d, d)); n = 0; k = 0; m = N // B
    while k < total:
        perm = None if replace else np.argsort(rng.random((R, N)), axis=1)   # reshuffle each epoch
        for j in range(m):
            W = sgd_step(W, eta, B, None if replace else perm[:, j * B:(j + 1) * B])
            k += 1
            if k > burn and k % 5 == 0:
                dW = W - w_star; acc += dW.T @ dW / R; n += 1
            if k >= total:
                break
    return acc / n

print(f"curvatures in [{hmin:.2f}, {hmax:.2f}],  tr Σ = {np.trace(Sigma):.3f}")
"""
    ),
    code(
        r"""
r = 0.01                                          # eta / B, held fixed
rows = []
for B in (1, 4, 16, 48, 96):
    eta = r * B
    P, Pd, Ps = sgd_cov(eta, B), pi_discrete(eta, B), pi_sde(eta, B)
    rows.append((eta, B, excess(P), excess(Pd), excess(Ps),
                 np.linalg.norm(P - Pd) / np.linalg.norm(Pd), np.linalg.norm(Ps - Pd) / np.linalg.norm(Pd)))
print("  eta    B  η·λmax   excess loss:  measured  exact-discrete  SDE (η trΣ/4B)   rel. error of Π:  measured  SDE")
for eta, B, a, b, c, e1, e2 in rows:
    print(f"{eta:5.2f} {B:4d}   {eta*hmax:5.2f}                {a:.5f}      {b:.5f}        {c:.5f}                   {e1:.3f}  {e2:.3f}")

eta_grid = r * np.linspace(1, 1.95 / (r * hmax), 60)   # from B = 1 up to the stability edge
plt.figure(figsize=(5.5, 3.6))
plt.plot(eta_grid * hmax, [excess(pi_discrete(e, e / r)) / excess(pi_sde(e, e / r)) for e in eta_grid],
         label="exact discrete theory")
plt.plot([row[0] * hmax for row in rows], [row[2] / row[4] for row in rows], "o", label="SGD (measured)")
plt.axhline(1, color="gray", ls="--", label="SDE: same for all rows")
plt.xlabel(r"$\eta\,\lambda_{\max}(H)$, with $\eta/B$ fixed"); plt.ylabel("excess loss / (η tr Σ / 4B)")
plt.yscale("log"); plt.legend(); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The measurement matches the exact discrete theory to within a few percent (Monte
Carlo error). The SDE is right at small $\eta$ and drifts away as $\eta\lambda_{\max}$
grows, though $\eta/B$ never changes: 50% above at $\eta\lambda_{\max}\approx0.9$, six
times above at $1.8$, divergent at the stability edge. In the $B=1$ row the small
gap comes mostly from the state-dependent noise term of 2.3.A.6.

Next, a learning-rate decay. The dashed line is not a fit: it iterates the exact
covariance map of 2.3.A.6 from the same start.
"""
    ),
    code(
        r"""
B, eta, R = 4, 0.2, 200
P0 = pi_discrete(eta, B)                          # start every chain in the stationary law at eta
W = w_star + rng.multivariate_normal(np.zeros(d), P0, R); trace = []
P = P0.ravel(); exact = []                        # exact expected covariance from the same start
maps = {lr: cov_map(lr, B) for lr in (eta, eta / 10)}
for k in range(3000):
    lr = eta if k < 1500 else eta / 10
    W = sgd_step(W, lr, B)
    dW = W - w_star
    trace.append(0.5 * np.einsum("ri,ij,rj->r", dW, H, dW).mean())
    op, c = maps[lr]; P = op @ P + c
    exact.append(excess(P.reshape(d, d)))
plt.figure(figsize=(6, 3.4))
plt.semilogy(trace, lw=1, alpha=0.8, label="SGD, mean over 200 chains")
plt.semilogy(exact, "k--", lw=1.2, label="exact expectation (2.3.A.6)")
plt.axvline(1500, color="gray", lw=0.8)
plt.xlabel("step"); plt.ylabel("excess loss  L(w) − L(w*)"); plt.legend()
plt.title("η → η/10 at step 1500")
plt.tight_layout(); plt.show()
print("relaxation times 1/(2 η' h_j) in steps, sharpest mode first:",
      np.round(1 / (2 * eta / 10 * np.linalg.eigvalsh(H)[::-1])).astype(int))
"""
    ),
    markdown(
        r"""
Finally, sampling with replacement against random reshuffling (2.1.A.2) at
$B=10$, so an epoch is $m=100$ steps. The second column, $m\,\eta\,h_{\min}$, is the
length of an epoch in relaxation times $1/(\eta h_{\min})$ of the flattest mode.
"""
    ),
    code(
        r"""
B = 10
print("  eta   m·η·hmin   with replacement   reshuffled   ratio   SDE value")
for eta in (0.01, 0.03, 0.1, 0.3):
    wr = excess(sgd_cov(eta, B, R=100, replace=True))
    rr = excess(sgd_cov(eta, B, R=100, replace=False))
    print(f"{eta:5.2f}     {N // B * eta * hmin:5.2f}        {wr:.5f}          {rr:.5f}     {rr / wr:.2f}    {eta / (4 * B) * np.trace(Sigma):.5f}")
"""
    ),
    markdown(
        r"""
When an epoch is short compared with the relaxation time, reshuffled SGD sits
well below the SDE's floor: the noise cancels over each epoch before the slow
modes integrate it. When an epoch spans several relaxation times, the schemes
agree and the SDE value applies (up to the discrete-time enhancement in the last
row).
"""
    ),
    # ================================================================ 2.4 (critical batch size)
    markdown(
        r"""
<a id="td22-cbs"></a>

## 2.4 Critical batch size

If you retune the learning rate for each batch size, past what batch size do
extra examples stop helping? A one-step model says: past the noise scale
$B_{\rm noise}=\mathrm{tr}(H\Sigma)/G^\top HG$, with $G$ the gradient, $H$ the Hessian
and $\Sigma$ the per-example gradient covariance. Below it, doubling the batch
nearly halves the steps; above it, extra examples are wasted.

**Paper:** [McCandlish et al.](https://arxiv.org/abs/1812.06162) (see [Readings](#td22-readings)), §2 for the model. Appendix A.1 has the two-batch estimator; Appendix C explains the temperature dependence of the noise scale (2.3.A.3 here).

With the best step size, one step at batch $B$ makes a fraction
$1/(1+B_{\rm noise}/B)$ of a full-batch step's progress. So steps and examples trade
off along a hyperbola whose knee is the critical batch size
$B_{\rm crit}\approx B_{\rm noise}$. The Hessian-free simple noise scale
$B_{\rm simple}=\mathrm{tr}\,\Sigma/\|G\|^2$ (§2.1) is usually within a small factor of
$B_{\rm noise}$. In McCandlish et al.'s experiments it predicted $B_{\rm crit}$ to within
an order of magnitude, for values from about 20 to millions, and both grew as the
loss fell.

<a id="td22-cbs-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Gradient $G=\nabla L$, Hessian $H$, per-example gradient covariance
$\Sigma$, minibatch gradient $g_B$ over $B$ examples drawn with replacement (so
$\mathrm{Cov}\,g_B=\Sigma/B$), and step size $\epsilon$ (McCandlish et al.'s symbol; it is
the $\eta$ of other sections). For one step $\theta\to\theta-\epsilon g_B$,

$$
\mathbb E\,L(\theta-\epsilon g_B)\approx L(\theta)-\epsilon\|G\|^2+\frac{\epsilon^2}2\Big(G^\top HG+\frac{\mathrm{tr}(H\Sigma)}B\Big).
$$

This rough local model needs $G\ne0$ and positive curvature along the step
($G^\top HG>0$, $\mathrm{tr}(H\Sigma)>0$), and ignores third-order terms. Minimizing
over $\epsilon$ (2.4.A.1) gives

$$
\epsilon_{\rm opt}(B)=\frac{\epsilon_{\max}}{1+B_{\rm noise}/B},\qquad
\Delta L_{\rm opt}(B)=\frac{\Delta L_{\max}}{1+B_{\rm noise}/B},\qquad
B_{\rm noise}=\frac{\mathrm{tr}(H\Sigma)}{G^\top HG},
$$

with $\epsilon_{\max}=\|G\|^2/G^\top HG$ and $\Delta L_{\max}=\|G\|^4/(2G^\top HG)$, the
full-batch values. For $B\ll B_{\rm noise}$, $\epsilon_{\rm opt}\approx\epsilon_{\max}B/B_{\rm noise}$:
the linear scaling rule. If $H$ is a multiple of the identity, $B_{\rm noise}$
reduces to $B_{\rm simple}=\mathrm{tr}\,\Sigma/\|G\|^2$.

**Steps versus examples.** If $B_{\rm noise}$ is roughly constant, the number of
steps $S$ (not the sharpness of other sections) and examples $E=SB$ needed to reach
a given loss obey

$$
\Big(\frac S{S_{\min}}-1\Big)\Big(\frac E{E_{\min}}-1\Big)=1,\qquad B_{\rm crit}\equiv\frac{E_{\min}}{S_{\min}}\approx B_{\rm noise}.
$$

At $B=B_{\rm crit}$ you pay twice the minimum steps and twice the minimum examples.

**Measuring it.** With replacement, $\mathbb E\|g_B\|^2=\|G\|^2+\mathrm{tr}\,\Sigma/B$, so
two batch sizes give two equations in two unknowns (2.4.A.3); in data-parallel
training the per-device and all-reduced gradients supply them for free. Two
warnings. The estimate of $\|G\|^2$ is a small difference of large numbers, so
average numerator and denominator separately before dividing. And $B_{\rm simple}$
depends on the temperature of the run you measure it in: at equilibrium in a
basin it is $\frac{2B}{\eta\bar h}$, with $\bar h=\mathrm{tr}(H\Sigma)/\mathrm{tr}\,\Sigma$
(2.3.A.3), so it rises tenfold when you cut $\eta$ tenfold. McCandlish et al.
recommend measuring on a run with a well-tuned learning rate.

**Exercise 2.4.A — The one-step model and the two-batch estimator.**

1. Derive $\epsilon_{\rm opt}$ and $\Delta L_{\rm opt}$ from the quadratic expansion. Why does the expansion use $\mathrm{tr}(H\Sigma)/B$, and how would it change for batches drawn without replacement from $N$ examples?
2. Assume each step at batch $B$ makes progress $\Delta L_{\max}/(1+B_{\rm noise}/B)$, with $\Delta L_{\max}$ and $B_{\rm noise}$ fixed. Show that $S=S_{\min}(1+B_{\rm noise}/B)$ and $E=E_{\min}(1+B/B_{\rm noise})$ with $E_{\min}=S_{\min}B_{\rm noise}$. Derive the tradeoff and $B_{\rm crit}=B_{\rm noise}$.
3. From $\mathbb E\|g_B\|^2=\|G\|^2+\mathrm{tr}\,\Sigma/B$ at $B_s<B_b$, derive unbiased estimators of $\|G\|^2$ and $\mathrm{tr}\,\Sigma$. Is their ratio an unbiased estimator of $B_{\rm simple}$?

<details><summary>Hint</summary>

For 1, $\mathbb E\,g_B^\top Hg_B=G^\top HG+\mathrm{tr}(H\,\mathrm{Cov}\,g_B)$. For 2, steps
needed are total progress over progress per step; $S_{\min}$ is the $B\to\infty$
limit and $E_{\min}$ the $B\to0$ limit of $SB$. For 3, write
$\mathbb E n_s$ and $\mathbb E n_b$ as two linear equations in $\|G\|^2$ and
$\mathrm{tr}\,\Sigma$, and solve.
</details>
"""
    ),
    answer(
        r"""
**1.** Expanding $L(\theta-\epsilon g_B)$ to second order and averaging over the batch: $\mathbb E\,G^\top g_B=\|G\|^2$, and $\mathbb E\,g_B^\top Hg_B=G^\top HG+\mathrm{tr}(H\Sigma)/B$ because $\mathrm{Cov}\,g_B=\Sigma/B$. Setting the $\epsilon$-derivative of $-\epsilon\|G\|^2+\frac{\epsilon^2}2(G^\top HG+\mathrm{tr}(H\Sigma)/B)$ to zero,

$$
\epsilon_{\rm opt}=\frac{\|G\|^2}{G^\top HG+\mathrm{tr}(H\Sigma)/B}=\frac{\|G\|^2/G^\top HG}{1+\frac{\mathrm{tr}(H\Sigma)}{B\,G^\top HG}}=\frac{\epsilon_{\max}}{1+B_{\rm noise}/B}.
$$

Substituting back, $\Delta L_{\rm opt}=\frac12\epsilon_{\rm opt}\|G\|^2=\frac{\|G\|^4}{2(G^\top HG+\mathrm{tr}(H\Sigma)/B)}=\frac{\Delta L_{\max}}{1+B_{\rm noise}/B}$. The noise enters only through $\mathrm{tr}(H\,\mathrm{Cov}\,g_B)$: noise along sharp directions costs more. Without replacement, $\mathrm{Cov}\,g_B=\frac{N-B}{N-1}\frac\Sigma B$, so $B_{\rm noise}/B$ is replaced by $\frac{N-B}{N-1}B_{\rm noise}/B$: negligible for $B\ll N$ and exactly zero at $B=N$.

**2.** To cover a fixed total loss decrease $\Delta_{\rm tot}$, $S=\Delta_{\rm tot}(1+B_{\rm noise}/B)/\Delta L_{\max}$. As $B\to\infty$, $S\to S_{\min}=\Delta_{\rm tot}/\Delta L_{\max}$, so $S=S_{\min}(1+B_{\rm noise}/B)$. Then $E=SB=S_{\min}(B+B_{\rm noise})$, which tends to $E_{\min}=S_{\min}B_{\rm noise}$ as $B\to0$, so $E=E_{\min}(1+B/B_{\rm noise})$. Hence $S/S_{\min}-1=B_{\rm noise}/B$ and $E/E_{\min}-1=B/B_{\rm noise}$, whose product is $1$, and $E_{\min}/S_{\min}=B_{\rm noise}$. Small batches minimize examples (compute), large batches minimize steps (wall-clock), and $B_{\rm crit}$ is the knee of the hyperbola.

**3.** Write $n_s=\|g_{B_s}\|^2$ and $n_b=\|g_{B_b}\|^2$. From $\mathbb En=\|G\|^2+\mathrm{tr}\,\Sigma/B$ at the two batch sizes:

$$
\widehat{\|G\|^2}=\frac{B_bn_b-B_sn_s}{B_b-B_s},\qquad\widehat{\mathrm{tr}\,\Sigma}=\frac{n_s-n_b}{1/B_s-1/B_b}.
$$

Both are linear in $(n_s,n_b)$, so they are unbiased; substitute the expectations to check. Their ratio is not unbiased (a ratio of unbiased estimators is biased in general), and since $\widehat{\|G\|^2}$ can be near zero or even negative, the ratio is badly behaved unless the denominator is averaged first.
"""
    ),
    markdown(
        r"""
**Code 2.4.B — The steps–examples tradeoff, measured.** Does the hyperbola of
2.4.A.2 hold, with $B_{\rm crit}\approx B_{\rm noise}$, on a problem where both can be
computed? The cell reuses the least-squares problem of Code 2.3.B ($d=8$, $N=1000$).
For each batch size from 1 to 512 it tries 18 learning rates (up to the stability
edge), runs 16 chains from a common start, and records the fewest steps $S(B)$
after which the mean excess loss falls below $0.01$ (it starts at about 3). It fits
$S=S_{\min}+E_{\min}/B$ and compares $B_{\rm crit}=E_{\min}/S_{\min}$ with $B_{\rm noise}$
measured along the way.
"""
    ),
    code(
        r"""
def steps_to_target(B, etas, w_init, target, R=16, max_steps=20000):
    # Fewest steps for the mean excess loss of R chains to reach `target`, for each learning rate.
    W = np.tile(w_init, (len(etas) * R, 1)); lr = np.repeat(etas, R)[:, None]
    hit = np.full(len(etas), -1)
    for k in range(1, max_steps + 1):
        W = sgd_step(W, lr, B)
        dW = W - w_star
        L = 0.5 * np.einsum("ri,ij,rj->r", dW, H, dW).reshape(len(etas), R).mean(1)
        hit[(hit < 0) & (L < target)] = k
        if (hit >= 0).all() or ((hit > 0).any() and k > 3 * hit[hit > 0].min()):
            break
    return hit

def noise_scales(w):
    g = X * (X @ w - y)[:, None]; G = g.mean(0); S = (g - G).T @ (g - G) / N
    return np.trace(H @ S) / (G @ H @ G), np.trace(S) / (G @ G)

w_init = w_star + 1.5 * rng.standard_normal(d); target = 0.01
etas = np.geomspace(0.002, 1.9 / hmax, 18)
Bs = 2 ** np.arange(10); S_of_B = []; best_eta = {}
for B in Bs:
    with np.errstate(over="ignore", invalid="ignore"):   # the largest learning rates diverge at small B
        hit = steps_to_target(B, etas, w_init, target)
    S_of_B.append(hit[hit > 0].min()); best_eta[B] = etas[hit > 0][np.argmin(hit[hit > 0])]
S_of_B = np.array(S_of_B, float)
A = np.vstack([np.ones(len(Bs)), 1 / Bs]).T / S_of_B[:, None]      # fit S = S_min + E_min/B in relative error
S_min, E_min = np.linalg.lstsq(A, np.ones(len(Bs)), rcond=None)[0]

plt.figure(figsize=(5.5, 3.8))
plt.loglog(S_of_B, S_of_B * Bs, "o", label="best over η, each B")
for B, S in zip(Bs, S_of_B):
    plt.annotate(f"{B}", (S, S * B), textcoords="offset points", xytext=(4, 2), fontsize=7)
Bf = np.geomspace(0.5, 2000, 200)
plt.loglog(S_min + E_min / Bf, (S_min + E_min / Bf) * Bf, "k--", lw=1, label="fit: (S/S_min−1)(E/E_min−1)=1")
plt.xlabel("steps S to reach the target"); plt.ylabel("examples E = S·B"); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"fit: S_min = {S_min:.1f} steps, E_min = {E_min:.0f} examples, B_crit = E_min/S_min = {E_min / S_min:.0f}")
for B in (8, 64):                                 # B_noise measured along the tuned runs themselves
    W = np.tile(w_init, (16, 1)); rec = []
    while True:
        dW = W - w_star; L = 0.5 * np.einsum("ri,ij,rj->r", dW, H, dW).mean()
        rec.append((L, np.median([noise_scales(w)[0] for w in W])))
        if L < target:
            break
        W = sgd_step(W, best_eta[B], B)
    rec = np.array(rec); late = rec[:, 0] < 0.1
    print(f"B = {B:3d}, η = {best_eta[B]:.3f}: {len(rec) - 1} steps, {late.sum()} of them with excess < 0.1;  "
          f"median B_noise over chains: {rec[0, 1]:.0f} at the start, {np.median(rec[late, 1]):.0f} over the "
          f"late steps, {rec[-1, 1]:.0f} at the end")
"""
    ),
    markdown(
        r"""
The points roughly follow the fitted hyperbola (residuals of about 10%, weak
evidence for a two-parameter fit to a monotone curve). The last two lines are more
informative: along the tuned runs $B_{\rm noise}$ grows twenty- to fortyfold as the
loss falls (median about 4 at the start, 75–190 late), the late stretch takes most
of the steps, and the fitted $B_{\rm crit}$ lies inside the range of $B_{\rm noise}$.
McCandlish et al. predict $B_{\rm crit}\approx B_{\rm noise}$ "appropriately averaged"
over training (their Appendix D).

**Exercise 2.4.C (code) — Estimate $B_{\rm simple}$.** Fill in
`noise_scale_estimates`, using the two-batch estimators of 2.4.A.3. The next cell
trains logistic regression with SGD ($B=16$, $\eta=0.5$, then $0.05$ from step 3000).
Every 100 steps it compares the exact $\mathrm{tr}\,\Sigma/\|G\|^2$ with your estimate
(2000 minibatches at each of $B_s=8$ and $B_b=128$), per checkpoint and as a ratio
of means over blocks of 5 checkpoints. The dashed lines are the OU prediction
$\frac{2B\,\mathrm{tr}\,\Sigma}{\eta\,\mathrm{tr}(H\Sigma)}$ (2.3.A.3), with $H$ and $\Sigma$ at
the minimizer.
"""
    ),
    code_answer(
        r"""
def noise_scale_estimates(g, b_small, b_big, draws, rng):
    # g: (N, d) per-example gradients at the current parameters.
    # Draw `draws` minibatches of each size, with replacement; average the squared norms of the
    # minibatch-mean gradients at each size; return unbiased estimates (G2_hat, trSigma_hat)
    # of |G|^2 and tr(Sigma).
    raise NotImplementedError
""",
        r"""
def noise_scale_estimates(g, b_small, b_big, draws, rng):
    # g: (N, d) per-example gradients at the current parameters.
    # Draw `draws` minibatches of each size, with replacement; average the squared norms of the
    # minibatch-mean gradients at each size; return unbiased estimates (G2_hat, trSigma_hat)
    # of |G|^2 and tr(Sigma).
    n = g.shape[0]
    ns = (g[rng.integers(0, n, (draws, b_small))].mean(1) ** 2).sum(1).mean()
    nb = (g[rng.integers(0, n, (draws, b_big))].mean(1) ** 2).sum(1).mean()
    G2_hat = (b_big * nb - b_small * ns) / (b_big - b_small)
    trSigma_hat = (ns - nb) / (1 / b_small - 1 / b_big)
    return G2_hat, trSigma_hat
""",
    ),
    code(
        r"""
N2, d2 = 4000, 20
X2 = rng.standard_normal((N2, d2)) * np.sqrt(np.geomspace(3, 0.3, d2))
y2 = (rng.random(N2) < 1 / (1 + np.exp(-X2 @ rng.standard_normal(d2)))).astype(float)

def per_example_grads(w):
    return X2 * (1 / (1 + np.exp(-X2 @ w)) - y2)[:, None]

w = np.zeros(d2)                                  # Newton's method for the minimizer, then H and Sigma there
for _ in range(30):
    q = 1 / (1 + np.exp(-X2 @ w))
    w -= np.linalg.solve((X2 * (q * (1 - q))[:, None]).T @ X2 / N2, X2.T @ (q - y2) / N2)
q = 1 / (1 + np.exp(-X2 @ w)); H2 = (X2 * (q * (1 - q))[:, None]).T @ X2 / N2
g = per_example_grads(w); S2 = g.T @ g / N2

try:
    noise_scale_estimates(g, 8, 128, 10, rng)
    ready = True
except NotImplementedError:
    ready = False
    print("Fill in noise_scale_estimates above to run this demo.")

if ready:
    B, draws = 16, 2000
    w = np.zeros(d2); log = []                    # columns: step, exact trS, exact |G|^2, est trS, est |G|^2
    for k in range(6001):
        eta = 0.5 if k < 3000 else 0.05
        if k % 100 == 0:
            g = per_example_grads(w); G = g.mean(0)
            G2_hat, trS_hat = noise_scale_estimates(g, 8, 128, draws, rng)
            log.append((k, ((g - G) ** 2).sum(1).mean(), G @ G, trS_hat, G2_hat))
        w -= eta * per_example_grads(w)[rng.integers(0, N2, B)].mean(0)
    log = np.array(log)
    blk = log[1:].reshape(-1, 5, 5).mean(1)       # block means over 5 checkpoints (skip the initial point)

    plt.figure(figsize=(6.5, 3.8))
    plt.semilogy(log[:, 0], log[:, 1] / log[:, 2], "-", lw=0.8, color="gray", label="exact tr Σ / |G|², per checkpoint")
    ok = log[:, 4] > 0
    plt.semilogy(log[ok, 0], log[ok, 3] / log[ok, 4], ".", ms=4, label="estimate, per checkpoint")
    plt.semilogy(blk[:, 0], blk[:, 1] / blk[:, 2], "s-", ms=4, label="exact, ratio of block means")
    plt.semilogy(blk[:, 0], blk[:, 3] / blk[:, 4], "o-", ms=4, label="estimate, ratio of block means")
    for lr, (a, b) in ((0.5, (0, 3000)), (0.05, (3000, 6000))):
        plt.hlines(2 * B * np.trace(S2) / (lr * np.trace(H2 @ S2)), a, b, colors="k", linestyles="--")
    plt.axvline(3000, color="gray", lw=0.8)
    plt.xlabel("step   (B = 16;  η = 0.5, then 0.05 from step 3000)"); plt.ylabel(r"$B_{\rm simple}$")
    plt.legend(fontsize=7); plt.tight_layout(); plt.show()
    print(f"B_simple at initialization: {log[0, 1] / log[0, 2]:.1f};  "
          f"checkpoints with a negative |G|^2 estimate (not plotted): {(~ok).sum()} of {len(log)}")
"""
    ),
    markdown(
        r"""
$B_{\rm simple}$ starts in the tens, rises quickly to the level predicted for
$\eta=0.5$, and jumps about tenfold after the learning-rate cut. Per-checkpoint
ratios fluctuate because at equilibrium the gradient is itself a fluctuation; the
prediction is for the ratio of means. A few single-checkpoint estimates are
negative, since $\widehat{\|G\|^2}$ is a small difference of large numbers; averaging
before dividing fixes most of this. "The" critical batch size depends on when you
ask, and at what learning rate.
"""
    ),
    # ================================================================ 2.5 (LSR)
    markdown(
        r"""
<a id="td22-lsr"></a>

## 2.5 The linear scaling rule and where it fails

Multiply the batch size $B$ by $\kappa$, multiply the learning rate $\eta$ by
$\kappa$. This *linear scaling rule* (LSR) works because the order-1 SME of §2.2
depends on $(\eta,B)$ only through $\eta/B$. It fails once $\eta$ times the top
Hessian eigenvalue is no longer small, or once $B$ passes the critical batch size
of §2.4.

**Papers:** [Li, Malladi and Arora](https://arxiv.org/abs/2102.12470), §2–§5: the failure criterion and SVAG; Goyal et al., *Accurate, large minibatch SGD*, §2, for the empirical rule (see [Readings](#td22-readings)).

Under $B\to\kappa B$, $\eta\to\kappa\eta$, the SDE is unchanged, and SDE time $t$
($t/\eta$ steps, $tB/\eta$ examples) is reached in $\kappa$ times fewer steps with the
same number of examples. It must fail eventually, in three ways. *Higher-order drift* (§2.4): the curvature term
$\frac{\eta^2}2G^\top HG$ grows like $\kappa$ per example, so the LSR holds only while
$B\ll B_{\rm noise}$. *Discreteness* (2.3.A.4): in a basin the floor grows by
$(1-\kappa\eta h/2)^{-1}$ for a curvature $h$, however small $\eta/B$ is. And
*everything the SME leaves out* (§2.7). Li, Malladi and Arora give a cheap test that
certifies failure, and a simulation, SVAG, that checks whether a run is at its SDE
limit.

<a id="td22-lsr-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Gradient $G=\nabla L$, Hessian $H$ with eigenvalues $h$, per-example
gradient covariance $\Sigma$, minibatch gradient $g_B$; order-1 SME
$d\theta=-\nabla L\,dt+\sqrt{\eta/B}\,\Sigma^{1/2}dW$.

**Li, Malladi and Arora's criterion.** For networks with normalization layers
trained with weight decay to an equilibrium (see
[`td2_5_normalization_weight_decay`](td2_5_normalization_weight_decay.html)), let
$\mathcal N$ and $\mathcal G$ be the equilibrium values of
$\mathbb E\,\mathrm{tr}\,\mathrm{Cov}(g_B)$ (the *minibatch* noise trace) and
$\mathbb E\|\nabla L\|^2$ in a run at $(B,\eta)$. They prove that if

$$
\kappa>C^2\Big(1+\frac{\mathcal N}{\mathcal G}\Big),
$$

the runs at $(B,\eta)$ and $(\kappa B,\kappa\eta)$ cannot agree to within a factor
$C$ on three test functions (weight norm, gradient norm, noise trace per example).
Since $\mathcal N=\mathbb E\,\mathrm{tr}\,\Sigma/B$, in the noisy regime the LSR fails
beyond $\kappa B\approx C^2\,\mathbb E\,\mathrm{tr}\,\Sigma/\mathbb E\|\nabla L\|^2$, i.e. $C^2$
times the equilibrium noise scale (ratio of means, as in 2.4.C), read off one
small-batch run (their Theorem 5.6, with $C^2=2$ in experiments). It certifies
failure, not success.


**SVAG** (Stochastic Variance Amplified Gradient). For an integer $l\ge1$, combine
two independent minibatch gradients with weights $a_\pm=\frac{1\pm\sqrt{2l-1}}2$ and
step with learning rate $\eta/l$. Each step matches an Euler–Maruyama step of size
$\eta/l$ for the *same* SDE (2.5.A.1), so as $l\to\infty$ SVAG converges weakly to it
(their Theorem 4.3); $l=1$ is SGD. If the curves for $l=1,2,4,\dots$ coincide, SGD is
at its SDE limit for those observables.

**Exercise 2.5.A — SVAG, and what the noise-to-signal ratio measures.** In items 2
and 3, $\Sigma$ is a scalar: the per-example gradient-noise variance along one mode.

1. *(SVAG.)* Show $a_++a_-=1$ and $a_+^2+a_-^2=l$. Conclude that the SVAG gradient has mean $G$ and covariance $l\,\Sigma/B$, and that one SVAG step has the conditional mean and covariance of an Euler–Maruyama step of size $\eta/l$ for $d\theta=-\nabla L\,dt+\sqrt{\eta/B}\,\Sigma^{1/2}dW$, the same SDE for every $l$.
2. *(The criterion in a basin.)* For one mode of curvature $h$ of discrete SGD with additive noise (2.3.A.4), show that at equilibrium $\mathcal N/\mathcal G=(2-\eta h)/(\eta h)$. (The OU model gives $2/(\eta h)$, its small-$\eta h$ limit; compare 2.3.A.3.) When does the noise dominate? Substitute into $\kappa>C^2(1+\mathcal N/\mathcal G)$ with $C^2=2$, and compare the threshold with the stability limit $\kappa\eta h<2$. Where is the criterion informative?
3. *(SVAG in one mode.)* For one mode of curvature $h$ with additive noise, show that SVAG($l$) at learning rate $\eta$ and batch $B$ has stationary excess loss $\frac{\eta\Sigma}{4B}\big(1-\frac{\eta h}{2l}\big)^{-1}$. How fast does it approach the SDE value, and why does the $\eta\lambda_{\max}\approx1.8$ curve in Code 2.5.B need larger $l$?

<details><summary>Hint</summary>

For 1, $\mathbb E(a_+g_1+a_-g_2)=(a_++a_-)G$ and the covariance is
$(a_+^2+a_-^2)\Sigma/B$ for independent batches. For 2, $\mathcal G=\mathbb E\,G^2=h^2\Pi$
with 2.3.A.4's $\Pi$ for one mode, and $\mathcal N=\Sigma/B$. For 3, SVAG is SGD
with learning rate $\eta/l$ and per-step noise covariance $l\Sigma/B$; redo 2.3.A.4(b).
</details>
"""
    ),
    answer(
        r"""
**1.** $a_++a_-=1$, and $a_+^2+a_-^2=\frac{(1+s)^2+(1-s)^2}4=\frac{1+s^2}2=\frac{1+2l-1}2=l$, with $s=\sqrt{2l-1}$. The SVAG gradient $a_+g_1+a_-g_2$ (two independent batches) has mean $G$ and covariance $l\,\Sigma/B$. One step with learning rate $\eta/l$ has mean $-\frac\eta lG$ and covariance $\frac{\eta^2}{l^2}\cdot l\frac\Sigma B=\frac\eta l\cdot\frac\eta B\Sigma$: the mean and covariance of an Euler–Maruyama step of size $\Delta t=\eta/l$ for drift $-G$ and diffusion $D=\frac\eta B\Sigma$. The SDE being matched does not depend on $l$; only the step size does. (Convergence also needs the higher moments to shrink fast enough; Li, Malladi and Arora check this, Lemma 4.6.)

**2.** From 2.3.A.4, one mode has $\Pi=\frac{\eta\Sigma}{Bh(2-\eta h)}$, so $\mathcal G=h^2\Pi=\frac{\eta h\Sigma}{B(2-\eta h)}$ and

$$
\frac{\mathcal N}{\mathcal G}=\frac{\Sigma/B}{\eta h\Sigma/(B(2-\eta h))}=\frac{2-\eta h}{\eta h}.
$$

The noise dominates ($\mathcal N/\mathcal G>1$) only for $\eta h<1$; at $\eta h=1.5$ the gradient, though produced entirely by fluctuations, is larger than its noise. For small $\eta h$ the ratio is $2/(\eta h)$, the equilibrium $B_{\rm simple}/B$ of 2.3.A.3.

With $C^2=2$ the criterion reads $\kappa>2\big(1+\frac{2-\eta h}{\eta h}\big)=\frac4{\eta h}$, that is $\kappa\eta h>4$ (a formal comparison: a quadratic is outside their scale-invariant setting). But the scaled run already diverges at $\kappa\eta h=2$. So in a frozen basin the criterion only fires after the run has blown up: true, but it adds nothing to 2.3.A.4. It is informative in their setting, where the equilibrium of a scale-invariant net with weight decay is not a frozen basin and $\mathcal N/\mathcal G$ is a cheap summary of it. The toy shows what $\mathcal N/\mathcal G$ tracks: learning rate times noise-weighted curvature, read off without computing $H$.

**3.** SVAG($l$) is SGD with step $\eta/l$ and gradient-noise covariance $l\Sigma/B$, so the per-step noise covariance is $(\eta/l)^2l\Sigma/B=\eta^2\Sigma/(lB)$. For one mode, $\Pi=(1-\frac{\eta h}l)^2\Pi+\frac{\eta^2\Sigma}{lB}$, so $\Pi\,\frac{\eta h}l\big(2-\frac{\eta h}l\big)=\frac{\eta^2\Sigma}{lB}$ and $\Pi=\frac{\eta\Sigma}{Bh(2-\eta h/l)}$. The excess loss is $\frac h2\Pi=\frac{\eta\Sigma}{4B}\big(1-\frac{\eta h}{2l}\big)^{-1}$. SVAG acts like SGD at the effective $\eta h/l$: the gap closes like $\frac{\eta h}{2l}$. At $\eta h=1.8$ the factor is $10$ at $l=1$, $1.8$ at $l=2$, $1.3$ at $l=4$ and $1.1$ at $l=8$, the pattern in Code 2.5.B.
"""
    ),
    markdown(
        r"""
**Code 2.5.B — SVAG on the least-squares problem.** Is SGD at its SDE limit at two
points of the family $\eta/B=0.01$ on the problem of Code 2.3.B:
$(\eta,B)=(0.16,16)$, where $\eta\lambda_{\max}\approx0.3$, and $(0.96,96)$, where
$\eta\lambda_{\max}\approx1.8$? For $l=1,2,4,8$ the cell divides SVAG's stationary
excess loss by the SDE's. "The SDE" keeps the state-dependent noise, so its floor
solves
$H\Pi+\Pi H=\frac\eta B\big(\Sigma^*+\frac1N\sum_iA_i\Pi A_i-H\Pi H\big)$, with
$A_i=x_ix_i^\top$ and $\Sigma^*=\Sigma$ the per-example gradient covariance at the
minimum (the continuous-time version of 2.3.A.6). It is about 4% above the
frozen-noise value $\frac{\eta}{4B}\mathrm{tr}\,\Sigma$.
"""
    ),
    code(
        r"""
def pi_sde_state(eta, B):
    # Stationary covariance of the SDE with the state-dependent least-squares noise (continuous 2.3.A.6).
    I = np.eye(d)
    A = np.kron(I, H) + np.kron(H, I) - eta / B * (K4 - np.kron(H, H))
    return np.linalg.solve(A, (eta / B * Sigma).ravel()).reshape(d, d)

def svag_floor(eta, B, l, R=100, burn_t=6, meas_t=10):
    # Stationary excess loss of SVAG(l): two independent batches combined with a_+, a_-, step eta/l.
    lr = eta / l; s = np.sqrt(2 * l - 1); a_plus, a_minus = (1 + s) / 2, (1 - s) / 2
    burn, total = int(burn_t / (hmin * lr)), int((burn_t + meas_t) / (hmin * lr))   # in SDE time units of 1/hmin
    W = np.tile(w_star, (R, 1)); acc, n = 0.0, 0
    for k in range(total):
        grads = []
        for _ in range(2):
            idx = rng.integers(0, N, (R, B)); Xb = X[idx]
            grads.append(np.einsum("rbi,rb->ri", Xb, np.einsum("rbi,ri->rb", Xb, W) - y[idx]) / B)
        W = W - lr * (a_plus * grads[0] + a_minus * grads[1])
        if k > burn and k % 5 == 0:
            dW = W - w_star; acc += 0.5 * np.einsum("ri,ij,rj->r", dW, H, dW).mean(); n += 1
    return acc / n

ls = (1, 2, 4, 8)
plt.figure(figsize=(5.5, 3.6))
for eta, B in ((0.16, 16), (0.96, 96)):
    ref = excess(pi_sde_state(eta, B))
    ratios = [svag_floor(eta, B, l) / ref for l in ls]
    plt.plot(ls, ratios, "o-", label=f"η={eta}, B={B}  (η·λmax={eta*hmax:.1f})")
    print(f"η·λmax = {eta*hmax:.2f}:  floor / SDE floor for l = {ls}:  " + ", ".join(f"{x:.2f}" for x in ratios))
plt.axhline(1, color="gray", ls="--", lw=0.8, label="SDE")
plt.xscale("log", base=2); plt.yscale("log"); plt.xticks(ls, [str(l) for l in ls])
plt.xlabel("SVAG parameter l  (l = 1 is SGD)"); plt.ylabel("stationary excess loss / SDE value")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
At $\eta\lambda_{\max}\approx0.3$ SGD is within about 10% of its SDE limit. At
$\eta\lambda_{\max}\approx1.8$ its floor is several times the SDE's, and SVAG walks it
down as $l$ grows: two runs of the "same" linear-scaling family are not both at
the SDE limit (2.5.A.3 gives the rate for one mode).
"""
    ),
    # ================================================================ 2.6
    markdown(
        r"""
<a id="td22-6"></a>

## 2.6 Adam and the square-root rule

For Adam and RMSprop the learning rate should scale like $\sqrt B$, not $B$, and
the moving-average parameters and $\epsilon$ must scale too.

**Paper:** Malladi, Lyu, Panigrahi and Arora, *On the SDEs and scaling rules for
adaptive gradient algorithms* (see [Readings](#td22-readings)).

RMSprop divides the minibatch gradient $g_B$ coordinatewise by $\sqrt v$, where $v$
is an exponential moving average (EMA) of $g_B^2$. When noise dominates,
$v\approx\mathrm{diag}(\Sigma)/B$ ($\Sigma$ the per-example gradient covariance), so the
update is roughly $-\eta\sqrt B\,\mathrm{diag}(\Sigma)^{-1/2}g_B$. Per step the drift is
$\propto\eta\sqrt B$ and the noise covariance $\propto\eta^2$, independent of $B$, so the
SDE temperature is $\eta/\sqrt B$: dividing by the noise scale already grows each
step like $\sqrt B$, and only the other $\sqrt B$ goes into the learning rate. Malladi et al. make this rigorous and get the
**square-root scaling rule** for $B\to\kappa B$:

$$
\eta\to\sqrt\kappa\,\eta,\qquad 1-\beta_i\to\kappa(1-\beta_i)\ \ (i=1,2),\qquad\epsilon\to\epsilon/\sqrt\kappa ,
$$

with $\beta_1,\beta_2$ Adam's EMA parameters and $\epsilon$ the constant added to $\sqrt v$.

<a id="td22-6-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Full gradient $G$; vector squares, roots and division are
coordinatewise. RMSprop steps $\theta\leftarrow\theta-\eta\,g/(\sqrt v+\epsilon)$, with $v$
an EMA of $g^2$ with parameter $\beta$; Adam adds an EMA of $g$ with parameter
$\beta_1$, and bias corrections.

> **Time-convention exception.** Elsewhere in this notebook $t=k\eta$. For RMSprop
> and Adam, Malladi et al.'s SDE uses $t=k\eta^2$. Durations in the two conventions
> are not comparable.

**Where the $\eta^2$ comes from.** In the noise-dominated regime
$v\approx G^2+\mathrm{diag}(\Sigma)/B\approx\mathrm{diag}(\Sigma)/B$. The per-step noise
covariance is then
$\eta^2B\cdot\mathrm{diag}(\Sigma)^{-1/2}\frac\Sigma B\,\mathrm{diag}(\Sigma)^{-1/2}\propto\eta^2$.
Matching to an SDE with time $\Delta t$ per step, the drift fixes
$\Delta t\propto\eta\sqrt B$, and the diffusion is $\eta^2/\Delta t\propto\eta/\sqrt B$.
Keeping it fixed under $B\to\kappa B$ requires $\eta\to\sqrt\kappa\,\eta$, and along
that family $\Delta t\propto\eta\sqrt B\propto\eta^2$.

**Malladi et al.'s version.** In a limit where $\eta\to0$ with the noise scale
growing like $1/\eta$, the EMA windows like $1/\eta^2$ steps and $\epsilon$ like
$1/\eta$, RMSprop and Adam (with their moment estimates) are weak order-1
approximated by SDEs in which one step is time $\eta^2$. Holding that limit fixed
under $B\to\kappa B$ gives the rule. The argument needs noise to dominate. In the
opposite regime $v\approx G^2$ and the update is close to sign descent
([`td2_4_optimizer_geometry`](td2_4_optimizer_geometry.html), whose §4.3.D also
covers where that picture fails).


**Exercise 2.6.A — Why a square root, and the floor.**

1. *(Malladi et al.'s warm-up.)* Take a linear loss $f(\theta)=\langle\theta,\bar g\rangle$ and isotropic noise, $g_k\sim\mathcal N(\bar g,\sigma^2I)$. Assume $v_k\approx\bar g^2+\sigma^2$ (concentrated) and $\sigma\gg\|\bar g\|$, and ignore $\epsilon$. Show that RMSprop gives, approximately, $\theta_k-\theta_0\sim\mathcal N\big(-k\eta\bar g/\sigma,\ k\eta^2I\big)$. What must happen to $\eta$ and $k$ to keep this law fixed when $\sigma\to\sigma/\sqrt\kappa$?
2. Why must $1-\beta$ scale like $\kappa$, and $\epsilon$ like $1/\sqrt\kappa$?
3. *(The floor.)* Assume all $\Sigma_{jj}>0$, and let $H\succ0$ be the Hessian of a quadratic basin. For preconditioned SGD $\theta\leftarrow\theta-\eta P(H\theta+\xi)$ with a fixed $P\succ0$ and $\mathrm{Cov}\,\xi=\Sigma/B$, the stationary covariance $\Pi$ of the OU model solves $PH\Pi+\Pi HP=\frac\eta BP\Sigma P$ (MDL [MDL1 §4](../notebooks/mdl1_statistical_learning.html#wb2-section-4), exercise 4.B.3, item 6, in this notebook's notation). Show that $\tfrac12\mathrm{tr}(H\Pi)=\frac{\eta}{4B}\mathrm{tr}(P\Sigma)$. With the noise-dominated RMSprop preconditioner $P=(\mathrm{diag}\,\Sigma/B)^{-1/2}$, show the floor is $\frac{\eta}{4\sqrt B}\sum_j\sqrt{\Sigma_{jj}}$. What happens to it under the square-root rule, and under the linear rule?
4. *(The exponent.)* Consider updates $\theta\leftarrow\theta-\eta\,(\mathrm{diag}\,\Sigma/B)^{-p}g_B$ with a fixed power $p\ge0$: $p=0$ is SGD and $p=\tfrac12$ is noise-dominated RMSprop. Using item 3, show that the floor depends on $(\eta,B)$ only through $\eta B^{p-1}$, so the scaling rule is $\eta\propto B^{1-p}$. Check that along this rule the number of steps per unit of drift scales like $1/B$, so the number of examples is unchanged for every $p$.

<details><summary>Hint</summary>

For 1, the update is $-\eta g_k/\sigma$ and the increments are independent
Gaussians. For 2, an EMA with parameter $\beta$ averages over about $1/(1-\beta)$
steps, and one step is SDE time $\eta^2$. For 3, multiply the Lyapunov equation on the left by $P^{-1}$ and
take the trace.
</details>
"""
    ),
    answer(
        r"""
**1.** With $v_k\approx\bar g^2+\sigma^2\approx\sigma^2$, the update is $\theta_{k+1}-\theta_k\approx-\eta g_k/\sigma\sim\mathcal N(-\eta\bar g/\sigma,\ \eta^2I)$, independently at each step. Summing $k$ of them gives mean $-k\eta\bar g/\sigma$ and covariance $k\eta^2I$. For this law not to change when $\sigma$ changes, both $k\eta/\sigma$ and $k\eta^2$ must be fixed. From $k\eta^2$ fixed, $k\propto1/\eta^2$; then $k\eta/\sigma\propto1/(\eta\sigma)$ fixed means $\eta\propto1/\sigma$. With $\sigma\propto1/\sqrt B$: $\eta\to\sqrt\kappa\,\eta$ and $k\to k/\kappa$, so the number of examples $kB$ is unchanged. The fixed quantity $k\eta^2$ is "time" in the $t=k\eta^2$ convention.

**2.** The EMA $v$ averages over about $1/(1-\beta)$ steps. For the adaptive part to look the same in SDE time, that window must cover a fixed time, i.e. $(1-\beta)^{-1}\eta^2$ is fixed, so $1-\beta\propto\eta^2\propto\kappa$. The same holds for $\beta_1$. The term $\epsilon$ is added to $\sqrt v\approx\sigma\sqrt{\mathrm{diag}\,\Sigma}\propto1/\sqrt B$; to keep its relative size fixed, $\epsilon\to\epsilon/\sqrt\kappa$. (Equivalently: $c_i=(1-\beta_i)/\eta^2$ and $\epsilon_0=\epsilon\eta$ are held fixed.)

**3.** Multiplying $PH\Pi+\Pi HP=\frac\eta BP\Sigma P$ on the left by $P^{-1}$ gives $H\Pi+P^{-1}\Pi HP=\frac\eta B\Sigma P$. The trace of $P^{-1}(\Pi H)P$ equals $\mathrm{tr}(H\Pi)$, so $2\,\mathrm{tr}(H\Pi)=\frac\eta B\mathrm{tr}(\Sigma P)$, which is the claim. With $P=\sqrt B\,\mathrm{diag}(\Sigma)^{-1/2}$, $\mathrm{tr}(P\Sigma)=\sqrt B\sum_j\sqrt{\Sigma_{jj}}$, so the floor is $\frac{\eta}{4\sqrt B}\sum_j\sqrt{\Sigma_{jj}}$. Under the square-root rule $\eta/\sqrt B$ is fixed, so the floor is unchanged. Under the linear rule $\eta/\sqrt B\to\sqrt\kappa\,\eta/\sqrt B$: the floor rises by $\sqrt\kappa$. (For Adam's momentum, see the note before the demo below.)

**4.** With $P=(\mathrm{diag}\,\Sigma/B)^{-p}=B^p\,\mathrm{diag}(\Sigma)^{-p}$, item 3 gives the floor $\frac{\eta}{4B}\mathrm{tr}(P\Sigma)=\frac{\eta B^{p-1}}4\sum_j\Sigma_{jj}^{1-p}$. More generally, the OU process has drift $-\eta B^pD_p H\theta$ and noise covariance $\eta^2B^{2p}D_p\frac\Sigma BD_p$ per step (with $D_p=\mathrm{diag}(\Sigma)^{-p}$). Taking $\Delta t=\eta B^p$ per step makes the drift rate $-D_pH\theta$, independent of $(\eta,B)$, and the diffusion $\eta B^{p-1}D_p\Sigma D_p$. So everything depends on $\eta B^{p-1}$ alone, and $B\to\kappa B$ requires $\eta\to\kappa^{1-p}\eta$: linear for $p=0$, square-root for $p=\frac12$, no change for $p=1$. The steps per unit time are $1/\Delta t=1/(\eta B^p)\propto1/B$, so the number of examples is invariant for every $p$.
"""
    ),
    markdown(
        r"""
**Code 2.6.B — The square-root rule on a noisy quadratic.** Adam on a
12-dimensional noisy quadratic (curvatures from 1 to 0.05, per-example noise
variances from 0.5 to 4) at base batch size 4, with the batch scaled by $\kappa=4$
and $16$ under each rule. The dashed lines are the floors of 2.6.A.3, derived
without momentum; momentum leaves them unchanged when each mode relaxes slowly
compared with its window of about $1/(1-\beta_1)$ steps. The linear-rule runs at
large $\kappa$ violate this, yet land within about 10% of the prediction.
"""
    ),
    code(
        r"""
dq = 12
hq = np.geomspace(1.0, 0.05, dq)                  # curvatures
sq = np.geomspace(0.5, 4.0, dq)                   # per-example gradient-noise variances (diagonal Sigma)

def adam_run(eta, B, b1, b2, eps, examples, R=32, theta0=1.0):
    th = np.full((R, dq), theta0); m = np.zeros((R, dq)); v = np.zeros((R, dq)); out = []
    for k in range(1, examples // B + 1):
        g = hq * th + np.sqrt(sq / B) * rng.standard_normal((R, dq))
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        th -= eta * (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + eps)
        out.append(0.5 * (hq * th**2).sum(1).mean())
    return np.arange(1, len(out) + 1) * B, np.array(out)

base = dict(eta=2e-3, B=4, b1=0.99, b2=0.999, eps=1e-8)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, rule in zip(axes, ("square-root rule", "linear rule")):
    for kap in (1, 4, 16):
        if rule == "square-root rule":
            hp = dict(eta=base["eta"] * np.sqrt(kap), b1=1 - kap * (1 - base["b1"]),
                      b2=1 - kap * (1 - base["b2"]), eps=base["eps"] / np.sqrt(kap))
        else:
            hp = dict(eta=base["eta"] * kap, b1=base["b1"], b2=base["b2"], eps=base["eps"])
        ex, L = adam_run(B=base["B"] * kap, examples=120_000, **hp)
        line, = ax.semilogy(ex, L, lw=1, label=f"κ={kap}: B={base['B']*kap}, η={hp['eta']:.4f}")
        ax.axhline(hp["eta"] / (4 * np.sqrt(base["B"] * kap)) * np.sqrt(sq).sum(), ls="--", lw=0.8,
                   color=line.get_color())
    ax.set_title(f"Adam, {rule}"); ax.set_xlabel("examples processed"); ax.legend(fontsize=8)
axes[0].set_ylabel("loss   (dashed: η Σ_j √Σ_jj / 4√B)")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
Under the square-root rule the runs follow nearly the same curve per example and
settle on one floor. Under the linear rule the floors separate by factors of 2 and
4 ($=\sqrt\kappa$). The early transient is not perfectly matched even under the
square-root rule: there the gradient is large compared with the noise in some
coordinates (discretization and bias corrections may also contribute).
"""
    ),
    # ================================================================ 2.7
    markdown(
        r"""
<a id="td22-valid"></a>

## 2.7 When is the SDE picture valid?

Everything above expands in the step size $\eta$, which the theorems justify only
as $\eta\to0$ on a fixed time horizon. Whether it is accurate at a practical
learning rate has to be checked. Below, $h$ is a Hessian eigenvalue and
$\lambda_{\max}$ the largest. ([`td2_1_flows`, §1.3](td2_1_flows.html#td21-s3) asks the same of the deterministic expansion.)

**Paper:** [Li, Malladi and Arora](https://arxiv.org/abs/2102.12470) (see [Readings](#td22-readings)), for SVAG and the evidence; [Smith, Dherin, Barrett and De](https://arxiv.org/abs/2101.12176) (§2.1), §4, for item 4.

1. **The bounds are not quantitative.** Weak order $p$ means error $C\eta^p$ with $C$ not estimated (§2.2).
2. **$\eta h$ is not small in the top directions.** In a basin the floor is 50% above the SDE value at $\eta\lambda_{\max}\approx0.9$ (Code 2.3.B), and the order-2 SME gets the sign of that correction wrong (2.3.A.4). Under full-batch GD networks sit at the edge of stability, $\eta\lambda_{\max}\approx2$, where a [central flow](td2_3_edge_of_stability.html#td23-3) (the flow of the trajectory averaged over the fast oscillation), not gradient flow, models the deterministic part; under SGD a [batch sharpness](td2_3_edge_of_stability.html#td23-6) plays this role.
3. **Long times and rare events.** Flat modes need about $1/(\eta h)$ steps to equilibrate, possibly longer than training. For isotropic Langevin dynamics, escape rates depend exponentially on barrier over temperature, so an $O(\eta h)$ relative error in the effective temperature (2.3.A.4) can change them by large factors; for anisotropic, state-dependent noise the relevant barrier is a quasipotential (the least-cost noise-driven path out of the basin; ask your agent), not the loss barrier. Arguments about which minima SGD "prefers" live here, where fixed-horizon bounds say nothing.
4. **The finite-$\eta$ drift may matter more than the noise.** Smith et al. ([`td2_1_flows`, §1.4](td2_1_flows.html#td21-s4)) argue, holding the temperature fixed while changing higher-order terms, that SGD's generalization benefit comes from the finite-$\eta$ drift corrections, not the SDE temperature. On that view the order-1 SME keeps the wrong small term.
5. **The one-step model of §2.4** assumes a roughly constant noise scale (Code 2.4.B saw $B_{\rm noise}$ grow twenty- to fortyfold) and ignores third-order terms.
6. **Reshuffling.** The theorems assume i.i.d. sampling; with reshuffling, slow directions fluctuate less than the SDE predicts (2.1.A.2, Code 2.3.B).
7. **Heavy tails.** [Şimşekli, Sagun and Gürbüzbalaban](https://arxiv.org/abs/1901.06053) argued minibatch noise is heavy-tailed, which would give a Lévy-driven limit. Li, Malladi and Arora argue that the measurement can mislead, and find that SGD and Gaussian noisy GD of matched covariance reach the same CIFAR-10 accuracy (89.4% vs 89.3%).
8. **Smoothness.** ReLU kinks, and normalization layers near the origin, violate the theorems' regularity assumptions.
9. **It can be tested.** SVAG (§2.5) checks whether a run is at its SDE limit. In Li, Malladi and Arora's CIFAR-10 runs it was at small and moderate batch sizes, and was not once the linear scaling rule broke.
"""
    ),
]

raise SystemExit(run(CELLS, "td2_2_sgd_sde", "td2_2_sgd_sde"))
