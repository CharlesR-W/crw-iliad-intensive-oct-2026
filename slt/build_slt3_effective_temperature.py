#!/usr/bin/env python3
"""Build SLT Notebook 3, "Effective exponents and temperature" (slt3_effective_temperature).

Shape (SLT_SPECS_v1.md, 2026-10-09): question, keystone reading, route; each `##`
section opens with its question and restates its notation, then default text,
exercises with worked solutions, and small CPU code cells. Sections: 3.1 the
pathology (lambda is defined only asymptotically and only for exact degeneracy;
the effective exponent and its crossover); 3.2 a marked STUB for CRW's
"effective theory / why temperature" argument; 3.3 sampling at temperature (the
WBIC, the local estimator, SGLD as the sampler, hyperparameters read as scales).

The SGLD update is the one in Furman's intro_lc notebook (Lau et al. 2023's
modification of Welling and Teh), rewritten in numpy with the same equation; the
examples follow devinterp's normal_crossing.ipynb and Furman's notebook.
Numerical checks: checks/slt3_checks.py.
"""
from nbkit_slt import answer, book, code, code_answer, markdown, reading, references, run

CELLS = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# SLT Notebook 3 — Effective exponents and temperature

**Singular Learning Theory, Iliad Intensive 2026**

*Question:* the learning coefficient $\lambda$ is defined only as $n\to\infty$,
and only for exactly degenerate losses. What is actually measurable?

Notebook 2 defined $\lambda$ from how the low-loss volume vanishes as the loss
tolerance goes to zero. Two things are wrong with that as a definition of
something to measure. A loss that is degenerate at one point and regular
everywhere nearby has $\lambda$ jump discontinuously at that point, and a
finite sample cannot tell the two apart. And nothing at finite $n$ is
asymptotic. This notebook replaces the limit by a quantity at a scale: the
*effective exponent* $\lambda_{\rm eff}(n)=n\,\mathbb E_{\pi_n}[K]$, which is
what the posterior at sample size $n$ actually sees, crosses over between the
exponents of the nearby and the exact loss at a calculable $n$ (§3.1). The
same quantity at inverse temperature $\beta$ is Watanabe's widely applicable
BIC and, with a localizing term, Lau et al.'s local learning coefficient
estimator, sampled by stochastic gradient Langevin dynamics (§3.3). The
estimator's hyperparameters are then scales, not nuisances.
"""
        + reading(
            "The Local Learning Coefficient: A Singularity-Aware Complexity Measure",
            "https://arxiv.org/abs/2308.12108",
            "Lau, Furman, Wang, Murfet & Wei (2023)",
            "The local learning coefficient and its SGLD estimator $\\hat\\lambda=n\\beta(\\mathbb E_\\beta[L_n]-L_n(w^*))$: read Sections 3–4 for the estimator and the normal-crossing and deep-linear-network checks that the code below reproduces at small scale.",
            "The hyperparameter discussion (Section 5 and the appendix) is §3.3's last topic.",
            importance=5,
            length=3,
        )
        + reading(
            "Intro to LC estimation",
            "https://github.com/zfurman56/intro-lc-estimation",
            "Furman (notebook, 2024)",
            "A from-scratch walk through the same estimator: volume scaling, the WBIC as a moment of the density of states, the infinite-data and finite-data settings, and a minimal SGLD implementation, which the code below follows.",
            importance=4,
            length=2,
        )
        + r"""
**Route.**

- [3.1 The pathology](#slt3-s1): $K=w^4+\delta w^2$ has $\lambda=\tfrac12$ for every $\delta>0$ and $\tfrac14$ at $\delta=0$; the effective exponent $\lambda_{\rm eff}(n)$ and its crossover at $n\sim\delta^{-2}$.
- [3.2 Effective theory, and why you need temperature](#slt3-s2): **stub**, to be supplied by CRW.
- [3.3 Sampling at temperature](#slt3-s3): the WBIC, the local estimator, SGLD, and hyperparameters read as scales. Code: the 1-D model swept over $\beta$, normal crossings, a two-layer linear network.

**Setup, repeated.** $K(w)=L(w)-L(w_0)\ge0$ population excess loss,
$K_n=L_n-L_n(w_0)$ its sample version; $\varphi$ prior;
$\bar Z_n=\int\varphi e^{-nK}$, $\bar F_n=-\log\bar Z_n$;
$V(t)$ the prior volume of $\{K<t\}$, with $V(t)\sim ct^\lambda(-\log t)^{m-1}$
defining $\lambda$ and $m$ (Notebook 2 §2.1). Tempered posterior
$\pi_{n,\beta}\propto\varphi e^{-\beta nL_n}$ (Notebook 1 §1.1), expectation
$\mathbb E_\beta[\cdot]$.
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(0)


def sgld_lambda(grad_minibatch, loss_full, w_star, n, batch, nbeta, gamma, eps, steps, burn, chains=3, rng=rng):
    # Local learning coefficient estimate by SGLD (Lau et al. 2023; the update in Furman's intro_lc notebook):
    #   w <- w - (eps/2) [ n*beta * (minibatch mean gradient of the per-sample loss) + gamma (w - w*) ] + sqrt(eps) * N(0, I)
    #   lambda_hat = n*beta * ( mean over the chain of L_n(w) - L_n(w*) )
    # grad_minibatch(w, idx) returns the mean gradient of the per-sample loss over the minibatch idx;
    # loss_full(w) returns L_n(w) on the whole data set.  Returns the per-chain estimates.
    L_star = loss_full(w_star)
    ests = []
    for _ in range(chains):
        w = np.array(w_star, float).copy()
        acc = 0.0
        for s in range(steps):
            idx = rng.integers(0, n, batch)
            g = grad_minibatch(w, idx)
            w = w - 0.5 * eps * (nbeta * g + gamma * (w - w_star)) + np.sqrt(eps) * rng.standard_normal(w.shape)
            if s >= burn:
                acc += loss_full(w)
        ests.append(nbeta * (acc / (steps - burn) - L_star))
    return np.array(ests)
"""
    ),
    # ------------------------------------------------------------------ 3.1
    markdown(
        r"""
<a id="slt3-s1"></a>

## 3.1 The pathology

*Question:* what does the posterior see at sample size $n$ when the loss is
nearly, but not exactly, degenerate?

*Notation:* $K$, $\bar Z_n$, $V(t)$, $\lambda$ as above. New: the *effective
exponent*

$$
\lambda_{\rm eff}(n)=-\frac{d\log\bar Z_n}{d\log n}=n\,\mathbb E_{\pi_n}[K],
$$

the posterior mean excess loss in units of $1/n$ (the identity is Notebook 2 Exercise
2.3.3). In a single basin $\lambda_{\rm eff}(n)\to\lambda$ as $n\to\infty$, so
$\lambda_{\rm eff}$ is the finite-$n$ object of which $\lambda$ is the limit.

Take $K(w)=w^4+\delta w^2$ with one parameter. For every $\delta>0$ the
minimum is regular ($K''(0)=2\delta>0$) and $\lambda=\tfrac12$; at $\delta=0$,
$\lambda=\tfrac14$. The learning coefficient is discontinuous in $\delta$, and
the discontinuity sits on a set of measure zero. Any real loss with a nearly
flat direction is of the first kind, and a finite sample, which fixes the loss
only to accuracy $O(n^{-1/2})$, cannot tell whether $\delta$ is zero. So the
question "what is $\lambda$?" has an answer that no experiment can determine.

The effective exponent resolves this. Rescaling $w=n^{-1/4}u$ in
$\bar Z_n=\int e^{-n(w^4+\delta w^2)}dw$ gives

$$
\bar Z_n\approx n^{-1/4}\,g(\delta\sqrt n),\qquad g(s)=\int e^{-u^4-su^2}du,
$$

so everything depends on the single combination $s=\delta\sqrt n$. For
$s\ll1$ the quartic term dominates and $\lambda_{\rm eff}\approx\tfrac14$; for
$s\gg1$ the quadratic dominates and $\lambda_{\rm eff}\to\tfrac12$. The
crossover is at $n\sim\delta^{-2}$: the posterior sees the degenerate exponent
until it has enough data to resolve the curvature $\delta$, which costs
$n\sim1/\delta^2$ samples. Below that $n$ the two losses are, for every purpose
Bayes cares about, the same loss. The asymptotic $\lambda$ is a statement about
the regime $n\gg\delta^{-2}$, and for a loss with many nearly flat directions of
different $\delta$ there is a hierarchy of such crossovers; what one measures at
a given $n$ is $\lambda_{\rm eff}(n)$, an exponent at a resolution scale.

**Exercise 3.1 (The effective exponent).**

1. Compute $\lambda$ (and $m$) for $K=w^4+\delta w^2$ at $\delta=0$ and at $\delta>0$, from $V(t)$ or from $\zeta$.
2. Verify $\lambda_{\rm eff}(n)=n\,\mathbb E_{\pi_n}[K]$ and show that in a single basin with $V(t)\sim ct^\lambda$, $\lambda_{\rm eff}(n)\to\lambda$.
3. Derive the scaling form $\bar Z_n\approx n^{-1/4}g(\delta\sqrt n)$ (take the prior constant near $0$), and from it $\lambda_{\rm eff}(n)=\tfrac14+\tfrac12\,s\,(-\log g)'(s)$ at $s=\delta\sqrt n$. Find $g(s)$ for $s\to\infty$ and check that $\lambda_{\rm eff}\to\tfrac12$. Where is the crossover?
4. Let the loss come from data: $y=\varepsilon$ (pure noise, variance $\sigma^2$), inputs $x=(x_1,x_2)$ standard normal, model $f_w(x)=w^2x_1+\sqrt\delta\,w\,x_2$, squared loss $\ell=\tfrac12(f_w(x)-y)^2$. Show $K(w)=\tfrac12(w^4+\delta w^2)$, and that the sample loss is $K_n(w)=\tfrac12(a_nw^4+2b_n\sqrt\delta\,w^3+c_n\delta w^2)-w^2u_n-\sqrt\delta\,w\,v_n$ with $a_n,c_n=1+O(n^{-1/2})$, $b_n=O(n^{-1/2})$ and $u_n,v_n=O(\sigma n^{-1/2})$. Conclude that even at $\delta=0$ the sample loss has a quadratic term of random size $O(n^{-1/2})$, so a sample of size $n$ cannot distinguish $\delta$ from $0$ when $\delta\lesssim n^{-1/2}$, which is the same threshold as in part 3.
"""
    ),
    answer(
        r"""
**1.** $\delta=0$: $V(t)\propto t^{1/4}$, $\lambda=\tfrac14$, $m=1$. $\delta>0$: near $0$, $K\approx\delta w^2$, so $\{K<t\}\approx(-\sqrt{t/\delta},\sqrt{t/\delta})$ and $V\propto t^{1/2}$: $\lambda=\tfrac12$, $m=1$. Via $\zeta$: $\int(w^4+\delta w^2)^z\,dw=\int w^{2z}(w^2+\delta)^zdw$, and near $w=0$ the second factor is $\delta^z$, a constant, so the first pole is that of $\int w^{2z}$, at $-\tfrac12$. The constant $c$, by contrast, is $\propto\delta^{-1/2}$ and blows up as $\delta\to0$: the discontinuity in $\lambda$ is accompanied by a divergence in the prefactor, which is how a continuous $\bar F_n$ can have a discontinuous limit of $\bar F_n/\log n$.

**2.** $\frac{d\log\bar Z_n}{d\log n}=n\frac{d\log\bar Z_n}{dn}=n\frac{\int(-K)\varphi e^{-nK}}{\int\varphi e^{-nK}}=-n\,\mathbb E_{\pi_n}[K]$. In a single basin $\bar Z_n\approx c\Gamma(\lambda+1)n^{-\lambda}$ (Notebook 2 Exercise 2.1.2), so $-d\log\bar Z_n/d\log n\to\lambda$.

**3.** With $w=n^{-1/4}u$, $dw=n^{-1/4}du$ and $n(w^4+\delta w^2)=u^4+\delta\sqrt n\,u^2$, so $\bar Z_n=\varphi(0)n^{-1/4}g(\delta\sqrt n)$ up to the prior's variation. Then $\log\bar Z_n=\text{const}-\tfrac14\log n+\log g(s)$ with $s=\delta n^{1/2}$, and $\frac{d}{d\log n}\log g(s)=\frac{ds}{d\log n}\frac{g'}{g}=\tfrac12s\frac{g'(s)}{g(s)}$, giving $\lambda_{\rm eff}=\tfrac14-\tfrac12s(\log g)'(s)=\tfrac14+\tfrac12s(-\log g)'(s)$. At $s=0$, $g'(0)=-\int u^2e^{-u^4}du$ is finite, so $\lambda_{\rm eff}=\tfrac14+O(s)$. For $s\to\infty$ the Gaussian factor confines $u$ to $|u|\lesssim s^{-1/2}$, where $u^4$ is negligible, so $g(s)\approx\int e^{-su^2}du=\sqrt{\pi/s}$, $(-\log g)'=\frac1{2s}$, and $\lambda_{\rm eff}\to\tfrac14+\tfrac14=\tfrac12$. The crossover is at $s\sim1$, i.e. $n\sim\delta^{-2}$, and the whole curve $\lambda_{\rm eff}$ against $\delta^2n$ is the same for every $\delta$ (the code below checks this collapse).

**4.** $K(w)=\tfrac12\mathbb E[(w^2x_1+\sqrt\delta wx_2-\varepsilon)^2]-\tfrac12\mathbb E[\varepsilon^2]=\tfrac12(w^4\mathbb E x_1^2+\delta w^2\mathbb Ex_2^2+2\sqrt\delta w^3\mathbb E[x_1x_2])=\tfrac12(w^4+\delta w^2)$, using $\mathbb Ex_1^2=\mathbb Ex_2^2=1$, $\mathbb E[x_1x_2]=0$ and $\mathbb E[x_i\varepsilon]=0$. Replacing expectations by sample averages: $a_n=\overline{x_1^2}$, $c_n=\overline{x_2^2}$, $b_n=\overline{x_1x_2}$, $u_n=\overline{x_1\varepsilon}$, $v_n=\overline{x_2\varepsilon}$, each an average of $n$ i.i.d. terms, so $a_n,c_n=1+O(n^{-1/2})$ and $b_n,u_n,v_n=O(n^{-1/2})$ (with $u_n,v_n\propto\sigma$). At $\delta=0$, $K_n(w)=\tfrac12a_nw^4-u_nw^2$: a quartic plus a quadratic of random coefficient $-u_n$ of size $\sigma/\sqrt n$, which is negative half the time (a double well with minima at $w^2=u_n/a_n$) and positive the other half (a regular minimum). Either way the sample loss is not $w^4$ at scales below $|u_n|\sim n^{-1/2}$: a sample of size $n$ fixes the quadratic coefficient only to $\pm n^{-1/2}$, so $\delta$ below that is invisible, exactly the $s=\delta\sqrt n\lesssim1$ regime of part 3. The two statements are the same statement: $\lambda_{\rm eff}(n)$ is the exponent at the resolution the data provide, and asking for $\lambda$ is asking for infinite resolution.
"""
    ),
    markdown(
        r"""
**Code: the crossover.** $\lambda_{\rm eff}(n)=n\,\mathbb E_{\pi_n}[K]$ by
quadrature for $K=w^4+\delta w^2$ and several $\delta$, against $n$ and against
$\delta^2n$.
"""
    ),
    code(
        r"""
w = np.linspace(-2, 2, 400_001)
dw = w[1] - w[0]
ns = np.logspace(0, 7, 57)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for delta in [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]:
    K = w**4 + delta * w**2
    lam_eff = []
    for n in ns:
        logp = -n * K
        p = np.exp(logp - logp.max()); p /= p.sum()
        lam_eff.append(n * (p * K).sum())
    lam_eff = np.array(lam_eff)
    ax[0].semilogx(ns, lam_eff, label=rf"$\delta={delta}$")
    if delta > 0:
        ax[1].semilogx(delta**2 * ns, lam_eff, label=rf"$\delta={delta}$")
for a in ax:
    a.axhline(0.25, color="k", ls=":"); a.axhline(0.5, color="k", ls=":"); a.set_ylabel(r"$\lambda_{\rm eff}(n)$")
ax[0].set_xlabel("n"); ax[0].legend(fontsize=8); ax[1].set_xlabel(r"$\delta^2 n$"); ax[1].set_title("the same curves against $\\delta^2 n$")
plt.tight_layout(); plt.show()
"""
    ),
    # ------------------------------------------------------------------ 3.2 (stub)
    markdown(
        r"""
<a id="slt3-s2"></a>

## 3.2 Effective theory, and why you need temperature

**STUB: CRW supplies the argument.** This section is a placeholder and is
not to be drafted by anyone else.

*Scope as planned:* §3.1 showed that the measurable object is an exponent at a
resolution scale, $\lambda_{\rm eff}$, not the asymptotic $\lambda$. The
argument to be supplied here says what plays the role of the scale and why
the inverse temperature $\beta$ (equivalently, the trade between $n$ and
$\beta$ in $\pi_{n,\beta}\propto\varphi e^{-\beta nL_n}$) is the knob that sets
it, so that §3.3's estimator is read as measuring $\lambda$ at the scale
$\beta$ selects. Use "effective singularity" only if unavoidable.
"""
    ),
    # ------------------------------------------------------------------ 3.3
    markdown(
        r"""
<a id="slt3-s3"></a>

## 3.3 Sampling at temperature

*Question:* how does one measure $\lambda_{\rm eff}$ for a loss given only by
data and a gradient oracle?

*Notation:* $L_n$ training loss on $n$ samples; $w^*$ a minimizer of $L_n$ in
the basin of interest (in practice, the trained parameter); tempered posterior
$\pi_{n,\beta}\propto\varphi e^{-\beta nL_n}$ with expectation $\mathbb E_\beta$;
$\bar Z_n(\beta)=\int\varphi e^{-\beta nK}$. New: a *localizing* parameter
$\gamma>0$, a minibatch size $b$, and a step size $\epsilon$.

**From the free energy to a posterior average.** In §3.1 the measurable
quantity was $n\,\mathbb E_{\pi_n}[K]=-d\log\bar Z_n/d\log n$. Changing $n$ is
not an option with a fixed data set, but by Notebook 1 Exercise 1.1.4 the same
derivative can be taken with respect to the temperature knob:
$-\partial_\beta\log\bar Z_n(\beta)=n\,\mathbb E_\beta[K_n]$. In a single basin
Notebook 2's volume expansion gives $\bar Z_n(\beta)\approx c\Gamma(\lambda+1)(\beta n)^{-\lambda}$,
so (Exercise 3.3.1)

$$
\mathbb E_\beta[nK_n]\approx\frac\lambda\beta,
\qquad\text{i.e.}\qquad
\lambda\approx\beta\,\mathbb E_\beta[nK_n]=n\beta\big(\mathbb E_\beta[L_n]-L_n(w^*)\big).
$$

At the particular value $\beta=1/\log n$ the left side is $\lambda\log n$,
which is the $\log n$ term of the free energy: $\mathbb E_{1/\log n}[nL_n]\approx
nL_n(w^*)+\lambda\log n\approx F_n$. This is Watanabe's *widely applicable
Bayesian information criterion* (WBIC): a posterior average at one specific
temperature reproduces the free energy's leading terms without knowing
$\lambda$, which is what makes it a usable model-selection criterion. Read the
other way, the same identity *measures* $\lambda$: the right-hand side is Lau
et al.'s *local learning coefficient estimator* $\hat\lambda$, with two
additions. The expectation is taken over the posterior restricted to the basin
of $w^*$ by adding $\tfrac\gamma2\|w-w^*\|^2$ to $\beta nL_n$ (so the chain
cannot wander to other basins, and the loss is measured relative to the local
minimum), and the expectation is computed by a sampler.

**SGLD as the sampler.** Stochastic gradient Langevin dynamics is gradient
descent on the energy $\beta nL_n(w)+\tfrac\gamma2\|w-w^*\|^2$ with Gaussian noise
added so that the chain's stationary distribution is the Boltzmann
distribution of that energy, and with the full gradient replaced by a
minibatch estimate:

$$
w\leftarrow w-\frac\epsilon2\Big(\beta n\,\hat\nabla L_n(w)+\gamma(w-w^*)\Big)+\sqrt\epsilon\,\xi,
\qquad \xi\sim\mathcal N(0,I),
$$

where $\hat\nabla L_n$ is the mean gradient over a minibatch of $b$ samples.
Without the minibatch approximation and as $\epsilon\to0$ this is the
Langevin equation, whose stationary density is $\propto e^{-E(w)}$ for energy
$E$ (the noise variance $\epsilon$ per step and the drift $-\tfrac\epsilon2\nabla E$
are tuned to each other for exactly this). With a finite $\epsilon$ and
minibatch noise it is biased, and there is no guarantee that it has mixed; it
is used because it costs one minibatch gradient per step, the same as a
training step, and works empirically. The estimate is then the average of
$L_n$ along the chain after a burn-in, minus $L_n(w^*)$, times $n\beta$.

**Hyperparameters as scales.** The estimator has four knobs, and each of them
sets a scale rather than being a nuisance to be tuned away.
$\beta$ sets the loss scale probed: the chain sits at excess loss
$\sim\lambda/(n\beta)$, so by §3.1 it measures $\lambda_{\rm eff}$ at resolution
$n\beta$, not $\lambda$; sweeping $\beta$ sweeps the resolution, and for
$K=w^4+\delta w^2$ the estimate moves from $\tfrac14$ to $\tfrac12$ as $n\beta$
passes $\delta^{-2}$.
$\gamma$ sets a length scale $\gamma^{-1/2}$ around $w^*$: directions in which
the loss is flat over longer distances than that are cut off, which lowers the
estimate; as $n\beta\to\infty$ at fixed $\gamma$ the posterior shrinks inside
that radius and $\gamma$ drops out, but at finite $n\beta$ it does not.
$\epsilon$ must be small compared with $2/(n\beta\Lambda)$ for $\Lambda$ the
largest curvature the chain meets, or the chain is unstable in that direction;
too small and it does not mix. The minibatch size $b$ sets the gradient noise,
which acts as an extra temperature. The code below shows the first two
dependences on examples where the answer is known.

**Exercise 3.3 (The estimator).**

1. Derive $\mathbb E_\beta[nK_n]\approx\lambda/\beta$ from Notebook 1 Exercise 1.1.4 and Notebook 2 §2.1, and say what the $-(m-1)\log\log n$ term does to it. Explain why $\beta=1/\log n$ is the temperature at which a posterior average reproduces the free energy's leading terms, and why that is useful for model selection.
2. Show that $\hat\lambda=n\beta(\mathbb E_\beta[L_n]-L_n(w^*))$ reduces to $\beta\,\mathbb E_\beta[nK_n]$ when $w^*$ minimizes $L_n$ in the basin, and that with the localizing term the expectation is over $\propto\varphi e^{-\beta nL_n-\gamma\|w-w^*\|^2/2}$. Using the scaling of §3.1, show that for $K\sim\|w-w^*\|^{2k}$ the localizing factor is $1+O\big(\gamma(n\beta)^{-1/k}\big)$ on the region that matters, so $\gamma$ drops out as $n\beta\to\infty$.
3. *(Predict before running.)* The code below takes the model of Exercise 3.1.4 with $n=1000$, $\sigma=\tfrac12$, $\delta\in\{0,0.03,0.1\}$ and $\beta\in\{0.01,0.03,0.1,0.3,1\}$. Using $K=\tfrac12(w^4+\delta w^2)$ and the rescaling of §3.1 (with $n\to n\beta$ and the factor $\tfrac12$), predict for each $(\delta,\beta)$ whether $\hat\lambda$ should be near $\tfrac14$, near $\tfrac12$, or in between. Then say what the finite-sample floor of Exercise 3.1.4 does at $\delta=0$ and large $\beta$.
4. *(Scale dependence through $\gamma$.)* For a regular basin with Hessian eigenvalues $h_1,\dots,h_d$ of $K$ and a Gaussian localizer, compute $\hat\lambda$ exactly: $\hat\lambda=\tfrac12\sum_i\frac{n\beta h_i}{n\beta h_i+\gamma}$. Interpret it as a count of directions, and say what it does as $\gamma\to0$ and $\gamma\to\infty$. <details><summary>Hint</summary>Both the energy and the localizer are quadratic, so the chain samples a Gaussian with precision $n\beta H+\gamma I$ and $\mathbb E[nK_n]=\tfrac12\mathrm{tr}(nH(n\beta H+\gamma I)^{-1})$.</details>
"""
    ),
    answer(
        r"""
**1.** Notebook 1 Exercise 1.1.4 with $K_n$ in place of $L_n$: $-\partial_\beta\log\bar Z_n(\beta)=n\,\mathbb E_\beta[K_n]$. With $\bar Z_n(\beta)\approx c\Gamma(\lambda+1)(\beta n)^{-\lambda}(\log\beta n)^{m-1}$ (Notebook 2 §2.1 with $n\to\beta n$), $-\partial_\beta\log\bar Z_n=\frac\lambda\beta-\frac{m-1}{\beta\log\beta n}$: the multiplicity gives a correction of relative size $1/\log(\beta n)$, which is why estimates for $m=2$ losses come out low at any practical $n\beta$ (visible in the normal-crossing code below). At $\beta=1/\log n$, $\mathbb E_\beta[nK_n]\approx\lambda\log n$, so $\mathbb E_\beta[nL_n]=nL_n(w^*)+\mathbb E_\beta[nK_n]\approx nL_n(w^*)+\lambda\log n$, the two leading terms of $F_n$ (Notebook 2 §2.1 with the $O(1)$ dropped). No other $\beta$ does this: at $\beta=1$ the average is $nL_n(w^*)+\lambda$, missing the $\log n$. For model selection one wants $F_n$ for each candidate model (Notebook 1 §1.3), and computing it needs either the full integral or $\lambda$; the WBIC needs neither, only samples at one temperature, and it is "widely applicable" because the derivation did not assume the model is regular.

**2.** $L_n-L_n(w^*)=K_n$ by definition when $w^*$ attains the minimum of $L_n$ in the basin, so $n\beta(\mathbb E_\beta[L_n]-L_n(w^*))=\beta\,\mathbb E_\beta[nK_n]$. If $w^*$ is not the minimizer, $L_n(w^*)-\min L_n$ is subtracted from every term and the estimate is biased by $n\beta$ times that, which is why one minimizes first. Adding $\tfrac\gamma2\|w-w^*\|^2$ to the energy multiplies the density by $e^{-\gamma\|w-w^*\|^2/2}$. With $K\sim\|w-w^*\|^{2k}$, the region where $e^{-\beta nK}$ is not small is $\|w-w^*\|\lesssim(n\beta)^{-1/2k}$, on which the localizer is $1-O(\gamma(n\beta)^{-1/k})$; it therefore changes the average by a relative $O(\gamma(n\beta)^{-1/k})$, which vanishes as $n\beta\to\infty$ (slowly for small $k$... i.e. for flat directions, large $k$, which is the point: flat directions are the ones $\gamma$ cuts off).

**3.** With $K=\tfrac12(w^4+\delta w^2)$, the rescaling $u=(n\beta/2)^{1/4}w$ turns $n\beta K$ into $u^4+s\,u^2$ with $s=\delta\sqrt{n\beta/2}$, so the crossover is at $s\sim1$, i.e. $n\beta\sim2/\delta^2$. For $\delta=0.1$: $n\beta\sim200$, so $\beta=0.01,0.03$ ($n\beta=10,30$, $s=0.2,0.4$) give near $\tfrac14$, $\beta=0.1$ ($s=0.7$) is in between, $\beta=0.3,1$ ($s=1.2,2.2$) are on the way to $\tfrac12$. For $\delta=0.03$: $n\beta\sim2200$, so every $\beta\le1$ ($s\le0.67$) gives $\tfrac14$ with a rise at the top. For $\delta=0$: $\tfrac14$ at all $\beta$, except that Exercise 3.1.4 says the sample loss has a quadratic term of size $|u_n|\sim\sigma/\sqrt n\approx0.016$, which acts like $\delta\approx0.03$ (if $u_n<0$; if $u_n>0$ it is a double well, with its own regular minima at $w^2=u_n/a_n$, which $w^*$ will sit in): so at $\beta=1$ the $\delta=0$ estimate should also drift upward, and the three $\delta$ curves should not be distinguishable below $\delta\sim0.03$. The plot below is to be compared against these predictions, and the exact tempered expectation by quadrature (same figure) separates what the estimator is estimating from what the sampler does to it.

**4.** The energy is $\tfrac12(w-w^*)^\top(n\beta H+\gamma I)(w-w^*)$ with $H$ the Hessian of $K_n$ (taking $K_n$ quadratic), so the chain samples $\mathcal N(w^*,(n\beta H+\gamma I)^{-1})$ and $\mathbb E[nK_n]=\tfrac12\mathrm{tr}\big(nH(n\beta H+\gamma I)^{-1}\big)=\tfrac12\sum_i\frac{nh_i}{n\beta h_i+\gamma}$; multiplying by $\beta$, $\hat\lambda=\tfrac12\sum_i\frac{n\beta h_i}{n\beta h_i+\gamma}$. Each direction contributes between $0$ and $\tfrac12$: $\tfrac12$ if its curvature $n\beta h_i$ is large compared with $\gamma$ (the loss, not the localizer, confines the chain there) and $0$ if small (the localizer does). So $\hat\lambda$ is half the number of directions that are stiff at the scale $\gamma$: as $\gamma\to0$ it is $d/2$, the regular answer; as $\gamma\to\infty$ it goes to $0$. For a non-quadratic basin the same reading applies with "curvature" replaced by the confinement the loss provides at the length $\gamma^{-1/2}$, and the DLN sweep below shows the resulting decline of $\hat\lambda$ with $\gamma$.
"""
    ),
    markdown(
        r"""
**Code 1: the 1-D model swept over $\beta$.** Exercise 3.1.4's model with
$n=1000$. For each $\delta$ and $\beta$ the cell runs SGLD from the minimizer
$w^*$ of $L_n$ and also computes the exact tempered expectation
$\beta\,\mathbb E_\beta[nK_n]$ on a grid (possible here because $w$ is one
number). Compare with your predictions from Exercise 3.3.3.
"""
    ),
    code(
        r"""
n, batch, sigma = 1000, 100, 0.5
betas = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
X = rng.standard_normal((n, 2))
y = sigma * rng.standard_normal(n)
w_grid = np.linspace(-1.5, 1.5, 60_001)

plt.figure(figsize=(6, 3.8))
for j, delta in enumerate([0.0, 0.03, 0.1]):
    def f(w, Xs): return w**2 * Xs[:, 0] + np.sqrt(delta) * w * Xs[:, 1]
    def loss_full(w): return 0.5 * np.mean((f(w, X) - y) ** 2)
    def grad_minibatch(w, idx):
        Xb, yb = X[idx], y[idx]
        return np.mean((f(w, Xb) - yb) * (2 * w * Xb[:, 0] + np.sqrt(delta) * Xb[:, 1]))
    L_grid = 0.5 * np.mean((w_grid[:, None] ** 2 * X[None, :, 0] + np.sqrt(delta) * w_grid[:, None] * X[None, :, 1] - y[None, :]) ** 2, axis=1)
    w_star = w_grid[np.argmin(L_grid)]
    exact, sgld = [], []
    for beta in betas:
        logp = -n * beta * L_grid
        p = np.exp(logp - logp.max()); p /= p.sum()
        exact.append(n * beta * ((p * L_grid).sum() - L_grid.min()))
        est = sgld_lambda(grad_minibatch, loss_full, np.array(w_star), n, batch, n * beta, gamma=1.0,
                          eps=1e-3, steps=12000, burn=2000, chains=3)
        sgld.append(est)
    sgld = np.array(sgld)
    plt.errorbar(n * betas, sgld.mean(1), sgld.std(1), fmt="o-", color=f"C{j}", capsize=3, label=rf"SGLD, $\delta={delta}$ ($w^*={w_star:.2f}$)")
    plt.plot(n * betas, exact, "--", color=f"C{j}", label=rf"exact tempered mean, $\delta={delta}$")
for d in [0.03, 0.1]:
    plt.axvline(2 / d**2, color="k", ls=":", lw=0.8)
plt.axhline(0.25, color="k", ls=":"); plt.axhline(0.5, color="k", ls=":")
plt.xscale("log"); plt.xlabel(r"$n\beta$"); plt.ylabel(r"$\hat\lambda$"); plt.legend(fontsize=7, ncol=2)
plt.title(r"$K=\frac{1}{2}(w^4+\delta w^2)$ from data, $n=1000$; dotted: $n\beta=2/\delta^2$"); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
**Code 2: normal crossings.** The model $f_w(x)=w_1^aw_2^bx$ fitting pure
noise, as in Lau et al.'s Table 1 and devinterp's `normal_crossing` notebook.
Notebook 2 Exercise 2.2.2 gives $\lambda=\min(\tfrac1{2a},\tfrac1{2b})$ with $m=2$
when $a=b$. SGLD is run from the origin at $\beta=1/\log n$ with $\gamma=1$;
next to it, the exact value of $\beta\,\mathbb E_\beta[nK_n]$ under the same
localized tempered posterior, computed on a two-dimensional grid, which is the
number the sampler is trying to produce.
"""
    ),
    code(
        r"""
n, batch = 1000, 100
nbeta = n / np.log(n)
Xc = rng.standard_normal(n)
yc = 0.25 * rng.standard_normal(n)
g1 = np.linspace(-2, 2, 801)
W1, W2 = np.meshgrid(g1, g1, indexing="ij")
sx2, sxy = np.mean(Xc**2), np.mean(Xc * yc)
print(f"{'(a,b)':8s} {'lambda':>7s} {'exact tempered':>15s} {'SGLD (3 chains)':>22s}")
for (a, b) in [(1, 1), (1, 2), (1, 3), (2, 2)]:
    lam = min(1 / (2 * a), 1 / (2 * b))
    def f(w, Xs): return w[0]**a * w[1]**b * Xs
    def loss_full(w): return 0.5 * np.mean((f(w, Xc) - yc) ** 2)
    def grad_minibatch(w, idx):
        Xb, yb = Xc[idx], yc[idx]; r = f(w, Xb) - yb
        return np.array([np.mean(r * Xb * a * w[0]**(a - 1) * w[1]**b), np.mean(r * Xb * b * w[0]**a * w[1]**(b - 1))])
    w_star = np.zeros(2)
    # exact: L_n(w) = 1/2 c^2 mean(x^2) - c mean(x y) + 1/2 mean(y^2), with c = w1^a w2^b
    C = W1**a * W2**b
    L_grid = 0.5 * C**2 * sx2 - C * sxy + 0.5 * np.mean(yc**2)
    logp = -nbeta * L_grid - 0.5 * 1.0 * (W1**2 + W2**2)
    p = np.exp(logp - logp.max()); p /= p.sum()
    exact = nbeta * ((p * L_grid).sum() - loss_full(w_star))
    est = sgld_lambda(grad_minibatch, loss_full, w_star, n, batch, nbeta, gamma=1.0, eps=1e-3, steps=10000, burn=2000, chains=3)
    print(f"({a},{b})    {lam:7.3f} {exact:15.3f} {est.mean():12.3f} +- {est.std():.3f}")
"""
    ),
    markdown(
        r"""
**Code 3: a two-layer linear network with known $\lambda$.** The model is
$f(x)=BAx$ with $A,B\in\mathbb R^{3\times3}$ ($d=18$ parameters, so a regular
model would have $\lambda=9$), fitting a teacher $B_0A_0$ of rank $r$ with a
little output noise. Aoyagi and Watanabe (2005) computed $\lambda$ for this
model; for square matrices of size $m$ it is $\lambda=\frac{(m+r)(3m-r)}8$,
plus $\tfrac18$ when $m+r$ is odd (in which case $m_{\rm mult}=2$), which for
$m=3$ gives $3.5,4,4.5,4.5$ for $r=0,1,2,3$. The chain starts at the sample's
minimizer near the teacher. The second part sweeps $\gamma$ at $r=3$: the
estimate falls as the localizer tightens, Exercise 3.3.4's effect.
"""
    ),
    code_answer(
        r"""
n, batch, m_dim = 1000, 100, 3
nbeta = n / np.log(n)
X3 = rng.standard_normal((n, m_dim))

def aoyagi_watanabe(m, r):
    return (m + r) * (3 * m - r) / 8 + (1 / 8 if (m + r) % 2 else 0.0)

def make_problem(r):
    s = np.zeros(m_dim); s[:r] = 1.0
    A0 = np.diag(np.sqrt(s)); B0 = np.diag(np.sqrt(s))
    Y = X3 @ (B0 @ A0).T + 0.1 * rng.standard_normal((n, m_dim))
    def unpack(w): return w[:9].reshape(3, 3), w[9:].reshape(3, 3)
    def loss_full(w):
        A, B = unpack(w); R = X3 @ (B @ A).T - Y
        return 0.5 * np.mean(np.sum(R**2, 1))
    def grad_minibatch(w, idx):
        A, B = unpack(w); Xb, Yb = X3[idx], Y[idx]
        R = Xb @ (B @ A).T - Yb
        return np.concatenate([(B.T @ R.T @ Xb / len(idx)).ravel(), (R.T @ Xb @ A.T / len(idx)).ravel()])
    w = np.concatenate([A0.ravel(), B0.ravel()])
    for _ in range(2000):                      # gradient descent to the sample's minimizer near the teacher
        w = w - 0.05 * grad_minibatch(w, np.arange(n))
    return loss_full, grad_minibatch, w

# TODO: for r in 3, 2, 1, 0 estimate lambda with sgld_lambda(..., nbeta, gamma=1.0, eps=3e-4, steps=6000, burn=1000)
#       and print it next to aoyagi_watanabe(3, r); then sweep gamma over [0.3, 1, 3, 10, 30, 100] at r = 3.
""",
        r"""
n, batch, m_dim = 1000, 100, 3
nbeta = n / np.log(n)
X3 = rng.standard_normal((n, m_dim))

def aoyagi_watanabe(m, r):
    return (m + r) * (3 * m - r) / 8 + (1 / 8 if (m + r) % 2 else 0.0)

def make_problem(r):
    s = np.zeros(m_dim); s[:r] = 1.0
    A0 = np.diag(np.sqrt(s)); B0 = np.diag(np.sqrt(s))
    Y = X3 @ (B0 @ A0).T + 0.1 * rng.standard_normal((n, m_dim))
    def unpack(w): return w[:9].reshape(3, 3), w[9:].reshape(3, 3)
    def loss_full(w):
        A, B = unpack(w); R = X3 @ (B @ A).T - Y
        return 0.5 * np.mean(np.sum(R**2, 1))
    def grad_minibatch(w, idx):
        A, B = unpack(w); Xb, Yb = X3[idx], Y[idx]
        R = Xb @ (B @ A).T - Yb
        return np.concatenate([(B.T @ R.T @ Xb / len(idx)).ravel(), (R.T @ Xb @ A.T / len(idx)).ravel()])
    w = np.concatenate([A0.ravel(), B0.ravel()])
    for _ in range(2000):                      # gradient descent to the sample's minimizer near the teacher
        w = w - 0.05 * grad_minibatch(w, np.arange(n))
    return loss_full, grad_minibatch, w

print(f"{'rank r':>6s} {'lambda (Aoyagi-Watanabe)':>26s} {'SGLD estimate':>22s}     (d/2 = 9)")
for r in [3, 2, 1, 0]:
    loss_full, grad_minibatch, w_star = make_problem(r)
    est = sgld_lambda(grad_minibatch, loss_full, w_star, n, batch, nbeta, gamma=1.0, eps=3e-4, steps=6000, burn=1000, chains=3)
    print(f"{r:6d} {aoyagi_watanabe(3, r):26.3f} {est.mean():14.2f} +- {est.std():.2f}")

loss_full, grad_minibatch, w_star = make_problem(3)
gammas = [0.3, 1, 3, 10, 30, 100]
sweep = np.array([sgld_lambda(grad_minibatch, loss_full, w_star, n, batch, nbeta, gamma=g, eps=3e-4, steps=6000, burn=1000, chains=3) for g in gammas])
plt.figure(figsize=(5, 3.3))
plt.errorbar(gammas, sweep.mean(1), sweep.std(1), fmt="o-", capsize=3)
plt.axhline(aoyagi_watanabe(3, 3), color="k", ls=":", label=r"$\lambda=4.5$"); plt.axhline(9, color="k", ls="--", lw=0.8, label=r"$d/2=9$")
plt.xscale("log"); plt.xlabel(r"localization $\gamma$"); plt.ylabel(r"$\hat\lambda$"); plt.title("rank-3 teacher: the estimate against the localizer")
plt.legend(); plt.tight_layout(); plt.show()
""",
    ),
    # ------------------------------------------------------------------ readings
    markdown(
        r"""
<a id="slt3-readings"></a>

## Readings

The keystone is Lau et al. (top of the page), with Furman's notebook as the
hands-on companion.
"""
        + reading(
            "A Widely Applicable Bayesian Information Criterion",
            "https://jmlr.org/papers/v14/watanabe13a.html",
            "Watanabe (JMLR 2013)",
            "The source of $\\mathbb E_{1/\\log n}[nL_n]\\approx F_n$ (§3.3), with the proof that it holds without assuming a regular model; Sections 1–3 give the statement and the heuristic, the rest the theorem.",
            importance=3,
            length=3,
        )
        + reading(
            "Deep Learning is Singular, and That's Good",
            "https://arxiv.org/abs/2010.11560",
            "Wei, Murfet, Gong, Li, Gell-Redman & Quella (2020)",
            "Why the asymptotic $\\lambda$ of a neural network is both the right object and a hard one to get at, and early experiments with its estimation; the case for the whole programme, written before the estimator of this notebook existed.",
            importance=3,
            length=3,
        )
        + references(
            book("Mathematical Theory of Bayesian Statistics", "https://doi.org/10.1201/9781315373010", "Watanabe (CRC Press, 2018)", "chapters 5–7 for the free energy at general $\\beta$ and the WBIC"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt3", "slt3_effective_temperature"))
