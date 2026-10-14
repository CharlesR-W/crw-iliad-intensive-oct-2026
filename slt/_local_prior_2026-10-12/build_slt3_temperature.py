#!/usr/bin/env python3
"""Build SLT3, "Effective exponents and temperature" (SLT day, NB3).

One linear path, three `##` sections (anchors slt3-1, slt3-2, slt3-3) plus a closing reading
list (slt3-readings), about two hours including the code:
1 the pathology: K = w^4 + delta w^2 has lambda = 1/2 for delta > 0 and 1/4 at delta = 0; the
  effective exponent lambda_eff(n) = -dlog Zbar_n/dlog n = n <K>_n and its crossover at n ~ delta^-2;
2 a marked stub: CRW supplies the effective-theory / why-temperature argument;
3 sampling at temperature: WBIC to leading order, the (local) estimator
  lambda-hat = n beta (E_beta[L_n] - L_n(w*)) with localisation gamma, what it measures (exact in two
  idealisations), SGLD as the sampler (a code exercise), then three checks: a beta-sweep on
  w^4 + delta w^2 against lambda_eff(n beta), normal crossings, and a two-layer linear network
  against Aoyagi & Watanabe's lambda. Hyperparameter dependence is read as scale dependence (an
  observation about the estimator only; the argument for temperature is CRW's, in section 2).
Exercises are bold paragraphs ("Exercise 3.1a"), solutions folded.

Code adapted from Furman's intro-lc-estimation notebook (the model y = w^2 x, the SGLD update) and
the devinterp v1.3.2 notebooks (normal_crossing.ipynb, dlns.ipynb); inline numpy SGLD, no devinterp.

Spec: SLT_SPECS_v1.md (NB3); brief: WRITER_BRIEF.md. Formula checks:
checks/slt3_temperature_formulas.py. Review log: ../reviews/slt/slt3_temperature.md.
"""
from nbkit_slt import answer, book, code, code_answer, markdown, reading, references, run  # noqa: F401

# ---------------------------------------------------------------- readings
LAU = reading(
    "The local learning coefficient: a singularity-aware complexity measure",
    "https://arxiv.org/abs/2308.12108",
    "Edmund Lau, Zach Furman, George Wang, Daniel Murfet and Susan Wei (AISTATS 2025)",
    "The estimator used in practice: a tempered posterior at $\\beta^*=1/\\log n$, localised at a "
    "parameter $w^*$ by a Gaussian of precision $\\gamma$, sampled with SGLD. Read §4 (Definition 2 and "
    "the SGLD update in §4.4) and §5.1, where the estimate is checked against known values for deep "
    "linear networks up to $10^8$ parameters. Appendix H says how to choose $\\gamma$, the step size "
    "and the chain length.",
    importance=5, length=3)
FURMAN = reading(
    "Introduction to RLCT estimation (notebook)",
    "https://github.com/zfurman56/intro-lc-estimation",
    "Zach Furman (2023)",
    "A short runnable notebook from volume scaling to WBIC to SGLD, on $K=w^4$ and the model "
    "$y=w^2x$. Its \"infinite-data\" and \"finite-data\" stages are the two idealisations of "
    "Exercise 3.3b; the sampler in §3.3 follows its SGLD class.",
    importance=4, length=2)
HVW = reading(
    "You're measuring model complexity wrong",
    "https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong",
    "Jesse Hoogland and Stan van Wingerden (LessWrong, 2023)",
    "The learning coefficient as the scaling exponent of low-loss volume, with pictures; the "
    "keystone of NB2. Read it here if you skipped NB2.",
    importance=3, length=2)
WBIC = reading(
    "A widely applicable Bayesian information criterion",
    "https://jmlr.org/papers/v14/watanabe13a.html",
    "Sumio Watanabe (JMLR 14, 2013)",
    "Proves that the posterior mean of $nL_n$ at $\\beta=1/\\log n$ has the same two leading terms "
    "as the free energy, with an $O_p(\\sqrt{\\log n})$ error, for singular and misspecified models "
    "under his conditions. Read §1 and §4 (Theorems 3–5, Corollaries 1–3); §5 has the proofs.",
    importance=3, length=3)
WELLING_TEH = reading(
    "Bayesian learning via stochastic gradient Langevin dynamics",
    "https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf",
    "Max Welling and Yee Whye Teh (ICML 2011)",
    "Where SGLD comes from: a Langevin step with minibatch gradients, and the argument that minibatch "
    "noise becomes negligible as the step size shrinks. Read §§1–3.",
    importance=2, length=2)

AW05 = "https://doi.org/10.1016/j.neunet.2005.03.014"

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# SLT3 — Effective exponents and temperature

**Singular learning theory day, Iliad Intensive 2026**

The learning coefficient $\lambda$ is defined by an $n\to\infty$ limit, and an arbitrarily small
change to the loss can change it. This notebook asks
what is actually measurable at finite $n$. It defines an effective exponent $\lambda_{\rm eff}(n)$,
shows how it crosses over between two values of $\lambda$, and then builds the estimator people
use: a posterior at an inverse temperature $\beta<1$, sampled with SGLD.

**Keystone readings.** Lau et al. for the estimator; Furman's notebook for a runnable route to it.
"""
        + LAU + FURMAN
        + r"""
**The path** (about two hours, most of it §3).

- [3.1 The pathology](#slt3-1) (35 min): $K=w^4+\delta w^2$, the effective exponent $\lambda_{\rm eff}(n)=n\,\mathbb E_n[K]$, and a crossover at $n\sim\delta^{-2}$.
- [3.2 Effective theory, and why you need temperature](#slt3-2): CRW's part (stub).
- [3.3 Sampling at temperature](#slt3-3) (80 min): WBIC, the local estimator $\hat\lambda$ with localisation $\gamma$, SGLD, and three checks in code (a $\beta$-sweep, normal crossings, a linear network).

It assumes $\lambda$ and $m$ as defined in NB2 §1. If you only want the estimator, start at §3.3.
"""
    ),
    code(
        r"""
import time
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)
"""
    ),
    # ================================================================ 3.1
    markdown(
        r"""
<a id="slt3-1"></a>

## 3.1 The pathology

**Question.** $\lambda$ is the exponent of a $t\to0$ limit. How stable is it, and what does a
finite sample see instead?

**Notation.** $K(w)=L(w)-L(w_0)$ is the population loss measured from its minimum, $\varphi$ the
prior, $\bar Z_n=\int\varphi\,e^{-nK}dw$ and $\bar F_n=-\log\bar Z_n$. The learning coefficient
$\lambda$ and its multiplicity $m$ are defined by the prior volume of a sublevel set,
$V(t)=\int_{K<t}\varphi\sim c\,t^{\lambda}(-\log t)^{m-1}$ as $t\to0$, and then
$\bar F_n=\lambda\log n-(m-1)\log\log n+O(1)$ (NB2 §1).
"""
        + HVW
        + r"""
**One parameter, two exponents.** Take $d=1$ and $K_\delta(w)=w^4+\delta w^2$ with $\delta\ge0$.

- $\delta=0$: $\{K<t\}=\{|w|<t^{1/4}\}$, so $V(t)\approx2\varphi(0)\,t^{1/4}$ and $\lambda=\tfrac14$.
- $\delta>0$: for $t\ll\delta^2$ the quadratic term dominates, $\{K<t\}\approx\{|w|<(t/\delta)^{1/2}\}$, so $V(t)\approx2\varphi(0)(t/\delta)^{1/2}$ and $\lambda=\tfrac12$.

So $\lambda$ jumps from $\tfrac14$ to $\tfrac12$ as soon as $\delta\neq0$: in this family the
value $\tfrac14$ occurs at a single point, $\delta=0$. Exercise 3.1a shows that a finite sample
cannot tell small $\delta$ from zero anyway.

**The effective exponent.** Since $\bar F_n\approx\lambda\log n$, define the local slope

$$
\lambda_{\rm eff}(n)=\frac{d\bar F_n}{d\log n}=-\frac{d\log\bar Z_n}{d\log n}.
$$

It is a function of $n$, it tends to $\lambda$ as $n\to\infty$, and it equals the mean loss
$n\,\mathbb E_n[K]$ under $\bar p_n\propto\varphi\,e^{-nK}$ (Exercise 3.1b). It is the slope of
a plot of $\bar F_n$ against $\log n$. For $K_\delta$ with a flat prior it depends on $n$ and
$\delta$ only through $n\delta^2$, and crosses over from $\tfrac14$ to $\tfrac12$ around
$n\sim\delta^{-2}$ (Exercise 3.1c).

<a id="slt3-ex3-1a"></a>

**Exercise 3.1a — Where $\delta$ comes from, and when data can see it.** *(The model is the one in
Furman's notebook.)* Let $x\sim N(0,1)$ and use the model $p(y\mid x,w)=N(y;\,w^2x,\,\sigma^2)$
with $\sigma^2=\tfrac12$. The truth is $y=-ax+\varepsilon$, $\varepsilon\sim N(0,\tfrac12)$, with
$a\ge0$. For $a>0$ the truth is outside the model (the model can only make slopes $w^2\ge0$), and
the best fit is $w_0=0$.

1. Show that $K(w)=w^4+2aw^2$, so $\delta=2a$.
2. Show that the empirical version $K_n(w)=L_n(w)-L_n(0)$ is $K_n(w)=m_2w^4+\delta_nw^2$ with $m_2=\frac1n\sum_ix_i^2$ and $\delta_n=-\frac2n\sum_ix_iy_i$, and that $\delta_n=\delta\,m_2+\xi$ with $\xi$ approximately $N(0,2/n)$.
3. For which $n$ can the data distinguish $\delta$ from $0$? Compare with the crossover scale $n\sim\delta^{-2}$ above.
4. What does $K_n$ look like when $\delta_n<0$? How often does that happen when $a=0$?
"""
    ),
    answer(
        r"""
1. $-\log p(y\mid x,w)=(y-w^2x)^2/(2\sigma^2)+\tfrac12\log(2\pi\sigma^2)=(y-w^2x)^2+c$. With
   $y=-ax+\varepsilon$, $y-w^2x=-(a+w^2)x+\varepsilon$, so
   $L(w)=\mathbb E[(a+w^2)^2x^2-2(a+w^2)x\varepsilon+\varepsilon^2]+c=(a+w^2)^2+\tfrac12+c$.
   For $a\ge0$ this is smallest at $w=0$, and $K(w)=(a+w^2)^2-a^2=w^4+2aw^2$.
2. $L_n(w)-L_n(0)=\frac1n\sum_i\big[(y_i-w^2x_i)^2-y_i^2\big]=m_2w^4-2w^2\cdot\frac1n\sum_ix_iy_i$,
   which is $m_2w^4+\delta_nw^2$. Substituting $y_i=-ax_i+\varepsilon_i$,
   $\delta_n=2a\,m_2-\frac2n\sum_ix_i\varepsilon_i$. The last term has mean $0$ and variance
   $\frac4n\,\mathbb E[x^2]\,\mathbb E[\varepsilon^2]=\frac2n$, and by the CLT it is close to
   Gaussian.
3. The noise in $\delta_n$ is about $\sqrt{2/n}$, so the data distinguish $\delta$ from $0$ only
   when $\delta\gtrsim\sqrt{2/n}$, that is $n\gtrsim2\delta^{-2}$. This is the same scale as the
   crossover: below it, neither the data nor the free energy can tell $\delta$ from $0$.
4. For $\delta_n<0$, $K_n$ is a symmetric double well with minima at $w=\pm\sqrt{-\delta_n/(2m_2)}$,
   of order $n^{-1/4}$, and depth $\delta_n^2/(4m_2)\sim1/n$. When $a=0$, $\delta_n$ is symmetric
   about $0$, so this happens about half the time. The minima sit at the posterior's own width
   $n^{-1/4}$ (Exercise 3.1c), and the depth changes $nK_n$ there by $O(1)$. So at $a=0$ the
   empirical loss the posterior sees is a single quartic well or a double well, decided by the
   sampling noise.
"""
    ),
    markdown(
        r"""
<a id="slt3-ex3-1b"></a>

**Exercise 3.1b — The effective exponent is a mean loss.** Write $\mathbb E_n$ for the average
over $\bar p_n(w)\propto\varphi(w)e^{-nK(w)}$.

1. Show that $\lambda_{\rm eff}(n)=n\,\mathbb E_n[K]$.
2. Regular case: $K(w)=\tfrac12w^\top Hw$ on $\mathbb R^d$ with $H$ positive definite and a flat prior. Show $\lambda_{\rm eff}(n)=d/2$ for every $n$.
3. Suppose $\bar F_n=\lambda\log n-(m-1)\log\log n+c+o(1)$, and that this expansion can be differentiated term by term (true when $v(t)$ has a power–log expansion, as in every example here). Show $\lambda_{\rm eff}(n)=\lambda-(m-1)/\log n+o(1/\log n)$. For $K=(ab)^2$ ($\lambda=\tfrac12$, $m=2$), roughly how large must $n$ be for $\lambda_{\rm eff}$ to come within $0.05$ of $\lambda$?

<details><summary>Hint</summary>

Differentiate $\log\bar Z_n$ under the integral. For part 2, substitute $w=u/\sqrt n$.

</details>
"""
    ),
    answer(
        r"""
1. $\frac{d}{dn}\log\bar Z_n=\frac{\int\varphi\,(-K)\,e^{-nK}}{\int\varphi\,e^{-nK}}=-\mathbb E_n[K]$,
   and $\frac{d}{d\log n}=n\frac{d}{dn}$, so $\lambda_{\rm eff}=n\,\mathbb E_n[K]$. (In the
   dictionary of NB1, $n$ plays the role of $\beta$ for the energy $K$, and this is the identity
   "mean energy $=\partial_\beta F$". It is exact, with no limit taken.)
2. With $w=u/\sqrt n$, $\bar Z_n=n^{-d/2}\int e^{-\frac12u^\top Hu}du$, so
   $\bar F_n=\frac d2\log n+\text{const}$ and $\lambda_{\rm eff}=d/2$ exactly. Equivalently,
   $\bar p_n=N(0,(nH)^{-1})$ and $n\,\mathbb E_n[\tfrac12w^\top Hw]=\tfrac12\operatorname{tr}(I_d)=d/2$.
3. $\frac{d}{d\log n}\log\log n=\frac1{\log n}$, so
   $\lambda_{\rm eff}=\lambda-\frac{m-1}{\log n}+o(1/\log n)$. For $m=2$, $1/\log n<0.05$ needs
   $\log n>20$, so $n\gtrsim e^{20}\approx5\times10^8$. Multiplicity is felt for a very long time.
   (The exact value for $K=(ab)^2$ with a uniform prior on $[-1,1]^2$ is $\lambda_{\rm eff}=0.45$
   at $n=10^8$.)
"""
    ),
    markdown(
        r"""
<a id="slt3-ex3-1c"></a>

**Exercise 3.1c — Scaling and the crossover.** Take $K_\delta=w^4+\delta w^2$ with a flat prior
on all of $\mathbb R$ (a good approximation for a prior that is nearly constant over the region
$|w|\lesssim n^{-1/4}$ where the posterior lives).

1. Substitute $w=n^{-1/4}u$ to show $\bar Z_n=n^{-1/4}g(s)$ with $s=\delta\sqrt n$ and $g(s)=\int e^{-u^4-su^2}du$. Deduce $\lambda_{\rm eff}(n)=\tfrac14+\tfrac s2\langle u^2\rangle_s$, where $\langle\cdot\rangle_s$ averages over the density $\propto e^{-u^4-su^2}$.
2. Show $\lambda_{\rm eff}\to\tfrac14$ as $s\to0$ and $\lambda_{\rm eff}\to\tfrac12$ as $s\to\infty$.
3. Conclude that $\lambda_{\rm eff}$ depends on $n$ and $\delta$ only through $n\delta^2=s^2$, and give a one-line Fermi argument for the crossover at $n\sim\delta^{-2}$.

<details><summary>Hint</summary>

$\frac{d}{ds}\log g=-\langle u^2\rangle_s$ and $\frac{ds}{d\log n}=\frac s2$. For large $s$
the $su^2$ term confines $u$ to $|u|\lesssim s^{-1/2}$, where $u^4$ is negligible.

</details>
"""
    ),
    answer(
        r"""
1. $\int e^{-n(w^4+\delta w^2)}dw=n^{-1/4}\int e^{-u^4-\delta\sqrt n\,u^2}du=n^{-1/4}g(s)$. So
   $\log\bar Z_n=-\tfrac14\log n+\log g(s)$ and
   $\lambda_{\rm eff}=\tfrac14-\frac{ds}{d\log n}\frac{g'(s)}{g(s)}=\tfrac14+\tfrac s2\langle u^2\rangle_s$.
2. As $s\to0$, $\langle u^2\rangle_s\to\langle u^2\rangle_0<\infty$, so the second term vanishes.
   As $s\to\infty$, the density is close to $N(0,1/(2s))$, so $\langle u^2\rangle_s\approx1/(2s)$ and
   $\lambda_{\rm eff}\to\tfrac14+\tfrac14=\tfrac12$.
3. $s^2=n\delta^2$, so $\lambda_{\rm eff}$ is a single function of $n\delta^2$; it equals $0.367$ at
   $n\delta^2=1$, about halfway. Fermi: $\bar p_n$ sits where $nK\sim1$. If the quartic term
   dominates, that is $|w|\sim n^{-1/4}$, where $\delta w^2\sim\delta n^{-1/2}$ against
   $w^4\sim n^{-1}$. Their ratio is $\delta\sqrt n=s$, so the quadratic term takes over once
   $n\gtrsim\delta^{-2}$.
"""
    ),
    markdown(
        r"""
The code computes $\lambda_{\rm eff}(n)=n\,\mathbb E_n[K]$ by quadrature, with a standard normal
prior, for several $\delta$, and replots the curves against $n\delta^2$. They collapse onto the
flat-prior scaling function of Exercise 3.1c except at $n\lesssim10$, where the prior, not $K$,
sets the slope.
"""
    ),
    code(
        r"""
def lam_eff(K, ns, gamma=1.0, wmax=12.0, N=200_001):
    '''n <K> under the density prop. to exp(-n K(w) - gamma w^2 / 2), for each n in ns (1-D quadrature).

    gamma = 1 is a standard normal prior; gamma = 0 a flat one (then K must confine w within wmax).'''
    s = np.linspace(-1, 1, N)
    w = wmax * np.sinh(8 * s) / np.sinh(8)          # grid dense near w = 0
    logdw, Kw = np.log(np.gradient(w)), K(w)
    out = []
    for n in np.atleast_1d(ns):
        a = -n * Kw - gamma * w**2 / 2 + logdw
        p = np.exp(a - a.max())
        out.append(n * (p * Kw).sum() / p.sum())
    return np.array(out)


ns = np.logspace(0, 12, 97)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for delta in [0, 1e-1, 1e-2, 1e-3, 1e-4]:
    le = lam_eff(lambda w: w**4 + delta * w**2, ns)
    line, = ax[0].semilogx(ns, le, label=f"δ = {delta:g}")
    if delta > 0:
        ax[0].axvline(delta**-2, color=line.get_color(), ls=":", lw=1)
        ax[1].semilogx(ns * delta**2, le, color=line.get_color(), label=f"δ = {delta:g}")
x = np.logspace(-4, 6, 61)          # x = n δ^2; flat prior: λ_eff = <u^4 + s u^2>_s with s = sqrt(x)
ax[1].semilogx(x, [lam_eff(lambda u: u**4 + np.sqrt(v) * u**2, 1.0, gamma=0.0, wmax=6.0)[0] for v in x],
               "k--", lw=1, label="flat prior (Exercise 3.1c)")
ax[0].set(xlabel="n", ylabel=r"$\lambda_{\rm eff}(n)$",
          title=r"$K=w^4+\delta w^2$, prior N(0,1); dotted: $n=\delta^{-2}$")
ax[1].set(xlabel=r"$n\delta^2$", title=r"the same curves against $n\delta^2$")
for a in ax:
    a.axhline(0.25, color="gray", lw=0.5)
    a.axhline(0.5, color="gray", lw=0.5)
    a.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("λ_eff at nδ² = 1 (flat prior):", lam_eff(lambda u: u**4 + u**2, 1.0, gamma=0.0, wmax=6.0)[0].round(3))
"""
    ),
    markdown(
        r"""
For $\delta=10^{-4}$, $\lambda_{\rm eff}$ stays between $0.24$ and $0.30$ from $n=10^2$ to
$n\approx10^7$, although $\lambda=\tfrac12$.
"""
    ),
    # ================================================================ 3.2
    markdown(
        r"""
<a id="slt3-2"></a>

## 3.2 Effective theory, and why you need temperature

**Question.** If a finite sample sees $\lambda_{\rm eff}$ rather than $\lambda$, what is the right
object to study, and where does temperature come in?

<div style="border: 2px dashed #c0392b; border-radius: 6px; padding: 0.6em 1em; margin: 1em 0;">

**Stub: CRW supplies this section.** Placeholder scope: $\lambda$ at a resolution scale, and
$\beta$ (or the trade between $n$ and $\beta$) as the knob that sets the scale.

</div>
"""
    ),
    # ================================================================ 3.3
    markdown(
        r"""
<a id="slt3-3"></a>

## 3.3 Sampling at temperature

**Question.** $Z_n$ is a $d$-dimensional integral we cannot compute. How do we measure $\lambda$
(or $\lambda_{\rm eff}$) from data, using only samples?

**Notation.** $L_n(w)=-\frac1n\sum_i\log p(X_i\mid w)$ is the empirical loss, $L$ its
population version, $w_0$ a minimiser of $L$, and $K_n(w)=L_n(w)-L_n(w_0)$, $K(w)=L(w)-L(w_0)$.
The tempered posterior is $p_\beta(w\mid D_n)\propto\varphi(w)e^{-n\beta L_n(w)}$, with average
$\mathbb E_\beta$; $\beta=1$ is Bayes. $Z_n(\beta)=\int\varphi\,e^{-n\beta L_n}$ and
$F_n(\beta)=-\log Z_n(\beta)$, so $F_n=F_n(1)$ is the free energy. $\lambda$ and $m$ are the
learning coefficient (the volume-scaling exponent) and its multiplicity, and
$\lambda_{\rm eff}(n)=n\,\mathbb E_n[K]$ is the effective exponent of §3.1.
"""
        + WBIC
        + r"""
### The estimator

Two exact facts start everything (Exercise 3.1b, part 1, with $\beta$ in place of $n$):

$$
\frac{dF_n(\beta)}{d\beta}=\mathbb E_\beta[nL_n],\qquad
\frac{d}{d\beta}\mathbb E_\beta[nL_n]=-\operatorname{Var}_\beta(nL_n)\le0 .
$$

In the dictionary of NB1 the first is "mean energy $=\partial_\beta F$"; both hold exactly for
any $n$ and $d$. Since $F_n(0)=0$ ($\varphi$ is normalised), $F_n=\int_0^1\mathbb E_\beta[nL_n]\,d\beta$,
so some $\beta_F\in(0,1)$ has $F_n=\mathbb E_{\beta_F}[nL_n]$. It is unique when $L_n$ is not
constant, because then $\mathbb E_\beta[nL_n]$ strictly decreases (Watanabe's Theorem 3). A single
posterior average at the right temperature gives the free energy.

<a id="slt3-ex3-3a"></a>

**Exercise 3.3a — WBIC to leading order.** Assume the free-energy formula holds with $n\beta$ in
place of $n$: $F_n(\beta)\approx n\beta L_n(w_0)+\lambda\log(n\beta)$.

1. Show $\mathbb E_\beta[nL_n]\approx nL_n(w_0)+\lambda/\beta$.
2. Show that $\mathbb E_\beta[nL_n]\approx F_n$ at $\beta=1/\log n$.
3. Rearrange part 1 into an estimator of $\lambda$.
4. Where does this derivation ignore the randomness of the data?
"""
    ),
    answer(
        r"""
1. $\frac{d}{d\beta}\big[n\beta L_n(w_0)+\lambda\log n+\lambda\log\beta\big]=nL_n(w_0)+\lambda/\beta$.
2. $F_n=F_n(1)\approx nL_n(w_0)+\lambda\log n$. Setting $nL_n(w_0)+\lambda/\beta$ equal to it
   gives $\beta=1/\log n$. The posterior average of $nL_n$ at that temperature is Watanabe's
   WBIC.
3. $\lambda\approx\beta\big(\mathbb E_\beta[nL_n]-nL_n(w_0)\big)=n\beta\big(\mathbb E_\beta[L_n]-L_n(w_0)\big)$.
4. In the assumption. The formula $\lambda\log(n\beta)$ is the one for $\bar Z$, built from the
   population $K$. With $K_n$ in its place, $nK_n(w)-nK(w)$ is a sum of $n$ centred terms
   $\log\frac{p(X_i\mid w_0)}{p(X_i\mid w)}-K(w)$. In the realisable case each has variance of
   order $K(w)$ near $w_0$, so the sum is of order $\sqrt{nK}$. Where the tempered posterior sits, $nK\sim\lambda/\beta$, so
   this is of order $\sqrt{\lambda/\beta}$: after multiplying by $\beta$, an error of order
   $\sqrt{\lambda\beta}$ in the estimator. That is the $U_n$ term below. Part 2 also drops the
   $-(m-1)\log\log n$ term.
"""
    ),
    markdown(
        r"""
**What Watanabe proves.** For $\beta=\beta_0/\log n$ with $\beta_0>0$ a constant, under his
conditions (which cover realisable models, singular or not, and regular ones), Theorem 4 of the
WBIC paper gives

$$
\mathbb E_\beta[nL_n]=nL_n(w_0)+\frac{\lambda}{\beta}+U_n\sqrt{\frac{\lambda}{2\beta}}+O_p(1),
$$

where $U_n$ has mean zero and tends in law to a Gaussian, with $\mathbb E[U_n^2]<1$ when the model
is realisable. At $\beta_0=1$ this is $\mathrm{WBIC}=nL_n(w_0)+\lambda\log n+O_p(\sqrt{\log n})$,
which has the same two leading terms as $F_n$. Multiplying by $\beta$, the estimator of part 3 is

$$
\hat\lambda=n\beta\big(\mathbb E_\beta[L_n]-L_n(w_0)\big)=\lambda+U_n\sqrt{\lambda\beta/2}+O_p(\beta).
$$

**The local estimator.** Lau et al. make two changes. They replace the prior by a Gaussian of
precision $\gamma$ centred on a parameter $w^*$ of interest (usually a trained network), and set
$\beta=\beta^*=1/\log n$:

$$
p(w;w^*,\beta,\gamma)\propto\exp\Big(-n\beta L_n(w)-\frac\gamma2\|w-w^*\|^2\Big),\qquad
\hat\lambda(w^*)=n\beta^*\big(\mathbb E_{w^*,\beta^*,\gamma}[L_n]-L_n(w^*)\big).
$$

This is the local learning coefficient (LLC) estimator. Lau et al. show that the corresponding
average estimates a localised free energy (their Appendix D), and conjecture that the limit is
the local learning coefficient $\lambda(w^*)$ under conditions on $\gamma$.

<a id="slt3-ex3-3b"></a>

**Exercise 3.3b — What $\hat\lambda$ measures, in two idealisations.** Use the global estimator
($\gamma=0$, flat prior, $w^*=w_0$). $J=\nabla^2L(w_0)$ is the Hessian (called $H$ in NB1) and
$C$ the covariance of $\nabla\log p(X\mid w_0)$.

1. *No data noise.* Replace $L_n$ by $L$, so $K_n$ becomes $K$. Show $\hat\lambda=\lambda_{\rm eff}(n\beta)$, the effective exponent of §3.1 at sample size $n\beta$.
2. *Regular, with data noise.* Take $K_n(w)=\tfrac12(w-\hat w)^\top J(w-\hat w)-\tfrac12\hat w^\top J\hat w$ on $\mathbb R^d$ (the quadratic expansion around $w_0=0$, with $\hat w$ the minimiser of $L_n$). Show $\hat\lambda=\frac d2-\frac{n\beta}2\hat w^\top J\hat w$. In a well-specified model $n\hat w^\top J\hat w$ is approximately $\chi^2_d$ (NB1 §2). Deduce $\mathbb E[\hat\lambda]\approx\frac d2(1-\beta)$. What changes under misspecification?
"""
    ),
    answer(
        r"""
1. With $K$ in place of $K_n$ the tempered posterior is $\propto\varphi\,e^{-n\beta K}$, which is
   $\bar p_{n'}$ with $n'=n\beta$. So $\hat\lambda=n'\,\mathbb E_{n'}[K]=\lambda_{\rm eff}(n')$ by
   Exercise 3.1b. Changing $\beta$ is the same as changing $n$ to $n\beta$.
2. The tempered posterior is $N\big(\hat w,(n\beta J)^{-1}\big)$, so
   $\mathbb E_\beta[K_n]=\tfrac12\operatorname{tr}\big(J(n\beta J)^{-1}\big)-\tfrac12\hat w^\top J\hat w=\frac d{2n\beta}-\tfrac12\hat w^\top J\hat w$
   and $\hat\lambda=n\beta\,\mathbb E_\beta[K_n]=\frac d2-\frac{n\beta}2\hat w^\top J\hat w$.
   With $n\hat w^\top J\hat w\approx\chi^2_d$ (mean $d$), $\mathbb E[\hat\lambda]\approx\frac d2(1-\beta)$.
   The reference point $L_n(w_0)$ lies above the minimum of $L_n$ by $\frac12\hat w^\top J\hat w$,
   and $n\beta$ times that is an $O(\beta)$ bias. Under misspecification
   $\hat w\approx N(0,J^{-1}CJ^{-1}/n)$ (NB1 §2), so $\mathbb E[n\hat w^\top J\hat w]=\operatorname{tr}(J^{-1}C)$
   and $\mathbb E[\hat\lambda]\approx\frac d2-\frac\beta2\operatorname{tr}(J^{-1}C)$.
"""
    ),
    markdown(
        r"""
So $\hat\lambda$ differs from $\lambda$ for three separate reasons, and the code below measures
each: the finite scale, $\lambda_{\rm eff}(n\beta)-\lambda$ (part 1); the data, an $O_p(\sqrt\beta)$
random term (root mean square below $\sqrt{\lambda\beta/2}$ in realisable models) and an
$O(\beta)$ bias (part 2 and Watanabe's theorem); and the sampler.

<a id="slt3-ex3-3c"></a>

**Exercise 3.3c — What $\gamma$ does.** In one dimension, take $K(w)=\tfrac12hw^2$ (no data
noise) and the localised posterior $\propto e^{-n\beta K(w)-\gamma w^2/2}$ centred at
$w^*=w_0=0$. Show $\hat\lambda=\tfrac12\big/\big(1+\gamma/(n\beta h)\big)$. Which directions does
the localisation remove from the count?
"""
    ),
    answer(
        r"""
The localised posterior is $N\big(0,1/(n\beta h+\gamma)\big)$, so
$\hat\lambda=n\beta\cdot\tfrac12h\cdot\frac1{n\beta h+\gamma}=\frac12\cdot\frac1{1+\gamma/(n\beta h)}$.
A direction with curvature $h\gg\gamma/(n\beta)$ counts fully ($\tfrac12$). A direction with
$h\ll\gamma/(n\beta)$ is confined by the Gaussian before the loss can confine it, and its
contribution is negligible; in between it is partly counted.
In distance terms, the localisation ignores how the loss behaves farther than about
$\gamma^{-1/2}$ from $w^*$.
"""
    ),
    markdown(
        r"""
### The sampler, and three checks

**SGLD.** We need samples from $\propto e^{-U(w)}$ with
$U(w)=n\beta L_n(w)+\frac\gamma2\|w-w^*\|^2$. The Langevin diffusion
$dw=-\tfrac12\nabla U\,dt+dB_t$ has exactly this stationary density. Its Euler step with step size
$\epsilon$ is $\Delta w=-\frac\epsilon2\nabla U(w)+\sqrt\epsilon\,\xi$, $\xi\sim N(0,I_d)$. Stochastic
gradient Langevin dynamics (SGLD) replaces $\nabla L_n$ by its average over a random minibatch of
$m$ samples. This is Lau et al.'s update (their §4.4):

$$
\Delta w_t=-\frac\epsilon2\Big(n\beta\,\widehat{\nabla L_n}(w_t)+\gamma(w_t-w^*)\Big)+\sqrt\epsilon\,\xi_t,
\qquad \widehat{\nabla L_n}=\frac1m\sum_{i\in B_t}\nabla\big(-\log p(X_i\mid w_t)\big).
$$

The tempering lives in $U$, through $n\beta$; the Langevin noise itself is at unit temperature.
SGLD samples $e^{-U}$ only as $\epsilon\to0$. For a well-behaved $U$ a finite step biases the
samples by $O(\epsilon)$; for a steep one such as $w^4$ a step too large for the region the chain
reaches can make it diverge. The minibatch adds gradient noise, and a finite chain may not have
mixed. These are sampler errors, not properties of the target. Lau et al. (Appendix H) tune $\epsilon$ until a Metropolis
acceptance check passes 90–95% of proposals and the loss trace is flat; here we keep $\epsilon$ a
small fraction of the inverse curvature.

<a id="slt3-ex3-3d"></a>

**Exercise 3.3d (code) — Write the sampler.** Fill in `sgld_llc`: run all chains in parallel
from $w^*$, take `steps` SGLD steps, and return $\hat\lambda$ for each chain, averaging
`loss(w)` $=L_n(w)-L_n(w^*)$ over the draws after a burn-in. The later cells need it; the solution
is on the reading page. *(Adapted from the `SGLD` class in Furman's notebook, itself following
Lau et al.'s update; no devinterp needed.)*
"""
    ),
    code_answer(
        r"""
def sgld_llc(grad, loss, w_star, nbeta, gamma, eps, steps, rng, burn=0.25):
    '''SGLD chains for the localised tempered posterior, prop. to exp(-nbeta L_n(w) - gamma/2 |w - w_star|^2).

    w_star : array (chains, d), every row the centre w* (each chain starts there)
    grad   : grad(w, rng) -> (chains, d), a minibatch estimate of grad L_n at each chain's w
    loss   : loss(w) -> (chains,), L_n(w) - L_n(w*) at each chain's w
    Returns lambda-hat for each chain: nbeta times the average of loss over the draws after burn-in.
    '''
    # your code here
    raise NotImplementedError
""",
        r"""
def sgld_llc(grad, loss, w_star, nbeta, gamma, eps, steps, rng, burn=0.25):
    '''SGLD chains for the localised tempered posterior, prop. to exp(-nbeta L_n(w) - gamma/2 |w - w_star|^2).

    w_star : array (chains, d), every row the centre w* (each chain starts there)
    grad   : grad(w, rng) -> (chains, d), a minibatch estimate of grad L_n at each chain's w
    loss   : loss(w) -> (chains,), L_n(w) - L_n(w*) at each chain's w
    Returns lambda-hat for each chain: nbeta times the average of loss over the draws after burn-in.
    '''
    w = w_star.copy()
    total, count = 0.0, 0
    for t in range(steps):
        drift = nbeta * grad(w, rng) + gamma * (w - w_star)
        w = w - 0.5 * eps * drift + np.sqrt(eps) * rng.standard_normal(w.shape)
        if t >= burn * steps:
            total, count = total + loss(w), count + 1
    return nbeta * total / count
""",
    ),
    markdown(
        r"""
<a id="slt3-ex3-3e"></a>

**Exercise 3.3e — Predict before you run.** The next cell uses the data of Exercise 3.1a with
$\delta=2a=10^{-2}$ and $n=10^6$, centres at $w^*=w_0=0$ with $\gamma=1$, and runs SGLD for
nine values of $\beta$ from $10^{-4}$ to $1$. Before running it, sketch $\hat\lambda$ against
$n\beta$. Where is $\beta^*=1/\log n$, and roughly what is $\hat\lambda$ there? Does the data noise
of Exercise 3.1a or the bias of Exercise 3.3b matter here?
"""
    ),
    answer(
        r"""
By Exercise 3.3b, part 1, $\hat\lambda(\beta)\approx\lambda_{\rm eff}(n\beta)$: the crossover curve of
§3.1 with $n\beta$ on the horizontal axis. It is near $\tfrac14$ for $n\beta\ll\delta^{-2}=10^4$,
near $\tfrac12$ for $n\beta\gg10^4$, and crosses over across a few decades around $n\beta\sim10^4$.
$\beta^*=1/\log10^6=0.072$, so $n\beta^*\approx7\times10^4$ and $n\beta^*\delta^2\approx7$, past
the middle of the crossover: $\hat\lambda(\beta^*)\approx0.45$, neither $\tfrac14$ nor
$\lambda=\tfrac12$.

The data noise is small: $\sqrt{2/n}=0.0014$ against $\delta=0.01$. The bias of Exercise 3.3b comes
from the linear term of $K_n$ at $w_0$, and here $K_n=m_2w^4+\delta_nw^2$ has none: the model is
even in $w$, so the minimiser of $L_n$ is $w_0=0$ itself whenever $\delta_n>0$. So the points
should follow the curve at every $\beta$, up to sampler error and small shifts from $m_2\ne1$ and
$\delta_n\ne\delta$ (a few thousandths here).
"""
    ),
    code(
        r"""
# Data from Exercise 3.1a: y = -a x + noise (variance 1/2), model y = w^2 x, so K = w^4 + δ w^2 with δ = 2a.
n, delta = 10**6, 1e-2
x = rng.standard_normal(n)
y = -(delta / 2) * x + np.sqrt(0.5) * rng.standard_normal(n)
m2, delta_n = np.mean(x * x), -2 * np.mean(x * y)      # K_n(w) = m2 w^4 + delta_n w^2 (Exercise 3.1a)
print(f"δ = {delta},  δ_n = {delta_n:.5f},  sampling noise sqrt(2/n) = {np.sqrt(2 / n):.5f}")
Kn = lambda w: (m2 * w**4 + delta_n * w**2)[:, 0]
gradKn = lambda w, rng: 4 * m2 * w**3 + 2 * delta_n * w      # full batch: the data enter only via m2, delta_n

t0 = time.time()
gamma, chains = 1.0, 200
betas = np.logspace(-4, 0, 9)
lam_hat = []
for beta in betas:
    nb = n * beta
    eps = 0.02 / (nb * (2 * delta + 12 / np.sqrt(nb)))         # a small fraction of 1 / (curvature of nβK_n)
    lam_hat.append(sgld_llc(gradKn, Kn, np.zeros((chains, 1)), nb, gamma, eps, 10000, rng).mean())
nbs = np.logspace(1.5, 6.5, 60)
plt.figure(figsize=(6.5, 3.8))
plt.semilogx(nbs, lam_eff(lambda w: w**4 + delta * w**2, nbs, gamma), "k-", lw=1,
             label=r"$\lambda_{\rm eff}(n\beta)$ of the population $K$ (§3.1)")
plt.semilogx(n * betas, lam_hat, "o", label=r"SGLD $\hat\lambda$, data $n=10^6$")
plt.axvline(n / np.log(n), color="gray", ls="--", lw=1, label=r"$\beta^*=1/\log n$")
plt.xlabel(r"$n\beta$")
plt.ylabel(r"$\hat\lambda$")
plt.legend(fontsize=8)
plt.title(rf"$K=w^4+\delta w^2$, $\delta={delta}$, $\gamma={gamma}$")
plt.tight_layout()
plt.show()
print("β:     ", "  ".join(f"{b:.0e}" for b in betas))
print("λ-hat: ", "  ".join(f"{v:5.3f}" for v in lam_hat))
print(f"({time.time() - t0:.1f} s)")
"""
    ),
    markdown(
        r"""
**Normal crossings.** Next, $K(w)=(w_1^{k_1}w_2^{k_2})^2$, from the model
$y=w_1^{k_1}w_2^{k_2}x+\varepsilon$ with the truth $y=\varepsilon$ (variance $\tfrac12$ as before).
Here $\lambda=\min\big(\tfrac1{2k_1},\tfrac1{2k_2}\big)$, with $m=2$ when $k_1=k_2$ and $m=1$
otherwise (NB2 §2). For each case the cell reports four numbers at
$n=10^4$, $\beta^*=1/\log n$, $\gamma=1$, $w^*=0$:

- $\lambda$;
- $\lambda_{\rm eff}(n\beta^*)$ for the same localised target, by quadrature (the finite-scale effect);
- $\hat\lambda$ from SGLD on the population $K$ (its gap to the previous number is sampler error);
- $\hat\lambda$ from SGLD on $K_n$ for six data sets: mean, and spread over data sets (the data effect, plus Monte Carlo error from 8 chains per data set).

*(Adapted from devinterp's `normal_crossing` notebook, which reproduces Lau et al.'s check.)*
"""
    ),
    code(
        r"""
# Normal crossings: model y = w1^k1 w2^k2 x, truth y = noise (variance 1/2), so K(w) = (w1^k1 w2^k2)^2.
def lam_eff_2d(k1, k2, nb, gamma, N=2001, wmax=8.0):
    '''nb <K> under exp(-nb K - gamma |w|^2 / 2), by quadrature on a 2-D grid.'''
    s = np.linspace(-1, 1, N)
    w = wmax * np.sinh(6 * s) / np.sinh(6)
    a, b = np.meshgrid(w, w, indexing="ij")
    K = (a**k1 * b**k2)**2
    lw = np.log(np.gradient(w))
    logp = -nb * K - gamma * (a**2 + b**2) / 2 + lw[:, None] + lw[None, :]
    p = np.exp(logp - logp.max())
    return nb * (p * K).sum() / p.sum()


def normal_crossings(cases, m2, mxy, chains, nb, gamma, eps, steps, rng):
    '''lambda-hat for each case, each data set (m2 = mean x^2, mxy = mean xy) and each chain.'''
    k = np.array(cases, float)
    k1, k2 = [np.repeat(k[:, j], len(m2) * chains)[:, None] for j in (0, 1)]
    m2 = np.tile(np.repeat(m2, chains), len(cases))[:, None]
    mxy = np.tile(np.repeat(mxy, chains), len(cases))[:, None]

    def f_and_grad(w):                                   # f = w1^k1 w2^k2 and its gradient
        a, b = w[:, :1], w[:, 1:]
        fa = np.where(k1 > 0, k1 * a**np.maximum(k1 - 1, 0) * b**k2, 0.0)
        fb = np.where(k2 > 0, k2 * b**np.maximum(k2 - 1, 0) * a**k1, 0.0)
        return a**k1 * b**k2, np.hstack([fa, fb])

    def loss(w):                                         # L_n(w) - L_n(0) = m2 f^2 - 2 mxy f
        f, _ = f_and_grad(w)
        return (m2 * f**2 - 2 * mxy * f)[:, 0]

    def grad(w, rng):                                    # full batch, via m2 and mxy
        f, df = f_and_grad(w)
        return (2 * m2 * f - 2 * mxy) * df

    lh = sgld_llc(grad, loss, np.zeros((len(k1), 2)), nb, gamma, eps, steps, rng)
    return lh.reshape(len(cases), -1, chains)


t0 = time.time()
n, gamma, eps, steps = 10**4, 1.0, 3e-5, 30000
nb = n / np.log(n)
cases = [(0, 1), (1, 1), (1, 2), (2, 2)]
pop = normal_crossings(cases, np.ones(1), np.zeros(1), 48, nb, gamma, eps, steps, rng)    # population K
X, Y = rng.standard_normal((6, n)), np.sqrt(0.5) * rng.standard_normal((6, n))           # 6 data sets
dat = normal_crossings(cases, np.mean(X * X, 1), np.mean(X * Y, 1), 8, nb, gamma, eps, steps, rng)
dat = dat.mean(2)                                                                          # one estimate per data set
print(f"n = {n}, β* = 1/log n = {1 / np.log(n):.3f}, γ = {gamma}")
print(" (k1,k2)    λ    m   λ_eff(nβ*)   λ-hat on K   λ-hat on K_n (mean ± sd over 6 data sets)")
for (p, q), pr, dr in zip(cases, pop, dat):
    lam = min(1 / (2 * p) if p else np.inf, 1 / (2 * q))
    print(f"  ({p},{q})    {lam:.3f}  {2 if p == q else 1}     {lam_eff_2d(p, q, nb, gamma):.3f}        "
          f"{pr.mean():.3f}        {dr.mean():.3f} ± {dr.std():.3f}")
print(f"({time.time() - t0:.1f} s)")
"""
    ),
    markdown(
        r"""
Reading the table column by column:

- *Finite scale* (column 3 against column 1) is the largest effect. At $m=2$, $\lambda_{\rm eff}(n\beta^*)$ sits well below $\lambda$, as Exercise 3.1b predicts: for $(1,1)$, $0.39$ against $0.5$, with $1/\log(n\beta^*)\approx0.14$.
- *Sampler* (column 4 against column 3): a few hundredths, largest for $(2,2)$, whose long flat valleys along both axes mix slowly.
- *Data* (column 5 against column 4): a downward pull of order $\beta^*\lambda$ (for the regular case $(0,1)$, Exercise 3.3b predicts $0.5\,\beta^*\approx0.05$), and a scatter from one data set to the next.

The ordering of the four values of $\lambda$ survives all three.

**A linear network with known $\lambda$.** The last check is a two-layer linear network
$f(x)=BAx$ with $A\in\mathbb R^{H\times M}$ and $B\in\mathbb R^{N\times H}$ (reduced-rank
regression), so $d=H(M+N)$. The truth is $y=B_0A_0x+\varepsilon$, $\varepsilon\sim N(0,I_N)$, with
$\operatorname{rank}(B_0A_0)=r$. ($H$ is a width here, not a Hessian.) [Aoyagi and Watanabe
(2005)]("""
        + AW05
        + r""") computed $\lambda$ and $m$ exactly for this model; `aoyagi_watanabe` in the next cell
implements their formula, as stated in the Iliad B.3 worksheet. For $M=3$ inputs, $N=5$ outputs
and $H=3$ hidden units ($d=24$, $d/2=12$) it gives $\lambda=4.5,\ 5.5,\ 6.5,\ 7.5$ for
$r=0,1,2,3$, with $m=2$ at $r=0$. Lau et al. run the same check on deeper linear networks (§5.1,
using Aoyagi's 2024 extension of the formula), with the estimator centred at a true parameter.
So do we: the chains start at a factorisation $(A_0,B_0)$ of the truth with the unused hidden
units switched off. We use $n=2000$ and minibatches of $b=200$. *(Adapted from devinterp's `dlns` notebook and Lau et al. §5.1.)*
"""
    ),
    code(
        r"""
def aoyagi_watanabe(M, N, H, r):
    '''Learning coefficient and multiplicity of the two-layer linear network / reduced-rank regression.'''
    if N + r <= M + H and M + r <= N + H and H + r <= M + N:
        s = 2 * (H + r) * (M + N) - (M - N)**2 - (H + r)**2
        return (s / 8, 1) if (M + H + N + r) % 2 == 0 else ((s + 1) / 8, 2)
    if M + H < N + r:
        return (H * M - H * r + N * r) / 2, 1
    if N + H < M + r:
        return (H * N - H * r + M * r) / 2, 1
    return M * N / 2, 1


t0 = time.time()
M, N, H = 3, 5, 3
n, b, gamma, eps, steps, chains = 2000, 200, 1.0, 1e-4, 10000, 4
nb = n / np.log(n)
print(f"M={M} inputs, N={N} outputs, H={H} hidden, d={H*(M+N)}, d/2={H*(M+N)/2}; "
      f"n={n}, batch b={b}, β*={1/np.log(n):.3f}")
for r in range(H + 1):
    U, _ = np.linalg.qr(rng.standard_normal((N, N)))
    V, _ = np.linalg.qr(rng.standard_normal((M, M)))
    s = np.r_[np.ones(r), np.zeros(H - r)]                    # B0 A0 has rank r, singular values 1
    A0, B0 = np.sqrt(s)[:, None] * V[:, :H].T, U[:, :H] * np.sqrt(s)
    X = rng.standard_normal((n, M))
    Y = X @ (B0 @ A0).T + rng.standard_normal((n, N))
    Sxx, Sxy = X.T @ X / n, X.T @ Y / n
    unpack = lambda w: (w[:, :H * M].reshape(-1, H, M), w[:, H * M:].reshape(-1, N, H))

    def grad_dln(w, rng):              # minibatch gradient of L_n(A, B) = mean |y - BAx|^2 / 2 (+ const)
        A, B = unpack(w)
        idx = rng.integers(0, n, (len(w), b))
        xb, yb = X[idx], Y[idx]
        h = xb @ A.transpose(0, 2, 1)
        res = h @ B.transpose(0, 2, 1) - yb
        gA = (res @ B).transpose(0, 2, 1) @ xb / b
        gB = res.transpose(0, 2, 1) @ h / b
        return np.hstack([gA.reshape(len(w), -1), gB.reshape(len(w), -1)])

    def Ln(w):                         # full-batch L_n, up to a constant, from the second moments
        A, B = unpack(w)
        P = B @ A
        return 0.5 * np.einsum("cij,jk,cik->c", P, Sxx, P) - np.einsum("cij,ji->c", P, Sxy)

    w_star = np.tile(np.r_[A0.ravel(), B0.ravel()], (chains, 1))
    L_star = Ln(w_star[:1])[0]
    lh = sgld_llc(grad_dln, lambda w: Ln(w) - L_star, w_star, nb, gamma, eps, steps, rng)
    lam, mult = aoyagi_watanabe(M, N, H, r)
    print(f"  r={r}:  λ = {lam:4.1f} (m={mult})   λ-hat = {lh.mean():.2f} ± {lh.std() / np.sqrt(chains):.2f}")
print(f"({time.time() - t0:.1f} s)")
"""
    ),
    markdown(
        r"""
The estimates track $\lambda$, increase with $r$ as $\lambda$ does, and sit far below $d/2=12$.
They run below $\lambda$ by up to about $\beta^*\lambda$ ($\beta^*=0.13$ here), the size of the
data bias in Exercise 3.3b, and the $m=2$ case $r=0$ also carries the $1/\log$ correction of
Exercise 3.1b. Each $r$ uses its own data set; change the seed to see the scatter.

**What the hyperparameters do.** Each knob of the estimator is either a scale or a sampler
setting:

- $\beta$. Without data noise and with $\gamma=0$, $\hat\lambda(\beta)=\lambda_{\rm eff}(n\beta)$ exactly (Exercise 3.3b), and the $\beta$-sweep above is the crossover curve of §3.1 drawn against $n\beta$. In that idealisation the $\beta$-dependence of $\hat\lambda$ is exactly the $n$-dependence of $\lambda_{\rm eff}$.
- $\gamma$. A distance scale, about $\gamma^{-1/2}$ around $w^*$: directions with curvature well below $\gamma/(n\beta)$ are suppressed, and those near it are partly counted (Exercise 3.3c).
- $\epsilon$, chain length, minibatch size. Sampler accuracy only. They have no counterpart in the target distribution, so tune them until $\hat\lambda$ stops depending on them.
"""
    ),
    # ================================================================ readings
    markdown(
        r"""
<a id="slt3-readings"></a>

## Readings

Keystones: Lau et al. and Furman's notebook (top). WBIC: §3.3. Hoogland and van Wingerden: §3.1.

**Claude-suggested extras.**
"""
        + WELLING_TEH
        + references(
            book("Algebraic Geometry and Statistical Learning Theory", "https://doi.org/10.1017/CBO9780511800474",
                 "Sumio Watanabe (Cambridge University Press, 2009)"),
            book("Mathematical Theory of Bayesian Statistics", "https://doi.org/10.1201/9781315373010",
                 "Sumio Watanabe (CRC Press, 2018)"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt3_temperature", "slt3_temperature"))
