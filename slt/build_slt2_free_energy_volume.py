#!/usr/bin/env python3
"""Build SLT Notebook 2, "Free energy and basin volume" (slt2_free_energy_volume).

Shape (SLT_SPECS_v1.md, 2026-10-09): question, keystone reading, route; each `##`
section opens with its question and restates its notation, then default text,
exercises with worked solutions, and small CPU code cells. Sections: 2.1 volume
scaling (the learning coefficient as a basin-volume exponent); 2.2 the Mellin
transform (poles of zeta give the terms of the large-n expansion); 2.3 several
basins (accuracy against volume, the critical sample size, a one-paragraph
thermodynamic gesture).

Conceptual figures (one per main idea, placed before its derivation) are drawn by
assets/figures/make_slt2_free_energy_volume_figures.py. Numerical checks of the closed
forms in the solutions: checks/slt2_checks.py.
"""
from nbkit_slt import answer, book, code, code_answer, image, markdown, reading, references, run

FIG = "assets/figures/slt2_free_energy_volume_"
POST = "https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong"

CELLS = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# SLT Notebook 2 — Free energy and basin volume

**Singular Learning Theory, Iliad Intensive 2026**

> **Read the blog posts first.** The best introductions to this material are blog posts, and they are probably better than these notebooks: Liam Carroll's sequence [Distilling Singular Learning Theory](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) (DSLT 0–4), Jesse Hoogland's [Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick) and, with Stan van Wingerden, [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong); then Zach Furman's [Singular learning theory: exercises](https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises) and Olli Järviniemi's [Dialogue introduction to Singular Learning Theory](https://www.lesswrong.com/posts/CmcarN6fGgTGwGuFp/dialogue-introduction-to-singular-learning-theory). The notebooks are provided for reference: derivations, worked exercises and code to check things.

*Question:* when the Hessian at the best parameter is degenerate, what replaces
$\tfrac d2\log n$ in the free energy?

Notebook 1's $F_n=nL_n(\hat w)+\tfrac d2\log n+O(1)$ needs a positive-definite
Hessian at the minimum. Neural networks typically have flat directions at a
minimum (continuous families of parameters computing the same function), and
there Laplace's method divides by zero. What always exists is
the prior volume $V(t)$ of the parameters within loss $t$ of the minimum. If
$V(t)\sim ct^\lambda(-\log t)^{m-1}$ as $t\to0$, then
$\bar F_n\approx\lambda\log n-(m-1)\log\log n$. The exponent $\lambda$, the
*learning coefficient*, is $d/2$ in Notebook 1's case and smaller otherwise.
"""
        + reading(
            "You're Measuring Model Complexity Wrong",
            "https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong",
            "Hoogland & van Wingerden (LessWrong, 2023)",
            "The learning coefficient as the exponent of low-loss volume, why the Hessian (and parameter count) get it wrong, and what it means for the Bayesian posterior; the argument of §2.1 with pictures.",
            importance=5,
            length=2,
        )
        + r"""
**Route.**

- [2.1 Volume scaling](#slt2-s1): slice parameter space by loss; $V(t)\sim ct^\lambda(-\log t)^{m-1}$ gives $\bar F_n\approx\lambda\log n-(m-1)\log\log n$; toy losses; $\lambda\le d/2$.
- [2.2 The Mellin transform](#slt2-s2): the poles of $\zeta(z)=\int K^z\varphi$ are the terms of the large-$n$ expansion.
- [2.3 Several basins](#slt2-s3): the posterior trades accuracy against volume and switches basin at a critical $n^*$.

**Setup** (Notebook 1's). Model $p(x\mid w)$, $w\in\mathbb R^d$, prior $\varphi$;
training and population losses $L_n$, $L$; $w_0$ a minimizer of $L$; excess
losses $K=L-L(w_0)\ge0$ and $K_n=L_n-L_n(w_0)$. The free energy splits as
$F_n=-\log\int\varphi e^{-nL_n}=nL_n(w_0)+\bar F_n$, with
$\bar F_n=-\log\int\varphi e^{-nK_n}$.

**One simplification.** We work with the population excess loss $K$ in place of
$K_n$: $\bar Z_n=\int\varphi e^{-nK}$, $\bar F_n=-\log\bar Z_n$, and the
corresponding Gibbs distribution $\propto\varphi e^{-nK}$, which we still call the
posterior. Under Watanabe's assumptions (realizable, or more generally "relatively
finite variance"), the free energy computed from $K_n$ differs from this one by a
random $O(1)$ (in Notebook 1's regular case, the $\tfrac n2g_n^\top H^{-1}g_n$ term,
of mean $\tfrac12\mathrm{tr}(H^{-1}C)$). So $K$ fixes the coefficients of $\log n$
and $\log\log n$, not the constant.
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def log_Zbar(logphi_grid, K_grid, cell, n):
    # log of  sum_grid  phi * exp(-n K) * cell volume,  a Riemann sum for Zbar_n; log space avoids underflow
    return logsumexp(logphi_grid - n * K_grid) + np.log(cell)


def slope(xs, ys):
    return np.polyfit(xs, ys, 1)[0]
"""
    ),
    # ------------------------------------------------------------------ 2.1
    markdown(
        r"""
<a id="slt2-s1"></a>

## 2.1 Volume scaling

*Question:* what property of the loss landscape sets the coefficient of $\log n$
in the free energy?

*Notation:* $K\ge0$ the excess loss; $\varphi$ the prior;
$\bar Z_n=\int\varphi e^{-nK}$, $\bar F_n=-\log\bar Z_n$; $d$ the number of
parameters. New: the prior volume of the *sublevel set* $\{K<t\}$ and its density,

$$
V(t)=\int_{\{w:\,K(w)<t\}}\varphi(w)\,dw ,\qquad v(t)=V'(t).
$$

"""
        + image(
            FIG + "lw_basin_volume.png",
            r"The prior volume of a sublevel set: the region of a two-parameter loss below the threshold $\epsilon$ (blue cap) projects to the grey region of parameter space, whose volume is $V(\epsilon)$ (here $V(t)$, with the prior uniform). Figure from Jesse Hoogland and Stan van Wingerden, [*You're Measuring Model Complexity Wrong*](%s) (LessWrong, 2023)." % POST,
            alt="A 3-D loss surface over two weights, with the part below a threshold epsilon shaded and its grey projection onto the weight plane labelled V(epsilon).",
        )
        + r"""
**Slice by loss.** The integrand of $\bar Z_n$ depends on $w$ only through
$K(w)$, so group parameters by their loss: $v(t)\,dt$ is the prior mass with
$K\in[t,t+dt]$, and

$$
\bar Z_n=\int_0^\infty e^{-nt}\,v(t)\,dt=n\int_0^\infty e^{-nt}\,V(t)\,dt .
$$
"""
        + image(
            FIG + "slicing.svg",
            r"Slicing by loss. The sublevel set $\{K<t\}$ (blue) has prior mass $V(t)$, the slab $\{t<K<t+dt\}$ (orange) has mass $v(t)\,dt$, and $\bar Z_n$ weights each slab by $e^{-nt}$ (right), so at large $n$ only $t\lesssim1/n$ counts.",
            alt="A quartic loss curve with a threshold t; the interval where the loss is below t is marked on the parameter axis, and a thin band of levels just above t is marked as two short slabs; beside it, the weight e to the minus n t decays with t.",
        )
        + r"""
In Notebook 1's dictionary $v$ is the density of states and this is
$Z(\beta)=\int e^{-\beta E}g(E)\,dE$. Since $e^{-nt}$ is negligible for
$t\gg1/n$, only the behaviour of $V$ near $t=0$ matters when $n$ is large.

**The learning coefficient.** If

$$
V(t)=c\,t^{\lambda}(-\log t)^{m-1}\,(1+o(1))\quad\text{as }t\to0,
$$

then (Exercise 2.1.2, with $\Gamma$ the Gamma function)

$$
\bar F_n=\lambda\log n-(m-1)\log\log n-\log\big(c\,\Gamma(\lambda+1)\big)+o(1).
$$

$\lambda$ is the *learning coefficient* and $m$ the *multiplicity*. In these
notebooks this is the whole definition: $\lambda$ is the exponent with which
low-loss volume vanishes, $\lambda=\lim_{t\to0}\log V(t)/\log t$. Watanabe proves
that for analytic $K$ and $\varphi$, $V$ always has this form, with $\lambda$ a
positive rational number and $m\le d$ a positive integer; §2.2 gives the tool
behind that theorem.
"""
        + image(
            FIG + "sublevel2d.svg",
            r"Sublevel sets at $t=1/4,1/16,1/64$, uniform prior on the box. Each time $t$ drops by $4$, the bowl's volume drops by $4$, that of $a^2+b^4$ by $4^{3/4}\approx2.8$, and the cross's by less still: the flatter the valley, the smaller $\lambda$.",
            alt="Three square panels with nested shaded sublevel sets: concentric disks for the bowl, nested ovals for a squared plus b to the fourth, and a nested cross shape for (ab) squared.",
        )
        + image(
            FIG + "logV.svg",
            r"$\lambda$ is the limiting slope of $\log V$ against $\log t$. For $(ab)^2$ the factor $\log(1/t)$ ($m=2$) makes the slope approach $\tfrac12$ slowly.",
            alt="Log of V against log of t for the three two-dimensional losses, with slope triangles 1, 3/4 and 1/2.",
        )
        + r"""
"""
        + image(
            FIG + "lw_hessian_infinite.png",
            r"Why the Hessian fails. Left: the sublevel set (blue) of a basin whose Hessian at the minimum is degenerate. Right: the quadratic approximation built from that Hessian is flat, so its sublevel set is an unbounded slab and Laplace's method predicts infinite volume. Figure from Jesse Hoogland and Stan van Wingerden, [*You're Measuring Model Complexity Wrong*](%s) (LessWrong, 2023)." % POST,
            alt="Left, a degenerate loss basin with its finite sublevel set; right, the same basin with the flat quadratic approximation, whose sublevel set is an unbounded plane slab.",
        )
        + r"""
**Bowl versus valley.** In the regular case $K\approx\tfrac12(w-w_0)^\top H(w-w_0)$,
the sublevel set is an ellipsoid of volume $\propto t^{d/2}$, so $\lambda=d/2$,
$m=1$, and Notebook 1's Laplace formula comes back, constant included
(Exercise 2.1.3). Flat directions make the sublevel set fatter, $V(t)$ larger and
$\lambda$ smaller: at a minimum where the prior is positive, $\lambda\le d/2$
(Exercise 2.1.6), with equality in the regular case. One can read $2\lambda$ as an effective number of parameters, but it
counts nothing and need not be an integer: $K=w^4$ has $\lambda=\tfrac14$ with one
parameter.

**Exercise 2.1 (Volume and free energy).** Take the prior uniform on $[-1,1]^d$ where a prior is needed.

1. Derive $\bar Z_n=\int_0^\infty e^{-nt}v(t)dt=n\int_0^\infty e^{-nt}V(t)dt$. <details><summary>Hint</summary>Write $e^{-nK(w)}=n\int_{K(w)}^\infty e^{-nt}dt$ and swap the order of integration; or insert $1=\int\delta(t-K(w))dt$.</details>
2. Show that $V(t)=ct^\lambda$ gives $\bar F_n=\lambda\log n-\log(c\Gamma(\lambda+1))+o(1)$, and that $V(t)=ct^\lambda(-\log t)^{m-1}$ gives the extra $-(m-1)\log\log n$. Say why only small $t$ matters even though the formula for $V$ is only assumed near $0$. <details><summary>Hint</summary>Substitute $t=s/n$ and use $\log n-\log s=\log n\,(1-\log s/\log n)$.</details>
3. Regular case: $K=\tfrac12(w-w_0)^\top H(w-w_0)$ with $H$ positive definite, and a smooth prior with $\varphi(w_0)>0$. Compute $V(t)$ to leading order (the volume of the $d$-dimensional unit ball is $\omega_d=\pi^{d/2}/\Gamma(d/2+1)$, and the ellipsoid $\{u:u^\top Au<1\}$ has volume $\omega_d(\det A)^{-1/2}$), read off $\lambda$ and $m$, and show that part 2 reproduces Notebook 1's $\bar F_n=\tfrac d2\log n+\tfrac12\log\det H-\log\varphi(w_0)-\tfrac d2\log2\pi+o(1)$.
4. Compute $V(t)$ and hence $\lambda$ and $m$ for: (a) $K=w^2$; (b) $K=w^4$; (c) $K=w^{2k}$; (d) $K=w^2+v^4$ (two parameters); (e) $K=(ab)^2$ (two parameters, zero on both axes); (f) the cubically reparametrized quadratic $K=\tfrac12(\mu^3-\mu_0^3)^2$, for $\mu_0=0$ and for $\mu_0\ne0$. What does (f) say about whether $\lambda$ is a property of the set of distributions $\{p_w\}$?
5. *(Sum rule.)* If $K(w,v)=K_1(w)+K_2(v)$ and $\varphi=\varphi_1(w)\varphi_2(v)$, show $\lambda=\lambda_1+\lambda_2$ and $m-1=(m_1-1)+(m_2-1)$. Check against 4(d).
6. *(Upper bound.)* Let $w_0$ be an interior minimizer of $K$ with $K(w_0)=0$, the prior continuous and positive at $w_0$, and $\Lambda$ an upper bound on the eigenvalues of $\nabla^2K$ on a ball around $w_0$. Show $K(w)\le\tfrac12\Lambda\|w-w_0\|^2$ there, deduce that the sublevel set contains a ball of radius $\sqrt{2t/\Lambda}$, and conclude $\lambda\le d/2$. <details><summary>Hint</summary>Taylor's theorem with the remainder in the form $K(w)=K(w_0)+\tfrac12(w-w_0)^\top\nabla^2K(\xi)(w-w_0)$ for some $\xi$ on the segment, and $u^\top Au\le\Lambda\|u\|^2$ when $\Lambda$ bounds the eigenvalues of the symmetric matrix $A$.</details>
"""
    ),
    answer(
        r"""
**1.** $\bar Z_n=\int\varphi(w)e^{-nK(w)}dw=\int\varphi(w)\,n\int_{K(w)}^\infty e^{-nt}dt\,dw=n\int_0^\infty e^{-nt}\Big(\int_{K(w)<t}\varphi(w)dw\Big)dt=n\int_0^\infty e^{-nt}V(t)dt$. Integrating by parts ($V(0)=0$, and $e^{-nt}V(t)\to0$ at $\infty$ since $V$ is bounded by the prior mass $1$) gives $\int_0^\infty e^{-nt}V'(t)dt=\int e^{-nt}v(t)dt$.

**2.** With $t=s/n$: $\bar Z_n=n\int_0^\infty e^{-nt}ct^\lambda dt=cn^{-\lambda}\int_0^\infty e^{-s}s^\lambda ds=c\,\Gamma(\lambda+1)\,n^{-\lambda}$, so $\bar F_n=\lambda\log n-\log(c\Gamma(\lambda+1))$. With the logarithm, $(-\log t)^{m-1}=(\log n-\log s)^{m-1}=(\log n)^{m-1}(1-\log s/\log n)^{m-1}$, and the integral against $e^{-s}s^\lambda$ of the bracket tends to $1$, so $\bar Z_n=c\Gamma(\lambda+1)n^{-\lambda}(\log n)^{m-1}(1+O(1/\log n))$, giving the extra $-(m-1)\log\log n$. (The correction is only $O(1/\log n)$: the $\log\log n$ term converges slowly, which matters in the code below.) Only small $t$ matters because the substitution shows that the integrand is concentrated at $s=O(1)$, i.e. $t=O(1/n)$: the region $t>\epsilon$ contributes at most $e^{-n\epsilon}$, exponentially small, whatever $V$ does there. (This is Watson's lemma: the large-$n$ expansion of a Laplace transform is determined by the small-$t$ expansion of the function.)

**3.** The sublevel set is the ellipsoid $\{u:u^\top Hu<2t\}=\{u:u^\top(H/2t)u<1\}$, of volume $\omega_d\det(H/2t)^{-1/2}=\omega_d(2t)^{d/2}(\det H)^{-1/2}$; multiplying by $\varphi(w_0)$ (the prior is approximately constant on a set of diameter $O(\sqrt t)$), $V(t)\approx\varphi(w_0)\omega_d(\det H)^{-1/2}2^{d/2}t^{d/2}$, so $\lambda=d/2$, $m=1$, $c=\varphi(w_0)2^{d/2}\pi^{d/2}(\det H)^{-1/2}/\Gamma(d/2+1)$. Then $c\Gamma(\lambda+1)=\varphi(w_0)(2\pi)^{d/2}(\det H)^{-1/2}$ and part 2 gives $\bar F_n=\tfrac d2\log n-\log\varphi(w_0)+\tfrac12\log\det H-\tfrac d2\log2\pi$, Notebook 1's formula. Volume scaling is Laplace's method with the Hessian replaced by what is actually there.

**4.** Prior density $2^{-d}$ on the box. (a) $\{w^2<t\}=(-\sqrt t,\sqrt t)$, $V=\sqrt t$ (after the factor $\tfrac12$): $\lambda=\tfrac12$, $m=1$. (b) $V=t^{1/4}$: $\lambda=\tfrac14$. (c) $V=t^{1/2k}$: $\lambda=1/2k$; the flatter the minimum, the smaller $\lambda$, although the Hessian is zero for every $k\ge2$ and cannot tell them apart. (d) $V(t)=\tfrac14\int_{|v|<t^{1/4}}2\sqrt{t-v^4}\,dv=\tfrac12t^{3/4}\int_{-1}^1\sqrt{1-u^4}\,du$ (substituting $v=t^{1/4}u$), so $\lambda=\tfrac34$, $m=1$. (e) $\{(ab)^2<t\}=\{|ab|<\sqrt t\}$. For $|a|<\sqrt t$ every $b\in[-1,1]$ qualifies; for $\sqrt t<|a|<1$ one needs $|b|<\sqrt t/|a|$. Area $=2\sqrt t\cdot2+2\int_{\sqrt t}^1\frac{2\sqrt t}{a}da=4\sqrt t+4\sqrt t\cdot\tfrac12(-\log t)=2\sqrt t(-\log t)+4\sqrt t$. Dividing by $4$: $V(t)=\tfrac12t^{1/2}(-\log t)\,(1+2/(-\log t))$, so $\lambda=\tfrac12$ and $m=2$: the zero set is two crossing lines, and where they cross the volume picks up a logarithm. (f) For $\mu_0=0$, $K=\mu^6/2$ and $V=\tfrac12\cdot2(2t)^{1/6}=(2t)^{1/6}$: $\lambda=\tfrac16$. For $\mu_0\ne0$ the minimum is at $\mu_0$ with $K''(\mu_0)=9\mu_0^4>0$, so $\lambda=\tfrac12$. The map $\theta=\mu^3$ is a bijection of $\mathbb R$, so both parametrizations describe exactly the same family of distributions, yet $\lambda$ differs. The reason: a uniform prior on $\mu$ and a uniform prior on $\theta$ are different priors on that family. The learning coefficient belongs to the parametrization together with its prior, not to the set of distributions alone; transforming the prior along with the coordinates (with its Jacobian) leaves $V(t)$, and so $\lambda$, unchanged.

**5.** $\bar Z_n=\int\varphi_1e^{-nK_1}\int\varphi_2e^{-nK_2}=\bar Z_n^{(1)}\bar Z_n^{(2)}$, so $\bar F_n=\bar F_n^{(1)}+\bar F_n^{(2)}$ and the coefficients of $\log n$ and $\log\log n$ add: $\lambda=\lambda_1+\lambda_2$, $(m-1)=(m_1-1)+(m_2-1)$. For $w^2+v^4$: $\tfrac12+\tfrac14=\tfrac34$, as in 4(d).

**6.** Taylor's theorem with $\nabla K(w_0)=0$ and $K(w_0)=0$ gives $K(w)=\tfrac12(w-w_0)^\top\nabla^2K(\xi)(w-w_0)\le\tfrac12\Lambda\|w-w_0\|^2$. Hence $\|w-w_0\|<\sqrt{2t/\Lambda}$ implies $K(w)<t$: the ball of radius $r=\sqrt{2t/\Lambda}$ sits inside the sublevel set, and $V(t)\ge\varphi_{\min}\omega_dr^d=\varphi_{\min}\omega_d(2/\Lambda)^{d/2}t^{d/2}$ for small $t$ (with $\varphi_{\min}>0$ the prior's minimum on the ball). Then $\lambda=\lim\log V/\log t\le\lim\log(Ct^{d/2})/\log t=d/2$ (recall $\log t<0$, so a larger $V$ means a smaller ratio). A bounded Hessian guarantees a ball inside the sublevel set, so the volume shrinks no faster than a ball's, $t^{d/2}$. (Without $\varphi(w_0)>0$ the bound fails: $K=w^2$ with prior $\tfrac32w^2$ on $[-1,1]$ has $V=t^{3/2}$.)
"""
    ),
    markdown(
        r"""
**Code: $\bar F_n$ against $\log n$ for the toy losses.** A Riemann sum for
$\bar Z_n$ on a grid, uniform prior on $[-1,1]^d$, for four losses of
Exercise 2.1.4, with the slope of $\bar F_n$ against $\log n$ fitted over
$n\ge300$. For $(ab)^2$ the $-\log\log n$ term pulls the slope below $\tfrac12$ by
about $1/\log n$, which decays slowly; fitting the full form
$\lambda\log n-(m-1)\log\log n+c$ with $m=2$ removes most of the bias.
"""
    ),
    code(
        r"""
w1 = np.linspace(-1, 1, 400_001)                      # 1-D grid
A, B = np.meshgrid(np.linspace(-1, 1, 2001), np.linspace(-1, 1, 2001), indexing="ij")   # 2-D grid
ns = np.logspace(1, 4, 13)

toys = {
    r"$w^2$ ($\lambda=1/2$)":        (np.full_like(w1, np.log(0.5)), w1**2, w1[1] - w1[0], 0.5, 1),
    r"$w^4$ ($\lambda=1/4$)":        (np.full_like(w1, np.log(0.5)), w1**4, w1[1] - w1[0], 0.25, 1),
    r"$w^2+v^4$ ($\lambda=3/4$)":    (np.full_like(A, np.log(0.25)), A**2 + B**4, (A[1, 0] - A[0, 0]) ** 2, 0.75, 1),
    r"$(ab)^2$ ($\lambda=1/2,\,m=2$)": (np.full_like(A, np.log(0.25)), (A * B) ** 2, (A[1, 0] - A[0, 0]) ** 2, 0.5, 2),
}
plt.figure(figsize=(6, 3.8))
for label, (logphi, K, cell, lam, m) in toys.items():
    Fbar = np.array([-log_Zbar(logphi, K, cell, n) for n in ns])
    plt.plot(np.log(ns), Fbar, "o-", ms=4, label=label)
    big = ns >= 300
    s_plain = slope(np.log(ns[big]), Fbar[big])
    # fit  Fbar = lam * log n - (m-1) log log n + c  with the known m, to recover lam
    X = np.stack([np.log(ns[big]), np.ones(big.sum())], 1)
    s_full = np.linalg.lstsq(X, Fbar[big] + (m - 1) * np.log(np.log(ns[big])), rcond=None)[0][0]
    print(f"{label:34s} slope vs log n (n>=300): {s_plain:.3f};  with the log log n term: {s_full:.3f};  lambda = {lam}")
plt.xlabel(r"$\log n$"); plt.ylabel(r"$\bar F_n$"); plt.legend(); plt.title("free energy of four toy losses, uniform prior")
plt.tight_layout(); plt.show()
"""
    ),
    # ------------------------------------------------------------------ 2.2
    markdown(
        r"""
<a id="slt2-s2"></a>

## 2.2 The Mellin transform

*Question:* can we get $\lambda$, $m$ and the corrections to the free energy
without computing $V(t)$ by hand?

*Notation:* $K$, $\varphi$, $V$, $v$, $\bar Z_n$ as in §2.1. New: for a complex
number $z$,

$$
\zeta(z)=\int K(w)^z\,\varphi(w)\,dw=\int_0^\infty t^z\,v(t)\,dt .
$$

The second form (the slicing of §2.1) makes $\zeta$ the *Mellin transform* of
$v$, which does for powers of $t$ what the Fourier transform does for
exponentials (the usual convention has $t^{z-1}$ in place of $t^z$). For bounded
$K$ and a prior of bounded support the integral converges for $\mathrm{Re}\,z>0$.
Moving left, it first diverges where $t^zv(t)$ stops being integrable at $t=0$,
which the small-$t$ behaviour of $v$ decides. Precisely (Exercise 2.2.1):

$$
\int_0^1t^{z}\cdot t^{\lambda-1}(-\log t)^{m-1}\,dt=\frac{(m-1)!}{(z+\lambda)^m}:
$$

a term $ct^{\lambda-1}(-\log t)^{m-1}$ in $v(t)$ is a pole of $\zeta$ at
$z=-\lambda$ of order $m$. Inverting the Mellin transform runs this backwards, and
§2.1 turns each such term of $v$ into a term of order $n^{-\lambda}(\log n)^{m-1}$
in $\bar Z_n$:

$$
\text{poles of }\zeta\ \longleftrightarrow\ \text{small-}t\text{ terms of }v\ \longleftrightarrow\ \text{large-}n\text{ terms of }\bar Z_n .
$$
"""
        + image(
            FIG + "poles.svg",
            r"Each pole of $\zeta$, at $-\lambda_k$ with order $m_k$, gives terms of $v(t)$ near $t=0$ and of $\bar Z_n$ at large $n$ with power $\lambda_k$ and at most $m_k-1$ logarithms; the figure shows the highest. The integral converges to the right of the first pole, which gives $\lambda$ and $m$.",
            alt="The real axis of the complex z-plane with poles marked at minus lambda, minus lambda two and further left; arrows connect each pole to a term of v(t) at small t and to a term of the partition function at large n.",
        )
        + r"""
The pole closest to the origin (the largest, since all are negative) gives
$\lambda$, its order gives $m$, and the further poles give the corrections.
Watanabe proves (we don't) that for analytic $K$ and $\varphi$, $\zeta$ extends to
the whole plane with only poles, all at negative rational numbers: a local change
of variables turns $K$ into a monomial $u_1^{2k_1}\cdots u_d^{2k_d}$, whose
$\zeta$ is a product of one-dimensional integrals.

**Monomials take one line.** Where $V(t)$ needed the case analysis of
Exercise 2.1.4(e), for $K=(ab)^2$ with the uniform prior on $[-1,1]^2$

$$
\zeta(z)=\frac14\int_{-1}^1\!\!\int_{-1}^1(ab)^{2z}\,da\,db=\frac14\Big(\frac2{2z+1}\Big)^2=\frac1{(2z+1)^2}=\frac14\cdot\frac1{(z+\tfrac12)^2},
$$

a double pole at $z=-\tfrac12$: $\lambda=\tfrac12$ and $m=2$.

**Exercise 2.2 (Poles and exponents).** Prior uniform on $[-1,1]^d$ unless stated.

1. Show $\zeta(z)=\int_0^\infty t^zv(t)dt$, and prove $\int_0^1t^{z+\lambda-1}(-\log t)^{m-1}dt=(m-1)!/(z+\lambda)^m$ for $\mathrm{Re}(z+\lambda)>0$. <details><summary>Hint</summary>Substitute $t=e^{-u}$; the result is a Gamma-function integral.</details>
2. Compute $\zeta$ and read off $(\lambda,m)$ for $K=w^2$, $w^4$, $w^{2k}$, $\tfrac12\mu^6$ (the cubic reparametrization at $\mu_0=0$), and the monomial $K=w_1^{2k_1}w_2^{2k_2}$. For the last one, state the rule for $\lambda$ and $m$ in terms of $k_1,k_2$ and extend it to $d$ variables.
3. For $K=w^2+v^4$ the integral does not factorize. Use $v(t)=V'(t)$ from Exercise 2.1.4(d) to locate the first pole of $\zeta$ and confirm $\lambda=\tfrac34$.
4. With a standard normal prior instead of the uniform one, compute $\zeta(z)=\mathbb E[w^{2z}]$ for $K=w^2$ and find all its poles. Which pole gives $\lambda$? What do the others give? <details><summary>Hint</summary>$\mathbb E[|w|^{p}]=2^{p/2}\Gamma(\tfrac{p+1}2)/\sqrt\pi$ for a standard normal $w$ and $p>-1$, and $\Gamma$ has simple poles at $0,-1,-2,\dots$.</details>
5. Explain in a sentence why changing the prior to any other smooth positive density cannot change $\lambda$ or $m$, but can change the constant $c$.
"""
    ),
    answer(
        r"""
**1.** $\int K(w)^z\varphi(w)dw=\int\varphi(w)\int_0^\infty t^z\delta(t-K(w))\,dt\,dw=\int_0^\infty t^z\Big(\int\varphi\,\delta(t-K)\Big)dt$, and the bracket is $v(t)=\frac d{dt}\int_{K<t}\varphi$. For the integral: with $t=e^{-u}$, $dt=-e^{-u}du$, $-\log t=u$, the integral becomes $\int_0^\infty e^{-(z+\lambda)u}u^{m-1}du=\Gamma(m)/(z+\lambda)^m=(m-1)!/(z+\lambda)^m$.

**2.** $K=w^2$: $\zeta=\tfrac12\int_{-1}^1w^{2z}dw=\frac1{2z+1}=\frac12\cdot\frac1{z+1/2}$, simple pole at $-\tfrac12$: $\lambda=\tfrac12$, $m=1$. $K=w^4$: $\frac1{4z+1}$, pole at $-\tfrac14$. $K=w^{2k}$: $\frac1{2kz+1}$, pole at $-\frac1{2k}$. $K=\tfrac12\mu^6$: $\zeta=2^{-z}\frac1{6z+1}$, pole at $-\tfrac16$ (the factor $2^{-z}$ is entire and only changes the residue, i.e. the constant). Monomial: $\zeta=\prod_i\frac1{2k_iz+1}$, simple poles at $z=-1/2k_i$; the pole closest to $0$ is at $-\min_i1/2k_i$, so $\lambda=\min_i\frac1{2k_i}$ and $m$ is the number of $i$ attaining the minimum (coincident simple poles multiply into a pole of that order). In $d$ variables the same: $\lambda=\min_i1/2k_i$, $m=\#\{i:1/2k_i=\lambda\}$. Note that variables that do not appear ($k_i=0$) contribute a factor $1$ and nothing to $\lambda$: directions along which the loss is flat cost nothing.

**3.** $V(t)=\tfrac12I\,t^{3/4}$ with $I=\int_{-1}^1\sqrt{1-u^4}du$, so $v(t)=\tfrac38I\,t^{-1/4}$ for small $t$, and $\int_0^1t^z\cdot t^{-1/4}dt=\frac1{z+3/4}$: first pole at $-\tfrac34$, $\lambda=\tfrac34$, $m=1$. In fact $V(t)=\tfrac12It^{3/4}$ exactly for $t\le1$, and the contribution $\int_1^2t^zv(t)\,dt$ is an entire function of $z$, so this is the only pole: here $\zeta(z)=\frac{3I/8}{z+3/4}+(\text{entire})$.

**4.** $\zeta(z)=\mathbb E[w^{2z}]=2^z\Gamma(z+\tfrac12)/\sqrt\pi$. Poles of $\Gamma(z+\tfrac12)$ are at $z+\tfrac12=0,-1,-2,\dots$, i.e. $z=-\tfrac12,-\tfrac32,-\tfrac52,\dots$, all simple. The first gives $\lambda=\tfrac12$, $m=1$ as with the uniform prior. The others give the terms $n^{-3/2},n^{-5/2},\dots$ in $\bar Z_n$, i.e. the $1/n,1/n^2,\dots$ corrections to $\bar F_n$; here $\bar Z_n=\int e^{-nw^2}e^{-w^2/2}/\sqrt{2\pi}\,dw=(1+2n)^{-1/2}$ exactly, whose expansion in $1/n$ has precisely those powers.

**5.** For a smooth positive prior, on a small neighbourhood of the zero set $\varphi$ is squeezed between two positive constants, $\varphi_{\min}\le\varphi\le\varphi_{\max}$, so $V(t)$ is squeezed between two multiples of the uniform-prior volume; the exponent and the power of the logarithm survive the squeeze, the constant $c$ does not. (The support matters: a prior that vanishes on part of the zero set changes which minima are available at all.)
"""
    ),
    # ------------------------------------------------------------------ 2.3
    markdown(
        r"""
<a id="slt2-s3"></a>

## 2.3 Several basins

*Question:* when the loss has several minima with different depths and
different volumes, where does the posterior go, and does that change with $n$?

*Notation:* basins $U_\alpha$ (regions each containing one minimum $w_\alpha$ of
$L$) with local free energies $F_n(U_\alpha)=-\log\int_{U_\alpha}\varphi e^{-nL_n}$
(Notebook 1, Exercise 1.1); $L_\alpha=L(w_\alpha)$; $\lambda_\alpha$, $m_\alpha$ the
learning coefficient and multiplicity of $L-L_\alpha$ inside $U_\alpha$.
"""
        + image(
            FIG + "two_basins.svg",
            r"Accuracy against volume: a narrow basin $U$ at excess loss $0$, a wide basin $V$ at $\Delta L$ (the code cell's loss with $\delta=0.03$), and the posterior at two sample sizes. At $n=10$ most of the mass is in the wide basin; at $n=300$ nearly all of it is in the accurate one.",
            alt="A loss with a narrow basin at zero and a wide basin slightly higher; below, the posterior at n = 10 has most mass in the wide basin, and at n = 300 nearly all in the narrow one.",
        )
        + r"""
Applying §2.1 inside each basin (population version, as throughout),

$$
F_n(U_\alpha)\approx nL_\alpha+\lambda_\alpha\log n-(m_\alpha-1)\log\log n+c_\alpha ,
\qquad
F_n=-\log\sum_\alpha e^{-F_n(U_\alpha)}\approx\min_\alpha F_n(U_\alpha)
$$

(Notebook 1, Exercise 1.1.3). Basin $\alpha$ holds posterior mass
$e^{-F_n(U_\alpha)}/\sum_\beta e^{-F_n(U_\beta)}$, so the posterior sits in the
basin of least free energy, trading accuracy, $nL_\alpha$, against volume,
$\lambda_\alpha\log n$. (With real data, $nL_n(w_\alpha)$ fluctuates around
$nL_\alpha$ by $O(\sqrt n)$, and between basins that compute different functions
these fluctuations do not cancel; they blur the comparison below.)

**The trade changes with $n$.** Take $U$ accurate but narrow and $V$ less accurate
but wide: $L_U<L_V$ and $\lambda_U>\lambda_V$. The log posterior odds are

$$
\log\frac{\pi_n(U)}{\pi_n(V)}=F_n(V)-F_n(U)\approx n\,\Delta L-\Delta\lambda\,\log n,
\qquad \Delta L=L_V-L_U>0,\ \ \Delta\lambda=\lambda_U-\lambda_V>0 .
$$

Keeping only these leading terms (dropping $c_\alpha$ and the $\log\log n$
terms), and when $\Delta\lambda/\Delta L>e$: at moderate $n$ the logarithm wins and
the posterior sits in the wide basin $V$; at large $n$ the linear term wins and it
moves to the accurate basin $U$. The switch is at

$$
\frac{n^*}{\log n^*}=\frac{\Delta\lambda}{\Delta L}.
$$

A small loss of accuracy bought with a large gain in volume keeps the posterior in
the wide basin for a long time. This is Notebook 1's BIC comparison between two
models, now between two regions of one model's parameter space.

**The thermodynamic reading.** Treating $n$ as continuous,
$\partial_n\bar F_n=\mathbb E_{\pi_n}[K]$ (Exercise 2.3.3): the free energy
differentiated in the "inverse temperature" is the "internal energy". Around $n^*$
it drops from about $\Delta L+\lambda_V/n$ to about $\lambda_U/n$, and in the
dictionary a jump in the first derivative of the free energy is a first-order
phase transition. The analogy stops there. No thermodynamic limit makes the jump
sharp: the log-odds have slope $\Delta L-\Delta\lambda/n^*$ at $n^*$, so they go from $-5$
to $5$ over about $10/(\Delta L-\Delta\lambda/n^*)$ samples ($60$ in Exercise 2.3.2). And the posterior has no dynamics, hence no metastable states:
past $n^*$ it is in $U$, whatever it was before. Metastability needs a procedure
that moves locally, such as a sampler or an optimizer started in $V$, which stays
there past $n^*$ until it crosses the barrier, on a time set by the barrier rather
than by the free energies. Whether that describes training is not a question
these notebooks answer.

**Exercise 2.3 (Two basins).**

1. Derive the log-odds formula and $n^*/\log n^*=\Delta\lambda/\Delta L$ from Notebook 1 Exercise 1.1.2 and §2.1. Show that if instead $\Delta L$ and $\Delta\lambda$ have opposite signs (one basin is both more accurate and wider), these leading terms never cross.
2. Suppose $F_n(U)\approx0.3n+20\log n$ and $F_n(V)\approx0.5n+2\log n$. Which basin is the accurate one, and which the wide one? Find $n^*$, and say what the posterior odds are at $n=100$ and at $n=2000$.
3. Treating $n$ as a continuous variable in $\bar Z_n=\int\varphi e^{-nK}$, show $\partial_n\bar F_n=\mathbb E_{\pi_n}[K]$ and $\partial_n^2\bar F_n=-\mathrm{Var}_{\pi_n}[K]$. Conclude that the posterior mean excess loss decreases with $n$, and say what it is in a single basin with learning coefficient $\lambda$ for large $n$.
4. *(Several basins, several scales.)* Three basins have $(L_\alpha,\lambda_\alpha)=(0,\,2),(0.01,\,1),(0.05,\,\tfrac14)$. Sketch which basin holds the posterior as $n$ runs from $10$ to $10^5$, and give the two crossover values.
"""
    ),
    answer(
        r"""
**1.** Notebook 1 Exercise 1.1.2: $\log\pi_n(U)/\pi_n(V)=F_n(V)-F_n(U)$. Insert the expansions: $F_n(V)-F_n(U)\approx n(L_V-L_U)+(\lambda_V-\lambda_U)\log n+O(\log\log n)$, which is $n\Delta L-\Delta\lambda\log n$ with the stated signs (dropping the slowly varying terms). For moderate $n$ the logarithmic term dominates and the expression is negative; for large $n$ the linear term wins and it is positive; it crosses zero where $n\Delta L=\Delta\lambda\log n$. The function $n/\log n$ is increasing for $n>e$, so the equation $n/\log n=\Delta\lambda/\Delta L$ has exactly one solution above $e$ when the right-hand side exceeds $e$ (the case of interest, $\Delta\lambda\gg\Delta L$). If $\Delta L$ and $\Delta\lambda$ have opposite signs, the two terms $n\Delta L$ and $-\Delta\lambda\log n$ have the same sign for every $n>1$: one basin is preferred throughout, and the odds only grow.

**2.** $U$ has the lower loss coefficient ($0.3<0.5$): it is the accurate basin, and it has the larger $\lambda$ ($20>2$): it is the narrow one. $\Delta L=0.2$, $\Delta\lambda=18$, so $n^*/\log n^*=90$, giving $n^*\approx570$ ($570/\log570=570/6.35\approx89.8$). At $n=100$: $F(V)-F(U)=0.2\cdot100-18\log100=20-82.9=-62.9$: the posterior is in $V$ by odds $e^{-63}$. At $n=2000$: $400-18\cdot7.6=263$: in $U$ by odds $e^{263}$. The crossover is sharp on the scale of the odds but it is not instantaneous in $n$: the log-odds pass through $[-5,5]$ over a window of about $60$ samples around $n^*$: the slope of the log-odds there is $\Delta L-\Delta\lambda/n^*=0.2-18/570\approx0.17$, and $10/0.17\approx60$.

**3.** $\partial_n\log\bar Z_n=\frac{\int(-K)\varphi e^{-nK}}{\int\varphi e^{-nK}}=-\mathbb E_{\pi_n}[K]$, so $\partial_n\bar F_n=\mathbb E_{\pi_n}[K]$; differentiating again as in Notebook 1 Exercise 1.1.4 gives $-\mathrm{Var}_{\pi_n}[K]\le0$. So $\mathbb E_{\pi_n}[K]$ is non-increasing in $n$: increasing $n$ in this fixed-$K$ Gibbs distribution lowers its mean excess loss. In a single basin, $\bar F_n\approx\lambda\log n-(m-1)\log\log n+c$ gives $\mathbb E_{\pi_n}[K]\approx\lambda/n$ (the $\log\log$ term contributes $-(m-1)/(n\log n)$, smaller): the posterior sits at excess loss $\lambda/n$, the "equipartition" of this problem, with $\lambda$ in place of $d/2$. Notebook 3 uses this identity as the definition of an estimator.

**4.** Pairwise crossovers (leading terms only): between basins 2 and 3, $n/\log n=\tfrac34/0.04=18.75$, so $n\approx83$; between 1 and 2, $n/\log n=1/0.01=100$, so $n\approx650$; between 1 and 3 (not needed, but for consistency) $n/\log n=1.75/0.05=35$, $n\approx180$, which lies between the other two, so basin 2 really does take over. The posterior is in basin 3 (widest, worst) for $n\lesssim83$, in basin 2 for $83\lesssim n\lesssim650$, and in basin 1 (narrowest, exact) after that: a staircase of increasingly accurate, increasingly narrow basins, with the range of $n$ over which each dominates set by the ratios $\Delta\lambda/\Delta L$.
"""
    ),
    markdown(
        r"""
**Code: a posterior that changes basin.** The loss of the figure above,
$K(w)=(w+1)^2(w-1)^4+\delta\,s(w)$, with $s$ a smooth step from $0$ (for
$w\le-\tfrac12$) to $1$ (for $w\ge\tfrac12$), now with $\delta=0.01$: a narrow
basin at $w=-1$ (excess loss $0$, $\lambda=\tfrac12$) and a wide one at $w=+1$
(excess loss $\delta$, $\lambda=\tfrac14$; the step is exactly flat on both
basins). The first cell plots the posterior mass of $w>0$, $\bar F_n$, and its
derivative $\mathbb E_{\pi_n}[K]$ against $n$. The marked $n^*$ solves
$n^*/\log n^*=\tfrac14/\delta$, where the leading terms balance; the $O(1)$
constants $c_\alpha$ move the actual crossing by a factor of order one. The second
cell is Exercise 2.3.2 as a plot.
"""
    ),
    code(
        r"""
delta = 0.01
w = np.linspace(-2.5, 2.5, 200_001)
x = np.clip(w + 0.5, 0, 1)
step = x**3 * (10 - 15 * x + 6 * x**2)              # smooth step: 0 for w <= -1/2, 1 for w >= 1/2
K = (w + 1) ** 2 * (w - 1) ** 4 + delta * step
logphi = np.full_like(w, -np.log(5.0))            # uniform prior on [-2.5, 2.5]
dw = w[1] - w[0]
ns = np.logspace(0.5, 3.5, 61)
right_mass, Fbar, EK = [], [], []
for n in ns:
    logp = logphi - n * K
    logZ = logsumexp(logp) + np.log(dw)
    p = np.exp(logp - logZ) * dw
    right_mass.append(p[w > 0].sum()); Fbar.append(-logZ); EK.append((p * K).sum())
# n* from  n / log n = (1/2 - 1/4) / delta
target = 0.25 / delta
n_star = ns[np.argmin(np.abs(ns / np.log(ns) - target))]

fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
ax[0].plot(ns, right_mass); ax[0].axvline(n_star, color="k", ls=":", label=f"$n^*\\approx{n_star:.0f}$")
ax[0].set_xscale("log"); ax[0].set_xlabel("n"); ax[0].set_ylabel("posterior mass of the wide basin (w > 0)"); ax[0].legend()
ax[1].plot(np.log(ns), Fbar); ax[1].set_xlabel(r"$\log n$"); ax[1].set_ylabel(r"$\bar F_n$")
ax[2].loglog(ns, EK, label=r"$\mathbb{E}_{\pi_n}[K]$")
ax[2].loglog(ns, delta + 0.25 / ns, ":", label=r"wide basin: $\delta + \lambda_V/n$, $\lambda_V=1/4$")
ax[2].loglog(ns, 0.5 / ns, ":", label=r"narrow basin: $\lambda_U/n$, $\lambda_U=1/2$")
ax[2].set_xlabel("n"); ax[2].legend(fontsize=8); ax[2].axvline(n_star, color="k", ls=":")
plt.tight_layout(); plt.show()
"""
    ),
    code_answer(
        r"""
# Exercise 2.3.2: plot F_n(U), F_n(V), the exact F_n = -log(e^{-F_U} + e^{-F_V}), and dF_n/dn, for n in [10, 3000].
n = np.linspace(10, 3000, 3000)
F_U = 0.3 * n + 20 * np.log(n)
F_V = 0.5 * n + 2 * np.log(n)
# TODO: F_total, the minimum of the two, and the derivative dF_total/dn (np.gradient); mark n* ~ 570
""",
        r"""
n = np.linspace(10, 3000, 3000)
F_U = 0.3 * n + 20 * np.log(n)
F_V = 0.5 * n + 2 * np.log(n)
F_tot = -np.logaddexp(-F_U, -F_V)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(n, F_tot - 0.3 * n, label=r"$F_n - 0.3n$ (exact)")
ax[0].plot(n, np.minimum(F_U, F_V) - 0.3 * n, "--", label=r"$\min(F_U, F_V) - 0.3n$")
ax[0].axvline(570, color="k", ls=":"); ax[0].set_xlabel("n"); ax[0].legend()
ax[1].plot(n, np.gradient(F_tot, n)); ax[1].axhline(0.5, ls=":", color="C1"); ax[1].axhline(0.3, ls=":", color="C1")
ax[1].axvline(570, color="k", ls=":"); ax[1].set_xlabel("n"); ax[1].set_ylabel(r"$dF_n/dn$")
plt.tight_layout(); plt.show()
print("log posterior odds U:V at n = 100, 570, 2000:", [round(float(0.2 * m - 18 * np.log(m)), 1) for m in (100, 570, 2000)])
""",
    ),
    # ------------------------------------------------------------------ readings
    markdown(
        r"""
<a id="slt2-readings"></a>

## Readings

The keystone is Hoogland and van Wingerden's post (top of the page).
"""
        + reading(
            "Singular learning theory: exercises",
            "https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises",
            "Furman (LessWrong, 2024)",
            "A problem set covering the volume computations of §2.1 and more, with the same toy losses; several exercises in the older Iliad worksheet were adapted from it.",
            importance=3,
            length=3,
        )
        + reading(
            "Recent Advances in Algebraic Geometry and Bayesian Statistics",
            "https://arxiv.org/abs/2211.10049",
            "Watanabe (2022)",
            "The author's own survey of the theory this notebook sketches: the free energy formula, the role of $\\zeta$ and its poles, and the generalization error, stated without the proofs.",
            importance=3,
            length=4,
        )
        + references(
            book("Algebraic Geometry and Statistical Learning Theory", "https://doi.org/10.1017/CBO9780511800474", "Watanabe (Cambridge University Press, 2009)", "chapter 4 for $\\zeta$ and the free energy expansion"),
            book("Mathematical Theory of Bayesian Statistics", "https://doi.org/10.1201/9781315373010", "Watanabe (CRC Press, 2018)", "chapters 5 and 6"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt2", "slt2_free_energy_volume"))
