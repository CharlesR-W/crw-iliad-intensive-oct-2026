#!/usr/bin/env python3
"""Build SLT Notebook 2, "Free energy and basin volume" (slt2_free_energy_volume).

Shape (SLT_SPECS_v1.md, 2026-10-09): question, keystone reading, route; each `##`
section opens with its question and restates its notation, then default text,
exercises with worked solutions, and small CPU code cells. Sections: 2.1 volume
scaling (the learning coefficient as a basin-volume exponent); 2.2 the Mellin
transform (poles of zeta give the terms of the large-n expansion); 2.3 several
basins (accuracy against volume, the critical sample size, a one-paragraph
thermodynamic gesture).

Numerical checks of the closed forms in the solutions: checks/slt2_checks.py.
"""
from nbkit_slt import answer, book, code, code_answer, markdown, reading, references, run

CELLS = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# SLT Notebook 2 — Free energy and basin volume

**Singular Learning Theory, Iliad Intensive 2026**

*Question:* when the Hessian at the best parameter is degenerate, what replaces
$\tfrac d2\log n$ in the free energy?

Notebook 1 derived $F_n=nL_n(\hat w)+\tfrac d2\log n+O(1)$ for a model whose loss has
a positive-definite Hessian at its minimum. Neural networks never satisfy that:
the set of parameters that compute a given function has positive dimension, so
some Hessian eigenvalues vanish, and Laplace's method divides by zero. This
notebook replaces the Hessian by something that always exists, the prior volume
of the set of parameters whose loss is within $t$ of the minimum. How that volume
scales as $t\to0$, $V(t)\sim ct^\lambda(-\log t)^{m-1}$, is all the free energy
needs: $\bar F_n\approx\lambda\log n-(m-1)\log\log n$. The exponent $\lambda$ is
the *learning coefficient*; it equals $d/2$ in Notebook 1's case and is smaller
otherwise.
"""
        + reading(
            "You're Measuring Model Complexity Wrong",
            "https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong",
            "Hoogland & van Wingerden (LessWrong, 2023)",
            "The learning coefficient as the exponent of low-loss volume, why the Hessian (and parameter count) get it wrong, and what it means for the Bayesian posterior; the same argument as §2.1 with pictures.",
            importance=5,
            length=2,
        )
        + r"""
**Route.**

- [2.1 Volume scaling](#slt2-s1): $\bar Z_n=\int e^{-nt}v(t)\,dt$; $V(t)\sim ct^\lambda(-\log t)^{m-1}$ gives $\bar F_n\approx\lambda\log n-(m-1)\log\log n$; toy losses; $\lambda\le d/2$.
- [2.2 The Mellin transform](#slt2-s2): $\zeta(z)=\int K^z\varphi$; its poles are the terms of the large-$n$ expansion; worked cases.
- [2.3 Several basins](#slt2-s3): the posterior picks the basin of least free energy, accuracy against volume, and switches basin at a critical $n^*$.

**Setup, repeated from Notebook 1.** Data $x_1,\dots,x_n$ i.i.d. from $q$; model
$p(x\mid w)$, $w\in\mathbb R^d$; prior $\varphi(w)$; $\ell(x,w)=-\log p(x\mid w)$;
$L_n$ training loss, $L$ population loss; $w_0$ a minimizer of $L$;
$K(w)=L(w)-L(w_0)\ge0$ and $K_n(w)=L_n(w)-L_n(w_0)$ the excess losses. The posterior
is $\pi_n\propto\varphi e^{-nL_n}$, the free energy is $F_n=-\log\int\varphi e^{-nL_n}$,
and $F_n=nL_n(w_0)+\bar F_n$ with $\bar F_n=-\log\bar Z_n$,
$\bar Z_n=\int\varphi e^{-nK_n}$.

**One simplification, made once.** This notebook works with the population
excess loss $K$ in place of the sample one $K_n$, i.e. with
$\bar Z_n=\int\varphi(w)e^{-nK(w)}dw$. The difference $n(K_n-K)$ is a sum of $n$
mean-zero terms, of size $\sqrt n\,\|w-w_0\|$ near $w_0$, and on the region
that matters it stays of order one (in Notebook 1's regular case it is the
$\tfrac n2g_n^\top H^{-1}g_n$ term, whose mean is $\tfrac12\mathrm{tr}(H^{-1}C)$).
Watanabe's theorem is that this remains true in general: the true $\bar F_n$ is
the one computed here plus a random $O(1)$. So the coefficients of $\log n$ and
$\log\log n$ can be read off from $K$, and the $O(1)$ cannot.
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

*Notation:* $K(w)\ge0$ the population excess loss, zero exactly on the set
$W_0=\{w:L(w)=L(w_0)\}$ of best parameters; $\varphi$ the prior;
$\bar Z_n=\int\varphi e^{-nK}$, $\bar F_n=-\log\bar Z_n$; $d$ the number of
parameters. New:

$$
V(t)=\int_{\{w:\,K(w)<t\}}\varphi(w)\,dw ,\qquad v(t)=V'(t).
$$

$V(t)$ is the prior volume of the *sublevel set*, the parameters whose excess
loss is below $t$; $v(t)$ is the prior volume per unit loss at loss $t$, and it
is the honest analogue of the density of states: $v(t)\,dt$ is the weight of
microstates with energy in $[t,t+dt]$. Slicing parameter space by the value of
$K$ turns the $d$-dimensional integral into a one-dimensional one,

$$
\bar Z_n=\int_0^\infty e^{-nt}\,v(t)\,dt=n\int_0^\infty e^{-nt}\,V(t)\,dt ,
$$

the Laplace transform of the density of states, exactly as $Z(\beta)=\int
e^{-\beta E}g(E)\,dE$ in statistical mechanics. Because $e^{-nt}$ is negligible
for $t\gg1/n$, only the behaviour of $V$ near $t=0$ matters when $n$ is large.
If

$$
V(t)=c\,t^{\lambda}(-\log t)^{m-1}\,(1+o(1))\quad\text{as }t\to0,
$$

then (Exercise 2.1.2)

$$
\bar F_n=\lambda\log n-(m-1)\log\log n-\log\big(c\,\Gamma(\lambda+1)\big)+o(1),
$$

where $\Gamma(\lambda+1)=\int_0^\infty e^{-s}s^\lambda ds$ is the Gamma function
(equal to $\lambda!$ for integer $\lambda$). The exponent $\lambda$ is called
the *learning coefficient* and $m$ the *multiplicity*. On the path through
these notebooks this is the whole definition: $\lambda$ is the exponent with
which low-loss volume vanishes, and it can be read off as
$\lambda=\lim_{t\to0}\log V(t)/\log t$. Watanabe proves that for analytic $K$
and $\varphi$ the expansion of $V$ always has this form, with $\lambda$ a
positive rational number and $m\le d$ a positive integer; §2.2 gives the tool
behind that theorem.

In the regular case $K\approx\tfrac12(w-w_0)^\top H(w-w_0)$ and the sublevel set
is an ellipsoid of volume $\propto t^{d/2}$, so $\lambda=d/2$ and $m=1$, and one
recovers Notebook 1's Laplace formula including its constant (Exercise 2.1.3).
Degenerate directions make the sublevel set fatter, $V(t)$ larger, and
$\lambda$ smaller: $2\lambda\le d$ always (Exercise 2.1.6), with equality
exactly when the Hessian is positive definite. One can read $2\lambda$ as the
effective number of parameters, but it is not an integer in general, and it is
not a count of anything: $K=w^4$ has $\lambda=\tfrac14$ with one parameter.

**Exercise 2.1 (Volume and free energy).** Take the prior uniform on $[-1,1]^d$ where a prior is needed.

1. Derive $\bar Z_n=\int_0^\infty e^{-nt}v(t)dt=n\int_0^\infty e^{-nt}V(t)dt$. <details><summary>Hint</summary>Write $e^{-nK(w)}=n\int_{K(w)}^\infty e^{-nt}dt$ and swap the order of integration; or insert $1=\int\delta(t-K(w))dt$.</details>
2. Show that $V(t)=ct^\lambda$ gives $\bar F_n=\lambda\log n-\log(c\Gamma(\lambda+1))+o(1)$, and that $V(t)=ct^\lambda(-\log t)^{m-1}$ gives the extra $-(m-1)\log\log n$. Say why only small $t$ matters even though the formula for $V$ is only assumed near $0$. <details><summary>Hint</summary>Substitute $t=s/n$ and use $\log n-\log s=\log n\,(1-\log s/\log n)$.</details>
3. Regular case: $K=\tfrac12(w-w_0)^\top H(w-w_0)$ with $H$ positive definite, and a smooth prior with $\varphi(w_0)>0$. Compute $V(t)$ to leading order (the volume of the $d$-dimensional unit ball is $\omega_d=\pi^{d/2}/\Gamma(d/2+1)$, and the ellipsoid $\{u:u^\top Au<1\}$ has volume $\omega_d(\det A)^{-1/2}$), read off $\lambda$ and $m$, and show that part 2 reproduces Notebook 1's $\bar F_n=\tfrac d2\log n+\tfrac12\log\det H-\log\varphi(w_0)-\tfrac d2\log2\pi+o(1)$.
4. Compute $V(t)$ and hence $\lambda$ and $m$ for: (a) $K=w^2$; (b) $K=w^4$; (c) $K=w^{2k}$; (d) $K=w^2+v^4$ (two parameters); (e) $K=(ab)^2$ (two parameters, zero on both axes); (f) the cubically reparametrized quadratic $K=\tfrac12(\mu^3-\mu_0^3)^2$, for $\mu_0=0$ and for $\mu_0\ne0$. What does (f) say about whether $\lambda$ is a property of the set of distributions $\{p_w\}$?
5. *(Sum rule.)* If $K(w,v)=K_1(w)+K_2(v)$ and $\varphi=\varphi_1(w)\varphi_2(v)$, show $\lambda=\lambda_1+\lambda_2$ and $m-1=(m_1-1)+(m_2-1)$. Check against 4(d).
6. *(Upper bound.)* Let $w_0$ be a minimizer of $K$ and $\Lambda$ an upper bound on the eigenvalues of $\nabla^2K$ on a ball around $w_0$. Show $K(w)\le\tfrac12\Lambda\|w-w_0\|^2$ there, deduce that the sublevel set contains a ball of radius $\sqrt{2t/\Lambda}$, and conclude $\lambda\le d/2$. <details><summary>Hint</summary>Taylor's theorem with the remainder in the form $K(w)=K(w_0)+\tfrac12(w-w_0)^\top\nabla^2K(\xi)(w-w_0)$ for some $\xi$ on the segment, and $u^\top Au\le\Lambda\|u\|^2$ when $\Lambda$ bounds the eigenvalues of the symmetric matrix $A$.</details>
"""
    ),
    answer(
        r"""
**1.** $\bar Z_n=\int\varphi(w)e^{-nK(w)}dw=\int\varphi(w)\,n\int_{K(w)}^\infty e^{-nt}dt\,dw=n\int_0^\infty e^{-nt}\Big(\int_{K(w)<t}\varphi(w)dw\Big)dt=n\int_0^\infty e^{-nt}V(t)dt$. Integrating by parts ($V(0)=0$, and $e^{-nt}V(t)\to0$ at $\infty$ since $V$ is bounded by the prior mass $1$) gives $\int_0^\infty e^{-nt}V'(t)dt=\int e^{-nt}v(t)dt$.

**2.** With $t=s/n$: $\bar Z_n=n\int_0^\infty e^{-nt}ct^\lambda dt=cn^{-\lambda}\int_0^\infty e^{-s}s^\lambda ds=c\,\Gamma(\lambda+1)\,n^{-\lambda}$, so $\bar F_n=\lambda\log n-\log(c\Gamma(\lambda+1))$. With the logarithm, $(-\log t)^{m-1}=(\log n-\log s)^{m-1}=(\log n)^{m-1}(1-\log s/\log n)^{m-1}$, and the integral against $e^{-s}s^\lambda$ of the bracket tends to $1$, so $\bar Z_n=c\Gamma(\lambda+1)n^{-\lambda}(\log n)^{m-1}(1+O(1/\log n))$, giving the extra $-(m-1)\log\log n$. (The correction is only $O(1/\log n)$: the $\log\log n$ term converges slowly, which matters in the code below.) Only small $t$ matters because the substitution shows that the integrand is concentrated at $s=O(1)$, i.e. $t=O(1/n)$: the region $t>\epsilon$ contributes at most $e^{-n\epsilon}$, exponentially small, whatever $V$ does there. (This is Watson's lemma: the large-$n$ expansion of a Laplace transform is determined by the small-$t$ expansion of the function.)

**3.** The sublevel set is the ellipsoid $\{u:u^\top Hu<2t\}=\{u:u^\top(H/2t)u<1\}$, of volume $\omega_d\det(H/2t)^{-1/2}=\omega_d(2t)^{d/2}(\det H)^{-1/2}$; multiplying by $\varphi(w_0)$ (the prior is approximately constant on a set of diameter $O(\sqrt t)$), $V(t)\approx\varphi(w_0)\omega_d(\det H)^{-1/2}2^{d/2}t^{d/2}$, so $\lambda=d/2$, $m=1$, $c=\varphi(w_0)2^{d/2}\pi^{d/2}(\det H)^{-1/2}/\Gamma(d/2+1)$. Then $c\Gamma(\lambda+1)=\varphi(w_0)(2\pi)^{d/2}(\det H)^{-1/2}$ and part 2 gives $\bar F_n=\tfrac d2\log n-\log\varphi(w_0)+\tfrac12\log\det H-\tfrac d2\log2\pi$, Notebook 1's formula. Volume scaling is Laplace's method with the Hessian replaced by what is actually there.

**4.** Prior density $2^{-d}$ on the box. (a) $\{w^2<t\}=(-\sqrt t,\sqrt t)$, $V=\sqrt t$ (after the factor $\tfrac12$): $\lambda=\tfrac12$, $m=1$. (b) $V=t^{1/4}$: $\lambda=\tfrac14$. (c) $V=t^{1/2k}$: $\lambda=1/2k$; the flatter the minimum, the smaller $\lambda$, although the Hessian is zero for every $k\ge2$ and cannot tell them apart. (d) $V(t)=\tfrac14\int_{|v|<t^{1/4}}2\sqrt{t-v^4}\,dv=\tfrac12t^{3/4}\int_{-1}^1\sqrt{1-u^4}\,du$ (substituting $v=t^{1/4}u$), so $\lambda=\tfrac34$, $m=1$. (e) $\{(ab)^2<t\}=\{|ab|<\sqrt t\}$. For $|a|<\sqrt t$ every $b\in[-1,1]$ qualifies; for $\sqrt t<|a|<1$ one needs $|b|<\sqrt t/|a|$. Area $=2\sqrt t\cdot2+2\int_{\sqrt t}^1\frac{2\sqrt t}{a}da=4\sqrt t+4\sqrt t\cdot\tfrac12(-\log t)=2\sqrt t(-\log t)+4\sqrt t$. Dividing by $4$: $V(t)=\tfrac12t^{1/2}(-\log t)\,(1+2/(-\log t))$, so $\lambda=\tfrac12$ and $m=2$: the zero set is two crossing lines, and where they cross the volume picks up a logarithm. (f) For $\mu_0=0$, $K=\mu^6/2$ and $V=\tfrac12\cdot2(2t)^{1/6}=(2t)^{1/6}$: $\lambda=\tfrac16$. For $\mu_0\ne0$ the minimum is at $\mu_0$ with $K''(\mu_0)=9\mu_0^4>0$, so $\lambda=\tfrac12$. The map $\theta=\mu^3$ is a bijection of $\mathbb R$, so both parametrizations describe exactly the same family of distributions, yet $\lambda$ differs: the learning coefficient is a property of the parametrization (and the prior's support), not of the set of distributions. That is not a defect: the prior lives on parameters, and it is the prior volume that Bayes integrates.

**5.** $\bar Z_n=\int\varphi_1e^{-nK_1}\int\varphi_2e^{-nK_2}=\bar Z_n^{(1)}\bar Z_n^{(2)}$, so $\bar F_n=\bar F_n^{(1)}+\bar F_n^{(2)}$ and the coefficients of $\log n$ and $\log\log n$ add: $\lambda=\lambda_1+\lambda_2$, $(m-1)=(m_1-1)+(m_2-1)$. For $w^2+v^4$: $\tfrac12+\tfrac14=\tfrac34$, as in 4(d).

**6.** Taylor's theorem with $\nabla K(w_0)=0$ and $K(w_0)=0$ gives $K(w)=\tfrac12(w-w_0)^\top\nabla^2K(\xi)(w-w_0)\le\tfrac12\Lambda\|w-w_0\|^2$. Hence $\|w-w_0\|<\sqrt{2t/\Lambda}$ implies $K(w)<t$: the ball of radius $r=\sqrt{2t/\Lambda}$ sits inside the sublevel set, and $V(t)\ge\varphi_{\min}\omega_dr^d=\varphi_{\min}\omega_d(2/\Lambda)^{d/2}t^{d/2}$ for small $t$ (with $\varphi_{\min}>0$ the prior's minimum on the ball). Then $\lambda=\lim\log V/\log t\le\lim\log(Ct^{d/2})/\log t=d/2$ (recall $\log t<0$, so a larger $V$ means a smaller ratio). Curvature can only make a basin narrower than a ball; the ball is the worst case, and it has $\lambda=d/2$.
"""
    ),
    markdown(
        r"""
**Code: $\bar F_n$ against $\log n$ for the toy losses.** The cell computes
$\bar Z_n$ by a Riemann sum on a grid for the losses of Exercise 2.1.4, with the
uniform prior on $[-1,1]^d$, and fits the slope of $\bar F_n$ against $\log n$
over a window of large $n$. For $(ab)^2$ the slope is pulled away from $\tfrac12$
by the $-\log\log n$ term, whose derivative $-1/(n\log n)$ decays slowly; fitting
the full form $\lambda\log n-(m-1)\log\log n+c$ with $m=2$ reduces it. In the two-basin figure further down, $n^*$ marks where the leading terms balance; the $O(1)$ constants $c_\alpha$ shift the actual crossing by a factor of order one.
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

*Question:* is there a way to get $\lambda$, $m$ and the corrections to the
free energy without computing $V(t)$ by hand?

*Notation:* $K$, $\varphi$, $V$, $v$, $\bar Z_n$ as in §2.1. New: for a complex
number $z$,

$$
\zeta(z)=\int K(w)^z\,\varphi(w)\,dw=\int_0^\infty t^z\,v(t)\,dt .
$$

The second form (same slicing as Exercise 2.1.1) says $\zeta$ is the *Mellin
transform* of the density of states: the Mellin transform of a function
$f$ on $(0,\infty)$ is $\int_0^\infty t^{z}f(t)\,dt$ (conventions differ by a
shift of $z$ by one), and it does for powers of $t$ what the Fourier transform
does for exponentials. The integral converges for $\mathrm{Re}\,z>0$ (the
integrand is bounded there, for a prior of bounded support and bounded $K$).
As $z$ moves left the integral blows up where $t^zv(t)$ stops being integrable
at $t=0$, and this is exactly where the small-$t$ behaviour of $v$ sits. The
computation that makes this precise is (Exercise 2.2.1)

$$
\int_0^1t^{z}\cdot t^{\lambda-1}(-\log t)^{m-1}\,dt=\frac{(m-1)!}{(z+\lambda)^m}:
$$

a term $ct^{\lambda-1}(-\log t)^{m-1}$ in $v(t)$ is the same thing as a pole of
$\zeta$ at $z=-\lambda$ of order $m$. Conversely, a Mellin transform can be
inverted, and each pole of $\zeta$ at $-\lambda_k$ of order $m_k$ gives back a
term $c_kt^{\lambda_k-1}(-\log t)^{m_k-1}$ in $v$ and so (§2.1) a term of order
$n^{-\lambda_k}(\log n)^{m_k-1}$ in $\bar Z_n$. In short:

$$
\text{poles of }\zeta\ \longleftrightarrow\ \text{small-}t\text{ terms of }v\ \longleftrightarrow\ \text{large-}n\text{ terms of }\bar Z_n .
$$

The pole closest to the origin (the largest, since all are negative) gives
$\lambda$, its order gives $m$, and the further poles give the corrections. What
Watanabe adds, and what we do not prove, is that for analytic $K$ and $\varphi$
the function $\zeta$ extends from $\mathrm{Re}\,z>0$ to the whole plane with
only poles, all at negative rational numbers; the proof changes variables
locally so that $K$ becomes a monomial $u_1^{2k_1}\cdots u_d^{2k_d}$, for which
$\zeta$ is a product of one-dimensional integrals.

**Why bother.** For a monomial, $\zeta$ factorizes and the poles are read off in
one line, where computing $V(t)$ needs the case analysis of Exercise 2.1.4(e).
The worked case is $K=(ab)^2$ with the uniform prior on $[-1,1]^2$:

$$
\zeta(z)=\frac14\int_{-1}^1\!\!\int_{-1}^1(ab)^{2z}\,da\,db=\frac14\Big(\frac2{2z+1}\Big)^2=\frac1{(2z+1)^2}=\frac14\cdot\frac1{(z+\tfrac12)^2},
$$

a double pole at $z=-\tfrac12$: $\lambda=\tfrac12$ and $m=2$, as found the hard
way in §2.1.

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

**3.** $V(t)=\tfrac12I\,t^{3/4}$ with $I=\int_{-1}^1\sqrt{1-u^4}du$, so $v(t)=\tfrac38I\,t^{-1/4}$ for small $t$, and $\int_0^1t^z\cdot t^{-1/4}dt=\frac1{z+3/4}$: first pole at $-\tfrac34$, $\lambda=\tfrac34$, $m=1$. (The exact $v$ has further terms at larger $t$ which give poles further left.)

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

*Notation:* regions $U_\alpha$ of parameter space, each containing one minimum
$w_\alpha$ of $L$ (a basin), with local free energies $F_n(U_\alpha)$ as in Notebook 1
Exercise 1.1; $L_\alpha=L(w_\alpha)$, $\lambda_\alpha$ and $m_\alpha$ the
learning coefficient and multiplicity of the basin, defined by the volume
scaling of $L-L_\alpha$ inside $U_\alpha$.

Applying §2.1 inside each basin,

$$
F_n(U_\alpha)\approx nL_\alpha+\lambda_\alpha\log n-(m_\alpha-1)\log\log n+c_\alpha ,
\qquad
F_n=-\log\sum_\alpha e^{-F_n(U_\alpha)}\approx\min_\alpha F_n(U_\alpha),
$$

(Notebook 1, Exercise 1.1.3; here with $L_n(w_\alpha)\approx L_\alpha$ by the law of
large numbers). The posterior mass of basin $\alpha$ is $e^{-F_n(U_\alpha)}/\sum_\beta e^{-F_n(U_\beta)}$,
so the posterior lives in the basin of least free energy, and the free energy
trades accuracy, $nL_\alpha$, against volume, $\lambda_\alpha\log n$. The trade
is not static. Take two basins, $U$ accurate but narrow and $V$ less accurate but
wide: $L_U<L_V$ and $\lambda_U>\lambda_V$. The log posterior odds are

$$
\log\frac{\pi_n(U)}{\pi_n(V)}=F_n(V)-F_n(U)\approx n\,\Delta L-\Delta\lambda\,\log n,
\qquad \Delta L=L_V-L_U>0,\ \ \Delta\lambda=\lambda_U-\lambda_V>0 .
$$

For small $n$ the logarithm wins and the posterior sits in the wide basin $V$;
for large $n$ the linear term wins and it moves to the accurate basin $U$. The
crossover is at the $n^*$ solving $n^*\Delta L=\Delta\lambda\log n^*$, i.e.

$$
\frac{n^*}{\log n^*}=\frac{\Delta\lambda}{\Delta L}.
$$

A small loss of accuracy against a large gain in volume keeps the posterior in
the simpler basin for a long time. This is the same mechanism as Notebook 1's BIC
comparison between two models, now happening between two regions of one model's
parameter space.

**The thermodynamic reading, in one paragraph.** Treating $n$ as continuous,
$\partial\bar F_n/\partial n=\mathbb E_{\pi_n}[K]$ (Exercise 2.3.3), the
posterior mean excess loss: the derivative of the free energy with respect to
the "inverse temperature" $n$ is the "internal energy", as in Notebook 1 Exercise
1.1.4. At $n^*$ the dominant basin changes, and $\partial\bar F_n/\partial n$
jumps from about $K_V$ to about $K_U$ over a window of $n$ around $n^*$ whose
width is set by how fast the odds change. A kink in the free energy with a
jump in its first derivative is, in the dictionary, a first-order phase
transition, and the jump in $\mathbb E[K]$ plays the part of a latent heat. The
analogy stops there. There is no thermodynamic limit that makes the kink sharp
(the window is of relative width $\sim1/\log n^*$), and the posterior has no
dynamics, so it has no metastable states: at $n>n^*$ it is in $U$ whatever it
was before. Metastability enters only for a procedure that moves locally, such
as a sampler or an optimizer started in $V$, which would stay there past $n^*$
until it crosses the barrier between the basins, for a time set by the barrier
and not by the free energies. Whether and how that describes training is not a
question these notebooks answer.

**Exercise 2.3 (Two basins).**

1. Derive the log-odds formula and $n^*/\log n^*=\Delta\lambda/\Delta L$ from Notebook 1 Exercise 1.1.2 and §2.1. Show that if instead $\Delta L$ and $\Delta\lambda$ have the same sign (one basin is both more accurate and wider), there is no crossover.
2. Suppose $F_n(U)\approx0.3n+20\log n$ and $F_n(V)\approx0.5n+2\log n$. Which basin is the accurate one, and which the wide one? Find $n^*$, and say what the posterior odds are at $n=100$ and at $n=2000$.
3. Treating $n$ as a continuous variable in $\bar Z_n=\int\varphi e^{-nK}$, show $\partial_n\bar F_n=\mathbb E_{\pi_n}[K]$ and $\partial_n^2\bar F_n=-\mathrm{Var}_{\pi_n}[K]$. Conclude that the posterior mean excess loss decreases with $n$, and say what it is in a single basin with learning coefficient $\lambda$ for large $n$.
4. *(Several basins, several scales.)* Three basins have $(L_\alpha,\lambda_\alpha)=(0,\,2),(0.01,\,1),(0.05,\,\tfrac14)$. Sketch which basin holds the posterior as $n$ runs from $10$ to $10^5$, and give the two crossover values.
"""
    ),
    answer(
        r"""
**1.** Notebook 1 Exercise 1.1.2: $\log\pi_n(U)/\pi_n(V)=F_n(V)-F_n(U)$. Insert the expansions: $F_n(V)-F_n(U)\approx n(L_V-L_U)+(\lambda_V-\lambda_U)\log n+O(\log\log n)$, which is $n\Delta L-\Delta\lambda\log n$ with the stated signs (dropping the slowly varying terms). For moderate $n$ the logarithmic term dominates and the expression is negative; for large $n$ the linear term wins and it is positive; it crosses zero where $n\Delta L=\Delta\lambda\log n$. The function $n/\log n$ is increasing for $n>e$, so the equation $n/\log n=\Delta\lambda/\Delta L$ has exactly one solution above $e$ when the right-hand side is positive. If $\Delta L$ and $\Delta\lambda$ have the same sign, the two terms have the same sign for every $n$: one basin is preferred throughout, and the odds only grow.

**2.** $U$ has the lower loss coefficient ($0.3<0.5$): it is the accurate basin, and it has the larger $\lambda$ ($20>2$): it is the narrow one. $\Delta L=0.2$, $\Delta\lambda=18$, so $n^*/\log n^*=90$, giving $n^*\approx570$ ($570/\log570=570/6.35\approx89.8$). At $n=100$: $F(V)-F(U)=0.2\cdot100-18\log100=20-82.9=-62.9$: the posterior is in $V$ by odds $e^{-63}$. At $n=2000$: $400-18\cdot7.6=263$: in $U$ by odds $e^{263}$. The crossover is sharp on the scale of the odds but it is not instantaneous in $n$: the log-odds pass through $[-5,5]$ over a window of width about $10/\Delta L=50$ samples around $n^*$.

**3.** $\partial_n\log\bar Z_n=\frac{\int(-K)\varphi e^{-nK}}{\int\varphi e^{-nK}}=-\mathbb E_{\pi_n}[K]$, so $\partial_n\bar F_n=\mathbb E_{\pi_n}[K]$; differentiating again as in Notebook 1 Exercise 1.1.4 gives $-\mathrm{Var}_{\pi_n}[K]\le0$. So $\mathbb E_{\pi_n}[K]$ is non-increasing in $n$: more data push the posterior to lower excess loss. In a single basin, $\bar F_n\approx\lambda\log n-(m-1)\log\log n+c$ gives $\mathbb E_{\pi_n}[K]\approx\lambda/n$ (the $\log\log$ term contributes $-(m-1)/(n\log n)$, smaller): the posterior sits at excess loss $\lambda/n$, the "equipartition" of this problem, with $\lambda$ in place of $d/2$. Notebook 3 uses this identity as the definition of an estimator.

**4.** Pairwise crossovers: between basins 2 and 3, $n/\log n=\tfrac34/0.04=18.75$, so $n\approx62$; between 1 and 2, $n/\log n=1/0.01=100$, so $n\approx650$; between 1 and 3 (not needed, but for consistency) $n/\log n=1.75/0.05=35$, $n\approx140$, which lies between the other two, so basin 2 really is visited. The posterior is in basin 3 (widest, worst) for $n\lesssim62$, in basin 2 for $62\lesssim n\lesssim650$, and in basin 1 (narrowest, exact) after that: a staircase of increasingly accurate, increasingly narrow basins, with the dwell time in each set by $\Delta\lambda/\Delta L$.
"""
    ),
    markdown(
        r"""
**Code: a posterior that changes basin.** The one-dimensional loss
$K(w)=(w+1)^2(w-1)^4+\delta\,\sigma(20w)$, with $\sigma$ the logistic function,
has a narrow regular basin at $w=-1$ (excess loss $0$, $\lambda=\tfrac12$) and a
wide degenerate basin at $w=+1$ (excess loss $\delta$, $\lambda=\tfrac14$). The
first cell computes the posterior mass of $w>0$ against $n$, with
$n^*/\log n^*=\tfrac14/\delta$ marked, together with $\bar F_n$ and its
derivative $\mathbb E_{\pi_n}[K]$. The second is Exercise 2.3.2 as a plot.
"""
    ),
    code(
        r"""
delta = 0.01
w = np.linspace(-2.5, 2.5, 200_001)
K = (w + 1) ** 2 * (w - 1) ** 4 + delta / (1 + np.exp(-20 * w))
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
