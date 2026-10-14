#!/usr/bin/env python3
"""Build SLT2, "Free energy and basin volume" (Singular learning theory day, NB2).

One linear path, three `##` sections (anchors slt2-1 ... slt2-3), about 90 minutes:
1 volume scaling (Zbar_n as a Laplace transform of V; lambda as the volume exponent; lambda <= d/2);
2 the zeta function (Mellin transform; poles <-> terms; (ab)^2 worked; w^2, w^4, Gaussian prior,
  the cubic reparametrisation); 3 several basins (F ~ min_alpha, the critical n*, one paragraph on
  metastability). Exercises are bold-labelled paragraphs with anchors, not headings.

Sources: the Iliad B.3 worksheet (Ogden, Farrugia-Roberts, Furman), Ex 3.4-3.7 and 4.1-4.2,
adapted. Formula checks: checks/slt2_volume_checks.py. Review log: ../reviews/slt/slt2_volume.md.
"""
from nbkit_slt import answer, code, code_answer, markdown, reading, references, book, run  # noqa: F401

# ---------------------------------------------------------------- readings
KEYSTONE = reading(
    "You're measuring model complexity wrong",
    "https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong",
    "Jesse Hoogland and Stan van Wingerden (LessWrong, 2023)",
    r"The volume picture of $\lambda$ that this notebook makes quantitative. Read the section "
    r"*From Basins to Model Complexity*: why basin volume and not Hessian flatness, "
    r"$V(\epsilon)\sim c\,\epsilon^{\lambda}$, two losses with the same (zero) Hessian and different "
    r"exponents ($w_1^4+w_2^4$ and $w_1^2w_2^4$), and a two-basin cartoon of §3. Exercise 2.4 shows a "
    r"change of coordinates (smooth, but with Jacobian vanishing at a point) that does change $\lambda$. "
    r"The estimator half of the post is NB3's subject.",
    importance=5, length=2)
WATANABE22 = reading(
    "Recent advances in algebraic geometry and Bayesian statistics",
    "https://arxiv.org/abs/2211.10049",
    "Sumio Watanabe (survey, 2022)",
    r"The source's own short statement of this notebook's results. Read §4.1 for $\zeta$ as the Mellin "
    r"transform of the state density $v(t)$ and the correspondence "
    r"$(z+\lambda)^{-m}\leftrightarrow t^{\lambda-1}(-\log t)^{m-1}$, and §5.2 for the expansion of "
    r"$F_n$ and of $\mathbb E[F_n]$ (the theorem quoted before the §2 code). Skip the resolution-map details; nothing here needs them.",
    importance=3, length=2)
FURMAN = reading(
    "Singular learning theory: exercises",
    "https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/",
    "Zach Furman (LessWrong, 2024)",
    r"Problem sets with more toy models. Problems 2 and 5 (the cubic parametrisation, and "
    r"$a^{k_1}b^{k_2}$ with its multiplicity) and 17 (invariance under smooth invertible changes of "
    r"variables) go with §2; problems 12–13 are the originals of the §2 code and Exercise 3.1 (also in the Iliad B.3 worksheet).",
    importance=2, length=2)
FLAJOLET = reading(
    "Mellin transforms and asymptotics: harmonic sums",
    "https://algo.inria.fr/flajolet/Publications/FlGoDu95.pdf",
    "Philippe Flajolet, Xavier Gourdon and Philippe Dumas (Theoretical Computer Science, 1995)",
    r"The general version of the §2 dictionary, with conditions: Theorems 3 and 4 of §2 say exactly when "
    r"poles of a Mellin transform give the terms of an asymptotic expansion. Read §§1–2; the rest is "
    r"about harmonic sums, which we don't need.",
    importance=2, length=2)

EXTRAS = "\n\n**Claude-suggested extras.**\n\n"

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# SLT2 — Free energy and basin volume

**Singular learning theory day, Iliad Intensive 2026**

In NB1 the free energy came out as $F_n\approx nL_n(\hat w)+\tfrac d2\log n+O(1)$, from
a Gaussian integral around a minimum whose Hessian is invertible. This notebook asks
what replaces $\tfrac d2\log n$ when the Hessian has zero eigenvalues. The answer is
$\lambda\log n-(m-1)\log\log n$, with $\lambda$ the exponent at which the prior volume of
near-optimal parameters shrinks (the *learning coefficient*; that is the only definition
used here). You will compute $\lambda$ for toy losses, see why a slope fitted at finite
$n$ misleads, and watch the posterior switch between basins.

**Keystone reading** (an alternative way into §1, or read it afterwards).
"""
        + KEYSTONE
        + r"""
**The path.** Sections 1–3, about 90 minutes. On the path, do Exercises 1.1, 1.3, 2.2,
2.4 and 3.1 (parts 1–3), and read the code output. Exercises 1.2, 2.1, 2.3 and 3.1
part 4 are optional.

- [1. Volume scaling](#slt2-1) (30 min): $\bar Z_n$ is a Laplace transform of the volume of near-optimal parameters, so $\bar F_n\approx\lambda\log n-(m-1)\log\log n$; and $\lambda\le d/2$.
- [2. The zeta function](#slt2-2) (35 min): $\lambda$ and $m$ from the poles of a zeta function; why finite-$n$ slope fits mislead; $\lambda$ depends on the parametrisation.
- [3. Several basins](#slt2-3) (25 min): the posterior trades accuracy against volume and switches basin at a critical $n^*$.

**Notation used throughout** (restated in each section). Parameter $w\in W\subseteq\mathbb R^d$,
prior density $\varphi(w)$, population loss $L(w)$ with minimiser $w_0$, and
$K(w)=L(w)-L(w_0)\ge0$. The averaged partition function and free energy are
$\bar Z_n=\int\varphi(w)\,e^{-nK(w)}\,dw$ and $\bar F_n=-\log\bar Z_n$. You need numpy,
scipy and matplotlib.
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, special

np.set_printoptions(precision=4, suppress=True)


def Z_pow(p, n):
    # int_0^1 exp(-n w^p) dw, exactly (incomplete gamma function). With a uniform prior on [-1, 1]
    # this is Zbar_n for K = |w|^p; products of these give separable losses in more dimensions.
    return n ** (-1 / p) * special.gamma(1 + 1 / p) * special.gammainc(1 / p, n)


def Z_ab(n):
    # Zbar_n for K = (ab)^2 with a uniform prior on [-1, 1]^2: the b-integral is an erf,
    # the a-integral is done by quadrature.
    f = lambda a: np.sqrt(np.pi) * special.erf(a * np.sqrt(n)) / (2 * a * np.sqrt(n))
    return integrate.quad(f, 0, 1, points=[1 / np.sqrt(n)] if n > 1 else None, limit=200)[0]
"""
    ),
    # ================================================================ 1
    markdown(
        r"""
<a id="slt2-1"></a>

## 1. Volume scaling

How does the free energy grow with $n$ when the Hessian at the minimum is not
invertible? This section shows that the answer depends on one function: the prior volume
of the parameters whose loss is within $t$ of optimal, as $t\to0$.

*Notation.* Prior $\varphi$ on $W\subseteq\mathbb R^d$; $K(w)=L(w)-L(w_0)\ge0$;
$\bar Z_n=\int\varphi\,e^{-nK}$, $\bar F_n=-\log\bar Z_n$. New here: the volume function
and its density,

$$
V(t)=\int_{K(w)<t}\varphi(w)\,dw,\qquad v(t)=V'(t).
$$

$V(t)$ is the prior probability that a parameter is within $t$ of optimal. It increases
from $V(0)=0$ to the total prior mass $1$.

**Where Laplace's method stops.** NB1 expanded $K(w)\approx\tfrac12(w-w_0)^\top H(w-w_0)$
and did the Gaussian integral, getting
$\bar Z_n\approx\varphi(w_0)(2\pi/n)^{d/2}(\det H)^{-1/2}$. If $H$ has a zero eigenvalue
this is infinite, while $\bar Z_n\le1$. The quadratic term simply does not control the
integral in that direction. For example, take $K=w^4$ on $[-1,1]$ with the uniform prior:

$$
\bar Z_n=\int_0^1e^{-nw^4}\,dw\approx\Gamma(\tfrac54)\,n^{-1/4},\qquad
\bar F_n\approx\tfrac14\log n .
$$

The coefficient of $\log n$ is $\tfrac14$, not $\tfrac d2=\tfrac12$. It is set by how
fast the set of good parameters shrinks, which is what $V(t)$ records.

**Sort parameters by their loss.** The integrand of $\bar Z_n$ depends on $w$ only
through $K(w)$. Group the parameters by their loss value: $v(t)\,dt$ is the prior mass
with $K\in[t,t+dt]$. Then

$$
\bar Z_n=\int_0^\infty e^{-nt}\,v(t)\,dt=n\int_0^\infty e^{-nt}\,V(t)\,dt. \tag{1}
$$

So $\bar Z_n$ is the Laplace transform of $v$. The factor $e^{-nt}$ discards everything
except $t\lesssim1/n$, so the large-$n$ behaviour of $\bar Z_n$ is the small-$t$ behaviour
of $V$.

*Dictionary.* In statistical mechanics $Z(\beta)=\int e^{-\beta E}g(E)\,dE$ with $g$ the
density of states. Here $n$ plays $\beta$, $t$ plays the energy, and $v$ is literally a
density of states; equation (1) is exact. One difference: a physical $g(E)$ grows like
$e^{Ns(E)}$ with a system size $N$, which is what makes thermodynamic limits sharp. Here
$v$ is a fixed function and there is no $N$.

**The volume exponent.** Suppose that, as $t\to0$,

$$
V(t)\simeq c\,t^{\lambda}(-\log t)^{m-1}
$$

for constants $c>0$, $\lambda>0$ and an integer $m\ge1$. $\lambda$ is the **learning
coefficient** and $m$ its **multiplicity**. Put $t=s/n$ in (1):
$\bar Z_n=\int_0^\infty e^{-s}\,V(s/n)\,ds$. For large $n$ only $s=O(1)$ matters, where
$V(s/n)\approx c\,s^\lambda n^{-\lambda}(\log n)^{m-1}$, so

$$
\bar Z_n\simeq c\,\Gamma(\lambda+1)\,n^{-\lambda}(\log n)^{m-1},\qquad
\bar F_n=\lambda\log n-(m-1)\log\log n-\log\big(c\,\Gamma(\lambda+1)\big)+o(1). \tag{2}
$$

This is what "λ is a basin-volume exponent" means. Shrinking the tolerance $t$ by a
factor 2 shrinks the near-optimal volume by $2^{\lambda}$ (up to the log); doubling $n$
shrinks the evidence by the same factor. When $m>1$, the $o(1)$ in (2) is of order
$1/\log n$, which decays very slowly; §2 makes this concrete. Everything here uses the
population loss $K$; the cubic-model code in §2 shows what the data add.

In the regular case, $V(t)$ is the prior mass of an ellipsoid and $\lambda=d/2$, $m=1$.
Exercise 1.1 checks that (2) then reproduces NB1's Laplace formula, constant included.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex1-1"></a>

**Exercise 1.1 — From volume to free energy.** Notation as above: $V(t)$ is the prior
mass of $\{K<t\}$ and $v=V'$.

1. Derive (1) by integrating by parts. Which property of $V$ kills the boundary terms?
2. Assume $V(t)=c\,t^\lambda(-\log t)^{m-1}(1+o(1))$ as $t\to0$. Fill in the step from $\bar Z_n=\int_0^\infty e^{-s}V(s/n)\,ds$ to (2), and say why the correction is relative order $1/\log n$ when $m>1$.
3. *Regular case.* Let $K(w)\approx\tfrac12(w-w_0)^\top H(w-w_0)$ with $H$ positive definite and $\varphi$ continuous at $w_0$. Show $V(t)\simeq\varphi(w_0)\,\omega_d\,(2t)^{d/2}(\det H)^{-1/2}$, where $\omega_d=\pi^{d/2}/\Gamma(\tfrac d2+1)$ is the volume of the unit ball. Read off $\lambda$ and $m$, and check that $c\,\Gamma(\lambda+1)$ is NB1's Laplace constant $\varphi(w_0)(2\pi)^{d/2}(\det H)^{-1/2}$.

<details><summary>Hint</summary>

For part 3, the ellipsoid $\{u:u^\top Au<1\}$ has volume $\omega_d(\det A)^{-1/2}$. Write the sublevel set as $u^\top\big(H/2t\big)u<1$.
</details>
"""
    ),
    answer(
        r"""
1. $v\,dt=dV$, so $\int_0^\infty e^{-nt}\,dV(t)=\big[e^{-nt}V(t)\big]_0^\infty+n\int_0^\infty e^{-nt}V(t)\,dt$. The boundary terms vanish because $V(0)=0$ (no prior mass has $K<0$) and $V\le1$, so $e^{-nt}V(t)\to0$ as $t\to\infty$. The first equality in (1) is the change of variables $t=K(w)$: $\int\varphi(w)g(K(w))\,dw=\int g(t)\,v(t)\,dt$ for any $g$, by the definition of $v$ as the density of the push-forward of $\varphi$ under $K$.
2. Substitute $t=s/n$, $dt=ds/n$: $n\int e^{-nt}V(t)\,dt=\int_0^\infty e^{-s}V(s/n)\,ds$. For fixed $s$ and $n\to\infty$, $s/n\to0$, so
   $$V(s/n)=c\,(s/n)^\lambda\big(\log n-\log s\big)^{m-1}(1+o(1))=c\,s^\lambda n^{-\lambda}(\log n)^{m-1}\Big(1-\frac{\log s}{\log n}\Big)^{m-1}(1+o(1)).$$
   The weight $e^{-s}$ makes large $s$ irrelevant (and $V\le1$ bounds the tail), so dominated convergence gives $\bar Z_n\simeq c\,n^{-\lambda}(\log n)^{m-1}\int_0^\infty e^{-s}s^\lambda\,ds=c\,\Gamma(\lambda+1)\,n^{-\lambda}(\log n)^{m-1}$. Taking $-\log$ gives (2). The factor $(1-\log s/\log n)^{m-1}=1-(m-1)\log s/\log n+\dots$ is what is dropped: its average against $e^{-s}s^\lambda$ is $1+O(1/\log n)$, a relative correction of order $1/\log n$ to $\bar Z_n$ and an additive $O(1/\log n)$ to $\bar F_n$. For $m=1$ it is absent and the corrections come from the next term of $V$ instead.
3. Put $u=w-w_0$. For small $t$ the set $\{K<t\}$ is, to leading order, $\{\tfrac12u^\top Hu<t\}=\{u^\top(H/2t)u<1\}$, an ellipsoid of volume $\omega_d\det(H/2t)^{-1/2}=\omega_d(2t)^{d/2}(\det H)^{-1/2}$. It shrinks to $w_0$, where $\varphi\approx\varphi(w_0)$, so
   $$V(t)\simeq\varphi(w_0)\,\omega_d\,2^{d/2}(\det H)^{-1/2}\;t^{d/2}:\qquad\lambda=\tfrac d2,\quad m=1,\quad c=\varphi(w_0)\,\omega_d\,2^{d/2}(\det H)^{-1/2}.$$
   Then $c\,\Gamma(\tfrac d2+1)=\varphi(w_0)\,\pi^{d/2}\,2^{d/2}(\det H)^{-1/2}=\varphi(w_0)(2\pi)^{d/2}(\det H)^{-1/2}$, and (2) gives $\bar F_n=\tfrac d2\log n+\tfrac12\log\det H-\tfrac d2\log2\pi-\log\varphi(w_0)+o(1)$: NB1's Laplace formula, constant included.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex1-2"></a>

**Exercise 1.2 — Volumes by hand** (optional; adapted from the Iliad B.3 worksheet, Ex
3.4 and 3.6). Take the uniform prior on $[-1,1]^d$. For small $t$, compute the prior
mass $V(t)$ of $\{K<t\}$, and read off $\lambda$ and $m$ from $V(t)\simeq c\,t^\lambda(-\log t)^{m-1}$.

1. $d=1$, $K=w^{2k}$ for $k=1,2,3,\dots$
2. $d=2$, $K=x^2$ (the loss ignores $y$).
3. $d=2$, $K=x^2+y^4$.
4. Compare the Hessians at the origin in parts 2 and 3. What does the Hessian miss?
"""
    ),
    answer(
        r"""
The prior density is $2^{-d}$, so $V(t)$ is $2^{-d}$ times the area of $\{K<t\}\cap[-1,1]^d$.

1. $\{w^{2k}<t\}=\{|w|<t^{1/2k}\}$ has length $2t^{1/2k}$, so $V(t)=t^{1/2k}$: $\lambda=\frac1{2k}$, $m=1$. For $k=2$ this is the $\tfrac14$ of the $w^4$ example above.
2. $\{x^2<t\}$ is the strip $|x|<\sqrt t$, $|y|\le1$, of area $4\sqrt t$. So $V(t)=\sqrt t$: $\lambda=\tfrac12$, $m=1$. The free direction $y$ contributes a constant factor, not a power of $t$.
3. For $|y|<t^{1/4}$, $x$ ranges over $|x|<\sqrt{t-y^4}$. With $y=t^{1/4}s$,
   $$V(t)=\tfrac14\int_{-t^{1/4}}^{t^{1/4}}2\sqrt{t-y^4}\,dy=\tfrac12\,t^{3/4}\int_{-1}^1\sqrt{1-s^4}\,ds\approx0.874\,t^{3/4}.$$
   So $\lambda=\tfrac34=\tfrac12+\tfrac14$, $m=1$: each coordinate contributes its own exponent.
4. Both Hessians are $\mathrm{diag}(2,0)$, of rank 1. Yet $\lambda=\tfrac12$ in part 2 and $\tfrac34$ in part 3. In part 2 the zero eigenvalue is a direction that is exactly flat; in part 3 the loss still grows along it, quartically. The Hessian sees only the second-order terms, and the volume depends on the first non-vanishing term in each direction.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex1-3"></a>

**Exercise 1.3 — $\lambda\le d/2$** (adapted from the Iliad B.3 worksheet, Ex 3.5). Let
$w_0$ be a minimiser in the interior of $W$, with $K(w_0)=0$; $V(t)$ is the prior mass of
$\{K<t\}$. Assume $K$ is twice continuously differentiable
near $w_0$, and $\varphi\ge\varphi_{\min}>0$ on a ball $B_\rho(w_0)$. Let $\Lambda>0$ bound
the eigenvalues of $\nabla^2K$ on that ball.

1. Show $K(w)\le\tfrac12\Lambda\,\lVert w-w_0\rVert^2$ on $B_\rho(w_0)$.
2. Deduce $V(t)\ge C\,t^{d/2}$ for small $t$, with $C>0$.
3. Conclude $\lambda\le d/2$. With Exercise 1.1, equality holds in the regular case.

<details><summary>Hint</summary>

Part 1: Taylor's theorem with the Lagrange remainder, using $\nabla K(w_0)=0$ (an interior minimum). Part 2: a small Euclidean ball around $w_0$ sits inside $\{K<t\}$.
</details>
"""
    ),
    answer(
        r"""
1. $w_0$ is an interior minimum, so $\nabla K(w_0)=0$. Taylor with the Lagrange remainder: for $w\in B_\rho(w_0)$ there is $\tilde w$ on the segment $[w_0,w]$ with $K(w)=\tfrac12(w-w_0)^\top\nabla^2K(\tilde w)(w-w_0)\le\tfrac12\Lambda\lVert w-w_0\rVert^2$.
2. Let $r=\sqrt{2t/\Lambda}$, and take $t$ small enough that $r\le\rho$. If $\lVert w-w_0\rVert<r$ then $K(w)<\tfrac12\Lambda r^2=t$. So the Euclidean ball $B_r(w_0)$ lies inside $\{K<t\}$, and
   $$V(t)\ge\varphi_{\min}\,\omega_d\,r^d=\varphi_{\min}\,\omega_d\,(2/\Lambda)^{d/2}\,t^{d/2}=C\,t^{d/2}.$$
3. If $V(t)\simeq c\,t^\lambda(-\log t)^{m-1}$ with $\lambda>d/2$, then $V(t)/t^{d/2}\to0$, contradicting part 2. So $\lambda\le d/2$. In the regular case Exercise 1.1 gave $\lambda=d/2$. Degenerate directions can only add volume near $w_0$, never remove it.
"""
    ),
    markdown(
        r"""
### Code: reading $\lambda$ off the slope

Four losses with the uniform prior on $[-1,1]^d$, with $\bar Z_n$ computed exactly by
quadrature: $w^2$ ($\lambda=\tfrac12$), $w^4$ ($\tfrac14$), $x^2+y^4$ ($\tfrac34$), and
$(ab)^2$ ($\lambda=\tfrac12$, $m=2$; §2 derives this). Left: $\bar F_n$ against $\log n$,
with dotted lines of slope $\lambda$. Right: the local slope $d\bar F_n/d\log n$, with
$\lambda$ dotted.
"""
    ),
    code(
        r"""
ns = np.logspace(1, 7, 49)
logn = np.log(ns)
cases = [  # (name, Zbar_n as a function of n, lambda, m)
    ("w^2", lambda n: Z_pow(2, n), 1 / 2, 1),
    ("w^4", lambda n: Z_pow(4, n), 1 / 4, 1),
    ("x^2+y^4", lambda n: Z_pow(2, n) * Z_pow(4, n), 3 / 4, 1),
    ("(ab)^2", Z_ab, 1 / 2, 2),
]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
print(f"{'K':>8}  lambda  m   slope of Fbar vs log n, fitted on 1e5 <= n <= 1e7")
for (name, Z, lam, m), col in zip(cases, ["C0", "C1", "C2", "C3"]):
    F = np.array([-np.log(Z(n)) for n in ns])
    ax[0].plot(logn, F, color=col, label=f"${name}$")
    ax[0].plot(logn, F[-1] + lam * (logn - logn[-1]), ":", color=col)
    ax[1].plot(logn, np.gradient(F, logn), color=col, label=f"${name}$")
    ax[1].axhline(lam, color=col, ls=":", lw=0.8)
    sel = ns >= 1e5
    print(f"{name:>8}  {lam:6.2f}  {m}   {np.polyfit(logn[sel], F[sel], 1)[0]:.3f}")
ax[0].set_xlabel(r"$\log n$"); ax[0].set_ylabel(r"$\bar F_n$"); ax[0].legend()
ax[0].set_title(r"$\bar F_n$ (dotted: slope $\lambda$)")
ax[1].set_xlabel(r"$\log n$"); ax[1].set_ylabel(r"$d\bar F_n/d\log n$"); ax[1].set_ylim(0, 0.85)
ax[1].set_title(r"local slope (dotted: $\lambda$)")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
For the three losses with $m=1$ the slope settles on $\lambda$ by $n\approx10^3$. For
$(ab)^2$ it is still near $0.44$ at $n=10^7$, well short of $\tfrac12$. The
$-(m-1)\log\log n$ term is responsible, and §2 shows exactly how.
"""
    ),
    # ================================================================ 2
    markdown(
        r"""
<a id="slt2-2"></a>

## 2. The zeta function

How do we get $\lambda$, $m$ and the corrections to (2) without working out the geometry
of the sets $\{K<t\}$? This section introduces a function that is often easy to compute
from $K$ and $\varphi$ directly, and whose poles are the terms of the large-$n$
expansion of $\bar Z_n$.

*Notation.* As in §1: prior $\varphi$, $K\ge0$, $V(t)$ the prior mass of $\{K<t\}$ and
$v=V'$, $\bar Z_n=\int\varphi e^{-nK}$, and $V(t)\simeq c\,t^\lambda(-\log t)^{m-1}$. New
here: for complex $z$,

$$
\zeta(z)=\int K(w)^z\,\varphi(w)\,dw=\int_0^\infty t^z\,v(t)\,dt .
$$

The second form (sort by loss, as in §1) says $\zeta$ is the Mellin transform of the
density of states $v$. The integral converges for $\mathrm{Re}\,z>-\lambda$, since
$v(t)\approx c\lambda\,t^{\lambda-1}(-\log t)^{m-1}$ near $0$. It then continues to a
function of $z$ on the whole plane, with poles. (For analytic $K$ this continuation
always exists, with poles at negative rational numbers. Below, every $\zeta$ is explicit.)

**The dictionary: a term in $v$, a pole of $\zeta$, a term in $\bar Z_n$.** Everything
follows from two integrals:

$$
\int_0^1t^{z+\lambda-1}\frac{(-\log t)^{j-1}}{(j-1)!}\,dt=\frac1{(z+\lambda)^{j}},\qquad
\int_0^\infty e^{-nt}\,t^{\lambda-1}\,dt=\Gamma(\lambda)\,n^{-\lambda}.
$$

So, near $t=0$:

- a term $a\,t^{\lambda-1}$ in $v$ gives a simple pole $a/(z+\lambda)$ of $\zeta$, and a term $a\,\Gamma(\lambda)\,n^{-\lambda}$ in $\bar Z_n$;
- a term $a\,t^{\lambda-1}(-\log t)^{m-1}/(m-1)!$ in $v$ gives a pole $a/(z+\lambda)^m$ of order $m$, and a leading term $a\,\Gamma(\lambda)\,n^{-\lambda}(\log n)^{m-1}/(m-1)!$ in $\bar Z_n$;
- the part of $v$ away from $t=0$ gives no poles (on a compact parameter set $v$ vanishes for large $t$), and only exponentially small terms in $\bar Z_n$.

Read the poles of $\zeta$ from right to left and you read the terms of $\bar Z_n$ from
largest to smallest. The pole nearest the origin sits at $z=-\lambda$, and its order is
$m$.

**The exact statement.** Multiply $\bar Z_n=\int_0^\infty e^{-nt}v(t)\,dt$ by $n^{-z-1}$
and integrate over $n$:

$$
\int_0^\infty n^{-z-1}\,\bar Z_n\,dn=\Gamma(-z)\,\zeta(z),\qquad-\lambda<\mathrm{Re}\,z<0 .
$$

So $\Gamma(-z)\zeta(z)$ is the Mellin transform of $\bar Z_n$ as a function of $n$, and
inverting it turns each pole of $\zeta$ into one term of the expansion (Exercise 2.3).

### Worked example: $K=(ab)^2$

Take $d=2$, $K(a,b)=a^2b^2$, and the uniform prior on $[-1,1]^2$, of density $\tfrac14$.
The zero set is the two axes, crossing at the origin.

*The zeta function.* The integrand factorises:

$$
\zeta(z)=\tfrac14\int_{-1}^1\!\int_{-1}^1|a|^{2z}|b|^{2z}\,da\,db=\Big(\int_0^1a^{2z}\,da\Big)^2=\frac1{(2z+1)^2}.
$$

A double pole at $z=-\tfrac12$ and nothing else: $\lambda=\tfrac12$, $m=2$.

*The density of states.* By the dictionary, a double pole $\tfrac14(z+\tfrac12)^{-2}$ comes
from $v(t)=\tfrac14\,t^{-1/2}(-\log t)$. Here this holds exactly for $0<t<1$ (and $v=0$
for $t>1$, since $K\le1$). Directly: $V(t)=P(|ab|<\sqrt t)=\sqrt t\,(1-\tfrac12\log t)$
for $a,b$ uniform, and differentiating gives the same $v$. The logarithm comes from the
hyperbolic arms $|ab|<\sqrt t$ running out along both axes.

*The free energy.* With $\int_0^\infty e^{-nt}t^{-1/2}(-\log t)\,dt=\Gamma(\tfrac12)\,n^{-1/2}(\log n-\psi(\tfrac12))$,
where $\psi=\Gamma'/\Gamma$ and $-\psi(\tfrac12)=\gamma+2\log2\approx1.96$ ($\gamma$ is
Euler's constant),

$$
\bar Z_n=\frac{\sqrt\pi}{4}\,n^{-1/2}\big(\log n+1.96\big)+O(e^{-n}),\qquad
\bar F_n=\tfrac12\log n-\log\big(\log n+1.96\big)+\log\frac{4}{\sqrt\pi}+O(e^{-n}).
$$

This is exact up to exponentially small terms, because $\zeta$ has a single pole. The
local slope is

$$
\frac{d\bar F_n}{d\log n}=\frac12-\frac1{\log n+1.96},
$$

which is $0.35$ at $n=10^2$, $0.41$ at $10^4$ and $0.44$ at $10^6$: the curve in the §1
plot. A straight line fitted to $\bar F_n$ against $\log n$ underestimates $\lambda$ at
every $n$, by about $1/\log n$.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex2-1"></a>

**Exercise 2.1 — Fitting $\lambda$ at finite $n$ (code; optional).** Below, $\bar F_n$ for
$K=(ab)^2$ is computed by quadrature for $10\le n\le10^{12}$. On each window of $n$, fit
(i) a straight line $\bar F_n\approx\hat\lambda\log n+C$; (ii) the two-term form
$\bar F_n\approx\hat\lambda\log n-\log\log n+C$, with $m=2$ assumed known; (iii) the same
form with $m$ free as well. Compare with $\lambda=\tfrac12$, $m=2$. What would you
conclude about $\lambda$ from data that stop at $n=10^5$?
"""
    ),
    code_answer(
        r"""
ns_fit = np.logspace(1, 12, 111)
F_ab = np.array([-np.log(Z_ab(n)) for n in ns_fit])
windows = [(1e1, 1e3), (1e3, 1e5), (1e5, 1e7), (1e7, 1e9), (1e10, 1e12)]
for lo, hi in windows:
    sel = (ns_fit >= 0.999 * lo) & (ns_fit <= 1.001 * hi)
    x, y = np.log(ns_fit[sel]), F_ab[sel]
    # (i) slope of a straight-line fit of y against x
    # (ii) lambda from y + log(x) = lambda * x + C   (the m = 2 form)
    # (iii) lambda and m from y = lambda * x - (m - 1) * log(x) + C
    ...
""",
        r"""
ns_fit = np.logspace(1, 12, 111)
F_ab = np.array([-np.log(Z_ab(n)) for n in ns_fit])
windows = [(1e1, 1e3), (1e3, 1e5), (1e5, 1e7), (1e7, 1e9), (1e10, 1e12)]
print("   window of n      (i) line   (ii) m=2 form   (iii) lambda, m free")
for lo, hi in windows:
    sel = (ns_fit >= 0.999 * lo) & (ns_fit <= 1.001 * hi)
    x, y = np.log(ns_fit[sel]), F_ab[sel]
    lam_line = np.polyfit(x, y, 1)[0]
    lam_m2 = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y + np.log(x), rcond=None)[0][0]
    lam_free, m_minus_1, _ = np.linalg.lstsq(np.c_[x, -np.log(x), np.ones_like(x)], y, rcond=None)[0]
    print(f"  {lo:7.0e}-{hi:7.0e}     {lam_line:.3f}        {lam_m2:.3f}          {lam_free:.3f}, {m_minus_1 + 1:.2f}")
"""
    ),
    answer(
        r"""
The straight line gives $0.34$–$0.46$ across eleven decades of $n$: from data ending
at $n=10^5$ you would report $\lambda\approx0.4$. Knowing $m=2$ fixes most of it, with a
small overshoot from the $O(1/\log n)$ remainder. Fitting $m$ as well recovers $\lambda$
to within a few percent but $m$ only roughly, because over any feasible window $\log n$
and $\log\log n$ are nearly collinear. The asymptotic value is a statement about
$n\to\infty$; a finite-$n$ fit returns an effective exponent, and when $m>1$ the two
differ for every practical $n$.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex2-2"></a>

**Exercise 2.2 — Zeta functions of toy losses.** Take the uniform prior on $[-1,1]^d$ and
$\zeta(z)=\int K^z\varphi$.

1. $d=1$: compute $\zeta$ for $K=w^2$ and $K=w^4$, and read off $\lambda$ and $m$. Compare with §1.
2. $d=2$, $K=a^{2k_1}b^{2k_2}$ with positive integers $k_1,k_2$: show $\zeta(z)=\big((2k_1z+1)(2k_2z+1)\big)^{-1}$. What are $\lambda$ and $m$? When is $m=2$?
3. The keystone's $K=w_1^2w_2^4$: what are $\lambda$, $m$, and the first correction to $\bar Z_n$?
"""
    ),
    answer(
        r"""
1. With density $\tfrac12$: $\zeta(z)=\int_0^1w^{2z}\,dw=\frac1{2z+1}$ for $K=w^2$, a simple pole at $-\tfrac12$: $\lambda=\tfrac12$, $m=1$. For $K=w^4$, $\zeta=\int_0^1w^{4z}dw=\frac1{4z+1}$: $\lambda=\tfrac14$, $m=1$. Both match §1 ($V=t^{1/2}$ and $t^{1/4}$).
2. The integrand factorises: $\zeta(z)=\int_0^1a^{2k_1z}da\int_0^1b^{2k_2z}db=\frac1{(2k_1z+1)(2k_2z+1)}$. The poles are at $-\frac1{2k_1}$ and $-\frac1{2k_2}$. The one nearest the origin gives $\lambda=\min\big(\frac1{2k_1},\frac1{2k_2}\big)=\frac1{2\max(k_1,k_2)}$. If $k_1\ne k_2$ the poles are distinct and simple, so $m=1$. If $k_1=k_2=k$ they coincide into a double pole, $\zeta=(2kz+1)^{-2}$, and $m=2$. Multiplicity is two exponents that happen to be equal: two directions that each give the same scaling, compounding into a logarithm. $(ab)^2$ is $k=1$.
3. $k_1=1$, $k_2=2$: $\zeta=\frac1{(2z+1)(4z+1)}$, poles at $-\tfrac14$ and $-\tfrac12$. So $\lambda=\tfrac14$, $m=1$. Near $z=-\tfrac14$, $\zeta\approx\frac{1}{(2z+1)\cdot4(z+\frac14)}\to\frac{1/2}{z+\frac14}$ (since $2z+1=\tfrac12$ there), and near $z=-\tfrac12$, $\zeta\approx\frac{-1/2}{z+\frac12}$. So $\bar Z_n=\tfrac12\Gamma(\tfrac14)\,n^{-1/4}-\tfrac12\Gamma(\tfrac12)\,n^{-1/2}+O(e^{-n})$: the second pole is the first correction, suppressed by $n^{-1/4}$.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex2-3"></a>

**Exercise 2.3 — Every pole is a term** (optional). $d=1$, $K=w^2$ on all of
$\mathbb R$, with the standard Gaussian prior $\varphi(w)=e^{-w^2/2}/\sqrt{2\pi}$.

Inverting the Mellin transform of §2,

$$
\bar Z_n=\frac1{2\pi i}\int_{c-i\infty}^{c+i\infty}\Gamma(-z)\,\zeta(z)\,n^{z}\,dz,\qquad-\lambda<c<0 .
$$

Moving the contour to the left picks up the residue of $\Gamma(-z)\zeta(z)n^z$ at each
pole of $\zeta$; the poles of $\Gamma(-z)$, at $z=0,1,2,\dots$, lie to the right and do not
contribute.

1. Show $\bar Z_n=(1+2n)^{-1/2}$ exactly.
2. Show $\zeta(z)=2^z\,\Gamma(z+\tfrac12)/\sqrt\pi$. Where are its poles?
3. Show that the residue of $\Gamma(-z)\zeta(z)n^z$ at the pole $z=-\tfrac12-k$ is the $k$-th term of the binomial expansion $(1+2n)^{-1/2}=(2n)^{-1/2}\sum_{k\ge0}\binom{-1/2}{k}(2n)^{-k}$.

So the regular case has subleading poles too. They come from the prior varying across
the basin, and they give the corrections in powers of $1/n$.
"""
    ),
    answer(
        r"""
1. $\bar Z_n=\int\frac{e^{-w^2/2-nw^2}}{\sqrt{2\pi}}\,dw$ is a Gaussian integral with precision $1+2n$, so $\bar Z_n=(1+2n)^{-1/2}$.
2. $\zeta(z)=\mathbb E\,|w|^{2z}$ for $w\sim N(0,1)$. With $u=w^2/2$: $\zeta(z)=\frac{2}{\sqrt{2\pi}}\int_0^\infty(2u)^z e^{-u}\frac{du}{\sqrt{2u}}=\frac{2^z}{\sqrt\pi}\int_0^\infty u^{z-1/2}e^{-u}du=\frac{2^z\,\Gamma(z+\frac12)}{\sqrt\pi}$. $\Gamma$ has simple poles at $0,-1,-2,\dots$ with residue $(-1)^k/k!$ at $-k$, so $\zeta$ has simple poles at $z=-\tfrac12-k$, $k=0,1,2,\dots$, with residues $\frac{2^{-1/2-k}(-1)^k}{k!\sqrt\pi}$. The largest is $-\tfrac12$: $\lambda=\tfrac12$, $m=1$.
3. The residue of $\Gamma(-z)\zeta(z)n^z$ at $z=-\tfrac12-k$ is
   $$\Gamma(\tfrac12+k)\,\frac{2^{-1/2-k}(-1)^k}{k!\sqrt\pi}\,n^{-1/2-k}.$$
   Using $\Gamma(\tfrac12+k)=\frac{(2k)!}{4^kk!}\sqrt\pi$, this is $(-1)^k\frac{(2k)!}{4^k(k!)^2}(2n)^{-1/2-k}$. And $\binom{-1/2}{k}=(-1)^k\frac{(2k)!}{4^k(k!)^2}$, so the residue is exactly $\binom{-1/2}{k}(2n)^{-1/2}(2n)^{-k}$. For $k=0$: $(2n)^{-1/2}$. For $k=1$: $-\tfrac14(2n)^{-1/2}n^{-1}$. Summing over all poles reproduces $(1+2n)^{-1/2}$, which converges for $n>\tfrac12$.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex2-4"></a>

**Exercise 2.4 — Same model, different $\lambda$: the cubic parametrisation** (adapted
from the Iliad B.3 worksheet, Ex 3.7). The model is $p(x\mid\theta)=N(\theta,1)$ and the
truth is $q=N(\theta_0,1)$, so $K=\tfrac12(\theta-\theta_0)^2$, the KL divergence from $q$ to
the model. Now parametrise by $\mu$ with $\theta=\mu^3$, so that
$K(\mu)=\tfrac12(\mu^3-\mu_0^3)^2$. Take the uniform prior on $\mu\in[-1,1]$.

1. Show that $\mu\mapsto\mu^3$ reaches every $\theta$, so both parametrisations describe the same set of distributions. Compute $K''(\mu_0)$. For $\mu_0\ne0$, what is $\lambda$?
2. Take $\mu_0=0$. Compute $\zeta(z)$ and read off $\lambda$ and $m$.
3. In the $\theta$ coordinate, with a uniform prior on $\theta$, $\lambda=\tfrac12$ at $\theta_0=0$. The two models are the same family of distributions with the same truth. Where does the extra volume at $\mu_0=0$ come from? Push the uniform prior on $\mu$ forward to $\theta$.
"""
    ),
    answer(
        r"""
1. $\mu\mapsto\mu^3$ is a bijection of $\mathbb R$, so every $\theta$ is $\mu^3$ for exactly one $\mu$, and both parametrisations describe the family $\{N(\theta,1)\}$. $K'(\mu)=3\mu^2(\mu^3-\mu_0^3)$, $K''(\mu)=6\mu(\mu^3-\mu_0^3)+9\mu^4$, so $K''(\mu_0)=9\mu_0^4$. For $\mu_0\ne0$ this is positive: the minimum is regular and $\lambda=\tfrac12$, $m=1$.
2. $K=\tfrac12\mu^6$ and the prior density is $\tfrac12$: $\zeta(z)=\int_0^1 2^{-z}\mu^{6z}\,d\mu=\frac{2^{-z}}{6z+1}$. A simple pole at $-\tfrac16$: $\lambda=\tfrac16$, $m=1$. (The factor $2^{-z}$ is entire and only changes the constant.) Directly: $V(t)=(2t)^{1/6}$.
3. The push-forward of the uniform density $\tfrac12$ on $\mu$ under $\theta=\mu^3$ is $\varphi_\theta(\theta)=\tfrac12\,\lvert d\mu/d\theta\rvert=\tfrac16\lvert\theta\rvert^{-2/3}$ on $[-1,1]$: infinite density at $\theta=0$. In $\theta$ coordinates, $\zeta(z)=\int(\theta^2/2)^z\,\tfrac16\lvert\theta\rvert^{-2/3}d\theta=\frac{2^{-z}}{6z+1}$, the same function, with the pole moved from $-\tfrac12$ to $-\tfrac16$ by the factor $\lvert\theta\rvert^{-2/3}$.
   So the evidence does not depend on coordinates; the prior does. "A prior that is smooth and positive at $w_0$" picks out different priors in different coordinates, and the uniform prior on $\mu$ piles mass onto $\theta=0$. $\lambda$ is a property of the model, its parametrisation and the prior class together. It is unchanged by smooth invertible changes of coordinates whose Jacobian is bounded and nonzero (the prior stays smooth and positive), and $\mu\mapsto\mu^3$ fails exactly there: its Jacobian vanishes at $0$.
"""
    ),
    markdown(
        r"""
### Code: the cubic model with data

So far everything used $K$, the population loss. The free energy of NB1 uses the
empirical loss $L_n$ instead. Under Watanabe's conditions (analytic model, compact
parameter set, prior positive at $w_0$; a realisable $q$ is the simplest case), his free
energy theorem reads

$$
F_n=nL_n(w_0)+\lambda\log n-(m-1)\log\log n+O_p(1),
$$

with the same $\lambda$ and $m$ as for $\bar F_n$. The data enter through $nL_n(w_0)$ and
a random $O(1)$ term; the rest of the $n$-dependence is that of $\bar F_n$. (NB1's
$nL_n(\hat w)$ differs from $nL_n(w_0)$ by $O_p(1)$ in the regular case.)

(Adapted from the Iliad B.3 worksheet, Ex 4.1.) Here $F_n$ is computed from actual
samples, to see this. Model
$p(x\mid\mu)=N(\mu^3,1)$, prior $\mu\sim N(0,1)$, truth $q=N(\mu_0^3,1)$. For a dataset
only the sample mean $\bar x$ matters:
$nL_n(\mu)-nL_n(\mu_0)=\tfrac n2\big[(\bar x-\mu^3)^2-(\bar x-\mu_0^3)^2\big]$.
The plot shows $F_n-nL_n(\mu_0)$ against $\log n$ for $\mu_0=0$ and $\mu_0=1$: the mean over
200 datasets (thick) and three single datasets (thin). Watanabe's theorem predicts slope
$\lambda$, with a random $O(1)$ offset per dataset. With this Gaussian prior,
$\zeta(z)=4^z\,\Gamma(3z+\tfrac12)/\sqrt\pi$ at $\mu_0=0$, whose largest pole is still
$-\tfrac16$.
"""
    ),
    code(
        r"""
rng = np.random.default_rng(0)
ns_c = np.logspace(1, 5, 17)
xi = rng.standard_normal(200)        # one draw per dataset; the sample mean is mu0^3 + xi / sqrt(n)
base = np.linspace(-5, 5, 4001)


def excess_free_energy(n, xbar, mu0):
    # F_n - n L_n(mu0) for p(x|mu) = N(mu^3, 1) with prior N(0, 1), as a 1-d integral over mu
    peak = np.cbrt(xbar)
    width = 1 / (3 * max(abs(peak), n ** (-1 / 6)) ** 2 * np.sqrt(n))   # posterior width near the peak
    mu = np.union1d(base, peak + width * np.linspace(-15, 15, 601))    # refine the grid there
    logf = -mu**2 / 2 - np.log(2 * np.pi) / 2 - n / 2 * ((xbar - mu**3) ** 2 - (xbar - mu0**3) ** 2)
    top = logf.max()
    return -(top + np.log(integrate.trapezoid(np.exp(logf - top), mu)))


fig, ax = plt.subplots(figsize=(6.5, 4))
for mu0, lam, col in [(0.0, 1 / 6, "C0"), (1.0, 1 / 2, "C3")]:
    Fx = np.array([[excess_free_energy(n, mu0**3 + x / np.sqrt(n), mu0) for x in xi] for n in ns_c])
    ax.plot(np.log(ns_c), Fx[:, :3], color=col, lw=0.6, alpha=0.6)
    mean = Fx.mean(axis=1)
    ax.plot(np.log(ns_c), mean, color=col, lw=2.5, label=rf"$\mu_0={mu0:g}$: mean of 200 datasets")
    ax.plot(np.log(ns_c), mean[-1] + lam * np.log(ns_c / ns_c[-1]), "k:", lw=1)
    sel = ns_c >= 1e3
    print(f"mu0 = {mu0:g}: lambda = {lam:.3f}, slope of the mean on 1e3 <= n <= 1e5 = "
          f"{np.polyfit(np.log(ns_c[sel]), mean[sel], 1)[0]:.3f}, spread across datasets (sd) = {Fx[-1].std():.2f}")
ax.set_xlabel(r"$\log n$"); ax.set_ylabel(r"$F_n - nL_n(\mu_0)$"); ax.legend(fontsize=8)
ax.set_title(r"cubic model: dotted lines have slope $\lambda$ (1/6 and 1/2)")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
Each dataset's curve is the mean curve shifted by an $O(1)$ amount that does not grow
with $n$, and every curve has slope $\lambda$: $\tfrac16$ at $\mu_0=0$ and $\tfrac12$ at
$\mu_0=1$. The data only add the random offset. Try $\mu_0=0.1$: for moderate $n$ the
slope looks like $\tfrac16$, not $\tfrac12$. NB3 §1 is about this.
"""
    ),
    markdown(
        r"""
**Readings for §2.**
"""
        + WATANABE22 + FURMAN + EXTRAS + FLAJOLET
    ),
    # ================================================================ 3
    markdown(
        r"""
<a id="slt2-3"></a>

## 3. Several basins

When the loss has several separated near-optimal regions, which one does the posterior
choose, and when does the choice change as $n$ grows? This section applies §1 to each
region and finds a trade between accuracy and volume.

*Notation.* Split $W$ into disjoint regions $U_\alpha$ ("basins"), each containing a local
minimum $w_\alpha$ of $L$. Let $w_0$ be the global minimiser, $K=L-L(w_0)$, and
$\Delta_\alpha=L(w_\alpha)-L(w_0)\ge0$ the excess loss of basin $\alpha$. The local
partition function and free energy are $\bar Z_n(U_\alpha)=\int_{U_\alpha}\varphi\,e^{-nK}$
and $\bar F_\alpha=-\log\bar Z_n(U_\alpha)$. Each basin has its own volume function, the
prior mass of $\{w\in U_\alpha:K(w)-\Delta_\alpha<t\}$, which behaves as
$c_\alpha t^{\lambda_\alpha}(-\log t)^{m_\alpha-1}$ for small $t$.

**Free energy as a minimum.** $\bar Z_n=\sum_\alpha\bar Z_n(U_\alpha)$, so

$$
\bar F_n=-\log\sum_\alpha e^{-\bar F_\alpha},\qquad
\min_\alpha\bar F_\alpha-\log(\#\text{basins})\le\bar F_n\le\min_\alpha\bar F_\alpha,
$$

and the posterior gives basin $\alpha$ mass proportional to $e^{-\bar F_\alpha}$. Applying
§1's result $\bar F\approx\lambda\log n-(m-1)\log\log n+\text{const}$ in each basin, after
taking out the factor $e^{-n\Delta_\alpha}$,

$$
\bar F_\alpha\approx n\Delta_\alpha+\lambda_\alpha\log n-(m_\alpha-1)\log\log n+C_\alpha,
\qquad C_\alpha=-\log\big(c_\alpha\Gamma(\lambda_\alpha+1)\big).
$$

**Two basins.** Basin $A$ fits worse, $\Delta_A=\Delta L>0$, but has more near-optimal
volume: $\lambda_A<\lambda_B$. Basin $B$ contains the global minimum. With $m=1$ in both,
the log posterior odds are

$$
\log\frac{P(A)}{P(B)}=\bar F_B-\bar F_A\approx-n\,\Delta L+(\lambda_B-\lambda_A)\log n+(C_B-C_A).
$$

At small $n$ the $\log n$ term wins and the posterior sits in $A$, the larger volume.
At large $n$ the linear term wins and it moves to $B$, the better fit. The switch is at
the critical $n^*$ where the odds are even:

$$
n^*\,\Delta L=\Delta\lambda\,\log n^*+\Delta C,\qquad\Delta\lambda=\lambda_B-\lambda_A,\ \Delta C=C_B-C_A,
$$

or $n^*/\log n^*\approx\Delta\lambda/\Delta L$ when $\Delta C$ is negligible.

**What changes at $n^*$.** Differentiating $\bar F_n$ gives the exact identity
$d\bar F_n/dn=\mathbb E_{\rm post}[K]$, the posterior mean of $K$ (the posterior here is
$\propto\varphi e^{-nK}$). Inside one basin, $\bar F\approx\lambda\log n$ gives
$\mathbb E_{\rm post}[nK]\to\lambda$: each quadratic direction contributes $\tfrac12$, a
quartic one $\tfrac14$. Across $n^*$, $\mathbb E_{\rm post}[K]$ drops by about $\Delta L$.

With data, $n\Delta L$ is replaced by $n\big(L_n(w_A)-L_n(w_B)\big)$, which fluctuates by
$O(\sqrt n)$ between datasets, so the switching point of one dataset scatters around the
$n^*$ computed here.
"""
    ),
    markdown(
        r"""
### Code: an accuracy-versus-volume switch

$d=8$ parameters per basin, with uniform prior mass $\tfrac12$ on a cube $[-1,1]^8$ around
each minimum. In basin $A$, $K=\Delta L+\sum_iw_i^4$ with $\Delta L=0.01$, so
$\lambda_A=8\cdot\tfrac14=2$. In basin $B$, $K=\sum_iw_i^2$, so $\lambda_B=8\cdot\tfrac12=4$.
Both local partition functions factorise into 1-d integrals, so everything is exact.
"""
    ),
    code(
        r"""
d, dL = 8, 0.01
lamA, lamB = d / 4, d / 2
ns3 = np.linspace(200, 4000, 381)
logZA = np.log(0.5) - ns3 * dL + d * np.log(Z_pow(4, ns3))   # basin A: K = dL + sum_i w_i^4
logZB = np.log(0.5) + d * np.log(Z_pow(2, ns3))              # basin B: K = sum_i w_i^2
F3 = -np.logaddexp(logZA, logZB)
pB = 1 / (1 + np.exp(logZA - logZB))                          # posterior mass of basin B
EK = np.gradient(F3, ns3)                                     # dFbar/dn = posterior mean of K

log_odds = lambda n: d * np.log(Z_pow(4, n)) - n * dL - d * np.log(Z_pow(2, n))
n_exact = optimize.brentq(log_odds, 100, 1e5)
n_lead = optimize.brentq(lambda n: n * dL - (lamB - lamA) * np.log(n), 50, 1e5)
lo, hi = ns3[np.searchsorted(pB, 0.27)], ns3[np.searchsorted(pB, 0.73)]
print(f"n* exact: {n_exact:.0f};  from n dL = (lamB - lamA) log n alone: {n_lead:.0f}")
print(f"P(B) goes from 27% to 73% between n = {lo:.0f} and {hi:.0f} (width {(hi - lo) / n_exact:.0%} of n*)")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(ns3, -logZA, "C1--", label=r"$\bar F_A$")
ax[0].plot(ns3, -logZB, "C0--", label=r"$\bar F_B$")
ax[0].plot(ns3, F3, "k", lw=1.5, label=r"$\bar F_n$")
ax[0].set_title("local and total free energies"); ax[0].legend()
ax[1].plot(ns3, pB, "k")
ax[1].axvline(n_exact, color="C3", ls=":", label=f"exact $n^*$ = {n_exact:.0f}")
ax[1].axvline(n_lead, color="C2", ls="--", lw=0.8, label=f"leading terms: {n_lead:.0f}")
ax[1].set_title("posterior mass of basin B"); ax[1].legend(fontsize=8)
ax[2].plot(ns3, EK / dL, "k")
ax[2].plot(ns3, (dL + lamA / ns3) / dL, "C1--", lw=0.8, label=r"basin A: $\Delta L+\lambda_A/n$")
ax[2].plot(ns3, (lamB / ns3) / dL, "C0--", lw=0.8, label=r"basin B: $\lambda_B/n$")
ax[2].set_title(r"$E_{\rm post}[K]\,/\,\Delta L$ (= $d\bar F_n/dn$ / $\Delta L$)"); ax[2].legend(fontsize=8)
for a in ax:
    a.set_xlabel("$n$")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The posterior moves from the wide, worse basin to the narrow, better one over a range of
about 16% of $n^*$. Here the constants $C_\alpha$ barely matter (the two leading terms
alone give $n^*\approx1457$); they would if the basins had different prior masses or
curvatures. Exercise 3.1 part 4 estimates the width.
"""
    ),
    markdown(
        r"""
<a id="slt2-ex3-1"></a>

**Exercise 3.1 — Locating and sharpening a basin switch** (adapted from the Iliad B.3
worksheet, Ex 4.2). Two basins $A$ and $B$ with local free energies $\bar F_A(n)$ and
$\bar F_B(n)$, and total $\bar F(n)=-\log\big(e^{-\bar F_A}+e^{-\bar F_B}\big)$. The
posterior mass of $B$ is $\sigma(\bar F_A-\bar F_B)$, with $\sigma(x)=1/(1+e^{-x})$.

1. Show $\min(\bar F_A,\bar F_B)-\log2\le\bar F\le\min(\bar F_A,\bar F_B)$.
2. Take $\bar F_A\approx0.5n+2\log n$ and $\bar F_B\approx0.3n+20\log n$. Which basin fits better, and which has more volume? Find $n^*$, and say which basin the posterior prefers on each side.
3. In general, write $\bar F_\alpha\approx n\Delta_\alpha+\lambda_\alpha\log n$. Show that a switch at some $n>e$ happens only if one basin is better in $\Delta$ and the other in $\lambda$. Give a leading-order formula for $n^*$ in terms of $r=\Delta\lambda/\Delta L$, where $\Delta L=\Delta_A-\Delta_B$ and $\Delta\lambda=\lambda_B-\lambda_A$.
4. *Width* (optional). The posterior mass of $B$ rises from 27% to 73% when the log odds $\bar F_A-\bar F_B$ go from $-1$ to $1$. Estimate the width $\delta n$ of this window for the numbers in part 2, and show that in general $\delta n/n^*\approx2/\big(\Delta\lambda(\log n^*-1)\big)$. Check it against the code above ($\Delta\lambda=2$, $n^*\approx1480$).

<details><summary>Hint</summary>

Part 3: $n/\log n$ is increasing for $n>e$, so it takes each value above $e$ once there. Iterate $n\leftarrow r\log n$ starting from $n=r\log r$.
</details>
"""
    ),
    answer(
        r"""
1. Let $F_{\min}=\min(\bar F_A,\bar F_B)$. Then $e^{-F_{\min}}\le e^{-\bar F_A}+e^{-\bar F_B}\le2e^{-F_{\min}}$. Taking $-\log$ reverses the inequalities: $F_{\min}-\log2\le\bar F\le F_{\min}$. So $\bar F$ is the minimum up to $\log2$, and the two agree away from the crossing, where one exponential dominates.
2. $B$ has the lower loss per sample ($0.3<0.5$); $A$ has the smaller $\lambda$ ($2<20$), so more volume. The crossing solves $0.2\,n=18\log n$, i.e. $n/\log n=90$: $n^*\approx571$ (iterating $n\leftarrow90\log n$ from $90\log90\approx405$: $540,\ 566,\ 571$). Below $n^*$, $\bar F_A<\bar F_B$ and the posterior sits in $A$ (volume wins); above it, in $B$ (fit wins). There is a second root near $n\approx1.01$, outside the range where the asymptotic forms mean anything.
3. The crossing $\bar F_A=\bar F_B$ solves $n\,\Delta L=\Delta\lambda\log n$. For $n>1$ the right side has the sign of $\Delta\lambda$, so a solution needs $\Delta L$ and $\Delta\lambda$ of the same sign: the basin with the lower loss must have the larger $\lambda$. Then $n/\log n=r$, which has exactly one solution with $n>e$ when $r>e$. Iterating, $n^*\approx r\log r\,(1+o(1))$; more precisely $n^*\approx r(\log r+\log\log r)$, with a relative error that shrinks as $r$ grows (5% at $r=90$). If one basin is better in both respects, it wins at every $n$.
4. The log odds $f(n)=\bar F_A-\bar F_B=n\,\Delta L-\Delta\lambda\log n$ (here $0.2n-18\log n$) have slope $f'(n^*)=\Delta L-\Delta\lambda/n^*$ at the crossing, so $\delta n\approx2/f'(n^*)$. For part 2: $f'=0.2-18/571\approx0.168$, $\delta n\approx12$, about 2% of $n^*$. In general, use $\Delta L=\Delta\lambda\log n^*/n^*$ at the crossing: $f'(n^*)=\Delta\lambda(\log n^*-1)/n^*$, so
   $$\frac{\delta n}{n^*}\approx\frac{2}{\Delta\lambda\,(\log n^*-1)} .$$
   For the code example, $2/(2\times(7.30-1))\approx0.16$, matching the 16% seen there. Part 2 is much sharper because $\Delta\lambda=18$. The sharpness is controlled by $\Delta\lambda\log n^*$, which grows with the number of parameters that differ between the basins, and only logarithmically with $n^*$. At finite $\Delta\lambda$ the switch is never a true discontinuity.
"""
    ),
    markdown(
        r"""
**Metastability, and the thermodynamic analogy.** In thermodynamic language the basin
switch is a first-order transition between two phases, with the kink in the free energy
rounded off; the analogy stops at the thermodynamic limit, since $n$ is an inverse
temperature here, not a system size. The posterior is an equilibrium object: at each $n$
it weights the basins by $e^{-\bar F_\alpha}$, with no history and no barrier to cross.
Metastability, a phase that persists after it has stopped being the free-energy minimum,
needs dynamics. It appears once the posterior is sampled by a local method, such as an
MCMC chain or the SGLD sampler of NB3, started in one basin. If the barrier between
basins is high, the chain stays in $A$ long after $n^*$, and what it measures is the local
free energy $\bar F_A$, not $\bar F_n$. NB3's local learning coefficient is defined for
exactly this kind of region-restricted posterior.

**Reading for §3.** The keystone's two-basin cartoon, in the same section as above.
"""
    ),
    markdown(
        r"""
<a id="slt2-readings"></a>

## Readings and references

The keystone is at the top of the page; the §2 readings follow the cubic-model code.
"""
        + references(
            book("Algebraic Geometry and Statistical Learning Theory",
                 "https://www.cambridge.org/core/books/algebraic-geometry-and-statistical-learning-theory/9C8FD1BDC817E2FC79117C7F41544A3A",
                 "Sumio Watanabe (Cambridge, 2009)",
                 "the free energy theorem and the zeta function in full generality"),
            book("Mathematical Theory of Bayesian Statistics",
                 "https://www.routledge.com/Mathematical-Theory-of-Bayesian-Statistics/Watanabe/p/book/9780367734817",
                 "Sumio Watanabe (CRC, 2018)",
                 "the same results for a statistics audience, beyond the realisable case"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt2_volume", "slt2_volume"))
