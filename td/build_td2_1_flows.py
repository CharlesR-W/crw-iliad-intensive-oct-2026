#!/usr/bin/env python3
"""Build NB2 §1, "Gradient descent as a flow plus corrections" (td2_1_flows).

Shape (streamlined 2026-10-09): a short pitch, the two key papers (Barrett-Dherin;
Smith-Dherin-Barrett-De) and a one-line route. Each `##` section then gives the
question and answer in plain words, its lead paper(s), at most a short core-idea
paragraph, and one `### Optional: annotated derivations and walkthrough` holding the
derivation narrative, the exercises (bold-labelled paragraphs, not headings), the
worked solutions and the code. Sections: 1.1 stability on a quadratic; 1.2 backward
error analysis and implicit gradient regularization; 1.3 validity of the expansion;
1.4 shuffled-minibatch SGD; 1.5 heavy-ball momentum. A short trailing reading list.

Symbolic checks of every closed form: checks/td2_1_flows_bea.py.
Review log: ../reviews/td/td2_1_flows.md.
"""
from nbkit_td import answer, book, code, code_answer, markdown, reading, references, run

CELLS = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# NB2 §1 — Gradient descent as a flow plus corrections

**Training Dynamics, Iliad Intensive 2026**

Continuous-time stories about training replace gradient descent (GD) by gradient
flow. *Backward error analysis* recovers what that drops: GD follows a slightly
different flow, whose first correction penalizes the squared gradient norm
(*implicit gradient regularization*, IGR). This notebook derives that correction, its
SGD and momentum versions, and where the expansion stops being trustworthy.
"""
        + reading(
            "Implicit gradient regularization",
            "https://arxiv.org/abs/2009.11162",
            "Barrett & Dherin (ICLR 2021)",
            "To first order in the step size $\\eta$, GD follows gradient flow on $L+\\frac\\eta4\\|\\nabla L\\|^2$; the paper derives this (Theorem 3.1) and tests the resulting preference for flat minima on a two-parameter model and on deep nets.",
            "Section 3 is §1.2 below.",
            importance=5,
            length=2,
        )
        + reading(
            "On the origin of implicit regularization in stochastic gradient descent",
            "https://arxiv.org/abs/2101.12176",
            "Smith, Dherin, Barrett & De (ICLR 2021)",
            "The same calculation for shuffled minibatch SGD (§1.4): an extra implicit penalty on the spread of minibatch gradients, scaling like learning rate over batch size, with experiments showing that adding it explicitly recovers much of the benefit of a large learning rate.",
            importance=4,
            length=3,
        )
        + r"""
**Route.** $S$ is the sharpness (the largest Hessian eigenvalue). Each section's
derivations, exercises and code are in its optional walkthrough.

- [1.1 Stability on a quadratic](#td21-s1): GD converges iff $\eta S<2$.
- [1.2 Backward error analysis](#td21-s2): GD follows gradient flow on $L+\tfrac\eta4\|\nabla L\|^2$, to first order.
- [1.3 Is the expansion valid?](#td21-s3): an asymptotic series in $\eta S$, blind to the $\eta S=2$ threshold where training often sits.
- [1.4 The SGD version](#td21-s4): shuffled minibatches add a penalty scaling like $\eta/(\text{batch size})$.
- [1.5 Momentum](#td21-s5): heavy ball's stability limit, effective learning rate and stronger penalty.
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from fractions import Fraction
from math import comb

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def rk4(F, y0, T, n=4000):
    # Fixed-step RK4 for ydot = F(y); n steps over [0, T]. Accurate to ~1e-12 for the smooth
    # 2D fields used here, far below the errors we measure.
    y, dt = np.array(y0, float), T / n
    for _ in range(n):
        k1 = F(y); k2 = F(y + dt / 2 * k1); k3 = F(y + dt / 2 * k2); k4 = F(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def slope(xs, ys):
    # least-squares slope of log(ys) against log(xs)
    return np.polyfit(np.log(xs), np.log(ys), 1)[0]


# The running nonquadratic example: a two-layer scalar "network" f = a*b fitting target 1,
# L(a, b) = (ab - 1)^2 / 2.  (Barrett & Dherin use the same model.)
def grad_ab(p):
    a, b = p
    r = a * b - 1
    return np.array([r * b, r * a])


def hess_ab(p):
    a, b = p
    return np.array([[b * b, 2 * a * b - 1], [2 * a * b - 1, a * a]])
"""
    ),
    # ------------------------------------------------------------------ 1.1
    markdown(
        r"""
<a id="td21-s1"></a>

## 1.1 Stability on a quadratic

On a quadratic loss, gradient descent with step size $\eta$ converges iff $\eta S<2$,
where the *sharpness* $S$ is the largest eigenvalue of the Hessian $H$. Near a minimum
the same holds for the linearization, so minima sharper than $2/\eta$ repel GD, while
gradient flow ($\dot\theta=-\nabla L$) converges to them. This threshold recurs in
the rest of the notebook and in `td2_2_edge_of_stability`.

The first section of Goh's [Why momentum really works](https://distill.pub/2017/momentum/)
(the lead reading of §1.5) does this case with interactive pictures.

<a id="td21-s1-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* GD is $\theta_{k+1}=\theta_k-\eta\nabla L(\theta_k)$; gradient flow uses
time $t=k\eta$, so one step advances time by $\eta$.

On $L(\theta)=\tfrac12\theta^\top H\theta$ with $H$ symmetric positive semidefinite,
GD is linear, $\theta_{k+1}=(I-\eta H)\theta_k$. In the eigenbasis $Hu_i=\lambda_iu_i$
each coordinate $c_i=u_i^\top\theta$ evolves on its own:

$$
c_i^{(k)}=(1-\eta\lambda_i)^k\,c_i^{(0)},
\qquad\text{while gradient flow gives}\qquad
c_i(t)=e^{-\lambda_it}c_i(0),\quad t=k\eta .
$$

One GD step multiplies by $1-\eta\lambda$ where the flow multiplies by
$e^{-\eta\lambda}=1-\eta\lambda+\tfrac12(\eta\lambda)^2-\dots$. They agree for
$\eta\lambda\ll1$ and part ways as $\eta\lambda$ grows:

- $0<\eta\lambda<1$: monotone decay, *faster* than the flow per unit time;
- $\eta\lambda=1$: that direction is solved in one step;
- $1<\eta\lambda<2$: the coordinate flips sign every step while it decays;
- $\eta\lambda=2$: a period-2 orbit; $\eta\lambda>2$: oscillating divergence.

Directions with $\lambda_i=0$ do not move, so GD converges iff $\eta S<2$.

**Exercise 1.1 — Best constant step; unstable minima.**

1. Take a quadratic whose nonzero eigenvalues lie in $[\mu,S]$. Find the fixed $\eta$ that minimizes the worst per-step contraction factor, and that factor in terms of $\kappa=S/\mu$. Where does the optimal $\eta$ put $\eta S$?
2. Let $\theta^*$ be a minimum of a smooth $L$ whose Hessian $H^*$ is positive definite, with sharpness $S^*=\lambda_{\max}(H^*)$. Show that if $\eta S^*>2$, GD started near $\theta^*$ generically does not converge to it, whereas gradient flow started near it does.

<details><summary>Hint</summary>

For 1, the worst factor is $\max(|1-\eta\mu|,|1-\eta S|)$; one term decreases in $\eta$ and the other eventually increases. For 2, linearize the GD map $\theta\mapsto\theta-\eta\nabla L(\theta)$ at $\theta^*$ and look at the eigenvalue of its Jacobian along the top eigenvector.
</details>
"""
    ),
    answer(
        r"""
**1.** The factor $|1-\eta\lambda|$ is largest at an endpoint of $[\mu,S]$, so the worst factor is $\max(|1-\eta\mu|,|1-\eta S|)$. For small $\eta$ the first dominates and decreases with $\eta$; once $\eta S>1$ the second grows. The minimum is where they balance with opposite signs, $1-\eta\mu=-(1-\eta S)$:

$$
\eta^*=\frac{2}{\mu+S},\qquad \text{factor}=\frac{S-\mu}{S+\mu}=\frac{\kappa-1}{\kappa+1}.
$$

Then $\eta^*S=2\kappa/(\kappa+1)\in[1,2)$. So the best constant step on an ill-conditioned quadratic deliberately makes the sharpest direction oscillate, already past where the small-$\eta$ expansion of §1.2 converges (§1.3).

**2.** Write $\theta=\theta^*+e$. Since $\nabla L(\theta^*)=0$, the GD map is $e\mapsto(I-\eta H^*)e+O(\|e\|^2)$. Its Jacobian at $e=0$ has eigenvalue $1-\eta S^*<-1$ along the top eigenvector $u$. A fixed point of a smooth map with a Jacobian eigenvalue of modulus above 1 cannot attract a whole neighbourhood: the starts that converge to it lie on a lower-dimensional local center-stable manifold, tangent to the eigenvectors with $|1-\eta\lambda_i|\le1$. A generic nearby start is pushed away, oscillating along $u$ with growth factor about $\eta S^*-1$ per step, until nonlinear terms take over. Gradient flow linearizes to $\dot e=-H^*e$, which with $H^*\succ0$ contracts every direction, so it converges from every nearby start. (With a zero Hessian eigenvalue a strict minimum need not attract all nearby flow lines, hence the positive-definiteness assumption.)

So minima sharper than $2/\eta$ repel GD, at every $\eta$. Exercise 1.2.c, item 3 shows why the expansion of §1.2 cannot see this.
"""
    ),
    # ------------------------------------------------------------------ 1.2
    markdown(
        r"""
<a id="td21-s2"></a>

## 1.2 Backward error analysis

GD does not follow gradient flow exactly, but to first order in the step size $\eta$
it follows gradient flow on a modified loss,

$$
\dot\theta=-\nabla\tilde L,\qquad \tilde L=L+\frac\eta4\|\nabla L\|^2 ,
$$

with time $t=k\eta$. The extra term penalizes the squared gradient norm: this is
*implicit gradient regularization* (IGR).

**Paper:** Barrett & Dherin, [Implicit gradient regularization](https://arxiv.org/abs/2009.11162)
(card at the top), Section 3. The method is standard numerical analysis: Hairer,
Lubich and Wanner, *Geometric Numerical Integration*, ch. IX (see Readings).

**Core idea.** Write $g=\nabla L$ and $H$ for the Hessian. Over a time $\eta$,
gradient flow moves by $-\eta g+\tfrac{\eta^2}2Hg+O(\eta^3)$, because the gradient
changes along the way. GD uses the gradient from the start of the step and moves by
exactly $-\eta g$, overshooting by $-\tfrac{\eta^2}2Hg$. A flow that reproduces GD must
add that overshoot as a drift, $-\tfrac\eta2Hg=-\nabla\big(\tfrac\eta4\|g\|^2\big)$.
Near a minimum $\|\nabla L\|^2\approx\delta^\top H^2\delta$ for a displacement
$\delta$, so the penalty is heaviest in sharp directions. It vanishes at every
critical point, so local minima of $L$ stay local minima of $\tilde L$ and IGR does
not help GD escape them. It matters when the minima
form a family, as in overparametrized models, where it changes *which* minimum GD
reaches (Exercise 1.2.d). That GD ends at flatter minima in general is Barrett and
Dherin's prediction, supported by experiments, not a theorem.

<a id="td21-s2-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* loss $L$, $g=\nabla L$, $H=\nabla^2L$, step $\eta$; GD is
$\theta_{k+1}=\theta_k-\eta g(\theta_k)$ and $t=k\eta$. For a general one-step
numerical method the state is $y$ and the step is $h$; for GD, $y=\theta$ and $h=\eta$.

*Forward error* asks how far the iterates are from the true flow; *backward error*
asks which nearby ODE reproduces them exactly. The difference between that ODE and the
original says what the discretization does. (Implicit biases of gradient flow itself,
such as max-margin solutions, are a different story.)

**The setup.** A one-step method with step $h$ maps $y\mapsto\Phi_h(y)$, with

$$
\Phi_h(y)=y+hf(y)+h^2d_2(y)+h^3d_3(y)+\cdots .
$$

GD is $\Phi_\eta(\theta)=\theta-\eta\nabla L(\theta)$: $h=\eta$, $f=-\nabla L$, all
$d_j=0$. Look for a *modified vector field*

$$
\tilde f=f+hf_2+h^2f_3+\cdots
$$

whose exact time-$h$ flow equals $\Phi_h$ as a power series in $h$. (Indexing as in
Hairer–Lubich–Wanner; Barrett and Dherin call the $O(h)$ correction $f_1$.)

**The recipe.** The time-$h$ flow of $\dot y=F(y)$ expands as

$$
y(h)=y+hF+\frac{h^2}{2}F'F+\frac{h^3}{6}\big(F''(F,F)+F'F'F\big)+O(h^4),
$$

with $F'$ the Jacobian and $F''(u,v)$ the second derivative applied to two vectors.
Substitute $F=\tilde f$ and match powers of $h$ against $\Phi_h$. At order $h^j$ the
unknown $f_j$ enters linearly with coefficient 1 and everything else is known from
lower orders, so the modified equation exists and is unique as a formal series. At
first order, $f_2=d_2-\tfrac12f'f$ (Exercise 1.2.a); that is all §1.4 and §1.5 use.

**What the truncation buys.** Keeping $r$ terms,
$\tilde f_{[r]}=f+hf_2+\dots+h^{r-1}f_r$, gives local error $O(h^{r+1})$ per step and
global error $O(h^r)$ over a fixed time, with constants depending on that time and on
derivatives of $f$. Gradient flow ($r=1$) is $O(\eta)$ off; the IGR flow ($r=2$) is
$O(\eta^2)$ off. The code after Exercise 1.2.d checks both slopes, and how much of GD's
actual shift the IGR flow accounts for.

**Exercise 1.2.a — The recipe.**

1. Derive the $h^3$ term of the flow expansion, $\tfrac16\big(F''(F,F)+F'F'F\big)$.
2. Match orders to show

$$
f_2=d_2-\tfrac12f'f,\qquad
f_3=d_3-\tfrac16\big(f''(f,f)+f'f'f\big)-\tfrac12\big(f'f_2+f_2'f\big).
$$

<details><summary>Hint</summary>

Along a solution, $\ddot y=F'(y)\dot y=F'F$. Differentiate once more with the product rule: the derivative of $F'(y)$ along the solution is $F''(y)(\dot y,\cdot)$. For 2, substitute $F=f+hf_2+h^2f_3$ into each term and keep everything up to $h^3$. Only the $\tfrac{h^2}{2}F'F$ term produces cross terms at $h^3$.
</details>
"""
    ),
    answer(
        r"""
**1.** Along a solution of $\dot y=F(y)$: $\dot y=F$, and $\ddot y=\frac{d}{dt}F(y)=F'(y)\dot y=F'F$. Differentiating again, both factors of $F'(y)F(y)$ change along the solution:

$$
\dddot y=\frac{d}{dt}\big[F'(y)\big]F+F'\frac{d}{dt}F(y)=F''(y)(\dot y,F)+F'F'\dot y=F''(F,F)+F'F'F .
$$

Taylor's theorem in $t$ gives $y(h)=y+h\dot y+\tfrac{h^2}{2}\ddot y+\tfrac{h^3}{6}\dddot y+O(h^4)$, which is the stated expansion.

**2.** Put $F=f+hf_2+h^2f_3$. Then, up to $O(h^4)$,

- $hF=hf+h^2f_2+h^3f_3$;
- $\tfrac{h^2}{2}F'F=\tfrac{h^2}{2}f'f+\tfrac{h^3}{2}\big(f'f_2+f_2'f\big)$, since $F'=f'+hf_2'+\dots$ and $F=f+hf_2+\dots$;
- $\tfrac{h^3}{6}\big(F''(F,F)+F'F'F\big)=\tfrac{h^3}{6}\big(f''(f,f)+f'f'f\big)$.

Matching with $\Phi_h=y+hf+h^2d_2+h^3d_3$:

- $h^1$: $f=f$. The method is consistent.
- $h^2$: $f_2+\tfrac12f'f=d_2$, so $f_2=d_2-\tfrac12f'f$.
- $h^3$: $f_3+\tfrac12(f'f_2+f_2'f)+\tfrac16(f''(f,f)+f'f'f)=d_3$, which gives the stated $f_3$.

Each order is linear in the new unknown with coefficient 1, which is the general pattern. (Checked symbolically on random polynomial vector fields in `checks/td2_1_flows_bea.py`.)
"""
    ),
    markdown(
        r"""
**Exercise 1.2.b — Gradient descent.**

1. For GD ($f=-g$, $d_2=d_3=0$), show $f_2=-\tfrac12Hg$ and that this equals $-\nabla\big(\tfrac14\|g\|^2\big)$. Conclude $\dot\theta=-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)$.
2. *(Optional.)* Show $f_3=-\tfrac13H^2g-\tfrac1{12}\nabla^3L[g,g]$, where $\nabla^3L[g,g]_i=\sum_{jk}\partial_i\partial_j\partial_kL\,g_jg_k$.
3. *(Optional.)* For $L(x,y)=\tfrac12x^2y$, compute $f_3$ and its curl $\partial_xf_{3,y}-\partial_yf_{3,x}$. What does a nonzero curl say about describing GD as "gradient flow on a modified loss"?

<details><summary>Hint</summary>

$\nabla\|g\|^2=2Hg$, because $\partial_i\sum_jg_j^2=2\sum_jg_j\partial_ig_j$ and $\partial_ig_j=H_{ij}$. For 2, you need $f'=-H$, $f''(u,v)=-\nabla^3L[u,v]$ and $f_2'v=-\tfrac12\big(\nabla^3L[v,g]+H^2v\big)$. For 3, the only nonzero third derivative of $\tfrac12x^2y$ is $\partial_x\partial_x\partial_yL=1$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $f=-g$, $f'=-H$ and $f'f=(-H)(-g)=Hg$, so $f_2=-\tfrac12Hg$. Since $\partial_i\|g\|^2=2\sum_jg_jH_{ji}$ and $H$ is symmetric, $\nabla\|g\|^2=2Hg$, so $f_2=-\nabla\big(\tfrac14\|g\|^2\big)$. The modified equation to first order is

$$
\dot\theta=-g-\frac{\eta}{2}Hg=-\nabla\Big(L+\frac\eta4\|\nabla L\|^2\Big).
$$

**2.** The pieces: $f'f'f=(-H)(-H)(-g)=-H^2g$; $f''(f,f)=-\nabla^3L[-g,-g]=-\nabla^3L[g,g]$; $f'f_2=(-H)(-\tfrac12Hg)=\tfrac12H^2g$; and $f_2'v=-\tfrac12\big(\nabla^3L[v,g]+H\,Hv\big)$ (product rule on $Hg$), so $f_2'f=f_2'(-g)=\tfrac12\big(\nabla^3L[g,g]+H^2g\big)$. Write $T=\nabla^3L[g,g]$. Then

$$
f_3=-\tfrac16\big(-T-H^2g\big)-\tfrac12\Big(\tfrac12H^2g+\tfrac12T+\tfrac12H^2g\Big)
=\tfrac16T+\tfrac16H^2g-\tfrac12H^2g-\tfrac14T
=-\tfrac13H^2g-\tfrac1{12}T .
$$

On a quadratic $T=0$ and $f_3=-\tfrac13H^2g$, which Exercise 1.2.c confirms independently.

**3.** For $L=\tfrac12x^2y$: $g=(xy,\ \tfrac12x^2)$ and $H=\begin{pmatrix}y&x\\x&0\end{pmatrix}$. Then $Hg=(xy^2+\tfrac12x^3,\ x^2y)$ and $H^2g=H(Hg)=(xy^3+\tfrac32x^3y,\ x^2y^2+\tfrac12x^4)$. The only third derivative is $L_{xxy}=1$, so $T_x=2L_{xxy}g_xg_y=x^3y$ and $T_y=L_{yxx}g_x^2=x^2y^2$. Hence

$$
f_3=\Big(-\tfrac13xy^3-\tfrac7{12}x^3y,\ \ -\tfrac5{12}x^2y^2-\tfrac16x^4\Big),
\qquad
\partial_xf_{3,y}-\partial_yf_{3,x}=-\tfrac56xy^2-\tfrac23x^3+xy^2+\tfrac7{12}x^3=-\tfrac{x}{12}\big(x^2-2y^2\big).
$$

The curl is not identically zero, so $f_3$ is not the gradient of any function. GD's modified equation is a (Euclidean) gradient flow at first order, $\tilde L=L+\tfrac\eta4\|\nabla L\|^2$, but not beyond. "GD minimizes a modified loss" is a first-order statement. (The curl was also checked with sympy, for this loss and for $\tfrac12(xy-1)^2$, in `checks/td2_1_flows_bea.py`.)
"""
    ),
    markdown(
        r"""
**Exercise 1.2.c — The quadratic, to all orders.**

For $L=\tfrac12\lambda x^2$ in one dimension:

1. Find the exact modified equation: a linear ODE $\dot x=a\,x$ whose time-$\eta$ flow is exactly one GD step. Expand $a$ in powers of $\eta$ and compare with $f_2$ and $f_3$ from 1.2.b.
2. For which $\eta\lambda$ does the series converge? What happens to the exact modified equation when $1<\eta\lambda<2$, where GD still converges? Can *any* real one-dimensional ODE $\dot x=F(x)$ reproduce GD there? In $d$ dimensions, can a real *gradient* flow?
3. Return to the instability of Exercise 1.1, item 2: a minimum with $\eta S^*>2$ is locally unstable for GD. Why can no truncation of the modified equation see this?

<details><summary>Hint</summary>

You need $e^{a\eta}=1-\eta\lambda$. For the one-dimensional question, solutions of an autonomous ODE with $F(0)=0$ cannot cross $x=0$. For the gradient-flow question, the time-$\eta$ map of a linear gradient flow $\dot x=-Ax$ ($A$ symmetric) is $e^{-\eta A}$. For 3, linearize the truncated modified equation at the minimum and use item 1 eigendirection by eigendirection.
</details>
"""
    ),
    answer(
        r"""
**1.** GD gives $x_k=(1-\eta\lambda)^kx_0$, and the flow of $\dot x=ax$ gives $x(k\eta)=e^{ak\eta}x_0$. They agree for all $k$ iff $e^{a\eta}=1-\eta\lambda$:

$$
a=\frac{\log(1-\eta\lambda)}{\eta}=-\lambda\Big(1+\frac{\eta\lambda}{2}+\frac{(\eta\lambda)^2}{3}+\frac{(\eta\lambda)^3}{4}+\cdots\Big)=-\lambda\sum_{j\ge0}\frac{(\eta\lambda)^j}{j+1}.
$$

The $O(\eta)$ term $-\tfrac12\eta\lambda^2x$ is $\eta f_2=-\tfrac\eta2Hg$ with $H=\lambda$, $g=\lambda x$. The $O(\eta^2)$ term $-\tfrac13\eta^2\lambda^3x$ is $\eta^2f_3=-\tfrac{\eta^2}{3}H^2g$ (the third-derivative term vanishes for a quadratic). For $\eta\lambda<1$ the exact modified equation is the gradient flow of $-\tfrac{a}{2}x^2$: on a quadratic with $\eta S<1$, GD *is* a gradient flow, with a step-dependent curvature $-a>\lambda$.

**2.** The series $\sum_jz^j/(j+1)=-\log(1-z)/z$ converges for $|z|<1$, so the modified equation converges only for $\eta\lambda<1$. At $\eta\lambda=1$ the rate $a$ diverges, because GD reaches $0$ in one step and a flow can only approach it asymptotically. For $1<\eta\lambda<2$, $1-\eta\lambda<0$ and $\log(1-\eta\lambda)$ is complex: GD flips the sign of $x$ every step. No real ODE $\dot x=F(x)$ can do that. If $F$ is smooth and $F(0)=0$ (as it must be, since $0$ is a fixed point of GD), uniqueness of solutions forbids a trajectory from crossing $0$. So for $\eta\lambda>1$ no real one-dimensional modified equation exists. In $d$ dimensions a linear gradient flow $\dot x=-Ax$ ($A$ symmetric) has time-$\eta$ map $e^{-\eta A}$, which is positive definite, so it cannot produce a negative multiplier either. (A non-gradient flow can, e.g. one that rotates a two-dimensional block by $\pi$ per step while shrinking it, but that is no longer a "modified loss".) The literature has two ways out. One is to allow complex flows ([Rosca et al.](https://arxiv.org/abs/2302.01952)'s "principal flow", $\dot\theta=\sum_i\frac{\log(1-\eta\lambda_i)}{\eta\lambda_i}(g^\top u_i)u_i$, exact on quadratics). The other is to model only the two-step map or a time average, whose multiplier $(1-\eta\lambda)^2$ is positive; that is the route of the central flows in `td2_2_edge_of_stability`.

**3.** Linearize at $\theta^*$. If $\tilde F$ is the modified field, its time-$\eta$ flow fixes $\theta^*$, and the Jacobian of that flow map at $\theta^*$ is $e^{\eta\,D\tilde F(\theta^*)}$. As formal series in $\eta$ this must equal the Jacobian of the GD map, $I-\eta H^*$. So $D\tilde F(\theta^*)=\frac1\eta\log(I-\eta H^*)=-\sum_{j\ge0}\frac{\eta^j}{j+1}(H^*)^{j+1}$, which is item 1 applied eigendirection by eigendirection. Truncating at any order leaves a sum of negative semidefinite terms (negative definite when $H^*\succ0$), so every truncation makes $\theta^*$ stable, whatever $\eta$ is. The GD instability at $\eta S^*>2$ is a threshold of the discrete map. It does not grow smoothly from $\eta=0$, so no finite truncation contains it.
"""
    ),
    code(
        r"""
# Decay rate per unit time of one quadratic direction, in units of lambda, as a function of
# z = eta*lambda: gradient flow (1), the IGR flow (1 + z/2), the next truncation (1 + z/2 + z^2/3),
# and GD's actual rate -log|1 - z| / z (negative = growth). The series for GD's rate converges
# only for z < 1, and GD itself diverges for z > 2.
z = np.linspace(1e-3, 2.4, 1000)
gd_rate = -np.log(np.abs(1 - z)) / z
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(z, gd_rate, "k", lw=2, label=r"GD: $-\log|1-\eta\lambda|/(\eta\lambda)$")
ax.plot(z, np.ones_like(z), label="gradient flow")
ax.plot(z, 1 + z / 2, label=r"IGR flow: $1+\eta\lambda/2$")
ax.plot(z, 1 + z / 2 + z**2 / 3, "--", label=r"next order: $+(\eta\lambda)^2/3$")
for x0, txt in [(1, "series radius"), (2, "GD unstable")]:
    ax.axvline(x0, color="gray", ls=":")
    ax.text(x0 + 0.03, -0.8, txt, color="gray", fontsize=8)
ax.set_ylim(-1, 4); ax.set_xlabel(r"$\eta\lambda$"); ax.set_ylabel(r"decay rate / $\lambda$")
ax.legend(fontsize=8, loc="upper right"); plt.tight_layout(); plt.show()
for zz in [0.1, 0.3, 0.6, 0.9]:
    exact = -np.log(1 - zz) / zz
    print(f"eta*lam={zz:.1f}: GD rate {exact:.4f}, IGR {1 + zz/2:.4f} (rel. err {abs(1 + zz/2 - exact)/exact:.1%})")
"""
    ),
    markdown(
        r"""
**Exercise 1.2.d — IGR picks a flatter minimum.**

Use the two-parameter model $L(a,b)=\tfrac12(ab-1)^2$, a scalar two-layer linear
"network" $f=ab$ fitting the target 1. Its minima form the hyperbola $ab=1$. Write
$r=ab-1$ and $D=a^2-b^2$ (the *imbalance*).

1. Show that gradient flow conserves $D$.
2. Show that one GD step maps $D\mapsto(1-\eta^2r^2)\,D$ exactly.
3. Compute $\|\nabla L\|^2$ and show that the IGR flow gives $\dot D=-\eta r^2D$. Check that this matches item 2 at leading order.
4. At a minimum on $ab=1$, show the Hessian has eigenvalues $0$ and $a^2+b^2=\sqrt{D^2+4}$. Conclude that GD (and the IGR flow) end at a flatter minimum than gradient flow from the same start, and say how the effect scales with $\eta$.

<details><summary>Hint</summary>

$\nabla L=r\,(b,a)$. For 3, $a\partial_a-b\partial_b$ annihilates any function of $ab$. At a minimum the Hessian is $\begin{pmatrix}b^2&1\\1&a^2\end{pmatrix}$; use the trace and determinant.
</details>
"""
    ),
    answer(
        r"""
**1.** $\dot a=-rb$, $\dot b=-ra$, so $\dot D=2a\dot a-2b\dot b=-2rab+2rab=0$. The conservation comes from the rescaling symmetry $(a,b)\mapsto(ca,b/c)$, which leaves $L$ unchanged.

**2.** $a'=a-\eta rb$, $b'=b-\eta ra$. Then

$$
a'^2-b'^2=a^2-2\eta rab+\eta^2r^2b^2-b^2+2\eta rab-\eta^2r^2a^2=(a^2-b^2)(1-\eta^2r^2).
$$

So $|D|$ shrinks every step in which $r\ne0$ (for $\eta|r|<\sqrt2$). The discrete step breaks the conservation law at order $\eta^2$ per step, which is order $\eta$ per unit time.

**3.** $\|\nabla L\|^2=r^2(a^2+b^2)$, so $\tilde L=L+\tfrac\eta4r^2(a^2+b^2)$. Along $\dot a=-\partial_a\tilde L$, $\dot b=-\partial_b\tilde L$,

$$
\dot D=-2\big(a\partial_a-b\partial_b\big)\tilde L .
$$

The operator $a\partial_a-b\partial_b$ kills any function of $ab$, so it kills $L$ and $r^2$, and it maps $a^2+b^2\mapsto2a^2-2b^2=2D$. Hence $\dot D=-2\cdot\tfrac\eta4r^2\cdot2D=-\eta r^2D$. Over one step of time $\eta$ this multiplies $D$ by $\exp\big(-\eta\int_t^{t+\eta}r(s)^2ds\big)=1-\eta^2r(t)^2+O(\eta^3)$, since $r$ changes by $O(\eta)$ during the step. This matches item 2 through order $\eta^2$ per step, the order to which the IGR flow is supposed to be accurate locally; the $O(\eta^3)$ terms generally differ.

**4.** At $ab=1$ ($r=0$), $\nabla^2L=\begin{pmatrix}b^2&2ab-1\\2ab-1&a^2\end{pmatrix}=\begin{pmatrix}b^2&1\\1&a^2\end{pmatrix}$, with determinant $a^2b^2-1=0$ and trace $a^2+b^2$. The eigenvalues are $0$ (along the hyperbola) and $S=a^2+b^2$. With $ab=1$, $(a^2+b^2)^2=(a^2-b^2)^2+4a^2b^2=D^2+4$, so $S=\sqrt{D^2+4}$: the smaller the imbalance, the flatter the minimum, with $S=2$ at the balanced points $a=b=\pm1$. Gradient flow keeps $D=D_0$ and ends at $S=\sqrt{D_0^2+4}$. The IGR flow shrinks $|D|$ by the factor $\exp\big(-\eta\int_0^\infty r(t)^2dt\big)$, and GD by $\prod_k|1-\eta^2r_k^2|$, the same to leading order. So if (i) the run converges to the hyperbola, (ii) $D_0\ne0$ and the start is not already a minimum, and (iii) $\eta|r_k|<\sqrt2$ at every step, then GD and the IGR flow end at a smaller $S$ than gradient flow. (If $\eta|r_k|>\sqrt2$ a GD step can *increase* $|D|$; (i) generically needs $\eta<2/S$ at the endpoint, by §1.1. A balanced start stays balanced under all three dynamics.) The effect is $O(\eta)$ and accumulates only while $r\ne0$, so it is larger for larger steps and for starts far from the solution set. The same quantity reappears in deep linear networks, where imbalance between layers controls lazy versus rich dynamics (`td1_linear_networks`).
"""
    ),
    code(
        r"""
# GD vs gradient flow vs the IGR flow on L = (ab-1)^2/2, from (a, b) = (2, 0.1), where S ~ 4.1-4.5.
p0, T = np.array([2.0, 0.1]), 2.0
F_gf = lambda p: -grad_ab(p)
F_igr = lambda eta: (lambda p: -(grad_ab(p) + 0.5 * eta * hess_ab(p) @ grad_ab(p)))


def gd(p, eta, K):
    traj = [p]
    for _ in range(K):
        p = p - eta * grad_ab(p); traj.append(p)
    return np.array(traj)


# (1) global error at fixed time T against eta: expect slope 1 (GF) and slope 2 (IGR flow)
etas = T / np.array([20, 40, 80, 160, 320, 640])
err_gf, err_igr = [], []
for eta in etas:
    pK = gd(p0, eta, int(round(T / eta)))[-1]
    err_gf.append(np.linalg.norm(pK - rk4(F_gf, p0, T)))
    err_igr.append(np.linalg.norm(pK - rk4(F_igr(eta), p0, T)))
print(f"log-log slope of |GD - flow| at t = {T}: gradient flow {slope(etas, err_gf):.2f}, "
      f"IGR flow {slope(etas, err_igr):.2f}")

# (2) where the long-time runs end up, and how much of GD's shift IGR accounts for
T_long = 40.0
p_gf = rk4(F_gf, p0, T_long, 8000)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
S_of = lambda p: np.linalg.eigvalsh(hess_ab(p))[-1]
t_gf = np.linspace(0, 8, 80)
axs[0].plot(t_gf, [S_of(rk4(F_gf, p0, t, 400)) for t in t_gf], "k", label="gradient flow")
for eta, col in [(0.05, "C0"), (0.2, "C1"), (0.35, "C3")]:
    tr = gd(p0, eta, int(T_long / eta))
    p_igr = rk4(F_igr(eta), p0, T_long, 8000)
    shift = np.linalg.norm(tr[-1] - p_gf)
    captured = np.dot(p_igr - p_gf, tr[-1] - p_gf) / shift**2
    S_end = np.linalg.eigvalsh(hess_ab(tr[-1]))[-1]
    print(f"eta={eta:<5} eta*S_init={eta * np.linalg.eigvalsh(hess_ab(p0))[-1]:.2f}  "
          f"final S: GF {np.linalg.eigvalsh(hess_ab(p_gf))[-1]:.3f}, GD {S_end:.3f};  "
          f"IGR flow captures {captured:.0%} of GD's shift away from the GF endpoint")
    ts = np.arange(len(tr)) * eta
    keep = ts <= 8
    axs[0].plot(ts[keep], [S_of(p) for p in tr[keep]], color=col, label=f"GD, eta={eta}")
    axs[2].plot(ts, tr[:, 0]**2 - tr[:, 1]**2, color=col, label=f"GD, eta={eta}")
axs[0].set_xlabel("t = k eta"); axs[0].set_ylabel("sharpness S")
axs[0].legend(fontsize=7); axs[0].set_title("sharpness along the run")
axs[1].loglog(etas, err_gf, "o-", label="GD vs gradient flow")
axs[1].loglog(etas, err_igr, "s-", label="GD vs IGR flow")
axs[1].set_xlabel(r"$\eta$"); axs[1].set_ylabel(f"error at t={T}"); axs[1].legend(fontsize=8)
axs[1].set_title("global error vs step size")
axs[2].axhline(p0[0]**2 - p0[1]**2, color="k", label="gradient flow (conserved)")
axs[2].set_xlabel("t = k eta"); axs[2].set_ylabel(r"imbalance $a^2-b^2$"); axs[2].legend(fontsize=7)
axs[2].set_title("GD shrinks the imbalance")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The slopes come out at 1 and 2: the IGR flow tracks GD one order better than
gradient flow. The second printout is more sobering. The IGR flow accounts for most of
GD's displacement from the gradient-flow endpoint at $\eta S\approx0.2$, less than half
at $\eta S\approx0.8$, and a small fraction at $\eta S\approx1.4$. The first-order
correction gets the *direction* of the effect right (toward balance and flatness) well
before it gets the *size* right. §1.3 asks why.
"""
    ),
    # ------------------------------------------------------------------ 1.3
    markdown(
        r"""
<a id="td21-s3"></a>

## 1.3 Is the expansion valid?

Only as an asymptotic series. The modified equation of §1.2 is a power series in the
step size $\eta$ that usually diverges; truncations come with rigorous error bounds,
controlled roughly by $\eta S$ (step size times sharpness, the largest Hessian
eigenvalue). The kept first correction degrades as $\eta S$ approaches 1, and no
truncation sees the GD instability at $\eta S=2$, which is where full-batch training of
neural networks typically sits.
"""
        + reading(
            "Asymptotic expansions and backward analysis for numerical integrators",
            "https://archive-ouverte.unige.ch/unige:12439",
            "Hairer & Lubich (2000)",
            "A short open-access account, by two of the textbook's authors, of why modified equations diverge and what truncating them still guarantees, on exactly solvable examples.",
            importance=2,
            length=1,
        )
        + reading(
            "Gradient descent on neural networks typically occurs at the edge of stability",
            "https://arxiv.org/abs/2103.00065",
            "Cohen, Kaur, Li, Kolter & Talwalkar (ICLR 2021)",
            "Under full-batch GD the sharpness typically rises until $\\eta S\\approx2$ and stays there ($2(1+\\beta)$ with heavy-ball momentum), so full-batch training usually leaves the regime where the modified flow is accurate; exceptions are discussed.",
            importance=3,
            length=3,
        )
        + r"""
<a id="td21-s3-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* GD with step $\eta$ on $L$, gradient $g$, Hessian $H$, sharpness $S$. A
one-step method $y\mapsto y+hf(y)+O(h^2)$ has modified field
$\tilde f=f+hf_2+h^2f_3+\cdots$ (§1.2) and truncation $\tilde f_{[r]}=f+\dots+h^{r-1}f_r$.
For GD, $h=\eta$, $f=-g$, $f_2=-\tfrac12Hg$.

**It usually diverges.** Even for analytic $f$ the series typically has zero radius of
convergence; the quadratic of Exercise 1.2.c is the exception. For GD on $L=y^3/3$
(Exercise 1.3.a and the code below) the coefficients grow like $(k-1)!/(2\pi)^k$ over
70 computed orders.

**What a truncation guarantees.**

1. *Fixed order:* local error $O(h^{r+1})$, global error $O(h^r)$ over a fixed time, with constants involving derivatives of $f$ up to order about $r+1$. The global constant can grow like $e^{\ell T}$, with $\ell$ the rate at which nearby solutions separate: errors do not pile up near a minimum with positive definite Hessian, but do near saddles.
2. *Optimal truncation:* for analytic $f$, truncating at order $N\sim c/h$ makes the one-step error $O(e^{-\gamma/h})$ for some $\gamma>0$ ([Benettin–Giorgilli 1994](https://doi.org/10.1007/BF02188219), [Hairer–Lubich 1997](https://doi.org/10.1007/s002110050271), [Reich 1999](https://doi.org/10.1137/S0036142997329797); HLW §IX.7). Roughly, $\gamma$ is the radius of a complex neighbourhood of the trajectory on which $f$ is bounded, divided by that bound.

**What sets the scale.** On a quadratic the parameter is exactly $\eta\lambda$ per
eigendirection, so $\eta S$ near a minimum. In general higher derivatives enter too,
and small $\eta S$ is not sufficient: $L(x)=x+K^{-2}\sin(Kx)$ has $|L''|\le1$, yet at
$x=0$ the formulas of Exercise 1.2.b give $f_2=0$ and $f_3=\tfrac{K}{12}(1+\tfrac1K)^2$.
The real comparison is the step length $\eta\|g\|$ against the distance over which the
Hessian changes (here $1/K$).

**How it fails, most important first.**

1. *The kept correction stops being enough.* Its relative error is $O((\eta S)^2)$: the IGR flow accounts for 81%, 42% and 16% of GD's shift at $\eta S\approx0.2$, 0.8, 1.4 (code after Exercise 1.2.d).
2. *No gradient flow reproduces the step* once a quadratic multiplier $1-\eta\lambda$ is $\le0$ (Exercise 1.2.c). For $y^3/3$ the GD map stops being invertible where $\Phi_h'(y)=1-2hy=1-\eta S=0$, and the truncation errors in the code below deteriorate as $\eta S\to1$.
3. *Divergence.* Harmless in practice: optimally truncated, the series is extremely accurate when $\eta S$ is small (least term $e^{-4\pi/(\eta S)}$ for $y^3/3$, Exercise 1.3.b), and nobody uses high orders anyway.

**Where training sits.** Beyond $\eta S=2$ a minimum is unstable for GD (Exercise
1.1), a threshold invisible to every truncation (Exercise 1.2.c, item 3). Full-batch
training typically sits at $\eta S\approx2$ (Cohen et al.), where the expansion
parameter is about 2 in the sharpest direction but still small for a much flatter bulk;
`td2_2_edge_of_stability` treats the oscillating directions separately. Under minibatch
SGD the full-batch sharpness settles below $2/\eta$, while a batch-level curvature,
$\mathbb E_B\big[g_B^\top H_Bg_B/\|g_B\|^2\big]$ with $g_B$, $H_B$ the gradient and Hessian
of a random minibatch's loss, hovers near $2/\eta$ ([Andreyev and
Beneventano](https://arxiv.org/abs/2412.20553), v5).

**Exercise 1.3.a — A divergent modified equation.**

Take GD on $L(y)=y^3/3$ for $y>0$, i.e. explicit Euler for $\dot y=-y^2$:
$\Phi_h(y)=y-hy^2$.

1. Show that the exact modified field $F$ (if it exists) satisfies $F(\Phi_h(y))=\Phi_h'(y)\,F(y)$.
2. Write $F(y)=\sum_{k\ge2}a_kh^{k-2}y^k$ with $a_2=-1$, and set $u=hy$, so $F=h^{-2}\varphi(u)$ with $\varphi(u)=\sum_ka_ku^k$. Turn item 1 into an equation for $\varphi$, then into a recursion for the $a_k$. Compute $a_3,a_4,a_5$, and check $a_3$ against $f_2=-\tfrac12f'f$.

<details><summary>Hint</summary>

For 1: the flow $\varphi_t$ of $F$ commutes with itself, $\varphi_t\circ\varphi_h=\varphi_h\circ\varphi_t$. Differentiate in $t$ at $t=0$. For 2: the equation becomes $\varphi(u-u^2)=(1-2u)\varphi(u)$. Compare coefficients of $u^n$. The $a_n$ terms cancel, and $a_{n-1}$ is the highest unknown left.
</details>
"""
    ),
    answer(
        r"""
**1.** If $\Phi_h=\varphi_h$, the time-$h$ flow of $F$, then $\varphi_t\circ\Phi_h=\Phi_h\circ\varphi_t$ for all $t$ (flows of an autonomous field commute). Differentiate at $t=0$: the left side gives $F(\Phi_h(y))$, the right side $\Phi_h'(y)F(y)$ by the chain rule.

**2.** With $\Phi_h(y)=y-hy^2$ and $\Phi_h'(y)=1-2hy$, and $F(y)=h^{-2}\varphi(hy)$:

$$
h^{-2}\varphi(hy-h^2y^2)=(1-2hy)\,h^{-2}\varphi(hy)\iff\varphi(u-u^2)=(1-2u)\varphi(u).
$$

The coefficient of $u^n$ on the left is $\sum_ka_k\,\mathrm{coef}_{u^n}\big[(u-u^2)^k\big]=\sum_ka_k\binom{k}{n-k}(-1)^{n-k}$; on the right it is $a_n-2a_{n-1}$. The $k=n$ term cancels $a_n$; the $k=n-1$ term is $-(n-1)a_{n-1}$. So

$$
(n-3)\,a_{n-1}=\sum_{k\le n-2}(-1)^{n-k}\binom{k}{n-k}a_k .
$$

At $n=3$ both sides vanish, which leaves $a_2$ free, fixed by consistency to $a_2=-1$ ($F=-y^2+O(h)$). Then:

- $n=4$: $a_3=\binom22a_2=-1$;
- $n=5$: $2a_4=-\binom23a_2+\binom32a_3=0-3$, so $a_4=-\tfrac32$;
- $n=6$: $3a_5=\binom24a_2-\binom33a_3+\binom42a_4=0+1-9=-8$, so $a_5=-\tfrac83$.

So $\dot y=-y^2-hy^3-\tfrac32h^2y^4-\tfrac83h^3y^5-\cdots$. Check: $f=-y^2$, $f'=-2y$, so $f_2=-\tfrac12(-2y)(-y^2)=-y^3$, matching $a_3=-1$. Note that the invariance equation alone does not prove the modified field exists as a convergent function. It determines the coefficients of the formal series, and the code's 70 coefficients are consistent with a divergent series.
"""
    ),
    markdown(
        r"""
**Exercise 1.3.b — The least term.**

$F(y)=\sum_{k\ge2}a_kh^{k-2}y^k$ is the modified field of GD with step $h$ on
$L(y)=y^3/3$, with $a_k$ from Exercise 1.3.a. Suppose $|a_k|$ grows like
$(k-1)!/(2\pi)^k$ up to factors varying slower than exponentially in $k$ (the next code
cell supports this over $k\le70$; a fit, not a theorem). At a point $y$ the $k$-th term
of $F$ has size about $y^2|a_k|u^{k-2}$, with $u=hy$.

1. Find the order $k^*$ of the smallest term and its size, to exponential accuracy.
2. Translate into the sharpness $S=L''(y)=2y$. How small is the least term at $\eta S=0.1$, $0.5$, $0.8$? Why should the estimate not be trusted as $\eta S\to1$?

<details><summary>Hint</summary>

Stirling: $(k-1)!\approx\sqrt{2\pi/k}\,(k/e)^k$. Minimize $\log\big[(k/e)^k(u/2\pi)^k\big]$ over $k$. For the last question, look at $\Phi_h'(y)$.
</details>
"""
    ),
    answer(
        r"""
**1.** Up to powers of $k$ and constants, the term is $(k-1)!\,(u/2\pi)^k\approx(k/e)^k(u/2\pi)^k=\exp\big(k\log\tfrac{ku}{2\pi e}\big)$. The derivative of $k\log\tfrac{ku}{2\pi e}$ in $k$ is $\log\tfrac{ku}{2\pi e}+1=\log\tfrac{ku}{2\pi}$, which vanishes at

$$
k^*=\frac{2\pi}{u}=\frac{2\pi}{hy},\qquad\text{where the term is}\quad \exp\big(k^*\log e^{-1}\big)=e^{-2\pi/u}.
$$

Terms shrink until $k\approx k^*$ and grow afterwards. The rule of thumb for asymptotic series is that truncating near the least term leaves an error of about that size, here $e^{-2\pi/(hy)}$. That is a heuristic, conditional on the fitted growth law, and does not predict the prefactor. The rigorous statement is the general theorem quoted above, which guarantees $O(e^{-\gamma/h})$ for *some* $\gamma>0$ without identifying it.

**2.** With $h=\eta$ and $S=2y$: $e^{-2\pi/(\eta y)}=e^{-4\pi/(\eta S)}$. At $\eta S=0.1$ this is $e^{-126}\approx10^{-55}$, irrelevant. At $\eta S=0.5$ it is $e^{-25}\approx10^{-11}$, and at $\eta S=0.8$ it is $e^{-15.7}\approx1.5\times10^{-7}$. But the useful truncation order $k^*\approx4\pi/(\eta S)$ is about 16 at $\eta S=0.8$; in practice one keeps only the first correction.

As $\eta S\to1$ the estimate stops meaning anything (failure 2 above): $\Phi_h'(y)=1-2hy=1-\eta S$ goes to zero, so the GD step is not invertible near $y$ and cannot be the time-$h$ map of any flow (flow maps are invertible). The code shows the best error at $\eta S=0.9$ already two orders of magnitude above the least-term estimate.
"""
    ),
    code(
        r"""
# Modified-equation coefficients for GD on L = y^3/3 (Euler for ydot = -y^2), from the
# recursion of Exercise 1.3.a in exact rational arithmetic.
N = 70
a = {2: Fraction(-1)}
for n in range(4, N + 2):
    s = sum(a[k] * comb(k, n - k) * (-1) ** (n - k) for k in a if k <= n - 2 and n - k <= k)
    a[n - 1] = s / (n - 3)
print("a_2..a_7 =", [str(a[k]) for k in range(2, 8)])
growth = {k: (np.log(abs(float(a[k]))) - np.sum(np.log(np.arange(1, k)))) / k for k in (30, 50, 70)}
print("(log|a_k| - log (k-1)!)/k at k = 30, 50, 70:", [round(float(v), 3) for v in growth.values()],
      f"-> drifting toward -log(2 pi) = {-np.log(2*np.pi):.3f} (a numerical fit)")

# One-step error of the truncated modified flow vs the GD step, as a function of the order n.
# A 1D flow moving down from y0 reaches y after time T(y) = int_y^y0 dz / (-F_n(z)); we find the
# endpoint T(y) = h by bisection (Gauss-Legendre for T). Orders whose field points the wrong way
# at y0 never reach the GD point at all; they are left blank (nan). The flow cannot cross an
# equilibrium of F_n, so the search stops above the first one.
af = np.array([float(a[k]) for k in range(2, N + 1)])
orders, pw = np.arange(1, 50), np.arange(N + 1)
xg, wg = np.polynomial.legendre.leggauss(120)


def truncation_errors(h, y0=1.0):
    y1 = y0 - h * y0**2                              # one GD step
    grid = np.linspace(y0, y1 - 0.5 * h, 4001)       # search window, descending from y0
    out = np.full(len(orders), np.nan)
    for i, n in enumerate(orders):
        c = np.zeros(N + 1)
        c[2:n + 2] = af[:n] * h ** np.arange(n)      # F_n(y) = sum_k a_k h^(k-2) y^k
        F = lambda y: (np.asarray(y)[..., None] ** pw) @ c
        bad = np.nonzero(F(grid) >= 0)[0]
        if len(bad) and bad[0] == 0:
            continue                                 # wrong direction at y0
        lo = grid[bad[0] - 1] if len(bad) else grid[-1]
        T = lambda y: 0.5 * (y0 - y) * np.sum(wg / -F(0.5 * (y0 + y) + 0.5 * (y0 - y) * xg))
        if T(lo) < h:
            continue
        top, bot = y0, lo
        for _ in range(60):
            mid = 0.5 * (top + bot)
            top, bot = (top, mid) if T(mid) > h else (mid, bot)
        out[i] = abs(0.5 * (top + bot) - y1)
    return out


fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].semilogy(np.arange(2, N + 1), np.abs(af), label=r"$|a_k|$")
axs[0].semilogy(np.arange(2, N + 1), [np.exp(np.sum(np.log(np.arange(1, k))) - k * np.log(2 * np.pi))
                                       for k in range(2, N + 1)], "--", label=r"$(k-1)!/(2\pi)^k$")
axs[0].set_xlabel("k"); axs[0].set_title("modified-equation coefficients"); axs[0].legend()
for h in [0.25, 0.3, 0.35, 0.4, 0.45]:
    e = truncation_errors(h)
    axs[1].semilogy(orders, e, label=f"eta*S = {2*h:.1f}")
    print(f"eta*S={2*h:.1f}: best one-step error {np.nanmin(e):.1e} at n={orders[np.nanargmin(e)]};"
          f"  exp(-2 pi/h) = {np.exp(-2*np.pi/h):.1e};  {np.isnan(e).sum()} orders never reach the GD point")
axs[1].set_xlabel("truncation order n"); axs[1].set_ylabel("one-step error")
axs[1].set_title("truncated modified flow vs one GD step"); axs[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The error first falls with the truncation order, then stops improving and grows;
gaps are orders whose truncated field points the wrong way. The step starts at
$y_0=1$, where $S=2$, so the least-term estimate of Exercise 1.3.b is $e^{-2\pi/h}$. Up
to $\eta S=0.8$ the best error lands within a factor of a few of it: consistent with
the exponent, not a confirmation of the prefactor (the left panel shows how far the
coefficients sit from the fitted curve). At $\eta S=0.9$ the best error is far above
the estimate, as the step approaches the point where it is no longer invertible.
"""
    ),
    # ------------------------------------------------------------------ 1.4
    markdown(
        r"""
<a id="td21-s4"></a>

## 1.4 The SGD version: shuffled minibatches

SGD with randomly shuffled minibatches, averaged over the order of the batches,
follows to second order in the step size $\eta$ a gradient flow on

$$
\mathbb E\,\tilde L_{\rm SGD}=L+\frac\eta4\|\nabla L\|^2+\frac{N-B}{N-1}\,\frac{\eta}{4B}\,\mathrm{tr}\,\Sigma ,
$$

with $N$ examples, batch size $B$, and $\Sigma$ the covariance of per-example
gradients. The first correction is GD's implicit gradient regularization (§1.2) and
depends on $\eta$ alone; the second penalizes the spread of per-example gradients and
scales like $\eta/B$.

**Paper:** Smith, Dherin, Barrett & De, [On the origin of implicit regularization in SGD](https://arxiv.org/abs/2101.12176)
(card at the top).

**Core idea.** To first order in $\eta$, one epoch of SGD equals one GD step of size
$m\eta$ ($m=N/B$ batches): with each example used once per epoch, the gradient noise
cancels at $O(\eta)$ and first shows up at $O(\eta^2)$, as the bias above. The result
is about the *mean* iterate over batch orders, and it needs the epoch length $m\eta$
times the *minibatch* curvatures to be small, a much stronger condition than for GD.

<a id="td21-s4-walkthrough"></a>

### Optional: annotated derivations and walkthrough

**Setup.** $N$ examples with losses $\ell_i$, $L=\frac1N\sum_i\ell_i$, split into
$m=N/B$ minibatches of size $B$ with losses $\hat L_k=\frac1B\sum_{i\in\text{batch }k}\ell_i$,
so $L=\frac1m\sum_k\hat L_k$. One epoch of SGD with *random shuffling* visits each
batch once, in a random order $\pi$: $\theta_{j+1}=\theta_j-\eta\nabla\hat L_{\pi(j)}(\theta_j)$.
Per-example gradient covariance:
$\Sigma=\frac1N\sum_i(\nabla\ell_i-\nabla L)(\nabla\ell_i-\nabla L)^\top$.

Smith et al. apply backward error analysis to a whole epoch. Averaged over batch orders
(batches held fixed), the iterate after one epoch follows, up to $O(m^3\eta^3)$,
gradient flow for time $m\eta$ on

$$
\tilde L_{\rm SGD}=L+\frac{\eta}{4m}\sum_{k=0}^{m-1}\|\nabla\hat L_k\|^2
=L+\frac\eta4\|\nabla L\|^2+\frac{\eta}{4m}\sum_k\|\nabla\hat L_k-\nabla L\|^2 .
$$

Averaging also over the random split into batches gives $\mathbb E\,\tilde L_{\rm SGD}$
above (Exercise 1.4.b). Individual epochs differ from the mean at order $m^2\eta^2$,
through the order-dependent term in Exercise 1.4.a.

**Exercise 1.4.a — One epoch.**

Write $\hat g_k=\nabla\hat L_k(\theta_0)$ and $\hat H_k=\nabla^2\hat L_k(\theta_0)$ for the
gradient and Hessian of batch $k$ at the start of the epoch, so $\sum_k\hat g_k=mg$ and
$\sum_k\hat H_k=mH$, where $g$ and $H$ are the full-batch gradient and Hessian at
$\theta_0$. Here $k$ labels batches and $j$ labels steps within the epoch.

1. For a fixed order (batches $0,1,\dots,m-1$), show $\theta_m=\theta_0-\eta\sum_k\hat g_k+\eta^2\sum_{j}\sum_{k<j}\hat H_j\hat g_k+O(m^3\eta^3)$.
2. Average over all $m!$ orders. Show $\mathbb E\,\theta_m=\theta_0-m\eta g+\tfrac{\eta^2}2\big(m^2Hg-\sum_k\hat H_k\hat g_k\big)+O(m^3\eta^3)$.
3. Match against the flow $\dot\theta=-g+\eta f_2$ run for time $m\eta$ (the §1.2 notation: $f_2$ is the $O(\eta)$ correction), and conclude $\tilde L_{\rm SGD}=L+\frac{\eta}{4m}\sum_k\|\nabla\hat L_k\|^2$. Then derive the second form of $\tilde L_{\rm SGD}$.

<details><summary>Hint</summary>

For 1, $\theta_j=\theta_0-\eta\sum_{k<j}\hat g_k+O(\eta^2)$, and $\nabla\hat L_j(\theta_j)=\hat g_j+\hat H_j(\theta_j-\theta_0)+O(\eta^2)$. For 2, in a uniformly random order each ordered pair $j\ne k$ has $k$ before $j$ with probability $\tfrac12$. For 3, the flow's Taylor expansion over time $\tau=m\eta$ is $\theta_0+\tau F+\tfrac{\tau^2}2F'F+\dots$ (from §1.2), and $\sum_k\hat H_k\hat g_k=\tfrac12\nabla\sum_k\|\hat g_k\|^2$.
</details>
"""
    ),
    answer(
        r"""
**1.** By induction $\theta_j-\theta_0=-\eta\sum_{k<j}\hat g_k+O(j^2\eta^2)$. Taylor-expanding each gradient around $\theta_0$,

$$
\nabla\hat L_j(\theta_j)=\hat g_j+\hat H_j(\theta_j-\theta_0)+O(\eta^2)=\hat g_j-\eta\sum_{k<j}\hat H_j\hat g_k+O(\eta^2),
$$

so $\theta_m=\theta_0-\eta\sum_j\nabla\hat L_j(\theta_j)=\theta_0-\eta\sum_j\hat g_j+\eta^2\sum_j\sum_{k<j}\hat H_j\hat g_k+O(m^3\eta^3)$. The error is $O(m^3\eta^3)$ because there are $O(m^3)$ third-order terms.

**2.** The first-order term $-\eta\sum_j\hat g_j=-m\eta g$ does not depend on the order. In the double sum, a uniformly random order puts $k$ before $j$ with probability $\tfrac12$ for each ordered pair $j\ne k$, so

$$
\mathbb E\sum_j\sum_{k\text{ before }j}\hat H_j\hat g_k=\tfrac12\sum_j\sum_{k\ne j}\hat H_j\hat g_k
=\tfrac12\Big(\sum_j\hat H_j\sum_k\hat g_k-\sum_j\hat H_j\hat g_j\Big)=\tfrac12\Big(m^2Hg-\sum_j\hat H_j\hat g_j\Big).
$$

**3.** Run $\dot\theta=F=-g+\eta f_2$ for time $\tau=m\eta$. With $F'F=(-H)(-g)+O(\eta)=Hg+O(\eta)$,

$$
\theta(m\eta)=\theta_0-m\eta g+m\eta^2f_2+\tfrac{m^2\eta^2}{2}Hg+O(m^3\eta^3).
$$

Equating with item 2, the $m^2Hg$ terms cancel and $m\eta^2f_2=-\tfrac{\eta^2}2\sum_k\hat H_k\hat g_k$, so

$$
f_2=-\frac1{2m}\sum_k\hat H_k\hat g_k=-\nabla\Big(\frac1{4m}\sum_k\|\nabla\hat L_k\|^2\Big),
$$

which gives $\tilde L_{\rm SGD}=L+\frac\eta{4m}\sum_k\|\nabla\hat L_k\|^2$. For the second form, write $\nabla\hat L_k=\nabla L+\delta_k$ with $\sum_k\delta_k=0$. Then $\sum_k\|\nabla L+\delta_k\|^2=m\|\nabla L\|^2+2\nabla L\cdot\sum_k\delta_k+\sum_k\|\delta_k\|^2=m\|\nabla L\|^2+\sum_k\|\delta_k\|^2$. With $m=1$ ($B=N$) the extra term vanishes and GD's IGR is recovered.
"""
    ),
    markdown(
        r"""
**Exercise 1.4.b — The $\eta/B$ term.**

1. A batch is a uniformly random subset of size $B$, drawn without replacement. Show $\mathbb E\|\nabla\hat L-\nabla L\|^2=\frac{N-B}{N-1}\frac{\mathrm{tr}\,\Sigma}{B}$, and conclude the formula for $\mathbb E\,\tilde L_{\rm SGD}$.
2. At fixed $\eta$, what does halving $B$ do to each correction term? Which term does the linear scaling rule ($\eta\propto B$) hold fixed, and which does it change?

<details><summary>Hint</summary>

Let $\delta_i=\nabla\ell_i-\nabla L$, so $\sum_i\delta_i=0$. Expand $\|\tfrac1B\sum_{i\in\text{batch}}\delta_i\|^2$ as a double sum and use $\Pr(i\in\text{batch})=B/N$ and $\Pr(i,j\in\text{batch})=\frac{B(B-1)}{N(N-1)}$ for $i\ne j$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $\delta_i$ as in the hint, $\nabla\hat L-\nabla L=\frac1B\sum_{i\in\text{batch}}\delta_i$, so

$$
\mathbb E\Big\|\frac1B\sum_{i\in\text{batch}}\delta_i\Big\|^2
=\frac1{B^2}\Big[\frac BN\sum_i\|\delta_i\|^2+\frac{B(B-1)}{N(N-1)}\sum_{i\ne j}\delta_i\cdot\delta_j\Big].
$$

Since $\sum_i\delta_i=0$, $\sum_{i\ne j}\delta_i\cdot\delta_j=\|\sum_i\delta_i\|^2-\sum_i\|\delta_i\|^2=-\sum_i\|\delta_i\|^2$. Using $\sum_i\|\delta_i\|^2=N\,\mathrm{tr}\,\Sigma$,

$$
\mathbb E\|\nabla\hat L-\nabla L\|^2=\frac{N\,\mathrm{tr}\,\Sigma}{B^2}\cdot\frac BN\Big(1-\frac{B-1}{N-1}\Big)=\frac{N-B}{N-1}\,\frac{\mathrm{tr}\,\Sigma}{B}.
$$

Each of the $m$ batches in a random partition is marginally such a uniform subset, so $\mathbb E\frac1m\sum_k\|\nabla\hat L_k-\nabla L\|^2$ takes the same value, which gives $\mathbb E\,\tilde L_{\rm SGD}$. The factor $\frac{N-B}{N-1}$ is the finite-population correction: the term vanishes at $B=N$.

**2.** The IGR term $\frac\eta4\|\nabla L\|^2$ does not change. The noise term $\frac{N-B}{N-1}\frac{\eta}{4B}\mathrm{tr}\,\Sigma$ is multiplied by $2\frac{N-B/2}{N-B}$: a bit more than 2 when $B\ll N$, and much more as $B$ approaches $N$ (where the term vanishes). Smaller batches strengthen the implicit penalty on gradient spread. Under linear scaling ($\eta\to2\eta$, $B\to2B$, $B\ll N$) the noise term is unchanged but the IGR term doubles. So linear scaling preserves the modified loss only when the noise term dominates, $\mathrm{tr}\,\Sigma/B\gg\|\nabla L\|^2$, the regime Smith et al. assume. `td2_3_sgd_sde` meets the same $\eta$ versus $\eta/B$ split in the stochastic modified equation.
"""
    ),
    markdown(
        r"""
**Exercise 1.4.c — Check it numerically (code).**

The cell below uses $N=8$ examples with loss $\ell_i=\log\cosh(x_i^\top w-y_i)$,
$w\in\mathbb R^2$, and $B=2$ ($m=4$ batches, $24$ orders). Complete `mean_epoch` (the
end-of-epoch iterate averaged *exactly* over all orders) and the `"sgd"` branch of
`flow_end` (the field $-\nabla\tilde L_{\rm SGD}$). The cell compares the mean iterate
with gradient flow, the IGR flow and the flow on $\tilde L_{\rm SGD}$, each run for time
$m\eta$. This one-epoch time shrinks with $\eta$, so each error is one power of $\eta$
higher than in the fixed-time §1.2 cell. Why does the IGR flow not improve the order of
the error, while the flow on $\tilde L_{\rm SGD}$ does?
"""
    ),
    code_answer(
        r"""
rng = np.random.default_rng(0)
Nex, B = 8, 2
m = Nex // B
X, yv = rng.normal(size=(Nex, 2)), rng.normal(size=Nex)
batches = [np.arange(k * B, (k + 1) * B) for k in range(m)]
everyone = np.arange(Nex)


def g_set(w, idx):        # gradient of the mean of log cosh(x_i.w - y_i) over examples idx
    r = X[idx] @ w - yv[idx]
    return (np.tanh(r)[:, None] * X[idx]).mean(0)


def H_set(w, idx):        # Hessian of the same
    r = X[idx] @ w - yv[idx]
    s = 1 / np.cosh(r) ** 2
    return (s[:, None, None] * X[idx][:, :, None] * X[idx][:, None, :]).mean(0)


def mean_epoch(w0, eta):
    # TODO: average over all m! orders (itertools.permutations) of one epoch of SGD:
    # for each order, take m steps w <- w - eta * g_set(w, batches[k]); return the mean final w.
    raise NotImplementedError


def flow_end(w0, eta, kind):
    def F(w):
        g = g_set(w, everyone)
        if kind == "gf":
            return -g
        if kind == "igr":
            return -(g + 0.5 * eta * H_set(w, everyone) @ g)
        # TODO (kind == "sgd"): return -grad of L_SGD = L + (eta / 4m) sum_k |grad L_k|^2,
        # using g_set(w, b) and H_set(w, b) for each batch b.
        raise NotImplementedError
    return rk4(F, w0, m * eta, 400)


w0 = np.array([1.5, -1.0])
etas_sgd = np.array([0.1, 0.05, 0.025, 0.0125])
# errs = {kind: [np.linalg.norm(mean_epoch(w0, eta) - flow_end(w0, eta, kind)) for eta in etas_sgd]
#         for kind in ["gf", "igr", "sgd"]}
# for kind, e in errs.items(): print(kind, f"slope {slope(etas_sgd, e):.2f}")
""",
        r"""
rng = np.random.default_rng(0)
Nex, B = 8, 2
m = Nex // B
X, yv = rng.normal(size=(Nex, 2)), rng.normal(size=Nex)
batches = [np.arange(k * B, (k + 1) * B) for k in range(m)]
everyone = np.arange(Nex)


def g_set(w, idx):        # gradient of the mean of log cosh(x_i.w - y_i) over examples idx
    r = X[idx] @ w - yv[idx]
    return (np.tanh(r)[:, None] * X[idx]).mean(0)


def H_set(w, idx):        # Hessian of the same
    r = X[idx] @ w - yv[idx]
    s = 1 / np.cosh(r) ** 2
    return (s[:, None, None] * X[idx][:, :, None] * X[idx][:, None, :]).mean(0)


def mean_epoch(w0, eta):
    finals = []
    for order in permutations(range(m)):
        w = w0.copy()
        for k in order:
            w = w - eta * g_set(w, batches[k])
        finals.append(w)
    return np.mean(finals, axis=0)


def flow_end(w0, eta, kind):
    def F(w):
        g = g_set(w, everyone)
        if kind == "gf":
            return -g
        if kind == "igr":
            return -(g + 0.5 * eta * H_set(w, everyone) @ g)
        # grad of (eta/4m) sum_k |g_k|^2 is (eta/2m) sum_k H_k g_k
        return -(g + eta / (2 * m) * sum(H_set(w, b) @ g_set(w, b) for b in batches))
    return rk4(F, w0, m * eta, 400)


w0 = np.array([1.5, -1.0])
etas_sgd = np.array([0.1, 0.05, 0.025, 0.0125])
errs = {kind: [np.linalg.norm(mean_epoch(w0, eta) - flow_end(w0, eta, kind)) for eta in etas_sgd]
        for kind in ["gf", "igr", "sgd"]}
names = {"gf": "gradient flow on L", "igr": "IGR flow (GD's modified loss)", "sgd": "flow on L_SGD"}
for kind, e in errs.items():
    print(f"{names[kind]:<32} slope {slope(etas_sgd, e):.2f}   errors {np.array2string(np.array(e), precision=1)}")
"""
    ),
    answer(
        r"""
The slopes come out at about 2, 2 and 3. The mean iterate differs from gradient flow at $O(\eta^2)$ through two terms: the full-batch IGR term and the batch-spread term $\frac{\eta}{4m}\sum_k\|\nabla\hat L_k-\nabla L\|^2$. The IGR flow fixes only the first, so an $O(\eta^2)$ error remains, at all but the largest $\eta$ even larger than gradient flow's. Gradient flow's residual is $-\tfrac{\eta^2}2\sum_k\hat H_k\hat g_k=-\tfrac{\eta^2}2\big(mHg+V\big)$ with $V=\tfrac12\nabla\sum_k\|\nabla\hat L_k-\nabla L\|^2$. The IGR flow removes the $mHg$ part and leaves $-\tfrac{\eta^2}2V$. Here $mHg$ and $V$ partly cancel, so correcting only one makes the residual bigger. The flow on $\tilde L_{\rm SGD}$ fixes both, leaving $O(\eta^3)$. (At the largest $\eta$ the IGR error sits off the asymptotic line, so its fitted slope reads somewhat below 2.)
"""
    ),
    # ------------------------------------------------------------------ 1.5
    markdown(
        r"""
<a id="td21-s5"></a>

## 1.5 Momentum

Heavy-ball momentum, $\theta_{k+1}=\theta_k-\eta\nabla L(\theta_k)+\beta(\theta_k-\theta_{k-1})$
with $\beta\in[0,1)$, does three things relative to GD with step $\eta$. On a quadratic
it stays stable up to $\eta S<2(1+\beta)$ ($S$ the sharpness). For small steps it
follows gradient flow sped up by $1/(1-\beta)$: an *effective learning rate*
$\eta/(1-\beta)$. And at matched effective learning rate its implicit gradient penalty
is $(1+\beta)/(1-\beta)$ times GD's (19 times at $\beta=0.9$), though only for
curvatures far smaller than those where GD's applies.
"""
        + reading(
            "Why momentum really works",
            "https://distill.pub/2017/momentum/",
            "Goh (Distill, 2017)",
            "Interactive: heavy ball on a convex quadratic, eigendirection by eigendirection, with the stability region, critical damping, the optimal tuning and the $\\sqrt\\kappa$ speedup that this section skips.",
            importance=4,
            length=2,
        )
        + reading(
            "Implicit regularization in heavy-ball momentum accelerated stochastic gradient descent",
            "https://arxiv.org/abs/2302.00849",
            "Ghosh, Lyu, Zhang & Wang (ICLR 2023)",
            "The heavy-ball modified flow with an $O(\\eta^2)$ bound for the method started from rest (their Theorem 4.1, with explicit $\\beta^k$ transient terms), and the SGD-with-momentum version, whose extra noise term mirrors §1.4.",
            importance=2,
            length=3,
        )
        + r"""
<a id="td21-s5-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* loss $L$, gradient $g=\nabla L$, Hessian $H$, sharpness $S=\lambda_{\max}(H)$,
step $\eta$, momentum $\beta\in[0,1)$, time $t=k\eta$. Recall from §1.1–1.2: for GD
($\beta=0$) a quadratic direction of curvature $\lambda$ is stable iff
$0<\eta\lambda<2$, and to first order GD follows gradient flow on
$L+\tfrac\eta4\|\nabla L\|^2$ (IGR).

Heavy ball ([Polyak 1964](https://doi.org/10.1016/0041-5553%2864%2990137-5)) keeps a velocity $v$:

$$
v_{k+1}=\beta v_k-\eta\,g(\theta_k),\qquad \theta_{k+1}=\theta_k+v_{k+1}.
$$

PyTorch's SGD with momentum ($b\leftarrow\beta b+g$, $\theta\leftarrow\theta-\eta b$) and
Goh's $z\leftarrow\beta z+\nabla f$, $w\leftarrow w-\alpha z$ are the same method
($v=-\eta b$, $\alpha=\eta$); Adam-style averaging, $m\leftarrow\beta m+(1-\beta)g$,
is the same with $\eta\to\eta(1-\beta)$.

**On a quadratic** (Exercise 1.5.a and the code after it). The characteristic roots
of $\theta_{k+1}=(1+\beta-\eta\lambda)\theta_k-\beta\theta_{k-1}$ give the stability
region $0<\eta\lambda<2(1+\beta)$, a spiral-in band with rate $\sqrt\beta$, and at
critical damping a rate $1-\sqrt{\eta\lambda}$ instead of GD's $1-\eta\lambda$: that
square root is the acceleration (Goh derives the optimal tuning).

**The flow limit** (Exercise 1.5.b). A slowly varying velocity gives
$v\approx-\eta g/(1-\beta)$, i.e. gradient flow sped up by $1/(1-\beta)$; at matched
effective learning rate momentum tolerates $(1+\beta)/(1-\beta)$ times sharper
directions. The next Taylor term gives a damped particle with mass $O(\eta)$
([Kovachki and Stuart](https://www.jmlr.org/papers/v22/19-466.html)), a good picture
of the start-up transient but globally only $O(\eta)$ accurate.

**The slow manifold** (Exercise 1.5.c). Heavy ball maps pairs $(\theta,v)$, so §1.2's
recipe does not apply directly. But a mismatch in $v$ decays like $\beta^k$, and after a
few times $1/(1-\beta)$ steps the velocity is *slaved* to the position. On this
attracting invariant "slow manifold" (Kovachki and Stuart prove it exists) the step is
a function of the current point, $\theta_{k+1}-\theta_k=D(\theta_k)$, and §1.2 applies:
a map $\theta\mapsto\theta+hf+h^2d_2+O(h^3)$ is followed, to $O(h^2)$ on fixed time
intervals, by $\dot\theta=f+hf_2$ with $f_2=d_2-\tfrac12f'f$ ($f'$ the Jacobian). The
result (Ghosh et al.) is

$$
\dot\theta=-\frac1{1-\beta}\nabla\Big(L+\frac{\eta(1+\beta)}{4(1-\beta)^2}\|\nabla L\|^2\Big),
$$

i.e. gradient flow on $L+\frac{\eta_{\rm eff}}4\cdot\frac{1+\beta}{1-\beta}\|\nabla L\|^2$
in units of the effective step $\eta_{\rm eff}=\eta/(1-\beta)$.

**Range of validity.** The velocity must relax much faster than the position drifts.
On a direction of curvature $\lambda$ the slow root $z\approx1-\eta\lambda/(1-\beta)$
and the fast root near $\beta$ meet at $\eta\lambda=(1-\sqrt\beta)^2$, so the expansion
needs $\eta\lambda\ll(1-\sqrt\beta)^2$, i.e.
$\eta_{\rm eff}\lambda\ll(1-\sqrt\beta)/(1+\sqrt\beta)\approx0.026$ at $\beta=0.9$,
against $\eta\lambda\ll1$ for GD. The 19-fold penalty applies to curvatures about 40
times smaller than GD's IGR does. *Starting from rest* ($v_0=0$) the velocity's initial
deficit, $\eta g/(1-\beta)$, decays like $\beta^k$ but leaves an $O(\eta)$ displacement
of about $\eta\beta g/(1-\beta)^2$, so the formula is then only $O(\eta)$ accurate;
Ghosh et al.'s theorem, which starts from rest, recovers $O(\eta^2)$ by keeping
explicit $\beta^k$ terms. The last code cell below shows both cases.

**Exercise 1.5.a — Roots and regimes.**

1. Derive the characteristic polynomial and show the stability region is $0<\eta\lambda<2(1+\beta)$ for $0\le\beta<1$.
2. Show that the roots are complex exactly for $(1-\sqrt\beta)^2<\eta\lambda<(1+\sqrt\beta)^2$, with $|z|=\sqrt\beta$ there.
3. For small $\eta\lambda$, find the root closest to 1 to first order in $\eta\lambda$, and read off the effective learning rate. For $0<\eta\lambda\le1$, at critical damping $\beta=(1-\sqrt{\eta\lambda})^2$, find the double root.

<details><summary>Hint</summary>

Write $p(z)=z^2-\tau z+\beta$ with $\tau=1+\beta-\eta\lambda$. For real roots, both lie in $(-1,1)$ iff $p(1)>0$, $p(-1)>0$ and the product $\beta$ has modulus below 1 (why?). For complex roots, $|z|^2=z\bar z=z_1z_2$. For 3, set $z=1-\epsilon$ in $p$ and keep first-order terms.
</details>
"""
    ),
    answer(
        r"""
**1.** Substituting $g=\lambda\theta$ into $\theta_{k+1}=\theta_k-\eta g+\beta(\theta_k-\theta_{k-1})$ gives $\theta_{k+1}=\tau\theta_k-\beta\theta_{k-1}$ with $\tau=1+\beta-\eta\lambda$; $\theta_k=z^k$ gives $p(z)=z^2-\tau z+\beta=0$. The general solution is $c_1z_1^k+c_2z_2^k$ (or $(c_1+c_2k)z^k$ for a double root), so convergence for all starts means $|z_1|,|z_2|<1$.

*Complex roots* ($\tau^2<4\beta$): the roots are conjugate, so $|z|^2=z_1z_2=\beta$ and $|z|=\sqrt\beta<1$. Convergent.

*Real roots* ($\tau^2\ge4\beta$): $p$ is an upward parabola. If both roots are in $(-1,1)$ then $p(\pm1)>0$. Conversely, if $p(1)>0$ and $p(-1)>0$, then $\pm1$ lie on the same side of both roots: either both roots are in $(-1,1)$, or both are $>1$, or both are $<-1$. The last two give $|z_1z_2|>1$, impossible since $z_1z_2=\beta<1$. Now $p(1)=1-\tau+\beta=\eta\lambda$ and $p(-1)=1+\tau+\beta=2(1+\beta)-\eta\lambda$. So the real-root case is stable iff $0<\eta\lambda<2(1+\beta)$. The complex band sits inside this interval (item 2), so the whole stability region is $0<\eta\lambda<2(1+\beta)$. At $\eta\lambda=2(1+\beta)$ a root reaches $-1$, the period-2 instability, exactly as at $\eta\lambda=2$ for GD ($\beta=0$).

**2.** Complex roots iff $\tau^2<4\beta$, i.e. $|1+\beta-\eta\lambda|<2\sqrt\beta$, i.e. $1+\beta-2\sqrt\beta<\eta\lambda<1+\beta+2\sqrt\beta$, which is $(1-\sqrt\beta)^2<\eta\lambda<(1+\sqrt\beta)^2$. There $|z|=\sqrt\beta$ by item 1, which the code below confirms. (Goh's text and figures use the same rate, but his displayed modulus expression drops the $\tfrac12$ of the root formula and evaluates to $2\sqrt\beta$.)

**3.** Put $z=1-\epsilon$: $p(1-\epsilon)=(1-\epsilon)^2-\tau(1-\epsilon)+\beta=(1-\tau+\beta)-\epsilon(2-\tau)+\epsilon^2=\eta\lambda-\epsilon(1-\beta+\eta\lambda)+\epsilon^2$. To first order, $\epsilon=\eta\lambda/(1-\beta)$. So the slow root is $z\approx1-\eta\lambda/(1-\beta)$, the GD multiplier with $\eta$ replaced by $\eta/(1-\beta)$: the effective learning rate. For $0<\eta\lambda\le1$, at critical damping $\tau=2\sqrt\beta$ (the lower edge of the complex band), the double root is $z=\tau/2=\sqrt\beta=1-\sqrt{\eta\lambda}$. For small $\eta\lambda$ that is far better than GD's $1-\eta\lambda$, and better than the overdamped slow root.
"""
    ),
    code(
        r"""
# Spectral radius max|z| of heavy ball on a quadratic over the (eta*lambda, beta) plane.
el = np.linspace(0.0, 4.4, 441)            # eta * lambda
be = np.linspace(0.0, 0.99, 199)           # beta
EL, BE = np.meshgrid(el, be)
tau = 1 + BE - EL
disc = np.sqrt((tau**2 - 4 * BE).astype(complex))
rho = np.maximum(np.abs((tau + disc) / 2), np.abs((tau - disc) / 2))

fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.pcolormesh(EL, BE, np.ma.masked_where(rho >= 1, rho), shading="auto", cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im, ax=ax, label=r"spectral radius $\max|z|$ (blank: unstable)")
ax.plot(2 * (1 + be), be, "r--", lw=1.2, label=r"$\eta\lambda=2(1+\beta)$")
ax.plot((1 - np.sqrt(be))**2, be, "w:", lw=1.2, label=r"$(1\mp\sqrt{\beta})^2$: complex band")
ax.plot((1 + np.sqrt(be))**2, be, "w:", lw=1.2)
ax.set_xlim(0, 4.4); ax.set_xlabel(r"$\eta\lambda$"); ax.set_ylabel(r"$\beta$")
ax.set_title("heavy ball on a quadratic: rate where stable")
ax.legend(fontsize=8, loc="upper left"); plt.tight_layout(); plt.show()

band = (EL > (1 - np.sqrt(BE))**2 + 1e-9) & (EL < (1 + np.sqrt(BE))**2 - 1e-9)
print("max |rho - sqrt(beta)| in the complex band:", np.abs(rho - np.sqrt(BE))[band].max())
stable = rho < 1
pred = (EL > 0) & (EL < 2 * (1 + BE))
print("grid points where the stability region disagrees with 0 < eta*lam < 2(1+beta):",
      int((stable != pred)[(np.abs(EL - 2 * (1 + BE)) > 0.02) & (EL > 0.02)].sum()),
      "(excluding a thin strip around the boundary, where rounding decides)")
# Critical damping: at beta = (1 - sqrt(eta*lam))^2 the rate is 1 - sqrt(eta*lam), vs GD's 1 - eta*lam.
for e in [0.01, 0.1]:
    b = (1 - np.sqrt(e))**2
    zs = np.roots([1, -(1 + b - e), b])
    print(f"eta*lam={e}: critical beta={b:.3f}, rate {np.abs(zs).max():.4f}; GD rate {1 - e:.4f}")
"""
    ),
    markdown(
        r"""
**Exercise 1.5.b — The flow limit.**

1. Show that if the iterates follow a smooth curve $\theta(t)$, $t=k\eta$, then to first order $\dot\theta=-\nabla L/(1-\beta)$.
2. Keep the next order of the Taylor expansion and derive $\tfrac{\eta(1+\beta)}2\ddot\theta+(1-\beta)\dot\theta=-\nabla L+O(\eta^2)$. What are the mass, the friction, and the velocity relaxation time in steps?
3. *(Optional; Exercise 1.5.c is the cleaner route.)* After the transient, the inertia term is small and can be treated perturbatively. Solve the ODE of item 2 for $\dot\theta$ to first order in $\eta$ (substitute the leading-order $\ddot\theta$) and compare with the modified flow quoted above.

<details><summary>Hint</summary>

$\theta_{k\pm1}=\theta\pm\eta\dot\theta+\tfrac{\eta^2}2\ddot\theta+O(\eta^3)$. Substitute into $\theta_{k+1}-\theta_k=\beta(\theta_k-\theta_{k-1})-\eta g(\theta_k)$. For 3, at leading order $\dot\theta=-g/(1-\beta)$, so $\ddot\theta=-H\dot\theta/(1-\beta)$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\theta_{k+1}-\theta_k=\eta\dot\theta+O(\eta^2)$ and $\theta_k-\theta_{k-1}=\eta\dot\theta+O(\eta^2)$. The recursion gives $\eta\dot\theta(1-\beta)=-\eta g+O(\eta^2)$, so $\dot\theta=-g/(1-\beta)$.

**2.** Keeping second-order terms, $\theta_{k+1}-\theta_k=\eta\dot\theta+\tfrac{\eta^2}2\ddot\theta$ and $\theta_k-\theta_{k-1}=\eta\dot\theta-\tfrac{\eta^2}2\ddot\theta$, up to $O(\eta^3)$. Then

$$
\eta\dot\theta+\tfrac{\eta^2}2\ddot\theta-\beta\big(\eta\dot\theta-\tfrac{\eta^2}2\ddot\theta\big)=-\eta g
\;\Longrightarrow\;
\frac{\eta(1+\beta)}2\ddot\theta+(1-\beta)\dot\theta=-\nabla L+O(\eta^2).
$$

This is a particle with mass $\mathsf m=\eta(1+\beta)/2$ and friction $\gamma=1-\beta$ in the potential $L$. Velocity relaxes on the time scale $\mathsf m/\gamma=\eta\frac{1+\beta}{2(1-\beta)}$, about $\frac{1+\beta}{2(1-\beta)}$ steps (about 10 at $\beta=0.9$), comparable to the $\beta^k$ transient; on longer time scales the particle follows the rescaled gradient flow. This is Kovachki–Stuart's damped equation at their $a=0$ (they call the momentum parameter $\lambda$), which they show is $O(\eta)$-accurate (Theorem 4, started from rest).

**3.** Write the ODE as $\dot\theta=-\frac{g}{1-\beta}-\frac{\eta(1+\beta)}{2(1-\beta)}\ddot\theta$. At leading order $\dot\theta=-g/(1-\beta)$, so $\ddot\theta=\frac{d}{dt}\big(-\frac{g}{1-\beta}\big)=-\frac{H\dot\theta}{1-\beta}=\frac{Hg}{(1-\beta)^2}$. Substituting,

$$
\dot\theta=-\frac{g}{1-\beta}-\eta\frac{1+\beta}{2(1-\beta)^3}Hg+O(\eta^2)
=-\frac1{1-\beta}\nabla\Big(L+\frac{\eta(1+\beta)}{4(1-\beta)^2}\|\nabla L\|^2\Big)+O(\eta^2),
$$

which is the heavy-ball modified flow: after the transient, the mass term *is* the $O(\eta)$ correction. Exercise 1.5.c gets this directly from the recursion, the safer route, since the second-order ODE was itself only derived to $O(\eta^2)$ in the step.
"""
    ),
    markdown(
        r"""
**Exercise 1.5.c — The heavy-ball modified flow.**

After transients, assume the step is a function of the current point:
$\theta_{k+1}-\theta_k=D(\theta_k)$ with $D=\eta D_1+\eta^2D_2+O(\eta^3)$ (an invariant
"slow manifold" in $(\theta,v)$ space).

1. Use the recursion to find $D_1=-g/(1-\beta)$ and $D_2=-\beta Hg/(1-\beta)^3$.
2. Apply the recipe of §1.2 to the one-step map $\theta\mapsto\theta+D(\theta)$ to obtain the modified flow above.
3. Compare with GD at the same effective learning rate.

<details><summary>Hint</summary>

The previous step is $\theta_k-\theta_{k-1}=D(\theta_{k-1})$ with $\theta_{k-1}=\theta_k-D(\theta_{k-1})$, so $D(\theta_{k-1})=D-D'D+O(\eta^3)$, everything evaluated at $\theta_k$. In the recipe, the method's $f$ and $d_2$ are $D_1$ and $D_2$ (with $h=\eta$).
</details>
"""
    ),
    answer(
        r"""
**1.** Evaluate everything at $\theta=\theta_k$. The previous step is $D(\theta_{k-1})=D-D'D+O(\eta^3)=\eta D_1+\eta^2(D_2-D_1'D_1)+O(\eta^3)$. The recursion $D=\beta D(\theta_{k-1})-\eta g$ gives, order by order:

- $\eta^1$: $D_1=\beta D_1-g$, so $D_1=-g/(1-\beta)$.
- $\eta^2$: $D_2=\beta(D_2-D_1'D_1)$, so $D_2=-\frac{\beta}{1-\beta}D_1'D_1$. With $D_1'=-H/(1-\beta)$, $D_1'D_1=Hg/(1-\beta)^2$, so $D_2=-\beta Hg/(1-\beta)^3$.

**2.** The one-step map is $\theta\mapsto\theta+\eta D_1+\eta^2D_2$, so with $h=\eta$, $f=D_1$ and $d_2=D_2$:

$$
f_2=d_2-\tfrac12f'f=-\frac{\beta Hg}{(1-\beta)^3}-\frac{Hg}{2(1-\beta)^2}=-\frac{2\beta+(1-\beta)}{2(1-\beta)^3}Hg=-\frac{1+\beta}{2(1-\beta)^3}Hg .
$$

With $Hg=\tfrac12\nabla\|g\|^2$, the modified flow $\dot\theta=D_1+\eta f_2$ is

$$
\dot\theta=-\frac{1}{1-\beta}\nabla\Big(L+\frac{\eta(1+\beta)}{4(1-\beta)^2}\|\nabla L\|^2\Big).
$$

At $\beta=0$ this is GD's IGR flow. (Checked symbolically, and also verified against the heavy-ball recursion directly, in `checks/td2_1_flows_bea.py`.)

**3.** Rescale time to $s=t/(1-\beta)$, which counts steps of size $\eta_{\rm eff}=\eta/(1-\beta)$: $\frac{d\theta}{ds}=-\nabla\big(L+\frac{\eta(1+\beta)}{4(1-\beta)^2}\|\nabla L\|^2\big)=-\nabla\big(L+\frac{\eta_{\rm eff}}4\frac{1+\beta}{1-\beta}\|\nabla L\|^2\big)$. GD with step $\eta_{\rm eff}$ has penalty $\frac{\eta_{\rm eff}}4\|\nabla L\|^2$. So at matched effective learning rate the momentum penalty is $(1+\beta)/(1-\beta)$ times stronger: 3 at $\beta=0.5$, 19 at $\beta=0.9$.
"""
    ),
    markdown(
        r"""
<details><summary>Note: a sign error in Kovachki–Stuart's eq. (19)</summary>

Kovachki and Stuart (2021) derive the modified potential for the family $\theta_{k+1}=\theta_k+\beta(\theta_k-\theta_{k-1})-\eta\nabla L\big(\theta_k+a(\theta_k-\theta_{k-1})\big)$ ($a=0$ heavy ball, $a=\beta$ Nesterov), written as $L+\tfrac\eta2c\|\nabla L\|^2$ inside $-(1-\beta)^{-1}\nabla$. With $\bar\lambda=1/(1-\beta)$ (their notation; not a curvature), they print $c=\bar\lambda(\bar\lambda-a+\tfrac12)$ (their eq. (19)). Redoing the calculation gives $c=\bar\lambda(\bar\lambda-a-\tfrac12)$. Checks: at $\beta=0$ the minus sign gives GD's $\tfrac\eta4\|\nabla L\|^2$ (the plus sign gives $\tfrac{3\eta}4$), and at $a=0$ it reproduces the heavy-ball modified flow above. The slip is in their $\ddot u$ just after eq. (21) ($u$ the parameters, $\Phi$ the loss), which should read $\ddot u=+\bar\lambda^2H\nabla\Phi$. This is our diagnosis (checked symbolically in `checks/td2_1_flows_bea.py`), not a published erratum. For Nesterov ($a=\beta$) the corrected penalty is $\frac{\eta(2\beta^2-\beta+1)}{4(1-\beta)^2}\|\nabla L\|^2$.
</details>
"""
    ),
    code(
        r"""
# Heavy ball vs its modified flow on L = (ab-1)^2/2 at beta = 0.5, error at a fixed time t = k*eta.
# Starting on the slow manifold, the rescaled gradient flow is O(eta) off and Ghosh et al.'s modified
# flow O(eta^2). Starting from rest (v0 = 0), the start-up transient shifts the trajectory by O(eta),
# so the modified flow is again only O(eta) accurate at this short time.
beta, T_hb = 0.5, 0.5
p0 = np.array([2.0, 0.1])


def heavy_ball(p, eta, K, v):
    for _ in range(K):
        v = beta * v - eta * grad_ab(p)
        p = p + v
    return p


def hb_flow(eta):
    c = eta * (1 + beta) / (2 * (1 - beta)**2)   # grad of eta(1+b)/(4(1-b)^2)|g|^2 is c * H g
    return lambda p: -(grad_ab(p) + c * hess_ab(p) @ grad_ab(p)) / (1 - beta)


etas_hb = T_hb / np.array([25, 50, 100, 200, 400, 800])
res = {"rescaled GF, slow-manifold start": [], "modified flow, slow-manifold start": [],
       "modified flow, start from rest": []}
for eta in etas_hb:
    K = int(round(T_hb / eta))
    g0, H0 = grad_ab(p0), hess_ab(p0)
    D1 = -g0 / (1 - beta)
    D2 = -beta * H0 @ g0 / (1 - beta)**3
    v_slow = eta * D1 + eta**2 * (D2 - (-H0 / (1 - beta)) @ D1)   # previous step D(theta_{-1}) to O(eta^2)
    p_slow = heavy_ball(p0, eta, K, v_slow)
    p_rest = heavy_ball(p0, eta, K, np.zeros(2))
    res["rescaled GF, slow-manifold start"].append(np.linalg.norm(p_slow - rk4(hb_flow(0.0), p0, T_hb, 2000)))
    res["modified flow, slow-manifold start"].append(np.linalg.norm(p_slow - rk4(hb_flow(eta), p0, T_hb, 2000)))
    res["modified flow, start from rest"].append(np.linalg.norm(p_rest - rk4(hb_flow(eta), p0, T_hb, 2000)))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for name, e in res.items():
    ax.loglog(etas_hb, e, "o-", label=name)
    print(f"{name:<38} slope (smallest 4 eta) {slope(etas_hb[2:], e[2:]):.2f}")
ax.set_xlabel(r"$\eta$"); ax.set_ylabel(f"|heavy ball - flow| at t={T_hb}"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
Read the slopes, not the heights. From a slow-manifold start the rescaled gradient
flow is off by $O(\eta)$ and the modified flow by $O(\eta^2)$. From rest, the modified
flow carries the $O(\eta)$ start-up offset derived above. Its constant happens to be
small here, so the from-rest curve sits *below* the slow-start curve at larger $\eta$
and only overtakes it at the smallest steps, where its slope of about 1 shows.
"""
    ),
    markdown(
        r"""
**Adam (quoted, not derived).** [Cattaneo, Klusowski and Shigida](https://arxiv.org/abs/2309.00079)
apply the same machinery to Adam. For full-batch Adam after the bias-correction
transient, with gradient components large compared with $\sqrt\epsilon$, they find a
sign-like flow on $L+\frac\eta2\big(\frac{1+\beta}{1-\beta}-\frac{1+\rho}{1-\rho}\big)\|\nabla L\|_1$
(their eq. (5); $\beta$, $\rho$ the first- and second-moment parameters). With the usual
$\rho>\beta$ the coefficient is negative: Adam *anti*-penalizes the $\ell_1$ norm of the
gradient.
"""
    ),
    # ------------------------------------------------------------------ readings
    markdown(
        r"""
<a id="td21-readings"></a>

## Readings

Each section's lead papers are at its top. Also:
[Rosca, Wu, Qin & Dherin, a continuous-time model of GD and instability](https://arxiv.org/abs/2302.01952)
(the complex "principal flow", exact on quadratics; Exercise 1.2.c);
[Kovachki & Stuart, continuous-time analysis of momentum methods](https://www.jmlr.org/papers/v22/19-466.html)
(the damped ODE and slow manifold of §1.5; see the sign-error note there);
[Cattaneo, Klusowski & Shigida, on the implicit bias of Adam](https://arxiv.org/abs/2309.00079)
(backward error analysis for Adam and RMSProp).
"""
        + references(
            book(
                "Geometric Numerical Integration",
                "https://doi.org/10.1007/3-540-30666-8_9",
                "Hairer, Lubich & Wanner (Springer, 2nd ed. 2006)",
                "ch. IX (backward error analysis; §IX.7 for optimal truncation)",
            )
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "td2_1_flows", "td2_1_flows"))
