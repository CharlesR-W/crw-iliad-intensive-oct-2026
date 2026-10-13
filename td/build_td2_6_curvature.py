#!/usr/bin/env python3
"""Build NB2 §6, "Curvature: what the Hessian looks like and when the basin is chosen" (Training Dynamics day).

Page shape (streamlined 2026-10-09): a short pitch, the two key papers, a one-line route per
section; then five `##` sections (anchors td26-1 ... td26-5), each with a plain-words statement,
its lead paper(s) via reading(), a short core paragraph, and one
`### Optional: annotated derivations and walkthrough` (anchor td26-N-walkthrough) holding the
derivation narrative, the exercises (bold labels, not headings) with worked answers, and the
code demos. A short "Readings" section (td26-readings) lists the remaining papers in one line.

Closed-form results are checked in checks/td2_6_curvature_check.py; the review log is
../reviews/td/td2_6_curvature.md.
"""
from nbkit_td import answer, code, code_answer, markdown, reading, run

# ---------------------------------------------------------------- papers
SAGUN_URL = "https://arxiv.org/abs/1706.04454"
FORT_URL = "https://arxiv.org/abs/2010.15110"


def sagun(blurb):
    return reading(
        "Empirical analysis of the Hessian of over-parametrized neural networks", SAGUN_URL,
        "Sagun, Evci, Güney, Dauphin and Bottou (2017)", blurb, importance=4, length=2)


def fort(blurb):
    return reading(
        "Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time "
        "evolution of the Neural Tangent Kernel", FORT_URL,
        "Fort, Dziugaite, Paul, Kharaghani, Roy and Ganguli (NeurIPS 2020)", blurb, importance=5, length=3)


MARTENS = reading(
    "New insights and perspectives on the natural gradient method",
    "https://arxiv.org/abs/1412.1193",
    "Martens (JMLR 2020)",
    "The clearest account of how the Gauss–Newton matrix, the Fisher and the empirical Fisher relate (§§8–11), and when they coincide.",
    importance=2,
    length=3,
)
PAPYAN = reading(
    "Traces of class/cross-class structure pervade deep learning spectra",
    "https://arxiv.org/abs/2008.11865",
    "Papyan (JMLR 2020)",
    "Class-level structure (about $C$ outliers) in the Hessians, Fisher matrices and gradients of real networks, a weaker cross-class layer below it, knockout experiments that attribute each level, and how BatchNorm shrinks the outliers.",
    "Long; the introduction's summary is enough to orient.",
    importance=3,
    length=4,
)
GUR_ARI = reading(
    "Gradient descent happens in a tiny subspace",
    "https://arxiv.org/abs/1812.04754",
    "Gur-Ari, Roberts and Dyer (2018)",
    "The observation: within tens of steps the gradient concentrates in the top-$k$ Hessian eigenspace, $k$ about the number of classes, and that subspace then changes little.",
    importance=3,
    length=2,
)
SONG = reading(
    "Does SGD really happen in tiny subspaces?",
    "https://arxiv.org/abs/2405.16002",
    "Song, Ahn and Yun (ICLR 2025)",
    "The intervention: projecting updates onto that dominant subspace stops learning, and projecting it out does not hurt; explained by noise in an ill-conditioned valley, or by edge-of-stability oscillation for GD.",
    importance=4,
    length=3,
)
FRANKLE = reading(
    "Linear mode connectivity and the lottery ticket hypothesis",
    "https://arxiv.org/abs/1912.05671",
    "Frankle, Dziugaite, Roy and Carbin (ICML 2020)",
    "Defines instability to SGD noise as the error barrier between forked children, and measures when networks of various sizes become stable.",
    importance=3,
    length=3,
)

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# NB2 §6 — Curvature: what the Hessian looks like and when the basin is chosen

**Training Dynamics, Iliad Intensive 2026**

The Hessian of a classifier's training loss is a bulk of near-zero eigenvalues
plus a few large outliers, about one per class. This notebook asks where the
outliers come from, whether the gradient's pull toward them means learning
happens there, and how early a run commits to one basin.
"""
        + sagun("The bulk-plus-outliers picture on small MLPs: adding parameters grows the bulk at zero; changing the data moves the outliers.")
        + fort("If you read one paper: basin commitment and the tangent kernel's motion measured side by side.")
        + r"""
**Route.**
- [6.1 The Gauss–Newton decomposition](#td26-1): which part of the Hessian you keep when you drop the network's second derivatives.
- [6.2 Gauss–Newton and the tangent kernel](#td26-2): curvature eigenvalues are kernel eigenvalues.
- [6.3 Bulk and outliers](#td26-3): about $C$ outliers from class means of gradients.
- [6.4 The tiny subspace](#td26-4): the gradient lives there; learning mostly does not.
- [6.5 When the basin is chosen](#td26-5): forks, loss barriers, and kernel growth.

The code (torch on CPU, a minute or two) uses MLPs of about 700 parameters
on Gaussian clusters, small enough to diagonalize the Hessian exactly; their
outliers are weak (for 4 classes, $\lambda_4/\lambda_5<2$). The setup cell
below defines the networks and the Hessian pieces.
"""
    ),
    code(
        r"""
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.func import grad, hessian, jacrev, jvp, vmap

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)          # the matrices here are small; more threads mostly add overhead


def make_data(C, n_per, d=20, sep=2.0, seed=0):
    # C Gaussian clusters in d dimensions: class means of norm ~sep, within-class noise of norm ~1.
    g = torch.Generator().manual_seed(seed)
    mu = sep * torch.randn(C, d, generator=g) / d**0.5
    X = torch.cat([mu[c] + torch.randn(n_per, d, generator=g) / d**0.5 for c in range(C)])
    y = torch.arange(C).repeat_interleave(n_per)
    return X, y


class MLP:
    # Two-hidden-layer MLP acting on one flat parameter vector, so the Hessian is a plain P x P matrix.
    def __init__(self, d, h, C, act=F.gelu, seed=0, zero_last=False):
        self.shapes = [(h, d), (h,), (h, h), (h,), (C, h), (C,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.act = act
        g = torch.Generator().manual_seed(seed)
        ps = [torch.randn(*s, generator=g) * (2 / s[1]) ** 0.5 if len(s) == 2 else torch.zeros(*s)
              for s in self.shapes]
        if zero_last:
            ps[4] = torch.zeros_like(ps[4])
        self.theta0 = torch.cat([p.flatten() for p in ps])
        self.P = self.theta0.numel()

    def features(self, theta, X):
        W1, b1, W2, b2 = [p.reshape(s) for p, s in zip(torch.split(theta, self.sizes), self.shapes)][:4]
        return self.act(self.act(X @ W1.T + b1) @ W2.T + b2)    # last hidden layer, shape (n, h)

    def __call__(self, theta, X):
        W3, b3 = [p.reshape(s) for p, s in zip(torch.split(theta, self.sizes), self.shapes)][4:]
        return self.features(theta, X) @ W3.T + b3              # logits z, shape (n, C)


def ce(z, y):
    return F.cross_entropy(z, y)


def mse(z, y):
    return 0.5 * ((z - F.one_hot(y, z.shape[-1]).to(z.dtype)) ** 2).sum(-1).mean()


def logit_hessian(z, loss):
    # A_i = d^2 ell_i / dz_i^2, shape (n, C, C).
    if loss is ce:
        p = torch.softmax(z, -1)
        return torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    return torch.eye(z.shape[-1]).expand(len(z), -1, -1)


def ggn(f, theta, X, loss):
    # H_GN = (1/n) sum_i J_i^T A_i J_i
    J = jacrev(f)(theta, X)                                     # (n, C, P)
    A = logit_hessian(f(theta, X), loss)
    return torch.einsum('ncp,ncd,ndq->pq', J, A, J) / len(X)


def residual(f, theta, X, y, loss):
    # H_res = (1/n) sum_{i,c} (d ell_i / d z_ic) Hess_theta z_ic: the Hessian of
    # theta -> sum_i <r_i, z_i(theta)> with the residuals r_i = (1/n) d ell_i/dz_i held fixed.
    z = f(theta, X).detach().requires_grad_(True)
    r, = torch.autograd.grad(loss(z, y), z)
    return hessian(lambda t: (r * f(t, X)).sum())(theta)


def eigs(M):
    return torch.linalg.eigvalsh(M).flip(0).numpy()            # descending
"""
    ),
    # ================================================================ 6.1
    markdown(
        r"""
<a id="td26-1"></a>

## 6.1 The Gauss–Newton decomposition

Second-order optimizers, influence functions and Laplace approximations use
the Gauss–Newton matrix in place of the Hessian: the part built from first
derivatives of the network only, positive semidefinite by construction. The
part it drops vanishes at interpolation for square loss, but for cross-entropy
it shrinks only like one over the logit scale.
"""
        + MARTENS
        + r"""
Take $n$ examples with logits $z_i=f(\theta;x_i)\in\mathbb R^C$, per-example
loss $\ell_i$, $L=\frac1n\sum_i\ell_i$, Jacobians
$J_i=\partial z_i/\partial\theta\in\mathbb R^{C\times P}$ and logit-space loss
Hessians $A_i=\nabla_z^2\ell_i$. Differentiating twice through the logits,

$$
H=\nabla^2_\theta L=\underbrace{\frac1n\sum_iJ_i^\top A_i\,J_i}_{H_{\rm GN}}
\;+\;\underbrace{\frac1n\sum_i\sum_{c=1}^C\frac{\partial\ell_i}{\partial z_{ic}}\,\nabla^2_\theta z_{ic}}_{H_{\rm res}} .
$$

$H_{\rm GN}\succeq0$ whenever $\ell$ is convex in the logits (square loss,
softmax cross-entropy), so every negative eigenvalue of $H$ comes from the
residual $H_{\rm res}$. For classifiers $H_{\rm GN}$ is also the Fisher
information of the model's predictive distribution (Martens §§8–9), which is
why "Fisher", "Gauss–Newton" and "GGN" are used interchangeably; the
*empirical* Fisher, built at the observed labels, is a different matrix
(§11). In practice the damped $H_{\rm GN}$ is inverted approximately with
K-FAC ([Martens and Grosse 2015](https://arxiv.org/abs/1503.05671)); the Data
Attribution day picks up from there.

<a id="td26-1-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* as above; $\theta\in\mathbb R^P$; $J\in\mathbb R^{nC\times P}$
stacks the $J_i$.

**Square loss.** For $\ell=\frac12\|z-y\|^2$ with one-hot $y$, $A_i=I_C$ and
$\partial\ell/\partial z=z_i-y_i$. At an interpolating minimum every residual
vanishes, so $H=H_{\rm GN}=\frac1nJ^\top J$ exactly.

**Cross-entropy.** With $p=\mathrm{softmax}(z)$ and $\ell=-\log p_y$,
$\nabla_z\ell=p-e_y$ ($e_y$ the $y$-th unit vector) and
$A=\mathrm{diag}(p)-pp^\top$. The residual $p-e_y$ never vanishes at finite
logits. Near interpolation, with $1-p_y=\varepsilon$, both $A$ and $p-e_y$ are
of order $\varepsilon$, so $H_{\rm GN}$ and $H_{\rm res}$ shrink together;
their ratio is set by the size of the logits, and improves only as the
margins (correct-class logit minus the largest other logit) grow, slowly
(Exercise 6.1.B).

**Rank.** $A\mathbf 1=0$, because adding a constant to every logit leaves $p$
unchanged; when all $p_c>0$ the rank of $A$ is exactly $C-1$. So
$\mathrm{rank}\,H_{\rm GN}\le n(C-1)$ ($\le nC$ for square loss), and when
$P>n(C-1)$ at least $P-n(C-1)$ eigenvalues of $H_{\rm GN}$ are exactly zero.
Part of the near-zero "bulk" of measured spectra (6.3) is this rank
deficiency, which $H_{\rm res}$ spreads into small eigenvalues of both signs.

**Exercise 6.1.A — The decomposition.** Notation as above.

1. Derive the decomposition with the chain rule. Show that $H_{\rm GN}\succeq0$ when every $A_i\succeq0$.
2. For square loss, show that $H_{\rm res}=0$ at any interpolating point, where $H=\frac1nJ^\top J$.
3. Take the scalar two-layer linear network $f=w_1w_2$ (one input $x=1$, target $s>0$, $L=\frac12(w_1w_2-s)^2$).
   - Compute $H_{\rm GN}$ and $H_{\rm res}$ at a general point, and show that $H_{\rm res}$ has eigenvalues $\pm(w_1w_2-s)$.
   - On the balanced line $w_1=w_2=a$, find where $H$ is indefinite.
4. Show that the eigenvalues of $H$ and $H_{\rm GN}$ differ by at most $\|H_{\rm res}\|_2$, and that $\lambda_{\min}(H)\ge\lambda_{\min}(H_{\rm res})$.

<details><summary>Hint</summary>

For 1, write $\partial L/\partial\theta_a$ with one sum over logits, then differentiate again. Two terms appear: one where the second derivative hits $\ell$ and one where it hits $z$. For 3, $\nabla f=(w_2,w_1)$ and $\nabla^2 f=\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$. For 4, use Weyl's inequality $\lambda_j(X)+\lambda_{\min}(Y)\le\lambda_j(X+Y)\le\lambda_j(X)+\lambda_{\max}(Y)$ for symmetric $X$ and $Y$.
</details>
"""
    ),
    answer(
        r"""
**1.** With one sum over logits,
$\partial L/\partial\theta_a=\frac1n\sum_i\sum_c(\partial\ell_i/\partial z_{ic})(\partial z_{ic}/\partial\theta_a)$, which is $\nabla L=\frac1n\sum_iJ_i^\top\nabla_z\ell_i$. Differentiating again, the product rule gives two terms:

$$
\frac{\partial^2L}{\partial\theta_a\partial\theta_b}=\frac1n\sum_i\Big[\sum_{c,d}\frac{\partial z_{ic}}{\partial\theta_a}\frac{\partial^2\ell_i}{\partial z_{ic}\partial z_{id}}\frac{\partial z_{id}}{\partial\theta_b}+\sum_c\frac{\partial\ell_i}{\partial z_{ic}}\frac{\partial^2z_{ic}}{\partial\theta_a\partial\theta_b}\Big].
$$

The first term is $\big(\frac1n\sum_iJ_i^\top A_iJ_i\big)_{ab}$ and the second is $(H_{\rm res})_{ab}$. For any $u\in\mathbb R^P$,
$u^\top H_{\rm GN}u=\frac1n\sum_i(J_iu)^\top A_i(J_iu)\ge0$ when every $A_i\succeq0$.

**2.** For $\ell=\frac12\|z-y\|^2$, $\nabla_z\ell_i=z_i-y_i$ and $A_i=I$. At an interpolating point $z_i=y_i$ for every $i$, so every coefficient of $\nabla^2_\theta z_{ic}$ in $H_{\rm res}$ is zero. Then $H=H_{\rm GN}=\frac1nJ^\top J$.

**3.** Write $r=w_1w_2-s$. Then $\ell'=r$ and $\ell''=1$, $\nabla f=(w_2,w_1)^\top$ and $\nabla^2f=\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$, so

$$
H_{\rm GN}=\begin{pmatrix}w_2^2&w_1w_2\\w_1w_2&w_1^2\end{pmatrix},\qquad
H_{\rm res}=r\begin{pmatrix}0&1\\1&0\end{pmatrix}.
$$

The eigenvalues of $H_{\rm res}$ are $\pm r$, with eigenvectors $(1,\pm1)/\sqrt2$. Away from the origin $H_{\rm GN}$ has rank one, with eigenvalue $w_1^2+w_2^2$ along $(w_2,w_1)$.

On the balanced line $w_1=w_2=a$:

$$
H=\begin{pmatrix}a^2&2a^2-s\\2a^2-s&a^2\end{pmatrix},
\qquad \det H=a^4-(2a^2-s)^2=(s-a^2)(3a^2-s).
$$

So $H$ is indefinite for $a^2<s/3$, near the saddle at the origin, and for $a^2>s$, past the minimum, where the residual has the other sign. It is positive definite for $s/3<a^2<s$, and singular at the minimum $a^2=s$, whose zero eigenvalue points along the hyperbola $w_1w_2=s$. Even in this two-parameter model, the residual is what makes the Hessian indefinite.

**4.** Weyl's inequality with $X=H_{\rm GN}$ and $Y=H_{\rm res}$ gives
$\lambda_j(H_{\rm GN})-\|H_{\rm res}\|_2\le\lambda_j(H)\le\lambda_j(H_{\rm GN})+\|H_{\rm res}\|_2$.
With $X=H_{\rm res}$ and $Y=H_{\rm GN}\succeq0$ instead, $\lambda_{\min}(H)\ge\lambda_{\min}(H_{\rm res})+\lambda_{\min}(H_{\rm GN})\ge\lambda_{\min}(H_{\rm res})$. Negative curvature is bounded by the residual, so near an interpolating minimum for square loss the negative eigenvalues must be small.
"""
    ),
    markdown(
        r"""
**Exercise 6.1.B — Softmax cross-entropy.** Let $p=\mathrm{softmax}(z)\in\mathbb R^C$,
$\ell=-\log p_y$ and $A=\nabla_z^2\ell$; $H_{\rm GN}$ and $H_{\rm res}$ as in 6.1.

1. Show that $\nabla_z\ell=p-e_y$ and $A=\mathrm{diag}(p)-pp^\top$.
2. Show that $A=\sum_cp_c(e_c-p)(e_c-p)^\top$, and deduce: $A\succeq0$; $A\mathbf 1=0$; the rank is exactly $C-1$ when every $p_c>0$; $\mathrm{tr}\,A=1-\|p\|^2$ and $\lambda_{\max}(A)\le\max_cp_c$.
3. **Fisher.** Show that $A=\mathbb E_{\hat y\sim p}\big[\nabla_z\ell(z,\hat y)\nabla_z\ell(z,\hat y)^\top\big]$, so $H_{\rm GN}$ is the Fisher information of the model's predictive distribution. How does the empirical Fisher $F_{\rm emp}=\frac1n\sum_iJ_i^\top(p_i-e_{y_i})(p_i-e_{y_i})^\top J_i$ scale near interpolation, compared with $H_{\rm GN}$?
4. **Near interpolation.** Let $p_y=1-\varepsilon$ and $p_c=\varepsilon q_c$ for $c\ne y$, with $\sum_{c\ne y}q_c=1$. Show that, exactly,
   $$p-e_y=\varepsilon\sum_{c\ne y}q_c(e_c-e_y),\qquad\text{and}\qquad A=\varepsilon\sum_{c\ne y}q_c(e_c-e_y)(e_c-e_y)^\top+O(\varepsilon^2).$$
   Write the resulting per-example contributions to $H_{\rm GN}$ and $H_{\rm res}$ in terms of $\nabla z_c-\nabla z_y$ and $\nabla^2z_c-\nabla^2z_y$.
5. **Logit scale.** Suppose the network is homogeneous of degree $k$, $f(\lambda\theta)=\lambda^kf(\theta)$ (e.g. a bias-free ReLU net with $k$ layers). Along a ray $\theta=\lambda\hat\theta$ on which every training point is classified correctly, show that $\|H_{\rm res}\|/\|H_{\rm GN}\|$ falls like $\lambda^{-k}$, one over the logit scale, even though both terms vanish like $\varepsilon$ (unless $\nabla^2z=0$, where the residual vanishes identically).

<details><summary>Hint</summary>

For 2, expand $\sum_cp_c(e_c-p)(e_c-p)^\top$ using $\sum_cp_c=1$. For the rank, $u^\top Au=\mathrm{Var}_{c\sim p}(u_c)$, which vanishes only for constant $u$. For $\lambda_{\max}$, note $A\preceq\mathrm{diag}(p)$. For 4, the term $c=y$ in item 2's sum is $O(\varepsilon^2)$. For 5, differentiate $f(\lambda\theta)=\lambda^kf(\theta)$ once and twice in $\theta$: $\nabla f$ scales as $\lambda^{k-1}$ and $\nabla^2f$ as $\lambda^{k-2}$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\ell=-z_y+\log\sum_ce^{z_c}$, so $\partial\ell/\partial z_c=p_c-\delta_{cy}$, that is, $\nabla_z\ell=p-e_y$. Differentiating $p_c=e^{z_c}/\sum_de^{z_d}$ gives $\partial p_c/\partial z_d=p_c\delta_{cd}-p_cp_d$, hence $A=\mathrm{diag}(p)-pp^\top$. The label $y$ drops out of the Hessian.

**2.** Expanding, with $\sum_cp_c=1$:

$$
\sum_cp_c(e_c-p)(e_c-p)^\top=\sum_cp_ce_ce_c^\top-\Big(\sum_cp_ce_c\Big)p^\top-p\Big(\sum_cp_ce_c\Big)^\top+pp^\top=\mathrm{diag}(p)-pp^\top.
$$

- **PSD.** $A$ is a nonnegative combination of outer products.
- **Null vector.** $A\mathbf 1=p-p(p^\top\mathbf 1)=0$.
- **Rank.** $u^\top Au=\sum_cp_cu_c^2-(\sum_cp_cu_c)^2=\mathrm{Var}_{c\sim p}(u_c)$. When every $p_c>0$, this vanishes only if all $u_c$ are equal, so the null space is exactly $\mathrm{span}(\mathbf 1)$ and the rank is $C-1$.
- **Trace.** $\mathrm{tr}\,A=\sum_cp_c-\sum_cp_c^2=1-\|p\|^2\le1-1/C$.
- **Largest eigenvalue.** $A\preceq\mathrm{diag}(p)$, so $\lambda_{\max}(A)\le\max_cp_c$.

**3.** With $\hat y\sim p$, $\nabla_z\ell(z,\hat y)=p-e_{\hat y}$, and

$$
\mathbb E_{\hat y\sim p}\big[(p-e_{\hat y})(p-e_{\hat y})^\top\big]=\sum_cp_c(e_c-p)(e_c-p)^\top=A
$$

by item 2. Therefore $H_{\rm GN}=\frac1n\sum_i\mathbb E_{\hat y\sim p_i}\big[\nabla_\theta\ell(\cdot,\hat y)\nabla_\theta\ell(\cdot,\hat y)^\top\big]$, the Fisher information of the model's conditional distribution averaged over the training inputs.

The empirical Fisher uses the observed label instead. Near interpolation $p_i-e_{y_i}=O(\varepsilon_i)$, so $F_{\rm emp}=O(\varepsilon^2)$ while $H_{\rm GN}=O(\varepsilon)$ (item 4). At the end of training the empirical Fisher is smaller by a factor of order $\varepsilon$, and its spectrum need not resemble the Hessian's.

**4.** Exactly,
$\sum_{c\ne y}p_c(e_c-e_y)=(p-p_ye_y)-(1-p_y)e_y=p-e_y$; with $p_c=\varepsilon q_c$ this is the first identity. For $A$, use item 2: the $c=y$ term is $p_y(e_y-p)(e_y-p)^\top=(1-\varepsilon)\,O(\varepsilon^2)$; for $c\ne y$ the weight is $\varepsilon q_c$ and $e_c-p=e_c-e_y+O(\varepsilon)$. Hence $A=\varepsilon\sum_{c\ne y}q_c(e_c-e_y)(e_c-e_y)^\top+O(\varepsilon^2)$.

Sandwiching between $J_i^\top$ and $J_i$, with $J_i^\top(e_c-e_y)=\nabla_\theta z_{ic}-\nabla_\theta z_{iy}$:

$$
H_{{\rm GN},i}\approx\varepsilon_i\sum_{c\ne y_i}q_{ic}(\nabla z_{ic}-\nabla z_{iy_i})(\nabla z_{ic}-\nabla z_{iy_i})^\top,\qquad
H_{{\rm res},i}=\varepsilon_i\sum_{c\ne y_i}q_{ic}(\nabla^2z_{ic}-\nabla^2z_{iy_i}).
$$

Both are first order in $\varepsilon_i$, with the same weights $q_{ic}$. Their ratio is set by how large $\nabla^2z$ is compared with $(\nabla z)^2$.

**5.** Differentiating $f(\lambda\theta)=\lambda^kf(\theta)$ gives $\nabla f(\lambda\hat\theta)=\lambda^{k-1}\nabla f(\hat\theta)$ and $\nabla^2f(\lambda\hat\theta)=\lambda^{k-2}\nabla^2f(\hat\theta)$. Along the ray the logits are $\lambda^k\hat z_i$. With every point correctly classified, $\varepsilon_i\approx\sum_{c\ne y_i}e^{-\lambda^km_{ic}}$, where $m_{ic}=\hat z_{iy_i}-\hat z_{ic}>0$ are the margins at $\hat\theta$. From item 4:

$$
\|H_{{\rm GN},i}\|\sim\varepsilon_i\,\lambda^{2k-2},\qquad\|H_{{\rm res},i}\|\sim\varepsilon_i\,\lambda^{k-2},\qquad
\frac{\|H_{\rm res}\|}{\|H_{\rm GN}\|}\sim\lambda^{-k}\propto\frac{1}{\text{logit scale}}.
$$

Both terms vanish exponentially in $\lambda^k$ (cross-entropy minima of separable data are at infinity), but the ratio falls only like one over the logit scale. So $H\approx H_{\rm GN}$ is controlled by margin growth, which under gradient descent is slow (logarithmic in time for separable data), not by the loss being small.
"""
    ),
    markdown(
        r"""
**Code 6.1.C — The decomposition, measured.** The first cell checks
$H=H_{\rm GN}+H_{\rm res}$ to machine precision and the rank of the softmax
Hessians (`residual` is the Hessian of $\theta\mapsto\sum_i\langle r_i,z_i(\theta)\rangle$
with the residuals $r_i=\frac1n\nabla_z\ell_i$ held fixed). The second trains
the network with cross-entropy and with square loss (128 points, 4 classes,
full-batch GD). Look for:
- square loss: the residual's share of the Hessian goes to zero; cross-entropy: it tracks one over the median margin, not the loss (from step 30 on, share × margin drifts only from about 1.6 to 1.85 while the loss falls 300-fold; the GELU net is not homogeneous, so this is beyond Exercise 6.1.B);
- at the end, the top four eigenvalues of $H$ and $H_{\rm GN}$ agree to 0.1–7%; further down, the residual adds to the bulk;
- the rest of the spectrum has both signs (466 positive, 193 negative) plus 17 exact zeros (adding the same vector to every last-layer weight row, and the same constant to every output bias, leaves the loss unchanged); the top four are only weakly separated ($\lambda_4/\lambda_5\approx1.6$).
"""
    ),
    code(
        r"""
t0 = time.time()
X, y = make_data(C=4, n_per=32)
net = MLP(d=20, h=16, C=4)
L = lambda t: ce(net(t, X), y)
theta = net.theta0.clone()
for _ in range(30):                                            # move to a generic, partly trained point
    theta = theta - 0.1 * grad(L)(theta)

for loss in (ce, mse):
    H = hessian(lambda t: loss(net(t, X), y))(theta)
    HGN, Hres = ggn(net, theta, X, loss), residual(net, theta, X, y, loss)
    err = (H - HGN - Hres).abs().max() / H.abs().max()
    print(f"{loss.__name__}: P = {net.P}, max|H - H_GN - H_res| / max|H| = {err:.1e}, "
          f"min eig H_GN = {eigs(HGN)[-1]:+.1e}, min eig H_res = {eigs(Hres)[-1]:+.2e}")

A = logit_hessian(net(theta, X), ce)
a = torch.linalg.eigvalsh(A)                                  # (n, C), ascending
print(f"softmax Hessians A_i: max |A_i 1| = {(A @ torch.ones(4)).abs().max():.1e}; "
      f"smallest eigenvalue / largest <= {(a[:, 0].abs() / a[:, -1]).max():.1e}; "
      f"second smallest / largest >= {(a[:, 1] / a[:, -1]).min():.1e}  (so rank C - 1 = 3)")
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    code(
        r"""
t0 = time.time()
ckpts = [0, 10, 30, 100, 300, 1000, 2000]
rec = {}
for loss in (ce, mse):
    L = lambda t: loss(net(t, X), y)
    theta, rows = net.theta0.clone(), []
    for s in range(ckpts[-1] + 1):
        if s in ckpts:
            HGN, Hres = ggn(net, theta, X, loss), residual(net, theta, X, y, loss)
            H = HGN + Hres
            z = net(theta, X)
            margin = (z.gather(1, y[:, None]).squeeze() - z.scatter(1, y[:, None], -1e9).max(1).values).median()
            rows.append((s, L(theta).item(), (torch.linalg.norm(Hres) / torch.linalg.norm(H)).item(), margin.item()))
            if s == ckpts[-1]:
                rec[loss.__name__ + "_spec"] = (eigs(H), eigs(HGN))
        theta = theta - 0.1 * grad(L)(theta)
    rec[loss.__name__] = np.array(rows)

for name in ("ce", "mse"):
    print(name)
    for s, l, r, m in rec[name]:
        extra = f"   median margin {m:5.2f}   ratio x margin {r * m:4.2f}" if name == "ce" else ""
        print(f"  step {int(s):5d}: loss {l:.2e}   ||H_res||/||H|| = {r:.3f}" + extra)
eH, eG = rec["ce_spec"]
print("CE, end of training, top 4 eigenvalues of H:", np.round(eH[:4], 5), " of H_GN:", np.round(eG[:4], 5),
      f" (relative differences {np.round(np.abs(eH[:4] - eG[:4]) / eH[:4], 3)})")
tol = 1e-10 * eH[0]                                              # numerical zeros: exact symmetries of CE
print(f"CE, end of training: {(eH > tol).sum()} positive, {(eH < -tol).sum()} negative, "
      f"{(np.abs(eH) <= tol).sum()} numerically zero eigenvalues of H (|lambda| <= {tol:.0e})")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for name, mk in (("ce", "o-"), ("mse", "s-")):
    ax[0].loglog(rec[name][:, 1], rec[name][:, 2], mk, label=name)
ax[0].set(xlabel="training loss", ylabel=r"$\|H_{\rm res}\|_F/\|H\|_F$", title="residual term along training")
ax[0].invert_xaxis(); ax[0].legend()
k = np.arange(1, 41)
ax[1].semilogy(k, eH[:40], "o", label="$H$"); ax[1].semilogy(k, eG[:40], "x", label=r"$H_{\rm GN}$")
ax[1].set(xlabel="index", title="CE, end of training: top 40 eigenvalues"); ax[1].legend()
bins = np.linspace(-9, -1, 41)
ax[2].hist(np.log10(eH[eH > tol]), bins=bins, alpha=0.7, label=f"positive ({(eH > tol).sum()})")
ax[2].hist(np.log10(-eH[eH < -tol]), bins=bins, alpha=0.7, label=f"negative ({(eH < -tol).sum()})")
ax[2].set_yscale("log"); ax[2].legend()
ax[2].set(xlabel=r"$\log_{10}|\lambda|$", ylabel="count", title=f"CE, end: nonzero eigenvalues of $H$ ($P={net.P}$)")
fig.tight_layout(); plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    markdown(
        r"""
Here $n(C-1)=384<P=676$, so $H_{\rm GN}$ has at least 292 exactly zero
eigenvalues. The bulk of $H$, of both signs, is mostly $H_{\rm res}$ acting on
that null space and on the small eigenvalues of $H_{\rm GN}$. The outliers are
$H_{\rm GN}$'s.
"""
    ),
    # ================================================================ 6.2
    markdown(
        r"""
<a id="td26-2"></a>

## 6.2 The Gauss–Newton matrix and the loss-weighted tangent kernel share their spectrum

Curvature lives in parameter space; the tangent kernel, which says how the
outputs move under a gradient step, lives in function space. Both are built
from the same Jacobian, so the Gauss–Newton matrix and the kernel weighted by
the loss Hessian have the same nonzero eigenvalues. Their eigenvectors are
tied only through the Jacobian.

This is linear algebra rather than a paper's result; Martens (6.1) covers the
Gauss–Newton side.

Stack the Jacobians $J_i=\partial z_i/\partial\theta$ ($n$ examples, $C$
logits, $P$ parameters) into $J\in\mathbb R^{nC\times P}$ and the logit-space
loss Hessians $A_i=\nabla_z^2\ell_i$ into the block-diagonal $A$. The
empirical tangent kernel is $\Theta=JJ^\top$: under gradient flow on
$L=\frac1n\sum_j\ell_j$ the logits move as
$\dot z_i=-\frac1n\sum_jJ_iJ_j^\top\nabla_z\ell_j$. Since
$H_{\rm GN}=\frac1nJ^\top AJ=M^\top M$ and $\frac1nA^{1/2}\Theta A^{1/2}=MM^\top$
with $M=n^{-1/2}A^{1/2}J$, with multiplicity

$$
\mathrm{spec}_{\neq0}\,H_{\rm GN}=\mathrm{spec}_{\neq0}\Big(\tfrac1n\,A^{1/2}\,\Theta\,A^{1/2}\Big).
$$

For square loss ($A=I$) at an interpolating minimum, the sharpness
$S=\lambda_{\max}(H)$ is the top kernel eigenvalue $\lambda_{\max}(\Theta)/n$:
progressive sharpening is the top of the kernel growing, and the edge of
stability holds it near $2n/\eta$ ($\eta$ the step size). For cross-entropy an
example fitted with confidence $1-\varepsilon_i$ has $A_i=O(\varepsilon_i)$
and drops out: the top of the Hessian is the kernel restricted to the examples
the model is still unsure about.

<a id="td26-2-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* as above; $H$ is the loss Hessian, $H_{\rm res}=H-H_{\rm GN}$ the
residual of 6.1.

**Directions.** If $v$ is a unit eigenvector of the weighted kernel with
eigenvalue $\lambda$, then $u=n^{-1/2}J^\top A^{1/2}v/\sqrt\lambda$ is one of
$H_{\rm GN}$, and $A^{1/2}Ju=\sqrt{n\lambda}\,v$: $u$ moves the
*loss-weighted* logits along $v$. Neither side's rotation forces the other's:
with square loss, $J\mapsto JQ$ ($Q$ orthogonal on parameter space) rotates
$H_{\rm GN}$'s eigenvectors and leaves $\Theta$ fixed, and $J\mapsto RJ$ ($R$
orthogonal on the output side) rotates $\Theta$'s and leaves $H_{\rm GN}$
fixed. So a kernel distance that mixes spectrum and rotation, as in 6.5,
speaks about curvature only through its spectral part, and only after the
$A$-reweighting (Code 6.2.B shows the unweighted kernel growing while the
weighted one shrinks).

**Exercise 6.2.A — The identity and its consequences.** Notation as above.
Items 2–3 show why, with square loss near interpolation, sustained
progressive sharpening means the kernel is not frozen.

1. Prove the spectral identity and the eigenvector map. Show that $\mathrm{rank}\,H_{\rm GN}=\mathrm{rank}(A^{1/2}J)\le\min(P,\,n(C-1))$ for cross-entropy.
2. For square loss with $L=\frac1{2n}\sum_i\|z_i-y_i\|^2$, linearize one GD step in function space and derive $r\mapsto(I-\eta\Theta/n)\,r$, with $r=z-y$ stacked. Conclude that near an interpolating minimum, GD is stable iff $\eta S<2$ with $S=\lambda_{\max}(\Theta)/n$.
3. For a model exactly linearized around a point (so $J$ is constant), explain in two sentences why $H$ does not change along the trajectory with square loss, while with cross-entropy it does.
4. *(Optional.)* **A two-layer linear network.** Take $L=\frac12\|\Sigma_{yx}-W_2W_1\|_F^2$, where $W_2W_1$ maps $\mathbb R^d\to\mathbb R^m$ and the target $\Sigma_{yx}=USV^\top$ has rank $r$, singular values $s_1>\dots>s_r>0$ and singular vectors $u_\alpha$ (columns of $U$) and $v_\beta$ (columns of $V$). Work at a balanced global minimum ($W_2W_2^\top=(WW^\top)^{1/2}$, $W_1^\top W_1=(W^\top W)^{1/2}$, $W=\Sigma_{yx}$).
   - Show that the nonzero eigenvalues of $H$ are those of the kernel operator $K[E]=W_2W_2^\top E+EW_1^\top W_1$ on $m\times d$ matrices.
   - Show that these eigenvalues are $s_\alpha+s_\beta$ for $\alpha,\beta\le r$, together with $s_\alpha$ (each with multiplicity $d-r$) and $s_\beta$ (each with multiplicity $m-r$).
   - Conclude that $S=2s_1$: the sharpness of the minimum is set by the strongest mode of the target.

<details><summary>Hint</summary>

For 1, use the SVD of $M$: $M=\sum_k\sigma_ka_kb_k^\top$ gives $M^\top M=\sum\sigma_k^2b_kb_k^\top$ and $MM^\top=\sum\sigma_k^2a_ka_k^\top$, and $M^\top a_k=\sigma_kb_k$. For 2, $\nabla L=\frac1nJ^\top r$ and $z(\theta-\eta\nabla L)\approx z-\eta J\nabla L$. For 4, at a global minimum the residual vanishes, so $H=J^\top J$ with $J=\partial\,\mathrm{vec}(W_2W_1)/\partial\theta$. The kernel $JJ^\top$ acting on $\mathrm{vec}\,E$ is $K$. Try $E=u_\alpha v_\beta^\top$, extending $\{u_\alpha\}$ and $\{v_\beta\}$ to orthonormal bases.
</details>
"""
    ),
    answer(
        r"""
**1.** Let $M=n^{-1/2}A^{1/2}J\in\mathbb R^{nC\times P}$, with $A^{1/2}$ the symmetric PSD square root of the block-diagonal $A$. Then $M^\top M=\frac1nJ^\top AJ=H_{\rm GN}$ and $MM^\top=\frac1nA^{1/2}JJ^\top A^{1/2}$.

Write the SVD $M=\sum_k\sigma_ka_kb_k^\top$ with $\sigma_k>0$. Then $M^\top M=\sum_k\sigma_k^2b_kb_k^\top$ and $MM^\top=\sum_k\sigma_k^2a_ka_k^\top$: both have nonzero eigenvalues $\{\sigma_k^2\}$ with the same multiplicities, and everything else is zero.

For the eigenvectors, if $MM^\top v=\lambda v$ with $\lambda>0$ and $\|v\|=1$, set $u=M^\top v/\sqrt\lambda=n^{-1/2}J^\top A^{1/2}v/\sqrt\lambda$. Then $M^\top Mu=M^\top(MM^\top v)/\sqrt\lambda=\lambda u$, and $\|u\|^2=v^\top MM^\top v/\lambda=1$.

The rank is $\mathrm{rank}\,M=\mathrm{rank}(A^{1/2}J)\le\min(P,\mathrm{rank}\,A)$, and for cross-entropy $\mathrm{rank}\,A=\sum_i\mathrm{rank}\,A_i\le n(C-1)$.

**2.** Here $\nabla_z\ell_i=z_i-y_i=r_i$, so $\nabla L=\frac1nJ^\top r$. One GD step moves $\theta$ by $-\frac\eta nJ^\top r$; to first order, $z$ moves by $J$ times that:

$$
r_{t+1}=r_t-\frac\eta nJJ^\top r_t=\Big(I-\frac\eta n\Theta\Big)r_t .
$$

Each kernel eigencomponent is multiplied by $1-\eta\mu/n$, with $\mu$ the kernel eigenvalue. This is stable iff $\eta\lambda_{\max}(\Theta)/n<2$. By item 1 with $A=I$, $\lambda_{\max}(\Theta)/n=\lambda_{\max}(H_{\rm GN})$, which equals $S$ at an interpolating minimum, where $H_{\rm res}=0$. This is the same condition, $\eta S<2$, as for GD on the quadratic $\tfrac12\lambda x^2$, which converges iff $\eta\lambda<2$.

**3.** For $z(\theta)=z_0+J(\theta-\theta_0)$, $\nabla^2_\theta z=0$, so $H_{\rm res}=0$ and $H=\frac1nJ^\top AJ$. With square loss $A=I$, so $H=\frac1nJ^\top J$ is constant along the whole trajectory. With cross-entropy the weights $A_i=\mathrm{diag}(p_i)-p_ip_i^\top$ depend on the current predictions; as examples become confidently fitted their $A_i$ shrink, and $H$ changes although $J$ does not.

For a nonlinear network a fixed kernel $JJ^\top$ is weaker than a constant $J$: it fixes the nonzero spectrum of $H_{\rm GN}$ (square loss) but not its eigenvectors, and says nothing about $H_{\rm res}$. So inferring kernel growth from progressive sharpening of the full Hessian also requires the residual to be negligible, as it is near interpolation with square loss.

**4.** At a global minimum $E=0$, so $H_{\rm res}=0$ and $H=J^\top J$, with $J$ the Jacobian of $\mathrm{vec}(W_2W_1)$ with respect to $(W_1,W_2)$. By item 1 with $A=I$, its nonzero eigenvalues are those of $JJ^\top$, the map $\delta W\mapsto$ (change of $W$ under the parameter step $J^\top\delta W$). The step $\Delta W_1=W_2^\top\delta W$, $\Delta W_2=\delta W\,W_1^\top$ changes $W$, to first order, by $\Delta W_2W_1+W_2\Delta W_1=\delta W\,W_1^\top W_1+W_2W_2^\top\delta W=K[\delta W]$.

At a balanced minimum $W_2W_2^\top=\sum_\alpha s_\alpha u_\alpha u_\alpha^\top$ and $W_1^\top W_1=\sum_\beta s_\beta v_\beta v_\beta^\top$. Extend $\{u_\alpha\}$ and $\{v_\beta\}$ to orthonormal bases of $\mathbb R^m$ and $\mathbb R^d$, with $s=0$ on the added vectors. Then

$$
K[u_\alpha v_\beta^\top]=(s_\alpha+s_\beta)\,u_\alpha v_\beta^\top ,
$$

so the eigenvalues of $K$ are all sums $s_\alpha+s_\beta$. The nonzero ones are $s_\alpha+s_\beta$ with both $\alpha,\beta\le r$; $s_\alpha$ with $\alpha\le r<\beta$ ($d-r$ copies each); and $s_\beta$ with $\beta\le r<\alpha$ ($m-r$ copies each). The largest is $S=2s_1$: the curvature spectrum at the minimum is read off from the target's singular values, and the top Hessian direction rescales the strongest mode.
"""
    ),
    markdown(
        r"""
**Exercise 6.2.B (code) — The identity, and what the weighting does.**
Train the network of Code 6.1.C with cross-entropy. At steps 10 and 2000,
check the identity numerically, then compare the unweighted kernel
$\Theta/n$ with the weighted $\frac1nA^{1/2}\Theta A^{1/2}$. For the top $C$
eigenvectors of each (vectors over (example, logit) pairs), measure the
fraction explained by patterns that depend only on (class of the example,
logit): the function-space side of the outliers of 6.3. Look for:
- the two spectra agree to machine precision, and $\mathrm{rank}\,H_{\rm GN}=n(C-1)$ exactly;
- the top kernel eigenvectors are almost entirely class patterns (about 0.97, against a random baseline of 0.03), and the unweighted kernel *grows* during training;
- the weighted kernel, which is the curvature, *shrinks* about 200-fold as examples become confidently fitted, and its top eigenvectors become less class-like: the weighting concentrates the curvature on the examples still uncertain.
"""
    ),
    code_answer(
        r"""
n, C = X.shape[0], 4


def class_score(v):
    # Fraction of ||v||^2 (v in R^{nC}, indexed by (example i, logit k)) explained by patterns that depend
    # only on (class of i, k): the "class-indicator-like" part.  A random vector scores about C^2/(nC).
    V = v.reshape(n, C)
    means = torch.stack([V[y == c].mean(0) for c in range(C)])        # (C classes, C logits)
    return (means[y] ** 2).sum().item() / (V ** 2).sum().item()


L = lambda t: ce(net(t, X), y)
# Your code: train with GD (lr 0.1) and, at steps 10 and 2000, form the stacked Jacobian J (nC x P), the
# block-diagonal A (use logit_hessian), a symmetric square root of A, and Theta = J J^T.
# 1. Check that J^T A J / n and A^1/2 Theta A^1/2 / n have the same top eigenvalues, and the rank of H_GN.
# 2. For the unweighted Theta / n and the weighted A^1/2 Theta A^1/2 / n, print the top-C eigenvalues and the
#    class_score of the top-C eigenvectors.
""",
        r"""
n, C = X.shape[0], 4


def class_score(v):
    # Fraction of ||v||^2 (v in R^{nC}, indexed by (example i, logit k)) explained by patterns that depend
    # only on (class of i, k): the "class-indicator-like" part.  A random vector scores about C^2/(nC).
    V = v.reshape(n, C)
    means = torch.stack([V[y == c].mean(0) for c in range(C)])        # (C classes, C logits)
    return (means[y] ** 2).sum().item() / (V ** 2).sum().item()


L = lambda t: ce(net(t, X), y)
theta = net.theta0.clone()
for step in range(2001):
    if step in (10, 2000):
        J = jacrev(net)(theta, X).reshape(n * C, -1)                 # stacked Jacobian (nC, P)
        A = torch.block_diag(*logit_hessian(net(theta, X), ce))      # (nC, nC)
        w, V = torch.linalg.eigh(A)
        A_half = V @ torch.diag(w.clamp(min=0).sqrt()) @ V.T
        Theta = J @ J.T                                              # empirical NTK Gram matrix
        K_w = A_half @ Theta @ A_half / n                            # loss-weighted kernel
        e_gn, e_w = eigs(J.T @ A @ J / n), eigs(K_w)
        print(f"step {step}: train loss {L(theta):.2e}; max rel. difference, top 50 eigenvalues of H_GN vs "
              f"A^1/2 Theta A^1/2 / n: {np.max(np.abs(e_gn[:50] - e_w[:50]) / e_gn[:50]):.0e}; "
              f"rank H_GN = {(e_gn > 1e-12 * e_gn[0]).sum()} (n(C-1) = {n * (C - 1)})")
        for name, K in (("unweighted Theta/n", Theta / n), ("weighted A^1/2 Theta A^1/2/n", K_w)):
            ev, U = torch.linalg.eigh(K)
            scores = [class_score(U[:, -j]) for j in range(1, C + 1)]
            print(f"   {name:30s}: top-{C} eigenvalues {np.round(ev.flip(0)[:C].numpy(), 4)}, "
                  f"class-pattern share of eigenvectors {np.round(scores, 2)}")
    theta = theta - 0.1 * grad(L)(theta)
print(f"(random-vector baseline for the class-pattern share: about {C * C / (n * C):.2f})")
""",
    ),
    # ================================================================ 6.3
    markdown(
        r"""
<a id="td26-3"></a>

## 6.3 Bulk and outliers

Measured Hessian spectra of classifiers show a bulk of eigenvalues near zero,
which grows when the network grows, and a few large outliers tied to the $C$
classes, which move when the data change. The outliers sit in the Gauss–Newton
part and come from class means of per-example gradient vectors; the
within-class scatter of those vectors forms the bulk.
"""
        + sagun("Short and readable: bulk plus outliers on Gaussian-mixture data and small MLPs, with the outliers in the Gauss–Newton part. Its appendix shows how much the outlier count depends on the cutoff rule (often a lot).")
        + PAPYAN
        + r"""
How many outliers? About $C$ dominant ones (about $k$ for small MLPs on $k$
Gaussian clusters), and below them a weaker cross-class layer of at most
$C(C-2)$ directions (Papyan measures about $C^2-2C$) that can merge into the
bulk; exactly $C-1$ in one special case, a zero last layer. Papers report
$C$, $C-1$ or about $C^2$ depending on what they count, the cutoff rule, the
loss, normalization (BatchNorm shrinks the outliers) and the stage of
training.

<a id="td26-3-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* $n$ examples with true classes $y_i$; logits $z_i$ with Jacobians
$J_i=\partial z_i/\partial\theta$; predicted probabilities
$p_i=\mathrm{softmax}(z_i)$ with entries $p_{ik}$; $e_k$ the $k$-th unit
vector in $\mathbb R^C$; $H_{\rm GN}=\frac1n\sum_iJ_i^\top A_iJ_i$ with the
softmax Hessian $A_i=\mathrm{diag}(p_i)-p_ip_i^\top$ (6.1).

**Why the outliers track classes.** Writing
$A_i=\sum_kp_{ik}(e_k-p_i)(e_k-p_i)^\top$ (Exercise 6.1.B, item 2) gives
$J_i^\top A_iJ_i=\sum_kp_{ik}v_{ik}v_{ik}^\top$, with

$$
v_{ik}=J_i^\top(e_k-p_i)=-\nabla_\theta\,\ell(z_i,k),
$$

minus the gradient example $i$ would give if it carried label $k$. So
$H_{\rm GN}$ is a weighted second moment of these vectors, labelled by the
true class $y_i$ and the hypothetical label $k$, and a second moment is means
plus covariances. *If* the vectors cluster by $(y_i,k)$, the group means give
large eigenvalues (at most $C(C-1)$, with $C$ dominant), and the within-group
scatter, spread over many directions, gives the bulk.

Why would they cluster? Restricted to the last-layer weights, $v_{ik}$ is
$(e_k-p_i)\otimes\tilde h_i$, with $\tilde h_i$ the last hidden features plus
a bias entry. Features of trained networks cluster by class, so these blocks
cluster by $(y_i,k)$; the observed clustering of logit gradients (Papyan;
[Fort and Ganguli](https://arxiv.org/abs/1910.05929)) is the empirical input.
Exercise 6.3.A derives the rest *given* that clustering. It also finds that,
for cross-entropy near interpolation, the $C$ class-level directions are
spanned to leading order by the class-mean *loss gradients*; when these are
the top directions, the gradient lives in the top subspace for a structural
reason (6.4).

**Exercise 6.3.A — The class-mean mechanism (derived given clustering).**
Notation as above, with $v_{ik}=J_i^\top(e_k-p_i)$. The core is item 1, the
class-level identity and Weyl bounds of item 3, and item 4; the counting parts
are optional.

1. Show that $H_{\rm GN}=\frac1n\sum_i\sum_kp_{ik}v_{ik}v_{ik}^\top$, and that $\sum_kp_{ik}v_{ik}=0$ for every $i$. Deduce $\sum_{k\ne y_i}p_{ik}v_{ik}=p_{iy_i}\nabla_\theta\ell_i$.
2. **Grouping.** Group the terms by $(c,k)$, with $c=y_i$ the true class and $k$ the hypothetical label. For each group define the weight $w_{ck}=\frac1n\sum_{i:y_i=c}p_{ik}$ and the weighted mean $m_{ck}$ (weights $p_{ik}$).
   - Show that $H_{\rm GN}=M_{\rm pair}+W$, with $M_{\rm pair}=\sum_{c,k}w_{ck}m_{ck}m_{ck}^\top$ and $W\succeq0$ (the weighted within-group covariances).
   - *(Optional, for the count.)* Show that $\mathrm{rank}\,M_{\rm pair}\le C(C-1)$.
3. **Class level.** Let $\bar m_c$ be the $w$-weighted mean of $m_{ck}$ over the wrong labels $k\ne c$, with total weight $\bar w_c=\sum_{k\ne c}w_{ck}$, and set $M_{\rm class}=\sum_c\bar w_c\bar m_c\bar m_c^\top$ (rank $\le C$).
   - Show $M_{\rm pair}\succeq M_{\rm class}$. *(Optional, for the count.)* Show that the range of $M_{\rm pair}$ is spanned by the $C$ class directions $\bar m_c$ together with at most $C(C-2)$ cross-class directions $m_{ck}-\bar m_c$ (a sum of spans, not necessarily orthogonal).
   - Use item 1 to show that $\bar m_c$ is proportional to $\sum_{i:y_i=c}p_{ic}\nabla_\theta\ell_i$, a weighted class-mean gradient.
   - Conclude with Weyl's inequality that $\lambda_j(H_{\rm GN})\ge\lambda_j(M_{\rm pair})$ and $\lambda_j(H_{\rm GN})\le\lambda_j(M_{\rm pair})+\|W\|_2$. When the within-group scatter is small in every direction, the top $C(C-1)$ eigenvalues are those of the group means.
4. **The special case at initialization.** Suppose the last layer is initialized to zero, so every $p_i$ is uniform and only the last layer has a nonzero Jacobian. Let $\tilde h_i=(h_i,1)$ be the last hidden features with a bias entry, and $M_2=\frac1n\sum_i\tilde h_i\tilde h_i^\top$.
   - Show that, restricted to the last layer (zeros elsewhere), $H_{\rm GN}=\frac1C\big(I_C-\tfrac1C\mathbf 1\mathbf 1^\top\big)\otimes M_2$ for cross-entropy and $H_{\rm GN}=I_C\otimes M_2$ for square loss.
   - Assuming $\lambda_{\max}(M_2)$ is simple and separated from the rest of $M_2$'s spectrum, conclude that the top eigenvalue of $H_{\rm GN}$ is $\lambda_{\max}(M_2)/C$ with multiplicity $C-1$ (cross-entropy), or $\lambda_{\max}(M_2)$ with multiplicity $C$ (square loss). Explain why $\lambda_{\max}(M_2)$ is large when the features have a sizeable mean, as GELU or ReLU features typically do.

<details><summary>Hint</summary>

For 1, combine 6.1.B, item 2 with $J_i^\top$ on both sides; then $\sum_kp_{ik}(e_k-p_i)=0$. For 2, within one group, $\sum_iw_i v_iv_i^\top=(\sum w_i)\,\bar v\bar v^\top+\sum_iw_i(v_i-\bar v)(v_i-\bar v)^\top$. For the rank, item 1 makes $\sum_kw_{ck}m_{ck}=0$ for each class $c$, so the $C$ vectors $\{m_{ck}\}_k$ span at most $C-1$ dimensions. For 3, apply the same "mean plus scatter" identity to the pair means within a class. For 4, $J_i$ restricted to the last layer is $I_C\otimes\tilde h_i^\top$, and $(X\otimes Y)$ has eigenvalues $\lambda_a(X)\lambda_b(Y)$.
</details>
"""
    ),
    answer(
        r"""
**1.** From 6.1.B, item 2, $A_i=\sum_kp_{ik}(e_k-p_i)(e_k-p_i)^\top$. Sandwiching between $J_i^\top$ and $J_i$ and averaging over $i$ gives $H_{\rm GN}=\frac1n\sum_{i,k}p_{ik}v_{ik}v_{ik}^\top$.

Also $\sum_kp_{ik}v_{ik}=J_i^\top\sum_kp_{ik}(e_k-p_i)=J_i^\top(p_i-p_i)=0$. Separating the $k=y_i$ term,
$\sum_{k\ne y_i}p_{ik}v_{ik}=-p_{iy_i}v_{iy_i}=p_{iy_i}J_i^\top(p_i-e_{y_i})=p_{iy_i}\nabla_\theta\ell_i$.

**2.** Within group $(c,k)$, with weights $p_{ik}$ summing to $nw_{ck}$ and weighted mean $m_{ck}$:

$$
\frac1n\sum_{i:y_i=c}p_{ik}v_{ik}v_{ik}^\top=w_{ck}\,m_{ck}m_{ck}^\top+\frac1n\sum_{i:y_i=c}p_{ik}(v_{ik}-m_{ck})(v_{ik}-m_{ck})^\top
$$

(expand the second term; the cross terms cancel by the definition of $m_{ck}$). Summing over groups gives $H_{\rm GN}=M_{\rm pair}+W$ with $W\succeq0$.

For the rank, item 1 summed over $i$ in class $c$ gives $\sum_kw_{ck}m_{ck}=0$. For each class the $C$ vectors $m_{c1},\dots,m_{cC}$ are linearly dependent and span at most $C-1$ dimensions, so all $C^2$ means span at most $C(C-1)$ and $\mathrm{rank}\,M_{\rm pair}\le C(C-1)$.

**3.** Apply the same identity to the pair means of class $c$ over $k\ne c$, with weights $w_{ck}$:

$$
\sum_{k\ne c}w_{ck}m_{ck}m_{ck}^\top=\bar w_c\,\bar m_c\bar m_c^\top+\sum_{k\ne c}w_{ck}(m_{ck}-\bar m_c)(m_{ck}-\bar m_c)^\top\succeq\bar w_c\,\bar m_c\bar m_c^\top .
$$

Summing over $c$, and adding the $k=c$ groups $w_{cc}m_{cc}m_{cc}^\top\succeq0$, gives $M_{\rm pair}\succeq M_{\rm class}$ exactly. (Near interpolation the $k=c$ groups are small anyway: $v_{ic}=-\nabla_\theta\ell_i=O(\varepsilon)$ with weight $p_{ic}\approx1$, so they contribute $O(\varepsilon^2)$ against $O(\varepsilon)$.)

For the range: by item 1, $w_{cc}m_{cc}=-\sum_{k\ne c}w_{ck}m_{ck}=-\bar w_c\bar m_c$, so $m_{cc}$ is parallel to $\bar m_c$. Within class $c$ the span of $\{m_{ck}\}_k$ is $\mathrm{span}\{\bar m_c\}$ plus the span of the $C-1$ deviations $m_{ck}-\bar m_c$ ($k\ne c$), whose $w$-weighted sum is zero, so they span at most $C-2$ dimensions. Over all classes: $C$ class directions plus at most $C(C-2)$ cross-class ones, consistent with $\mathrm{rank}\,M_{\rm pair}\le C(C-1)$.

By item 1, $\bar w_c\bar m_c=\sum_{k\ne c}w_{ck}m_{ck}=\frac1n\sum_{i:y_i=c}p_{ic}\nabla_\theta\ell_i$, a weighted class-mean gradient of the actual loss. Near interpolation $p_{ic}=1-\varepsilon_i$, so it differs from the unweighted class sum $\sum_i\nabla\ell_i$ by $\sum_i\varepsilon_i\nabla\ell_i$, which is $O(\varepsilon^2)$ against $O(\varepsilon)$. So the $\bar m_c$ are, to that accuracy, proportional to the $C$ class-mean gradients (turning this into an angle bound needs those gradients to be well-conditioned, with no near-cancellation), and the full-batch gradient, their average, lies approximately in their span. Whether this span is the *top* eigenspace of $H_{\rm GN}$ depends on how large $M_{\rm class}$ is compared with the rest; the decomposition does not decide it, the code measures it.

For the eigenvalues, Weyl's inequality with $W\succeq0$ gives $\lambda_j(M_{\rm pair})\le\lambda_j(H_{\rm GN})\le\lambda_j(M_{\rm pair})+\|W\|_2$, and $M_{\rm pair}\succeq M_{\rm class}$ gives $\lambda_j(M_{\rm pair})\ge\lambda_j(M_{\rm class})$. If $\|W\|_2$ (the largest within-group variance in any single direction) is small compared with the top eigenvalues of the means, those eigenvalues appear as outliers. The within-group scatter has large total trace but is spread over many directions, so it forms the bulk: a spiked-covariance picture, with the clustering of logit gradients by class as the empirical input.

**4.** With $W_3=0$ the backpropagated signal to earlier layers vanishes, so only $(W_3,b_3)$ has a nonzero Jacobian, and since $z_i=W_3h_i+b_3$ that Jacobian is $I_C\otimes\tilde h_i^\top$. All logits are zero, so every $p_i=\mathbf 1/C$ and every $A_i$ equals $\bar A=\frac1C(I-\frac1C\mathbf 1\mathbf 1^\top)$ for cross-entropy, or $I$ for square loss. Then

$$
H_{\rm GN}=\frac1n\sum_i(I_C\otimes\tilde h_i)\,\bar A\,(I_C\otimes\tilde h_i^\top)=\bar A\otimes\frac1n\sum_i\tilde h_i\tilde h_i^\top=\bar A\otimes M_2 .
$$

The eigenvalues of a Kronecker product are products of eigenvalues. For cross-entropy, $\bar A$ has eigenvalue $1/C$ with multiplicity $C-1$, plus $0$ on $\mathbf 1$, so $\lambda_{\max}(M_2)/C$ appears $C-1$ times. For square loss, $\bar A=I_C$, so $\lambda_{\max}(M_2)$ appears $C$ times.

$M_2=\mu\mu^\top+\mathrm{Cov}(\tilde h)$, with $\mu$ the mean feature vector. For ReLU features, and typically for GELU features, most coordinates have a sizeable positive mean (dead units excepted), so $\|\mu\|^2$ grows like the width, while the covariance is spread over many directions; the top eigenvalue is then dominated by the mean.

These degenerate outliers are the feature mean tensored with the logit structure, the mechanism Papyan (2020, §9.3) identifies for the Hessian's outliers, in its simplest exact form. The degeneracy comes from the isotropic softmax factor $\bar A$ on $\mathbf 1^\perp$ (uniform predictions, whatever class information the features carry), and the softmax null direction removes exactly one. If $\lambda_{\max}(M_2)$ had multiplicity $q$, the counts would be $q(C-1)$ and $qC$.
"""
    ),
    markdown(
        r"""
**Code 6.3.B — The class-mean part reproduces the outliers.** For $C=2,4,8$,
train to 100% training accuracy and compare the spectrum of $H_{\rm GN}$ with
$M_{\rm pair}$ and $M_{\rm class}$ of Exercise 6.3.A; then check where the
gradient sits, and the zero-last-layer case. Look for:
- $M_{\rm class}$ (rank $\le C$) lower-bounds and tracks the top $C$ eigenvalues of $H_{\rm GN}$ (for $C=8$, 0.84 against 1.00 for the largest), with a visible knee after the $C$-th;
- $M_{\rm pair}$ (rank $\le C(C-1)$) tracks the next eigenvalues too; for $C=2,4$ its spectrum ends where the bulk begins, while for $C=8$ there is no gap near index 56 and its smaller eigenvalues sit inside the bulk;
- with a zero last layer the counts are exactly $C-1$ for cross-entropy and $C$ for square loss.
"""
    ),
    code(
        r"""
t0 = time.time()


def pair_means_part(f, theta, X, y, C):
    # H_GN = (1/n) sum_{i,k} p_ik v_ik v_ik^T with v_ik = J_i^T (e_k - p_i).  Group the (i, k) terms by
    # (true class y_i, logit k), and keep only each group's weighted mean ("pair means"), or only each class's
    # mean over its wrong logits k != y ("class means").
    J = jacrev(f)(theta, X)
    p = torch.softmax(f(theta, X), -1)
    v = torch.einsum('ncp,nkc->nkp', J, torch.eye(C)[None] - p[:, None, :])   # v[i, k] = J_i^T (e_k - p_i)
    n = len(X)
    pair, cls = torch.zeros(f.P, f.P), torch.zeros(f.P, f.P)
    for c in range(C):
        idx = y == c
        wsum, msum = 0.0, 0.0
        for k in range(C):
            w = p[idx, k]
            m = (w[:, None] * v[idx, k]).sum(0) / w.sum()
            pair += (w.sum() / n) * torch.outer(m, m)
            if k != c:
                wsum, msum = wsum + w.sum(), msum + w.sum() * m
        cls += (wsum / n) * torch.outer(msum / wsum, msum / wsum)
    G = torch.einsum('nk,nkp,nkq->pq', p, v, v) / n
    return G, pair, cls


fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for j, C in enumerate((2, 4, 8)):
    Xc, yc = make_data(C, n_per=50)
    netc = MLP(d=20, h=16, C=C)
    Lc = lambda t: ce(netc(t, Xc), yc)
    th = netc.theta0.clone()
    for _ in range(300):
        th = th - 0.1 * grad(Lc)(th)
    G, pair, cls = pair_means_part(netc, th, Xc, yc, C)
    assert torch.allclose(G, ggn(netc, th, Xc, ce))
    eG, eP, eC = eigs(G), eigs(pair), eigs(cls)
    K = C * C + 6
    k = np.arange(1, K + 1)
    ax[j].semilogy(k, eG[:K], "o-", label=r"$H_{\rm GN}$")
    ax[j].semilogy(k[:C * (C - 1)], eP[:C * (C - 1)], "x--", label=f"pair means (rank $\\leq C(C-1)={C*(C-1)}$)")
    ax[j].semilogy(k[:C], eC[:C], "+:", ms=10, label=f"class means (rank $\\leq C={C}$)")
    ax[j].axvline(C + 0.5, color="gray", ls=":")
    ax[j].set(ylim=(eG[K - 1] / 3, eG[0] * 2), xlabel="index", title=f"C = {C}, after 300 GD steps")
    ax[j].legend(fontsize=8)
    print(f"C={C}: top C+2 of H_GN / top: {np.round(eG[:C + 2] / eG[0], 2)};  class-means part: {np.round(eC[:C + 2] / eG[0], 2)}")
fig.tight_layout(); plt.show()

# Where the gradient sits: the top-C eigenspace of H_GN versus the span of the class-mean gradients (C = 8 run).
Q, _ = torch.linalg.qr(torch.stack([grad(lambda t: ce(netc(t, Xc[yc == c]), yc[yc == c]))(th) for c in range(C)], 1))
U = torch.linalg.eigh(G)[1][:, -C:]
gfull = grad(Lc)(th)
print(f"C={C}: overlap of top-{C} eigenspace with span of class-mean gradients, tr(P_U P_Q)/C = {(U.T @ Q).norm()**2 / C:.2f}; "
      f"fraction of the full gradient in the top-{C} eigenspace = {(U.T @ gfull).norm()**2 / gfull.norm()**2:.2f}")

# At a zero-initialized last layer: every p_i is uniform and only the last layer has a nonzero Jacobian.
for C in (4, 8):
    Xc, yc = make_data(C, n_per=50)
    netc = MLP(d=20, h=16, C=C, zero_last=True)
    hb = torch.cat([netc.features(netc.theta0, Xc), torch.ones(len(Xc), 1)], 1)    # features with a bias entry
    m2_top = eigs(hb.T @ hb / len(Xc))[0]
    for loss, pred in ((ce, m2_top / C), (mse, m2_top)):
        e = eigs(ggn(netc, netc.theta0, Xc, loss))
        gap = int(np.argmax(e[:2 * C] / e[1:2 * C + 1])) + 1
        print(f"zero last layer, C={C}, {loss.__name__}: biggest gap after eigenvalue {gap}; those {gap} equal to "
              f"{e[0]:.4f} (spread {np.ptp(e[:gap]):.0e}); predicted {pred:.4f}; next eigenvalue {e[gap]:.4f}")
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    markdown(
        r"""
So the class level is clean on this data: a knee after $C$, and a top-$C$
eigenspace that nearly coincides with the span of the class-mean gradients.
The cross-class level is clean for small $C$ and merges into the bulk at
$C=8$. That merging, more pronounced on real data, is one reason the papers
disagree about the count.
"""
    ),
    # ================================================================ 6.4
    markdown(
        r"""
<a id="td26-4"></a>

## 6.4 The tiny subspace: the gradient is there, learning (mostly) is not

Early in training the gradient comes to point almost entirely into the few
top-curvature directions of the Hessian. Yet an optimizer restricted to that
subspace stops making progress, and one that avoids it trains normally. The
top component is the restoring force of an oscillation across a steep valley;
progress is the slow drift along the valley floor, in the bulk.
"""
        + GUR_ARI
        + SONG
        + r"""
Take an eigendirection of the Hessian with curvature $\lambda$, and let $g$ be
the gradient's component along it. A step $-\eta g$ ($\eta$ the step size)
changes the loss, to second order, by

$$
\Delta L=-\eta g^2\Big(1-\frac{\eta\lambda}{2}\Big).
$$

At the edge of stability, $\eta\lambda\approx2$, the decrease cancels: the
step overshoots to the opposite wall. Below the edge with SGD noise, the step
lowers the loss only by removing what the noise injected, a fixed stationary
floor. Either way the large gradient component along steep directions is not
progress.

<a id="td26-4-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Prerequisites.* **Edge of stability**
([`td2_2`](td2_2_edge_of_stability.html)): GD on $\tfrac12\lambda x^2$
converges iff $\eta\lambda<2$; in full-batch training the sharpness
$S=\lambda_{\max}(H)$ rises to about $2/\eta$ and hovers there
([Cohen et al. 2021](https://arxiv.org/abs/2103.00065)), with the iterate
oscillating along the top eigenvector
([Damian, Nichani and Lee](https://arxiv.org/abs/2209.15594)). **SGD noise
floor** ([`td2_3`](td2_3_sgd_sde.html)): near a minimum, SGD with step $\eta$
and batch $B$ fluctuates at a stationary excess loss of about
$\frac{\eta}{4B}\mathrm{tr}\,\Sigma$ ($\Sigma$ the per-example gradient
covariance; quadratic basin, $\eta h\ll1$ in the noisy directions).

*Notation:* $H$ is the Hessian of the training loss $L$, $P_k$ the orthogonal
projector onto its top-$k$ eigenvectors, $\nabla L$ the full-batch gradient,
$g_B$ a minibatch gradient.

**The observation** (Gur-Ari, Roberts and Dyer). With $k$ the number of
classes, the fraction of the full-batch gradient in the top-$k$ eigenspace,

$$
f_{\rm top}=\frac{\|P_k\nabla L\|^2}{\|\nabla L\|^2},
$$

rises close to 1 within tens of steps of SGD (in their MNIST MLP, top-20:
from 0.20 to 0.94 in 50 steps), and the subspace then changes little.

**The intervention** (Song, Ahn and Yun). Replace the minibatch update
$-\eta g_B$ by Dom-SGD, $-\eta P_kg_B$, or by Bulk-SGD, $-\eta(I-P_k)g_B$,
with $P_k$ recomputed every step. Switched on in mid-training, once the
alignment is high (at 53% and 69% training accuracy where reported), Dom-SGD
stops decreasing the training loss and Bulk-SGD trains as well as SGD. Dom
also fails for GD at the edge, momentum SGD, Adam and SAM. (Settings: small
subsets of MNIST, CIFAR-10 and SST-2; small MLPs, CNN and Transformer; mostly
square loss.) Their explanation: near the trajectory the landscape is an
ill-conditioned valley, and minibatch noise keeps knocking the iterate up the
steep walls, where the gradient is large and points back across. Small-step
GD shows no alignment ($f_{\rm top}\to0$); GD at the edge aligns through its
self-stabilizing oscillation. The Mysteries day's
[MDL4 §9](../notebooks/mdl4_mysteries.html#wb6-section-9) makes the same
point with a pure quadratic.

**Where this holds.** The valley picture is about mid-training. Late in
training with cross-entropy, on fitted data, alignment can be
non-oscillatory. Heuristically, if one competitor dominates an example's
loss, $\ell\approx e^{-m}$ with margin $m$, then $\nabla\ell\approx-\ell\nabla m$
and $\nabla^2\ell\approx\ell(\nabla m\nabla m^\top-\nabla^2m)$; if the margin
Hessian is negligible and $\eta\ell\|\nabla m\|^2\ll2$, the gradient points
along a top curvature direction and a step along it realizes its full
first-order decrease. Code 6.4.B panel (b) is consistent with this (there
Dom-GD trains like GD and Bulk-GD barely moves) but does not establish the
mechanism.

**Exercise 6.4.A — A valley.** Why can the gradient point almost entirely
along a steep direction while all the progress is along a flat one? Take
$L(x,y)=\frac12ax^2+\ell(y)$, where $x,y$ are coordinates in parameter space,
$x$ is steep (curvature $a$), and along the floor direction $y$ the loss
decreases slowly: $\ell'(y)=-c<0$, $\ell''\approx0$. Dom-GD updates only $x$;
Bulk-GD updates only $y$; $\eta$ is the step size.

1. For full GD, Dom-GD and Bulk-GD, compute the one-step change of $L$ exactly.
2. **Edge of stability.** Suppose $\eta a=2-\delta$ with $0<\delta\ll1$ and the iterate starts at $x_0\ne0$.
   - Compute $f_{\rm top}$ (the fraction of $\|\nabla L\|^2$ along $x$) as a function of $x$, and show it is close to 1 whenever $|x|\gg c/a$.
   - How does the loss evolve under each scheme over many steps?
3. **SGD noise.** Let the $x$-update carry noise, $x_{t+1}=(1-\eta a)x_t+\eta\xi_t$ with $\mathbb E\xi_t=0$ and $\mathbb E\xi_t^2=\sigma^2$, independent across steps.
   - Find the stationary $\mathbb E x^2$, the stationary loss in $x$, and the mean squared gradient component $a^2\mathbb Ex^2$.
   - Compute the ratio $a^2\mathbb Ex^2/(a^2\mathbb Ex^2+c^2)$, a proxy for $f_{\rm top}$. For Gaussian noise, explain why $f_{\rm top}$ is typically near 1 once $a\eta\sigma^2\gg c^2(2-\eta a)$, and why Dom-SGD stalls while Bulk-SGD keeps descending.
4. **No noise, small step.** Under GD with $\eta a<2$ and no noise, what does $f_{\rm top}$ do over time? Compare with the small-learning-rate GD curve in Code 6.4.B.
5. What would make Bulk-GD fail in this model? Relate your answer to why Song, Ahn and Yun switch to Dom/Bulk only after the iterate is near the valley floor.

<details><summary>Hint</summary>

The update along $x$ is $x\mapsto(1-\eta a)x$, so $\frac12ax^2$ changes by $\frac12ax^2[(1-\eta a)^2-1]=-\frac12\eta a^2x^2(2-\eta a)$. Along $y$, $\ell$ changes by $\ell(y+\eta c)-\ell(y)=-\eta c^2$ when $\ell$ is linear. For 3, take the expectation of $x_{t+1}^2$ and set it equal to $\mathbb Ex_t^2$.
</details>
"""
    ),
    answer(
        r"""
**1.** The gradient is $(ax,\,-c)$.
- **Dom-GD** moves only $x$, to $(1-\eta a)x$: $\Delta L_{\rm Dom}=\frac12a x^2\big[(1-\eta a)^2-1\big]=-\frac12\eta a^2x^2(2-\eta a)$.
- **Bulk-GD** moves only $y$, to $y+\eta c$: $\Delta L_{\rm Bulk}=-\eta c^2$ for linear $\ell$.
- **Full GD** does both, and the changes add.

Both are instances of $\Delta L=-\eta g^2(1-\eta\lambda/2)$: along $x$, $g=ax$ and $\lambda=a$; along $y$, $g=-c$ and $\lambda=0$.

**2.** With $\eta a=2-\delta$,
$$
f_{\rm top}=\frac{a^2x^2}{a^2x^2+c^2},
$$
which is close to 1 whenever $|x|\gg c/a$. Over many steps:
- **The $x$-coordinate.** Each step multiplies $x$ by $1-\eta a=-(1-\delta)$: $x$ flips sign and shrinks by $1-\delta$. The loss in $x$ falls by $\frac12\eta a^2x^2\delta=a x^2\delta(1-\delta/2)$, a tiny fraction of $\frac12ax^2$ per step. The large gradient along $x$ buys almost nothing.
- **Dom-GD** only bleeds $\frac12ax^2$ away at rate about $2\delta$ per step and never lowers $\ell$: it stalls at $\ell(y_0)$.
- **Bulk-GD** lowers $\ell$ by $\eta c^2$ every step, which is all the progress there is, but leaves the oscillation energy $\frac12ax_0^2$ in place.
- **Full GD** does both.

In a real network at the edge, self-stabilization keeps $\delta\approx0$, so the oscillation persists, and Bulk-GD tracks full GD up to that constant oscillation energy.

**3.** Squaring and averaging, $\mathbb Ex_{t+1}^2=(1-\eta a)^2\mathbb Ex_t^2+\eta^2\sigma^2$. At stationarity,

$$
\mathbb Ex^2=\frac{\eta^2\sigma^2}{1-(1-\eta a)^2}=\frac{\eta\sigma^2}{a(2-\eta a)},\qquad
\tfrac12a\,\mathbb Ex^2=\frac{\eta\sigma^2}{2(2-\eta a)},\qquad
a^2\,\mathbb Ex^2=\frac{a\eta\sigma^2}{2-\eta a}.
$$

The mean squared gradient along $x$ grows with the curvature $a$, while along $y$ it is $c^2$. The ratio is $\frac{a\eta\sigma^2/(2-\eta a)}{a\eta\sigma^2/(2-\eta a)+c^2}$. It is a proxy, not the mean of $f_{\rm top}$ (a mean of ratios); with heavy-tailed noise the two can differ wildly. For Gaussian $\xi$, $x$ is Gaussian at stationarity, $|x|$ is typically of order its standard deviation, and the proxy describes the typical $f_{\rm top}$: near 1 once $a\eta\sigma^2\gg c^2(2-\eta a)$. Steep walls plus noise give alignment, even far below the edge.

- **Dom-SGD** keeps $x$ fluctuating at its stationary level, so its loss sits at $\ell(y_0)+\frac{\eta\sigma^2}{2(2-\eta a)}$ forever.
- **Bulk-SGD** freezes $x$ (no update and no noise along $x$) and moves $y$ at the full rate.

This is the mechanism of Song et al.'s toy model, $L=\frac12(1000x^2+y^2)$ with two per-sample losses $L\pm100xy$, whose gradient noise is $\pm100\,(y,x)$: the noise along the steep $x$ direction is proportional to the position $y$ along the floor, and does not vanish on the floor.

**4.** Without noise, $x_t=(1-\eta a)^tx_0\to0$ geometrically while the $y$-gradient stays $-c$, so $f_{\rm top}\to0$. This is the small-learning-rate GD curve in the code, where $f_{\rm top}$ drops to zero within a few dozen steps. Alignment needs something that keeps re-exciting the steep direction: noise, or an edge-of-stability oscillation.

**5.** Bulk-GD never corrects $x$. If the iterate starts far up the wall, the energy $\frac12ax_0^2$ is never removed. In a curved valley, where the floor bends and the steep direction rotates, steps along the bulk also drift off the floor at second order, and nothing pulls the iterate back. Bulk updates work when the iterate is already near the floor and the remaining steep-direction energy is small or self-limiting: the regime Song et al. intervene in, after an initial phase of ordinary training. (Their appendix also reports that Bulk-SGD raises the sharpness faster than SGD, consistent with nothing in the bulk update holding the steep directions in check.)
"""
    ),
    markdown(
        r"""
**Code 6.4.B — Alignment, and what projecting does.** The first cell trains a
tanh network with square loss (4 overlapping classes, 256 points) in three
regimes, tracking the loss, the sharpness and $f_{\rm top}$ for the top-4
eigenspace. That eigenspace comes from a warm-started power iteration and is
approximate (an exact check gave subspace overlap 0.985), so read the curves
as qualitative. The second cell forks unprojected, Dom and Bulk updates from
(a) the SGD run, Song et al.'s regime, and (b) a cross-entropy run on
separable data after 100% training accuracy, the late regime. Look for:
- GD below the edge: $f_{\rm top}$ drops to zero.
- GD at the edge ($\eta=0.5$, $2/\eta=4$): the sharpness starts at 10 and drops within a few steps (a catapult; [`td2_2_edge_of_stability` §2.7](td2_2_edge_of_stability.html#td22-7)), then sits near $2/\eta$ from about step 80. $f_{\rm top}$ stays near 0 until roughly step 180, then jumps to near 1 once an oscillation along the top directions has grown; the loss spikes are bursts of it. "At the edge" and "aligned" are not simultaneous here.
- SGD: $f_{\rm top}$ is high on average (about 0.7 in this single run), driven by noise, far below the edge.
- In (a), Dom stalls (loss near 0.032) while Bulk tracks SGD (Bulk 0.032 to about 0.026, SGD 0.031 to about 0.025 over 250 steps): a modest effect, in a single run. In (b) it is the other way round.
"""
    ),
    code(
        r"""
t0 = time.time()


def top_subspace(gL, theta, V, iters=2):
    # Top-k eigenpairs of the full-batch Hessian by warm-started block power iteration plus Rayleigh-Ritz.
    # gL is the full-batch gradient function; H V is computed by forward-over-reverse differentiation.
    HV = lambda V: vmap(lambda v: jvp(gL, (theta,), (v,))[1], in_dims=1, out_dims=1)(V)
    for _ in range(iters):
        V, _ = torch.linalg.qr(HV(V))
    ev, R = torch.linalg.eigh(V.T @ HV(V))
    return ev.flip(0), (V @ R).flip(1)


def run(net, X, y, loss, theta, lr, steps, B=None, mode="full", k=4, V=None, seed=0, every=1):
    # mode "full": (S)GD.  "dom": update projected onto the top-k Hessian eigenspace.  "bulk": projected off it.
    # The eigenspace is recomputed every `every` steps.  Returns (theta, V, rows of (loss, f_top, sharpness)).
    gL = grad(lambda t: loss(net(t, X), y))
    g = torch.Generator().manual_seed(seed)
    if V is None:
        _, V = top_subspace(gL, theta, torch.linalg.qr(torch.randn(net.P, k, generator=g))[0], iters=20)
    out = []
    for s in range(steps):
        if s % every == 0:
            ev, V = top_subspace(gL, theta, V)
            gf = gL(theta)
            frac = ((V.T @ gf).norm() ** 2 / gf.norm() ** 2).item()   # Gur-Ari et al.'s f_top (full-batch gradient)
        idx = torch.arange(len(X)) if B is None else torch.randint(len(X), (B,), generator=g)
        gb = grad(lambda t: loss(net(t, X[idx]), y[idx]))(theta)
        if mode == "dom":
            gb = V @ (V.T @ gb)
        elif mode == "bulk":
            gb = gb - V @ (V.T @ gb)
        theta = theta - lr * gb
        out.append((loss(net(theta, X), y).item(), frac, ev[0].item()))
    return theta, V, np.array(out)


Xs, ys = make_data(4, n_per=64, sep=1.0)                       # overlapping clusters: not separable at once
nets = MLP(d=20, h=16, C=4, act=torch.tanh)
regimes = {"GD, lr 0.05 (far below the edge)": (0.05, None),
           "GD, lr 0.5 (edge of stability)": (0.5, None),
           "SGD, lr 0.1, batch 8": (0.1, 8)}
states, curves = {}, {}
for name, (lr, B) in regimes.items():
    th, V, out = run(nets, Xs, ys, mse, nets.theta0.clone(), lr, 400, B, every=5)
    states[name], curves[name] = (th, V), out

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for i, (name, out) in enumerate(curves.items()):
    lr = regimes[name][0]
    ax[0].plot(out[:, 1], color=f"C{i}", lw=0.8, label=name)
    ax[1].semilogy(out[:, 0], color=f"C{i}", lw=0.8)
    ax[2].plot(out[:, 2], color=f"C{i}", lw=0.8)
    if 2 / lr < 12:
        ax[2].axhline(2 / lr, color=f"C{i}", ls=":", lw=1)
    print(f"{name}: mean f_top over steps 200-400 = {out[200:, 1].mean():.2f}, "
          f"final sharpness {out[-1, 2]:.2f} (2/lr = {2 / lr:.0f})")
ax[0].set(xlabel="step", ylabel=r"$f_{\rm top}$", title=r"$\|P_4\nabla L\|^2/\|\nabla L\|^2$ (MSE)")
ax[0].legend(fontsize=8)
ax[1].set(xlabel="step", title="training loss")
ax[2].set(xlabel="step", ylim=(0, 12), title=r"sharpness; dotted: $2/\eta=4$ (others: 40, 20)")
fig.tight_layout(); plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    code(
        r"""
t0 = time.time()
labels = {"full": "unprojected", "dom": "Dom (top-4 only)", "bulk": "Bulk (top-4 removed)"}
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))

# (a) Song-Ahn-Yun's setting: SGD, square loss, mid-training.  Fork the SGD run of the previous cell at step 400.
th, V = states["SGD, lr 0.1, batch 8"]
for mode, ls in (("full", "-"), ("dom", "--"), ("bulk", ":")):
    _, _, out = run(nets, Xs, ys, mse, th, 0.1, 250, 8, mode=mode, V=V, seed=1, every=2)
    ax[0].semilogy(out[:, 0], ls, lw=1.2, label=labels[mode])
    print(f"(a) MSE, SGD, {mode:4s}: loss {out[0, 0]:.4f} -> mean of last 50 steps {out[-50:, 0].mean():.4f}")
ax[0].set(title="(a) MSE, SGD lr 0.1, batch 8, forked at step 400", xlabel="steps after the fork", ylabel="training loss")

# (b) Outside that setting: cross-entropy on separable data, small-lr GD, after interpolation.
Xc, yc = make_data(4, n_per=50)
netc = MLP(d=20, h=16, C=4)
thc = netc.theta0.clone()
for _ in range(100):
    thc = thc - 0.05 * grad(lambda t: ce(netc(t, Xc), yc))(thc)
for mode, ls in (("full", "-"), ("dom", "--"), ("bulk", ":")):
    _, _, out = run(netc, Xc, yc, ce, thc, 0.05, 200, None, mode=mode, every=2)
    ax[1].semilogy(out[:, 0], ls, lw=1.2, label=labels[mode])
    print(f"(b) CE, GD,  {mode:4s}: loss {out[0, 0]:.4f} -> {out[-1, 0]:.4f}, mean f_top {out[:, 1].mean():.2f}")
ax[1].set(title="(b) CE, GD lr 0.05, from step 100 (100% train accuracy)", xlabel="further steps")
for a in ax:
    a.legend()
fig.tight_layout(); plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    # ================================================================ 6.5
    markdown(
        r"""
<a id="td26-5"></a>

## 6.5 When the basin is chosen

Fork a run at step $t$, train two children with independent minibatch noise,
and measure the loss on the straight line between them: once there is no
barrier, the run has committed to a basin. Commitment comes early, and in a
ResNet on CIFAR it coincides with the window in which the tangent kernel moves
fastest. In a solvable model the basin is chosen while the relevant kernel
(Gauss–Newton) eigenvalue is still in its early exponential growth.
"""
        + fort("ResNet-20 on CIFAR: a chaotic transient of 2–3 epochs fixes the basin, the same window in which the empirical NTK moves fastest; the kernel then keeps moving slowly.")
        + FRANKLE
        + r"""
In a model linear in its parameters (a fixed kernel, the lazy limit) the loss
is convex, so there is no basin to choose. The smallest model with a choice
is the scalar two-layer net $f=w_1w_2$ with loss $\frac12(w_1w_2-s)^2$,
$s>0$: its minima form two branches ($w_1,w_2>0$ or both $<0$) on either side
of a saddle at the origin. From small balanced initialization, noise fixes the
branch early in the escape from the saddle, while the kernel
$\Theta=w_1^2+w_2^2$ is still one to two orders of magnitude below its final
value. A conserved imbalance $d=w_2^2-w_1^2$ well above the noise fixes the
branch at initialization, even when it is far too small to make the dynamics
lazy.

<a id="td26-5-walkthrough"></a>

### Optional: annotated derivations and walkthrough

*Notation:* $\theta_A,\theta_B$ are the final parameters of two children
trained from a common checkpoint; $\mathcal E$ is the error (test or
training); $\Theta_t=J_tJ_t^\top$ is the empirical tangent-kernel Gram matrix
at step $t$, with $J_t$ the Jacobian of the network's outputs on a data set.

**Operational definition** (Frankle et al.). Copy the parameters at step $t$,
train two children to the end with independent minibatch noise, and evaluate
the error along the straight line between them. The *error barrier* is

$$
\max_{\alpha\in[0,1]}\mathcal E\big((1-\alpha)\theta_A+\alpha\theta_B\big)-\tfrac12\big(\mathcal E(\theta_A)+\mathcal E(\theta_B)\big),
$$

and the run is *committed* (stable, in their terms) at step $t$ if the barrier
is near zero, under 2% in their experiments. With the loss in place of
$\mathcal E$ it is the *loss barrier*, used below.

**What is measured.**
- Frankle et al.: stable at initialization for LeNet, after about 1.5–3% of training for CIFAR ResNet-20/VGG-16, and after about 16–20% for ImageNet ResNet-50/Inception-v3 (test error; on training error ResNet-50 commits later and Inception-v3 not within their runs).
- Fort et al.: in ResNet-20 on CIFAR, commitment within 2–3 epochs coincides with the window in which the kernel distance $1-\langle\Theta_t,\Theta_{t'}\rangle_F/(\|\Theta_t\|_F\|\Theta_{t'}\|_F)$ changes fastest. Training linearized around the epoch-3–4 kernel gives at least 3× lower error than around the initial one.
- A related early-phase signature: in the critical-periods experiments of Achille, Rovere and Soatto, the Fisher information (the Gauss–Newton matrix, for cross-entropy) rises rapidly early in training and then decreases ([`td2_8_further_readings` §4](td2_8_further_readings.html#td28-plasticity)), the same shape as the weighted kernel in Code 6.2.B.

**How curvature enters.** By 6.2, $H_{\rm GN}$ and the loss-weighted kernel
have the same nonzero eigenvalues, so growth of kernel eigenvalues is growth
of curvature outliers; Fort et al.'s unweighted cosine relates to curvature
only through that spectral part.

**The solvable model.** For $f=w_1w_2$ (Exercise 6.1.A, item 3) both branches
compute the same function, so here "the basin" is a choice between symmetric
copies; in real networks the copies are related by permutations and the
choice can also select different functions. The kernel $\Theta=w_1^2+w_2^2$
is also the nonzero eigenvalue of the Gauss–Newton matrix. Gradient flow
conserves the imbalance $d$; *balanced* means $d=0$, and for $|d|\gg s$ the
dynamics are lazy (the kernel barely moves; cf. the linear-network regimes of
[`td1_linear_networks`](td1_linear_networks.html)). Exercise 6.5.A and Code
6.5.B add two findings to the summary above: the barrier between branches is
large while within a branch it is fourth order in the difference; and with
two modes of different strength, commitment happens in two stages, of which
a whole-kernel cosine distance (dominated by the largest eigenvalue)
registers only the first.

So the model supports a spectral statement: *a basin is chosen while the
corresponding kernel (Gauss–Newton) eigenvalue is in its early exponential
growth, before it is an outlier in absolute terms*. That is consistent with
Fort et al.'s timing for the dominant directions without explaining it, and
with the silent alignment of
[Atanasov, Bordelon and Pehlevan](https://arxiv.org/abs/2111.00034): in
linear networks from small initialization, the kernel aligns to the task
while still small, before the loss moves.

**Exercise 6.5.A — Basins of the scalar linear network.** Take
$L=\frac12(w_1w_2-s)^2$ with $s>0$, branches $w_1,w_2>0$ and $w_1,w_2<0$,
kernel $\Theta=w_1^2+w_2^2$ and imbalance $d=w_2^2-w_1^2$. The *loss barrier*
between two endpoints is the maximum loss on the straight line between them
minus the mean of the endpoint losses; Fort et al.'s kernel distance between
times $t,t'$ is $1-\cos(\Theta_t,\Theta_{t'})$.

1. Let the children end on opposite branches, at $(a,s/a)$ and $(-b,-s/b)$ with $a,b>0$. Show that the loss along the linear path has barrier $\frac{s^2}2\big(\frac{(a+b)^2}{4ab}\big)^2\ge\frac{s^2}2$.
2. On the same branch, at $(a,s/a)$ and $(b,s/b)$, show $w_1w_2=s\big[1+\alpha(1-\alpha)\frac{(a-b)^2}{ab}\big]$ along the path, and a barrier of $\frac{s^2(a-b)^4}{32a^2b^2}$. Evaluate it for $s=1$, $a=1.1$, $b=0.9$.
3. **Choosing the branch.** Gradient flow conserves $w_2^2-w_1^2$. Near the origin, use $u=(w_1+w_2)/\sqrt2$ and $v=(w_1-w_2)/\sqrt2$, and linearize:
   - show that $\dot u=su$ and $\dot v=-sv$;
   - argue that from a small random initialization with gradient noise, the branch is fixed by the sign of $u$ once its growth outruns the noise;
   - show that this happens while the loss is still close to its initial value $s^2/2$, early in a window in which $\Theta$ grows by orders of magnitude;
   - *(optional)* with $du=su\,dt+\sigma\,dW$ and $u(0)\sim\mathcal N(0,a_0^2)$, show that two children forked at time $t$ end on opposite branches with probability $\arccos\rho(t)/\pi$, where $\rho(t)$ is the correlation of the two children's limits of $e^{-st}u(t)$ as $t\to\infty$ (the "bracket" of the hint), and show that $\rho(t)=1-\frac{q}{a_0^2+q}e^{-2st}$ with $q=\sigma^2/2s$ (Code 6.5.B plots this prediction).
4. **Convexity.** For a model linear in $\theta$ with a convex loss, show that the barrier is at most $\frac12|L(\theta_A)-L(\theta_B)|$.
5. **Imbalance removes the choice.** Under gradient flow, suppose $d=w_2(0)^2-w_1(0)^2>0$ with $w_1(0)^2\ll d$.
   - Show that $w_2$ never changes sign, so the branch is $\mathrm{sign}\,w_2(0)$, fixed at initialization.
   - Show that at the minimum reached $\Theta=\sqrt{d^2+4s^2}$, while initially $\Theta\approx d$. Conclude that two thresholds differ: imbalance well above the noise fixes the branch, while laziness (a nearly frozen kernel) needs $d\gg s$. In between, the kernel can change by a large factor (about $2s/d$ when $d\ll s$) although nothing is being chosen.
6. **Two modes.** Take $L=\sum_{m=1,2}\frac12(w_1^{(m)}w_2^{(m)}-s_m)^2$ with $s_1>s_2$: two decoupled modes of a linear network on two orthonormal input directions, with kernel $\Theta=\mathrm{diag}(\Theta_1,\Theta_2)$, $\Theta_m=(w_1^{(m)})^2+(w_2^{(m)})^2$.
   - From small balanced initialization, when is each mode's branch chosen?
   - Compare the time at which each mode's branch is fixed with the time at which $\Theta_m$ grows from its small initial value to $O(s_m)$.
   - The direction $\Theta/\|\Theta\|_F$ ends at $(s_1,s_2)/\sqrt{s_1^2+s_2^2}$. When does Fort et al.'s kernel distance register mode 2, and why does it barely register mode 2's commitment?

<details><summary>Hint</summary>

For 1, both coordinates change sign along the path, at $\alpha=a/(a+b)$ and $\alpha=b/(a+b)$; between these $w_1w_2<0$. Write $w_1w_2$ as a product of two linear functions of $\alpha$ and minimize it.

For 3, $\Theta$ is $\|\nabla f\|^2=w_2^2+w_1^2=u^2+v^2$. Without stochastic calculus: in discrete time near the origin, $u_{k+1}=(1+\eta s)u_k+\sqrt\eta\,\sigma\zeta_k$ with independent standard normal $\zeta_k$. Unroll: $u_k=(1+\eta s)^k\big[u_0+\sqrt\eta\,\sigma\sum_{j<k}(1+\eta s)^{-j-1}\zeta_j\big]$; later noise enters the bracket suppressed by $(1+\eta s)^{-j}$. For the optional part, the bracket at the fork is shared and the later noise is independent; for jointly Gaussian, mean-zero $X,Y$ with correlation $\rho$, $\Pr(\mathrm{sign}X\ne\mathrm{sign}Y)=\arccos\rho/\pi$ (Sheppard's formula).

For 4, use $L(\alpha)\le(1-\alpha)L_A+\alpha L_B\le\max(L_A,L_B)$. For 5, $w_2^2=w_1^2+d\ge d$; at the minimum solve $w_1w_2=s$, $w_2^2-w_1^2=d$. For 6, each mode is item 3 with its own $s_m$: its branch is fixed after a time of order $1/s_m$, while its eigenvalue, seeded by $a_{0,m}^2+q_m$ with $q_m=\sigma^2/2s_m$ (item 3), needs a time of order $s_m^{-1}\log\big(s_m/(a_{0,m}^2+q_m)\big)$ to grow.
</details>
"""
    ),
    answer(
        r"""
**1.** Along the path,
$w_1(\alpha)=(1-\alpha)a-\alpha b$ and $w_2(\alpha)=s\big[(1-\alpha)/a-\alpha/b\big]$. The product factorizes:

$$
w_1w_2=\frac{s}{ab}\big(\alpha(a+b)-a\big)\big(\alpha(a+b)-b\big).
$$

It vanishes at $\alpha=a/(a+b)$ and $\alpha=b/(a+b)$ and is negative between them. Its minimum is at the midpoint of the roots, $\alpha=\frac12$, where

$$
w_1w_2=\frac{s}{ab}\cdot\frac{b-a}{2}\cdot\frac{a-b}{2}=-\frac{s(a-b)^2}{4ab}.
$$

The loss there is $\frac12\big(s+\frac{s(a-b)^2}{4ab}\big)^2=\frac{s^2}2\Big(\frac{(a+b)^2}{4ab}\Big)^2$. The endpoint losses are zero, so this is the barrier. It is at least $s^2/2$ (by AM–GM, $(a+b)^2\ge4ab$), with equality iff $a=b$: at least the loss at the origin.

**2.** On the same branch,
$w_1w_2=\big[(1-\alpha)a+\alpha b\big]\,s\big[(1-\alpha)/a+\alpha/b\big]=s\big[(1-\alpha)^2+\alpha^2+\alpha(1-\alpha)(a/b+b/a)\big]$.
Since $(1-\alpha)^2+\alpha^2=1-2\alpha(1-\alpha)$ and $a/b+b/a-2=(a-b)^2/(ab)$, this is $s\big[1+\alpha(1-\alpha)(a-b)^2/(ab)\big]$. The loss $\frac12s^2\alpha^2(1-\alpha)^2(a-b)^4/(ab)^2$ is largest at $\alpha=\frac12$, giving a barrier of $\frac{s^2(a-b)^4}{32a^2b^2}$. For $s=1$, $a=1.1$, $b=0.9$: $(0.2)^4/(32\times0.9801)\approx5.1\times10^{-5}$, four orders of magnitude below the opposite-branch barrier.

**3.** The gradient is $\nabla L=r\,(w_2,w_1)$ with $r=w_1w_2-s$. Near the origin $r\approx-s$, so $\dot w_1=sw_2$ and $\dot w_2=sw_1$, hence $\dot u=su$ and $\dot v=-sv$: the origin is a saddle, unstable along $u$ and stable along $v$. Exactly, $\frac{d}{dt}(w_2^2-w_1^2)=2w_2(-rw_1)-2w_1(-rw_2)=0$.

From an initialization of size $a_0$, add gradient noise: $du=su\,dt+\sigma\,dW$ (similarly for $v$, which stays small).
- The solution is $u(t)=e^{st}\big[u(0)+\sigma\int_0^te^{-st'}dW_{t'}\big]$ (in discrete time, the unrolled sum of the hint). The bracket converges (its noise part has variance $\frac{\sigma^2}{2s}(1-e^{-2st})$), so after a few multiples of $1/s$ it stops changing, and later noise is irrelevant against the growing factor $e^{st}$.
- The sign of the bracket is therefore decided early. Since $v$ stays at the noise scale (stationary variance $\sigma^2/2s$) while $u$ grows, it is the common sign of $w_1$ and $w_2$ at the end: the branch.
- During the escape the loss is $\frac12(w_1w_2-s)^2\approx\frac12(\frac{u^2-v^2}{2}-s)^2\approx\frac{s^2}2$ until $u^2$ becomes comparable to $s$.
- Meanwhile $\Theta=u^2+v^2$ grows like $e^{2st}$, from $O(a_0^2)$ to $O(s)$.

So the branch is chosen early in the escape, while $\Theta$ is still far below its final size and before the loss moves much; most of the kernel's growth comes afterwards.

*Optional part.* In continuous time the limiting bracket is $\beta_\infty=u(0)+\sigma\int_0^\infty e^{-st'}dW_{t'}$, Gaussian with variance $a_0^2+q$, where $q=\sigma^2/2s$. Two children forked at $t$ share $u(0)$ and the noise up to $t$ (variance $a_0^2+q(1-e^{-2st})$) and add independent noise afterwards (variance $qe^{-2st}$ each). Their limiting brackets are jointly Gaussian with correlation

$$
\rho(t)=\frac{a_0^2+q(1-e^{-2st})}{a_0^2+q}=1-\frac{q}{a_0^2+q}\,e^{-2st},
$$

and by Sheppard's formula they end on opposite branches with probability $\arccos\rho(t)/\pi$. For a disagreement tolerance $\epsilon$, commitment takes

$$
t_\epsilon=\frac1{2s}\log\frac{q/(a_0^2+q)}{1-\cos\pi\epsilon}\quad(\text{or }0\text{ if negative}),
$$

a time of order $1/s$ that depends on the noise-to-initialization ratio $q/a_0^2$ but not on their common scale. The escape is seeded by $a_0^2+q$: $\mathbb E\,u(t)^2=(a_0^2+q)e^{2st}-q$, so $\Theta$ reaches $O(s)$ after a time of order $s^{-1}\log\big(s/(a_0^2+q)\big)$. When initialization and noise are both small, this is logarithmically later than commitment, so the choice is made while $\Theta$ is still well below its final value (though already growing). After the escape, children differ only by small amounts of imbalance, and by item 2 their barrier is fourth order in that difference.

**4.** With $L$ convex in $\theta$, $L((1-\alpha)\theta_A+\alpha\theta_B)\le(1-\alpha)L_A+\alpha L_B\le\max(L_A,L_B)$. So the barrier is at most $\max(L_A,L_B)-\frac12(L_A+L_B)=\frac12|L_A-L_B|$, small as soon as both children are trained. A fixed-kernel model has nothing to choose.

**5.** The imbalance is conserved (item 3), so $w_2^2=w_1^2+d\ge d>0$ for all time: $w_2$ is continuous and never zero, so it keeps its initial sign, and at the end $w_1w_2=s>0$ gives $w_1$ the same sign. The branch is $\mathrm{sign}\,w_2(0)$, decided at initialization. (Conservation is exact for gradient flow only. With small noise and a discrete step the imbalance drifts slightly, but as long as it stays well above zero no child changes branch, which is why the imbalanced runs in the code show no opposite-branch pairs, only tiny within-branch barriers as in item 2.)

At the minimum, $w_1^2w_2^2=s^2$ and $w_2^2-w_1^2=d$, so $w_1^2=\frac{-d+\sqrt{d^2+4s^2}}2$, $w_2^2=\frac{d+\sqrt{d^2+4s^2}}2$ and $\Theta=\sqrt{d^2+4s^2}$. Initially $\Theta=d+2w_1(0)^2\approx d$. The ratio $\sqrt{1+4s^2/d^2}$ tends to 1 for $d\gg s$: the kernel is nearly frozen, the lazy regime. For $d\asymp s$ it changes by an $O(1)$ factor, and for $d\ll s$ by a factor of about $2s/d$, although the branch was fixed at initialization: a moving kernel does not by itself mean a basin is being chosen. What matters is whether the dynamics pass near the symmetric saddle, which needs $|d|$ comparable to the noise, a much smaller threshold than the $d\gg s$ needed for laziness.

**6.** The loss is a sum of two copies of the scalar problem, so each mode follows item 3 with its own $s_m$. Mode $m$'s branch is fixed once its bracket has settled, after a few times $1/s_m$, and its eigenvalue $\Theta_m\approx u_m^2$ grows like $e^{2s_mt}$, seeded by $a_{0,m}^2+q_m$ with $q_m=\sigma^2/2s_m$: the linearized $\mathbb E\,u_m^2$ reaches $s_m$ at $t=\frac1{2s_m}\log\frac{s_m+q_m}{a_{0,m}^2+q_m}$. So each mode commits during its eigenvalue's early exponential growth, while it is still one to two orders of magnitude below its final value; the logarithm makes most of the growth come after the choice. With $s_1>s_2$, commitment happens in two stages, mode 2 later by a factor of about $s_1/s_2$.

The kernel distance $1-\cos(\Theta_t,\Theta_{t'})$ only sees changes in the *direction* of $(\Theta_1,\Theta_2)$. While mode 1 escapes, $\Theta_1$ outgrows $\Theta_2$ and the direction swings toward $(1,0)$: that window overlaps mode 1's commitment, so for the dominant mode the cosine does register the commitment window. Afterwards $\Theta_1\gg\Theta_2$ and the direction is about $(1,0)$ however $\Theta_2$ fluctuates, so mode 2's commitment, made while $\Theta_2$ is tiny, is nearly invisible. The direction moves again only when $\Theta_2$ grows to a sizeable fraction of $\Theta_1$, ending at $(s_1,s_2)/\sqrt{s_1^2+s_2^2}$: after mode 2 has committed. A whole-kernel cosine is dominated by the top eigenvalue and is blind to weak modes; the per-mode eigenvalues are the informative quantity.
"""
    ),
    markdown(
        r"""
**Code 6.5.B — Forks of a two-mode linear network, and imbalance.** The model
of Exercise 6.5.A, item 6, with $s_1=1$, $s_2=0.25$, trained by noisy GD
$w\leftarrow w-\eta\nabla L+\sqrt\eta\,\sigma\xi$ ($\xi$ standard normal,
$\eta=0.05$, $\sigma=0.02$) from three starts: small balanced (rich),
$w_2(0)=\pm0.3$ ($d=0.09$: committed but not lazy), and $w_2(0)=\pm3$ (lazy).
For 60 parents per setting it forks 4 pairs of children at several times;
children train 1000 steps, with the noise switched off for the last 200. It
reports the loss barrier, the fraction of pairs on opposite branches per mode
(against $\arccos\rho(t)/\pi$ from item 3), the per-mode $\Theta_m$, and Fort
et al.'s kernel velocity (cosine distance ten steps apart, per step). This
shows a mechanism that makes commitment early in a model where every step can
be followed; it is not evidence that the same mechanism operates in a ResNet.
Look for:
- Balanced: disagreement roughly follows the prediction (tails off by a few percentage points either way, e.g. mode 1 at step 60: 0.05 measured vs 0.02 predicted; mode 2 at step 120: 0.03 vs 0.10). Mode 1 commits by about step 60–100, mode 2 by about step 120 (a few percent of pairs persist to step 200).
- Over steps 0–60, $\Theta_1$ grows about a hundredfold but stays tens of times below its final $2s_1$; mode 2 commits while $\Theta_2$ is a few percent of $2s_2$.
- Kernel velocity: noise for about 20 steps, then mode 1's escape (steps 20–60, overlapping its commitment), at its minimum (about $10^{-7}$ per step) while mode 2 commits, rising again over steps 150–300 as $\Theta_2$ catches up.
- Imbalanced: no opposite-branch pairs; within-branch barriers of $10^{-5}$ to $10^{-6}$. With $w_2(0)=\pm0.3$ the kernel still moves a lot early, so a moving kernel does not imply a choice; with $\pm3$ it barely moves.
"""
    ),
    code(
        r"""
t0 = time.time()
rng = np.random.default_rng(0)
s_modes = np.array([1.0, 0.25])            # two decoupled modes of a linear net, s1 > s2
eta, sigma, a0 = 0.05, 0.02, 0.01          # step size, gradient-noise level, init scale


def noisy_gd(w, steps, noise=sigma):
    # w[..., m, 0], w[..., m, 1]: first- and second-layer weight of mode m.  Loss sum_m (w1 w2 - s_m)^2 / 2.
    for _ in range(steps):
        r = w[..., 0] * w[..., 1] - s_modes
        g = np.stack([r * w[..., 1], r * w[..., 0]], -1)
        w = w - eta * g + np.sqrt(eta) * noise * rng.standard_normal(w.shape)
    return w


def loss(w):
    return 0.5 * ((w[..., 0] * w[..., 1] - s_modes) ** 2).sum(-1)


def kernel(w):
    # The NTK Gram matrix on the two (orthonormal) input directions is diagonal: Theta = diag(w1^2 + w2^2).
    return (w ** 2).sum(-1)


def init(n, w2_scale):
    w = a0 * rng.standard_normal((n, 2, 2))
    if w2_scale > a0:                        # imbalanced: second layer starts at +-w2_scale
        w[..., 1] = w2_scale * np.sign(rng.standard_normal((n, 2)))
    return w


setups = {"balanced, init 0.01": a0, "imbalanced, w2(0) = ±0.3": 0.3, "strongly imbalanced (lazy), w2(0) = ±3": 3.0}
forks = [0, 10, 20, 40, 60, 80, 120, 160, 200, 300, 400]
n_par, n_pair, T = 60, 4, 1000
alpha = np.linspace(0, 1, 21).reshape(-1, 1, 1, 1, 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for j, (name, w2s) in enumerate(setups.items()):
    parents, last, rows = init(n_par, w2s), 0, []
    for tf in forks:
        parents, last = noisy_gd(parents, tf - last), tf
        kids = [noisy_gd(noisy_gd(np.repeat(parents[:, None], n_pair, 1), T - 200), 200, noise=0.0) for _ in range(2)]
        Lp = loss((1 - alpha) * kids[0] + alpha * kids[1])
        barrier = Lp.max(0) - 0.5 * (Lp[0] + Lp[-1])
        opposite = (np.sign(kids[0][..., 0]) != np.sign(kids[1][..., 0])).mean((0, 1))   # per mode
        rows.append((tf, barrier.mean(), *opposite, np.median(loss(parents))))
    rows = np.array(rows)
    print(name)
    for tf, b, o1, o2, l in rows:
        print(f"  fork at step {int(tf):3d}: mean loss barrier {b:.1e}; pairs on opposite branches: "
              f"mode 1 {o1:.2f}, mode 2 {o2:.2f}; parent loss {l:.3f}")
    ls = ["-", "--", ":"][j]
    if j == 0:
        tt = np.linspace(0, forks[-1], 200)
        for m, mk in ((0, "o"), (1, "s")):
            ax[0].plot(rows[:, 0], rows[:, 2 + m], mk, color=f"C{m}", label=f"mode {m + 1} ($s={s_modes[m]}$), measured")
            q = sigma**2 / (2 * s_modes[m])                    # linearized-escape prediction (Exercise 6.5.A, item 3)
            rho = 1 - q / (a0**2 + q) * np.exp(-2 * s_modes[m] * eta * tt)
            ax[0].plot(tt, np.arccos(rho) / np.pi, "-", color=f"C{m}", lw=1, label=r"$\arccos\rho(t)/\pi$")

    # Kernel velocity along fresh parent trajectories: Fort et al.'s cosine distance per step, 10 steps apart.
    w, vel, eig = init(200, w2s), [], []
    for k in range(40):
        K1 = kernel(w); w = noisy_gd(w, 10); K2 = kernel(w)
        cos = (K1 * K2).sum(-1) / np.linalg.norm(K1, axis=-1) / np.linalg.norm(K2, axis=-1)
        vel.append(np.median(1 - cos) / 10); eig.append(np.median(K2, 0))
    eig = np.array(eig)
    if j < 2:
        for m in range(2):
            ax[1].semilogy(10 * np.arange(1, 41), eig[:, m], ls, color=f"C{m}", label=f"$\\Theta_{m + 1}$, {name}")
    ax[2].semilogy(10 * np.arange(1, 41), np.array(vel) + 1e-9, ls, color="C2", label=name)
    print(f"  kernel velocity, median over 200 runs, steps 10/100/200/300/400: {np.round(np.array(vel)[[0, 9, 19, 29, 39]], 7)}")
ax[0].set(xlabel="fork step", ylabel="fraction of child pairs", title="balanced init: children on opposite branches")
ax[0].legend(fontsize=7)
ax[1].set(xlabel="step", title=r"per-mode kernel eigenvalues $\Theta_m=w_1^2+w_2^2$ (median)"); ax[1].legend(fontsize=7)
ax[2].set(xlabel="step", title="whole-kernel velocity: (1 - cos) per step"); ax[2].legend(fontsize=7)
fig.tight_layout(); plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    # ================================================================ readings
    markdown(
        r"""
<a id="td26-readings"></a>

## Readings

The lead papers are at the top of each section. Also:
[Ghorbani, Krishnan and Xiao 2019](https://arxiv.org/abs/1901.10159) (full Hessian densities of ImageNet-scale networks by stochastic Lanczos; without BatchNorm, outliers appear early and hold the gradient);
[Atanasov, Bordelon and Pehlevan 2022](https://arxiv.org/abs/2111.00034) (silent alignment, the matrix version of 6.5's timing);
[Sagun, Bottou and LeCun 2016](https://arxiv.org/abs/1611.07476);
Papyan [2018](https://arxiv.org/abs/1811.07062) and [2019](https://arxiv.org/abs/1901.08244).
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "td2_6_curvature", "td2_6_curvature"))
