#!/usr/bin/env python3
"""Build NB2 §5: Normalization, weight decay, and instabilities.

Shape (streamlined 2026-10-09, see ../reviews/2026-10-09-td-streamline-brief.md): a short
pitch with the two key papers (Kosson et al.; Wortsman et al.) and a route; then four
sections (5.1 scale invariance, 5.2 weight-decay equilibrium, 5.3 loss spikes, 5.4
clipping), each opening with its question, answer and lead paper, followed by one
"Optional: annotated derivations and walkthrough" subsection (anchor s5N-walkthrough)
holding the exercises (bold labels, not headings), solutions and code; a one-line
"Also:" reading list closes the page.

Closed forms are checked in checks/td2_5_normalization_weight_decay_checks.py.
Source verification and critique rounds: ../reviews/td/td2_5_normalization_weight_decay.md.
"""
from nbkit_td import answer, code, code_answer, markdown, reading, run

CELLS = [
    # ================================================================== title and pitch
    markdown(
        r"""
# NB2 §5 — Normalization, weight decay, and instabilities

**{{VARIANT}} · Training Dynamics, Iliad Intensive 2026**

Most weights in a modern network feed a normalization layer, so the loss sees only their
direction. The learning rate $\eta$ and weight decay $\lambda$ then act only through the
weight norm: they set an effective step size $\eta/\|w\|^2$, and at equilibrium each block
turns by about $\sqrt{2\eta\lambda}$ radians per step, whatever its gradient scale. The
second half asks why large runs spike, and what gradient clipping is for.

"""
        + reading(
            "Rotational equilibrium: how weight decay balances learning across neural networks",
            "https://arxiv.org/abs/2305.17212",
            "Kosson, Messmer & Jaggi (ICML 2024)",
            "Equilibrium norms and per-step rotations for SGD with momentum, AdamW, Adam+L² and Lion (Table 1), measured in ResNets and GPT-2; the backbone of 5.1–5.2.",
            importance=3,
            length=3,
        )
        + reading(
            "Small-scale proxies for large-scale Transformer training instabilities",
            "https://arxiv.org/abs/2309.14322",
            "Wortsman et al. (ICLR 2024)",
            "Reproduces attention-logit growth and output-logit divergence in small transformers at high learning rate, and removes them with QK-layernorm and z-loss; the core reading for 5.3.",
            importance=4,
            length=3,
        )
        + r"""

**Route.** Each section opens with its question, answer and paper; the derivations, exercises
and code sit under "Optional: annotated derivations and walkthrough".
- [5.1 Scale invariance](#s51): one step is exactly a step on the sphere with learning rate $\eta/\lVert w\rVert^2$.
- [5.2 The weight-decay equilibrium](#s52): SGD and AdamW; only $\eta\lambda$ matters; weight decay as an exponential learning-rate schedule.
- [5.3 Loss spikes](#s53): four mechanisms, and which are edge-of-stability phenomena.
- [5.4 Clipping](#s54): the right step when curvature grows with the gradient.
- [Readings](#s5r)
"""
    ),
    code(
        r"""
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)   # the models here are tiny; extra threads only add overhead
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
"""
    ),
    # ================================================================== 5.1
    markdown(
        r"""
<a id="s51"></a>

## 5.1 Scale invariance

Take one block of weights $w$, with norm $r=\|w\|$ and direction $u=w/r$, that feeds a
LayerNorm, RMSNorm or BatchNorm. Then the loss obeys $L(cw)=L(w)$ for every $c>0$: it is a
function on the unit sphere. Its gradient is orthogonal to $w$ and shrinks like $1/r$, its
sharpness (top Hessian eigenvalue) shrinks like $1/r^2$, and one SGD step with learning rate
$\eta$ and weight decay $\lambda$ is *exactly* a gradient step on the sphere (then
renormalized) with the effective learning rate $\eta_{\rm eff}=\eta/((1-\eta\lambda)r^2)$.

"""
        + reading(
            "L2 regularization versus batch and weight normalization",
            "https://arxiv.org/abs/1706.05350",
            "van Laarhoven (2017)",
            "A short note: on normalized weights, L² regularization cannot change the function; it only sets the weight scale, and with it the effective learning rate $\\eta/\\|w\\|^2$.",
            importance=3,
            length=1,
        )
        + r"""

**How far this reaches.** In a BatchNorm convnet almost every weight is scale-invariant
(BatchNorm's invariance is per output channel, i.e. per row of $W$); the exceptions are the
final classifier, biases and the norm layers' gains. In a pre-LN transformer few matrices
are *exactly* invariant: scaling an attention or MLP output projection changes that block's
share of the residual stream. QK-layernorm (5.3) makes each head's $W_Q,W_K$ invariant.
Everything below is exact for scale-invariant blocks and a model elsewhere.

<a id="s51-walkthrough"></a>

### Optional: annotated derivations and walkthrough

**Setup.** LayerNorm and RMSNorm divide a vector by its own scale, so
$\mathrm{LN}(c\,h)=\mathrm{LN}(h)$ for $c>0$, exactly when the stabilizer $\epsilon$ is zero
and to excellent approximation when $h$'s variance (for RMSNorm, mean square) is
$\gg\epsilon$. If $h=Wx$ feeds it, the loss is unchanged by $W\to cW$. Notation:
$S(w)=\lambda_{\max}(\nabla^2L(w))$ is the sharpness; $\lambda$ is always the weight decay. A
reference like "5.1.2.3" means Exercise 5.1.2, part 3.

**Exercise 5.1.1 — five identities.** Let $L$ be scale-invariant and twice differentiable,
with $u=w/\|w\|$. Show:

1. $\langle\nabla L(w),w\rangle=0$: the gradient is tangent to the sphere through $w$.
2. $\nabla L(cw)=\nabla L(w)/c$.
3. $\nabla^2L(cw)=\nabla^2L(w)/c^2$. Hence $S(w)=S(u)/\|w\|^2$: shrinking the weights makes the loss sharper.
4. $\nabla^2L(w)\,w=-\nabla L(w)$. In particular $w$ is a null direction of the Hessian at every critical point.
5. Under gradient flow $\dot w=-\nabla L(w)$, $\|w\|$ is conserved.

(The Mysteries day's [WB6 §10](../notebooks/06_mysteries_myriad.html#wb6-section-10), Exercise 10.1, derives 1–4 for batch normalization.)

<details><summary>Hint</summary>

Differentiate the identity $L(cw)=L(w)$, once with respect to $c$ at $c=1$ and once (and
twice) with respect to $w$. For 4, differentiate the identity from 1 with respect to $w$.
</details>
"""
    ),
    answer(
        r"""
**1.** Differentiate $L(cw)=L(w)$ with respect to $c$: $\langle\nabla L(cw),w\rangle=0$. Set $c=1$.

**2.** Differentiate $L(cw)=L(w)$ with respect to $w$. The chain rule gives $c\,\nabla L(cw)=\nabla L(w)$, so $\nabla L(cw)=\nabla L(w)/c$. The gradient is homogeneous of degree $-1$.

**3.** Differentiate $c\,\nabla L(cw)=\nabla L(w)$ with respect to $w$: $c^2\,\nabla^2L(cw)=\nabla^2L(w)$. With $c=1/\|w\|$, $\nabla^2L(w)=\nabla^2L(u)/\|w\|^2$, so every eigenvalue, in particular the largest, scales as $1/\|w\|^2$: $S(w)=S(u)/\|w\|^2$. Here $S(u)$ is the sharpness at the unit-norm point; it depends only on the direction.

**4.** Differentiate $\sum_iw_i\,\partial_iL(w)=0$ with respect to $w_j$: $\partial_jL+\sum_iw_i\,\partial_i\partial_jL=0$, i.e. $\nabla^2L(w)\,w=-\nabla L(w)$. At a critical point the right side vanishes. (This is Euler's theorem for the degree-$(-1)$ homogeneous function $\nabla L$.)

**5.** $\frac{d}{dt}\|w\|^2=2\langle w,\dot w\rangle=-2\langle w,\nabla L(w)\rangle=0$ by 1. So any change in $\|w\|$ under gradient descent is a *discrete-step* effect, second order in $\eta$ per step. Gradient flow cannot see it. The first correction of NB2 §1's modified flow, $\dot w=-\nabla L-\tfrac\eta2\nabla^2L\,\nabla L$, does: with 1 and 4, $\frac{d}{dt}\|w\|^2=-\eta\langle w,\nabla^2L\,\nabla L\rangle=\eta\|\nabla L\|^2$, which over one step of duration $\eta$ is the growth $r_{k+1}^2-r_k^2=\eta^2G_k^2/r_k^2$ that 5.1.2 finds exactly (at $\lambda=0$; there $r_k=\|w_k\|$ and $G_k=\|\nabla L(u_k)\|$ is the gradient norm at the unit-norm point, so $\|\nabla L(w_k)\|=G_k/r_k$).
"""
    ),
    markdown(
        r"""
The next cell checks 1–4 numerically, by autograd, on a two-layer net whose first-layer
weights feed a LayerNorm (with $\epsilon=0$, so the invariance is exact).
"""
    ),
    code(
        r"""
torch.manual_seed(0)
X, Y = torch.randn(32, 8, dtype=torch.float64), torch.randn(32, 1, dtype=torch.float64)
W2 = torch.randn(1, 16, dtype=torch.float64) / 4

def layer_norm(h):                      # LayerNorm with eps = 0 and no gain: exactly scale-invariant
    h = h - h.mean(1, keepdim=True)
    return h / h.pow(2).mean(1, keepdim=True).sqrt()

def loss(W1_flat):                      # L(W1) for a 8 -> 16 -> LN -> tanh -> 1 network
    h = layer_norm(X @ W1_flat.view(16, 8).T)
    return 0.5 * ((torch.tanh(h) @ W2.T - Y) ** 2).mean()

w = torch.randn(128, dtype=torch.float64)
grad = lambda v: torch.autograd.functional.jacobian(loss, v)
hess = lambda v: torch.autograd.functional.hessian(loss, v)
g, H = grad(w), hess(w)
sharp = lambda M: torch.linalg.eigvalsh(M)[-1].item()

print(f"L(3w) - L(w)                      = {(loss(3 * w) - loss(w)).item():.1e}")
print(f"cos(grad L(w), w)                 = {(g @ w / g.norm() / w.norm()).item():.1e}")
print(f"|3 grad L(3w) - grad L(w)|        = {(3 * grad(3 * w) - g).norm().item():.1e}")
print(f"|H(w) w + grad L(w)|              = {(H @ w + g).norm().item():.1e}")
print(f"S(w) / S(2w)  (expect 4)          = {sharp(H) / sharp(hess(2 * w)):.6f}")
"""
    ),
    markdown(
        r"""
**Exercise 5.1.2 — one step, exactly.** What does one step of SGD with weight decay do to
the direction and to the norm? It is GD on the unit sphere with a step size set by the norm,
exactly, not just for small steps; the exactness is what makes 5.2 work.

Write $w_k=r_ku_k$ with $r_k=\|w_k\|$, $\|u_k\|=1$, and take one step of SGD with learning
rate $\eta$ and weight decay $\lambda$, $w_{k+1}=(1-\eta\lambda)w_k-\eta\nabla L(w_k)$, with
$0\le\eta\lambda<1$. (Read $\nabla L$ as a minibatch gradient; each minibatch loss is itself
scale-invariant.) Write $G_k=\|\nabla L(u_k)\|$ for the gradient norm *at the unit-norm
point*, so that $\|\nabla L(w_k)\|=G_k/r_k$. Recall that GD on a quadratic of curvature $h$
is stable iff $\eta h<2$.

1. Show that the new direction is exactly
$$
u_{k+1}=\frac{u_k-\eta_{\mathrm{eff},k}\nabla L(u_k)}{\big\|u_k-\eta_{\mathrm{eff},k}\nabla L(u_k)\big\|},
\qquad
\eta_{\mathrm{eff},k}=\frac{\eta}{(1-\eta\lambda)\,r_k^2}.
$$
This is a gradient step on the sphere followed by renormalization, with learning rate $\eta_{\mathrm{eff}}$. Without weight decay, $\eta_{\mathrm{eff}}=\eta/r_k^2$.
2. Show that the norm obeys the exact recursion
$$
r_{k+1}^2=(1-\eta\lambda)^2r_k^2+\frac{\eta^2G_k^2}{r_k^2}.
$$
3. Let $\theta_k$ be the angle between $w_k$ and $w_{k+1}$. Show that $\cos\theta_k=(1-\eta\lambda)\,r_k/r_{k+1}$ and $\tan\theta_k=\eta_{\mathrm{eff},k}G_k$.
4. Let $u^*$ be a minimum of $L$ on the sphere. Show that the condition for the direction dynamics to be linearly stable at $u^*$, $\eta_{\mathrm{eff}}\,S(u^*)<2$, is the same as the Euclidean condition $\eta\,S(w)<2$ (up to the factor $1-\eta\lambda$): "effective learning rate" and "sharpness of $w$" are two ways of saying the same thing.

<details><summary>Hint</summary>

Use $\nabla L(w_k)=\nabla L(u_k)/r_k$ and factor $(1-\eta\lambda)r_k$ out of $w_{k+1}$.
Then use orthogonality: $\|u-a\nabla L(u)\|^2=1+a^2G^2$. For 3, compute $\langle w_k,w_{k+1}\rangle$ in two ways. For 4, near $u^*$ the direction dynamics are GD on $L$ restricted to the sphere, whose Hessian in tangent directions at $u^*$ has top eigenvalue $S(u^*)$ (by 5.1.1.4 the radial direction is a null direction).
</details>
"""
    ),
    answer(
        r"""
**1.** By 5.1.1.2, $\nabla L(w_k)=\nabla L(u_k)/r_k$. So
$$
w_{k+1}=(1-\eta\lambda)r_ku_k-\frac{\eta}{r_k}\nabla L(u_k)
=(1-\eta\lambda)r_k\Big[u_k-\frac{\eta}{(1-\eta\lambda)r_k^2}\nabla L(u_k)\Big].
$$
The prefactor is a positive scalar (this is where $\eta\lambda<1$ is used), so the direction of $w_{k+1}$ is the direction of the bracket, which is the claimed formula. Nothing was expanded in $\eta$; the statement is exact.

**2.** The bracket has $\|u_k-\eta_{\rm eff}\nabla L(u_k)\|^2=1+\eta_{\rm eff}^2G_k^2$, because $\nabla L(u_k)\perp u_k$ (5.1.1.1). So
$$
r_{k+1}^2=(1-\eta\lambda)^2r_k^2\big(1+\eta_{\rm eff}^2G_k^2\big)=(1-\eta\lambda)^2r_k^2+\frac{\eta^2G_k^2}{r_k^2}.
$$
Geometrically this is Pythagoras: the decayed vector $(1-\eta\lambda)w_k$ and the step $-\eta\nabla L(w_k)$ are orthogonal.

**3.** $\langle w_k,w_{k+1}\rangle=(1-\eta\lambda)r_k^2$, since the gradient term is orthogonal to $w_k$. Dividing by $r_kr_{k+1}$ gives $\cos\theta_k=(1-\eta\lambda)r_k/r_{k+1}$. The component of $w_{k+1}$ along $u_k$ is $(1-\eta\lambda)r_k$ and the orthogonal component has length $\eta G_k/r_k$, so $\tan\theta_k=\eta G_k/((1-\eta\lambda)r_k^2)=\eta_{\rm eff,k}G_k$.

**4.** Near $u^*$, part 1 is gradient descent on the sphere with step $\eta_{\rm eff}$; linearizing in the tangent coordinates, the iteration matrix is $I-\eta_{\rm eff}\nabla^2_{\rm tan}L(u^*)$ (the renormalization contributes only at second order, because the gradient vanishes at $u^*$). It is stable iff $\eta_{\rm eff}S(u^*)<2$. By 5.1.1.3 and 5.1.1.4, the Euclidean Hessian at $w=r u^*$ is $\nabla^2L(u^*)/r^2$, with the radial direction a null direction, so $S(w)=S(u^*)/r^2$ and
$$
\eta_{\rm eff}\,S(u^*)=\frac{\eta\,S(u^*)}{(1-\eta\lambda)r^2}=\frac{\eta\,S(w)}{1-\eta\lambda}.
$$
Shrinking $r$ raises the effective learning rate and the Euclidean sharpness by the same factor; the stability condition does not care which description you use. This is the hinge for 5.3.1.
"""
    ),
    # ================================================================== 5.2
    markdown(
        r"""
<a id="s52"></a>

## 5.2 The weight-decay equilibrium

On a scale-invariant block (notation of 5.1: $r=\|w\|$, learning rate $\eta$, weight decay
$\lambda$), gradient noise grows the norm and weight decay shrinks it. Without weight decay
the norm only grows, so the effective learning rate $\eta/r^2$ decays by itself, like
$1/\sqrt k$ at step $k$. With it, the norm settles where each step turns the direction by
$\theta^*=\arccos(1-\eta\lambda)\approx\sqrt{2\eta\lambda}$, independent of the gradient
scale: exactly under SGD, approximately under AdamW.

The lead paper is Kosson et al. (top of page), who call this state *rotational
equilibrium*; the exact SGD statement below is Li and Arora's.

"""
        + reading(
            "An exponential learning rate schedule for deep learning",
            "https://arxiv.org/abs/1910.07454",
            "Li & Arora (ICLR 2020)",
            "For a scale-invariant loss, SGD with weight decay (and momentum) equals, in function space, SGD with an exponentially increasing learning rate and no weight decay; BatchNorm ResNets train fine this way.",
            importance=5,
            length=3,
        )
        + r"""

**The core idea.** On these weights weight decay cannot prefer one function over another;
its only channel is the norm, hence $\eta_{\rm eff}$ (any effect on generalization goes
through that, e.g. through the SGD noise scale, [NB2 §3](td2_3_sgd_sde.html)). So under
SGD, $\eta$ and $\lambda$ matter only through $\eta\lambda$, Li, Lyu and Arora's *intrinsic
learning rate*, and weight decay can be traded exactly for an exponentially growing learning
rate. Under AdamW the norm settles at $\approx\sqrt{\eta C/(2\lambda)}$ for a block of $C$
weights, and the step felt by the direction scales as $\eta/\|w\|$, not $\eta/\|w\|^2$.
Reaching equilibrium takes $\sim1/(4\eta\lambda)$ steps under SGD, often a sizeable part of
a run.

<a id="s52-walkthrough"></a>

### Optional: annotated derivations and walkthrough

**Setup.** SGD with weight decay is $w_{k+1}=(1-\eta\lambda)w_k-\eta\nabla L(w_k)$ (PyTorch's
`SGD(weight_decay=λ)` without momentum). With $r_k=\|w_k\|$, $G_k$ the gradient norm at the
unit-norm point $u_k=w_k/r_k$, and $\theta_k$ the angle between $w_k$ and $w_{k+1}$,
Exercise 5.1.2 gives, exactly,
$$
r_{k+1}^2=(1-\eta\lambda)^2r_k^2+\frac{\eta^2G_k^2}{r_k^2},\qquad
\eta_{\mathrm{eff},k}=\frac{\eta}{(1-\eta\lambda)r_k^2},\qquad
\cos\theta_k=(1-\eta\lambda)\frac{r_k}{r_{k+1}},\qquad
\tan\theta_k=\eta_{\mathrm{eff},k}G_k.
$$

**Exercise 5.2.1 — norm growth without weight decay.** With $\lambda=0$ the norm only grows,
so the effective learning rate keeps falling. What schedule is that over a long run?
([Arora, Li and Lyu 2019](https://arxiv.org/abs/1812.03981) show that GD with a fixed
learning rate on scale-invariant weights still converges at the rate of well-tuned GD.)

1. Show $r_{k+1}^4\ge r_k^4+2\eta^2G_k^2$, with near-equality when the angular step $\tan\theta_k=\eta G_k/r_k^2$ is small.
2. Suppose $G_k\approx G$ is roughly constant (as when minibatch noise dominates). Show that for large $k$, $\eta_{\rm eff,k}\approx1/(G\sqrt{2k})$ and $\theta_k\approx1/\sqrt{2k}$, *independently of $\eta$*.
3. What does this say about tuning the learning rate of normalized layers trained without weight decay for a long time?

<details><summary>Hint</summary>

Square the recursion $r_{k+1}^2=r_k^2+\eta^2G_k^2/r_k^2$ and drop a non-negative term. For 2, sum over steps and substitute into $\eta/r_k^2$.
</details>
"""
    ),
    answer(
        r"""
**1.** Square the recursion with $\lambda=0$: $r_{k+1}^4=r_k^4+2\eta^2G_k^2+\eta^4G_k^4/r_k^4$. The last term is non-negative, and relative to the middle one it is $\tfrac12\eta^2G_k^2/r_k^4=\tfrac12\tan^2\theta_k$, small when the angular steps are small.

**2.** Summing, $r_k^4\approx r_0^4+2\eta^2\sum_{j<k}G_j^2\approx r_0^4+2\eta^2G^2k$, provided the angular steps stay small; this needs the first one, $\eta G/r_0^2$, to be small too. (Exactly, $z_k=r_k^2/(\eta G)$ obeys $z_{k+1}=z_k+1/z_k$; if $z_0\ll1$, the first step inflates the norm to $z_1\approx1/z_0$, and the transient lasts $\sim1/(2z_0^2)$ steps instead.) Once $k\gg r_0^4/(2\eta^2G^2)$,
$$
\eta_{\rm eff,k}=\frac{\eta}{r_k^2}\approx\frac{\eta}{\eta G\sqrt{2k}}=\frac1{G\sqrt{2k}},\qquad
\theta_k\approx\eta_{\rm eff,k}G\approx\frac1{\sqrt{2k}}.
$$
The base learning rate has cancelled. A larger $\eta$ grows the norm faster, and the faster growth exactly compensates.

**3.** Late in training, the base learning rate of such layers barely matters: they run on an automatic $1/\sqrt k$ schedule. What $\eta$ does control is how long the transient lasts, $k\sim r_0^4/(2\eta^2G^2)$ when the first step is small, so the initialization scale and $\eta$ trade off against each other only early on. If you find that changing the learning rate of a normalized layer "does nothing", this is a candidate reason.
"""
    ),
    markdown(
        r"""
**A toy for checking the formulas.** Take the scale-invariant loss $L(w)=\tfrac12\,u^\top Au$
with $u=w/\|w\|$ and $A=\mathrm{diag}(a_1,\dots,a_d)$ (a Rayleigh quotient). Minibatch noise
is modelled by adding $\sigma\xi$, $\xi$ standard Gaussian, to the gradient at $u$ and
projecting onto the tangent space. (Real minibatch gradients of a scale-invariant network
are also orthogonal to $w$, so only the noise distribution is a model.) The simulator runs
several $(\eta,\lambda,\sigma)$ settings in parallel.
"""
    ),
    code(
        r"""
d_toy = 20
a_toy = np.linspace(0.0, 1.0, d_toy)          # L(u) = 1/2 u^T diag(a) u on the unit sphere

def sgd_scale_invariant(eta, lam, sigma, T, r0=1.0, seed=0, schedule=None):
    # SGD + weight decay on the noisy scale-invariant toy; one row per (eta, lam, sigma) setting.
    # schedule(k) -> per-step learning rates (overrides eta); used for the exponential-LR check.
    eta, lam, sigma, r0 = (np.array(z, float) for z in np.broadcast_arrays(
        np.atleast_1d(eta), np.atleast_1d(lam), np.atleast_1d(sigma), np.atleast_1d(r0)))
    B = len(eta); rng = np.random.default_rng(seed)
    w = rng.standard_normal((B, d_toy))
    w *= r0[:, None] / np.linalg.norm(w, axis=1, keepdims=True)
    R, COS, G2 = np.empty((T, B)), np.empty((T, B)), np.empty((T, B)); U = np.empty((T, B, d_toy))
    for k in range(T):
        r = np.linalg.norm(w, axis=1, keepdims=True); u = w / r
        gu = a_toy * u + sigma[:, None] * rng.standard_normal(w.shape)
        gu -= (gu * u).sum(1, keepdims=True) * u          # tangent part: the (noisy) gradient at u
        lr = eta if schedule is None else schedule(k)
        w_new = (1 - lr * lam)[:, None] * w - lr[:, None] * gu / r
        R[k], G2[k], U[k] = r[:, 0], (gu**2).sum(1), u
        COS[k] = (w * w_new).sum(1) / r[:, 0] / np.linalg.norm(w_new, axis=1)
        w = w_new
    return dict(r=R, cos=np.clip(COS, -1, 1), G2=G2, u=U)

# --- no weight decay: the effective learning rate forgets eta ---
etas = np.array([0.01, 0.1, 1.0])
out = sgd_scale_invariant(etas, 0.0, 0.5, 20000)
k = np.arange(1, 20001)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, e in enumerate(etas):
    ax[0].loglog(k, e / out["r"][:, i] ** 2, label=f"eta = {e}")
    ax[1].loglog(k, out["r"][:, i], label=f"eta = {e}")
G_rms = np.sqrt(out["G2"].mean())
ax[0].loglog(k, 1 / (G_rms * np.sqrt(2 * k)), "k--", label=r"$1/(G\sqrt{2k})$")
ax[0].set(xlabel="step k", ylabel=r"$\eta/\|w\|^2$", title="effective LR, no weight decay")
ax[1].set(xlabel="step k", ylabel=r"$\|w\|$", title="the norm grows like $k^{1/4}$")
ax[0].legend(); ax[1].legend(); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
After a transient (longer for smaller $\eta$), all three effective learning rates fall onto
the dashed line $1/(G\sqrt{2k})$, $G$ the RMS gradient norm: the base learning rate has been
forgotten, as 5.2.1.2 predicts.

**Exercise 5.2.2 — the weight-decay equilibrium.** With weight decay on, where does the
norm settle, and how far does the direction turn per step once it is there? Use the
recursions in the setup above. Take $\lambda>0$, and first take $G_k=G$ constant.

1. Find the fixed point of the norm recursion and show
$$
r_*^4=\frac{\eta G^2}{\lambda(2-\eta\lambda)}\approx\frac{\eta G^2}{2\lambda}.
$$
2. Show that at the fixed point $\cos\theta^*=1-\eta\lambda$ exactly, so $\theta^*=\arccos(1-\eta\lambda)=\sqrt{2\eta\lambda}\,\big(1+O(\eta\lambda)\big)$, independent of $G$. Show also that $\eta_{\rm eff}^*=\tan\theta^*/G\approx\sqrt{2\eta\lambda}/G$.
3. Linearize the recursion for $s=r^2$ about the fixed point and show that deviations shrink by the factor $2(1-\eta\lambda)^2-1\approx1-4\eta\lambda$ per step. How many steps does it take to reach equilibrium with $\eta=0.1$, $\lambda=5\times10^{-4}$?
4. **Noise.** Now let $G_k$ be random (minibatch noise), with $\mathbb E[G_k^2\mid w_k]=\bar G^2$. Using 5.1.2.3, show that whenever $\log r_k$ is stationary (its distribution no longer changes with $k$),
$$
\mathbb E\big[\log\cos\theta_k\big]=\log(1-\eta\lambda)\qquad\text{exactly.}
$$
Conclude that $\mathbb E\,\theta_k^2\le-2\log(1-\eta\lambda)$, with near-equality, so that the RMS angle is $\approx\sqrt{2\eta\lambda}$, when the per-step angles are all small. What about the *mean* angle?
5. Still with noise, show that $\mathbb E[r_{k+1}^2\mid w_k]=(1-\eta\lambda)^2r_k^2+\eta^2\bar G^2/r_k^2$ exactly, and hence, if $r$ fluctuates little, $r_*^4\approx\eta\bar G^2/(\lambda(2-\eta\lambda))$ with $\bar G$ the RMS gradient norm at the unit-norm point. (The next cell compares this with the measured norm.)

<details><summary>Hint</summary>

For 2, at the fixed point $r_{k+1}=r_k$, and 5.1.2.3 does the rest. For 4, take logs in $\cos\theta_k=(1-\eta\lambda)r_k/r_{k+1}$ and use $\log\cos\theta\approx-\theta^2/2$. For the mean angle, Jensen.
</details>
"""
    ),
    answer(
        r"""
**1.** Put $s=r^2$: $s_{k+1}=a\,s_k+\eta^2G^2/s_k$ with $a=(1-\eta\lambda)^2$. A fixed point satisfies $(1-a)s_*=\eta^2G^2/s_*$, so $s_*^2=\eta^2G^2/(1-a)$. Since $1-a=\eta\lambda(2-\eta\lambda)$,
$$
r_*^4=s_*^2=\frac{\eta G^2}{\lambda(2-\eta\lambda)}.
$$
The norm scales as $(\eta/\lambda)^{1/4}G^{1/2}$: a larger gradient scale buys a larger norm, not a larger rotation.

**2.** At the fixed point $r_{k+1}=r_k$, so $\cos\theta^*=(1-\eta\lambda)r_k/r_{k+1}=1-\eta\lambda$. Expanding $\arccos(1-x)=\sqrt{2x}\,(1+x/12+\dots)$ gives $\theta^*\approx\sqrt{2\eta\lambda}$. From $\tan\theta=\eta_{\rm eff}G$,
$$
\eta_{\rm eff}^*=\frac{\tan\theta^*}{G}=\frac{\sqrt{\eta\lambda(2-\eta\lambda)}}{(1-\eta\lambda)\,G}\approx\frac{\sqrt{2\eta\lambda}}{G}.
$$
The effective learning rate adjusts itself to the gradient scale so that the *angle* comes out the same. A layer with larger gradients sits at a larger norm and so takes the same angular steps. This is the "balancing" in Kosson et al.'s title.

**3.** $\frac{d}{ds}\big(as+\eta^2G^2/s\big)=a-\eta^2G^2/s^2$. At $s_*$, $\eta^2G^2/s_*^2=1-a$, so the slope is $2a-1=2(1-\eta\lambda)^2-1=1-4\eta\lambda+2\eta^2\lambda^2$. Deviations decay like $e^{-4\eta\lambda k}$, with time constant $1/(4\eta\lambda)$ steps. For $\eta=0.1$, $\lambda=5\times10^{-4}$: $\eta\lambda=5\times10^{-5}$, time constant $5000$ steps, and a few time constants (tens of thousands of steps) to forget the initial norm. That is a sizeable fraction of a typical schedule, so the equilibrium is often approached while the learning rate is still being changed.

**4.** Take logs: $\log\cos\theta_k=\log(1-\eta\lambda)+\log r_k-\log r_{k+1}$. This uses only scale invariance (the gradient term is orthogonal to $w_k$, step by step, for every minibatch), not the distribution of $G_k$. Take expectations. In a stationary state $\mathbb E\log r_{k+1}=\mathbb E\log r_k$, so
$$
\mathbb E\log\cos\theta_k=\log(1-\eta\lambda).
$$
Since $-\log\cos\theta\ge\theta^2/2$ for $|\theta|<\pi/2$, $\mathbb E\,\theta_k^2\le-2\log(1-\eta\lambda)\approx2\eta\lambda$, for any noise. Equality to leading order needs the large-angle contributions to $\mathbb E[-\log\cos\theta]$ to be negligible, which holds if the per-step rotations are uniformly small. (Light tails alone are not enough: if the gradient is zero on most steps and of order one on rare steps, the rare large rotations carry most of $\mathbb E[-\log\cos\theta]$ and the RMS angle can be far below $\sqrt{2\eta\lambda}$.) By Jensen, $\mathbb E\,\theta_k\le\sqrt{\mathbb E\,\theta_k^2}$, with equality only if $\theta_k$ is deterministic; in the toy below the two agree to about 1%.

**5.** $\mathbb E[r_{k+1}^2\mid w_k]=a\,r_k^2+\eta^2\bar G^2/r_k^2$ holds exactly. If $r$ fluctuates little over the $\sim1/(4\eta\lambda)$ steps it takes to relax, then $r_*^4\approx\eta\bar G^2/(\lambda(2-\eta\lambda))$, with $\bar G$ now the RMS gradient norm at the unit-norm point.
"""
    ),
    code(
        r"""
# Weight-decay equilibrium: five settings, including two pairs with the same eta*lam.
eta = np.array([0.1, 0.1, 0.4, 0.05, 0.2])
lam = np.array([1e-2, 1e-2, 2.5e-3, 2e-2, 2e-2])
sig = np.array([0.3, 1.0, 1.0, 0.3, 0.1])
T = 8000
out = sgd_scale_invariant(eta, lam, sig, T)
theta = np.arccos(out["cos"]); burn = T // 4

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
win = 400
for i in range(len(eta)):
    rms = np.sqrt(np.convolve(theta[:, i] ** 2, np.ones(win) / win, mode="valid"))
    ax[0].plot(np.arange(len(rms)) + win, rms, label=f"eta={eta[i]}, lam={lam[i]}, sigma={sig[i]}")
    ax[1].plot(out["r"][:, i])
for el in np.unique(eta * lam):
    ax[0].axhline(np.arccos(1 - el), color="k", ls="--", lw=1)
ax[0].set(xlabel="step", ylabel="RMS angle per step (window 400)", title=r"dashed: $\arccos(1-\eta\lambda)$")
ax[1].set(xlabel="step", ylabel=r"$\|w\|$", title=r"norm relaxes on a $1/(4\eta\lambda)$ timescale")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()

print(" eta     lam     sigma | RMS angle  mean angle  arccos(1-el)  sqrt(2el) | E[-log cos]  -log(1-el) | <r>     r* pred")
for i in range(len(eta)):
    th, el = theta[burn:, i], eta[i] * lam[i]
    r_pred = (eta[i] * out["G2"][burn:, i].mean() / (lam[i] * (2 - el))) ** 0.25
    print(f"{eta[i]:5.2f} {lam[i]:7.4f} {sig[i]:6.2f}   | {np.sqrt((th**2).mean()):.5f}    {th.mean():.5f}"
          f"     {np.arccos(1 - el):.5f}      {np.sqrt(2 * el):.5f} | {-np.log(out['cos'][burn:, i]).mean():.6f}"
          f"     {-np.log(1 - el):.6f}  | {out['r'][burn:, i].mean():.3f}   {r_pred:.3f}")
"""
    ),
    markdown(
        r"""
(In the table, `el` is $\eta\lambda$.) Four of the five settings share $\eta\lambda=10^{-3}$,
with three different $(\eta,\lambda)$ pairs and two noise levels, and rotate by the same
angle. The noise level $\sigma$ (equivalently the gradient scale) moves only the norm. The
stationarity identity $\mathbb E[-\log\cos\theta]=-\log(1-\eta\lambda)$ holds to about 0.2%.

**Exercise 5.2.3 — only $\eta\lambda$ matters.** On scale-invariant weights, do $\eta$ and
$\lambda$ matter separately, or only through $\eta\lambda$ (Li, Lyu and Arora's *intrinsic
learning rate*)? With the setup's recursions, define the rescaled squared norm
$\tilde s_k=r_k^2/\eta$.

1. Show that $\tilde s_{k+1}=(1-\eta\lambda)^2\tilde s_k+G_k^2/\tilde s_k$ and $\eta_{\mathrm{eff},k}=1/\big((1-\eta\lambda)\tilde s_k\big)$.
2. Conclude: two runs with the same product $\eta\lambda$, the same initial direction, the same initial $r_0^2/\eta$ and the same minibatch sequence produce *identical* direction trajectories $u_k$, hence identical network functions on scale-invariant layers, even if $\eta$ and $\lambda$ differ individually. With different $r_0^2/\eta$, what can you say?
3. Suppose you multiply $\eta$ by 10 and divide $\lambda$ by 10 on a scale-invariant layer. What changes?

<details><summary>Hint</summary>

For 2, note that $u_{k+1}$ is determined by $u_k$, the minibatch and $\eta_{{\rm eff},k}$ (5.1.2.1), and by part 1 the pair $(u_k,\tilde s_k)$ evolves by a map involving only $\eta\lambda$. For different $r_0^2/\eta$, think about what is local (constant-$G$ relaxation of $\tilde s$, as in 5.2.2.3) and what is only conjectured.
</details>
"""
    ),
    answer(
        r"""
**1.** Divide the norm recursion by $\eta$: $r_{k+1}^2/\eta=(1-\eta\lambda)^2r_k^2/\eta+\eta G_k^2/r_k^2=(1-\eta\lambda)^2\tilde s_k+G_k^2/\tilde s_k$. And $\eta_{\rm eff}=\eta/((1-\eta\lambda)r_k^2)=1/((1-\eta\lambda)\tilde s_k)$.

**2.** By 5.1.2.1, $u_{k+1}$ is a function of $u_k$, the minibatch at step $k$ (through $\nabla L(u_k)$ and $G_k$) and $\eta_{{\rm eff},k}$. By part 1, $(u_k,\tilde s_k)$ evolves by a map that involves $\eta,\lambda$ only through $\eta\lambda$. Same $\eta\lambda$, same $(u_0,\tilde s_0)$ and same minibatches give the same $(u_k,\tilde s_k)$ for all $k$. Different $\tilde s_0$: the runs now see different directions and hence different $G_k$, so they are no longer coupled exactly. If $G$ is roughly constant, the $\tilde s$ recursion is the one from 5.2.2.3 with $\eta^2G^2$ replaced by $G^2$, and it relaxes toward its fixed point at rate $\approx4\eta\lambda$; beyond that local statement there is no theorem. That the *distribution* of the network function forgets the initial norm in $O(1/\eta\lambda)$ steps is Li, Lyu and Arora's Fast Equilibrium Conjecture, supported by experiments but not proved.

**3.** On a scale-invariant layer, if you also scale the initialization so that $r_0^2/\eta$ is unchanged (i.e. $r_0\to\sqrt{10}\,r_0$), nothing at all in function space, by part 2. With the same initialization, the run starts with a 10 times larger effective learning rate; if both runs then reach the same equilibrium (the conjecture above), they end up statistically the same, with the norm $\sqrt{10}$ times larger ($r_*^2\propto\sqrt{\eta/\lambda}$) and the same rotation per step. What does change in any case: every parameter that is *not* scale-invariant (biases, gains, the output layer), for which $\eta$ is the learning rate in the ordinary sense.
"""
    ),
    markdown(
        r"""
**Exercise 5.2.4 — weight decay is an exponentially increasing learning rate.** Weight decay
acts here only by shrinking the norm, which raises $\eta_{\rm eff}$. Can it be traded
*exactly* for a learning-rate schedule? This is Li and Arora's theorem, without momentum.
Let $L$ be scale-invariant (the paper freezes the last layer and the norm layers' affine
parameters to arrange this), and run SGD with weight decay,
$w_{k+1}=(1-\eta\lambda)w_k-\eta\nabla L_k(w_k)$, from $w_0$, where $L_k$ is the minibatch
loss at step $k$.

1. Define $\tilde w_k=(1-\eta\lambda)^{-k}w_k$. Show that $\tilde w_{k+1}=\tilde w_k-\tilde\eta_k\nabla L_k(\tilde w_k)$ with
$$
\tilde\eta_k=\eta\,(1-\eta\lambda)^{-2k-1},
$$
i.e. plain SGD, no weight decay, with an exponentially increasing learning rate, from the same $\tilde w_0=w_0$. *Hint:* divide the update by $(1-\eta\lambda)^{k+1}$, then use $\nabla L(cw)=\nabla L(w)/c$ (5.1.1.2) to rewrite $\nabla L_k(w_k)$ in terms of $\tilde w_k$.
2. In what sense are the two runs "equivalent"? Are the iterates equal?
3. With $\eta\lambda=5\times10^{-5}$ (e.g. $\eta=0.1$, $\lambda=5\times10^{-4}$), by what factor does $\tilde\eta$ grow per epoch of 391 steps (CIFAR-10 at batch 128)? Why do the two pictures (shrinking weights at fixed $\eta$, versus growing $\eta$ at growing weights) give the same effective learning rate?

The momentum version (stated, not derived): with heavy-ball momentum $\gamma$, the same holds with $\tilde\eta_k=\eta\,\alpha^{-2k-1}$, where $\alpha$ is the larger root (it $\to1$ as $\eta\lambda\to0$) of $x^2-(1+\gamma-\eta\lambda)x+\gamma=0$ (real if $\eta\lambda\le(1-\sqrt\gamma)^2$), plus a matching initial momentum buffer. (If you read the paper's Theorem 2.9: the rescaling
should be $\tilde\theta_t=\alpha^{-t}\theta_t$, as its own initial condition and Theorem 2.1
require; the printed $\alpha^t$ is a typo.)
"""
    ),
    answer(
        r"""
**1.** Divide the update by $(1-\eta\lambda)^{k+1}$:
$$
\tilde w_{k+1}=\tilde w_k-\eta(1-\eta\lambda)^{-(k+1)}\nabla L_k(w_k).
$$
Now $w_k=(1-\eta\lambda)^k\tilde w_k$, and by 5.1.1.2, $\nabla L_k(c\,\tilde w)=\nabla L_k(\tilde w)/c$ with $c=(1-\eta\lambda)^k$. So $\nabla L_k(w_k)=(1-\eta\lambda)^{-k}\nabla L_k(\tilde w_k)$ and
$$
\tilde w_{k+1}=\tilde w_k-\eta(1-\eta\lambda)^{-2k-1}\nabla L_k(\tilde w_k).
$$

**2.** The iterates differ by the positive factor $(1-\eta\lambda)^{-k}$, which grows exponentially. They have the same direction at every step, so the network functions agree at every step, and so do the training and test losses. "Equivalent" means equal in function space, not in parameter space.

**3.** Per step $\tilde\eta$ grows by $(1-\eta\lambda)^{-2}\approx1+2\eta\lambda=1.0001$; per 391 steps, by $(1-5\times10^{-5})^{-782}\approx e^{0.039}\approx1.04$. With $q=1-\eta\lambda$: $\tilde r_k=q^{-k}r_k$ exactly, so $\tilde\eta_k/\tilde r_k^2=\eta q^{-2k-1}/(q^{-2k}r_k^2)=\eta/(q\,r_k^2)$, which is the $\eta_{\rm eff}$ of 5.1.2. Whatever the weight-decay run's norm does (shrink, or grow, or sit at equilibrium), the exponential run's norm is the same thing times $q^{-k}$, and the step size seen by the direction is identical. Only the ratio of step size to squared norm is visible to the direction, so a learning-rate schedule and a weight-decay schedule on scale-invariant weights are interchangeable.
"""
    ),
    markdown(
        r"""
**Code exercise — check 5.2.3 and 5.2.4.** `sgd_scale_invariant` (above) draws the same
noise for the same `seed` and accepts a per-step learning-rate `schedule`. (a) Run SGD with
weight decay and the exponential-learning-rate run without it from the same $w_0$; compare
their directions $u_k$. (b) Compare $(\eta,\lambda,r_0)=(0.1,10^{-2},1)$ with
$(0.4,2.5\times10^{-3},2)$, which share $\eta\lambda$ and $r_0^2/\eta$.
"""
    ),
    code_answer(
        r"""
# (a) Li-Arora: SGD + WD versus plain SGD with lr_k = eta * (1 - eta*lam)**(-2k-1), same seed.
# (b) same eta*lam and same r0^2/eta: are the direction trajectories identical?
# Hint: sgd_scale_invariant(..., schedule=lambda k: np.array([...])) and compare out["u"].
...
""",
        r"""
T, e, l = 400, 0.1, 1e-2
wd = sgd_scale_invariant(e, l, 0.3, T, seed=7)
exp_lr = sgd_scale_invariant(e, 0.0, 0.3, T, seed=7,
                             schedule=lambda k: np.array([e * (1 - e * l) ** (-2 * k - 1)]))
print("(a) max |u_WD - u_expLR| over 400 steps:", np.abs(wd["u"] - exp_lr["u"]).max())
print("    norm ratio r_expLR / r_WD at k = 0, 200, 399:",
      (exp_lr["r"] / wd["r"])[[0, 200, 399], 0], " vs (1-eta*lam)^-k:", (1 - e * l) ** -np.array([0, 200, 399]))

A = sgd_scale_invariant(0.1, 1e-2, 0.3, 2000, r0=1.0, seed=3)
B = sgd_scale_invariant(0.4, 2.5e-3, 0.3, 2000, r0=2.0, seed=3)
print("(b) max |u_A - u_B| over 2000 steps:", np.abs(A["u"] - B["u"]).max())
print("    r^2/eta at the end:", A["r"][-1, 0] ** 2 / 0.1, B["r"][-1, 0] ** 2 / 0.4)
""",
    ),
    markdown(
        r"""
Both differences are at roundoff level (roundoff in the exponential run grows slowly with
its norm). The two runs in (b) have learning rates four times apart and compute the same
function at every step.

**AdamW.** The SGD analysis rested on $\|\nabla L(w)\|=G/r$. Adam's update does not scale
like $1/r$, because dividing by $\sqrt{\hat v}$ cancels the gradient's scale, so the norm
recursion changes. Notation: AdamW with gradient $g_k$ keeps
$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k$ and $v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2$ (coordinatewise),
bias-corrects them to $\hat m_k=m_k/(1-\beta_1^k)$, $\hat v_k=v_k/(1-\beta_2^k)$, and on one
block of $C$ weights steps $w_{k+1}=(1-\eta\lambda)w_k-\eta U_k$ with
$U_k=\hat m_k/(\sqrt{\hat v_k}+\epsilon)$.

What comes out (derived for $\beta_1=0$ in Exercise 5.2.5; the momentum factors are Kosson
et al.'s), assuming noise-dominated gradients, small relative updates, and
$\eta\lambda\ll1-\beta_1$ with $\hat v$ tracking $\mathbb Eg^2$:
- *At equilibrium* the norm is $r_*\approx\sqrt{\eta C/(2\lambda)}$, independent of the gradient scale, and the RMS angle per step is $\approx\sqrt{2\eta\lambda\,(1-\beta_1)/(1+\beta_1)}$.
- *At any norm $r$* the RMS angle per step is $\approx(\eta/r)\sqrt{C(1-\beta_1)/(1+\beta_1)}$: the effective learning rate scales as $\eta/\|w\|$, not $\eta/\|w\|^2$.
- *Approach:* the norm relaxes on a timescale $\sim1/(2\eta\lambda)$ steps. Unrolled, $w_k=(1-\eta\lambda)^kw_0-\eta\sum_{j<k}(1-\eta\lambda)^{k-1-j}U_j$: the weights are an exponential average of past updates with memory $\sim1/(\eta\lambda)$ steps.

With Adam plus an $L^2$ penalty instead, the decay is normalized along with the gradient,
so the rotation rate depends on each layer's gradient scale: Kosson et al. measure a roughly
$30\times$ spread across layers, against nearly uniform rotation for AdamW, and hypothesize
that this balance is AdamW's main benefit.

**Exercise 5.2.5 — the AdamW equilibrium.** Take $\beta_1=0$ (so $\hat m_k=g_k$),
$\epsilon\approx0$, and $r_k=\|w_k\|$. Assume (as Kosson et al. do) a noise-dominated "random
walk": $\mathbb E\langle U_k,w_k\rangle=0$ and $\mathbb E\|U_k\|^2=C$, the latter because each
coordinate of $U_k=g_k/\sqrt{\hat v_k}$ has second moment $\approx1$ when $\hat v$ tracks
$\mathbb Eg^2$.

1. Show that the expected squared norm obeys $\mathbb E r_{k+1}^2=(1-\eta\lambda)^2\,\mathbb Er_k^2+\eta^2C$, with equilibrium $r_*^2=\eta C/(\lambda(2-\eta\lambda))\approx\eta C/(2\lambda)$, approached at rate $\approx2\eta\lambda$ per step.
2. Show that the RMS angle per step at equilibrium is again $\theta^*\approx\sqrt{2\eta\lambda}$, i.e. $\cos\theta^*\approx1-\eta\lambda$.
3. Show that the angle per step is $\approx\eta\sqrt C/r$. In what sense is the effective learning rate of AdamW $\propto\eta/\|w\|$ rather than $\eta/\|w\|^2$?

<details><summary>Hint</summary>

Expand $\|(1-\eta\lambda)w_k-\eta U_k\|^2$ and take expectations. For 2, split $U_k$ into its component along $u_k$ and the tangential rest; the angle is set by the ratio of the tangential part to the radial part of $w_{k+1}$.
</details>
"""
    ),
    answer(
        r"""
**1.** $r_{k+1}^2=(1-\eta\lambda)^2r_k^2-2\eta(1-\eta\lambda)\langle w_k,U_k\rangle+\eta^2\|U_k\|^2$. Take expectations with the two assumptions: $\mathbb Er_{k+1}^2=(1-\eta\lambda)^2\mathbb Er_k^2+\eta^2C$. This is a linear recursion with fixed point $r_*^2=\eta^2C/(1-(1-\eta\lambda)^2)=\eta C/(\lambda(2-\eta\lambda))$ and contraction factor $(1-\eta\lambda)^2\approx1-2\eta\lambda$. The norm now scales as $\sqrt{\eta C/\lambda}$, with no dependence on the gradient scale at all.

**2.** This needs more assumptions, which Kosson et al. also make: the relative update is uniformly small, $\eta\|U\|\ll r$ on every step (so the denominator below stays near $(1-\eta\lambda)r$; rare large updates would break the expectation), the radial part of the update is small, $\mathbb E(u^\top U)^2\ll C$ (so nearly all of $\|U\|^2$ is rotation), and $r$ concentrates near $r_*$. Exactly, $\tan\theta=\eta\|U_\perp\|/\big((1-\eta\lambda)r-\eta\,u^\top U\big)$ with $U_\perp$ the tangential part. Under the assumptions, the component of $w_{k+1}$ along $w_k$ is $\approx(1-\eta\lambda)r_*$, and the orthogonal part has mean square $\approx\eta^2C$. So $\mathbb E\tan^2\theta\approx\eta^2C/((1-\eta\lambda)^2r_*^2)=\eta\lambda(2-\eta\lambda)/(1-\eta\lambda)^2$, the same expression as for SGD, giving $\cos\theta^*\approx1-\eta\lambda$ and $\theta^*\approx\sqrt{2\eta\lambda}$.

**3.** $\tan\theta\approx\eta\|U\|/r\approx\eta\sqrt C/r$. For SGD the step was $\eta G/r^2$: one factor of $1/r$ from the step size and one from $\nabla L(w)=\nabla L(u)/r$. Adam's normalization removes the second, so the rotation is $\eta/r$ times a gradient-independent constant.
"""
    ),
    markdown(
        r"""
**With momentum** (Kosson et al., Table 1). For AdamW the equilibrium norm is unchanged but
the RMS angle per step shrinks to $\sqrt{2\eta\lambda(1-\beta_1)/(1+\beta_1)}$: momentum
spreads each gradient over $\sim1/(1-\beta_1)$ correlated steps, so each step rotates less.
For SGD with heavy-ball momentum $\gamma$ their prediction is $\sqrt{2\eta\lambda/(1+\gamma)}$.
D'Angelo et al. (Readings) report, for sign-SGD, that matching $\eta/\|w\|$ (not
$\eta/\|w\|^2$) reproduces AdamW's training curves without weight decay in float32, while the
bfloat16 runs diverge. The next cell tests the random-walk assumption.
"""
    ),
    code(
        r"""
def adamw_scale_invariant(eta, lam, sigma, T, b1=0.9, b2=0.999, eps=1e-12, d=64, seed=0):
    # AdamW on the noisy Rayleigh toy in d dimensions; returns per-step angles and norms.
    a = np.linspace(0, 1, d); rng = np.random.default_rng(seed)
    w = rng.standard_normal(d); m = np.zeros(d); v = np.zeros(d); ang, rs, rad = [], [], []
    for k in range(1, T + 1):
        r = np.linalg.norm(w); u = w / r
        q = a * u + sigma * rng.standard_normal(d); q -= (q @ u) * u
        g = q / r
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w_new = (1 - eta * lam) * w - eta * (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + eps)
        U = (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + eps)
        ang.append(np.arccos(np.clip(w @ w_new / r / np.linalg.norm(w_new), -1, 1))); rs.append(r)
        rad.append(u @ U / np.linalg.norm(U))        # cosine between the Adam update and w
        w = w_new
    return np.array(ang), np.array(rs), np.array(rad)

print(" sigma  beta1 | RMS angle  Kosson sqrt(2 el (1-b1)/(1+b1)) | <r>    sqrt(eta C/(2 lam)) | <cos(u, U)>")
eta_a, lam_a, C = 1e-2, 0.1, 64
for sigma in [1.0, 0.1]:
    for b1 in [0.0, 0.9]:
        T = int(10 / (eta_a * lam_a)); th, rs, rad = adamw_scale_invariant(eta_a, lam_a, sigma, T, b1=b1)
        print(f"  {sigma:4.1f}   {b1:3.1f}  |  {np.sqrt((th[T//2:]**2).mean()):.4f}       "
              f"{np.sqrt(2*eta_a*lam_a*(1-b1)/(1+b1)):.4f}                     | {rs[T//2:].mean():.2f}   "
              f"{np.sqrt(eta_a*C/(2*lam_a)):.2f}               | {rad[T//2:].mean():+.3f}")
"""
    ),
    markdown(
        r"""
On the toy in $C=64$ dimensions ($\sigma$ the noise level), the predictions hold to about
5–10% when noise dominates ($\sigma=1$). When signal is comparable to noise ($\sigma=0.1$) they
fail: the norm settles about twice as large and rotates about half as fast. The last column
shows why: Adam's coordinatewise normalization breaks orthogonality to $w$. At $\beta_1=0$ the
cosine of $u$ with $U$ is only about $-0.04$, but this radial term enters the norm recursion
at first order in $\eta$ ($-2\eta(1-\eta\lambda)\langle w,U\rangle\approx0.02$ per step) and
rotation only at second order ($\eta^2C\approx0.006$). (With $\beta_1=0.9$ the cosine is
larger even when the predictions hold, because momentum points partly along *past* tangent
directions.) The SGD identity $\mathbb E\log\cos\theta=\log(1-\eta\lambda)$, by contrast,
needs only scale invariance and stationarity.

**What this means for setting $\eta$ and $\lambda$** on normalized weights:
- With a fixed initialization, changing $\eta$ and $\lambda$ at fixed product changes the transient; that the difference is later forgotten is conjectured, not proved (5.2.3).
- Decaying $\eta$ at fixed $\lambda$ shrinks the equilibrium rotation only like $\sqrt\eta$, and only once the norm has re-equilibrated, which takes $\sim1/(4\eta\lambda)$ steps (SGD) or $1/(2\eta\lambda)$ (AdamW). If the schedule moves faster, the equilibrium is a target, not the state.
- With little gradient noise the equilibrium can be unstable, and the dynamics turn periodic (5.3, mechanism 1).
- Parameters that are not scale-invariant (gains, biases, embeddings, the output layer) behave in the ordinary way: weight decay can change their function directly.
- A pre-LN transformer is not exactly scale-invariant, but Kosson et al. find the predicted rotation holding very well in a GPT-2 (124M) under AdamW with weight standardization, and "often close" without it.
"""
    ),
    # ================================================================== 5.3
    markdown(
        r"""
<a id="s53"></a>

## 5.3 Loss spikes

Why do runs sometimes spike (a sudden rise in training loss that recovers) or slowly
diverge (a drift upward that does not)? There is no single cause. Below are four documented
mechanisms, each asked whether it is an *edge-of-stability phenomenon*: an instability that
appears when some curvature, measured in the optimizer's own geometry, crosses its stability
threshold (for GD with learning rate $\eta$ and sharpness $S$, $\eta S=2$;
[NB2 §2](td2_2_edge_of_stability.html)).

| Mechanism | Edge of stability? | Spike or divergence | Remedy |
|---|---|---|---|
| 1. Weight decay shrinks a normalized layer until $\eta_{\rm eff}S$ reaches 2 | Yes | Periodic spikes | LR decay; gradient noise; fixing the norm |
| 2. Adam's second moment goes stale in a quiet phase | Yes, in Adam's preconditioned geometry | Spikes that recover | Lower $\beta_2$ |
| 3. Attention logits grow with $\lVert W_Q\rVert\lVert W_K\rVert$ | Not shown | Slow divergence | QK-layernorm |
| 4. Adam drifts output logits along a flat direction | The drift is not (zero curvature) | Slow divergence | z-loss; weight decay |

The lead papers are Wortsman et al. (top of page) for 3 and 4 (they study slow divergences,
not fast spikes) and, for 2:

"""
        + reading(
            "Adaptive preconditioners trigger loss spikes in Adam",
            "https://arxiv.org/abs/2506.04805",
            "Bai et al. (ICML 2026)",
            "Mechanism 2, analyzed on a 1D quadratic and shown in small networks and transformers up to 187M parameters; lowering β₂ reduces spikes.",
            importance=2,
            length=3,
        )
        + r"""

Mechanism 1 is the equilibrium of 5.2 going unstable when gradient noise is small. The
walkthrough reproduces 1, 2 and the drift of 4 on a laptop in seconds.

<a id="s53-walkthrough"></a>

### Optional: annotated derivations and walkthrough

**Setup.** GD on $\tfrac12hx^2$ gives $x_{k+1}=(1-\eta h)x_k$, stable iff $0<\eta h<2$; near
a minimum with Hessian $H$ the condition is $\eta S<2$, $S=\lambda_{\max}(H)$. For a scale-invariant block
$w=ru$ (5.1), the direction moves with $\eta_{\rm eff}=\eta/((1-\eta\lambda)r^2)$, is stable
near a minimum iff $\eta_{\rm eff}S(u)<2$ ($S(u)$ the sharpness at the unit-norm point), and
the norm obeys $r_{k+1}^2=(1-\eta\lambda)^2r_k^2+\eta^2G_k^2/r_k^2$, $G_k$ the gradient norm
at $u_k$.

**Mechanism 1: weight decay drives a normalized layer to the edge of stability.** Near a
minimum the gradient is small, so the norm recursion is dominated by decay: $r$ shrinks and
$\eta_{\rm eff}$ rises until $\eta_{\rm eff}S(u)$ crosses 2. The direction goes unstable, the
loss jumps, the large gradients inflate $r$, and training re-converges; then the decay
starts again. [Lobacheva et al.](https://arxiv.org/abs/2106.15739) document these periodic
destabilizations in long constant-learning-rate BatchNorm training, and
[Lyu, Li and Arora](https://arxiv.org/abs/2206.07085) prove (locally) that GD with weight
decay reaches $\eta_{\rm eff}\approx2/S(u)$.

**Exercise 5.3.1 — the cycle.** How long does weight decay take to push a converged
normalized layer to instability, and what ends each spike? Let $u^*$ be a minimum of a
scale-invariant $L$ on the sphere, with sharpness $S(u^*)$ in tangent directions, and run
full-batch GD with weight decay from near $u^*$.

1. Using 5.1.2.4, show that the direction is linearly stable at $u^*$ iff $r^2>r_c^2=\eta S(u^*)/(2(1-\eta\lambda))$.
2. While $u$ sits at $u^*$, the gradient vanishes. Show that $r_k^2$ then decays like $(1-\eta\lambda)^{2k}$, and estimate the number of steps it takes to go from $r^2$ to $r_c^2$.
3. Describe the cycle that follows. Which term in the norm recursion ends each spike?
4. In the toy $L(x,y)=x^2/(x^2+y^2)$ (used by Li, Lyu and Arora and by Lobacheva et al.), the minimum is $u^*=(0,\pm1)$. Compute $S(u^*)$ and $r_c$ for $\eta=0.1$.

<details><summary>Hint</summary>

For 4, the tangent direction at $(0,1)$ is $x$; expand $L(x,1)$ to second order in $x$. For 3, look at which term of $r_{k+1}^2=(1-\eta\lambda)^2r_k^2+\eta^2G_k^2/r_k^2$ becomes large when the loss is high.
</details>
"""
    ),
    answer(
        r"""
**1.** By 5.1.2.4, the direction dynamics are stable iff $\eta_{\rm eff}S(u^*)<2$, i.e. $\eta S(u^*)/((1-\eta\lambda)r^2)<2$, i.e. $r^2>\eta S(u^*)/(2(1-\eta\lambda))$.

**2.** With $G_k=0$ the norm recursion is $r_{k+1}^2=(1-\eta\lambda)^2r_k^2$. Going from $r^2$ to $r_c^2$ takes $k=\log(r^2/r_c^2)/(-2\log(1-\eta\lambda))\approx\log(r^2/r_c^2)/(2\eta\lambda)$ steps: a time of order $1/\eta\lambda$ times a logarithm.

**3.** Once $r<r_c$, any component of $u-u^*$ along the top tangent eigendirection is multiplied by $|1-\eta_{\rm eff}S(u^*)|>1$ each step, with alternating sign. That component starts tiny (roundoff, or minibatch noise), so the loss stays low for a while and then rises abruptly: a spike. The large gradient feeds the term $\eta^2G_k^2/r_k^2$ in the norm recursion, which inflates $r$, lowers $\eta_{\rm eff}$ below $2/S(u^*)$, and lets the direction re-converge. Then the decay phase starts again. The period is part 2 plus the time the oscillation takes to grow from its seed, which depends on the seed's size: just after crossing, $\eta_{\rm eff}S$ exceeds 2 by an amount growing like $4\eta\lambda j$ after $j$ steps, so the log-amplitude grows like $2\eta\lambda j^2$, and a seed $A_0$ needs $j\sim\sqrt{\log(A/A_0)/(2\eta\lambda)}$ steps. Both pieces shrink as $\eta\lambda$ grows; Lobacheva et al. find empirically that periods shorten as $\eta\lambda$ grows.

**4.** At $u=(0,1)$ and in the tangent direction $x$: $L(x,1)=x^2/(1+x^2)=x^2-x^4+\dots$, so $S(u^*)=2$. Then $r_c^2=\eta S/(2(1-\eta\lambda))\approx\eta$, so $r_c\approx\sqrt{0.1}\approx0.32$ for $\eta=0.1$. The next cell shows the norm sawing around this value.
"""
    ),
    code(
        r"""
def gd_wd_2d(eta, lam, T, noise=0.0, w0=(1.0, 0.5), seed=0):
    # GD + weight decay on L(x, y) = x^2 / (x^2 + y^2); 'noise' adds a tiny isotropic kick per step.
    rng = np.random.default_rng(seed); w = np.array(w0); rec = []
    for _ in range(T):
        x, y = w; r2 = x * x + y * y
        g = np.array([2 * x * y * y, -2 * x * x * y]) / r2**2
        rec.append((x * x / r2, eta * 2.0 / ((1 - eta * lam) * r2), math.sqrt(r2)))  # loss, eta_eff*S(u*), r
        w = (1 - eta * lam) * w - eta * g + noise * rng.standard_normal(2)
    return np.array(rec)

eta, lam, T = 0.1, 0.01, 6000
quiet = gd_wd_2d(eta, lam, T, noise=1e-6)          # noise ~ roundoff: periodic edge-of-stability cycles
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
ax[0].semilogy(quiet[:, 0]); ax[0].set(title="loss", xlabel="step")
ax[1].plot(quiet[:, 1]); ax[1].axhline(2, color="k", ls="--"); ax[1].set(title=r"$\eta_{\rm eff}\,S(u^*)$", xlabel="step")
ax[2].plot(quiet[:, 2]); ax[2].axhline(math.sqrt(eta / (1 - eta * lam)), color="k", ls="--")
ax[2].set(title=r"$\|w\|$ (dashed: $r_c$)", xlabel="step"); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The loss axis is logarithmic: each spike jumps many orders of magnitude out of a converged
state, then re-converges. Exercise 5.3.1.2 predicts the first crossing of $r_c$ after
$\log(r_0^2/r_c^2)/(2\eta\lambda)\approx\log(1.25/0.1)/0.002\approx1260$ steps ($r_0^2=1.25$
for $w_0=(1,0.5)$, $r_c^2\approx\eta=0.1$). The loss starts rising at about step 1340 (the
extra delay is the oscillation growing from its tiny seed), and the cycle then repeats every
$\approx130$ steps. The tiny `noise` stands in for roundoff and minibatch noise: without it the
instability never visibly triggers within 6000 steps (the unstable component underflows) and
$\|w\|$ shrinks to about $0.003$. With much larger noise you are back in 5.2's smooth
equilibrium.

**Mechanism 2: Adam's second moment goes stale.** Adam (with gradient $g$) divides each
coordinate's step $\eta\hat m$ by $\sqrt{\hat v}+\epsilon$, where $\hat m$ and $\hat v$ are
running averages of $g$ and $g^2$ with decay rates $\beta_1,\beta_2$ (5.2). Late in training,
freeze the preconditioner $P=\mathrm{diag}(\sqrt{\hat v}+\epsilon)$ and define the
*preconditioned sharpness* $S_P=\lambda_{\max}(P^{-1}H)$, $H$ the Hessian. On a quadratic the
positive-curvature modes are then stable iff $\eta S_P<(2+2\beta_1)/(1-\beta_1)$, which is 38
at $\beta_1=0.9$ ([NB2 §2.5](td2_2_edge_of_stability.html#td22-5);
[Cohen et al. 2022](https://arxiv.org/abs/2207.14484)), and full-batch Adam drives $\eta S_P$
up to this threshold and holds it there.

In a quiet phase where some coordinates' gradients fall well below $\sqrt{\hat v}$, $\hat v$
decays instead of tracking $g^2$, so $\eta S_P$ grows past the threshold; the oscillation
produces large gradients, $\hat v$ catches up on its $1/(1-\beta_2)$ timescale, and the loss
recovers. That is Bai et al.'s account. Not every crossing spikes: they find the better
predictor is the preconditioned curvature along the gradient, $\eta\,g^\top P^{-1}Hg/\|g\|^2$,
crossing the threshold. ([Molybog et al.](https://arxiv.org/abs/2304.09871) give a related
account for 65B–546B-parameter models in which $\epsilon$, not $\hat v$'s decay, sets the
scale.)

**Exercise 5.3.2 — Adam's quiet-phase overshoot, in one dimension.** How fast does a quiet
phase push Adam past its threshold, and how large a gradient does it take to recover? Take $L=\tfrac12hx^2$ and Adam with $\beta_1=0$ (i.e. RMSprop), $\epsilon=0$, late in
training (ignore bias correction): $g_k=hx_k$, $v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2$,
$x_{k+1}=x_k-\eta g_k/\sqrt{v_k}$. Write $q_k=\eta h/\sqrt{v_k}$; this is $\eta S_P$ in one
dimension, and at $\beta_1=0$ the threshold $(2+2\beta_1)/(1-\beta_1)$ is 2.

1. With $v$ frozen, show $x_{k+1}=(1-q)x_k$, stable iff $q<2$.
2. *Quiet phase.* Suppose $g_k^2\ll v_k$ for a stretch of steps. Show $v_k\approx\beta_2^kv_0$, so $q$ grows by a factor $F$ in $k_F=2\log F/\log(1/\beta_2)\approx2\log F/(1-\beta_2)$ steps. Evaluate $k_2$ for $\beta_2=0.999$ and $0.9$.
3. *Overshoot.* Suppose $q$ has drifted to $2(1+\delta)$ with small $\delta>0$. Show that the oscillation amplitude then grows by a factor $\approx1+2\delta$ per step, and that $v$ must grow by a factor $(1+\delta)^2\approx1+2\delta$ to bring $q$ back to 2. Writing $c_k=g_k^2/v_{k-1}$, show $v_k/v_{k-1}=\beta_2+(1-\beta_2)c_k$. Conclude that correcting the overshoot within $m$ steps requires $c\gtrsim1+2\delta/\big(m(1-\beta_2)\big)$ on those steps. Evaluate for $\delta=0.1$, $m=10$, with $\beta_2=0.999$ and $\beta_2=0.9$.
4. Why do spikes of this kind need a quiet phase, and why are they rare in the noisy early phase of training?

<details><summary>Hint</summary>

For 3, $|1-q|=1+2\delta$ when $q=2(1+\delta)$, and $q\propto v^{-1/2}$. Over $m$ steps, $\log v$ changes by $\sum\log(\beta_2+(1-\beta_2)c_k)\approx(1-\beta_2)\sum(c_k-1)$, which must reach $\approx2\delta$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $v$ fixed, $x_{k+1}=x_k-\eta hx_k/\sqrt v=(1-q)x_k$; $|1-q|<1$ iff $0<q<2$.

**2.** $v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2\approx\beta_2v_{k-1}$ when $(1-\beta_2)g_k^2\ll\beta_2v_{k-1}$, so $v_k\approx\beta_2^kv_0$ and $q_k\approx\beta_2^{-k/2}q_0$. Growth by $F$ needs $\beta_2^{-k/2}=F$, i.e. $k_F=2\log F/\log(1/\beta_2)$. For $F=2$: $k_2\approx1386$ steps at $\beta_2=0.999$, $k_2\approx13$ at $\beta_2=0.9$.

**3.** At $q=2(1+\delta)$, $x_{k+1}=(1-q)x_k=-(1+2\delta)x_k$: the amplitude grows by $1+2\delta$ per step, with alternating sign. Since $q\propto v^{-1/2}$, returning to $q=2$ needs $v\to(1+\delta)^2v\approx(1+2\delta)v$. Dividing the $v$ recursion by $v_{k-1}$: $v_k/v_{k-1}=\beta_2+(1-\beta_2)c_k$. Over $m$ steps, $\log(v_{\rm end}/v_{\rm start})=\sum\log\big(1+(1-\beta_2)(c_k-1)\big)\approx(1-\beta_2)\sum_k(c_k-1)$, which must reach $2\delta$. So the average of $c_k-1$ over the $m$ steps must be about $2\delta/(m(1-\beta_2))$. With $\delta=0.1$, $m=10$: $\beta_2=0.999$ needs $c\gtrsim21$, i.e. squared gradients twenty times the remembered $v$, which in this quadratic means the loss has risen by more than an order of magnitude above the level $v$ remembers: a spike. $\beta_2=0.9$ needs only $c\gtrsim1.2$: $v$ catches up while the gradient is still ordinary, and $q$ is held near 2 without a visible spike. (In a 1D simulation, $\eta=0.01$, $h=1$, $x_0=1$, gradient noise $10^{-8}$, the peak $c$ in the first overshoot is about 20–30 for $\beta_2$ between 0.99 and 0.9995.) This is the asymmetry behind the two panels of the next cell; a small $\beta_2$ has its own cost, a noisier preconditioner and, in the next cell, slower progress.

**4.** The trigger is a long stretch in which $g^2$ is small compared with the remembered $v$: the coordinate has converged, or its gradients have been small for longer than $1/(1-\beta_2)$ steps. Early in training gradients are large and changing, $v$ is refreshed constantly, and the preconditioned sharpness is pinned by the ordinary edge-of-stability mechanism. Bai et al. note that loss spikes in their LLaMA-style runs are often preceded by unusually small gradients.
"""
    ),
    markdown(
        r"""
The next cell reproduces Bai et al.'s two-layer contrast: a width-20 tanh network fitting
$\sin x+\sin4x$ with full-batch Adam ($\eta=0.02$, $\beta_1=0.9$), at $\beta_2=0.999$ and
$\beta_2=0.9$. Every 20 steps it computes the exact Hessian (61 parameters) and
$\eta S_P=\eta\,\lambda_{\max}\big(P^{-1/2}HP^{-1/2}\big)$ (same eigenvalues as $P^{-1}H$).
"""
    ),
    code(
        r"""
x_fit = torch.linspace(-math.pi, math.pi, 64, dtype=torch.float64)
y_fit = torch.sin(x_fit) + torch.sin(4 * x_fit)
WIDTH = 20

def fit_loss(theta):                    # theta = (W1, b1, W2, b2) for a 1 -> 20 -> 1 tanh network
    W1, c1, W2, c2 = torch.split(theta, [WIDTH, WIDTH, WIDTH, 1])
    pred = torch.tanh(x_fit[:, None] * W1 + c1) @ W2 + c2
    return 0.5 * ((pred - y_fit) ** 2).mean()

def adam_full_batch(eta, b2, T, b1=0.9, eps=1e-8, every=20, seed=0):
    gen = torch.Generator().manual_seed(seed)
    theta = torch.cat([torch.randn(WIDTH, generator=gen, dtype=torch.float64), torch.zeros(WIDTH, dtype=torch.float64),
                       torch.randn(WIDTH, generator=gen, dtype=torch.float64) / math.sqrt(WIDTH),
                       torch.zeros(1, dtype=torch.float64)])
    m, v = torch.zeros_like(theta), torch.zeros_like(theta); losses, psharp = [], []
    for k in range(1, T + 1):
        th = theta.clone().requires_grad_(True); L = fit_loss(th); g, = torch.autograd.grad(L, th)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1**k), v / (1 - b2**k); losses.append(L.item())
        if k % every == 0:
            H = torch.autograd.functional.hessian(fit_loss, theta, vectorize=True)
            s = (vh.sqrt() + eps).rsqrt()
            psharp.append((k, eta * torch.linalg.eigvalsh(s[:, None] * H * s[None, :])[-1].item()))
        theta = theta - eta * mh / (vh.sqrt() + eps)
    return np.array(losses), np.array(psharp)

fig, ax = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
for j, b2 in enumerate([0.999, 0.9]):
    losses, ps = adam_full_batch(0.02, b2, 5000)
    ax[0, j].semilogy(losses); ax[0, j].set(title=rf"full-batch Adam, $\beta_2={b2}$", ylabel="loss")
    ax[1, j].plot(ps[:, 0], ps[:, 1]); ax[1, j].axhline(38, color="k", ls="--")
    ax[1, j].set(xlabel="step", ylabel=r"$\eta S_P$", ylim=(0, 50))
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
With $\beta_2=0.999$, $\eta S_P$ climbs to 38 and then saws around it (each slow rise is
$\hat v$ decaying), and the loss spikes by one to two orders of magnitude, though not at
every crossing. With $\beta_2=0.9$, $\eta S_P$ overshoots once early (to about 45), then sits
a little below 38, and the loss never spikes; but it ends near $10^{-3}$, against about
$2\times10^{-5}$ for $\beta_2=0.999$ despite its spikes. Here lowering $\beta_2$ trades spikes
for slower progress. That this is what happens in language models rests on Bai et al.'s
larger experiments, not on this cell.

**Mechanism 3: attention-logit growth.** An attention head with query and key matrices
$W_Q,W_K$ has logits $z_{ij}=\langle W_Qx_i,W_Kx_j\rangle/\sqrt{d_h}$ ($x_i$ the inputs, $d_h$
the head dimension). They are bilinear in $W_Q,W_K$, so they grow quadratically with the
parameter scale; when they get large, attention collapses to near one-hot and training
diverges. [Dehghani et al.](https://arxiv.org/abs/2302.05442) saw this in ViTs of about 8B
parameters (logits past $5\times10^4$). Wortsman et al. reproduce it in small models at high
learning rate, find the growth is in $\|q\|,\|k\|$ rather than their alignment, show it
persists with a pointwise (non-softmax) attention variant, and remove it with QK-layernorm.
Nobody has shown it to be an edge-of-stability crossing. The exercise shows why logits grow
with the weights and why a collapsed head stays collapsed, not the divergence itself.

**Exercise 5.3.3 — attention logits and QK-layernorm.** What sets the size of attention
logits, and how does QK-layernorm cap it? Let $x_i\in\mathbb R^d$ be the (normalized) inputs to an attention head, $q_i=W_Qx_i$,
$k_j=W_Kx_j\in\mathbb R^{d_h}$, logits $z_{ij}=\langle q_i,k_j\rangle/\sqrt{d_h}$ and
weights $p_{ij}=\mathrm{softmax}_j(z_{ij})$.

1. Show $z(cW_Q,c'W_K)=cc'\,z(W_Q,W_K)$ and $|z_{ij}|\le\|W_Q\|_2\|W_K\|_2\|x_i\|\|x_j\|/\sqrt{d_h}$. If both matrices grow in norm by a factor $\rho$ (as can happen at high learning rate without weight decay), how do the logits scale?
2. The Jacobian of $p_{i\cdot}$ with respect to $z_{i\cdot}$ is $\mathrm{diag}(p)-pp^\top$. Show that if the largest attention probability in the row, say $p_{i1}$, is $1-\delta$, every entry of this Jacobian is $O(\delta)$. What does that do to the gradient signal reaching the losing keys?
3. QK-layernorm replaces $q$ and $k$ by $\hat q=\mathrm{LN}(q)$, $\hat k=\mathrm{LN}(k)$ (no gain). Show $|z_{ij}|\le\sqrt{d_h}$. With elementwise gains $\gamma_q,\gamma_k$, what is the bound? What does QK-layernorm do to $W_Q$ and $W_K$ in the sense of 5.1?

<details><summary>Hint</summary>

For 3, LayerNorm without gain outputs a vector with zero mean and unit mean square over its $d_h$ coordinates; what is its Euclidean norm? Then use Cauchy–Schwarz.
</details>
"""
    ),
    answer(
        r"""
**1.** $z$ is linear in $q_i$ and linear in $k_j$, which are linear in $W_Q$ and $W_K$: scaling them by $c,c'$ scales $z$ by $cc'$. Cauchy–Schwarz and the operator-norm bound give $|z_{ij}|\le\|q_i\|\|k_j\|/\sqrt{d_h}\le\|W_Q\|_2\|x_i\|\,\|W_K\|_2\|x_j\|/\sqrt{d_h}$. If both norms grow by $\rho$ and the directions stay put, logits grow by $\rho^2$. Wortsman et al. note that attention logits are the only quantity they examine that depends quadratically on the parameter scale, and that the measured growth is in $\|q\|,\|k\|$, not in the alignment between them.

**2.** The entries are $p_a(\mathbb 1[a=b]-p_b)$. Let $p_1=1-\delta$, so $\sum_{a\ne1}p_a=\delta$. Diagonal: $p_1(1-p_1)=\delta(1-\delta)$, and $p_a(1-p_a)\le p_a\le\delta$ for $a\ne1$. Off-diagonal: $|p_ap_b|\le\delta$ whenever either index is not 1. So the whole Jacobian is $O(\delta)$, and the gradient reaching every logit of that row is suppressed by $\delta$. The losing keys get almost no signal, so the head cannot easily un-collapse: the saturated state is self-sustaining.

**3.** With no gain, $\mathrm{LN}(q)$ has mean zero and mean square one over its $d_h$ coordinates, so $\|\hat q\|=\|\hat k\|=\sqrt{d_h}$ and $|z_{ij}|\le\|\hat q\|\|\hat k\|/\sqrt{d_h}=\sqrt{d_h}$ (8 for $d_h=64$). With gains, $\|\gamma\odot\hat q\|\le\|\gamma_q\|_\infty\sqrt{d_h}$, so $|z_{ij}|\le\|\gamma_q\|_\infty\|\gamma_k\|_\infty\sqrt{d_h}$. The logit scale can now only grow through $2d_h$ gain parameters, not through the full matrices. And $W_Q$, $W_K$ become scale-invariant (per head), so by 5.2 their norms now only set an effective learning rate: the growth that was the instability has been moved into the harmless quantity of 5.2.1.
"""
    ),
    markdown(
        r"""
**Mechanism 4: output-logit divergence.** For logits $z\in\mathbb R^K$, write
$\log Z=\log\sum_ie^{z_i}$; the cross-entropy for label $y$ is $\log Z-z_y$. It is invariant
under shifting all logits by a constant, so the overall level of the logits is a flat
direction of the loss. Wortsman et al. observe, in models trained without weight decay, that
late in training the output logits drift to very negative values and the loss diverges.
z-loss [(from PaLM)](https://arxiv.org/abs/2204.02311), an auxiliary term $10^{-4}\log^2Z$,
removes it, and weight decay also helps at larger scales. They do not say why the drift
eventually breaks training; the exercise shows why it happens at all under Adam. The drift
runs along a direction of zero curvature, so it is not an edge-of-stability crossing; whether
what finally diverges is one is not established.

**Exercise 5.3.4 — a flat direction that Adam does not respect.** Why does Adam push the
output logits along a direction the loss cannot see, when SGD does not? Take a final layer $z=Wh+b$, $W\in\mathbb R^{K\times n}$, on features $h\in\mathbb R^n$, with
softmax cross-entropy: $p=\mathrm{softmax}(z)$, $\log Z=\log\sum_ie^{z_i}$, and loss
$\mathrm{CE}=\log Z-z_y$ for label $y$, whose gradient is $\partial\,\mathrm{CE}/\partial z=p-e_y$ ($e_y$ the one-hot vector of $y$). Write $\bar z=\frac1K\sum_iz_i$ for the mean logit.

1. Show that the cross-entropy is invariant under $b\to b+c\mathbf 1$ and under $W\to W+\mathbf 1v^\top$ for any $c,v$. Show that the gradient is orthogonal to these directions, so under SGD (any batch size, no weight decay) the column sums $\mathbf 1^\top W$ and the sum $\mathbf 1^\top b$ are exactly conserved.
2. Now take sign descent (Adam with $\beta_1=\beta_2=\epsilon=0$) with a batch of one example with label $y$, and assume $0<p_i<1$. Show that the logits of that example (holding $h$ fixed) change by $\Delta z_i=-\eta\,s_i(\|h\|_1+1)$, with $s_i=+1$ for $i\ne y$ and $s_y=-1$. Deduce that the mean logit falls by $\eta(\|h\|_1+1)(K-2)/K$ per step, and that to first order $\Delta\log Z=-\eta(\|h\|_1+1)(1-2p_y)$. When does $\log Z$ fall?
3. Show that z-loss, $\alpha_z(\log Z)^2$ with coefficient $\alpha_z>0$, has gradient $2\alpha_z\log Z\,p$ with respect to $z$, whose component along $\mathbf 1$ is $2\alpha_z\log Z$. What does it pin, and what does $\log Z=0$ mean for the logits?

<details><summary>Hint</summary>

For 2, $\partial\mathrm{CE}/\partial W_{ij}=(p_i-\mathbb 1[i=y])h_j$, so the sign of each coordinate is $s_i\,\mathrm{sign}(h_j)$. Then $\Delta z_i=\sum_j\Delta W_{ij}h_j+\Delta b_i$, and $\sum_js\,\mathrm{sign}(h_j)h_j=s\|h\|_1$. For $\Delta\log Z$, use $\partial\log Z/\partial z_i=p_i$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathrm{CE}=\log\sum_ie^{z_i}-z_y$ is unchanged by $z\to z+c\mathbf 1$: $\log Z$ gains $c$ and so does $z_y$. Both $b\to b+c\mathbf 1$ and $W\to W+\mathbf 1v^\top$ shift every logit by the same amount ($c$, respectively $v^\top h$). The gradient with respect to $z$ is $p-e_y$, and $\mathbf 1^\top(p-e_y)=1-1=0$. So $\nabla_b\mathrm{CE}=p-e_y$ has zero sum, and $\nabla_W\mathrm{CE}=(p-e_y)h^\top$ has zero column sums. Averaging over a batch preserves this. Every SGD step therefore leaves $\mathbf 1^\top b$ and $\mathbf 1^\top W$ unchanged. The mean logit $\bar z=\frac1K(\mathbf 1^\top Wh+\mathbf 1^\top b)$ can still change through $h$, but the parameters along the flat direction never move.

**2.** Sign descent updates $\Delta b_i=-\eta\,\mathrm{sign}(p_i-\mathbb 1[i=y])=-\eta s_i$ and $\Delta W_{ij}=-\eta\,s_i\,\mathrm{sign}(h_j)$. On the same example, $\Delta z_i=\sum_j\Delta W_{ij}h_j+\Delta b_i=-\eta s_i(\sum_j|h_j|+1)$. Averaging over $i$: $\sum_is_i=(K-1)-1=K-2$, so $\Delta\bar z=-\eta(\|h\|_1+1)(K-2)/K$, strictly negative for $K>2$ (zero for binary classification). To first order, $\Delta\log Z=\sum_ip_i\Delta z_i=-\eta(\|h\|_1+1)\big(\sum_{i\ne y}p_i-p_y\big)=-\eta(\|h\|_1+1)(1-2p_y)$, which is negative whenever $p_y<\tfrac12$, the typical case for a language model whose loss is a few nats. The loss still decreases: $\Delta(\log Z-z_y)=-2\eta(\|h\|_1+1)(1-p_y)<0$. The coordinatewise sign map does not preserve "sums to zero", so every step has a component along the flat direction, and it points down. Real Adam averages over steps and batches, which weakens and can change this, but there is no conservation law to stop the drift.

**3.** $\partial(\log Z)/\partial z=p$, so $\nabla_z\,\alpha_z(\log Z)^2=2\alpha_z\log Z\,p$, whose sum is $2\alpha_z\log Z$. This is a restoring force along exactly the direction cross-entropy cannot see, proportional to $\log Z$; it pins $\log Z$ near 0. With $\log Z=0$, $z_i=\log p_i$: the logits *are* the log-probabilities. Wortsman et al. describe the instability as the output logits diverging from the log-probabilities; z-loss removes that freedom. Weight decay is another restoring force (toward $W=0$, $b=0$), which is consistent with their finding that it also helps.
"""
    ),
    code(
        r"""
# A small classifier on overlapping Gaussian classes (loss stays ~2.5 nats, like an LM).
gen = torch.Generator().manual_seed(0)
K_cls, d_in, N, hid = 32, 16, 8192, 32
means = torch.randn(K_cls, d_in, generator=gen)
labels = torch.randint(0, K_cls, (N,), generator=gen)
inputs = means[labels] + 3 * torch.randn(N, d_in, generator=gen)

def train_classifier(opt_name, lr, z_coef, steps=2000, batch=64):
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(d_in, hid), torch.nn.LayerNorm(hid), torch.nn.ReLU(),
                              torch.nn.Linear(hid, K_cls))
    with torch.no_grad():                      # start near uniform predictions: log Z ~ 0 at init
        net[-1].weight.mul_(0.1); net[-1].bias.fill_(-math.log(K_cls))
    opt = (torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.95)) if opt_name == "adam"
           else torch.optim.SGD(net.parameters(), lr=lr))
    hist = []
    for t in range(steps):
        idx = torch.randint(0, N, (batch,))
        z = net(inputs[idx]); logZ = torch.logsumexp(z, 1)
        ce = (logZ - z.gather(1, labels[idx, None])[:, 0]).mean()
        opt.zero_grad(); (ce + z_coef * (logZ ** 2).mean()).backward(); opt.step()
        hist.append((ce.item(), logZ.mean().item(), z.mean().item()))
    return np.array(hist)

runs = {"SGD (lr 0.3)": ("sgd", 0.3, 0.0), "Adam (lr 1e-2)": ("adam", 1e-2, 0.0),
        "Adam + z-loss 1e-2": ("adam", 1e-2, 1e-2)}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
for name, args in runs.items():
    h = train_classifier(*args)
    smooth = lambda s: np.convolve(s, np.ones(50) / 50, mode="valid")
    for j in range(3):
        ax[j].plot(smooth(h[:, j]), label=name)
    print(f"{name:20s} final CE {h[-200:, 0].mean():.3f}   log Z {h[-200:, 1].mean():6.2f}   mean logit {h[-200:, 2].mean():6.2f}")
for j, t in enumerate(["cross-entropy", r"batch mean of $\log Z$", "mean logit"]):
    ax[j].set(title=t, xlabel="step")
ax[1].axhline(0, color="k", ls="--", lw=1)
ax[0].legend(); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
All three runs start with $\log Z\approx0$ (uniform predictions). Under SGD the final layer's
column sums and bias sum are exactly conserved (5.3.4.1); the mean logit could still move
through the features $h$, but here it stays put, and $\log Z$ rises only because the logits
spread out. Adam reaches the same cross-entropy while steadily dragging the mean logit down
and $\log Z$ well below zero. With z-loss, $\log Z$ stops near zero (about $-0.5$, where its
pull balances Adam's drift). The z-loss coefficient is $10^{-2}$ so the effect shows within
2000 steps; PaLM and Wortsman et al. use $10^{-4}$ over $10^5$ steps. This cell shows the
drift and the pinning, **not** the divergence: nothing breaks here in float32.
"""
    ),
    # ================================================================== 5.4
    markdown(
        r"""
<a id="s54"></a>

## 5.4 Clipping

Is gradient clipping (rescaling the gradient when its norm exceeds a threshold) only a guard
against rare huge gradients? Not only: when the local curvature grows with the gradient norm,
the right step size is $\propto1/\|\nabla L\|$ where the gradient is large, which is what
clipping does, and clipped GD can then be arbitrarily faster than GD with any fixed step.
Under Adam the step is already bounded, and clipping mainly protects the optimizer state.

"""
        + reading(
            "Why gradient clipping accelerates training: a theoretical justification for adaptivity",
            "https://arxiv.org/abs/1905.11881",
            "Zhang, He, Sra & Jadbabaie (ICLR 2020)",
            "Introduces (L₀,L₁)-smoothness, measures it along training trajectories, and proves that clipped GD can be arbitrarily faster than fixed-step GD on such functions.",
            importance=2,
            length=3,
        )
        + r"""

**The core idea.** $L$ is $(L_0,L_1)$-*smooth* if $\|\nabla^2L(x)\|\le L_0+L_1\|\nabla L(x)\|$
(operator norm on the Hessian). This contains ordinary $L$-smoothness ($L_1=0$) and also, e.g.,
every univariate polynomial of degree at least 3. A fixed step must be safe at the worst
point of the run and is then too small everywhere else. Zhang et al. see curvature tracking
the gradient norm clearly in LSTM language models, but in ResNet-20 on CIFAR-10 only with
clipping and a large learning rate; whether clipping speeds up a given run depends on that.

<a id="s54-walkthrough"></a>

### Optional: annotated derivations and walkthrough

**Setup.** Clipped GD is
$$
x_{k+1}=x_k-h_k\nabla L(x_k),\qquad h_k=\min\Big(\eta,\ \frac{\gamma\eta}{\|\nabla L(x_k)\|}\Big):
$$
step $\eta$ where the gradient is small, a step of fixed length $\gamma\eta$ where it is
large. Exercise 5.4.1 shows on a two-line example why it can beat every fixed step;
Exercise 5.4.2 derives the descent lemma behind the general statement.

**Exercise 5.4.1 — a two-line example.** How much faster can clipped GD be than GD with the
best fixed step? Take $L(x,y)=\cosh x+\tfrac14y^2$, started at $(x_0,1)$ with $x_0>0$ large. The $x$
direction has curvature growing with the gradient; the $y$ direction is an ordinary
quadratic with curvature $\tfrac12$.

1. Show that $L$ is $(1,1)$-smooth, i.e. $\|\nabla^2L\|\le1+\|\nabla L\|$, using $\cosh x\le1+|\sinh x|$.
2. Show that fixed-step GD with $h\sinh x_0>2x_0$ overshoots to $|x_1|>x_0$ and then diverges. Conclude that *every* fixed step that converges has $h<2x_0/\sinh x_0\approx4x_0e^{-x_0}$, and that the $y$ coordinate then needs at least $\approx2\log(1/\epsilon)/h\approx\sinh(x_0)\log(1/\epsilon)/x_0$ steps to reach $|y|<\epsilon$.
3. Clipped GD on the global gradient, with $\eta=1$, $\gamma=1$ (step length at most 1): roughly how many steps?
4. Why is the second coordinate needed? (What happens in one dimension with $h=x_0/\sinh x_0$?)

Then check it numerically: implement clipped GD, and compare it with fixed-step GD over a grid of step sizes, on the number of steps to reach $\|(x,y)\|<10^{-8}$.

<details><summary>Hint</summary>

The two coordinates decouple: $x\mapsto x-h\sinh x$ and $y\mapsto(1-h/2)y$. For 2, compare $|x_1|$ with $x_0$, and use that $\sinh t/t$ increases in $t>0$. Then the $y$ coordinate alone gives the step count.
</details>
"""
    ),
    answer(
        r"""
**1.** $\cosh x-|\sinh x|=e^{-|x|}\le1$. The Hessian is $\mathrm{diag}(\cosh x,\tfrac12)$, so $\|\nabla^2L\|=\cosh x\le1+|\sinh x|\le1+\|\nabla L\|$.

**2.** The $x$ update is $x\mapsto x-h\sinh x$, independent of $y$. $x_1<-x_0$ iff $h\sinh x_0>2x_0$. Then $|x_1|>x_0$, and since $\sinh|x|/|x|$ increases with $|x|$, the condition $h\sinh|x|>2|x|$ holds even more strongly at $x_1$, so $|x_k|$ keeps growing. At equality $x$ lands on the period-2 orbit $\pm x_0$. Every convergent fixed step therefore has $h<2x_0/\sinh x_0\approx4x_0e^{-x_0}$. The $y$ update is $y\mapsto(1-h/2)y$, so $|y_k|=(1-h/2)^k$, and $|y_k|<\epsilon$ needs $k\ge\log(1/\epsilon)/(-\log(1-h/2))\ge\log(1/\epsilon)/\big(-\log(1-x_0/\sinh x_0)\big)\approx\sinh(x_0)\log(1/\epsilon)/x_0$: exponential in $x_0$, for every fixed step, not just badly tuned ones.

**3.** While $\|\nabla L\|>1$, the step has length exactly 1, and since $|\sinh x|\gg|y|/2$ it points almost along $x$: about $x_0$ steps to reach $|x|\lesssim\operatorname{arcsinh}1\approx0.88$ (while $y$ barely moves). From there the step is unclipped with $h=1$: $x\mapsto x-\sinh x\approx-x^3/6$ converges in a few steps and $y$ halves each step, so about $\log_2(1/\epsilon)\approx27$ more steps. Total $\approx x_0+30$: linear in $x_0$ against exponential.

**4.** In one dimension, the fixed step $h=x_0/\sinh x_0$ sends $x_0$ exactly to the minimum in one step: an oracle step tuned to the starting point. A fixed step has to be safe at the worst point of the run and is then too small everywhere else; with a second, well-conditioned direction, "everywhere else" cannot be skipped. (Zhang et al.'s lower bound is likewise a worst case over a class of functions.)
"""
    ),
    code_answer(
        r"""
def clipped_gd_steps(x0, eta=1.0, gamma=1.0, tol=1e-8):
    # TODO: clipped GD on L = cosh(x) + y^2/4 from (x0, 1); return the number of steps to |(x, y)| < tol.
    ...

def fixed_gd_steps(x0, hs, tol=1e-8):
    # TODO: fixed-step GD for each h in the grid hs (vectorize over hs); return the fewest steps and its h.
    ...

# for x0 in [1.7, 3.7, 5.7, 7.7, 9.7, 11.7]: print both, and the lower bound from Exercise 5.4.1.2
""",
        r"""
def clipped_gd_steps(x0, eta=1.0, gamma=1.0, tol=1e-8):
    x, y = float(x0), 1.0
    for k in range(10**6):
        if math.hypot(x, y) < tol:
            return k
        gx, gy = math.sinh(x), y / 2
        h = min(eta, gamma * eta / math.hypot(gx, gy))
        x, y = x - h * gx, y - h * gy

def fixed_gd_steps(x0, hs, tol=1e-8, max_steps=3 * 10**5):
    x, y = np.full(len(hs), float(x0)), np.ones(len(hs))
    for k in range(max_steps):
        ok = np.hypot(x, y) < tol
        if ok.any():                          # the first step size to converge is the best on the grid
            return k, hs[np.argmax(ok)]
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.where(np.abs(x) < 50, x - hs * np.sinh(np.clip(x, -50, 50)), np.inf)
        y = y - hs * y / 2
    return None, None

print("  x0 | clipped GD | best fixed step on grid (h)  | lower bound for any fixed step")
for x0 in [1.7, 3.7, 5.7, 7.7, 9.7, 11.7]:
    n, h = fixed_gd_steps(x0, np.geomspace(1e-5, 1.9, 400))
    print(f"{x0:5.1f}| {clipped_gd_steps(x0):6d}     | {n:8d}   (h = {h:.2e})     | {math.log(1e8) / -math.log(1 - x0 / math.sinh(x0)):10.0f}")
""",
    ),
    markdown(
        r"""
The best fixed step on the grid sits just below $2x_0/\sinh x_0$, and its step count sits
just above the lower bound, which holds for every fixed step; clipped GD's grows by about one
step per unit of $x_0$ ($\approx x_0+26$), so its advantage is clear by $x_0\approx3$ (at
$x_0=1.7$ the tuned fixed step still wins, 20 steps to 28).
"""
    ),
    markdown(
        r"""
**Exercise 5.4.2 — the descent lemma and the optimal step.** What step size does
$(L_0,L_1)$-smoothness certify as safe, and how does it compare with clipping? Assume $\|\nabla^2L\|\le L_0+L_1\|\nabla L\|$ everywhere (operator norm on the Hessian). Fix $x$, write $g=\nabla L(x)$, and let $\|d\|\le1/L_1$.

1. Let $\phi(s)=\|\nabla L(x+sd)\|$. Show $\phi'(s)\le(L_0+L_1\phi(s))\|d\|$ (where differentiable), and conclude by Grönwall that $L_0+L_1\phi(s)\le(L_0+L_1\|g\|)e^{L_1s\|d\|}$.
2. Deduce $\|\nabla L(x+sd)-g\|\le(L_0+L_1\|g\|)\big(e^{L_1s\|d\|}-1\big)/L_1$, and then
$$
L(x+d)\le L(x)+\langle g,d\rangle+(e-2)\,(L_0+L_1\|g\|)\,\|d\|^2 .
$$
3. Take $d=-hg$. Find the $h$ minimizing the bound, check it satisfies $\|d\|\le1/L_1$, and show it equals clipped GD (step $\min(\eta,\gamma\eta/\|g\|)$ for suitable $\eta,\gamma$) up to a factor of 2. How much does each step decrease $L$, for small and for large $\|g\|$?
4. Fixed-step GD uses a single $h$ for the whole run. For which $h$ does the descent lemma certify descent at every iterate, and what does that suggest about its speed?

<details><summary>Hint</summary>

For 2, write the Taylor remainder as $\int_0^1\langle\nabla L(x+sd)-g,d\rangle\,ds$, substitute the bound, and use that $u\mapsto\big((e^u-1)/u-1\big)/u$ is increasing on $(0,1]$ with value $e-2$ at $u=1$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\phi'(s)\le\|\frac{d}{ds}\nabla L(x+sd)\|=\|\nabla^2L(x+sd)\,d\|\le(L_0+L_1\phi(s))\|d\|$. Let $\psi=L_0+L_1\phi$. Then $\psi'\le L_1\|d\|\,\psi$, so $\psi(s)\le\psi(0)e^{L_1s\|d\|}$. (If $\phi$ is not differentiable at isolated points, the same holds for its upper Dini derivative.)

**2.** $\|\nabla L(x+sd)-g\|\le\int_0^s\|\nabla^2L(x+\sigma d)\|\|d\|\,d\sigma\le\int_0^s\psi(\sigma)\|d\|\,d\sigma\le(L_0+L_1\|g\|)\,\|d\|\int_0^se^{L_1\sigma\|d\|}d\sigma=(L_0+L_1\|g\|)\frac{e^{L_1s\|d\|}-1}{L_1}$.

Then, with $u=L_1\|d\|\le1$,
$$
L(x+d)-L(x)-\langle g,d\rangle=\int_0^1\langle\nabla L(x+sd)-g,d\rangle ds
\le\frac{(L_0+L_1\|g\|)\|d\|}{L_1}\int_0^1(e^{us}-1)\,ds
=(L_0+L_1\|g\|)\|d\|^2\cdot\frac{(e^u-1)/u-1}{u}.
$$
The last factor increases from $\tfrac12$ (as $u\to0$) to $e-2\approx0.718$ (at $u=1$), giving the claim. As $L_1\to0$ this recovers the usual descent lemma up to the constant ($\tfrac12$ becomes $e-2$ because we allowed $u$ up to 1).

**3.** With $d=-hg$ and $\kappa=2(e-2)\approx1.44$: $L(x-hg)\le L(x)-h\|g\|^2+\tfrac\kappa2(L_0+L_1\|g\|)h^2\|g\|^2$. Minimizing over $h$:
$$
h^*=\frac1{\kappa(L_0+L_1\|g\|)},\qquad L(x-h^*g)\le L(x)-\frac{\|g\|^2}{2\kappa(L_0+L_1\|g\|)}.
$$
Constraint: $h^*\|g\|=\|g\|/(\kappa(L_0+L_1\|g\|))<1/(\kappa L_1)<1/L_1$ since $\kappa>1$. Because $\max(a,b)\le a+b\le2\max(a,b)$,
$$
\tfrac12\min\Big(\frac1{\kappa L_0},\frac{1}{\kappa L_1\|g\|}\Big)\le h^*\le\min\Big(\frac1{\kappa L_0},\frac1{\kappa L_1\|g\|}\Big),
$$
and the right side is clipped GD with $\eta=1/(\kappa L_0)$ and $\gamma\eta=1/(\kappa L_1)$. The decrease per step is at least $\min\big(\|g\|^2/(4\kappa L_0),\ \|g\|/(4\kappa L_1)\big)$: quadratic in $\|g\|$ when the gradient is small (as for ordinary GD), but *linear* in $\|g\|$ when it is large, because the step length saturates at about $1/L_1$, the radius over which the curvature bound can be trusted.

**4.** The lemma certifies descent at every iterate for $h\lesssim1/(\kappa(L_0+L_1M))$, where $M$ bounds the gradient norm along the trajectory. This is sufficient, not necessary, but it suggests the problem: a fixed step that is safe at the worst point of the run (typically the start) is paid for at every step, including the many steps near the optimum where the curvature is only $\sim L_0$, giving a cost $\sim(L_0+L_1M)\Delta/\epsilon^2$ to reach $\|\nabla L\|\le\epsilon$, where $\Delta=L(x_0)-\inf L$. For the optimal step of part 3, by contrast, the decrease per step, $\min\big(\|g\|^2/(4\kappa L_0),\|g\|/(4\kappa L_1)\big)$, is at least $\epsilon^2/(4\kappa L_0)$ whenever $\|g\|\ge\epsilon$ and $\epsilon\le L_0/L_1$ (check the two cases $\|g\|\lessgtr L_0/L_1$), so it needs at most $4\kappa L_0\Delta/\epsilon^2$ steps in that small-$\epsilon$ regime. Zhang et al.'s lower bound shows this is not just a weakness of the proof: it is a worst case over the class, saying that for *each* fixed step there are $(L_0,L_1)$-smooth functions on which that step is slower by a factor growing with $L_1M$. Exercise 5.4.1 is stronger in one respect: there a single function defeats every fixed step.
"""
    ),
    markdown(
        r"""
**Clipping under Adam.** In large-model training, clipping is usually applied to the global
gradient norm before the gradient enters Adam (Wortsman et al. clip at norm 1), and the GD
theory above does not transfer directly. With Adam's coordinatewise
$v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2$, bias correction $\hat v_k=v_k/(1-\beta_2^k)$, $\beta_1=0$
(so the step is $\eta g_k/(\sqrt{\hat v_k}+\epsilon)$) and any $\epsilon\ge0$, we have
$\hat v_k\ge v_k\ge(1-\beta_2)g_k^2$, so
$$
|\Delta w_i|=\frac{\eta\,|g_{k,i}|}{\sqrt{\hat v_{k,i}}+\epsilon}\le\frac{\eta}{\sqrt{1-\beta_2}}.
$$
So clipping is not needed to keep Adam's step finite. It still changes the current step
(with $\beta_2=0.999$ and remembered $v=1$, a gradient of $1000$ moves a coordinate by about
$31.6\eta$; clipped to 1, by $\eta$), and it protects the optimizer state: an unclipped huge
gradient inflates $\hat v$, which then suppresses those coordinates' updates until it decays
back, which takes $\approx\log F/(1-\beta_2)$ steps to fall by a factor $F$ late in training if
later gradients are much smaller. This is a plausible role, not an established theory; we
know of no analysis of clipped Adam comparable to the GD story above.
"""
    ),
    # ================================================================== readings
    markdown(
        r"""
<a id="s5r"></a>

## Readings

The lead papers are linked at the top of the page and of each section. Also:
[Li, Lyu & Arora, the intrinsic learning rate](https://arxiv.org/abs/2010.02916) (names
$\eta\lambda$, conjectures equilibrium in $O(1/\eta\lambda)$ steps; 5.2);
[Lobacheva et al., periodic behavior with BatchNorm and weight decay](https://arxiv.org/abs/2106.15739)
(mechanism 1);
[D'Angelo et al., why we need weight decay](https://arxiv.org/abs/2310.04415) (argues it is
never useful as an explicit regularizer; acts through $\eta/\|w\|$ under AdamW);
[Wan et al. 2021, spherical motion dynamics](https://proceedings.neurips.cc/paper_files/paper/2021/file/326a8c055c0d04f5b06544665d8bb3ea-Paper.pdf)
(the per-step SGD equilibrium angle under explicit noise assumptions).
"""
    ),
]

raise SystemExit(run(CELLS, "td2_5_normalization_weight_decay", "td2_5_normalization_weight_decay"))
